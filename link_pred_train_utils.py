from tqdm import tqdm
import torch
import time
import copy
import os
import json

import numpy as np
from torch_sparse import SparseTensor
from torch_geometric.utils import coalesce, add_self_loops

from data_process_utils import pre_compute_subgraphs, get_random_inds, get_subgraph_sampler
from construct_subgraph import construct_mini_batch_giant_graph, print_subgraph_data

from utils import row_norm, sym_norm
from rl_selector import Policy, Critic
import torch.nn.functional as F

# === RL policy filter ======================================================
def rl_filter(policy, g_emb, edge_feat_full, edge_w, max_k=512):
    # Candidate set: edges whose weights are in (0, 1)
    cand = (edge_w > 0) & (edge_w < 1)
    idx = cand.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        # No candidates -> keep all edges
        mask = torch.ones_like(edge_w, dtype=torch.bool, device=edge_w.device)
        return edge_w, None, None, mask

    # Limit the maximum number of examined edges
    if idx.numel() > max_k:
        perm = torch.randperm(idx.numel(), device=edge_w.device)[:max_k]
        idx = idx[perm]

    # Build states and perform Bernoulli sampling
    state = torch.cat([g_emb.repeat(idx.numel(), 1), edge_feat_full[idx]], dim=1)
    keep_prob = policy(state)
    m = torch.distributions.Bernoulli(keep_prob)
    a = m.sample().to(torch.bool)

    # Apply sampled actions to edge weights
    ew = edge_w.clone()
    ew[idx] = ew[idx] * a.float()

    # Build the mask: keep all by default, then set entries in idx with a == 0 to False
    mask = torch.ones_like(edge_w, dtype=torch.bool, device=edge_w.device)
    mask[idx] = a

    return ew, state.detach(), m.log_prob(a.float()).detach(), mask


# structural features (placeholder)
def compute_sign_feats(node_feats, df, start_i, num_links, root_nodes, args):
    if node_feats is None:
        return None
    idx = torch.tensor(root_nodes, dtype=torch.long, device=node_feats.device)
    return node_feats[idx]


# single-mode run (train/valid/test)
def run(model, optimizer, args, subgraphs, df, node_feats, edge_feats, mode,
        policy=None, buffer=None):
    time_aggre = 0
    is_train = (mode == 'train')
    model.train(is_train)

    # Select a slice of the DataFrame and sampling parameters according to the mode
    if mode == 'train':
        cur_df = df[:args.train_edge_end]
        neg, extra_neg = args.neg_samples, args.extra_neg_samples
    elif mode == 'valid':
        cur_df = df[args.train_edge_end:args.val_edge_end]
        neg = extra_neg = 1
    else:
        cur_df = df[args.val_edge_end:]
        neg = extra_neg = 1

    loader = cur_df.groupby(cur_df.index // args.batch_size)
    pbar = tqdm(total=len(loader))
    pbar.set_description(f"{mode} mode with negative samples {neg} ...")

    all_ap, all_auc = [], []
    if not args.use_cached_subgraph and is_train:
        subgraphs.sampler.reset()

    # Pre-allocate all_inds (upper bound: batch_size * max_edges)
    max_inds = args.batch_size * args.max_edges
    all_inds = torch.empty(max_inds, dtype=torch.long, device=args.device)

    for batch_idx in range(len(loader)):
        # === 1. Sample subgraph mini-batch ===
        if not args.use_cached_subgraph and is_train:
            roots = subgraphs.all_root_nodes[batch_idx]
            samp = get_random_inds(len(roots), extra_neg, neg)
            batch_data = subgraphs.mini_batch(batch_idx, samp)
        else:
            roots = subgraphs[batch_idx]
            samp = get_random_inds(len(roots), extra_neg, neg)
            batch_data = [roots[j] for j in samp]

        data = construct_mini_batch_giant_graph(batch_data, args.max_edges)

        # === 2. Raw features ===
        feats = edge_feats[data['eid']].to(args.device)
        edts  = torch.from_numpy(data['edts']).float().to(args.device)
        ws    = torch.from_numpy(data['weights']).float().to(args.device)
        ptr   = data['all_edge_indptr']  # length = number of subgraphs + 1

        # === 3. RL filtering & actual edge removal ===
        if args.rl_on and policy is not None:
            # 3.1 Global graph embedding
            roots_tensor = torch.tensor(data['root_nodes'], device=args.device)
            g_emb = node_feats[roots_tensor].mean(0, keepdim=True)

            # 3.2 Time encoding
            t = edts.unsqueeze(1) / 1e7
            freq = torch.arange(100, device=args.device) / 100.
            time_phi = torch.cos(t * freq)

            # 3.3 Full state vector
            full_feat = torch.cat([ws.unsqueeze(1), feats, time_phi], dim=1)

            # 3.4 Apply RL-based filtering
            ws, sb, lp, mask = rl_filter(policy, g_emb, full_feat, ws, args.max_k)
            if sb is not None and buffer is not None:
                buffer.append((sb, lp))

            # 3.5 Physically remove edges with weight 0
            feats = feats[mask]
            edts  = edts[mask]
            ws    = ws[mask]

            # 3.6 Recompute ptr
            new_ptr = [0]
            for i in range(len(ptr)-1):
                kept = int(mask[ptr[i]:ptr[i+1]].sum().item())
                new_ptr.append(new_ptr[-1] + kept)
            ptr = new_ptr

        # === 4. Structural features (optional) ===
        sf = compute_sign_feats(
            node_feats, df, None, None, data['root_nodes'], args
        )

        # === 5. Build all_inds & mask_list ===
        cnt = 0
        for sub_i in range(len(ptr)-1):
            num = ptr[sub_i+1] - ptr[sub_i]
            if num > 0:
                start = sub_i * args.max_edges
                all_inds[cnt:cnt+num] = torch.arange(start, start+num, device=args.device)
                cnt += num
        mask_list = [(ptr[i+1] - ptr[i]) > 0 for i in range(len(ptr)-1)]

        # === 6. Forward & backward ===
        inputs = [feats, edts, ws.unsqueeze(1), len(ptr)-1, all_inds[:cnt]]
        start_time = time.time()
        loss, ap, auc = model(inputs, mask_list, neg, sf)
        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        time_aggre += (time.time() - start_time)

        all_ap.append(ap)
        all_auc.append(auc)
        pbar.update(1)

    pbar.close()
    avg_ap  = sum(all_ap)  / len(all_ap)
    avg_auc = sum(all_auc) / len(all_auc)
    print(f"{mode} mode with time {time_aggre:.4f}, "
          f"average precision {avg_ap:.4f}, auc score {avg_auc:.4f}, loss {loss.item():.4f}")
    return avg_ap, avg_auc, loss.item()


# =============== Main training loop ===============
def link_pred_train(model, args, g, df, node_feats, edge_feats):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.rl_on:
        dim_node  = node_feats.shape[1] if node_feats is not None else 0
        state_dim = dim_node + edge_feats.shape[1] + 1 + 100
        policy    = Policy(state_dim).to(args.device)
        critic    = Critic(state_dim).to(args.device)
        opt_pi    = torch.optim.Adam(policy.parameters(), lr=3e-4)
        opt_v     = torch.optim.Adam(critic.parameters(), lr=3e-4)
        buffer, baseline = [], None
    else:
        policy = critic = opt_pi = opt_v = buffer = None
        baseline = None

    # Pre-compute all subgraphs
    train_s = (pre_compute_subgraphs(args, g, df, 'train')
               if args.use_cached_subgraph
               else get_subgraph_sampler(args, g, df, 'train'))
    valid_s = pre_compute_subgraphs(args, g, df, 'valid')
    test_s  = pre_compute_subgraphs(args, g, df, 'test')

    history = {k: [] for k in [
        'train_ap','valid_ap','test_ap',
        'train_auc','valid_auc','test_auc',
        'train_loss','valid_loss','test_loss'
    ]}
    best_ap = -1

    for epoch in range(args.epochs):
        print(f">>> Epoch {epoch+1}")
        ta, tu, tl = run(model, optimizer, args, train_s, df, node_feats, edge_feats, 'train', policy, buffer)
        va, vu, vl = run(model, None,      args, valid_s, df, node_feats, edge_feats, 'valid', policy, None)
        na, nu, nl = run(model, None,      args, test_s,  df, node_feats, edge_feats, 'test',  policy, None)

        # RL parameter update
        if args.rl_on and buffer:
            r = 0.0 if baseline is None else max(min((vu-baseline)/(baseline+1e-8),1),-1)
            if baseline is None: baseline = vu
            rt = torch.tensor(r, device=args.device)
            for sb, lp in buffer:
                vp = critic(sb)
                adv = rt - vp.detach()
                opt_pi.zero_grad();   (-lp * adv).mean().backward(); opt_pi.step()
                opt_v.zero_grad();    F.mse_loss(vp, rt).backward(); opt_v.step()
            buffer.clear()

        # Logging
        history['train_ap'].append(ta);    history['train_auc'].append(tu);    history['train_loss'].append(tl)
        history['valid_ap'].append(va);    history['valid_auc'].append(vu);    history['valid_loss'].append(vl)
        history['test_ap'].append(na);     history['test_auc'].append(nu);     history['test_loss'].append(nl)

        if va > best_ap:
            best_ap  = va
            best_idx = epoch
            best_model = copy.deepcopy(model).cpu()
        if epoch > best_idx + 20:
            break

    print(f"average precision {history['test_ap'][best_idx]:.4f}, "
          f"auc score {history['test_auc'][best_idx]:.4f}")

    history['final_test_ap']  = history['test_ap'][best_idx]
    history['final_test_auc'] = history['test_auc'][best_idx]
    os.makedirs(os.path.dirname(args.link_pred_result_fn), exist_ok=True)
    json.dump(history, open(args.link_pred_result_fn, 'w'))
    return best_model


@torch.no_grad()
def fetch_all_predict(model, optimizer, args, subgraphs, df, node_feats, edge_feats, mode):
    raise NotImplementedError("Not implemented")
