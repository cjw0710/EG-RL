import numpy as np, os, argparse
from collections import defaultdict

def build(src, dst):
    raw = np.load(src)                           # (u,v,t,val)
    out, bucket = [], defaultdict(list)

    for u, v, t, val in raw:
        bucket[(u, v)].append((t, val))

    for (u, v), seq in bucket.items():
        seq.sort()
        prev_val, prev_t = seq[0][1], seq[0][0]
        for t_cur, val_cur in seq[1:]:
            if val_cur == prev_val:
                continue
            L = t_cur - prev_t
            sign = 1 if val_cur == 1 else -1
            for step, τ in enumerate(range(prev_t + 1, t_cur + 1), 1):
                w = step / L                     # 线性渐变
                out.append((u, v, τ, sign, w))
            prev_val, prev_t = val_cur, t_cur

    out = np.asarray(out, dtype=np.float32)
    np.save(dst, out)
    print(f"saved {out.shape[0]} events → {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/edge_list.npy")
    ap.add_argument("--dst", default="data/edge_list_grad.npy")
    build(**vars(ap.parse_args()))
