import numpy as np
import torch
from bisect import bisect
from collections import defaultdict

_grad_map = defaultdict(list)

def set_grad_data(evts: np.ndarray):
    """
    evts: numpy array of shape (N, 5), each row is (u, v, t, sign, w)
    """
    global _grad_map
    _grad_map.clear()
    for u, v, t, sign, w in evts:
        key = (int(u), int(v))
        _grad_map[key].append((float(t), float(w)))
    # Sort events by time for each (u, v)
    for key in _grad_map:
        _grad_map[key].sort(key=lambda x: x[0])

def compute_interpolated_weights(eids, edts, df):
    """
    eids: list or 1D tensor, indices of edges in the original df
    edts: 1D tensor, timestamps for each edge in the subgraph
    df: pandas.DataFrame, original edge table, must contain 'src' and 'dst' columns
    Returns: torch tensor of interpolated weights with shape (len(edts),)
    """
    device = edts.device
    out = []
    # Convert eids and edts to Python lists
    eids_list = eids.tolist() if isinstance(eids, torch.Tensor) else list(eids)
    edts_list = edts.tolist()
    for eid, t in zip(eids_list, edts_list):
        u = int(df.at[eid, 'src'])
        v = int(df.at[eid, 'dst'])
        # Find gradient records in either forward or reverse direction
        events = _grad_map.get((u, v), []) + _grad_map.get((v, u), [])
        if not events:
            out.append(0.0)
            continue
        times, weights = zip(*events)
        idx = bisect(times, t)
        if idx == 0:
            w_interp = weights[0]
        elif idx >= len(times):
            w_interp = weights[-1]
        else:
            t0, w0 = times[idx-1], weights[idx-1]
            t1, w1 = times[idx],   weights[idx]
            w_interp = w0 + (w1 - w0) * (t - t0) / (t1 - t0)
        out.append(w_interp)
    return torch.tensor(out, device=device, dtype=torch.float32)

