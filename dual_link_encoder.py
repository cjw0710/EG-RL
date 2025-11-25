import torch
import torch.nn as nn
from model import TimeEncode          

class _SingleLink(nn.Module):
    """(Δt, weight) → timeEnc × weight → Linear → mean-pool"""
    def __init__(self, time_dim, hid_dim):
        super().__init__()
        self.tenc = TimeEncode(time_dim)
        self.proj = nn.Linear(time_dim, hid_dim, bias=False)

    def forward(self, seq):           # seq [B,L,2]
        dt, w = seq[..., :1], seq[..., 1:2]
        z = self.proj(self.tenc(dt) * w)        # [B,L,hid]
        return z.mean(1)                        # [B,hid]

class DualLinkEncoder(nn.Module):
    """正负通道各跑一次 _SingleLink, 输出拼接"""
    def __init__(self, time_dim, hid_dim):
        super().__init__()
        self.pos = _SingleLink(time_dim, hid_dim)
        self.neg = _SingleLink(time_dim, hid_dim)

    def forward(self, pos_seq, neg_seq):
        return torch.cat([self.pos(pos_seq), self.neg(neg_seq)], -1)
