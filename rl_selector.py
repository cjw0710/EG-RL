# rl_selector.py
import torch, torch.nn as nn, torch.nn.functional as F

class Policy(nn.Module):          # πθ
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
    def forward(self, s):                         # s: [B, state_dim]
        return torch.sigmoid(self.net(s)).squeeze(-1)     # [B]

class Critic(nn.Module):           # Qψ
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
    def forward(self, s):
        return self.net(s).squeeze(-1)
