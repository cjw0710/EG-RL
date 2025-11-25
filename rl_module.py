# rl_module.py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNet(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)          # keep / drop

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class EdgeSelectEnv:
    """
    Single-edge episode; reward = +1 (hit) / -1 (false positive) minus baseline
    """
    def __init__(self, y_true):
        self.y_true = y_true          # 0/1 tensor aligned with y_pred_pos
        self.i = 0

    def reset(self):
        self.i = 0

    def step(self, action):
        r = 1. if action == self.y_true[self.i] else -1.
        self.i += 1
        done = self.i == len(self.y_true)
        return r, done
