"""Scale estimation (Sec. 3.1).

A two-layer MLP that maps the global feature ``F_thumb`` to the absolute
illuminance scale ``Q_max`` (a single positive scalar per image, in stops).
Estimating the scale from the whole-image thumbnail keeps it globally
consistent rather than patch-dependent, which the paper notes is essential to
avoid train/test inconsistency.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleHead(nn.Module):
    def __init__(self, feat_dim: int = 64, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns Q_max of shape (B, 1, 1, 1), strictly positive (softplus)."""
        x = F.relu(self.fc1(feat), inplace=True)
        q = F.softplus(self.fc2(x))          # positive scale in stops
        return q.view(-1, 1, 1, 1)
