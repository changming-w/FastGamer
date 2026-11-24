"""Lightweight neural modulator for global refinement (Sec. 3.4).

To inject global context into the gain-map prediction without expensive spatial
alignment, a small MLP predicts channel-wise ``(alpha, beta)`` vectors from the
global feature ``F_thumb``. They are broadcast over the full resolution and
applied to the LUT-transformed log-GM via an efficient affine transform followed
by ``tanh`` (Eq. 4):

    GM_norm^log = tanh( GM_LUT^log * (1 + alpha) + beta )   in [-1, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralModulator(nn.Module):
    def __init__(self, feat_dim: int = 64, hidden: int = 32, num_channels: int = 3):
        super().__init__()
        self.C = num_channels
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_channels * 2)   # alpha, beta per channel
        # Zero-init -> starts as an identity affine (alpha=0, beta=0).
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feat: torch.Tensor, gm_lut: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(feat), inplace=True)
        ab = self.fc2(x)                                  # (B, 2C)
        alpha, beta = ab[:, :self.C], ab[:, self.C:]
        alpha = alpha.view(-1, self.C, 1, 1)
        beta = beta.view(-1, self.C, 1, 1)
        gm_norm = torch.tanh(gm_lut * (1.0 + alpha) + beta)   # Eq. 4
        return gm_norm
