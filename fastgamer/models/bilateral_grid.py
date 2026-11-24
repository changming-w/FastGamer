"""Dynamic bilateral grid for local inversion (Sec. 3.2).

To undo spatially-varying local tone-mapping degradations, a lightweight MLP
predicts ``K`` bilateral grids of dimension ``Nb x Nb x Nb`` (depth x H x W) from
the global feature ``F_thumb`` (K=3, Nb=8 in the paper). The grids are *sliced*
against the full-resolution input using the input RGB channels themselves as the
range guidance (reusing RGB avoids extra computation). The sliced maps are then
fused with the input RGB through a 1x1 projection to form a spatially-modulated
base ``I_grid`` for the subsequent LUT transformation.

Slicing is a single 3D ``grid_sample`` per grid (network-free at high
resolution): for output pixel ``(y, x)`` and grid ``k`` we sample the volume at
``(x_spatial, y_spatial, guidance = I_HR[:, k])``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilateralGridGenerator(nn.Module):
    """Generate K bilateral grids from F_thumb and slice+fuse on the full-res input."""

    def __init__(self, feat_dim: int = 64, hidden: int = 32, num_grids: int = 3,
                 grid_size: int = 8, in_nc: int = 3):
        super().__init__()
        self.K = num_grids           # number of grids (== range-guidance channels)
        self.Nb = grid_size          # per-axis resolution of each grid
        self.in_nc = in_nc
        assert num_grids == in_nc, "reuse of RGB as range guidance requires K == in_nc"

        out_dim = num_grids * grid_size ** 3
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        # 1x1 projection fusing sliced grids (K) with input RGB (in_nc) -> in_nc.
        self.fuse = nn.Conv2d(num_grids + in_nc, in_nc, kernel_size=1, bias=True)

        # Initialize so that the module starts close to identity (I_grid ~= I_HR):
        # zero the last MLP layer (grids -> ~0 slices) and set the fuse conv to
        # copy the RGB channels through.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.fuse.weight)
        with torch.no_grad():
            for c in range(in_nc):
                self.fuse.weight[c, num_grids + c, 0, 0] = 1.0
        nn.init.zeros_(self.fuse.bias)

    def generate(self, feat: torch.Tensor) -> torch.Tensor:
        """F_thumb -> grids of shape (B, K, Nb, Nb, Nb)."""
        b = feat.shape[0]
        grids = self.mlp(feat).view(b, self.K, self.Nb, self.Nb, self.Nb)
        return grids

    def slice(self, grids: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Slice each grid with (spatial x, y, guidance=img[:,k]) -> (B, K, H, W).

        The K grids are sliced in a single batched 3D ``grid_sample`` (batch dim
        B*K), so grid k uses input channel k as its range guidance -- exactly one
        network-free kernel over the high-resolution image.
        """
        b, _, h, w = img.shape
        bk = b * self.K
        device, dtype = img.device, img.dtype
        # Shared spatial coordinates in [-1, 1], broadcast to the B*K batch.
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(bk, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(bk, 1, h, w)
        # guidance: grid k <- channel k  ->  reshape (B, K, H, W) to (B*K, 1, H, W).
        guide = (img.clamp(0.0, 1.0) * 2.0 - 1.0).reshape(bk, 1, h, w)
        grid = torch.stack([xs, ys, guide], dim=-1)                # (B*K,1,H,W,3): (x=W,y=H,z=guide)
        vol = grids.reshape(bk, 1, self.Nb, self.Nb, self.Nb)      # (B*K,1,Nb,Nb,Nb)
        out = F.grid_sample(vol, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)  # (B*K,1,1,H,W)
        return out.view(b, self.K, h, w)

    def forward(self, feat: torch.Tensor, img: torch.Tensor):
        grids = self.generate(feat)
        sliced = self.slice(grids, img)
        i_grid = self.fuse(torch.cat([sliced, img], dim=1))       # (B, in_nc, H, W)
        return i_grid, grids
