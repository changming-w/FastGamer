#!/usr/bin/env python
"""Single-image inference: SDR -> reconstructed HDR + gain-map visualization.

Outputs:
  <out>_hdr_pq.png    HDR reconstruction, PQ-encoded for 8-bit preview
  <out>_hdr.npy       relative-linear HDR (float32)
  <out>_gm.png        normalized color gain map visualized to [0, 1]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastgamer.models import build_model
from fastgamer import metrics, utils


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fastgamer.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--input", required=True, help="SDR image path")
    ap.add_argument("--out", default="infer_out/result")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    model = build_model(cfg).to(device).eval()
    if args.ckpt:
        utils.load_checkpoint(args.ckpt, model, map_location=device)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("WARNING: no checkpoint -- using an untrained model.")

    sdr = utils.imread_srgb(args.input)                       # HWC RGB [0,1]
    i_hr = utils.to_chw_tensor(sdr).unsqueeze(0).to(device)
    pred = model(i_hr)                                         # thumbnail auto from I_HR

    hdr_lin = pred["hdr_lin"][0].permute(1, 2, 0).cpu().numpy()
    gm_norm = pred["gm_norm"][0].permute(1, 2, 0).cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    utils.imwrite_srgb(f"{args.out}_hdr_pq.png", np.clip(metrics.to_pq(hdr_lin), 0, 1))
    np.save(f"{args.out}_hdr.npy", hdr_lin.astype(np.float32))
    utils.imwrite_srgb(f"{args.out}_gm.png", np.clip(gm_norm * 0.5 + 0.5, 0, 1))
    print(f"q_max={float(pred['q_max'][0]):.3f}  hdr_peak={hdr_lin.max():.2f}")
    print(f"Saved: {args.out}_hdr_pq.png, {args.out}_hdr.npy, {args.out}_gm.png")


if __name__ == "__main__":
    main()
