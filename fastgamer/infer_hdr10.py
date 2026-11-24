#!/usr/bin/env python
"""Single-image SDR (8-bit) -> HDR10 (16-bit PQ, BT.2020) inference via FastGaMer.

Mirrors the per-frame color pipeline in ``video_to_hdr10.py`` so the single-image
output is directly comparable with the video pipeline:

    SDR sRGB image (/255)
        -> model(i_hr) on GPU            [relative-linear HDR, BT.709/sRGB]
        -> clip>=0 -> BT.709->BT.2020 -> clip>=0
        -> PQ OETF (x203 nits) -> clip[0,1]
        -> 16-bit PNG (rgb48 / yuv-independent uint16 triplets)

Outputs:
  <out>_hdr10.png       16-bit PQ-encoded BT.2020 image (same data as video rgb48le)
  <out>_hdr.npy         relative-linear HDR (float32, BT.709, for further use)

Usage:
    python scripts/infer_hdr10.py \
        --config configs/fastgamer_multi.yaml \
        --ckpt experiments/fastgamer_multi/model_latest.pth \
        --input /path/to/sdr.jpg \
        --out /path/to/out_prefix
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastgamer.models import build_model
from fastgamer import gainmap, utils


# BT.709/sRGB -> BT.2020 primaries (linear light); mirrors video_to_hdr10.py.
M_709_TO_2020 = [
    [0.6274040, 0.3292820, 0.0433136],
    [0.0690970, 0.9195400, 0.0113612],
    [0.0163916, 0.0880132, 0.8955950],
]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="Single-image SDR 8-bit -> HDR10 16-bit")
    ap.add_argument("--config", default="configs/fastgamer_multi.yaml")
    ap.add_argument("--ckpt", default="experiments/fastgamer_multi/model_latest.pth")
    ap.add_argument("--input", required=True, help="SDR image path (8-bit)")
    ap.add_argument("--out", required=True, help="Output prefix (no extension)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--half", action="store_true", help="fp16 inference (fp32 default)")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    model = build_model(cfg).to(device).eval()
    utils.load_checkpoint(args.ckpt, model, map_location=device)
    if args.half:
        model.half()
    print(f"Loaded checkpoint: {args.ckpt}  (device={device}, half={args.half})")

    # ---- load SDR (imread_srgb reads 8/16-bit -> float [0,1] HWC RGB)
    sdr = utils.imread_srgb(args.input)
    H, W = sdr.shape[:2]
    i_hr = utils.to_chw_tensor(sdr).unsqueeze(0).to(device)   # (1,3,H,W)
    if args.half:
        i_hr = i_hr.half()
    print(f"Input: {args.input}  {W}x{H}")

    M = torch.tensor(M_709_TO_2020, dtype=torch.float32, device=device)
    REF = gainmap.REF_WHITE_NITS

    # ---- model inference
    hdr = model(i_hr)["hdr_lin"].float()[0]                    # (3,H,W) rel-linear, 709
    hdr = hdr.clamp_(min=0.0)
    hdr = (M @ hdr.reshape(3, -1)).reshape(3, H, W).clamp_(min=0.0)   # BT.2020
    pq = gainmap.pq_oetf(hdr * REF).clamp_(0.0, 1.0)                  # PQ [0,1]
    u16 = (pq * 65535.0 + 0.5).to(torch.int32).permute(1, 2, 0).contiguous()
    out16 = u16.cpu().numpy().astype(np.uint16)                       # (H,W,3) RGB

    print(f"peak~{float(hdr.max()) * REF:.0f} nits  pq_max={float(pq.max()):.3f}")

    # ---- save 16-bit PNG (cv2 expects BGR for multi-channel uint16)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    hdr_pq_path = f"{args.out}_hdr10.png"
    import cv2
    cv2.imwrite(hdr_pq_path, out16[..., ::-1])

    # ---- also save relative-linear HDR for downstream use
    hdr_lin_path = f"{args.out}_hdr.npy"
    hdr_lin = hdr.permute(1, 2, 0).cpu().numpy().astype(np.float32)   # (H,W,3) BT.2020
    np.save(hdr_lin_path, hdr_lin)

    print(f"Saved: {hdr_pq_path}")
    print(f"Saved: {hdr_lin_path}")


if __name__ == "__main__":
    main()
