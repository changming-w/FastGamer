#!/usr/bin/env python
"""Evaluate FastGaMer and report metrics in the paper's layout (Tab. 1 / Tab. 2).

Linear domain (peak-normalized): PSNR / SSIM / SRSIM
PQ domain (pq_oetf(linear*203)): PSNR / SSIM / SRSIM
HDR domain: Delta E_ITP, (optional) HDR-VDP-3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastgamer.models import build_model
from fastgamer.data import GainMapDataset
from fastgamer import metrics, utils


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fastgamer.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--with_hdrvdp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    device = args.device if torch.cuda.is_available() else "cpu"

    model = build_model(cfg).to(device).eval()
    ckpt = args.ckpt or cfg["test"].get("ckpt")
    if ckpt:
        utils.load_checkpoint(ckpt, model, map_location=device)
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("WARNING: no checkpoint -- evaluating an untrained model.")

    d = cfg["data"]
    dataset = GainMapDataset(d["root"], args.split, d["patch_size"], d["thumb_size"],
                             d["mode"], augment_data=False, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    agg = {"linear": {}, "pq": {}, "hdr": {}}
    n = 0
    for i, batch in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        i_hr = batch["i_hr"].to(device)
        i_thumb = batch["i_thumb"].to(device)
        pred = model(i_hr, i_thumb)
        pred_lin = pred["hdr_lin"][0].permute(1, 2, 0).cpu().numpy()
        gt_lin = batch["hdr_lin"][0].permute(1, 2, 0).numpy()

        m = metrics.evaluate(pred_lin, gt_lin, with_hdrvdp=args.with_hdrvdp)
        for dom in m:
            for k, v in m[dom].items():
                agg[dom].setdefault(k, []).append(v)
        n += 1

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            pq = metrics.to_pq(pred_lin)
            utils.imwrite_srgb(os.path.join(args.save_dir, f"{batch['name'][0]}_pred_pq.png"), pq)

        print(f"[{n}] {batch['name'][0]:>24s}  "
              f"lin.PSNR {m['linear']['psnr']:.2f}  "
              f"PQ.PSNR {m['pq']['psnr']:.2f}  dEITP {m['hdr']['deltaEITP']:.2f}")

    def mean(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("nan")

    print("\n================ FastGaMer results ({} samples) ================".format(n))
    print(f"{'':8s} | {'PSNR':>8s} {'SSIM':>8s} {'SRSIM':>8s}")
    for dom in ("linear", "pq"):
        print(f"{dom:8s} | {mean(agg[dom]['psnr']):8.3f} "
              f"{mean(agg[dom]['ssim']):8.4f} {mean(agg[dom]['srsim']):8.4f}")
    print(f"HDR      | dEITP {mean(agg['hdr']['deltaEITP']):.3f}", end="")
    if args.with_hdrvdp and "hdrvdp3" in agg["hdr"]:
        print(f"  HDR-VDP3 {mean(agg['hdr']['hdrvdp3']):.3f}", end="")
    print("\n===============================================================")


if __name__ == "__main__":
    main()
