#!/usr/bin/env python
"""Train FastGaMer (Sec. 4 protocol).

Adam(beta1=0.9, beta2=0.99), lr=2e-4 with MultiStepLR x0.5 at
{200k,400k,600k,800k} and no warm-up, 256x256 random crops with flips/rotations,
batch size 16. Objective is Eq. 5 (L1 on normalized + scaled GM + LUT
smoothness/monotonicity).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastgamer.models import build_model
from fastgamer.losses import FastGaMerLoss
from fastgamer.data import GainMapDataset
from fastgamer import utils


def build_loaders(cfg, batch_size):
    d = cfg["data"]
    train_set = GainMapDataset(d["root"], "train", d["patch_size"], d["thumb_size"],
                               d["mode"], d.get("augment", True), is_train=True,
                               seed=cfg["train"]["seed"])
    drop_last = len(train_set) >= batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=d.get("num_workers", 4),
                              drop_last=drop_last, pin_memory=True)
    return train_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fastgamer.yaml")
    ap.add_argument("--max_iter", type=int, default=None, help="override niter (smoke test)")
    ap.add_argument("--batch", type=int, default=None, help="override batch_size")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = utils.load_config(args.config)
    utils.set_seed(cfg["train"]["seed"])
    device = args.device if torch.cuda.is_available() else "cpu"

    exp_dir = os.path.join(cfg["path"]["experiments_root"], cfg["name"])
    os.makedirs(exp_dir, exist_ok=True)

    model = build_model(cfg).to(device)
    print(f"Model parameters: {utils.count_parameters(model)/1e6:.4f} M")
    criterion = FastGaMerLoss(cfg["train"]["lambda1"], cfg["train"]["lambda2"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                                 betas=tuple(cfg["train"]["betas"]),
                                 weight_decay=cfg["train"].get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["train"]["lr_steps"], gamma=cfg["train"]["lr_gamma"])

    start_step = 0
    if cfg["path"].get("resume"):
        start_step = utils.load_checkpoint(cfg["path"]["resume"], model, optimizer,
                                           scheduler, map_location=device)
    elif cfg["path"].get("pretrain"):
        utils.load_checkpoint(cfg["path"]["pretrain"], model, map_location=device)

    batch_size = args.batch or cfg["train"]["batch_size"]
    loader = build_loaders(cfg, batch_size)
    niter = args.max_iter or cfg["train"]["niter"]
    print(f"Training for {niter} iterations on {device} "
          f"(dataset: {len(loader.dataset)} pairs, batch {batch_size}).")

    model.train()
    step = start_step
    t0 = time.time()
    data_iter = iter(loader)
    while step < niter:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        i_hr = batch["i_hr"].to(device, non_blocking=True)
        i_thumb = batch["i_thumb"].to(device, non_blocking=True)
        target = {"gm_norm": batch["gm_norm"].to(device),
                  "gm_pred": batch["gm_pred"].to(device)}

        pred = model(i_hr, i_thumb)
        losses = criterion(pred, target)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg["train"]["print_freq"] == 0 or step == 1:
            ips = cfg["train"]["print_freq"] / max(time.time() - t0, 1e-6)
            t0 = time.time()
            print(f"iter {step:>7d} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"total {losses['total'].item():.4f} | norm {losses['l_norm']:.4f} | "
                  f"scale {losses['l_scale']:.4f} | smooth {losses['l_smooth']:.4f} | "
                  f"mono {losses['l_mono']:.4f} | {ips:.1f} it/s")

        if step % cfg["train"]["save_freq"] == 0 or step == niter:
            ckpt = os.path.join(exp_dir, f"model_{step}.pth")
            utils.save_checkpoint(ckpt, model, optimizer, scheduler, step)
            utils.save_checkpoint(os.path.join(exp_dir, "model_latest.pth"),
                                  model, optimizer, scheduler, step)
            print(f"  saved {ckpt}")

    print("Training done.")


if __name__ == "__main__":
    main()
