#!/usr/bin/env python3
"""train.py - Training loop for the needle mask + keypoint model.

Train split uses augmentation (Step 4) + the domain-balanced sampler (Step 5);
val split is plain and sequential. AMP/half precision on CUDA, cosine LR,
per-term loss logging, per-epoch validation, and checkpoints that reload to
identical metrics.

Usage:
    python train.py --manifest dataset/manifest.jsonl --splits dataset/splits.json \
        --out runs/exp1 --epochs 50 --batch-size 8 --target-noboard-frac 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from needle_dataset import NeedleDataset, NATIVE
from needle_augment import NeedleAugmentation
from needle_sampler import build_sampler, load_train_records
from needle_model import NeedleNet
from needle_loss import NeedleLoss


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def build_loaders(args):
    aug = NeedleAugmentation(
        rot_deg=args.rot_deg, fov_aware=args.fov_aware, seed=args.seed,
    )
    train_ds = NeedleDataset(args.manifest, args.splits, split="train",
                             input_size=args.input_size, transform=aug)
    val_ds = NeedleDataset(args.manifest, args.splits, split="val",
                           input_size=args.input_size, transform=None)

    train_recs = load_train_records(args.manifest, args.splits)
    sampler = build_sampler(train_recs, target_noboard_frac=args.target_noboard_frac,
                            num_samples=len(train_recs), seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# validation metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate(model, loader, crit, device, heatmap_stride: int = 4) -> dict:
    model.eval()
    agg = {"loss": 0.0, "mask": 0.0, "hm": 0.0, "vis": 0.0,
           "dice": 0.0, "iou": 0.0, "n": 0}
    kp_err_sum, kp_n = 0.0, 0
    vis_correct, vis_n = 0, 0

    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        out = model(batch["image"])
        losses = crit(out, batch)
        bs = batch["image"].shape[0]
        agg["loss"] += losses["total"].item() * bs
        for t in ("mask", "hm", "vis"):
            agg[t] += losses[t].item() * bs

        # mask dice / iou at 0.5
        prob = torch.sigmoid(out["mask_logits"]) > 0.5
        tgt = batch["mask"] > 0.5
        inter = (prob & tgt).sum((1, 2, 3)).float()
        union = (prob | tgt).sum((1, 2, 3)).float()
        psum = prob.sum((1, 2, 3)).float() + tgt.sum((1, 2, 3)).float()
        agg["dice"] += ((2 * inter + 1e-6) / (psum + 1e-6)).sum().item()
        agg["iou"] += ((inter + 1e-6) / (union + 1e-6)).sum().item()
        agg["n"] += bs

        # keypoint pixel error in NATIVE coords, visible points only
        pred = model.soft_argmax(out["heatmaps"])             # (B,K,2) heatmap units
        B, K, H, W = out["heatmaps"].shape
        tgt_flat = batch["heatmaps"].reshape(B, K, -1).argmax(-1)
        gt = torch.stack([tgt_flat % W, tgt_flat // W], -1).float()  # (B,K,2)
        to_native = heatmap_stride * (NATIVE / batch["image"].shape[-1])
        err = ((pred - gt) * to_native).pow(2).sum(-1).sqrt()  # (B,K)
        m = batch["hm_mask"] > 0
        kp_err_sum += err[m].sum().item()
        kp_n += int(m.sum().item())

        # visibility accuracy on labeled points
        pred_vis = torch.sigmoid(out["vis_logits"]) > 0.5
        tgt_vis = batch["vis_target"] > 0.5
        vm = batch["vis_mask"] > 0
        vis_correct += int((pred_vis == tgt_vis)[vm].sum().item())
        vis_n += int(vm.sum().item())

    n = max(agg["n"], 1)
    return {
        "val_loss": agg["loss"] / n,
        "val_mask": agg["mask"] / n, "val_hm": agg["hm"] / n, "val_vis": agg["vis"] / n,
        "val_dice": agg["dice"] / n, "val_iou": agg["iou"] / n,
        "val_kp_err_px": kp_err_sum / max(kp_n, 1),
        "val_vis_acc": vis_correct / max(vis_n, 1),
    }


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, crit, opt, scaler, device, use_amp) -> dict:
    model.train()
    agg = {"total": 0.0, "mask": 0.0, "hm": 0.0, "vis": 0.0, "n": 0}
    for batch in loader:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(device)
        opt.zero_grad()
        with torch.autocast(device_type=device.split(":")[0], enabled=use_amp):
            out = model(batch["image"])
            losses = crit(out, batch)
        scaler.scale(losses["total"]).backward()
        scaler.step(opt)
        scaler.update()
        bs = batch["image"].shape[0]
        for t in ("total", "mask", "hm", "vis"):
            agg[t] += losses[t].item() * bs
        agg["n"] += bs
    n = max(agg["n"], 1)
    return {k: agg[k] / n for k in ("total", "mask", "hm", "vis")}


def save_checkpoint(path, model, opt, epoch, best):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "best": best}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--target-noboard-frac", type=float, default=0.25)
    ap.add_argument("--rot-deg", type=float, default=180.0)
    ap.add_argument("--fov-aware", action="store_true", default=True)
    ap.add_argument("--no-fov-aware", dest="fov_aware", action="store_false")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    use_amp = device.startswith("cuda")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_loaders(args)
    model = NeedleNet(num_keypoints=2, pretrained=args.pretrained).to(device)
    crit = NeedleLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log = []
    best = -1.0
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, crit, opt, scaler, device, use_amp)
        val = validate(model, val_loader, crit, device, heatmap_stride=4)
        sched.step()
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], **tr, **val}
        log.append(row)
        print(f"ep {epoch:3d} | train {tr['total']:.4f} (mask {tr['mask']:.3f} "
              f"hm {tr['hm']:.4f} vis {tr['vis']:.4f}) | val dice {val['val_dice']:.3f} "
              f"kp_err {val['val_kp_err_px']:.1f}px vis_acc {val['val_vis_acc']:.3f}")
        save_checkpoint(out_dir / "last.pt", model, opt, epoch, best)
        if val["val_dice"] > best:
            best = val["val_dice"]
            save_checkpoint(out_dir / "best.pt", model, opt, epoch, best)
        (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    print(f"done. best val dice={best:.3f}  checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
