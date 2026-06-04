#!/usr/bin/env python3
"""Verify train.py mechanics on CPU: no NaNs, val metrics, checkpoint round-trip."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

import train as T
from needle_model import NeedleNet
from needle_loss import NeedleLoss

UP = Path("/mnt/user-data/uploads")
WORK = Path("/home/claude/step8")
WORK.mkdir(exist_ok=True)


def build_tiny():
    """2-bag manifest from real uploaded frames; bagA=board (train), bagB=no-board (val)."""
    kp = json.load(open(UP / "keypoints.json"))["frames"]
    def rec(idx, bag, board):
        f = kp[f"{idx:06d}"]
        def e(n):
            xy = f.get(n)
            return {"xy": [float(xy[0]), float(xy[1])] if xy else None,
                    "visible": xy is not None}
        return {"bag": bag, "frame": idx,
                "image": str(UP / f"{idx:06d}.jpg"),
                "mask": str(UP / f"frame_{idx:06d}_needle_mask.png"),
                "has_board": board, "board_detected": False, "pose_status": "ok",
                "keypoints": {n: e(n) for n in
                              ("needle_tip", "needle_tail", "left_arm_tip", "right_arm_tip")}}
    recs = [rec(62, "bagA", True), rec(63, "bagA", True),
            rec(64, "bagB", False), rec(65, "bagB", False)]
    man = WORK / "manifest.jsonl"
    man.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    splits = {"manifest": str(man), "seed": 0,
              "splits": {"train": ["bagA"], "val": ["bagB"], "test": []}}
    sp = WORK / "splits.json"
    sp.write_text(json.dumps(splits))
    return man, sp


def main():
    man, sp = build_tiny()
    args = SimpleNamespace(
        manifest=str(man), splits=str(sp), out=str(WORK / "run"),
        epochs=2, batch_size=2, lr=1e-3, weight_decay=1e-4, input_size=256,
        target_noboard_frac=0.5, rot_deg=180.0, fov_aware=True,
        num_workers=0, pretrained=False, device="cpu", seed=0,
    )

    torch.manual_seed(0)
    train_loader, val_loader = T.build_loaders(args)
    model = NeedleNet(2, pretrained=False)
    crit = NeedleLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    for ep in range(2):
        tr = T.train_one_epoch(model, train_loader, crit, opt, scaler, "cpu", use_amp=False)
        assert all(math.isfinite(v) for v in tr.values()), f"NaN/inf in train losses: {tr}"
        print(f"ep{ep} train: total={tr['total']:.4f} mask={tr['mask']:.4f} "
              f"hm={tr['hm']:.4f} vis={tr['vis']:.4f}")

    val = T.validate(model, val_loader, crit, "cpu")
    for k, v in val.items():
        assert math.isfinite(v), f"non-finite val metric {k}={v}"
    print("val metrics:", {k: round(v, 4) for k, v in val.items()})
    # sanity ranges
    assert 0 <= val["val_dice"] <= 1 and 0 <= val["val_iou"] <= 1
    assert 0 <= val["val_vis_acc"] <= 1
    assert val["val_kp_err_px"] >= 0

    # --- checkpoint round-trip: reload -> identical val metrics ---
    ckpt = WORK / "ck.pt"
    T.save_checkpoint(ckpt, model, opt, epoch=1, best=val["val_dice"])
    model2 = NeedleNet(2, pretrained=False)
    sd = torch.load(ckpt)["model"]
    model2.load_state_dict(sd)
    val2 = T.validate(model2, val_loader, crit, "cpu")
    for k in val:
        assert abs(val[k] - val2[k]) < 1e-6, f"metric {k} not reproduced: {val[k]} vs {val2[k]}"
    print("checkpoint reload reproduces val metrics exactly: OK")
    print("All Step-8 checks passed.")


if __name__ == "__main__":
    main()
