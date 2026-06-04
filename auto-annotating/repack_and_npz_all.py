#!/usr/bin/env python3
"""Repack all annotated bags as `topics` and generate NPZs from them.

Walks the annotation dir, finds bags that have annotations, optionally filters to
only those that are "complete enough", repacks each as topics, then runs the NPZ
exporter on the repacked bags.

Safe to re-run: skips a stage if its output already exists unless --force.

Usage:
  python repack_and_npz_all.py \
      --ann-dir /path/to/annotations \
      --bags-root /path/to/original_bags \
      --out-dir /path/to/annotated_out \
      --npz-dir /path/to/npz_out \
      --scripts-dir /path/to/scripts          # where repack_bag.py / generate_npz live
      [--min-frames 50] [--include-partial] [--force]

Notes on partial annotations:
  A bag is considered annotated if it has keypoints.json AND a masks/ dir with at
  least one mask. "Ready frames" = frames having BOTH a mask and keypoints. By
  default a bag is repacked only if it has >= --min-frames ready frames (so you
  don't silently train on half-finished bags). Pass --include-partial to repack
  every annotated bag regardless of how few frames are ready.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def ready_frame_count(ann_dir: Path, stem: str) -> tuple[int, int]:
    """Return (ready_frames, total_keypoint_frames) for a bag.
    ready = frames present in BOTH keypoints.json and the masks dir."""
    kp_path = ann_dir / stem / "keypoints.json"
    masks_dir = ann_dir / stem / "masks"
    if not kp_path.exists():
        return (0, 0)
    try:
        frames = json.loads(kp_path.read_text()).get("frames", {})
    except Exception:
        return (0, 0)
    kp_idx = set()
    for k in frames:
        try:
            kp_idx.add(int(k))
        except ValueError:
            pass
    mask_idx = set()
    if masks_dir.exists():
        for p in masks_dir.glob("frame_*_needle_mask.png"):
            parts = p.stem.split("_")
            for tok in parts:
                if tok.isdigit():
                    mask_idx.add(int(tok)); break
    return (len(kp_idx & mask_idx), len(kp_idx))


def find_original_bag(bags_root: Path, stem: str) -> Path | None:
    """Find the original bag directory matching an annotation stem."""
    cand = bags_root / stem
    if cand.is_dir():
        return cand
    # fall back to a recursive search by name
    for p in bags_root.rglob(stem):
        if p.is_dir():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--bags-root", required=True,
                    help="Directory containing the ORIGINAL bag folders.")
    ap.add_argument("--out-dir", required=True,
                    help="Where annotated (repacked) bags are written.")
    ap.add_argument("--npz-dir", required=True,
                    help="Where NPZ files are written.")
    ap.add_argument("--scripts-dir", default=".",
                    help="Dir containing repack_bag.py and generate_npz_from_topics.py")
    ap.add_argument("--min-frames", type=int, default=50,
                    help="Skip bags with fewer ready frames (ignored if --include-partial).")
    ap.add_argument("--include-partial", action="store_true",
                    help="Repack every annotated bag regardless of ready-frame count.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages even if outputs already exist.")
    args = ap.parse_args()

    ann_dir = Path(args.ann_dir).resolve()
    bags_root = Path(args.bags_root).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = Path(args.npz_dir).resolve(); npz_dir.mkdir(parents=True, exist_ok=True)
    scripts = Path(args.scripts_dir).resolve()
    repack_py = scripts / "repack_bag.py"
    npz_py = scripts / "generate_npz_from_topics.py"
    for f in (repack_py, npz_py):
        if not f.exists():
            sys.exit(f"missing script: {f}")

    # discover annotated bags (have a keypoints.json)
    stems = sorted(d.name for d in ann_dir.iterdir()
                   if d.is_dir() and (d / "keypoints.json").exists())
    if not stems:
        sys.exit(f"no annotated bags (keypoints.json) found under {ann_dir}")

    print(f"Found {len(stems)} annotated bag(s) under {ann_dir}\n")
    repacked, skipped, failed = [], [], []

    for stem in stems:
        ready, total = ready_frame_count(ann_dir, stem)
        tag = f"{stem}: {ready} ready / {total} keypoint frames"
        if not args.include_partial and ready < args.min_frames:
            print(f"  SKIP  {tag}  (< {args.min_frames} ready; use --include-partial to force)")
            skipped.append(stem); continue

        orig = find_original_bag(bags_root, stem)
        if orig is None:
            print(f"  SKIP  {tag}  (no original bag dir found under {bags_root})")
            skipped.append(stem); continue

        annotated_out = out_dir / f"{stem}_annotated_snapshot"
        if annotated_out.exists() and not args.force:
            print(f"  HAVE  {tag}  (repacked already; --force to redo)")
        else:
            print(f"  PACK  {tag}")
            cmd = [sys.executable, str(repack_py),
                   "--bag", str(orig), "--ann-dir", str(ann_dir),
                   "--out-dir", str(out_dir), "--output-mode", "snapshot",
                   "--poses", "auto"]
            r = subprocess.run(cmd, cwd=str(scripts))
            if r.returncode != 0:
                print(f"        repack FAILED for {stem}")
                failed.append(stem); continue
        repacked.append((stem, annotated_out))

    # # NPZ generation from the repacked topic bags
    # print("\n--- generating NPZs ---")
    # for stem, annotated_out in repacked:
    #     npz_out = npz_dir / f"{stem}.npz"
    #     if npz_out.exists() and not args.force:
    #         print(f"  HAVE  {stem}.npz")
    #         continue
    #     # the repacked dir contains the .mcap; point the exporter at the dir
    #     cmd = [sys.executable, str(npz_py),
    #            "--input", str(annotated_out), "--out-dir", str(npz_dir)]
    #     r = subprocess.run(cmd, cwd=str(scripts))
    #     if r.returncode != 0:
    #         print(f"  NPZ FAILED for {stem}")
    #         failed.append(stem)

    print("\n=== summary ===")
    print(f"  repacked: {len(repacked)}")
    print(f"  skipped:  {len(skipped)}  {skipped if skipped else ''}")
    print(f"  failed:   {len(failed)}  {failed if failed else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
