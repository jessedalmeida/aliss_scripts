#!/usr/bin/env python3
"""
Batch smooth all checkerboard poses in annotation directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from temporal_interpolation import smooth_poses_se3, save_se3_smoothed_poses


def main():
    parser = argparse.ArgumentParser(
        description="Batch smooth all checkerboard poses"
    )
    parser.add_argument(
        "ann_dir",
        type=Path,
        help="Annotation directory (containing ch_* subdirs)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=2.0,
        help="Process noise scale (default: 2.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done, don't actually smooth",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite original poses.json (after backup)",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix for backup files (default: .bak)",
    )
    
    args = parser.parse_args()
    
    ann_dir = args.ann_dir.resolve()
    if not ann_dir.exists():
        print(f"Error: {ann_dir} not found")
        return 1
    
    # Find all ch_* sequence directories
    seq_dirs = sorted([d for d in ann_dir.iterdir() if d.is_dir() and d.name.startswith("ch_")])
    
    if not seq_dirs:
        print(f"No ch_* sequences found in {ann_dir}")
        return 1
    
    print(f"Found {len(seq_dirs)} sequences to smooth:")
    for d in seq_dirs:
        print(f"  - {d.name}")
    print()
    
    results_summary = {}
    
    for seq_dir in seq_dirs:
        poses_json = seq_dir / "poses.json"
        if not poses_json.exists():
            print(f"⊘ {seq_dir.name}: poses.json not found, skipping")
            continue
        
        output_json = seq_dir / "poses_smooth.json"
        
        print(f"→ {seq_dir.name}...", end=" ", flush=True)
        
        if args.dry_run:
            print("(DRY RUN - would smooth)")
            continue
        
        try:
            # Smooth
            results = smooth_poses_se3(
                poses_json,
                process_noise_scale=args.noise_scale,
                only_ok_frames=True,
            )
            
            # Save
            save_se3_smoothed_poses(poses_json, results, output_json)
            
            # Statistics
            residuals = [r["residual_norm"] for r in results.values()]
            mean_res = sum(residuals) / len(residuals) if residuals else 0
            max_res = max(residuals) if residuals else 0
            
            print(f"✓ ({len(results)} frames, mean residual: {mean_res:.4f} m)")
            
            results_summary[seq_dir.name] = {
                "frames_smoothed": len(results),
                "mean_residual": mean_res,
                "max_residual": max_res,
            }
            
            # Apply to original if requested
            if args.inplace:
                backup = poses_json.with_suffix(poses_json.suffix + args.backup_suffix)
                poses_json.rename(backup)
                output_json.rename(poses_json)
                print(f"  → Backed up to {backup.name}, replaced original")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            return 1
    
    if not args.dry_run:
        print()
        print("=== Summary ===")
        total_frames = sum(r["frames_smoothed"] for r in results_summary.values())
        print(f"Total frames smoothed: {total_frames}")
        print(f"\nResiduals by sequence:")
        for seq, stats in results_summary.items():
            print(f"  {seq}: mean={stats['mean_residual']:.5f}m, "
                  f"max={stats['max_residual']:.5f}m")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
