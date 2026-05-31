#!/usr/bin/env python3
"""
Smooth checkerboard pose detections using SE(3) manifold-based approach.

This applies principled covariance-weighted smoothing that respects measurement
uncertainty and follows Lie group geometry for rotation manifolds.

Usage:
    python smooth_poses_se3.py --ann-dir ./annotations --bag ch_circlexy --noise-scale 1.0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from temporal_interpolation import smooth_poses_se3, save_se3_smoothed_poses


def main():
    parser = argparse.ArgumentParser(
        description="Smooth checkerboard poses using SE(3) manifold smoothing"
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        required=True,
        help="Path to annotations directory containing bag subdirectories",
    )
    parser.add_argument(
        "--bag",
        type=str,
        required=True,
        help="Bag folder name under the annotation directory",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=100.0,
        help="Process noise scale (higher = smoother). Default: 1.0. Try 0.1-10.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Default: poses_smooth.json in the bag directory.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Launch compare_poses.py on the original and smoothed output after smoothing.",
    )
    parser.add_argument(
        "--only-ok",
        action="store_true",
        default=True,
        help="Only smooth frames with status='ok' (default: True)",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="Smooth all frames with valid poses, not just status='ok'",
    )
    
    args = parser.parse_args()
    
    bag_dir = args.ann_dir.resolve() / Path(args.bag)
    poses_json_path = bag_dir / "poses.json"
    if not bag_dir.exists():
        print(f"Error: bag directory not found: {bag_dir}")
        return 1
    if not poses_json_path.exists():
        print(f"Error: poses.json not found at {poses_json_path}")
        return 1
    
    output_path = args.output or bag_dir / "poses_smooth.json"
    output_path = output_path.resolve()
    
    only_ok_frames = not args.all_frames
    
    print(f"Loading poses from: {poses_json_path}")
    print(f"Process noise scale: {args.noise_scale}")
    print(f"Smoothing mode: {'ok frames only' if only_ok_frames else 'all frames with valid poses'}")
    print()
    
    # Apply smoothing
    print("Running SE(3) manifold smoothing...")
    results = smooth_poses_se3(
        poses_json_path,
        process_noise_scale=args.noise_scale,
        only_ok_frames=only_ok_frames,
    )
    
    if not results:
        print("Error: No poses smoothed. Check input file and frame status.")
        return 1
    
    print(f"Successfully smoothed {len(results)} frames")
    
    # Compute statistics
    residuals = [r["residual_norm"] for r in results.values()]
    print(f"\nSmoothing residuals (log-manifold distance):")
    print(f"  Mean: {sum(residuals) / len(residuals):.6f}")
    print(f"  Max:  {max(residuals):.6f}")
    print(f"  Min:  {min(residuals):.6f}")
    
    # Save results
    print(f"\nSaving smoothed poses to: {output_path}")
    save_se3_smoothed_poses(poses_json_path, results, output_path)
    
    print(f"✓ Done!")
    print()
    print("Next steps:")
    print(f"1. Review the smoothed poses: {output_path}")
    print(f"2. If satisfied, backup original and replace:")
    print(f"   cp {poses_json_path} {poses_json_path}.bak")
    print(f"   cp {output_path} {poses_json_path}")
    print(f"3. If you want more/less smoothing, try different --noise-scale values")
    
    if args.compare:
        compare_script = Path(__file__).parent / "compare_poses.py"
        print()
        print("Launching compare_poses.py to inspect smoothing results...")
        result = subprocess.run([
            sys.executable,
            str(compare_script),
            "--ann-dir",
            str(args.ann_dir.resolve()),
            "--bag",
            args.bag,
        ])
        if result.returncode != 0:
            return result.returncode
    
    return 0


if __name__ == "__main__":
    exit(main())
