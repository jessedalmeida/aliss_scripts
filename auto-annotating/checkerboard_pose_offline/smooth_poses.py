#!/usr/bin/env python3
"""
Smooth noisy checkerboard pose detections using covariance-weighted Rauch-Tung-Striebel filter.

This script smooths successful pose detections while respecting their measurement uncertainty.
Higher detection uncertainty (higher covariance) allows more smoothing. Lower uncertainty
keeps the pose closer to the measured value.

Usage:
    # Smooth a single bag
    python smooth_poses.py --ann-dir ./annotations --bag ch_circlexy --smooth-strength 1.0
    
    # Smooth all bags with stronger smoothing (10x more smoothing)
    python smooth_poses.py --ann-dir ./annotations --all --smooth-strength 10.0
    
    # Use the output
    cp annotations/ch_circlexy/poses_smoothed.json annotations/ch_circlexy/poses.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from temporal_interpolation import smooth_poses_rts, save_smoothed_poses


def smooth_bag(
    bag_dir: Path,
    smooth_strength: float = 1.0,
    output_suffix: str = "_smoothed",
) -> tuple[bool, str]:
    """
    Smooth poses for a single bag.
    
    Args:
        bag_dir: Path to annotation directory (e.g., annotations/ch_circlexy)
        smooth_strength: Process noise scale (0.1-10). Higher = smoother.
        output_suffix: Suffix for output file (poses{suffix}.json)
    
    Returns:
        (success, message)
    """
    poses_json = bag_dir / "poses.json"
    output_json = bag_dir / f"poses{output_suffix}.json"
    
    if not poses_json.exists():
        return False, f"poses.json not found: {poses_json}"
    
    try:
        print(f"  Smoothing {bag_dir.name}...")
        results = smooth_poses_rts(poses_json, process_noise_scale=smooth_strength)
        
        if not results:
            return False, f"No frames to smooth in {bag_dir.name}"
        
        save_smoothed_poses(poses_json, results, output_json)
        
        # Count stats
        num_frames = len(results)
        avg_error = sum(r["original_position_error"] for r in results.values()) / num_frames
        
        msg = f"  ✓ Smoothed {num_frames} frames. Avg position deviation: {avg_error*1000:.2f}mm"
        msg += f"\n    Output: {output_json.name}"
        return True, msg
    
    except Exception as e:
        return False, f"Error smoothing {bag_dir.name}: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Smooth noisy checkerboard pose detections using RTS filter."
    )
    parser.add_argument(
        "--ann-dir",
        type=Path,
        required=True,
        help="Path to annotations directory (contains bag subdirectories)",
    )
    parser.add_argument(
        "--bag",
        type=str,
        help="Smooth only this bag (e.g., ch_circlexy)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Smooth all bags found in ann-dir",
    )
    parser.add_argument(
        "--smooth-strength",
        type=float,
        default=1.0,
        help="Process noise scale. Higher = smoother. Default 1.0 (conservative).\n"
             "Try 0.1-0.5 for slight smoothing, 1.0-5.0 for moderate, 10+ for aggressive.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_smoothed",
        help="Suffix for output poses file (default: _smoothed -> poses_smoothed.json)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original poses.json (careful!). Default creates poses_smoothed.json.",
    )
    
    args = parser.parse_args()
    
    if args.in_place:
        args.output_suffix = ""
    
    # Collect bags to process
    bags = []
    if args.bag:
        bag_dir = args.ann_dir / args.bag
        if not bag_dir.exists():
            print(f"ERROR: Bag directory not found: {bag_dir}")
            return 1
        bags = [bag_dir]
    elif args.all:
        bags = sorted([
            p for p in args.ann_dir.iterdir() 
            if p.is_dir() and (p / "poses.json").exists()
        ])
    else:
        print("ERROR: Specify --bag or --all")
        return 1
    
    if not bags:
        print("ERROR: No bags found with poses.json")
        return 1
    
    print(f"\nSmoothing {len(bags)} bag(s) with strength={args.smooth_strength}")
    print(f"Output files: poses{args.output_suffix}.json\n")
    
    successes = 0
    for bag_dir in bags:
        success, message = smooth_bag(bag_dir, args.smooth_strength, args.output_suffix)
        print(message)
        if success:
            successes += 1
    
    print(f"\n✓ Successfully smoothed {successes}/{len(bags)} bags")
    print("\nTo use the smoothed poses:")
    if args.in_place:
        print("  Poses have been overwritten in-place.")
    else:
        print(f"  cp annotations/<bag>/poses{args.output_suffix}.json annotations/<bag>/poses.json")
    
    return 0


if __name__ == "__main__":
    exit(main())
