#!/usr/bin/env python3
"""
Quick comparison of smoothed poses against original.

Usage:
    python quick_compare.py annotations/ch_circlexy
"""
from __future__ import annotations

import sys
from pathlib import Path
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_compare.py <annotation_dir>")
        print("  e.g.: python quick_compare.py annotations/ch_circlexy")
        return 1
    
    seq_dir = Path(sys.argv[1]).resolve()
    poses_orig = seq_dir / "poses.json"
    poses_smooth = seq_dir / "poses_smooth.json"
    
    if not poses_orig.exists():
        print(f"Error: {poses_orig} not found")
        return 1
    
    if not poses_smooth.exists():
        print(f"Error: {poses_smooth} not found")
        print(f"Have you run smoothing yet? Try:")
        print(f"  python smooth_poses_se3.py {seq_dir}/poses.json")
        return 1
    
    print(f"Comparing {seq_dir.name}...")
    print(f"  Original: {poses_orig}")
    print(f"  Smoothed: {poses_smooth}")
    print(f"  Plot:     interactive display")
    print()
    
    # Run comparison (use full path to compare_poses.py in same directory)
    compare_script = Path(__file__).parent / "compare_poses.py"
    result = subprocess.run([
        sys.executable, str(compare_script),
        str(poses_orig),
        str(poses_smooth),
    ])
    
    if result.returncode == 0:
        print()
        print(f"✓ Comparison saved to: {output_img}")
        print(f"  Open in image viewer to inspect")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
