#!/usr/bin/env python3
"""
Quick comparison of smoothed poses against original.

Usage:
    python quick_compare.py annotations/ch_circlexy [--output compare.png]
"""
from __future__ import annotations

import sys
from pathlib import Path
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_compare.py <annotation_dir> [--output IMG]")
        print("  e.g.: python quick_compare.py annotations/ch_circlexy")
        return 1

    seq_dir = Path(sys.argv[1]).resolve()
    output = None
    if "--output" in sys.argv:
        output = sys.argv[sys.argv.index("--output") + 1]

    poses_orig = seq_dir / "poses.json"
    poses_smooth = seq_dir / "poses_smooth.json"

    if not poses_orig.exists():
        print(f"Error: {poses_orig} not found")
        return 1
    if not poses_smooth.exists():
        print(f"Error: {poses_smooth} not found")
        print("Have you run smoothing yet? Try:")
        print(f"  python smooth_poses_se3.py --ann-dir {seq_dir.parent} --bag {seq_dir.name}")
        return 1

    print(f"Comparing {seq_dir.name}...")
    print(f"  Original: {poses_orig}")
    print(f"  Smoothed: {poses_smooth}")
    print(f"  Plot:     {output or 'interactive display'}")
    print()

    # compare_poses.py takes --ann-dir / --bag (not positional pose files)
    compare_script = Path(__file__).parent / "compare_poses.py"
    cmd = [
        sys.executable, str(compare_script),
        "--ann-dir", str(seq_dir.parent),
        "--bag", seq_dir.name,
    ]
    if output:
        cmd += ["--output", output]
    result = subprocess.run(cmd)

    if result.returncode == 0 and output:
        print()
        print(f"\u2713 Comparison saved to: {output}")
        print("  Open in image viewer to inspect")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
