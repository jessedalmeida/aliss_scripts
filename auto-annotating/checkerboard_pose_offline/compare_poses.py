#!/usr/bin/env python3
"""
Compare original vs smoothed poses visually and statistically.

Plots trajectories and shows frame-by-frame deviations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def load_poses_json(path: Path) -> dict:
    """Load poses.json file."""
    with open(path, "r") as fh:
        return json.load(fh)


def extract_trajectory(frames: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract position trajectory from frames dict.
    
    Returns:
        (frame_indices, positions) where positions is (N, 3) array
    """
    frame_keys_sorted = sorted(
        [k for k in frames.keys() if k.isdigit()],
        key=lambda x: int(x)
    )
    
    positions = []
    indices = []
    
    for frame_key in frame_keys_sorted:
        frame_data = frames[frame_key]
        pose = frame_data.get("pose")
        if not pose or not pose.get("position"):
            continue
        
        pos = pose.get("position")
        positions.append(pos)
        indices.append(int(frame_key))
    
    return np.array(indices), np.array(positions)


def compute_trajectory_stats(pos_orig: np.ndarray, pos_smooth: np.ndarray) -> dict:
    """Compute statistics of smoothing."""
    diffs = pos_smooth - pos_orig
    distances = np.linalg.norm(diffs, axis=1)
    
    return {
        "mean_shift_m": float(np.mean(distances)),
        "median_shift_m": float(np.median(distances)),
        "max_shift_m": float(np.max(distances)),
        "std_shift_m": float(np.std(distances)),
        "total_frames": len(distances),
    }


def plot_comparison(
    frames_orig: dict,
    frames_smooth: dict,
    output_path: Path | None = None,
) -> None:
    """
    Create comparison plots.
    
    Args:
        frames_orig: Original frames dict
        frames_smooth: Smoothed frames dict
        output_path: Where to save the figure (None = display)
    """
    idx_orig, pos_orig = extract_trajectory(frames_orig)
    idx_smooth, pos_smooth = extract_trajectory(frames_smooth)
    
    # Match them up (should be same frames)
    common_indices = sorted(set(idx_orig) & set(idx_smooth))
    common_idx = np.array(common_indices)
    
    # Get positions for common frames
    pos_orig_matched = np.array([pos_orig[list(idx_orig).index(i)] for i in common_indices])
    pos_smooth_matched = np.array([pos_smooth[list(idx_smooth).index(i)] for i in common_indices])
    
    # Compute stats
    stats = compute_trajectory_stats(pos_orig_matched, pos_smooth_matched)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3D trajectory comparison
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax3d.plot(pos_orig_matched[:, 0], pos_orig_matched[:, 1], pos_orig_matched[:, 2],
              'b-', alpha=0.6, linewidth=2, label="Original")
    ax3d.plot(pos_smooth_matched[:, 0], pos_smooth_matched[:, 1], pos_smooth_matched[:, 2],
              'r-', alpha=0.6, linewidth=2, label="Smoothed")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Position Trajectory")
    ax3d.legend()
    ax3d.grid(True, alpha=0.3)
    
    # XY view
    ax_xy = fig.add_subplot(2, 3, 2)
    ax_xy.plot(pos_orig_matched[:, 0], pos_orig_matched[:, 1],
               'b-', alpha=0.6, linewidth=2, label="Original")
    ax_xy.plot(pos_smooth_matched[:, 0], pos_smooth_matched[:, 1],
               'r-', alpha=0.6, linewidth=2, label="Smoothed")
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY Projection")
    ax_xy.legend()
    ax_xy.grid(True, alpha=0.3)
    ax_xy.axis("equal")
    
    # XZ view
    ax_xz = fig.add_subplot(2, 3, 3)
    ax_xz.plot(pos_orig_matched[:, 0], pos_orig_matched[:, 2],
               'b-', alpha=0.6, linewidth=2, label="Original")
    ax_xz.plot(pos_smooth_matched[:, 0], pos_smooth_matched[:, 2],
               'r-', alpha=0.6, linewidth=2, label="Smoothed")
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("XZ Projection")
    ax_xz.legend()
    ax_xz.grid(True, alpha=0.3)
    ax_xz.axis("equal")
    
    # Frame-by-frame deviation
    diffs = pos_smooth_matched - pos_orig_matched
    distances = np.linalg.norm(diffs, axis=1)
    
    ax_dev = fig.add_subplot(2, 3, 4)
    ax_dev.plot(common_idx, distances * 1000, 'g-', linewidth=1.5, alpha=0.8)
    ax_dev.fill_between(common_idx, distances * 1000, alpha=0.3, color="green")
    ax_dev.set_xlabel("Frame Index")
    ax_dev.set_ylabel("Deviation (mm)")
    ax_dev.set_title("Frame-by-Frame Smoothing Magnitude")
    ax_dev.grid(True, alpha=0.3)
    
    # XYZ component deviations
    ax_comp = fig.add_subplot(2, 3, 5)
    ax_comp.plot(common_idx, np.abs(diffs[:, 0]) * 1000, label="ΔX", linewidth=1.5, alpha=0.7)
    ax_comp.plot(common_idx, np.abs(diffs[:, 1]) * 1000, label="ΔY", linewidth=1.5, alpha=0.7)
    ax_comp.plot(common_idx, np.abs(diffs[:, 2]) * 1000, label="ΔZ", linewidth=1.5, alpha=0.7)
    ax_comp.set_xlabel("Frame Index")
    ax_comp.set_ylabel("Abs Deviation (mm)")
    ax_comp.set_title("Component-wise Deviations")
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3)
    
    # Statistics text
    ax_stats = fig.add_subplot(2, 3, 6)
    ax_stats.axis("off")
    stats_text = f"""
Smoothing Statistics
{'='*40}

Total Frames: {stats['total_frames']}

Mean Shift:    {stats['mean_shift_m']*1000:.3f} mm
Median Shift:  {stats['median_shift_m']*1000:.3f} mm
Max Shift:     {stats['max_shift_m']*1000:.3f} mm
Std Dev:       {stats['std_shift_m']*1000:.3f} mm

Distance Range: {stats['mean_shift_m']*1000:.1f}–{stats['max_shift_m']*1000:.1f} mm
    """
    ax_stats.text(0.1, 0.5, stats_text, family="monospace", fontsize=11,
                  verticalalignment="center")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        print(f"✓ Comparison plot saved: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs smoothed poses"
    )
    parser.add_argument(
        "poses_original",
        type=Path,
        help="Path to original poses.json",
    )
    parser.add_argument(
        "poses_smoothed",
        type=Path,
        help="Path to smoothed poses.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save comparison plot to image file (default: display interactively)",
    )
    
    args = parser.parse_args()
    
    poses_orig = args.poses_original.resolve()
    poses_smooth = args.poses_smoothed.resolve()
    
    if not poses_orig.exists():
        print(f"Error: {poses_orig} not found")
        return 1
    if not poses_smooth.exists():
        print(f"Error: {poses_smooth} not found")
        return 1
    
    print(f"Loading original poses from: {poses_orig}")
    data_orig = load_poses_json(poses_orig)
    
    print(f"Loading smoothed poses from: {poses_smooth}")
    data_smooth = load_poses_json(poses_smooth)
    
    print("Creating comparison plots...")
    plot_comparison(
        data_orig["frames"],
        data_smooth["frames"],
        output_path=args.output,
    )
    if args.output is None:
        print()
        print("✓ Interactive plot displayed. Close the window when finished.")

    return 0


if __name__ == "__main__":
    exit(main())
