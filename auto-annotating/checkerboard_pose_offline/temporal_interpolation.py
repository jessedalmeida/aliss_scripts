#!/usr/bin/env python3
"""
Temporal pose interpolation for failed frames.
Fills in missing poses by interpolating from neighboring successful frames.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def quaternion_slerp(q1: list[float], q2: list[float], t: float) -> list[float]:
    """
    Spherical linear interpolation between two quaternions.
    Args:
        q1: First quaternion [qx, qy, qz, qw]
        q2: Second quaternion [qx, qy, qz, qw]
        t: Interpolation factor (0 = q1, 1 = q2)
    Returns:
        Interpolated quaternion [qx, qy, qz, qw]
    """
    q1 = np.array(q1, dtype=np.float64)
    q2 = np.array(q2, dtype=np.float64)
    
    # Compute dot product
    dot = float(np.dot(q1, q2))
    
    # If dot product is negative, negate one quaternion to take the shorter path
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot product
    dot = np.clip(dot, -1.0, 1.0)
    
    # Compute angle
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    
    # Compute interpolated quaternion
    q3 = (q2 - q1 * dot)
    q3_norm = np.linalg.norm(q3)
    if q3_norm > 1e-8:
        q3 = q3 / q3_norm
    else:
        q3 = np.array([0.0, 0.0, 0.0, 1.0])
    
    result = q1 * math.cos(theta) + q3 * math.sin(theta)
    return [float(v) for v in result]


def interpolate_pose(
    pose_before: dict,
    pose_after: dict,
    t: float,
) -> dict[str, Any]:
    """
    Interpolate a pose between two poses.
    Args:
        pose_before: Pose dict with position and quaternion
        pose_after: Pose dict with position and quaternion
        t: Interpolation factor (0 = before, 1 = after)
    Returns:
        Interpolated pose dict
    """
    pos_before = np.array(pose_before.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    pos_after = np.array(pose_after.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
    pos_interp = (1.0 - t) * pos_before + t * pos_after
    
    quat_before = pose_before.get("quaternion", [0.0, 0.0, 0.0, 1.0])
    quat_after = pose_after.get("quaternion", [0.0, 0.0, 0.0, 1.0])
    quat_interp = quaternion_slerp(quat_before, quat_after, t)
    
    covariance_before = pose_before.get("covariance", [0.0] * 36)
    covariance_after = pose_after.get("covariance", [0.0] * 36)
    covariance_interp = [(1.0 - t) * v1 + t * v2 for v1, v2 in zip(covariance_before, covariance_after)]
    
    return {
        "frame_id": pose_before.get("frame_id", "endoscope_optical"),
        "position": [float(v) for v in pos_interp],
        "quaternion": quat_interp,
        "covariance": covariance_interp,
    }


def find_nearest_successful_frames(
    frames: dict[str, dict],
    failed_frame_idx: int,
) -> tuple[int | None, dict | None, int | None, dict | None]:
    """
    Find the nearest successful frames before and after a failed frame.
    Args:
        frames: Dict of all frames from poses.json
        failed_frame_idx: Index of the failed frame
    Returns:
        Tuple of (before_idx, before_frame_data, after_idx, after_frame_data)
        None values if not found
    """
    before_idx = None
    before_data = None
    after_idx = None
    after_data = None
    
    # Find nearest successful frame before
    for offset in range(1, failed_frame_idx + 1):
        candidate_idx = failed_frame_idx - offset
        frame_data = frames.get(f"{candidate_idx:06d}")
        if frame_data and frame_data.get("status") == "ok" and frame_data.get("pose"):
            before_idx = candidate_idx
            before_data = frame_data
            break
    
    # Find nearest successful frame after
    max_frame_idx = max(int(k) for k in frames.keys() if k.isdigit())
    for offset in range(1, max_frame_idx - failed_frame_idx + 1):
        candidate_idx = failed_frame_idx + offset
        frame_data = frames.get(f"{candidate_idx:06d}")
        if frame_data and frame_data.get("status") == "ok" and frame_data.get("pose"):
            after_idx = candidate_idx
            after_data = frame_data
            break
    
    return before_idx, before_data, after_idx, after_data


def compute_temporal_fill(
    poses_json_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Compute temporal interpolation for all failed frames.
    Returns dict of frame_key -> interpolation result with keys:
      - interpolated_pose: The computed pose
      - before_idx, before_distance: Nearest successful frame before
      - after_idx, after_distance: Nearest successful frame after
      - interpolation_factor: t value used
    """
    with open(poses_json_path, "r") as fh:
        payload = json.load(fh)
    
    frames = payload.get("frames", {})
    results = {}
    
    for frame_key, frame_data in frames.items():
        if frame_data.get("status") == "ok":
            continue
        
        try:
            frame_idx = int(frame_key)
        except (ValueError, TypeError):
            continue
        
        before_idx, before_data, after_idx, after_data = find_nearest_successful_frames(frames, frame_idx)
        
        if before_data is None or after_data is None:
            results[frame_key] = {
                "status": "no_neighbors",
                "reason": "Could not find successful frames before and after",
                "before_idx": before_idx,
                "after_idx": after_idx,
            }
            continue
        
        before_distance = frame_idx - before_idx
        after_distance = after_idx - frame_idx
        total_distance = before_distance + after_distance
        interpolation_factor = float(before_distance) / float(total_distance)
        
        before_pose = before_data.get("pose")
        after_pose = after_data.get("pose")
        
        if not before_pose or not after_pose:
            results[frame_key] = {
                "status": "no_pose",
                "reason": "Neighbor frame(s) missing pose data",
            }
            continue
        
        try:
            interpolated_pose = interpolate_pose(before_pose, after_pose, interpolation_factor)
        except Exception as e:
            results[frame_key] = {
                "status": "interpolation_failed",
                "reason": str(e),
            }
            continue
        
        results[frame_key] = {
            "status": "success",
            "interpolated_pose": interpolated_pose,
            "before_idx": before_idx,
            "before_distance": before_distance,
            "after_idx": after_idx,
            "after_distance": after_distance,
            "interpolation_factor": interpolation_factor,
        }
    
    return results


def save_interpolated_preview(
    poses_json_path: Path,
    temporal_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save a preview poses.json with temporal interpolations filled in.
    Allows user to review before accepting.
    """
    with open(poses_json_path, "r") as fh:
        payload = json.load(fh)
    
    frames = payload.get("frames", {})
    
    for frame_key, result in temporal_results.items():
        if result.get("status") != "success":
            continue
        
        frame_data = frames.get(frame_key, {})
        frame_data["status"] = "interpolated"
        frame_data["pose"] = result["interpolated_pose"]
        frame_data["detector"] = "temporal_interpolation"
        frame_data["rms_reprojection_error"] = 0.0
        frame_data["failure_reason"] = ""
        frame_data["temporal_source"] = {
            "before_idx": result["before_idx"],
            "after_idx": result["after_idx"],
            "interpolation_factor": result["interpolation_factor"],
        }
        frames[frame_key] = frame_data
    
    payload["frames"] = frames
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
