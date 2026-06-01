#!/usr/bin/env python3
"""
Temporal pose interpolation for failed frames.
Fills in missing poses by interpolating from neighboring successful frames.

Also includes SE(3) manifold-based smoothing using principled Lie group mechanics.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


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


def smooth_poses_rts(
    poses_json_path: Path,
    process_noise_scale: float = 1.0,
    only_ok_frames: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Smooth successful pose detections using Rauch-Tung-Striebel (RTS) backward filter.
    
    This is a principled approach that:
    - Uses pose covariances as measurement noise (higher cov → allow more smoothing)
    - Assumes small process noise (poses shouldn't jump between frames)
    - Does optimal backward smoothing for lowest total error
    
    Args:
        poses_json_path: Path to poses.json
        process_noise_scale: Scales the assumed process noise between frames.
                             Higher = smoother. Default 1.0 is conservative.
                             Try 0.1-10 depending on how much smoothing you want.
        only_ok_frames: If True, only smooth frames with status="ok". 
                        If False, smooth all frames with valid poses.
    
    Returns:
        Dict mapping frame_key -> {"smoothed_pose": pose_dict, "original_pose": pose_dict}
    """
    with open(poses_json_path, "r") as fh:
        payload = json.load(fh)
    
    frames = payload.get("frames", {})
    
    # Collect all frames to smooth (in order)
    frame_keys_sorted = sorted(
        [k for k in frames.keys() if k.isdigit()],
        key=lambda x: int(x)
    )
    
    frame_data_list = []
    for frame_key in frame_keys_sorted:
        frame_data = frames[frame_key]
        if only_ok_frames and frame_data.get("status") != "ok":
            continue
        pose = frame_data.get("pose")
        if not pose or not pose.get("position") or not pose.get("quaternion"):
            continue
        frame_data_list.append((frame_key, frame_data, pose))
    
    if len(frame_data_list) < 2:
        return {}
    
    # Extract poses and covariances
    num_frames = len(frame_data_list)
    positions = np.array([pose[2]["position"] for pose in frame_data_list], dtype=np.float64)  # (N, 3)
    quaternions = [pose[2]["quaternion"] for pose in frame_data_list]  # List of [qx, qy, qz, qw]
    covariances = np.array([
        np.array(pose[2].get("covariance", [0.0] * 36), dtype=np.float64).reshape(6, 6)
        for pose in frame_data_list
    ])  # (N, 6, 6)
    
    # Process noise: assume small constant motion between frames
    # Split into position (3x3) and orientation (3x3) blocks
    Q_pos = np.eye(3) * (0.001 ** 2) * process_noise_scale  # position process noise
    Q_rot = np.eye(3) * (0.001 ** 2) * process_noise_scale  # rotation (axis-angle) process noise
    
    # ===== Forward pass (Kalman filter) =====
    x_filt = np.copy(positions)  # Filtered positions
    P_filt = np.zeros((num_frames, 3, 3))  # Filtered covariance for positions
    
    # Initial state
    P_filt[0] = covariances[0, :3, :3]  # Use measurement covariance
    
    for k in range(1, num_frames):
        # Predict
        x_pred = x_filt[k-1]  # Simple constant-velocity model: x_k|k-1 = x_{k-1}
        P_pred = P_filt[k-1] + Q_pos
        
        # Measurement update
        z_k = positions[k]
        R_k = covariances[k, :3, :3]  # Measurement covariance (position part)
        
        # Kalman gain
        S_k = P_pred + R_k
        K_k = P_pred @ np.linalg.inv(S_k)
        
        # Update
        x_filt[k] = x_pred + K_k @ (z_k - x_pred)
        P_filt[k] = (np.eye(3) - K_k) @ P_pred
    
    # ===== Backward pass (RTS smoother) =====
    x_smooth = np.copy(x_filt)
    P_smooth = np.copy(P_filt)
    
    for k in range(num_frames - 2, -1, -1):
        # Predicted covariance at k+1
        P_pred_next = P_filt[k] + Q_pos
        
        # Smoother gain
        C_k = P_filt[k] @ np.linalg.inv(P_pred_next)
        
        # Smoothing update
        x_smooth[k] = x_filt[k] + C_k @ (x_smooth[k+1] - x_filt[k])
        P_smooth[k] = P_filt[k] + C_k @ (P_smooth[k+1] - P_pred_next) @ C_k.T
    
    # ===== Smooth quaternions using weighted SLERP =====
    # For rotations, use covariance trace as weighting (larger cov = less confident)
    quaternions_smooth = [q.copy() for q in quaternions]
    
    for k in range(1, num_frames - 1):
        q_before = quaternions_smooth[k - 1]
        q_curr = quaternions[k]
        q_after = quaternions_smooth[k + 1]
        
        # Weight by inverse covariance (more confident → stronger influence)
        cov_curr = covariances[k, 3:6, 3:6]
        cov_trace = np.trace(cov_curr)
        if cov_trace > 1e-8:
            w_curr = 1.0 / cov_trace
        else:
            w_curr = 1.0
        
        # Blend: interpolate between current and neighbors
        # Forward interpolation
        q_fwd = quaternion_slerp(q_before, q_curr, 0.5)
        # Backward interpolation  
        q_bwd = quaternion_slerp(q_curr, q_after, 0.5)
        # Blend with confidence-based weighting
        alpha = min(0.3, 1.0 / (1.0 + w_curr))  # Cap smoothing strength
        quaternions_smooth[k] = quaternion_slerp(q_curr, 
                                                 quaternion_slerp(q_fwd, q_bwd, 0.5),
                                                 alpha)
    
    # Build results
    results = {}
    for i, (frame_key, frame_data, pose) in enumerate(frame_data_list):
        # Reconstruct smoothed pose
        smoothed_pose = {
            "frame_id": pose.get("frame_id", "endoscope_optical"),
            "position": [float(v) for v in x_smooth[i]],
            "quaternion": [float(v) for v in quaternions_smooth[i]],
            "covariance": [float(v) for v in P_smooth[i].flatten()],
        }
        
        results[frame_key] = {
            "original_pose": pose,
            "smoothed_pose": smoothed_pose,
            "original_position_error": float(np.linalg.norm(positions[i] - x_smooth[i])),
        }
    
    return results


def save_smoothed_poses(
    poses_json_path: Path,
    smoothing_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save smoothed poses to a new poses.json file.
    
    Args:
        poses_json_path: Original poses.json path
        smoothing_results: Results from smooth_poses_rts()
        output_path: Where to write the smoothed poses.json
    """
    with open(poses_json_path, "r") as fh:
        payload = json.load(fh)
    
    frames = payload.get("frames", {})
    
    for frame_key, result in smoothing_results.items():
        frame_data = frames.get(frame_key, {})
        frame_data["pose"] = result["smoothed_pose"]
        frame_data["smoothing_info"] = {
            "method": "rts_filter",
            "original_position_error_m": result["original_position_error"],
        }
        frames[frame_key] = frame_data
    
    payload["frames"] = frames
    payload["smoothing_applied"] = True
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


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


# ============================================================================
# SE(3) MANIFOLD-BASED SMOOTHING
# ============================================================================

def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation (3D) to rotation matrix.
    Args:
        axis_angle: (3,) array with direction=axis, magnitude=angle
    Returns:
        (3, 3) rotation matrix
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.eye(3)
    axis = axis_angle / angle
    # Rodrigues formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to axis-angle representation.
    Args:
        R: (3, 3) rotation matrix
    Returns:
        (3,) axis-angle vector
    """
    rot = Rotation.from_matrix(R)
    rotvec = rot.as_rotvec()
    return rotvec


def pose_to_se3_matrix(position: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
    """
    Build SE(3) matrix from position and axis-angle rotation.
    Args:
        position: (3,) translation vector
        axis_angle: (3,) axis-angle rotation
    Returns:
        (4, 4) SE(3) matrix [R | t; 0 | 1]
    """
    R = axis_angle_to_matrix(axis_angle)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def se3_matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract position and axis-angle from SE(3) matrix.
    Args:
        T: (4, 4) SE(3) matrix
    Returns:
        Tuple of (position (3,), axis_angle (3,))
    """
    position = T[:3, 3]
    axis_angle = matrix_to_axis_angle(T[:3, :3])
    return position, axis_angle


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map from se(3) algebra to SE(3) group.
    xi is a 6D vector [rho, phi] where:
      - phi: (3,) rotation (axis-angle)
      - rho: (3,) translation
    Args:
        xi: (6,) tangent space element [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]
    Returns:
        (4, 4) SE(3) matrix
    """
    rho = xi[:3]  # Translation part
    phi = xi[3:]  # Rotation part (axis-angle)
    
    # Rotation part: standard exp map
    R = axis_angle_to_matrix(phi)
    
    # Translation: affected by rotation
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        # First-order approximation when angle ≈ 0
        V = np.eye(3)
    else:
        axis = phi / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        V = np.eye(3) + (1 - np.cos(angle)) * K / angle + (angle - np.sin(angle)) * (K @ K) / (angle ** 2)
    
    t = V @ rho
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithm map from SE(3) group to se(3) algebra.
    Args:
        T: (4, 4) SE(3) matrix
    Returns:
        (6,) tangent space element [rho_x, rho_y, rho_z, phi_x, phi_y, phi_z]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    
    # Rotation log (axis-angle)
    phi = matrix_to_axis_angle(R)
    
    # Translation: need to invert V
    angle = np.linalg.norm(phi)
    if angle < 1e-8:
        V_inv = np.eye(3)
    else:
        axis = phi / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        V = np.eye(3) + (1 - np.cos(angle)) * K / angle + (angle - np.sin(angle)) * (K @ K) / (angle ** 2)
        V_inv = np.linalg.inv(V)
    
    rho = V_inv @ t
    
    xi = np.concatenate([rho, phi])
    return xi


def smooth_poses_se3(
    poses_json_path: Path,
    process_noise_scale: float = 1.0,
    only_ok_frames: bool = True,
    use_cov_drift: bool = True,
    pos_process_noise: float = 1e-7,
) -> dict[str, dict[str, Any]]:
    """
    Smooth pose detections using SE(3) manifold-based covariance-weighted smoothing.

    Implements a Rauch-Tung-Striebel (RTS) smoother directly on the ABSOLUTE pose
    measurements, treating each detected pose as an independent observation of the
    true pose at that frame.

    Position is smoothed with a standard linear RTS filter.
    Rotation is smoothed with a geodesic SO(3) RTS filter that linearises in the
    tangent space of the current filtered estimate, avoiding cumulative drift.

    Previous versions smoothed the *relative* frame-to-frame increments (xi_rel)
    and then reconstructed the trajectory by chaining exp(xi_smooth) from the
    first frame.  That is a dead-reckoning reconstruction: any per-step smoothing
    deviation accumulates into a growing constant offset from the true absolute
    measurements.  This version keeps the filter state anchored to the absolute
    measurements so no such drift can occur.

    Args:
        poses_json_path: Path to poses.json
        process_noise_scale: Scales the assumed process noise between frames.
                             Higher = smoother trajectory. Default 1.0.
                             Typical range: 0.1-10.0
        only_ok_frames: If True, only smooth frames with status="ok"
        use_cov_drift: Unused; kept for API compatibility.

    Returns:
        Dict mapping frame_key -> {
            "original_pose": pose_dict,
            "smoothed_pose": pose_dict,
            "residual_norm": float (log-manifold distance between smoothed and original)
        }
    """
    with open(poses_json_path, "r") as fh:
        payload = json.load(fh)

    frames = payload.get("frames", {})

    # Collect frames in order
    frame_keys_sorted = sorted(
        [k for k in frames.keys() if k.isdigit()],
        key=lambda x: int(x)
    )

    frame_data_list = []
    for frame_key in frame_keys_sorted:
        frame_data = frames[frame_key]
        if frame_data.get("status") == "failed":
            continue
        if only_ok_frames and frame_data.get("status") != "ok":
            continue
        pose = frame_data.get("pose")
        if not pose or not pose.get("position") or not pose.get("quaternion"):
            continue
        frame_data_list.append((frame_key, frame_data, pose))

    if len(frame_data_list) < 2:
        return {}

    num_frames = len(frame_data_list)

    # Parse measurements
    positions = np.array([f[2]["position"] for f in frame_data_list], dtype=np.float64)
    quats_meas = [Rotation.from_quat(np.array(f[2]["quaternion"], dtype=np.float64))
                  for f in frame_data_list]
    covs = [
        np.array(f[2].get("covariance", np.eye(6) * 0.001), dtype=np.float64).reshape(6, 6)
        for f in frame_data_list
    ]

    # Process noise (random-walk model between frames).
    # Use per-frame, per-axis process noise scaled from the current
    # measurement covariance. This keeps the gain balanced for each axis
    # and prevents rare high-uncertainty measurements from over-smoothing.
    # At scale=1.0, Q_k ~ R_k for each frame.
    # scale > 1  →  Q_k > R_k  →  trust measurements more  →  less smoothing.
    # scale < 1  →  Q_k < R_k  →  trust the motion model more  →  more smoothing.

    # ── Position: standard linear RTS ──────────────────────────────────────────
    m_pos_filt = np.zeros((num_frames, 3))
    P_pos_filt = np.zeros((num_frames, 3, 3))
    m_pos_filt[0] = positions[0]
    P_pos_filt[0] = covs[0][:3, :3]

    # Position process noise is a FIXED small random-walk term, NOT proportional
    # to the measurement covariance. Tying Q to R (the previous behaviour) meant
    # that at planar-ambiguous frames — where the measurement Z-variance balloons —
    # Q ballooned too, so the filter lost its anchor and the RTS backward pass swung
    # Z by up to ~11 mm with no support in the raw data. A fixed Q keeps the gain
    # well-behaved: confident frames are still followed, noisy frames are smoothed
    # toward the local trend rather than allowed to wander.
    Q_pos_fixed = np.eye(3) * pos_process_noise * process_noise_scale
    for k in range(1, num_frames):
        P_pred = P_pos_filt[k - 1] + Q_pos_fixed
        S = P_pred + covs[k][:3, :3]
        K = P_pred @ np.linalg.inv(S)
        m_pos_filt[k] = m_pos_filt[k - 1] + K @ (positions[k] - m_pos_filt[k - 1])
        P_pos_filt[k] = (np.eye(3) - K) @ P_pred

    m_pos_smooth = m_pos_filt.copy()
    P_pos_smooth = P_pos_filt.copy()
    for k in range(num_frames - 2, -1, -1):
        P_pred = P_pos_filt[k] + Q_pos_fixed
        G = P_pos_filt[k] @ np.linalg.inv(P_pred)
        m_pos_smooth[k] = m_pos_filt[k] + G @ (m_pos_smooth[k + 1] - m_pos_filt[k])
        P_pos_smooth[k] = P_pos_filt[k] + G @ (P_pos_smooth[k + 1] - P_pred) @ G.T

    # ── Rotation: geodesic SO(3) RTS ───────────────────────────────────────────
    # State: a Rotation R_filt[k], covariance 3×3 in the local tangent space.
    # Prediction: R_pred = R_filt[k-1], P_pred = P_filt[k-1] + Q_rot (const-vel prior).
    # Update: innovation = log(R_meas * R_pred^{-1}) computed in tangent space of R_pred.
    R_filt: list[Rotation] = [None] * num_frames  # type: ignore[assignment]
    P_rot_filt = [None] * num_frames

    R_filt[0] = quats_meas[0]
    P_rot_filt[0] = covs[0][3:, 3:]

    for k in range(1, num_frames):
        # Predict
        Q_rot = np.diag(np.diag(covs[k][3:, 3:]) * process_noise_scale)
        P_pred = P_rot_filt[k - 1] + Q_rot
        R_pred = R_filt[k - 1]
        # Innovation in tangent space of R_pred
        delta = (quats_meas[k] * R_pred.inv()).as_rotvec()
        S = P_pred + covs[k][3:, 3:]
        K = P_pred @ np.linalg.inv(S)
        delta_upd = K @ delta
        R_filt[k] = Rotation.from_rotvec(delta_upd) * R_pred
        P_rot_filt[k] = (np.eye(3) - K) @ P_pred

    # Backward RTS on SO(3): smoother gain applied in tangent space of R_filt[k]
    R_smooth: list[Rotation] = list(R_filt)
    P_rot_smooth = list(P_rot_filt)
    for k in range(num_frames - 2, -1, -1):
        Q_rot = np.diag(np.diag(covs[k + 1][3:, 3:]) * process_noise_scale)
        P_pred = P_rot_filt[k] + Q_rot
        G = P_rot_filt[k] @ np.linalg.inv(P_pred)
        # Smoother correction: tangent vector from R_filt[k] toward R_smooth[k+1]
        delta_smooth = (R_smooth[k + 1] * R_filt[k].inv()).as_rotvec()
        delta_upd = G @ delta_smooth
        R_smooth[k] = Rotation.from_rotvec(delta_upd) * R_filt[k]
        P_rot_smooth[k] = P_rot_filt[k] + G @ (P_rot_smooth[k + 1] - P_pred) @ G.T

    # ── Build results ───────────────────────────────────────────────────────────
    results = {}
    for i, (frame_key, frame_data, pose) in enumerate(frame_data_list):
        pos_smooth = m_pos_smooth[i]
        quat_smooth = R_smooth[i].as_quat()

        # Residual: SE(3) log distance between smoothed and original poses
        T_orig = pose_to_se3_matrix(
            np.array(pose["position"], dtype=np.float64),
            Rotation.from_quat(np.array(pose["quaternion"], dtype=np.float64)).as_rotvec(),
        )
        T_sm = pose_to_se3_matrix(pos_smooth, R_smooth[i].as_rotvec())
        residual_norm = float(np.linalg.norm(se3_log(np.linalg.inv(T_orig) @ T_sm)))

        smoothed_pose = {
            "frame_id": pose.get("frame_id", "endoscope_optical"),
            "position": [float(v) for v in pos_smooth],
            "quaternion": [float(v) for v in quat_smooth],
            "covariance": [float(v) for v in
                           np.array(pose.get("covariance", np.eye(6) * 0.001),
                                    dtype=np.float64).reshape(6, 6).flatten()],
        }

        results[frame_key] = {
            "original_pose": pose,
            "smoothed_pose": smoothed_pose,
            "residual_norm": residual_norm,
        }

    return results


def save_se3_smoothed_poses(
    poses_json_path: Path,
    smoothing_results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save SE(3) smoothed poses to a new poses.json file.
    
    Args:
        poses_json_path: Original poses.json path
        smoothing_results: Results from smooth_poses_se3()
        output_path: Where to write the smoothed poses.json
    """
    with open(poses_json_path, "r") as fh:
        payload = json.load(fh)
    
    frames = payload.get("frames", {})
    
    for frame_key, result in smoothing_results.items():
        frame_data = frames.get(frame_key, {})
        frame_data["pose"] = result["smoothed_pose"]
        frame_data["smoothing_info"] = {
            "method": "se3_manifold_smoothing",
            "residual_norm_m": result["residual_norm"],
        }
        frames[frame_key] = frame_data
    
    payload["frames"] = frames
    payload["smoothing_applied"] = True
    payload["smoothing_method"] = "se3_information_filter"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")