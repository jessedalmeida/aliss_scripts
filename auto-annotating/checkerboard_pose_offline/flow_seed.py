#!/usr/bin/env python3
"""
Optical-flow based seeding for failed checkerboard detections.
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Any

import cv2
import numpy as np

from checkerboard_pose_offline import (
    compute_pose_covariance,
    _solve_pnp_refine,
    flatten_covariance,
    rvec_to_quaternion,
    generate_checkerboard_info,
    to_gray_image,
)
from visualize_poses import collect_frames, load_camera_yaml


def forward_backward_check(prev_pts, next_pts, next_back_pts, max_err=1.5):
    # prev_pts, next_pts, next_back_pts are Nx2 arrays
    err = np.linalg.norm(prev_pts - next_back_pts, axis=1)
    good = err <= max_err
    return good, err


def try_optical_flow_seed(
    bag_dir: Path,
    frame_key: str,
    poses_payload: dict,
    camera_yaml_path: Path,
    neighbor_radius: int = 2,
    fb_max_err: float = 1.5,
    min_tracked: int = 4,
    pixel_noise_sigma: float = 10.0,
) -> dict[str, Any] | None:
    """
    Attempt to seed corners via optical flow from nearest successful neighbor frames.
    Returns a pose dict if successful, otherwise None.
    """
    frames = poses_payload.get("frames", {})
    try:
        idx = int(frame_key)
    except Exception:
        return None

    # collect neighbor frames indices sorted by distance
    success_idxs = sorted([int(k) for k, v in frames.items() if v.get("status") == "ok" and v.get("pose") and v.get("corners")])
    if not success_idxs:
        return None

    # find nearest neighbors within radius
    neighbors = []
    for d in range(1, neighbor_radius + 1):
        before = idx - d
        after = idx + d
        if before in success_idxs:
            neighbors.append(before)
        if after in success_idxs:
            neighbors.append(after)
    if not neighbors:
        return None

    frames_dir = bag_dir / "frames"
    frame_paths = collect_frames(frames_dir)

    # load target image
    target_path = None
    for p in frame_paths:
        if p.stem.endswith(frame_key) or frame_key in p.stem:
            target_path = p
            break
    if target_path is None:
        target_path = frames_dir / f"{idx:06d}.jpg"
        if not target_path.exists():
            target_path = frames_dir / f"frame_{idx:06d}.jpg"
            if not target_path.exists():
                return None

    target_img = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if target_img is None:
        return None
    target_gray = to_gray_image(target_img)

    camera = load_camera_yaml(camera_yaml_path)
    camera_matrix = np.asarray(camera["projection_matrix"]["data"], dtype=np.float64).reshape(3,4)[:,:3]
    distortion = np.asarray(camera["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1,1)

    params = poses_payload.get("parameters", {})
    squares_x = int(params.get("squares_x", 4))
    squares_y = int(params.get("squares_y", 5))
    square_size = float(params.get("square_size", 0.002))
    checker = generate_checkerboard_info(squares_x, squares_y, square_size)
    object_points = np.asarray(checker.object_points, dtype=np.float64)

    lk_params = dict(winSize=(21,21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    for nb in neighbors:
        nb_key = f"{nb:06d}"
        nb_entry = frames.get(nb_key)
        if not nb_entry:
            continue
        corners = nb_entry.get("corners")
        if not corners:
            continue
        # corners stored as list of [x,y]
        prev_pts = np.asarray(corners, dtype=np.float32)
        # find neighbor image path
        nb_path = None
        for p in frame_paths:
            if p.stem.endswith(nb_key) or nb_key in p.stem:
                nb_path = p
                break
        if nb_path is None:
            nb_path = frames_dir / f"{nb:06d}.jpg"
            if not nb_path.exists():
                nb_path = frames_dir / f"frame_{nb:06d}.jpg"
                if not nb_path.exists():
                    continue
        nb_img = cv2.imread(str(nb_path), cv2.IMREAD_COLOR)
        if nb_img is None:
            continue
        nb_gray = to_gray_image(nb_img)

        # track forward nb -> target
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(nb_gray, target_gray, prev_pts, None, **lk_params)
        # track backward target -> nb
        next_back, st2, err2 = cv2.calcOpticalFlowPyrLK(target_gray, nb_gray, next_pts, None, **lk_params)

        # forward-backward check
        good_mask, fb_err = forward_backward_check(prev_pts, next_pts, next_back, max_err=fb_max_err)
        good_idx = np.nonzero(good_mask)[0]

        if len(good_idx) < min_tracked:
            continue

        tracked_image_points = next_pts[good_idx]
        tracked_object_points = np.asarray(object_points, dtype=np.float64)[good_idx]

        # Solve PnP
        res = _solve_pnp_refine(tracked_object_points, tracked_image_points, camera_matrix, distortion)
        if res is None:
            continue
        rvec, tvec = res
        cov = compute_pose_covariance(tracked_object_points, tracked_image_points, rvec, tvec, camera_matrix, distortion, pixel_noise_sigma)
        projected = cv2.projectPoints(tracked_object_points, rvec, tvec, camera_matrix, distortion)[0].reshape(-1, 2)
        if len(tracked_image_points) > 0:
            diffs = tracked_image_points - projected
            rms = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))
        else:
            rms = 0.0

        pose_dict = {
            "frame_id": "endoscope_optical",
            "position": [float(tvec[0,0]), float(tvec[1,0]), float(tvec[2,0])],
            "quaternion": rvec_to_quaternion(rvec),
            "covariance": flatten_covariance(cov),
        }
        result = {
            "status": "ok",
            "pose": pose_dict,
            "corners": tracked_image_points.tolist(),
            "corner_indices": good_idx.tolist(),
            "detector": "optical_flow_seed",
            "rms_reprojection_error": float(rms),
            "used_neighbor": nb,
            "num_tracked": int(len(good_idx)),
        }
        return result

    return None
