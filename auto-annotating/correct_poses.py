#!/usr/bin/env python3
"""
correct_poses.py - the corner-picker that review_failed_frames.py imports.

This was referenced but missing from the pipeline. It opens a small OpenCV
window, lets you click the checkerboard corners in board order, solves for the
pose, and returns a poses.json-style result dict (or None if cancelled).

Drop this next to the other scripts. (The GUI provides the same capability as
a canvas mode; this keeps the legacy CLI path working too.)

open_corner_picker(image, camera, checkerboard, pixel_noise_sigma=10.0) -> dict | None
"""

from __future__ import annotations

import numpy as np
import cv2


def _cb():
    import importlib
    last = None
    for name in ("checkerboard_pose_offline",
                 "checkerboard_pose_offline.checkerboard_pose_offline"):
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            last = exc
            continue
        if hasattr(mod, "_solve_pnp_refine"):
            return mod
    raise ImportError("Could not import checkerboard_pose_offline with its API (flat or nested).") from last


def open_corner_picker(image, camera, checkerboard, pixel_noise_sigma: float = 10.0):
    _m = _cb()
    _solve_pnp_refine = _m._solve_pnp_refine
    compute_pose_covariance = _m.compute_pose_covariance
    pose_rms_reprojection_error = _m.pose_rms_reprojection_error
    rvec_to_quaternion = _m.rvec_to_quaternion
    flatten_covariance = _m.flatten_covariance
    object_points = np.asarray(checkerboard.object_points, dtype=np.float64)
    n = object_points.shape[0]
    K = np.asarray(camera.camera_matrix, dtype=np.float64)
    dist = np.asarray(camera.distortion_coeffs, dtype=np.float64)

    clicks: list[list[float]] = []
    win = "Corner picker - click corners in board order (U=undo, ENTER=solve, Q=cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < n:
            clicks.append([float(x), float(y)])

    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = image.copy()
        for i, (px, py) in enumerate(clicks):
            cv2.circle(disp, (int(px), int(py)), 4, (0, 255, 255), -1)
            cv2.putText(disp, str(i + 1), (int(px) + 6, int(py) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(disp, f"{len(clicks)}/{n} corners",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(win)
            return None
        if key in (ord("u"), ord("U")) and clicks:
            clicks.pop()
        if key in (13, 10) and len(clicks) == n:
            break

    cv2.destroyWindow(win)
    image_points = np.asarray(clicks, dtype=np.float64).reshape(-1, 2)
    refined = _solve_pnp_refine(object_points, image_points, K, dist)
    if refined is None:
        print("[WARN] solvePnP failed on picked corners")
        return None
    rvec, tvec = refined
    cov = compute_pose_covariance(object_points, image_points, rvec, tvec, K, dist, pixel_noise_sigma)
    rms = pose_rms_reprojection_error(object_points, image_points, rvec, tvec, K, dist)
    return {
        "status": "ok",
        "detector": "manual_correction",
        "corners": image_points.tolist(),
        "pose": {
            "frame_id": "endoscope_optical",
            "position": [float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])],
            "quaternion": rvec_to_quaternion(rvec),
            "covariance": flatten_covariance(cov),
        },
        "rms_reprojection_error": float(rms),
    }
