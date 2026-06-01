"""
needle_pipeline.pose_ops - recompute a checkerboard pose from corner pixels.

Used by the GUI's corner-fix mode and the legacy review path. Reuses the
PnP / covariance functions from the existing checkerboard module rather than
reimplementing them. The legacy algorithm scripts must be importable (the
server adds --scripts-dir to sys.path at startup).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import cv2

def _import_checkerboard():
    """Return the checkerboard_pose_offline module whether it's a flat module
    (checkerboard_pose_offline.py) or nested in a same-named package folder
    (checkerboard_pose_offline/checkerboard_pose_offline.py). Verifies the
    resolved module actually exposes the API, since the bare name can resolve
    to an empty package __init__ that imports fine but has none of the symbols."""
    import importlib
    last = None
    for name in ("checkerboard_pose_offline",
                 "checkerboard_pose_offline.checkerboard_pose_offline"):
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            last = exc
            continue
        if hasattr(mod, "generate_checkerboard_info"):
            return mod
    raise ImportError(
        "Could not import checkerboard_pose_offline with its API. If it lives in "
        "a subfolder, ensure that folder is on --scripts-dir; either expose the "
        "functions from the package __init__.py or keep the module file inside it."
    ) from last


def _import_estimate():
    import importlib
    last = None
    for name in ("estimate_checkerboard_pose_offline",
                 "checkerboard_pose_offline.estimate_checkerboard_pose_offline"):
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            last = exc
            continue
        if hasattr(mod, "choose_camera_yaml"):
            return mod
    raise ImportError("Could not import estimate_checkerboard_pose_offline with its API.") from last


def _camera_yaml(ctx, bag: str) -> Path:
    if ctx.camera_yaml:
        return Path(ctx.camera_yaml)
    choose_camera_yaml = _import_estimate().choose_camera_yaml
    return choose_camera_yaml(ctx.ann_dir, ctx.bag_dir(bag), None)


def expected_corner_count(ctx) -> int:
    b = ctx.board
    return (b["squares_x"] - 1) * (b["squares_y"] - 1)


def recompute_from_corners(ctx, bag: str, key: str, corners, initial_pose=None) -> dict:
    """corners: list of [x, y] in board order. Returns a poses.json frame dict."""
    cb = _import_checkerboard()
    generate_checkerboard_info = cb.generate_checkerboard_info
    load_camera_model = cb.load_camera_model
    _solve_pnp_refine = cb._solve_pnp_refine
    compute_pose_covariance = cb.compute_pose_covariance
    pose_rms_reprojection_error = cb.pose_rms_reprojection_error
    rvec_to_quaternion = cb.rvec_to_quaternion
    flatten_covariance = cb.flatten_covariance

    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    board = generate_checkerboard_info(
        ctx.board["squares_x"], ctx.board["squares_y"], ctx.board["square_size"])
    object_points = np.asarray(board.object_points, dtype=np.float64)

    if image_points.shape[0] != object_points.shape[0]:
        raise ValueError(
            f"need {object_points.shape[0]} corners "
            f"({ctx.board['squares_x']-1}x{ctx.board['squares_y']-1}), "
            f"got {image_points.shape[0]}")

    camera = load_camera_model(_camera_yaml(ctx, bag), rectified_input=True)
    # IMPORTANT: when rectified_input=True the estimator solves PnP with the
    # PROJECTION matrix (P[:, :3]), not the raw camera_matrix. Using the wrong
    # one (they differ here: fx 646 vs 794) throws Z off by ~focal-ratio.
    K = cb.camera_matrix_for_model(camera, rectified_input=True)
    dist = np.asarray(camera.distortion_coeffs, dtype=np.float64)

    initial_rvec = None
    initial_tvec = None

    if initial_pose:

        pos = initial_pose.get("position")
        quat = initial_pose.get("quaternion")

        if pos is not None and quat is not None:
            initial_tvec = np.asarray(pos, dtype=np.float64).reshape(3, 1)

            qx, qy, qz, qw = quat
            n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
            qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

            R = np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
            ], dtype=np.float64)

            initial_rvec, _ = cv2.Rodrigues(R)

    # refined = _solve_pnp_refine(object_points, image_points, K, dist)
    refined = _solve_pnp_refine(
        object_points,
        image_points,
        K,
        dist,
        rvec=initial_rvec,
        tvec=initial_tvec,
    )
    if refined is None:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = refined

    cov = compute_pose_covariance(object_points, image_points, rvec, tvec, K, dist, 1.0)
    rms = pose_rms_reprojection_error(object_points, image_points, rvec, tvec, K, dist)

    return {
        "frame_key": key,
        "status": "ok",
        "detector": "manual_corner",
        "failure_reason": "",
        "corners": image_points.tolist(),
        "pose": {
            "frame_id": "endoscope_optical",
            "position": [float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])],
            "quaternion": rvec_to_quaternion(rvec),
            "covariance": flatten_covariance(cov),
        },
        "rms_reprojection_error": float(rms),
    }


def _rvec_tvec_for_frame(ctx, bag: str, frame: dict):
    """Recover (rvec, tvec) for a solved frame. Prefer the stored corners
    (exact solve); fall back to the stored quaternion + position."""
    cb = _import_checkerboard()
    pose = frame.get("pose") or {}
    pos = pose.get("position")
    if pos is None:
        return None
    tvec = np.asarray(pos, dtype=np.float64).reshape(3, 1)

    corners = frame.get("corners")
    if corners:
        board = cb.generate_checkerboard_info(
            ctx.board["squares_x"], ctx.board["squares_y"], ctx.board["square_size"])
        obj = np.asarray(board.object_points, dtype=np.float64)
        img = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if img.shape[0] == obj.shape[0]:
            camera = load_camera_model_cached(ctx, bag)
            refined = cb._solve_pnp_refine(obj, img, camera[0], camera[1])
            if refined is not None:
                return refined
    # fallback: quaternion [qx,qy,qz,qw] -> rotation matrix -> rvec
    q = pose.get("quaternion")
    if q is None:
        return None
    qx, qy, qz, qw = q
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec


def load_camera_model_cached(ctx, bag: str):
    cb = _import_checkerboard()
    camera = cb.load_camera_model(_camera_yaml(ctx, bag), rectified_input=True)
    # match the estimator: projection matrix when rectified, not raw camera_matrix
    K = cb.camera_matrix_for_model(camera, rectified_input=True)
    return (np.asarray(K, dtype=np.float64),
            np.asarray(camera.distortion_coeffs, dtype=np.float64))


def project_axes(ctx, bag: str, key: str, frame: dict, axis_len: float | None = None) -> dict | None:
    """Project the board coordinate frame (origin + X/Y/Z tips) to pixel coords
    for overlay. Returns {origin:[x,y], x:[x,y], y:[x,y], z:[x,y]} or None."""
    rt = _rvec_tvec_for_frame(ctx, bag, frame)
    if rt is None:
        return None
    rvec, tvec = rt
    K, dist = load_camera_model_cached(ctx, bag)
    # default axis length: 2 squares, so it's visibly scaled to the board
    L = axis_len if axis_len else 2.0 * float(ctx.board["square_size"])
    pts3d = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]], dtype=np.float64)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    p = proj.reshape(-1, 2)
    return {"origin": p[0].tolist(), "x": p[1].tolist(),
            "y": p[2].tolist(), "z": p[3].tolist()}


def _import_temporal():
    import importlib
    for name in ("temporal_interpolation",
                 "checkerboard_pose_offline.temporal_interpolation"):
        try:
            m = importlib.import_module(name)
            if hasattr(m, "smooth_poses_se3"):
                return m
        except ImportError:
            continue
    raise ImportError("Could not import temporal_interpolation with smooth_poses_se3.")


def resmooth(ctx, bag: str, z_downweight: float = 1.0, process_noise_scale: float = 1.0,
             to_preview: bool = False) -> dict:
    """Re-run SE(3) smoothing on the bag's current poses.json.

    z_downweight > 1 inflates each frame's Z (and out-of-plane) position variance
    so the smoother trusts Z measurements less and leans on the temporal prior —
    the right lever when the board is small in-frame and depth is ill-conditioned.
    Writes poses_smooth.json (or poses_smooth_preview.json if to_preview).
    """
    import json
    import tempfile
    tmp = _import_temporal()
    bd = ctx.bag_dir(bag)
    poses_path = bd / "poses.json"
    if not poses_path.exists():
        raise RuntimeError("no poses.json to smooth")

    src_path = poses_path
    if z_downweight and z_downweight != 1.0:
        data = json.loads(poses_path.read_text())
        for fr in data.get("frames", {}).values():
            pose = fr.get("pose") if isinstance(fr, dict) else None
            if not pose:
                continue
            cov = pose.get("covariance")
            if cov and len(cov) == 36:
                c = list(cov)
                # 6x6 row-major; position Z variance is index 2*6+2 = 14
                c[14] *= float(z_downweight) ** 2
                # also down-weight the two in-plane tilt rotations (rx, ry) -> indices 21, 28
                c[21] *= float(z_downweight)
                c[28] *= float(z_downweight)
                pose["covariance"] = c
        tf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        tf.write(json.dumps(data)); tf.close()
        src_path = Path(tf.name)

    results = tmp.smooth_poses_se3(src_path, process_noise_scale=process_noise_scale)
    out_name = "poses_smooth_preview.json" if to_preview else "poses_smooth.json"
    out_path = bd / out_name
    tmp.save_se3_smoothed_poses(src_path, results, out_path)
    if src_path != poses_path:
        try:
            src_path.unlink()
        except OSError:
            pass
    return {"smoothed_frames": len(results), "output": out_name,
            "z_downweight": z_downweight}
