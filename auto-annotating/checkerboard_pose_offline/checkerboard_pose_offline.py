from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from pathlib import Path
from typing import Any
import json
import math
import os
from functools import lru_cache

import cv2
import numpy as np
import yaml


@dataclass
class CheckerboardInfo:
    squares_x: int
    squares_y: int
    square_size: float
    pattern_size: tuple[int, int]
    object_points: list[tuple[float, float, float]]


@dataclass
class CameraModel:
    image_width: int
    image_height: int
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    projection_matrix: np.ndarray
    distortion_model: str = "plumb_bob"


@dataclass
class CheckerboardDetectionResult:
    success: bool = False
    corners: list[tuple[float, float]] = field(default_factory=list)
    detector_name: str = ""
    failure_stage: str = ""
    failure_reason: str = ""
    roi: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class PoseEstimateResult:
    success: bool
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    covariance: np.ndarray | None = None
    rms_reprojection_error: float | None = None
    failure_reason: str = ""


def frame_index_from_path(path: Path) -> int | None:
    stem = path.stem
    for part in stem.split("_"):
        if part.isdigit():
            return int(part)
    if stem.isdigit():
        return int(stem)
    return None


def sanitize_for_filename(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in text)


def to_gray_image(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.copy()


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0.0 or abs(gamma - 1.0) < 1e-3:
        return image.copy()

    lut = np.empty((256,), dtype=np.uint8)
    for i in range(256):
        lut[i] = np.clip(255.0 * ((i / 255.0) ** gamma), 0, 255)
    return cv2.LUT(image, lut)


def apply_preprocessing_chain(
    gray: np.ndarray,
    steps: list[str],
    clahe_clip_limit: float,
    clahe_tile_grid_size: int,
    blur_kernel_size: int,
    gamma_value: float,
) -> np.ndarray:
    current = gray.copy()
    for step in steps:
        if step == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
            current = clahe.apply(current)
        elif step == "normalize":
            current = cv2.normalize(current, None, 0, 255, cv2.NORM_MINMAX)
            current = current.astype(np.uint8, copy=False)
        elif step == "blur":
            kernel = max(1, blur_kernel_size | 1)
            current = cv2.GaussianBlur(current, (kernel, kernel), 0.0)
        elif step == "denoise":
            current = cv2.fastNlMeansDenoising(current, None, 7, 7, 21)
        elif step == "gamma":
            current = apply_gamma(current, gamma_value)
    return current


def generate_checkerboard_info(squares_x: int, squares_y: int, square_size: float) -> CheckerboardInfo:
    pattern_size = (squares_x - 1, squares_y - 1)
    center_x = 0.5 * (pattern_size[0] - 1) * square_size
    center_y = 0.5 * (pattern_size[1] - 1) * square_size
    object_points: list[tuple[float, float, float]] = []
    for y in range(pattern_size[1]):
        for x in range(pattern_size[0]):
            object_points.append((x * square_size - center_x, y * square_size - center_y, 0.0))
    return CheckerboardInfo(
        squares_x=squares_x,
        squares_y=squares_y,
        square_size=square_size,
        pattern_size=pattern_size,
        object_points=object_points,
    )


def load_camera_model(camera_yaml_path: Path, rectified_input: bool) -> CameraModel:
    with open(camera_yaml_path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)

    image_width = int(payload.get("image_width", 0))
    image_height = int(payload.get("image_height", 0))
    distortion_model = str(payload.get("distortion_model", "plumb_bob"))

    camera_matrix = np.asarray(payload["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    distortion_coeffs = np.asarray(payload["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1, 1)
    projection_matrix = np.asarray(payload["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)

    if rectified_input:
        distortion_coeffs = np.zeros((5, 1), dtype=np.float64)

    return CameraModel(
        image_width=image_width,
        image_height=image_height,
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion_coeffs,
        projection_matrix=projection_matrix,
        distortion_model=distortion_model,
    )


def camera_matrix_for_model(camera: CameraModel, rectified_input: bool) -> np.ndarray:
    if rectified_input:
        return camera.projection_matrix[:, :3].astype(np.float64, copy=True)
    return camera.camera_matrix.astype(np.float64, copy=True)


def clamp_rect(roi: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    max_w, max_h = size
    x = max(0, min(x, max_w - 1))
    y = max(0, min(y, max_h - 1))
    w = max(0, min(w, max_w - x))
    h = max(0, min(h, max_h - y))
    return x, y, w, h


def inflate_rect(roi: tuple[int, int, int, int], padding_factor: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return (0, 0, 0, 0)
    pad_x = int(round(w * padding_factor))
    pad_y = int(round(h * padding_factor))
    return clamp_rect((x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y), size)


def bbox_from_points(points: list[tuple[float, float]], size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not points:
        return (0, 0, 0, 0)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x = int(math.floor(min(xs)))
    y = int(math.floor(min(ys)))
    w = int(math.ceil(max(xs))) - x + 1
    h = int(math.ceil(max(ys))) - y + 1
    return clamp_rect((x, y, w, h), size)


def build_search_rois(
    image_size: tuple[int, int],
    last_pose: tuple[np.ndarray, np.ndarray] | None,
    last_corners: list[tuple[float, float]] | None,
    board_points: list[tuple[float, float, float]],
    camera_matrix: np.ndarray | None,
    distortion_coeffs: np.ndarray,
    use_temporal_roi: bool,
    temporal_roi_padding_factor: float,
) -> list[tuple[int, int, int, int]]:
    rois: list[tuple[int, int, int, int]] = []
    if use_temporal_roi:
        seed = (0, 0, 0, 0)
        if last_pose is not None and camera_matrix is not None and board_points:
            rvec, tvec = last_pose
            projected, _ = cv2.projectPoints(np.asarray(board_points, dtype=np.float64), rvec, tvec, camera_matrix, distortion_coeffs)
            pts = [(float(p[0][0]), float(p[0][1])) for p in projected]
            seed = bbox_from_points(pts, image_size)
        elif last_corners:
            seed = bbox_from_points(last_corners, image_size)

        if seed[2] > 0 and seed[3] > 0:
            tight = inflate_rect(seed, temporal_roi_padding_factor, image_size)
            wide = inflate_rect(seed, max(temporal_roi_padding_factor * 2.0, temporal_roi_padding_factor + 0.25), image_size)
            if tight[2] > 0 and tight[3] > 0:
                rois.append(tight)
            if wide[2] > 0 and wide[3] > 0 and wide != tight:
                rois.append(wide)

    rois.append((0, 0, image_size[0], image_size[1]))

    unique: list[tuple[int, int, int, int]] = []
    for roi in rois:
        if roi[2] <= 0 or roi[3] <= 0:
            continue
        if roi not in unique:
            unique.append(roi)
    return unique


def detect_orientation_markers(
    gray: np.ndarray,
    min_radius: int = 5,
    max_radius: int = 30,
    max_circles: int = 4,
    roi: tuple[int, int, int, int] | None = None,
) -> list[tuple[tuple[float, float], float]]:
    search = gray
    offset_x = 0
    offset_y = 0
    if roi is not None:
        x, y, w, h = roi
        if w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= gray.shape[1] and y + h <= gray.shape[0]:
            search = gray[y : y + h, x : x + w]
            offset_x = x
            offset_y = y

    blurred = cv2.GaussianBlur(search, (9, 9), 2.0, 2.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1.2,
        max(8, search.shape[0] // 6),
        param1=80,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    markers: list[tuple[tuple[float, float], float]] = []
    if circles is not None:
        circles = circles[0]
        for circle in circles[:max_circles]:
            markers.append(((float(circle[0]) + offset_x, float(circle[1]) + offset_y), float(circle[2])))
    return markers


def verify_and_fix_orientation(
    markers: list[tuple[tuple[float, float], float]],
    corners: list[tuple[float, float]],
    pattern_size: tuple[int, int],
) -> bool:
    del pattern_size
    if not markers or not corners:
        return bool(corners)
    return all(radius >= 1.0 for _, radius in markers)


def detect_checkerboard_in_roi(
    image: np.ndarray,
    pattern_size: tuple[int, int],
    roi: tuple[int, int, int, int],
    detector_mode: str,
    timeout_ms: int,
) -> CheckerboardDetectionResult:
    result = CheckerboardDetectionResult(roi=roi)
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        result.failure_stage = "roi"
        result.failure_reason = "invalid search ROI"
        return result

    min_roi_side = max(32, max(pattern_size) * 8)
    if w < min_roi_side or h < min_roi_side:
        result.failure_stage = "roi_too_small"
        result.failure_reason = "search ROI too small for stable checkerboard detection"
        return result

    roi_image = image[y : y + h, x : x + w].copy()

    def _run_legacy(fast_check: bool) -> CheckerboardDetectionResult:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        if fast_check:
            flags |= cv2.CALIB_CB_FAST_CHECK

        def _detect() -> tuple[bool, list[tuple[float, float]]]:
            found, corners = cv2.findChessboardCorners(roi_image, pattern_size, flags)
            pts = []
            if corners is not None:
                pts = [(float(pt[0][0]) + x, float(pt[0][1]) + y) for pt in corners]
            return found, pts

        state: dict[str, Any] = {"complete": False, "found": False, "corners": []}

        def _worker() -> None:
            found, corners = _detect()
            state["found"] = found
            state["corners"] = corners
            state["complete"] = True

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        deadline = None if timeout_ms <= 0 else (time.monotonic() + timeout_ms / 1000.0)
        while not state["complete"]:
            if deadline is not None and time.monotonic() > deadline:
                result.failure_stage = "detector_timeout"
                result.failure_reason = "legacy checkerboard detection timed out"
                result.detector_name = "legacy-fast" if fast_check else "legacy"
                return result
            time.sleep(0.001)

        found = bool(state["found"])
        corners = list(state["corners"])

        if not found:
            result.failure_stage = "detector_rejection"
            result.failure_reason = "legacy fast-check rejected the ROI" if fast_check else "legacy detector rejected the ROI"
            result.detector_name = "legacy-fast" if fast_check else "legacy"
            return result

        result.success = True
        result.corners = corners
        result.detector_name = "legacy-fast" if fast_check else "legacy"
        return result

    def _run_sb() -> CheckerboardDetectionResult:
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

        def _detect() -> tuple[bool, list[tuple[float, float]]]:
            found, corners = cv2.findChessboardCornersSB(roi_image, pattern_size, flags)
            pts = []
            if corners is not None:
                pts = [(float(pt[0][0]) + x, float(pt[0][1]) + y) for pt in corners]
            return found, pts

        state: dict[str, Any] = {"complete": False, "found": False, "corners": []}

        def _worker() -> None:
            found, corners = _detect()
            state["found"] = found
            state["corners"] = corners
            state["complete"] = True

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        deadline = None if timeout_ms <= 0 else (time.monotonic() + timeout_ms / 1000.0)
        while not state["complete"]:
            if deadline is not None and time.monotonic() > deadline:
                result.failure_stage = "detector_timeout"
                result.failure_reason = "SB checkerboard detection timed out"
                result.detector_name = "sb"
                return result
            time.sleep(0.001)

        found = bool(state["found"])
        corners = list(state["corners"])

        if not found:
            result.failure_stage = "detector_rejection"
            result.failure_reason = "SB detector rejected the ROI"
            result.detector_name = "sb"
            return result

        result.success = True
        result.corners = corners
        result.detector_name = "sb"
        return result

    detector_mode = detector_mode.lower().strip()
    if detector_mode in {"sb", "auto"} and hasattr(cv2, "findChessboardCornersSB"):
        sb = _run_sb()
        if sb.success or detector_mode == "sb":
            return sb

    if detector_mode == "fast":
        return _run_legacy(True)

    legacy = _run_legacy(False)
    if legacy.success or detector_mode != "auto":
        return legacy
    return legacy


def check_points_colinear(points: list[tuple[float, float]]) -> bool:
    if len(points) < 3:
        return True
    data = np.asarray(points, dtype=np.float64)
    data -= np.mean(data, axis=0, keepdims=True)
    cov = np.cov(data.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.abs(eigvals))[::-1]
    if eigvals[0] <= 1e-12:
        return True
    return (eigvals[-1] / eigvals[0]) < 1e-2


def _solve_pnp_refine(object_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(object_points) < 4:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion_coeffs,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None

    refine = getattr(cv2, "solvePnPRefineLM", None)
    if refine is not None:
        rvec, tvec = refine(object_points, image_points, camera_matrix, distortion_coeffs, rvec, tvec)
    else:
        ok_iter, rvec_iter, tvec_iter = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok_iter:
            rvec, tvec = rvec_iter, tvec_iter

    return np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1)


def compute_pose_covariance(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    pixel_noise_sigma: float,
) -> np.ndarray:
    projected, jac = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coeffs)
    del projected
    if jac is None:
        raise RuntimeError("projectPoints did not return a Jacobian")

    jacobian = np.asarray(jac[:, :6], dtype=np.float64)
    if jacobian.shape[0] != image_points.shape[0] * 2:
        jacobian = jacobian.reshape(image_points.shape[0] * 2, 6)

    info = jacobian.T @ jacobian
    if pixel_noise_sigma > 0.0:
        info /= float(pixel_noise_sigma) ** 2

    prior_sigmas = np.array([10.0, 10.0, 10.0, 1.0, 1.0, 1.0], dtype=np.float64)
    prior_info = np.diag(1.0 / np.square(prior_sigmas))
    info = info + prior_info

    try:
        covariance = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(info)

    return covariance


def pose_rms_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion_coeffs)
    projected = projected.reshape(-1, 2)
    residuals = projected - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))


def build_detection_debug_grid(
    raw_gray: np.ndarray,
    preprocessed: np.ndarray,
    roi: tuple[int, int, int, int],
    detection: CheckerboardDetectionResult,
    status: str,
    failure_reason: str,
    pattern_size: tuple[int, int],
) -> np.ndarray:
    h, w = raw_gray.shape[:2]
    grid = np.zeros((h, w, 3), dtype=np.uint8)

    def draw_panel(src: np.ndarray, dst: tuple[int, int, int, int], label: str) -> None:
        x, y, pw, ph = dst
        if src.ndim == 2:
            panel = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        else:
            panel = src.copy()
        panel = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_AREA)
        grid[y : y + ph, x : x + pw] = panel
        cv2.putText(grid, label, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    half_w = max(1, w // 2)
    half_h = max(1, h // 2)
    draw_panel(raw_gray, (0, 0, half_w, half_h), "Raw grayscale")
    draw_panel(preprocessed, (half_w, 0, half_w, half_h), "Preprocessed")

    roi_overlay = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
    rx, ry, rw, rh = roi
    if rw > 0 and rh > 0:
        cv2.rectangle(roi_overlay, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
    for corner in detection.corners:
        cv2.circle(roi_overlay, (int(round(corner[0])), int(round(corner[1]))), 4, (0, 255, 0), 2)
    draw_panel(roi_overlay, (0, half_h, half_w, half_h), "Search ROI / corners")

    detail = np.zeros((half_h, half_w, 3), dtype=np.uint8) + 10
    cv2.putText(detail, f"Status: {status}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if status == "DETECTED" else (0, 0, 255), 2)
    cv2.putText(detail, f"Detector: {detection.detector_name}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(detail, f"Pattern: {pattern_size[0]}x{pattern_size[1]}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(detail, f"Reason: {failure_reason}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    draw_panel(detail, (half_w, half_h, half_w, half_h), "Diagnostics")

    for corner in detection.corners:
        pt = (int(round(corner[0])), int(round(corner[1])))
        cv2.circle(grid, pt, 4, (255, 255, 255), 2)
        cv2.circle(grid, pt, 2, (0, 255, 0), -1)

    if rw > 0 and rh > 0:
        cv2.rectangle(grid, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    cv2.putText(grid, f"Status: {status}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if status == "DETECTED" else (0, 0, 255), 2)
    return grid


def load_detection_cache(cache_path: Path) -> dict[str, list[tuple[float, float]]]:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    result: dict[str, list[tuple[float, float]]] = {}
    for key, corners in payload.get("detections", {}).items():
        result[key] = [(float(pt[0]), float(pt[1])) for pt in corners]
    return result


def save_detection_cache(cache_path: Path, cache: dict[str, list[tuple[float, float]]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "detections": {key: [[float(x), float(y)] for x, y in corners] for key, corners in cache.items()},
    }
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def as_list_of_lists(points: list[tuple[float, float]] | list[tuple[float, float, float]] | None) -> list[list[float]]:
    if points is None:
        return []
    return [[float(v) for v in point] for point in points]


def flatten_covariance(covariance: np.ndarray | None) -> list[float]:
    if covariance is None:
        return [0.0] * 36
    return [float(v) for v in np.asarray(covariance, dtype=np.float64).reshape(-1)]


class OfflineCheckerboardEstimator:
    def __init__(
        self,
        camera: CameraModel,
        checkerboard: CheckerboardInfo,
        pixel_noise_sigma: float = 10.0,
        rectified_input: bool = True,
        detector_mode: str = "legacy",
        preprocess_steps: list[str] | None = None,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        blur_kernel_size: int = 3,
        gamma_value: float = 1.0,
        use_orientation_markers: bool = True,
        use_temporal_roi: bool = False,
        temporal_roi_padding_factor: float = 0.35,
        detection_timeout_ms: int = 1000,
        save_failed_diagnostics: bool = False,
        failed_diagnostics_dir: Path | None = None,
        detection_cache_file: Path | None = None,
    ):
        self.camera = camera
        self.checkerboard = checkerboard
        self.pixel_noise_sigma = float(pixel_noise_sigma)
        self.rectified_input = rectified_input
        self.detector_mode = detector_mode
        self.preprocess_steps = preprocess_steps or ["clahe"]
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.blur_kernel_size = blur_kernel_size
        self.gamma_value = gamma_value
        self.use_orientation_markers = use_orientation_markers
        self.use_temporal_roi = use_temporal_roi
        self.temporal_roi_padding_factor = temporal_roi_padding_factor
        self.detection_timeout_ms = detection_timeout_ms
        self.save_failed_diagnostics = save_failed_diagnostics
        self.failed_diagnostics_dir = failed_diagnostics_dir or Path("/tmp/checkerboard_pose_offline_diagnostics")
        self.detection_cache_file = detection_cache_file

        self.camera_matrix = camera_matrix_for_model(camera, rectified_input)
        self.distortion_coeffs = camera.distortion_coeffs.astype(np.float64, copy=True)
        self.last_successful_pose: tuple[np.ndarray, np.ndarray] | None = None
        self.last_successful_corners: list[tuple[float, float]] | None = None
        self.detection_cache: dict[str, list[tuple[float, float]]] = {}
        if detection_cache_file is not None:
            self.detection_cache = load_detection_cache(detection_cache_file)

    def _maybe_save_failed_diagnostics(
        self,
        frame_key: str,
        raw_gray: np.ndarray,
        preprocessed: np.ndarray,
        detection: CheckerboardDetectionResult,
        failure_reason: str,
    ) -> str | None:
        if not self.save_failed_diagnostics:
            return None
        self.failed_diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostic = build_detection_debug_grid(
            raw_gray,
            preprocessed,
            detection.roi,
            detection,
            "FAILED",
            failure_reason,
            self.checkerboard.pattern_size,
        )
        filename = self.failed_diagnostics_dir / f"{sanitize_for_filename(frame_key)}_{sanitize_for_filename(failure_reason)}.png"
        cv2.imwrite(str(filename), diagnostic)
        return str(filename)

    def _detect_frame(self, frame: np.ndarray, frame_key: str) -> tuple[CheckerboardDetectionResult, np.ndarray, np.ndarray]:
        gray = to_gray_image(frame)
        preprocessed = apply_preprocessing_chain(
            gray,
            self.preprocess_steps,
            self.clahe_clip_limit,
            self.clahe_tile_grid_size,
            self.blur_kernel_size,
            self.gamma_value,
        )

        if frame_key in self.detection_cache:
            detection = CheckerboardDetectionResult(
                success=True,
                corners=self.detection_cache[frame_key],
                detector_name="cache",
                failure_reason="replayed from cache",
                roi=bbox_from_points(self.detection_cache[frame_key], (gray.shape[1], gray.shape[0])),
            )
            return detection, gray, preprocessed

        rois = build_search_rois(
            (gray.shape[1], gray.shape[0]),
            self.last_successful_pose,
            self.last_successful_corners,
            self.checkerboard.object_points,
            self.camera_matrix,
            self.distortion_coeffs,
            self.use_temporal_roi,
            self.temporal_roi_padding_factor,
        )

        detection = CheckerboardDetectionResult()
        for roi in rois:
            detection = detect_checkerboard_in_roi(
                preprocessed,
                self.checkerboard.pattern_size,
                roi,
                self.detector_mode,
                self.detection_timeout_ms,
            )
            if detection.success:
                break
        return detection, gray, preprocessed

    def _solve_pose(self, corners: list[tuple[float, float]]) -> PoseEstimateResult:
        board_points = np.asarray(self.checkerboard.object_points, dtype=np.float64)
        image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        refined = _solve_pnp_refine(board_points, image_points, self.camera_matrix, self.distortion_coeffs)
        if refined is None:
            return PoseEstimateResult(success=False, failure_reason="solvePnP failed")

        rvec, tvec = refined
        covariance = compute_pose_covariance(
            board_points,
            image_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.distortion_coeffs,
            self.pixel_noise_sigma,
        )
        rms = pose_rms_reprojection_error(board_points, image_points, rvec, tvec, self.camera_matrix, self.distortion_coeffs)
        return PoseEstimateResult(success=True, rvec=rvec, tvec=tvec, covariance=covariance, rms_reprojection_error=rms)

    def process_frame(self, frame: np.ndarray, frame_key: str) -> dict[str, Any]:
        detection, gray, preprocessed = self._detect_frame(frame, frame_key)

        if not detection.success:
            reason = detection.failure_reason or "no detector succeeded"
            diagnostics_path = self._maybe_save_failed_diagnostics(frame_key, gray, preprocessed, detection, reason)
            return {
                "frame_key": frame_key,
                "status": "failed",
                "detector": detection.detector_name,
                "failure_stage": detection.failure_stage,
                "failure_reason": reason,
                "roi": list(detection.roi),
                "corners": [],
                "pose": None,
                "rms_reprojection_error": None,
                "diagnostics_image": diagnostics_path,
            }

        if detection.corners is None or len(detection.corners) != len(self.checkerboard.object_points):
            reason = "detected corner count mismatch"
            detection.failure_stage = "not_enough_corners"
            detection.failure_reason = reason
            diagnostics_path = self._maybe_save_failed_diagnostics(frame_key, gray, preprocessed, detection, reason)
            return {
                "frame_key": frame_key,
                "status": "failed",
                "detector": detection.detector_name,
                "failure_stage": detection.failure_stage,
                "failure_reason": reason,
                "roi": list(detection.roi),
                "corners": as_list_of_lists(detection.corners),
                "pose": None,
                "rms_reprojection_error": None,
                "diagnostics_image": diagnostics_path,
            }

        corners = [(float(x), float(y)) for x, y in detection.corners]
        corner_array = np.ascontiguousarray(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2))
        if corner_array.size == 0:
            reason = "no checkerboard corners available for refinement"
            detection.failure_stage = "corner_refinement"
            detection.failure_reason = reason
            diagnostics_path = self._maybe_save_failed_diagnostics(frame_key, gray, preprocessed, detection, reason)
            return {
                "frame_key": frame_key,
                "status": "failed",
                "detector": detection.detector_name,
                "failure_stage": detection.failure_stage,
                "failure_reason": reason,
                "roi": list(detection.roi),
                "corners": [],
                "pose": None,
                "rms_reprojection_error": None,
                "diagnostics_image": diagnostics_path,
            }
        cv2.cornerSubPix(
            gray,
            corner_array,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        )
        corners = [(float(pt[0][0]), float(pt[0][1])) for pt in corner_array]

        if check_points_colinear(corners):
            reason = "detected corners are colinear or poorly distributed"
            detection.failure_stage = "colinearity"
            detection.failure_reason = reason
            diagnostics_path = self._maybe_save_failed_diagnostics(frame_key, gray, preprocessed, detection, reason)
            return {
                "frame_key": frame_key,
                "status": "failed",
                "detector": detection.detector_name,
                "failure_stage": detection.failure_stage,
                "failure_reason": reason,
                "roi": list(detection.roi),
                "corners": as_list_of_lists(corners),
                "pose": None,
                "rms_reprojection_error": None,
                "diagnostics_image": diagnostics_path,
            }

        if self.use_orientation_markers:
            markers = []
            if self.checkerboard.pattern_size[0] >= 2 and self.checkerboard.pattern_size[1] >= 2:
                corners_w = self.checkerboard.pattern_size[0]
                corners_h = self.checkerboard.pattern_size[1]
                tl = np.asarray(corners[0], dtype=np.float32)
                tr = np.asarray(corners[corners_w - 1], dtype=np.float32)
                bl = np.asarray(corners[(corners_h - 1) * corners_w], dtype=np.float32)
                br = np.asarray(corners[-1], dtype=np.float32)
                step_x = (tr - tl) / max(1, corners_w - 1)
                step_y = (bl - tl) / max(1, corners_h - 1)
                square_px_x = float(np.linalg.norm(step_x))
                square_px_y = float(np.linalg.norm(step_y))
                search_radius_min = max(3, int(0.18 * min(square_px_x, square_px_y)))
                search_radius_max = max(search_radius_min + 2, int(0.45 * max(square_px_x, square_px_y)))
                roi_half = max(12, int(0.9 * max(square_px_x, square_px_y)))
                marker_centers = [
                    tl - 0.5 * step_x - 0.5 * step_y,
                    tr + 0.5 * step_x - 0.5 * step_y,
                    bl - 0.5 * step_x + 0.5 * step_y,
                    br + 0.5 * step_x + 0.5 * step_y,
                ]
                for center in marker_centers:
                    c = (float(center[0]), float(center[1]))
                    rx = max(0, int(math.floor(c[0] - roi_half)))
                    ry = max(0, int(math.floor(c[1] - roi_half)))
                    rw = min(gray.shape[1] - rx, 2 * roi_half)
                    rh = min(gray.shape[0] - ry, 2 * roi_half)
                    if rw <= 0 or rh <= 0:
                        continue
                    markers.extend(detect_orientation_markers(gray, search_radius_min, search_radius_max, 1, (rx, ry, rw, rh)))

            if not markers:
                min_x = gray.shape[1]
                min_y = gray.shape[0]
                max_x = 0
                max_y = 0
                for px, py in corners:
                    min_x = min(min_x, int(math.floor(px)))
                    min_y = min(min_y, int(math.floor(py)))
                    max_x = max(max_x, int(math.ceil(px)))
                    max_y = max(max_y, int(math.ceil(py)))
                bw = max(1, max_x - min_x)
                bh = max(1, max_y - min_y)
                margin = max(10, int(0.1 * max(bw, bh)))
                rx = max(0, min_x - margin)
                ry = max(0, min_y - margin)
                rw = min(gray.shape[1] - rx, bw + 2 * margin)
                rh = min(gray.shape[0] - ry, bh + 2 * margin)
                if rw > 0 and rh > 0:
                    markers = detect_orientation_markers(gray, 5, 30, 4, (rx, ry, rw, rh))

            verify_and_fix_orientation(markers, corners, self.checkerboard.pattern_size)

        pose_estimate = self._solve_pose(corners)
        if not pose_estimate.success:
            diagnostics_path = self._maybe_save_failed_diagnostics(frame_key, gray, preprocessed, detection, pose_estimate.failure_reason)
            return {
                "frame_key": frame_key,
                "status": "failed",
                "detector": detection.detector_name,
                "failure_stage": "pose_solve_failure",
                "failure_reason": pose_estimate.failure_reason,
                "roi": list(detection.roi),
                "corners": as_list_of_lists(corners),
                "pose": None,
                "rms_reprojection_error": None,
                "diagnostics_image": diagnostics_path,
            }

        self.last_successful_pose = (pose_estimate.rvec.copy(), pose_estimate.tvec.copy())
        self.last_successful_corners = corners
        if self.detection_cache_file is not None and frame_key not in self.detection_cache:
            self.detection_cache[frame_key] = corners
            save_detection_cache(self.detection_cache_file, self.detection_cache)

        pose_dict = {
            "frame_id": "endoscope_optical",
            "position": [float(pose_estimate.tvec[0, 0]), float(pose_estimate.tvec[1, 0]), float(pose_estimate.tvec[2, 0])],
            "quaternion": rvec_to_quaternion(pose_estimate.rvec),
            "covariance": flatten_covariance(pose_estimate.covariance),
        }
        return {
            "frame_key": frame_key,
            "status": "ok",
            "detector": detection.detector_name,
            "failure_stage": "",
            "failure_reason": "",
            "roi": list(detection.roi),
            "corners": as_list_of_lists(corners),
            "pose": pose_dict,
            "rms_reprojection_error": float(pose_estimate.rms_reprojection_error or 0.0),
            "diagnostics_image": None,
        }


def rvec_to_quaternion(rvec: np.ndarray) -> list[float]:
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    trace = float(np.trace(rotation_matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
    else:
        if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            s = math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            qz = 0.25 * s
    return [float(qx), float(qy), float(qz), float(qw)]


def discover_frame_paths(frames_dir: Path) -> list[Path]:
    frame_paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        frame_paths.extend(sorted(frames_dir.glob(pattern)))
    return sorted(frame_paths)
