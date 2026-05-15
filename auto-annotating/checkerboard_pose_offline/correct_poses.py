#!/usr/bin/env python3
"""
Interactive corner-click correction UI for manual pose estimation.
Allows user to click checkerboard corners on a failed frame and automatically
computes pose and covariance from those corners.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np

from checkerboard_pose_offline import (
    CameraModel,
    CheckerboardInfo,
    _solve_pnp_refine,
    compute_pose_covariance,
    pose_rms_reprojection_error,
    flatten_covariance,
    rvec_to_quaternion,
)


class CornerClickUI:
    """Interactive UI for selecting checkerboard corners by clicking."""

    def __init__(
        self,
        image: np.ndarray,
        camera: CameraModel,
        checkerboard: CheckerboardInfo,
        pixel_noise_sigma: float = 10.0,
    ):
        self.image = image.copy()
        self.display = image.copy()
        self.camera = camera
        self.checkerboard = checkerboard
        self.pixel_noise_sigma = pixel_noise_sigma

        self.clicked_corners: list[tuple[int, int]] = []
        self.valid_corners: list[tuple[float, float]] = []
        self.pose_result: dict[str, Any] | None = None
        self.completed = False

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse clicks to record corner positions."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_corners.append((x, y))
            self.redraw()
            print(f"Corner {len(self.clicked_corners)}: ({x}, {y})")

    def redraw(self) -> None:
        """Redraw image with clicked corners and instructions."""
        self.display = self.image.copy()

        expected_corners = self.checkerboard.pattern_size[0] * self.checkerboard.pattern_size[1]
        remaining = max(0, expected_corners - len(self.clicked_corners))

        for i, (x, y) in enumerate(self.clicked_corners):
            cv2.circle(self.display, (x, y), 1, (0, 255, 0), -1)
            cv2.circle(self.display, (x, y), 2, (0, 255, 0), 2)
            cv2.putText(self.display, str(i + 1), (x + 10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        lines = [
            f"Clicked: {len(self.clicked_corners)} corners (expected {expected_corners})",
            "Left-click: Add corner",
            "C: Clear all  |  U: Undo  |  Enter: Solve  |  Esc: Cancel",
        ]
        h, w = self.display.shape[:2]
        panel_h = 100
        overlay = self.display.copy()
        cv2.rectangle(overlay, (0, h - panel_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, self.display, 0.3, 0, self.display)
        for i, line in enumerate(lines):
            cv2.putText(self.display, line, (10, h - panel_h + 28 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    def run(self) -> dict[str, Any] | None:
        """Run the interactive corner picker UI. Returns pose dict or None if cancelled."""
        window_name = "Corner Click UI - Select Checkerboard Corners"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self.redraw()
        cv2.imshow(window_name, self.display)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # Esc
                cv2.destroyWindow(window_name)
                return None
            elif key == ord("c") or key == ord("C"):
                self.clicked_corners = []
                self.redraw()
                cv2.imshow(window_name, self.display)
            elif key == ord("u") or key == ord("U"):
                if self.clicked_corners:
                    self.clicked_corners.pop()
                    self.redraw()
                    cv2.imshow(window_name, self.display)
            elif key == 13:  # Enter
                result = self._solve_and_return()
                if result:
                    self.pose_result = result
                    self.completed = True
                    cv2.destroyWindow(window_name)
                    return result
                else:
                    print("[ERROR] Failed to solve pose. Please ensure corners are correct.")
                    cv2.imshow(window_name, self.display)

            cv2.imshow(window_name, self.display)

        cv2.destroyWindow(window_name)
        return None

    def _solve_and_return(self) -> dict[str, Any] | None:
        """Solve PnP from clicked corners and return pose dict, or None if failed."""
        if len(self.clicked_corners) < 4:
            print(f"[ERROR] Need at least 4 corners, got {len(self.clicked_corners)}")
            return None

        expected_corners = self.checkerboard.pattern_size[0] * self.checkerboard.pattern_size[1]
        if len(self.clicked_corners) != expected_corners:
            print(f"[WARN] Clicked {len(self.clicked_corners)} corners, expected {expected_corners}. Attempting solve anyway...")

        object_points = np.asarray(self.checkerboard.object_points, dtype=np.float64)
        image_points = np.asarray(self.clicked_corners, dtype=np.float64)

        if len(image_points) < len(object_points):
            print(f"[ERROR] Clicked {len(image_points)} corners but need {len(object_points)}")
            return None

        if len(image_points) > len(object_points):
            print(f"[WARN] Clicked {len(image_points)} corners but only need {len(object_points)}. Using first {len(object_points)} corners.")
            image_points = image_points[: len(object_points)]

        result = _solve_pnp_refine(object_points, image_points, self.camera.camera_matrix, self.camera.distortion_coeffs)
        if result is None:
            print("[ERROR] solvePnP failed")
            return None

        rvec, tvec = result

        covariance = compute_pose_covariance(
            object_points,
            image_points,
            rvec,
            tvec,
            self.camera.camera_matrix,
            self.camera.distortion_coeffs,
            self.pixel_noise_sigma,
        )

        rms = pose_rms_reprojection_error(object_points, image_points, rvec, tvec, self.camera.camera_matrix, self.camera.distortion_coeffs)

        pose_dict = {
            "frame_id": "endoscope_optical",
            "position": [float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0])],
            "quaternion": rvec_to_quaternion(rvec),
            "covariance": flatten_covariance(covariance),
        }
        
        result_dict = {
            "status": "ok",
            "pose": pose_dict,
            "detector": "manual_correction",
            "rms_reprojection_error": float(rms) if rms is not None else 0.0,
        }

        print(f"[SUCCESS] Pose computed. RMS reprojection error: {result_dict['rms_reprojection_error']:.3f}px")

        return result_dict


def open_corner_picker(
    frame_image: np.ndarray,
    camera: CameraModel,
    checkerboard: CheckerboardInfo,
    pixel_noise_sigma: float = 10.0,
) -> dict[str, Any] | None:
    """
    Open interactive corner picker UI.
    Args:
        frame_image: Input image (BGR or grayscale)
        camera: CameraModel with calibration
        checkerboard: CheckerboardInfo with pattern details
        pixel_noise_sigma: Pixel noise for covariance computation
    Returns:
        Dict with keys: status, pose (containing frame_id, position, quaternion, covariance), 
        detector, rms_reprojection_error. Or None if user cancelled.
    """
    if frame_image.ndim == 2:
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_GRAY2BGR)

    ui = CornerClickUI(frame_image, camera, checkerboard, pixel_noise_sigma)
    return ui.run()
