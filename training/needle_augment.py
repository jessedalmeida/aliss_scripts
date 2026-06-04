#!/usr/bin/env python3
"""needle_augment.py - Augmentation for needle mask + keypoint training.

Operates on RAW arrays (image, mask, keypoint coords, visibility) BEFORE
heatmap rendering, so geometry stays exact and visibility can be recomputed.

Geometric (image + mask + keypoints in lockstep):
    - horizontal flip
    - rotation over a configurable range (default full 360 deg: the needle
      appears at every orientation in the data, so the model must be
      orientation-invariant)
    - scale + translation jitter

Photometric (image only):
    - hue / saturation jitter   (breaks the green checkerboard-print tail cue)
    - brightness / contrast     (endoscope illumination varies a lot)
    - additive Gaussian noise   (grainy sensor)

FOV-aware visibility (PARAMETERIZABLE via fov_aware):
    A keypoint is marked not-visible after a transform if it leaves the image
    bounds OR leaves the circular endoscope field of view. This matters while
    the FOV is not yet auto-cropped/centered: rotation about the image center
    can push an in-FOV point out of an OFF-center field. Once an upstream
    auto-crop guarantees the FOV is centered, centered rotation preserves
    distance-from-center and cannot move an in-FOV point out -- set
    fov_aware=False then to skip the check.

    Visibility is only ever turned OFF here (a point that rotates out of view
    becomes unsupervised); it is never turned on. Real `occluded` labels from
    the annotation pipeline compose with this by AND -- geometric out-of-view
    and labeled-occluded stay distinct concepts.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_fov_mask(image: np.ndarray, thresh: int = 15) -> np.ndarray:
    """Binary mask of the lit circular field (True inside the endoscope view).

    Derived from the image rather than a hardcoded radius, since the FOV is not
    yet guaranteed centered. Threshold the near-black border, fill, keep largest
    component.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    fov = (gray > thresh).astype(np.uint8)
    # close small gaps so the lit field is one blob, then keep the largest
    # component. Interior bright spots (specular highlights) are already > thresh,
    # so no separate hole-fill is needed.
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fov)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fov = (lab == largest)
    return fov.astype(bool)


class NeedleAugmentation:
    def __init__(
        self,
        rot_deg: float = 180.0,        # rotation drawn from [-rot_deg, +rot_deg]; 180 => full circle
        flip_p: float = 0.5,
        scale_range: tuple[float, float] = (0.85, 1.15),
        translate_frac: float = 0.05,  # fraction of image size
        hue_deg: float = 10.0,
        sat_scale: float = 0.2,        # +/- fraction
        bright_scale: float = 0.25,
        contrast_scale: float = 0.25,
        noise_std: float = 8.0,        # in 0-255 units
        fov_aware: bool = True,
        fov_thresh: int = 15,
        p_geom: float = 1.0,
        seed: int | None = None,
    ):
        self.rot_deg = rot_deg
        self.flip_p = flip_p
        self.scale_range = scale_range
        self.translate_frac = translate_frac
        self.hue_deg = hue_deg
        self.sat_scale = sat_scale
        self.bright_scale = bright_scale
        self.contrast_scale = contrast_scale
        self.noise_std = noise_std
        self.fov_aware = fov_aware
        self.fov_thresh = fov_thresh
        self.p_geom = p_geom
        self.rng = np.random.RandomState(seed)

    # ---- geometric ---- #
    def _affine(self, image, mask, fov, kps, S):
        cx = cy = S / 2.0
        angle = self.rng.uniform(-self.rot_deg, self.rot_deg)
        scale = self.rng.uniform(*self.scale_range)
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        t = self.translate_frac * S
        M[0, 2] += self.rng.uniform(-t, t)
        M[1, 2] += self.rng.uniform(-t, t)

        image = cv2.warpAffine(image, M, (S, S), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, M, (S, S), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fov = cv2.warpAffine(fov.astype(np.uint8), M, (S, S), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        # kps: (K,2) -> homogeneous
        ones = np.ones((kps.shape[0], 1), np.float32)
        kps_h = np.hstack([kps, ones])
        kps_t = (M @ kps_h.T).T  # (K,2)
        return image, mask, fov, kps_t.astype(np.float32)

    def _maybe_flip(self, image, mask, fov, kps, S):
        if self.rng.rand() < self.flip_p:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            fov = fov[:, ::-1].copy()
            kps = kps.copy()
            kps[:, 0] = (S - 1) - kps[:, 0]
        return image, mask, fov, kps

    # ---- photometric (image only) ---- #
    def _photometric(self, image):
        img = image.astype(np.float32)
        # hue/sat in HSV
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + self.rng.uniform(-self.hue_deg, self.hue_deg) / 2.0) % 180.0
        hsv[..., 1] *= 1.0 + self.rng.uniform(-self.sat_scale, self.sat_scale)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        # brightness (additive) + contrast (multiplicative about mean)
        img *= 1.0 + self.rng.uniform(-self.bright_scale, self.bright_scale)
        c = 1.0 + self.rng.uniform(-self.contrast_scale, self.contrast_scale)
        img = (img - img.mean()) * c + img.mean()
        # noise
        if self.noise_std > 0:
            img += self.rng.randn(*img.shape) * self.noise_std
        return np.clip(img, 0, 255).astype(np.uint8)

    # ---- visibility recompute ---- #
    def _update_visibility(self, kps, vis, fov, S):
        vis = vis.copy()
        for k in range(kps.shape[0]):
            if vis[k] <= 0:
                continue
            x, y = kps[k]
            in_bounds = (0 <= x < S) and (0 <= y < S)
            if not in_bounds:
                vis[k] = 0.0
                continue
            if self.fov_aware and not fov[int(round(y)), int(round(x))]:
                vis[k] = 0.0
        return vis

    def __call__(self, raw: dict) -> dict:
        """raw: image (S,S,3 uint8), mask (S,S float), kps (K,2 float; NaN if absent),
        vis (K,) float, labeled (K,) float.  Returns same structure, augmented."""
        image, mask, kps = raw["image"], raw["mask"], raw["kps"].copy()
        vis, labeled = raw["vis"], raw["labeled"]
        S = image.shape[0]

        fov = detect_fov_mask(image, self.fov_thresh) if self.fov_aware else np.ones(image.shape[:2], bool)

        # NaN-safe: park absent kps at center for warping, they stay unsupervised
        absent = ~np.isfinite(kps).all(axis=1)
        kps_safe = kps.copy()
        kps_safe[absent] = S / 2.0

        if self.rng.rand() < self.p_geom:
            image, mask, fov, kps_safe = self._maybe_flip(image, mask, fov, kps_safe, S)
            image, mask, fov, kps_safe = self._affine(image, mask, fov, kps_safe, S)

        vis = self._update_visibility(kps_safe, vis, fov, S)
        kps_safe[absent] = np.nan  # restore absent markers

        image = self._photometric(image)

        return {"image": image, "mask": mask, "kps": kps_safe,
                "vis": vis, "labeled": labeled}
