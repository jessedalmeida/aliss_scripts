"""
needle_pipeline.keypoint_ops - suggest tip/tail from a needle mask.

Reuses the skeleton/endpoint helpers from the existing extract_keypoints.py so
the GUI's "suggest from mask" button matches what the batch keypoint stage does.
Pure cv2/numpy - no GPU - runs in the web process. The legacy scripts must be
importable (the server puts --scripts-dir on sys.path at startup).
"""

from __future__ import annotations

from pathlib import Path

import cv2


def suggest_from_mask(mask_path: Path, tip_hint=None, tail_hint=None,
                      prev_tip=None, prev_tail=None) -> dict | None:
    """Return {"needle_tip":[x,y], "needle_tail":[x,y]} suggested from the mask,
    or None if the mask is empty / no endpoints can be found.

    Hints (any may be None) disambiguate which endpoint is tip vs tail:
      tip_hint/tail_hint  - a click the user already placed this frame
      prev_tip/prev_tail  - tip/tail from the previous frame (temporal continuity)
    """
    from extract_keypoints import load_mask, choose_endpoints, order_tip_tail

    mask = load_mask(Path(mask_path))
    if mask is None or not mask.any():
        return None
    pair = choose_endpoints(mask)
    if pair is None:
        return None
    tip, tail = order_tip_tail(
        pair,
        tuple(tip_hint) if tip_hint else None,
        tuple(tail_hint) if tail_hint else None,
        tuple(prev_tip) if prev_tip else None,
        tuple(prev_tail) if prev_tail else None,
    )
    return {"needle_tip": [int(tip[0]), int(tip[1])],
            "needle_tail": [int(tail[0]), int(tail[1])]}
