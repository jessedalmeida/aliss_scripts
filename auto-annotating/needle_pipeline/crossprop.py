#!/usr/bin/env python3
"""
needle_pipeline.crossprop - segment a bag using ANOTHER bag's seed as a teacher.

Instead of placing clicks on the target bag's own frame, this uses a source
bag's annotated frame (the image + its positive/negative clicks) as a reference,
and lets SAM2's video memory carry that learned needle appearance into the
target bag's frames. This is robust to the needle being in a different pixel
position than the source, because SAM2 matches on appearance, not coordinates.

Mechanism: build a temporary frame sequence [reference_frame, target_f0,
target_f1, ...], place the clicks on frame 0 (the reference), propagate, and
keep only the target frames' masks (output index i>=1 maps to target frame i-1).

Driven by <target_bag>/reference_seed.json:
    {"source_bag": "...", "source_frame_key": "000012",
     "points": [[x,y], ...], "labels": [1,0, ...]}

Run:
    python -m needle_pipeline.crossprop --ann-dir ./annotations --bag target1 \
        --sam2-repo /path/to/sam2 --model large

Reuses load_sam2_predictor / cleanup_binary_mask / collect_frame_paths from the
existing propagate.py, so SAM2 is driven exactly as the normal propagate stage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

REF_SEED_NAME = "reference_seed.json"


def _frame_for_key(frames_dir: Path, key: str, collect) -> Path | None:
    for p in collect(frames_dir):
        import re
        m = re.search(r"(\d+)", p.stem)
        if m and f"{int(m.group(1)):06d}" == key:
            return p
    return None


def crossbag_propagate(target_bag: str, ann_dir: Path, predictor, device: str,
                       threshold: float = 0.0, cleanup: bool = True,
                       min_component_area: int = 100) -> bool:
    """Propagate a reference seed (from another bag) across target_bag's frames."""
    from propagate import collect_frame_paths, cleanup_binary_mask

    target_dir = ann_dir / target_bag
    ref_path = target_dir / REF_SEED_NAME
    if not ref_path.exists():
        print(f"  [SKIP] {target_bag} - no {REF_SEED_NAME}")
        return False
    ref = json.loads(ref_path.read_text())

    src_bag = ref["source_bag"]
    src_frames = ann_dir / src_bag / "frames"
    ref_frame = _frame_for_key(src_frames, ref["source_frame_key"], collect_frame_paths)
    if ref_frame is None:
        print(f"  [ERROR] reference frame {src_bag}:{ref['source_frame_key']} not found")
        return False

    tgt_frames_dir = target_dir / "frames"
    tgt_frames = collect_frame_paths(tgt_frames_dir) if tgt_frames_dir.exists() else []
    if not tgt_frames:
        print(f"  [SKIP] {target_bag} - no frames")
        return False

    masks_dir = target_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    done_sentinel = target_dir / "propagation_done"

    points = np.array(ref.get("points", []), dtype=np.float32)
    labels = np.array(ref.get("labels", []), dtype=np.int32)
    if points.size == 0 or not (labels == 1).any():
        print(f"  [ERROR] reference seed has no positive points")
        return False

    # Build a combined sequence: 000000 = reference frame, 000001.. = target frames.
    tmp = Path(tempfile.mkdtemp(prefix=f"crossprop_{target_bag}_"))
    try:
        # reference frame, normalized to .jpg so SAM2's loader is happy
        ref_img = cv2.imread(str(ref_frame))
        if ref_img is None:
            print(f"  [ERROR] could not read reference frame {ref_frame}")
            return False
        cv2.imwrite(str(tmp / "000000.jpg"), ref_img)
        for i, fp in enumerate(tgt_frames, start=1):
            img = cv2.imread(str(fp))
            if img is None:
                print(f"  [WARN] skip unreadable target frame {fp}")
                continue
            cv2.imwrite(str(tmp / f"{i:06d}.jpg"), img)

        import torch
        autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
                        else torch.autocast("cpu", dtype=torch.float32))
        t0 = time.time()
        with torch.inference_mode(), autocast_ctx:
            print(f"  Teaching from {src_bag}:{ref['source_frame_key']} -> {target_bag} "
                  f"({len(tgt_frames)} frames)")
            state = predictor.init_state(video_path=str(tmp),
                                         offload_video_to_cpu=True,
                                         offload_state_to_cpu=True)
            predictor.reset_state(state)
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=1,
                points=points, labels=labels)

            saved = 0
            for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(state):
                if out_idx == 0:
                    continue  # the reference frame itself - discard
                tgt_i = out_idx - 1
                if tgt_i >= len(tgt_frames):
                    continue
                mask = (out_logits[0] > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255
                if cleanup:
                    mask = cleanup_binary_mask(mask, min_component_area=min_component_area)
                cv2.imwrite(str(masks_dir / f"frame_{tgt_i:06d}_needle_mask.png"), mask)
                saved += 1
        print(f"  ✓ saved {saved} masks in {time.time()-t0:.1f}s -> {masks_dir}")
        done_sentinel.touch()
        _sanity(target_dir, tgt_frames_dir, masks_dir, collect_frame_paths)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _sanity(bag_dir: Path, frames_dir: Path, masks_dir: Path, collect, n: int = 6) -> None:
    frames = collect(frames_dir)
    if not frames:
        return
    picks = frames[:: max(1, len(frames) // n)][:n]
    panels = []
    for fp in picks:
        import re
        m = re.search(r"(\d+)", fp.stem)
        key = f"{int(m.group(1)):06d}" if m else fp.stem
        img = cv2.imread(str(fp))
        mk = masks_dir / f"frame_{key}_needle_mask.png"
        if img is not None and mk.exists():
            mask = cv2.imread(str(mk), cv2.IMREAD_GRAYSCALE)
            overlay = img.copy()
            overlay[mask > 127] = (0, 255, 0)
            panels.append(cv2.addWeighted(img, 0.6, overlay, 0.4, 0))
    if panels:
        h = min(p.shape[0] for p in panels)
        strip = np.hstack([cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels])
        cv2.imwrite(str(bag_dir / "sanity_check.png"), strip)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-bag SAM2 propagation (teach from another bag)")
    ap.add_argument("--ann-dir", required=True)
    ap.add_argument("--bag", required=True, help="target bag (must have reference_seed.json)")
    ap.add_argument("--sam2-repo", required=True)
    ap.add_argument("--model", default="large", choices=["tiny", "small", "base", "large"])
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.sam2_repo)))
    from propagate import load_sam2_predictor, get_device
    device = get_device()
    predictor = load_sam2_predictor(Path(args.sam2_repo), args.model, device)
    ok = crossbag_propagate(args.bag, Path(args.ann_dir), predictor, device)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
