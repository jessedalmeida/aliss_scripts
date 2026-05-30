#!/usr/bin/env python3
"""
02_propagate.py  —  Run on your GPU machine (CUDA)
====================================================
Reads the seeds.json files produced by 01_seed_annotator.py and runs SAM2
video propagation to generate per-frame segmentation masks.

OUTPUT
------
For each bag processed:
  <output_dir>/<bag_stem>/masks/frame_XXXXXX_needle_mask.png   — binary mask (0/255)
  <output_dir>/<bag_stem>/propagation_done                     — sentinel file

USAGE
-----
# Process all annotated bags under a directory
python 02_propagate.py --ann-dir ./annotations --sam2-repo /path/to/sam2

# Process a single bag
python 02_propagate.py --ann-dir ./annotations --bag suture1 --sam2-repo /path/to/sam2

# Use a smaller/faster model
python 02_propagate.py --ann-dir ./annotations --sam2-repo /path/to/sam2 --model small

DEPENDENCIES (GPU machine)
--------------------------
  # Clone SAM2 and install
  git clone https://github.com/facebookresearch/sam2.git
  cd sam2
  pip install -e .
  # Download checkpoints
  cd checkpoints && bash download_ckpts.sh

  # Also need:
  pip install opencv-python numpy tqdm

SAM2 MODEL OPTIONS
------------------
  tiny   — fastest, least accurate  (sam2.1_hiera_tiny)
  small  — good balance             (sam2.1_hiera_small)
  base   — better accuracy          (sam2.1_hiera_base_plus)
  large  — best accuracy, slowest   (sam2.1_hiera_large)   [DEFAULT]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
#  MODEL CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml",  "checkpoints/sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml",  "checkpoints/sam2.1_hiera_small.pt"),
    "base":  ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml",  "checkpoints/sam2.1_hiera_large.pt"),
}


# ════════════════════════════════════════════════════════════════════════════
#  DEVICE DETECTION
# ════════════════════════════════════════════════════════════════════════════

def get_device() -> str:
    if not TORCH_AVAILABLE:
        print("[ERROR] PyTorch not installed.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: CUDA — {name}  ({mem:.1f} GB VRAM)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  Device: Apple Silicon MPS (will be slower than CUDA)")
    else:
        device = "cpu"
        print("  Device: CPU (very slow — consider using a GPU)")

    return device


def frame_index_from_path(path: Path) -> int | None:
    """Extract the numeric frame index from a filename stem."""
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else None


def collect_frame_paths(frames_dir: Path) -> list[Path]:
    """Collect frame images in numeric order, deduplicating by frame index."""
    candidates: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        candidates.extend(frames_dir.glob(pattern))

    ordered: dict[int, Path] = {}
    fallback: list[Path] = []
    for path in candidates:
        idx = frame_index_from_path(path)
        if idx is None:
            fallback.append(path)
        elif idx not in ordered:
            ordered[idx] = path

    return [ordered[idx] for idx in sorted(ordered)] + sorted(set(fallback))


def clear_gpu_memory(device: str):
    """Clear GPU cache and collect garbage to free memory between bags."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()


# ════════════════════════════════════════════════════════════════════════════
#  SAM2 LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_sam2_predictor(sam2_repo: Path, model_size: str, device: str):
    """Load the SAM2 video predictor from a local repo clone."""
    if str(sam2_repo) not in sys.path:
        sys.path.insert(0, str(sam2_repo))

    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam2.build_sam import _load_checkpoint
    except ImportError as e:
        print(f"[ERROR] Cannot import sam2. Make sure --sam2-repo points to the cloned repo: {e}")
        sys.exit(1)

    cfg_rel, ckpt_rel = MODEL_CONFIGS[model_size]
    cfg_name = cfg_rel.removeprefix("configs/").removeprefix("sam2/configs/")
    ckpt_path = sam2_repo / ckpt_rel

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("  Run:  cd <sam2_repo>/checkpoints && bash download_ckpts.sh")
        sys.exit(1)

    print(f"  Loading SAM2 ({model_size}) from {ckpt_path.name}…")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(sam2_repo / "sam2" / "configs"), version_base="1.2")
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
    ]
    hydra_overrides += [
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        "++model.binarize_mask_from_pts_for_mem_enc=true",
        "++model.fill_hole_area=8",
    ]

    cfg = compose(config_name=cfg_name, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    predictor = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(predictor, str(ckpt_path))
    predictor = predictor.to(device)
    predictor.eval()
    print("  ✓ Model loaded.")
    return predictor


def cleanup_binary_mask(mask: np.ndarray, min_component_area: int = 200) -> np.ndarray:
    """Lightly clean a binary mask by removing tiny islands and filling small holes."""
    binary = (mask > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return binary

    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_labels = [idx + 1 for idx, area in enumerate(areas) if area >= min_component_area]
    if not keep_labels:
        keep_labels = [1 + int(np.argmax(areas))]

    cleaned = np.zeros_like(binary)
    for label in keep_labels:
        cleaned[labels == label] = 255

    flood = cleaned.copy()
    flood_mask = np.zeros((cleaned.shape[0] + 2, cleaned.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    cleaned = cleaned | cv2.bitwise_not(flood)
    return cleaned


def ensure_jpg_frames(src_dir: Path) -> Path:
    """Return a folder path with JPG frames, renaming to numeric stems.

    Collects common image files in `src_dir` and writes numeric-named JPGs
    into a sibling folder `<src_dir>_jpg`. Returns the path to use for SAM2.
    """
    if not src_dir.exists():
        return src_dir

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    files = []
    for pat in patterns:
        files.extend(sorted(src_dir.glob(pat)))

    if not files:
        return src_dir

    out_dir = src_dir.with_name(src_dir.name + "_jpg")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(sorted(files)):
        img = cv2.imread(str(p))
        if img is None:
            continue
        out_p = out_dir / f"{i:06d}.jpg"
        cv2.imwrite(str(out_p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return out_dir


# ════════════════════════════════════════════════════════════════════════════
#  PROPAGATION
# ════════════════════════════════════════════════════════════════════════════

def propagate_bag(
    bag_stem: str,
    ann_dir: Path,
    predictor,
    device: str,
    threshold: float = 0.0,
    cleanup_mask: bool = False,
    min_component_area: int = 100,
):
    """
    Run SAM2 propagation for one bag and save mask PNGs.

    Args:
        bag_stem:   name of the bag subdirectory under ann_dir
        ann_dir:    root annotation directory (output from 01_seed_annotator)
        predictor:  loaded SAM2 video predictor
        device:     "cuda", "mps", or "cpu"
        threshold:  logit threshold for mask binarisation (0.0 is standard)
        cleanup_mask: If True, lightly clean masks before writing them to disk.
        min_component_area: Minimum connected-component area kept when cleaning.
    """
    bag_dir = ann_dir / bag_stem
    seeds_path = bag_dir / "seeds.json"
    frames_dir = bag_dir / "frames"
    masks_dir = bag_dir / "masks"
    done_sentinel = bag_dir / "propagation_done"

    if done_sentinel.exists():
        print(f"  [SKIP] {bag_stem} — already propagated (delete 'propagation_done' to redo)")
        return True

    if not seeds_path.exists():
        print(f"  [SKIP] {bag_stem} — no seeds.json found")
        return False

    # Accept common image extensions (png/jpg/jpeg). If none present, skip.
    imgs = collect_frame_paths(frames_dir) if frames_dir.exists() else []
    if not frames_dir.exists() or not imgs:
        print(f"  [SKIP] {bag_stem} — no frames found in {frames_dir}")
        return False

    with open(seeds_path) as f:
        seeds = json.load(f)

    frame_paths = collect_frame_paths(frames_dir)
    n_frames = len(frame_paths)

    # Support legacy single-seed format or new multi-seed format.
    if "seed_frames" in seeds:
        seed_entries = seeds["seed_frames"]
    else:
        seed_entries = [{
            "seed_frame_idx": int(seeds.get("seed_frame_idx", 0)),
            "objects": seeds.get("objects", {}),
        }]

    # (objects presence is validated after collecting labels below)

    masks_dir.mkdir(parents=True, exist_ok=True)

    seed_idxs = [int(e.get("seed_frame_idx", 0)) for e in seed_entries]
    print(f"  Frames: {n_frames}  |  Seed frames: {seed_idxs}")
    # Collect union of object labels across seeds for reporting
    all_labels = set()
    for e in seed_entries:
        all_labels.update(e.get("objects", {}).keys())
    print(f"  Objects to propagate: {list(all_labels)}")
    if not all_labels:
        print(f"  [SKIP] {bag_stem} — seeds.json has no object prompts")
        return False

    # ── Set up autocast ────────────────────────────────────────────────────
    import torch
    if device == "cuda":
        # Enable TF32 for Ampere+ GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif device == "mps":
        autocast_ctx = torch.autocast("cpu", dtype=torch.float32)  # MPS fallback
    else:
        autocast_ctx = torch.autocast("cpu", dtype=torch.float32)

    t_start = time.time()

    with torch.inference_mode(), autocast_ctx:
        print("  Initialising SAM2 state…")
        # frames_for_sam = ensure_jpg_frames(frames_dir)
        # We'll propagate from each seed entry independently, then merge per-frame masks
        video_segments_best: dict[int, dict[str, tuple[int, np.ndarray, bool]]] = {}

        for entry in seed_entries:
            seed_frame_idx = int(entry.get("seed_frame_idx", 0))
            objects_data = entry.get("objects", {})

            print(f"  Initialising SAM2 state for seed frame {seed_frame_idx}…")
            inference_state = predictor.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
            )
            predictor.reset_state(inference_state)

            # Map labels to local obj ids for this seed
            obj_id_map = {}
            for obj_idx, (obj_label, obj_data) in enumerate(objects_data.items()):
                obj_id = obj_idx + 1
                obj_id_map[obj_label] = obj_id

                points = np.array(obj_data.get("points", []), dtype=np.float32)
                labels = np.array(obj_data.get("labels", []), dtype=np.int32)

                if points.size == 0:
                    continue

                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=seed_frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

                # Save the seed-frame mask immediately for verification
                for i, oid in enumerate(out_obj_ids):
                    if oid == obj_id:
                        seed_mask = (out_mask_logits[i] > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255
                        if cleanup_mask:
                            seed_mask = cleanup_binary_mask(seed_mask, min_component_area=min_component_area)
                        if seed_frame_idx not in video_segments_best:
                            video_segments_best[seed_frame_idx] = {}
                        # store (src_seed, mask, is_seed_frame=True) — the explicit
                        # click prompt is authoritative and must outrank any
                        # propagated mask that lands on this frame from another seed.
                        video_segments_best[seed_frame_idx][obj_label] = (seed_frame_idx, seed_mask, True)

            # ── Propagate forward for this seed ─────────────────────────────────
            print(f"  Propagating forward from seed {seed_frame_idx}…")
            iter_ = predictor.propagate_in_video(inference_state)
            if TQDM_AVAILABLE:
                iter_ = tqdm(iter_, total=n_frames, desc="  Forward", unit="frame")
            for out_frame_idx, out_obj_ids, out_mask_logits in iter_:
                for i, oid in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255
                    if cleanup_mask:
                        mask = cleanup_binary_mask(mask, min_component_area=min_component_area)
                    label = next((lbl for lbl, mid in obj_id_map.items() if mid == oid), f"obj_{oid}")
                    # if we already have a mask for this frame+label, keep the one from the nearer seed
                    prev = video_segments_best.get(out_frame_idx, {}).get(label)
                    replace = True
                    if prev is not None:
                        prev_seed_idx, _, prev_is_seed = prev
                        # an explicit seed-frame mask is authoritative — never overwrite it
                        if prev_is_seed:
                            replace = False
                        elif abs(out_frame_idx - prev_seed_idx) <= abs(out_frame_idx - seed_frame_idx):
                            replace = False
                    if replace:
                        if out_frame_idx not in video_segments_best:
                            video_segments_best[out_frame_idx] = {}
                        video_segments_best[out_frame_idx][label] = (seed_frame_idx, mask, False)

        # ── Propagate backward from seed frame ────────────────────────────
            if seed_frame_idx > 0:
                print(f"  Propagating backward from frame {seed_frame_idx}…")
                iter_rev = predictor.propagate_in_video(inference_state, reverse=True)
                if TQDM_AVAILABLE:
                    iter_rev = tqdm(iter_rev, total=seed_frame_idx + 1, desc="  Backward", unit="frame")

                for out_frame_idx, out_obj_ids, out_mask_logits in iter_rev:
                    for i, oid in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > threshold).squeeze().cpu().numpy().astype(np.uint8) * 255
                        if cleanup_mask:
                            mask = cleanup_binary_mask(mask, min_component_area=min_component_area)
                        label = next((lbl for lbl, mid in obj_id_map.items() if mid == oid), f"obj_{oid}")
                        prev = video_segments_best.get(out_frame_idx, {}).get(label)
                        replace = True
                        if prev is not None:
                            prev_seed_idx, _, prev_is_seed = prev
                            if prev_is_seed:
                                replace = False
                            elif abs(out_frame_idx - prev_seed_idx) <= abs(out_frame_idx - seed_frame_idx):
                                replace = False
                        if replace:
                            if out_frame_idx not in video_segments_best:
                                video_segments_best[out_frame_idx] = {}
                            video_segments_best[out_frame_idx][label] = (seed_frame_idx, mask, False)

        # After processing all seeds, convert best entries to plain masks
        video_segments: dict[int, dict[str, np.ndarray]] = {}
        for fidx, labels in video_segments_best.items():
            video_segments[fidx] = {}
            for lbl, entry in labels.items():
                src_seed, mask = entry[0], entry[1]
                video_segments[fidx][lbl] = mask

    # ── Save all masks ─────────────────────────────────────────────────────
    print("  Saving masks…")
    saved_count = 0
    seed_frame_set = {int(e.get("seed_frame_idx", 0)) for e in seed_entries}
    for frame_idx in sorted(video_segments.keys()):
        for label, mask in video_segments[frame_idx].items():
            fname = masks_dir / f"frame_{frame_idx:06d}_{label}.png"
            cv2.imwrite(str(fname), mask)
            saved_count += 1
        # Diagnostic: report which seed each seed-frame's mask actually came from.
        if frame_idx in seed_frame_set:
            prov = video_segments_best.get(frame_idx, {})
            for label, entry in prov.items():
                src_seed, _, is_seed = entry
                tag = "authoritative click" if is_seed else f"PROPAGATED from seed {src_seed}"
                if not is_seed or src_seed != frame_idx:
                    print(f"    [seed-frame {frame_idx:06d}/{label}] mask source: {tag}"
                          + ("  <-- unexpected; expected its own click mask" if src_seed != frame_idx else ""))

    elapsed = time.time() - t_start
    fps = n_frames / elapsed if elapsed > 0 else 0

    print(f"  ✓ Saved {saved_count} masks in {elapsed:.1f}s  ({fps:.1f} frames/s)")
    done_sentinel.touch()

    # ── Quick sanity check visualisation ──────────────────────────────────
    _save_sanity_strip(bag_dir, frames_dir, masks_dir, video_segments)

    return True


def _save_sanity_strip(bag_dir: Path, frames_dir: Path, masks_dir: Path,
                        video_segments: dict, n_samples: int = 8):
    """Save a contact-sheet PNG with frame + mask overlay for quick visual QC."""
    frame_idxs = sorted(video_segments.keys())
    if not frame_idxs:
        return

    # Pick evenly-spaced sample frames
    step = max(1, len(frame_idxs) // n_samples)
    sampled = frame_idxs[::step][:n_samples]

    panels = []
    for idx in sampled:
        frame_path = frames_dir / f"frame_{idx:06d}.png"
        if not frame_path.exists():
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Overlay all masks for this frame
        overlay = img.copy()
        colours = [(0, 255, 100), (255, 100, 0), (100, 0, 255)]
        for col_idx, (label, mask) in enumerate(video_segments[idx].items()):
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            col = colours[col_idx % len(colours)]
            colour_layer = np.zeros_like(overlay)
            colour_layer[mask > 0] = col
            overlay = cv2.addWeighted(overlay, 0.7, colour_layer, 0.3, 0)

        # Resize to small thumbnail
        thumb = cv2.resize(overlay, (320, 240))
        cv2.putText(thumb, f"f{idx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        panels.append(thumb)

    if panels:
        strip = np.hstack(panels)
        out_path = bag_dir / "sanity_check.png"
        cv2.imwrite(str(out_path), strip)
        print(f"  QC strip saved → {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SAM2 propagation for needle tracking bags")
    parser.add_argument("--ann-dir", required=True,
                        help="Annotation root directory (output from 01_seed_annotator.py)")
    parser.add_argument("--sam2-repo", required=True,
                        help="Path to cloned facebookresearch/sam2 repository")
    parser.add_argument("--bag", default=None,
                        help="Process only this bag stem (default: all in ann-dir)")
    parser.add_argument("--model", default="large",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="SAM2 model size (default: large)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Mask logit threshold (default: 0.0)")
    parser.add_argument("--cleanup-mask", action="store_true",
                        help="Lightly clean masks before saving them")
    parser.add_argument("--min-component-area", type=int, default=200,
                        help="Minimum component area kept when --cleanup-mask is enabled")
    parser.add_argument("--device", default=None,
                        help="Force device: cuda / mps / cpu (default: auto-detect)")
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    sam2_repo = Path(args.sam2_repo)

    if not ann_dir.exists():
        print(f"[ERROR] ann-dir does not exist: {ann_dir}")
        sys.exit(1)

    if not sam2_repo.exists():
        print(f"[ERROR] sam2-repo does not exist: {sam2_repo}")
        sys.exit(1)

    # ── Device ────────────────────────────────────────────────────────────
    device = args.device if args.device else get_device()

    # ── Find bags to process ──────────────────────────────────────────────
    if args.bag:
        bag_stems = [args.bag]
    else:
        bag_stems = sorted(
            d.name for d in ann_dir.iterdir()
            if d.is_dir() and (d / "seeds.json").exists()
        )

    if not bag_stems:
        print(f"[ERROR] No bags with seeds.json found in {ann_dir}")
        sys.exit(1)

    print(f"\nFound {len(bag_stems)} bag(s) to propagate: {bag_stems}")

    # ── Process each bag ──────────────────────────────────────────────────
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for bag_stem in bag_stems:
        print(f"\n{'='*60}")
        print(f"BAG: {bag_stem}")
        print(f"{'='*60}")

        # Clear GPU memory before loading model for this bag
        clear_gpu_memory(device)

        # Load model fresh for each bag (avoids OOM from memory fragmentation)
        predictor = load_sam2_predictor(sam2_repo, args.model, device)

        result = propagate_bag(
            bag_stem=bag_stem,
            ann_dir=ann_dir,
            predictor=predictor,
            device=device,
            threshold=args.threshold,
            cleanup_mask=args.cleanup_mask,
            min_component_area=args.min_component_area,
        )

        # Clean up predictor after bag is done
        del predictor
        clear_gpu_memory(device)

        if result is True:
            n_ok += 1
        elif result == "skip":
            n_skip += 1
        else:
            n_fail += 1

    print(f"\n{'='*60}")
    print(f"DONE — {n_ok} propagated, {n_skip} skipped, {n_fail} failed")
    print(f"{'='*60}")
    print("\nNext step: run 03_repack_bag.py to write masks back into .mcap files")


if __name__ == "__main__":
    main()
