#!/usr/bin/env python3
"""
01_seed_annotator.py  —  Run on your Mac (no GPU needed)
=========================================================
Extracts frames from one or more ROS2 .mcap bag files and presents a
click-based UI for placing SAM2 seed prompts (positive / negative clicks).

OUTPUT
------
For each bag processed, writes:
  <output_dir>/<bag_stem>/frames/        — extracted PNGs (subsampled)
  <output_dir>/<bag_stem>/seeds.json     — click prompts + metadata

The seeds.json is consumed by 02_propagate.py on the GPU machine.

USAGE
-----
# Single bag
python 01_seed_annotator.py --bag /path/to/suture1 --out ./annotations

# Open a file picker instead of typing paths
python 01_seed_annotator.py --select-bag --out ./annotations

# Multiple bags (glob)
python 01_seed_annotator.py --bag /path/to/bags/*.mcap --out ./annotations

# Skip extraction if frames already exist (re-annotate only)
python 01_seed_annotator.py --bag /path/to/suture1 --out ./annotations --skip-extract

# Edit an existing seeds.json in place
python 01_seed_annotator.py --bag /path/to/suture1 --out ./annotations --skip-extract --edit-existing

DEPENDENCIES (Mac)
------------------
  pip install opencv-python numpy rosbags tqdm

The rosbags library reads mcap files without a full ROS2 installation.

CONTROLS (in the annotation window)
------------------------------------
  Left-click          Add positive prompt (foreground — the needle)
  Right-click         Add negative prompt (background — not the needle)
  Middle-click        Remove nearest prompt
  Z                   Undo last prompt
  C                   Clear all prompts for this object
  ENTER / SPACE       Accept and move on
  Q / ESC             Quit without saving remaining bags

OBJECTS ANNOTATED (in order)
-----------------------------
  1. needle_mask      — the suturing needle
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

# ── Optional: rosbags for mcap reading ──────────────────────────────────────
try:
    from rosbags.highlevel import AnyReader
    ROSBAGS_AVAILABLE = True
except ImportError:
    ROSBAGS_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

IMAGE_TOPICS = [
    "/ves_camera/image_rect",   # preferred (already rectified)
    "/ves_camera/image",        # fallback
]

OBJECTS = ["needle_mask"]   # annotation order

# Colours for rendering prompts (BGR)
COL_POS = (0, 255, 0)      # green  — foreground click
COL_NEG = (0, 0, 255)      # red    — background click
COL_RADIUS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Preview mask tuning.
PREVIEW_GC_ITERS = 3
PREVIEW_COLOR = (0, 255, 255)  # cyan-ish preview overlay


# ════════════════════════════════════════════════════════════════════════════
#  FRAME EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

def extract_frames_from_bag(bag_path: Path, out_dir: Path, every_n: int = 3) -> list[Path]:
    """
    Read image messages from the bag and save every Nth frame as a PNG.
    Returns sorted list of saved frame paths.
    """
    if not ROSBAGS_AVAILABLE:
        print("[ERROR] rosbags not installed. Run:  pip install rosbags")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    frame_idx = 0
    saved_idx = 0

    # rosbags works on the bag directory (rosbag2 metadata folder)
    bag_dir = bag_path if bag_path.is_dir() else bag_path.parent

    print(f"  Scanning bag: {bag_dir}")

    with AnyReader([bag_dir]) as reader:
        # Pick whichever image topic is present
        available = set(reader.topics)
        topic = next((t for t in IMAGE_TOPICS if t in available), None)
        if topic is None:
            print(f"  [WARN] No image topic found in {bag_dir}. Available: {available}")
            return []

        print(f"  Using topic: {topic}")

        connections = [c for c in reader.connections if c.topic == topic]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if frame_idx % every_n == 0:
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    # Convert ROS Image → numpy BGR
                    img = ros_image_to_cv2(msg)
                    if img is None:
                        frame_idx += 1
                        continue
                    fname = out_dir / f"{saved_idx:06d}.jpg"
                    cv2.imwrite(str(fname), img)
                    saved.append(fname)
                    saved_idx += 1
                    if saved_idx % 50 == 0:
                        print(f"    Saved {saved_idx} frames…")
                except Exception as e:
                    print(f"  [WARN] Frame {frame_idx} decode error: {e}")
            frame_idx += 1

    print(f"  Extracted {len(saved)} frames (every {every_n} from {frame_idx} total)")
    return sorted(saved)


def ros_image_to_cv2(msg) -> np.ndarray | None:
    """Convert a sensor_msgs/Image message to a BGR numpy array."""
    encoding = msg.encoding.lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)

    try:
        if encoding in ("bgr8", "rgb8", "bgra8", "rgba8"):
            channels = 4 if "a" in encoding else 3
            img = data.reshape((h, w, channels))
            if encoding.startswith("rgb"):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif encoding == "bgra8":
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif encoding == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif encoding == "mono8":
            img = data.reshape((h, w))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif encoding == "16uc1":
            img = data.view(np.uint16).reshape((h, w))
            img = (img / 256).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            # Attempt generic reshape
            step = msg.step
            img = data.reshape((h, step))[:, :w * 3].reshape((h, w, 3))
    except Exception:
        return None


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

    return img


# ════════════════════════════════════════════════════════════════════════════
#  ANNOTATION UI
# ════════════════════════════════════════════════════════════════════════════

class AnnotationSession:
    """
    Interactive click-prompt collector for a single frame sequence.
    Maintains separate prompt lists for each object label.
    """

    def __init__(self, frame_paths: list[Path], bag_stem: str, existing_seeds: dict | None = None):
        self.frame_paths = frame_paths
        self.bag_stem = bag_stem

        # prompts[obj_label] = list of {"x": int, "y": int, "label": 1|0}
        self.prompts: dict[str, list[dict]] = {obj: [] for obj in OBJECTS}
        self.seed_frame_idx = 0         # which frame to seed from (default 0)
        self.active_click_label = 1     # 1=positive, 0=negative
        self.done = False
        self.quit_all = False

        if existing_seeds:
            self.seed_frame_idx = int(existing_seeds.get("seed_frame_idx", 0))
            existing_objects = existing_seeds.get("objects", {})
            obj_data = existing_objects.get("needle_mask")
            if obj_data:
                points = obj_data.get("points", [])
                labels = obj_data.get("labels", [])
                self.prompts["needle_mask"] = [
                    {"x": int(pt[0]), "y": int(pt[1]), "label": int(lbl)}
                    for pt, lbl in zip(points, labels)
                ]

        # Load the seed frame image
        self.seed_frame_idx = max(0, min(self.seed_frame_idx, len(self.frame_paths) - 1))
        self.seed_image = cv2.imread(str(self.frame_paths[self.seed_frame_idx]))
        if self.seed_image is None:
            raise RuntimeError(f"Cannot read seed frame: {self.frame_paths[self.seed_frame_idx]}")

        self.display_image = self.seed_image.copy()
        self.preview_mask: np.ndarray | None = None
        self.preview_visible = False
        self.preview_dirty = True
        self.window_name = f"SAM2 Seeder - {bag_stem}"
        # Collected multi-seed entries: list of {"seed_frame_idx": int, "objects": {...}}
        self.collected_seeds: list[dict] = []

    @property
    def current_obj(self) -> str:
        return "needle_mask"

    def current_click_mode(self) -> str:
        return "positive" if self.active_click_label == 1 else "negative"

    def _build_preview_mask(self) -> np.ndarray:
        """Build a quick segmentation preview from current clicks.

        Uses GrabCut seeded by positive/negative clicks on the seed frame.
        This is only for live feedback while annotating.
        """
        h, w = self.seed_image.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((0, 0), dtype=np.uint8)

        prompts = self.prompts["needle_mask"]
        if not prompts:
            return np.zeros((h, w), dtype=np.uint8)

        pos_points = [(int(p["x"]), int(p["y"])) for p in prompts if p["label"] == 1]
        neg_points = [(int(p["x"]), int(p["y"])) for p in prompts if p["label"] == 0]
        if not pos_points:
            return np.zeros((h, w), dtype=np.uint8)

        # Seed GrabCut with a rectangle around positives, expanded a bit.
        xs = [p[0] for p in pos_points]
        ys = [p[1] for p in pos_points]
        pad = max(20, int(0.12 * max(h, w)))
        x0 = max(0, min(xs) - pad)
        y0 = max(0, min(ys) - pad)
        x1 = min(w - 1, max(xs) + pad)
        y1 = min(h - 1, max(ys) + pad)

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[y0:y1 + 1, x0:x1 + 1] = cv2.GC_PR_FGD
        for x, y in neg_points:
            cv2.circle(gc_mask, (x, y), 8, cv2.GC_BGD, -1)
        for x, y in pos_points:
            cv2.circle(gc_mask, (x, y), 4, cv2.GC_FGD, -1)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(self.seed_image, gc_mask, None, bgd_model, fgd_model, PREVIEW_GC_ITERS, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return np.zeros((h, w), dtype=np.uint8)

        mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if not np.any(mask):
            return mask

        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _refresh_preview(self):
        """Recompute the preview mask from current clicks when requested."""
        self.preview_mask = self._build_preview_mask()
        self.preview_dirty = False

    def _render(self):
        """Redraw the display image with prompts and optional preview overlay."""
        vis = self.seed_image.copy()

        if self.preview_visible and self.preview_mask is not None and np.any(self.preview_mask):
            preview_layer = np.zeros_like(vis)
            preview_layer[self.preview_mask > 0] = PREVIEW_COLOR
            blended = cv2.addWeighted(vis, 0.75, preview_layer, 0.25, 0)
            vis[self.preview_mask > 0] = blended[self.preview_mask > 0]

            contours, _ = cv2.findContours(self.preview_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, PREVIEW_COLOR, 2)

        # Draw prompts for the needle only.
        for p in self.prompts["needle_mask"]:
            col = COL_POS if p["label"] == 1 else COL_NEG
            cv2.circle(vis, (p["x"], p["y"]), COL_RADIUS, col, -1)
            cv2.circle(vis, (p["x"], p["y"]), COL_RADIUS + 2, (255, 255, 255), 1)

        # HUD overlay
        h, w = vis.shape[:2]
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        obj_label = self.current_obj
        n_pos = sum(1 for p in self.prompts[obj_label] if p["label"] == 1)
        n_neg = sum(1 for p in self.prompts[obj_label] if p["label"] == 0)

        lines = [
            f"Bag: {self.bag_stem}  |  Frame {self.seed_frame_idx}/{len(self.frame_paths)-1}",
            f"Object: {obj_label}   +{n_pos} pos  -{n_neg} neg   Click mode: {self.current_click_mode()}",
            "L-click=place current mode  T=toggle +/−  Z=undo  C=clear  P=preview  ENTER=done  Q=quit",
        ]
        if self.preview_visible:
            preview_state = "fresh" if not self.preview_dirty else "stale"
            lines.append(f"Preview: {preview_state} (press P to recompute/toggle)")
        else:
            lines.append("Preview: off (press P to compute from current clicks)")
        for i, line in enumerate(lines):
            cv2.putText(vis, line, (10, h - 80 + i * 25), FONT, 0.55, (200, 255, 200), 1, cv2.LINE_AA)

        self.display_image = vis

    def _mouse_callback(self, event, x, y, flags, param):
        obj_label = self.current_obj

        if event == cv2.EVENT_LBUTTONDOWN:
            self.prompts[obj_label].append({"x": x, "y": y, "label": self.active_click_label})
            self.preview_dirty = True
            self._render()

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Remove nearest prompt
            if self.prompts[obj_label]:
                dists = [math.hypot(p["x"] - x, p["y"] - y) for p in self.prompts[obj_label]]
                nearest = int(np.argmin(dists))
                self.prompts[obj_label].pop(nearest)
                self.preview_dirty = True
                self._render()

    def run(self) -> dict | None:
        """
        Launch the annotation window.
        Returns the seeds dict, or None if the user quit.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1080, 720)
        self._render()
        cv2.imshow(self.window_name, self.display_image)

        cv2.waitKey(1)

        try:
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV created the window but could not attach the annotation callback."
            ) from exc

        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('z'):
                obj_label = self.current_obj
                if self.prompts[obj_label]:
                    self.prompts[obj_label].pop()
                    self._render()

            elif key == ord('t'):
                self.active_click_label = 0 if self.active_click_label == 1 else 1
                self._render()

            elif key == ord('c'):
                self.prompts[self.current_obj] = []
                self.preview_dirty = True
                self._render()

            elif key == ord('p'):
                self.preview_visible = not self.preview_visible
                if self.preview_visible and self.preview_dirty:
                    self._refresh_preview()
                self._render()

            elif key in (13, 32):   # ENTER or SPACE
                # Must have at least one positive prompt for the needle
                if not any(p["label"] == 1 for p in self.prompts["needle_mask"]):
                    print("  [!] Please add at least one positive click on the needle before accepting.")
                    continue
                break

            elif key in (ord('q'), 27):   # Q or ESC
                self.quit_all = True
                cv2.destroyWindow(self.window_name)
                return None

            # Allow changing seed frame with left/right arrow keys
            elif key == 81 or key == 2:   # left arrow
                self.seed_frame_idx = max(0, self.seed_frame_idx - 1)
                self.seed_image = cv2.imread(str(self.frame_paths[self.seed_frame_idx]))
                self.preview_dirty = True
                self._render()

            elif key == 83 or key == 3:   # right arrow
                self.seed_frame_idx = min(len(self.frame_paths) - 1, self.seed_frame_idx + 1)
                self.seed_image = cv2.imread(str(self.frame_paths[self.seed_frame_idx]))
                self.preview_dirty = True
                self._render()

            elif key == ord('n') or key == ord('N'):
                # Add current prompts as a seed entry and advance to next frame
                pts = self.prompts["needle_mask"]
                if not any(p["label"] == 1 for p in pts):
                    print("  [!] Cannot add seed without at least one positive click.")
                    continue
                entry = {
                    "seed_frame_idx": int(self.seed_frame_idx),
                    "objects": {}
                }
                entry["objects"]["needle_mask"] = {
                    "points": [[p["x"], p["y"]] for p in pts],
                    "labels": [p["label"] for p in pts],
                }
                self.collected_seeds.append(entry)
                print(f"  Added seed for frame {self.seed_frame_idx}. Total seeds: {len(self.collected_seeds)}")
                # Clear current prompts and move forward one frame if possible
                self.prompts = {obj: [] for obj in OBJECTS}
                if self.seed_frame_idx < len(self.frame_paths) - 1:
                    self.seed_frame_idx += 1
                    self.seed_image = cv2.imread(str(self.frame_paths[self.seed_frame_idx]))
                self.preview_dirty = True
                self._render()

        cv2.destroyWindow(self.window_name)

        # Build seeds dict. If multiple seeds were collected via 'N', include them.
        if self.collected_seeds:
            # Optionally include current prompts as final seed if present
            pts = self.prompts["needle_mask"]
            if pts and any(p["label"] == 1 for p in pts):
                entry = {
                    "seed_frame_idx": int(self.seed_frame_idx),
                    "objects": {
                        "needle_mask": {
                            "points": [[p["x"], p["y"]] for p in pts],
                            "labels": [p["label"] for p in pts],
                        }
                    }
                }
                self.collected_seeds.append(entry)

            seeds = {
                "bag_stem": self.bag_stem,
                "frame_count": len(self.frame_paths),
                "seed_frames": self.collected_seeds,
            }
            # For backward compatibility, also include first seed as legacy keys
            first = self.collected_seeds[0]
            seeds["seed_frame_idx"] = first["seed_frame_idx"]
            seeds["objects"] = first["objects"]
            return seeds
        else:
            seeds = {
                "bag_stem": self.bag_stem,
                "seed_frame_idx": self.seed_frame_idx,
                "frame_count": len(self.frame_paths),
                "objects": {}
            }
            pts = self.prompts["needle_mask"]
            if pts:
                seeds["objects"]["needle_mask"] = {
                    "points": [[p["x"], p["y"]] for p in pts],
                    "labels": [p["label"] for p in pts],
                }
            return seeds


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def process_bag(bag_path: Path, out_root: Path, every_n: int, skip_extract: bool, edit_existing: bool):
    """Extract frames (if needed) and run the annotation UI for one bag."""
    bag_stem = bag_path.stem if bag_path.is_file() else bag_path.name
    bag_out = out_root / bag_stem
    frames_dir = bag_out / "frames"
    seeds_path = bag_out / "seeds.json"
    existing_seeds = None

    print(f"\n{'='*60}")
    print(f"BAG: {bag_stem}")
    print(f"{'='*60}")

    # ── Frame extraction ──────────────────────────────────────────────────
    # Accept either JPG or PNG frames; extraction writes JPG by default.
    if skip_extract and frames_dir.exists() and (
        any(frames_dir.glob("*.jpg")) or any(frames_dir.glob("*.jpeg")) or any(frames_dir.glob("*.png"))
    ):
        frame_paths = collect_frame_paths(frames_dir)
        print(f"  Skipping extraction — found {len(frame_paths)} existing frames.")
    else:
        frame_paths = extract_frames_from_bag(bag_path, frames_dir, every_n=every_n)

    if not frame_paths:
        print(f"  [SKIP] No frames found for {bag_stem}")
        return False

    if edit_existing and seeds_path.exists():
        try:
            with open(seeds_path) as f:
                existing_seeds = json.load(f)
            print(f"  Editing existing seeds: {seeds_path}")
        except Exception as exc:
            print(f"  [WARN] Could not load existing seeds.json: {exc}")
            existing_seeds = None

    # ── Annotation ────────────────────────────────────────────────────────
    print(f"\n  Launching annotation UI for {len(frame_paths)} frames…")
    print("  Controls: L-click=positive  R-click=negative  Z=undo  C=clear")
    print("            N=next object  LEFT/RIGHT arrows=change seed frame")
    print("            ENTER=accept  Q=quit")

    session = AnnotationSession(frame_paths, bag_stem, existing_seeds=existing_seeds)
    seeds = session.run()

    if session.quit_all:
        print("  User quit — stopping.")
        return "quit"

    if seeds is None:
        print(f"  [SKIP] No annotation saved for {bag_stem}")
        return False

    # ── Save seeds ────────────────────────────────────────────────────────
    bag_out.mkdir(parents=True, exist_ok=True)
    with open(seeds_path, "w") as f:
        json.dump(seeds, f, indent=2)
    print(f"  ✓ Seeds saved → {seeds_path}")
    for obj, data in seeds["objects"].items():
        n_pos = sum(1 for l in data["labels"] if l == 1)
        n_neg = sum(1 for l in data["labels"] if l == 0)
        print(f"    {obj}: {n_pos} positive, {n_neg} negative")

    return True


def choose_bag_paths_gui() -> list[Path]:
    """Open a native picker to select a bag folder."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    selected_path = filedialog.askdirectory(
        title="Select a bag folder",
    )

    root.destroy()

    if not selected_path:
        sys.exit("ERROR: No bag folder selected.")

    return [Path(selected_path).expanduser().resolve()]


def main():
    parser = argparse.ArgumentParser(description="SAM2 seed annotator for needle tracking bags")
    parser.add_argument("--bag", nargs="*",
                        help="Path(s) to bag directory/directories (supports glob)")
    parser.add_argument("--select-bag", action="store_true",
                        help="Open a file picker to choose one or more bag files (default if --bag is omitted)")
    parser.add_argument("--out", required=True,
                        help="Output root directory")
    parser.add_argument("--every-n", type=int, default=3,
                        help="Keep every Nth frame during extraction (default: 3)")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip frame extraction if frames already exist")
    parser.add_argument("--edit-existing", action="store_true",
                        help="Load an existing seeds.json (if present) and edit it in place")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.select_bag or not args.bag:
        bag_paths = choose_bag_paths_gui()
    else:
        bag_paths = []
        for b in args.bag:
            p = Path(b)
            if p.exists():
                bag_paths.append(p)
            else:
                # Try glob
                matches = sorted(Path(".").glob(b))
                bag_paths.extend(matches)

    if not bag_paths:
        print("[ERROR] No bag paths found.")
        sys.exit(1)

    print(f"Found {len(bag_paths)} bag(s) to process.")

    for bag_path in bag_paths:
        result = process_bag(bag_path, out_root, args.every_n, args.skip_extract, args.edit_existing)
        if result == "quit":
            break

    print("\nAll done. Transfer the output directory to your GPU machine and run 02_propagate.py")


if __name__ == "__main__":
    main()
