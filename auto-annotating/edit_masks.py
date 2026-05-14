#!/usr/bin/env python3
"""
Simple interactive mask editor

Usage:
  python edit_masks.py --frames-dir /path/to/frames --masks-dir /path/to/masks

Controls:
    Left-drag: apply current mode (paint or erase)
    T: toggle paint/erase mode
    C: clear all mask pixels for this frame
    [ / -: decrease brush size
    ] / +: increase brush size
    V: copy mask from previous frame
    S: save current mask
    Left/Right arrows or A/D: prev/next frame (autosave if modified)
    Q / ESC: quit (autosave prompt)
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import sys


def find_mask_for_frame(masks_dir: Path, frame_idx: int):
    # Prefer files like frame_000012_*.png
    pattern = f"frame_{frame_idx:06d}_*.png"
    matches = sorted(masks_dir.glob(pattern))
    return matches[0] if matches else None


class MaskEditor:
    def __init__(self, frames_dir: Path, masks_dir: Path, brush: int = 3):
        self.frames_dir = frames_dir
        self.masks_dir = masks_dir
        self.frame_paths = sorted([p for p in frames_dir.glob("*.png")] + [p for p in frames_dir.glob("*.jpg")])
        if not self.frame_paths:
            raise RuntimeError(f"No frames found in {frames_dir}")
        self.n = len(self.frame_paths)
        self.idx = 0
        self.brush = max(1, brush)

        self.window = "Mask Editor"
        self.painting = False
        self.paint_mode = 1  # 1=paint, 0=erase
        self.modified = False
        self.cursor_x = None
        self.cursor_y = None

        self.frame = None
        self.mask = None
        self.display = None

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse_cb)

    def _load(self):
        fp = self.frame_paths[self.idx]
        self.frame = cv2.imread(str(fp))
        if self.frame is None:
            raise RuntimeError(f"Cannot load frame {fp}")
        # find corresponding mask
        mask_path = find_mask_for_frame(self.masks_dir, self.idx)
        if mask_path and mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
            else:
                self.mask = (m > 127).astype(np.uint8) * 255
        else:
            self.mask = np.zeros(self.frame.shape[:2], dtype=np.uint8)
        self.modified = False
        self._update_display()

    def _save(self):
        # Write mask back to masks_dir with same basename if exists, else create name
        mask_path = find_mask_for_frame(self.masks_dir, self.idx)
        if mask_path is None:
            # create a new mask filename matching frame index
            mask_path = self.masks_dir / f"frame_{self.idx:06d}_mask.png"
            self.masks_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), self.mask)
        self.modified = False
        print(f"Saved mask → {mask_path}")

    def _update_display(self):
        overlay = self.frame.copy()
        coloured = np.zeros_like(overlay)
        coloured[self.mask > 0] = (0, 255, 100)
        overlay = cv2.addWeighted(overlay, 0.7, coloured, 0.3, 0)
        if self.cursor_x is not None and self.cursor_y is not None:
            cursor_color = (0, 255, 255) if self.paint_mode == 1 else (0, 128, 255)
            cv2.circle(overlay, (self.cursor_x, self.cursor_y), self.brush, cursor_color, 1, cv2.LINE_AA)
            cv2.circle(overlay, (self.cursor_x, self.cursor_y), 2, cursor_color, -1, cv2.LINE_AA)
        # HUD
        h, w = overlay.shape[:2]
        cv2.rectangle(overlay, (0, h - 80), (w, h), (0, 0, 0), -1)
        lines = [
            f"Frame {self.idx}/{self.n-1}  Brush={self.brush}  Mode={'paint' if self.paint_mode==1 else 'erase'}",
            "L-drag=paint/erase  T=toggle  C=clear  [/-=size  S=save  A/D=nav  V=copy  Q=quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(overlay, line, (8, h - 56 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
        self.display = overlay

    def _mouse_cb(self, event, x, y, flags, param):
        self.cursor_x = x
        self.cursor_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.painting = True
            self._stroke(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.painting:
            self._stroke(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.painting = False
        else:
            self._update_display()

    def _stroke(self, x, y):
        cv2.circle(self.mask, (x, y), self.brush, 255 if self.paint_mode == 1 else 0, -1)
        self.modified = True
        self._update_display()

    def run(self):
        self._load()
        while True:
            cv2.imshow(self.window, self.display)
            key = cv2.waitKeyEx(20)
            if key == -1:
                continue
            # handle printable keys
            if key == ord('q') or key == 27:
                if self.modified:
                    print("Unsaved changes — saving before exit.")
                    self._save()
                break
            if key == ord('s'):
                self._save()
                continue
            if key == ord('t'):
                self.paint_mode = 0 if self.paint_mode == 1 else 1
                self._update_display()
                continue
            if key == ord('c'):
                self.mask[:] = 0
                self.modified = True
                self._update_display()
                continue
            if key in (ord('a'), 81, 2424832):  # left (include multiple codes)
                # prev frame
                if self.modified:
                    print("Autosaving modified mask before navigation...")
                    self._save()
                self.idx = max(0, self.idx - 1)
                self._load()
                continue
            if key in (ord('d'), 83, 2555904):  # right
                if self.modified:
                    print("Autosaving modified mask before navigation...")
                    self._save()
                self.idx = min(self.n - 1, self.idx + 1)
                self._load()
                continue
            if key in (ord('+'), ord('='), ord(']')):
                self.brush = min(100, self.brush + 1)
                self._update_display()
                continue
            if key in (ord('-'), ord('[')):
                self.brush = max(1, self.brush - 1)
                self._update_display()
                continue
            if key == ord('v'):
                # copy mask from previous frame
                if self.idx > 0:
                    prev_mask_path = find_mask_for_frame(self.masks_dir, self.idx - 1)
                    if prev_mask_path and prev_mask_path.exists():
                        m = cv2.imread(str(prev_mask_path), cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            self.mask = (m > 127).astype(np.uint8) * 255
                            self.modified = True
                            self._update_display()
                            print(f"Copied mask from frame {self.idx - 1}")
                continue

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Interactive mask editor")
    grp = parser.add_mutually_exclusive_group(required=False)
    grp.add_argument("--frames-dir", help="Directory with frame images")
    grp.add_argument("--ann-dir", help="Annotation root directory (output from seed_annotator)")

    parser.add_argument("--bag", help="Bag subdirectory under --ann-dir (required if --ann-dir is used)")
    parser.add_argument("--masks-dir", help="Directory with mask images (pattern frame_######_*.png). If omitted with --ann-dir, uses <ann-dir>/<bag>/masks")
    parser.add_argument("--brush", type=int, default=5, help="Initial brush radius (px)")
    args = parser.parse_args()

    # Resolve directories: prefer --ann-dir + --bag; else require --frames-dir and --masks-dir
    if args.ann_dir:
        if not args.bag:
            print("When using --ann-dir you must also provide --bag <bag_stem>")
            sys.exit(1)
        ann = Path(args.ann_dir)
        frames_dir = ann / args.bag / "frames"
        masks_dir = Path(args.masks_dir) if args.masks_dir else (ann / args.bag / "masks")
    else:
        if not args.frames_dir or not args.masks_dir:
            print("Either --ann-dir and --bag or both --frames-dir and --masks-dir must be provided")
            sys.exit(1)
        frames_dir = Path(args.frames_dir)
        masks_dir = Path(args.masks_dir)

    if not frames_dir.exists():
        print(f"Frames dir not found: {frames_dir}")
        sys.exit(1)
    editor = MaskEditor(frames_dir, masks_dir, brush=args.brush)
    editor.run()


if __name__ == '__main__':
    main()
