#!/usr/bin/env python3
"""
cleanup_masks.py - Morphological cleanup and speckle filtering for mask post-processing.

Applies binary morphology (opening/closing) and connected-component filtering to remove
noise and small artifacts from SAM2-propagated masks.

Usage
-----
python3 cleanup_masks.py --ann-dir ./annotations --bag ch_linearx
python3 cleanup_masks.py --ann-dir ./annotations --bag ch_linearx --kernel-size 5 --min-area 50

Arguments
---------
--ann-dir       Annotation root directory (required)
--bag           Bag name to clean (required)
--kernel-size   Morphology kernel size: 3, 5, 7, etc. (default: 5)
--min-area      Minimum connected component area in pixels to keep (default: 100)
--operation     'opening', 'closing', or 'both' (default: 'opening')
--inplace       Overwrite masks in place; otherwise save to 'masks_clean/' (default: False)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def collect_masks(masks_dir: Path) -> list[Path]:
    """Collect all PNG masks from the masks directory."""
    mask_paths = sorted(masks_dir.glob("*.png"))
    return mask_paths


def cleanup_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    min_area: int = 100,
    operation: str = "opening",
) -> np.ndarray:
    """
    Apply morphology and connected-component filtering to a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (values 0 or 1)
    kernel_size : int
        Morphology kernel size (must be odd)
    min_area : int
        Minimum component area in pixels to retain
    operation : str
        'opening' (erosion then dilation), 'closing' (dilation then erosion), or 'both'

    Returns
    -------
    np.ndarray
        Cleaned binary mask
    """
    if mask.max() == 0:
        return mask  # Empty mask, no-op

    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = mask.copy()

    # Apply morphology
    if operation in ("opening", "both"):
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    if operation in ("closing", "both"):
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Filter small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    filtered = np.zeros_like(cleaned)

    for label_idx in range(1, num_labels):  # Skip background (0)
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label_idx] = 1

    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean up masks with morphology and speckle filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ann-dir", required=True, help="Annotation root directory")
    parser.add_argument("--bag", required=True, help="Bag name to clean")
    parser.add_argument("--kernel-size", type=int, default=5, help="Morphology kernel size (default: 5)")
    parser.add_argument("--min-area", type=int, default=100, help="Min component area in pixels (default: 100)")
    parser.add_argument(
        "--operation",
        choices=["opening", "closing", "both"],
        default="opening",
        help="Morphology operation (default: opening)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite masks in place; otherwise save to 'masks_clean/'",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    bag_dir = ann_dir / args.bag
    masks_dir = bag_dir / "masks"

    if not masks_dir.exists():
        print(f"[ERROR] Masks directory not found: {masks_dir}")
        return 1

    mask_paths = collect_masks(masks_dir)
    if not mask_paths:
        print(f"[ERROR] No masks found in {masks_dir}")
        return 1

    print(f"Found {len(mask_paths)} masks in {masks_dir}")
    print(f"Parameters: kernel_size={args.kernel_size}, min_area={args.min_area}, operation={args.operation}")

    # Determine output directory
    if args.inplace:
        output_dir = masks_dir
        print(f"Cleaning masks in-place at {output_dir}")
    else:
        output_dir = bag_dir / "masks_clean"
        output_dir.mkdir(exist_ok=True)
        print(f"Saving cleaned masks to {output_dir}")

    processed_count = 0
    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Could not load mask: {mask_path}")
            continue

        # Normalize to binary
        binary_mask = (mask > 128).astype(np.uint8)

        # Cleanup
        cleaned = cleanup_mask(
            binary_mask,
            kernel_size=args.kernel_size,
            min_area=args.min_area,
            operation=args.operation,
        )

        # Save (as 0/255 for standard binary mask format)
        output_mask = (cleaned * 255).astype(np.uint8)
        output_path = output_dir / mask_path.name
        cv2.imwrite(str(output_path), output_mask)
        processed_count += 1

    print(f"Successfully cleaned {processed_count}/{len(mask_paths)} masks")
    if not args.inplace:
        print(f"\nTo use cleaned masks, copy them back:")
        print(f"  cp {output_dir}/* {masks_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
