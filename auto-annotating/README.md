# SAM2 Needle Mask Annotation Pipeline

Three-script pipeline for annotating needle segmentation masks across ROS2 `.mcap` bag files using SAM2 video propagation.

```
01_seed_annotator.py   ← Mac (no GPU needed)
02_propagate.py        ← GPU machine (CUDA)
03_repack_bag.py       ← Either machine
```

---

## Overview

```
[Mac]                          [GPU machine]               [Either]
 bags/ ──► extract frames ──► seeds.json ──► SAM2 prop ──► masks/ ──► annotated bags/
           click to seed                      (auto)
```

You click once per clip on the Mac to place seed prompts. SAM2 propagates those masks to every frame automatically on the GPU machine. The final step writes the masks back into new `.mcap` files alongside the original topics.

---

## Setup

### Mac (Script 01)

```bash
pip install opencv-python numpy rosbags tqdm
```

`rosbags` lets you read `.mcap` files without a full ROS2 installation.

### GPU Machine (Script 02)

```bash
# Clone SAM2
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[demo]"

# Download checkpoints (all sizes, ~5GB total, or pick one)
cd checkpoints
bash download_ckpts.sh
cd ..

# Other deps
pip install opencv-python numpy tqdm
```

---

## Step 1 — Seed Annotation (Mac)

If you do not want to type bag paths, launch the annotator with the picker:

```bash
python 01_seed_annotator.py --select-bag --out ./annotations
```

```bash
# Single bag directory
python 01_seed_annotator.py --bag /path/to/suture1 --out ./annotations

# Multiple bags
python 01_seed_annotator.py \
    --bag /path/to/bags/suture1 /path/to/bags/suture2 \
    --out ./annotations

# Keep every 3rd frame (default). Use --every-n 5 for larger bags.
python 01_seed_annotator.py --bag /path/to/suture1 --out ./annotations --every-n 5

# Re-annotate without re-extracting frames
python 01_seed_annotator.py --bag /path/to/suture1 --out ./annotations --skip-extract
```

### Annotation controls

| Action | Control |
|--------|---------|
| Add positive click (needle) | Left-click |
| Add negative click (background) | Right-click |
| Remove nearest click | Middle-click |
| Undo last click | Z |
| Clear all clicks for this object | C |
| Switch to next object (arm_mask) | N |
| Change seed frame | ← / → arrows |
| Accept and move to next bag | ENTER or SPACE |
| Quit | Q or ESC |

**Tips:**
- You only need to annotate the **first object** (needle_mask). Press ENTER to skip arm_mask if you don't need it.
- A few positive clicks on the needle body + one negative click on a nearby instrument is usually enough.
- Use ← / → to find a frame where the needle is clearly visible before placing prompts.
- The seed frame doesn't have to be frame 0 — SAM2 will propagate both forward and backward.

### Output structure

```
annotations/
  suture1/
    frames/          ← extracted PNGs
      frame_000000.png
      frame_000001.png
      ...
    seeds.json       ← your click prompts
  suture2/
    ...
```

Transfer the entire `annotations/` directory to your GPU machine.

---

## Step 2 — Propagation (GPU Machine)

```bash
# Process all annotated bags
python 02_propagate.py \
    --ann-dir ./annotations \
    --sam2-repo /path/to/sam2

# Single bag
python 02_propagate.py \
    --ann-dir ./annotations \
    --sam2-repo /path/to/sam2 \
    --bag suture1

# Faster (smaller model, slightly less accurate)
python 02_propagate.py \
    --ann-dir ./annotations \
    --sam2-repo /path/to/sam2 \
    --model small
```

### Model size tradeoffs

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| tiny  | fastest | lowest | ~3 GB |
| small | fast | good | ~4 GB |
| base  | moderate | better | ~6 GB |
| large | slowest | best | ~9 GB |

`large` is recommended since you're offline-processing, not real-time.

### Output

```
annotations/
  suture1/
    frames/          ← original extracted frames
    seeds.json
    masks/
      frame_000000_needle_mask.png    ← binary (0 or 255)
      frame_000001_needle_mask.png
      ...
      frame_000000_arm_mask.png       ← only if you seeded arm_mask
    sanity_check.png ← contact sheet for quick visual QC
    propagation_done ← sentinel; delete to re-run propagation
```

**Check `sanity_check.png` first** before repacking. If the masks look wrong:
1. Delete `propagation_done`
2. Re-run `01_seed_annotator.py --skip-extract` to update seeds.json
3. Re-run `02_propagate.py`

---

## Step 3 — Repack Bags (Either Machine)

```bash
python 03_repack_bag.py \
    --bag /path/to/original/suture1 \
    --ann-dir ./annotations \
    --out-dir ./annotated_bags
```

This writes a new bag at `./annotated_bags/suture1_annotated/` containing all original topics plus:

- `/needle_tracking/needle_mask` — `sensor_msgs/msg/Image` (mono8)
- `/needle_tracking/arm_mask` — `sensor_msgs/msg/Image` (mono8)

---

## Replaying annotated bags

```bash
ros2 bag play ./annotated_bags/suture1_annotated \
    --rate 1.0
```

Your `NeedleTrackerNode` subscribes to `/needle_tracking/needle_mask` via the `needle_mask_callback`, so it will pick up the masks automatically during playback.

---

## Troubleshooting

**`rosbags` can't find the image topic**
The script tries `/ves_camera/image_rect` first, then `/ves_camera/image`. If your bags use a different topic name, edit the `IMAGE_TOPICS` list at the top of `01_seed_annotator.py`.

**Annotation window doesn't open on Mac**
Try: `export DISPLAY=:0` or install `opencv-python-headless` and use `pip install opencv-python` instead.

**SAM2 import error**
Make sure you added the sam2 repo to your Python path:
```bash
cd /path/to/sam2 && pip install -e .
```

**Masks look blurry or wrong**
- Add more negative clicks around the arm/tissue that SAM2 is incorrectly including.
- Try a larger model (`--model large`).
- Try a different seed frame (use the arrow keys to pick a frame with clear needle visibility).

**Out of GPU memory**
- Use `--model small` or `--model tiny`.
- Process shorter clips by splitting bags first.
