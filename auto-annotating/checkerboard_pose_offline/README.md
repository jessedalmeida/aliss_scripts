# Offline Checkerboard Pose Estimation

This folder contains an offline version of the checkerboard pose pipeline from `pose-estimation-uncertainty`.

It processes extracted frames, detects the checkerboard with the same detector fallback chain and preprocessing ideas as the ROS node, and writes one `poses.json` file per bag containing:

- per-frame pose
- 6x6 covariance
- detection metadata
- failure reasons and diagnostics image paths for frames that could not be detected automatically

## Scripts

- `estimate_checkerboard_pose_offline.py`: batch process one bag or all bags under `--ann-dir`
- `review_failed_frames.py`: interactively step through failed frames and save accept/reject/manual-edit decisions
- `visualize_poses.py`: browse frames and overlay the saved checkerboard pose, axes, and projected corners

## Typical usage

```bash
cd /home/jesse/aliss_core/src/aliss_scripts/auto-annotating
./.venv2/bin/python checkerboard_pose_offline/estimate_checkerboard_pose_offline.py \
  --ann-dir ./annotations \
  --all \
  --save-failed-diagnostics
```

The script writes `poses.json` into each bag directory unless `--output-json` is provided.

The reviewer writes `pose_reviews.json` next to the bag annotations by default. Each reviewed failed frame is marked as one of:

- `accepted`
- `rejected`
- `manual_edit`

## Reviewing failed frames

```bash
cd /home/jesse/aliss_core/src/aliss_scripts/auto-annotating
./.venv2/bin/python checkerboard_pose_offline/review_failed_frames.py \
  --ann-dir ./annotations \
  --bag ch_linearx
```

Controls:

- `Y`: mark accepted
- `N`: mark rejected
- `M`: mark for manual edit and open the pose overlay for that frame
- Left/Right arrows: previous/next failed frame
- `S`: save review decisions
- `Q` or `Esc`: quit

## Visualizing poses

```bash
cd /home/jesse/aliss_core/src/aliss_scripts/auto-annotating
./.venv2/bin/python checkerboard_pose_offline/visualize_poses.py \
  --ann-dir ./annotations \
  --bag ch_linearx
```

Controls:

- `D`, right arrow, or space: next frame
- `A` or left arrow: previous frame
- `Q` or `Esc`: quit

This viewer overlays the saved pose axes on top of the extracted frame and optionally projects the checkerboard corners back into the image.

## Repacking

The top-level `repack_bag.py` can consume `poses.json` and publish `/pose_estimator/pose` on the repacked bag so the pose does not need to be recomputed during playback.
