# Pose Smoothing for Checkerboard Detection

## Problem
Successful checkerboard detections can have "jumpiness" or noise frame-to-frame due to:
- Subpixel detection noise in corner localization
- Small variations in lighting/reflections
- Numerical precision in pose estimation

## Solution: Covariance-Weighted RTS Filtering

The `smooth_poses_rts()` function implements an **optimal** pose smoother using a **Rauch-Tung-Striebel (RTS) filter**. This is principled because it:

1. **Respects measurement uncertainty**: Your `poses.json` already contains 6×6 covariance matrices for each pose
   - Higher covariance (less confident detection) → allow more smoothing
   - Lower covariance (confident detection) → stay closer to measurement

2. **Uses constrained motion model**: Assumes poses change smoothly between frames (small process noise)
   - Doesn't assume constant velocity or acceleration
   - Conservative by default (set `process_noise_scale=1.0`)

3. **Optimal backward smoothing**: RTS filter is provably optimal in the least-squares sense
   - Forward Kalman filter → backward smoother correction
   - Lower total estimation error than forward-only filtering

### Mathematical Outline

For position (rotation handled separately via SLERP):

```
Forward pass (Kalman filter):
  Predict: x̂⁻ = x̂⁺(k-1)  [constant-velocity model]
  Update:  x̂⁺ = x̂⁻ + K(z - x̂⁻)  [Kalman gain K uses covariance]

Backward pass (RTS smoother):
  x̃ = x̂⁺ + C(x̃₊₁ - x̂⁻₊₁)  [smoother gain C blends forward/backward]
```

Where:
- Measurement covariance R comes from pose.covariance[0:3,0:3]
- Process noise Q is small and configurable
- Rotation uses weighted quaternion SLERP (confidence-weighted)

## Usage

### Basic Usage (Single Bag)

```bash
cd auto-annotating
python checkerboard_pose_offline/smooth_poses.py \
  --ann-dir ./annotations \
  --bag ch_circlexy \
  --smooth-strength 1.0
```

This creates `annotations/ch_circlexy/poses_smoothed.json`.

### Try Different Smoothing Strengths

```bash
# Slight smoothing (conservative)
python ... --smooth-strength 0.3

# Moderate smoothing (default)
python ... --smooth-strength 1.0

# Aggressive smoothing
python ... --smooth-strength 10.0
```

- **0.1-0.5**: Barely noticeable smoothing (best for low-noise data)
- **1.0-5.0**: Moderate smoothing (most common)
- **5-20+**: Strong smoothing (use carefully; may lose valid motion)

### Smooth All Bags

```bash
python checkerboard_pose_offline/smooth_poses.py \
  --ann-dir ./annotations \
  --all \
  --smooth-strength 2.0
```

### Apply to Original (Careful!)

```bash
# Preview first:
python ... --ann-dir ./annotations --bag ch_circlexy

# If happy with poses_smoothed.json:
python ... --ann-dir ./annotations --bag ch_circlexy --in-place
```

## Choosing Smooth Strength

The `process_noise_scale` parameter controls how much the filter "trusts" its motion model:

1. **Visualize the results**:
   ```bash
   python visualize_poses.py --ann-dir ./annotations --bag ch_circlexy
   ```
   Compare `poses.json` (original) vs `poses_smoothed.json` (smoothed)

2. **Start conservative** (0.3-1.0) and increase if:
   - Jumps are still visible
   - Motion looks natural
   - Reprojection errors stay reasonable

3. **Reduce if**:
   - Trajectories become unrealistically smooth
   - Real motion is being lost
   - Object appears to "lag" behind actual movement

## Verification

After smoothing, check:

1. **Reprojection error** (unchanged by smoothing):
   ```python
   # Already in poses_smoothed.json
   frame["rms_reprojection_error"]  
   ```

2. **Position deviation** (how much we smoothed):
   ```python
   # Added by smoother:
   frame["smoothing_info"]["original_position_error_m"]  # deviation in meters
   ```

3. **Visual verification**:
   ```bash
   python visualize_poses.py --ann-dir ./annotations --bag ch_circlexy
   ```

## How to Integrate

### Option 1: Replace Original (Recommended if Good)
```bash
cp annotations/ch_circlexy/poses_smoothed.json annotations/ch_circlexy/poses.json
```

### Option 2: Keep Both, Use Smoothed for Training
```bash
# In your training pipeline:
from pathlib import Path
import json

poses_path = Path("annotations/ch_circlexy/poses_smoothed.json")  # or poses.json
with open(poses_path) as f:
    data = json.load(f)
```

### Option 3: Selective Smoothing
```python
from temporal_interpolation import smooth_poses_rts

# Smooth only specific bags or frames
results = smooth_poses_rts(
    Path("annotations/ch_circlexy/poses.json"),
    process_noise_scale=1.5
)

# Inspect results before saving
for frame_key, result in results.items():
    dev = result["original_position_error"]
    if dev > 0.01:  # > 1 cm deviation: inspect this frame
        print(f"Frame {frame_key}: {dev*1000:.1f}mm smoothed")
```

## Troubleshooting

### "No frames to smooth"
- Check poses.json has frames with `status="ok"` and valid `pose` field
- Verify JSON is not corrupted

### Smoothed poses look worse
- Reduce `smooth_strength` (0.3 instead of 1.0)
- Check if motion is actually supposed to be jerky (grasping/manipulation)
- May need different smoothing per-segment

### Large deviations from original
- Normal if original had significant noise
- Check covariance values (very small = over-confident detection)
- Consider adjusting detection parameters instead

## References

- **Rauch-Tung-Striebel smoother**: Gelb et al. "Applied Optimal Estimation"
- **Quaternion SLERP**: Shoemake, "Animating Rotation with Quaternion Curves" (SIGGRAPH 1985)
- Your existing code already uses quaternion SLERP in `temporal_interpolation.py`
