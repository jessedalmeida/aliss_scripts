# SE(3) Manifold-Based Pose Smoothing

## Overview

Your checkerboard pose detections can have jitter due to subpixel detection noise and local minima in the pose solver. This module implements **principled covariance-weighted smoothing on SE(3)** — the Lie group of rigid transformations.

## Mathematical Approach

Rather than treating poses as 6D vectors, we:
1. Work on the **SE(3) manifold** using the exponential map: `pose_new = pose_old * exp(xi)`
2. Use **Rauch-Tung-Striebel (RTS) smoothing** on the **se(3) tangent space** (algebra)
3. Weight smoothing by **measurement covariances** (higher uncertainty = allow more smoothing)
4. Use **process noise** to couple neighboring frames (prevents overfitting)

### Key Equations

**SE(3) exponential map:**
```
T = exp(xi) = [R | t; 0 | 1]
where:
  xi = [rho, phi]  (6D: translation + axis-angle rotation)
  R = exp(phi)  (rotation matrix from axis-angle)
  t = V(phi) @ rho  (translation adjusted by rotation)
```

**Information filter form:**
```
sigma_inv_k = Y_pred + Y_k  (information addition)
where Y_k is the measurement precision (inverse covariance)
```

**RTS backward smoothing:**
```
m_smooth_k = m_filt_k + G_k @ (m_smooth_{k+1} - m_filt_k)
where G_k is the smoother gain balancing process and measurement noise
```

## Why This Approach?

✓ **Respects Lie group geometry** — rotations stay on SO(3), not in flat ℝ³
✓ **Uses measurement uncertainty** — high-uncertainty poses get smoothed more
✓ **Optimal solution** — RTS smoother minimizes squared error on the manifold
✓ **No parameter tuning** (mostly) — process noise is the main knob

## Usage

### Basic Usage

```bash
# Smooth with default settings (noise_scale=1.0)
python smooth_poses_se3.py annotations/ch_circlexy/poses.json

# More aggressive smoothing (less noise assumption)
python smooth_poses_se3.py annotations/ch_circlexy/poses.json --noise-scale 10.0

# Softer smoothing (more noise assumption)
python smooth_poses_se3.py annotations/ch_circlexy/poses.json --noise-scale 0.1

# Custom output location
python smooth_poses_se3.py annotations/ch_circlexy/poses.json \
  --noise-scale 2.0 \
  --output annotations/ch_circlexy/poses_smooth.json
```

### Understanding Process Noise Scale

**`--noise-scale`** controls how much the smoother trusts temporal continuity:

| Scale | Effect | Use Case |
|-------|--------|----------|
| 0.1 | Very soft smoothing | Minor jitter, trust detections |
| 1.0 | Moderate (default) | Typical jitter from detection noise |
| 5-10 | Aggressive smoothing | Significant jumpiness, allow larger corrections |
| 50+ | Very aggressive | Only use if trajectory is very noisy |

**Formula:** `Q = noise_scale * 0.0001 * I₆`

Higher `noise_scale` = assumes larger between-frame changes are normal = smoother result

### Workflow

1. **Test different settings** on one sequence:
   ```bash
   python smooth_poses_se3.py annotations/ch_circlexy/poses.json \
     --noise-scale 1.0 --output poses_scale1.json
   python smooth_poses_se3.py annotations/ch_circlexy/poses.json \
     --noise-scale 5.0 --output poses_scale5.json
   ```

2. **Visualize results** (e.g., with your existing visualization tools)

3. **Apply to all sequences:**
   ```bash
   for seq in ch_circlexy ch_circlexz ch_figure8 ch_ingrasp_rot*; do
     python smooth_poses_se3.py annotations/$seq/poses.json \
       --noise-scale 2.0 \
       --output annotations/$seq/poses_smooth.json
   done
   ```

4. **Accept the smoothed poses** (after verification):
   ```bash
   # Backup original
   cp annotations/ch_circlexy/poses.json annotations/ch_circlexy/poses.json.bak
   
   # Replace with smoothed
   cp poses_scale2.json annotations/ch_circlexy/poses.json
   ```

## Implementation Details

### File: `temporal_interpolation.py`

**New SE(3) functions:**
- `axis_angle_to_matrix()` — Rodrigues formula
- `matrix_to_axis_angle()` — scipy.spatial.transform.Rotation
- `pose_to_se3_matrix()` / `se3_matrix_to_pose()` — conversions
- `se3_exp()` / `se3_log()` — Lie group exponential/log maps
- `smooth_poses_se3()` — main RTS smoothing algorithm
- `save_se3_smoothed_poses()` — write results back to JSON

### File: `smooth_poses_se3.py`

Command-line wrapper with:
- Argument parsing
- Progress reporting
- Residual statistics
- Output validation

## Interpreting Output

```
Smoothing residuals (log-manifold distance):
  Mean: 0.018162  ← average shift per frame (meters)
  Max:  0.219902  ← largest single-frame adjustment
  Min:  0.000299  ← smallest adjustment
```

- **Mean ~0.01-0.05 m** = typical for jitter removal
- **Max >> Mean** = probably caught a jump
- **Mean > 0.1 m** = smoothing might be too aggressive

## Advanced: Tuning the Process Noise

The process noise is currently fixed at `0.0001 * noise_scale * I₆`. You can modify this by editing the line in `smooth_poses_se3()`:

```python
# Current (near line 640):
Q = np.eye(6) * (process_noise_scale * 0.0001)

# More aggressive (assumes faster dynamics):
Q = np.eye(6) * (process_noise_scale * 0.001)
```

**Larger Q** = smoother (less trust in constant velocity model)
**Smaller Q** = stiffer (more trust in constant velocity model)

## Debugging

If smoothing has no effect:
1. Check that `--noise-scale` is not too small (try 10.0)
2. Verify poses.json is being read correctly
3. Check that poses have non-zero covariances

If smoothing is too aggressive:
1. Reduce `--noise-scale` (try 0.5)
2. Or reduce the base process noise in the code

## References

The approach is based on:
- **Rauch-Tung-Striebel smoother:** Classic optimal backward pass for filtering
- **SE(3) Lie groups:** See Barfoot's "State Estimation for Robotics"
- **Information filtering:** Parallel to RTS but works with precision matrices

## Dependencies

- `numpy`
- `scipy.spatial.transform.Rotation`
- Standard library: `json`, `pathlib`, `argparse`

All should be available in your existing environment.
