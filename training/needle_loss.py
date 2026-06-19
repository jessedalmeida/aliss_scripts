#!/usr/bin/env python3
"""needle_loss.py - Multi-task loss for the needle model.

Terms (each returned separately for logging):
  mask : BCE-with-logits + (1 - soft Dice)   on the needle mask
  hm   : MSE between predicted and target heatmaps, MASKED by hm_mask so
         occluded/absent keypoints contribute exactly zero
  vis  : BCE-with-logits on visibility, MASKED by vis_mask so only labeled
         keypoints contribute

total = w_mask*mask + w_hm*hm + w_vis*vis
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_argmax(hm: torch.Tensor) -> torch.Tensor:
    """(B,K,H,W) -> (B,K,2) expected (x,y) in heatmap-grid units via spatial softmax."""
    B, K, H, W = hm.shape
    prob = F.softmax(hm.reshape(B, K, -1), dim=-1).reshape(B, K, H, W)
    xs = torch.linspace(0, W - 1, W, device=hm.device)
    ys = torch.linspace(0, H - 1, H, device=hm.device)
    ex = (prob.sum(2) * xs).sum(-1)
    ey = (prob.sum(3) * ys).sum(-1)
    return torch.stack([ex, ey], dim=-1)   # (B,K,2)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()


class NeedleLoss(nn.Module):
    def __init__(self, w_mask: float = 1.0, w_hm: float = 1.0, w_vis: float = 0.5):
        super().__init__()
        self.w_mask, self.w_hm, self.w_vis = w_mask, w_hm, w_vis

    def forward(self, out: dict, batch: dict) -> dict:
        # ---- mask ----
        mask_bce = F.binary_cross_entropy_with_logits(out["mask_logits"], batch["mask"])
        mask_dice = soft_dice_loss(out["mask_logits"], batch["mask"])
        mask_loss = mask_bce + mask_dice

        # ---- keypoint coordinate loss (masked by hm_mask) ----
        # Soft-argmax gives differentiable (x,y) in heatmap-grid units (0..H-1).
        # L2 distance normalized by heatmap height puts the loss in [0,1] range,
        # making it comparable in scale to the mask and vis terms.
        pred_xy = _soft_argmax(out["heatmaps"])               # (B,K,2) heatmap units
        tgt_xy = batch["kps_hm"]                              # (B,K,2)
        hm_mask = batch["hm_mask"]                            # (B,K)
        H = out["heatmaps"].shape[2]
        per_kp = ((pred_xy - tgt_xy) ** 2).sum(-1).sqrt() / H  # (B,K) normalized L2
        denom = hm_mask.sum().clamp(min=1.0)
        hm_loss = (per_kp * hm_mask).sum() / denom

        # ---- visibility (masked by vis_mask) ----
        vis_mask = batch["vis_mask"]                          # (B,K)
        per_kp_vis = F.binary_cross_entropy_with_logits(
            out["vis_logits"], batch["vis_target"], reduction="none")
        vdenom = vis_mask.sum().clamp(min=1.0)
        vis_loss = (per_kp_vis * vis_mask).sum() / vdenom

        total = self.w_mask * mask_loss + self.w_hm * hm_loss + self.w_vis * vis_loss
        return {"total": total, "mask": mask_loss, "hm": hm_loss, "vis": vis_loss,
                "mask_bce": mask_bce.detach(), "mask_dice": mask_dice.detach()}
