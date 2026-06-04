#!/usr/bin/env python3
"""needle_model.py - Multi-task net: needle mask + tip/tail heatmaps + visibility.

Shared ResNet-34 encoder -> U-Net decoder (with skips) -> three heads:
  - mask_head : 1-channel segmentation logit, upsampled to input resolution
  - hm_head   : K heatmap logits at stride 4 (160x160 for 640 input)
  - vis_head  : K visibility logits from the encoder bottleneck

Keypoint coordinates come from soft-argmax over the heatmaps: a spatial softmax
followed by the expected (x, y), which is differentiable and sub-pixel (so
localization is not capped at the 160-grid resolution). Training (Step 7) can
supervise the heatmaps directly (MSE vs target Gaussians, masked by visibility)
and/or the soft-argmax coordinates; both are exposed here.

Real-time validated in Step 2: the equivalent-cost network ran ~163 FPS at 640
on the RTX 4070 Mobile in half precision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class _UpBlock(nn.Module):
    """Upsample x2, concat skip, two 3x3 convs."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class NeedleNet(nn.Module):
    def __init__(self, num_keypoints: int = 2, pretrained: bool = True,
                 softargmax_temp: float = 1.0):
        super().__init__()
        self.K = num_keypoints
        self.temp = softargmax_temp

        weights = "IMAGENET1K_V1" if pretrained else None
        enc = resnet34(weights=weights)
        # encoder stages (for input 640): strides 4,4,8,16,32
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # stride 2, 64
        self.pool = enc.maxpool                                   # stride 4, 64
        self.layer1, self.layer2 = enc.layer1, enc.layer2         # 64@s4, 128@s8
        self.layer3, self.layer4 = enc.layer3, enc.layer4         # 256@s16, 512@s32

        # decoder back up to stride 4
        self.up3 = _UpBlock(512, 256, 256)   # s32 -> s16
        self.up2 = _UpBlock(256, 128, 128)   # s16 -> s8
        self.up1 = _UpBlock(128, 64, 64)     # s8  -> s4

        self.mask_head = nn.Conv2d(64, 1, 1)
        self.hm_head = nn.Conv2d(64, self.K, 1)
        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, self.K),
        )

    def forward(self, x: torch.Tensor) -> dict:
        in_hw = x.shape[-2:]
        s = self.stem(x)            # s2,  64
        p = self.pool(s)            # s4,  64
        c1 = self.layer1(p)         # s4,  64
        c2 = self.layer2(c1)        # s8,  128
        c3 = self.layer3(c2)        # s16, 256
        c4 = self.layer4(c3)        # s32, 512

        d = self.up3(c4, c3)        # s16
        d = self.up2(d, c2)         # s8
        d = self.up1(d, c1)         # s4  (160x160 for 640 input)

        mask_logits = F.interpolate(self.mask_head(d), size=in_hw,
                                    mode="bilinear", align_corners=False)
        heatmaps = self.hm_head(d)              # (B,K,160,160) logits
        vis_logits = self.vis_head(c4)          # (B,K)
        return {"mask_logits": mask_logits, "heatmaps": heatmaps, "vis_logits": vis_logits}

    def soft_argmax(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """(B,K,H,W) logits -> (B,K,2) expected (x,y) in heatmap-grid units."""
        B, K, H, W = heatmaps.shape
        flat = heatmaps.reshape(B, K, -1) / self.temp
        prob = F.softmax(flat, dim=-1).reshape(B, K, H, W)
        xs = torch.linspace(0, W - 1, W, device=heatmaps.device)
        ys = torch.linspace(0, H - 1, H, device=heatmaps.device)
        ex = (prob.sum(dim=2) * xs).sum(dim=-1)   # (B,K)
        ey = (prob.sum(dim=3) * ys).sum(dim=-1)   # (B,K)
        return torch.stack([ex, ey], dim=-1)      # (B,K,2)


def keypoints_to_input(coords_hm: torch.Tensor, heatmap_stride: int = 4) -> torch.Tensor:
    """Heatmap-grid coords -> input-image (640) pixel coords."""
    return coords_hm * heatmap_stride
