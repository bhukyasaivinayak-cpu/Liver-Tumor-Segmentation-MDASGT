"""Runnable, paper-inspired PFWF + CRes2Net + MDA-SGT segmentation model."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.net(x)


class Res2SEBlock(nn.Module):
    """Four-scale hierarchical Res2Net block with channel recalibration."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, scale: int = 4):
        super().__init__()
        width = out_ch // scale
        self.scale, self.width = scale, width
        self.reduce = nn.Sequential(nn.Conv2d(in_ch, width * scale, 1, stride=stride, bias=False),
                                    nn.BatchNorm2d(width * scale), nn.ReLU(inplace=True))
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                          nn.BatchNorm2d(width), nn.ReLU(inplace=True)) for _ in range(scale - 1)
        ])
        self.fuse = nn.Sequential(nn.Conv2d(width * scale, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.se = SEBlock(out_ch)
        self.skip = nn.Identity() if in_ch == out_ch and stride == 1 else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        residual = self.skip(x)
        groups = self.reduce(x).split(self.width, 1)
        outputs = [groups[0]]
        running = groups[1]
        for i, conv in enumerate(self.branches):
            if i > 0:
                running = running + groups[i + 1]
            running = conv(running)
            outputs.append(running)
        return F.relu(self.se(self.fuse(torch.cat(outputs, 1))) + residual, inplace=True)


class TransformerSpatialAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 4, layers: int = 2):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 1024, channels))
        layer = nn.TransformerEncoderLayer(channels, heads, channels * 4, dropout=.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        if tokens.size(1) <= self.pos.size(1):
            pos = self.pos[:, :tokens.size(1)]
        else:
            pos = F.interpolate(self.pos.transpose(1, 2), size=tokens.size(1), mode="linear").transpose(1, 2)
        return self.encoder(tokens + pos).transpose(1, 2).reshape(b, c, h, w)


class GlobalSpatialAttention(nn.Module):
    """Efficient spatial affinity approximation to the paper's GSA branch."""
    def __init__(self, channels: int):
        super().__init__()
        qk = max(channels // 8, 8)
        self.q, self.k, self.v = nn.Conv2d(channels, qk, 1), nn.Conv2d(channels, qk, 1), nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(x).flatten(2)
        a = torch.softmax(torch.bmm(q, k) / math.sqrt(k.size(1)), dim=-1)
        v = self.v(x).flatten(2)
        context = torch.bmm(v, a.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.gamma * context


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch),
                                 nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)


class MDASGT(nn.Module):
    """2D binary segmentation network faithful to the paper's named components."""
    def __init__(self, in_channels: int = 1, base: int = 32):
        super().__init__()
        self.stem = ConvBlock(in_channels, base)
        self.e2 = Res2SEBlock(base, base * 2, stride=2)
        self.e3 = Res2SEBlock(base * 2, base * 4, stride=2)
        self.e4 = Res2SEBlock(base * 4, base * 8, stride=2)
        self.tsa = TransformerSpatialAttention(base * 8)
        self.gsa = GlobalSpatialAttention(base * 8)
        self.mix = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.d3 = ConvBlock(base * 8 + base * 4, base * 4)
        self.d2 = ConvBlock(base * 4 + base * 2, base * 2)
        self.d1 = ConvBlock(base * 2 + base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        f1 = self.stem(x); f2 = self.e2(f1); f3 = self.e3(f2); f4 = self.e4(f3)
        weights = torch.softmax(self.mix, dim=0)
        z = weights[0] * self.tsa(f4) + weights[1] * self.gsa(f4)
        z = self.d3(torch.cat([F.interpolate(z, f3.shape[-2:], mode="bilinear", align_corners=False), f3], 1))
        z = self.d2(torch.cat([F.interpolate(z, f2.shape[-2:], mode="bilinear", align_corners=False), f2], 1))
        z = self.d1(torch.cat([F.interpolate(z, f1.shape[-2:], mode="bilinear", align_corners=False), f1], 1))
        return self.out(z)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum((1, 2, 3))
    dice = (2 * intersection + eps) / (prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return 1 - dice.mean()


def combined_loss(logits, target):
    return .5 * F.binary_cross_entropy_with_logits(logits, target) + .5 * dice_loss(logits, target)

