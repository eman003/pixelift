"""DeOldify "Artistic" colorization generator.

A re-implementation of the inference network from DeOldify
(https://github.com/jantic/DeOldify, MIT, Copyright (c) 2018 Jason Antic):
fastai's ``DynamicUnetDeep`` on a ResNet-34 encoder with ``nf_factor=1.5``,
self-attention and blurred pixel-shuffle upsampling.

The training-time spectral and weight normalisation are folded into plain
convolution weights by :func:`convert_state_dict` when the checkpoint is
loaded; in evaluation mode that gives exactly the same weights (no power
iteration runs at inference).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# Images go in and out normalised with the ImageNet statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(out + identity)


def _resnet34_body() -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(3, 64, 7, 2, 3, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, 2, 1),
    ]
    inplanes = 64
    for planes, blocks, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
        stage = [_BasicBlock(inplanes, planes, stride)]
        stage += [_BasicBlock(planes, planes) for _ in range(blocks - 1)]
        layers.append(nn.Sequential(*stage))
        inplanes = planes
    return nn.Sequential(*layers)


def _conv(ni: int, nf: int, ks: int = 3, bn: bool = True, relu: bool = True) -> nn.Sequential:
    """fastai ``custom_conv_layer``: conv [, ReLU] [, BatchNorm] (bias only without BN)."""
    layers: list[nn.Module] = [nn.Conv2d(ni, nf, ks, padding=(ks - 1) // 2, bias=not bn)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


class _SelfAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.key = nn.Conv1d(channels, channels // 8, 1, bias=False)
        self.value = nn.Conv1d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        flat = x.view(*size[:2], -1)
        f, g, h = self.query(flat), self.key(flat), self.value(flat)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        out = self.gamma * torch.bmm(h, beta) + flat
        return out.view(*size).contiguous()


class _PixelShuffleBlur(nn.Module):
    """``CustomPixelShuffle_ICNR``: conv + BN, ReLU, pixel shuffle, 2×2 blur."""

    def __init__(self, ni: int, nf: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(ni, nf * 4, 1, bias=False), nn.BatchNorm2d(nf * 4))
        self.shuf = nn.PixelShuffle(2)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blur(self.pad(self.shuf(F.relu(self.conv(x)))))


class _UnetBlockDeep(nn.Module):
    def __init__(
        self, up_in_c: int, x_in_c: int, final_div: bool, self_attention: bool, nf_factor: float
    ) -> None:
        super().__init__()
        self.shuf = _PixelShuffleBlur(up_in_c, up_in_c // 2)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = int((ni if final_div else ni // 2) * nf_factor)
        self.conv1 = _conv(ni, nf)
        self.conv2 = _conv(nf, nf)
        if self_attention:
            self.conv2.append(_SelfAttention(nf))

    def forward(self, up_in: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up_out = self.shuf(up_in)
        if skip.shape[-2:] != up_out.shape[-2:]:
            up_out = F.interpolate(up_out, skip.shape[-2:], mode="nearest")
        cat = F.relu(torch.cat([up_out, self.bn(skip)], dim=1))
        return self.conv2(self.conv1(cat))


class _PixelShuffleICNR(nn.Module):
    """fastai ``PixelShuffle_ICNR`` with the default weight-normalised conv, no blur."""

    def __init__(self, ni: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(ni, ni * 4, 1))
        self.shuf = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuf(F.relu(self.conv(x)))


class _ResBlock(nn.Module):
    def __init__(self, nf: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_conv(nf, nf, bn=False), _conv(nf, nf, bn=False)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[1](self.layers[0](x)) + x


class DeOldifyDeep(nn.Module):
    """Input: (N, 3, H, W) grey image as RGB, ImageNet-normalised.

    Output: (N, 3, H, W) colour image, ImageNet-normalised. H and W should be
    multiples of 32 (DeOldify renders at ``render_factor × 16`` square).
    """

    _SKIPS = (6, 5, 4, 2)  # encoder layers whose outputs feed the decoder

    def __init__(self, nf_factor: float = 1.5) -> None:
        super().__init__()
        encoder = _resnet34_body()
        blocks = []
        up_in = 512
        for i, x_in in enumerate((256, 128, 64, 64)):
            final = i == 3
            block = _UnetBlockDeep(
                up_in, x_in, not final, self_attention=i == 1, nf_factor=nf_factor
            )
            blocks.append(block)
            up_in = block.conv2[0].out_channels
        last = up_in
        # Indices match fastai's SequentialEx so checkpoint keys line up;
        # parameter-free layers are placeholders.
        self.layers = nn.ModuleList(
            [
                encoder,
                nn.BatchNorm2d(512),
                nn.Identity(),  # ReLU
                nn.Sequential(_conv(512, 1024), _conv(1024, 512)),
                *blocks,
                _PixelShuffleICNR(last),
                nn.Identity(),  # merge with the input
                _ResBlock(last + 3),
                _conv(last + 3, 3, ks=1, bn=False, relu=False),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder = self.layers[0]
        feats: dict[int, torch.Tensor] = {}
        h = x
        for index, layer in enumerate(encoder):
            h = layer(h)
            if index in self._SKIPS:
                feats[index] = h
        h = self.layers[3](F.relu(self.layers[1](h)))
        for block, index in zip(self.layers[4:8], self._SKIPS, strict=True):
            h = block(h, feats[index])
        h = self.layers[8](h)
        h = torch.cat([h, x], dim=1)
        h = self.layers[11](self.layers[10](h))
        return torch.sigmoid(h) * 6.0 - 3.0  # fastai SigmoidRange(-3, 3)


def convert_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Fold spectral norm (``weight_orig/_u/_v``) and weight norm (``weight_g/_v``)."""
    state = checkpoint.get("model", checkpoint)
    out: dict[str, Any] = {}
    for key, value in state.items():
        prefix, _, name = key.rpartition(".")
        if name == "weight_orig":
            u = state[f"{prefix}.weight_u"].float()
            v = state[f"{prefix}.weight_v"].float()
            weight = value.float()
            sigma = torch.dot(u, weight.reshape(weight.shape[0], -1) @ v)
            out[f"{prefix}.weight"] = weight / sigma
        elif name == "weight_g":
            v = state[f"{prefix}.weight_v"].float()
            norm = v.reshape(v.shape[0], -1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            out[f"{prefix}.weight"] = value.float() * v / norm
        elif name in ("weight_u", "weight_v"):
            continue
        else:
            out[key] = value
    return out
