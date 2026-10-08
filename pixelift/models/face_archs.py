"""Face restoration and face detection networks.

GFPGANv1Clean is a re-implementation of the "clean" GFPGAN architecture
(https://github.com/TencentARC/GFPGAN, Apache-2.0, Copyright (c) 2021 THL A29
Limited, a Tencent company), whose StyleGAN2 decoder derives from
stylegan2-pytorch (MIT, Kim Seonghyeon). The clean variant needs no compiled
CUDA extensions, so it runs on any PyTorch device.

RetinaFace is a re-implementation of the detector shipped with facexlib
(https://github.com/xinntao/facexlib, MIT), itself based on
Pytorch_Retinaface (MIT, biubug6), with a torchvision-compatible ResNet-50.

Parameter names match the official weights so they load with ``strict=True``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# --- GFPGAN -----------------------------------------------------------------


class _NormStyleCode(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class _ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_style_feat: int,
        demodulate: bool = True,
        upsample: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.eps = eps
        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
            / math.sqrt(in_channels * kernel_size**2)
        )
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)
        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        h, w = x.shape[2:]
        out = F.conv2d(x.reshape(1, b * c, h, w), weight, padding=self.padding, groups=b)
        return out.view(b, self.out_channels, *out.shape[2:4])


class _StyleConv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_style_feat: int, upsample: bool
    ) -> None:
        super().__init__()
        self.modulated_conv = _ModulatedConv2d(
            in_channels, out_channels, 3, num_style_feat, upsample=upsample
        )
        self.weight = nn.Parameter(torch.zeros(1))  # noise strength
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        out = self.modulated_conv(x, style) * 2**0.5
        out = out + self.weight * noise.to(out.dtype)
        return self.activate(out + self.bias)


class _ToRGB(nn.Module):
    def __init__(self, in_channels: int, num_style_feat: int, upsample: bool = True) -> None:
        super().__init__()
        self.upsample = upsample
        self.modulated_conv = _ModulatedConv2d(in_channels, 3, 1, num_style_feat, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.modulated_conv(x, style) + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip
        return out


class _ConstantInput(nn.Module):
    def __init__(self, num_channel: int, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, num_channel, size, size))

    def forward(self, batch: int) -> torch.Tensor:
        return self.weight.repeat(batch, 1, 1, 1)


def _channels(narrow: float, channel_multiplier: int) -> dict[int, int]:
    return {
        4: int(512 * narrow),
        8: int(512 * narrow),
        16: int(512 * narrow),
        32: int(512 * narrow),
        64: int(256 * channel_multiplier * narrow),
        128: int(128 * channel_multiplier * narrow),
        256: int(64 * channel_multiplier * narrow),
        512: int(32 * channel_multiplier * narrow),
        1024: int(16 * channel_multiplier * narrow),
    }


class _StyleGAN2GeneratorCSFT(nn.Module):
    """StyleGAN2 decoder with spatial feature transform (SFT) conditions.

    Always uses the stored noise maps, so restoring the same face twice gives
    the same result (the reference implementation draws fresh noise).
    """

    def __init__(
        self,
        out_size: int,
        num_style_feat: int = 512,
        num_mlp: int = 8,
        channel_multiplier: int = 2,
        narrow: float = 1,
        sft_half: bool = True,
    ) -> None:
        super().__init__()
        self.sft_half = sft_half
        layers: list[nn.Module] = [_NormStyleCode()]
        for _ in range(num_mlp):
            layers += [nn.Linear(num_style_feat, num_style_feat), nn.LeakyReLU(0.2, inplace=True)]
        self.style_mlp = nn.Sequential(*layers)  # unused: GFPGAN feeds latents directly
        channels = _channels(narrow, channel_multiplier)
        self.constant_input = _ConstantInput(channels[4], size=4)
        self.style_conv1 = _StyleConv(channels[4], channels[4], num_style_feat, upsample=False)
        self.to_rgb1 = _ToRGB(channels[4], num_style_feat, upsample=False)
        self.log_size = int(math.log(out_size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        for layer_idx in range(self.num_layers):
            resolution = 2 ** ((layer_idx + 5) // 2)
            self.noises.register_buffer(
                f"noise{layer_idx}", torch.randn(1, 1, resolution, resolution)
            )
        in_channels = channels[4]
        for i in range(3, self.log_size + 1):
            out_channels = channels[2**i]
            self.style_convs.append(
                _StyleConv(in_channels, out_channels, num_style_feat, upsample=True)
            )
            self.style_convs.append(
                _StyleConv(out_channels, out_channels, num_style_feat, upsample=False)
            )
            self.to_rgbs.append(_ToRGB(out_channels, num_style_feat))
            in_channels = out_channels

    def forward(self, latent: torch.Tensor, conditions: list[torch.Tensor]) -> torch.Tensor:
        noise = [getattr(self.noises, f"noise{i}") for i in range(self.num_layers)]
        out = self.constant_input(latent.shape[0])
        out = self.style_conv1(out, latent[:, 0], noise[0])
        skip = self.to_rgb1(out, latent[:, 1])
        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.style_convs[::2],
            self.style_convs[1::2],
            noise[1::2],
            noise[2::2],
            self.to_rgbs,
            strict=False,
        ):
            out = conv1(out, latent[:, i], noise1)
            if i < len(conditions):
                if self.sft_half:
                    same, sft = torch.split(out, out.size(1) // 2, dim=1)
                    out = torch.cat([same, sft * conditions[i - 1] + conditions[i]], dim=1)
                else:
                    out = out * conditions[i - 1] + conditions[i]
            out = conv2(out, latent[:, i + 1], noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2
        return skip


class _ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.scale_factor = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        out = F.interpolate(
            out, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return out + self.skip(x)


class GFPGANv1Clean(nn.Module):
    """U-Net encoder + StyleGAN2 decoder (the configuration of GFPGAN v1.3 / v1.4).

    Input: aligned 512×512 RGB face, values in [-1, 1]. Output: same.
    """

    def __init__(
        self,
        out_size: int = 512,
        num_style_feat: int = 512,
        channel_multiplier: int = 2,
        num_mlp: int = 8,
        narrow: float = 1,
        sft_half: bool = True,
    ) -> None:
        super().__init__()
        self.num_style_feat = num_style_feat
        channels = _channels(narrow * 0.5, channel_multiplier)
        self.log_size = int(math.log(out_size, 2))
        self.conv_body_first = nn.Conv2d(3, channels[2**self.log_size], 1)
        in_channels = channels[2**self.log_size]
        self.conv_body_down = nn.ModuleList()
        for i in range(self.log_size, 2, -1):
            out_channels = channels[2 ** (i - 1)]
            self.conv_body_down.append(_ResBlock(in_channels, out_channels, 0.5))
            in_channels = out_channels
        self.final_conv = nn.Conv2d(in_channels, channels[4], 3, 1, 1)
        in_channels = channels[4]
        self.conv_body_up = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[2**i]
            self.conv_body_up.append(_ResBlock(in_channels, out_channels, 2))
            in_channels = out_channels
        # Intermediate RGB outputs (training only); kept so the weights load strictly.
        self.toRGB = nn.ModuleList(
            nn.Conv2d(channels[2**i], 3, 1) for i in range(3, self.log_size + 1)
        )
        self.final_linear = nn.Linear(channels[4] * 4 * 4, (self.log_size * 2 - 2) * num_style_feat)
        self.stylegan_decoder = _StyleGAN2GeneratorCSFT(
            out_size, num_style_feat, num_mlp, channel_multiplier, narrow, sft_half
        )
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            out_channels = channels[2**i]
            sft_out = out_channels if sft_half else out_channels * 2
            for group in (self.condition_scale, self.condition_shift):
                group.append(
                    nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                        nn.LeakyReLU(0.2, True),
                        nn.Conv2d(out_channels, sft_out, 3, 1, 1),
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conditions: list[torch.Tensor] = []
        skips: list[torch.Tensor] = []
        feat = F.leaky_relu_(self.conv_body_first(x), negative_slope=0.2)
        for block in self.conv_body_down:
            feat = block(feat)
            skips.insert(0, feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)
        latent = self.final_linear(feat.reshape(feat.size(0), -1))
        latent = latent.view(latent.size(0), -1, self.num_style_feat)
        for i, block in enumerate(self.conv_body_up):
            feat = block(feat + skips[i])
            conditions.append(self.condition_scale[i](feat))
            conditions.append(self.condition_shift[i](feat))
        return self.stylegan_decoder(latent, conditions)


# --- RetinaFace -------------------------------------------------------------


class _Bottleneck(nn.Module):
    """torchvision ResNet-50 bottleneck (stride on the 3×3 convolution)."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, 1, stride, bias=False), nn.BatchNorm2d(planes * 4)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        identity = x if self.downsample is None else self.downsample(x)
        return self.relu(out + identity)


class _ResNet50Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        inplanes = 64
        for index, (planes, blocks, stride) in enumerate(
            ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)), start=1
        ):
            layers = [_Bottleneck(inplanes, planes, stride)]
            inplanes = planes * 4
            layers += [_Bottleneck(inplanes, planes) for _ in range(blocks - 1)]
            setattr(self, f"layer{index}", nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer2(self.layer1(x))
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c2, c3, c4]


def _conv_bn(inp: int, oup: int, leaky: float = 0, relu: bool = True, k: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(inp, oup, k, 1, k // 2, bias=False), nn.BatchNorm2d(oup)]
    if relu:
        layers.append(nn.LeakyReLU(negative_slope=leaky, inplace=True))
    return nn.Sequential(*layers)


class _SSH(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3X3 = _conv_bn(channels, channels // 2, relu=False)
        self.conv5X5_1 = _conv_bn(channels, channels // 4)
        self.conv5X5_2 = _conv_bn(channels // 4, channels // 4, relu=False)
        self.conv7X7_2 = _conv_bn(channels // 4, channels // 4)
        self.conv7x7_3 = _conv_bn(channels // 4, channels // 4, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c5_1 = self.conv5X5_1(x)
        out = torch.cat(
            [self.conv3X3(x), self.conv5X5_2(c5_1), self.conv7x7_3(self.conv7X7_2(c5_1))], dim=1
        )
        return F.relu(out)


class _FPN(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int) -> None:
        super().__init__()
        self.output1 = _conv_bn(in_channels[0], out_channels, k=1)
        self.output2 = _conv_bn(in_channels[1], out_channels, k=1)
        self.output3 = _conv_bn(in_channels[2], out_channels, k=1)
        self.merge1 = _conv_bn(out_channels, out_channels)
        self.merge2 = _conv_bn(out_channels, out_channels)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        o1, o2, o3 = self.output1(feats[0]), self.output2(feats[1]), self.output3(feats[2])
        o2 = self.merge2(o2 + F.interpolate(o3, size=o2.shape[2:], mode="nearest"))
        o1 = self.merge1(o1 + F.interpolate(o2, size=o1.shape[2:], mode="nearest"))
        return [o1, o2, o3]


class _Head(nn.Module):
    def __init__(self, channels: int, outputs: int) -> None:
        super().__init__()
        self.outputs = outputs
        self.conv1x1 = nn.Conv2d(channels, 2 * outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1)
        return out.reshape(out.shape[0], -1, self.outputs)


class RetinaFace(nn.Module):
    """RetinaFace with a ResNet-50 backbone (facexlib ``retinaface_resnet50``).

    Input: BGR float tensor with the per-channel mean (104, 117, 123)
    subtracted. Output: (box offsets, class probabilities, landmark offsets)
    per prior box; see ``pixelift.core.restoration.faces`` for decoding.
    """

    MIN_SIZES = ((16, 32), (64, 128), (256, 512))
    STEPS = (8, 16, 32)
    VARIANCE = (0.1, 0.2)

    def __init__(self) -> None:
        super().__init__()
        self.body = _ResNet50Body()
        self.fpn = _FPN([512, 1024, 2048], 256)
        self.ssh1 = _SSH(256)
        self.ssh2 = _SSH(256)
        self.ssh3 = _SSH(256)
        self.ClassHead = nn.ModuleList(_Head(256, 2) for _ in range(3))
        self.BboxHead = nn.ModuleList(_Head(256, 4) for _ in range(3))
        self.LandmarkHead = nn.ModuleList(_Head(256, 10) for _ in range(3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fpn = self.fpn(self.body(x))
        feats = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        boxes = torch.cat([h(f) for h, f in zip(self.BboxHead, feats, strict=True)], dim=1)
        scores = torch.cat([h(f) for h, f in zip(self.ClassHead, feats, strict=True)], dim=1)
        marks = torch.cat([h(f) for h, f in zip(self.LandmarkHead, feats, strict=True)], dim=1)
        return boxes, F.softmax(scores, dim=-1), marks


def strip_module_prefix(state: dict) -> dict:
    """Checkpoints saved from ``nn.DataParallel`` prefix every key with ``module.``."""
    return {k.removeprefix("module."): v for k, v in state.items()}
