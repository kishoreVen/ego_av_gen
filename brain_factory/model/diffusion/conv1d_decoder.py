"""Hierarchical 1D convolutional decoder for latent-to-waveform generation.

A multi-stage transposed-convolution decoder with Block1D residual blocks,
depthwise convolution mixers, and causal padding for streaming support.

Used by VibeVoice's acoustic tokenizer to decode 64-dim latents into 24kHz audio.

Weight key layout for the decoder (checkpoint compatibility):
    .upsample_layers.0.0.conv.conv.*        (first stage: SConv1d)
    .upsample_layers.{1-N}.0.convtr.convtr.* (remaining: SConvTranspose1d)
    .stages.{0-N}.{block_idx}.*             (Block1D residual blocks)
    .head.conv.conv.*                        (final SConv1d: channels -> audio)
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------------------------


def _get_extra_padding_for_conv1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0,
) -> int:
    """Compute extra padding needed to ensure output length is correct."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + kernel_size - padding_total
    return ideal_length - length


def _pad1d(
    x: torch.Tensor,
    paddings: tuple[int, int],
    mode: str = "zero",
    value: float = 0.0,
) -> torch.Tensor:
    """Pad 1D tensor handling edge cases for reflect mode."""
    left, right = paddings
    if mode == "reflect":
        max_pad = x.shape[-1] - 1
        if left > max_pad or right > max_pad:
            # Fall back to constant padding when input is too short
            return F.pad(x, (left, right), mode="constant", value=value)
    if mode == "zero":
        mode = "constant"
    return F.pad(x, (left, right), mode=mode, value=value)


def _unpad1d(x: torch.Tensor, paddings: tuple[int, int]) -> torch.Tensor:
    """Remove left/right padding from 1D tensor."""
    left, right = paddings
    end = x.shape[-1] - right if right > 0 else x.shape[-1]
    return x[..., left:end]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class ConvRMSNorm(nn.Module):
    """RMS normalization for 1D convolution tensors [B, C, T].

    Transposes to [B, T, C] for normalization, then transposes back.

    Weight key: .weight [dim]
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(dim))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        input_dtype = x.dtype
        x = x.transpose(1, 2).float()  # [B, T, C]
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = (x * self.weight).to(input_dtype)
        return x.transpose(1, 2)  # [B, C, T]


# ---------------------------------------------------------------------------
# Convolution wrappers
# ---------------------------------------------------------------------------


class NormConv1d(nn.Module):
    """Conv1d with optional weight normalization.

    Weight keys: .conv.weight, .conv.bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class NormConvTranspose1d(nn.Module):
    """ConvTranspose1d with optional weight normalization.

    Weight keys: .convtr.weight, .convtr.bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.convtr: nn.ConvTranspose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convtr(x)


class SConv1d(nn.Module):
    """Causal 1D convolution with streaming support.

    For causal mode: left-pads with (kernel_size - 1) * dilation to ensure
    output depends only on past + current input.

    Weight keys: .conv.conv.weight, .conv.conv.bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.conv: NormConv1d = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.causal: bool = causal
        self.pad_mode: str = pad_mode
        self.stride: int = stride
        self.kernel_size: int = kernel_size
        self.dilation: int = dilation

        # Compute total padding
        self._padding_total: int = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T]

        Returns:
            [B, C_out, T_out] where T_out = ceil(T / stride) for causal.
        """
        extra_padding = _get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, self._padding_total
        )

        if self.causal:
            # All padding on the left for causal
            x = _pad1d(
                x,
                (self._padding_total, int(extra_padding)),
                mode=self.pad_mode,
            )
        else:
            # Symmetric padding
            left = self._padding_total // 2
            right = self._padding_total - left + int(extra_padding)
            x = _pad1d(x, (left, right), mode=self.pad_mode)

        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """Causal transposed 1D convolution for upsampling.

    Weight keys: .convtr.convtr.weight, .convtr.convtr.bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.convtr: NormConvTranspose1d = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            bias=bias,
        )
        self.causal: bool = causal
        self.stride: int = stride
        self.kernel_size: int = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, T]

        Returns:
            [B, C_out, T * stride]
        """
        y = self.convtr(x)

        # Remove padding to get exact output length
        padding_total = self.kernel_size - self.stride
        if self.causal:
            # Remove right padding for causal
            y = _unpad1d(y, (0, padding_total))
        else:
            right = padding_total // 2
            left = padding_total - right
            y = _unpad1d(y, (left, right))

        return y


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Convlayer(nn.Module):
    """Depthwise convolution mixer for Block1D.

    Weight keys: .conv.conv.conv.weight, .conv.conv.conv.bias
    (triple-nested: Convlayer → SConv1d → NormConv1d → Conv1d)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        causal: bool = True,
        pad_mode: str = "constant",
    ) -> None:
        super().__init__()
        self.conv: SConv1d = SConv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            groups=dim,  # depthwise
            causal=causal,
            pad_mode=pad_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FFN(nn.Module):
    """Two-layer GELU feed-forward network.

    Weight keys: .linear1.weight, .linear1.bias, .linear2.weight, .linear2.bias
    """

    def __init__(self, embed_dim: int, ffn_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear1: nn.Linear = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.linear2: nn.Linear = nn.Linear(ffn_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class Block1D(nn.Module):
    """Residual block: depthwise conv mixer + FFN with layer scale.

    Weight keys:
        .norm.weight, .ffn_norm.weight
        .mixer.conv.conv.conv.weight, .mixer.conv.conv.conv.bias
        .ffn.linear1.weight, .ffn.linear1.bias
        .ffn.linear2.weight, .ffn.linear2.bias
        .gamma [dim], .ffn_gamma [dim]
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        causal: bool = True,
        pad_mode: str = "constant",
        norm_eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm: ConvRMSNorm = ConvRMSNorm(dim, eps=norm_eps)
        self.mixer: Convlayer = Convlayer(
            dim, kernel_size=kernel_size, causal=causal, pad_mode=pad_mode
        )
        self.gamma: nn.Parameter = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )

        ffn_dim = int(dim * 4)
        self.ffn_norm: ConvRMSNorm = ConvRMSNorm(dim, eps=norm_eps)
        self.ffn: FFN = FFN(dim, ffn_dim, bias=bias)
        self.ffn_gamma: nn.Parameter = nn.Parameter(
            layer_scale_init_value * torch.ones(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]

        Returns:
            [B, C, T]
        """
        # Depthwise conv branch
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = x * self.gamma.unsqueeze(-1)
        x = residual + x

        # FFN branch
        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)  # [B, T, C]
        x = self.ffn(x)
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class HierarchicalConv1dDecoder(nn.Module):
    """Multi-stage hierarchical 1D convolutional decoder.

    Progressively upsamples latent representations through transposed
    convolution stages, each followed by residual Block1D blocks.

    Default VibeVoice config:
        vae_dim=64, channels=1, n_filters=32
        ratios=[8,5,5,4,2,2], depths="3-3-3-3-3-3-8"

    Channel progression (decoder):
        Input: vae_dim (64)
        Stage 0: vae_dim -> n_filters * 2^(N_ratios) = 32 * 64 = 2048 (SConv1d, no upsample)
        Stage 1: 2048 -> 1024 (SConvTranspose1d, upsample by 2)
        Stage 2: 1024 -> 512 (SConvTranspose1d, upsample by 2)
        Stage 3: 512 -> 256 (SConvTranspose1d, upsample by 4)
        Stage 4: 256 -> 128 (SConvTranspose1d, upsample by 5)
        Stage 5: 128 -> 64 (SConvTranspose1d, upsample by 5)
        Stage 6: 64 -> 32 (SConvTranspose1d, upsample by 8)
        Head: 32 -> channels (1) (SConv1d)

    Total upsample: 2*2*4*5*5*8 = 3200x

    Weight keys:
        .upsample_layers.{i}.0.*    (SConv1d or SConvTranspose1d)
        .stages.{i}.{j}.*           (Block1D)
        .head.*                     (SConv1d)
    """

    def __init__(
        self,
        vae_dim: int = 64,
        channels: int = 1,
        n_filters: int = 32,
        ratios: Optional[List[int]] = None,
        depths: str = "3-3-3-3-3-3-8",
        causal: bool = True,
        pad_mode: str = "constant",
        layer_scale_init_value: float = 1e-6,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if ratios is None:
            ratios = [8, 5, 5, 4, 2, 2]

        # Parse depths: reversed for decoder (encoder had them forward)
        depth_list: List[int] = [int(d) for d in depths.split("-")]
        # Decoder depths are reversed from encoder depths
        decoder_depths: List[int] = list(reversed(depth_list))

        n_ratios = len(ratios)
        n_stages = n_ratios + 1  # +1 for the initial projection stage

        # Channel progression for decoder (from deep to shallow)
        # In the decoder: channels go from largest to smallest
        channels_list: List[int] = []
        for i in range(n_ratios + 1):
            channels_list.append(n_filters * (2 ** (n_ratios - i)))
        # channels_list for default: [2048, 1024, 512, 256, 128, 64, 32]

        # Upsample layers: one per stage
        self.upsample_layers: nn.ModuleList = nn.ModuleList()

        # Stage 0: SConv1d projection (no upsampling)
        self.upsample_layers.append(
            nn.ModuleList(
                [
                    SConv1d(
                        vae_dim,
                        channels_list[0],
                        kernel_size=7,
                        causal=causal,
                        pad_mode=pad_mode,
                    )
                ]
            )
        )

        # Stages 1..N: SConvTranspose1d upsampling
        for i in range(n_ratios):
            ratio = ratios[i]
            self.upsample_layers.append(
                nn.ModuleList(
                    [
                        SConvTranspose1d(
                            channels_list[i],
                            channels_list[i + 1],
                            kernel_size=ratio * 2,
                            stride=ratio,
                            causal=causal,
                        )
                    ]
                )
            )

        # Residual blocks per stage
        self.stages: nn.ModuleList = nn.ModuleList()
        for i in range(n_stages):
            n_blocks = decoder_depths[i] if i < len(decoder_depths) else 3
            stage_blocks = nn.ModuleList(
                [
                    Block1D(
                        dim=channels_list[i],
                        kernel_size=7,
                        layer_scale_init_value=layer_scale_init_value,
                        causal=causal,
                        pad_mode=pad_mode,
                        norm_eps=norm_eps,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.stages.append(stage_blocks)

        # Final head: project to audio channels
        self.head: SConv1d = SConv1d(
            channels_list[-1],
            channels,
            kernel_size=7,
            causal=causal,
            pad_mode=pad_mode,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to audio waveform.

        Args:
            latents: [B, vae_dim, T_latent] latent codes (channel-first).

        Returns:
            [B, channels, T_audio] reconstructed audio at full sample rate.
        """
        x = latents

        for i, (upsample, blocks) in enumerate(
            zip(self.upsample_layers, self.stages)
        ):
            # Upsample (or project for stage 0)
            x = upsample[0](x)

            # Apply residual blocks
            for block in blocks:
                x = block(x)

        # Final projection to audio channels
        x = self.head(x)
        return x
