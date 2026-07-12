"""Adaptive Layer Normalization (adaLN) diffusion head.

A lightweight feedforward denoiser conditioned on external hidden states
and diffusion timesteps via adaLN modulation. No self-attention layers —
all heavy lifting is done by the upstream backbone.

Used by VibeVoice for DDPM v_prediction denoising of acoustic latents.

Weight key layout (for checkpoint compatibility):
    .noisy_images_proj.weight   [hidden_size, latent_size]
    .t_embedder.mlp.0.weight    [hidden_size, frequency_dim]
    .t_embedder.mlp.2.weight    [hidden_size, hidden_size]
    .cond_proj.weight           [cond_dim, hidden_size]
    .layers.{i}.norm.weight
    .layers.{i}.adaLN_modulation.1.weight  [3*hidden_size, cond_dim]
    .layers.{i}.ffn.gate_proj.weight
    .layers.{i}.ffn.up_proj.weight
    .layers.{i}.ffn.down_proj.weight
    .final_layer.adaLN_modulation.1.weight [2*hidden_size, cond_dim]
    .final_layer.linear.weight  [latent_size, hidden_size]
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root mean square normalization with optional learned scale."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: Optional[nn.Parameter] = (
            nn.Parameter(torch.ones(dim)) if elementwise_affine else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x.to(input_dtype)


def _modulate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Apply adaptive modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by a 2-layer MLP.

    Weight keys:
        .mlp.0.weight [hidden_size, frequency_dim]
        .mlp.2.weight [hidden_size, hidden_size]

    The mlp is nn.Sequential(Linear, SiLU, Linear) so indices are .0 and .2.
    """

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size: int = frequency_embedding_size

    @staticmethod
    def _sinusoidal_embedding(
        t: torch.Tensor, dim: int, max_period: float = 10000.0
    ) -> torch.Tensor:
        """Create sinusoidal positional embedding from integer timesteps.

        Args:
            t: [N] integer or float timesteps.
            dim: Embedding dimension (must be even).
            max_period: Controls frequency range.

        Returns:
            [N, dim] sinusoidal embedding.
        """
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half_dim, dtype=torch.float32, device=t.device)
            / half_dim
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] timestep indices (integer or float).

        Returns:
            [B, hidden_size] timestep embedding.
        """
        t_embed = self._sinusoidal_embedding(t, self.frequency_embedding_size)
        t_embed = t_embed.to(self.mlp[0].weight.dtype)
        return self.mlp(t_embed)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network for diffusion head layers.

    Weight keys:
        .gate_proj.weight [ffn_dim, embed_dim]
        .up_proj.weight   [ffn_dim, embed_dim]
        .down_proj.weight [embed_dim, ffn_dim]
    """

    def __init__(self, embed_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj: nn.Linear = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj: nn.Linear = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj: nn.Linear = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HeadLayer(nn.Module):
    """Single adaLN diffusion head layer.

    Structure:
        1. RMSNorm → adaLN modulation (shift, scale, gate)
        2. SwiGLU FFN
        3. Gated residual connection

    Weight keys:
        .norm.weight
        .adaLN_modulation.1.weight  [3*embed_dim, cond_dim]
        .ffn.{gate,up,down}_proj.weight

    The adaLN_modulation is nn.Sequential(SiLU(), Linear(...)), so the
    Linear is at index .1.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        cond_dim: int,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm: RMSNorm = RMSNorm(embed_dim, eps=norm_eps)
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        )
        self.ffn: SwiGLUFFN = SwiGLUFFN(embed_dim, ffn_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, embed_dim] input features.
            c: [B, T, cond_dim] or [B, cond_dim] conditioning (broadcast).

        Returns:
            [B, T, embed_dim]
        """
        modulation = self.adaLN_modulation(c)
        shift, scale, gate = modulation.chunk(3, dim=-1)
        h = _modulate(self.norm(x), shift, scale)
        h = self.ffn(h)
        return x + gate * h


class FinalLayer(nn.Module):
    """Final output projection with adaLN modulation.

    Weight keys:
        .adaLN_modulation.1.weight [2*hidden_size, cond_size]
        .linear.weight             [output_size, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        cond_size: int,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm: RMSNorm = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.adaLN_modulation: nn.Sequential = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        )
        self.linear: nn.Linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        modulation = self.adaLN_modulation(c)
        shift, scale = modulation.chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        return self.linear(x)


class AdaLNDiffusionHead(nn.Module):
    """Adaptive LayerNorm diffusion prediction head.

    A lightweight denoiser that predicts noise/velocity from noisy latents,
    conditioned on external hidden states and diffusion timesteps.

    Architecture:
        noisy_images_proj: Linear(latent_size → hidden_size)
        t_embedder: TimestepEmbedder → [B, hidden_size]
        cond_proj: Linear(hidden_size → cond_dim)
        c = cond_proj(condition) + t_embedder(timesteps)
        layers: N × HeadLayer (adaLN + SwiGLU FFN)
        final_layer: FinalLayer (adaLN + Linear → latent_size)
    """

    def __init__(
        self,
        hidden_size: int = 896,
        latent_size: int = 64,
        head_layers: int = 4,
        head_ffn_ratio: float = 3.0,
        rms_norm_eps: float = 1e-5,
        frequency_embedding_size: int = 256,
    ) -> None:
        super().__init__()
        self.cond_dim: int = hidden_size
        ffn_dim: int = int(hidden_size * head_ffn_ratio)

        self.noisy_images_proj: nn.Linear = nn.Linear(
            latent_size, hidden_size, bias=False
        )
        self.t_embedder: TimestepEmbedder = TimestepEmbedder(
            self.cond_dim, frequency_embedding_size
        )
        self.cond_proj: nn.Linear = nn.Linear(
            hidden_size, self.cond_dim, bias=False
        )

        self.layers: nn.ModuleList = nn.ModuleList(
            [
                HeadLayer(
                    embed_dim=hidden_size,
                    ffn_dim=ffn_dim,
                    cond_dim=self.cond_dim,
                    norm_eps=rms_norm_eps,
                )
                for _ in range(head_layers)
            ]
        )

        self.final_layer: FinalLayer = FinalLayer(
            hidden_size=hidden_size,
            output_size=latent_size,
            cond_size=self.cond_dim,
            norm_eps=rms_norm_eps,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Zero-init adaLN modulation and final output for stable training."""
        # Zero-init all adaLN modulation output weights
        for layer in self.layers:
            nn.init.zeros_(layer.adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].weight)

        # Zero-init final output projection
        nn.init.zeros_(self.final_layer.linear.weight)

        # Normal init for timestep embedder MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_latents: [B, latent_size] or [B, T, latent_size] noisy latents.
            timesteps: [B] diffusion timestep indices.
            condition: [B, hidden_size] or [B, T, hidden_size] backbone states.

        Returns:
            Same shape as noisy_latents with predicted noise/velocity.
        """
        x = self.noisy_images_proj(noisy_latents)
        t = self.t_embedder(timesteps)  # [B, cond_dim]
        cond = self.cond_proj(condition)

        # Fuse condition + timestep
        if cond.dim() == 3:
            c = cond + t.unsqueeze(1)  # [B, T, cond_dim]
        else:
            c = cond + t  # [B, cond_dim]

        for layer in self.layers:
            x = layer(x, c)

        x = self.final_layer(x, c)
        return x
