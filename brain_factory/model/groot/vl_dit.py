"""Vision-Language Diffusion Transformer (VLDiT) for GR00T action prediction.

Port of NVIDIA Isaac-GR00T gr00t/model/modules/dit.py — DiT and AlternateVLDiT.
Diffusers ModelMixin/ConfigMixin replaced with plain nn.Module so the module
fits naturally into brain_factory. All attribute names are preserved for
state dict compatibility.

Load from a GR00T checkpoint via:
    CheckpointMapping(path=ckpt, source_prefix="action_head.model.", target_prefix="")

Weight key layout:

TimestepEncoder:
    .time_proj                                  (Timesteps — no learnable params)
    .timestep_embedder.linear_1.weight          [time_embed_dim, 256]
    .timestep_embedder.linear_1.bias            [time_embed_dim]
    .timestep_embedder.linear_2.weight          [time_embed_dim, time_embed_dim]
    .timestep_embedder.linear_2.bias            [time_embed_dim]

AdaLayerNorm (custom — NOT diffusers'):
    .silu                                       (no params)
    .linear.weight                              [2*embedding_dim, embedding_dim]
    .linear.bias                                [2*embedding_dim]
    .norm.weight / .norm.bias                   (only when norm_elementwise_affine=True)

BasicTransformerBlock — cross-attn blocks (even idx in AlternateVLDiT):
    .norm1.linear.weight / .norm1.linear.bias
    .norm1.norm.weight / .norm1.norm.bias       (omitted if norm_elementwise_affine=False)
    .attn1.to_q.weight                          Q from hidden_states
    .attn1.to_k.weight                          K from encoder_hidden_states [cross_attention_dim]
    .attn1.to_v.weight                          V from encoder_hidden_states [cross_attention_dim]
    .attn1.to_out.0.weight / .attn1.to_out.0.bias
    .norm3.weight / .norm3.bias
    .ff.net.0.proj.weight / .ff.net.0.proj.bias (GEGLU gate+up fused)
    .ff.net.2.weight / .ff.net.2.bias           (output projection)

BasicTransformerBlock — self-attn blocks (odd idx in AlternateVLDiT):
    same as above but .attn1.to_k / .attn1.to_v project from inner_dim (not cross_attention_dim)

DiT / AlternateVLDiT:
    .timestep_encoder.*
    .transformer_blocks.{0..N-1}.*
    .norm_out.weight / .norm_out.bias           (elementwise_affine=False → no weights)
    .proj_out_1.weight / .proj_out_1.bias       [2*inner_dim, inner_dim]  (adaLN shift/scale)
    .proj_out_2.weight / .proj_out_2.bias       [output_dim, inner_dim]   (final projection)

Architecture notes:
    inner_dim = num_attention_heads * attention_head_dim  (e.g. 32*48 = 1536 for N1.6)
    output_dim = hidden_size of the action head           (e.g. 1024 for N1.6)

    AlternateVLDiT alternates cross-attention and self-attention blocks:
        even blocks → cross-attn to VL embeddings (attending text or image tokens alternately)
        odd blocks  → self-attention over action+state sequence

    The final output layer uses the timestep embedding for adaLN shift/scale:
        shift, scale = proj_out_1(silu(temb)).chunk(2)
        out = proj_out_2(norm_out(x) * (1 + scale) + shift)
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


# ---------------------------------------------------------------------------
# SDPA workaround for NVIDIA Spark (sm121) GPUs
# ---------------------------------------------------------------------------


def _should_force_math_sdpa() -> bool:
    override = os.environ.get("GR00T_DIT_SDPA_MODE")
    if override == "math":
        return True
    if override == "default":
        return False
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return (major, minor) == (12, 1)
    return False


def _sdpa_context():
    if not _should_force_math_sdpa():
        return nullcontext()
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
        enable_cudnn=False,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VLDiTConfig:
    """Construction parameters for VLDiT / AlternateVLDiT.

    Defaults match GR00T N1.6 (AlternateVLDiT).
    """

    num_attention_heads: int = 32
    attention_head_dim: int = 48
    output_dim: int = 1024           # hidden_size of the action head
    num_layers: int = 32
    dropout: float = 0.2
    attention_bias: bool = True
    activation_fn: str = "geglu"
    upcast_attention: bool = False
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-5
    max_num_positional_embeddings: int = 512
    final_dropout: bool = True
    positional_embeddings: Optional[str] = None    # None = no sinusoidal pos embed in blocks
    interleave_self_attention: bool = True          # alternates cross / self attention blocks
    cross_attention_dim: Optional[int] = 2048      # backbone_embedding_dim
    attend_text_every_n_blocks: int = 2            # AlternateVLDiT only


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class TimestepEncoder(nn.Module):
    """Converts integer diffusion timestep bucket indices to a dense embedding.

    Uses a 256-channel sinusoidal projection (Timesteps, no learnable params)
    followed by a two-layer MLP (TimestepEmbedding).

    Weight keys:
        .time_proj                              (non-learnable)
        .timestep_embedder.linear_1.weight/bias
        .timestep_embedder.linear_2.weight/bias
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj: Timesteps = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder: TimestepEmbedding = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] int64 discretised timestep indices
        Returns:
            [B, embedding_dim] timestep embedding
        """
        dtype = next(self.parameters()).dtype
        proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(proj)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalisation conditioned on a continuous embedding.

    Custom implementation matching GR00T's (not diffusers' AdaLayerNorm).
    The conditioning vector (temb) is projected to shift+scale via a linear
    layer, then applied after LayerNorm.

    forward(x, temb):
        temb = linear(silu(temb))            # [B, 2*D]
        scale, shift = temb.chunk(2, dim=1)
        out  = norm(x) * (1+scale[:,None]) + shift[:,None]

    Weight keys:
        .silu                           (no params)
        .linear.weight                  [2*embedding_dim, embedding_dim]
        .linear.bias                    [2*embedding_dim]
        .norm.weight / .norm.bias       (only if norm_elementwise_affine=True)
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.silu: nn.SiLU = nn.SiLU()
        self.linear: nn.Linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm: nn.LayerNorm = nn.LayerNorm(
            embedding_dim, norm_eps, norm_elementwise_affine
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, T, embedding_dim]
            temb: [B, embedding_dim] timestep embedding
        Returns:
            [B, T, embedding_dim]
        """
        temb = self.linear(self.silu(temb))             # [B, 2*D]
        scale, shift = temb.chunk(2, dim=1)             # each [B, D]
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    """Single transformer block: (optional adaLN) self/cross-attn + FFN.

    This is the fundamental building block of VLDiT.  Each block has:
      norm1  → attn1  (self-attn if cross_attention_dim=None, else cross-attn to VL)
      norm3  → ff     (GEGLU feed-forward)

    norm1 is AdaLayerNorm when norm_type="ada_norm" (timestep-conditioned),
    plain LayerNorm otherwise.

    Weight keys (cross-attn block, ada_norm):
        .norm1.linear.weight / .norm1.linear.bias
        .norm1.norm.weight / .norm1.norm.bias       (if norm_elementwise_affine=True)
        .attn1.to_q.weight
        .attn1.to_k.weight
        .attn1.to_v.weight
        .attn1.to_out.0.weight / .attn1.to_out.0.bias
        .norm3.weight / .norm3.bias
        .ff.net.0.proj.weight / .ff.net.0.proj.bias
        .ff.net.2.weight / .ff.net.2.bias
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "ada_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm_type: str = norm_type

        # 1. Self-attention (or cross-attention when cross_attention_dim is not None)
        if norm_type == "ada_norm":
            self.norm1: nn.Module = AdaLayerNorm(
                dim,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1: Attention = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Feed-forward
        self.norm3: nn.LayerNorm = nn.LayerNorm(
            dim, norm_eps, norm_elementwise_affine
        )
        self.ff: FeedForward = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout: Optional[nn.Dropout] = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:          [B, T, dim] action+state sequence
            attention_mask:         [B, T] optional self-attention mask
            encoder_hidden_states:  [B, S, cross_attention_dim] VL embeddings (cross-attn only)
            encoder_attention_mask: [B, S] VL attention mask
            temb:                   [B, dim] timestep embedding for adaLN
        Returns:
            [B, T, dim]
        """
        # Normalise + attend
        if self.norm_type == "ada_norm":
            norm_hidden = self.norm1(hidden_states, temb)
        else:
            norm_hidden = self.norm1(hidden_states)

        with _sdpa_context():
            attn_out = self.attn1(
                norm_hidden,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=(
                    encoder_attention_mask
                    if encoder_hidden_states is not None
                    else attention_mask
                ),
            )

        if self.final_dropout is not None:
            attn_out = self.final_dropout(attn_out)
        hidden_states = attn_out + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # Feed-forward
        ff_out = self.ff(self.norm3(hidden_states))
        hidden_states = ff_out + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


class VLDiT(nn.Module):
    """Diffusion Transformer conditioned on VL embeddings.

    This is a brain_factory port of GR00T's DiT class.
    Load pretrained weights with:
        CheckpointMapping(path=ckpt, source_prefix="action_head.model.", target_prefix="")

    Forward: (noisy_actions, vl_embeddings, timestep) → denoised_actions

    Weight keys:
        .timestep_encoder.*
        .transformer_blocks.{i}.*
        .norm_out.weight / .norm_out.bias   (elementwise_affine=False → no weights)
        .proj_out_1.weight / .proj_out_1.bias
        .proj_out_2.weight / .proj_out_2.bias
    """

    def __init__(self, cfg: VLDiTConfig) -> None:
        super().__init__()
        self.cfg: VLDiTConfig = cfg
        self.inner_dim: int = cfg.num_attention_heads * cfg.attention_head_dim

        self.timestep_encoder: TimestepEncoder = TimestepEncoder(self.inner_dim)

        blocks: List[BasicTransformerBlock] = []
        for idx in range(cfg.num_layers):
            # Interleave: odd blocks are self-attention only (no VL cross-attention)
            use_self_attn = (idx % 2 == 1) and cfg.interleave_self_attention
            block_cross_dim = None if use_self_attn else cfg.cross_attention_dim
            blocks.append(
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=cfg.num_attention_heads,
                    attention_head_dim=cfg.attention_head_dim,
                    dropout=cfg.dropout,
                    activation_fn=cfg.activation_fn,
                    attention_bias=cfg.attention_bias,
                    upcast_attention=cfg.upcast_attention,
                    norm_elementwise_affine=cfg.norm_elementwise_affine,
                    norm_type="ada_norm",
                    norm_eps=cfg.norm_eps,
                    final_dropout=cfg.final_dropout,
                    cross_attention_dim=block_cross_dim,
                )
            )
        self.transformer_blocks: nn.ModuleList = nn.ModuleList(blocks)

        # Output projection — adaLN shift/scale from timestep, then linear to output_dim
        self.norm_out: nn.LayerNorm = nn.LayerNorm(
            self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out_1: nn.Linear = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2: nn.Linear = nn.Linear(self.inner_dim, cfg.output_dim)

    def _run_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run all transformer blocks. Odd blocks are self-attn, even are cross-attn."""
        for idx, block in enumerate(self.transformer_blocks):
            use_self_attn = (idx % 2 == 1) and self.cfg.interleave_self_attention
            block_enc_states = None if use_self_attn else encoder_hidden_states
            block_enc_mask = None if use_self_attn else encoder_attention_mask
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=block_enc_states,
                encoder_attention_mask=block_enc_mask,
                temb=temb,
            )
        return hidden_states

    def _output_projection(
        self, hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        """Apply adaLN shift/scale from timestep then project to output_dim."""
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:          [B, T, inner_dim]  state+action embeddings
            encoder_hidden_states:  [B, S, cross_dim]  VL backbone embeddings
            timestep:               [B] int64 discretised timestep bucket
            encoder_attention_mask: [B, S] optional VL attention mask
        Returns:
            [B, T, output_dim] predicted velocity
        """
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = self._run_blocks(
            hidden_states, encoder_hidden_states, temb, encoder_attention_mask
        )
        return self._output_projection(hidden_states, temb)


# ---------------------------------------------------------------------------
# AlternateVLDiT
# ---------------------------------------------------------------------------


class AlternateVLDiT(VLDiT):
    """VLDiT variant that alternates between text and image cross-attention.

    GR00T N1.6 uses this. Each even block cross-attends to either the text
    tokens or image tokens of the VL backbone (alternating per attend_text_every_n_blocks).

    No extra weights vs VLDiT — same state dict structure, same CheckpointMapping.
    The attend_text_every_n_blocks parameter is a runtime property.
    """

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:           [B, T, inner_dim]
            encoder_hidden_states:   [B, S, cross_dim]  full VL sequence
            timestep:                [B] int64
            image_mask:              [B, S] bool — True for image tokens
            backbone_attention_mask: [B, S] bool — True for valid tokens
            encoder_attention_mask:  unused (kept for API compatibility)
        Returns:
            [B, T, output_dim]
        """
        assert self.cfg.interleave_self_attention, (
            "AlternateVLDiT requires interleave_self_attention=True"
        )

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        # Separate masks for image vs text tokens
        image_attn_mask = image_mask & backbone_attention_mask        # [B, S]
        non_image_attn_mask = (~image_mask) & backbone_attention_mask  # [B, S]

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Odd → pure self-attention
                hidden_states = block(hidden_states, temb=temb)
            else:
                # Even → cross-attention; alternate text vs image attention mask
                if idx % (2 * self.cfg.attend_text_every_n_blocks) == 0:
                    curr_mask = non_image_attn_mask   # text tokens
                else:
                    curr_mask = image_attn_mask        # image tokens
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=curr_mask,
                    temb=temb,
                )

        return self._output_projection(hidden_states, temb)
