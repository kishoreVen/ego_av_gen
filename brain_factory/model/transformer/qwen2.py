"""Pure PyTorch Qwen2 transformer implementation.

No HuggingFace dependencies. Weight key names match the Qwen2 checkpoint
format so that pretrained weights load via safetensors prefix remapping.

Supports KV cache for autoregressive inference (past_key_values).

Architecture reference: Qwen2.5-0.5B
  - Hidden size: 896
  - Attention heads: 14 (GQA with 2 KV heads)
  - Head dim: 64
  - Intermediate size: 4864 (SwiGLU FFN)
  - RoPE theta: 1,000,000
  - Vocab size: 151,936
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# KV cache type: list of (key, value) tuples, one per layer.
# Each key/value shape: [B, num_kv_heads, seq_len, head_dim]
KVCache = List[Tuple[torch.Tensor, torch.Tensor]]


class Qwen2RMSNorm(nn.Module):
    """Root mean square layer normalization (weight-only, no bias).

    Weight key: .weight [hidden_size]
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))
        self.eps: float = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class Qwen2RotaryEmbedding(nn.Module):
    """Rotary positional embedding with configurable theta.

    No learned parameters — precomputes cos/sin tables.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 1_000_000.0,
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.max_position_embeddings: int = max_position_embeddings
        self.base: float = base

        inv_freq: torch.Tensor = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for RoPE.

        Args:
            x: Input tensor (used only for dtype/device).
            position_ids: [B, T] position indices.

        Returns:
            (cos, sin) each of shape [B, T, dim].
        """
        # inv_freq: [dim/2]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )  # [B, dim/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [B, 1, T]

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(
            1, 2
        )  # [B, T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, T, dim]

        cos = emb.cos()
        sin = emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors.

    Args:
        q: [B, num_heads, T, head_dim]
        k: [B, num_kv_heads, T, head_dim]
        cos: [B, T, head_dim] (will be unsqueezed for heads dim)
        sin: [B, T, head_dim]
    """
    cos = cos.unsqueeze(1)  # [B, 1, T, head_dim]
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (GQA).

    Args:
        hidden_states: [B, num_kv_heads, T, head_dim]
        n_rep: number of repetitions (num_heads // num_kv_heads)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class Qwen2Attention(nn.Module):
    """Grouped Query Attention with rotary positional embeddings.

    Supports KV cache for autoregressive generation.

    Weight keys:
        .q_proj.weight [num_heads * head_dim, hidden_size]
        .q_proj.bias   [num_heads * head_dim]
        .k_proj.weight [num_kv_heads * head_dim, hidden_size]
        .k_proj.bias   [num_kv_heads * head_dim]
        .v_proj.weight [num_kv_heads * head_dim, hidden_size]
        .v_proj.bias   [num_kv_heads * head_dim]
        .o_proj.weight [hidden_size, num_heads * head_dim]
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_attention_heads
        self.num_kv_heads: int = num_key_value_heads
        self.head_dim: int = head_dim
        self.num_kv_groups: int = num_attention_heads // num_key_value_heads
        self.attention_dropout: float = attention_dropout

        self.q_proj: nn.Linear = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=True
        )
        self.k_proj: nn.Linear = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=True
        )
        self.v_proj: nn.Linear = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=True
        )
        self.o_proj: nn.Linear = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, hidden_size]
            attention_mask: [B, 1, T, T_total] additive mask (0=attend, -inf=mask)
            position_ids: [B, T] (unused when cos/sin provided directly)
            cos: [B, T, head_dim] precomputed RoPE cos
            sin: [B, T, head_dim] precomputed RoPE sin
            past_key_value: optional (past_k, past_v) each [B, kv_heads, S_cache, head_dim]

        Returns:
            (output, (new_k, new_v)) where output is [B, T, hidden_size]
        """
        bsz, seq_len, _ = hidden_states.shape

        q: torch.Tensor = self.q_proj(hidden_states)
        k: torch.Tensor = self.k_proj(hidden_states)
        v: torch.Tensor = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Concatenate with cached KV (before GQA repeat)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k.to(k.dtype), k], dim=2)
            v = torch.cat([past_v.to(v.dtype), v], dim=2)

        # Store updated KV cache (at KV-head granularity, before repeat)
        new_past_key_value = (k, v)

        # GQA: repeat KV heads
        k = _repeat_kv(k, self.num_kv_groups)
        v = _repeat_kv(v, self.num_kv_groups)

        # Scaled dot-product attention
        attn_output: torch.Tensor = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output), new_past_key_value


class Qwen2MLP(nn.Module):
    """SwiGLU feed-forward network.

    Weight keys:
        .gate_proj.weight [intermediate_size, hidden_size]
        .up_proj.weight   [intermediate_size, hidden_size]
        .down_proj.weight [hidden_size, intermediate_size]
    """

    def __init__(
        self,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
    ) -> None:
        super().__init__()
        self.gate_proj: nn.Linear = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.up_proj: nn.Linear = nn.Linear(
            hidden_size, intermediate_size, bias=False
        )
        self.down_proj: nn.Linear = nn.Linear(
            intermediate_size, hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    """Single Qwen2 transformer decoder layer.

    Weight keys:
        .input_layernorm.weight
        .self_attn.{q,k,v,o}_proj.*
        .post_attention_layernorm.weight
        .mlp.{gate,up,down}_proj.weight
    """

    def __init__(
        self,
        hidden_size: int = 896,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_layernorm: Qwen2RMSNorm = Qwen2RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn: Qwen2Attention = Qwen2Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
        )
        self.post_attention_layernorm: Qwen2RMSNorm = Qwen2RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.mlp: Qwen2MLP = Qwen2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            past_key_value=past_key_value,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv


class Qwen2Model(nn.Module):
    """Qwen2 backbone: embedding + N transformer layers + optional final norm.

    Supports KV cache for autoregressive inference via past_key_values.

    Weight keys:
        .embed_tokens.weight [vocab_size, hidden_size]
        .layers.{i}.*  (Qwen2DecoderLayer keys)
        .norm.weight   (only if include_final_norm=True)

    Args:
        include_final_norm: If True, adds a final RMSNorm (.norm). The VibeVoice
            text backbone (4 layers) does NOT have this; the TTS backbone
            (20 layers) DOES.
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 896,
        num_layers: int = 4,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 8192,
        attention_dropout: float = 0.0,
        include_final_norm: bool = False,
    ) -> None:
        super().__init__()
        self.embed_tokens: nn.Embedding = nn.Embedding(vocab_size, hidden_size)

        self.layers: nn.ModuleList = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm: Optional[Qwen2RMSNorm] = (
            Qwen2RMSNorm(hidden_size, eps=rms_norm_eps)
            if include_final_norm
            else None
        )

        self.rotary_emb: Qwen2RotaryEmbedding = Qwen2RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[KVCache] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, KVCache]]:
        """Run the Qwen2 backbone.

        Provide either input_ids OR inputs_embeds (not both).

        Args:
            input_ids: [B, T] token ids (used with embed_tokens).
            inputs_embeds: [B, T, hidden_size] pre-computed embeddings.
            attention_mask: [B, T_total] binary mask (1=attend, 0=mask).
                T_total = cache_len + T for cached generation.
            position_ids: [B, T] position indices. Auto-generated if None.
            past_key_values: optional KV cache from previous forward pass.

        Returns:
            When past_key_values is None: hidden_states [B, T, hidden_size]
            When past_key_values is provided: (hidden_states, new_past_key_values)
        """
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        use_cache = past_key_values is not None

        # Determine cache length for position offset
        cache_len = 0
        if use_cache and len(past_key_values) > 0:
            cache_len = past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(
                cache_len, cache_len + seq_len, device=device
            ).unsqueeze(0).expand(bsz, -1)

        # Compute RoPE cos/sin once for all layers
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Build causal attention mask
        causal_mask: Optional[torch.Tensor] = None
        if attention_mask is not None or seq_len > 1 or cache_len > 0:
            causal_mask = self._make_causal_mask(
                bsz, seq_len, device, hidden_states.dtype,
                attention_mask, cache_len,
            )

        new_past_key_values: KVCache = []

        for i, layer in enumerate(self.layers):
            layer_past_kv = past_key_values[i] if use_cache else None
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                past_key_value=layer_past_kv,
            )
            if use_cache:
                new_past_key_values.append(new_kv)

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        if use_cache:
            return hidden_states, new_past_key_values
        return hidden_states

    @staticmethod
    def _make_causal_mask(
        bsz: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        attention_mask: Optional[torch.Tensor] = None,
        cache_len: int = 0,
    ) -> torch.Tensor:
        """Create a combined causal + padding attention mask.

        Args:
            bsz: Batch size.
            seq_len: Number of new tokens.
            device: Target device.
            dtype: Target dtype.
            attention_mask: [B, T_total] binary mask (1=attend, 0=mask).
                T_total = cache_len + seq_len.
            cache_len: Number of cached positions to attend to.

        Returns:
            [B, 1, seq_len, cache_len + seq_len] additive mask.
        """
        total_len = cache_len + seq_len

        # Build boolean mask: True = attend, False = masked
        attend = torch.zeros(
            seq_len, total_len, dtype=torch.bool, device=device
        )
        for i in range(seq_len):
            attend[i, : cache_len + i + 1] = True

        # Combine with padding mask
        if attention_mask is not None:
            # attention_mask: [B, T_total] with 1=attend, 0=mask
            padding = attention_mask[:, None, :].bool()  # [B, 1, T_total]
            attend = attend.unsqueeze(0) & padding.unsqueeze(1)  # [B, 1, S, T_total]
        else:
            attend = attend.unsqueeze(0).unsqueeze(0)  # [1, 1, S, T_total]

        # Convert to additive mask: True → 0.0, False → -inf
        causal = torch.where(
            attend,
            torch.zeros(1, device=device, dtype=dtype),
            torch.full((1,), float("-inf"), device=device, dtype=dtype),
        )

        return causal
