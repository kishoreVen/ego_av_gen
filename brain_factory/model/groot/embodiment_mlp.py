"""Multi-embodiment MLP modules for GR00T.

Port of NVIDIA Isaac-GR00T gr00t/model/modules/embodiment_conditioned_mlp.py.
Attribute names are preserved exactly so weights load directly from the GR00T
checkpoint via CheckpointMapping with source_prefix="action_head.state_encoder.",
"action_head.action_encoder.", or "action_head.action_decoder.".

Weight key layout:

CategorySpecificLinear:
    .W    [num_embodiments, input_dim, hidden_dim]
    .b    [num_embodiments, hidden_dim]

CategorySpecificMLP:
    .layer1.W / .layer1.b
    .layer2.W / .layer2.b

MultiEmbodimentActionEncoder:
    .W1.W / .W1.b    (action_dim → hidden_size)
    .W2.W / .W2.b    (2*hidden_size → hidden_size)
    .W3.W / .W3.b    (hidden_size → hidden_size)
    # SinusoidalPositionalEncoding has no learnable params

Architecture notes:
    CategorySpecificLinear: maintains one weight matrix and bias per embodiment.
    Indexing with cat_ids performs a batched gather so each sample uses its own
    embodiment's weights — this is how GR00T handles heterogeneous robots.

    MultiEmbodimentActionEncoder fuses per-timestep action embeddings with
    sinusoidal timestep encodings before passing through two more linear layers.
    The sinusoidal encoding here is over the *diffusion timestep t*, not position.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of diffusion timesteps.

    No learnable parameters — produces [B, T, embedding_dim] from
    integer/float timestep indices of shape [B, T].
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim: int = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B, T] float or int timestep values.
        Returns:
            [B, T, embedding_dim] sinusoidal embedding.
        """
        timesteps = timesteps.float()
        B, T = timesteps.shape
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            math.log(10000.0) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # [B, T, half_dim]
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # [B, T, embedding_dim]


# ---------------------------------------------------------------------------
# Category-specific linear layers
# ---------------------------------------------------------------------------


class CategorySpecificLinear(nn.Module):
    """Linear layer with per-embodiment weight and bias tensors.

    Weight keys:
        .W    [num_embodiments, input_dim, hidden_dim]
        .b    [num_embodiments, hidden_dim]
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_categories: int = num_categories
        self.W: nn.Parameter = nn.Parameter(
            0.02 * torch.randn(num_categories, input_dim, hidden_dim)
        )
        self.b: nn.Parameter = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, T, input_dim]
            cat_ids: [B] int64 embodiment indices
        Returns:
            [B, T, hidden_dim]
        """
        selected_W = self.W[cat_ids]   # [B, input_dim, hidden_dim]
        selected_b = self.b[cat_ids]   # [B, hidden_dim]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with per-embodiment weights (ReLU activation between layers).

    Used for both the state encoder and action decoder in GR00T.

    Weight keys:
        .layer1.W / .layer1.b
        .layer2.W / .layer2.b
    """

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.num_categories: int = num_categories
        self.layer1: CategorySpecificLinear = CategorySpecificLinear(
            num_categories, input_dim, hidden_dim
        )
        self.layer2: CategorySpecificLinear = CategorySpecificLinear(
            num_categories, hidden_dim, output_dim
        )

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, T, input_dim]
            cat_ids: [B] int64 embodiment indices
        Returns:
            [B, T, output_dim]
        """
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


# ---------------------------------------------------------------------------
# Multi-embodiment action encoder
# ---------------------------------------------------------------------------


class MultiEmbodimentActionEncoder(nn.Module):
    """Encodes noisy action trajectories with diffusion timestep conditioning.

    Fuses the per-embodiment action embedding (W1) with a sinusoidal timestep
    encoding, then applies two more per-embodiment linear layers (W2, W3).

    Architecture:
        a_emb   = W1(actions)                     [B, T, hidden_size]
        tau_emb = SinusoidalPE(timestep)           [B, T, hidden_size]
        x       = swish(W2(cat([a_emb, tau_emb]))) [B, T, hidden_size]
        out     = W3(x)                            [B, T, hidden_size]

    Weight keys:
        .W1.W / .W1.b    [num_embodiments, action_dim,   hidden_size]
        .W2.W / .W2.b    [num_embodiments, 2*hidden_size, hidden_size]
        .W3.W / .W3.b    [num_embodiments, hidden_size,  hidden_size]
        # pos_encoding has no learnable params
    """

    def __init__(
        self,
        action_dim: int,
        hidden_size: int,
        num_embodiments: int,
    ) -> None:
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_embodiments: int = num_embodiments

        self.W1: CategorySpecificLinear = CategorySpecificLinear(
            num_embodiments, action_dim, hidden_size
        )
        self.W2: CategorySpecificLinear = CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size
        )
        self.W3: CategorySpecificLinear = CategorySpecificLinear(
            num_embodiments, hidden_size, hidden_size
        )
        self.pos_encoding: SinusoidalPositionalEncoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            actions:   [B, T, action_dim] noisy action trajectory
            timesteps: [B] int64 discretised diffusion timestep per sample
            cat_ids:   [B] int64 embodiment indices
        Returns:
            [B, T, hidden_size] encoded action features
        """
        B, T, _ = actions.shape

        # Broadcast scalar timestep across the action horizon.
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)  # [B, T]
        else:
            raise ValueError(f"Expected timesteps shape [B], got {timesteps.shape}")

        a_emb = self.W1(actions, cat_ids)                                         # [B, T, W]
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)             # [B, T, W]
        x = _swish(self.W2(torch.cat([a_emb, tau_emb], dim=-1), cat_ids))        # [B, T, W]
        return self.W3(x, cat_ids)                                                 # [B, T, W]
