"""GR00T action head — flow-matching diffusion policy.

Port of NVIDIA Isaac-GR00T gr00t/model/gr00t_n1d6/gr00t_n1d6.py (Gr00tN1d6ActionHead).
HuggingFace BatchFeature replaced with plain dicts. Attribute names match the
original for state dict compatibility.

Load pretrained weights with:
    CheckpointMapping(path=groot_ckpt, source_prefix="action_head.", target_prefix="")
    # (when used standalone)

or as part of Gr00tModel:
    CheckpointConfig(path=groot_ckpt)  # full model — keys already match

Weight key layout:
    .model.*                          VLDiT / AlternateVLDiT (see vl_dit.py)
    .state_encoder.layer1.W / .b
    .state_encoder.layer2.W / .b
    .action_encoder.W1.W / .b
    .action_encoder.W2.W / .b
    .action_encoder.W3.W / .b
    .action_decoder.layer1.W / .b
    .action_decoder.layer2.W / .b
    .vlln.weight / .vlln.bias         (only when use_vlln=True)
    .position_embedding.weight        (only when add_pos_embed=True)
    .mask_token                       (only when state_dropout_prob>0)

Training — flow matching:
    1. Sample continuous t from Beta(alpha, beta) scaled to [0, noise_s].
    2. Interpolate: x_t = (1-t)*noise + t*actions  (noise→clean as t: 0→1)
    3. Target velocity: v* = actions - noise
    4. Discretise t → bucket index for the action encoder and DiT timestep.
    5. Encode (state, noisy_actions, VL_embeddings) → DiT → predict velocity.
    6. Loss: MSE(pred_velocity, v*) masked by action_mask.

Inference — Euler integration (num_inference_timesteps steps):
    Start from x_0 ~ N(0,I), integrate dx/dt = v(x,t) from t=0 to t=1.

Architecture notes:
    state_encoder:  [B, 1, max_state_dim]   → [B, 1, input_embedding_dim]
    action_encoder: [B, H, max_action_dim]  → [B, H, input_embedding_dim]
    cat along T dim → [B, 1+H, input_embedding_dim]  (= inner_dim of DiT)
    DiT output:     [B, 1+H, hidden_size]
    action_decoder: [B, 1+H, hidden_size]   → [B, 1+H, max_action_dim]
    take last H positions → [B, H, max_action_dim]  (action prediction)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_factory.noise_schedule.rectified_flow_schedule import RectifiedFlowSchedule
from brain_factory.model.groot.embodiment_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from brain_factory.model.groot.vl_dit import AlternateVLDiT, VLDiT, VLDiTConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Gr00tActionHeadConfig:
    """Construction parameters for Gr00tActionHead.

    Defaults match GR00T N1.6.
    """

    # Dimensions
    backbone_embedding_dim: int = 2048   # Eagle output dim (cross-attn dim for DiT)
    hidden_size: int = 1024              # DiT output dim / action decoder input dim
    input_embedding_dim: int = 1536      # = num_attention_heads * attention_head_dim
    max_state_dim: int = 29
    max_action_dim: int = 29
    action_horizon: int = 16
    max_num_embodiments: int = 32

    # VLDiT
    dit: VLDiTConfig = field(default_factory=lambda: VLDiTConfig(
        num_attention_heads=32,
        attention_head_dim=48,      # inner_dim = 32*48 = 1536 = input_embedding_dim ✓
        output_dim=1024,            # = hidden_size ✓
        num_layers=32,
        dropout=0.2,
        attention_bias=True,
        activation_fn="geglu",
        norm_elementwise_affine=False,
        norm_eps=1e-5,
        final_dropout=True,
        positional_embeddings=None,
        interleave_self_attention=True,
        cross_attention_dim=2048,   # = backbone_embedding_dim ✓
        attend_text_every_n_blocks=2,
    ))
    use_alternate_vl_dit: bool = True

    # Position embedding inside the action head sequence
    add_pos_embed: bool = True
    max_seq_len: int = 1024

    # VL layer norm (normalises backbone features before cross-attention)
    use_vlln: bool = True

    # State augmentation
    state_dropout_prob: float = 0.0
    state_additive_noise_scale: float = 0.0

    # Flow matching noise schedule
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Fine-tuning control
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True


# ---------------------------------------------------------------------------
# Action head
# ---------------------------------------------------------------------------


class Gr00tActionHead(nn.Module):
    """Flow-matching diffusion action head for GR00T.

    Attribute names match GR00T's Gr00tN1d6ActionHead exactly so that
    weights load from the pretrained checkpoint via CheckpointMapping
    with source_prefix="action_head.".
    """

    def __init__(self, cfg: Gr00tActionHeadConfig) -> None:
        super().__init__()
        self.cfg: Gr00tActionHeadConfig = cfg

        # --- VLDiT (velocity prediction network) ---
        if cfg.use_alternate_vl_dit:
            self.model: nn.Module = AlternateVLDiT(cfg.dit)
        else:
            self.model = VLDiT(cfg.dit)

        # --- Embodiment-specific encoders / decoders ---
        # Note: attribute names .state_encoder, .action_encoder, .action_decoder
        # must match GR00T exactly for state dict compatibility.
        self.state_encoder: CategorySpecificMLP = CategorySpecificMLP(
            num_categories=cfg.max_num_embodiments,
            input_dim=cfg.max_state_dim,
            hidden_dim=cfg.hidden_size,
            output_dim=cfg.input_embedding_dim,
        )
        self.action_encoder: MultiEmbodimentActionEncoder = MultiEmbodimentActionEncoder(
            action_dim=cfg.max_action_dim,
            hidden_size=cfg.input_embedding_dim,
            num_embodiments=cfg.max_num_embodiments,
        )
        self.action_decoder: CategorySpecificMLP = CategorySpecificMLP(
            num_categories=cfg.max_num_embodiments,
            input_dim=cfg.hidden_size,
            hidden_dim=cfg.hidden_size,
            output_dim=cfg.max_action_dim,
        )

        # --- VL layer norm: normalises backbone features ---
        self.vlln: nn.Module = (
            nn.LayerNorm(cfg.backbone_embedding_dim) if cfg.use_vlln else nn.Identity()
        )

        # --- Optional position embedding over the (state+action) sequence ---
        if cfg.add_pos_embed:
            self.position_embedding: nn.Embedding = nn.Embedding(
                cfg.max_seq_len, cfg.input_embedding_dim
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # --- State dropout mask token ---
        self.mask_token: Optional[nn.Parameter] = (
            nn.Parameter(0.02 * torch.randn(1, 1, cfg.input_embedding_dim))
            if cfg.state_dropout_prob > 0
            else None
        )

        # --- Rectified flow schedule (noise sampling + interpolation) ---
        self._schedule: RectifiedFlowSchedule = RectifiedFlowSchedule(
            num_inference_steps=cfg.num_inference_timesteps,
            num_timestep_buckets=cfg.num_timestep_buckets,
            noise_beta_alpha=cfg.noise_beta_alpha,
            noise_beta_beta=cfg.noise_beta_beta,
            noise_s=cfg.noise_s,
        )

        self._apply_grad_settings()

    # ------------------------------------------------------------------
    # Gradient / trainability
    # ------------------------------------------------------------------

    def _apply_grad_settings(self) -> None:
        cfg = self.cfg
        for p in self.parameters():
            p.requires_grad = True
        if not cfg.tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if cfg.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.mask_token is not None:
                self.mask_token.requires_grad_(False)
        if not cfg.tune_diffusion_model:
            self.model.requires_grad_(False)
        if not cfg.tune_vlln:
            self.vlln.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self) -> None:
        """Keep frozen sub-modules in eval mode during training."""
        if self.training:
            if not self.cfg.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.cfg.add_pos_embed:
                    self.position_embedding.eval()
            if not self.cfg.tune_diffusion_model:
                self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_time(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return self._schedule.sample_t(batch_size, device, dtype)

    def _encode_vl(self, backbone_output: Dict) -> Dict:
        """Apply VL layer norm to backbone features (in-place copy)."""
        features = self.vlln(backbone_output["backbone_features"])
        return {**backbone_output, "backbone_features": features}

    def _encode_state(
        self,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        """Encode per-embodiment state and optionally apply dropout / noise."""
        state_features = self.state_encoder(state, embodiment_id)  # [B, S, input_embedding_dim]

        # State dropout: replace with learned mask token
        if self.cfg.state_dropout_prob > 0 and self.mask_token is not None:
            drop = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.cfg.state_dropout_prob
            ).to(dtype=state_features.dtype)[:, None, None]
            state_features = state_features * (1 - drop) + self.mask_token * drop

        # State additive noise
        if self.training and self.cfg.state_additive_noise_scale > 0:
            state_features = state_features + (
                torch.randn_like(state_features) * self.cfg.state_additive_noise_scale
            )

        return state_features

    def _add_pos_embed_to_actions(self, action_features: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to action features only (not state features).

        Matches GR00T exactly: pos embed is applied to action_features before
        concatenation with state_features, so position 0..H-1 are the action positions.
        State features receive no positional embedding.
        """
        if not self.cfg.add_pos_embed:
            return action_features
        T = action_features.shape[1]
        pos_ids = torch.arange(T, dtype=torch.long, device=action_features.device)
        return action_features + self.position_embedding(pos_ids).unsqueeze(0)

    def _run_dit(
        self,
        sa_embs: torch.Tensor,
        backbone_output: Dict,
        t_discretized: torch.Tensor,
    ) -> torch.Tensor:
        vl_embeds = backbone_output["backbone_features"]
        vl_attn_mask = backbone_output["backbone_attention_mask"]
        if self.cfg.use_alternate_vl_dit:
            return self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_discretized,
                image_mask=backbone_output["image_mask"],
                backbone_attention_mask=backbone_output["backbone_attention_mask"],
            )
        else:
            return self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_discretized,
                encoder_attention_mask=vl_attn_mask,
            )

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        backbone_output: Dict,
        action_input: Dict,
    ) -> Dict:
        """Compute flow-matching training loss.

        Args:
            backbone_output: from EagleVLMBackbone.forward()
                backbone_features:       [B, S, backbone_embedding_dim]
                backbone_attention_mask: [B, S] bool
                image_mask:              [B, S] bool
            action_input: dict with keys
                state:         [B, state_horizon, max_state_dim] float
                action:        [B, action_horizon, max_action_dim] float
                embodiment_id: [B] int64
                action_mask:   [B, action_horizon, max_action_dim] float (1=active, 0=pad)
        Returns:
            dict with keys
                loss:            scalar training loss
                action_loss:     [B, H, max_action_dim] per-element MSE before masking
                backbone_features: [B, S, backbone_embedding_dim]
                state_features:  [B, state_horizon, input_embedding_dim]
        """
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self._encode_vl(backbone_output)

        embodiment_id: torch.Tensor = action_input["embodiment_id"]
        actions: torch.Tensor = action_input["action"]         # [B, H, action_dim]

        state_features = self._encode_state(action_input["state"], embodiment_id)

        # --- Flow matching: sample t, interpolate, compute velocity target ---
        noise = torch.randn_like(actions)
        t_cont = self._sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy_trajectory, velocity, t_discretized = self._schedule.training_step(
            actions, noise, t_cont
        )

        # --- Encode and run DiT ---
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        action_features = self._add_pos_embed_to_actions(action_features)
        sa_embs = torch.cat([state_features, action_features], dim=1)

        dit_output = self._run_dit(sa_embs, backbone_output, t_discretized)  # [B, S+H, hidden_size]

        pred = self.action_decoder(dit_output, embodiment_id)       # [B, S+H, action_dim]
        pred_actions = pred[:, -actions.shape[1]:]                  # [B, H, action_dim]

        action_mask: torch.Tensor = action_input["action_mask"]
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": backbone_output["backbone_features"],
            "state_features": state_features,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: Dict,
        action_input: Dict,
    ) -> Dict:
        """Generate actions via Euler integration of the learned velocity field.

        Args:
            backbone_output: same structure as training forward
            action_input: dict with keys
                state:         [B, state_horizon, max_state_dim]
                embodiment_id: [B] int64
        Returns:
            dict with keys
                action_pred:       [B, action_horizon, max_action_dim]
                backbone_features: [B, S, backbone_embedding_dim]
                state_features:    [B, state_horizon, input_embedding_dim]
        """
        backbone_output = self._encode_vl(backbone_output)
        embodiment_id: torch.Tensor = action_input["embodiment_id"]
        B = embodiment_id.shape[0]
        device = embodiment_id.device
        dtype = backbone_output["backbone_features"].dtype

        state_features = self._encode_state(action_input["state"], embodiment_id)

        # Start from pure noise; Euler-integrate from t=0 to t=1
        actions = torch.randn(
            B, self.cfg.action_horizon, self.cfg.max_action_dim,
            dtype=dtype, device=device,
        )

        dt = 1.0 / self.cfg.num_inference_timesteps

        for t_cont, t_disc_int in self._schedule.inference_schedule():
            t_tensor = torch.full((B,), t_disc_int, dtype=torch.long, device=device)

            action_features = self.action_encoder(actions, t_tensor, embodiment_id)
            action_features = self._add_pos_embed_to_actions(action_features)
            sa_embs = torch.cat([state_features, action_features], dim=1)

            dit_output = self._run_dit(sa_embs, backbone_output, t_tensor)
            pred = self.action_decoder(dit_output, embodiment_id)
            pred_velocity = pred[:, -self.cfg.action_horizon:]

            actions = self._schedule.euler_step(actions, pred_velocity, dt)

        return {
            "action_pred": actions,
            "backbone_features": backbone_output["backbone_features"],
            "state_features": state_features,
        }
