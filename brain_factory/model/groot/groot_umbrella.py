"""GR00T UmbrellaModel — composes Eagle backbone + flow-matching action head.

Port of NVIDIA Isaac-GR00T Gr00tN1d6 into brain_factory's UmbrellaModelBase.
HuggingFace PreTrainedModel replaced with UmbrellaModelBase.
forward() accepts and returns plain dicts instead of BatchFeature.

Attribute names .backbone and .action_head match GR00T exactly, so the full
pretrained checkpoint loads without any key remapping:
    CheckpointConfig(path="/path/to/nvidia/GR00T-N1.6-3B")

To load only the action head (e.g. to swap the backbone):
    CheckpointMapping(path=ckpt, source_prefix="action_head.", target_prefix="action_head.")

State dict layout:
    backbone.model.language_model.*
    backbone.model.vision_model.*
    backbone.model.mlp1.*
    action_head.model.*                 VLDiT
    action_head.state_encoder.*
    action_head.action_encoder.*
    action_head.action_decoder.*
    action_head.vlln.*
    action_head.position_embedding.*
    action_head.mask_token              (if state_dropout_prob > 0)

Training mode  — call forward(batch):
    batch keys expected:
        input_ids, attention_mask, pixel_values    (Eagle inputs)
        state, action, embodiment_id, action_mask  (action head inputs)
    returns: {"loss": scalar, "action_loss": tensor, ...}

Inference mode — call get_action(batch):
    batch keys expected:
        input_ids, attention_mask, pixel_values    (Eagle inputs)
        state, embodiment_id                       (action head inputs)
    returns: {"action_pred": [B, H, action_dim], ...}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from brain_factory.model.backbone.eagle import EagleBackboneConfig, EagleVLMBackbone
from brain_factory.model.groot.action_head import Gr00tActionHead, Gr00tActionHeadConfig
from brain_factory.scaffold.umbrella_model_base import UmbrellaModelBase


@dataclass
class Gr00tModelConfig:
    """Top-level config for the full GR00T model.

    Composes backbone and action head configs. All defaults match GR00T N1.6.
    """

    backbone: EagleBackboneConfig = field(default_factory=EagleBackboneConfig)
    action_head: Gr00tActionHeadConfig = field(default_factory=Gr00tActionHeadConfig)


class Gr00tModel(UmbrellaModelBase):
    """Full GR00T model: Eagle VLM backbone + flow-matching diffusion action head.

    Instantiate for training from a pretrained checkpoint:
        cfg = Gr00tModelConfig()
        model = Gr00tModel(cfg)
        load_checkpoint(CheckpointConfig(path="nvidia/GR00T-N1.6-3B"), model)

    Or construct from scratch (random weights) by just calling Gr00tModel(cfg).
    """

    def __init__(self, cfg: Gr00tModelConfig) -> None:
        super().__init__()
        self.backbone: EagleVLMBackbone = EagleVLMBackbone(cfg.backbone)
        self.action_head: Gr00tActionHead = Gr00tActionHead(cfg.action_head)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Training forward — returns loss dict.

        Args:
            batch: flat dict containing both Eagle inputs and action head inputs.
                Eagle keys:       input_ids, attention_mask, pixel_values
                Action head keys: state, action, embodiment_id, action_mask
        Returns:
            dict with keys: loss, action_loss, action_mask, backbone_features, state_features
        """
        backbone_output = self.backbone(batch)
        return self.action_head(backbone_output, batch)

    @torch.no_grad()
    def get_action(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Inference forward — returns predicted actions.

        Args:
            batch: flat dict with Eagle inputs + state + embodiment_id
        Returns:
            dict with keys: action_pred, backbone_features, state_features
        """
        backbone_output = self.backbone(batch)
        return self.action_head.get_action(backbone_output, batch)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
