"""Eagle VLM backbone wrapper for brain_factory.

Port of NVIDIA Isaac-GR00T gr00t/model/modules/eagle_backbone.py.
Eagle-Block2A-2B-v2 is NVIDIA's internal VLM (vision-language model) that
processes images and text instructions into a sequence of hidden states that
the GR00T action head cross-attends to.

The model is loaded from the GR00T checkpoint via HuggingFace AutoModel with
trust_remote_code=True because Eagle's architecture is bundled with the model
card, not published as a standard HF class.

State dict prefix in the GR00T checkpoint: "backbone.*"

Load pretrained weights with:
    CheckpointMapping(path=groot_ckpt, source_prefix="backbone.", target_prefix="backbone.")
    # (when this module is used as self.backbone inside Gr00tModel)

or for a standalone backbone:
    CheckpointMapping(path=groot_ckpt, source_prefix="backbone.", target_prefix="")

Weight key layout (inside .model):
    .model.language_model.model.embed_tokens.weight
    .model.language_model.model.layers.{i}.*        (pruned to select_layer)
    .model.vision_model.*
    .model.mlp1.*                                   (vision→language projection)

Architecture notes:
    Eagle takes (input_ids, attention_mask, pixel_values) and returns the last
    hidden state of the language model layers (up to select_layer). The output
    is a sequence [B, T, 2048] that contains fused vision and language tokens.

    image_mask identifies which token positions correspond to image patches;
    the action head uses this to separate text vs. image cross-attention.

Requirements:
    pip install transformers>=4.51.3
    The Eagle model card must be accessible (local copy in gr00t/model/modules/nvidia/
    or via HuggingFace Hub with trust_remote_code=True).
    Requires flash-attn and bfloat16 for the default nvidia/Eagle-Block2A-2B-v2 checkpoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class EagleBackboneConfig:
    """Configuration for the Eagle VLM backbone.

    Defaults match the GR00T N1.6 checkpoint (nvidia/Eagle-Block2A-2B-v2).
    """

    model_name: str = "nvidia/Eagle-Block2A-2B-v2"
    select_layer: int = 16           # Truncate LLM to this many layers (saves memory)
    tune_llm: bool = False           # Unfreeze language model weights
    tune_visual: bool = False        # Unfreeze vision encoder + mlp1 weights
    tune_top_llm_layers: int = 4     # Unfreeze this many top LLM layers even if tune_llm=False
    use_flash_attention: bool = True # Required for Eagle-Block2A-2B-v2
    load_bf16: bool = True           # Required for Eagle-Block2A-2B-v2
    trainable_params_fp32: bool = True  # Cast trainable params to fp32 after bf16 load
    eagle_model_dir: Optional[str] = None  # Path to local Eagle model directory


class EagleVLMBackbone(nn.Module):
    """Eagle vision-language backbone.

    Takes tokenised text + pixel values and returns fused VL hidden states
    that the GR00T action head cross-attends to.

    forward() output dict:
        backbone_features:      [B, T, 2048]  last hidden state
        backbone_attention_mask: [B, T] bool  valid token positions
        image_mask:             [B, T] bool   True at image patch positions

    Weight keys (state dict prefix is relative to this module):
        .model.language_model.*
        .model.vision_model.*
        .model.mlp1.*
    """

    def __init__(self, cfg: EagleBackboneConfig) -> None:
        super().__init__()
        self.cfg: EagleBackboneConfig = cfg

        extra_kwargs: Dict = {}
        if cfg.use_flash_attention:
            extra_kwargs["attn_implementation"] = "flash_attention_2"
        if cfg.load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        # Resolve the Eagle model directory
        if cfg.eagle_model_dir is not None:
            eagle_path = cfg.eagle_model_dir
        else:
            # Fall back to the bundled copy shipped with Isaac-GR00T
            _here = os.path.dirname(os.path.abspath(__file__))
            eagle_path = os.path.join(_here, "..", "groot", "nvidia", "Eagle-Block2A-2B-v2")
            eagle_path = os.path.normpath(eagle_path)

        config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)
        self.model: nn.Module = AutoModel.from_config(config, trust_remote_code=True, **extra_kwargs)

        # Truncate LLM to select_layer (reduces memory and matches GR00T checkpoint shape)
        while len(self.model.language_model.model.layers) > cfg.select_layer:
            self.model.language_model.model.layers.pop(-1)

        self._set_trainable_parameters()

        if cfg.load_bf16 and cfg.trainable_params_fp32:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)

    def _set_trainable_parameters(self) -> None:
        cfg = self.cfg
        for p in self.parameters():
            p.requires_grad = True

        if not cfg.tune_llm:
            self.model.language_model.requires_grad_(False)
        if not cfg.tune_visual:
            self.model.vision_model.requires_grad_(False)
            self.model.mlp1.requires_grad_(False)

        if cfg.tune_top_llm_layers > 0:
            for layer in self.model.language_model.model.layers[-cfg.tune_top_llm_layers :]:
                for param in layer.parameters():
                    param.requires_grad = True

    def set_frozen_modules_to_eval_mode(self) -> None:
        """Call at the start of each training forward to keep frozen sub-modules in eval.

        HuggingFace Trainer calls model.train() before each step, which would
        erroneously enable dropout in frozen layers. This corrects that.
        """
        if self.training:
            if not self.cfg.tune_llm:
                self.model.language_model.eval()
            if not self.cfg.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys
                input_ids:       [B, T] int64
                attention_mask:  [B, T] int64 (1=valid, 0=pad)
                pixel_values:    [B, C, H, W] or list of image tensors
        Returns:
            dict with keys
                backbone_features:       [B, T, 2048]
                backbone_attention_mask: [B, T] bool
                image_mask:              [B, T] bool
        """
        self.set_frozen_modules_to_eval_mode()
        vl_input = {k: batch[k] for k in ("input_ids", "attention_mask", "pixel_values")}
        outputs = self.model(**vl_input, output_hidden_states=True)
        hidden = outputs["hidden_states"][-1]  # [B, T, 2048]
        image_mask = batch["input_ids"] == self.model.config.image_token_index
        attention_mask = batch["attention_mask"] == 1
        return {
            "backbone_features": hidden,
            "backbone_attention_mask": attention_mask,
            "image_mask": image_mask,
        }
