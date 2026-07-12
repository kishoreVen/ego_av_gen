"""SAM 3.1 local inference model.

Takes an RGB image and a text prompt, returns predicted object masks with
confidence scores using SAM 3.1's text-prompted segmentation.

Requires the sam3 package installed from:
    https://github.com/facebookresearch/sam3  (see setup.sh)

Checkpoint is downloaded from HuggingFace on first load:
    facebook/sam3.1  (gated — run `hf auth login` first)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from model_router.scaffold.model_base import LocalModel

logger = logging.getLogger(__name__)

_HF_REPO = "facebook/sam3"
_CHECKPOINT_FILENAME = "sam3.pt"

_PRETRAINED_DIR = Path(__file__).parents[2] / "pretrained_models" / "sam3"


class SAMModel(LocalModel):
    """SAM 3.1: image + text prompt → object masks + scores."""

    def __init__(
        self,
        model_id: str = _HF_REPO,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__(model_id, device)
        self.confidence_threshold = confidence_threshold
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # SAM3 uses bfloat16 — enable autocast to avoid dtype mismatches
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast(self.device, dtype=torch.bfloat16).__enter__()

        ckpt_path = _PRETRAINED_DIR / _CHECKPOINT_FILENAME
        load_from_hf = not ckpt_path.exists()
        if load_from_hf:
            logger.info("[sam] checkpoint not found locally, will download from HF...")
        else:
            logger.info(f"[sam] loading from {ckpt_path}")

        logger.info("[sam] building SAM 3.1 image model...")
        self.model = build_sam3_image_model(
            checkpoint_path=str(ckpt_path) if not load_from_hf else None,
            load_from_HF=load_from_hf,
            device=self.device,
        )

        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=self.confidence_threshold,
        )

        self._loaded = True
        logger.info("[sam] loaded")

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        """Run SAM 3.1 text-prompted segmentation on an image.

        Kwargs:
            image: PIL.Image — RGB input image.
            prompt: str — text description of objects to segment (default "object").
            confidence_threshold: float — override threshold for this call (optional).

        Returns:
            {
                "masks":  List[np.ndarray],  # H×W boolean arrays, one per object
                "labels": List[str],          # "object_0", "object_1", ...
                "scores": List[float],        # confidence score per mask
                "usage":  {"input_pixels": int, "output_masks": int},
            }
        """
        image: Image.Image = kwargs["image"]
        prompt: str = kwargs.get("prompt", "object")

        if not self._loaded:
            self.load()
        assert self.processor is not None

        import torch

        inference_state = self.processor.set_image(image)
        self.processor.reset_all_prompts(inference_state)
        inference_state = self.processor.set_text_prompt(prompt, inference_state)

        # Extract masks and scores from inference state
        # Sam3Processor stores: masks (bool tensor), scores (prob tensor), boxes
        masks: list[np.ndarray] = []
        scores: list[float] = []
        labels: list[str] = []

        raw_masks = inference_state.get("masks", torch.empty(0))
        raw_scores = inference_state.get("scores", torch.empty(0))

        if raw_masks.numel() > 0:
            # masks shape: [N, 1, H, W], scores shape: [N]
            for idx in range(raw_masks.shape[0]):
                mask_np = raw_masks[idx, 0].cpu().numpy().astype(bool)
                masks.append(mask_np)
                score = float(raw_scores[idx].cpu()) if idx < raw_scores.shape[0] else 0.0
                scores.append(score)
                labels.append(f"object_{idx}")

        # Build text summary for compatibility with demo/router
        summary_parts = [f"Found {len(masks)} object(s) for prompt '{prompt}'."]
        for i, (label, score) in enumerate(zip(labels, scores)):
            summary_parts.append(f"  {label}: score={score:.3f}")

        return {
            "text": "\n".join(summary_parts),
            "masks": masks,
            "labels": labels,
            "scores": scores,
            "usage": {
                "input_pixels": image.width * image.height,
                "output_masks": len(masks),
            },
        }

    def unload(self) -> None:
        self.model = None
        self.processor = None
        super().unload()
