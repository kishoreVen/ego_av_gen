"""Qwen2.5-VL local inference model.

Supports text and image/video inputs → text output.
Uses transformers + qwen_vl_utils for processing.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PIL import Image

from model_router.scaffold.model_base import LocalModel

logger = logging.getLogger(__name__)


class QwenVLModel(LocalModel):
    """Qwen2.5-VL vision-language model for local inference."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 2048,
    ) -> None:
        super().__init__(model_id, device)
        self.max_new_tokens = max_new_tokens
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(f"[qwen_vl] loading {self.model_id} → {self.device}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self._loaded = True
        logger.info(f"[qwen_vl] loaded")

    def generate(self, **kwargs: Any) -> dict[str, Any]:
        """Run Qwen2.5-VL inference.

        Kwargs:
            text: User query text.
            images: Optional list of PIL images.
            video_frames: Optional list of PIL images (sampled video frames).
            system_prompt: Optional system prompt.

        Returns:
            {"text": str, "usage": {"input_tokens": int, "output_tokens": int}}
        """
        text: str = kwargs["text"]
        images: list[Image.Image] | None = kwargs.get("images")
        video_frames: list[Image.Image] | None = kwargs.get("video_frames")
        system_prompt: str | None = kwargs.get("system_prompt")

        if not self._loaded:
            self.load()
        assert self.model is not None and self.processor is not None

        messages = self._build_messages(text, images, video_frames, system_prompt)

        from qwen_vl_utils import process_vision_info

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = output_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return {
            "text": output_text,
            "usage": {
                "input_tokens": input_len,
                "output_tokens": generated_ids.shape[1],
            },
        }

    def unload(self) -> None:
        self.model = None
        self.processor = None
        super().unload()

    @staticmethod
    def _build_messages(
        text: str,
        images: list[Image.Image] | None,
        video_frames: list[Image.Image] | None,
        system_prompt: str | None,
    ) -> list[dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content: list[dict] = []

        if images:
            for img in images:
                content.append({"type": "image", "image": img})

        if video_frames:
            content.append({"type": "video", "video": video_frames})

        content.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content})
        return messages
