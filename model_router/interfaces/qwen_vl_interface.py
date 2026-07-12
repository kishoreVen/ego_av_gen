"""Qwen2.5-VL local inference interface."""
from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from model_router.model_interface import Capability, Query
from model_router.scaffold.local_interface import (
    LocalModelInterface,
    b64_to_image,
    extract_images,
)


class QwenVLInterface(LocalModelInterface):
    """Qwen2.5-VL via local inference — supports TEXT + IMAGE_ENC + VIDEO_ENC."""

    def __init__(self, seed: int | None = None) -> None:
        from model_router.models.qwen_vl import QwenVLModel

        super().__init__(model=QwenVLModel(), seed=seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.IMAGE_ENC, Capability.VIDEO_ENC]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        text = query.make_query()
        system_prompt = query.get_system_prompt()
        images = extract_images(query)

        video_frames: list[Image.Image] | None = None
        if query.video:
            video_frames = [
                f if isinstance(f, Image.Image) else b64_to_image(f)
                for f in query.video
            ]

        return self.model.generate(
            text=text,
            images=images or None,
            video_frames=video_frames,
            system_prompt=system_prompt,
        )
