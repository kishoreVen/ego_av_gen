"""Base class for image generation model interfaces that support optional prompt compaction.

Some models (like diffusion models) benefit from having prompts compacted before generation,
while others (like OpenAI and Gemini) handle long prompts natively and don't need compaction.
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import replace
from typing import Any, Dict, List

from model_router.model_interface import (
    ModelInterface,
    Query,
    ImageGenQuery,
    Capability,
)

logger = logging.getLogger(__name__)


class ImageGenerationModelInterface(ModelInterface):
    """Base class for image generation models that support optional prompt compaction.

    Models that need prompt compaction should extend this class and implement
    ``_fetch_image_response``. The compaction logic is handled automatically in
    ``fetch_response`` if a compaction_prompt is provided in the query.

    Models that don't need compaction (like OpenAI and Gemini native interfaces)
    should extend ModelInterface directly and implement fetch_response.
    """

    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)
        self._router = None

    def _get_router(self):
        """Lazily initialize the ModelRouter for compaction queries."""
        if self._router is None:
            from model_router.router import ModelRouter

            self._router = ModelRouter()
        return self._router

    def _compact_prompt(self, query: ImageGenQuery) -> str:
        compaction_query = Query(
            system_prompt=query.compaction_prompt,
            query_text=query.make_query(),
        )

        router = self._get_router()
        response = router.get_response(
            compaction_query,
            Capability.TEXT,
            query.compaction_model,
        )

        compacted_text = response["text"]
        logger.debug(
            f"Compacted prompt to {len(compacted_text)} chars"
        )
        return compacted_text

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        if not isinstance(query, ImageGenQuery):
            raise ValueError("Image generation requires ImageGenQuery")

        if query.compaction_prompt:
            compacted_text = self._compact_prompt(query)
            compacted_query = replace(
                query,
                query_text=compacted_text,
                system_prompt=None,
                compaction_prompt=None,
            )
            return self._fetch_image_response(compacted_query, capability)
        else:
            return self._fetch_image_response(query, capability)

    @abstractmethod
    def _fetch_image_response(
        self, query: ImageGenQuery, capability: Capability | None = None
    ) -> Dict[str, Any]: ...

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.IMAGE_GEN]
