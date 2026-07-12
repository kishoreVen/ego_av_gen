"""Base interface for locally-hosted models.

Layer 2 of the local inference stack:
  models/       — load weights, run forward / generate  (Layer 1)
  interfaces/   — bridges model_router protocol <> local models  (Layer 2)
"""
from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

from PIL import Image

from model_router.model_interface import Capability, ModelInterface, Query
from model_router.scaffold.model_base import LocalModel


class LocalModelInterface(ModelInterface):
    """Generic bridge from model_router protocol to a LocalModel.

    Subclasses set ``self.model`` to a ``LocalModel`` instance and implement
    ``supported_capabilities()`` and ``fetch_response()``.
    """

    def __init__(self, model: LocalModel, seed: int | None = None) -> None:
        super().__init__(seed)
        self.model = model

    def initialize_client(self) -> None:
        if not self.model.loaded:
            self.model.load()

    def requires_initialization(self) -> bool:
        return not self.model.loaded

    def supported_capabilities(self) -> List[Capability]:
        raise NotImplementedError

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Shared helpers for local interfaces                                          #
# --------------------------------------------------------------------------- #


def extract_images(query: Query) -> list[Image.Image]:
    """Normalize query.images into a flat list of PIL Images."""
    if query.images is None:
        return []
    if isinstance(query.images, Image.Image):
        return [query.images]
    if isinstance(query.images, str):
        return [b64_to_image(query.images)]
    if isinstance(query.images, dict):
        return [
            v if isinstance(v, Image.Image) else b64_to_image(v)
            for v in query.images.values()
        ]
    return [
        i if isinstance(i, Image.Image) else b64_to_image(i)
        for i in query.images
    ]


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))
