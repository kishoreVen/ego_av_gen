"""Base class for locally-hosted models.

Layer 1 of the local inference stack:
  models/base.py   — load weights, run forward / generate
  interfaces/      — bridges model_router protocol to local models

Subclasses implement load(), generate(), and unload().
The interface layer handles Query → model input and model output → response dict.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class LocalModel(ABC):
    """Abstract base for a locally-loaded model."""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights and tokenizer/processor onto device."""
        ...

    @abstractmethod
    def generate(self, **kwargs: Any) -> dict[str, Any]:
        """Run inference. Kwargs are model-specific.

        Returns a dict with at least a ``"text"`` key for text outputs,
        or other keys depending on the model type.
        """
        ...

    def unload(self) -> None:
        """Free GPU memory. Override for custom cleanup."""
        import torch

        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"[{self.model_id}] unloaded")
