"""Base classes for model_router."""

from model_router.scaffold.model_base import LocalModel
from model_router.scaffold.local_interface import LocalModelInterface
from model_router.scaffold.image_generation_interface import (
    ImageGenerationModelInterface,
)

__all__ = ["LocalModel", "LocalModelInterface", "ImageGenerationModelInterface"]
