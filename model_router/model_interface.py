
from abc import abstractmethod
from enum import StrEnum
from typing import Any, Dict, List

# Re-export Query classes for backward compatibility
from model_router.query import (
    StructuredPrompt,
    Query,
    ImageGenQuery,
    AudioGenQuery,
    SoundEffectsGenQuery,
    VideoGenQuery,
)

# Re-export for convenience
__all__ = [
    "Capability",
    "StructuredPrompt",
    "Query",
    "ImageGenQuery",
    "AudioGenQuery",
    "SoundEffectsGenQuery",
    "VideoGenQuery",
    "ModelInterface",
]


class Capability(StrEnum):
    """
    Defines all the capabilities that will be allowed for generation
    """

    AUDIO_GEN = "audio_generation"
    AUDIO_ENC = "audio_encoding"

    IMAGE_GEN = "image_generation"
    IMAGE_ENC = "image_encoding"
    IMAGE_OUTPAINT = "image_outpaint"
    IMAGE_INPAINT = "image_inpaint"

    VIDEO_GEN = "video_generation"
    VIDEO_ENC = "video_encoding"

    MESH_GEN = "mesh_generation"

    SOUND_EFFECTS_GEN = "sound_effects_generation"

    TEXT = "text"
    TEXT_THINKING = "text_thinking"


class ModelInterface:
    """
    Wrapper to create a client from the API source.

    Subclasses should override the `formatter` property to provide
    model-specific prompt formatting.
    """

    def __init__(self, seed: int | None) -> None:
        self.seed = seed

    @property
    def formatter(self):
        """Get the prompt formatter for this model.

        Override in subclasses to provide model-specific formatting.
        Default returns the generic formatter.

        Returns:
            A PromptFormatter instance appropriate for this model family.
        """
        from model_router.lib.prompt_formatter import DEFAULT_FORMATTER

        return DEFAULT_FORMATTER

    @abstractmethod
    def initialize_client(self) -> None: ...

    def requires_initialization(self) -> bool:
        return True

    @abstractmethod
    def supported_capabilities(self) -> List[Capability]: ...

    @abstractmethod
    def fetch_response(
        self, query: Query, capability: "Capability | None" = None
    ) -> Dict[str, Any]: ...
