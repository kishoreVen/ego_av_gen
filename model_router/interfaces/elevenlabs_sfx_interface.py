
"""
ElevenLabs Sound Effects Interface Module

┌──────────────────────────────────────────────────────────────┐
│                        ModelInterface                        │
│                     (Abstract Base Class)                    │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ - initialize_client()                                  │  │
│  │ - requires_initialization()                            │  │
│  │ - supported_capabilities()                             │  │
│  │ - fetch_response()                                     │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                                │
                                │ Inherits
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                   ElevenLabsSoundEffects                      │
│                  (Sound Effects Generator)                    │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Capabilities:                                          │   │
│  │ - SOUND_EFFECTS_GEN                                    │   │
│  │                                                        │   │
│  │ Features:                                              │   │
│  │ - Text-to-sound-effect generation                      │   │
│  │ - Duration control (0.5-22 seconds)                    │   │
│  │ - Prompt influence adjustment (0.0-1.0)                │   │
│  │ - Base64 audio output                                  │   │
│  │                                                        │   │
│  │ Default Settings:                                      │   │
│  │ - default_duration: None (auto-determine)              │   │
│  │ - default_prompt_influence: 0.3                        │   │
│  └────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
"""

import base64
import os
from typing import Any, Dict, List, cast
import logging

from elevenlabs.client import ElevenLabs
from model_router.model_interface import (
    ModelInterface,
    Capability,
    Query,
    SoundEffectsGenQuery,
)

logger = logging.getLogger(__name__)


class ElevenLabsSoundEffects(ModelInterface):
    """ElevenLabs Sound Effects generation model."""

    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)
        self.client = None

        # Default settings for sound effects
        self.default_duration = None  # Let API auto-determine
        self.default_prompt_influence = 0.3

    def initialize_client(self) -> None:
        """Initialize the ElevenLabs client."""
        self.client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    def requires_initialization(self) -> bool:
        return self.client is None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.SOUND_EFFECTS_GEN]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        """Generate sound effects from text using ElevenLabs API."""
        query = cast(SoundEffectsGenQuery, query)

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Prepare text input
        text = self._prepare_text(query)

        # Generate sound effect
        sound_data = self._generate_sound_effect(text, query)

        # Convert audio to base64 for consistent output format
        sound_base64 = self._audio_to_base64(sound_data)

        return {
            "audio": sound_base64,
            "format": query.output_format or "mp3_44100_128",
            "duration": query.duration_seconds,
            "type": "sound_effect",
        }

    def _prepare_text(self, query: SoundEffectsGenQuery) -> str:
        """Prepare text description for sound effect generation."""
        text_parts = []

        if query.system_prompt:
            text_parts.append(query.system_prompt)

        if query.query_text:
            text_parts.append(query.query_text)

        if not text_parts:
            raise ValueError("Query must have either system_prompt or query_text")

        return " ".join(text_parts).strip()

    def _generate_sound_effect(self, text: str, query: SoundEffectsGenQuery) -> bytes:
        """Generate sound effect using ElevenLabs API."""
        try:
            # Prepare generation parameters
            params = {
                "text": text,
            }

            # Add optional duration
            if query.duration_seconds is not None:
                # Validate duration range (0.5 to 22 seconds)
                duration = max(0.5, min(22.0, query.duration_seconds))
                params["duration_seconds"] = duration

            # Add prompt influence
            if query.prompt_influence is not None:
                # Validate prompt influence range (0.0 to 1.0)
                influence = max(0.0, min(1.0, query.prompt_influence))
                params["prompt_influence"] = influence
            else:
                params["prompt_influence"] = self.default_prompt_influence

            # Generate sound effect
            logger.info(f"Generating sound effect with params: {params}")
            result = self.client.text_to_sound_effects.convert(**params)

            # Collect all audio chunks
            audio_chunks = []
            for chunk in result:
                audio_chunks.append(chunk)

            # Combine chunks into single bytes object
            return b"".join(audio_chunks)

        except Exception as e:
            logger.error(f"Error generating sound effect: {e}")
            raise ValueError(f"Sound effect generation failed: {e}")

    def _audio_to_base64(self, audio_data: bytes) -> str:
        """Convert audio bytes to base64 string."""
        return base64.b64encode(audio_data).decode("utf-8")
