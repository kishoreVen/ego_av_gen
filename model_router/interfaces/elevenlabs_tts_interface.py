
"""
ElevenLabs Text-to-Speech Interface Module

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
         ┌──────────────────────────────────────────────────────────────┐
         │                    ElevenLabsTTSInterface                    │
         │                  (Base TTS Implementation)                   │
         │  ┌────────────────────────────────────────────────────────┐  │
         │  │ Capabilities: AUDIO_GEN                                │  │
         │  │                                                        │  │
         │  │ Common Features:                                       │  │
         │  │ - Text-to-speech generation                            │  │
         │  │ - Voice selection and settings                         │  │
         │  │ - Streaming/non-streaming modes                        │  │
         │  │ - Base64 audio output                                  │  │
         │  │                                                        │  │
         │  │ Voice Settings:                                        │  │
         │  │ - stability (0.0-1.0)                                  │  │
         │  │ - similarity_boost (0.0-1.0)                           │  │
         │  │ - style (v2+ models only)                              │  │
         │  │ - use_speaker_boost (v2+ models only)                  │  │
         │  └────────────────────────────────────────────────────────┘  │
         └──────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────┬─────────────────────┬───────────────────┐
          ▼                   ▼                     ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Multilingual V2  │ │    Flash V2      │ │    Turbo V2      │ │ Monolingual V1   │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ - 29 languages   │ │ - 32 languages   │ │ - Balanced       │ │ - English only   │
│ - High quality   │ │ - Ultra-low      │ │   quality/speed  │ │ - High quality   │
│ - Standard speed │ │   latency        │ │ - Good for most  │ │ - No style param │
│                  │ │ - Optimized for  │ │   use cases      │ │                  │
│                  │ │   speed          │ │                  │ │                  │
└──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘
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
    AudioGenQuery,
)

logger = logging.getLogger(__name__)


class ElevenLabsTTSInterface(ModelInterface):
    """Base interface for ElevenLabs text-to-speech models."""

    def __init__(self, model_name: str, seed: int | None) -> None:
        super().__init__(seed)
        self.model_name: str = model_name
        self.client = None

        # Default voice settings
        self.default_stability = 0.5
        self.default_similarity_boost = 0.75
        self.default_style = 0.0
        self.default_use_speaker_boost = True

    def initialize_client(self) -> None:
        """Initialize the ElevenLabs client."""
        self.client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

    def requires_initialization(self) -> bool:
        return self.client is None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.AUDIO_GEN]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        """Generate audio from text using ElevenLabs API."""
        query = cast(AudioGenQuery, query)

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Prepare text input
        text = self._prepare_text(query)

        # Get voice ID (use default if not specified)
        voice_id = query.voice_id or self._get_default_voice()

        # Prepare voice settings
        voice_settings = self._prepare_voice_settings(query)

        # Generate audio
        if query.stream:
            audio_data = self._generate_streaming(text, voice_id, query, voice_settings)
        else:
            audio_data = self._generate_standard(text, voice_id, query, voice_settings)

        # Convert audio to base64 for consistent output format
        audio_base64 = self._audio_to_base64(audio_data)

        return {
            "audio": audio_base64,
            "format": query.output_format or "mp3_44100_128",
            "voice_id": voice_id,
            "model": self.model_name,
        }

    def _prepare_text(self, query: AudioGenQuery) -> str:
        """Prepare text from query."""
        text = query.make_query()
        if not text:
            raise ValueError("Query must have either system_prompt or query_text")
        return text

    def _get_default_voice(self) -> str:
        """Get default voice ID for the model."""
        # Default to a common voice
        # In production, this could be configurable
        return "21m00Tcm4TlvDq8ikWAM"  # Rachel voice

    def _prepare_voice_settings(self, query: AudioGenQuery) -> Dict[str, Any]:
        """Prepare voice settings dictionary."""
        settings = {}

        settings["stability"] = query.stability or self.default_stability
        settings["similarity_boost"] = (
            query.similarity_boost or self.default_similarity_boost
        )

        # Style is only for v2 models
        if "v2" in self.model_name:
            settings["style"] = query.style or self.default_style
            settings["use_speaker_boost"] = (
                query.use_speaker_boost
                if query.use_speaker_boost is not None
                else self.default_use_speaker_boost
            )

        return settings

    def _generate_standard(
        self,
        text: str,
        voice_id: str,
        query: AudioGenQuery,
        voice_settings: Dict[str, Any],
    ) -> bytes:
        """Generate audio using standard (non-streaming) method."""
        try:
            # Prepare generation parameters
            params = {
                "text": text,
                "voice_id": voice_id,
                "model_id": self.model_name,
            }

            # Add optional parameters
            if query.output_format:
                params["output_format"] = query.output_format

            if query.language_code and "v2.5" in self.model_name:
                params["language_code"] = query.language_code

            if query.generation_seed is not None:
                params["seed"] = query.generation_seed

            if query.previous_text:
                params["previous_text"] = query.previous_text

            if query.next_text:
                params["next_text"] = query.next_text

            # Add voice settings
            params["voice_settings"] = voice_settings

            # Generate audio
            audio_generator = self.client.text_to_speech.convert(**params)

            # Collect all audio chunks
            audio_chunks = []
            for chunk in audio_generator:
                audio_chunks.append(chunk)

            # Combine chunks into single bytes object
            return b"".join(audio_chunks)

        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            raise ValueError(f"Audio generation failed: {e}")

    def _generate_streaming(
        self,
        text: str,
        voice_id: str,
        query: AudioGenQuery,
        voice_settings: Dict[str, Any],
    ) -> bytes:
        """Generate audio using streaming method."""
        try:
            # For streaming, we'll still collect all chunks and return as bytes
            # In a real streaming scenario, you'd yield chunks as they arrive
            params = {
                "text": text,
                "voice_id": voice_id,
                "model_id": self.model_name,
                "stream": True,
            }

            # Add optional parameters
            if query.output_format:
                params["output_format"] = query.output_format

            if query.language_code and "v2.5" in self.model_name:
                params["language_code"] = query.language_code

            if query.generation_seed is not None:
                params["seed"] = query.generation_seed

            # Add voice settings
            params["voice_settings"] = voice_settings

            # Generate audio stream
            audio_stream = self.client.text_to_speech.convert_as_stream(**params)

            # Collect streamed chunks
            audio_chunks = []
            for chunk in audio_stream:
                audio_chunks.append(chunk)

            return b"".join(audio_chunks)

        except Exception as e:
            logger.error(f"Error generating streaming audio: {e}")
            raise ValueError(f"Streaming audio generation failed: {e}")

    def _audio_to_base64(self, audio_data: bytes) -> str:
        """Convert audio bytes to base64 string."""
        return base64.b64encode(audio_data).decode("utf-8")


class ElevenLabsMultilingualV2(ElevenLabsTTSInterface):
    """ElevenLabs Multilingual v2 model - supports 29 languages."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("eleven_multilingual_v2", seed)

    def _get_default_voice(self) -> str:
        """Get default voice for multilingual model."""
        return "21m00Tcm4TlvDq8ikWAM"  # Rachel - good for multiple languages


class ElevenLabsFlashV2(ElevenLabsTTSInterface):
    """ElevenLabs Flash v2.5 model - ultra-low latency, supports 32 languages."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("eleven_flash_v2_5", seed)
        # Flash model optimized for speed
        self.default_stability = 0.5
        self.default_similarity_boost = 0.75

    def _get_default_voice(self) -> str:
        """Get default voice for flash model."""
        return "21m00Tcm4TlvDq8ikWAM"  # Rachel - optimized for speed


class ElevenLabsTurboV2(ElevenLabsTTSInterface):
    """ElevenLabs Turbo v2.5 model - balance of quality and speed."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("eleven_turbo_v2_5", seed)
        # Turbo model balanced settings
        self.default_stability = 0.5
        self.default_similarity_boost = 0.8

    def _get_default_voice(self) -> str:
        """Get default voice for turbo model."""
        return "21m00Tcm4TlvDq8ikWAM"  # Rachel - balanced quality


class ElevenLabsMonolingualV1(ElevenLabsTTSInterface):
    """ElevenLabs Monolingual v1 model - English only, high quality."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("eleven_monolingual_v1", seed)
        # V1 model doesn't support style parameter
        self.default_style = None

    def _get_default_voice(self) -> str:
        """Get default voice for monolingual model."""
        return "21m00Tcm4TlvDq8ikWAM"  # Rachel - optimized for English

    def _prepare_voice_settings(self, query: AudioGenQuery) -> Dict[str, Any]:
        """Prepare voice settings for v1 model (no style parameter)."""
        settings = {
            "stability": query.stability or self.default_stability,
            "similarity_boost": query.similarity_boost or self.default_similarity_boost,
        }
        # V1 doesn't support style or use_speaker_boost
        return settings
