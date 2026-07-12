
"""
Query data classes for model router requests.

This module contains pure data classes with no dependencies on formatters
or model interfaces. The formatting logic lives in PromptFormatter.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Tuple

import numpy.typing as npt
from PIL import Image


@dataclass
class StructuredPrompt:
    """A prompt with separated components for model-specific formatting.

    Use this to define prompts that can be formatted differently for each
    model backend (Claude uses XML tags, Gemini uses Markdown headers,
    OpenAI uses delimiters).

    Attributes:
        base_instruction: The main role/task description
        sections: Named sections (format specs, examples, context)
        critical_requirements: Must-follow requirements (placed first)
        requirements: Standard requirements

    Usage:
        prompt = StructuredPrompt(
            base_instruction="You are a story planner...",
            sections={"Input Format": "...", "Output Format": "..."},
            critical_requirements=["Must output valid JSON", ...],
            requirements=["Use age-appropriate content", ...],
        )

        # Create a Query with the structured prompt
        query = Query(structured_prompt=prompt, query_text="Create a story...")

        # Format using the model's formatter
        formatted_query = formatter(query)
    """

    base_instruction: str
    sections: Dict[str, str] = field(default_factory=dict)
    critical_requirements: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)

    def to_flat_prompt(self) -> str:
        """Convert to a flat string for backwards compatibility.

        This produces a generic format without model-specific optimizations.
        Prefer using the formatter for best results.

        Returns:
            A flat prompt string
        """
        result = self.base_instruction

        for name, content in self.sections.items():
            result += f"\n\n{name}:\n{content}"

        if self.critical_requirements:
            result += "\n\nCritical Requirements:\n"
            result += "\n".join(f"* {r}" for r in self.critical_requirements)

        if self.requirements:
            result += "\n\nRequirements:\n"
            result += "\n".join(f"* {r}" for r in self.requirements)

        return result


@dataclass
class Query:
    """Query object for model router requests.

    Supports both simple string prompts (system_prompt) and structured prompts
    (structured_prompt). When using structured prompts, use the model's
    formatter to get model-optimized formatting.
    """

    # Simple string system prompt (used when structured_prompt is None)
    system_prompt: str | None = None

    # Structured prompt for model-specific formatting
    structured_prompt: StructuredPrompt | None = None

    query_text: str | None = None

    # Support base64 or Pillow Image. Assume format is specified in Client Config.
    images: (
        Image.Image
        | str
        | List[Image.Image | str]
        | Dict[str, Image.Image | str]
        | None
    ) = None

    # Video specific format can be added later. Assume format is specified in Client Config.
    video: List[Image.Image | str] | None = None

    # Audio specific format can be added later. Assume numpy array for now and
    # format to be specified in Client Config
    audio: npt.NDArray | None = None

    # Generation parameter overrides. When set, model interfaces use these
    # instead of their hard-coded defaults.
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    # Service tier for OpenAI API ("default", "flex", "auto", or None)
    # Flex processing provides ~50% cost reduction with slower response times
    service_tier: str | None = None

    # Number of times to repeat the prompt body in make_query()
    repetitions: int = 1

    def is_empty(self) -> bool:
        """
        Checks if all attributes of a dataclass instance are None.
        """
        for f in fields(self):
            if getattr(self, f.name) is not None:
                return False

        return True

    def make_query(self) -> str:
        """Build the full prompt string for model interfaces.

        Combines system prompt and query text, replacing the manual
        concatenation pattern (system_prompt + '\\n\\n' + query_text)
        scattered across model interfaces. When repetitions > 1, the
        prompt is repeated with a separator for multi-sample generation.

        Returns:
            Full prompt string with repetitions joined by '\\n---\\n'.
        """
        parts = []
        system = self.get_system_prompt()
        if system:
            parts.append(system)
        if self.query_text:
            parts.append(self.query_text)
        full = "\n\n".join(parts)
        return "\n---\n".join([full] * self.repetitions)

    def get_system_prompt(self) -> str | None:
        """Get the effective system prompt.

        Returns:
            The system_prompt if set, otherwise the flat version of
            structured_prompt if set, otherwise None.
        """
        if self.system_prompt is not None:
            return self.system_prompt
        if self.structured_prompt is not None:
            return self.structured_prompt.to_flat_prompt()
        return None


@dataclass
class ImageGenQuery(Query):
    # Image resolution as a tuple (width, height)
    image_resolution: Tuple[int, int] | None = None

    image_format: str | None = None

    number_of_results: int | None = None

    generation_steps: int | None = None

    negative_prompt: str | None = None

    mask_image: str | Image.Image | None = None

    # Position for outpainting (x, y) - where to place the original image in the canvas
    image_position: Tuple[int, int] | None = None

    # Compaction prompt for diffusion model optimization - if provided, the query_text
    # and system_prompt will be compacted before image generation
    compaction_prompt: str | None = None

    # Model to use for compaction (defaults to "anthropic_haiku45")
    compaction_model: str = "anthropic_haiku45"


@dataclass
class AudioGenQuery(Query):
    # Voice identifier for the text-to-speech model
    voice_id: str | None = None

    # Model identifier (e.g., "eleven_multilingual_v2", "eleven_flash_v2_5")
    model_id: str | None = None

    # Output format (e.g., "mp3_44100_128", "mp3_22050_32")
    output_format: str | None = None

    # Voice settings
    stability: float | None = None  # 0.0 to 1.0
    similarity_boost: float | None = None  # 0.0 to 1.0
    style: float | None = None  # 0.0 to 1.0 (for v2 models)
    use_speaker_boost: bool | None = None

    # Whether to stream the audio
    stream: bool = False

    # Language code (ISO 639-1) for language enforcement
    language_code: str | None = None

    # Seed for deterministic generation
    generation_seed: int | None = None

    # Previous text for continuity
    previous_text: str | None = None

    # Next text for continuity
    next_text: str | None = None


@dataclass
class SoundEffectsGenQuery(Query):
    # Duration of the sound effect in seconds (0.5 to 22 seconds)
    # If None, API will automatically determine optimal duration
    duration_seconds: float | None = None

    # Prompt influence (0.0 to 1.0)
    # Higher values make generation follow prompt more closely
    prompt_influence: float | None = None

    # Output format for the sound effect
    output_format: str | None = None


@dataclass
class VideoGenQuery(Query):
    # Video resolution as a tuple (width, height)
    video_resolution: Tuple[int, int] | None = None

    # Duration of the video in seconds
    duration: float | None = None

    # Frames per second
    fps: int | None = None

    # Seed/reference image for image-to-video generation (first frame)
    seed_image: str | Image.Image | None = None

    # End frame image for image-to-video (e.g., Kling's image_tail, pro mode only)
    tail_image: str | Image.Image | None = None

    # Number of inference steps
    generation_steps: int | None = None

    # Guidance scale (CFG scale)
    cfg_scale: float | None = None

    # Negative prompt to avoid certain elements
    negative_prompt: str | None = None

    # Number of videos to generate
    number_of_results: int | None = None

    # Output format (MP4, WEBM)
    video_format: str | None = None

    # Aspect ratio string (e.g., "16:9", "9:16", "1:1")
    aspect_ratio: str | None = None

    # Generation mode (e.g., "std" for standard, "pro" for professional)
    mode: str | None = None

    # Whether to generate native audio alongside the video
    generate_audio: bool | None = None
