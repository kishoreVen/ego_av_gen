
from google import genai

from typing import Any, Dict, List, cast
from PIL import Image
import io
import base64
import os
import time

from model_router.model_interface import (
    ModelInterface,
    Capability,
    Query,
    ImageGenQuery,
    VideoGenQuery,
)
from model_router.lib.prompt_formatter import PromptFormatter
import logging

logger = logging.getLogger(__name__)


class GeminiPromptFormatter(PromptFormatter):
    """Formatter optimized for Gemini models.

    Uses Markdown headers and structured formatting as recommended
    by Google's prompting strategies documentation.

    Gemini-specific optimizations:
    - Markdown headers (##) for sections
    - Clear hierarchical structure
    - Position essential constraints first
    """

    def wrap_section(self, name: str, content: str) -> str:
        """Wrap content with Markdown header."""
        return f"## {name}\n{content}"

    def format_requirements(
        self, requirements: List[str], critical: List[str]
    ) -> str:
        """Format requirements with Markdown headers, critical first."""
        result = ""

        if critical:
            result += "## Critical Requirements\n"
            result += "\n".join(f"* {r}" for r in critical)

        if requirements:
            if result:
                result += "\n\n"
            result += "## Requirements\n"
            result += "\n".join(f"* {r}" for r in requirements)

        return result


# Singleton formatter instance for Gemini models
_GEMINI_FORMATTER = GeminiPromptFormatter()


# Gemini-supported aspect ratios mapped to their numeric values
_ASPECT_RATIOS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "21:9": 21 / 9,
}


def _resolution_to_aspect_ratio(width: int, height: int) -> str:
    """Convert resolution to closest Gemini-supported aspect ratio.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Gemini aspect ratio string (e.g., "1:1", "4:3", "21:9")
    """
    if height == 0:
        return "1:1"

    target_ratio = width / height

    # Find the closest supported aspect ratio
    closest_ratio = "1:1"
    min_diff = float("inf")

    for ratio_str, ratio_val in _ASPECT_RATIOS.items():
        diff = abs(target_ratio - ratio_val)
        if diff < min_diff:
            min_diff = diff
            closest_ratio = ratio_str

    return closest_ratio


def _extract_images_from_query(
    images: Image.Image | str | List | Dict | None,
) -> List[Image.Image | str]:
    """Extract images from various input formats.

    Args:
        images: Images in various formats (single, list, or dict)

    Returns:
        List of images (PIL Image or base64 string)
    """
    if not images:
        return []

    images_to_process = []

    if isinstance(images, (Image.Image, str)):
        images_to_process = [images]
    elif isinstance(images, list):
        images_to_process = images
    elif isinstance(images, dict):
        # Extract images from dict of str -> list -> list structure
        for _, value in images.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, list) and len(item) > 0:
                        images_to_process.append(item[0])
                    else:
                        images_to_process.append(item)
            else:
                images_to_process.append(value)

    return images_to_process


def _convert_image_to_part(img: Image.Image | str) -> Any:
    """Convert an image to a Gemini Part object.

    Args:
        img: PIL Image or base64 string

    Returns:
        Gemini Part object
    """
    if isinstance(img, str):
        img_bytes = base64.b64decode(img)
        return genai.types.Part.from_bytes(data=img_bytes, mime_type="image/webp")
    elif isinstance(img, Image.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return genai.types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    return None


def _build_multi_modal_content(
    query: Query,
    include_video: bool = False,
    raise_on_empty: bool = False,
) -> str | List:
    """Build multi-modal content for Gemini API.

    Args:
        query: Query containing text and/or images/video
        include_video: Whether to process video frames
        raise_on_empty: Whether to raise ValueError if no content

    Returns:
        Content for Gemini API (string or list with images and text)
    """
    text_prompt = query.make_query()

    # For text-only, return string directly
    has_images = query.images is not None
    has_video = include_video and query.video is not None
    if not has_images and not has_video:
        if raise_on_empty and not text_prompt:
            raise ValueError("No content provided")
        return text_prompt

    # For multi-modal, build a list with images as Part objects and text
    contents = []
    image_parts = []

    # Process images
    if query.images:
        images_to_process = _extract_images_from_query(query.images)
        if images_to_process:
            logger.info(f"Extracted {len(images_to_process)} images for processing")

        for img in images_to_process:
            part = _convert_image_to_part(img)
            if part:
                image_parts.append(part)

    # Process video frames (treated as sequence of images)
    if include_video and query.video:
        for frame in query.video:
            part = _convert_image_to_part(frame)
            if part:
                image_parts.append(part)

    # Images first, then text
    if image_parts:
        contents.extend(image_parts)
    if text_prompt:
        contents.append(text_prompt)

    if raise_on_empty and not contents:
        raise ValueError("No content provided for image generation")

    return contents if contents else text_prompt


class GeminiInterface(ModelInterface):
    """Base interface for Google Gemini models.

    Uses Gemini-optimized prompt formatting with Markdown headers.
    """

    # Default temperature for Gemini text generation
    generation_temperature: float = 0.7

    def __init__(
        self, model_name: str, seed: int | None, thinking: bool = False
    ) -> None:
        super().__init__(seed)
        self.model_name: str = model_name
        self.thinking: bool = thinking
        self.client = None

    @property
    def formatter(self) -> PromptFormatter:
        """Gemini models use Markdown header formatting."""
        return _GEMINI_FORMATTER

    def initialize_client(self) -> None:
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"timeout": 180000},
        )

    def requires_initialization(self) -> bool:
        return super().requires_initialization()

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        contents = _build_multi_modal_content(query, include_video=True)

        effective_temperature = query.temperature if query.temperature is not None else self.generation_temperature

        thinking_budget = 0 if not self.thinking else -1
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=genai.types.GenerateContentConfig(
                temperature=effective_temperature,
                top_p=query.top_p,
                top_k=query.top_k,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_budget=thinking_budget
                ),
            ),
        )
        return {
            "text": response.text,
            "usage": {
                "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            },
        }


class GeminiPro25(GeminiInterface):
    def __init__(self, seed: int | None) -> None:
        # Gemini Pro 2.5 requires thinking
        super().__init__("gemini-2.5-pro", seed, thinking=True)


class GeminiPro3(GeminiInterface):
    def __init__(self, seed: int | None) -> None:
        # Gemini Pro 3 supports thinking via TEXT_THINKING capability
        super().__init__("gemini-3-pro-preview", seed, thinking=False)

    def supported_capabilities(self) -> List[Capability]:
        return [
            Capability.TEXT,
            Capability.TEXT_THINKING,
            Capability.IMAGE_ENC,
            Capability.VIDEO_ENC,
        ]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        contents = _build_multi_modal_content(query, include_video=True)

        effective_temperature = query.temperature if query.temperature is not None else self.generation_temperature

        # Build config based on capability
        if capability == Capability.TEXT_THINKING:
            config = genai.types.GenerateContentConfig(
                temperature=effective_temperature,
                top_p=query.top_p,
                top_k=query.top_k,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_level=genai.types.ThinkingLevel.HIGH
                ),
            )
        else:
            config = genai.types.GenerateContentConfig(
                temperature=effective_temperature,
                top_p=query.top_p,
                top_k=query.top_k,
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return {
            "text": response.text,
            "usage": {
                "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            },
        }


class GeminiFlash25(GeminiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gemini-2.5-flash", seed)


class GeminiFlashLite25(GeminiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gemini-2.5-flash-lite", seed)


class GeminiFlash3(GeminiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gemini-3-flash-preview", seed, thinking=False)

    def supported_capabilities(self) -> List[Capability]:
        return [
            Capability.TEXT,
            Capability.TEXT_THINKING,
            Capability.IMAGE_ENC,
            Capability.VIDEO_ENC,
        ]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        contents = _build_multi_modal_content(query, include_video=True)

        effective_temperature = query.temperature if query.temperature is not None else self.generation_temperature

        if capability == Capability.TEXT_THINKING:
            config = genai.types.GenerateContentConfig(
                temperature=effective_temperature,
                top_p=query.top_p,
                top_k=query.top_k,
                thinking_config=genai.types.ThinkingConfig(
                    thinking_level=genai.types.ThinkingLevel.HIGH
                ),
            )
        else:
            config = genai.types.GenerateContentConfig(
                temperature=effective_temperature,
                top_p=query.top_p,
                top_k=query.top_k,
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return {
            "text": response.text,
            "usage": {
                "input_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "output_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
            },
        }


class GeminiImageInterface(ModelInterface):
    """Base interface for Gemini image generation models."""

    def __init__(self, model_name: str, seed: int | None) -> None:
        super().__init__(seed)
        self.model_name: str = model_name
        self.client = None

    @property
    def formatter(self) -> PromptFormatter:
        """Gemini image models also use Markdown header formatting."""
        return _GEMINI_FORMATTER

    def initialize_client(self) -> None:
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"timeout": 180000},
        )

    def requires_initialization(self) -> bool:
        return super().requires_initialization()

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.IMAGE_GEN]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        query = cast(ImageGenQuery, query)

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Build multi-modal content using shared function
        contents = _build_multi_modal_content(
            query, include_video=False, raise_on_empty=True
        )

        # Get number of results requested (default to 1)
        num_results = query.number_of_results if query.number_of_results else 1

        # Derive aspect ratio from image_resolution
        aspect_ratio = "1:1"  # default
        if query.image_resolution:
            aspect_ratio = _resolution_to_aspect_ratio(*query.image_resolution)

        # Build config for image generation
        config = genai.types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=genai.types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        )

        all_generated_images = []
        total_input_tokens = 0
        total_output_tokens = 0

        # Gemini generates one image per request, so loop if multiple requested
        for i in range(num_results):
            if num_results > 1:
                logger.info(f"Generating image {i+1}/{num_results}")

            # Generate image using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            # Accumulate usage metadata
            if response.usage_metadata:
                total_input_tokens += response.usage_metadata.prompt_token_count or 0
                total_output_tokens += response.usage_metadata.candidates_token_count or 0

            # Extract generated image from response
            if not response.candidates or not response.candidates[0].content.parts:
                # Log any text the model returned instead of an image
                text_response = None
                if response.candidates and response.candidates[0].content:
                    text_parts = [
                        p.text for p in response.candidates[0].content.parts
                        if hasattr(p, "text") and p.text
                    ] if response.candidates[0].content.parts else []
                    text_response = " ".join(text_parts) if text_parts else None
                logger.warning(
                    f"No image generated in response. "
                    f"Model text: {text_response or '(empty response)'}"
                )
                raise ValueError(f"No image generated in response. Model text: {text_response or '(empty response)'}")

            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and hasattr(part.inline_data, "data"):
                    # Image data is in the response
                    image_data = part.inline_data.data

                    # Convert to base64 if needed
                    if isinstance(image_data, bytes):
                        base64_string = base64.b64encode(image_data).decode("utf-8")
                    else:
                        base64_string = image_data

                    all_generated_images.append(base64_string)

        if not all_generated_images:
            # Collect any text the model returned instead of images
            text_parts = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                text_parts = [
                    p.text for p in response.candidates[0].content.parts
                    if hasattr(p, "text") and p.text
                ]
            text_response = " ".join(text_parts) if text_parts else "(no text)"
            logger.error(
                f"No images found in response. "
                f"Model text: {text_response}, "
                f"query_text={query.query_text[:500] if query.query_text else None}, "
                f"num_input_images={len(query.images) if query.images else 0}"
            )
            raise ValueError("No images found in response")

        return {
            "images": all_generated_images,
            "usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            },
        }


class GeminiNanoBanana(GeminiImageInterface):
    """Gemini's native image generation model with a fun name!"""

    def __init__(self, seed: int | None) -> None:
        super().__init__("gemini-2.5-flash-image", seed)


class GeminiNanoBanana2(GeminiImageInterface):
    """Gemini 3.1 Flash image generation model, optimized for speed and high-volume use."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("gemini-3.1-flash-image-preview", seed)


class GeminiPro3Image(GeminiImageInterface):
    """Gemini 3 Pro image generation model."""

    def __init__(self, seed: int | None) -> None:
        super().__init__("gemini-3-pro-image-preview", seed)


def _to_pil_image(img: Image.Image | str) -> Image.Image | None:
    """Convert a base64 string or PIL Image to a PIL Image.

    Args:
        img: PIL Image or base64 string (with or without data URI prefix)

    Returns:
        PIL Image or None if conversion fails
    """
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, str):
        b64_data = img
        if b64_data.startswith("data:"):
            b64_data = b64_data.split(",", 1)[1] if "," in b64_data else b64_data
        img_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(img_bytes))
    return None


class GeminiVeo31(ModelInterface):
    """Veo 3.1 video generation via Google's Gemini API.

    Supports text-to-video, image-to-video (via seed_image), and
    reference image guidance (via query.images, up to 3).
    """

    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)
        self.model_name = "veo-3.1-generate-preview"
        self.client = None

    @property
    def formatter(self) -> PromptFormatter:
        return _GEMINI_FORMATTER

    def initialize_client(self) -> None:
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"],
            http_options={"timeout": 180000},
        )

    def requires_initialization(self) -> bool:
        return self.client is None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.VIDEO_GEN]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        query = cast(VideoGenQuery, query)

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Build text prompt
        text_prompt = query.system_prompt or query.get_system_prompt() or ""
        if query.query_text:
            if text_prompt:
                text_prompt += "\n\n"
            text_prompt += query.query_text

        if not text_prompt:
            raise ValueError("Video generation requires a text prompt")

        if query.fps:
            text_prompt += f"\n\nGenerate at {query.fps} frames per second."

        # Process seed image for image-to-video
        seed_image_part = None
        if query.seed_image:
            seed_image_pil = _to_pil_image(query.seed_image)
            if seed_image_pil:
                buf = io.BytesIO()
                seed_image_pil.save(buf, format="PNG")
                seed_image_part = genai.types.Image(
                    image_bytes=buf.getvalue(), mime_type="image/png"
                )

        # Process reference images from query.images (up to 3)
        reference_images = None
        if query.images:
            images_list = _extract_images_from_query(query.images)
            if images_list:
                reference_images = []
                for img in images_list[:3]:
                    pil_img = _to_pil_image(img)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        ref_image = genai.types.Image(
                            image_bytes=buf.getvalue(), mime_type="image/png"
                        )
                        reference_images.append(
                            genai.types.VideoGenerationReferenceImage(
                                image=ref_image,
                                reference_type="asset",
                            )
                        )

        # Determine aspect ratio and resolution from video_resolution
        aspect_ratio = "16:9"
        resolution = "720p"
        if query.video_resolution:
            w, h = query.video_resolution
            aspect_ratio = "9:16" if h > w else "16:9"
            short_side = min(w, h)
            if short_side >= 2160:
                resolution = "4k"
            elif short_side >= 1080:
                resolution = "1080p"

        # Map duration to Veo-supported values: "4", "6", "8"
        duration_str = "8"
        if query.duration:
            if query.duration <= 4:
                duration_str = "4"
            elif query.duration <= 6:
                duration_str = "6"

        # 1080p and 4k require duration "8"
        if resolution in ("1080p", "4k"):
            duration_str = "8"

        # Build config
        config_kwargs: Dict[str, Any] = {
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "duration_seconds": duration_str,
            "number_of_videos": query.number_of_results or 1,
        }
        if query.negative_prompt:
            config_kwargs["negative_prompt"] = query.negative_prompt
        if reference_images:
            config_kwargs["reference_images"] = reference_images

        config = genai.types.GenerateVideosConfig(**config_kwargs)

        # Build generate_videos kwargs
        generate_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": text_prompt,
            "config": config,
        }
        if seed_image_part:
            generate_kwargs["image"] = seed_image_part

        logger.info(
            f"Generating video with Veo 3.1: "
            f"resolution={resolution}, aspect_ratio={aspect_ratio}, "
            f"duration={duration_str}s, "
            f"has_seed_image={seed_image_part is not None}, "
            f"num_reference_images={len(reference_images) if reference_images else 0}"
        )

        # Submit video generation (async operation)
        operation = self.client.models.generate_videos(**generate_kwargs)

        # Poll for completion (Veo takes 11s to 6min)
        while not operation.done:
            time.sleep(10)
            operation = self.client.operations.get(operation)

        if not operation.response or not operation.response.generated_videos:
            raise ValueError("No video generated in Veo response")

        # Extract video data
        video_results = []
        for generated_video in operation.response.generated_videos:
            video_bytes = self.client.files.download(file=generated_video.video)
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")
            video_results.append({"base64": video_b64})

        logger.info(f"Veo 3.1 generated {len(video_results)} video(s)")
        return {"videos": video_results}
