
import os
from typing import Any, Dict, List
from model_router.model_interface import ModelInterface, Capability, Query
from model_router.lib.prompt_formatter import PromptFormatter
from model_router.lib.media import (
    image_to_base64_with_mime,
    convert_query_images_to_base64_list,
    detect_image_mime_type,
)

from anthropic import Anthropic
from anthropic.types.message_param import MessageParam as AnthropicMessageParam
from anthropic.types import TextBlockParam, ImageBlockParam
from PIL import Image


class ClaudePromptFormatter(PromptFormatter):
    """Formatter optimized for Claude models.

    Uses XML tags for clear section delineation, which Claude was
    specifically trained to recognize as a prompt organizing mechanism.

    Claude-specific optimizations:
    - XML tags like <instructions>, <context>, <requirements>
    - Critical constraints at the beginning
    - Avoids the word "think" (triggers extended thinking in Claude 4.x)
    """

    # Words to replace to avoid triggering extended thinking
    THINK_REPLACEMENTS = {
        "Think about": "Consider",
        "think about": "consider",
        "Think through": "Work through",
        "think through": "work through",
        "Think carefully": "Evaluate carefully",
        "think carefully": "evaluate carefully",
    }

    def wrap_section(self, name: str, content: str) -> str:
        """Wrap content in XML tags."""
        tag = name.lower().replace(" ", "_").replace("-", "_")
        return f"<{tag}>\n{content}\n</{tag}>"

    def format_requirements(
        self, requirements: List[str], critical: List[str]
    ) -> str:
        """Format requirements with XML tags, critical first."""
        result = ""

        if critical:
            result += "<critical_requirements>\n"
            result += "\n".join(f"* {r}" for r in critical)
            result += "\n</critical_requirements>"

        if requirements:
            if result:
                result += "\n\n"
            result += "<requirements>\n"
            result += "\n".join(f"* {r}" for r in requirements)
            result += "\n</requirements>"

        return result

    def format_system_prompt(self, prompt: str) -> str:
        """Apply Claude-specific transformations.

        Replaces "think" variants to avoid triggering extended thinking.
        """
        result = prompt
        for old, new in self.THINK_REPLACEMENTS.items():
            result = result.replace(old, new)
        return result


# Singleton formatter instance for Claude models
_CLAUDE_FORMATTER = ClaudePromptFormatter()


class AnthropicInterface(ModelInterface):
    """Base interface for Anthropic Claude models.

    Uses Claude-optimized prompt formatting with XML tags.
    """

    def __init__(self, model_name: str, seed: int | None) -> None:
        super().__init__(seed)

        # Client Config -- Hard Coded
        self.max_tokens = 8192
        self.generation_temperature = 0.7
        self.generation_top_k = 5
        self.generation_top_p = 0.6

        self.client = None

        self.model_name: str = model_name

    @property
    def formatter(self) -> PromptFormatter:
        """Claude models use XML tag formatting."""
        return _CLAUDE_FORMATTER

    def initialize_client(self) -> None:
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def requires_initialization(self) -> bool:
        return self.client == None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.IMAGE_ENC, Capability.VIDEO_ENC]

    def _build_message_content(
        self, query: Query
    ) -> List[TextBlockParam | ImageBlockParam]:
        content = []

        # Add text content
        if query.query_text:
            content.append(TextBlockParam(type="text", text=query.query_text))

        # Add image content
        if query.images:
            base64_images = convert_query_images_to_base64_list(query.images)
            for img_base64 in base64_images:
                content.append(self._create_image_block(img_base64))

        # Add video content (treated as sequence of images)
        if query.video:
            for frame in query.video:
                content.append(self._create_image_block(frame))

        return content

    def _create_image_block(self, image: Image.Image | str) -> ImageBlockParam:
        if isinstance(image, str):
            # Detect actual mime type from base64 data
            mime_type = detect_image_mime_type(image)
            # Strip data URI prefix if present
            if image.startswith("data:"):
                image = image.split(",", 1)[1]
            return ImageBlockParam(
                type="image",
                source={"type": "base64", "media_type": mime_type, "data": image},
            )
        elif isinstance(image, Image.Image):
            # Convert PIL Image to base64 with correct mime type
            base64_data, mime_type = image_to_base64_with_mime(image)
            return ImageBlockParam(
                type="image",
                source={
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_data,
                },
            )
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        content = self._build_message_content(query)

        anthropic_messages: List[AnthropicMessageParam] = [
            AnthropicMessageParam(role="user", content=content)
        ]

        system_prompt = query.system_prompt or query.get_system_prompt() or "You are a helpful assistant."

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        effective_temperature = query.temperature if query.temperature is not None else self.generation_temperature
        effective_top_p = query.top_p if query.top_p is not None else self.generation_top_p
        effective_top_k = query.top_k if query.top_k is not None else self.generation_top_k

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=anthropic_messages,
            max_tokens=self.max_tokens,
            temperature=effective_temperature,
            top_p=effective_top_p,
            top_k=effective_top_k,
        )

        if len(response.content) <= 0:
            raise ValueError("Content generation failed")

        response_text: str = response.content[0].text  # type: ignore

        return {
            "text": response_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }


class AnthropicHaiku35(AnthropicInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("claude-3-5-haiku-20241022", seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT]


class AnthropicSonnet37(AnthropicInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("claude-3-7-sonnet-20250219", seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT]


class AnthropicSonnet4(AnthropicInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("claude-sonnet-4-20250514", seed)


class AnthropicNoTopPInterface(AnthropicInterface):
    """Base class for models that don't allow both temperature and top_p."""

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        content = self._build_message_content(query)

        anthropic_messages: List[AnthropicMessageParam] = [
            AnthropicMessageParam(role="user", content=content)
        ]

        system_prompt = query.system_prompt or query.get_system_prompt() or "You are a helpful assistant."

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        effective_temperature = query.temperature if query.temperature is not None else self.generation_temperature
        effective_top_k = query.top_k if query.top_k is not None else self.generation_top_k

        response = self.client.messages.create(
            model=self.model_name,
            system=system_prompt,
            messages=anthropic_messages,
            max_tokens=self.max_tokens,
            temperature=effective_temperature,
            top_k=effective_top_k,
        )

        if len(response.content) <= 0:
            raise ValueError("Content generation failed")

        response_text: str = response.content[0].text  # type: ignore

        return {
            "text": response_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }


class AnthropicHaiku45(AnthropicNoTopPInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("claude-haiku-4-5-20251001", seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT]


class AnthropicSonnet45(AnthropicNoTopPInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("claude-sonnet-4-5-20250929", seed)


class AnthropicOpusInterface(AnthropicNoTopPInterface):
    """Base class for Opus models that support extended thinking."""

    # Default thinking budget in tokens
    thinking_budget_tokens: int = 10000

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.TEXT_THINKING, Capability.IMAGE_ENC, Capability.VIDEO_ENC]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        content = self._build_message_content(query)

        anthropic_messages: List[AnthropicMessageParam] = [
            AnthropicMessageParam(role="user", content=content)
        ]

        system_prompt = query.system_prompt or query.get_system_prompt() or "You are a helpful assistant."

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Build request kwargs
        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "system": system_prompt,
            "messages": anthropic_messages,
        }

        # Add thinking config if TEXT_THINKING capability requested
        if capability == Capability.TEXT_THINKING:
            # max_tokens must be greater than thinking budget
            request_kwargs["max_tokens"] = self.thinking_budget_tokens + self.max_tokens
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
        else:
            request_kwargs["max_tokens"] = self.max_tokens
            # Only add temperature/top_k when not thinking
            effective_temperature = query.temperature if query.temperature is not None else self.generation_temperature
            effective_top_k = query.top_k if query.top_k is not None else self.generation_top_k
            request_kwargs["temperature"] = effective_temperature
            request_kwargs["top_k"] = effective_top_k

        response = self.client.messages.create(**request_kwargs)

        if len(response.content) <= 0:
            raise ValueError("Content generation failed")

        # Extract text from response, handling thinking blocks
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text = block.text
                break

        return {
            "text": response_text,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }


class AnthropicOpus45(AnthropicOpusInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("claude-opus-4-5-20251101", seed)


class AnthropicOpus41(AnthropicOpusInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("claude-opus-4-1-20250805", seed)
