
import io
import os
from typing import Any, Dict, List
from model_router.model_interface import (
    ModelInterface,
    Capability,
    Query,
    ImageGenQuery,
)
from model_router.lib.prompt_formatter import PromptFormatter
from model_router.lib.media import (
    image_to_base64,
    convert_query_images_to_base64_list,
    base64_to_image,
)

import openai
from openai.types import chat
from PIL import Image


class OpenAIPromptFormatter(PromptFormatter):
    """Formatter optimized for OpenAI models.

    Uses delimiter-based sections as recommended by OpenAI's
    prompt engineering best practices.

    OpenAI-specific optimizations:
    - Triple-dash delimiters for sections
    - Clear separation between instructions and content
    - Explicit section markers
    """

    def wrap_section(self, name: str, content: str) -> str:
        """Wrap content with delimiter markers."""
        delimiter = name.upper().replace(" ", "_").replace("-", "_")
        return f"---{delimiter}---\n{content}"

    def format_requirements(
        self, requirements: List[str], critical: List[str]
    ) -> str:
        """Format requirements with delimiters, critical first."""
        result = ""

        if critical:
            result += "---CRITICAL_REQUIREMENTS---\n"
            result += "\n".join(f"* {r}" for r in critical)

        if requirements:
            if result:
                result += "\n\n"
            result += "---REQUIREMENTS---\n"
            result += "\n".join(f"* {r}" for r in requirements)

        return result


# Singleton formatter instance for OpenAI models
_OPENAI_FORMATTER = OpenAIPromptFormatter()


class OpenAiInterface(ModelInterface):
    """Base interface for OpenAI models.

    Uses OpenAI-optimized prompt formatting with delimiters.
    """

    def __init__(self, model_name: str, seed: int | None) -> None:
        super().__init__(seed)

        # Client Config -- Hard Coded
        self.generation_temperature = 0.7
        self.generation_top_p = 0.6
        self.max_tokens = 4096  # Default value, override in subclasses

        self.client = None
        self.model_name: str = model_name

    @property
    def formatter(self) -> PromptFormatter:
        """OpenAI models use delimiter-based formatting."""
        return _OPENAI_FORMATTER

    def initialize_client(self) -> None:
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def requires_initialization(self) -> bool:
        return self.client == None

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.IMAGE_ENC]

    def _build_message_content(self, query: Query) -> List[Dict[str, Any]]:
        content = []

        # Add text content
        if query.query_text:
            content.append({"type": "text", "text": query.query_text})

        # Add image content
        if query.images:
            base64_images = convert_query_images_to_base64_list(query.images)
            for img_base64 in base64_images:
                content.append(self._create_image_content(img_base64))

        # Add video content (treated as sequence of images)
        if query.video:
            for frame in query.video:
                content.append(self._create_image_content(frame))

        return content

    def _create_image_content(self, image: Image.Image | str) -> Dict[str, Any]:
        if isinstance(image, str):
            # Assume it's already base64 encoded
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image}"},
            }
        elif isinstance(image, Image.Image):
            # Convert PIL Image to base64
            base64_data = image_to_base64(image)
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_data}"},
            }
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        content = self._build_message_content(query)

        system_prompt = query.system_prompt or query.get_system_prompt() or "You are a helpful assistant."

        openai_messages: List[chat.ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": content,
            },
        ]

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Use max_completion_tokens for newer models, max_tokens for older ones
        completion_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "seed": self.seed,
        }

        effective_temperature = query.temperature if query.temperature is not None else self.generation_temperature
        effective_top_p = query.top_p if query.top_p is not None else self.generation_top_p

        # GPT-5 models have restrictions on parameters
        if "gpt-5" in self.model_name.lower():
            completion_params["max_completion_tokens"] = self.max_tokens
            completion_params["reasoning_effort"] = "high"
            # GPT-5 models only support temperature=1 (default)
            # Don't include temperature and top_p parameters
        else:
            completion_params["max_tokens"] = self.max_tokens
            completion_params["temperature"] = effective_temperature
            completion_params["top_p"] = effective_top_p

        # Pass through service_tier for flex processing support
        if query.service_tier:
            completion_params["service_tier"] = query.service_tier

        response = self.client.chat.completions.create(**completion_params)

        if len(response.choices) <= 0:
            raise ValueError("Content generation failed")

        response_text: str = response.choices[0].message.content  # type: ignore

        return {
            "text": response_text,
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }


class OpenAiGPT4o(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gpt-4o", seed)
        self.max_tokens = 16384


class OpenAiGPT4oMini(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gpt-4o-mini", seed)
        self.max_tokens = 16384

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.IMAGE_ENC]


class OpenAiGPT4Turbo(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gpt-4-turbo", seed)
        self.max_tokens = 4096

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.IMAGE_ENC]


class OpenAiO1Mini(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("o1-mini", seed)
        self.max_tokens = 65536

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        # O1 models don't support system prompts, temperature, or top_p
        content = self._build_message_content(query)

        openai_messages: List[chat.ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": content,
            }
        ]

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # O1 models use max_completion_tokens instead of max_tokens
        completion_params = {
            "model": self.model_name,
            "messages": openai_messages,
            "max_completion_tokens": self.max_tokens,
        }

        # Pass through service_tier for flex processing support
        if query.service_tier:
            completion_params["service_tier"] = query.service_tier

        response = self.client.chat.completions.create(**completion_params)

        if len(response.choices) <= 0:
            raise ValueError("Content generation failed")

        response_text: str = response.choices[0].message.content  # type: ignore

        return {
            "text": response_text,
            "usage": {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }


class OpenAiGPT5(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gpt-5", seed)
        self.max_tokens = 8192 * 5

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.TEXT_THINKING, Capability.IMAGE_ENC]


class OpenAiGPT5Mini(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gpt-5-mini", seed)
        self.max_tokens = 8192 * 5

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.TEXT_THINKING, Capability.IMAGE_ENC]


class OpenAiGPT5Nano(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gpt-5-nano", seed)
        self.max_tokens = 8192 * 5

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.TEXT_THINKING]


class OpenAiGPT52(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gpt-5.2", seed)
        self.max_tokens = 8192 * 5

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.TEXT_THINKING, Capability.IMAGE_ENC]


class OpenAiImageGeneration(OpenAiInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__("gpt-image-1", seed)

        self.output_limit = 8

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.IMAGE_GEN]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        if not isinstance(query, ImageGenQuery):
            raise ValueError("Image generation requires ImageGenQuery")

        if self.client is None:
            raise ValueError("Call initialize_client before querying")

        # Prepare size - gpt-image-1 only supports 1024x1024
        size = "1024x1024"

        # Get number of images to generate
        num_outputs = min(query.number_of_results or 1, self.output_limit)

        # Build prompt with style if provided
        effective_prompt = query.system_prompt or query.get_system_prompt()
        if effective_prompt is None and query.query_text is None:
            raise ValueError("Image generation requires a query_text or system_prompt")

        prompt = effective_prompt if effective_prompt else ""

        if query.query_text:
            prompt = f"{prompt}\n\n{query.query_text}" if prompt else query.query_text

        prompt = prompt.strip()

        if query.negative_prompt:
            prompt = f"{prompt}. Avoid: {query.negative_prompt}"

        try:
            # Check if we have reference images for editing
            if query.images:
                # Convert reference images to base64 list
                base64_images = convert_query_images_to_base64_list(query.images)

                # Convert base64 strings to BytesIO for the API
                image_bytes_list = []
                for b64_str in base64_images:
                    # Convert base64 to PIL Image, then to bytes
                    pil_image = base64_to_image(b64_str)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format="PNG")
                    buffer.seek(0)
                    image_bytes_list.append(buffer)

                # Use images for editing (API accepts multiple images)
                response = self.client.images.edit(
                    model="gpt-image-1",
                    image=image_bytes_list,
                    prompt=prompt,
                    size=size,
                    input_fidelity="high",
                    n=num_outputs,
                )
            else:
                # Standard image generation
                response = self.client.images.generate(
                    model=self.model_name,
                    prompt=prompt,
                    size=size,
                    n=num_outputs,
                )

            # Return the first generated image as base64
            if response.data and len(response.data) > 0:
                responses = [data.b64_json for data in response.data]

                return {"images": responses}
            else:
                raise ValueError("No images generated")

        except Exception as e:
            raise ValueError(f"Image generation failed: {str(e)}")
