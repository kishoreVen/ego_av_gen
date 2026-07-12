
"""Hardcoded pricing configuration for model interfaces.

Prices are in USD per million tokens. Update these values when provider
pricing changes.
"""

from dataclasses import dataclass


@dataclass
class ModelPricing:
    """Pricing per million tokens."""

    input_price_per_million: float
    output_price_per_million: float


# Pricing configuration keyed by interface name (from INTERFACE_REGISTRY)
# Prices as of January 2026
MODEL_PRICING: dict[str, ModelPricing] = {
    # Anthropic models
    "anthropic_haiku35": ModelPricing(0.80, 4.00),
    "anthropic_sonnet37": ModelPricing(3.00, 15.00),
    "anthropic_sonnet4": ModelPricing(3.00, 15.00),
    "anthropic_haiku45": ModelPricing(1.00, 5.00),
    "anthropic_sonnet45": ModelPricing(3.00, 15.00),
    "anthropic_opus45": ModelPricing(15.00, 75.00),
    "anthropic_opus41": ModelPricing(15.00, 75.00),
    # OpenAI models (https://platform.openai.com/docs/pricing)
    "openai_gpt4o": ModelPricing(2.50, 10.00),
    "openai_gpt4o_mini": ModelPricing(0.15, 0.60),
    "openai_gpt4_turbo": ModelPricing(10.00, 30.00),
    "openai_gpt5": ModelPricing(1.25, 10.00),
    "openai_gpt5_mini": ModelPricing(0.25, 2.00),
    "openai_gpt5_nano": ModelPricing(0.05, 0.40),
    "openai_gpt52": ModelPricing(1.75, 14.00),
    # Gemini models (https://ai.google.dev/gemini-api/docs/pricing)
    "gemini_pro3": ModelPricing(2.00, 12.00),
    "gemini_flash3": ModelPricing(0.50, 3.00),
    "gemini_pro25": ModelPricing(1.25, 10.00),
    "gemini_flash25": ModelPricing(0.30, 2.50),
    "gemini_flash_lite25": ModelPricing(0.10, 0.40),
    # Gemini image generation models (output tokens represent generated images)
    "gemini_nano_banana": ModelPricing(0.30, 30.00),
    "gemini_pro3_image": ModelPricing(2.00, 120.00),
    # Audio models have per-character pricing, not tracked here
}


def calculate_cost(
    interface_name: str,
    input_tokens: int,
    output_tokens: int,
    service_tier: str | None = None,
) -> float:
    """Calculate cost in USD for a given LLM call.

    Args:
        interface_name: The interface name from INTERFACE_REGISTRY
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        service_tier: Optional service tier (e.g., "flex" for 50% OpenAI discount)

    Returns:
        Cost in USD (0.0 if interface not found in pricing config)
    """
    pricing = MODEL_PRICING.get(interface_name)
    if not pricing:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing.input_price_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_price_per_million
    total_cost = input_cost + output_cost

    # Apply flex tier discount (50% off for OpenAI models)
    if service_tier == "flex" and interface_name.startswith("openai_"):
        total_cost *= 0.5

    return total_cost
