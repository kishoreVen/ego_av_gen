
import dataclasses
import logging
from typing import Any, Dict

from model_router.model_interface import (
    ModelInterface,
    Query,
    Capability,
)
from model_router import interfaces
from model_router.lib.retry import RetryConfig
from model_router.lib.pricing import calculate_cost
from model_router.lib.cost_tracker import get_current_accumulator

logger: logging.Logger = logging.getLogger(__name__)

# Fallback chains: when a model exhausts all retries, try the next provider.
# Maps interface name → fallback interface name.
DEFAULT_FALLBACK_MAP: Dict[str, str] = {
    # Gemini → Anthropic fallbacks
    "gemini_flash3": "openai_gpt4o_mini",
    "gemini_flash25": "openai_gpt4o_mini",
    "gemini_flash_lite25": "openai_gpt4o_mini",
    "gemini_pro3": "anthropic_sonnet4",
    "gemini_pro25": "anthropic_sonnet4",
    # Anthropic → OpenAI fallbacks
    "anthropic_haiku35": "openai_gpt4o_mini",
    "anthropic_haiku45": "openai_gpt4o_mini",
    "anthropic_sonnet4": "openai_gpt4o",
    "anthropic_sonnet45": "openai_gpt4o",
    "anthropic_opus45": "openai_gpt5",
    # OpenAI → Anthropic fallbacks
    "openai_gpt4o_mini": "anthropic_haiku45",
    "openai_gpt4o": "anthropic_sonnet4",
    "openai_gpt5": "anthropic_opus45",
}


class ModelRouter:
    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        default_service_tier: str | None = None,
        fallback_map: Dict[str, str] | None = None,
    ) -> None:
        """
        Args:
            retry_config: Configuration for retry behavior on API failures.
            default_service_tier: Default service tier for OpenAI API calls.
                Use "flex" for batch jobs to get ~50% cost reduction.
            fallback_map: Mapping of interface name to fallback interface name.
                When a model exhausts all retries, the fallback is attempted once.
                Defaults to DEFAULT_FALLBACK_MAP.
        """
        self.loaded_registry: Dict[str, ModelInterface] = {
            interface_name: interface(seed=None)
            for interface_name, interface in interfaces.INTERFACE_REGISTRY.items()
        }
        self.retry_config = retry_config or RetryConfig()
        self.default_service_tier = default_service_tier
        self.fallback_map = fallback_map if fallback_map is not None else DEFAULT_FALLBACK_MAP

    def determine_interface(self, query: Query, capability: Capability) -> str:
        # Route video generation to Runware Minimax
        if capability == Capability.VIDEO_GEN:
            return "runware_minimax"
        return "anthropic_haiku35"

    def get_response(
        self,
        query: Query,
        capability: Capability,
        interface_type: str | None,
    ) -> Dict[str, Any]:
        """
        Args
        ----
        query: The query to be run on the model. Use `build_query` to make the
                Query dataclass.

        capability: The capability (eg. image gen or video gen) that we would like from
                    the inferface.

        interface_type: the interface to be used for the query and the capability.
                        If None, the router will automatically determine the interface
                        in `determine_interface`

        Returns
        -------
            response from model for the requested `capability`
        """
        if query.is_empty():
            raise ValueError("Query empty. Can we be serious?")

        # Apply default service_tier if query doesn't have one set
        if self.default_service_tier and not query.service_tier:
            query = dataclasses.replace(query, service_tier=self.default_service_tier)

        if interface_type is None:
            interface_type = self.determine_interface(query, capability)
            logger.info(
                f"Automatically determined {interface_type} interface type as appropriate "
                f"for capability {capability}."
            )

        try:
            return self._call_interface(query, capability, interface_type)
        except Exception as primary_error:
            fallback_type = self.fallback_map.get(interface_type)
            if fallback_type is None:
                raise

            # Check fallback supports the requested capability
            fallback_interface = self.loaded_registry.get(fallback_type)
            if fallback_interface is None or capability not in fallback_interface.supported_capabilities():
                raise

            logger.warning(
                f"Primary interface {interface_type} failed after all retries: "
                f"{primary_error!r}. Falling back to {fallback_type}."
            )
            return self._call_interface(query, capability, fallback_type)

    def _call_interface(
        self, query: Query, capability: Capability, interface_type: str,
    ) -> Dict[str, Any]:
        """Execute a query against a specific interface with retries."""
        model_interface = self.loaded_registry[interface_type]

        if capability not in model_interface.supported_capabilities():
            raise ValueError(
                f"{capability} not supported by interface_type ({interface_type})"
            )

        if model_interface.requires_initialization():
            model_interface.initialize_client()

        # Apply model-specific prompt formatting
        formatted_query = model_interface.formatter(query)

        # Execute with exponential backoff retry
        for attempt in self.retry_config.attempts(interface_type):
            with attempt:
                response = model_interface.fetch_response(formatted_query, capability)

                # Calculate cost if usage data is present
                if "usage" in response:
                    usage = response["usage"]
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                    cost = calculate_cost(
                        interface_type, input_tokens, output_tokens, query.service_tier
                    )
                    response["cost"] = cost
                    response["interface"] = interface_type

                    # Register with thread-local accumulator if active
                    accumulator = get_current_accumulator()
                    if accumulator:
                        accumulator.add(input_tokens, output_tokens, cost)

                return response
