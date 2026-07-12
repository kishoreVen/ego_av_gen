
"""Dummy interface for testing purposes.

This interface returns pre-configured responses without making API calls.
Useful for unit testing components that depend on the model router.
"""

from typing import Any, Dict, List

from model_router.model_interface import Capability, ModelInterface, Query


class DummyInterface(ModelInterface):
    """A dummy model interface that returns pre-configured responses.

    Use this for testing by setting responses before calls:

        # In test setup
        DummyInterface.set_responses([
            {"text": '{"action": "proceed", "feedback": "Looks good"}'},
            {"text": '{"action": "revise", "feedback": "Fix this"}'},
        ])

        # Then use "dummy" as interface_type in router calls
        response = router.get_response(query, Capability.TEXT, "dummy")
    """

    # Class-level state for test responses
    _responses: List[Dict[str, Any]] = []
    _call_count: int = 0
    _calls: List[Dict[str, Any]] = []

    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)

    @classmethod
    def set_responses(cls, responses: List[Dict[str, Any]]) -> None:
        """Set the responses to return in sequence.

        Args:
            responses: List of response dicts to return. Each should have
                      the same structure as real interface responses
                      (e.g., {"text": "..."} for TEXT capability).
        """
        cls._responses = responses
        cls._call_count = 0
        cls._calls = []

    @classmethod
    def get_calls(cls) -> List[Dict[str, Any]]:
        """Get all recorded calls for inspection in tests."""
        return cls._calls

    @classmethod
    def reset(cls) -> None:
        """Reset all state between tests."""
        cls._responses = []
        cls._call_count = 0
        cls._calls = []

    def initialize_client(self) -> None:
        pass

    def requires_initialization(self) -> bool:
        return False

    def supported_capabilities(self) -> List[Capability]:
        return list(Capability)

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        """Return the next pre-configured response.

        Records the call for later inspection.

        Raises:
            IndexError: If called more times than responses configured.
        """
        DummyInterface._calls.append({
            "query": query,
            "capability": capability,
        })

        if DummyInterface._call_count >= len(DummyInterface._responses):
            raise IndexError(
                f"DummyInterface called {DummyInterface._call_count + 1} times "
                f"but only {len(DummyInterface._responses)} responses configured. "
                f"Use DummyInterface.set_responses() to configure more responses."
            )

        response = DummyInterface._responses[DummyInterface._call_count]
        DummyInterface._call_count += 1
        return response
