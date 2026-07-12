
from typing import Any, Dict, List
from model_router.model_interface import ModelInterface, Capability, Query


class ReplicateInterface(ModelInterface):
    def __init__(self, seed: int | None) -> None:
        super().__init__(seed)

    def supported_capabilities(self) -> List[Capability]:
        return [Capability.TEXT, Capability.IMAGE_ENC]

    def fetch_response(
        self, query: Query, capability: Capability | None = None
    ) -> Dict[str, Any]:
        return {}
