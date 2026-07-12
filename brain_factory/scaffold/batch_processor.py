from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn


class BatchProcessorBase(nn.Module):
    """Base class for batch preprocessing applied before model forward.

    Subclasses implement apply() to transform a batch and invert() to reverse it.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a batch before model forward."""
        ...

    @abstractmethod
    def invert(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Reverse the preprocessing."""
        ...

    def to(self, *args: Any, **kwargs: Any) -> BatchProcessorBase:
        return super().to(*args, **kwargs)  # type: ignore[return-value]

    def cuda(self, device: Optional[torch.device | int] = None) -> BatchProcessorBase:
        return super().cuda(device)  # type: ignore[return-value]

    def cpu(self) -> BatchProcessorBase:
        return super().cpu()  # type: ignore[return-value]


class BatchProcessorChain(nn.Module):
    """Ordered collection of BatchProcessorBase instances.

    Applies/inverts only processors matching a given process_profile.
    """

    def __init__(self, processors: List[BatchProcessorBase]) -> None:
        super().__init__()
        self._processors: nn.ModuleList = nn.ModuleList(processors)

    def apply(
        self,
        batch: Dict[str, Any],
        process_profile: Set[str] | None = None,
    ) -> Dict[str, Any]:
        """Apply processors in order. If process_profile is set, only run matching ones."""
        for processor in self._processors:
            if process_profile is None or type(processor).__name__ in process_profile:
                batch = processor.apply(batch)
        return batch

    def invert(
        self,
        batch: Dict[str, Any],
        process_profile: Set[str] | None = None,
    ) -> Dict[str, Any]:
        """Invert processors in reverse order. If process_profile is set, only run matching ones."""
        for processor in reversed(self._processors):
            if process_profile is None or type(processor).__name__ in process_profile:
                batch = processor.invert(batch)
        return batch

    def to(self, *args: Any, **kwargs: Any) -> BatchProcessorChain:
        return super().to(*args, **kwargs)  # type: ignore[return-value]

    def cuda(self, device: Optional[torch.device | int] = None) -> BatchProcessorChain:
        return super().cuda(device)  # type: ignore[return-value]

    def cpu(self) -> BatchProcessorChain:
        return super().cpu()  # type: ignore[return-value]
