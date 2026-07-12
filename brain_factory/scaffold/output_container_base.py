from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class OutputContainerBase(ABC):
    """Base class for output containers that save/visualize model outputs.

    Output containers are plugged into Monitor and called at configurable
    intervals to persist predictions, targets, or derived visualizations.
    """

    def __init__(self, output_folder: str = "") -> None:
        self._output_folder: str = output_folder

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @abstractmethod
    def save(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        output_dir: str,
        device: torch.device,
    ) -> None:
        """Save predictions/targets to output_dir.

        Args:
            predictions: Dict from model forward pass.
            targets: Dict of ground truth.
            output_dir: Directory to write outputs into.
            device: Current device (for tensor operations if needed).
        """
        ...

    def cleanup(self) -> None:
        """Cleanup resources. Default is no-op."""
        pass

    def to(self, device: torch.device) -> OutputContainerBase:
        """Move internal PyTorch modules to device. Default returns self."""
        return self
