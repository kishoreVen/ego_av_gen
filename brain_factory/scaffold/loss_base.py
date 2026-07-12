from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

import torch


class LossBase(torch.nn.Module):
    """Abstract base for individual loss computations.

    Each LossBase subclass computes a single loss term from model predictions
    and targets. Returns a per-sample loss tensor of shape (batch_size,)
    to allow downstream weighting and reduction.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        step: int,
    ) -> torch.Tensor:
        """Compute per-sample loss.

        Args:
            predictions: Dict from model forward pass (e.g. {"logits": ...}).
            targets: Dict of ground truth (e.g. {"target": ...}).
            step: Current training step (for iteration-based tuning).

        Returns:
            Tensor of shape (batch_size,) — one loss value per sample.
        """
        ...
