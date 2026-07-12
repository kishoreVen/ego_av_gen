from __future__ import annotations

from typing import Any, Dict

import torch

from brain_factory.scaffold.loss_base import LossBase


class DummyCrossEntropyLoss(LossBase):
    """Cross-entropy loss for classification tasks.

    Computes per-sample cross-entropy between model logits and batch targets.
    Returns a tensor of shape (batch_size,).
    """

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        step: int,
    ) -> torch.Tensor:
        logits: torch.Tensor = predictions["logits"]
        target: torch.Tensor = targets["target"].long()
        if target.dim() == 0:
            target = target.unsqueeze(0)
        return torch.nn.functional.cross_entropy(logits, target, reduction="none")
