from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from brain_factory.scaffold.loss_base import LossBase
from brain_factory.scaffold.weight_schedule_base import WeightScheduleBase


@dataclass
class LossTermConfig:
    """Configuration for a single loss term: the loss function and its weight schedule."""

    loss: LossBase
    weight_schedule: WeightScheduleBase


@dataclass
class LossStatistics:
    """Container for loss computation results.

    Attributes:
        total_loss: Scalar tensor — the weighted sum of all loss terms.
            This is the value to call .backward() on.
        loss_breakdown: Dict mapping loss name -> weighted scalar loss.
        unweighted_loss_breakdown: Dict mapping loss name -> unweighted scalar loss
            (raw mean before weight is applied).
    """

    total_loss: torch.Tensor
    loss_breakdown: Dict[str, torch.Tensor]
    unweighted_loss_breakdown: Dict[str, torch.Tensor]


class LossGatherer(torch.nn.Module):
    """Aggregates multiple weighted loss terms into a single scalar loss for backprop.

    Each loss term is a LossTermConfig containing a LossBase and a WeightScheduleBase.
    At each step:
    1. LossBase.forward() produces per-sample loss of shape (batch_size,)
    2. Mean across batch -> unweighted scalar
    3. WeightScheduleBase.forward(step) -> scalar weight
    4. weighted = unweighted * weight
    5. Sum all weighted terms -> total_loss
    """

    def __init__(self, losses: Dict[str, LossTermConfig]) -> None:
        super().__init__()
        self._loss_modules: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {name: term.loss for name, term in losses.items()}
        )
        self._weight_schedules: torch.nn.ModuleDict = torch.nn.ModuleDict(
            {name: term.weight_schedule for name, term in losses.items()}
        )

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        step: int,
    ) -> LossStatistics:
        """Compute all loss terms, apply weights, and return statistics.

        Args:
            predictions: Dict from model forward pass.
            targets: Dict of ground truth.
            step: Current training step (for weight schedule and loss lookup).

        Returns:
            LossStatistics with total_loss, per-term weighted, and unweighted breakdowns.
        """
        loss_breakdown: Dict[str, torch.Tensor] = {}
        unweighted_loss_breakdown: Dict[str, torch.Tensor] = {}

        for name, loss_module in self._loss_modules.items():
            per_sample_loss: torch.Tensor = loss_module(predictions, targets, step)
            unweighted_scalar: torch.Tensor = per_sample_loss.mean()

            weight: float = self._weight_schedules[name](step)
            weighted_scalar: torch.Tensor = unweighted_scalar * weight

            unweighted_loss_breakdown[name] = unweighted_scalar
            loss_breakdown[name] = weighted_scalar

        total_loss: torch.Tensor = sum(loss_breakdown.values())  # type: ignore[assignment]

        return LossStatistics(
            total_loss=total_loss,
            loss_breakdown=loss_breakdown,
            unweighted_loss_breakdown=unweighted_loss_breakdown,
        )
