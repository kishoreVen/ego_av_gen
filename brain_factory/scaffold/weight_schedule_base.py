from __future__ import annotations

from abc import abstractmethod

import torch


class WeightScheduleBase(torch.nn.Module):
    """Base class for loss weight schedules.

    Controls the scalar weight applied to a loss term at each training step.
    Subclass to implement warm-up, decay, cyclic, or other scheduling strategies.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, step: int) -> float:
        """Return the weight for this loss term at the given training step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            A float weight to multiply the loss term by.
        """
        ...
