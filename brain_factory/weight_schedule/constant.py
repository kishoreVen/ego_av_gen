from __future__ import annotations

from brain_factory.scaffold.weight_schedule_base import WeightScheduleBase


class ConstantWeightSchedule(WeightScheduleBase):
    """A weight schedule that returns a fixed weight at every step."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self._weight: float = weight

    def forward(self, step: int) -> float:
        return self._weight
