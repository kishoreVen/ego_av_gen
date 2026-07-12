from __future__ import annotations

from typing import Any, Dict

import torch

from brain_factory.scaffold.batch_processor import BatchProcessorBase


class DummyNormProcessor(BatchProcessorBase):
    """A dummy batch processor that normalizes the input tensor."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        super().__init__()
        self._mean: float = mean
        self._std: float = std

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["input"] = (batch["input"] - self._mean) / self._std
        return batch

    def invert(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["input"] = batch["input"] * self._std + self._mean
        return batch
