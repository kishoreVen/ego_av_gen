from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from brain_factory.scaffold.umbrella_model_base import UmbrellaModelBase


class DummyUmbrellaModel(UmbrellaModelBase):
    """A dummy umbrella model for testing the framework."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone: nn.Linear = nn.Linear(input_dim, hidden_dim)
        self.head: nn.Linear = nn.Linear(hidden_dim, num_classes)
        self.activation: nn.ReLU = nn.ReLU()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        x: torch.Tensor = batch["input"]
        hidden: torch.Tensor = self.activation(self.backbone(x))
        logits: torch.Tensor = self.head(hidden)
        return {"logits": logits}
