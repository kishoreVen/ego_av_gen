from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class UmbrellaModelBase(nn.Module):
    """Base wrapper for a model and its heads.

    An umbrella model bundles a backbone with one or more task heads,
    providing a unified forward interface.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run forward pass on a batch."""
        ...
