from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AssetBase(ABC):
    """Base class for globally shared assets (e.g. 3D model templates, token vocabularies)."""

    def __init__(self, name: str) -> None:
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def move_to_device(self, device: torch.device | str) -> None:
        """Move this asset's tensors/data to the specified device."""
        ...
