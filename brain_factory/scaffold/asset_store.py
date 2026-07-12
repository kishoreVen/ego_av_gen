from __future__ import annotations

import logging
from typing import Dict

import torch

from brain_factory.scaffold.asset_base import AssetBase

logger: logging.Logger = logging.getLogger(__name__)


class AssetStore:
    """Singleton store for globally shared assets.

    Lifecycle is owned by RecipeBase: populated in prepare(), cleared in cleanup().
    Sub-components (model, loss, etc.) access assets via AssetStore.get().
    """

    _instance: AssetStore | None = None
    _assets: Dict[str, AssetBase] = {}

    def __new__(cls) -> AssetStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._assets = {}
        return cls._instance

    @classmethod
    def register(cls, asset: AssetBase) -> None:
        """Register an asset by name."""
        logger.info(f"Registering asset: {asset.name}")
        cls._assets[asset.name] = asset

    @classmethod
    def get(cls, name: str) -> AssetBase:
        """Retrieve a registered asset by name."""
        if name not in cls._assets:
            raise KeyError(f"Asset '{name}' not found. Registered: {list(cls._assets.keys())}")
        return cls._assets[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered assets. Called by RecipeBase.cleanup()."""
        logger.info(f"Clearing {len(cls._assets)} assets")
        cls._assets.clear()

    @classmethod
    def move_all_to_device(cls, device: torch.device | str) -> None:
        """Move all registered assets to the specified device."""
        for asset in cls._assets.values():
            asset.move_to_device(device)

    @classmethod
    def names(cls) -> list[str]:
        """Return names of all registered assets."""
        return list(cls._assets.keys())
