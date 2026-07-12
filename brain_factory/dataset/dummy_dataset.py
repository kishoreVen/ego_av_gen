from __future__ import annotations

from typing import Any, Dict

import torch

from brain_factory.scaffold.dataset_base import DatasetBase


class DummyDataset(DatasetBase):
    """A dummy dataset that returns random tensors."""

    def __init__(self, num_samples: int = 100, input_dim: int = 16, use_cache: bool = False) -> None:
        super().__init__(use_cache=use_cache)
        self._num_samples: int = num_samples
        self._input_dim: int = input_dim

    def __len__(self) -> int:
        return self._num_samples

    def pre_cache_get_item(self, idx: int) -> Dict[str, Any]:
        return {
            "input": torch.randn(self._input_dim),
            "target": torch.randint(0, 2, (1,)).item(),
        }

    def cached_get_item(self, idx: int) -> Dict[str, Any]:
        # Dummy: same as pre_cache since there's no real caching
        return self.pre_cache_get_item(idx)
