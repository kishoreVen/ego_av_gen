from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

from torch.utils.data import Dataset


class DatasetBase(Dataset[Dict[str, Any]]):
    """Base dataset with a two-phase get pattern.

    Supports a pre-caching workflow:
    1. Run pre_cache_get_item() over the full dataset to compute and cache intermediates.
    2. During training, __getitem__ calls cached_get_item() to load cached results directly.

    This keeps the dataset configured in one place while supporting offline preprocessing
    (e.g. pre-tokenization, embedding computation) followed by fast cached training.
    """

    def __init__(self, use_cache: bool = False) -> None:
        super().__init__()
        self._use_cache: bool = use_cache

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._use_cache:
            return self.cached_get_item(idx)
        return self.pre_cache_get_item(idx)

    @abstractmethod
    def pre_cache_get_item(self, idx: int) -> Dict[str, Any]:
        """Full item retrieval — may include expensive computation.

        Used during the caching pass or when cache is not available.
        """
        ...

    @abstractmethod
    def cached_get_item(self, idx: int) -> Dict[str, Any]:
        """Fast item retrieval from cached/preprocessed data.

        Used during training after a caching pass has been run.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...
