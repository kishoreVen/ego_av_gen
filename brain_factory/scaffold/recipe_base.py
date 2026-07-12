from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Final, Optional, final

from omegaconf import DictConfig

from torch.utils.data import Dataset

from brain_factory.lib.checkpoint import CheckpointConfig
from brain_factory.scaffold.asset_store import AssetStore
from brain_factory.scaffold.runner import ProcessorConfig, RunConfig
from brain_factory.scaffold.umbrella_model_base import UmbrellaModelBase

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class RecipeConfigBase:
    recipe_name: str
    dataset: Dataset[Any]
    umbrella_model: UmbrellaModelBase
    model_checkpoint: CheckpointConfig | None = None


class RecipeBase:
    """Base class for all recipes.

    Lifecycle: master_chef() calls prepare() -> cook() -> cleanup() in sequence.
    """

    def __init__(self, config: RecipeConfigBase) -> None:
        self.config: RecipeConfigBase = config
        logger.info(f"Initializing recipe {config.recipe_name}")

    def prepare(
        self,
        raw_recipe_config: DictConfig,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """Setup hook. Override to add recipe-specific initialization.

        NOTE: If overriding, make sure to call the parent method.
        """
        pass

    @abstractmethod
    def cook(
        self,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """Main loop (training or inference). Concrete recipes implement this."""
        ...

    def cleanup(
        self,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """Teardown hook. Override to add recipe-specific cleanup.

        NOTE: If overriding, make sure to call the parent method.
        """
        AssetStore.clear()

    @final
    def master_chef(
        self,
        raw_recipe_config: DictConfig,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """Orchestrator that runs the full recipe lifecycle."""
        self.prepare(
            raw_recipe_config=raw_recipe_config,
            processor_config=processor_config,
            run_config=run_config,
        )

        logger.info(f"Running recipe {self.__class__.__name__}")

        try:
            self.cook(
                processor_config=processor_config,
                run_config=run_config,
            )
        finally:
            self.cleanup(
                processor_config=processor_config,
                run_config=run_config,
            )
