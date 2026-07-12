from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass

from brain_factory.scaffold.recipe_base import RecipeBase, RecipeConfigBase
from brain_factory.scaffold.runner import ProcessorConfig, RunConfig

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class InferenceRecipeConfigBase(RecipeConfigBase):
    inference_limit: int | None = None


class InferenceRecipeBase(RecipeBase):
    """Base class for inference recipes.

    Extends RecipeBase with an inference limit to cap inference at N samples/steps.
    """

    def __init__(self, config: InferenceRecipeConfigBase) -> None:
        self.config = config
        super().__init__(config)

    @abstractmethod
    def cook(
        self,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """The main inference loop. Concrete inference recipes implement this."""
        ...
