from __future__ import annotations

import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Dict, List, Optional, Tuple, final

import torch
import torch.distributed

from omegaconf import DictConfig

from brain_factory.lib.learning.per_module_optimization import (
    gather_optimizer_param_groups,
)
from brain_factory.scaffold.loss_gatherer import LossGatherer
from brain_factory.scaffold.monitor import Monitor
from brain_factory.scaffold.recipe_base import RecipeBase, RecipeConfigBase
from brain_factory.scaffold.runner import ProcessorConfig, RunConfig
from brain_factory.scaffold.umbrella_model_base import UmbrellaModelBase

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class TrainerRecipeConfigBase(RecipeConfigBase):
    max_steps: int = 1000
    optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW  # type: ignore[assignment]
    learning_rate: float | Dict[str, float] = 1e-4
    learning_rate_scheduler: (
        Callable[..., torch.optim.lr_scheduler.LRScheduler] | None
    ) = None
    loss_gatherer: LossGatherer | None = None
    train_monitor: Monitor | None = None


class TrainingRecipeBase(RecipeBase):
    """Base class for training recipes.

    Extends RecipeBase with optimizer/scheduler construction, LR resolution,
    and training-specific prepare/cleanup (seeding, DDP, tensorboard).
    """

    def __init__(self, config: TrainerRecipeConfigBase) -> None:
        self.config = config
        super().__init__(config)

    def prepare(
        self,
        raw_recipe_config: DictConfig,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """Prepare for training: seed, DDP init, tensorboard setup.

        NOTE: If overriding, make sure to call the parent method.
        """
        super().prepare(raw_recipe_config, processor_config, run_config)

        torch.manual_seed(run_config.cook_settings.seed)
        if run_config.debug.torch_deterministic:
            torch.use_deterministic_algorithms(True)

        if (train_monitor := self.config.train_monitor) is not None:
            train_monitor.set_unresolved_recipe_config(raw_recipe_config)
            train_monitor.set_tb_writer(
                out_dir=run_config.output_dir,
                rank=processor_config.global_rank,
            )
            if processor_config.global_rank == 0:
                train_monitor.export_model_graph(
                    self.config.umbrella_model, run_config.output_dir
                )

    def close_kitchen(
        self,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """Cleanup after training: DDP teardown, tensorboard close.

        NOTE: If overriding, make sure to call the parent method.
        """
        if processor_config.ddp_enabled:
            torch.distributed.destroy_process_group()

        if self.config.train_monitor is not None:
            self.config.train_monitor.close()

    def cleanup(
        self,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """Full cleanup: close_kitchen + parent cleanup."""
        self.close_kitchen(processor_config, run_config)
        super().cleanup(processor_config, run_config)

    @abstractmethod
    def cook(
        self,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        """The main training loop. Concrete training recipes implement this."""
        ...

    @final
    @cached_property
    def default_learning_rate(self) -> float:
        """Resolve the default learning rate from config.

        If float, returns directly. If dict, requires a 'default' key.
        """
        if isinstance(lr := self.config.learning_rate, float):
            return lr
        elif isinstance(lr, dict):
            if "default" not in lr:
                raise ValueError(
                    f"Learning rate config must contain a 'default' key. Got {lr}"
                )
            return lr["default"]
        else:
            raise TypeError(
                f"learning_rate must be a float or Dict[str, float], got {type(lr)}"
            )

    def make_optimizer_and_scheduler(
        self,
        model: torch.nn.Module,
        max_steps_per_rank: int | None = None,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        """Build optimizer and LR scheduler from config.

        Uses gather_optimizer_parameter_groups() for param groups,
        defaults to ConstantLR if no scheduler is configured.
        """
        umbrella_model: UmbrellaModelBase = model  # type: ignore[assignment]

        optimizer: torch.optim.Optimizer = self.config.optimizer(
            self.gather_optimizer_parameter_groups(model=umbrella_model),
            lr=self.default_learning_rate,
        )

        if max_steps_per_rank is None:
            max_steps_per_rank = self.config.max_steps

        scheduler: torch.optim.lr_scheduler.LRScheduler
        if self.config.learning_rate_scheduler is None:
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=max_steps_per_rank
            )
        else:
            scheduler = self.config.learning_rate_scheduler(optimizer)

        return optimizer, scheduler

    @final
    def gather_optimizer_parameter_groups(
        self,
        model: UmbrellaModelBase,
    ) -> List[Dict[str, Any]]:
        """Build optimizer parameter groups based on learning rate type.

        If learning_rate is a float: single group with all requires_grad params.
        If learning_rate is a dict: delegates to gather_optimizer_param_groups().
        """
        if isinstance(lr := self.config.learning_rate, float):
            return [
                {
                    "params": [p for p in model.parameters() if p.requires_grad],
                    "name": self.__class__.__name__,
                    "lr": lr,
                }
            ]
        elif isinstance(lr, dict):
            return gather_optimizer_param_groups(model, lr)
        else:
            raise TypeError(
                f"learning_rate must be a float or Dict[str, float], got {type(lr)}"
            )
