from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from brain_factory.scaffold.loss_gatherer import LossStatistics
from brain_factory.scaffold.runner import ProcessorConfig, RunConfig
from brain_factory.scaffold.training_recipe_base import (
    TrainerRecipeConfigBase,
    TrainingRecipeBase,
)

logger: logging.Logger = logging.getLogger(__name__)


class DummyTrainingRecipe(TrainingRecipeBase):
    """A dummy training recipe for testing the framework end-to-end."""

    def cook(
        self,
        processor_config: ProcessorConfig,
        run_config: RunConfig,
    ) -> None:
        model: torch.nn.Module = self.config.umbrella_model
        dataloader: DataLoader[Dict[str, Any]] = DataLoader(self.config.dataset, batch_size=8)

        optimizer, scheduler = self.make_optimizer_and_scheduler(model)

        device: torch.device = torch.device(run_config.device if run_config.device != "gpu" else "cuda")
        model.to(device)
        if self.config.loss_gatherer is not None:
            self.config.loss_gatherer.to(device)
        if self.config.train_monitor is not None:
            self.config.train_monitor.to(device)

        model.train()
        step: int = 0
        for batch in dataloader:
            if step >= self.config.max_steps:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            output: Dict[str, Any] = model(batch)

            loss_stats: LossStatistics = self.config.loss_gatherer(output, batch, step)
            loss: torch.Tensor = loss_stats.total_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1

            if self.config.train_monitor is not None:
                self.config.train_monitor.update_train_progress(
                    step=step,
                    loss_stats=loss_stats,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model=model,
                    run_config=run_config,
                    processor_config=processor_config,
                    predictions=output,
                    targets=batch,
                    samples_processed=step * 8,
                )
            elif step % 10 == 0:
                logger.info(f"Step {step}/{self.config.max_steps} | Loss: {loss.item():.4f}")

        logger.info(f"Training complete after {step} steps")
