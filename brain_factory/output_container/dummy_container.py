from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import torch

from brain_factory.scaffold.output_container_base import OutputContainerBase

logger: logging.Logger = logging.getLogger(__name__)


class DummyOutputContainer(OutputContainerBase):
    """A dummy output container that saves prediction/target summaries as JSON.

    Useful for testing and as a reference implementation.
    """

    def save(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        output_dir: str,
        device: torch.device,
    ) -> None:
        summary: Dict[str, Any] = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                summary[f"pred_{key}_shape"] = list(value.shape)
                summary[f"pred_{key}_mean"] = value.float().mean().item()
            else:
                summary[f"pred_{key}"] = str(value)

        output_path: str = os.path.join(output_dir, f"{self.output_folder}_summary.json")
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.debug(f"DummyOutputContainer saved to {output_path}")
