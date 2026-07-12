from __future__ import annotations

import logging

from brain_factory.system import configure_hydra
from brain_factory.scaffold.recipe_base import RecipeBase
from brain_factory.scaffold.runner import ProcessorConfig, RunConfig

import hydra
from dacite import from_dict
from omegaconf import DictConfig, OmegaConf

logger: logging.Logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("--- Run Config ---")
    logger.info(OmegaConf.to_yaml(cfg.run))
    logger.info("--- Recipe Config ---")
    logger.info(OmegaConf.to_yaml(cfg.projects))

    run_config: RunConfig = from_dict(data_class=RunConfig, data=OmegaConf.to_container(cfg.run, resolve=True))
    processor_config: ProcessorConfig = ProcessorConfig()

    recipe: RecipeBase = hydra.utils.instantiate(cfg.projects)

    recipe.master_chef(
        raw_recipe_config=cfg.projects,
        processor_config=processor_config,
        run_config=run_config,
    )


if __name__ == "__main__":
    with configure_hydra():
        main()
