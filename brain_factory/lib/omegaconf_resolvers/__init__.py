from __future__ import annotations

from brain_factory.lib.omegaconf_resolvers.dir_resolver import register_datetime_dir_resolver


def register_resolvers() -> None:
    """Register all custom OmegaConf resolvers for brain_factory."""
    register_datetime_dir_resolver()
