from __future__ import annotations

from datetime import datetime

from omegaconf import OmegaConf


def _datetime_dir(base: str = "./brain_factory_out") -> str:
    """Generate a timestamped output directory path.

    Returns ``{base}/{YYYY-MM-DD_HH-MM-SS}``, e.g.
    ``./brain_factory_out/2026-02-15_14-30-00``.
    """
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}/{timestamp}"


def register_datetime_dir_resolver() -> None:
    """Register the ``datetime_dir`` OmegaConf resolver.

    Usage in YAML: ``output_dir: ${datetime_dir:./brain_factory_out}``
    """
    OmegaConf.register_new_resolver("datetime_dir", _datetime_dir, replace=True)
