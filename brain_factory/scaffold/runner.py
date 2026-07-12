from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DebugConfig:
    torch_deterministic: bool = False


@dataclass
class CookSettings:
    seed: int = 42


@dataclass
class RunConfig:
    output_dir: str = "./brain_factory_out"
    experiment_name: str = "default"
    resume_using: str | None = None
    device: str = "gpu"
    num_nodes: int = 1
    num_gpus: int = 1
    num_cores: int = 1
    seed: int = 42
    cook_settings: CookSettings = field(default_factory=CookSettings)
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class ProcessorConfig:
    ddp_enabled: bool = False
    local_rank: int = 0
    global_rank: int = 0
    world_size: int = 1
