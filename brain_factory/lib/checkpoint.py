from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_MAX_SHARD_SIZE_BYTES: int = 5 * 1024 * 1024 * 1024  # 5 GB


@dataclass
class CheckpointMapping:
    """Maps a region of a checkpoint file to a region of a model's state_dict.

    Args:
        path: Path to a .safetensors file, .safetensors.index.json, or directory.
        source_prefix: Prefix to select from checkpoint keys (e.g. "model.backbone.encoder.").
        target_prefix: Prefix to map into in the model's state_dict (e.g. "encoder.").
        strict: If True, all model keys under target_prefix must be found in the checkpoint.
    """

    path: str
    source_prefix: str = ""
    target_prefix: str = ""
    strict: bool = True


@dataclass
class CheckpointConfig:
    """Configuration for loading model checkpoints.

    Args:
        path: Base model checkpoint path (full state_dict load, applied first).
        mappings: Overlay mappings applied in order after the base load. Last wins on conflicts.
    """

    path: Optional[str] = None
    mappings: List[CheckpointMapping] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is None and len(self.mappings) == 0:
            raise ValueError("CheckpointConfig must specify either path or mappings (or both).")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_safetensors_path(path: str) -> Tuple[str, Dict[str, str]]:
    """Resolve a checkpoint path to a mapping of tensor keys to shard file paths.

    Handles three cases:
    - Single .safetensors file
    - Sharded .safetensors.index.json
    - Directory (auto-detects single vs sharded)

    Returns:
        Tuple of (base_dir, key_to_shard_path) where key_to_shard_path maps
        tensor names to absolute file paths.
    """
    if os.path.isdir(path):
        index_path: str = os.path.join(path, "model.safetensors.index.json")
        single_path: str = os.path.join(path, "model.safetensors")
        if os.path.exists(index_path):
            return _resolve_sharded_index(index_path)
        elif os.path.exists(single_path):
            return _resolve_single_file(single_path)
        else:
            raise FileNotFoundError(
                f"Directory {path} contains neither model.safetensors.index.json "
                f"nor model.safetensors."
            )
    elif path.endswith(".safetensors.index.json"):
        return _resolve_sharded_index(path)
    elif path.endswith(".safetensors"):
        return _resolve_single_file(path)
    else:
        raise ValueError(
            f"Unsupported checkpoint path: {path}. "
            f"Must be a .safetensors file, .safetensors.index.json, or directory."
        )


def _resolve_single_file(path: str) -> Tuple[str, Dict[str, str]]:
    """Resolve a single .safetensors file to a key->path mapping.

    Only reads the file header (key index), not tensor data.
    """
    abs_path: str = os.path.abspath(path)
    with safetensors.torch.safe_open(abs_path, framework="pt") as f:
        keys: List[str] = list(f.keys())
    key_to_path: Dict[str, str] = {key: abs_path for key in keys}
    return os.path.dirname(abs_path), key_to_path


def _resolve_sharded_index(index_path: str) -> Tuple[str, Dict[str, str]]:
    """Resolve a sharded safetensors index to a key->shard path mapping."""
    abs_index: str = os.path.abspath(index_path)
    base_dir: str = os.path.dirname(abs_index)

    with open(abs_index, "r") as f:
        index_data: dict = json.load(f)

    weight_map: Dict[str, str] = index_data["weight_map"]
    key_to_path: Dict[str, str] = {
        key: os.path.join(base_dir, shard_file) for key, shard_file in weight_map.items()
    }
    return base_dir, key_to_path


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _apply_mapping(
    key_to_shard: Dict[str, str],
    source_prefix: str,
    target_prefix: str,
) -> Dict[str, Tuple[str, str]]:
    """Filter keys by source_prefix and remap to target_prefix.

    Returns:
        Dict mapping target_key -> (shard_path, source_key).
    """
    resolved: Dict[str, Tuple[str, str]] = {}
    for source_key, shard_path in key_to_shard.items():
        if source_prefix == "" or source_key.startswith(source_prefix):
            suffix: str = source_key[len(source_prefix):]
            target_key: str = target_prefix + suffix
            resolved[target_key] = (shard_path, source_key)
    return resolved


def _load_tensors_by_shard(
    resolved: Dict[str, Tuple[str, str]],
) -> Dict[str, torch.Tensor]:
    """Load tensors grouped by shard file for efficient I/O.

    Opens each shard file once and loads only the needed tensors.
    """
    shard_to_keys: Dict[str, List[Tuple[str, str]]] = {}
    for target_key, (shard_path, source_key) in resolved.items():
        if shard_path not in shard_to_keys:
            shard_to_keys[shard_path] = []
        shard_to_keys[shard_path].append((target_key, source_key))

    state_dict: Dict[str, torch.Tensor] = {}
    for shard_path, key_pairs in shard_to_keys.items():
        with safetensors.torch.safe_open(shard_path, framework="pt") as f:
            for target_key, source_key in key_pairs:
                state_dict[target_key] = f.get_tensor(source_key)

    return state_dict


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_checkpoint(config: CheckpointConfig, model: torch.nn.Module) -> None:
    """Load checkpoint(s) into a model according to the config.

    Algorithm:
    1. If config.path is set, resolve all keys as a full model load (no prefix remapping).
    2. For each mapping in config.mappings, resolve with prefix remapping.
       Last wins on key conflicts.
    3. Validate strict mappings against model state_dict.
    4. Group by shard file, load only needed tensors.
    5. Apply via model.load_state_dict(strict=False).
    """
    model_keys: set[str] = set(model.state_dict().keys())
    resolved: Dict[str, Tuple[str, str]] = {}

    # Step 1: Base model load
    if config.path is not None:
        logger.debug(f"Resolving base checkpoint: {config.path}")
        _, key_to_shard = _resolve_safetensors_path(config.path)
        base_resolved: Dict[str, Tuple[str, str]] = _apply_mapping(
            key_to_shard, source_prefix="", target_prefix=""
        )
        resolved.update(base_resolved)
        logger.debug(f"Base checkpoint: {len(base_resolved)} keys resolved.")

    # Step 2: Overlay mappings
    for i, mapping in enumerate(config.mappings):
        logger.debug(
            f"Resolving mapping [{i}]: {mapping.path} "
            f"(source_prefix={mapping.source_prefix!r} -> target_prefix={mapping.target_prefix!r})"
        )
        _, key_to_shard = _resolve_safetensors_path(mapping.path)
        mapping_resolved: Dict[str, Tuple[str, str]] = _apply_mapping(
            key_to_shard, mapping.source_prefix, mapping.target_prefix
        )

        # Strict validation: all model keys under target_prefix must be found
        if mapping.strict:
            target_model_keys: set[str] = {
                k for k in model_keys if k.startswith(mapping.target_prefix)
            }
            missing: set[str] = target_model_keys - set(mapping_resolved.keys())
            if missing:
                raise ValueError(
                    f"Strict checkpoint mapping [{i}] ({mapping.path}) is missing "
                    f"{len(missing)} keys under target_prefix={mapping.target_prefix!r}: "
                    f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
                )

        overwritten: int = len(set(mapping_resolved.keys()) & set(resolved.keys()))
        if overwritten > 0:
            logger.debug(f"Mapping [{i}] overwrites {overwritten} keys from previous mappings.")
        resolved.update(mapping_resolved)
        logger.debug(f"Mapping [{i}]: {len(mapping_resolved)} keys resolved.")

    # Step 3: Warn about uncovered model keys
    resolved_keys: set[str] = set(resolved.keys())
    uncovered: set[str] = model_keys - resolved_keys
    if uncovered:
        logger.warning(
            f"{len(uncovered)} model keys not covered by any checkpoint mapping "
            f"(will stay randomly initialized): {sorted(uncovered)[:10]}"
            f"{'...' if len(uncovered) > 10 else ''}"
        )

    # Filter to only keys that exist in the model
    extra_keys: set[str] = resolved_keys - model_keys
    if extra_keys:
        logger.warning(
            f"{len(extra_keys)} checkpoint keys do not match any model parameter "
            f"(will be ignored): {sorted(extra_keys)[:10]}"
            f"{'...' if len(extra_keys) > 10 else ''}"
        )
        resolved = {k: v for k, v in resolved.items() if k in model_keys}

    # Step 4: Load tensors grouped by shard
    state_dict: Dict[str, torch.Tensor] = _load_tensors_by_shard(resolved)

    # Step 5: Apply to model
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Checkpoint loaded: {len(state_dict)} tensors applied to model.")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def _tensor_byte_size(tensor: torch.Tensor) -> int:
    """Return the size of a tensor in bytes."""
    return tensor.nelement() * tensor.element_size()


def _assign_tensors_to_shards(
    state_dict: Dict[str, torch.Tensor],
    max_shard_size_bytes: int,
) -> List[Dict[str, torch.Tensor]]:
    """Split a state dict into shards that each fit within max_shard_size_bytes.

    Tensors are assigned greedily in iteration order. A tensor that exceeds
    max_shard_size_bytes on its own gets its own shard.
    """
    shards: List[Dict[str, torch.Tensor]] = []
    current_shard: Dict[str, torch.Tensor] = {}
    current_size: int = 0

    for key, tensor in state_dict.items():
        tensor_size: int = _tensor_byte_size(tensor)
        if current_size + tensor_size > max_shard_size_bytes and len(current_shard) > 0:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size

    if len(current_shard) > 0:
        shards.append(current_shard)

    return shards


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: str,
    metadata: Optional[Dict[str, str]] = None,
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
) -> str:
    """Save a model checkpoint in safetensors format.

    Saves as a single file if the model fits within max_shard_size_bytes,
    otherwise shards across multiple files with an index.

    Args:
        model: The model to save.
        output_dir: Directory to write checkpoint files into.
        metadata: Optional metadata dict (str->str) stored in safetensors header.
        max_shard_size_bytes: Max bytes per shard file. Defaults to 5 GB.

    Returns:
        Path to the saved checkpoint (single file or index json).
    """
    os.makedirs(output_dir, exist_ok=True)
    state_dict: Dict[str, torch.Tensor] = model.state_dict()

    total_size: int = sum(_tensor_byte_size(t) for t in state_dict.values())
    logger.debug(f"Saving checkpoint: {len(state_dict)} tensors, {total_size / 1e9:.2f} GB total.")

    # Single file case
    if total_size <= max_shard_size_bytes:
        output_path: str = os.path.join(output_dir, "model.safetensors")
        safetensors.torch.save_file(state_dict, output_path, metadata=metadata)
        logger.info(f"Checkpoint saved: {output_path}")
        return output_path

    # Sharded case
    shards: List[Dict[str, torch.Tensor]] = _assign_tensors_to_shards(
        state_dict, max_shard_size_bytes
    )
    num_shards: int = len(shards)
    weight_map: Dict[str, str] = {}

    for shard_idx, shard_dict in enumerate(shards):
        shard_name: str = f"model-{shard_idx + 1:05d}-of-{num_shards:05d}.safetensors"
        shard_path: str = os.path.join(output_dir, shard_name)
        shard_metadata: Optional[Dict[str, str]] = metadata if shard_idx == 0 else None
        safetensors.torch.save_file(shard_dict, shard_path, metadata=shard_metadata)
        for key in shard_dict:
            weight_map[key] = shard_name
        logger.debug(f"Saved shard {shard_idx + 1}/{num_shards}: {shard_name}")

    index: Dict[str, object] = {
        "metadata": metadata or {},
        "weight_map": weight_map,
    }
    index_path: str = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Sharded checkpoint saved: {num_shards} shards in {output_dir}")
    return index_path


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def read_checkpoint_metadata(path: str) -> Dict[str, str]:
    """Read metadata from a checkpoint file, index, or directory.

    Handles the same path formats as load_checkpoint:
    - Single .safetensors file: reads header metadata.
    - .safetensors.index.json: reads the "metadata" field from the index JSON.
    - Directory: auto-detects single vs sharded and reads accordingly.

    Returns:
        Metadata dict (empty dict if no metadata is stored).
    """
    if os.path.isdir(path):
        index_path: str = os.path.join(path, "model.safetensors.index.json")
        single_path: str = os.path.join(path, "model.safetensors")
        if os.path.exists(index_path):
            return _read_index_metadata(index_path)
        elif os.path.exists(single_path):
            return _read_safetensors_metadata(single_path)
        else:
            raise FileNotFoundError(
                f"Directory {path} contains neither model.safetensors.index.json "
                f"nor model.safetensors."
            )
    elif path.endswith(".safetensors.index.json"):
        return _read_index_metadata(path)
    elif path.endswith(".safetensors"):
        return _read_safetensors_metadata(path)
    else:
        raise ValueError(
            f"Unsupported checkpoint path: {path}. "
            f"Must be a .safetensors file, .safetensors.index.json, or directory."
        )


def _read_safetensors_metadata(path: str) -> Dict[str, str]:
    """Read metadata from a single .safetensors file header."""
    with safetensors.torch.safe_open(path, framework="pt") as f:
        metadata: Optional[Dict[str, str]] = f.metadata()
    return metadata or {}


def _read_index_metadata(index_path: str) -> Dict[str, str]:
    """Read metadata from a sharded index JSON file."""
    with open(index_path, "r") as f:
        index_data: dict = json.load(f)
    return index_data.get("metadata", {})


TRAINING_STATE_FILENAME: str = "training_state.pt"


# ---------------------------------------------------------------------------
# Training state (optimizer + scheduler)
# ---------------------------------------------------------------------------


def save_training_state(
    output_dir: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> str:
    """Save optimizer and scheduler state dicts to a single .pt file.

    Args:
        output_dir: Directory to write the training_state.pt file into.
        optimizer: Optional optimizer whose state_dict will be saved.
        scheduler: Optional LR scheduler whose state_dict will be saved.

    Returns:
        Path to the saved training_state.pt file.
    """
    os.makedirs(output_dir, exist_ok=True)
    state: Dict[str, Any] = {}

    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()

    output_path: str = os.path.join(output_dir, TRAINING_STATE_FILENAME)
    torch.save(state, output_path)
    logger.info(f"Training state saved: {output_path}")
    return output_path


def load_training_state(
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Dict[str, Any]:
    """Load optimizer and scheduler state dicts from a training_state.pt file.

    Args:
        path: Path to a .pt file or a directory containing training_state.pt.
        optimizer: Optional optimizer to load state into via load_state_dict.
        scheduler: Optional LR scheduler to load state into via load_state_dict.

    Returns:
        The full loaded dict (callers can inspect keys like "optimizer", "scheduler").
    """
    if os.path.isdir(path):
        path = os.path.join(path, TRAINING_STATE_FILENAME)

    state: Dict[str, Any] = torch.load(path, weights_only=True)

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        logger.debug("Optimizer state loaded.")
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
        logger.debug("Scheduler state loaded.")

    logger.info(f"Training state loaded: {path}")
    return state
