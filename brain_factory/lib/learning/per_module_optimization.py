from __future__ import annotations

from typing import Any, Dict, List

import torch


def gather_optimizer_param_groups(
    model: torch.nn.Module,
    learning_rate_config: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Recursively gather all the parameters from the submodules
    and return a List[Dict[str, Any]] that can be passed to the optimizer.

    The learning rate for each group is set by the learning_rate_config
    where key is a dot separated path to the module whose parameters are being optimized and
    value is the learning rate for that group.

    Goal: Recursively construct and gather parameter groups for each sub-module whose
    override is specified in the config. If not specified set the learning rate to
    the default key's value.
    """
    # Validates that default learning rate exists
    if "default" not in learning_rate_config:
        raise ValueError("learning_rate dictionary must contain 'default' key")

    default_lr = learning_rate_config["default"]

    param_groups: List[Dict[str, Any]] = []

    # Start recursion with branch-specific depth calculation
    _recursive_gather_params(
        "", model, param_groups, default_lr, learning_rate_config
    )

    return param_groups


def _recursive_gather_params(
    current_path: str,
    module: torch.nn.Module,
    param_groups: List[Dict[str, Any]],
    default_lr: float,
    learning_rate_config: Dict[str, float],
) -> None:
    """Recursively gather parameters with branch-specific depth optimization."""
    # For each named child module
    for name, child_module in module.named_children():
        if not child_module.training:
            continue

        # Construct dot-separated path
        child_path = f"{current_path}.{name}" if current_path else name

        # Get learning rate for this specific module path
        lr = _get_lr_for_module(child_path, default_lr, learning_rate_config)

        # Check if there are any deeper overrides with different learning rates
        has_lr_conflicts = _has_learning_rate_conflicts(
            child_path, lr, learning_rate_config
        )

        if not has_lr_conflicts:
            # OPTIMIZATION: No deeper overrides with different learning rates
            # Use recursive=True to get all parameters below this level
            direct_params = [
                p for p in child_module.parameters(recurse=True) if p.requires_grad
            ]

            if direct_params:
                param_group = {
                    "params": direct_params,
                    "name": child_path,
                    "lr": lr,
                }
                param_groups.append(param_group)
        else:
            # There are deeper overrides with different learning rates - continue granular recursion

            # Get direct parameters for this module
            direct_params = [
                p for p in child_module.parameters(recurse=False) if p.requires_grad
            ]

            if direct_params:
                param_group = {
                    "params": direct_params,
                    "name": child_path,
                    "lr": lr,
                }
                param_groups.append(param_group)

            # Continue recursing since we have deeper overrides with different LRs
            _recursive_gather_params(
                child_path, child_module, param_groups, default_lr, learning_rate_config
            )


def _has_learning_rate_conflicts(
    current_path: str,
    current_lr: float,
    learning_rate_config: Dict[str, float],
) -> bool:
    """Check if there are any deeper config paths with a different learning rate.

    Simple check: if any deeper config path has a different learning rate
    than current_lr, return True.
    """
    for config_path, config_lr in learning_rate_config.items():
        if config_path.startswith(current_path + "."):
            if config_lr != current_lr:
                return True
    return False


def _get_lr_for_module(
    child_path: str,
    default_lr: float,
    learning_rate_config: Dict[str, float],
) -> float:
    """Get learning rate for a specific module path with hierarchical matching.

    Looks for the most specific config key that matches the module path.
    Walks up the path hierarchy from exact match to progressively shorter prefixes.
    Falls back to default_lr if no match is found.
    """
    # Check for exact match
    if child_path in learning_rate_config:
        return learning_rate_config[child_path]

    # Walk up the path hierarchy to find the most specific ancestor match
    path = child_path
    while "." in path:
        path = path.rsplit(".", 1)[0]
        if path in learning_rate_config:
            return learning_rate_config[path]

    return default_lr
