from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from brain_factory.lib.checkpoint import save_checkpoint, save_training_state
from brain_factory.scaffold.loss_gatherer import LossStatistics
from brain_factory.scaffold.output_container_base import OutputContainerBase
from brain_factory.scaffold.runner import ProcessorConfig, RunConfig

logger: logging.Logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE: bool = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore[misc, assignment]


def _build_dummy_batch(model: torch.nn.Module) -> Optional[Dict[str, Any]]:
    """Build a minimal dummy batch for tracing execution order.

    Inspects the model's forward signature and class name to produce
    appropriately shaped dummy tensors. Returns None if unsupported.
    """
    import inspect

    cls_name: str = model.__class__.__name__
    device: torch.device = next(model.parameters()).device if len(list(model.parameters())) > 0 else torch.device("cpu")

    # Check if forward() expects a 'batch' dict (our UmbrellaModelBase pattern)
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())

    if len(params) >= 1 and params[0] == "batch":
        # Try to infer from common patterns
        batch: Dict[str, torch.Tensor] = {}

        # Look at conv_in or first Conv2d for spatial dims
        spatial_h, spatial_w = 32, 32
        in_channels = 3

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d) and name in ("conv_in",):
                in_channels = mod.in_channels
                break

        # Look for known model patterns
        if "FlowUNet" in cls_name or "UNet" in cls_name:
            batch["x_t"] = torch.randn(1, 3, spatial_h, spatial_w, device=device)
            batch["t"] = torch.tensor([0.5], device=device)
            batch["character_image"] = torch.randn(1, 3, spatial_h, spatial_w, device=device)
            batch["current_frame"] = torch.randn(1, 3, spatial_h, spatial_w, device=device)
            # Check for info_proj linear layer to get info_vector dim
            for name, mod in model.named_modules():
                if name == "info_proj" and isinstance(mod, torch.nn.Linear):
                    batch["info_vector"] = torch.randn(1, mod.in_features, device=device)
                    break
            if "info_vector" not in batch:
                batch["info_vector"] = torch.randn(1, 128, device=device)
        elif "Dummy" in cls_name:
            # DummyUmbrellaModel expects "input"
            for name, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    batch["input"] = torch.randn(1, mod.in_features, device=device)
                    break
        else:
            return None

        return batch

    return None


class Monitor:
    """Concrete training monitor that handles logging, TensorBoard, checkpoints,
    and output container orchestration.

    Lifecycle (managed by TrainingRecipeBase):
        1. Hydra instantiates the Monitor via ``_target_`` in config.
        2. ``prepare()`` calls ``set_unresolved_recipe_config()`` and ``set_tb_writer()``.
        3. The training loop calls ``update_train_progress()`` each step.
        4. ``close_kitchen()`` calls ``close()`` for cleanup.

    Interval gating: each interval parameter controls how often that action fires.
    A value of 0 disables the action. Otherwise, the action fires when
    ``step % interval == 0``.
    """

    def __init__(
        self,
        containers: List[OutputContainerBase] | None = None,
        visualization_folder_name: str = "visualizations",
        log_loss_interval: int = 0,
        tb_loss_interval: int = 0,
        output_save_interval: int = 0,
        checkpoint_interval: int = 0,
    ) -> None:
        self._containers: List[OutputContainerBase] = containers or []
        self._visualization_folder_name: str = visualization_folder_name
        self._log_loss_interval: int = log_loss_interval
        self._tb_loss_interval: int = tb_loss_interval
        self._output_save_interval: int = output_save_interval
        self._checkpoint_interval: int = checkpoint_interval
        self._tb_writer: Optional[SummaryWriter] = None  # type: ignore[type-arg]
        self._raw_recipe_config: Optional[DictConfig] = None
        self._best_loss: float = float("inf")
        self._metrics_log_path: Optional[str] = None

    # -----------------------------------------------------------------------
    # Interval gating
    # -----------------------------------------------------------------------

    def should_log_loss(self, step: int) -> bool:
        """Check if loss should be logged at this step."""
        return self._log_loss_interval > 0 and step % self._log_loss_interval == 0

    def should_log_to_tb(self, step: int) -> bool:
        """Check if metrics should be logged to TensorBoard at this step."""
        return self._tb_loss_interval > 0 and step % self._tb_loss_interval == 0

    def should_save_monitor(self, step: int) -> bool:
        """Check if monitor should save outputs at this step."""
        return self._output_save_interval > 0 and step % self._output_save_interval == 0

    def should_save_checkpoint(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step."""
        if self._checkpoint_interval <= 0:
            return False
        return step == 0 or step % self._checkpoint_interval == 0

    # -----------------------------------------------------------------------
    # Config storage
    # -----------------------------------------------------------------------

    def set_unresolved_recipe_config(self, raw_recipe_config: DictConfig) -> None:
        """Store the raw (unresolved) recipe config for logging/reproducibility.

        Called automatically by TrainingRecipeBase.prepare().
        """
        self._raw_recipe_config = raw_recipe_config

    # -----------------------------------------------------------------------
    # TensorBoard management
    # -----------------------------------------------------------------------

    def get_tb_writer(self) -> Optional[SummaryWriter]:  # type: ignore[type-arg]
        """Return the TensorBoard writer, or None if not initialized."""
        return self._tb_writer

    def set_tb_writer(self, out_dir: str, rank: int = 0) -> None:
        """Initialize the TensorBoard SummaryWriter and metrics log.

        Args:
            out_dir: Base output directory for this experiment.
            rank: Global rank for distributed training. Rank > 0 gets
                  a rank-specific subdirectory.
        """
        # Metrics log (always available, Claude-friendly)
        self._metrics_log_path = os.path.join(out_dir, "metrics.jsonl")
        os.makedirs(out_dir, exist_ok=True)

        if not _TENSORBOARD_AVAILABLE:
            logger.warning(
                "TensorBoard not available. Install tensorboard to enable TB logging. "
                "Metrics will still be written to metrics.jsonl."
            )
            return

        tb_dir: str
        if rank > 0:
            tb_dir = os.path.join(out_dir, "tb", f"rank_{rank}")
        else:
            tb_dir = os.path.join(out_dir, "tb")
        self._tb_writer = SummaryWriter(log_dir=tb_dir)
        logger.info(f"TensorBoard writer initialized at {tb_dir}")

    def flush_tb_writer(self) -> None:
        """Flush the TensorBoard writer to disk."""
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def add_scalar_to_tb_writer(self, tag: str, value: float, step: int) -> None:
        """Write a single scalar to TensorBoard."""
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)

    def log_optimizer_state_to_tb_writer(
        self, optimizer: torch.optim.Optimizer, step: int
    ) -> None:
        """Log learning rate per param group to TensorBoard."""
        for i, group in enumerate(optimizer.param_groups):
            group_name: str = group.get("name", f"group_{i}")
            self.add_scalar_to_tb_writer(f"lr/{group_name}", group["lr"], step)

    # -----------------------------------------------------------------------
    # Metrics log (Claude-friendly JSON Lines)
    # -----------------------------------------------------------------------

    def _append_metrics_log(self, scalars: Dict[str, float], step: int) -> None:
        """Append a JSON line to metrics.jsonl with all scalars for this step."""
        if self._metrics_log_path is None:
            return
        entry: Dict[str, Any] = {"step": step, **scalars}
        with open(self._metrics_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # -----------------------------------------------------------------------
    # Loss logging
    # -----------------------------------------------------------------------

    def log_loss_breakdown(self, loss_stats: LossStatistics, step: int) -> None:
        """Log loss breakdown to the Python logger."""
        if not self.should_log_loss(step):
            return

        parts: List[str] = [
            f"Step {step} | total_loss: {loss_stats.total_loss.item():.6f}"
        ]
        for name, value in loss_stats.loss_breakdown.items():
            unweighted: float = loss_stats.unweighted_loss_breakdown[name].item()
            weighted: float = value.item()
            parts.append(f"  {name}: {weighted:.6f} (unweighted: {unweighted:.6f})")
        logger.info("\n".join(parts))

    def log_to_tensorboard(
        self,
        loss_stats: LossStatistics,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> None:
        """Write loss scalars and optimizer LR to TensorBoard and metrics.jsonl."""
        if not self.should_log_to_tb(step):
            return

        scalars: Dict[str, float] = {}

        # Loss terms
        scalars["loss/total"] = loss_stats.total_loss.item()
        for name, value in loss_stats.loss_breakdown.items():
            scalars[f"loss/{name}"] = value.item()
        for name, value in loss_stats.unweighted_loss_breakdown.items():
            scalars[f"loss_unweighted/{name}"] = value.item()

        # Optimizer LR
        for i, group in enumerate(optimizer.param_groups):
            group_name: str = group.get("name", f"group_{i}")
            scalars[f"lr/{group_name}"] = group["lr"]

        # Write to TensorBoard
        for tag, value in scalars.items():
            self.add_scalar_to_tb_writer(tag, value, step)
        self.flush_tb_writer()

        # Write to metrics.jsonl
        self._append_metrics_log(scalars, step)

    # -----------------------------------------------------------------------
    # Checkpoint saving
    # -----------------------------------------------------------------------

    def save_training_checkpoint(
        self,
        model: torch.nn.Module,
        run_config: RunConfig,
        processor_config: ProcessorConfig,
        step: int,
        checkpoint_name: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        total_loss: Optional[float] = None,
        samples_processed: int = 0,
    ) -> None:
        """Save a training checkpoint with model, optimizer, scheduler, and config.

        Args:
            model: The model to checkpoint.
            run_config: Run configuration (for output paths).
            processor_config: Processor configuration (for rank gating).
            step: Current training step.
            checkpoint_name: Custom name for the checkpoint directory.
                Defaults to ``step_{step}``.
            optimizer: Optional optimizer to save state_dict.
            scheduler: Optional scheduler to save state_dict.
            total_loss: Optional current loss value (for best-loss tracking).
            samples_processed: Number of samples processed so far.
        """
        if not self.should_save_checkpoint(step):
            return

        name: str = checkpoint_name or f"step_{step}"
        checkpoint_dir: str = os.path.join(
            run_config.output_dir, "checkpoints", name
        )

        # Model checkpoint via lib/checkpoint.py
        metadata: Dict[str, str] = {
            "step": str(step),
            "samples_processed": str(samples_processed),
            "experiment_name": run_config.experiment_name,
        }
        if total_loss is not None:
            metadata["total_loss"] = str(total_loss)
            if total_loss < self._best_loss:
                self._best_loss = total_loss
                metadata["is_best"] = "true"
        save_checkpoint(model, checkpoint_dir, metadata=metadata)

        # Optimizer + scheduler state
        if optimizer is not None or scheduler is not None:
            save_training_state(checkpoint_dir, optimizer=optimizer, scheduler=scheduler)

        # Save raw recipe config for reproducibility
        if self._raw_recipe_config is not None:
            config_path: str = os.path.join(checkpoint_dir, "recipe_config.yaml")
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(self._raw_recipe_config))

        logger.info(f"Training checkpoint saved: {checkpoint_dir}")

    # -----------------------------------------------------------------------
    # Output container saving
    # -----------------------------------------------------------------------

    def save_outputs(
        self,
        run_config: RunConfig,
        processor_config: ProcessorConfig,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        step: int,
    ) -> None:
        """Save outputs from all containers at the given step."""
        if not self.should_save_monitor(step):
            return

        output_dir: str = os.path.join(
            run_config.output_dir,
            self._visualization_folder_name,
            f"step_{step}",
        )
        os.makedirs(output_dir, exist_ok=True)
        device: torch.device = torch.device("cpu")
        for container in self._containers:
            container.save(predictions, targets, output_dir, device)

    # -----------------------------------------------------------------------
    # High-level progress methods (main entry points for training loops)
    # -----------------------------------------------------------------------

    def update_train_progress(
        self,
        step: int,
        loss_stats: LossStatistics,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        model: torch.nn.Module,
        run_config: RunConfig,
        processor_config: ProcessorConfig,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        samples_processed: int = 0,
    ) -> None:
        """Handle all periodic logging operations during training.

        This is the single call site from the training loop. Internally
        dispatches to loss logging, TensorBoard, checkpointing, and output
        saving based on interval configuration.
        """
        self.log_loss_breakdown(loss_stats, step)
        self.log_to_tensorboard(loss_stats, optimizer, step)
        self.save_training_checkpoint(
            model=model,
            run_config=run_config,
            processor_config=processor_config,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            total_loss=loss_stats.total_loss.item(),
            samples_processed=samples_processed,
        )
        self.save_outputs(run_config, processor_config, predictions, targets, step)

    def update_val_progress(
        self,
        step: int,
        loss_stats: LossStatistics,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        run_config: RunConfig,
        processor_config: ProcessorConfig,
    ) -> None:
        """Handle periodic logging operations during validation.

        Logs with a ``val/`` prefix and saves validation outputs.
        """
        # Log validation loss
        parts: List[str] = [
            f"[Validation] Step {step} | total_loss: {loss_stats.total_loss.item():.6f}"
        ]
        for name, value in loss_stats.loss_breakdown.items():
            parts.append(f"  {name}: {value.item():.6f}")
        logger.info("\n".join(parts))

        # Write val metrics to TB and metrics.jsonl
        scalars: Dict[str, float] = {
            "val/loss/total": loss_stats.total_loss.item(),
        }
        for name, value in loss_stats.loss_breakdown.items():
            scalars[f"val/loss/{name}"] = value.item()

        for tag, val in scalars.items():
            self.add_scalar_to_tb_writer(tag, val, step)
        self.flush_tb_writer()
        self._append_metrics_log(scalars, step)

        # Save validation outputs
        val_output_dir: str = os.path.join(
            run_config.output_dir,
            self._visualization_folder_name,
            f"val_step_{step}",
        )
        os.makedirs(val_output_dir, exist_ok=True)
        device: torch.device = torch.device("cpu")
        for container in self._containers:
            container.save(predictions, targets, val_output_dir, device)

    def update_inference_progress(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        run_config: RunConfig,
        processor_config: ProcessorConfig,
        step: int | None = None,
        loss_stats: Optional[LossStatistics] = None,
    ) -> None:
        """Handle periodic operations during inference."""
        if loss_stats is not None and step is not None:
            logger.info(
                f"[Inference] Step {step} | total_loss: {loss_stats.total_loss.item():.6f}"
            )

        path_parts = [
            run_config.output_dir,
            self._visualization_folder_name,
        ]
        if step is not None:
            path_parts.append(f"inference_step_{step}")
        inf_output_dir: str = os.path.join(*path_parts)
        os.makedirs(inf_output_dir, exist_ok=True)
        device: torch.device = torch.device("cpu")
        for container in self._containers:
            container.save(predictions, targets, inf_output_dir, device)

    # -----------------------------------------------------------------------
    # Model graph export
    # -----------------------------------------------------------------------

    def export_model_graph(
        self, model: torch.nn.Module, output_dir: str
    ) -> None:
        """Export the model's module hierarchy and data flow as JSON.

        Walks ``model.named_modules()`` to build a recursive tree of modules
        with parameter counts and shape descriptions, then records forward-pass
        execution order via hooks to produce flow edges.

        The result is written to ``<output_dir>/architecture.json``.
        """
        graph_path: str = os.path.join(output_dir, "architecture.json")

        modules_tree = self._build_module_tree(model)
        flow_edges = self._trace_flow(model)

        graph: Dict[str, Any] = {
            "model_name": model.__class__.__name__,
            "modules": modules_tree,
            "flow": flow_edges,
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(graph_path, "w") as f:
            json.dump(graph, f, indent=2)
        logger.info(f"Model architecture graph exported to {graph_path}")

    @staticmethod
    def _get_shape_desc(module: torch.nn.Module) -> Optional[str]:
        """Build a human-readable shape description for a leaf module."""
        cls_name: str = module.__class__.__name__

        if isinstance(module, torch.nn.Linear):
            return f"Linear({module.in_features}\u2192{module.out_features})"
        elif isinstance(module, torch.nn.Conv2d):
            k = module.kernel_size
            ks = f"{k[0]}\u00d7{k[1]}" if isinstance(k, tuple) else str(k)
            return f"Conv2d({module.in_channels}\u2192{module.out_channels}, {ks})"
        elif isinstance(module, torch.nn.Conv1d):
            k = module.kernel_size
            ks = str(k[0]) if isinstance(k, tuple) else str(k)
            return f"Conv1d({module.in_channels}\u2192{module.out_channels}, k={ks})"
        elif isinstance(module, torch.nn.ConvTranspose2d):
            k = module.kernel_size
            ks = f"{k[0]}\u00d7{k[1]}" if isinstance(k, tuple) else str(k)
            return f"ConvTranspose2d({module.in_channels}\u2192{module.out_channels}, {ks})"
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            return f"{cls_name}({module.num_features})"
        elif isinstance(module, torch.nn.LayerNorm):
            return f"LayerNorm({list(module.normalized_shape)})"
        elif isinstance(module, torch.nn.GroupNorm):
            return f"GroupNorm({module.num_groups}, {module.num_channels})"
        elif isinstance(module, torch.nn.Embedding):
            return f"Embedding({module.num_embeddings}, {module.embedding_dim})"
        elif isinstance(module, torch.nn.MultiheadAttention):
            return f"MultiheadAttention(d={module.embed_dim}, heads={module.num_heads})"
        elif isinstance(
            module,
            (
                torch.nn.MaxPool2d,
                torch.nn.AvgPool2d,
                torch.nn.AdaptiveAvgPool2d,
            ),
        ):
            return f"{cls_name}({getattr(module, 'kernel_size', '?')})"
        elif isinstance(module, torch.nn.Dropout):
            return f"Dropout(p={module.p})"

        return None

    @staticmethod
    def _param_count(module: torch.nn.Module) -> int:
        """Count parameters owned directly by this module (not children)."""
        child_params = set()
        for child in module.children():
            for p in child.parameters():
                child_params.add(id(p))
        return sum(
            p.numel()
            for p in module.parameters()
            if id(p) not in child_params
        )

    def _build_module_tree(
        self, module: torch.nn.Module, name: str = ""
    ) -> Dict[str, Any]:
        """Recursively build the module tree."""
        children: List[Dict[str, Any]] = []
        for child_name, child_module in module.named_children():
            children.append(self._build_module_tree(child_module, child_name))

        total_params: int = sum(p.numel() for p in module.parameters())
        own_params: int = self._param_count(module)

        return {
            "name": name,
            "type": module.__class__.__name__,
            "params": total_params,
            "own_params": own_params,
            "shape_desc": self._get_shape_desc(module),
            "children": children,
        }

    @staticmethod
    def _trace_flow(model: torch.nn.Module) -> List[Dict[str, Any]]:
        """Trace execution order by recording which leaf modules fire.

        Registers forward hooks on all leaf modules (those with no children),
        runs a dummy forward to capture ordering, then builds flow edges.
        """
        execution_order: List[Dict[str, Any]] = []
        hooks: List[torch.utils.hooks.RemovableHook] = []

        # Build name map for leaf modules
        leaf_names: Dict[int, str] = {}
        for full_name, mod in model.named_modules():
            if len(list(mod.children())) == 0:
                leaf_names[id(mod)] = full_name

        def make_hook(mod_name: str):  # type: ignore[no-untyped-def]
            def hook(
                module: torch.nn.Module,
                input: Any,
                output: Any,
            ) -> None:
                out_shape: Optional[List[int]] = None
                if isinstance(output, torch.Tensor):
                    out_shape = list(output.shape)
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    first = output[0]
                    if isinstance(first, torch.Tensor):
                        out_shape = list(first.shape)
                execution_order.append(
                    {"name": mod_name, "output_shape": out_shape}
                )

            return hook

        for full_name, mod in model.named_modules():
            if len(list(mod.children())) == 0:
                hooks.append(mod.register_forward_hook(make_hook(full_name)))

        # Try a dummy forward pass to capture execution order
        try:
            model.eval()
            with torch.no_grad():
                dummy_batch = _build_dummy_batch(model)
                if dummy_batch is not None:
                    model(dummy_batch)
        except Exception as e:
            logger.warning(f"Could not trace model flow: {e}")
        finally:
            for h in hooks:
                h.remove()

        # Build edges: consecutive leaf modules form an edge
        flow: List[Dict[str, Any]] = []
        for i in range(1, len(execution_order)):
            flow.append(
                {
                    "from": execution_order[i - 1]["name"],
                    "to": execution_order[i]["name"],
                    "shape": execution_order[i - 1].get("output_shape"),
                }
            )
        return flow

    # -----------------------------------------------------------------------
    # Device and lifecycle
    # -----------------------------------------------------------------------

    def to(self, device: torch.device) -> Monitor:
        """Move all container modules to the specified device."""
        for container in self._containers:
            container.to(device)
        return self

    def close(self) -> None:
        """Shutdown: cleanup containers and close TensorBoard writer."""
        for container in self._containers:
            container.cleanup()
        self.flush_tb_writer()
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        logger.info("Monitor shutdown complete.")
