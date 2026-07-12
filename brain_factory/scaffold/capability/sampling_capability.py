from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

from brain_factory.scaffold.umbrella_model_base import UmbrellaModelBase


class SamplerType(Enum):
    """ODE solver type for sampling."""

    EULER = "euler"
    HEUN = "heun"


@dataclass
class SamplingConfig:
    """Configuration for sampling.

    Attributes:
        num_steps: Number of ODE integration steps.
        sampler: Type of ODE solver.
        guidance_scale: Classifier-free guidance scale (1.0 = no guidance).
    """

    num_steps: int = 50
    sampler: SamplerType = SamplerType.EULER
    guidance_scale: float = 1.0


class SamplingCapability(nn.Module):
    """Wraps a velocity model for sampling via ODE integration.

    Provides a clean API for generating samples from noise using
    rectified flow ODE integration. Supports Euler and Heun solvers.

    The model should be in eval mode when sampling (model.eval()).

    Usage:
        capability = SamplingCapability(model)
        samples = capability.sample(
            noise=torch.randn(B, C, H, W),
            conditioning={"character_image": char_img, "current_frame": frame},
            config=SamplingConfig(num_steps=50),
        )
    """

    def __init__(self, model: UmbrellaModelBase) -> None:
        """Initialize sampling capability.

        Args:
            model: Velocity prediction model. In eval mode, forward() takes
                   Dict[str, Any] with keys "x_t", "t", and conditioning,
                   returns {"velocity": Tensor}.
        """
        super().__init__()
        self._model: UmbrellaModelBase = model

    @property
    def model(self) -> UmbrellaModelBase:
        """The underlying velocity model."""
        return self._model

    def sample(
        self,
        noise: torch.Tensor,
        conditioning: Dict[str, Any],
        config: Optional[SamplingConfig] = None,
        progress_callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        """Generate samples from noise via ODE integration.

        Integrates the ODE dx/dt = v(x, t) from t=1 (noise) to t=0 (data).

        Args:
            noise: Starting noise tensor. Shape: (B, C, H, W)
            conditioning: Dict of conditioning tensors passed to model.
                Keys depend on model (e.g., "character_image", "current_frame").
            config: Sampling configuration. Defaults to SamplingConfig().
            progress_callback: Optional callback called at each step with
                (current_step, total_steps, current_sample).

        Returns:
            Generated samples. Shape: (B, C, H, W)
        """
        if config is None:
            config = SamplingConfig()

        device: torch.device = noise.device
        batch_size: int = noise.shape[0]

        # Time schedule from t=1 to t=0
        schedule: torch.Tensor = torch.linspace(
            1.0, 0.0, config.num_steps + 1, device=device
        )

        x_t: torch.Tensor = noise

        for step_idx in range(config.num_steps):
            t: torch.Tensor = schedule[step_idx].expand(batch_size)
            t_next: torch.Tensor = schedule[step_idx + 1].expand(batch_size)

            if config.sampler == SamplerType.HEUN:
                x_t = self._heun_step(x_t, t, t_next, conditioning, config)
            else:
                x_t = self._euler_step(x_t, t, t_next, conditioning, config)

            if progress_callback is not None:
                progress_callback(step_idx + 1, config.num_steps, x_t)

        return x_t

    def _euler_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        conditioning: Dict[str, Any],
        config: SamplingConfig,
    ) -> torch.Tensor:
        """Single Euler ODE step.

        x_{t+dt} = x_t + dt * v(x_t, t)
        """
        velocity: torch.Tensor = self._get_velocity(x_t, t, conditioning, config)
        dt: torch.Tensor = t_next - t
        dt = self._expand_t(dt, x_t.ndim)
        return x_t + dt * velocity

    def _heun_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        conditioning: Dict[str, Any],
        config: SamplingConfig,
    ) -> torch.Tensor:
        """Heun's method (2nd order) ODE step.

        x_euler = x_t + dt * v(x_t, t)
        x_{t+dt} = x_t + 0.5 * dt * (v(x_t, t) + v(x_euler, t_next))
        """
        v_t: torch.Tensor = self._get_velocity(x_t, t, conditioning, config)

        dt: torch.Tensor = t_next - t
        dt_expanded: torch.Tensor = self._expand_t(dt, x_t.ndim)

        # Euler prediction
        x_euler: torch.Tensor = x_t + dt_expanded * v_t

        # Velocity at Euler point
        v_euler: torch.Tensor = self._get_velocity(x_euler, t_next, conditioning, config)

        # Heun update: average of two velocities
        v_avg: torch.Tensor = 0.5 * (v_t + v_euler)
        return x_t + dt_expanded * v_avg

    def _get_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: Dict[str, Any],
        config: SamplingConfig,
    ) -> torch.Tensor:
        """Get velocity from model, optionally with CFG."""
        batch: Dict[str, Any] = {
            "x_t": x_t,
            "t": t,
            **conditioning,
        }

        with torch.no_grad():
            output: Dict[str, Any] = self._model(batch)
            velocity: torch.Tensor = output["velocity"]

        if config.guidance_scale != 1.0:
            velocity = self._apply_cfg(velocity, x_t, t, conditioning, config)

        return velocity

    def _apply_cfg(
        self,
        velocity_cond: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: Dict[str, Any],
        config: SamplingConfig,
    ) -> torch.Tensor:
        """Apply classifier-free guidance.

        v_guided = v_uncond + scale * (v_cond - v_uncond)
        """
        # Create null conditioning (zeros for tensor values)
        null_conditioning: Dict[str, Any] = {}
        for key, value in conditioning.items():
            if isinstance(value, torch.Tensor):
                null_conditioning[key] = torch.zeros_like(value)
            else:
                null_conditioning[key] = value

        batch_uncond: Dict[str, Any] = {
            "x_t": x_t,
            "t": t,
            **null_conditioning,
        }

        with torch.no_grad():
            output_uncond: Dict[str, Any] = self._model(batch_uncond)
            velocity_uncond: torch.Tensor = output_uncond["velocity"]

        # CFG formula
        return velocity_uncond + config.guidance_scale * (velocity_cond - velocity_uncond)

    def _expand_t(self, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Expand t to broadcast with (B, C, H, W) tensors."""
        while t.ndim < ndim:
            t = t.unsqueeze(-1)
        return t

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass delegates to model for compatibility.

        Note: For sampling, use the sample() method instead.
        """
        return self._model(batch)
