"""Abstract base class for diffusion noise schedules.

All concrete noise schedules in brain_factory/noise_schedule/ inherit from
NoiseScheduleBase and implement its three abstract methods.

Contract
--------
add_noise(x_0, noise, timestep) → x_t
    Forward (noising) process. Produces the noisy sample that the model sees
    during training.  The semantics of `timestep` are schedule-specific:
    an integer bucket index for DDPM, a continuous float in [0,1] for
    rectified flow.

get_target(x_0, noise, timestep) → target
    Returns what the model is trained to predict. Three common conventions:
    - epsilon:      the raw noise (DDPM default)
    - v_prediction: sqrt(α_t)*noise − sqrt(1−α_t)*x_0  (DDPM cosine, DPM-Solver)
    - velocity:     x_0 − noise  (rectified flow — constant along the path)

step(model_output, x_t, timestep, **kwargs) → x_{t-1 / t+dt}
    Single inference denoising (or integration) step.  For DDPM this is a
    stochastic reverse step; for rectified flow it is an Euler integration
    step; for DPM-Solver it is a high-order multistep update.

Why a base class?
-----------------
- Enforces a consistent training-loop interface so recipes can swap schedules.
- Makes it easy to register new schedules (inherit, implement three methods).
- Does NOT force a common config type — each schedule manages its own
  hyperparameters because the parameter spaces are incompatible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class NoiseScheduleBase(ABC):
    """Abstract interface for diffusion / flow-matching noise schedules.

    Pure math objects — no nn.Module, no learned parameters.
    Subclasses must implement add_noise, get_target, and step.
    """

    @abstractmethod
    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process: corrupt clean data with noise.

        Args:
            x_0:      [B, ...] clean data samples.
            noise:    [B, ...] sampled noise (same shape as x_0).
            timestep: [B] timestep indices or values (schedule-specific type).
        Returns:
            [B, ...] noisy samples x_t.
        """
        ...

    @abstractmethod
    def get_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the regression target for the model at this timestep.

        For v_prediction schedules: sqrt(α_t)*noise − sqrt(1−α_t)*x_0
        For rectified flow:         x_0 − noise  (independent of timestep)
        For epsilon prediction:     noise

        Args:
            x_0:      [B, ...] clean data.
            noise:    [B, ...] sampled noise.
            timestep: [B] timestep indices or values.
        Returns:
            [B, ...] regression target tensor.
        """
        ...

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Single inference denoising or integration step.

        Args:
            model_output: [B, ...] model's prediction (velocity, epsilon, etc.).
            x_t:          [B, ...] current noisy sample.
            timestep:     [B] or scalar current timestep.
            **kwargs:     Schedule-specific extras (e.g. noise for SDE variants,
                          dt for Euler integration).
        Returns:
            [B, ...] updated sample.
        """
        ...
