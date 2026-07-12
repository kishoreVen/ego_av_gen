"""Rectified flow noise schedule.

Pure math utilities — no nn.Module, no learned parameters.

Rectified flow (Liu et al., 2022 "Flow Straight and Fast") uses straight-line
ODE paths between the noise distribution and the data distribution:

    x_t = (1 - t) * noise + t * x_0          t ∈ [0, 1]

The velocity target along each path is constant:

    v* = dx_t / dt = x_0 - noise

The model learns v(x_t, t) ≈ v*, then at inference we integrate from t=0
(pure noise) to t=1 (clean data) via Euler steps.

Contrast with DDPM: paths are curved (cosine schedule), direction is reversed
(T→0), target is epsilon or v_prediction, and t is an integer bucket index.

GR00T-specific additions:
    - Beta-distributed time sampling (skewed toward t≈1 where signal dominates)
    - A noise_s ceiling that keeps t slightly below 1.0 for numerical stability
    - Discretisation: continuous t is mapped to an integer bucket for the DiT's
      timestep encoder (which expects integer indices, not floats)

Reference: "Flow Straight and Fast: Learning to Generate and Transfer Data with
Rectified Flow" (Liu et al., 2022). GR00T implementation: NVIDIA Isaac-GR00T
gr00t/model/gr00t_n1d6/gr00t_n1d6.py (Gr00tN1d6ActionHead).
"""
from __future__ import annotations

import torch
from torch.distributions import Beta

from brain_factory.scaffold.diffusion.noise_schedule_base import NoiseScheduleBase


class RectifiedFlowSchedule(NoiseScheduleBase):
    """Rectified flow schedule for training and Euler inference.

    All methods work with continuous t ∈ [0, 1].
    No schedule tensors to precompute — the math is closed-form.

    Convention (opposite of DDPM):
        t = 0  →  pure Gaussian noise
        t = 1  →  clean data
    """

    def __init__(
        self,
        num_inference_steps: int = 4,
        num_timestep_buckets: int = 1000,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
    ) -> None:
        """
        Args:
            num_inference_steps:   Number of Euler steps at inference.
            num_timestep_buckets:  Continuous t is mapped to an integer in
                                   [0, num_timestep_buckets) for model conditioning.
            noise_beta_alpha:      Alpha parameter of the Beta time sampler.
            noise_beta_beta:       Beta  parameter of the Beta time sampler.
            noise_s:               Upper bound on sampled t (keeps training away
                                   from t=1 where x_t = x_0 exactly).
        """
        self.num_inference_steps: int = num_inference_steps
        self.num_timestep_buckets: int = num_timestep_buckets
        self.noise_s: float = noise_s
        self._beta_dist: Beta = Beta(noise_beta_alpha, noise_beta_beta)

    # ------------------------------------------------------------------
    # NoiseScheduleBase interface
    # ------------------------------------------------------------------

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process: x_t = (1-t)*noise + t*x_0.

        Args:
            x_0:      [B, ...] clean data
            noise:    [B, ...] Gaussian noise
            timestep: [B] continuous float t values in [0, 1]
        Returns:
            [B, ...] noisy samples x_t
        """
        t = _broadcast(timestep, x_0.ndim)
        return (1.0 - t) * noise + t * x_0

    def get_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity target: v* = x_0 - noise (constant along each straight path).

        The timestep argument is accepted for API uniformity but is not used —
        rectified flow targets are independent of t.
        """
        return x_0 - noise

    def step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Single Euler integration step: x_{t+dt} = x_t + dt * v(x_t, t).

        Args:
            model_output: [B, ...] predicted velocity from the model
            x_t:          [B, ...] current sample
            timestep:     unused (kept for interface compatibility)
            dt:           step size (kwarg, default = 1 / num_inference_steps)
        Returns:
            [B, ...] updated sample
        """
        dt: float = kwargs.get("dt", 1.0 / self.num_inference_steps)
        return x_t + dt * model_output

    # ------------------------------------------------------------------
    # Rectified-flow-specific training utilities
    # ------------------------------------------------------------------

    def sample_t(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample continuous t ∈ [0, noise_s] from a scaled Beta distribution.

        Beta(1.5, 1.0) is right-skewed, placing more mass near t=1 (signal-rich
        regime), which empirically improves training convergence.

        Returns:
            [B] float tensor of continuous timestep values
        """
        t = self._beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return (1.0 - t) * self.noise_s

    def discretize_t(self, t_cont: torch.Tensor) -> torch.Tensor:
        """Map continuous t to integer bucket indices for model conditioning.

        Args:
            t_cont: [B] continuous t values in [0, noise_s]
        Returns:
            [B] int64 bucket indices in [0, num_timestep_buckets)
        """
        return (t_cont * self.num_timestep_buckets).long()

    def training_step(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience: compute noisy sample, velocity target, and discrete t.

        Args:
            x_0:   [B, ...] clean data
            noise: [B, ...] Gaussian noise
            t:     [B] continuous timestep from sample_t()
        Returns:
            (x_t, v_target, t_discrete)
        """
        x_t = self.add_noise(x_0, noise, t)
        v_target = self.get_target(x_0, noise, t)
        t_discrete = self.discretize_t(t)
        return x_t, v_target, t_discrete

    def inference_schedule(self) -> list[tuple[float, int]]:
        """Return the (t_continuous, t_discrete) pairs for each inference step.

        Steps uniformly from t=0 to t=1 (exclusive of endpoint).
        """
        n = self.num_inference_steps
        return [
            (step / float(n), int((step / float(n)) * self.num_timestep_buckets))
            for step in range(n)
        ]

    def sample(
        self,
        noise: torch.Tensor,
        velocity_fn,
    ) -> torch.Tensor:
        """Run the full Euler inference loop.

        Args:
            noise:       [B, ...] starting noise tensor
            velocity_fn: callable(x_t, t_discrete_tensor) → velocity [B, ...]
        Returns:
            [B, ...] generated clean samples
        """
        x = noise
        dt = 1.0 / self.num_inference_steps
        B, device = noise.shape[0], noise.device

        for t_cont, t_disc_int in self.inference_schedule():
            t_disc = torch.full((B,), t_disc_int, dtype=torch.long, device=device)
            velocity = velocity_fn(x, t_disc)
            x = self.step(velocity, x, t_disc, dt=dt)

        return x

    # Alias for backward compatibility
    def get_velocity_target(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return x_0 - noise


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _broadcast(t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Expand [B] tensor to broadcast with [B, ...] data tensors."""
    while t.ndim < ndim:
        t = t.unsqueeze(-1)
    return t
