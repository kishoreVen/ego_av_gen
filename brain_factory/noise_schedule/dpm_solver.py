"""DPM-Solver++ multistep scheduler for inference (pure PyTorch, no HF deps).

Ported from the diffusers DPMSolverMultistepScheduler, configured for
VibeVoice's `sde-dpmsolver++` with v_prediction and cosine beta schedule.

This is primarily an inference scheduler. Training noise / velocity targets
should be computed using DDPMCosineSchedule (add_noise, get_target).

Reference:
    "DPM-Solver++: Fast Solver for Guided Sampling" (Lu et al., 2022)
    https://arxiv.org/abs/2211.01095
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from brain_factory.scaffold.diffusion.noise_schedule_base import NoiseScheduleBase


@dataclass
class DPMSolverConfig:
    """Configuration matching the reference VibeVoice scheduler setup."""

    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "v_prediction"
    algorithm_type: str = "sde-dpmsolver++"
    solver_order: int = 2
    solver_type: str = "midpoint"
    lower_order_final: bool = True
    euler_at_final: bool = False
    thresholding: bool = False
    final_sigmas_type: str = "zero"
    timestep_spacing: str = "linspace"
    lambda_min_clipped: float = -float("inf")


def _betas_for_alpha_bar(num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine beta schedule (squaredcos_cap_v2)."""
    def alpha_bar_fn(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DPMSolverSchedule(NoiseScheduleBase):
    """DPM-Solver++ multistep scheduler.

    Implements NoiseScheduleBase: add_noise and get_target for training,
    step for inference (multi-step with internal state buffer).

    Note: step() is stateful — call set_timesteps() before each inference
    run to reset the internal model_outputs buffer and step index.
    """

    def __init__(self, config: Optional[DPMSolverConfig] = None) -> None:
        if config is None:
            config = DPMSolverConfig()
        self.config = config

        if config.beta_schedule in ("squaredcos_cap_v2", "cosine"):
            self.betas = _betas_for_alpha_bar(config.num_train_timesteps)
        else:
            raise NotImplementedError(f"beta_schedule={config.beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1.0 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        self.sigmas = ((1.0 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # Mutable inference state — reset by set_timesteps()
        self.num_inference_steps: Optional[int] = None
        self.timesteps: torch.Tensor = torch.tensor([])
        self.model_outputs: List[Optional[torch.Tensor]] = [None] * config.solver_order
        self.lower_order_nums: int = 0
        self._step_index: Optional[int] = None

    # ------------------------------------------------------------------
    # NoiseScheduleBase interface
    # ------------------------------------------------------------------

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: x_t = α_t * x_0 + σ_t * noise."""
        alpha = self._gather(self.alpha_t, timestep, x_0.dim())
        sigma = self._gather(self.sigma_t, timestep, x_0.dim())
        return alpha * x_0 + sigma * noise

    def get_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute regression target.

        v_prediction: α_t * noise − σ_t * x_0
        epsilon:      noise
        """
        if self.config.prediction_type == "epsilon":
            return noise
        alpha = self._gather(self.alpha_t, timestep, x_0.dim())
        sigma = self._gather(self.sigma_t, timestep, x_0.dim())
        return alpha * noise - sigma * x_0

    def step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Single DPM-Solver++ denoising step.

        Stateful: internally buffers previous model outputs for multistep
        update. Call set_timesteps() before each new inference run.

        kwargs:
            variance_noise: optional pre-sampled noise for SDE variant
            generator:      optional torch.Generator for reproducibility
        """
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps() before step().")

        t_int = int(timestep.item()) if hasattr(timestep, "item") else int(timestep)
        if self._step_index is None:
            self._init_step_index(torch.tensor([t_int]))

        lower_order_final = (
            self._step_index == len(self.timesteps) - 1
        ) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            self._step_index == len(self.timesteps) - 2
        ) and self.config.lower_order_final and len(self.timesteps) < 15

        model_output_converted = self.convert_model_output(model_output, sample=x_t)

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output_converted

        sample = x_t.to(torch.float32)

        noise = kwargs.get("variance_noise")
        generator = kwargs.get("generator")
        if self.config.algorithm_type in ("sde-dpmsolver", "sde-dpmsolver++"):
            if noise is None:
                noise = torch.randn(
                    model_output.shape,
                    generator=generator,
                    device=model_output.device,
                    dtype=torch.float32,
                )
            else:
                noise = noise.to(device=model_output.device, dtype=torch.float32)

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self._first_order_update(model_output_converted, sample, noise)
        else:
            prev_sample = self._second_order_update(self.model_outputs, sample, noise)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        self._step_index += 1
        return prev_sample.to(model_output.dtype)

    # ------------------------------------------------------------------
    # Inference setup
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Compute timestep schedule and reset state for a new inference run."""
        cfg = self.config
        clipped_idx = torch.searchsorted(
            torch.flip(self.lambda_t, [0]), cfg.lambda_min_clipped
        )
        last_timestep = int((cfg.num_train_timesteps - clipped_idx).item())

        if cfg.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, last_timestep - 1, num_inference_steps + 1)
                .round()[::-1][:-1]
                .copy()
                .astype(np.int64)
            )
        else:
            raise NotImplementedError(f"timestep_spacing={cfg.timestep_spacing}")

        sigmas_np = ((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt().numpy()
        sigmas_np = np.interp(timesteps, np.arange(len(sigmas_np)), sigmas_np)

        sigma_last = 0.0 if cfg.final_sigmas_type == "zero" else float(sigmas_np[-1])
        sigmas_np = np.concatenate([sigmas_np, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas_np)
        self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.int64)
        self.num_inference_steps = len(timesteps)
        self.model_outputs = [None] * cfg.solver_order
        self.lower_order_nums = 0
        self._step_index = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather(self, values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
        out = values.to(timesteps.device)[timesteps.long()].float()
        while out.dim() < ndim:
            out = out.unsqueeze(-1)
        return out

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor):
        alpha_t = 1.0 / ((sigma ** 2 + 1) ** 0.5)
        return alpha_t, sigma * alpha_t

    def _init_step_index(self, timestep: torch.Tensor) -> None:
        timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        self._step_index = int(indices[0].item()) if len(indices) > 0 else 0

    def convert_model_output(
        self, model_output: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        """Convert v_prediction → x0_pred for dpmsolver++ update."""
        sigma = self.sigmas[self._step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        if self.config.prediction_type == "v_prediction":
            return alpha_t * sample - sigma_t * model_output
        elif self.config.prediction_type == "epsilon":
            return (sample - sigma_t * model_output) / alpha_t
        raise ValueError(f"prediction_type={self.config.prediction_type}")

    def _first_order_update(self, m0, sample, noise):
        sigma_t = self.sigmas[self._step_index + 1]
        sigma_s = self.sigmas[self._step_index]
        at, st = self._sigma_to_alpha_sigma_t(sigma_t)
        as_, ss = self._sigma_to_alpha_sigma_t(sigma_s)
        lt = torch.log(at) - torch.log(st)
        ls = torch.log(as_) - torch.log(ss)
        h = lt - ls
        if self.config.algorithm_type == "sde-dpmsolver++":
            return (
                (st / ss * torch.exp(-h)) * sample
                + (at * (1 - torch.exp(-2.0 * h))) * m0
                + st * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        return (st / ss) * sample - (at * (torch.exp(-h) - 1.0)) * m0

    def _second_order_update(self, model_outputs, sample, noise):
        sigma_t = self.sigmas[self._step_index + 1]
        sigma_s0 = self.sigmas[self._step_index]
        sigma_s1 = self.sigmas[self._step_index - 1]
        at, st = self._sigma_to_alpha_sigma_t(sigma_t)
        as0, ss0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        as1, ss1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        lt = torch.log(at) - torch.log(st)
        ls0 = torch.log(as0) - torch.log(ss0)
        ls1 = torch.log(as1) - torch.log(ss1)
        m0, m1 = model_outputs[-1], model_outputs[-2]
        h = lt - ls0
        r0 = (ls0 - ls1) / h
        D0 = m0
        D1 = (1.0 / r0) * (m0 - m1)
        if self.config.algorithm_type == "sde-dpmsolver++":
            if self.config.solver_type == "midpoint":
                return (
                    (st / ss0 * torch.exp(-h)) * sample
                    + (at * (1 - torch.exp(-2.0 * h))) * D0
                    + 0.5 * (at * (1 - torch.exp(-2.0 * h))) * D1
                    + st * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            return (
                (st / ss0 * torch.exp(-h)) * sample
                + (at * (1 - torch.exp(-2.0 * h))) * D0
                + (at * ((1 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                + st * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        if self.config.solver_type == "midpoint":
            return (
                (st / ss0) * sample
                - (at * (torch.exp(-h) - 1.0)) * D0
                - 0.5 * (at * (torch.exp(-h) - 1.0)) * D1
            )
        return (
            (st / ss0) * sample
            - (at * (torch.exp(-h) - 1.0)) * D0
            + (at * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
        )

    # Alias kept for backward compatibility
    def get_velocity(self, original, noise, timesteps):
        return self.get_target(original, noise, timesteps)
