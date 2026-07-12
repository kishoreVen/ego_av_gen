"""DDPM diffusion schedule with cosine beta schedule and v_prediction.

Pure math utilities — no nn.Module, no learned parameters.
Precomputes alpha/beta schedules at construction time.

Reference: "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021).
"""
from __future__ import annotations

import math
from typing import Literal

import torch

from brain_factory.scaffold.diffusion.noise_schedule_base import NoiseScheduleBase


class DDPMCosineSchedule(NoiseScheduleBase):
    """DDPM noise schedule with cosine beta schedule.

    Prediction target is v_prediction by default:
        v = sqrt(alpha_bar) * noise − sqrt(1 − alpha_bar) * x_0

    All schedule tensors are stored as 1D float64 arrays indexed by timestep
    (0 to num_train_timesteps - 1).  Timesteps are integer bucket indices.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: Literal["v_prediction", "epsilon"] = "v_prediction",
        s: float = 0.008,
    ) -> None:
        self.num_train_timesteps: int = num_train_timesteps
        self.prediction_type: str = prediction_type

        # Cosine schedule: alpha_bar(t) = cos^2((t/T + s) / (1 + s) * pi/2)
        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / num_train_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f_t / f_t[0]
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-6, max=1.0 - 1e-6)

        self.alphas_cumprod: torch.Tensor = alphas_cumprod[:num_train_timesteps]
        self.sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - self.alphas_cumprod)

        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas: torch.Tensor = torch.clamp(betas, max=0.999)
        self.alphas: torch.Tensor = 1.0 - self.betas

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _gather(self, values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
        """Index into a schedule tensor, broadcast to data shape."""
        out = values.to(timesteps.device)[timesteps.long()].float()
        while out.dim() < ndim:
            out = out.unsqueeze(-1)
        return out

    # ------------------------------------------------------------------
    # NoiseScheduleBase interface
    # ------------------------------------------------------------------

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(α_bar_t)*x_0 + sqrt(1−α_bar_t)*noise."""
        sqrt_a = self._gather(self.sqrt_alphas_cumprod, timestep, x_0.dim())
        sqrt_1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, timestep, x_0.dim())
        return sqrt_a * x_0 + sqrt_1ma * noise

    def get_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute model regression target.

        v_prediction: sqrt(α_bar_t)*noise − sqrt(1−α_bar_t)*x_0
        epsilon:      noise
        """
        if self.prediction_type == "epsilon":
            return noise
        # v_prediction
        sqrt_a = self._gather(self.sqrt_alphas_cumprod, timestep, x_0.dim())
        sqrt_1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, timestep, x_0.dim())
        return sqrt_a * noise - sqrt_1ma * x_0

    def step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Single DDPM reverse step: x_t → x_{t-1}.

        Recovers x_0 from model output, then samples from the posterior.
        """
        t = int(timestep.item()) if timestep.numel() == 1 else int(timestep[0].item())
        t_tensor = torch.tensor([t], device=x_t.device)

        x_0_pred = self.predict_original(model_output, t_tensor, x_t)
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        if t == 0:
            return x_0_pred

        alpha_bar_t = self.alphas_cumprod[t].to(x_t.device).float()
        alpha_bar_prev = self.alphas_cumprod[t - 1].to(x_t.device).float()
        beta_t = self.betas[t].to(x_t.device).float()

        coeff_x0 = (alpha_bar_prev.sqrt() * beta_t) / (1.0 - alpha_bar_t)
        coeff_xt = (self.alphas[t].to(x_t.device).float().sqrt() * (1.0 - alpha_bar_prev)) / (
            1.0 - alpha_bar_t
        )
        mean = coeff_x0 * x_0_pred + coeff_xt * x_t
        variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        return mean + variance.sqrt() * torch.randn_like(x_t)

    # ------------------------------------------------------------------
    # Additional utilities (not part of base contract)
    # ------------------------------------------------------------------

    def predict_original(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Recover x_0 from the model's output at timestep t."""
        sqrt_a = self._gather(self.sqrt_alphas_cumprod, timestep, sample.dim())
        sqrt_1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, timestep, sample.dim())
        if self.prediction_type == "v_prediction":
            return sqrt_a * sample - sqrt_1ma * model_output
        else:  # epsilon
            return (sample - sqrt_1ma * model_output) / sqrt_a

    # Alias kept for backward compatibility with existing code
    def get_velocity(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.get_target(original, noise, timesteps)
