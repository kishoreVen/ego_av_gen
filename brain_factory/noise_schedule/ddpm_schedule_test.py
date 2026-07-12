"""Tests for DDPMCosineSchedule."""
import sys
import unittest

import torch

sys.path.insert(0, ".")
from brain_factory.noise_schedule.ddpm_schedule import DDPMCosineSchedule
from brain_factory.scaffold.diffusion.noise_schedule_base import NoiseScheduleBase


class TestDDPMCosineScheduleInheritance(unittest.TestCase):
    def test_inherits_base(self):
        self.assertTrue(issubclass(DDPMCosineSchedule, NoiseScheduleBase))


class TestDDPMScheduleValues(unittest.TestCase):
    def setUp(self):
        self.schedule = DDPMCosineSchedule(num_train_timesteps=1000)

    def test_alphas_cumprod_range(self):
        self.assertTrue((self.schedule.alphas_cumprod > 0).all())
        self.assertTrue((self.schedule.alphas_cumprod <= 1.0).all())

    def test_alphas_cumprod_monotone_decreasing(self):
        # Higher t = more noise = lower alpha_bar
        a = self.schedule.alphas_cumprod
        self.assertTrue((a[:-1] >= a[1:]).all())

    def test_betas_positive_and_bounded(self):
        self.assertTrue((self.schedule.betas > 0).all())
        self.assertTrue((self.schedule.betas <= 0.999).all())


class TestDDPMAddNoise(unittest.TestCase):
    def setUp(self):
        self.schedule = DDPMCosineSchedule()

    def test_output_shape(self):
        x0 = torch.randn(4, 16, 29)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (4,))
        self.assertEqual(self.schedule.add_noise(x0, noise, t).shape, x0.shape)

    def test_at_t0_approx_clean(self):
        # alpha_bar[0] ≈ 1 → x_t ≈ x_0 when noise is zero
        x0 = torch.randn(2, 8)
        noise = torch.zeros_like(x0)
        t = torch.zeros(2, dtype=torch.long)
        x_t = self.schedule.add_noise(x0, noise, t)
        self.assertTrue(torch.allclose(x_t, x0, atol=1e-3))

    def test_at_tmax_approx_noise(self):
        # alpha_bar[T-1] ≈ 0 → x_t ≈ noise when x_0 is zero
        x0 = torch.zeros(2, 8)
        noise = torch.ones(2, 8)
        t = torch.full((2,), 999, dtype=torch.long)
        x_t = self.schedule.add_noise(x0, noise, t)
        self.assertTrue(torch.allclose(x_t, noise, atol=1e-2))


class TestDDPMGetTarget(unittest.TestCase):
    def setUp(self):
        self.schedule = DDPMCosineSchedule(prediction_type="v_prediction")
        self.eps_schedule = DDPMCosineSchedule(prediction_type="epsilon")

    def test_vpred_output_shape(self):
        x0 = torch.randn(3, 10)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (3,))
        self.assertEqual(self.schedule.get_target(x0, noise, t).shape, x0.shape)

    def test_epsilon_returns_noise(self):
        x0 = torch.randn(2, 5)
        noise = torch.randn_like(x0)
        t = torch.tensor([100, 500])
        self.assertTrue(torch.equal(self.eps_schedule.get_target(x0, noise, t), noise))

    def test_get_velocity_alias(self):
        x0 = torch.randn(2, 5)
        noise = torch.randn_like(x0)
        t = torch.tensor([100, 500])
        self.assertTrue(torch.equal(
            self.schedule.get_target(x0, noise, t),
            self.schedule.get_velocity(x0, noise, t),
        ))


class TestDDPMPredictOriginal(unittest.TestCase):
    def setUp(self):
        self.schedule = DDPMCosineSchedule()

    def test_roundtrip(self):
        x0 = torch.randn(2, 8)
        noise = torch.randn_like(x0)
        t = torch.tensor([200, 500])
        x_t = self.schedule.add_noise(x0, noise, t)
        v = self.schedule.get_target(x0, noise, t)
        x0_rec = self.schedule.predict_original(v, t, x_t)
        self.assertTrue(torch.allclose(x0_rec, x0, atol=1e-5))


class TestDDPMStep(unittest.TestCase):
    def setUp(self):
        self.schedule = DDPMCosineSchedule()

    def test_output_shape(self):
        x_t = torch.randn(2, 8)
        model_out = torch.randn_like(x_t)
        out = self.schedule.step(model_out, x_t, torch.tensor([500]))
        self.assertEqual(out.shape, x_t.shape)

    def test_at_t0_is_deterministic(self):
        # No noise is added at t=0
        x_t = torch.randn(2, 4)
        model_out = torch.randn_like(x_t)
        t = torch.tensor([0])
        out1 = self.schedule.step(model_out, x_t, t)
        out2 = self.schedule.step(model_out, x_t, t)
        self.assertTrue(torch.equal(out1, out2))


if __name__ == "__main__":
    unittest.main()
