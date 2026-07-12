"""Tests for DPMSolverSchedule."""
import sys
import unittest

import torch

sys.path.insert(0, ".")
from brain_factory.noise_schedule.dpm_solver import DPMSolverConfig, DPMSolverSchedule
from brain_factory.scaffold.diffusion.noise_schedule_base import NoiseScheduleBase


class TestDPMSolverInheritance(unittest.TestCase):
    def test_inherits_base(self):
        self.assertTrue(issubclass(DPMSolverSchedule, NoiseScheduleBase))


class TestDPMSolverScheduleValues(unittest.TestCase):
    def setUp(self):
        self.schedule = DPMSolverSchedule()

    def test_alphas_cumprod_shape(self):
        self.assertEqual(self.schedule.alphas_cumprod.shape, (1000,))

    def test_sigma_t_range(self):
        self.assertTrue((self.schedule.sigma_t >= 0).all())
        self.assertTrue((self.schedule.sigma_t <= 1.0).all())


class TestDPMSolverAddNoise(unittest.TestCase):
    def setUp(self):
        self.schedule = DPMSolverSchedule()

    def test_output_shape(self):
        x0 = torch.randn(3, 64)
        noise = torch.randn_like(x0)
        t = torch.randint(0, 1000, (3,))
        self.assertEqual(self.schedule.add_noise(x0, noise, t).shape, x0.shape)

    def test_at_t0_approx_clean(self):
        # alpha_t[0] ≈ 1, sigma_t[0] ≈ 0 → x_t ≈ x_0 when noise is zero
        x0 = torch.randn(2, 8)
        noise = torch.zeros_like(x0)
        t = torch.zeros(2, dtype=torch.long)
        x_t = self.schedule.add_noise(x0, noise, t)
        self.assertTrue(torch.allclose(x_t, x0, atol=1e-3))


class TestDPMSolverGetTarget(unittest.TestCase):
    def setUp(self):
        self.schedule = DPMSolverSchedule()

    def test_vpred_output_shape(self):
        x0 = torch.randn(2, 8)
        noise = torch.randn_like(x0)
        t = torch.tensor([100, 500])
        self.assertEqual(self.schedule.get_target(x0, noise, t).shape, x0.shape)

    def test_epsilon_returns_noise(self):
        sched = DPMSolverSchedule(DPMSolverConfig(prediction_type="epsilon"))
        x0 = torch.randn(2, 8)
        noise = torch.randn_like(x0)
        t = torch.tensor([100, 500])
        self.assertTrue(torch.equal(sched.get_target(x0, noise, t), noise))

    def test_get_velocity_alias(self):
        x0 = torch.randn(2, 5)
        noise = torch.randn_like(x0)
        t = torch.tensor([100, 500])
        self.assertTrue(torch.equal(
            self.schedule.get_target(x0, noise, t),
            self.schedule.get_velocity(x0, noise, t),
        ))


class TestDPMSolverSetTimesteps(unittest.TestCase):
    def setUp(self):
        self.schedule = DPMSolverSchedule()

    def test_sets_correct_count(self):
        self.schedule.set_timesteps(20)
        self.assertEqual(self.schedule.num_inference_steps, 20)
        self.assertEqual(len(self.schedule.timesteps), 20)

    def test_resets_state(self):
        self.schedule.set_timesteps(10)
        self.schedule._step_index = 5
        self.schedule.set_timesteps(10)
        self.assertIsNone(self.schedule._step_index)
        self.assertEqual(self.schedule.lower_order_nums, 0)


class TestDPMSolverStep(unittest.TestCase):
    def setUp(self):
        self.schedule = DPMSolverSchedule()

    def test_requires_set_timesteps(self):
        with self.assertRaises(ValueError):
            self.schedule.step(torch.randn(1, 4), torch.randn(1, 4), torch.tensor([999]))

    def test_inference_loop_output_shape(self):
        self.schedule.set_timesteps(5)
        x = torch.randn(2, 16)
        for t in self.schedule.timesteps:
            x = self.schedule.step(torch.randn_like(x), x, t)
        self.assertEqual(x.shape, (2, 16))

    def test_inference_loop_does_not_explode(self):
        torch.manual_seed(42)
        self.schedule.set_timesteps(10)
        x = torch.randn(32, 64)
        initial_std = x.std().item()
        for t in self.schedule.timesteps:
            x = self.schedule.step(torch.zeros_like(x), x, t)
        self.assertLess(x.std().item(), initial_std * 10)


if __name__ == "__main__":
    unittest.main()
