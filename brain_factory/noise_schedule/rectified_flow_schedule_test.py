"""Tests for RectifiedFlowSchedule."""
import sys
import unittest

import torch

sys.path.insert(0, ".")
from brain_factory.noise_schedule.rectified_flow_schedule import RectifiedFlowSchedule
from brain_factory.scaffold.diffusion.noise_schedule_base import NoiseScheduleBase


class TestRectifiedFlowInheritance(unittest.TestCase):
    def test_inherits_base(self):
        self.assertTrue(issubclass(RectifiedFlowSchedule, NoiseScheduleBase))


class TestSampleT(unittest.TestCase):
    def setUp(self):
        self.schedule = RectifiedFlowSchedule(num_inference_steps=4, noise_s=0.999)

    def test_shape(self):
        t = self.schedule.sample_t(8, torch.device("cpu"), torch.float32)
        self.assertEqual(t.shape, (8,))
        self.assertEqual(t.dtype, torch.float32)

    def test_in_range(self):
        torch.manual_seed(0)
        t = self.schedule.sample_t(1000, torch.device("cpu"), torch.float32)
        self.assertTrue((t >= 0).all())
        self.assertTrue((t <= self.schedule.noise_s).all())


class TestAddNoise(unittest.TestCase):
    def setUp(self):
        self.schedule = RectifiedFlowSchedule()

    def test_output_shape(self):
        x0 = torch.randn(4, 16, 29)
        noise = torch.randn_like(x0)
        t = torch.full((4,), 0.5)
        self.assertEqual(self.schedule.add_noise(x0, noise, t).shape, x0.shape)

    def test_at_t0_is_noise(self):
        x0 = torch.randn(2, 8)
        noise = torch.randn_like(x0)
        x_t = self.schedule.add_noise(x0, noise, torch.zeros(2))
        self.assertTrue(torch.allclose(x_t, noise))

    def test_at_t1_is_clean(self):
        x0 = torch.randn(2, 8)
        noise = torch.randn_like(x0)
        x_t = self.schedule.add_noise(x0, noise, torch.ones(2))
        self.assertTrue(torch.allclose(x_t, x0))

    def test_midpoint(self):
        x0 = torch.ones(2, 4)
        noise = torch.zeros(2, 4)
        x_t = self.schedule.add_noise(x0, noise, torch.full((2,), 0.5))
        self.assertTrue(torch.allclose(x_t, torch.full_like(x_t, 0.5)))


class TestGetTarget(unittest.TestCase):
    def setUp(self):
        self.schedule = RectifiedFlowSchedule()

    def test_constant_across_timesteps(self):
        x0 = torch.randn(3, 10)
        noise = torch.randn_like(x0)
        v1 = self.schedule.get_target(x0, noise, torch.full((3,), 0.1))
        v2 = self.schedule.get_target(x0, noise, torch.full((3,), 0.9))
        self.assertTrue(torch.equal(v1, v2))

    def test_equals_x0_minus_noise(self):
        x0 = torch.randn(2, 5)
        noise = torch.randn_like(x0)
        t = torch.full((2,), 0.5)
        self.assertTrue(torch.equal(self.schedule.get_target(x0, noise, t), x0 - noise))

    def test_get_velocity_target_alias(self):
        x0 = torch.randn(2, 5)
        noise = torch.randn_like(x0)
        t = torch.full((2,), 0.5)
        self.assertTrue(torch.equal(
            self.schedule.get_target(x0, noise, t),
            self.schedule.get_velocity_target(x0, noise),
        ))


class TestDiscreteT(unittest.TestCase):
    def setUp(self):
        self.schedule = RectifiedFlowSchedule(num_timestep_buckets=1000)

    def test_dtype_and_range(self):
        t = torch.tensor([0.0, 0.5, 0.999])
        disc = self.schedule.discretize_t(t)
        self.assertEqual(disc.dtype, torch.int64)
        self.assertTrue((disc >= 0).all())
        self.assertTrue((disc < self.schedule.num_timestep_buckets).all())

    def test_monotone(self):
        t = torch.linspace(0, 0.999, 100)
        disc = self.schedule.discretize_t(t)
        self.assertTrue((disc[1:] >= disc[:-1]).all())


class TestTrainingStep(unittest.TestCase):
    def setUp(self):
        self.schedule = RectifiedFlowSchedule()

    def test_output_shapes(self):
        B = 4
        x0 = torch.randn(B, 16, 29)
        noise = torch.randn_like(x0)
        t = self.schedule.sample_t(B, x0.device, x0.dtype)
        x_t, v_target, t_disc = self.schedule.training_step(x0, noise, t)
        self.assertEqual(x_t.shape, x0.shape)
        self.assertEqual(v_target.shape, x0.shape)
        self.assertEqual(t_disc.shape, (B,))
        self.assertEqual(t_disc.dtype, torch.int64)

    def test_velocity_value(self):
        x0 = torch.ones(2, 5)
        noise = torch.zeros(2, 5)
        t = torch.full((2,), 0.5)
        _, v_target, _ = self.schedule.training_step(x0, noise, t)
        self.assertTrue(torch.allclose(v_target, torch.ones_like(v_target)))


class TestEulerStep(unittest.TestCase):
    def setUp(self):
        self.schedule = RectifiedFlowSchedule(num_inference_steps=4)

    def test_output_shape(self):
        x_t = torch.randn(2, 8)
        v = torch.randn_like(x_t)
        out = self.schedule.step(v, x_t, torch.zeros(2), dt=0.25)
        self.assertEqual(out.shape, x_t.shape)

    def test_euler_value(self):
        x_t = torch.zeros(1, 4)
        v = torch.ones(1, 4)
        out = self.schedule.step(v, x_t, torch.zeros(1), dt=0.25)
        self.assertTrue(torch.allclose(out, torch.full_like(out, 0.25)))

    def test_default_dt(self):
        x_t = torch.zeros(1, 4)
        v = torch.ones(1, 4)
        out = self.schedule.step(v, x_t, torch.zeros(1))
        expected = 1.0 / self.schedule.num_inference_steps
        self.assertTrue(torch.allclose(out, torch.full_like(out, expected)))


class TestInferenceSchedule(unittest.TestCase):
    def setUp(self):
        self.schedule = RectifiedFlowSchedule(num_inference_steps=4)

    def test_length(self):
        self.assertEqual(len(self.schedule.inference_schedule()), 4)

    def test_starts_at_zero(self):
        t_cont, t_disc = self.schedule.inference_schedule()[0]
        self.assertEqual(t_cont, 0.0)
        self.assertEqual(t_disc, 0)

    def test_monotone(self):
        t_conts = [s[0] for s in self.schedule.inference_schedule()]
        self.assertTrue(all(t_conts[i] <= t_conts[i + 1] for i in range(len(t_conts) - 1)))


if __name__ == "__main__":
    unittest.main()
