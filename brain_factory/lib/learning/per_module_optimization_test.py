from __future__ import annotations

import unittest
from typing import Any, Dict

import torch
import torch.nn as nn

from brain_factory.lib.learning.per_module_optimization import (
    _get_lr_for_module,
    _has_learning_rate_conflicts,
    gather_optimizer_param_groups,
)


class SimpleBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


class SimpleHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TwoPartModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SimpleBackbone()
        self.head = SimpleHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class TestGetLrForModule(unittest.TestCase):
    def test_exact_match(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-5}
        self.assertEqual(_get_lr_for_module("backbone", 1e-4, config), 1e-5)

    def test_hierarchical_match(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-5}
        self.assertEqual(_get_lr_for_module("backbone.layer1", 1e-4, config), 1e-5)

    def test_deep_hierarchical_match(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-5}
        self.assertEqual(
            _get_lr_for_module("backbone.layer1.weight", 1e-4, config), 1e-5
        )

    def test_most_specific_wins(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-5, "backbone.layer1": 1e-6}
        self.assertEqual(_get_lr_for_module("backbone.layer1", 1e-4, config), 1e-6)

    def test_falls_back_to_default(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-5}
        self.assertEqual(_get_lr_for_module("head", 1e-4, config), 1e-4)

    def test_no_partial_name_match(self) -> None:
        """'back' should not match 'backbone'."""
        config = {"default": 1e-4, "back": 1e-5}
        self.assertEqual(_get_lr_for_module("backbone", 1e-4, config), 1e-4)


class TestHasLearningRateConflicts(unittest.TestCase):
    def test_no_conflicts(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-4}
        self.assertFalse(_has_learning_rate_conflicts("backbone", 1e-4, config))

    def test_deeper_conflict(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-4, "backbone.layer1": 1e-6}
        self.assertTrue(_has_learning_rate_conflicts("backbone", 1e-4, config))

    def test_deeper_same_lr_no_conflict(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-4, "backbone.layer1": 1e-4}
        self.assertFalse(_has_learning_rate_conflicts("backbone", 1e-4, config))

    def test_sibling_not_a_conflict(self) -> None:
        config = {"default": 1e-4, "backbone": 1e-4, "head": 1e-6}
        self.assertFalse(_has_learning_rate_conflicts("backbone", 1e-4, config))


class TestGatherOptimizerParamGroups(unittest.TestCase):
    def test_missing_default_raises(self) -> None:
        model = TwoPartModel()
        with self.assertRaises(ValueError):
            gather_optimizer_param_groups(model, {"backbone": 1e-5})

    def test_uniform_lr_single_groups(self) -> None:
        """With only a default LR, each top-level child gets one group."""
        model = TwoPartModel()
        groups = gather_optimizer_param_groups(model, {"default": 1e-4})
        names = [g["name"] for g in groups]
        self.assertIn("backbone", names)
        self.assertIn("head", names)
        for g in groups:
            self.assertEqual(g["lr"], 1e-4)

    def test_per_module_lr(self) -> None:
        """Different LR for backbone vs head."""
        model = TwoPartModel()
        config = {"default": 1e-4, "backbone": 1e-5}
        groups = gather_optimizer_param_groups(model, config)
        lr_by_name = {g["name"]: g["lr"] for g in groups}
        self.assertEqual(lr_by_name["backbone"], 1e-5)
        self.assertEqual(lr_by_name["head"], 1e-4)

    def test_deep_override_splits_groups(self) -> None:
        """Override for backbone.layer1 should split backbone into finer groups."""
        model = TwoPartModel()
        config = {"default": 1e-4, "backbone.layer1": 1e-6}
        groups = gather_optimizer_param_groups(model, config)
        lr_by_name = {g["name"]: g["lr"] for g in groups}
        self.assertEqual(lr_by_name["backbone.layer1"], 1e-6)
        self.assertEqual(lr_by_name["backbone.layer2"], 1e-4)
        self.assertEqual(lr_by_name["head"], 1e-4)

    def test_all_params_covered(self) -> None:
        """Every requires_grad parameter should appear in exactly one group."""
        model = TwoPartModel()
        config = {"default": 1e-4, "backbone.layer1": 1e-6}
        groups = gather_optimizer_param_groups(model, config)

        all_group_params = set()
        for g in groups:
            for p in g["params"]:
                param_id = id(p)
                self.assertNotIn(param_id, all_group_params, "Duplicate param found")
                all_group_params.add(param_id)

        expected_params = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(all_group_params, expected_params)

    def test_frozen_params_excluded(self) -> None:
        """Parameters with requires_grad=False should not appear in any group."""
        model = TwoPartModel()
        for p in model.backbone.layer1.parameters():
            p.requires_grad = False

        config = {"default": 1e-4}
        groups = gather_optimizer_param_groups(model, config)

        all_group_params = set()
        for g in groups:
            for p in g["params"]:
                all_group_params.add(id(p))

        frozen_params = {
            id(p)
            for p in model.backbone.layer1.parameters()
        }
        self.assertTrue(all_group_params.isdisjoint(frozen_params))

    def test_eval_module_skipped(self) -> None:
        """Modules in eval mode should be skipped."""
        model = TwoPartModel()
        model.backbone.eval()

        config = {"default": 1e-4}
        groups = gather_optimizer_param_groups(model, config)
        names = [g["name"] for g in groups]
        self.assertNotIn("backbone", names)
        self.assertIn("head", names)


if __name__ == "__main__":
    unittest.main()
