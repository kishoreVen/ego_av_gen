from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Dict, List

import torch

from brain_factory.lib.checkpoint import (
    CheckpointConfig,
    CheckpointMapping,
    load_checkpoint,
    load_training_state,
    read_checkpoint_metadata,
    save_checkpoint,
    save_training_state,
)


class EncoderDecoder(torch.nn.Module):
    """Simple encoder-decoder model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(4, 8)
        self.decoder = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class BigModel(torch.nn.Module):
    """Larger model whose encoder can be extracted into a smaller model."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.ModuleDict(
            {"encoder": torch.nn.Linear(4, 8)}
        )
        self.classifier = torch.nn.Linear(8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone["encoder"](x))


class TestLoadCheckpointFullModel(unittest.TestCase):
    def test_full_model_load_from_file(self) -> None:
        """Load a complete model state dict from a single safetensors file."""
        model: EncoderDecoder = EncoderDecoder()
        reference: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(reference, tmpdir)

            config: CheckpointConfig = CheckpointConfig(
                path=os.path.join(tmpdir, "model.safetensors")
            )
            load_checkpoint(config, model)

        for key in reference.state_dict():
            torch.testing.assert_close(
                model.state_dict()[key], reference.state_dict()[key]
            )

    def test_full_model_load_from_directory(self) -> None:
        """Load a complete model state dict from a directory."""
        model: EncoderDecoder = EncoderDecoder()
        reference: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(reference, tmpdir)

            config: CheckpointConfig = CheckpointConfig(path=tmpdir)
            load_checkpoint(config, model)

        for key in reference.state_dict():
            torch.testing.assert_close(
                model.state_dict()[key], reference.state_dict()[key]
            )


class TestLoadCheckpointPrefixRemap(unittest.TestCase):
    def test_cross_model_remap(self) -> None:
        """Load encoder from a BigModel checkpoint into an EncoderDecoder."""
        big_model: BigModel = BigModel()
        target_model: EncoderDecoder = EncoderDecoder()

        original_encoder_weight: torch.Tensor = big_model.backbone["encoder"].weight.clone()
        original_encoder_bias: torch.Tensor = big_model.backbone["encoder"].bias.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(big_model, tmpdir)

            config: CheckpointConfig = CheckpointConfig(
                mappings=[
                    CheckpointMapping(
                        path=tmpdir,
                        source_prefix="backbone.encoder.",
                        target_prefix="encoder.",
                        strict=True,
                    )
                ]
            )
            load_checkpoint(config, target_model)

        torch.testing.assert_close(target_model.encoder.weight, original_encoder_weight)
        torch.testing.assert_close(target_model.encoder.bias, original_encoder_bias)

    def test_source_prefix_to_empty_target(self) -> None:
        """Load a sub-module from a checkpoint as the entire model."""
        standalone: torch.nn.Linear = torch.nn.Linear(4, 8)
        big_model: BigModel = BigModel()

        original_weight: torch.Tensor = big_model.backbone["encoder"].weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(big_model, tmpdir)

            config: CheckpointConfig = CheckpointConfig(
                mappings=[
                    CheckpointMapping(
                        path=tmpdir,
                        source_prefix="backbone.encoder.",
                        target_prefix="",
                        strict=True,
                    )
                ]
            )
            load_checkpoint(config, standalone)

        torch.testing.assert_close(standalone.weight, original_weight)


class TestLoadCheckpointMultiSource(unittest.TestCase):
    def test_encoder_and_decoder_from_different_files(self) -> None:
        """Load encoder and decoder from separate checkpoint files."""
        encoder_source: EncoderDecoder = EncoderDecoder()
        decoder_source: EncoderDecoder = EncoderDecoder()
        target: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            enc_dir: str = os.path.join(tmpdir, "enc")
            dec_dir: str = os.path.join(tmpdir, "dec")
            save_checkpoint(encoder_source, enc_dir)
            save_checkpoint(decoder_source, dec_dir)

            config: CheckpointConfig = CheckpointConfig(
                mappings=[
                    CheckpointMapping(
                        path=enc_dir,
                        source_prefix="encoder.",
                        target_prefix="encoder.",
                    ),
                    CheckpointMapping(
                        path=dec_dir,
                        source_prefix="decoder.",
                        target_prefix="decoder.",
                    ),
                ]
            )
            load_checkpoint(config, target)

        torch.testing.assert_close(
            target.encoder.weight, encoder_source.encoder.weight
        )
        torch.testing.assert_close(
            target.decoder.weight, decoder_source.decoder.weight
        )


class TestLoadCheckpointBaseWithOverlay(unittest.TestCase):
    def test_base_plus_overlay(self) -> None:
        """Load full base model, then overwrite encoder from a different source."""
        base_model: EncoderDecoder = EncoderDecoder()
        overlay_source: BigModel = BigModel()
        target: EncoderDecoder = EncoderDecoder()

        original_decoder_weight: torch.Tensor = base_model.decoder.weight.clone()
        overlay_encoder_weight: torch.Tensor = overlay_source.backbone["encoder"].weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir: str = os.path.join(tmpdir, "base")
            overlay_dir: str = os.path.join(tmpdir, "overlay")
            save_checkpoint(base_model, base_dir)
            save_checkpoint(overlay_source, overlay_dir)

            config: CheckpointConfig = CheckpointConfig(
                path=base_dir,
                mappings=[
                    CheckpointMapping(
                        path=overlay_dir,
                        source_prefix="backbone.encoder.",
                        target_prefix="encoder.",
                    )
                ],
            )
            load_checkpoint(config, target)

        # Encoder should come from overlay
        torch.testing.assert_close(target.encoder.weight, overlay_encoder_weight)
        # Decoder should come from base
        torch.testing.assert_close(target.decoder.weight, original_decoder_weight)


class TestLoadCheckpointLastWins(unittest.TestCase):
    def test_last_mapping_wins_on_conflict(self) -> None:
        """When two mappings target the same keys, the last one wins."""
        source_a: EncoderDecoder = EncoderDecoder()
        source_b: EncoderDecoder = EncoderDecoder()
        target: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a: str = os.path.join(tmpdir, "a")
            dir_b: str = os.path.join(tmpdir, "b")
            save_checkpoint(source_a, dir_a)
            save_checkpoint(source_b, dir_b)

            config: CheckpointConfig = CheckpointConfig(
                mappings=[
                    CheckpointMapping(path=dir_a, source_prefix="encoder.", target_prefix="encoder."),
                    CheckpointMapping(path=dir_b, source_prefix="encoder.", target_prefix="encoder."),
                ]
            )
            load_checkpoint(config, target)

        # source_b (last) should win
        torch.testing.assert_close(target.encoder.weight, source_b.encoder.weight)


class TestLoadCheckpointSharded(unittest.TestCase):
    def test_load_from_sharded_checkpoint(self) -> None:
        """Save sharded, load back, verify round-trip."""
        reference: EncoderDecoder = EncoderDecoder()
        target: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(reference, tmpdir, max_shard_size_bytes=64)

            config: CheckpointConfig = CheckpointConfig(path=tmpdir)
            load_checkpoint(config, target)

        for key in reference.state_dict():
            torch.testing.assert_close(
                target.state_dict()[key], reference.state_dict()[key]
            )

    def test_sharded_partial_load(self) -> None:
        """Save sharded, load only encoder via mapping."""
        reference: EncoderDecoder = EncoderDecoder()
        target: EncoderDecoder = EncoderDecoder()
        original_decoder_weight: torch.Tensor = target.decoder.weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(reference, tmpdir, max_shard_size_bytes=64)

            config: CheckpointConfig = CheckpointConfig(
                mappings=[
                    CheckpointMapping(
                        path=tmpdir,
                        source_prefix="encoder.",
                        target_prefix="encoder.",
                        strict=True,
                    )
                ]
            )
            load_checkpoint(config, target)

        torch.testing.assert_close(target.encoder.weight, reference.encoder.weight)
        torch.testing.assert_close(target.decoder.weight, original_decoder_weight)


class TestLoadCheckpointStrictValidation(unittest.TestCase):
    def test_strict_mapping_missing_keys_raises(self) -> None:
        """Strict mapping should fail if checkpoint doesn't cover all model keys under target prefix."""
        target: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            # bias=False produces only a "weight" key — encoder expects weight + bias
            save_checkpoint(torch.nn.Linear(4, 8, bias=False), tmpdir)

            config: CheckpointConfig = CheckpointConfig(
                mappings=[
                    CheckpointMapping(
                        path=tmpdir,
                        source_prefix="",
                        target_prefix="encoder.",
                        strict=True,
                    )
                ]
            )
            with self.assertRaises(ValueError):
                load_checkpoint(config, target)

    def test_non_strict_mapping_allows_missing_keys(self) -> None:
        """Non-strict mapping should not fail on missing keys."""
        target: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            # bias=False produces only a "weight" key — decoder expects weight + bias
            save_checkpoint(torch.nn.Linear(8, 4, bias=False), tmpdir)

            config: CheckpointConfig = CheckpointConfig(
                mappings=[
                    CheckpointMapping(
                        path=tmpdir,
                        source_prefix="",
                        target_prefix="decoder.",
                        strict=False,
                    )
                ]
            )
            # Should not raise
            load_checkpoint(config, target)


class TestSaveCheckpointSingleFile(unittest.TestCase):
    def test_save_and_reload(self) -> None:
        """Save a model, then load it back and verify weights match."""
        source: EncoderDecoder = EncoderDecoder()
        target: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path: str = save_checkpoint(source, tmpdir)
            self.assertTrue(result_path.endswith("model.safetensors"))
            self.assertTrue(os.path.exists(result_path))

            config: CheckpointConfig = CheckpointConfig(path=tmpdir)
            load_checkpoint(config, target)

        for key in source.state_dict():
            torch.testing.assert_close(
                target.state_dict()[key], source.state_dict()[key]
            )

    def test_save_with_metadata(self) -> None:
        """Save with metadata and verify it's stored in the file header."""
        model: EncoderDecoder = EncoderDecoder()
        metadata: Dict[str, str] = {"model_type": "encoder_decoder", "version": "1"}

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path: str = save_checkpoint(model, tmpdir, metadata=metadata)
            stored_metadata: Dict[str, str] = read_checkpoint_metadata(result_path)

            self.assertEqual(stored_metadata["model_type"], "encoder_decoder")
            self.assertEqual(stored_metadata["version"], "1")

    def test_save_creates_output_dir(self) -> None:
        """save_checkpoint should create the output directory if it doesn't exist."""
        model: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir: str = os.path.join(tmpdir, "a", "b", "c")
            result_path: str = save_checkpoint(model, nested_dir)
            self.assertTrue(os.path.exists(result_path))


class TestSaveCheckpointSharded(unittest.TestCase):
    def test_sharded_save_and_reload(self) -> None:
        """Force sharding with a small max_shard_size and verify round-trip."""
        source: EncoderDecoder = EncoderDecoder()
        target: EncoderDecoder = EncoderDecoder()

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path: str = save_checkpoint(source, tmpdir, max_shard_size_bytes=64)
            self.assertTrue(result_path.endswith("model.safetensors.index.json"))
            self.assertTrue(os.path.exists(result_path))

            # Verify index file is valid
            with open(result_path, "r") as f:
                index_data: dict = json.load(f)
            self.assertIn("weight_map", index_data)
            self.assertEqual(
                set(index_data["weight_map"].keys()),
                set(source.state_dict().keys()),
            )

            # Verify multiple shard files exist
            shard_files: List[str] = [
                f for f in os.listdir(tmpdir) if f.startswith("model-") and f.endswith(".safetensors")
            ]
            self.assertGreater(len(shard_files), 1)

            # Load back and verify
            config: CheckpointConfig = CheckpointConfig(path=tmpdir)
            load_checkpoint(config, target)

        for key in source.state_dict():
            torch.testing.assert_close(
                target.state_dict()[key], source.state_dict()[key]
            )

    def test_sharded_save_with_metadata(self) -> None:
        """Metadata should be stored in both the index and the first shard."""
        model: EncoderDecoder = EncoderDecoder()
        metadata: Dict[str, str] = {"experiment": "test_run"}

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path: str = save_checkpoint(model, tmpdir, metadata=metadata, max_shard_size_bytes=64)

            # Check index metadata
            index_metadata: Dict[str, str] = read_checkpoint_metadata(result_path)
            self.assertEqual(index_metadata["experiment"], "test_run")

            # Check first shard has metadata
            shard_files: List[str] = sorted(
                f for f in os.listdir(tmpdir) if f.startswith("model-") and f.endswith(".safetensors")
            )
            first_shard_path: str = os.path.join(tmpdir, shard_files[0])
            shard_meta: Dict[str, str] = read_checkpoint_metadata(first_shard_path)
            self.assertEqual(shard_meta["experiment"], "test_run")

    def test_sharded_save_partial_load(self) -> None:
        """Save sharded, then load only the encoder via mapping."""
        source: EncoderDecoder = EncoderDecoder()
        target: EncoderDecoder = EncoderDecoder()
        original_decoder: torch.Tensor = target.decoder.weight.clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(source, tmpdir, max_shard_size_bytes=64)

            config: CheckpointConfig = CheckpointConfig(
                mappings=[
                    CheckpointMapping(
                        path=tmpdir,
                        source_prefix="encoder.",
                        target_prefix="encoder.",
                    )
                ]
            )
            load_checkpoint(config, target)

        torch.testing.assert_close(target.encoder.weight, source.encoder.weight)
        torch.testing.assert_close(target.decoder.weight, original_decoder)


class TestSaveLoadTrainingStateRoundTrip(unittest.TestCase):
    def test_optimizer_and_scheduler_round_trip(self) -> None:
        """Save optimizer + scheduler state, load into fresh instances, verify match."""
        model: EncoderDecoder = EncoderDecoder()
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )

        # Simulate a few training steps to populate optimizer state
        for _ in range(5):
            loss: torch.Tensor = model(torch.randn(2, 4)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        original_opt_state: dict = optimizer.state_dict()
        original_sched_state: dict = scheduler.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_training_state(tmpdir, optimizer=optimizer, scheduler=scheduler)

            # Fresh optimizer + scheduler
            model2: EncoderDecoder = EncoderDecoder()
            opt2: torch.optim.Optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
            sched2: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(
                opt2, step_size=10, gamma=0.5
            )

            load_training_state(tmpdir, optimizer=opt2, scheduler=sched2)

        self.assertEqual(opt2.state_dict()["param_groups"], original_opt_state["param_groups"])
        self.assertEqual(sched2.state_dict(), original_sched_state)


class TestLoadTrainingStateFromFile(unittest.TestCase):
    def test_load_from_explicit_file_path(self) -> None:
        """Load training state by providing the .pt file path directly."""
        model: EncoderDecoder = EncoderDecoder()
        optimizer: torch.optim.Optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        loss: torch.Tensor = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path: str = save_training_state(tmpdir, optimizer=optimizer)
            self.assertTrue(saved_path.endswith("training_state.pt"))

            model2: EncoderDecoder = EncoderDecoder()
            opt2: torch.optim.Optimizer = torch.optim.SGD(model2.parameters(), lr=0.1)
            load_training_state(saved_path, optimizer=opt2)

        self.assertEqual(
            opt2.state_dict()["param_groups"], optimizer.state_dict()["param_groups"]
        )


class TestLoadTrainingStateSelective(unittest.TestCase):
    def test_load_only_optimizer(self) -> None:
        """Load with only optimizer, no scheduler — should not raise."""
        model: EncoderDecoder = EncoderDecoder()
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10
        )

        loss: torch.Tensor = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_training_state(tmpdir, optimizer=optimizer, scheduler=scheduler)

            model2: EncoderDecoder = EncoderDecoder()
            opt2: torch.optim.Optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
            state: dict = load_training_state(tmpdir, optimizer=opt2)

        self.assertIn("optimizer", state)
        self.assertIn("scheduler", state)
        self.assertEqual(
            opt2.state_dict()["param_groups"], optimizer.state_dict()["param_groups"]
        )

    def test_load_only_scheduler(self) -> None:
        """Load with only scheduler, no optimizer — should not raise."""
        model: EncoderDecoder = EncoderDecoder()
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10
        )
        scheduler.step()
        scheduler.step()

        original_sched_state: dict = scheduler.state_dict()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_training_state(tmpdir, optimizer=optimizer, scheduler=scheduler)

            opt_fresh: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            sched_fresh: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(
                opt_fresh, step_size=10
            )
            load_training_state(tmpdir, scheduler=sched_fresh)

        self.assertEqual(sched_fresh.state_dict(), original_sched_state)


class TestSaveTrainingStatePartial(unittest.TestCase):
    def test_save_optimizer_only(self) -> None:
        """Save with only optimizer (no scheduler) and load back."""
        model: EncoderDecoder = EncoderDecoder()
        optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        loss: torch.Tensor = model(torch.randn(2, 4)).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_training_state(tmpdir, optimizer=optimizer)
            state: dict = load_training_state(tmpdir)

        self.assertIn("optimizer", state)
        self.assertNotIn("scheduler", state)


if __name__ == "__main__":
    unittest.main()
