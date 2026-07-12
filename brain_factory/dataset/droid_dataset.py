from __future__ import annotations

import glob
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from brain_factory.scaffold.dataset_base import DatasetBase

# ---------------------------------------------------------------------------
# DROID TFRecord schema (inspected from actual records)
#
# Each episode is ONE tf.train.Example.  Step values are packed flat:
#   float fields  → float_list  of length n_steps * dim  (cast to float32)
#   bytes fields  → bytes_list  of length n_steps        (JPEG or string)
#   bool fields   → int64_list  of length n_steps
# ---------------------------------------------------------------------------

# (feature_key, per-step dimension)
_FLOAT_FIELDS: Dict[str, int] = {
    "steps/action":                          7,
    "steps/observation/joint_position":      7,
    "steps/observation/cartesian_position":  6,
    "steps/observation/gripper_position":    1,
    "steps/reward":                          1,
    "steps/discount":                        1,
}

_BYTES_FIELDS = [
    "steps/observation/wrist_image_left",
    "steps/observation/exterior_image_1_left",
    "steps/observation/exterior_image_2_left",
    "steps/language_instruction",
]

_INT64_FIELDS = [
    "steps/is_first",
    "steps/is_last",
    "steps/is_terminal",
]


class DroidDataset(DatasetBase):
    """Dataset for DROID (r2d2_faceblur) episodes stored as RLDS TFRecords.

    Reads with tensorflow directly; does NOT use tensorflow-datasets, avoiding
    the protobuf 4.x / tfds FieldDescriptor.label compatibility bug.

    Accepts either the versioned directory (e.g. data/droid_100/1.0.0) or the
    dataset root (e.g. data/droid_100) and auto-detects the version dir.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Optional[Tuple[int, int]] = None,
        max_episodes: Optional[int] = None,
        use_cache: bool = False,
    ) -> None:
        super().__init__(use_cache=use_cache)
        self._image_size = image_size
        self._steps: List[Dict[str, Any]] = []
        self._load(data_dir, split, max_episodes)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _resolve_shard_dir(self, data_dir: str) -> Path:
        p = Path(data_dir)
        if (p / "dataset_info.json").exists():
            return p
        candidates = sorted(p.glob("*/dataset_info.json"))
        if not candidates:
            raise FileNotFoundError(f"No dataset_info.json found under {data_dir}")
        return candidates[0].parent

    def _build_parse_spec(self) -> Dict[str, Any]:
        import tensorflow as tf

        spec: Dict[str, Any] = {
            "episode_metadata/file_path": tf.io.FixedLenFeature([], tf.string, ""),
        }
        for key in _FLOAT_FIELDS:
            spec[key] = tf.io.VarLenFeature(tf.float32)
        for key in _BYTES_FIELDS:
            spec[key] = tf.io.VarLenFeature(tf.string)
        for key in _INT64_FIELDS:
            spec[key] = tf.io.VarLenFeature(tf.int64)
        return spec

    def _load(self, data_dir: str, split: str, max_episodes: Optional[int]) -> None:
        import tensorflow as tf

        shard_dir = self._resolve_shard_dir(data_dir)
        pattern = str(shard_dir / f"*-{split}.tfrecord-*")
        shards = sorted(glob.glob(pattern))
        if not shards:
            raise FileNotFoundError(f"No shards found matching {pattern}")

        spec = self._build_parse_spec()
        raw_ds = tf.data.TFRecordDataset(shards)
        ep_count = 0
        for raw_record in raw_ds:
            if max_episodes is not None and ep_count >= max_episodes:
                break
            parsed = tf.io.parse_single_example(raw_record, spec)
            self._steps.extend(self._unpack_episode(parsed))
            ep_count += 1

    def _unpack_episode(self, parsed: Any) -> List[Dict[str, Any]]:
        import tensorflow as tf

        n_steps = int(tf.sparse.to_dense(parsed["steps/is_first"]).shape[0])

        # float fields: (n_steps * dim,) → (n_steps, dim)
        float_arrays: Dict[str, np.ndarray] = {}
        for key, dim in _FLOAT_FIELDS.items():
            flat = tf.sparse.to_dense(parsed[key]).numpy()  # (n_steps * dim,)
            float_arrays[key] = flat.reshape(n_steps, dim)

        # bytes fields: (n_steps,)
        bytes_arrays: Dict[str, np.ndarray] = {}
        for key in _BYTES_FIELDS:
            bytes_arrays[key] = tf.sparse.to_dense(parsed[key]).numpy()  # (n_steps,)

        # int64 fields: (n_steps,)
        int_arrays: Dict[str, np.ndarray] = {}
        for key in _INT64_FIELDS:
            int_arrays[key] = tf.sparse.to_dense(parsed[key]).numpy()  # (n_steps,)

        steps = []
        for i in range(n_steps):
            step: Dict[str, Any] = {}
            for key in _FLOAT_FIELDS:
                step[key] = float_arrays[key][i]       # float32 ndarray (dim,)
            for key in _BYTES_FIELDS:
                step[key] = bytes_arrays[key][i]        # bytes
            for key in _INT64_FIELDS:
                step[key] = bool(int_arrays[key][i])
            steps.append(step)
        return steps

    # ------------------------------------------------------------------
    # DatasetBase interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._steps)

    def pre_cache_get_item(self, idx: int) -> Dict[str, Any]:
        raw = self._steps[idx]
        return {
            "wrist_image":      self._jpeg_to_tensor(raw["steps/observation/wrist_image_left"]),
            "exterior_image_1": self._jpeg_to_tensor(raw["steps/observation/exterior_image_1_left"]),
            "exterior_image_2": self._jpeg_to_tensor(raw["steps/observation/exterior_image_2_left"]),
            "joint_position":   torch.from_numpy(raw["steps/observation/joint_position"]),
            "cartesian_position": torch.from_numpy(raw["steps/observation/cartesian_position"]),
            "gripper_position": torch.from_numpy(raw["steps/observation/gripper_position"]),
            "action":           torch.from_numpy(raw["steps/action"]),
            "language_instruction": raw["steps/language_instruction"].decode("utf-8"),
            "is_first":  raw["steps/is_first"],
            "is_last":   raw["steps/is_last"],
        }

    def cached_get_item(self, idx: int) -> Dict[str, Any]:
        return self.pre_cache_get_item(idx)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _jpeg_to_tensor(self, jpeg_bytes: bytes) -> torch.Tensor:
        """JPEG bytes → float32 HWC tensor in [0, 1], optionally resized."""
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        if self._image_size is not None:
            img = img.resize((self._image_size[1], self._image_size[0]))
        return torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
