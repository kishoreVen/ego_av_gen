from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from brain_factory.scaffold.dataset_base import DatasetBase


class _RecordingIndex:
    """One recording's worth of data: video.mov + manifest.json +
    frames.jsonl + raw/ depth-confidence dumps, written by the ObjectRecorder
    iOS app (see data_collector/object_recorder)."""

    def __init__(self, recording_dir: Path, name: str, mode: str) -> None:
        self.dir = recording_dir
        self.name = name
        self.mode = mode
        with open(recording_dir / "manifest.json") as f:
            self.manifest: Dict[str, Any] = json.load(f)
        self.has_depth: bool = bool(self.manifest.get("has_lidar_depth"))
        self.depth_width: Optional[int] = self.manifest.get("depth_width")
        self.depth_height: Optional[int] = self.manifest.get("depth_height")
        self.frames: List[Dict[str, Any]] = self._read_frames(recording_dir / "frames.jsonl")
        self._video_frame_indices: List[int] = self._align_frames_to_video(recording_dir / "video.mov")
        self._cap: Optional[cv2.VideoCapture] = None

    @staticmethod
    def _read_frames(path: Path) -> List[Dict[str, Any]]:
        frames = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        return frames

    def _align_frames_to_video(self, video_path: Path) -> List[int]:
        """Maps each frames.jsonl entry to a video.mov frame number.

        frames.jsonl entries and video.mov frames are NOT guaranteed 1:1 (the
        video encoder can momentarily fall behind the metadata stream) -- see
        experiments/ios_data_viewer/loader.py for the original report of this
        behavior. So instead of trusting frame_index as a video frame number,
        we do one full decode pass to record every video frame's presentation
        time, then match each metadata frame to the video frame with the
        closest timestamp.
        """
        if not self.frames:
            return []

        cap = cv2.VideoCapture(str(video_path))
        presentation_times: List[float] = []
        try:
            while cap.grab():
                presentation_times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        finally:
            cap.release()
        if not presentation_times:
            raise RuntimeError(f"Could not decode any frames from {video_path}")

        # ARKit anchors the video's own pts at 0 using the first metadata
        # frame's timestamp as the origin (see ARKitCaptureController); Max
        # FPS mode does the same relative to its own first sample buffer.
        start = self.frames[0]["timestamp"]
        indices = []
        for frame in self.frames:
            target = frame["timestamp"] - start
            i = bisect_left(presentation_times, target)
            if i <= 0:
                best = 0
            elif i >= len(presentation_times):
                best = len(presentation_times) - 1
            else:
                before, after = presentation_times[i - 1], presentation_times[i]
                best = i if (after - target) < (target - before) else i - 1
            indices.append(best)
        return indices

    def _capture(self) -> cv2.VideoCapture:
        if self._cap is None:
            self._cap = cv2.VideoCapture(str(self.dir / "video.mov"))
        return self._cap

    def read_image_rgb(self, local_idx: int) -> np.ndarray:
        video_frame = self._video_frame_indices[local_idx]
        cap = self._capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read video frame {video_frame} from {self.dir / 'video.mov'}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def read_depth(self, raw_relative_path: str) -> np.ndarray:
        raw = np.fromfile(self.dir / raw_relative_path, dtype=np.float32)
        return raw.reshape(self.depth_height, self.depth_width)

    def read_confidence(self, raw_relative_path: str) -> np.ndarray:
        raw = np.fromfile(self.dir / raw_relative_path, dtype=np.uint8)
        return raw.reshape(self.depth_height, self.depth_width)


class ObjectRecorderSessionDataset(DatasetBase):
    """PyTorch Dataset over one ObjectRecorder capture session (see
    data_collector/object_recorder), flattening frames across all of the
    session's recordings into a single indexable sequence.

    A session directory looks like:

        <session_dir>/
            session_manifest.json      # [{"name": "01_arkit", "mode": "arkit"}, ...]
            01_arkit/
                video.mov, manifest.json, frames.jsonl, raw/
            02_maxfps/
                ...

    A session is captures of one physical object, so every item also carries
    `object_name` -- the name given at "New Session" time (`label` in
    session_manifest.json), falling back to the session folder name for
    older sessions that predate that field.

    Schema notes:
    - `intrinsics` is always present (zero-filled + `has_intrinsics=False`
      when the underlying frame didn't carry one -- observed to happen for
      Max FPS recordings on some devices/formats).
    - `extrinsics`/`has_extrinsics` are only included in the item dict if at
      least one recording in this dataset instance is ARKit mode (Max FPS
      never has extrinsics -- see the app's README). Likewise `depth`,
      `confidence`, `has_depth`, `has_confidence` are only included if at
      least one recording actually has LiDAR depth.
    - `light_estimate`/`exposure` are JSON strings (empty string if absent)
      rather than tensors, since their fields vary by device (e.g. only
      LiDAR devices report directional light) -- `json.loads()` them if
      non-empty.
    - ARKit and Max FPS frames come from different camera formats and are
      NOT the same image shape by default (e.g. landscape 1920x1440 vs
      portrait 1440x1920 in the sample aa_battery session) -- pass
      `image_size` to resize everything to a common shape if you plan to
      batch across mixed modes, or filter with `modes=["arkit"]` /
      `modes=["maxfps"]` to keep a batch's images natively uniform.
    - Random-access reads seek into a compressed video per call, which is
      not fast for training-scale iteration (no frame-cache pass exists
      yet -- `cached_get_item` currently just calls `pre_cache_get_item`,
      same as the other datasets in this package).
    """

    def __init__(
        self,
        session_dir: str,
        modes: Optional[Sequence[str]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        use_cache: bool = False,
    ) -> None:
        super().__init__(use_cache=use_cache)
        self._session_dir = Path(session_dir)
        self._image_size = image_size
        self._object_name, self._recordings = self._load_session(modes)
        self._flat_index: List[Tuple[int, int]] = [
            (rec_idx, local_idx)
            for rec_idx, rec in enumerate(self._recordings)
            for local_idx in range(len(rec.frames))
        ]

        self._has_arkit = any(rec.mode == "arkit" for rec in self._recordings)
        depth_recordings = [rec for rec in self._recordings if rec.has_depth]
        self._has_depth = bool(depth_recordings)
        self._depth_shape: Optional[Tuple[int, int]] = (
            (depth_recordings[0].depth_height, depth_recordings[0].depth_width) if depth_recordings else None
        )

    def _load_session(self, modes: Optional[Sequence[str]]) -> Tuple[str, List[_RecordingIndex]]:
        manifest_path = self._session_dir / "session_manifest.json"
        with open(manifest_path) as f:
            session_manifest = json.load(f)

        # A session is captures of one physical object; the label the user
        # gave it at "New Session" time is the object's identity. Older
        # session_manifest.json files (predating that field) don't have it,
        # so fall back to the session folder name -- e.g. "aa_battery".
        object_name = session_manifest.get("label") or self._session_dir.name

        recordings = []
        for entry in session_manifest["recordings"]:
            if modes is not None and entry["mode"] not in modes:
                continue
            recordings.append(_RecordingIndex(self._session_dir / entry["name"], entry["name"], entry["mode"]))
        if not recordings:
            raise ValueError(f"No recordings found under {self._session_dir} (modes={modes})")
        return object_name, recordings

    @property
    def object_name(self) -> str:
        return self._object_name

    def __len__(self) -> int:
        return len(self._flat_index)

    def pre_cache_get_item(self, idx: int) -> Dict[str, Any]:
        rec_idx, local_idx = self._flat_index[idx]
        rec = self._recordings[rec_idx]
        frame = rec.frames[local_idx]

        image = rec.read_image_rgb(local_idx)
        if self._image_size is not None:
            height, width = self._image_size
            image = cv2.resize(image, (width, height))

        item: Dict[str, Any] = {
            "image": torch.from_numpy(image.astype(np.float32) / 255.0),
            "intrinsics": self._matrix_or_zero(frame.get("intrinsics"), (3, 3)),
            "has_intrinsics": torch.tensor(frame.get("intrinsics") is not None),
            "object_name": self._object_name,
            "mode": rec.mode,
            "recording_name": rec.name,
            "frame_index": torch.tensor(frame["frame_index"], dtype=torch.int64),
            "timestamp": torch.tensor(frame["timestamp"], dtype=torch.float64),
            "light_estimate": self._json_or_empty(frame.get("light_estimate")),
            "exposure": self._json_or_empty(frame.get("exposure")),
        }

        if self._has_arkit:
            item["extrinsics"] = self._matrix_or_zero(frame.get("extrinsics"), (4, 4))
            item["has_extrinsics"] = torch.tensor(frame.get("extrinsics") is not None)

        if self._has_depth:
            assert self._depth_shape is not None
            depth_file = frame.get("depth_file")
            confidence_file = frame.get("confidence_file")
            item["depth"] = (
                torch.from_numpy(rec.read_depth(depth_file))
                if depth_file
                else torch.zeros(self._depth_shape, dtype=torch.float32)
            )
            item["has_depth"] = torch.tensor(depth_file is not None)
            item["confidence"] = (
                torch.from_numpy(rec.read_confidence(confidence_file))
                if confidence_file
                else torch.zeros(self._depth_shape, dtype=torch.uint8)
            )
            item["has_confidence"] = torch.tensor(confidence_file is not None)

        return item

    def cached_get_item(self, idx: int) -> Dict[str, Any]:
        return self.pre_cache_get_item(idx)

    @staticmethod
    def _matrix_or_zero(value: Optional[List[List[float]]], shape: Tuple[int, int]) -> torch.Tensor:
        if value is None:
            return torch.zeros(shape, dtype=torch.float32)
        return torch.tensor(value, dtype=torch.float32)

    @staticmethod
    def _json_or_empty(value: Optional[Dict[str, Any]]) -> str:
        return json.dumps(value) if value is not None else ""
