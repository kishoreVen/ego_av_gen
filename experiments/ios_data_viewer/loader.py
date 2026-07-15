"""Loader for ObjectRecorder iOS app session output (ARKit or Max-FPS mode).

A session directory (pulled from the phone, see README.md) looks like:

    session_<mode>_<timestamp>/
        video.mov
        manifest.json
        frames.jsonl
        raw/
            depth_000123.bin
            confidence_000123.bin

Usage:
    from loader import Session
    session = Session("session_arkit_20260714_213402")
    for frame in session.frames:
        depth = session.depth_map(frame)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


@dataclass
class Frame:
    frame_index: int
    timestamp: float
    intrinsics: Optional[np.ndarray]  # 3x3, full-resolution RGB intrinsics
    extrinsics: Optional[np.ndarray]  # 4x4 camera-to-world transform (ARKit mode only)
    depth_file: Optional[str]
    confidence_file: Optional[str]
    light_estimate: Optional[dict]
    exposure: Optional[dict]


class Session:
    """Reads a session_<mode>_<timestamp>/ folder written by the ObjectRecorder app."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with open(self.path / "manifest.json") as f:
            self.manifest: dict = json.load(f)
        self.frames: list[Frame] = list(self._read_frames())

    @property
    def mode(self) -> str:
        return self.manifest.get("mode", "unknown")

    @property
    def video_path(self) -> Path:
        return self.path / "video.mov"

    @property
    def has_depth(self) -> bool:
        return bool(self.manifest.get("has_lidar_depth"))

    @property
    def width(self) -> int:
        return self.manifest["width"]

    @property
    def height(self) -> int:
        return self.manifest["height"]

    def _read_frames(self) -> Iterator[Frame]:
        frames_path = self.path / "frames.jsonl"
        with open(frames_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield Frame(
                    frame_index=obj["frame_index"],
                    timestamp=obj["timestamp"],
                    intrinsics=np.array(obj["intrinsics"], dtype=np.float32) if obj.get("intrinsics") else None,
                    extrinsics=np.array(obj["extrinsics"], dtype=np.float32) if obj.get("extrinsics") else None,
                    depth_file=obj.get("depth_file"),
                    confidence_file=obj.get("confidence_file"),
                    light_estimate=obj.get("light_estimate"),
                    exposure=obj.get("exposure"),
                )

    def depth_map(self, frame: Frame) -> Optional[np.ndarray]:
        """Float32 depth in meters, shape (depth_height, depth_width)."""
        if not frame.depth_file:
            return None
        w, h = self.manifest["depth_width"], self.manifest["depth_height"]
        raw = np.fromfile(self.path / frame.depth_file, dtype=np.float32)
        return raw.reshape(h, w)

    def confidence_map(self, frame: Frame) -> Optional[np.ndarray]:
        """UInt8 confidence (0=low, 1=medium, 2=high), shape (depth_height, depth_width)."""
        if not frame.confidence_file:
            return None
        w, h = self.manifest["depth_width"], self.manifest["depth_height"]
        raw = np.fromfile(self.path / frame.confidence_file, dtype=np.uint8)
        return raw.reshape(h, w)

    @property
    def start_timestamp(self) -> float:
        """The ARKit uptime of the first frame; frame.timestamp - this equals
        the video's own presentation time in seconds (see ARKitCaptureController,
        which uses exactly this offset as the AVAssetWriter pts)."""
        return self.frames[0].timestamp

    def video_frames(self) -> Iterator[tuple[float, np.ndarray]]:
        """Yields (presentation_time_seconds, bgr_frame) from video.mov, where
        presentation_time_seconds is directly comparable to
        `frame.timestamp - session.start_timestamp` for any Frame. Frame
        metadata (frames.jsonl) and video frames are NOT guaranteed 1:1 (a
        metadata frame can be logged with no corresponding video frame if the
        encoder was momentarily behind) -- match by timestamp, not index."""
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield t, frame
        finally:
            cap.release()
