from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict

import cv2
import torch

from brain_factory.scaffold.output_container_base import OutputContainerBase

logger: logging.Logger = logging.getLogger(__name__)


class ExtractedVideoContainer(OutputContainerBase):
    """Output container that persists generated videos and extracted frames.

    Expects ``predictions`` to contain:
      - ``video_bytes`` (``bytes``): Raw video data (e.g. mp4).
      - ``action_name`` (``str``, optional): Used as a subdirectory name.

    Depending on *output_format*, the container writes:
      - ``video.mp4`` — the raw video file.
      - ``frames/frame_NNNN.{frame_format}`` — one image per video frame.

    Args:
        output_format: ``"video"``, ``"frames"``, or ``"both"``.
        frame_format: Image extension for extracted frames (e.g. ``"png"``).
        output_folder: Passed to ``OutputContainerBase``.
    """

    def __init__(
        self,
        output_format: str,
        frame_format: str,
        output_folder: str = "",
    ) -> None:
        super().__init__(output_folder=output_folder)
        self._output_format: str = output_format
        self._frame_format: str = frame_format

    def save(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
        output_dir: str,
        device: torch.device,
    ) -> None:
        video_bytes: bytes | None = predictions.get("video_bytes")
        if video_bytes is None:
            return

        action_name: str = predictions.get("action_name", "")
        if action_name:
            output_dir = os.path.join(output_dir, action_name)
            os.makedirs(output_dir, exist_ok=True)

        if self._output_format in ("video", "both"):
            video_path: str = os.path.join(output_dir, "video.mp4")
            with open(video_path, "wb") as f:
                f.write(video_bytes)

        if self._output_format in ("frames", "both"):
            self._extract_frames(video_bytes, output_dir)

    def _extract_frames(self, video_bytes: bytes, output_dir: str) -> None:
        frames_dir: str = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path: str = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            frame_idx: int = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_path: str = os.path.join(
                    frames_dir, f"frame_{frame_idx:04d}.{self._frame_format}"
                )
                cv2.imwrite(out_path, frame)
                frame_idx += 1
            cap.release()
            logger.info(f"Extracted {frame_idx} frames to {frames_dir}")
        finally:
            os.unlink(tmp_path)
