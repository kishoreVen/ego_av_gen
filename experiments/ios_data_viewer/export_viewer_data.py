#!/usr/bin/env python3
"""Exports an ObjectRecorder session into the format the browser viewer
(viewer/index.html) consumes:

  - video.mov, copied verbatim, for direct playback
  - depth/depth_<frame_index>.png, a colorized heatmap per depth frame
    (warm/red = near, cool/blue = far), for frame-by-frame inspection
  - trajectory.json, one entry per posed frame (position/rotation/timestamp
    relative to video start, and a depth_image path when available)
  - positions.f32 / colors.u8, a downsampled point cloud unprojected from
    depth and colored by sampling the nearest video frame (by timestamp,
    not index -- frames.jsonl entries and video frames are not guaranteed
    1:1 if the encoder ever falls behind)

Usage:
    python export_viewer_data.py /path/to/session_arkit_20260714_213402
    python serve.py   # then open the printed URL
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from geometry import unproject_depth
from loader import Session

VIEWER_DATA_DIR = Path(__file__).parent / "viewer" / "data"


def _load_video_index(session: Session) -> tuple[np.ndarray, list[np.ndarray]]:
    """Decodes video.mov once into (timestamps[N] seconds, frames[N] BGR),
    sorted by time (video.mov is written in presentation-time order)."""
    times: list[float] = []
    frames: list[np.ndarray] = []
    for t, frame in session.video_frames():
        times.append(t)
        frames.append(frame)
    return np.array(times, dtype=np.float64), frames


def _nearest_video_frame(video_times: np.ndarray, video_frames: list[np.ndarray], t: float) -> np.ndarray | None:
    if len(video_times) == 0:
        return None
    idx = int(np.searchsorted(video_times, t))
    idx = min(idx, len(video_times) - 1)
    if idx > 0 and abs(video_times[idx - 1] - t) < abs(video_times[idx] - t):
        idx -= 1
    return video_frames[idx]


def _export_video(source: Path, out_dir: Path) -> bool:
    """Transcodes to H.264/mp4 so it plays in any browser (source is HEVC in
    ARKit mode, which Chrome/Chromium's <video> element cannot decode). Falls
    back to a verbatim copy if ffmpeg isn't available, with a warning."""
    if shutil.which("ffmpeg") is None:
        print("warning: ffmpeg not found on PATH, copying video verbatim -- "
              "it may not play in Chrome/Chromium if it's HEVC-encoded.")
        shutil.copyfile(source, out_dir / "video.mp4")
        return True
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            "-movflags", "+faststart",
            str(out_dir / "video.mp4"),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"warning: ffmpeg transcode failed, falling back to verbatim copy:\n{result.stderr[-500:]}")
        shutil.copyfile(source, out_dir / "video.mp4")
    return True


def _colorize_depth(depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Near = warm (red), far = cool (blue); invalid/no-return pixels = dark gray."""
    valid = depth > 0
    norm = np.clip(depth / max_depth, 0, 1)
    inverted_gray = ((1.0 - norm) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(inverted_gray, cv2.COLORMAP_JET)
    colored[~valid] = (30, 30, 30)
    return colored


def export_session(
    session_dir: Path,
    out_name: str,
    max_points: int,
    frame_stride: int,
    pixel_stride: int,
    min_confidence: int,
    max_depth: float,
) -> None:
    session = Session(session_dir)
    out_dir = VIEWER_DATA_DIR / out_name
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)

    t0 = session.start_timestamp
    video_times, video_frames = _load_video_index(session)
    has_video = len(video_frames) > 0

    if session.video_path.exists():
        _export_video(session.video_path, out_dir)

    # --- per-frame depth heatmaps (all frames, not strided -- these are small
    # and drive frame-by-frame scrubbing in the inspector tab) ---
    depth_image_by_frame: dict[int, str] = {}
    for frame in session.frames:
        if not frame.depth_file:
            continue
        depth = session.depth_map(frame)
        if depth is None:
            continue
        colored = _colorize_depth(depth, max_depth)
        filename = f"depth_{frame.frame_index:06d}.png"
        cv2.imwrite(str(out_dir / "depth" / filename), colored)
        depth_image_by_frame[frame.frame_index] = f"depth/{filename}"

    # --- trajectory (camera poses), timestamps relative to video start ---
    trajectory = [
        {
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp - t0,
            "position": frame.extrinsics[:3, 3].tolist(),
            "rotation": frame.extrinsics[:3, :3].flatten().tolist(),
            "depth_image": depth_image_by_frame.get(frame.frame_index),
        }
        for frame in session.frames
        if frame.extrinsics is not None
    ]
    (out_dir / "trajectory.json").write_text(json.dumps(trajectory))

    # --- point cloud, colored by the nearest video frame at each depth frame's timestamp ---
    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    frames_with_depth = [fr for fr in session.frames if fr.depth_file][::frame_stride]

    for frame in frames_with_depth:
        depth = session.depth_map(frame)
        confidence = session.confidence_map(frame)
        if depth is None or frame.extrinsics is None or frame.intrinsics is None:
            continue

        world, pixel_uv = unproject_depth(
            depth,
            frame.intrinsics,
            frame.extrinsics,
            full_width=session.width,
            full_height=session.height,
            confidence=confidence,
            min_confidence=min_confidence,
            stride=pixel_stride,
            max_depth=max_depth,
        )
        if world.shape[0] == 0:
            continue

        img = _nearest_video_frame(video_times, video_frames, frame.timestamp - t0) if has_video else None
        if img is not None:
            u = np.clip(pixel_uv[:, 0].astype(np.int32), 0, session.width - 1)
            v = np.clip(pixel_uv[:, 1].astype(np.int32), 0, session.height - 1)
            rgb = img[v, u][:, ::-1]
        else:
            rgb = np.full((world.shape[0], 3), 180, dtype=np.uint8)

        positions.append(world)
        colors.append(rgb.astype(np.uint8))

    if positions:
        all_positions = np.concatenate(positions, axis=0)
        all_colors = np.concatenate(colors, axis=0)
    else:
        all_positions = np.zeros((0, 3), dtype=np.float32)
        all_colors = np.zeros((0, 3), dtype=np.uint8)

    if all_positions.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(all_positions.shape[0], size=max_points, replace=False)
        all_positions = all_positions[idx]
        all_colors = all_colors[idx]

    all_positions.astype(np.float32).tofile(out_dir / "positions.f32")
    all_colors.astype(np.uint8).tofile(out_dir / "colors.u8")

    meta = {
        "mode": session.mode,
        "point_count": int(all_positions.shape[0]),
        "pose_count": len(trajectory),
        "depth_frame_count": len(depth_image_by_frame),
        "width": session.width,
        "height": session.height,
        "has_video": has_video,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    _update_index(out_name, meta)
    print(
        f"Exported {meta['point_count']} points, {meta['pose_count']} poses, "
        f"{meta['depth_frame_count']} depth frames -> {out_dir}"
    )


def _update_index(name: str, meta: dict) -> None:
    index_path = VIEWER_DATA_DIR / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [entry for entry in index if entry["name"] != name]
    index.append({"name": name, **meta})
    index_path.write_text(json.dumps(index, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="Path to a session_<mode>_<timestamp> folder")
    parser.add_argument("--name", help="Name to export under (defaults to the session folder name)")
    parser.add_argument("--max-points", type=int, default=400_000)
    parser.add_argument("--frame-stride", type=int, default=3, help="Use every Nth depth frame for the point cloud")
    parser.add_argument("--pixel-stride", type=int, default=2, help="Sample every Nth pixel in each depth map")
    parser.add_argument("--min-confidence", type=int, default=2, help="0=low 1=medium 2=high")
    parser.add_argument("--max-depth", type=float, default=3.0, help="Discard/clip depth beyond this many meters")
    args = parser.parse_args()

    export_session(
        args.session_dir,
        out_name=args.name or args.session_dir.name,
        max_points=args.max_points,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        min_confidence=args.min_confidence,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
