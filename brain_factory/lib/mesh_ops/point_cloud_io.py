"""Point-cloud file reading and normalization/resampling helpers.

Backs brain_factory.lib.data_structures.mesh.Mesh's point-cloud loading path
(is_point_cloud_file, load_rope_points). Port of meshflow/meshflow/utils/mesh.py
(~/code/meshflow). Meshflow's original code is distributed under Meta's FAIR
Noncommercial Research License (noncommercial research use only); this repo is
Apache-2.0, so confirm license compatibility before any commercial use of this
module.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import trimesh

POINT_CLOUD_EXTS = (".pcd", ".xyz", ".pts", ".npy", ".npz")


def normalize_points(
    points: torch.Tensor,
    normalize_by: str = "bsphere",
    size: float = 2.0,
) -> torch.Tensor:
    if normalize_by == "bbox":
        bbox_min, bbox_max = points.min(0).values, points.max(0).values
        return (points - (bbox_min + bbox_max) / 2) * (size / (bbox_max - bbox_min).max())
    if normalize_by == "bsphere":
        center = (points.min(0).values + points.max(0).values) / 2
        radius = torch.linalg.norm(points - center, dim=-1).max() * 2
        return (points - center) * (size / radius)
    raise NotImplementedError(f"Invalid normalize_by: {normalize_by}")


def resample_point_cloud(points: torch.Tensor, num_points: int) -> torch.Tensor:
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty")
    if points.shape[0] < num_points:
        raise ValueError(
            f"Point cloud must have at least {num_points} points, got {int(points.shape[0])}"
        )
    idx = torch.randperm(points.shape[0], device=points.device)[:num_points]
    return points[idx]


def read_point_cloud_file(filename: str) -> torch.Tensor:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".npy":
        arr = np.load(filename)
    elif ext == ".npz":
        data = np.load(filename)
        for key in ("points", "pts", "xyz", "vertices", "data"):
            if key in data:
                arr = data[key]
                break
        else:
            raise ValueError(f"No point array found in {filename}")
    elif ext in (".xyz", ".pts"):
        arr = np.loadtxt(filename, dtype=np.float32)
    elif ext == ".pcd":
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        start = 0
        for i, line in enumerate(lines):
            header = line.upper()
            if header.startswith("POINTS") or header.startswith("WIDTH"):
                start = i + 1
                break
        arr = np.loadtxt(lines[start:], dtype=np.float32)
    else:
        raise NotImplementedError(f"Unsupported point cloud format: {ext}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] % 3 != 0:
            raise ValueError(f"Invalid point array shape in {filename}")
        arr = arr.reshape(-1, 3)
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 coordinates per point in {filename}")
    return torch.from_numpy(arr[..., :3].copy())


def try_read_point_cloud_ply(filename: str) -> Optional[torch.Tensor]:
    """Return points if `filename` is a faceless (point-cloud) .ply, else None."""
    loaded = trimesh.load(filename, process=False)
    if isinstance(loaded, trimesh.PointCloud):
        return torch.from_numpy(np.asarray(loaded.vertices, dtype=np.float32))
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) == 0:
        return torch.from_numpy(np.asarray(loaded.vertices, dtype=np.float32))
    if isinstance(loaded, trimesh.Scene):
        points = []
        for geometry in loaded.geometry.values():
            if isinstance(geometry, trimesh.PointCloud):
                points.append(geometry.vertices)
            elif isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) == 0:
                points.append(geometry.vertices)
        if points:
            return torch.from_numpy(np.concatenate(points, axis=0).astype(np.float32))
    return None
