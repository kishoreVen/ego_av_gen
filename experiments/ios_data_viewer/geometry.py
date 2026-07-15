"""Depth unprojection matching ARKit's coordinate conventions.

ARKit camera-local space: +X right, +Y up, +Z toward the viewer (i.e. the
camera looks down its local -Z axis). `ARFrame.camera.transform` maps this
local space to world space. Depth values are the distance from the camera
to the scene, i.e. along -Z.
"""
from __future__ import annotations

import numpy as np


def unproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    full_width: int,
    full_height: int,
    confidence: np.ndarray | None = None,
    min_confidence: int = 2,
    stride: int = 2,
    max_depth: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unprojects a depth map into world-space points.

    `intrinsics` is the full-resolution RGB camera intrinsic matrix (3x3);
    it's rescaled here to the depth map's (lower) resolution. `extrinsics`
    is the camera-to-world transform for the same frame.

    Returns (points_world[N, 3] float32, pixel_uv[N, 2] float32) where
    pixel_uv are full-resolution pixel coordinates, useful for sampling a
    matching color from the RGB video frame.
    """
    h, w = depth.shape
    scale_x = w / full_width
    scale_y = h / full_height
    fx = intrinsics[0, 0] * scale_x
    fy = intrinsics[1, 1] * scale_y
    cx = intrinsics[0, 2] * scale_x
    cy = intrinsics[1, 2] * scale_y

    us, vs = np.meshgrid(np.arange(0, w, stride), np.arange(0, h, stride))
    us = us.ravel()
    vs = vs.ravel()
    d = depth[vs, us]

    valid = (d > 0) & (d < max_depth)
    if confidence is not None:
        valid &= confidence[vs, us] >= min_confidence

    us, vs, d = us[valid], vs[valid], d[valid]
    if d.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    x_local = (us - cx) / fx * d
    y_local = -(vs - cy) / fy * d
    z_local = -d

    local = np.stack([x_local, y_local, z_local, np.ones_like(d)], axis=1)  # [N, 4]
    world = (extrinsics @ local.T).T[:, :3]

    pixel_uv = np.stack([us / scale_x, vs / scale_y], axis=1)
    return world.astype(np.float32), pixel_uv.astype(np.float32)
