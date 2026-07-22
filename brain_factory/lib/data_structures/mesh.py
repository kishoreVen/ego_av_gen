"""Mesh data structure: loading, augmentation, and topology feature extraction.

Wraps trimesh; caches derived per-vertex normals and unique edges. Supporting
free functions live under brain_factory.lib.mesh_ops (point_cloud_io, topology,
reconstruction) rather than on this class, matching the split between
data structures and operations in brain_factory/lib.

Port of meshflow/meshflow/utils/mesh.py (~/code/meshflow). Meshflow's original
code is distributed under Meta's FAIR Noncommercial Research License
(noncommercial research use only); this repo is Apache-2.0, so confirm license
compatibility before any commercial use of this module.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from brain_factory.lib.mesh_ops.point_cloud_io import (
    POINT_CLOUD_EXTS,
    normalize_points,
    read_point_cloud_file,
    resample_point_cloud,
    try_read_point_cloud_ply,
)


class Mesh:
    """Mesh class for loading, augmentation, and topology feature extraction."""

    def __init__(self, verts: torch.Tensor, faces: torch.Tensor, device: torch.device | None = None) -> None:
        """Args:
            verts: (N, 3) vertex positions.
            faces: (F, 3) triangle indices.
            device: target device; defaults to verts.device.
        """
        self.device = device if device is not None else verts.device
        self.verts = verts.to(self.device, dtype=torch.float32)
        self.faces = faces.to(self.device, dtype=torch.int64)
        self._v_nrm: Optional[torch.Tensor] = None
        self._edges: Optional[torch.Tensor] = None

    @classmethod
    def is_point_cloud_file(cls, filename: str) -> bool:
        """True for point-cloud formats, including faceless `.ply` point clouds."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in POINT_CLOUD_EXTS:
            return True
        if ext == ".ply":
            return try_read_point_cloud_ply(filename) is not None
        return False

    @classmethod
    def load_mesh(
        cls,
        filename: str,
        normalize: bool = False,
        normalize_by: str = "bbox",
        obj_scale: float = 2.0,
        preprocess: bool = True,
        device: torch.device | None = None,
    ) -> Mesh:
        """Load mesh from disk (.stl/.obj/.ply/.glb/.gltf).

        STL files are rotated -90 degrees about X during load.
        Scene graphs are flattened into a single mesh.

        Args:
            normalize: whether to call normalize() after load.
            normalize_by: "bbox" or "bsphere".
            obj_scale: target extent after normalization (passed as size).
            preprocess: whether to merge vertices and drop degenerate faces.
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext not in (".stl", ".obj", ".ply", ".glb", ".gltf"):
            raise NotImplementedError(f"Unsupported mesh format: {ext}")
        if cls.is_point_cloud_file(filename):
            raise ValueError(
                f"{filename} is a point cloud; use load_rope_points() instead of load_mesh()."
            )

        mesh = trimesh.load(filename, force="mesh", process=False, skip_materials=True)
        if ext == ".stl":
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
        if isinstance(mesh, trimesh.Scene):
            parts = []
            for _, node in mesh.graph.to_flattened().items():
                name = node["geometry"]
                if name in mesh.geometry and isinstance(mesh.geometry[name], trimesh.Trimesh):
                    parts.append(mesh.geometry[name].apply_transform(node["transform"]))
            mesh = trimesh.util.concatenate(parts)

        m = cls(
            torch.tensor(mesh.vertices, dtype=torch.float32),
            torch.tensor(mesh.faces, dtype=torch.int64),
            device=device,
        )
        if preprocess:
            m.preprocess()
        if normalize:
            m.normalize(normalize_by=normalize_by, size=obj_scale)
        return m

    @classmethod
    def load_rope_points(
        cls,
        filename: str,
        n_verts: int,
        normalize_by: str = "bsphere",
        obj_scale: float = 2.0,
        preprocess: bool = True,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Load a mesh or point cloud and return fixed-size points for RoPE conditioning."""
        device = device or torch.device("cpu")
        ext = os.path.splitext(filename)[1].lower()

        if ext in POINT_CLOUD_EXTS:
            points = read_point_cloud_file(filename).to(device)
        elif ext == ".ply":
            points = try_read_point_cloud_ply(filename)
            if points is None:
                mesh = cls.load_mesh(
                    filename,
                    normalize=True,
                    normalize_by=normalize_by,
                    obj_scale=obj_scale,
                    preprocess=preprocess,
                    device=device,
                )
                surface_pts, _ = mesh.sample_surface(n_verts)
                return surface_pts
            points = points.to(device)
        else:
            mesh = cls.load_mesh(
                filename,
                normalize=True,
                normalize_by=normalize_by,
                obj_scale=obj_scale,
                preprocess=preprocess,
                device=device,
            )
            surface_pts, _ = mesh.sample_surface(n_verts)
            return surface_pts

        points = normalize_points(points, normalize_by=normalize_by, size=obj_scale)
        return resample_point_cloud(points, n_verts)

    def to_trimesh(self) -> trimesh.Trimesh:
        """Export to a host-side mesh object (detached from autograd)."""
        return trimesh.Trimesh(
            vertices=self.verts.detach().cpu().numpy(),
            faces=self.faces.detach().cpu().numpy(),
            process=False,
        )

    @property
    def v_nrm(self) -> torch.Tensor:
        """Per-vertex normals (area-weighted face normal sum), shape (N, 3)."""
        if self._v_nrm is None:
            v0, v1, v2 = self.verts[self.faces[:, 0]], self.verts[self.faces[:, 1]], self.verts[self.faces[:, 2]]
            fn = torch.linalg.cross(v1 - v0, v2 - v0)
            v_nrm = torch.zeros_like(self.verts)
            v_nrm.scatter_add_(0, self.faces[:, 0:1].expand(-1, 3), fn)
            v_nrm.scatter_add_(0, self.faces[:, 1:2].expand(-1, 3), fn)
            v_nrm.scatter_add_(0, self.faces[:, 2:3].expand(-1, 3), fn)
            v_nrm = torch.where(
                (v_nrm * v_nrm).sum(-1, keepdim=True) > 1e-20,
                v_nrm,
                torch.tensor([0.0, 0.0, 1.0], device=self.device),
            )
            self._v_nrm = F.normalize(v_nrm, dim=1)
        return self._v_nrm

    @property
    def edges(self) -> torch.Tensor:
        """Unique undirected edges, shape (E, 2)."""
        if self._edges is None:
            e = torch.cat([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]], dim=0)
            self._edges = torch.unique(torch.sort(e, dim=1)[0], dim=0)
        return self._edges

    def preprocess(self) -> Mesh:
        """Clean mesh in-place: merge vertices, remove degenerate/unreferenced geometry."""
        tm = self.to_trimesh()
        tm.process()
        tm.merge_vertices(merge_tex=True, merge_norm=True)
        tm.update_faces(tm.nondegenerate_faces())
        tm.update_faces(tm.unique_faces())
        tm.remove_unreferenced_vertices()
        self.verts = torch.from_numpy(tm.vertices).to(self.verts).float()
        self.faces = torch.from_numpy(tm.faces).to(self.faces).long()
        self._v_nrm = self._edges = None
        return self

    def normalize(self, normalize_by: str = "bsphere", size: float = 2.0) -> Mesh:
        """Center and scale vertices in-place to fit a canonical volume.

        Args:
            normalize_by: "bbox" (axis-aligned box) or "bsphere" (bounding sphere).
            size: target extent along the longest axis / diameter.
        """
        if normalize_by == "bbox":
            bbox_min, bbox_max = self.verts.min(0).values, self.verts.max(0).values
            self.verts = (self.verts - (bbox_min + bbox_max) / 2) * (size / (bbox_max - bbox_min).max())
        elif normalize_by == "bsphere":
            center = (self.verts.min(0).values + self.verts.max(0).values) / 2
            self.verts = (self.verts - center) * (
                size / (torch.linalg.norm(self.verts - center, dim=-1).max() * 2)
            )
        else:
            raise NotImplementedError(f"Invalid normalize_by: {normalize_by}")
        self._v_nrm = self._edges = None
        return self

    def rotate(self, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> Mesh:
        """Apply ZYX Euler rotation (degrees) and return a new Mesh."""
        y, p, r = np.radians(yaw), np.radians(pitch), np.radians(roll)
        cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
        R = torch.tensor(
            [
                [cy * cr + sy * sp * sr, -cy * sr + sy * sp * cr, sy * cp],
                [cp * sr, cp * cr, -sp],
                [-sy * cr + cy * sp * sr, sy * sr + cy * sp * cr, cy * cp],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return Mesh(self.verts @ R.T, self.faces, device=self.device)

    @torch.no_grad()
    def sample_surface(self, num_points: int, return_face_idx: bool = False):
        """Uniformly sample points on the mesh surface.

        Returns:
            points, face_normals; with return_face_idx=True also per-sample face indices.
        """
        tm = self.to_trimesh()
        points, face_idx = tm.sample(num_points, return_index=True)
        points = torch.from_numpy(points.astype(np.float32)).to(self.device)
        normals = torch.from_numpy(tm.face_normals[face_idx].astype(np.float32)).to(self.device)
        if return_face_idx:
            return points, normals, torch.from_numpy(face_idx).to(self.device, dtype=torch.long)
        return points, normals

    @torch.no_grad()
    def build_topology(
        self,
        topology_feats_type: Optional[List[str]] = None,
        n_verts: int = 8192,
        max_degree: int = 50,
    ) -> Dict[str, Any]:
        """Build padded topology tensors for MeshFlow VAE / inference.

        Supported features: verts_normal, neighbor_points, degree, adjacency_matrix.

        Returns:
            Dict with padded_verts, verts_mask, verts, faces, num_verts, and requested features.
        """
        if topology_feats_type is None:
            topology_feats_type = ["verts_normal", "neighbor_points", "degree", "adjacency_matrix"]

        m = self.preprocess().normalize(size=2.0)
        num_verts = int(m.verts.shape[0])
        if num_verts > n_verts:
            raise ValueError(f"Mesh has {num_verts} verts > n_verts={n_verts}")
        if num_verts == 0:
            raise ValueError("Mesh has no vertices")

        device = m.device
        edges, v_nrm = m.edges, m.v_nrm
        n_negative_verts = n_verts - num_verts

        negative_verts = torch.zeros(0, 3, dtype=m.verts.dtype, device=device)
        negative_normals = torch.zeros(0, 3, dtype=v_nrm.dtype, device=device)
        nearest_verts_idx = None
        if n_negative_verts > 0:
            negative_verts, negative_normals, negative_face_idx = m.sample_surface(
                n_negative_verts, return_face_idx=True
            )
            face_vertices = m.faces[negative_face_idx]
            dist2 = (face_vertices - negative_verts[:, None, :]).pow(2).sum(dim=-1)
            nearest_choice = dist2.argmin(dim=1)
            nearest_verts_idx = face_vertices[
                torch.arange(n_negative_verts, device=device),
                nearest_choice,
            ]

        padded = torch.zeros(n_verts, 3, dtype=m.verts.dtype, device=device)
        padded[:num_verts] = m.verts
        if n_negative_verts > 0:
            padded[num_verts:] = negative_verts.to(dtype=m.verts.dtype, device=device)

        verts_mask = (torch.arange(n_verts, device=device) < num_verts).float().unsqueeze(-1)
        ret: Dict[str, Any] = {
            "padded_verts": padded,
            "verts_mask": verts_mask,
            "num_verts": num_verts,
            "num_faces": int(m.faces.shape[0]),
            "num_edges": int(edges.shape[0]),
            "verts": m.verts,
            "faces": m.faces,
        }

        for feat in topology_feats_type:
            if feat == "verts_normal":
                out = torch.zeros(n_verts, 3, dtype=v_nrm.dtype, device=device)
                out[:num_verts] = v_nrm
                if n_negative_verts > 0:
                    out[num_verts:] = negative_normals.to(dtype=v_nrm.dtype, device=device)
            elif feat == "adjacency_matrix":
                out = torch.zeros(n_verts, n_verts, dtype=torch.bool, device=device)
                out[edges[:, 0], edges[:, 1]] = True
                out[edges[:, 1], edges[:, 0]] = True
                if n_negative_verts > 0:
                    neg_slice = slice(num_verts, n_verts)
                    out[neg_slice, :] = out[nearest_verts_idx, :]
                    out[:, neg_slice] = out[:, nearest_verts_idx]
            elif feat == "degree":
                degrees = torch.bincount(edges.flatten(), minlength=num_verts)
                if (degrees > max_degree).any():
                    raise ValueError(f"Vertex degree exceeds max_degree ({max_degree}).")
                out = torch.zeros(n_verts, dtype=torch.long, device=device)
                out[:num_verts] = degrees
                if n_negative_verts > 0:
                    out[num_verts:] = out[nearest_verts_idx]
                out = out.unsqueeze(-1)
            elif feat == "neighbor_points":
                all_e = torch.cat([edges, edges.flip(1)], dim=0)
                src, dst = all_e[:, 0], all_e[:, 1]
                order = torch.argsort(src)
                src, dst = src[order], dst[order]
                if torch.bincount(src, minlength=num_verts).max() > max_degree:
                    raise ValueError(f"Vertex degree exceeds max_degree ({max_degree}).")
                first = torch.searchsorted(src, src)
                out = torch.zeros(n_verts, max_degree, 3, dtype=m.verts.dtype, device=device)
                out[src, torch.arange(len(src), device=device) - first] = m.verts[dst]
                if n_negative_verts > 0:
                    out[num_verts:] = out[nearest_verts_idx]
                out = out.view(n_verts, -1)
            else:
                raise ValueError(f"Unknown topology_feats_type: {feat}")
            ret[feat] = out
        return ret
