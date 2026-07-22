"""Reconstruct a Mesh (triangles) from predicted vertices/normals/edges.

Port of meshflow/meshflow/utils/mesh.py (~/code/meshflow), used by
brain_factory.model.meshflow.meshflow_vae.MeshFlowVAE.extract_mesh. Meshflow's
original code is distributed under Meta's FAIR Noncommercial Research License
(noncommercial research use only); this repo is Apache-2.0, so confirm license
compatibility before any commercial use of this module.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from brain_factory.lib.data_structures.mesh import Mesh
from brain_factory.lib.mesh_ops.topology import detect_and_triangulate_holes


def extract_mesh_from_verts_normals_edges(
    vertices: torch.Tensor,
    vertices_normal: torch.Tensor,
    edges: torch.Tensor,
    fill_holes: bool = False,
    max_cycle_size: int = 10,
    device: torch.device | None = None,
) -> Mesh:
    """Build a Mesh from predicted vertices, normals, and edge pairs."""
    if device is None:
        device = vertices.device

    vertices = vertices.to(device, dtype=torch.float32)
    vertices_normal = vertices_normal.to(device, dtype=torch.float32)
    edges = edges.to(device, dtype=torch.long)

    num_vertices = vertices.shape[0]
    if edges.numel() == 0 or num_vertices < 3:
        raise ValueError(
            f"Cannot extract mesh: invalid input (edges={edges.numel()}, vertices={num_vertices})"
        )

    edges = torch.clamp(edges, 0, num_vertices - 1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    if edges.numel() == 0:
        raise ValueError("Cannot extract mesh: no edges after removing self-loops")

    adjacency_list: List[List[int]] = [[] for _ in range(num_vertices)]
    edge_set: set[Tuple[int, int]] = set()
    for e in edges.cpu().numpy():
        v1, v2 = int(e[0]), int(e[1])
        if (v1, v2) not in edge_set and (v2, v1) not in edge_set:
            adjacency_list[v1].append(v2)
            adjacency_list[v2].append(v1)
            edge_set.add((v1, v2))

    max_neighbors = max(len(adj) for adj in adjacency_list) if adjacency_list else 0
    if max_neighbors == 0:
        raise ValueError("Cannot extract mesh: no vertex neighbors in edge graph")

    adj_tensor = torch.full((num_vertices, max_neighbors), -1, dtype=torch.long, device=device)
    for i, neighbors in enumerate(adjacency_list):
        if neighbors:
            adj_tensor[i, : len(neighbors)] = torch.tensor(neighbors, dtype=torch.long, device=device)

    triangles: List[Tuple[int, int, int]] = []
    for v1, v2 in edge_set:
        n1_mask = adj_tensor[v1] >= 0
        n2_mask = adj_tensor[v2] >= 0
        if not (n1_mask.any() and n2_mask.any()):
            continue
        neighbors_v1 = adj_tensor[v1][n1_mask]
        neighbors_v2 = adj_tensor[v2][n2_mask]
        matches = neighbors_v1.unsqueeze(1) == neighbors_v2.unsqueeze(0)
        if not matches.any():
            continue
        v1_indices, _ = torch.where(matches)
        for v3 in neighbors_v1[v1_indices].unique():
            if v3 != v1 and v3 != v2:
                triangles.append(tuple(sorted([v1, v2, int(v3.item())])))

    triangles = list(set(triangles))
    if not triangles:
        raise ValueError("Cannot extract mesh: no triangles formed from edges")

    if fill_holes:
        extra = detect_and_triangulate_holes(triangles, max_cycle_size=max_cycle_size)
        if extra:
            triangles = list(set(triangles + extra))

    triangle_tensor = torch.tensor(triangles, dtype=torch.long, device=device)
    triangle_vertices = vertices[triangle_tensor]
    edge1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    edge2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    face_normals = torch.linalg.cross(edge1, edge2, dim=1)
    face_normals_norm = torch.linalg.norm(face_normals, dim=1, keepdim=True)
    fallback = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
    face_normals = torch.where(
        face_normals_norm > 1e-8,
        face_normals / face_normals_norm,
        fallback.expand_as(face_normals),
    )

    triangle_normals = vertices_normal[triangle_tensor]
    avg_vertex_normals = triangle_normals.mean(dim=1)
    avg_normals_norm = torch.linalg.norm(avg_vertex_normals, dim=1, keepdim=True)
    avg_vertex_normals = torch.where(
        avg_normals_norm > 1e-8,
        avg_vertex_normals / avg_normals_norm,
        fallback.expand_as(avg_vertex_normals),
    )

    faces = triangle_tensor.clone()
    flip_mask = (face_normals * avg_vertex_normals).sum(dim=1) < 0
    faces[flip_mask, 1], faces[flip_mask, 2] = faces[flip_mask, 2], faces[flip_mask, 1]
    faces = torch.clamp(faces, 0, num_vertices - 1)

    mesh = Mesh(verts=vertices, faces=faces, device=device)
    mesh._v_nrm = vertices_normal
    return mesh
