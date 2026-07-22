"""Boundary-loop detection and hole fan-triangulation over a triangle soup.

Port of meshflow/meshflow/utils/mesh.py (~/code/meshflow), used by
brain_factory.lib.mesh_ops.reconstruction when fill_holes=True. Meshflow's
original code is distributed under Meta's FAIR Noncommercial Research License
(noncommercial research use only); this repo is Apache-2.0, so confirm license
compatibility before any commercial use of this module.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def detect_and_triangulate_holes(
    triangles: List[Tuple[int, int, int]],
    max_cycle_size: int = 20,
) -> List[Tuple[int, int, int]]:
    """Find boundary loops and fan-triangulate holes in a triangle soup."""
    edge_to_triangles: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    for tri in triangles:
        edges_in_tri = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[0], tri[2]])),
        ]
        for edge in edges_in_tri:
            edge_to_triangles.setdefault(edge, []).append(tri)

    boundary_edges = {edge for edge, tris in edge_to_triangles.items() if len(tris) == 1}
    if not boundary_edges:
        return []

    boundary_vertices: set[int] = set()
    boundary_adjacency: Dict[int, List[int]] = {}
    for v1, v2 in boundary_edges:
        boundary_vertices.update((v1, v2))
        boundary_adjacency.setdefault(v1, []).append(v2)
        boundary_adjacency.setdefault(v2, []).append(v1)

    def find_cycles_dfs(start_vertex: int) -> List[List[int]]:
        cycles: List[List[int]] = []
        visited_in_path: set[int] = set()
        path: List[int] = []

        def dfs(current: int) -> None:
            visited_in_path.add(current)
            path.append(current)
            has_unvisited_neighbor = False
            for neighbor in boundary_adjacency.get(current, []):
                if neighbor == start_vertex and len(path) >= 3:
                    if 4 <= len(path) <= max_cycle_size:
                        cycles.append(path[:])
                elif neighbor not in visited_in_path and len(path) < max_cycle_size:
                    has_unvisited_neighbor = True
                    dfs(neighbor)
            if not has_unvisited_neighbor and len(path) >= 3:
                if 4 <= len(path) + 1 <= max_cycle_size:
                    if start_vertex not in boundary_adjacency.get(current, []):
                        missing_edge = tuple(sorted([current, start_vertex]))
                        if missing_edge in edge_to_triangles:
                            cycles.append(path[:])
            path.pop()
            visited_in_path.remove(current)

        dfs(start_vertex)
        return cycles

    all_cycles: List[List[int]] = []
    processed_cycles: set[Tuple[int, ...]] = set()
    for start_v in boundary_vertices:
        for cycle in find_cycles_dfs(start_v):
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
            reversed_cycle = tuple(reversed(normalized))
            if normalized not in processed_cycles and reversed_cycle not in processed_cycles:
                processed_cycles.add(normalized)
                all_cycles.append(cycle)

    new_triangles: List[Tuple[int, int, int]] = []
    for loop in all_cycles:
        v0 = loop[0]
        for i in range(1, len(loop) - 1):
            new_triangles.append(tuple(sorted([v0, loop[i], loop[i + 1]])))
    return new_triangles
