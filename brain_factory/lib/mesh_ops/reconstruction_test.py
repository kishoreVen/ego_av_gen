from __future__ import annotations

import unittest

import torch

from brain_factory.lib.data_structures.mesh import Mesh
from brain_factory.lib.mesh_ops.reconstruction import extract_mesh_from_verts_normals_edges

# Regular tetrahedron centered at the origin, so per-vertex outward normals
# are just the normalized vertex positions.
_TETRAHEDRON_VERTS = torch.tensor(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
)
_TETRAHEDRON_NORMALS = torch.nn.functional.normalize(_TETRAHEDRON_VERTS, dim=-1)
_TETRAHEDRON_EDGES = torch.tensor(
    [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
)


class TestExtractMeshFromVertsNormalsEdges(unittest.TestCase):
    def test_complete_graph_yields_tetrahedron_faces(self):
        """All 6 edges of 4 points -> every 3-clique becomes a face."""
        mesh = extract_mesh_from_verts_normals_edges(
            _TETRAHEDRON_VERTS, _TETRAHEDRON_NORMALS, _TETRAHEDRON_EDGES
        )
        self.assertIsInstance(mesh, Mesh)
        faces = {tuple(sorted(f.tolist())) for f in mesh.faces}
        self.assertEqual(
            faces, {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)}
        )

    def test_empty_edges_raises(self):
        with self.assertRaises(ValueError):
            extract_mesh_from_verts_normals_edges(
                _TETRAHEDRON_VERTS, _TETRAHEDRON_NORMALS, torch.zeros(0, 2, dtype=torch.long)
            )

    def test_too_few_vertices_raises(self):
        verts = _TETRAHEDRON_VERTS[:2]
        normals = _TETRAHEDRON_NORMALS[:2]
        edges = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            extract_mesh_from_verts_normals_edges(verts, normals, edges)

    def test_self_loops_only_raises(self):
        edges = torch.tensor([[0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            extract_mesh_from_verts_normals_edges(
                _TETRAHEDRON_VERTS, _TETRAHEDRON_NORMALS, edges
            )

    def test_no_shared_neighbors_forms_no_triangles(self):
        """A single edge with no common neighbor can't form a triangle."""
        edges = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            extract_mesh_from_verts_normals_edges(
                _TETRAHEDRON_VERTS, _TETRAHEDRON_NORMALS, edges
            )

    def test_fill_holes_true_does_not_crash_on_closed_mesh(self):
        """fill_holes is a no-op when the reconstructed mesh is already closed."""
        mesh = extract_mesh_from_verts_normals_edges(
            _TETRAHEDRON_VERTS, _TETRAHEDRON_NORMALS, _TETRAHEDRON_EDGES, fill_holes=True
        )
        self.assertEqual(mesh.faces.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
