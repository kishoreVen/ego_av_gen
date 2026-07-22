from __future__ import annotations

import unittest

from brain_factory.lib.mesh_ops.topology import detect_and_triangulate_holes


class TestDetectAndTriangulateHoles(unittest.TestCase):
    def test_watertight_mesh_has_no_holes(self):
        """A closed tetrahedron has every edge shared by exactly two faces."""
        triangles = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
        self.assertEqual(detect_and_triangulate_holes(triangles), [])

    def test_fills_quad_hole(self):
        """Four pyramid side faces (no base) leave a 4-vertex boundary loop."""
        triangles = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)]
        extra = detect_and_triangulate_holes(triangles)

        self.assertGreater(len(extra), 0)
        boundary_verts = {1, 2, 3, 4}
        for tri in extra:
            self.assertTrue(set(tri).issubset(boundary_verts))

        # Filling the hole should leave no boundary edge (one shared by exactly one face).
        combined = triangles + extra
        self.assertEqual(detect_and_triangulate_holes(combined), [])

    def test_max_cycle_size_excludes_larger_holes(self):
        """A hole larger than max_cycle_size should not be fan-triangulated."""
        triangles = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)]
        self.assertEqual(detect_and_triangulate_holes(triangles, max_cycle_size=3), [])

    def test_no_triangles_returns_empty(self):
        self.assertEqual(detect_and_triangulate_holes([]), [])


if __name__ == "__main__":
    unittest.main()
