from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import torch
import trimesh

from brain_factory.lib.mesh_ops.point_cloud_io import (
    normalize_points,
    read_point_cloud_file,
    resample_point_cloud,
    try_read_point_cloud_ply,
)


class TestNormalizePoints(unittest.TestCase):
    def test_bbox_centers_and_scales(self):
        points = torch.tensor([[0.0, 0.0, 0.0], [4.0, 2.0, 0.0]])
        out = normalize_points(points, normalize_by="bbox", size=2.0)
        torch.testing.assert_close(out.mean(0), torch.zeros(3))
        self.assertAlmostEqual(float(out.max() - out.min()), 2.0, places=5)

    def test_bsphere_centers_and_scales(self):
        points = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        out = normalize_points(points, normalize_by="bsphere", size=2.0)
        torch.testing.assert_close(out.mean(0), torch.zeros(3))
        radius = torch.linalg.norm(out, dim=-1).max()
        self.assertAlmostEqual(float(radius), 1.0, places=5)

    def test_unknown_normalize_by_raises(self):
        points = torch.zeros(3, 3)
        with self.assertRaises(NotImplementedError):
            normalize_points(points, normalize_by="unknown")


class TestResamplePointCloud(unittest.TestCase):
    def test_downsamples_without_replacement(self):
        points = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        out = resample_point_cloud(points, 4)
        self.assertEqual(out.shape, (4, 3))
        # Every sampled row must be one of the original rows (no fabricated points).
        for row in out:
            self.assertTrue(any(torch.equal(row, src) for src in points))

    def test_exact_count_returns_all_points_once(self):
        points = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        out = resample_point_cloud(points, 3)
        self.assertEqual(sorted(out.flatten().tolist()), sorted(points.flatten().tolist()))

    def test_too_few_points_raises(self):
        points = torch.zeros(2, 3)
        with self.assertRaises(ValueError):
            resample_point_cloud(points, 5)

    def test_empty_raises(self):
        points = torch.zeros(0, 3)
        with self.assertRaises(ValueError):
            resample_point_cloud(points, 1)


class TestReadPointCloudFile(unittest.TestCase):
    def setUp(self):
        self.points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )

    def test_npy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pts.npy")
            np.save(path, self.points)
            out = read_point_cloud_file(path)
            torch.testing.assert_close(out, torch.from_numpy(self.points))

    def test_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pts.npz")
            np.savez(path, points=self.points)
            out = read_point_cloud_file(path)
            torch.testing.assert_close(out, torch.from_numpy(self.points))

    def test_npz_missing_known_key_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pts.npz")
            np.savez(path, unrelated=self.points)
            with self.assertRaises(ValueError):
                read_point_cloud_file(path)

    def test_xyz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pts.xyz")
            np.savetxt(path, self.points)
            out = read_point_cloud_file(path)
            torch.testing.assert_close(out, torch.from_numpy(self.points))

    def test_pcd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pts.pcd")
            with open(path, "w") as f:
                f.write("# .PCD ascii point cloud\n")
                f.write("POINTS 3\n")
                for row in self.points:
                    f.write(f"{row[0]} {row[1]} {row[2]}\n")
            out = read_point_cloud_file(path)
            torch.testing.assert_close(out, torch.from_numpy(self.points))

    def test_unsupported_extension_raises(self):
        with self.assertRaises(NotImplementedError):
            read_point_cloud_file("pts.unsupported")


class TestTryReadPointCloudPly(unittest.TestCase):
    def test_faceless_ply_returns_points(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        cloud = trimesh.PointCloud(verts)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cloud.ply")
            cloud.export(path)
            out = try_read_point_cloud_ply(path)
            self.assertIsNotNone(out)
            self.assertEqual(out.shape, (3, 3))

    def test_faced_ply_returns_none(self):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]])
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mesh.ply")
            mesh.export(path)
            self.assertIsNone(try_read_point_cloud_ply(path))


if __name__ == "__main__":
    unittest.main()
