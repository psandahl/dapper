import math
import numpy as np
import unittest

import dapper.common.settings as settings
from dapper.math.frustum import Frustum
import dapper.math.helpers as mat_hlp
from dapper.math.ray import Ray


class TestFrustum(unittest.TestCase):
    def test_frustum_does_contain(self):
        w = 1024
        h = 768

        image_size = (w, h)
        K_inv = np.linalg.inv(mat_hlp.ideal_intrinsic_matrix(image_size, 40))

        frustum = Frustum(K_inv, image_size)

        # Mid point.
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w / 2, h / 2,
                                    settings.FRUSTUM_NEAR_PLANE + 1e-05)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w / 2, h / 2, 1)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w / 2, h / 2, 100)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w / 2, h / 2, 1000)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w / 2, h / 2,
                                    settings.FRUSTUM_FAR_PLANE - 1e-05)))

        # Corners with some epsilon margin.
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, 1e-09, 1e-09,
                                    settings.FRUSTUM_NEAR_PLANE + 1e-09)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, 1e-09, 1e-09, 1.0)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, 1e-09, 1e-09,
                                    settings.FRUSTUM_FAR_PLANE - 1e-09)))

        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w - 1 - 1e-09, 1e-09,
                                    settings.FRUSTUM_NEAR_PLANE + 1e-09)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w - 1 - 1e-09, 1e-09, 1.0)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w - 1 - 1e-09, 1e-09,
                                    settings.FRUSTUM_FAR_PLANE - 1e-09)))

        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w - 1 - 1e-09, h - 1 - 1e-09,
                                    settings.FRUSTUM_NEAR_PLANE + 1e-09)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w - 1 - 1e-09, h - 1 - 1e-09, 1.0)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, w - 1 - 1e-09, h - 1 - 1e-09,
                                    settings.FRUSTUM_FAR_PLANE - 1e-09)))

        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, 1e-09, h - 1 - 1e-09,
                                    settings.FRUSTUM_NEAR_PLANE + 1e-09)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, 1e-09, h - 1 - 1e-09, 1.0)))
        self.assertTrue(frustum.contains(
            mat_hlp.unproject_image(K_inv, 1e-09, h - 1 - 1e-09,
                                    settings.FRUSTUM_FAR_PLANE - 1e-09)))

    def test_frustum_does_not_contain(self):
        w = 1024
        h = 768

        image_size = (w, h)
        K_inv = np.linalg.inv(mat_hlp.ideal_intrinsic_matrix(image_size, 40))

        frustum = Frustum(K_inv, image_size)

        # Middle, but outside far/near.
        self.assertFalse(frustum.contains(
            mat_hlp.unproject_image(K_inv, w / 2, h / 2,
                                    settings.FRUSTUM_NEAR_PLANE - 1e-09)))
        self.assertFalse(frustum.contains(
            mat_hlp.unproject_image(K_inv, w / 2, h / 2,
                                    settings.FRUSTUM_FAR_PLANE + 1e-09)))

        # Random outside image.
        self.assertFalse(frustum.contains(
            mat_hlp.unproject_image(K_inv, 1100, 899, 100)))

    def test_frustum_intersect_inside(self):
        w = 1024.0
        h = 768.0
        half_fov = 20.0

        image_size = (w, h)
        K_inv = np.linalg.inv(
            mat_hlp.ideal_intrinsic_matrix(image_size, half_fov * 2.0))

        frustum = Frustum(K_inv, image_size)

        origin = np.array([0, 0, 286.7])

        # Front
        ray = Ray(origin, origin + np.array([0, 0, 1.0]))
        t = frustum.intersect_inside(ray)
        self.assertAlmostEqual(t, settings.FRUSTUM_FAR_PLANE - origin[2])
        self.assertTrue(frustum.contains(ray.point_at(t)))

        # Back
        ray = Ray(origin, origin + np.array([0, 0, -1.0]))
        t = frustum.intersect_inside(ray)
        self.assertAlmostEqual(t, origin[2] - settings.FRUSTUM_NEAR_PLANE)
        self.assertTrue(frustum.contains(ray.point_at(t)))

        # Right
        ray = Ray(origin, origin + np.array([1.0, 0.0, 0.0]))
        t = frustum.intersect_inside(ray)
        dist = origin[2] * math.tan(math.radians(half_fov))
        self.assertAlmostEqual(t, dist)
        self.assertTrue(frustum.contains(ray.point_at(t)))

        # Left
        ray = Ray(origin, origin + np.array([-1.0, 0.0, 0.0]))
        t = frustum.intersect_inside(ray)
        dist = origin[2] * math.tan(math.radians(half_fov))
        self.assertAlmostEqual(t, dist)
        self.assertTrue(frustum.contains(ray.point_at(t)))

        # Up
        ray = Ray(origin, origin + np.array([0.0, -1.0, 0.0]))
        t = frustum.intersect_inside(ray)
        dist = dist / (w / h)  # Scale with aspect.
        self.assertAlmostEqual(t, dist)

        # Down
        ray = Ray(origin, origin + np.array([0.0, 1.0, 0.0]))
        t = frustum.intersect_inside(ray)
        self.assertAlmostEqual(t, dist)

        # Not intersect.
        origin = np.array([10.0, 0.0, 1.0])
        ray = Ray(origin, origin + np.array([0, 0, -1.0]))
        t = frustum.intersect_inside(ray)
        self.assertIsNone(t)

    def test_frustum_intersect_outside(self):
        w = 1024.0
        h = 768.0
        half_fov = 20.0

        image_size = (w, h)
        K_inv = np.linalg.inv(
            mat_hlp.ideal_intrinsic_matrix(image_size, half_fov * 2.0))

        frustum = Frustum(K_inv, image_size)

        # Setup where the intended hit is the right plane.
        origin = np.array([10.0, 0.0, -1.0])
        ray = Ray(origin, np.array([0, 0, 10.0]))
        t = frustum.intersect_outside(ray)
        self.assertAlmostEqual(t, 11.0022205)  # Hard coded.

        # Not intersect.
        origin = np.array([0, 0, 286.7])
        ray = Ray(origin, origin + np.array([0, 0, -1.0]))
        t = frustum.intersect_outside(ray)
        self.assertIsNone(t)
