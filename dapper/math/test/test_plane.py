import numpy as np
import math
import unittest

import dapper.math.plane as pl
from dapper.math.ray import Ray


class TestPlane(unittest.TestCase):
    def test_distance_to(self):
        # Simplest case. The point's z value shall be echoed back.
        plane = pl.plane(np.array([0, 0, 1]), 0)

        p = np.array([1, 2, 3])
        self.assertAlmostEqual(pl.distance_to(plane, p), 3)

        p = np.array([1, 2, 0])
        self.assertAlmostEqual(pl.distance_to(plane, p), 0)

        p = np.array([1, 2, -23.9])
        self.assertAlmostEqual(pl.distance_to(plane, p), -23.9)

        # Set the plane at some distance from origo.
        plane = pl.plane(np.array([0, 0, 1]), 5)

        p = np.array([1, 2, 3])
        self.assertAlmostEqual(pl.distance_to(plane, p), -2)

    def test_infront_of(self):
        plane = pl.plane(np.array([0, 0, 1]), 10)
        self.assertFalse(pl.infront_of(plane, np.array([1, 2, 9])))
        self.assertTrue(pl.infront_of(plane, np.array([1, 2, 11])))

        plane = pl.plane(np.array([0, 0, -1]), -10)
        self.assertTrue(pl.infront_of(plane, np.array([1, 2, 9])))
        self.assertFalse(pl.infront_of(plane, np.array([1, 2, 11])))

    def test_ray_does_intersect(self):
        plane = pl.plane(np.array([0, 0, 1]), 0)

        # Straight to the plane.
        orig = np.array([0, 0, 10.0])
        ray = Ray(orig, np.array([0, 0, 0]))
        self.assertAlmostEqual(pl.raycast(plane, ray), 10)

        # With some angle.
        target = np.array([5, 0, 0])
        ray = Ray(orig, target)
        self.assertAlmostEqual(pl.raycast(plane, ray),
                               np.linalg.norm(target - orig))

    def test_ray_does_not_intersect(self):
        plane = pl.plane(np.array([0, 0, 1]), 0)

        orig = np.array([0, 0, 10.0])

        # Point in opposite direction.
        ray = Ray(orig, np.array([0, 0, 20.0]))
        self.assertIsNone(pl.raycast(plane, ray))
