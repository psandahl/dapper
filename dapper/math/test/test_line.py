import math
import numpy as np
import unittest

from dapper.math.line import Line


class TestLine(unittest.TestCase):
    def test_create_simple_lines(self):
        line = Line(np.array([0, 0]), np.array([1, 1]))
        self.assertTrue(line.is_normal())
        self.assertAlmostEqual(line.length, math.sqrt(2))
        self.assertAlmostEqual(line.k, 1.0)
        self.assertAlmostEqual(line.m, 0.0)

        line = Line(np.array([1, 2]), np.array([2, 1]))
        self.assertTrue(line.is_normal())
        self.assertAlmostEqual(line.length, math.sqrt(2))
        self.assertAlmostEqual(line.k, -1.0)
        self.assertAlmostEqual(line.m, 3.0)

        line = Line(np.array([2, 2]), np.array([3, 2]))
        self.assertTrue(line.is_horizontal())
        self.assertAlmostEqual(line.length, 1.0)
        self.assertAlmostEqual(line.k, 0.0)
        self.assertAlmostEqual(line.m, 2.0)

        line = Line(np.array([2, 2]), np.array([2, 3]))
        self.assertTrue(line.is_vertical())
        self.assertAlmostEqual(line.length, 1.0)
        self.assertTrue(math.isinf(line.k))
        self.assertTrue(math.isnan(line.m))

    def test_y_at(self):
        line = Line(np.array([0, 0]), np.array([1, 1]))
        self.assertAlmostEqual(line.y_at(11.1), 11.1)

        line = Line(np.array([2, 2]), np.array([3, 2]))
        self.assertAlmostEqual(line.y_at(11.1), 2.0)

        line = Line(np.array([2, 2]), np.array([2, 3]))
        self.assertTrue(math.isnan(line.y_at(2)))

    def test_x_at(self):
        line = Line(np.array([0, 0]), np.array([1, 1]))
        self.assertAlmostEqual(line.x_at(11.1), 11.1)

        line = Line(np.array([2, 2]), np.array([3, 2]))
        self.assertTrue(math.isnan(line.x_at(2)))

        line = Line(np.array([2, 2]), np.array([2, 3]))
        self.assertAlmostEqual(line.x_at(11.1), 2.0)
