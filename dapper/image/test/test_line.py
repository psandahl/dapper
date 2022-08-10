import math
import numpy as np
import unittest

from dapper.image.line import Line


class TestLine(unittest.TestCase):
    def test_simple_lines(self):
        size = (800, 600)

        # Diagonal.
        begin = np.array([10, 10])
        end = np.array([20, 20])

        line = Line(begin, end, size)
        self.assertTrue(line.ok)
        np.testing.assert_array_equal(line.begin, begin)
        np.testing.assert_array_equal(line.end, end)
        self.assertAlmostEqual(line.min_distance, 0.0)
        self.assertAlmostEqual(line.max_distance, np.linalg.norm(end - begin))
        self.assertAlmostEqual(line.curr_distance, 0.0)
        self.assertAlmostEqual(line.length, np.linalg.norm(end - begin))
        self.assertAlmostEqual(line.gradient[0], math.sqrt(0.5))
        self.assertAlmostEqual(line.gradient[1], math.sqrt(0.5))

        # Opposite diagonal.
        end, begin = begin, end
        line = Line(begin, end, size)
        self.assertTrue(line.ok)
        np.testing.assert_array_equal(line.begin, begin)
        np.testing.assert_array_equal(line.end, end)
        self.assertAlmostEqual(line.min_distance, 0.0)
        self.assertAlmostEqual(line.max_distance, np.linalg.norm(end - begin))
        self.assertAlmostEqual(line.curr_distance, 0.0)
        self.assertAlmostEqual(line.length, np.linalg.norm(end - begin))
        self.assertAlmostEqual(line.gradient[0], -math.sqrt(0.5))
        self.assertAlmostEqual(line.gradient[1], -math.sqrt(0.5))

    def test_nonintersecting_lines(self):
        size = (800, 600)

        begin = np.array([-10, -10])
        end = np.array([20, -20])

        line = Line(begin, end, size)
        self.assertFalse(line.ok)

        end, begin = begin, end
        line = Line(begin, end, size)
        self.assertFalse(line.ok)

    def test_partly_outside_lines(self):
        size = (800, 600)

        begin = np.array([-10, 10])
        end = np.array([20, 10])

        line = Line(begin, end, size)
        self.assertTrue(line.ok)
        np.testing.assert_array_equal(line.begin, begin)
        np.testing.assert_array_equal(line.end, end)
        self.assertAlmostEqual(line.min_distance, 2 - begin[0])
        self.assertAlmostEqual(line.curr_distance, line.min_distance)
        self.assertAlmostEqual(line.max_distance, np.linalg.norm(end - begin))
        self.assertAlmostEqual(line.length, np.linalg.norm(end - begin))

        begin = np.array([700, 10])
        end = np.array([810, 10])

        line = Line(begin, end, size)
        self.assertTrue(line.ok)
        np.testing.assert_array_equal(line.begin, begin)
        np.testing.assert_array_equal(line.end, end)
        self.assertAlmostEqual(line.min_distance, 0.0)
        self.assertAlmostEqual(line.curr_distance, line.min_distance)
        self.assertAlmostEqual(line.max_distance, 797 - begin[0])
        self.assertAlmostEqual(line.length, np.linalg.norm(end - begin))

    def test_line_navigation(self):
        size = (800, 600)

        begin = np.array([10, 10])
        end = np.array([20, 10])

        line = Line(begin, end, size)

        self.assertAlmostEqual(line.ratio(), 0.0)
        np.testing.assert_array_almost_equal(line.point(), begin)
        np.testing.assert_array_almost_equal(
            line.point(1), begin + np.array([1, 0]))

        line.forward()
        np.testing.assert_array_almost_equal(
            line.point(), begin + np.array([1, 0]))
        np.testing.assert_array_almost_equal(line.point(-1), begin)
        self.assertAlmostEqual(line.ratio(), 0.1)
