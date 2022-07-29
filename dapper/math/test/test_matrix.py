import numpy as np
import unittest

import dapper.math.matrix as mat


class MatrixTest(unittest.TestCase):
    def test_rotation_matrix(self):
        """
        Test creation and decomposition of rotation matrices.
        """
        yaw = 0.0
        pitch = 0.0
        roll = 0.0
        m = mat.rotation_matrix(yaw, pitch, roll)

        self.assertTupleEqual(m.shape, (4, 4))
        yaw1, pitch1, roll1 = mat.decompose_rotation(m)
        self.assertAlmostEqual(yaw1, yaw)
        self.assertAlmostEqual(pitch1, pitch)
        self.assertAlmostEqual(roll1, roll)

        yaw = -34.66
        pitch = 12.1
        roll = -128.784
        m = mat.rotation_matrix(yaw, pitch, roll)

        self.assertTupleEqual(m.shape, (4, 4))
        yaw1, pitch1, roll1 = mat.decompose_rotation(m)
        self.assertAlmostEqual(yaw1, yaw)
        self.assertAlmostEqual(pitch1, pitch)
        self.assertAlmostEqual(roll1, roll)

    def test_translation_matrix(self):
        """
        Test creation and decomposition of translation matrices.
        """
        translate = np.array([1.0, 2.0, 3.0])
        m = mat.translation_matrix(translate)

        self.assertTupleEqual(m.shape, (4, 4))
        translate1 = mat.decompose_translation(m)
        self.assertTupleEqual(translate1.shape, (3,))
        np.testing.assert_equal(translate1, translate)
