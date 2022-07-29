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

        self.assertTupleEqual((4, 4), m.shape)
        yaw1, pitch1, roll1 = mat.decompose_rotation_matrix(m)
        self.assertAlmostEqual(yaw, yaw1)
        self.assertAlmostEqual(pitch, pitch1)
        self.assertAlmostEqual(roll, roll1)

        yaw = -34.66
        pitch = 12.1
        roll = -128.784
        m = mat.rotation_matrix(yaw, pitch, roll)

        self.assertTupleEqual((4, 4), m.shape)
        yaw1, pitch1, roll1 = mat.decompose_rotation_matrix(m)
        self.assertAlmostEqual(yaw, yaw1)
        self.assertAlmostEqual(pitch, pitch1)
        self.assertAlmostEqual(roll, roll1)
