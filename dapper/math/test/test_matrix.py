import numpy as np
import unittest

import dapper.math.matrix as mat


class MatrixTest(unittest.TestCase):
    def test_rotation_matrix(self):
        """
        Test creation and decomposition of rotation matrices.
        """
        rotation1 = np.array([0.0, 0.0, 0.0])
        m = mat.rotation_matrix(rotation1)

        self.assertTupleEqual(m.shape, (4, 4))
        rotation2 = mat.decompose_rotation(m)
        self.assertTupleEqual((3,), rotation2.shape)
        np.testing.assert_almost_equal(rotation2, rotation1)

        rotation1 = np.array([-34.66, 12.1, -128.784])
        m = mat.rotation_matrix(rotation1)

        self.assertTupleEqual(m.shape, (4, 4))
        rotation2 = mat.decompose_rotation(m)
        self.assertTupleEqual((3,), rotation2.shape)
        np.testing.assert_almost_equal(rotation2, rotation1)

    def test_translation_matrix(self):
        """
        Test creation and decomposition of translation matrices.
        """
        translation1 = np.array([1.0, 2.0, 3.0])
        m = mat.translation_matrix(translation1)

        self.assertTupleEqual(m.shape, (4, 4))
        translation2 = mat.decompose_translation(m)
        self.assertTupleEqual(translation2.shape, (3,))
        np.testing.assert_equal(translation2, translation1)
