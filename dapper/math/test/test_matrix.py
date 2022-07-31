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

    def test_pose_matrix(self):
        """
        Test creation and decomposition of pose matrix.
        """
        rotation1 = np.array([-34.66, 12.1, -128.784])
        translation1 = np.array([1.0, 2.0, 3.0])
        m = mat.pose_matrix(rotation1, translation1)

        self.assertTupleEqual(m.shape, (4, 4))
        rotation2, translation2 = mat.decompose_pose(m)
        self.assertTupleEqual(rotation2.shape, (3,))
        self.assertTupleEqual(translation2.shape, (3,))
        np.testing.assert_almost_equal(rotation2, rotation1)
        np.testing.assert_equal(translation2, translation1)

    def test_relative_pose(self):
        """
        Test of relative pose between two pose matrices.
        """
        # Simplest case: 'from' is 'zero', then the result
        # shall be equal to 'to'.
        rotation_from = np.array([0.0, 0.0, 0.0])
        translation_from = np.array([0.0, 0.0, 0.0])
        pose_from = mat.pose_matrix(rotation_from, translation_from)

        rotation_to = np.array([45.0, 17.0, 5.0])
        translation_to = np.array([1.0, 2.0, 3.0])
        pose_to = mat.pose_matrix(rotation_to, translation_to)

        m = mat.relative_pose(pose_from, pose_to)
        self.assertTupleEqual(m.shape, (4, 4))
        rotation, translation = mat.decompose_pose(m)
        np.testing.assert_almost_equal(rotation, rotation_to)
        np.testing.assert_almost_equal(translation, translation_to)

        # The little more complicated case is when 'from' is
        # rotated and translated. 'To' has the same global
        # rotation, and there shall thus be no relative rotation.
        rotation_from = np.array([90.0, 0.0, 0.0])
        translation_from = np.array([10.0, 0.0, 10.0])
        pose_from = mat.pose_matrix(rotation_from, translation_from)

        rotation_to = np.array([90.0, 0.0, 0.0])
        translation_to = np.array([12.0, -1.0, 8.0])
        pose_to = mat.pose_matrix(rotation_to, translation_to)

        m = mat.relative_pose(pose_from, pose_to)
        rotation, translation = mat.decompose_pose(m)
        np.testing.assert_almost_equal(rotation, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(translation, np.array([2.0, -1.0, 2.0]))

        # The most complicated case is when there also are a relative
        # rotation.
        rotation_from = np.array([90.0, 0.0, 0.0])
        translation_from = np.array([10.0, 0.0, 10.0])
        pose_from = mat.pose_matrix(rotation_from, translation_from)

        rotation_to = np.array([45.0, -10.5, 5.0])
        translation_to = np.array([12.0, -1.0, 8.0])
        pose_to = mat.pose_matrix(rotation_to, translation_to)

        m = mat.relative_pose(pose_from, pose_to)
        rotation, translation = mat.decompose_pose(m)
        np.testing.assert_almost_equal(rotation, np.array([-45.0, -10.5, 5.0]))
        np.testing.assert_almost_equal(translation, np.array([2.0, -1.0, 2.0]))

    def test_apply_pose(self):
        """
        Test of applying a pose to another (inverse of the relative pose).
        """
        rotation_from = np.array([90.0, 0.0, 0.0])
        translation_from = np.array([10.0, 0.0, 10.0])
        pose_from = mat.pose_matrix(rotation_from, translation_from)

        rotation_to = np.array([45.0, -10.5, 5.0])
        translation_to = np.array([12.0, -1.0, 8.0])
        pose_to = mat.pose_matrix(rotation_to, translation_to)

        pose_delta = mat.relative_pose(pose_from, pose_to)

        # With the apply we shall now have reconstructed pose_to.
        m = mat.apply_pose(pose_from, pose_delta)
        self.assertTupleEqual(m.shape, (4, 4))
        rotation, translation = mat.decompose_pose(m)
        np.testing.assert_almost_equal(rotation, rotation_to)
        np.testing.assert_almost_equal(translation, translation_to)

    def test_extract_3x3(self):
        """
        Test the extraction of a 3x3 matrix from a matrix.
        """
        m44 = np.arange(0, 16).reshape(4, 4)

        m33 = mat.extract_3x3(m44)
        self.assertTupleEqual((3, 3), m33.shape)
        np.testing.assert_equal(m33, np.array(
            [0, 1, 2, 4, 5, 6, 8, 9, 10]).reshape(3, 3))

        # Shall return the input reference, as no extraction is needed.
        m33_1 = mat.extract_3x3(m33)
        self.assertIs(m33_1, m33)

    def test_add_row(self):
        """
        Test addition of a row to a matrix.
        """
        m34 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]).reshape(3, 4)

        m44 = mat.add_row(m34)
        self.assertTupleEqual((4, 4), m44.shape)

        # Result shall be 4x4 identity matrix.
        np.testing.assert_equal(m44, np.eye(4, 4))
