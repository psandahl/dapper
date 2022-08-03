import numpy as np
import unittest

import dapper.math.helpers as hlp


class TestHelpers(unittest.TestCase):
    def test_ideal_intrinsic_matrix(self):
        """
        Silly simple at the moment ...
        """
        K = hlp.ideal_intrinsic_matrix((1024, 768), 30)
        self.assertTupleEqual((3, 3), K.shape)

    def test_project_and_unproject(self):
        """
        Test projection and unprojection on image.
        """
        K = hlp.ideal_intrinsic_matrix((1024, 768), 30)
        K_inv = np.linalg.inv(K)

        xyz = np.array([0, 0, 100])
        px = hlp.project_image(K, xyz)
        self.assertTupleEqual((2,), px.shape)
        xyz2 = hlp.unproject_image(K_inv, px[0], px[1], 100)
        self.assertTupleEqual((3,), xyz2.shape)
        np.testing.assert_almost_equal(xyz2, xyz)

        xyz = np.array([12.0, -8.8, 171])
        px = hlp.project_image(K, xyz)
        self.assertTupleEqual((2,), px.shape)
        xyz2 = hlp.unproject_image(K_inv, px[0], px[1], 171)
        self.assertTupleEqual((3,), xyz2.shape)
        np.testing.assert_almost_equal(xyz2, xyz)
