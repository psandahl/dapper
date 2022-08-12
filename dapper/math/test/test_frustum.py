import numpy as np
import unittest

import dapper.common.settings as settings
from dapper.math.frustum import Frustum
import dapper.math.helpers as mat_hlp


class TestFrustum(unittest.TestCase):
    def test_frustum_does_contain(self):
        w = 1024
        h = 768

        image_size = (1024, 768)
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

        image_size = (1024, 768)
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
