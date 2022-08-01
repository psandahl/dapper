import numpy as np
import unittest

import dapper.image.gradient as gr


class TestGradient(unittest.TestCase):
    def test_pixel_gradient(self):
        """
        Test the computation of a gradient.
        """

        # Constant image.
        base = 25

        img = np.zeros((3, 3), dtype=np.uint8)
        img[:, :] = base

        np.testing.assert_equal(
            gr.pixel_gradient(img, 1, 1), np.array([0, 0]))

        #     31
        #  32 px 22
        #     23
        img[0, 1] = 31
        img[1, 0] = 32
        img[1, 2] = 22
        img[2, 1] = 23

        np.testing.assert_almost_equal(
            gr.pixel_gradient(img, 1, 1), np.array([(22 - 32) / 2.0, (23 - 31) / 2.0]))

        img[:, :] = base

        #     9
        #  9 px 22
        #    23
        img[0, 1] = 9
        img[1, 0] = 9
        img[1, 2] = 22
        img[2, 1] = 23

        np.testing.assert_almost_equal(
            gr.pixel_gradient(img, 1, 1), np.array([(22 - 9) / 2.0, (23 - 9) / 2.0]))

    def test_gradient_orientation(self):
        """
        Test the orientation of a gradient.
        """
        img = np.zeros((3, 3), dtype=np.uint8)

        orientation = gr.gradient_orientation(gr.pixel_gradient(img, 1, 1))
        self.assertEqual(orientation, 0.0)

        # 0  0 1
        # 0 px 1
        # 0  0 1
        img[0, 2] = 1
        img[1, 2] = 1
        img[2, 2] = 1

        # Shall also be zero degrees ...
        orientation = gr.gradient_orientation(gr.pixel_gradient(img, 1, 1))
        self.assertEqual(orientation, 0.0)

        img[:, :] = 0

        # 0  0 0
        # 0 px 1
        # 0 1  1
        img[1, 2] = 1
        img[2, 1] = 1
        img[2, 2] = 1

        orientation = gr.gradient_orientation(gr.pixel_gradient(img, 1, 1))
        self.assertEqual(orientation, 45.0)

        img[:, :] = 0

        # 0  0 0
        # 0 px 0
        # 1 1  1
        img[2, 0] = 1
        img[2, 1] = 1
        img[2, 2] = 1

        orientation = gr.gradient_orientation(gr.pixel_gradient(img, 1, 1))
        self.assertEqual(orientation, 90.0)

        img[:, :] = 0

        # 0  0 0
        # 1 px 0
        # 1 1  0
        img[1, 0] = 1
        img[2, 0] = 1
        img[2, 1] = 1

        orientation = gr.gradient_orientation(gr.pixel_gradient(img, 1, 1))
        self.assertEqual(orientation, 135.0)

        img[:, :] = 0

        # 1  0 0
        # 1 px 0
        # 1 0  0
        img[0, 0] = 1
        img[1, 0] = 1
        img[2, 0] = 1

        orientation = gr.gradient_orientation(gr.pixel_gradient(img, 1, 1))
        self.assertEqual(orientation, 180.0)

        img[:, :] = 0

        # And a few in the negative direction ...

        # 0  1 1
        # 0 px 1
        # 0 0  0
        img[0, 1] = 1
        img[0, 2] = 1
        img[1, 2] = 1

        orientation = gr.gradient_orientation(gr.pixel_gradient(img, 1, 1))
        self.assertEqual(orientation, -45.0)

        img[:, :] = 0

        # 1  1 1
        # 0 px 0
        # 0 0  0
        img[0, 0] = 1
        img[0, 1] = 1
        img[0, 2] = 1

        orientation = gr.gradient_orientation(gr.pixel_gradient(img, 1, 1))
        self.assertEqual(orientation, -90.0)

        img[:, :] = 0
