import numpy as np
import unittest

import dapper.image.helpers as hlp


class HelpersTest(unittest.TestCase):
    def test_is_image(self):
        """
        Test checking if something is an image.
        """
        self.assertFalse(hlp.is_image(None))
        self.assertFalse(hlp.is_image(1))
        self.assertFalse(hlp.is_image([1, 2, 3]))
        self.assertFalse(hlp.is_image('something'))
        self.assertFalse(hlp.is_image(np.arange(0, 100, dtype=np.uint8)))

        self.assertTrue(hlp.is_image(np.zeros((10, 10), dtype=np.uint8)))
        self.assertTrue(hlp.is_image(np.zeros((10, 10, 2), dtype=np.uint8)))
        self.assertTrue(hlp.is_image(np.zeros((10, 10, 3), dtype=np.uint8)))
        self.assertTrue(hlp.is_image(np.zeros((10, 10, 4), dtype=np.uint8)))

        self.assertFalse(hlp.is_image(np.zeros((10, 10, 5), dtype=np.uint8)))

    def test_image_size(self):
        """
        Test checking of image size.
        """
        self.assertTupleEqual((30, 20), hlp.image_size(
            np.zeros((20, 30), dtype=np.uint8)))
        self.assertTupleEqual((30, 20), hlp.image_size(
            np.zeros((20, 30, 2), dtype=np.uint8)))
        self.assertTupleEqual((30, 20), hlp.image_size(
            np.zeros((20, 30, 3), dtype=np.uint8)))
        self.assertTupleEqual((30, 20), hlp.image_size(
            np.zeros((20, 30, 4), dtype=np.uint8)))

    def test_image_channels(self):
        """
        Test checking of image channels.
        """
        self.assertEqual(1, hlp.image_channels(
            np.zeros((10, 10), dtype=np.uint8)))
        self.assertEqual(2, hlp.image_channels(
            np.zeros((10, 10, 2), dtype=np.uint8)))
        self.assertEqual(3, hlp.image_channels(
            np.zeros((10, 10, 3), dtype=np.uint8)))
        self.assertEqual(4, hlp.image_channels(
            np.zeros((10, 10, 4), dtype=np.uint8)))

    def test_within_image_extent(self):
        """
        Test checking of coordinate within image extent.
        """
        image_size = (30, 20)

        self.assertTrue(hlp.within_image_extent(image_size, 0, 0))
        self.assertTrue(hlp.within_image_extent(image_size, 29, 19))
        self.assertTrue(hlp.within_image_extent(image_size, 0, 0))
        self.assertTrue(hlp.within_image_extent(image_size, 15, 10))

        self.assertFalse(hlp.within_image_extent(image_size, 30, 19))
        self.assertFalse(hlp.within_image_extent(image_size, 29, 20))
        self.assertFalse(hlp.within_image_extent(image_size, -1, -1))

        self.assertFalse(hlp.within_image_extent(image_size, 0, 0, 1))
        self.assertFalse(hlp.within_image_extent(image_size, 29, 19, 1))
