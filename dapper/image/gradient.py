import logging
import math
import numpy as np

import dapper.image.helpers as hlp

logger = logging.getLogger(__name__)


def pixel_gradient(image: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    Compute the gradient for a pixel.

    Parameters:
        image: The image which to calculate the gradient for.
        x: x coordinate.
        y: y coordinate.

    Returns:
        Image gradient as numpy array.
    """
    assert hlp.is_image(image)
    assert hlp.image_channels(image) == 1
    assert hlp.within_image_extent(hlp.image_size(image), x, y, 1)

    east = float(image[y, x + 1])
    west = float(image[y, x - 1])
    south = float(image[y + 1, x])
    north = float(image[y - 1, x])

    Gx = (east - west) / 2.0
    Gy = (south - north) / 2.0

    return np.array([Gx, Gy], dtype=np.float64)


def gradient_magnitude_sq(gradient: np.ndarray) -> float:
    """
    Compute the squared magnitude of a gradient.

    Parameters:
        gradient: The gradient.

    Returns:
        The squared magnitude.
    """
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == (2,)

    x, y = gradient
    return x * x + y * y


def gradient_orientation(gradient: np.ndarray) -> float:
    """
    Compute the orientation of a gradient [0, 180] degrees.
    """
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == (2,)

    x, y = gradient
    return math.degrees(math.atan2(y, x))


def gradient_visualization_images(image: np.ndarray, threshold: float) -> tuple:
    assert hlp.is_image(image)
    assert hlp.image_channels(image) == 1
    assert image.dtype == np.uint8

    w, h = hlp.image_size(image)

    Gx = np.ndarray((h, w), dtype=np.uint8)
    Gx[:, :] = 128

    Gy = np.ndarray((h, w), dtype=np.uint8)
    Gy[:, :] = 128

    strongest = np.zeros((h, w), dtype=np.uint8)

    threshold_sq = threshold * threshold

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            g = pixel_gradient(image, x, y)

            xx, yy = (g + 128.0).astype(np.uint8)
            Gx[y, x] = xx
            Gy[y, x] = yy

            if gradient_magnitude_sq(g) > threshold_sq:
                strongest[y, x] = 255

    return Gx, Gy, strongest
