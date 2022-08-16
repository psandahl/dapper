import cv2 as cv
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


def is_image(value: any) -> bool:
    """
    Check if something is an image.

    Parameters:
        value: Something to check.

    Returns:
        True if image.
    """
    if isinstance(value, np.ndarray):
        if len(value.shape) == 3:
            _, _, c = value.shape
            return c <= 4  # Color + alpha
        else:
            return len(value.shape) == 2
    else:
        return False


def image_size(image: np.ndarray) -> tuple:
    """
    Return the size in pixels for the image.

    Parameters:
        image: The image.

    Returns:
        Tuple (width, height).
    """
    assert is_image(image)

    if len(image.shape) == 3:
        h, w, _ = image.shape
        return w, h
    else:
        h, w = image.shape
        return w, h


def image_channels(image: np.ndarray) -> int:
    """
    Return the channels for an image.

    Parameters:
        image: The image.

    Returns:
        The number of channels.
    """
    assert is_image(image)

    if len(image.shape) == 3:
        _, _, c = image.shape
        return c
    else:
        return 1


def within_image_extent(image_size: tuple, px: any, border: int = 0) -> bool:
    """
    Check that an image coordinate is within the extent of the image.

    Parameters:
        image_size: Tuple (width, height).
        px: Tuple or array (x, y).
        border: Extra border inside of image.

    Returns:
        True if the coordinate is within the image (+border).
    """
    assert len(image_size) == 2
    assert len(px) == 2

    w, h = image_size
    x, y = px
    return (x - border) >= 0 and (x + border) < w and (y - border) >= 0 and (y + border) < h


def pixel_to_index(image_size: tuple, x: int, y: int) -> int:
    """
    Convert a pixel coordinate to index image.

    Parameters:
        image_size: Tuple (w, h).
        x: x coordinate.
        y: y coordinate.

    Returns:
        Image index.
    """
    assert len(image_size) == 2
    assert within_image_extent(image_size, (x, y))

    w, _ = image_size
    return y * w + x


def index_to_pixel(image_size: tuple, index: int) -> tuple:
    """
    Convert an image index to pixel coordinate.

    Parameters:
        image_size: Tuple (w, h).
        index: Image index.

    Returns:
        Tuple (x, y).
    """
    assert len(image_size) == 2

    w, _ = image_size
    return index % w, index // w


def gray_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Create a BGR copy of the given gray scale image.
    """
    assert is_image(image)
    assert image_channels(image)

    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)


def px_interpolate(image: np.ndarray, px: any) -> any:
    """
    Read an image, interpolated between four pixels.

    Parameters:
        image: The image to read.
        px: A pixel (x, y), tuple, list or array.

    Returns:
        The color.
    """
    assert is_image(image)
    assert len(px) == 2

    x, y = px
    assert x >= 0.0
    assert y >= 0.0

    # Get the integer part of pixel.
    i_x = math.floor(x)
    i_y = math.floor(y)

    w, h = image_size(image)
    if i_x < w - 1 and i_y < h - 1:
        # Fractional part of pixel.
        f_x = x - i_x
        f_y = y - i_y

        w00 = (1.0 - f_x) * (1.0 - f_y)
        w10 = f_x * (1.0 - f_y)
        w01 = (1.0 - f_x) * f_y
        w11 = f_x * f_y

        px00 = image[i_y, i_x]
        px10 = image[i_y, i_x + 1]
        px01 = image[i_y + 1, i_x]
        px11 = image[i_y + 1, i_x + 1]

        return w00 * px00 + w10 * px10 + w01 * px01 + w11 * px11
    elif i_x < w and i_y < h:
        # No interpolation at the border.
        return image[i_y, i_x]
    else:
        return None
