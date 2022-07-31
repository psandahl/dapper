import cv2 as cv
import logging
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
