import logging
import numpy as np
import math

logger = logging.getLogger(__name__)


def ideal_intrinsic_matrix(image_size: tuple, fov: float) -> np.ndarray:
    """
    Create an ideal intrinsic matrix (K) from image size and horizontal field
    of view.

    Parameters:
        image_size: Tuple (width, height).
        fov: Horizontal field of view, in degrees.
    """
    assert len(image_size) == 2

    w, h = image_size
    fov = math.radians(fov)

    fx = (w / 2.0) / math.tan(fov / 2.0)
    fy = fx
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    mat = [fx, 0, cx, 0, fy, 0, 0, 0, 1]

    return np.array(mat, dtype=np.float64).reshape(3, 3)


def project_image(K: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Project a viewspace coordinate onto the image plane.

    Parameters:
        K: Intrinsic matrix.
        xyz: Viewspace coordinate.

    Returns:
        Pixel coordinate, array of length 2.
    """
    assert isinstance(K, np.ndarray)
    assert (3, 3) == K.shape
    assert isinstance(xyz, np.ndarray)
    assert (3,) == xyz.shape

    px = K @ xyz
    px /= px[2]

    return px[:2]


def unproject_image(K_inv: np.ndarray, x: float, y: float, depth: float = 1.0) -> np.ndarray:
    """
    Unproject a pixel coordinate and bring back into viewspace.

    Parameters:
        K_inv: The inverse intrinsic matrix.
        x: X coordinate.
        y: Y coordinate.
        depth: Scene depth.

    Returns:
        Viewspace coordinate, array of length 3.
    """
    assert isinstance(K_inv, np.ndarray)
    assert (3, 3) == K_inv.shape

    px = np.array([x, y, 1.0])
    return depth * (K_inv @ px)


def homogeneous(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """
    Homogeneous multiplication of a 3d vector with a 4x4 matrix.

    Parameters:
        mat: A 4x4 matrix.
        xyz: A 3d vector, which will be appended with '1' during
        multiplication.

    Returns:
        A transformed 3d vector.
    """
    assert isinstance(mat, np.ndarray)
    assert (4, 4) == mat.shape

    assert isinstance(xyz, np.ndarray)
    assert (3,) == xyz.shape

    xyz2 = mat @ np.append(xyz, 1.0)
    xyz2 /= xyz2[3]

    return xyz2[:3]
