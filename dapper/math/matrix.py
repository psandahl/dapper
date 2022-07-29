import logging
import numpy as np
import math

logger = logging.getLogger(__name__)


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    Compute an Euler rotation matrix in axis order y, x and z.

    Parameters:
        yaw: Yaw angle in degrees.
        pitch: Pitch angle in degrees.
        roll: Roll angle in degrees.

    Returns:
        A 4x4 rotation matrix.
    """
    y = np.radians(yaw)
    x = np.radians(pitch)
    z = np.radians(roll)

    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    cz = math.cos(z)
    sz = math.sin(z)

    mat = [cy * cz + sx * sy * sz, cz * sx * sy - cy * sz, cx * sy, 0.0,
           cx * sz, cx * cz, -sx, 0.0,
           -cz * sy + cy * sx * sz, cy * cz * sx + sy * sz, cx * cy, 0.0,
           0.0, 0.0, 0.0, 1.0
           ]

    return np.array(mat).reshape(4, 4)


def decompose_rotation_matrix(mat: np.ndarray) -> tuple:
    """
    Decompose an Euler rotation matrix in order y, x, and z into
    yaw, pitch and roll.

    Parameters:
        mat: 4x4 matrix.

    Returns:
        Tuple (yaw, pitch, roll) in degrees.
    """
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (4, 4)

    y = math.atan2(mat[0, 2], mat[2, 2])
    x = math.asin(-mat[1, 2])
    z = math.atan2(mat[1, 0], mat[1, 1])

    return (math.degrees(y), math.degrees(x), math.degrees(z))
