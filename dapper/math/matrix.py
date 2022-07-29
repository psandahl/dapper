import logging
import numpy as np
import math

logger = logging.getLogger(__name__)


def rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """
    Create an Euler rotation matrix in axis order y, x and z.

    Parameters:
        rotation: A rotation vector of length three [yaw, pitch, roll]
        in degrees.

    Returns:
        A 4x4 rotation matrix.
    """
    assert isinstance(rotation, np.ndarray)
    assert rotation.shape == (3,)

    radians = np.radians(rotation)
    cy, cx, cz = np.cos(radians)
    sy, sx, sz = np.sin(radians)

    mat = [cy * cz + sx * sy * sz, cz * sx * sy - cy * sz, cx * sy, 0.0,
           cx * sz, cx * cz, -sx, 0.0,
           -cz * sy + cy * sx * sz, cy * cz * sx + sy * sz, cx * cy, 0.0,
           0.0, 0.0, 0.0, 1.0
           ]

    return np.array(mat).reshape(4, 4)


def decompose_rotation(mat: np.ndarray) -> np.ndarray:
    """    
    Decompose a 4x4 matrix rotation part into Euler rotations. Axis
    order y, x and z.

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

    return np.degrees((y, x, z))


def translation_matrix(translation: np.ndarray) -> np.ndarray:
    """
    Create a translation matrix.

    Parameters:
        translation: Translation vector of length three.

    Returns:
        A 4x4 translation matrix.
    """
    assert isinstance(translation, np.ndarray)
    assert translation.shape == (3,)

    mat = np.eye(4, 4, dtype=np.float64)
    mat[0, 3] = translation[0]
    mat[1, 3] = translation[1]
    mat[2, 3] = translation[2]

    return mat


def decompose_translation(mat: np.ndarray) -> np.ndarray:
    """
    Decompose a 4x4 matrix translation part into an array.

    Parameters:
        mat: 4x4 matrix.

    Returns:
        Array with translation x, y and z.
    """
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (4, 4)

    x = mat[0, 3]
    y = mat[1, 3]
    z = mat[2, 3]

    return np.array([x, y, z])
