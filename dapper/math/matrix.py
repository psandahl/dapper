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


def pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Create a pose matrix from rotation and translation (the inversion of a 
    pose matrix is the extrinsic matrix).

    Parameters:        
        rotation: A rotation vector of length three [yaw, pitch, roll]
        in degrees.
        translation: Translation vector of length three.

    Returns:
        A 4x4 pose matrix.
    """
    return translation_matrix(translation) @ rotation_matrix(rotation)


def decompose_pose(mat: np.ndarray) -> tuple:
    """
    Decompose pose into rotation and translation.

    Parameters:
        mat: 4x4 pose matrix.

    Returns:
        Tuple ([yaw, pitch, roll], [x, y, z]). Angles are degrees.
    """
    return decompose_rotation(mat), decompose_translation(mat)


def relative_pose(pose_from: np.ndarray, pose_to: np.ndarray) -> np.ndarray:
    """
    Compute the relative pose between two poses. The relative pose
    is the inverse compared to what OpenCV return in 'reconverPose()'.

    Parameters:
        pose_from: 4x4 pose matrix.
        pose_to: 4x4 pose matrix.

    Returns:
        A 4x4 matrix with the relative pose describing pose_to
        relative to pose_from.
    """
    assert isinstance(pose_from, np.ndarray)
    assert pose_from.shape == (4, 4)
    assert isinstance(pose_to, np.ndarray)
    assert pose_to.shape == (4, 4)

    return np.linalg.inv(pose_from) @ pose_to


def apply_pose(pose_from: np.ndarray, pose_delta: np.ndarray) -> np.ndarray:
    """
    Apply a delta pose, to get a new pose.

    Parameters:
        pose_from: 4x4 pose matrix.
        pose_delta: 4x4 pose matrix.

    Returns:
        A 4x4 matrix with the new pose.
    """
    assert isinstance(pose_from, np.ndarray)
    assert pose_from.shape == (4, 4)
    assert isinstance(pose_delta, np.ndarray)
    assert pose_delta.shape == (4, 4)

    return pose_from @ pose_delta


def extract_3x3(mat: np.ndarray) -> np.ndarray:
    """
    Extract a 3x3 section from a matrix. If the matrix is too
    small the input is returned.

    Parameters:
        mat: Input matrix.

    Returns:
        A 3x3 matrix, from the input.
    """
    assert isinstance(mat, np.ndarray)

    rows, cols = mat.shape
    if min(rows, cols) >= 3 and max(rows, cols) > 3:
        return mat[:3, :3]
    else:
        logger.warning(
            f'Original matrix returned - size=({rows}, {cols})')
        return mat


def add_row(mat: np.ndarray) -> np.ndarray:
    """
    Add a new row to a matrix. The row will be ended with a one.

    Parameters:
        mat: A MxN matrix.

    Returns:
        A (M+1)xN matrix.
    """
    assert isinstance(mat, np.ndarray)

    _, cols = mat.shape
    row = np.zeros(cols, dtype=mat.dtype)
    row[-1] = 1

    return np.vstack((mat, row))
