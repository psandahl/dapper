import dapper.math.helpers as mat_hlp
from dapper.math.ray import Ray
import logging
import numpy as np

logger = logging.getLogger(__name__)


def plane(direction: np.ndarray, distance: float) -> np.ndarray:
    """
    Create a plane with an orientation and a distance from origo.

    Parameters:
        direction: The direction of the plane.
        distance: Distance from origo along the direction.

    Returns:
        A plane as array [normal, distance].
    """
    assert isinstance(direction, np.ndarray)
    assert (3,) == direction.shape

    return np.append(mat_hlp.normalize(direction), distance)


def distance_to(plane: np.ndarray, point: np.ndarray) -> float:
    """
    Compute the distance from a point to the plane.
    > 0 means infront of plane, == 0 on plane and < 0
    behind plane.

    Parameters:
        plane: The plane
        point: The point.

    Returns:
        The distance.
    """
    assert isinstance(plane, np.ndarray)
    assert (4,) == plane.shape
    assert isinstance(point, np.ndarray)
    assert (3,) == point.shape

    return np.dot(point, plane[:3]) - plane[3]


def infront_of(plane: np.ndarray, point: np.ndarray) -> bool:
    """
    Check if a point is infront of a plane.

    Parameters:
        plane: The plane
        point: The point.

    Returns:
        True if infront of, False otherwise.
    """
    return distance_to(plane, point) > 0.0


def raycast(plane: np.ndarray, ray: Ray) -> float:
    """
    Raycast a plane.

    Parameters:
        plane: The plane.
        ray: The ray.

    Returns:
        The distance to the plane along the ray, or None.
    """
    assert isinstance(plane, np.ndarray)
    assert (4,) == plane.shape
    assert isinstance(ray, Ray)

    # The dot product of the plane and the ray direction must
    # be negative.
    normal = plane[:3]

    nd = np.dot(ray.direction, normal)
    if nd >= 0.0:
        return None

    pn = np.dot(ray.origin, normal)

    # Compute the length at intersection point.
    t = (plane[3] - pn) / nd

    # If positive, intersection.
    return t if t >= 0.0 else None
