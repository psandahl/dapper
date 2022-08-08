import logging
import numpy as np

logger = logging.getLogger(__name__)


class Ray():
    """
    Representation of a ray with origin and a direction.
    """

    def __init__(self, origin: np.ndarray, look_at: np.ndarray) -> None:
        """
        Construct a ray.

        Parameters:
            origin: The origin of the ray.
            look_at: A point where the ray look.
        """
        assert isinstance(origin, np.ndarray)
        assert (3,) == origin.shape
        assert isinstance(look_at, np.ndarray)
        assert (3,) == look_at.shape

        self.origin = origin
        self.direction = look_at - origin
        self.direction /= np.linalg.norm(self.direction)

    def point_at(self, distance: float) -> np.ndarray:
        """
        Get a point along the ray.

        Parameters:
            distance: The distance.

        Returns:
            The point.
        """
        return self.origin + self.direction * distance

    @staticmethod
    def distance(point0: np.ndarray, point1: np.ndarray) -> float:
        """
        Static method to 
        """
        assert isinstance(point0, np.ndarray)
        assert (3,) == point1.shape
        assert isinstance(point1, np.ndarray)
        assert (3,) == point1.shape

        return np.linalg.norm(point1 - point0)
