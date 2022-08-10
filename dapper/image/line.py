import logging
import numpy as np

import dapper.common.settings as settings
import dapper.image.helpers as hlp

logger = logging.getLogger(__name__)


class Line():
    """
    A line abstraction for a 2d line that is clamped to and extent, and
    that can be tranversed.
    """

    def __init__(self, begin: np.ndarray, end: np.ndarray, size: tuple, border: int = 2) -> None:
        """
        Construct the line. The attribute ok must be checked after construction.

        Parameters:
            begin: The where the line starts.
            end: The point where the line ends.
            size: Tuple (width, height) for the target extent.
            border: Offset for a border inside the extent.            
        """
        assert isinstance(begin, np.ndarray)
        assert (2,) == begin.shape
        assert isinstance(end, np.ndarray)
        assert (2,) == end.shape
        assert len(size) == 2

        self.begin = begin
        self.end = end
        self.length = np.linalg.norm(end - begin)

        self.step_size = 1.0 / settings.EPILINE_SAMPLE_SIZE
        self.num_steps = self.length / self.step_size

        self.min_distance = 0.0
        self.max_distance = self.length

        self.gradient = end - begin
        self.gradient = self.gradient / np.linalg.norm(self.gradient)

        while not hlp.within_image_extent(size, self.point_at(self.min_distance), border) \
                and self.min_distance < self.max_distance:
            self.min_distance += self.step_size

        while not hlp.within_image_extent(size, self.point_at(self.max_distance), border) \
                and self.max_distance > self.min_distance:
            self.max_distance -= self.step_size

        self.curr_distance = self.min_distance

        self.ok = self.min_distance < self.max_distance

    def point(self, offset: float = 0.0) -> np.ndarray:
        """
        Return the current point on the line.

        Parameters:
            offset: The offset (in step_size) from the current distance.

        Returns:
            The requested point.
        """
        return self.point_at(self.curr_distance + offset * self.step_size)

    def point_at(self, distance: float) -> np.ndarray:
        """
        Return a point at the given distance along the line.

        Parameters:
            distance: The distance along the line.

        Returns:
            The requested point.
        """
        return self.begin + distance * self.gradient

    def forward(self) -> None:
        """
        Step forward one step_size.
        """
        self.curr_distance += self.step_size

    def ratio(self) -> float:
        """
        Get the ratio [0, 1] of the current distance relative the line's length.
        """
        return self.curr_distance / self.length
