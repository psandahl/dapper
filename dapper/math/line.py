import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class Line():
    """
    Class implementing a few simple line functions related to
    the line equation.
    """

    def __init__(self, begin: np.ndarray, end: np.ndarray) -> None:
        """
        Create a line from two points. The line is represented
        as a line equation: y = kx + m, where k is the slope
        and m is the y intercept.

        Special treatment is done to horizontal and vertical lines.

        Parameters:
            begin: The line's start.
            end: The line's end.
        """
        assert isinstance(begin, np.ndarray)
        assert (2,) == begin.shape
        assert isinstance(end, np.ndarray)
        assert (2,) == end.shape
        assert begin[0] <= end[0]

        self.begin = begin
        self.end = end
        self.length = np.linalg.norm(end - begin)

        dx = end[0] - begin[0]
        dy = end[1] - begin[1]

        if math.isclose(0.0, abs(dx), abs_tol=1e-12):
            self.k = math.inf
            self.m = math.nan
        elif math.isclose(0.0, abs(dy), abs_tol=1e-12):
            self.k = 0.0
            self.m = begin[1]
        else:
            # Lines captured by line equation.
            self.k = dy / dx
            self.m = end[1] - self.k * end[0]

    def y_at(self, x: float) -> float:
        """
        Get the y coordinate for the given x.

        Parameters:
            x: x coordinate.

        Returns:
            The y coordinate. If the line is verical, nan is returned.
        """
        if not self.is_vertical():
            return x * self.k + self.m
        else:
            return math.nan

    def x_at(self, y: float) -> float:
        """
        Get the x coordinate for the given y.

        Parameters:
            y: y coordinate.

        Returns:
            The coordinate. If the line is horizontal, nan is returned.
        """
        if self.is_horizontal():
            return math.nan
        elif self.is_vertical():
            return self.begin[0]
        else:
            return (y - self.m) / self.k

    def is_horizontal(self) -> bool:
        """
        Check if the line is identified as horizontal.
        """
        return math.isclose(0.0, abs(self.k), abs_tol=1e-12)

    def is_vertical(self) -> bool:
        """
        Check is the line is identified as vertical.
        """
        return math.isnan(self.m)

    def is_normal(self) -> bool:
        """
        Check if the line is normal.
        """
        return not (self.is_vertical() or self.is_horizontal())
