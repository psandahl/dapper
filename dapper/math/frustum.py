import logging
import numpy as np

import dapper.common.settings as settings
import dapper.math.helpers as img_hlp
import dapper.math.plane as plane

logger = logging.getLogger(__name__)


class Frustum():
    """
    Frustum representation.
    """

    def __init__(self, K_inv: np.ndarray, image_size: tuple) -> None:
        """
        Construct frustum.

        Parameters:
            K_inv: Inverse K matrix.
            image_size: Tuple (width, height).
        """
        assert isinstance(K_inv, np.ndarray)
        assert (3, 3) == K_inv.shape
        assert len(image_size) == 2

        w, h = image_size

        ul = img_hlp.unproject_image(K_inv, 0, 0)
        ur = img_hlp.unproject_image(K_inv, w - 1, 0)
        ll = img_hlp.unproject_image(K_inv, 0, h - 1)
        lr = img_hlp.unproject_image(K_inv, w - 1, h - 1)

        self.near = plane.plane(np.array([0.0, 0.0, 1.0]),
                                settings.FRUSTUM_NEAR_PLANE)
        self.far = plane.plane(np.array([0.0, 0.0, -1.0]),
                               -settings.FRUSTUM_FAR_PLANE)
        self.top = plane.plane(np.cross(ul, ur), 0.0)
        self.bottom = plane.plane(np.cross(lr, ll), 0.0)
        self.left = plane.plane(np.cross(ll, ul), 0.0)
        self.right = plane.plane(np.cross(ur, lr), 0.0)

    def contains(self, point: np.ndarray) -> bool:
        """
        Check if a point is contained in the frustum.

        Parameters:
            point: Point in the frustums coordinate frame.

        Returns:
            True if contained in frustum, False otherwise.
        """
        assert isinstance(point, np.ndarray)
        assert (3,) == point.shape

        return plane.infront_of(self.near, point) and \
            plane.infront_of(self.far, point) and \
            plane.infront_of(self.top, point) and \
            plane.infront_of(self.bottom, point) and \
            plane.infront_of(self.left, point) and \
            plane.infront_of(self.right, point)
