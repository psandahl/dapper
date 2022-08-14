import copy
import enum
import logging
import numpy as np

import dapper.common.settings as settings
import dapper.math.helpers as img_hlp
import dapper.math.plane as plane
from dapper.math.ray import Ray

logger = logging.getLogger(__name__)


class PlaneId(enum.IntEnum):
    NEAR = 0
    FAR = 1
    TOP = 2
    BOTTOM = 3
    LEFT = 4
    RIGHT = 5


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

        # Corner rays at depth one.
        ul = img_hlp.unproject_image(K_inv, 0, 0)
        ur = img_hlp.unproject_image(K_inv, w - 1, 0)
        ll = img_hlp.unproject_image(K_inv, 0, h - 1)
        lr = img_hlp.unproject_image(K_inv, w - 1, h - 1)

        # Corner to be used for filtering.
        self.corner = lr

        # Setup all planes.
        self.planes = [None] * 6

        self.planes[PlaneId.NEAR] = plane.plane(np.array([0.0, 0.0, 1.0]),
                                                settings.FRUSTUM_NEAR_PLANE)
        self.planes[PlaneId.FAR] = plane.plane(np.array([0.0, 0.0, -1.0]),
                                               -settings.FRUSTUM_FAR_PLANE)
        self.planes[PlaneId.TOP] = plane.plane(np.cross(ul, ur), 0.0)
        self.planes[PlaneId.BOTTOM] = plane.plane(np.cross(lr, ll), 0.0)
        self.planes[PlaneId.LEFT] = plane.plane(np.cross(ll, ul), 0.0)
        self.planes[PlaneId.RIGHT] = plane.plane(np.cross(ur, lr), 0.0)

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

        for p in self.planes:
            if not plane.infront_of(p, point):
                return False

        return True

    def intersect_inside(self, ray: Ray) -> float:
        """
        Intersect a ray (given its origin is inside the frustum)
        with the frustum boundaries.

        Parmeters:
            ray: The ray.

        Returns:
            The distance to frustum border, or None.
        """
        assert isinstance(ray, Ray)

        if not self.contains(ray.origin):
            return None

        for p in self.planes:
            t = plane.raycast(p, ray)
            if not t is None:
                return t - 1e-09

        return None

    def intersect_outside(self, ray: Ray) -> float:
        """
        Intersect a ray (given its origin is outside the frustum)
        with the frustum boundaries.

        Parmeters:
            ray: The ray.

        Returns:
            The distance to frustum border, or None.
        """
        assert isinstance(ray, Ray)

        if self.contains(ray.origin):
            return None

        for i, p in enumerate(self.planes):
            t = plane.raycast(-p, ray)  # Plane is negated.
            if not t is None:
                # Filter plane outside of core frustum.
                x, y, z = ray.point_at(t)
                if i == PlaneId.NEAR:
                    corner = self.corner * settings.FRUSTUM_NEAR_PLANE
                    if abs(x) > corner[0] or abs(y) > corner[1]:
                        continue
                elif i == PlaneId.FAR:
                    corner = self.corner * settings.FRUSTUM_FAR_PLANE
                    if abs(x) > corner[0] or abs(y) > corner[1]:
                        continue
                elif i == PlaneId.TOP or PlaneId.BOTTOM:
                    corner = self.corner * z
                    if z < settings.FRUSTUM_NEAR_PLANE or z > settings.FRUSTUM_FAR_PLANE \
                            or abs(x) > corner[0]:
                        continue
                elif i == PlaneId.LEFT or PlaneId.RIGHT:
                    corner = self.corner * z
                    if z < settings.FRUSTUM_NEAR_PLANE or z > settings.FRUSTUM_FAR_PLANE \
                            or abs(y) > corner[1]:
                        continue

                return t - 1e-09

        return None

    def clamp_ray(self, ray: Ray, near: float, far: float) -> tuple:
        """
        Clamp a ray to the frustum, i.e. near and far must be contained
        inside, given that the ray intersects the frustum.

        Parameters:
            ray: The ray.
            near: The near distance along the ray.
            far: The far distance along the ray.

        Returns:
            Tuple (new near, new far), or None it ray does not intersect.
        """
        assert isinstance(ray, Ray)

        if self.contains(ray.point_at(near)) and self.contains(ray.point_at(far)):
            # Simplest case: both near and far are inside the frustum.
            return near, far
        elif self.contains(ray.origin) and self.contains(ray.point_at(near)) \
                and not self.contains(ray.point_at(far)):
            # The next case is when the ray origin and the near is inside the
            # frustum, but the far is outside. Clamp far to the inside.
            t = self.intersect_inside(ray)
            assert not t is None
            assert self.contains(ray.point_at(t))

            return near, t
        elif not self.contains(ray.origin) and not self.contains(ray.point_at(near)):
            # The last case is when both origin and near is outside, and far can be either
            # way.
            t = self.intersect_outside(ray)

            if t is None:
                # Ray does not intersect frustum at all.
                return None

            # Set new near, just inside frustum.
            new_near = t + 2 * 1e-09
            assert self.contains(ray.point_at(new_near))

            # Check if far is contained inside frustum.
            if self.contains(ray.point_at(far)):
                # Yes. We're done.
                return new_near, far

            # Create a new ray, with origin at new_near.
            tmp_ray = copy.deepcopy(ray)
            tmp_ray.origin = ray.point_at(new_near)

            # Clamp to frustum border.
            t = self.intersect_inside(tmp_ray)
            assert not t is None
            assert self.contains(tmp_ray.point_at(t))

            return new_near, new_near + t

        # A case we do not solve. Return None.
        return None
