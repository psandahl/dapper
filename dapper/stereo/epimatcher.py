from cmath import isnan
import cv2 as cv
import logging
import math
import numpy as np

import dapper.image.gradient as gr
import dapper.image.helpers as img_hlp
from dapper.image.line import Line
import dapper.math.matrix as mat
import dapper.math.helpers as mat_hlp
from dapper.math.ray import Ray

logger = logging.getLogger(__name__)


class EpiMatcher():
    """
    Class that is matching a keyframe against another frame along
    epipolar line.
    """

    def __init__(self, visualize: bool = False) -> None:
        """
        Construct the EpiMatcher.
        """
        logger.debug('Construct EpiMatcher object')

        # Initialize the matcher with a blank state.
        self.keyframe_id = None
        self.keyframe_image = None
        self.keyframe_pose = None
        self.keyframe_K = None
        self.keyframe_K_inv = None
        self.keyframe_strong_gradients = None
        self.keyframe_depth_map = None

        self.other_id = None
        self.other_image = None
        self.other_pose = None
        self.other_K = None

        self.keyframe_to_other_vec = None
        self.other_to_keyframe_vec = None
        self.keyframe_to_other = None
        self.other_to_keyframe = None

        self.visualize = visualize
        self.keyframe_visual_image = None
        self.other_visual_image = None

    def set_keyframe(self, frame_id: int, image: np.ndarray,
                     pose: np.ndarray, K: np.ndarray, depth_map: any) -> None:
        """
        Assign a new keyframe to the EpiMatcher.
        """
        assert img_hlp.is_image(image)
        assert img_hlp.image_channels(image) == 1
        assert isinstance(pose, np.ndarray)
        assert (4, 4) == pose.shape
        assert isinstance(K, np.ndarray)
        assert (3, 3) == K.shape

        logger.info(f'Set new keyframe id={frame_id}')

        # Set and compute stuff for the keyframe.
        self.keyframe_id = frame_id
        self.keyframe_image = image
        self.keyframe_pose = pose
        self.keyframe_K = K
        self.keyframe_K_inv = np.linalg.inv(K)
        self.keyframe_strong_gradients = gr.strong_gradients(image)
        self.keyframe_depth_map = depth_map

    def match(self, frame_id: int, image: np.ndarray, pose: np.ndarray, K: np.ndarray) -> None:
        """
        Match the given frame against the keyframe assigned to the matcher.
        """
        assert not self.keyframe_id is None and self.keyframe_id < frame_id
        assert img_hlp.is_image(image)
        assert img_hlp.image_channels(image) == 1
        assert isinstance(pose, np.ndarray)
        assert (4, 4) == pose.shape
        assert isinstance(K, np.ndarray)
        assert (3, 3) == K.shape

        logger.info(
            f'Start matching between keyframe id={self.keyframe_id} and id={frame_id}')
        logger.debug(
            f'{len(self.keyframe_strong_gradients)} strong image points are available in keyframe')

        # Set and compute stuff for the frame.
        self.other_id = frame_id
        self.other_image = image
        self.other_pose = pose
        self.other_K = K

        # Setup stuff that relates this frame and the keyframe.
        self.keyframe_to_other_vec = mat.decompose_pose(
            self.other_pose)[1] - mat.decompose_pose(self.keyframe_pose)[1]

        self.other_to_keyframe = mat.relative_pose(self.keyframe_pose,
                                                   self.other_pose)
        self.keyframe_to_other = np.linalg.inv(self.other_to_keyframe)

        # If visualization is active, prepare visualization images.
        if self.visualize:
            self.keyframe_visual_image = img_hlp.gray_to_bgr(
                self.keyframe_image)
            self.other_visual_image = img_hlp.gray_to_bgr(self.other_image)

        # Bugs ... try find why things project incorrectly.
        #p0 = np.array([0, 0, 3.0])
        #p1 = np.array([0, 0, 9.0])
        #p2 = np.array([0, 0, 27.0])
        # ds = [10, 100, 1000]
        # cs = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        # u = 100
        # v = 200

        # for depth, col in zip(ds, cs):
        #     p_key = mat_hlp.unproject_image(self.keyframe_K_inv, u, v, depth)

        #     px_key = mat_hlp.project_image(self.keyframe_K, p_key)
        #     p_oth = mat_hlp.homogeneous(self.keyframe_to_other, p_key)
        #     px_oth = mat_hlp.project_image(self.other_K, p_oth)

        #     print(f'p_key={p_key}')
        #     print(f'p_oth={p_oth}')
        #     print(f'px_key={px_key}')
        #     print(f'px_oth={px_oth}')

        #     cv.drawMarker(self.keyframe_visual_image, px_key.astype(int), col)

        #     if p_oth[2] > 0.0:
        #         cv.drawMarker(self.other_visual_image, px_oth.astype(int), col)

        # Dummy for testing ... just sample one from gradients.
        index = self.keyframe_strong_gradients[len(
            self.keyframe_strong_gradients) // 7]
        px = img_hlp.index_to_pixel(
            img_hlp.image_size(self.keyframe_image), index)

        self._search_epiline(px)

        # self._match_pixel(px)
        # self._visualize_epi(px)

    def _search_epiline(self, px: tuple) -> None:
        # Get near, mean and far samples along the epipolar ray.
        samples = self._epiline_ray(px)
        if samples is None:
            logger.debug(
                f'Failed to extract epipolar ray for px={px}')
            return

        ray, near_distance, mean_distance, far_distance = samples

        # Project the near and far distances to image positions
        # in the 'other' frame.
        near_px = mat_hlp.project_image(
            self.other_K, ray.point_at(near_distance))
        far_px = mat_hlp.project_image(
            self.other_K, ray.point_at(far_distance))

        epiline = Line(near_px, far_px, img_hlp.image_size(self.other_image))
        if not epiline.ok:
            logger.debug(f'Failed to extract epipolar line for px={px})')
            return

        if self.visualize:
            # Visualization in keyframe: marker for selected pixel
            # and circle for epipole.
            cv.drawMarker(self.keyframe_visual_image, px, (0, 255, 0))

            other = mat_hlp.homogeneous(
                self.other_to_keyframe, np.array([0, 0, 0]))
            epipole_key = mat_hlp.project_image(self.keyframe_K, other)
            cv.circle(self.keyframe_visual_image, epipole_key.astype(int),
                      3, (0, 255, 255), cv.FILLED)

            # Visualization in other frame: markers for the depth samples,
            # epipolar line from epipole to near point.
            mean_px = mat_hlp.project_image(
                self.other_K, ray.point_at(mean_distance))

            keyframe = mat_hlp.homogeneous(
                self.keyframe_to_other, np.array([0, 0, 0]))
            if math.isclose(keyframe[2], 0.0, abs_tol=1e-9):
                # Hack for horizontal stereo.
                keyframe[2] = 1e-8
            epipole_oth = mat_hlp.project_image(self.other_K, keyframe)

            cv.circle(self.other_visual_image, epipole_oth.astype(int),
                      3, (0, 255, 255), cv.FILLED)

            cv.line(self.other_visual_image, near_px.astype(int),
                    epipole_oth.astype(int), (255, 255, 255))
            cv.drawMarker(self.other_visual_image,
                          near_px.astype(int), (0, 0, 255))
            cv.drawMarker(self.other_visual_image,
                          mean_px.astype(int), (0, 255, 0))
            cv.drawMarker(self.other_visual_image,
                          far_px.astype(int), (255, 0, 0))

    def _epiline_ray(self, px: tuple) -> tuple:
        # TODO: Fetch real depth.
        near_depth = 5
        mean_depth = 25
        far_depth = 250

        # Pick points from the keyframe's depth distribution for this pixel.
        u, v = px
        near_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, u, v, near_depth)
        mean_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, u, v, mean_depth)
        far_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, u, v, far_depth)

        # Transform to other's frame.
        near_point = mat_hlp.homogeneous(self.keyframe_to_other, near_point)
        mean_point = mat_hlp.homogeneous(self.keyframe_to_other, mean_point)
        far_point = mat_hlp.homogeneous(self.keyframe_to_other, far_point)

        # If both near is behind the image plane, just give up.
        if near_point[2] < 0 and far_point[2] < 0:
            logger.debug('All epiline samples are behind camera')
            return None

        origin = mat_hlp.homogeneous(
            self.keyframe_to_other, np.array([0, 0, 0]))

        near_distance = Ray.distance(origin, near_point)
        mean_distance = Ray.distance(origin, mean_point)
        far_distance = Ray.distance(origin, far_point)

        ray = Ray(origin, far_point)

        # Check if the near or far distance must be changed.
        behind = origin[2] < 0
        if behind and near_point[2] < 0:
            offset = 1e-05 - near_point[2]
            grow = offset / math.cos(ray.angle(np.array([0, 0, 1])))

            logger.debug(f'Near was behind camera, grow distance with={grow}')

            near_distance += grow
        elif not behind and far_point[2] < 0:
            offset = 1e-05 - far_point[2]
            shrink = offset / math.cos(ray.angle(np.array([0, 0, -1])))

            logger.debug(
                f'Far was behind camera, shrink distance with={shrink}')

            far_distance += shrink

        return ray, near_distance, mean_distance, far_distance

    def _match_pixel(self, px: tuple) -> None:
        """
        Match the given pixel from the assigned keyframe with the new frame.
        """
        # TODO: Fetch depth and ranges from depth map.
        near_depth = 1
        curr_depth = 50
        far_depth = 250

        # Compute the epiline to get target pixels from keyframe.
        target_epiline = self._compute_target_epiline(px)
        if target_epiline is None:
            logger.debug(f'Failed to compute target epiline for px={px}')
            return

        target_epx, target_epy = target_epiline

        # Compute the epiline to search in the other frame. Here it is described
        # as a ray with lengths,  in the keyframe's camera frame.
        epiline, near_length, curr_length, far_length = self._compute_search_epiline(
            px, near_depth, curr_depth, far_depth)

        # Transform the ray into other's camera frame.
        epi_origin = mat_hlp.homogeneous(
            self.keyframe_to_other, np.array([0, 0, 0]))
        epi_dir = mat_hlp.homogeneous(
            self.keyframe_to_other, epiline, 0.0)
        assert np.linalg.norm(epi_dir) > 1.0 - \
            1e-06 and np.linalg.norm(epi_dir) < 1.0 + 1e-06

        # Adjust the near and far lengths from geometric constraints.
        adjusted_epilength = self._adjust_epilength(
            epi_origin, epi_dir, near_length, far_length)
        if adjusted_epilength is None:
            logger.debug(f'Failed to adjust epilength for px={px}')
            return

        near_depth, far_length = adjusted_epilength

        # Get points for the three depth/epipolar lengths.
        near_point = epi_origin + epi_dir * near_length
        near_px = mat_hlp.project_image(self.other_K, near_point)

        curr_point = epi_origin + epi_dir * curr_length
        curr_px = mat_hlp.project_image(self.other_K, curr_point)

        far_point = epi_origin + epi_dir * far_length
        far_px = mat_hlp.project_image(self.other_K, far_point)

        # Visualization stuff.
        if self.visualize:
            # Pixel in keyframe.
            cv.drawMarker(self.keyframe_visual_image, px, (0, 255, 0))

            # Epiline in keyframe, prolonged for visualization.
            u, v = px

            key_epi_start = (int(u - 20 * target_epx),
                             int(v - 20 * target_epy))
            key_epi_end = (int(u + 20 * target_epx), int(v + 20 * target_epy))
            cv.line(self.keyframe_visual_image, key_epi_start,
                    key_epi_end, (255, 255, 255))

            # Epiline in other
            cv.line(self.other_visual_image, near_px.astype(
                int), far_px.astype(int), (255, 255, 255))

            # Min, curr and max in other.
            cv.drawMarker(self.other_visual_image,
                          near_px.astype(int), (0, 0, 255))
            cv.drawMarker(self.other_visual_image,
                          curr_px.astype(int), (0, 255, 0))
            cv.drawMarker(self.other_visual_image,
                          far_px.astype(int), (255, 0, 0))

    def _adjust_epilength(self, epi_origin: np.ndarray, epi_dir: np.ndarray,
                          near_length: float, far_length: float) -> tuple:
        """
        Adjust the epilength from geometric constraints in viewspace for
        the other frame.
        """
        near = epi_origin + epi_dir * near_length
        far = epi_origin + epi_dir * far_length
        behind = epi_origin[2] < 0.0

        min_dist = 0.1

        if behind and near[2] <= min_dist:
            theta = math.acos(np.dot(epi_dir, np.array([0.0, 0.0, 1.0])))

            dist = min_dist - near[2]
            grow = dist / math.cos(theta)

            if near_length + grow >= far_length:
                logger.debug('Near length is far than far length')
                return None

            logger.debug(f'Grow near length with={grow}')

            near_length += grow
        elif not behind and far[2] <= min_dist:
            theta = math.acos(np.dot(epi_dir, np.array([0.0, 0.0, 1.0])))

            dist = min_dist - near[2]
            shrink = dist / math.cos(theta)

            if far_length - shrink <= near_length:
                logger.debug('Far length is near than near length')
                return None

            logger.debug(f'Shrink far length with={shrink}')

            far_length -= shrink
        elif behind and far[2] <= min_dist:
            logger.debug(
                'Far length cannot be adjusted when keyframe is behind')
            return None
        elif not behind and near[2] <= min_dist:
            logger.debug(
                'Near length cannot be adjusted when keyframe is in front')
            return None

        # adj_near = epi_origin + epi_dir * near_length
        # adj_far = epi_origin + epi_dir * far_length
        # logger.debug(f'near={near} adj near={adj_near}')
        # logger.debug(f'far={far} adj far={adj_far}')

        return near_length, far_length

    def _compute_target_epiline(self, px: tuple) -> tuple:
        """
        Compute the epiline in terms of pixel offsets
        to be used in the keyframe.
        """
        fx = self.keyframe_K[0, 0]
        fy = self.keyframe_K[1, 1]
        cx = self.keyframe_K[0, 2]
        cy = self.keyframe_K[1, 2]

        x, y = px

        epx = -fx * \
            self.keyframe_to_other_vec[0] + \
            (x - cx) * self.keyframe_to_other_vec[2]
        epy = -fy * \
            self.keyframe_to_other_vec[1] + \
            (y - cy) * self.keyframe_to_other_vec[2]

        if math.isnan(epx + epy):
            return None

        print(f'keyframe_to_other={self.keyframe_to_other_vec}')
        print(f'epx={epx:.2f} epy={epy:.2f}')

        sample_size = 1.0
        scale = sample_size / math.hypot(epx, epy)

        return epx * scale, epy * scale

    def _compute_search_epiline(self, px: tuple, near_depth: float,
                                curr_depth: float, far_depth: float) -> tuple:
        """
        Compute the epiline from the keyframe, in terms of
        viewspace ray. In keyframe's frame.
        """
        x, y = px
        min_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, x, y, near_depth)
        curr_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, x, y, curr_depth)
        max_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, x, y, far_depth)

        near_epilength = np.linalg.norm(min_point)
        curr_epilength = np.linalg.norm(curr_point)
        far_epilength = np.linalg.norm(max_point)

        epiline = min_point / near_epilength

        return (epiline, near_epilength, curr_epilength, far_epilength)

    def _visualize_epi(self, px: tuple) -> None:
        keyframe = img_hlp.gray_to_bgr(self.keyframe_image)
        other = img_hlp.gray_to_bgr(self.other_image)

        # Visualize the other camera's position in the keyframe image.
        other_in_key = mat_hlp.homogeneous(
            self.other_to_keyframe, np.array([0, 0, 0]))
        if other_in_key[2] > 0.0:
            other_in_key_px = mat_hlp.project_image(
                self.keyframe_K, other_in_key).astype(int)
            cv.circle(keyframe, other_in_key_px, 3, (0, 255, 0), cv.FILLED)
        else:
            logger.warning(
                f'Cannot visualize other camera, as it not is infront of camera')

        # Visualize the selected pixel in the keyframe image.
        # cv.circle(keyframe, px, 3, (255, 0, 0))
        cv.drawMarker(keyframe, px, (255, 0, 0))

        # Mid coordinate in other image.
        mid_px = mat_hlp.project_image(self.keyframe_K, np.array((0, 0, 50)))
        cv.circle(other, mid_px.astype(int), 3, (0, 255, 0), cv.FILLED)

        min_depth = 5.0
        max_depth = 100.0

        u, v = px
        near = mat_hlp.homogeneous(self.keyframe_to_other,
                                   mat_hlp.unproject_image(self.keyframe_K_inv, u, v, min_depth))
        far = mat_hlp.homogeneous(self.keyframe_to_other,
                                  mat_hlp.unproject_image(self.keyframe_K_inv, u, v, max_depth))

        print(f'near={near} far={far}')

        sample_size = 1.0

        near_px = mat_hlp.project_image(self.other_K, near)
        far_px = mat_hlp.project_image(self.other_K, far)
        print(f'near_px={near_px} far_px={far_px}')

        radiusx_px = far_px[0] - near_px[0]
        radiusy_px = far_px[1] - near_px[1]
        epilength_px = math.hypot(radiusx_px, radiusy_px)
        stepx_px = radiusx_px * (sample_size / epilength_px)
        stepy_px = radiusy_px * (sample_size / epilength_px)

        num_steps = epilength_px / sample_size

        print(f'epilength_px={epilength_px}')
        print(f'stepx_px={stepx_px} stepy_px={stepy_px}')

        print(f'Num steps={num_steps}')

        cv.drawMarker(other, near_px.astype(int), (0, 0, 255))
        cv.drawMarker(other, far_px.astype(int), (255, 0, 0))
        cv.line(other, near_px.astype(int),
                far_px.astype(int), (255, 255, 255))

        radiusx_view = far[0] - near[0]
        radiusy_view = far[1] - near[1]

        epilength_view = math.hypot(radiusx_view, radiusy_view)
        stepx_view = radiusx_view / num_steps
        stepy_view = radiusy_view / num_steps

        depth_range = max_depth - min_depth
        step_depth = depth_range / num_steps

        # step = 0
        # px = near_px
        # x = near[0]
        # y = near[1]
        # z = near[2]
        # while step <= num_steps:
        #     # print(f'px={px}')
        #     px[0] += stepx_px
        #     px[1] += stepy_px

        #     x += stepx_view
        #     y += stepy_view
        #     z += step_depth

        #     cv.drawMarker(other, px.astype(int), (255, 255, 255))

        #     step += 1

        # print(f'at end={np.array((x, y, z))}')

        cv.setWindowTitle('keyframe', f'keyframe={self.keyframe_id}')
        cv.setWindowTitle('other', f'other frame={self.other_id}')
        cv.imshow('keyframe', keyframe)
        cv.imshow('other', other)
