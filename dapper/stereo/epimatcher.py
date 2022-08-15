import cv2 as cv
import logging
import math
import numpy as np

import dapper.common.settings as settings
from dapper.math.frustum import Frustum
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
        self.other_frustum = None

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
        self.other_frustum = Frustum(
            np.linalg.inv(K), img_hlp.image_size(image))

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

        # Dummy for testing ... just sample one from gradients.
        index = self.keyframe_strong_gradients[len(
            self.keyframe_strong_gradients) // 4]
        px = img_hlp.index_to_pixel(
            img_hlp.image_size(self.keyframe_image), index)

        self._search_along_epiline(px)

        # self._match_pixel(px)
        # self._visualize_epi(px)

    def _search_along_epiline(self, px: tuple) -> None:
        # Get near, mean and far samples along the epipolar ray.
        samples = self._epiline_ray(px)
        if samples is None:
            logger.debug(
                f'Failed to extract epipolar ray for px={px}')
            return

        ray, near_distance, mean_distance, far_distance = samples
        near = ray.point_at(near_distance)
        far = ray.point_at(far_distance)

        # Project the near and far distances to image positions
        # in the 'other' frame, in order to get an epipolar line
        # to search along in the other frame.
        near_px = mat_hlp.project_image(self.other_K, near)
        far_px = mat_hlp.project_image(self.other_K, far)

        epiline = Line(near_px, far_px, img_hlp.image_size(self.other_image))
        if not epiline.ok:
            logger.debug(f'Failed to extract epipolar line for px={px})')
            return

        # Get the epiline where to find the template pixels in
        # the keyframe.
        epiline_key = self._epiline_key(px)

        if self.visualize:
            # Visualization in keyframe: marker for selected pixel
            # and circle for epipole.
            other_in_key = mat_hlp.homogeneous(
                self.other_to_keyframe, np.array([0, 0, 0]))
            if math.isclose(other_in_key[2], 0.0, abs_tol=1e-9):
                # Hack for horizontal stereo.
                other_in_key[2] = 1e-8
            epipole_key_px = mat_hlp.project_image(
                self.keyframe_K, other_in_key)

            cv.line(self.keyframe_visual_image, px, epipole_key_px.astype(int),
                    (128, 128, 128))
            cv.drawMarker(self.keyframe_visual_image, px, (0, 255, 0))
            cv.circle(self.keyframe_visual_image, epipole_key_px.astype(int),
                      3, (0, 255, 255), cv.FILLED)

            begin = px - 10.0 * settings.EPILINE_SAMPLE_SIZE * epiline_key
            end = px + 10.0 * settings.EPILINE_SAMPLE_SIZE * epiline_key
            cv.line(self.keyframe_visual_image, begin.astype(int),
                    end.astype(int), (255, 0, 0), 3)

            # Visualization in other frame: markers for the depth samples,
            # epipolar line from epipole to near point.

            key_in_other = mat_hlp.homogeneous(
                self.keyframe_to_other, np.array([0, 0, 0]))
            if math.isclose(key_in_other[2], 0.0, abs_tol=1e-9):
                # Hack for horizontal stereo.
                key_in_other[2] = 1e-8
            epipole_oth_px = mat_hlp.project_image(self.other_K, key_in_other)

            cv.line(self.other_visual_image, far_px.astype(int),
                    epipole_oth_px.astype(int), (128, 128, 128))
            cv.circle(self.other_visual_image, epipole_oth_px.astype(int),
                      3, (0, 255, 255), cv.FILLED)

            epinear_px = epiline.point_at(epiline.min_distance)
            epifar_px = epiline.point_at(epiline.max_distance)
            cv.line(self.other_visual_image, epinear_px.astype(int),
                    epifar_px.astype(int), (255, 0, 0), 3)

    def _epiline_ray(self, px: tuple) -> tuple:
        """
        Compute the epiline ray, from the pixel in the keyframe, and
        produce a ray, together with distance ranges, in the other frame. 
        If the computation fails, None is returned.
        """
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

        # If both near and far is behind the image plane, just give up.
        if near_point[2] < 0 and far_point[2] < 0:
            logger.debug('All epiline samples are behind camera')
            return None

        origin = mat_hlp.homogeneous(
            self.keyframe_to_other, np.array([0, 0, 0]))

        near_distance = Ray.distance(origin, near_point)
        mean_distance = Ray.distance(origin, mean_point)
        far_distance = Ray.distance(origin, far_point)

        # Clamp the ray to the frustum.
        ray = Ray(origin, far_point)
        distances = self.other_frustum.clamp_ray(
            ray, near_distance, far_distance)
        if distances is None:
            logger.debug('Failed to clamp ray to frustum')
            return None

        near_distance, far_distance = distances

        return ray, near_distance, mean_distance, far_distance

    def _epiline_key(self, px: tuple) -> np.ndarray:
        """
        Compute the epiline for use in the keyframe. The
        epiline is represented as normalized gradient.
        """
        # The epiline in the keyframe is between the epipole
        # (other camera position) and the pixel.
        epipole = mat_hlp.homogeneous(
            self.other_to_keyframe, np.array([0.0, 0.0, 0.0]))
        if math.isclose(epipole[2], 0.0, abs_tol=1e-9):
            # Hack for horizontal stereo.
            epipole[2] = 1e-8

        epipole_px = mat_hlp.project_image(self.keyframe_K, epipole)
        gradient = px - epipole_px

        return gradient / np.linalg.norm(gradient)
