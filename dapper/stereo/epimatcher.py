from cmath import isnan
import cv2 as cv
import logging
import math
import numpy as np

import dapper.image.gradient as gr
import dapper.image.helpers as img_hlp
import dapper.math.matrix as mat
import dapper.math.helpers as mat_hlp

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

        # Dummy for testing ... just sample one from gradients.
        index = self.keyframe_strong_gradients[len(
            self.keyframe_strong_gradients) // 2]
        px = img_hlp.index_to_pixel(
            img_hlp.image_size(self.keyframe_image), index)

        self._match_pixel(px)

        # self._keyframe_epiline(px)

        # self._visualize_epi(px)

    def _match_pixel(self, px: tuple) -> None:
        """
        Match the given pixel from the assigned keyframe with the new frame.
        """
        keyframe_epiline = self._keyframe_epiline(px)
        if keyframe_epiline is None:
            logger.debug(f'Failed to compute keyframe epiline for px={px}')
            return

        other_epiline = self._other_epiline(px)

    def _keyframe_epiline(self, px: tuple) -> tuple:
        """
        Compute the epiline in terms of pixel offsets 
        to be used in the keyframe.
        """
        fx = self.keyframe_K[0, 0]
        fy = self.keyframe_K[1, 1]
        cx = self.keyframe_K[0, 2]
        cy = self.keyframe_K[1, 2]

        x, y = px

        epx = - fx * \
            self.keyframe_to_other_vec[0] + \
            (x - cx) * self.keyframe_to_other_vec[2]
        epy = - fy * \
            self.keyframe_to_other_vec[1] + \
            (y - cy) * self.keyframe_to_other_vec[2]

        if math.isnan(epx + epy):
            return None

        sample_size = 1.0
        scale = sample_size / math.hypot(epx, epy)

        return epx * scale, epy * scale

    def _other_epiline(self, px: tuple) -> tuple:
        """
        Compute the epiline from the keyframe, in terms of
        viewspace ray. In keyframe's frame.
        """
        # TODO: Fetch depth and ranges from depth map.
        min_depth = 15
        curr_depth = 31
        max_depth = 38

        x, y = px
        min_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, x, y, min_depth)
        curr_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, x, y, curr_depth)
        max_point = mat_hlp.unproject_image(
            self.keyframe_K_inv, x, y, max_depth)

        min_epilength = np.linalg.norm(min_point)
        curr_epilength = np.linalg.norm(curr_point)
        max_epilength = np.linalg.norm(max_point)

        epiline = min_point / min_epilength

        return (epiline, min_epilength, curr_epilength, max_epilength)

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
        #cv.circle(keyframe, px, 3, (255, 0, 0))
        cv.drawMarker(keyframe, px, (255, 0, 0))

        # Mid coordinate in other image.
        mid_px = mat_hlp.project_image(self.keyframe_K, np.array((0, 0, 50)))
        cv.circle(other, mid_px.astype(int), 3, (0, 255, 0), cv.FILLED)

        min_depth = 5.0
        max_depth = 50.0

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

        radiusx_view = far[0] - near[0]
        radiusy_view = far[1] - near[1]

        epilength_view = math.hypot(radiusx_view, radiusy_view)
        stepx_view = radiusx_view / num_steps
        stepy_view = radiusy_view / num_steps

        depth_range = max_depth - min_depth
        step_depth = depth_range / num_steps

        step = 0
        px = near_px
        x = near[0]
        y = near[1]
        z = near[2]
        while step <= num_steps:
            # print(f'px={px}')
            px[0] += stepx_px
            px[1] += stepy_px

            x += stepx_view
            y += stepy_view
            z += step_depth

            cv.drawMarker(other, px.astype(int), (255, 255, 255))

            step += 1

        print(f'at end={np.array((x, y, z))}')

        cv.setWindowTitle('keyframe', f'keyframe={self.keyframe_id}')
        cv.setWindowTitle('other', f'other frame={self.other_id}')
        cv.imshow('keyframe', keyframe)
        cv.imshow('other', other)
