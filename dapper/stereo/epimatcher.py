import cv2 as cv
import logging
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

    def __init__(self) -> None:
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

        self.keyframe_to_other = None
        self.other_to_keyframe = None

    def set_keyframe(self, frame_id: int, image: np.ndarray,
                     pose: np.ndarray, K: np.ndarray, depth_map: any) -> None:
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

        # Compute matrices to transform between keyframe and the other
        self.other_to_keyframe = mat.relative_pose(self.keyframe_pose,
                                                   self.other_pose)
        self.keyframe_to_other = np.linalg.inv(self.other_to_keyframe)

        # Dummy for testing ... just sample one from gradients.
        index = self.keyframe_strong_gradients[len(
            self.keyframe_strong_gradients) // 2]
        px = img_hlp.index_to_pixel(
            img_hlp.image_size(self.keyframe_image), index)

        self._visualize_epi(px)

    def _visualize_epi(self, px: tuple) -> None:
        keyframe = img_hlp.gray_to_bgr(self.keyframe_image)
        other = img_hlp.gray_to_bgr(self.other_image)

        # Visualize the other camera's position in the keyframe image.
        other_in_key = mat_hlp.homogeneous(
            self.other_to_keyframe, np.array([0, 0, 0]))
        if other_in_key[2] > 0.0:
            other_in_key_px = mat_hlp.project_image(
                self.keyframe_K, other_in_key).astype(int)
            cv.circle(keyframe, other_in_key_px, 2, (0, 255, 0), cv.FILLED)
        else:
            logger.warning(
                f'Cannot visualize other camera, as it not is infront of camera')

        cv.setWindowTitle('keyframe', f'keyframe={self.keyframe_id}')
        cv.setWindowTitle('other', f'other frame={self.other_id}')
        cv.imshow('keyframe', keyframe)
        cv.imshow('other', other)
