import cv2 as cv
import logging
import numpy as np

import dapper.image.helpers as hlp
import dapper.math.matrix as mat
from dapper.stereo.epimatcher import EpiMatcher
from dapper.util.groundtruthiterator import GroundTruthIterator

logger = logging.getLogger(__name__)


class StereoDevApp():
    """
    An application for development of depth algorithms in isolation, 
    using calibrated datasets with ground truth poses.
    """

    def __init__(self) -> None:
        logger.debug('Construct StereoDevApp object')

        self.frame_id = 0
        self.epi_matcher = EpiMatcher(visualize=True)

        self.keyframe_images = list()
        self.keyframe_poses = list()
        self.keyframe_strong_gradients = list()

        self.current_image = None
        self.current_pose = np.eye(4, 4, dtype=np.float64)

    def run(self, data_dir: str) -> bool:
        logger.info(f"Start StereoDevApp with data_dir='{data_dir}'")

        self.frame_id = 0
        self.keyframe_images = list()
        self.keyframe_poses = list()

        self.current_image = None
        self.current_pose = np.eye(4, 4, dtype=np.float64)
        self.current_K = np.eye(3, 3, dtype=np.float64)

        # Iterate through all data in the dataset.
        dataset = GroundTruthIterator(data_dir)
        if not dataset.is_ok:
            logger.error('Failed to initialize the dataset')
            return False

        for image, K, pose in dataset:
            assert hlp.is_image(image)
            assert hlp.image_channels(image) == 1

            self.current_image = image
            self.current_pose = pose
            self.current_K = K  # Assume never change.

            if not self.epi_matcher.keyframe_visual_image is None:
                cv.setWindowTitle(
                    'keyframe', f'keyframe={self.epi_matcher.keyframe_id}')
                cv.imshow('keyframe', self.epi_matcher.keyframe_visual_image)

            if not self.epi_matcher.other_visual_image is None:
                cv.setWindowTitle(
                    'frame', f'frame={self.epi_matcher.other_id}')
                cv.imshow('frame', self.epi_matcher.other_visual_image)

            print(f'Current frame id={self.frame_id}')
            self._print_relative_latest_keyframe()
            print('Commands:')

            print(' ESQ => quit')
            print(" 'k' => use this frame as new keyframe")
            print(' ANY => compute algorithms and step')

            key = cv.waitKey(0)
            if key == 27:
                break
            elif key == ord('k') or not self.keyframe_poses:
                self._new_keyframe()
            else:
                self._new_frame()

            self.frame_id += 1

        cv.destroyAllWindows()

        return True

    def _new_keyframe(self):
        logger.info(f'Add frame id={self.frame_id} as new keyframe')

        # Store information about the keyframe.
        self.keyframe_images.append(self.current_image)
        self.keyframe_poses.append(self.current_pose)

        # Add the keyframe to the epi matcher.
        self.epi_matcher.set_keyframe(self.frame_id, self.current_image,
                                      self.current_pose, self.current_K,
                                      None)

    def _new_frame(self):
        logger.info(f'Processing frame id={self.frame_id}')

        # Match the frame with latest keyframe.
        self.epi_matcher.match(
            self.frame_id, self.current_image, self.current_pose, self.current_K)

    def _print_relative_latest_keyframe(self):
        if self.keyframe_poses:
            relative_pose = mat.relative_pose(
                self.keyframe_poses[-1], self.current_pose)

            ypr, xyz = mat.decompose_pose(relative_pose)
            yaw, pitch, roll = ypr
            x, y, z = xyz
            print(
                f'Relative pose yaw={yaw:.2f} pitch={pitch:.2f} roll={roll:.2f} x={x:.2f} y={y:.2f} z={z:2f}')
        else:
            print('No previous keyframe')
