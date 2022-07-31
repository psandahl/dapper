import cv2 as cv
import logging

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

    def run(self, data_dir: str) -> bool:
        logger.info(f"Start StereoDevApp with data_dir='{data_dir}'")

        self.frame_id = 0

        dataset = GroundTruthIterator(data_dir)
        if not dataset.is_ok:
            logger.error('Failed to initialize the dataset')
            return False

        for image, K, pose in dataset:

            cv.imshow('frame image', image)
            key = cv.waitKey(0)
            if key == 27:
                break

            self.frame_id += 1

        return True
