import cv2 as cv
import logging
import os

import dapper.math.matrix as mat
import dapper.util.dataread as dr

logger = logging.getLogger(__name__)


class GroundTruthIterator:
    """
    Iterator of directory with the following content:

    File 'calib.txt', with one or several 3x4 calibration matrices.
    File 'poses.txt', with 3x4 pose matrices.
    Directory 'image_l', with images named such that they can be sorted.

    Check is_ok before use.
    """

    def __init__(self, data_dir: str) -> None:
        """
        Construct the iterator, given the path to the data directory.

        Parameters:
            data_dir: Path to data directory.
        """
        logger.debug('Construct GroundTruthIterator object')

        calib_matrices = dr.read_3x4_matrices(
            os.path.join(data_dir, 'calib.txt'))

        self.K = mat.extract_3x3(calib_matrices[0])
        logger.debug(f'Intrinsic matrix K=\n{self.K}')

        self.pose_matrices = dr.read_3x4_matrices(
            os.path.join(data_dir, 'poses.txt'))
        num_pose_matrices = len(self.pose_matrices)
        logger.debug(f'Number of pose matrices={num_pose_matrices}')

        image_dir = os.path.join(data_dir, 'image_l')
        self.image_paths = [os.path.join(image_dir, file)
                            for file in sorted(os.listdir(image_dir))]

        num_images = len(self.image_paths)
        logger.debug(f'Number of images={num_images}')

        self.is_ok = num_pose_matrices == num_images
        if not self.is_ok:
            logger.warning(f'Number of images and poses does not match')

        self.num_data = num_images
        self.current_data = 0

    def __iter__(self):
        """
        Reset the iteration index.
        """
        self.current_data = 0

        return self

    def __next__(self):
        """
        Get the next item in the dataset.

        Returns:
            Tuple (grayscale image, 3x3 K matrix, 4x4 pose matrix).
        """
        if self.current_data < self.num_data:
            image = cv.imread(
                self.image_paths[self.current_data], cv.IMREAD_GRAYSCALE)
            pose = mat.add_row(self.pose_matrices[self.current_data])

            self.current_data += 1

            return image, self.K, pose
        else:
            logger.debug('Reached last data item')
            raise StopIteration
