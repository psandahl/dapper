import logging
import numpy as np

logger = logging.getLogger(__name__)


def read_3x4_matrices(filename: str) -> list():
    """
    Read a text file, where each line represents a 3x4 matrix.

    Parameters:
        filename: The name of the file.

    Returns:
        A list of matrices.
    """
    matrices = list()
    with open(filename, 'r') as f:
        lineno = 0
        for line in f.readlines():
            lineno += 1
            array = np.fromstring(line, dtype=np.float64, sep=' ')
            if len(array) == 12:
                matrices.append(array.reshape(3, 4))
            else:
                logger.warning(
                    f'Line did not contain 12 values in {filename}:{lineno}')

    return matrices
