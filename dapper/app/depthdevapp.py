import logging

from dapper.util.groundtruthiterator import GroundTruthIterator

logger = logging.getLogger(__name__)


class DepthDevApp():
    """
    An application for development of depth algorithms in isolation, 
    using calibrated datasets with ground truth poses.
    """

    def __init__(self) -> None:
        logger.debug('Construct DepthDevApp object')

    def run(self, data_dir: str) -> bool:
        logger.info(f"Start DepthDevApp with data_dir='{data_dir}'")

        itr = GroundTruthIterator(data_dir)
        if not itr.is_ok:
            logger.error('Failed to initialize the data iterator')
            return False

        return True
