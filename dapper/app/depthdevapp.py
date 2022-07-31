import logging

logger = logging.getLogger(__name__)


class DepthDevApp():
    """
    An application for development of depth algorithms in isolation, 
    using calibrated datasets with ground truth poses.
    """

    def __init__(self) -> None:
        logger.debug('Construct DepthDevApp object')

    def run(self, data_dir: str) -> bool:
        logger.debug(f'Start DepthDevApp with data_dir={data_dir}')
        return True
