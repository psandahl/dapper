import argparse
import logging
import sys

from dapper.app.stereodevapp import StereoDevApp

logger = None
handler = None


def setup_logging() -> None:
    """
    Global setup of logging system. Module loggers then register
    as getLogger(__name__) to end up in logger tree.
    """
    global logger
    logger = logging.getLogger('dapper')
    logger.setLevel(logging.DEBUG)

    global handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        '[%(levelname)s %(name)s:%(lineno)d] %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def main() -> None:
    """
    Entry point for the dapper execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log',
                        help='set the effective log level (DEBUG, INFO, WARNING or ERROR)')
    parser.add_argument('-s', '--stereo', action='store_true',
                        help='run the stereo development application')
    parser.add_argument('-d', '--dataset',
                        help='dataset for the application')
    args = parser.parse_args()

    # Check if the effective log level shall be altered.
    if not args.log is None:
        log_level = args.log.upper()
        if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            num_log_level = getattr(logging, log_level)
            handler.setLevel(num_log_level)
        else:
            parser.print_help()
            sys.exit(1)

    # Check for which application to run.
    if args.stereo and not args.dataset is None:
        app = StereoDevApp()
        if not app.run(args.dataset):
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    # Successful exit.
    sys.exit(0)


if __name__ == '__main__':
    setup_logging()

    try:
        main()
    except Exception as e:
        logger.exception(f"Global exception handler caught: '{e}'")
        sys.exit(1)
