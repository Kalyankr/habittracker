import logging


def init_logger(level=logging.INFO):
    """
    Initialize root logger with a consistent format.
    Call this once at the start of your script.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger()
