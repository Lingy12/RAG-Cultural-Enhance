import logging
import sys

def get_logger(name):
    # Create a logger object
    logger = logging.getLogger(name)
    logger.propagate = False
    # Set the logging level
    logger.setLevel(logging.DEBUG)

    # Create a handler for outputting log messages to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # Create a formatter for the log messages
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


