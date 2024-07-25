import logging
import logging.config
import json
import os


def setup_logging():
    with open(os.path.join(os.path.dirname(__file__), "logging.json")) as f:
        config = json.load(f)
    logging.config.dictConfig(config)


def get_logger(name):
    return logging.getLogger(name)


# Setup at import time to ensure all child processes have the same config
setup_logging()
