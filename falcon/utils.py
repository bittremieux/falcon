import logging
import random
import sys

import numpy as np


def set_seeds(my_seed=42):
    random.seed(my_seed)
    np.random.seed(my_seed)


def configure_logger():
    logger = logging.getLogger("falcon")
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
                "{message}",
                style="{",
            )
        )
        root.addHandler(handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    return logger
