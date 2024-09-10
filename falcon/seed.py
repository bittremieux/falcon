import random

import numpy as np


def set_seeds(my_seed=42):
    random.seed(my_seed)
    np.random.seed(my_seed)
