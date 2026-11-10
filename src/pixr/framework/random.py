import random

import numpy as np

import torch

from chugr.common import rank_seed


def random_seed(seed=42, rank=0):
    # ranks spaced by a large stride (not seed + rank), so runs w/ nearby seeds don't share random streams
    seed = rank_seed(seed, rank)
    torch.manual_seed(seed)
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)
