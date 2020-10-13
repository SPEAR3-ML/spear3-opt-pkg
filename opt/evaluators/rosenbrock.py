# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import pandas as pd
import time
from ..utils.helpers import make_async

from operator import itemgetter

async def evaluate(X, configs={
        'vrange': [0, 1],
        'wall_time': 1,
        'noise_level': 0
    }):
    assert type(X) == np.ndarray, 'Input X must be a numpy array'

    vrange, wall_time, noise_level = itemgetter('vrange', 'wall_time', 'noise_level')(configs)
    
    await asyncio.sleep(wall_time)
    # denormalize the parameters
    X1 = vrange[0] + (vrange[1] - vrange[0]) * X

    # Rosenbrock
    Y = np.sum(100 * (X1[:, 1:] - X1[:, :-1] ** 2) ** 2 + \
               (1 - X1[:, :-1]) ** 2, axis=1).reshape(-1, 1).astype('float64')

    # add noise
    Y += noise_level * np.random.randn(*Y.shape)
    
    return Y
