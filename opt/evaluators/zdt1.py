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
        'wall_time': 1
    }):
    assert type(X) == np.ndarray, 'Input X must be a numpy array'
    
    vrange, wall_time = itemgetter('vrange', 'wall_time')(configs)

    # start_time = time.time()
    await asyncio.sleep(wall_time)
    # denormalize the parameters
    X = vrange[0] + (vrange[1] - vrange[0]) * X

    # ZDT1
    G = 1 + 9 * np.mean(X[:, 1:], axis=1)
    F1 = X[:, 0]
    F2 = G * (1 - np.sqrt(F1 / G))
    Y = np.vstack((F1, F2)).transpose()
    # end_time = time.time()
    
    return Y
