# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import pandas as pd
import time
from ..utils.helpers import make_awaitable

def evaluator_generator(vrange=(0, 1), noise_level=0):
    @make_awaitable
    def evaluate(X, hook=None):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'
        
        start_time = time.time()

        # denormalize the parameters
        X = vrange[0] + (vrange[1] - vrange[0]) * X

        # Rosenbrock
        Y = np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + \
                   (1 - X[:, :-1]) ** 2, axis=1).reshape(-1, 1).astype('float64')
        # add noise
        Y += noise_level * np.random.randn(*Y.shape)

        end_time = time.time()
        
        if hook:
            hook(X, Y, end_time, start_time)

        return Y
    
    return evaluate
