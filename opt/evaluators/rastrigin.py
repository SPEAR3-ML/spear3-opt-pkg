# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import pandas as pd
import time
from ..utils.helpers import make_async

def evaluator_generator(vrange=(-5.12, 5.12)):
    def evaluate(X, hook=None):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'
        
        start_time = time.time()

        # denormalize the parameters
        X = vrange[0] + (vrange[1] - vrange[0]) * X

        # Rastrigin
        Y = 10 * X.shape[1] + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X),
                                     axis=1).reshape(-1, 1)
        end_time = time.time()
        
        if hook:
            hook(X, Y, end_time, start_time)

        return Y
    
    return evaluate
