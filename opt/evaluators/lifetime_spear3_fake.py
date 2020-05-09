# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import asyncio
import datetime as dt

from ..utils.spear3_lt_opt import x_to_pv

async def random(x):
    await asyncio.sleep(0.5)
    
    return 100 * np.random.rand()

async def rosenbrock(x):
    await asyncio.sleep(0.5)
    
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

async def _get_objective(connection, x, noise_level=0):
#     assert len(x) == 13, 'Input dimension must be equal to 13'
    
    y = (await rosenbrock(x)) + noise_level * np.random.randn()
    obj = np.array([y])
    
    return obj

async def get_objective(connection, x, noise_level=0):
    return await _get_objective(connection, x, noise_level)

def evaluator_generator(connection, vrange=(0, 1), noise_level=0):
    async def evaluate(X, hook):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'
        
        X1 = X
        if X.shape[1] < 13:
            Xf = np.tile(np.zeros(13 - X.shape[1]), (X.shape[0], 1))
            X1 = np.hstack((X, Xf))

        PV = x_to_pv(X1, vrange[0], vrange[1])
        
        Y = np.zeros((PV.shape[0], 1))
        for i, pv in enumerate(PV):
            obj = await get_objective(connection, pv[:X.shape[1]], noise_level)
            finished_time = dt.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            Y[i] = obj
            if hook:
                hook(pv, obj, finished_time)

        return Y
    
    return evaluate
