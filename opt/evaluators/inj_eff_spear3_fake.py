# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import datetime as dt

async def get_inj_eff(connection):
    await asyncio.sleep(0.5)
    return 6 * np.random.rand()

async def get_objective(x, connection):
    assert len(x) == 10, 'Input dimension must be equal to 10'
    
    await asyncio.sleep(0.5)
    
    # calculate objective
    inj_eff = await get_inj_eff(connection)
    obj = np.array([-inj_eff])
    
    return obj

def evaluator_generator(origin, connection):
    async def evaluate(X, hook=None):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'

        # denormalize the parameters
        vrange = np.zeros((X.shape[0], 2))
        vrange[:, 0] = -20
        vrange[:, 1] = 20
        lower_limit = vrange[:, 0].reshape(-1, 1)
        upper_limit = vrange[:, 1].reshape(-1, 1)
        P = X * (upper_limit - lower_limit) + lower_limit

        # convert to the real PV values
        d_chrod_sa = np.array([
            [0.0591, 0.0608, 0.0608, 0.0608, 0.0304, -0.0065, -0.0155, -0.0155, -0.0155, -0.0078],
            [-0.0388, -0.0714, -0.0714, -0.0714, -0.0357, 0.0445, 0.0863, 0.0863, 0.0863, 0.0431]
        ])
        mat_trans = np.linalg.svd(d_chrod_sa)[2][:, 2:].transpose()
        PV = origin + P.dot(mat_trans)
        
        Y = np.zeros((PV.shape[0], 1))
        for i, pv in enumerate(PV):
            obj = await get_objective(pv, connection)
            finished_time = dt.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            Y[i] = obj
            if hook:
                hook(pv, obj, finished_time)

        return Y
    
    return evaluate
