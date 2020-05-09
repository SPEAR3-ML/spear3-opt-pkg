# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import asyncio
import datetime as dt

from ..utils.spear3_api import within_spear3, _get_lifetime
from ..utils.spear3_lt_opt import _set_origin, x_to_pv

async def _get_objective(x, connection):
    assert len(x) == 13, 'Input dimension must be equal to 13'
    
    # set quadrupole PV values
    df = _set_origin(x.reshape(1, -1), connection)
    await asyncio.sleep(2)
    
    # calculate objective
    lifetime = await _get_lifetime(connection)
    obj = np.array([-lifetime])
    
    return obj

@within_spear3
async def get_objective(x, connection):
    return await _get_objective(x, connection)

def evaluator_generator(connection):
    async def evaluate(X, hook):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'

        PV = x_to_pv(X)
        
        Y = np.zeros((PV.shape[0], 1))
        for i, pv in enumerate(PV):
            obj = await get_objective(pv, connection)
            finished_time = dt.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            Y[i] = obj
            if hook:
                hook(pv, obj, finished_time)

        return Y
    
    return evaluate
