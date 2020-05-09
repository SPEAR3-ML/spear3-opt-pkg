# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import asyncio
import datetime as dt

from ..utils.spear3_api import within_spear3, _get_loss_rate
from ..utils.spear3_lt_opt import _get_origin, _set_origin, x_to_pv

async def _get_objective(x, connection, duration=1, max_step=None):
    assert len(x) == 13, 'Input dimension must be equal to 13'
    
    if max_step is None:
        x1 = x.reshape(1, -1)
        df = _set_origin(x1, connection)
        await asyncio.sleep(0.5)
    else:
        x0 = _get_origin(connection)
        x1 = x.reshape(1, -1)
        dx_max = np.max(np.abs(x1 - x0))
        step_num = np.ceil(dx_max / max_step)

        steps = np.linspace(x0, x1, step_num + 1)[1:]
        for step in steps:
            # set quadrupole PV values
            df = _set_origin(step, connection)
            await asyncio.sleep(0.5)
    
    # calculate objective
    lossrate = await _get_loss_rate(connection, duration)
    obj = np.array([-lossrate])
    
    return obj

@within_spear3
async def get_objective(x, connection, duration=1, max_step=None):
    return await _get_objective(x, connection, duration, max_step)

def evaluator_generator(connection, vrange=(-20, 20), duration=1, max_step=None, fixed_values=[]):
    async def evaluate(X, hook):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'
        
        X1 = X
        if X.shape[1] < 13:
            Xf = np.tile(np.array(fixed_values), (X.shape[0], 1))
            X1 = np.hstack((X, Xf))
            assert X1.shape[1] == 13, f'Length of fixed values must be {13 - X.shape[1]}'

        PV = x_to_pv(X1, vrange[0], vrange[1])
        
        Y = np.zeros((PV.shape[0], 1))
        for i, pv in enumerate(PV):
            obj = await get_objective(pv, connection, duration, max_step)
            finished_time = dt.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            Y[i] = obj
            if hook:
                hook(pv, obj, finished_time)

        return Y
    
    return evaluate
