# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import asyncio
import datetime as dt

from ..utils.spear3_api import within_spear3, _write_sext_value, \
    _turn_off_injection, _turn_on_injection, _get_injection_state, _get_inj_eff
from ..utils import spear3_da_opt as da_opt

async def _get_objective(x, connection):
    assert len(x) == 10, 'Input dimension must be equal to 10'
    
    inj_state = _get_injection_state(connection)
    
    # turn off injection if it's on
    if inj_state:
        _turn_off_injection(connection)
    
    # set sextupole PV values
    df = da_opt._set_origin(x.reshape(1, -1), connection)
    await asyncio.sleep(0.5)
    
    # calculate objective
    inj_eff = await _get_inj_eff(connection)
    obj = np.array([-inj_eff])
    
    # recover injection state if needed
    if inj_state:
        _turn_on_injection(connection)
    
    return obj

@within_spear3
async def get_objective(x, connection):
    return await _get_objective(x, connection)

def evaluator_generator(origin, connection):
    async def evaluate(X, hook):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'

        PV = da_opt.x_to_pv(X, origin)
        
        Y = np.zeros((PV.shape[0], 1))
        for i, pv in enumerate(PV):
            obj = await get_objective(pv, connection)
            finished_time = dt.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
            Y[i] = obj
            if hook:
                hook(pv, obj, finished_time)

        return Y
    
    return evaluate
