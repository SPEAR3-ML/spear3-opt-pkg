# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import time
from pymoo.factory import get_problem

def evaluator_generator(prob_id='zdt1', vrange=(0, 1), wall_time=1):
    async def evaluate(X):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'
        
        start_time = time.time()
        await asyncio.sleep(wall_time)
        # denormalize the parameters
        X = vrange[0] + (vrange[1] - vrange[0]) * X

        prob = get_problem(prob_id)
        Y = prob.evaluate(X)
        end_time = time.time()
        
        return Y
    
    return evaluate
