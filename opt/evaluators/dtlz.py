# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import time
import pygmo as pg

from operator import itemgetter

async def evaluate(X, configs={
        'prob_id': 1,
        'D': 30,
        'O': 2,
        'wall_time': 1
    }):
    assert type(X) == np.ndarray, 'Input X must be a numpy array'

    prob_id, D, O, wall_time = itemgetter('prob_id', 'D', 'O', 'wall_time')(configs)
    
    prob = pg.problem(pg.dtlz(prob_id=prob_id, dim=D, fdim=O))
    lower_bound, upper_bound = prob.get_bounds()
    # denormalize the parameters
    X = lower_bound + (upper_bound - lower_bound) * X
    
    Y = []
    start_time = time.time()
    await asyncio.sleep(wall_time)
    for x in X:
        y = prob.fitness(x)
        Y.append(y)
    end_time = time.time()
    Y = np.array(Y)
    print(f'time cost: {(end_time - start_time) * 1e3:.3f} ms')
    
    return Y
