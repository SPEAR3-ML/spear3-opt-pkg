# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np

from ..utils.operators import initialize, \
    particle_swarm_init, particle_swarm_iter, \
    pbest_update, gbest_update
from ..utils.helpers import make_async
from operator import itemgetter

async def init(evaluate, configs):
    # Note: evaluate must be awaitable!
    
    D, N0, method = itemgetter('D', 'N0', 'method')(configs)
    
    i0 = 0
    X0 = initialize((N0, D), method)
    Y0 = await evaluate(X0)
    V0, pbest0, gbest0 = particle_swarm_init(X0, Y0)
    
    return i0, X0, V0, Y0, pbest0, gbest0

async def loop(evaluate, i0, X0, Y0, V0, pbest0, gbest0, configs):
    # Note: evaluate must be awaitable!
    
    Ng = itemgetter('Ng')(configs)
    
    # increase generation index
    i1 = i0 + 1
    
    X_pbest0, Y_pbest0 = pbest0

    X1, V1 = particle_swarm_iter(X0, V0, pbest0, gbest0)

    Y1 = await evaluate(X1)

    pbest1 = pbest_update(pbest0, X1, Y1)
    gbest1 = gbest_update(pbest1)
    
    # termination condition
    end = i1 >= (Ng - 1)

    return i1, X1, Y1, V1, pbest1, gbest1, end

async def optimize(evaluate, configs):
    # config
    # D = 8  decision space dimension
    # N0 = 100  initial population
    # Ng = 100  total generation
    # method = 'lhs'  initialization method
    evaluate = make_async(evaluate)
    
    # initialize
    i0, X0, V0, Y0, pbest0, gbest0 = await init(evaluate, configs)
    
    # run
    i, X, Y, V, pbest, gbest, end = i0, X0, Y0, V0, pbest0, gbest0, False
    while not end:
        i, X, Y, V, pbest, gbest, end = await loop(evaluate, i, X, Y, V, pbest, gbest, configs)
    
    return pbest
