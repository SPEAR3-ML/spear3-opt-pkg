# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np

from ..utils.operators import initialize, \
    particle_swarm_init, particle_swarm_iter, \
    pbest_update, gbest_update, combine_populations, \
    gaussian_process_generator, \
    offset, crossover, non_dominated_sorting, select, \
    upper_confidence_bound, expected_improvement, surrogate_based_optimization
from ..utils.helpers import make_awaitable
from operator import itemgetter

async def init(evaluate, params, hooks):
    # Note: evaluate must be awaitable!
    
    D, N0, method = itemgetter('D', 'N0', 'method')(params)
    hook_init, hook_eval = itemgetter('init', 'evaluate')(hooks)
    
    i0 = 0
    X0 = initialize((N0, D), method)
    Y0 = await evaluate(X0, hook_eval)
    V0, pbest0, gbest0 = particle_swarm_init(X0, Y0)
    
    if hook_init:
        hook_init(i0, pbest0[1], Y0)
    
    return i0, X0, V0, Y0, pbest0, gbest0

async def loop(evaluate, i0, X0, Y0, V0, pbest0, gbest0, params, hooks):
    # Note: evaluate must be awaitable!
    
    Ng, acquisition, A, No, Nc = itemgetter('Ng', 'acquisition', 'A', 'No', 'Nc')(params)
    a_name, a_params = itemgetter('name', 'params')(acquisition)
    hook_loop, hook_eval = itemgetter('loop', 'evaluate')(hooks)
    
    # increase generation index
    i1 = i0 + 1
    
    X_pbest0, Y_pbest0 = pbest0

    _X0 = combine_populations((X0, X_pbest0))
    _Y0 = combine_populations((Y0, Y_pbest0))

    gp = gaussian_process_generator(_X0, _Y0)
    gp.optimize()

    _X1, _V1 = particle_swarm_iter(X0, V0, pbest0, gbest0)
    _X2, _V2 = offset(_X1, _V1, amplitude=A, factor=No)
    _X3, _V3 = crossover(_X1, _V1, factor=Nc)

    _X4 = combine_populations((_X1, _X2, _X3))
    _V4 = combine_populations((_V1, _V2, _V3))
    
    assert a_name in ['ucb', 'sbo', 'ei'], 'Specified acquisition function is not supported'
    if a_name == 'ucb':
        kappa = itemgetter('kappa')(a_params)
        _Y4 = upper_confidence_bound(_X4, gp, kappa)
    elif a_name == 'sbo':
        _Y4 = surrogate_based_optimization(_X4, gp)
    elif a_name == 'ei':
        xi = itemgetter('xi')(a_params)
        _Y0_mean, _ = gp.predict(_X0)
        _Y0_opt = np.min(_Y0_mean, axis=0)
        _Y4 = -expected_improvement(_X4, gp, _Y0_opt, xi)
    _I4 = non_dominated_sorting(_Y4)

    X1, V1 = select(_I4, [_X4, _V4], n=X0.shape[0])
    Y1 = await evaluate(X1, hook_eval)

    pbest1 = pbest_update(pbest0, X1, Y1)
    gbest1 = gbest_update(pbest1)
    
    if hook_loop:
        hook_loop(i1, pbest1[1], Y1)
    
    # termination condition
    end = i1 >= (Ng - 1)

    return i1, X1, Y1, V1, pbest1, gbest1, end

async def optimize(evaluate, params, hooks):
    # config
    # D = 8  decision space dimension
    # N0 = 100  initial population
    # Ng = 100  total generation
    # method = 'lhs'  initialization method
    evaluate = make_awaitable(evaluate)
    
    # initialize
    i0, X0, V0, Y0, pbest0, gbest0 = await init(evaluate, params, hooks)
    
    # run
    i, X, Y, V, pbest, gbest, end = i0, X0, Y0, V0, pbest0, gbest0, False
    while not end:
        i, X, Y, V, pbest, gbest, end = \
            await loop(evaluate, i, X, Y, V, pbest, gbest, params, hooks)
    
    return pbest
