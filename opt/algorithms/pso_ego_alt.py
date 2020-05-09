# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np

from . import pso
from . import pso_ego
from ..utils.operators import initialize, combine_populations, \
    gaussian_process_generator, \
    non_dominated_sorting, select, \
    upper_confidence_bound, expected_improvement
from ..utils.helpers import make_awaitable
from operator import itemgetter

async def get_best_solutions(N, task=None):
    assert task, 'No task specified'
    
    algorithm, evaluate = \
        itemgetter('algorithm', 'evaluate')(task)
    name, params, hooks = \
        itemgetter('name', 'params', 'hooks')(algorithm)
    assert name in ['pso', 'psoego'], \
        'Specified algorithm is not supported'
    
    pbest = None
    if name == 'pso':
        pbest = await pso.optimize(evaluate, params, hooks)
    elif name == 'psoego':
        pbest = await pso_ego.optimize(evaluate, params, hooks)
    
    # select
    I = non_dominated_sorting(pbest[1])
    X1, = select(I, [pbest[0]], N)
    
    return X1

async def init(evaluate, params, hooks):
    # Note: evaluate must be awaitable!
    
    D, N0, method, vrange, add_center = \
        itemgetter('D', 'N0', 'method', 'vrange', 'add_center')(params)
    
    # initialize
    X0 = initialize((N0, D), method, vrange, add_center)
    archive0 = [np.zeros((0, D)), np.zeros((0, 1))]
    
    return X0, archive0

async def loop(evaluate, X0, archive0, params, hooks):
    # Note: evaluate must be awaitable!
    
    Nd, T, acquisition, algorithm = \
        itemgetter('Nd', 'T', 'acquisition', 'sub_algo')(params)
    a_name, a_params = itemgetter('name', 'params')(acquisition)
    hook_eval = itemgetter('evaluate')(hooks)
    
    # form the next archive
    Y0 = await evaluate(X0, hook_eval)
    X_archive0, Y_archive0 = archive0
    X_archive1 = combine_populations((X_archive0, X0))
    Y_archive1 = combine_populations((Y_archive0, Y0))
    archive1 = (X_archive1, Y_archive1)
    
    # build gp model from the new archive
    gp = gaussian_process_generator(X_archive1, Y_archive1)
    gp.optimize()
    
    if a_name == 'ucb':
        kappa = itemgetter('kappa')(a_params)
        
        def gp_evaluate(X, hook=None):
            return upper_confidence_bound(X, gp, kappa)
    elif a_name == 'ei':
        xi = itemgetter('xi')(a_params)
        Y_archive1_mean, _ = gp.predict(X_archive1)
        Y_archive1_opt = np.min(Y_archive1_mean, axis=0)
        
        def gp_evaluate(X, hook=None):
            return -expected_improvement(X, gp, Y_archive1_opt, xi)
    
    # seek for the best Nd solutions
    # we can start from the known best solution set, say, the archive
    # but for now we just make a clean start
    task = {
        'algorithm': algorithm,
        'evaluate': gp_evaluate
    }
    X1 = await get_best_solutions(Nd, task)
    
    # termination condition
    end = Y_archive1.shape[0] >= T
    
    return X1, archive1, end

async def optimize(evaluate, params, hooks):
    # config
    # D = 8  decision space dimension
    # N0 = 87  initial population
    # Nd = 5  selected point number per iteration
    # T = 300  total evaluation number
    # ALGO = 'psoego'  sub-algorithm    
    evaluate = make_awaitable(evaluate)
    
    # initialize
    X0, archive0 = await init(evaluate, params, hooks)
    
    # run
    X, archive, end = X0, archive0, False
    while not end:
        X, archive, end = await loop(evaluate, X, archive, params, hooks)
        
    return archive
