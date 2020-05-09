# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import pygmo as pg

import GPyOpt
import GPy
from ..utils.operators import initialize, combine_populations
from ..utils.helpers import make_awaitable
from operator import itemgetter

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
    hook_inspect = itemgetter('inspect')(hooks)
    
    # form the next archive
    Y0 = await evaluate(X0, hook_eval)
    X_archive0, Y_archive0 = archive0
    X_archive1 = combine_populations((X_archive0, X0))
    Y_archive1 = combine_populations((Y_archive0, Y0))
    archive1 = (X_archive1, Y_archive1)
    
    # using gpyopt to build gp model from the new archive
    X1 = []
    domain = [dict(name=f'x{i}', type='continuous', domain=(0, 1))
              for i in range(X0.shape[1])]
    if a_name == 'ucb':
        kappa = itemgetter('kappa')(a_params)
        kappa_list = np.linspace(0, kappa, Nd)
        
        for _kappa in kappa_list:
            kernel = GPy.kern.RBF(X0.shape[1])
            bo_step = GPyOpt.methods.BayesianOptimization(f=None, domain=domain,
                                                          kernel=kernel,
#                                                           acquisition_type='LCB',
                                                          X=X_archive1, Y=Y_archive1)
            X1.append(bo_step.suggest_next_locations())
    
    if hook_inspect:
        hook_inspect(bo_step)
    
    X1 = np.vstack(X1)
    
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
