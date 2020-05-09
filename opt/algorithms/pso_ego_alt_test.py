# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import pygmo as pg

from . import de
from . import pso_pygmo
from ..utils.operators import initialize, combine_populations, \
    gaussian_process_generator, \
    upper_confidence_bound, expected_improvement
from ..utils.helpers import make_awaitable
from operator import itemgetter

async def get_best_solutions(task=None):
    assert task, 'No task specified'
    
    algorithm, problems = \
        itemgetter('algorithm', 'problems')(task)
    name, params, hooks = \
        itemgetter('name', 'params', 'hooks')(algorithm)
    assert name in ['de', 'pso'], \
        'Specified algorithm is not supported'
    
    X = []
    if name == 'de':
        # build the archipelago
        archi = pg.archipelago()
        for prob in problems:
            isl = de.get_island(prob, params, hooks)
            archi.push_back(isl)
            
        archi.evolve()
        await asyncio.sleep(0)
        archi.wait()
        await asyncio.sleep(0)
        X = archi.get_champions_x()
        Y = archi.get_champions_f()
        
        hook_report = itemgetter('report')(hooks)
        if hook_report is not None:
            for i in range(len(Y)):
                gbest = (X[i], Y[i])
                hook_report(gbest)
    elif name == 'pso':
        # build the archipelago
        archi = pg.archipelago()
        for prob in problems:
            isl = pso_origin.get_island(prob, params, hooks)
            archi.push_back(isl)
            
        archi.evolve()
        await asyncio.sleep(0)
        archi.wait()
        await asyncio.sleep(0)
        X = archi.get_champions_x()
        Y = archi.get_champions_f()
        
        hook_report = itemgetter('report')(hooks)
        if hook_report is not None:
            for i in range(len(Y)):
                gbest = (X[i], Y[i])
                hook_report(gbest)
    X = np.array(X)
    
    return X

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
    Y0 = await evaluate(X0)
    X_archive0, Y_archive0 = archive0
    X_archive1 = combine_populations((X_archive0, X0))
    Y_archive1 = combine_populations((Y_archive0, Y0))
    archive1 = (X_archive1, Y_archive1)
    
    # build gp model from the new archive
    Y_mean_archive1 = np.mean(Y_archive1, axis=0)
    _Y_archive1 = Y_archive1 - Y_mean_archive1
    gp = gaussian_process_generator(X_archive1, _Y_archive1, ARD=False)
    gp['.*lengthscale'].constrain_bounded(0.1, 10)
#     gp.Gaussian_noise.variance.constrain_bounded(0.1, 1.0)
#     gp.Gaussian_noise.variance.constrain_fixed(0)
#     gp.rbf.lengthscale.constrain_bounded(0.1, 10)
#     gp.sum.rbf.lengthscale.constrain_bounded(0.1, 10)
    gp.optimize_restarts(num_restarts=20, verbose=False, parallel=True)
#     gp.optimize()
    
    if hook_inspect:
        hook_inspect(gp, Y_mean_archive1)
    
    problems = []
    if a_name == 'ucb':
        kappa = itemgetter('kappa')(a_params)
        kappa_list = np.linspace(kappa, 0, Nd)
        
        for _kappa in kappa_list:
            def gp_evaluate(X, hook=None, _kappa=_kappa):
                return upper_confidence_bound(X, gp, _kappa)
            problems.append(gp_evaluate)
    elif a_name == 'ei':
        xi = itemgetter('xi')(a_params)
        xi_list = np.linspace(0, xi, Nd)
#         Y_archive1_mean, _ = gp.predict(X_archive1)
#         Y_archive1_opt = np.min(Y_archive1_mean, axis=0)
        Y_archive1_opt = np.min(Y_archive1, axis=0)
        
        for _xi in xi_list:
            def gp_evaluate(X, hook=None, _xi=_xi):
                return -expected_improvement(X, gp, Y_archive1_opt - Y_mean_archive1, _xi)
            problems.append(gp_evaluate)
    
    # seek for the best Nd solutions in some sense
    # we can start from the known best solution set, say, the archive
    # but for now we just make a clean start
    task = {
        'algorithm': algorithm,
        'problems': problems
    }
    X1 = await get_best_solutions(task)
    
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
