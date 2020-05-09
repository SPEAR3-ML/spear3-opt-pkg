# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import pygmo as pg
# from numba import jit

from operator import itemgetter

class evaluate_wrapper:
    def __init__(self, D, evaluate, name='X'):
        self.D = D
        self.evaluate = evaluate
        self.name = name
#     @jit
    def fitness(self, x):
        return [self.evaluate(x.reshape(1, -1))[0, 0]]

    def get_bounds(self):
        return (np.zeros(self.D), np.ones(self.D))

    def get_name(self):
        return self.name

    def get_extra_info(self):
        return f'\tDimensions: {self.D}'

async def optimize(evaluate, params, hooks):
    # config
    # D = 8  decision space dimension
    # N0 = 100  initial population
    # Ng = 100  total generation
    D, Ng, N0 = itemgetter('D', 'Ng', 'N0')(params)
    hook_report = itemgetter('report')(hooks)
    
    algo = pg.algorithm(pg.pso(gen=Ng))
    algo.set_verbosity(int(Ng / 10))
    prob = pg.problem(evaluate_wrapper(D, evaluate))
    pop = pg.population(prob, N0)
    
    await asyncio.sleep(0)
    pop = algo.evolve(pop)
    gbest = (pop.champion_x, pop.champion_f)
    
    if hook_report:
        hook_report(gbest)
    
    return gbest

def get_island(evaluate, params, hooks):
    # config
    # D = 8  decision space dimension
    # N0 = 100  initial population
    # Ng = 100  total generation
    D, Ng, N0 = itemgetter('D', 'Ng', 'N0')(params)
    
    algo = pg.algorithm(pg.pso(gen=Ng))
    algo.set_verbosity(int(Ng / 10))
    prob = pg.problem(evaluate_wrapper(D, evaluate))
    island = pg.island(algo=algo, prob=prob, size=N0, udi=pg.mp_island())
    
    return island
