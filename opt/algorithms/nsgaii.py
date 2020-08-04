# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import autograd.numpy as anp
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize

from operator import itemgetter

class Evaluator(Problem):
    def __init__(self, evaluate, D=30):
        super().__init__(n_var=D,
                         n_obj=2,
                         n_constr=0,
                         xl=anp.array([0] * D),
                         xu=anp.array([1] * D))
        self.evaluate_origin = evaluate

    def _evaluate(self, x, out, *args, **kwargs):
        Y = self.evaluate_origin(x)
        out["F"] = Y
#         out["F"] = anp.column_stack([f1, f2])

async def optimize(evaluate, configs):
    # config
    # D = 8  decision space dimension
    # N0 = 100  initial population
    # Ng = 100  total generation
    # seed = None  random seed
    D, Ng, N0, seed = itemgetter('D', 'Ng', 'N0', 'seed')(configs)
    
    problem = Evaluator(evaluate, D)
    algorithm = NSGA2(
        pop_size=N0,
        n_offsprings=N0,
        sampling=get_sampling('real_random'),
        crossover=get_crossover('real_sbx', prob=0.9, eta=15),
        mutation=get_mutation('real_pm', eta=20),
        eliminate_duplicates=True
    )
    termination = get_termination('n_gen', Ng)
    await asyncio.sleep(0)
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=True,
                   verbose=False)
    gbest = (res.X, res.F)
    
    return gbest
