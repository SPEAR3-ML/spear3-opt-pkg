# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import six
from smt.applications import EGO

from ..utils.operators import initialize
from ..utils.helpers import make_sync
from operator import itemgetter

async def optimize(evaluate, params):
    # config
    # D = 8  decision space dimension
    # N0 = 30  initial population
    # Ng = 200  total generation
    # method = 'lhs'  initialization method
    evaluate = make_sync(evaluate)
    
    # initialize
    D, N0, method, vrange, add_center, criterion, T = \
        itemgetter('D', 'N0', 'method', 'vrange', 'add_center', 'criterion', 'T')(params)
    xdoe = initialize((N0, D), method, vrange, add_center)
    
    xlimits = np.ones((D, 2))
    xlimits[:, 0] = 0
    
    n_iter = T - N0 + 1
    ego = EGO(n_iter=n_iter, criterion=criterion, xdoe=xdoe, xlimits=xlimits)

    # run
    x_opt, y_opt, ind_best, x_data, y_data, x_doe, y_doe = ego.optimize(
        fun=evaluate
    )
    
    return x_opt, y_opt
