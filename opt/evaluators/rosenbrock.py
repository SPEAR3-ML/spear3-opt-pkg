# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import numpy as np
import pandas as pd
import datetime as dt

def evaluator_generator(vrange=(0, 1), wall_time=1):
    async def evaluate(X, hook=None):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'

        # denormalize the parameters
        X = vrange[0] + (vrange[1] - vrange[0]) * X

        # Rosenbrock
        Y = np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + \
                   (1 - X[:, :-1]) ** 2, axis=1).reshape(-1, 1)

        if hook:
            for i, y in enumerate(Y):
                await asyncio.sleep(wall_time)
                finished_time = dt.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                # it's very strange that if the obj is negative, the bokeh
                # plot will not update. however if use obj[0] instead of obj
                # when update the plot, the plot will update...
                # will try to figure this out when have more time
                # if i % 2:
                #     hook(X[i], -y, finished_time)
                # else:
                #     hook(X[i], y, finished_time)
                hook(X[i], y, finished_time)

        return Y
    
    return evaluate
