# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import numpy.random as random
from scipy.stats import norm

import GPy
from pyDOE import lhs

from .helpers import make_awaitable

def initialize(shape, method='rand', vrange=(0, 1), add_center=False):
    assert method in ['rand', 'lhs'], 'Specified method is not supported'
    
    X0 = None
    if not add_center:
        if method == 'rand':
            X0 = random.rand(*shape)
        elif method == 'lhs':
            X0 = lhs(shape[1], samples=shape[0], criterion='c')
    else:
        center = 0.5 * np.ones((1, shape[1]))
        if method == 'rand':
            X0 = random.rand(*shape)
            X0[0] = center
        elif method == 'lhs':
            X0 = lhs(shape[1], samples=shape[0] - 1, criterion='c')
            X0 = np.vstack((center, X0))
    
    # scale
    X0 = vrange[0] + (vrange[1] - vrange[0]) * X0
    
    return X0

def particle_swarm_init(X0, Y0):
    V0 = np.zeros(X0.shape)
    pbest0 = (X0, Y0)
    gbest0 = np.argmin(Y0)
    
    return V0, pbest0, gbest0

def particle_swarm_iter(X, V, pbest, gbest, w=1.0, c1=2.0, c2=2.0):
    r1 = random.rand()
    r2 = random.rand()
    
    Vn = w * V + c1 * r1 * (pbest[0] - X) + c2 * r2 * (pbest[0][gbest] - X)
    Xn = X + Vn
    
    # Prevent the parameters from running out of the range
    Xn[Xn < 0] = 0
    Xn[Xn > 1] = 1
    
    return Xn, Vn

def offset(X, V, amplitude=0.05, factor=20):
    if not factor:
        Xn = np.zeros((0, X.shape[1]))
        Vn = np.zeros((0, V.shape[1]))
        
        return Xn, Vn
    
    Xn = np.tile(X, (factor, 1))
    Vn = np.tile(V, (factor, 1))
#     Vn = np.zeros(Xn.shape)
    Vn[:X.shape[0]] = V

    Xn += amplitude * (1 - 2 * random.rand(*Xn.shape))
    Vn += amplitude * (1 - 2 * random.rand(*Vn.shape))
    
    Xn[Xn < 0] = 0
    Xn[Xn > 1] = 1
    
    return Xn, Vn

def crossover(X, V, factor=20):
    if not factor:
        Xn = np.zeros((0, X.shape[1]))
        Vn = np.zeros((0, V.shape[1]))
        
        return Xn, Vn
    
    Xn = np.tile(X, (factor, 1))
    Vn = np.tile(V, (factor, 1))
    
    I = random.randint(0, X.shape[0], Xn.shape[0])
    D = random.rand(Xn.shape[0]).reshape(-1, 1)
    
    Xn = (1 - D) * Xn + D * Xn[I]
    Vn = (1 - D) * Vn + D * Vn[I]
#     Vn = np.zeros(Xn.shape)
    amplitude = 0.05
    Vn += amplitude * (1 - 2 * random.rand(*Vn.shape))

    return Xn, Vn

def non_dominated_sorting(Y):
    return np.argsort(Y, axis=0)[:, 0]

def select(I, S, n=100):
    return [X[I[:n]] for X in S]

def pbest_update(pbest, X, Y):
    X_pbest, Y_pbest = pbest
    X_pbestn = np.copy(X_pbest)
    Y_pbestn = np.copy(Y_pbest)
    for i in range(X.shape[0]):
        if Y_pbestn[i] > Y[i]:
            Y_pbestn[i] = Y[i]
            X_pbestn[i] = X[i]
    pbestn = (X_pbestn, Y_pbestn)
    
    return pbestn

def gbest_update(pbest):
    # Note that this pbest has to be the updated one
    gbestn = np.argmin(pbest[1])
    
    return gbestn

def gaussian_process_generator(X, Y, bias=False, ARD=False):
    kernel = None
    if bias:
        kernel = GPy.kern.RBF(X.shape[1], ARD=ARD) + GPy.kern.Bias(X.shape[1])
    else:
        kernel = GPy.kern.RBF(X.shape[1], ARD=ARD)
    model = GPy.models.GPRegression(X, Y, kernel)
    
    return model

def combine_populations(populations):
    return np.concatenate(populations, axis=0)

def upper_confidence_bound(X, gp, kappa=3):
    Y_mean, Y_variance = gp.predict(X)
    ucb = Y_mean - kappa * np.sqrt(Y_variance)
    
    return ucb

def surrogate_based_optimization(X, gp):
    sbo, _ = gp.predict(X)
    
    return sbo

def expected_improvement(X, gp, Y_sample_opt, xi=0.01):
    Y_mean, Y_variance = gp.predict(X)
    Y_std = np.sqrt(Y_variance)
    
    return _expected_improvement(Y_mean, Y_std, Y_sample_opt, xi)

def _expected_improvement(Y_mean, Y_std, Y_sample_opt, xi=0.01):
    with np.errstate(divide='warn'):
        imp = Y_sample_opt - Y_mean - xi
        Z = imp / Y_std
        ei = imp * norm.cdf(Z) + Y_std * norm.pdf(Z)
        ei[Y_std == 0.0] = 0.0

    return ei
