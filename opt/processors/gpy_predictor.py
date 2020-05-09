# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy

def model_gpy(X, Y, params=None):
#     theta = params['theta']
#     sigma_f = np.sqrt(params['var'])
#     sigma_n = params['sigma_n']
    
    D = X.shape[1]
    model_list = []
    for i in range(Y.shape[1]):
        kernel = GPy.kern.RBF(D)
        model = GPy.models.GPRegression(X, Y[:, i:i+1], kernel)
        
#         model.rbf.variance = sigma_f[0, i] ** 2
#         model.rbf.lengthscale = theta[0, 0]
#         model.Gaussian_noise.variance = sigma_n[0, 0] ** 2
        model['.*lengthscale'].constrain_bounded(0.1, 10)
        model.optimize_restarts(num_restarts=20, verbose=False, parallel=True)
#         print(f'sigma_f: {np.sqrt(model.rbf.variance)}, theta: {model.rbf.lengthscale}, sigma_n: {np.sqrt(model.Gaussian_noise.variance)}')
        
        model_list.append(model)

    def predict(Xs):
        mu_list = []
        sigma_list = []
        for i in range(Y.shape[1]):
            model = model_list[i]
            
            mu, var = model.predict(Xs)
            mu_list.append(mu)
            sigma_list.append(np.sqrt(var))
        
        mu = np.hstack(mu_list)
        sigma = np.hstack(sigma_list)
        return mu, sigma
    return predict

def process(X):
    X0 = np.array(X[0][0])
    Y0 = np.array(X[0][1])
    X1 = np.array(X[1])
    
    Y0_mean = np.mean(Y0, axis=0)
    _Y0 = Y0 - Y0_mean
    
    predict = model_gpy(X0, _Y0)
    _Y1_mu, Y1_sigma = predict(X1)
    Y1_mu = _Y1_mu + Y0_mean
    
    P = [Y1_mu.tolist(), Y1_sigma.tolist()]
    return P
