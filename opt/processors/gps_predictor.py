# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

from ..utils.helpers import SE_kernel, hinv, hinv_auto

def model(X, Y, params=None):
    theta = None
    sigma_f = None
    sigma_n = None
    C = None
    lamb = None
    try:
        theta = params['theta']
    except KeyError:
        pass
    try:
        sigma_f = np.sqrt(params['var'])
    except KeyError:
        pass
    except TypeError:
        sigma_f = np.std(Y, axis=0).reshape(1, -1)
    try:
        sigma_n = params['sigma_n']
    except KeyError:
        pass
    try:
        C = params['C']
    except KeyError:
        pass
    try:
        lamb = params['lamb']
    except KeyError:
        pass
    if (lamb is None) and (C is None):
        raise Exception('C or lamb must be specified!')
 
    kernel_list = []
    invA_list = []
    A_list = []
    for i in range(Y.shape[1]):
        try:
            kernel = SE_kernel(sigma_f[0, i], theta[0, 0])
            K = kernel(X, X)
            A = K + sigma_n[0, 0] ** 2 * np.eye(K.shape[0])
            kernel_list.append(kernel)
            A_list.append(A)
            if lamb is None:
                invA = hinv_auto(A, C)
            else:
                invA = hinv(A, lamb)
            invA_list.append(invA)
        except:
            # Need to optimize the params here!
            raise Exception('params optimizer not implemented yet')
        
    def predict(Xs):
        mu_list = []
        sigma_list = []
        for i in range(Y.shape[1]):
            kernel = kernel_list[i]
            invA = invA_list[i]
            
            Ks = kernel(Xs, X)
            mu = np.linalg.multi_dot([Ks, invA, Y[:, i:i+1]])
            mu_list.append(mu)
            
            Kss = kernel(Xs, Xs)
            cov = Kss - np.linalg.multi_dot([Ks, invA, Ks.T])
            sigma = np.sqrt(cov.diagonal().reshape(-1, 1))
            sigma_list.append(sigma)
        
        mu = np.hstack(mu_list)
        sigma = np.hstack(sigma_list)
            
        return mu, sigma, A_list
    
    return predict

def process(X, params=None):
    X0 = np.array(X[0][0])
    Y0 = np.array(X[0][1])
    X1 = np.array(X[1])
    
    Y0_mean = np.mean(Y0, axis=0)
    _Y0 = Y0 - Y0_mean
    
    predict = model(X0, _Y0, params)
    _Y1_mu, Y1_sigma, _ = predict(X1)
    Y1_mu = _Y1_mu + Y0_mean
    
    P = [Y1_mu.tolist(), Y1_sigma.tolist()]
    return P
