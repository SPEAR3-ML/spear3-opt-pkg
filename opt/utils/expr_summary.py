# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import os
import pickle
import numpy as np
import pandas as pd

# general purpose
def read_expr_log(fname):
    logs = []

    with open(fname, 'rb') as f:
        while True:
            try:
                logs.append(pickle.load(f))
            except EOFError:
                break
    
    return logs

def combine_logs(logs):
    X = logs[0]['X']
    Y = logs[0]['Y']
    for log in logs[1:]:
        X = np.vstack((X, log['X']))
        Y = np.vstack((Y, log['Y']))
    combined_log = {
        'X': X,
        'Y': Y
    }
    
    return combined_log

def expr_read_data(root, data_filter=None, timestamp=False):
    measured_points = [p for p in os.listdir(root) if not data_filter or data_filter(os.path.join(root, p))]
    assert len(measured_points), 'no measured data found'
    
    X = []
    Y = []
    if timestamp:
        T = []
    for point in measured_points:
        data_point = pickle.load(open(os.path.join(root, point), 'rb'))
        X.append(data_point['pv'])
        obj = data_point['obj']
        if type(obj) is not np.ndarray:
            obj = [obj]
        Y.append(obj)
        if timestamp:
            T.append(data_point['time'])
    X = np.array(X)
    Y = np.array(Y)
    I = np.argsort(Y, axis=0)[:, 0]
    X = X[I]
    Y = Y[I]
    
    if timestamp:
        T = np.array(T)
        T = T[I]
        
        return X, Y, T
    
    return X, Y

def expr_data_filter(p):
    fname = os.path.split(p)[-1]
    return not fname.startswith('.') and not os.path.splitext(fname)[1]\
        and not os.path.isdir(p)

# da-opt summary
da_opt_read_data = expr_read_data
da_opt_data_filter = expr_data_filter

def da_opt_test_summary(root, data_filter=None):
    X, Y = da_opt_read_data(root, data_filter)

    data = np.hstack((X, Y))
    columns = [f'D{i + 1}' for i in range(X.shape[1])] + [f'O{i + 1}' for i in range(Y.shape[1])]
    summary = pd.DataFrame(data=data, columns=columns)
    
    return summary

def da_opt_expr_summary(root, data_filter=None, timestamp=False):
    if timestamp:
        X, Y, T = expr_read_data(root, data_filter, timestamp)
        
        summary = pd.DataFrame({
            'MS1-SFM': X[:, 0],
            'MS2-SF': X[:, 1],
            'MS3-SF': X[:, 2],
            'MS4-SF': X[:, 3],
            'MS5-SF': X[:, 4],
            'MS1-SDM': X[:, 5],
            'MS2-SD': X[:, 6],
            'MS3-SD': X[:, 7],
            'MS4-SD': X[:, 8],
            'MS5-SD': X[:, 9],
            'OBJ': Y[:, 0],
            'T': T
        })
    else:
        X, Y = expr_read_data(root, data_filter, timestamp)

        summary = pd.DataFrame({
            'MS1-SFM': X[:, 0],
            'MS2-SF': X[:, 1],
            'MS3-SF': X[:, 2],
            'MS4-SF': X[:, 3],
            'MS5-SF': X[:, 4],
            'MS1-SDM': X[:, 5],
            'MS2-SD': X[:, 6],
            'MS3-SD': X[:, 7],
            'MS4-SD': X[:, 8],
            'MS5-SD': X[:, 9],
            'OBJ': Y[:, 0]
        })

    return summary

# lifetime opt summary
def lt_opt_expr_summary(root, data_filter=None, timestamp=False):
    if timestamp:
        X, Y, T = expr_read_data(root, data_filter, timestamp)
        
        summary = pd.DataFrame({
            '01G-QSS4': X[:, 0],
            '02G-QSS3': X[:, 1],
            '05G-QSS3': X[:, 2],
            '07G-QSS2': X[:, 3],
            '08G-QSS2': X[:, 4],
            '09G-QSS1': X[:, 5],
            '10G-QSS4': X[:, 6],
            '11G-QSS3': X[:, 7],
            '12G-QSS3': X[:, 8],
            '14G-QSS2': X[:, 9],
            '16G-QSS2': X[:, 10],
            '17G-QSS2': X[:, 11],
            '18G-QSS1': X[:, 12],
            'OBJ': Y[:, 0],
            'T': T,
        })
    else:
        X, Y = expr_read_data(root, data_filter, timestamp)

        summary = pd.DataFrame({
            '01G-QSS4': X[:, 0],
            '02G-QSS3': X[:, 1],
            '05G-QSS3': X[:, 2],
            '07G-QSS2': X[:, 3],
            '08G-QSS2': X[:, 4],
            '09G-QSS1': X[:, 5],
            '10G-QSS4': X[:, 6],
            '11G-QSS3': X[:, 7],
            '12G-QSS3': X[:, 8],
            '14G-QSS2': X[:, 9],
            '16G-QSS2': X[:, 10],
            '17G-QSS2': X[:, 11],
            '18G-QSS1': X[:, 12],
            'OBJ': Y[:, 0],
        })

    return summary
