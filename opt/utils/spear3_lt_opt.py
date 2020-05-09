# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import asyncio
import pickle

from .spear3_api import within_spear3, \
    _read_qss_value, _write_qss_value, \
    _get_lifetime

# api for the lossrate opt
def _get_origin(connection):
    # get skew quadrupole PV values
    pv_name_list = [
        '01G-QSS4',
        '02G-QSS3',
        '05G-QSS3',
        '07G-QSS2',
        '08G-QSS2',
        '09G-QSS1',
        '10G-QSS4',
        '11G-QSS3',
        '12G-QSS3',
        '14G-QSS2',
        '16G-QSS2',
        '17G-QSS2',
        '18G-QSS1',
    ]
    df = _read_qss_value(pv_name_list, connection)
    origin = np.array(df['CurrSetpt'].tolist()).reshape(1, -1)
    
    return origin

@within_spear3
def get_origin(connection):
    return _get_origin(connection)

def _set_origin(origin, connection):
    assert origin.shape[1] == 13, 'Origin dimension must be equal to 13'
    
    # set quadrupole PV values
    pv_name_list = [
        '01G-QSS4',
        '02G-QSS3',
        '05G-QSS3',
        '07G-QSS2',
        '08G-QSS2',
        '09G-QSS1',
        '10G-QSS4',
        '11G-QSS3',
        '12G-QSS3',
        '14G-QSS2',
        '16G-QSS2',
        '17G-QSS2',
        '18G-QSS1',
    ]
    df = _write_qss_value(pv_name_list, origin[0], connection)
    
    return df

@within_spear3
def set_origin(origin, connection):
    return _set_origin(origin, connection)

async def _collect_lifetime(connection, condition_array, repeat=20, duration=1, verbose=True, output='collected_lifetime'):
    origin = _get_origin(connection)
    
    data = []
    for i in range(len(condition_array)):
        condition = condition_array[i:i+1, :]
        _set_origin(condition, connection)
        
        lifetime_list = []
        for j in range(repeat):
            if verbose:
                print(f'measuring lifetime for condition {i + 1}: {j + 1}/{repeat} ...', end='\r')
            lifetime = await _get_lifetime(connection, duration)
            lifetime_list.append(lifetime)
            if output:
                pickle.dump((condition_array, data, lifetime_list), open(output, 'wb'))
            await asyncio.sleep(1)
        
        data.append(lifetime_list)
    
    # reset to origin
    _set_origin(origin, connection)
    
    if output:
        pickle.dump((condition_array, data, []), open(output, 'wb'))
        
    data = np.array(data)
    return data

@within_spear3
async def collect_lifetime(connection, condition_array, repeat=20, duration=1, verbose=True, output='collected_lifetime'):
    return await _collect_lifetime(connection, condition_array, repeat, duration, verbose, output)

def x_to_pv(X, vmin=None, vmax=None):
    assert X.shape[1] == 13, 'Input dimension must be equal to 13'
    
    # denormalize the parameters
    if vmin is None:
        vmin = -20 * np.ones(X.shape[1])
    if vmax is None:
        vmax = 20 * np.ones(X.shape[1])
    PV = X * (vmax - vmin) + vmin

    return PV

def pv_to_x(PV, vmin=None, vmax=None):
    assert PV.shape[1] == 13, 'Input dimension must be equal to 13'

    if vmin is None:
        vmin = -20 * np.ones(PV.shape[1])
    if vmax is None:
        vmax = 20 * np.ones(PV.shape[1])
    X = (PV - vmin) / (vmax - vmin)
    
    return X
