# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import asyncio
import pickle

from .spear3_api import within_spear3, \
    _read_sext_value, _write_sext_value, \
    _get_injection_state, _turn_off_injection, _turn_on_injection, \
    _get_inj_eff

# api for the da opt
def _get_origin(connection):
    # get sextupole PV values
    pv_name_list = [
        'MS1-SFM',
        'MS2-SF',
        'MS3-SF',
        'MS4-SF',
        'MS5-SF',
        'MS1-SDM',
        'MS2-SD',
        'MS3-SD',
        'MS4-SD',
        'MS5-SD'
    ]
    df = _read_sext_value(pv_name_list, connection)
    origin = np.array(df['CurrSetpt'].tolist()).reshape(1, -1)
    
    return origin

@within_spear3
def get_origin(connection):
    return _get_origin(connection)

def _set_origin(origin, connection):
    assert origin.shape[1] == 10, 'Origin dimension must be equal to 10'
    
    inj_state = _get_injection_state(connection)
    
    if inj_state:
        _turn_off_injection(connection)
    
    # set sextupole PV values
    pv_name_list = [
        'MS1-SFM',
        'MS2-SF',
        'MS3-SF',
        'MS4-SF',
        'MS5-SF',
        'MS1-SDM',
        'MS2-SD',
        'MS3-SD',
        'MS4-SD',
        'MS5-SD'
    ]
    _write_sext_value(pv_name_list, origin[0], connection)
    df = _read_sext_value(pv_name_list, connection)
    
    if inj_state:
        _turn_on_injection(connection)
    
    return df

@within_spear3
def set_origin(origin, connection):
    return _set_origin(origin, connection)

async def _collect_inj_eff(connection, condition_array, repeat=20, limit=17.5, verbose=True, output='collected_inj_eff'):
    origin = _get_origin(connection)
    
    data = []
    for i in range(len(condition_array)):
        condition = condition_array[i:i+1, :]
        _set_origin(condition, connection)
        
        inj_eff_list = []
        for j in range(repeat):
            if verbose:
                print(f'measuring injection efficiency for condition {i + 1}: {j + 1}/{repeat} ...', end='\r')
            inj_eff = await _get_inj_eff(connection, limit)
            inj_eff_list.append(inj_eff)
            if output:
                pickle.dump((condition_array, data, inj_eff_list), open(output, 'wb'))
            await asyncio.sleep(1)
        
        data.append(inj_eff_list)
    
    # reset to origin
    _set_origin(origin, connection)
    
    if output:
        pickle.dump((condition_array, data, []), open(output, 'wb'))
        
    data = np.array(data)
    return data

@within_spear3
async def collect_inj_eff(connection, condition_array, repeat=20, limit=17.5, verbose=True, output='collected_inj_eff'):
    return await _collect_inj_eff(connection, condition_array, repeat, limit, verbose, output)
    
def x_to_pv(X, origin, vmin=None, vmax=None):
    assert X.shape[1] == 8, 'Input dimension must be equal to 8'
    
    # denormalize the parameters
    if vmin is None:
        vmin = -20 * np.ones(X.shape[1])
    if vmax is None:
        vmax = 20 * np.ones(X.shape[1])
    P = X * (vmax - vmin) + vmin

    # convert to the real PV values
    d_chrod_sa = np.array([
        [0.0591, 0.0608, 0.0608, 0.0608, 0.0304, -0.0065, -0.0155, -0.0155, -0.0155, -0.0078],
        [-0.0388, -0.0714, -0.0714, -0.0714, -0.0357, 0.0445, 0.0863, 0.0863, 0.0863, 0.0431]
    ])
    mat_trans = np.linalg.svd(d_chrod_sa)[2][:, 2:].transpose()
    PV = origin + P.dot(mat_trans)
    
    return PV

def pv_to_x(PV, origin, vmin=None, vmax=None):
    assert PV.shape[1] == 10, 'Input dimension must be equal to 10'

    d_chrod_sa = np.array([
        [0.0591, 0.0608, 0.0608, 0.0608, 0.0304, -0.0065, -0.0155, -0.0155, -0.0155, -0.0078],
        [-0.0388, -0.0714, -0.0714, -0.0714, -0.0357, 0.0445, 0.0863, 0.0863, 0.0863, 0.0431]
    ])
    mat_trans = np.linalg.svd(d_chrod_sa)[2][:, 2:]
    P = (PV - origin).dot(mat_trans)
    
    if vmin is None:
        vmin = -20 * np.ones(P.shape[1])
    if vmax is None:
        vmax = 20 * np.ones(P.shape[1])
    X = (P - vmin) / (vmax - vmin)
    
    return X
