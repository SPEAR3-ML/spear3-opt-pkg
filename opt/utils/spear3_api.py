# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import pandas as pd
from fabric import Connection
import asyncio
import inspect
import time

from .helpers import parse_caget_result, get_str

# decorators
def within_spear3(func):
    if inspect.iscoroutinefunction(func):
        async def do_within_spear3(*args, **kwargs):
            c_list = [arg for arg in list(args) + list(kwargs.values()) \
                      if type(arg) is Connection]
            assert len(c_list) == 1, \
                'Input arguments must contain one and only one connection'

            c = c_list[0]
            opened = c.is_connected

            commands = [
                'source /afs/slac/g/spear/epics/epicsSetup',
                'setenv EPICS_CA_ADDR_LIST "spearca1:5100 spearca2:5100"'
            ]
            with c.prefix(commands[0]), c.prefix(commands[1]):
                results = await func(*args, **kwargs)
                if not opened:
                    c.close()

                return results

        return do_within_spear3
    else:
        def do_within_spear3(*args, **kwargs):
            c_list = [arg for arg in list(args) + list(kwargs.values()) \
                      if type(arg) is Connection]
            assert len(c_list) == 1, \
                'Input arguments must contain one and only one connection'

            c = c_list[0]
            opened = c.is_connected

            commands = [
                'source /afs/slac/g/spear/epics/epicsSetup',
                'setenv EPICS_CA_ADDR_LIST "spearca1:5100 spearca2:5100"'
            ]
            with c.prefix(commands[0]), c.prefix(commands[1]):
                results = func(*args, **kwargs)
                if not opened:
                    c.close()

                return results

        return do_within_spear3

# read/write PVs
def _read_pv_value(pv_name_list, connection):
    cmd_list = [f'caget {pv_name}' for pv_name in pv_name_list]
    results = connection.run(' && '.join(cmd_list), hide=True)
    
    pv_value_list = [parse_caget_result(line) \
                     for line in results.stdout.splitlines()]
    
    return pv_value_list

@within_spear3
def read_pv_value(pv_name_list, connection):
    return _read_pv_value(pv_name_list, connection)

def _write_pv_value(pv_name_list, pv_value_list, connection):
    # Update the values
    cmd_list = [f'caput {pv_name} {get_str(pv_value_list[i])}' \
                for i, pv_name in enumerate(pv_name_list)]
    connection.run(' && '.join(cmd_list), hide=True)

@within_spear3
def write_pv_value(pv_name_list, pv_value_list, connection):
    return _write_pv_value(pv_name_list, pv_value_list, connection)

# specific APIs
def _read_qss_value(pv_name_list, connection):
    _pv_name_list = [f'{pv_name}:Curr1' for pv_name in pv_name_list]
    curr1_values = _read_pv_value(_pv_name_list, connection)
    
    _pv_name_list = [f'{pv_name}:CurrSetpt' for pv_name in pv_name_list]
    setpt_values_raw = _read_pv_value(_pv_name_list, connection)
    setpt_values = [setpt_value_raw[1] for setpt_value_raw in setpt_values_raw]
    
    df = pd.DataFrame({
        'PV': pv_name_list,
        'Curr1': curr1_values,
        'CurrSetpt': setpt_values
    })
    # df.style.hide_index()
    
    return df

@within_spear3
def read_qss_value(pv_name_list, connection):
    return _read_qss_value(pv_name_list, connection)

def _write_qss_value(pv_name_list, pv_value_list, connection):
    # Update the values
    cmd_list = []
    for i, pv_name in enumerate(pv_name_list):
        pv_cmd_list = [
#             f'caput {pv_name}:ControlState 0',  # ARM -> HALT
            f'caput {pv_name}:CurrSetpt {pv_value_list[i]}',  # update value
#             f'caput {pv_name}:ControlState 1',  # HALT -> ARM
#             f'caput {pv_name}:ControlState 2'  # ARM -> RUN -> ARM
        ]
        cmd_list += pv_cmd_list
    connection.run(' && '.join(cmd_list), hide=True)
    
    # Report the results
    df = _read_qss_value(pv_name_list, connection)
    
    return df

@within_spear3
def write_qss_value(pv_name_list, pv_value_list, connection):
    return _write_qss_value(pv_name_list, pv_value_list, connection)

def _read_sext_value(pv_name_list, connection):
    _pv_name_list = [f'{pv_name}:Curr' for pv_name in pv_name_list]
    curr_values = _read_pv_value(_pv_name_list, connection)
    
    _pv_name_list = [f'{pv_name}:CurrSetpt' for pv_name in pv_name_list]
    setpt_values = _read_pv_value(_pv_name_list, connection)
    
    df = pd.DataFrame({
        'PV': pv_name_list,
        'Curr': curr_values,
        'CurrSetpt': setpt_values
    })
    # df.style.hide_index()
    
    return df

@within_spear3
def read_sext_value(pv_name_list, connection):
    return _read_sext_value(pv_name_list, connection)

def _write_sext_value(pv_name_list, pv_value_list, connection):
    # Update the values
    cmd_list = [
        # f'caput Fofb:ControlState 0'
    ]
    for i, pv_name in enumerate(pv_name_list):
        cmd_list.append(f'caput {pv_name}:CurrSetpt {pv_value_list[i]}')
    cmd_list += [
        # f'caput Fofb:ControlState 1',
        # f'caput Fofb:ControlState 2'
    ]
    connection.run(' && '.join(cmd_list), hide=True)
    
    # Report the results
    df = _read_sext_value(pv_name_list, connection)
    
    return df

@within_spear3
def write_sext_value(pv_name_list, pv_value_list, connection):
    return _write_sext_value(pv_name_list, pv_value_list, connection)

def _get_beam_current(connection):
    dcct = _read_pv_value(['SPEAR:Beam2CurrAvg'], connection)[0]
    
    return dcct

@within_spear3
def get_beam_current(connection):
    return _get_beam_current(connection)

def _get_injection_state(connection):
    inj_state_raw = _read_pv_value(['BUCKET:SelectStateSetpt'], connection)[0]
    inj_state = 0 if inj_state_raw == 'Off' else 1
    
    return inj_state 

@within_spear3
def get_injection_state(connection):
    return _get_injection_state(connection)

def _turn_off_injection(connection):
    _write_pv_value(['BUCKET:SelectStateSetpt'], [0], connection)

@within_spear3
def turn_off_injection(connection):
    return _turn_off_injection(connection)

def _turn_on_injection(connection):
    _write_pv_value(['BUCKET:SelectStateSetpt'], [1], connection)

@within_spear3
def turn_on_injection(connection):
    return _turn_on_injection(connection)

def _turn_off_kicker(connection):
    _write_pv_value(['SPEAR-KICKER:TriggerEnable'], [0], connection)

@within_spear3
def turn_off_kicker(connection):
    return _turn_off_kicker(connection)

def _turn_on_kicker(connection):
    _write_pv_value(['SPEAR-KICKER:TriggerEnable'], [1], connection)

@within_spear3
def turn_on_kicker(connection):
    return _turn_on_kicker(connection)

async def _discharge_kicker(connection, duration=8):
    _turn_on_kicker(connection)
    await asyncio.sleep(duration)
    _turn_off_kicker(connection)

@within_spear3
async def discharge_kicker(connection, duration=8):
    return await _discharge_kicker(connection, duration)

def _set_kick_position(pos, connection):
    _write_pv_value(['BUCKET:Setpt'], [pos], connection)

@within_spear3
def set_kick_position(pos, connection):
    return _set_kick_position(pos, connection)

async def _get_inj_eff(connection, limit=78, measure_time=11, hook=None):
    assert measure_time >= 2, 'Measure time must be no less than 2s'
    
    inj_state = _get_injection_state(connection)
    
    if inj_state:
        _turn_off_injection(connection)
    
    dcct0 = _get_beam_current(connection)
    if dcct0 > limit:
        print(f'dcct exceeded limition {limit}: {dcct0}, action required.')
        input('press Enter to continue...')
        dcct0 = _get_beam_current(connection)
    
    _turn_on_injection(connection)
    await asyncio.sleep(measure_time - 2)
    
    _turn_off_injection(connection)
    await asyncio.sleep(0.5)
    
    booQ9_raw = _read_pv_value(['BOO-QM:Buffer9'], connection)[0]
    booQ9 = np.array(booQ9_raw[1:]) - (-0.0344)
    await asyncio.sleep(1.5)
    
    dcct1 = _get_beam_current(connection)
    
    fillrate = (dcct1 - dcct0) / measure_time * 60  # mA/min
    
    booQ9_eff = booQ9[booQ9 > 0.015]
    if len(booQ9_eff) <= 8:
        print('no booster beam')
        avg_booQ9 = 0.5  # very large for detuned beam
        # rms_booQ9 = 0.0
    else:
        avg_booQ9 = np.mean(booQ9_eff)
        # rms_booQ9 = np.std(booQ9_eff)
    
    inj_eff = fillrate / avg_booQ9 * 0.24
    
    if inj_state:
        _turn_on_injection(connection)
    
    return inj_eff

@within_spear3
async def get_inj_eff(connection, limit=78, measure_time=11):
    return await _get_inj_eff(connection, limit, measure_time)

async def _inject_beam(connection, target=3, inj_point=None):
    inj_state = _get_injection_state(connection)
    
    if inj_state:
        _turn_off_injection(connection)
    
    pv_value_init = None
    if inj_point is None:
        print('no injection point found, please set it manually.')
        input('press Enter to continue...')
    else:
        pv_name_list, pv_value_list = inj_point
        pv_value_init = _read_pv_value(pv_name_list, connection)
        _write_pv_value(pv_name_list, pv_value_list, connection)
    _turn_on_injection(connection)

    while dcct < target:
        await asyncio.sleep(1)
        dcct = _get_beam_current(connection)
    _turn_off_injection(connection)

    print(f'dcct reached target {target}: {dcct}, injection done.')

    # recover previous working point if needed
    if inj_point is None:
        print('do not foget to recover the sext PVs if you modified them.')
        input('press Enter to continue...')
    else:
        _write_pv_value(pv_name_list, pv_value_init, connection)
        
    if inj_state:
        _turn_on_injection(connection)

@within_spear3
async def inject_beam(connection, target=3, inj_point=None):
    return await _inject_beam(connection, target, inj_point)

async def _get_da(connection, kicker_volt_list=np.arange(0.5, 1.1, 0.025), \
                  min_dcct=3, target_dcct=3, inj_point=None, hook=None):
    inj_state = _get_injection_state(connection)
    
    if inj_state:
        _turn_off_injection(connection)
    
    # save kicker init value
    kicker_name_list = [
        '02S-K1:VoltSetpt',
        '03S-K2:VoltSetpt',
        '04S-K3:VoltSetpt'
    ]
    kicker_init_value = _read_pv_value(kicker_name_list, connection)
    
    # check beam current, inject if needed
    dcct = _get_beam_current(connection)
    if dcct < min_dcct:
        assert target_dcct >= min_dcct, 'Target dcct must be no lower than min dcct'
        
        print(f'dcct lower than {min_dcct}: {dcct}, start injection...')
        await _inject_beam(connection, target_dcct, inj_point)
    
    # config the kicker
    _set_kick_position(5, connection)
    _turn_off_kicker(connection)
    await asyncio.sleep(0.1)
    
    _write_pv_value(kicker_name_list, [0, 0, 0], connection)
    await _discharge_kicker(connection, 8)
    
    # scan kicker voltage and collect data
    for kicker_volt in kicker_volt_list:
        _turn_off_kicker(connection)
        await asyncio.sleep(0.2)
        _write_pv_value(kicker_name_list[0:1], [kicker_volt], connection)
        await asyncio.sleep(2)
        _turn_on_kicker(connection)
        await asyncio.sleep(0.5)
        
        # do spear_acquire_eight_rxoneturn
        _turn_off_kicker(connection)
        # get data?
        
        volt = _read_pv_value(['02S-K1:VoltReadback'], connection)[0]
        dcct = _get_beam_current(connection)
        if hook:
            hook(volt, dcct)
        await asyncio.sleep(0.2)
        if dcct < 0.001:
            break
    
    # restore kickers
    _turn_off_kicker(connection)
    await asyncio.sleep(0.1)
    _write_pv_value(kicker_name_list, kicker_init_value, connection)
    await _discharge_kicker(connection, 2)
    
    # calculate da based on measured data
    da = 0
    
    if inj_state:
        _turn_on_injection(connection)
    
    return da

@within_spear3
async def get_da(connection, kicker_volt_list=np.arange(0.5, 1.1, 0.025), \
                 min_dcct=3, target_dcct=3, inj_point=None, hook=None):
    return await _get_da(connection, kicker_volt_list, min_dcct, \
                         target_dcct, inj_point, hook)

async def _monitor_beam_current(connection, freq=1, duration=None, hook=None):
    t = 0
    while (duration is None) or t <= duration:
        dcct = _get_beam_current(connection)
        if hook:
            hook(t, dcct)
        await asyncio.sleep(1 / freq)
        t += 1 / freq

@within_spear3
async def monitor_beam_current(connection, freq=1, duration=None, hook=None):
    return await _monitor_beam_current(connection, freq, duration, hook)

def _get_spear_injection_state(connection):
    spear_inj_state = _read_pv_value(['SPEAR:InjectState'], connection)[0]
    
    return spear_inj_state[0] != 'No'

@within_spear3
def get_spear_injection_state(connection):
    return _get_spear_injection_state(connection)

async def _get_loss_rate(connection, duration=1, timeout=30, return_dcct=False, hook=None):
    # usually injection would last no more than 10s
    # and the top-off injection mode injects beam every 5 minutes
    t_origin = time.time()
    
    measuring = False
    t0 = None
    dcct0 = None
    while True:
        if not _get_spear_injection_state(connection):
            if not measuring:
                measuring = True
                t0 = time.time()
                dcct0 = _get_beam_current(connection)
            else:
                dt = time.time() - t0
                if dt >= duration:
                    dcct1 = _get_beam_current(connection)
                    # make sure it's still not injecting after we measured
                    # the beam current
                    if not _get_spear_injection_state(connection):
                        loss_rate = (dcct0 - dcct1) * 60 / dt
                        if loss_rate > 0:
                            if return_dcct:
                                dcct_mean = (dcct0 + dcct1) / 2
                                return loss_rate, dcct_mean
                            else:
                                return loss_rate
                        else:  # something went wrong
                            measuring = False
                            t0 = None
                            dcct0 = None
                    else:  # reset
                        measuring = False
                        t0 = None
                        dcct0 = None
        else:
            if measuring:  # reset
                measuring = False
                t0 = None
                dcct0 = None
        
        await asyncio.sleep(1)
        if time.time() - t_origin >= timeout:
            print('loss rate measurement timeout!')
            
            if return_dcct:
                return None, None
            else:
                return None

@within_spear3
async def get_loss_rate(connection, duration=1, timeout=30, return_dcct=False, hook=None):
    return await _get_loss_rate(connection, duration, timeout, return_dcct, hook)

async def _get_lifetime(connection, duration=1, timeout=30, hook=None):
    loss_rate, dcct = await _get_loss_rate(connection, duration, timeout, True)
    if loss_rate is None:
        print('lifetime measurement timeout!')
            
        return None
    else:
        lifetime = loss_rate / (dcct / 500) ** 2
        
        return lifetime

@within_spear3
async def get_lifetime(connection, duration=1, timeout=30, hook=None):
    return await _get_lifetime(connection, duration, timeout, hook)
