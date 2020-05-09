# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import asyncio
import inspect
import time
import datetime
import numpy as np

def print_args(func):
    def do_print_args(*args, **kwargs):
        print(args, kwargs)

    return do_print_args

def curry(func):
    f_args = []
    f_kwargs = {}
    def f(*args, **kwargs):
        nonlocal f_args, f_kwargs
        if args or kwargs:
            f_args += args
            f_kwargs.update(kwargs)
            return f
        else:
            return func(*f_args, *f_kwargs)
    return f

def make_awaitable(func):
    if inspect.iscoroutinefunction(func):
        return func
    else:
        async def func_a(*args, **kwargs):
            await asyncio.sleep(0)
            return func(*args, **kwargs)
        
        return func_a
    
make_async = make_awaitable

def make_sync(func):
    if inspect.iscoroutinefunction(func):
        def func_d(*args, **kwargs):
            return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
        
        return func_d
    else:
        return func
    
def try_get_number(s):
    try:
        f = float(s)
    except ValueError:
        return s
    else:
        return f
    
def parse_caget_result(line):
    tokens = line.split()
    size = len(tokens)
    
    assert size >= 2, 'Invalid caget result'
    
    if len(tokens) == 2:
        return try_get_number(tokens[1])
    
    return [try_get_number(token) for token in tokens[1:]]

def get_str(l):
    if isinstance(l, str):
        return l
    
    try:
        s = ' '.join([str(x) for x in l])
    except TypeError:
        return str(l)
    else:
        return s

def run_time(func, repeat=1):
    def cal_run_time(*args, **kwargs):
        start_time = time.time()
        for i in range(repeat):
            result = func(*args, **kwargs)
        end_time = time.time()
        dt = (end_time - start_time) / repeat
        
        return dt
    return cal_run_time

def get_timestamp(time_in_sec=None, time_str='%x %X.%f'):
    t = time_in_sec
    if t is None:
        t = time.time()
        
    return datetime.datetime.fromtimestamp(t).strftime(time_str)

def make_evaluate_2d(evaluate):
    # evaluate must be sync function!
    def evaluate_2d(x, y, hook=None):
        xx, yy = np.meshgrid(x, y)
        X = np.vstack((xx.flatten(), yy.flatten())).transpose()
        Y = evaluate(X, hook)
        Y.shape = xx.shape

        return Y
    
    return evaluate_2d
