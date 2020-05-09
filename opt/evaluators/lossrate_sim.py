# Copyright (c) 2019-2020, SPEAR3 authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import datetime
import json

from ..utils.spear3_lt_opt import x_to_pv

class LossRateNet(nn.Module):
    def __init__(self):
        super(LossRateNet, self).__init__()
        self.fc1 = nn.Linear(13, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 80)
        self.fc4 = nn.Linear(80, 60)
        self.fc5 = nn.Linear(60, 40)
        self.fc6 = nn.Linear(40, 20)
        self.fc7 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return x

def evaluator_generator(model_fname, vrange=(0, 1), noise_level=0, dt=0.5):
    info = None
    with open(model_fname, 'r') as f:
        info = json.load(f)
        
    # load the pre-trained net
    net = LossRateNet()
    if torch.cuda.is_available():
        model = torch.load(info['model_path'])
    else:
        print('gpu not found, use cpu version')
        model = torch.load(info['model_path'], map_location=torch.device('cpu'))
    net.load_state_dict(model)
    
    # load other properties
    mu_X = np.array(info['mu_X'])
    sigma_X = np.array(info['sigma_X'])
    mu_Y = np.array(info['mu_Y'])
    sigma_Y = np.array(info['sigma_Y'])
    
    async def evaluate(X, hook=None):
        assert type(X) == np.ndarray, 'Input X must be a numpy array'
        
        PV = x_to_pv(X, vrange[0], vrange[1])
        
        Y = None
        with torch.no_grad():
            _X = (PV - mu_X) / sigma_X
            _Y = net(torch.from_numpy(_X).float()).numpy()
            Y = _Y * sigma_Y + mu_Y
            
        if hook:
            for i, pv in enumerate(PV):
                await asyncio.sleep(dt)
                finished_time = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
                hook(pv, Y[i], finished_time)

        return Y
    
    return evaluate
