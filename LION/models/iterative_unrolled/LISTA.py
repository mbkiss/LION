# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# From: https://github.com/milesial/Pytorch-UNet
# Authors: Max Kiss, Ander Biguri
# =============================================================================


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from LION.models import LIONmodel
from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
import LION.CTtools.ct_utils as ct_utils
import LION.utils.utils as ai_utils

#%% 

def soft_thr(input_, theta_):
    return F.relu(input_-theta_)-F.relu(-input_-theta_)

class LISTA(LIONmodel.LIONmodel):
    def __init__(self,geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None):
        super().__init__(model_parameters, geometry_parameters)
        self._W = nn.Linear(in_features = self.model_parameters.m, out_features = self.model_parameters.n, bias=False)
        self._S = nn.Linear(in_features = self.model_parameters.n, out_features = self.model_parameters.n, bias=False)

        self.thr = nn.Parameter(torch.rand(self.model_parameters.numIter,1), requires_grad=True)
        self.numIter = self.model_parameters.numIter
        self.A = self.model_parameters.A
        self.alpha = self.model_parameters.alpha
        self.device = self.model_parameters.device


    @staticmethod
    def default_parameters():
        LISTA_params = LIONParameter()
        LISTA_params.numIter = 15
        LISTA_params.thr = nn.Parameter(torch.rand(LISTA_params.numIter,1), requires_grad=True)
        LISTA_params.m = 956 #70
        LISTA_params.n = 3600 #100
        # create the random matrix
        seed = 80
        rng = np.random.RandomState(seed)
        D = rng.normal(0, 1/np.sqrt(LISTA_params.m), [LISTA_params.m, LISTA_params.n])
        D /= np.linalg.norm(D,2,axis=0)

        LISTA_params.A = D
        LISTA_params.alpha = 5e-4
        LISTA_params.device = torch.device("cuda:0")

        return LISTA_params

    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(A.T, A))
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*A.T)
        B = B.float().to(self.device)
        
        thr = torch.ones(self.numIter, 1) * 0.1 / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
            
        for iter in range(self.numIter):
            d = soft_thr(self._W(y) + self._S(d), self.thr[iter])
            x.append(d)
        return x
