# This file is part of LION library
# License : GPL-3
#
# Author: Max Kiss
# Modifications: Ander Biguri
# Based on: https://arxiv.org/pdf/1704.04058.pdf
# =============================================================================

from LION.models import LIONmodel

from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
import LION.CTtools.ct_utils as ct_utils
import LION.utils.utils as ai_utils

import numpy as np
from pathlib import Path
import warnings

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import fdk

import torch
import torch.nn as nn
import torch.nn.functional as F

M=5


class update(LIONmodel.LIONmodel):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None):
        super().__init__(model_parameters, geometry_parameters)
        #size = self.geo.image_shape
        #angles = self.geo.angles


        #geometry = odl.tomo.parallel_beam_geometry(config.space, num_angles=angles)
        #self.fwd_op = odl.tomo.RayTransform(config.space, geometry, impl='astra_cuda')
        #self.grad_op =  odl.Gradient(config.space)
        self._make_operator()

        #Remove BatchNorm2d(32) because CT data is not normalized
        self.conv = nn.Sequential(
            nn.Conv2d(3+M, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1+M, 3, padding=1),
            nn.BatchNorm2d(1+M),
        )

    @staticmethod
    def default_parameters():
        pass

    def gradientTVnormBackward2D(self,f):
        Gx = torch.diff(f, dim=2)
        Gy = torch.diff(f, dim=3)
        tvg = torch.zeros_like(f)

        Gx = torch.cat((torch.zeros_like(Gx[:, :, :1, :]), Gx), dim=2)  # Pad Gx with zeros
        Gy = torch.cat((torch.zeros_like(Gy[:, :, :, :1]), Gy), dim=3)  # Pad Gy with zeros

        nrm = torch.sqrt(Gx**2 + Gy**2 +1e-6)
        
        tvg[:, :, :, :] = tvg[:, :, :, :] + (Gx[:, :, :, :] + Gy[:, :, :, :]) / nrm[:, :, :, :]
        tvg[:, :, :-1, :] = tvg[:, :, :-1, :] - Gx[:, :, 1:, :] / nrm[:, :, 1:, :]
        tvg[:, :, :, :-1] = tvg[:, :, :, :-1] - Gy[:, :, :, 1:] / nrm[:, :, :, 1:]
    
        return tvg

    def forward(self, f, s, x):
        del_L = self.AT(self.A(f)-x)
        del_S = self.gradientTVnormBackward2D(f)

        output = self.conv(torch.cat([f, s, del_L, del_S], dim=1)*1.0)
        f = f + output[:, M:M+1]
        s = nn.ReLU()(output[:, 0:M])
        return (f, s)

class LG(LIONmodel.LIONmodel):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None):
        super().__init__(model_parameters, geometry_parameters)
        self.geo = geometry_parameters
        self.M = 5
        self._make_operator()
        self.update = update(geometry_parameters)
        self.iterates = self.model_parameters.n_iters
    
    @staticmethod
    def default_parameters():
        LG_params = LIONParameter()
        LG_params.M = 5
        LG_params.n_iters = 5

        return LG_params
    
    def forward(self, x):

        B, C, W, H = x.shape

        f = x.new_zeros(B, 1, *self.geo.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, x[i, 0])
            aux = torch.clip(aux, min=0)
            f[i] = aux
        tmp = torch.zeros(f.shape).type_as(f)
        s = tmp.clone()
        for i in range(self.M-1):
            s = torch.cat((s, tmp), dim=1)
        for i in range(self.iterates):
            (f, s) = self.update(f, s, x)
        return f
    
    def init_weights(self, m):
        pass
