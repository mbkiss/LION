# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# Author: Max Kiss
# Modifications: Ander Biguri
# Based on: https://github.com/jinxixiang/FISTA-Net/
# Reproduce ISTA-Net (DOI 10.1109/CVPR.2018.00196)
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
import os

import tomosipo as ts
from tomosipo.torch_support import to_autograd
from ts_algorithms import fdk

import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


# Define ISTA-Net-plus Block
class BasicBlock(LIONmodel.LIONmodel):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None):
        super().__init__(geometry_parameters, model_parameters)
        self.geo = geometry_parameters
        self._make_operator()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))


        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    @staticmethod
    def default_parameters():
        pass

    def forward(self, x, PhiTPhi, PhiTb): #, mask):
        
        # print("lambda_step: ", self.lambda_step)
        # print("soft_thr: ", self.soft_thr)

        # convert data format from (batch_size, channel, pnum, pnum) to (circle_num, batch_size)
        #pnum = x.size()[2]
        #x = x.view(x.size()[0], x.size()[1], pnum*pnum, -1)   # (batch_size, channel, pnum*pnum, 1)
        #x = torch.squeeze(x, 1)
        #x = torch.squeeze(x, 2).t()             
        #x = mask.mm(x)  
        B, C, W, H = x.shape

        f = x.new_zeros(B, 1, *self.geo.image_shape[1:])
        for i in range(B):
            aux = fdk(self.op, x[i, 0])
            aux = torch.clip(aux, min=0)
            f[i] = aux
        x = f

        # rk block in the paper
        x = x - self.lambda_step  * PhiTPhi.mm(x) + self.lambda_step * PhiTb

        # convert (circle_num, batch_size) to (batch_size, channel, pnum, pnum)
        #x = torch.mm(mask.t(), x)
        #x = x.view(pnum, pnum, -1)
        #x = x.unsqueeze(0)
        x_input = x# x.permute(3, 0, 1, 2)

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define ISTA-Net
# got rid of , mask, after Phi
class ISTANet(LIONmodel.LIONmodel):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None, LayerNo = 6):
        super().__init__(model_parameters, geometry_parameters)
        onelayer = []
        self.LayerNo = LayerNo
        self.geo = geometry_parameters
        self._make_operator()
        self.Phi = self.A #Phi
        #self.mask = mask

        for i in range(LayerNo):
            onelayer.append(BasicBlock(geometry_parameters))

        self.fcs = nn.ModuleList(onelayer)

    @staticmethod
    def default_parameters():
        ISTA_params = LIONParameter()

        return ISTA_params

    def forward(self, Qinit, b):
        
        # convert data format from (batch_size, channel, vector_row, vector_col) to (vector_row, batch_size)
        #b = torch.squeeze(b, 1)
        #b = torch.squeeze(b, 2)
        #b = b.t()

        PhiTPhi = self.Phi.t().mm(self.Phi)
        PhiTb = self.Phi.t().mm(b)
        x = Qinit
        layers_sym = []   # for computing symmetric loss
        xnews = [] # iteration result
        xnews.append(x)
        
        for i in range(self.LayerNo):
            # print("iteration #{}:".format(i))
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb) #, self.mask)
            layers_sym.append(layer_sym)
            xnews.append(x)

        x_final = x

        return [x_final, layers_sym]
