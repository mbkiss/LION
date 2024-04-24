# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# From: https://github.com/milesial/Pytorch-UNet
# Authors: Max Kiss, Ander Biguri
# =============================================================================
import torch.nn as nn
import torch
import math
import pdb

from LION.models import LIONmodel
from LION.utils.math import power_method
from LION.utils.parameter import LIONParameter
import LION.CTtools.ct_geometry as ct
import LION.CTtools.ct_utils as ct_utils
import LION.utils.utils as ai_utils

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1,
                 padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(nOut),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck,
                 bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                   stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down',
                                bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal',
                                   bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal',
                                   bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]

        return torch.cat(res, dim=1)

class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * args.grFactor[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * args.grFactor[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
        self.layers.append(conv)

        nIn = nOut * args.grFactor[0]

        for i in range(1, args.nScales):
            self.layers.append(ConvBasic(nIn, nOut * args.grFactor[i],
                                         kernel=3, stride=2, padding=1))
            nIn = nOut * args.grFactor[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res

class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, args, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else args.nScales
        self.outScales = outScales if outScales is not None else args.nScales

        self.nScales = args.nScales
        self.discard = self.inScales - self.outScales

        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * args.grFactor[self.offset - 1]
            nIn2 = nIn * args.grFactor[self.offset]
            _nOut = nOut * args.grFactor[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[self.offset - 1],
                                              args.bnFactor[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * args.grFactor[self.offset],
                                          nOut * args.grFactor[self.offset],
                                          args.bottleneck,
                                          args.bnFactor[self.offset]))

        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * args.grFactor[i - 1]
            nIn2 = nIn * args.grFactor[i]
            _nOut = nOut * args.grFactor[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, args.bottleneck,
                                              args.bnFactor[i - 1],
                                              args.bnFactor[i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """
    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res

class MSDNet(LIONmodel.LIONmodel):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None):
        super().__init__(model_parameters, geometry_parameters)
        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.nBlocks = self.model_parameters.nBlocks
        self.steps = [self.model_parameters.base]
        #self.model_parameters = model_parameters
        
        n_layers_all, n_layer_curr = self.model_parameters.base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(self.model_parameters.step if self.model_parameters.stepmode == 'even'
                             else self.model_parameters.step * i + 1)
            n_layers_all += self.steps[-1]

        print("building network of steps: ")
        print(self.steps, n_layers_all)

        nIn = self.model_parameters.nChannels
        for i in range(self.nBlocks):
            print(' ********************** Block {} '
                  ' **********************'.format(i + 1))
            m, nIn = \
                self._build_block(nIn, model_parameters, self.steps[i],
                                  n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, model_parameters, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(3, nIn, self.model_parameters)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            inScales = self.model_parameters.nScales
            outScales = self.model_parameters.nScales
            if self.model_parameters.prune == 'min':
                inScales = min(self.model_parameters.nScales, n_layer_all - n_layer_curr + 2)
                outScales = min(self.model_parameters.nScales, n_layer_all - n_layer_curr + 1)
            elif self.model_parameters.prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / self.model_parameters.nScales)
                inScales = self.model_parameters.nScales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = self.model_parameters.nScales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            layers.append(MSDNLayer(nIn, self.model_parameters.growthRate, self.model_parameters, inScales, outScales))
            print('|\t\tinScales {} outScales {} inChannels {} outChannels {}\t\t|'.format(inScales, outScales, nIn, self.model_parameters.growthRate))

            nIn += self.model_parameters.growthRate
            if self.model_parameters.prune == 'max' and inScales > outScales and \
                    self.model_parameters.reduction > 0:
                offset = self.model_parameters.nScales - outScales
                layers.append(
                    self._build_transition(nIn, math.floor(1.0 * self.model_parameters.reduction * nIn),
                                           outScales, offset, model_parameters))
                _t = nIn
                nIn = math.floor(1.0 * self.model_parameters.reduction * nIn)
                print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(1.0 * self.model_parameters.reduction * _t)))
            elif self.model_parameters.prune == 'min' and self.model_parameters.reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = self.model_parameters.nScales - outScales
                layers.append(self._build_transition(nIn, math.floor(1.0 * self.model_parameters.reduction * nIn),
                                                     outScales, offset, model_parameters))

                nIn = math.floor(1.0 * self.model_parameters.reduction * nIn)
                print('|\t\tTransition layer inserted! (min)\t|')
            print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, model_parameters):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * self.model_parameters.grFactor[offset + i],
                                 nOut * self.model_parameters.grFactor[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)

    @staticmethod
    def default_parameters():
        MSD_params = LIONParameter()
        MSD_params.nBlocks = 1
        MSD_params.nChannels = 32
        MSD_params.base =4
        MSD_params.stepmode = 'even' #'lin_grow'
        MSD_params.step = 1
        MSD_params.growthRate = 6
        MSD_params.grFactor = [1,2,4]
        MSD_params.prune = 'max' #'min'
        MSD_params.bnFactor = [1,2,4]
        MSD_params.bottleneck = True
        MSD_params.nScales = len(MSD_params.grFactor)
        MSD_params.reduction = 0.5

        return MSD_params


    def forward(self, x):
        res = []
        for i in range(self.nBlocks):
            x = self.blocks[i](x)
            res.append(self.classifier[i](x))
        return res
