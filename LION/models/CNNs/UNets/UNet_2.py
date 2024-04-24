# =============================================================================
# This file is part of LION library
# License : GPL-3
#
# From: Clone of `dival.reconstructors.networks.unet` that supports a custom norm instead of `BatchNorm2d`, optionally.
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


def get_unet_model(in_ch=1, out_ch=1, scales=5, skip=4,
                   channels=(32, 32, 64, 64, 128, 128), use_sigmoid=True,
                   use_norm=True, norm_constructor=None):
    assert (1 <= scales <= 6)
    skip_channels = [skip] * (scales)
    return UNet(in_ch=in_ch, out_ch=out_ch, channels=channels[:scales],
                skip_channels=skip_channels, use_sigmoid=use_sigmoid,
                use_norm=use_norm, norm_constructor=norm_constructor)


class UNet(LIONmodel.LIONmodel):
    def __init__(self, geometry_parameters: ct.Geometry, model_parameters: LIONParameter = None):
        super().__init__(model_parameters, geometry_parameters)
        assert (len(self.model_parameters.channels) == len(self.model_parameters.skip_channels))
        #self.scales = len(channels)
        #self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = InBlock(self.model_parameters.in_ch, self.model_parameters.channels[0], use_norm=self.model_parameters.use_norm,
                           norm_constructor=self.model_parameters.norm_constructor)
        for i in range(1, self.model_parameters.scales):
            self.down.append(DownBlock(in_ch=self.model_parameters.channels[i - 1],
                                       out_ch=self.model_parameters.channels[i],
                                       use_norm=self.model_parameters.use_norm,
                                       norm_constructor=self.model_parameters.norm_constructor))
        for i in range(1, self.model_parameters.scales):
            self.up.append(UpBlock(in_ch=self.model_parameters.channels[-i],
                                   out_ch=self.model_parameters.channels[-i - 1],
                                   skip_ch=self.model_parameters.skip_channels[-i],
                                   use_norm=self.model_parameters.use_norm,
                                   norm_constructor=self.model_parameters.norm_constructor))
        self.outc = OutBlock(in_ch=self.model_parameters.channels[0],
                             out_ch=self.model_parameters.out_ch)

    @staticmethod
    def default_parameters():
        UNet = LIONParameter()
        UNet.in_ch =1
        UNet.out_ch = 1
        UNet.scales = 5
        UNet.skip = 4
        UNet.channels = (32, 32, 64, 64, 128)
        UNet.skip_channels = [UNet.skip] * (UNet.scales)
        UNet.use_sigmoid = True
        UNet.use_norm = True
        UNet.norm_constructor = None

        return UNet

    def forward(self, x0):
        xs = [self.inc(x0), ]
        for i in range(self.model_parameters.scales - 1):
            xs.append(self.down[i](xs[-1]))
        x = xs[-1]
        for i in range(self.model_parameters.scales - 1):
            x = self.up[i](x, xs[-2 - i])
        return torch.sigmoid(self.outc(x)) if self.model_parameters.use_sigmoid else self.outc(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True,
                 norm_constructor=None):
        super(DownBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            if norm_constructor is None:
                norm_constructor = lambda ch: nn.BatchNorm2d(ch)
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                norm_constructor(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                norm_constructor(out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=2, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True,
                 norm_constructor=None):
        super(InBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        if use_norm:
            if norm_constructor is None:
                norm_constructor = lambda ch: nn.BatchNorm2d(ch)
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                norm_constructor(out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3, use_norm=True,
                 norm_constructor=None):
        super(UpBlock, self).__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.skip = skip_ch > 0
        if skip_ch == 0:
            skip_ch = 1
        if use_norm and norm_constructor is None:
            norm_constructor = lambda ch: nn.BatchNorm2d(ch)
        if use_norm:
            self.conv = nn.Sequential(
                norm_constructor(in_ch + skip_ch),
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                norm_constructor(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                norm_constructor(out_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, stride=1,
                          padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size,
                          stride=1, padding=to_pad),
                nn.LeakyReLU(0.2, inplace=True))

        if use_norm:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                norm_constructor(skip_ch),
                nn.LeakyReLU(0.2, inplace=True))
        else:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch, skip_ch, kernel_size=1, stride=1),
                nn.LeakyReLU(0.2, inplace=True))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.skip_conv(x2)
        if not self.skip:
            x2 = x2 * 0
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if (np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == min(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2,
                                   diff3:diff3 + target_shape3])
        return torch.cat(inputs_, dim=1)


class OutBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x

    def __len__(self):
        return len(self._modules)
