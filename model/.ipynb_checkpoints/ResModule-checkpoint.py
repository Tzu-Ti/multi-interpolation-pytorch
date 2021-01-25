__author__ = 'Titi'

import torch
import torch.nn as nn
import numpy as np
import cv2

from utils.pixelShuffle_torch import pixel_shuffle

__author__ = 'Titi'

import torch
import torch.nn as nn

class ConvNorm(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, norm=False):
        super(ConvNorm, self).__init__()

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv = nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=kernel_size, bias=True)

        self.norm = norm
        if norm == 'IN':
            self.norm = nn.InstanceNorm2d(out_channel, track_running_stats=True)
        elif norm == 'BN':
            self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm:
            out = self.norm(out)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y
    
## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, reduction, bias=True, norm=False, act=nn.ReLU(True), downscale=False, return_ca=False):
        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_channel, out_channel, kernel_size, stride=2 if downscale else 1, norm='BN'),
            act,
            ConvNorm(out_channel, out_channel, kernel_size, stride=1, norm='BN'),
            CALayer(out_channel, reduction)
        )
        self.downscale = downscale
        if downscale:
            self.downConv = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=2, padding=1)
        self.return_ca = return_ca

    def forward(self, x):
        res = x
        out, ca = self.body(x)
        if self.downscale:
            res = self.downConv(res)
        out += res

        if self.return_ca:
            return out, ca
        else:
            return out

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_resblocks, n_channel, kernel_size, reduction, act, norm=False):
        super(ResidualGroup, self).__init__()

        modules_body = [RCAB(n_channel, n_channel, kernel_size, reduction, bias=True, norm=norm, act=act)
            for _ in range(n_resblocks)]
        modules_body.append(ConvNorm(n_channel, n_channel, kernel_size, stride=1, norm=norm))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

class RES(nn.Module):
    def __init__(self, n_resgroups, n_resblocks, n_channel, reduction=8, act=nn.ReLU(True), norm=False):
        super(RES, self).__init__()

        # define modules: body, tail

        modules_body = [
            ResidualGroup(
                n_resblocks=n_resblocks,
                n_channel=n_channel,
                kernel_size=3,
                reduction=reduction, 
                act=act, 
                norm=norm)
            for _ in range(n_resgroups)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = conv3x3(n_channel, n_channel)

    def forward(self, x):
        res = self.body(x)
        res += x
        out = self.tailConv(res)
        return out
