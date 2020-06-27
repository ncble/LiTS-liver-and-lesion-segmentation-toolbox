"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import spectral_norm
from torchsummary import summary


def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def ConvSN2d(in_channels, out_channels, kernel_size,
             stride=1,
             padding=0,
             dilation=1,
             groups=1,
             bias=True,
             padding_mode='zeros'):
    A = spectral_norm(nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias))
    # A.__class__.__name__ = 'ConvSN2d' ## [TODO]
    return A


def ConvSN3d(in_channels, out_channels, kernel_size,
             stride=1,
             padding=0,
             dilation=1,
             groups=1,
             bias=True,
             padding_mode='zeros'):
    A = spectral_norm(nn.Conv3d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias))
    # A.__class__.__name__ = 'ConvSN3d' ## [TODO]
    return A


def ConvTransposeSN2d(in_channels, out_channels, kernel_size,
                      stride=1,
                      padding=0,
                      output_padding=0,
                      groups=1,
                      bias=True,
                      dilation=1,
                      padding_mode='zeros'):
    A = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         output_padding=output_padding,
                                         groups=groups,
                                         bias=bias,
                                         dilation=dilation))
    # A.__class__.__name__ = 'ConvTransposeSN2d' ## [TODO]
    return A


def ConvTransposeSN3d(in_channels, out_channels, kernel_size,
                      stride=1,
                      padding=0,
                      output_padding=0,
                      groups=1,
                      bias=True,
                      dilation=1,
                      padding_mode='zeros'):
    A = spectral_norm(nn.ConvTranspose3d(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         output_padding=output_padding,
                                         groups=groups,
                                         bias=bias,
                                         dilation=dilation))
    # A.__class__.__name__ = 'ConvTransposeSN3d' ## [TODO]
    return A


def LinearSN(in_features, out_features, bias=True):
    A = spectral_norm(nn.Linear(in_features, out_features, bias=bias))
    # A.__class__.__name__ = 'LinearSN' ## [TODO]
    return A


if __name__ == "__main__":
    print("Start")
    # img_shape = (3, 128, 128)
    # model = ...
    # A = summary(model, img_shape)
