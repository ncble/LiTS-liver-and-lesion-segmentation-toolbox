"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0

TODO: merge this to unet.py

"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
try:
    # relative import
    from base_models import *
except:
    from models.base_models import *


class UNet3D(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        depth=4,
        start_ch=64,
        inc_rate=2,
        kernel_size=3,
        padding=True,
        batch_norm=True,
        spec_norm=False,
        dropout=0.5,
        up_mode='upconv',
        include_top=True,
        include_last_act=False,
        ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            depth (int): depth of the network
            start_ch (int): number of filters in the first layer is start_ch
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            dropout (None or float): Use dropout (if not None) in Conv block.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet3D, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.include_top = include_top
        self.padding = padding
        self.out_ch = out_ch
        self.depth = depth
        self.spec_norm = spec_norm
        self.include_last_act = include_last_act
        prev_channels = in_ch
        self.down_path = nn.ModuleList()
        for i in range(depth+1):
            self.down_path.append(
                UNetConvBlock(prev_channels, (inc_rate ** i) * start_ch,
                              padding, batch_norm, dropout, spec_norm=self.spec_norm, kernel_size=kernel_size)
            )
            prev_channels = (inc_rate ** i) * start_ch

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UNetUpBlock(prev_channels, (inc_rate ** i) * start_ch, up_mode,
                            padding, batch_norm, dropout, spec_norm=self.spec_norm)
            )
            prev_channels = (inc_rate ** i) * start_ch

        if self.include_top:
            if self.spec_norm:
                self.last = ConvSN3d(prev_channels, out_ch, kernel_size=1)
            else:
                self.last = nn.Conv3d(prev_channels, out_ch, kernel_size=1)
        else:
            self.out_ch = prev_channels
        if self.include_last_act:
            self.tanh_act = nn.Tanh()

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool3d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        if self.include_top:
            x = self.last(x)
            if self.include_last_act:
                return self.tanh_act(x)
            else:
                return x
        else:
            # no reason to apply activation here
            return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, dropout, spec_norm=False, kernel_size=3):
        super(UNetConvBlock, self).__init__()
        self.spec_norm = spec_norm
        block = []
        if padding:
            # [stride=1] padding = (k-1)/2
            padding = (kernel_size-1)/2

        if self.spec_norm:
            # [stride=1] padding = (k-1)/2
            block.append(ConvSN3d(in_size, out_size,
                                  kernel_size=kernel_size, padding=int(padding)))
        else:
            # [stride=1] padding = (k-1)/2
            block.append(nn.Conv3d(in_size, out_size,
                                   kernel_size=kernel_size, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))
        if dropout is not None:
            block.append(nn.Dropout(p=dropout))
        if self.spec_norm:
            block.append(ConvSN3d(out_size, out_size,
                                  kernel_size=kernel_size, padding=int(padding)))
        else:
            block.append(nn.Conv3d(out_size, out_size,
                                   kernel_size=kernel_size, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm3d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, dropout, spec_norm=False, kernel_size=3):
        super(UNetUpBlock, self).__init__()
        self.padding = padding
        self.spec_norm = spec_norm
        if padding:
            # [stride=2] output_padding + kernel_size = 2(padding + 1)
            if kernel_size%2 == 1:
                # odd number 
                output_padding = 1
                padding = int(kernel_size/2)
            else:
                raise ValueError("ConvTranspose for U-net doesn't support even kernel_size !")
                # even number
                # output_padding = 0
                # padding = int(kernel_size/2) - 1
        else:
            output_padding = 1
            padding = 1

        if up_mode == 'upconv':
            if self.spec_norm:
                # [stride=2] output_padding + kernel_size = 2(padding + 1)
                self.up = ConvTransposeSN3d(
                    in_size, out_size, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
            else:
                # [stride=2] output_padding + kernel_size = 2(padding + 1)
                self.up = nn.ConvTranspose3d(
                    in_size, out_size, kernel_size=kernel_size, stride=2, padding=padding, output_padding=output_padding)
        elif up_mode == 'upsample':

            if self.spec_norm:
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    ConvSN3d(in_size, out_size, kernel_size=1),
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    nn.Conv3d(in_size, out_size, kernel_size=1),
                )
        self.relu_act = nn.ReLU()
        self.conv_block = UNetConvBlock(
            in_size, out_size, padding, batch_norm, dropout, spec_norm=self.spec_norm, kernel_size=kernel_size)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.relu_act(up)
        # print("Brige shape: {}, target size: {}".format(bridge.shape, up.shape[2:]))
        if self.padding:
            crop1 = bridge
        else:
            crop1 = self.center_crop(bridge, up.shape[2:])
        # print(up.shape)
        # print(crop1.shape)
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

if __name__ == "__main__":
    print("Start")
    img_shape = (1, 32, 32, 32)
    model = UNet3D(in_ch=1,
                 out_ch=3,
                 depth=3,
                 start_ch=32,
                 inc_rate=2,
                 kernel_size=3,
                 padding=True,
                 batch_norm=True,
                 spec_norm=False,
                 dropout=0.5,
                 up_mode='upconv',
                 include_top=True,
                 )
    A = summary(model, img_shape)
