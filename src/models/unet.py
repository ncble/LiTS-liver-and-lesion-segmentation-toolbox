"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0


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


class UNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        depth=4,
        start_ch=64,
        inc_rate=2,
        padding=True,
        batch_norm=True,
        spec_norm=False,
        dropout=0.5,
        up_mode='upconv',
        include_top=True
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
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.include_top = include_top
        self.padding = padding
        self.out_ch = out_ch
        self.depth = depth
        self.spec_norm = spec_norm
        prev_channels = in_ch
        self.down_path = nn.ModuleList()
        for i in range(depth+1):
            self.down_path.append(
                UNetConvBlock(prev_channels, (inc_rate ** i) * start_ch,
                              padding, batch_norm, dropout, spec_norm=self.spec_norm)
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
                self.last = ConvSN2d(prev_channels, out_ch, kernel_size=1)
            else:
                self.last = nn.Conv2d(prev_channels, out_ch, kernel_size=1)
        else:
            self.out_ch = prev_channels
        self.tanh_act = nn.Tanh()

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        if self.include_top:
            x = self.last(x)
            return self.tanh_act(x)
        else:
            return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, dropout, spec_norm=False):
        super(UNetConvBlock, self).__init__()
        self.spec_norm = spec_norm
        block = []
        if self.spec_norm:
            # [stride=1] padding = (k-1)/2
            block.append(ConvSN2d(in_size, out_size,
                                  kernel_size=3, padding=int(padding)))
        else:
            # [stride=1] padding = (k-1)/2
            block.append(nn.Conv2d(in_size, out_size,
                                   kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        if dropout is not None:
            block.append(nn.Dropout(p=dropout))
        if self.spec_norm:
            block.append(ConvSN2d(out_size, out_size,
                                  kernel_size=3, padding=int(padding)))
        else:
            block.append(nn.Conv2d(out_size, out_size,
                                   kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, dropout, spec_norm=False):
        super(UNetUpBlock, self).__init__()
        self.padding = padding
        self.spec_norm = spec_norm
        if up_mode == 'upconv':
            if self.spec_norm:
                self.up = ConvTransposeSN2d(
                    in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1)
            else:
                # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
                self.up = nn.ConvTranspose2d(
                    in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif up_mode == 'upsample':

            if self.spec_norm:
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    ConvSN2d(in_size, out_size, kernel_size=1),
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    nn.Conv2d(in_size, out_size, kernel_size=1),
                )
        self.relu_act = nn.ReLU()
        self.conv_block = UNetConvBlock(
            in_size, out_size, padding, batch_norm, dropout, spec_norm=self.spec_norm)

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
    img_shape = (1, 512, 512)
    model = UNet(in_ch=1,
                 out_ch=3,
                 depth=4,
                 start_ch=64,
                 inc_rate=2,
                 padding=True,
                 batch_norm=True,
                 spec_norm=False,
                 dropout=0.5,
                 up_mode='upconv',
                 include_top=True)
    A = summary(model, img_shape)
