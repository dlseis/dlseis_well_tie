"""Base blocks for the neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wtie.utils.types_ import Tensor


_padding_mode = 'zeros' #'replicate' REPLICATE IS BROKEN!


class ConvBnLrelu1d(nn.Module):
    """ """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 padding_mode: str = _padding_mode,
                 inplace: bool = False) -> None:


        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              padding_mode=padding_mode,
                              bias=False)

        self.bn = nn.BatchNorm1d(out_channels)

        self.act = nn.LeakyReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LinearLrelu(nn.Module):
    def __init__(self, in_features: int, out_features: int, inplace: bool = False):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.LeakyReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = self.act(x)
        return x




class DoubleConv1d(nn.Module):
    """ """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 padding_mode: str = _padding_mode,
                 residual: bool = True) -> None:

        super().__init__()
        self.out_channels = out_channels

        self.residual = residual

        ckwargs = dict(kernel_size=kernel_size, padding=padding,
                       padding_mode=padding_mode, inplace=False)


        self.one = ConvBnLrelu1d(in_channels,out_channels,**ckwargs)
        self.two = ConvBnLrelu1d(out_channels,out_channels,**ckwargs)



    def forward(self, x: Tensor) -> Tensor:
        x = self.one(x)
        if self.residual:
            res = x
        x = self.two(x)
        if self.residual:
            x += res
            x /= 2.
        return x


class SingleConv1d(nn.Module):
    """Conv, Bn, LRelu"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 padding_mode: str = _padding_mode) -> None:

        super().__init__()

        ckwargs = dict(kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)


        #self.out_channels = out_channels

        self.one = ConvBnLrelu1d(in_channels,out_channels,**ckwargs)


    def forward(self, x: Tensor) -> Tensor:
        return self.one(x)








class Down1d(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 factor: int,
                 kernel_size: int,
                 padding: int,
                 padding_mode: str = _padding_mode) -> None:

        super().__init__()

        #self.out_channels = out_channels

        self.mp = nn.MaxPool1d(factor)
        self.conv = DoubleConv1d(in_channels, out_channels,
                         kernel_size, padding, padding_mode)


    def forward(self, x: Tensor) -> Tensor:
        x = self.mp(x)
        x = self.conv(x)
        return x



class Up1d(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 factor: int,
                 kernel_size: int,
                 padding: int,
                 padding_mode: str = _padding_mode) -> None:

        super().__init__()

        #self.out_channels = out_channels

        self.up = nn.Upsample(scale_factor=factor, mode='nearest')
        #self.up = nn.Upsample(scale_factor=factor, mode='trilinear', align_corners=True)
        self.conv = DoubleConv1d(in_channels, out_channels, kernel_size,
                                 padding, padding_mode=padding_mode)



    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        x = self.conv(x)
        return x






class OutConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 padding_mode: str = _padding_mode) -> int:
        super().__init__()
        self.out_channels = out_channels

        #self.conv1 = DoubleConv3d(in_channels, out_channels, kernel_size,
                                 #padding, padding_mode=padding_mode)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=1, padding=padding, padding_mode=padding_mode)

    def forward(self, x: Tensor) -> Tensor:
        #x = self.conv1(x)
        x = self.conv(x)
        return x






class MatchSizeToRef1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, ref: Tensor) -> Tensor:
        # input shape is NCL
        diff = ref.size()[2] - x.size()[2]

        x = F.pad(x, [diff // 2, diff - diff // 2])
        # if you have padding issues, see
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x