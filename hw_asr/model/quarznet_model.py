from torch import nn
from typing import List
from math import ceil

from hw_asr.base import BaseModel


def conv_bn(in_channels: int, out_channels: int, *args, **kwargs):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm1d(out_channels)
    )


class ConvBnReLUBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super(ConvBnReLUBlock, self).__init__()

        self.block = nn.Sequential(
            *conv_bn(in_channels, out_channels, *args, **kwargs),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.block(X)


class TCSConvBnBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super(TCSConvBnBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, groups=in_channels, *args, **kwargs),  # depthwise
            *conv_bn(in_channels, out_channels, kernel_size=1, bias=False)  # pointwise + norm
        )

    def forward(self, X):
        return self.block(X)


class MainBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super(MainBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.blocks = nn.ModuleList(
            [TCSConvBnBlock(self._apply_first(i), out_channels, *args, **kwargs) for i in range(5)]
        )

        self.residual_layer = conv_bn(self.in_channels, self.out_channels, kernel_size=1)

    def _apply_first(self, index: int):
        return self.in_channels if index == 0 else self.out_channels

    def forward(self, X):

        residual = self.residual_layer(X)
        for index, unit_block in enumerate(self.blocks):
            X = unit_block(X)
            if index == len(self.blocks) - 1:
                X += residual
            X = nn.ReLU(inplace=True)(X)

        return X


class QuarzNet(BaseModel):
    def __init__(self, n_feats: int, n_class: int, multiplier: int = 1, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        block_sizes = [256, 256, 512, 512, 512, 512]
        kernels = [33, 39, 51, 63, 75, 87]

        # C1 Block: Downsampling, stride = 2, padding = ceil((K - 2) / 2)
        self.conv1 = nn.Sequential(
            TCSConvBnBlock(n_feats, block_sizes[0], kernel_size=kernels[0], stride=2,
                           padding=int(ceil((kernels[0] - 2) / 2)), bias=False),
            nn.ReLU(inplace=True)
        )

        current_in_channels = block_sizes[0]

        # B1-B5 Blocks: each BxR TCSConv-Bn-ReLU with residual connections, B=multiplier, R=5
        # padding maintains separate time channel width
        self.main_blocks = nn.ModuleList()

        for i in range(5):
            for j in range(multiplier):
                self.main_blocks.append(
                    MainBlock(current_in_channels, block_sizes[i], kernel_size=kernels[i],
                              padding=int(ceil((kernels[i] - 1) / 2)), bias=False)
                )
                current_in_channels = block_sizes[i]  # sequentially update in_channels

        # C2 Block
        self.conv2 = nn.Sequential(
            TCSConvBnBlock(current_in_channels, block_sizes[5], kernel_size=kernels[5], dilation=2,
                           padding=kernels[5] - 1, bias=False),
            nn.ReLU(inplace=True)
        )

        current_in_channels = block_sizes[5]

        # C3 Block
        self.conv3 = ConvBnReLUBlock(current_in_channels, current_in_channels * 2, kernel_size=1, bias=False)

        # C4 Block
        self.conv4 = ConvBnReLUBlock(current_in_channels * 2, n_class, kernel_size=1, bias=False)

    def forward(self, X, *args, **kwargs):
        X = self.conv1(X)
        for block in self.main_blocks:
            X = block(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        return {"logits": X}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
