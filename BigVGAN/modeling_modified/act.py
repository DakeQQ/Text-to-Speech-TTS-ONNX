# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from alias_free_activation.torch.resample import UpSample1d, DownSample1d


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    # x: [B,C,T]
    def forward(self, x, x_shape, up_filter_pad, up_pad_zeros, down_filter_pad, down_pad_zeros_L, down_pad_zeros_R):
        x = self.upsample(x, x_shape, up_filter_pad, up_pad_zeros)
        x = self.act(x)
        x = self.downsample(x, x_shape, down_filter_pad, down_pad_zeros_L, down_pad_zeros_R)
        return x
