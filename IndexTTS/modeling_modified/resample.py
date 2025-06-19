# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn as nn
from torch.nn import functional as F

from .filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)
        self.x_shape = [768, 384, 192, 96, 48, 24]
        self.filter_pad = [filter.expand(768, -1, -1)]
        self.filter_pad.append(filter.expand(384, -1, -1))
        self.filter_pad.append(filter.expand(192, -1, -1))
        self.filter_pad.append(filter.expand(96, -1, -1))
        self.filter_pad.append(filter.expand(48, -1, -1))
        self.filter_pad.append(filter.expand(24, -1, -1))

        self.pad_zeros = [torch.zeros((1, 768, 5), dtype=torch.float32)]
        self.pad_zeros.append(torch.zeros((1, 384, 5), dtype=torch.float32))
        self.pad_zeros.append(torch.zeros((1, 192, 5), dtype=torch.float32))
        self.pad_zeros.append(torch.zeros((1, 96, 5), dtype=torch.float32))
        self.pad_zeros.append(torch.zeros((1, 48, 5), dtype=torch.float32))
        self.pad_zeros.append(torch.zeros((1, 24, 5), dtype=torch.float32))
        self.pad_zeros.append(torch.zeros((1, 24, 15), dtype=torch.float32))

    # x: [B, C, T]
    def forward(self, x, idx):
        x = torch.cat([self.pad_zeros[idx], x, self.pad_zeros[idx]], dim=-1)
        x = self.ratio * F.conv_transpose1d(x, self.filter_pad[idx], stride=self.stride, groups=self.x_shape[idx])
        x = x[..., self.pad_left:-self.pad_right]
        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x, idx):
        return self.lowpass(x, idx)
