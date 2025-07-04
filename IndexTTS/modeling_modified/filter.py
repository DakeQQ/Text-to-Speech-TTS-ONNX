# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    # This code is adopted from adefossez's julius.core.sinc under the MIT License
    # https://adefossez.github.io/julius/julius/core.html
    #   LICENSE is in incl_licenses directory.
    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(x == 0,
                           torch.tensor(1., device=x.device, dtype=x.dtype),
                           torch.sin(math.pi * x) / math.pi / x)


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # return filter [1,1,kernel_size]
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)
        self.x_shape = [768, 384, 192, 96, 48, 24]
        self.filter_pad = [filter.expand(768, -1, -1)]
        self.filter_pad.append(filter.expand(384, -1, -1))
        self.filter_pad.append(filter.expand(192, -1, -1))
        self.filter_pad.append(filter.expand(96, -1, -1))
        self.filter_pad.append(filter.expand(48, -1, -1))
        self.filter_pad.append(filter.expand(24, -1, -1))

        self.pad_zeros_L = [torch.zeros((1, 768, self.pad_left), dtype=torch.float32)]
        self.pad_zeros_L.append(torch.zeros((1, 384, self.pad_left), dtype=torch.float32))
        self.pad_zeros_L.append(torch.zeros((1, 192, self.pad_left), dtype=torch.float32))
        self.pad_zeros_L.append(torch.zeros((1, 96, self.pad_left), dtype=torch.float32))
        self.pad_zeros_L.append(torch.zeros((1, 48, self.pad_left), dtype=torch.float32))
        self.pad_zeros_L.append(torch.zeros((1, 24, self.pad_left), dtype=torch.float32))
        self.pad_zeros_L.append(torch.zeros((1, 24, 15), dtype=torch.float32))

        self.pad_zeros_R = [torch.zeros((1, 768, self.pad_right), dtype=torch.float32)]
        self.pad_zeros_R.append(torch.zeros((1, 384, self.pad_right), dtype=torch.float32))
        self.pad_zeros_R.append(torch.zeros((1, 192, self.pad_right), dtype=torch.float32))
        self.pad_zeros_R.append(torch.zeros((1, 96, self.pad_right), dtype=torch.float32))
        self.pad_zeros_R.append(torch.zeros((1, 48, self.pad_right), dtype=torch.float32))
        self.pad_zeros_R.append(torch.zeros((1, 24, self.pad_right), dtype=torch.float32))
        self.pad_zeros_R.append(torch.zeros((1, 24, 15), dtype=torch.float32))

    #input [B, C, T]
    def forward(self, x, idx):
        if self.padding:
            x = torch.cat([self.pad_zeros_L[idx], x, self.pad_zeros_R[idx]], dim=-1)
        out = F.conv1d(x, self.filter_pad[idx], stride=self.stride, groups=self.x_shape[idx])
        return out
