import gc
import sys
from pathlib import Path

import onnx
import torch

script_dir = Path(__file__).resolve().parent
onnx_folder = script_dir / "BigVGAN_ONNX"
onnx_folder.mkdir(parents=True, exist_ok=True)

model_path          = str(Path.home() / "Downloads" / "bigvgan_v2_24khz_100band_256x")  # The BigVGAN project path.    URL: https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x / https://github.com/NVIDIA/BigVGAN
onnx_model_A        = str(onnx_folder / "BigVGAN.onnx")                            # The exported onnx model path.
onnx_model_Metadata = str(onnx_folder / "BigVGAN_Metadata.onnx")                   # Static runtime contract.


# ONNX Runtime Settings
ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
MAX_THREADS = 0                         # Max CPU parallel threads.
DEVICE_ID = 0                           # The GPU id, default to 0.

# Model Parameters
OPSET = 20
DYNAMIC_AXIS = True                     # The default dynamic axis is mel feature length.
USE_TANH = True                         # Option to apply tanh(x) at the final output.
MAX_SIGNAL_LENGTH = 512                 # Max frames for audio length after STFT processed. For static axis setting.
DYNAMIC_TRACE_LENGTH = 4                # Trace exemplar only; the exported time dimension remains dynamic.
OUT_SAMPLE_RATE = 24000                 # Public generated-waveform ONNX output rate.
OUT_AUDIO_DTYPE = "INT16"              # "F16" | "F32" | "INT16".
MODEL_SAMPLE_RATE = 24000               # Native checkpoint sample rate; do not edit.

_OUTPUT_AUDIO_DTYPES = {"F16", "F32", "INT16"}
PATCHED_BIGVGAN_SOURCES = {
    "bigvgan.py": r'''# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import os
import json
from pathlib import Path
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

import activations
from utils import init_weights, get_padding
from alias_free_activation.torch.act import Activation1d as TorchActivation1d
from env import AttrDict

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))


class ExportPeriodicActivation(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        alpha = activation.alpha.detach()
        scale = activation.beta.detach() if isinstance(activation, activations.SnakeBeta) else alpha
        if activation.alpha_logscale:
            alpha = torch.exp(alpha)
            scale = torch.exp(scale)
        self.register_buffer("alpha", alpha.reshape(1, -1, 1).contiguous())
        self.register_buffer(
            "inv_scale",
            torch.reciprocal(scale + activation.no_div_by_zero).reshape(1, -1, 1).contiguous(),
        )

    def forward(self, x):
        periodic = torch.sin(x * self.alpha)
        return x + periodic * periodic * self.inv_scale


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()
        
        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )

    def forward(
        self,
        x,
        first_upsampled,
        channels,
        resample_filter,
        pad_zeros,
        down_pad_zeros_R,
        upsample_scale,
        crop_starts,
        crop_ends,
        crop_axes,
        crop_steps,
    ):
        a1, a2 = self.activations[0], self.activations[1]
        xt = a1.activate_and_downsample(first_upsampled, channels, resample_filter, pad_zeros, down_pad_zeros_R)
        xt = self.convs1[0](xt)
        xt = a2(xt, channels, resample_filter, pad_zeros, down_pad_zeros_R, upsample_scale, crop_starts, crop_ends, crop_axes, crop_steps)
        xt = self.convs2[0](xt)
        x = xt + x

        acts1, acts2 = self.activations[2::2], self.activations[3::2]
        for c1, c2, a1, a2 in zip(self.convs1[1:], self.convs2[1:], acts1, acts2):
            xt = a1(x, channels, resample_filter, pad_zeros, down_pad_zeros_R, upsample_scale, crop_starts, crop_ends, crop_axes, crop_steps)
            xt = c1(xt)
            xt = a2(xt, channels, resample_filter, pad_zeros, down_pad_zeros_R, upsample_scale, crop_starts, crop_ends, crop_axes, crop_steps)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    Unlike AMPBlock1, AMPBlock2 does not contain extra Conv1d layers with fixed dilation=1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        h: AttrDict,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()
        
        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.Snake(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=activations.SnakeBeta(
                            channels, alpha_logscale=h.snake_logscale
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(
    torch.nn.Module,
    PyTorchModelHubMixin,
    library_name="bigvgan",
    repo_url="https://github.com/NVIDIA/BigVGAN",
    docs_url="https://github.com/NVIDIA/BigVGAN/blob/main/README.md",
    pipeline_tag="audio-to-audio",
    license="mit",
    tags=["neural-vocoder", "audio-generation", "arxiv:2206.04658"],
):
    """
    BigVGAN is a neural vocoder model that applies anti-aliased periodic activation for residual blocks (resblocks).
    New in BigVGAN-v2: it can optionally use optimized CUDA kernels for AMP (anti-aliased multi-periodicity) blocks.

    Args:
        h (AttrDict): Hyperparameters.
        use_cuda_kernel (bool): If set to True, loads optimized CUDA kernels for AMP. This should be used for inference only, as training is not supported with CUDA kernels.

    Note:
        - The `use_cuda_kernel` parameter should be used for inference only, as training with CUDA kernels is not supported.
        - Ensure that the activation function is correctly specified in the hyperparameters (h.activation).
    """

    def __init__(self, h: AttrDict, use_cuda_kernel: bool = False):
        super().__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        if h.resblock == "1":
            resblock_class = AMPBlock1
        elif h.resblock == "2":
            resblock_class = AMPBlock2

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_class(h, ch, k, d, activation=h.activation)
                )

        # Post-conv
        activation_post = (
            activations.Snake(ch, alpha_logscale=h.snake_logscale)
            if h.activation == "snake"
            else (
                activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
                if h.activation == "snakebeta"
                else None
            )
        )
        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

        self.register_buffer(
            "inv_num_kernels",
            torch.tensor(1.0 / self.num_kernels, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "upsample_scale",
            torch.tensor(float(self.activation_post.upsample.ratio), dtype=torch.float32),
            persistent=False,
        )
        crop_left = self.activation_post.upsample.pad_left
        crop_right = self.activation_post.upsample.pad_right
        self.register_buffer("crop_starts", torch.tensor([crop_left], dtype=torch.int64), persistent=False)
        self.register_buffer("crop_ends", torch.tensor([-crop_right], dtype=torch.int64), persistent=False)
        self.register_buffer("crop_axes", torch.tensor([2], dtype=torch.int64), persistent=False)
        self.register_buffer("crop_steps", torch.tensor([1], dtype=torch.int64), persistent=False)

        resample_filter = self.activation_post.upsample.filter
        down_filter = self.activation_post.downsample.lowpass.filter

        self.x_shape = []
        for i in range(self.num_upsamples):
            shape = self.ups._modules[f"{i}"]._modules["0"].out_channels
            self.x_shape.append(shape)
            self.register_buffer(f"resample_filter_{i}", resample_filter.expand(shape, -1, -1), persistent=False)
            self.register_buffer(
                f"pad_zeros_{i}",
                torch.zeros((1, shape, self.activation_post.upsample.pad), dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                f"down_pad_zeros_R_{i}",
                torch.zeros((1, shape, self.activation_post.downsample.lowpass.pad_right), dtype=torch.float32),
                persistent=False,
            )
        self.register_buffer("post_pad_zeros", torch.zeros((1, shape, 15), dtype=torch.float32), persistent=False)

    def forward(self, x):
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            channels = self.x_shape[i]
            resample_filter = getattr(self, f"resample_filter_{i}")
            pad_zeros = getattr(self, f"pad_zeros_{i}")
            down_pad_zeros_R = getattr(self, f"down_pad_zeros_R_{i}")
            first_upsampled = self.activation_post.upsample(x, channels, resample_filter, pad_zeros, self.upsample_scale, self.crop_starts, self.crop_ends, self.crop_axes, self.crop_steps)
            xs = self.resblocks[i * self.num_kernels](x, first_upsampled, channels, resample_filter, pad_zeros, down_pad_zeros_R, self.upsample_scale, self.crop_starts, self.crop_ends, self.crop_axes, self.crop_steps)
            for j in range(1, self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x, first_upsampled, channels, resample_filter, pad_zeros, down_pad_zeros_R, self.upsample_scale, self.crop_starts, self.crop_ends, self.crop_axes, self.crop_steps)
            x = xs * self.inv_num_kernels

        # Post-conv
        x = self.activation_post(
            x,
            self.x_shape[-1],
            getattr(self, f"resample_filter_{self.num_upsamples - 1}"),
            self.post_pad_zeros,
            self.post_pad_zeros,
            self.upsample_scale,
            self.crop_starts,
            self.crop_ends,
            self.crop_axes,
            self.crop_steps,
        )
        x = self.conv_post(x)
        # Final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass

    def prepare_for_export(self):
        export_buffer_prefixes = ("resample_filter_", "pad_zeros_", "down_pad_zeros_R_")
        export_buffer_names = {
            "inv_num_kernels",
            "upsample_scale",
            "crop_starts",
            "crop_ends",
            "crop_axes",
            "crop_steps",
            "post_pad_zeros",
        }
        for name in tuple(self._non_persistent_buffers_set):
            if name in export_buffer_names or name.startswith(export_buffer_prefixes):
                self._non_persistent_buffers_set.remove(name)
        activation_modules = [
            module for module in self.modules() if isinstance(module, TorchActivation1d)
        ]
        for module in activation_modules:
            if isinstance(module.act, ExportPeriodicActivation):
                continue
            module.act = ExportPeriodicActivation(module.act)

    # Additional methods for huggingface_hub support
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config.json from a Pytorch model to a local directory."""

        model_path = save_directory / "bigvgan_generator.pt"
        torch.save({"generator": self.state_dict()}, model_path)

        config_path = save_directory / "config.json"
        with open(config_path, "w") as config_file:
            json.dump(self.h, config_file, indent=4)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str,
        cache_dir: str,
        force_download: bool,
        proxies: Optional[Dict] = None,
        resume_download: bool = False,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",  # Additional argument
        strict: bool = False,  # Additional argument
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""

        # Download and load hyperparameters (h) used by BigVGAN
        if os.path.isdir(model_id):
            print("Loading config.json from local directory")
            config_file = os.path.join(model_id, "config.json")
        else:
            config_file = hf_hub_download(
                repo_id=model_id,
                filename="config.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
        h = load_hparams_from_json(config_file)

        # instantiate BigVGAN using h
        if use_cuda_kernel:
            print(
                f"[WARNING] You have specified use_cuda_kernel=True during BigVGAN.from_pretrained(). Only inference is supported (training is not implemented)!"
            )
            print(
                f"[WARNING] You need nvcc and ninja installed in your system that matches your PyTorch build is using to build the kernel. If not, the model will fail to initialize or generate incorrect waveform!"
            )
            print(
                f"[WARNING] For detail, see the official GitHub repository: https://github.com/NVIDIA/BigVGAN?tab=readme-ov-file#using-custom-cuda-kernel-for-synthesis"
            )
        model = cls(h, use_cuda_kernel=use_cuda_kernel)

        # Download and load pretrained generator weight
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, "bigvgan_generator.pt")
        else:
            print(f"Loading weights from {model_id}")
            model_file = hf_hub_download(
                repo_id=model_id,
                filename="bigvgan_generator.pt",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )

        checkpoint_dict = torch.load(model_file, map_location=map_location)

        try:
            model.load_state_dict(checkpoint_dict["generator"])
        except RuntimeError:
            print(
                f"[INFO] the pretrained checkpoint does not contain weight norm. Loading the checkpoint after removing weight norm!"
            )
            model.remove_weight_norm()
            model.load_state_dict(checkpoint_dict["generator"])

        return model
''',
    "alias_free_activation/torch/resample.py": r'''# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch.nn as nn
from torch.nn import functional as F
from alias_free_activation.torch.filter import LowPassFilter1d
from alias_free_activation.torch.filter import kaiser_sinc_filter1d
import torch


class ExportStaticSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, starts, ends, axes, steps):
        return torch.ops.aten.slice.Tensor(x, 2, starts[0], ends[0], steps[0])

    @staticmethod
    def symbolic(g, x, starts, ends, axes, steps):
        return g.op("Slice", x, starts, ends, axes, steps)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = (
            self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        )
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
        )
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x, channels, up_filter, up_pad_zeros, up_scale, crop_starts, crop_ends, crop_axes, crop_steps):
        x = torch.cat([up_pad_zeros, x, up_pad_zeros], dim=-1)
        x = up_scale * F.conv_transpose1d(x, up_filter, stride=self.stride, groups=channels)
        return ExportStaticSlice.apply(x, crop_starts, crop_ends, crop_axes, crop_steps)


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x, channels, down_filter, down_pad_zeros_L, down_pad_zeros_R):
        return self.lowpass(x, channels, down_filter, down_pad_zeros_L, down_pad_zeros_R)
''',
    "alias_free_activation/torch/filter.py": r'''# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
#   LICENSE is in incl_licenses directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

if "sinc" in dir(torch):
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
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


# This code is adopted from adefossez's julius.lowpass.LowPassFilters under the MIT License
# https://adefossez.github.io/julius/julius/lowpass.html
#   LICENSE is in incl_licenses directory.
def kaiser_sinc_filter1d(
    cutoff, half_width, kernel_size
):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        """
        Normalize filter to have sum = 1, otherwise we will have a small leakage of the constant component in the input signal.
        """
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ):
        """
        kernel_size should be even number for stylegan3 setup, in this implementation, odd number is also possible.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    # Input [B, C, T]
    def forward(self, x, channels, down_filter, down_pad_zeros_L, down_pad_zeros_R):
        if self.padding:
            x = torch.cat([down_pad_zeros_L, x, down_pad_zeros_R], dim=-1)
        return F.conv1d(x, down_filter, stride=self.stride, groups=channels)
''',
    "alias_free_activation/torch/act.py": r'''# Adapted from https://github.com/junjun3518/alias-free-torch under the Apache License 2.0
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
    def forward(self, x, channels, resample_filter, pad_zeros, down_pad_zeros_R, upsample_scale, crop_starts, crop_ends, crop_axes, crop_steps):
        x = self.upsample(x, channels, resample_filter, pad_zeros, upsample_scale, crop_starts, crop_ends, crop_axes, crop_steps)
        return self.activate_and_downsample(x, channels, resample_filter, pad_zeros, down_pad_zeros_R)

    def activate_and_downsample(self, x, channels, resample_filter, pad_zeros, down_pad_zeros_R):
        x = self.act(x)
        x = self.downsample(x, channels, resample_filter, pad_zeros, down_pad_zeros_R)
        return x
''',
}


def install_patched_bigvgan_sources(destination):
    destination = Path(destination)
    for relative_path, source in PATCHED_BIGVGAN_SOURCES.items():
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8") == source:
            continue
        target.write_text(source, encoding="utf-8")


install_patched_bigvgan_sources(model_path)


if model_path not in sys.path:
    sys.path.insert(0, model_path)


from bigvgan import BigVGAN


class BIGVGAN(torch.nn.Module):
    def __init__(self, bigvgan, use_tanh):
        super(BIGVGAN, self).__init__()
        self.bigvgan = bigvgan
        self.bigvgan.use_tanh_at_final = use_tanh
        self.output_resample_scale = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)

    def forward(self, mel_features):
        generated_wav = self.bigvgan(mel_features)
        if self.output_resample_scale != 1.0:
            generated_wav = torch.nn.functional.interpolate(
                generated_wav,
                scale_factor=self.output_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            return (generated_wav * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return generated_wav.float()
        return generated_wav.half()


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


with torch.inference_mode():
    model = BigVGAN.from_pretrained(model_path, use_cuda_kernel=False)
    model.remove_weight_norm()  # remove weight norm in the model and set to eval mode
    model.prepare_for_export()
    model = model.eval().to('cpu').float()
    N_MEL = model.conv_pre.in_channels
    model = BIGVGAN(model, USE_TANH).eval()

    print("\nStart to Export the BigVGAN...\n")
    trace_length = DYNAMIC_TRACE_LENGTH if DYNAMIC_AXIS else MAX_SIGNAL_LENGTH
    mel_features = torch.ones((1, N_MEL, trace_length), dtype=torch.float32)
    torch.onnx.export(
        model,
        (mel_features,),
        onnx_model_A,
        input_names=['mel_features'],
        output_names=['generated_wav'],
        dynamic_axes={
            'mel_features': {2: 'mel_features_len'},
            'generated_wav': {2: 'generated_len'}
        } if DYNAMIC_AXIS else None,
        dynamo=False,
        opset_version=OPSET)
    metadata = {
        "graph_layout": "mel_to_waveform",
        "out_sample_rate": str(OUT_SAMPLE_RATE),
        "model_file_name_vocoder": Path(onnx_model_A).name,
    }
    metadata_marker = torch.zeros(1, dtype=torch.int64)
    torch.onnx.export(
        METADATA_CARRIER(),
        (metadata_marker,),
        onnx_model_Metadata,
        input_names=["metadata_marker"],
        output_names=["metadata_marker_out"],
        opset_version=OPSET,
        dynamo=False,
    )
    metadata_model = onnx.load(onnx_model_Metadata, load_external_data=False)
    del metadata_model.metadata_props[:]
    for key, value in metadata.items():
        metadata_model.metadata_props.add(key=key, value=value)
    onnx.save(metadata_model, onnx_model_Metadata)
    del metadata_model, metadata_marker
    del model
    del mel_features
    gc.collect()
    print("\nExport Done.")


if model_path in sys.path:
    sys.path.remove(model_path)

print("\nExport done!")
