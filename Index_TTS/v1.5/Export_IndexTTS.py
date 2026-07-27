import gc
import importlib
import math
import shutil
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.
from Shared_Weights import (
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    audit_shared_bundle,
    build_decode_step_graphs,
    bundle_shared_initializers,
)


# User configuration
home_path = Path.home()
project_path = str(home_path / "Downloads" / "index-tts-main")   # IndexTTS project: https://github.com/index-tts/index-tts
models_path = str(home_path / "Downloads" / "IndexTTS-1.5")      # Model folder: https://modelscope.cn/models/IndexTeam/IndexTTS-1.5/files
MAX_SIGNAL_LENGTH = 2048                                         # Maximum sequence length baked into position tables and attention masks.
USE_F16_KV = True                                                # Store the KV cache in float16 to reduce memory bandwidth.
COMPUTE_IN_F32 = False                                           # With float16 KV, upcast attention computation to float32 for accuracy.


# Derived export paths
script_dir = Path(__file__).resolve().parent
onnx_folder = script_dir / "IndexTTS_ONNX"
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
onnx_model_Reference_Preprocess = str(onnx_folder / "IndexTTS_ReferencePreprocess.onnx")
onnx_model_Target_Preprocess    = str(onnx_folder / "IndexTTS_TargetPreprocess.onnx")
onnx_model_Decode_Embed         = str(onnx_folder / "IndexTTS_DecodeEmbed.onnx")
onnx_model_Decoder              = str(onnx_folder / "IndexTTS_Decoder.onnx")
onnx_model_Metadata             = str(onnx_folder / "IndexTTS_Metadata.onnx")
DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")
onnx_model_Main_Prefill = {
    strategy: str(onnx_folder / f"IndexTTS_MainPrefill_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Main_Decode = {
    strategy: str(onnx_folder / f"IndexTTS_MainDecode_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Decode_Step = {
    strategy: str(onnx_folder / f"IndexTTS_DecodeStep_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}


# Audio boundary and fixed model contract
IN_SAMPLE_RATE = 24000
OUT_SAMPLE_RATE = 24000
IN_AUDIO_DTYPE = "F32"                  # "F16" | "F32" | "INT16".
OUT_AUDIO_DTYPE = "F32"                 # "F16" | "F32" | "INT16".
MODEL_SAMPLE_RATE = 24000
STOP_TOKEN = [8193]
N_MELS = 100
NFFT = 1024
HOP_LENGTH = 256
WINDOW_LENGTH = 1024
WINDOW_TYPE = "hann"
OPSET = 20

_AUDIO_DTYPES = {"F16": torch.float16, "F32": torch.float32, "INT16": torch.int16}
if IN_SAMPLE_RATE < 1 or OUT_SAMPLE_RATE < 1:
    raise ValueError("IN_SAMPLE_RATE and OUT_SAMPLE_RATE must be positive.")
if IN_AUDIO_DTYPE.upper() not in _AUDIO_DTYPES:
    raise ValueError(f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; expected one of {tuple(_AUDIO_DTYPES)}.")
if OUT_AUDIO_DTYPE.upper() not in _AUDIO_DTYPES:
    raise ValueError(f"Unsupported OUT_AUDIO_DTYPE={OUT_AUDIO_DTYPE!r}; expected one of {tuple(_AUDIO_DTYPES)}.")

# Representative dynamic-control inputs used while tracing the strategy graphs.
PENALTY_VALUE = 0.8
PENALTY_RANGE = 10
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_K = 50
SAMPLING_TOP_P = 0.95
SAMPLING_REPETITION_PENALTY = 1.1


if project_path not in sys.path:
    sys.path.append(project_path)


_BIGVGAN_EXPORT_CHANNELS = (768, 384, 192, 96, 48, 24)


if 'sinc' in dir(torch):
    sinc = torch.sinc
else:
    def sinc(x: torch.Tensor):
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    a_value = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if a_value > 50.0:
        beta = 0.1102 * (a_value - 8.7)
    elif a_value >= 21.0:
        beta = 0.5842 * (a_value - 21.0) ** 0.4 + 0.07886 * (a_value - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_kernel = torch.zeros_like(time)
    else:
        filter_kernel = 2 * cutoff * window * sinc(2 * cutoff * time)
        filter_kernel /= filter_kernel.sum()
    return filter_kernel.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12):
        super().__init__()
        if cutoff < -0.0:
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
        filter_kernel = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter_kernel)
        self.x_shape = list(_BIGVGAN_EXPORT_CHANNELS)
        self.filter_pad = [filter_kernel.expand(channel, -1, -1) for channel in _BIGVGAN_EXPORT_CHANNELS]
        self.pad_zeros_L = [torch.zeros((1, channel, self.pad_left), dtype=torch.float32) for channel in _BIGVGAN_EXPORT_CHANNELS]
        self.pad_zeros_R = [torch.zeros((1, channel, self.pad_right), dtype=torch.float32) for channel in _BIGVGAN_EXPORT_CHANNELS]
        self.pad_zeros_L.append(torch.zeros((1, 24, 15), dtype=torch.float32))
        self.pad_zeros_R.append(torch.zeros((1, 24, 15), dtype=torch.float32))

    def forward(self, x, idx):
        if self.padding:
            x = torch.cat([self.pad_zeros_L[idx], x, self.pad_zeros_R[idx]], dim=-1)
        return F.conv1d(x, self.filter_pad[idx], stride=self.stride, groups=self.x_shape[idx])


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter_kernel = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                             half_width=0.6 / ratio,
                             kernel_size=self.kernel_size)
        self.register_buffer("filter", filter_kernel)
        self.x_shape = list(_BIGVGAN_EXPORT_CHANNELS)
        self.filter_pad = [filter_kernel.expand(channel, -1, -1) for channel in _BIGVGAN_EXPORT_CHANNELS]
        self.pad_zeros = [torch.zeros((1, channel, 5), dtype=torch.float32) for channel in _BIGVGAN_EXPORT_CHANNELS]
        self.pad_zeros.append(torch.zeros((1, 24, 15), dtype=torch.float32))
        self.scale_folded = False

    def forward(self, x, idx):
        x = torch.cat([self.pad_zeros[idx], x, self.pad_zeros[idx]], dim=-1)
        x = F.conv_transpose1d(x, self.filter_pad[idx], stride=self.stride, groups=self.x_shape[idx])
        if not self.scale_folded:
            x = self.ratio * x
        return x[..., self.pad_left:-self.pad_right]


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


class Activation1d(nn.Module):
    def __init__(self,
                 activation,
                 up_ratio: int = 2,
                 down_ratio: int = 2,
                 up_kernel_size: int = 12,
                 down_kernel_size: int = 12):
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x, idx):
        x = self.upsample(x, idx)
        x = self.act(x)
        return self.downsample(x, idx)


class FrozenSnakeActivation(nn.Module):
    def __init__(self, activation):
        super().__init__()
        alpha = torch.exp(activation.alpha) if activation.alpha_logscale else activation.alpha
        denominator = activation.beta if hasattr(activation, "beta") else activation.alpha
        denominator = torch.exp(denominator) if activation.alpha_logscale else denominator
        self.register_buffer("alpha", alpha.detach().view(1, -1, 1))
        self.register_buffer(
            "inv_denominator",
            torch.reciprocal(denominator.detach() + activation.no_div_by_zero).view(1, -1, 1),
        )

    def forward(self, x):
        return x + self.inv_denominator * torch.sin(x * self.alpha).square()


def share_bigvgan_resample_buffers(owner, activation_modules):
    if not activation_modules:
        raise RuntimeError("BigVGAN has no exportable Activation1d modules.")
    if any(module.upsample.scale_folded for module in activation_modules):
        raise RuntimeError("BigVGAN upsample scale was already folded.")
    upsample_ratio = activation_modules[0].upsample.ratio
    if any(module.upsample.ratio != upsample_ratio for module in activation_modules[1:]):
        raise RuntimeError("BigVGAN upsample ratio mismatch.")

    def get_groups(module):
        return {
            "up_filter": module.upsample.filter_pad,
            "up_padding": module.upsample.pad_zeros,
            "down_filter": module.downsample.lowpass.filter_pad,
            "down_left": module.downsample.lowpass.pad_zeros_L,
            "down_right": module.downsample.lowpass.pad_zeros_R,
        }

    template_groups = get_groups(activation_modules[0])
    for module in activation_modules[1:]:
        candidate_groups = get_groups(module)
        for group_name, reference_tensors in template_groups.items():
            candidate_tensors = candidate_groups[group_name]
            if len(candidate_tensors) != len(reference_tensors):
                raise RuntimeError(f"BigVGAN resample buffer count mismatch for {group_name}.")
            if any(not torch.equal(candidate, reference) for candidate, reference in zip(candidate_tensors, reference_tensors)):
                raise RuntimeError(f"BigVGAN resample buffer value mismatch for {group_name}.")

    shared_groups = {}
    for group_name, tensors in template_groups.items():
        shared_tensors = []
        for index, tensor in enumerate(tensors):
            buffer_name = f"resample_{group_name}_{index}"
            if group_name == "up_filter":
                tensor = tensor * upsample_ratio
            owner.register_buffer(buffer_name, tensor)
            shared_tensors.append(getattr(owner, buffer_name))
        shared_groups[group_name] = shared_tensors

    for module in activation_modules:
        module.upsample.filter_pad = shared_groups["up_filter"]
        module.upsample.pad_zeros = shared_groups["up_padding"]
        module.downsample.lowpass.filter_pad = shared_groups["down_filter"]
        module.downsample.lowpass.pad_zeros_L = shared_groups["down_left"]
        module.downsample.lowpass.pad_zeros_R = shared_groups["down_right"]
        module.upsample.scale_folded = True


def _bigvgan_activation_class(use_cuda_kernel):
    if use_cuda_kernel:
        from indextts.BigVGAN.alias_free_activation.cuda.activation1d import Activation1d as CudaActivation1d
        return CudaActivation1d
    return Activation1d


class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        from indextts.BigVGAN.utils import get_padding, init_weights
        import indextts.BigVGAN.activations as activations

        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)
        activation_cls = _bigvgan_activation_class(self.h.get("use_cuda_kernel", False))
        if activation == 'snake':
            self.activations = nn.ModuleList([
                activation_cls(activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta':
            self.activations = nn.ModuleList([
                activation_cls(activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x, idx):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x, idx)
            xt = c1(xt)
            xt = a2(xt, idx)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        from indextts.BigVGAN.utils import get_padding, init_weights
        import indextts.BigVGAN.activations as activations

        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)
        activation_cls = _bigvgan_activation_class(self.h.get("use_cuda_kernel", False))
        if activation == 'snake':
            self.activations = nn.ModuleList([
                activation_cls(activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta':
            self.activations = nn.ModuleList([
                activation_cls(activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x, idx=0):
        for conv, activation in zip(self.convs, self.activations):
            xt = activation(x, idx)
            xt = conv(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class BigVGAN(torch.nn.Module):
    def __init__(self, h, use_cuda_kernel=False):
        super(BigVGAN, self).__init__()
        from indextts.BigVGAN.ECAPA_TDNN import ECAPA_TDNN
        from indextts.BigVGAN.utils import init_weights
        import indextts.BigVGAN.activations as activations

        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.inv_num_kernels = float(1.0 / self.num_kernels)
        self.num_upsamples = len(h.upsample_rates)
        self.feat_upsample = h.feat_upsample
        self.cond_in_each_up_layer = h.cond_d_vector_in_each_upsampling_layer

        self.conv_pre = weight_norm(Conv1d(h.gpt_dim, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = AMPBlock1 if h.resblock == "1" else AMPBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(self.h, ch, k, d, activation=h.activation))

        activation_cls = _bigvgan_activation_class(use_cuda_kernel)
        if h.activation == "snake":
            activation_post = activations.Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = activation_cls(activation=activation_post)
        elif h.activation == "snakebeta":
            activation_post = activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = activation_cls(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        self.speaker_encoder = ECAPA_TDNN(h.num_mels, lin_neurons=h.speaker_embedding_dim)
        self.cond_layer = nn.Conv1d(h.speaker_embedding_dim, h.upsample_initial_channel, 1)
        if self.cond_in_each_up_layer:
            self.conds = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = h.upsample_initial_channel // (2 ** (i + 1))
                self.conds.append(nn.Conv1d(h.speaker_embedding_dim, ch, 1))

    def forward(self, x, mel_ref, lens=None):
        speaker_embedding = self.speaker_encoder(mel_ref, lens)
        n_batch = x.size(0)
        contrastive_loss = None
        if n_batch * 2 == speaker_embedding.size(0):
            spe_emb_chunk1 = speaker_embedding[:n_batch, :, :]
            spe_emb_chunk2 = speaker_embedding[n_batch:, :, :]
            contrastive_loss = self.cal_clip_loss(spe_emb_chunk1.squeeze(1), spe_emb_chunk2.squeeze(1), self.logit_scale.exp())
            speaker_embedding = speaker_embedding[:n_batch, :, :]
        speaker_embedding = speaker_embedding.transpose(1, 2)

        if self.feat_upsample:
            x = F.interpolate(x.transpose(1, 2), scale_factor=[4], mode="linear").squeeze(1)
        else:
            x = x.transpose(1, 2)

        x = self.conv_pre(x)
        x = x + self.cond_layer(speaker_embedding)

        for i in range(self.num_upsamples):
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)

            if self.cond_in_each_up_layer:
                x = x + self.conds[i](speaker_embedding)

            xs = None
            for j in range(self.num_kernels):
                resblock_out = self.resblocks[i * self.num_kernels + j](x, i)
                if xs is None:
                    xs = resblock_out
                else:
                    xs += resblock_out
            x = xs * self.inv_num_kernels

        x = self.activation_post(x, -1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x, contrastive_loss

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for layers in self.ups:
            for layer in layers:
                remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    def cal_clip_loss(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text


def _register_inline_module(module_name, symbols):
    module = types.ModuleType(module_name)
    module.__dict__.update(symbols)
    module.__package__ = module_name.rpartition(".")[0]
    sys.modules[module_name] = module
    parent_name, _, attribute_name = module_name.rpartition(".")
    parent_module = importlib.import_module(parent_name)
    setattr(parent_module, attribute_name, module)
    return module


def _assert_device_map(device_map, num_blocks):
    assigned_blocks = [block for blocks in device_map.values() for block in blocks]
    duplicates = sorted({block for block in assigned_blocks if assigned_blocks.count(block) > 1})
    missing = sorted(set(range(num_blocks)) - set(assigned_blocks))
    extra = sorted(set(assigned_blocks) - set(range(num_blocks)))
    if duplicates or missing or extra:
        raise ValueError(
            "Invalid device map: "
            f"duplicate blocks={duplicates}, missing blocks={missing}, extra blocks={extra}."
        )


def _get_device_map(num_blocks, devices):
    devices = list(devices)
    if not devices:
        return {"cpu": list(range(num_blocks))}
    blocks_per_device = math.ceil(num_blocks / len(devices))
    return {
        device: list(range(start, min(start + blocks_per_device, num_blocks)))
        for device_index, device in enumerate(devices)
        if (start := device_index * blocks_per_device) < num_blocks
    }


def _install_transformers_compatibility_modules():
    """Bypass IndexTTS's version-pinned Transformers copies during export."""
    from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel

    model_parallel_module = "transformers.utils.model_parallel_utils"
    try:
        importlib.import_module(model_parallel_module)
    except ModuleNotFoundError as error:
        if error.name != model_parallel_module:
            raise
        _register_inline_module(
            model_parallel_module,
            {
                "assert_device_map": _assert_device_map,
                "get_device_map": _get_device_map,
            },
        )

    _register_inline_module(
        "indextts.gpt.transformers_gpt2",
        {
            "GPT2Model": GPT2Model,
            "GPT2PreTrainedModel": GPT2PreTrainedModel,
        },
    )


def _install_inline_bigvgan_modules():
    _register_inline_module(
        "indextts.BigVGAN.alias_free_torch.filter",
        {
            "math": math,
            "torch": torch,
            "nn": nn,
            "F": F,
            "sinc": sinc,
            "kaiser_sinc_filter1d": kaiser_sinc_filter1d,
            "LowPassFilter1d": LowPassFilter1d,
        },
    )
    _register_inline_module(
        "indextts.BigVGAN.alias_free_torch.resample",
        {
            "torch": torch,
            "nn": nn,
            "F": F,
            "LowPassFilter1d": LowPassFilter1d,
            "kaiser_sinc_filter1d": kaiser_sinc_filter1d,
            "UpSample1d": UpSample1d,
            "DownSample1d": DownSample1d,
        },
    )
    _register_inline_module(
        "indextts.BigVGAN.alias_free_torch.act",
        {
            "nn": nn,
            "DownSample1d": DownSample1d,
            "UpSample1d": UpSample1d,
            "Activation1d": Activation1d,
        },
    )

    alias_package = importlib.import_module("indextts.BigVGAN.alias_free_torch")
    alias_package.LowPassFilter1d = LowPassFilter1d
    alias_package.kaiser_sinc_filter1d = kaiser_sinc_filter1d
    alias_package.UpSample1d = UpSample1d
    alias_package.DownSample1d = DownSample1d
    alias_package.Activation1d = Activation1d

    models_module = _register_inline_module(
        "indextts.BigVGAN.models",
        {
            "torch": torch,
            "nn": nn,
            "F": F,
            "Conv1d": Conv1d,
            "ConvTranspose1d": ConvTranspose1d,
            "remove_weight_norm": remove_weight_norm,
            "weight_norm": weight_norm,
            "AMPBlock1": AMPBlock1,
            "AMPBlock2": AMPBlock2,
            "BigVGAN": BigVGAN,
        },
    )
    bigvgan_package = importlib.import_module("indextts.BigVGAN")
    bigvgan_package.models = models_module
    bigvgan_package.BigVGAN = BigVGAN


_install_transformers_compatibility_modules()
_install_inline_bigvgan_modules()


def _compute_statistics(x, m, dim=2):
    mean = (m * x).sum(dim, keepdim=True)
    centered = x - mean
    std = torch.sqrt((m * (centered * centered)).sum(dim, keepdim=True).clamp(1e-6))
    return mean, std


def rel_shift(x):
    x_padded = F.pad(x, (1, 0))
    return x_padded.reshape(x.shape[0], x.shape[1] + 1, x.shape[2])[:, 1:]


class IndexTTS_Encoder(torch.nn.Module):
    def __init__(self, indexTTS, custom_stft, nfft, n_mels, sample_rate, max_signal_len):
        super(IndexTTS_Encoder, self).__init__()
        self.bigvgan = indexTTS.bigvgan.eval()
        self.indexTTS = indexTTS.gpt.eval()
        self.custom_stft = custom_stft
        if getattr(self.custom_stft, "input_scale_folded", False):
            raise RuntimeError("STFT input scale was already folded.")
        self.input_resample_scale = float(sample_rate / IN_SAMPLE_RATE)
        if "int" in IN_AUDIO_DTYPE.lower():
            self.custom_stft.stft_kernel.mul_(float(1.0 / 32768.0))
            self.custom_stft.input_scale_folded = True
        self.register_buffer(
            "fbank",
            torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, None, 'htk').transpose(0, 1).unsqueeze(0),
        )
        asp_channels = self.bigvgan.speaker_encoder.channels[-1]
        asp_conv = self.bigvgan.speaker_encoder.asp.tdnn.conv.conv
        self.register_buffer("asp_feature_weight", asp_conv.weight.data[:, :asp_channels].contiguous())
        self.register_buffer("asp_statistics_weight", asp_conv.weight.data[:, asp_channels:].contiguous())
        self.register_buffer("asp_bias", asp_conv.bias.data.contiguous())
        self.indexTTS.conditioning_encoder.embed.pos_enc.pe = self.indexTTS.conditioning_encoder.embed.pos_enc.pe[:, :max_signal_len].half()
        self.indexTTS.conditioning_encoder.embed.out._modules['0'].weight.data *= self.indexTTS.conditioning_encoder.embed.pos_enc.xscale
        self.indexTTS.conditioning_encoder.embed.out._modules['0'].bias.data *= self.indexTTS.conditioning_encoder.embed.pos_enc.xscale
        self.perceiver_encoder_head = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].heads
        self.perceiver_encoder_head_dim = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.out_features // self.perceiver_encoder_head
        self.register_buffer("latents", self.indexTTS.perceiver_encoder.latents.data.unsqueeze(0))
        num_heads = self.indexTTS.conditioning_encoder.encoders._modules['0'].self_attn.h
        head_dim = self.indexTTS.conditioning_encoder.encoders._modules['0'].self_attn.d_k
        hidden_size = self.indexTTS.conditioning_encoder.encoders._modules['0'].self_attn.linear_q.in_features
        scaling = float(head_dim ** -0.25)
        for layer in self.indexTTS.conditioning_encoder.encoders:
            qkv_weight = torch.cat((
                layer.self_attn.linear_q.weight.data * scaling,
                layer.self_attn.linear_k.weight.data * scaling,
                layer.self_attn.linear_v.weight.data,
            ), dim=0).view(3, num_heads, head_dim, hidden_size).permute(1, 3, 0, 2).reshape(num_heads, hidden_size, 3 * head_dim).contiguous()
            qkv_bias = torch.cat((
                layer.self_attn.linear_q.bias.data * scaling,
                layer.self_attn.linear_k.bias.data * scaling,
                layer.self_attn.linear_v.bias.data,
            ), dim=0).view(3, num_heads, head_dim).permute(1, 0, 2).reshape(num_heads, 1, 3 * head_dim).contiguous()
            layer.self_attn.register_buffer("qkv_weight", qkv_weight)
            layer.self_attn.register_buffer("qkv_bias", qkv_bias)
            layer.self_attn.register_buffer(
                "position_weight",
                (layer.self_attn.linear_pos.weight.data * scaling).view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous(),
            )
            layer.self_attn.register_buffer(
                "position_bias_u",
                layer.self_attn.pos_bias_u.data.unsqueeze(1) * scaling,
            )
            layer.self_attn.register_buffer(
                "position_bias_v",
                layer.self_attn.pos_bias_v.data.unsqueeze(1) * scaling,
            )
            layer.self_attn.register_buffer(
                "out_proj_weight",
                layer.self_attn.linear_out.weight.data.view(hidden_size, num_heads, head_dim).permute(1, 2, 0).contiguous(),
            )
            layer.self_attn.register_buffer(
                "out_proj_bias",
                layer.self_attn.linear_out.bias.data.view(1, 1, -1).contiguous(),
            )

        num_heads = self.perceiver_encoder_head
        head_dim = self.perceiver_encoder_head_dim
        hidden_size = self.indexTTS.perceiver_encoder.layers._modules['0']._modules['0'].to_q.in_features
        scaling = float(head_dim ** -0.25)
        for attn, _ in self.indexTTS.perceiver_encoder.layers:
            attn.register_buffer(
                "query_weight",
                (attn.to_q.weight.data * scaling).view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous(),
            )
            attn.register_buffer(
                "key_weight",
                (attn.to_kv.weight.data[:attn.to_q.out_features] * scaling).view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous(),
            )
            attn.register_buffer(
                "value_weight",
                attn.to_kv.weight.data[attn.to_q.out_features:].view(num_heads, head_dim, hidden_size).transpose(1, 2).contiguous(),
            )
            attn.register_buffer(
                "out_proj_weight",
                attn.to_out.weight.data.view(hidden_size, num_heads, head_dim).permute(1, 2, 0).contiguous(),
            )

    def forward(self, audio: torch.ShortTensor):
        audio = audio.float()
        if self.input_resample_scale != 1.0:
            audio = F.interpolate(
                audio,
                scale_factor=self.input_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        real_part, imag_part = self.custom_stft(audio)
        mel_signal = torch.matmul(self.fbank, torch.sqrt(real_part * real_part + imag_part * imag_part)).clamp(min=1e-7).log()
        x = self.indexTTS.conditioning_encoder.embed.conv(mel_signal.transpose(1, 2).unsqueeze(1))
        enc_len = x.shape[2].unsqueeze(0)
        x = self.indexTTS.conditioning_encoder.embed.out(x.transpose(1, 2).contiguous().view(1, enc_len, -1))
        pos_emb = self.indexTTS.conditioning_encoder.embed.pos_enc.pe[:, :enc_len].float()
        for encoder_layer in self.indexTTS.conditioning_encoder.encoders:
            x1 = encoder_layer.norm_mha(x)
            qkv = torch.matmul(x1, encoder_layer.self_attn.qkv_weight) + encoder_layer.self_attn.qkv_bias
            q, k, v = torch.split(qkv, encoder_layer.self_attn.d_k, dim=-1)
            k = k.transpose(1, 2)
            p = torch.matmul(pos_emb, encoder_layer.self_attn.position_weight).transpose(1, 2)
            q_with_bias_u = q + encoder_layer.self_attn.position_bias_u
            q_with_bias_v = q + encoder_layer.self_attn.position_bias_v
            matrix_ac = torch.matmul(q_with_bias_u, k)
            matrix_bd = torch.matmul(q_with_bias_v, p)
            matrix_bd = rel_shift(matrix_bd)
            attn_out = torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v)
            attn_out = torch.matmul(attn_out, encoder_layer.self_attn.out_proj_weight).sum(dim=0, keepdim=True) + encoder_layer.self_attn.out_proj_bias
            x += attn_out
            residual = x
            x = encoder_layer.norm_conv(x).transpose(1, 2)
            x = encoder_layer.conv_module.pointwise_conv1(x)
            x = torch.nn.functional.glu(x, dim=1)
            x = encoder_layer.conv_module.depthwise_conv(x).transpose(1, 2)
            x = encoder_layer.conv_module.activation(encoder_layer.conv_module.norm(x)).transpose(1, 2)
            x = encoder_layer.conv_module.pointwise_conv2(x).transpose(1, 2)
            x += residual
            x = x + encoder_layer.feed_forward(encoder_layer.norm_ff(x))
            x = encoder_layer.norm_final(x)
        x = self.indexTTS.conditioning_encoder.after_norm(x)
        x = self.indexTTS.perceiver_encoder.proj_context(x)
        latents = self.latents
        for attn, ff in self.indexTTS.perceiver_encoder.layers:
            q = torch.matmul(latents, attn.query_weight)
            cat_latent_x = torch.cat([latents, x], dim=1)
            k = torch.matmul(cat_latent_x, attn.key_weight).transpose(1, 2)
            v = torch.matmul(cat_latent_x, attn.value_weight)
            attn_out = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v)
            attn_out = torch.matmul(attn_out, attn.out_proj_weight).sum(dim=0, keepdim=True)
            latents = attn_out + latents
            latents = ff(latents) + latents
        conds_latent = self.indexTTS.perceiver_encoder.norm(latents)

        # bigvgan part
        ref_signal_len = mel_signal.shape[-1].unsqueeze(0)
        speaker_embedding = []
        for i, layer in enumerate(self.bigvgan.speaker_encoder.blocks):
            mel_signal = layer(mel_signal)
            if i > 0:
                speaker_embedding.append(mel_signal)
        speaker_embedding = torch.cat(speaker_embedding, dim=1)
        speaker_embedding = self.bigvgan.speaker_encoder.mfa(speaker_embedding)
        mean, std = _compute_statistics(speaker_embedding, 1.0 / ref_signal_len)
        statistics = torch.cat((mean, std), dim=1)
        attn = F.conv1d(speaker_embedding, self.asp_feature_weight, self.asp_bias)
        attn = attn + F.conv1d(statistics, self.asp_statistics_weight)
        attn = self.bigvgan.speaker_encoder.asp.tdnn.norm(self.bigvgan.speaker_encoder.asp.tdnn.activation(attn))
        attn = self.bigvgan.speaker_encoder.asp.conv(self.bigvgan.speaker_encoder.asp.tanh(attn))
        attn = torch.nn.functional.softmax(attn, dim=2)
        mean, std = _compute_statistics(speaker_embedding, attn)
        speaker_embedding = torch.cat((mean, std), dim=1)
        speaker_embedding = self.bigvgan.speaker_encoder.asp_bn(speaker_embedding)
        speaker_embedding = self.bigvgan.speaker_encoder.fc(speaker_embedding)
        bigvgan_cond_layer_speaker_embedding = self.bigvgan.cond_layer(speaker_embedding)
        save_bigvgan_conds = []
        for i in range(self.bigvgan.num_upsamples):
            save_bigvgan_conds.append(self.bigvgan.conds[i](speaker_embedding))
        return *save_bigvgan_conds, bigvgan_cond_layer_speaker_embedding, conds_latent


class IndexTTS_Target_Preprocess(torch.nn.Module):
    def __init__(self, indexTTS):
        super(IndexTTS_Target_Preprocess, self).__init__()
        self.indexTTS = indexTTS.gpt.eval()
        self.register_buffer("start_ids", torch.tensor([[0]], dtype=torch.int32))
        self.register_buffer("end_ids", torch.tensor([[1]], dtype=torch.int32))
        self.register_buffer(
            "start_mel_ids",
            torch.tensor([[self.indexTTS.start_mel_token]], dtype=torch.int32),
        )

    def forward(self, conds_latent, text_ids):
        text_ids = torch.cat([self.start_ids, text_ids, self.end_ids], dim=-1)
        text_ids_len = torch._shape_as_tensor(text_ids)[1:2]
        text_emb = self.indexTTS.text_embedding(text_ids) + self.indexTTS.text_pos_embedding.emb.weight[:text_ids_len]
        gpt_hidden = self.indexTTS.inference_model.embeddings(self.start_mel_ids)
        gpt_hidden = gpt_hidden + self.indexTTS.inference_model.text_pos_embedding.emb.weight[:1]
        return torch.cat([conds_latent, text_emb, gpt_hidden], dim=1)


class IndexTTS_Decode_Embed(torch.nn.Module):
    def __init__(self, indexTTS):
        super(IndexTTS_Decode_Embed, self).__init__()
        self.indexTTS = indexTTS.gpt.eval()

    def forward(self, current_token, save_ids_in):
        generated_len = torch._shape_as_tensor(save_ids_in)[1:2]
        hidden_states = self.indexTTS.inference_model.embeddings(current_token)
        hidden_states = hidden_states + self.indexTTS.inference_model.text_pos_embedding.emb.weight[generated_len]
        return hidden_states


class IndexTTS_Main(torch.nn.Module):
    def __init__(self, indexTTS, num_layers, max_seq_len):
        super(IndexTTS_Main, self).__init__()
        self.indexTTS = indexTTS.gpt.eval()
        self.num_layers = num_layers
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        # When True (and USE_F16_KV): keep the f16 KV storage but run the attention matmuls in f32.
        self.compute_in_f32 = COMPUTE_IN_F32
        # Mask dtype tracks the attention-compute dtype: float16 only for the minimum-cast f16 KV attention
        # (f16 KV computing in f16); float32 otherwise (including compute_in_f32).
        self.register_buffer(
            "attention_mask",
            ((1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128),
        )

        num_heads = self.indexTTS.inference_model.transformer.h._modules['0'].attn.num_heads
        head_dim = self.indexTTS.inference_model.transformer.h._modules['0'].attn.head_dim
        hidden_size = self.indexTTS.inference_model.transformer.h._modules['0'].attn.embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        scaling = float(head_dim ** -0.25)
        for layer in self.indexTTS.inference_model.transformer.h:
            qkv_weight = layer.attn.c_attn.weight.data.clone()
            qkv_bias = layer.attn.c_attn.bias.data.clone()
            qkv_weight[:, :2 * hidden_size] *= scaling
            qkv_bias[:2 * hidden_size] *= scaling
            layer.attn.register_buffer("qkv_weight", qkv_weight)
            layer.attn.register_buffer("qkv_bias", qkv_bias)
            layer.attn.register_buffer(
                "out_proj_weight",
                layer.attn.c_proj.weight.data.view(num_heads, head_dim, hidden_size).contiguous(),
            )
            layer.attn.register_buffer(
                "out_proj_bias",
                layer.attn.c_proj.bias.data.view(1, 1, -1).contiguous(),
            )
        mel_head = self.indexTTS.inference_model.lm_head._modules['1']
        self.register_buffer("mel_head_weight", mel_head.weight.data.transpose(0, 1).contiguous())
        self.register_buffer("mel_head_bias", mel_head.bias.data.clone())

    def forward(self, *all_inputs):
        hidden_state = all_inputs[-2]                                  # (batch, ids_len, hidden_size)
        history_len = all_inputs[-1]
        ids_len = torch._shape_as_tensor(hidden_state)[1:2]
        kv_seq_len = history_len + ids_len
        attention_mask = self.attention_mask[:, :, history_len:kv_seq_len, :kv_seq_len]
        if USE_F16_KV and not self.compute_in_f32:
            attention_mask = attention_mask.half()
        else:
            attention_mask = attention_mask.float()

        for i, layer in enumerate(self.indexTTS.inference_model.transformer.h):
            hidden_states_norm = layer.ln_1(hidden_state)
            qkv = torch.matmul(hidden_states_norm, layer.attn.qkv_weight) + layer.attn.qkv_bias
            qkv = qkv.reshape(1, -1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(0, 3, 1, 2, 4).reshape(1, self.num_heads, -1, 3 * self.head_dim)

            if USE_F16_KV and not self.compute_in_f32:
                qkv = qkv.half()  # one cast covers q, k, and v for f16 attention and cache storage
            q, k, v = torch.split(qkv, layer.attn.head_dim, dim=-1)
            k = k.transpose(-1, -2)
            if USE_F16_KV and self.compute_in_f32:
                k = k.half()       # store-only cast — keeps the growing KV cache in f16
                v = v.half()

            k = torch.cat((all_inputs[i], k), dim=-1)                   # (batch, num_heads, head_dim, kv_seq_len)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)  # (batch, num_heads, kv_seq_len, head_dim)
            self.save_key[i] = k
            self.save_value[i] = v
            if USE_F16_KV and self.compute_in_f32:
                # f16 KV storage, f32 compute: upcast the cache at the matmul use points.
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1), v.float())  # (batch, num_heads, ids_len, head_dim)
            else:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v)  # (batch, num_heads, ids_len, head_dim)
                if USE_F16_KV:
                    hidden_state_attn = hidden_state_attn.float()
            hidden_state_attn = torch.matmul(hidden_state_attn, layer.attn.out_proj_weight).sum(dim=1) + layer.attn.out_proj_bias
            hidden_state = hidden_state + hidden_state_attn
            ffn = torch.matmul(layer.ln_2(hidden_state), layer.mlp.c_fc.weight) + layer.mlp.c_fc.bias
            ffn = layer.mlp.act(ffn)
            ffn = torch.matmul(ffn, layer.mlp.c_proj.weight) + layer.mlp.c_proj.bias
            hidden_state = hidden_state + ffn
        last_hidden_state = self.indexTTS.inference_model.transformer.ln_f(hidden_state[:, -1])  # (batch, hidden_size)
        logits_hidden_state = self.indexTTS.inference_model.lm_head._modules['0'](last_hidden_state)
        logits = torch.matmul(logits_hidden_state, self.mel_head_weight) + self.mel_head_bias
        return *self.save_key, *self.save_value, last_hidden_state, logits, kv_seq_len


class IndexTTS_Decoder(torch.nn.Module):
    def __init__(self, indexTTS):
        super(IndexTTS_Decoder, self).__init__()
        self.gpt = indexTTS.gpt.eval()
        self.bigvgan = indexTTS.bigvgan.eval()
        activation_modules = [module for module in self.bigvgan.modules() if isinstance(module, Activation1d)]
        for module in activation_modules:
            module.act = FrozenSnakeActivation(module.act)
        share_bigvgan_resample_buffers(self, activation_modules)
        self.inv_num_kernels = float(1.0 / self.bigvgan.num_kernels)
        self.output_resample_scale = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)

    def forward(self, *all_inputs):
        latent = self.gpt.final_norm(all_inputs[-1].unsqueeze(0))
        latent = self.bigvgan.conv_pre(latent.transpose(1, 2)) + all_inputs[-2]
        for i in range(self.bigvgan.num_upsamples):
            for i_up in range(len(self.bigvgan.ups[i])):
                latent = self.bigvgan.ups[i][i_up](latent)
            if self.bigvgan.cond_in_each_up_layer:
                latent = latent + all_inputs[i]
            x = self.bigvgan.resblocks[i * self.bigvgan.num_kernels](latent, i)
            for j in range(1, self.bigvgan.num_kernels):
                x = x + self.bigvgan.resblocks[i * self.bigvgan.num_kernels + j](latent, i)
            latent = x * self.inv_num_kernels
        latent = self.bigvgan.conv_post(self.bigvgan.activation_post(latent, -1))
        generated_wav = torch.tanh(latent)
        if self.output_resample_scale != 1.0:
            generated_wav = F.interpolate(
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


# ─────────────────────────────────────────────────────────────────────────────
# Decoding strategy modules. Each final graph contains exactly one strategy, so
# inference loads only the selected prefill/decode pair.
# ─────────────────────────────────────────────────────────────────────────────
class APPLY_PENALTY(torch.nn.Module):
    """Apply a repetition penalty over the most recent ``penalty_range`` tokens."""

    def __init__(self):
        super(APPLY_PENALTY, self).__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized      = logits.gather(1, target_indices) * penalty_value
        logits         = logits.scatter(1, target_indices, penalized)
        return logits


class SIGN_AWARE_REPETITION_PENALTY(torch.autograd.Function):
    """Apply sampling repetition penalty while preserving int32 ONNX indices."""

    @staticmethod
    def forward(ctx, logits, repetition_penalty, previous_ids):
        previous_ids_long = previous_ids.long()
        previous_logits = torch.gather(logits, 1, previous_ids_long)
        previous_scores = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        return torch.scatter(logits, 1, previous_ids_long, previous_scores)

    @staticmethod
    def symbolic(g, logits, repetition_penalty, previous_ids):
        previous_logits = g.op("GatherElements", logits, previous_ids, axis_i=1)
        zero = g.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))
        previous_scores = g.op(
            "Where",
            g.op("Less", previous_logits, zero),
            g.op("Mul", previous_logits, repetition_penalty),
            g.op("Div", previous_logits, repetition_penalty),
        )
        return g.op("ScatterElements", logits, previous_ids, previous_scores, axis_i=1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    """Top-K/Top-P categorical sampling with sign-aware repetition penalty."""

    def __init__(self, vocab_size):
        super(TOPK_TOPP_SAMPLING, self).__init__()
        self.register_buffer("one", torch.tensor([1], dtype=torch.int64), persistent=False)
        self.register_buffer("vocab_size", torch.tensor([vocab_size], dtype=torch.int64), persistent=False)

    def sample(self, scores, temperature, top_k, top_p, greedy_scores):
        top_k = torch.minimum(torch.maximum(top_k, self.one), self.vocab_size)
        sorted_scores, sorted_indices = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=True)
        sorted_probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        keep = (cumulative_probabilities - sorted_probabilities) <= top_p

        kept_mass = torch.where(keep, cumulative_probabilities, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax((cumulative_probabilities >= threshold).int(), dim=-1, keepdim=True)
        sampled_id = torch.gather(sorted_indices, 1, winner).int()
        greedy_id = torch.argmax(greedy_scores, dim=-1, keepdim=True).int()
        return torch.where(top_k == self.one, greedy_id, sampled_id)

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        scores = SIGN_AWARE_REPETITION_PENALTY.apply(logits, repetition_penalty, previous_ids)
        sampled_id = self.sample(scores, temperature, top_k, top_p, logits)
        save_ids = torch.cat([previous_ids, sampled_id], dim=-1)
        return sampled_id, save_ids


class INDEXTTS_TOKEN_STRATEGY(torch.nn.Module):
    """Select and append one mel token using exactly one configured strategy."""

    def __init__(self, strategy, vocab_size):
        super(INDEXTTS_TOKEN_STRATEGY, self).__init__()
        if strategy not in DECODE_STRATEGIES:
            raise ValueError(f"Unsupported decode strategy: {strategy!r}")
        self.strategy = strategy
        self.penalty = APPLY_PENALTY()
        self.sampling = TOPK_TOPP_SAMPLING(vocab_size)

    def forward(self, logits, save_ids, *controls):
        if self.strategy == "penalty_greedy":
            penalty_value, penalty_range = controls
            penalized_logits = self.penalty(logits, save_ids, penalty_value, penalty_range)
            use_penalty = torch._shape_as_tensor(save_ids)[1:2] >= penalty_range
            logits = torch.where(use_penalty, penalized_logits, logits)
        elif self.strategy == "sampling":
            return self.sampling(logits, *controls, save_ids)

        next_token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return next_token, torch.cat([save_ids, next_token], dim=-1)


class INDEXTTS_MAIN_PREFILL_STRATEGY(torch.nn.Module):
    """Run the full prompt with empty KV state and select the first mel token."""

    def __init__(self, main_core, strategy, vocab_size):
        super(INDEXTTS_MAIN_PREFILL_STRATEGY, self).__init__()
        self.main_core = main_core
        self.strategy_name = strategy
        self.strategy = INDEXTTS_TOKEN_STRATEGY(strategy, vocab_size)
        self.num_layers = main_core.num_layers
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        self.register_buffer(
            "empty_key",
            torch.zeros(1, main_core.num_heads, main_core.head_dim, 0, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_value",
            torch.zeros(1, main_core.num_heads, 0, main_core.head_dim, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer("zero_history_len", torch.zeros(1, dtype=torch.int64), persistent=False)

    def forward(self, hidden_states, *controls):
        outputs = self.main_core(
            *([self.empty_key] * self.num_layers),
            *([self.empty_value] * self.num_layers),
            hidden_states,
            self.zero_history_len,
        )
        state_count = self.num_layers * 2
        logits = outputs[state_count + 1]
        if self.strategy_name == "sampling":
            next_token = self.strategy.sampling.sample(logits, *controls, logits)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return (
            *outputs[:state_count],
            outputs[state_count],
            next_token,
            outputs[state_count + 2],
        )


class INDEXTTS_MAIN_DECODE_STRATEGY(torch.nn.Module):
    """Advance Main from one embedded mel token and select the following token."""

    def __init__(self, main_core, strategy, vocab_size):
        super(INDEXTTS_MAIN_DECODE_STRATEGY, self).__init__()
        self.main_core = main_core
        self.strategy = INDEXTTS_TOKEN_STRATEGY(strategy, vocab_size)
        self.state_count = main_core.num_layers * 2

    def forward(self, *args):
        states = args[:self.state_count]
        hidden_states = args[self.state_count]
        save_ids_in = args[self.state_count + 1]
        history_len = args[self.state_count + 2]
        controls = args[self.state_count + 3:]
        outputs = self.main_core(*states, hidden_states, history_len)
        next_token, save_ids_out = self.strategy(
            outputs[self.state_count + 1],
            save_ids_in,
            *controls,
        )
        return (
            *outputs[:self.state_count],
            outputs[self.state_count],
            next_token,
            save_ids_out,
            outputs[self.state_count + 2],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers — export the pipeline geometry once so inference never has to
# hand-duplicate the fixed-at-export constants (mirrors the ASR repo).
# ─────────────────────────────────────────────────────────────────────────────
def build_model_metadata(*sections):
    """Flatten metadata sections into a ``{str: str}`` dict for ``metadata_props``."""
    metadata = {}
    for section in sections:
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                metadata[str(key)] = "1" if value else "0"
            elif isinstance(value, (list, tuple)):
                metadata[str(key)] = ",".join(str(item) for item in value)
            else:
                metadata[str(key)] = str(value)
    return metadata


def write_onnx_metadata(onnx_path, metadata):
    """Add/overwrite ``metadata_props`` in place, leaving any external-weight sidecar untouched."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, onnx_path)


def replace_onnx_metadata(onnx_path, metadata):
    """Replace the metadata carrier contract without loading external weights."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, onnx_path)


class METADATA_CARRIER(torch.nn.Module):
    """Tiny identity graph that carries the static package contract."""

    def forward(self, marker):
        return marker


if onnx_folder.exists():
    shutil.rmtree(onnx_folder)
onnx_folder.mkdir(parents=True)

print("\n\nStart compact IndexTTS export...\n")
with torch.inference_mode():
    from indextts.infer import IndexTTS
    from indextts.utils.front import TextNormalizer as ProjectTextNormalizer

    original_normalizer_load = ProjectTextNormalizer.load
    ProjectTextNormalizer.load = lambda self: None
    try:
        indexTTS = IndexTTS(model_dir=models_path, cfg_path=models_path + "/config.yaml", use_fp16=False, device='cpu')
    finally:
        ProjectTextNormalizer.load = original_normalizer_load
    for para in indexTTS.gpt.parameters():
        para.requires_grad = False
    for para in indexTTS.bigvgan.parameters():
        para.requires_grad = False

    NUM_HEADS = indexTTS.gpt.heads
    NUM_LAYERS = indexTTS.gpt.layers
    HIDDEN_SIZE = indexTTS.gpt.model_dim
    HEAD_DIM = indexTTS.gpt.inference_model.transformer.h._modules['0'].attn.head_dim
    SPEAKER_EMBED_SIZE = indexTTS.bigvgan.cond_layer.out_channels
    MEL_CODE_SIZE = indexTTS.gpt.number_mel_codes
    kv_dtype = torch.float16 if USE_F16_KV else torch.float32   # float16 KV cache storage when USE_F16_KV.

    # Package contract metadata is built while the model is loaded, then
    # stamped onto every exported graph at the end of the export block.
    onnx_metadata = build_model_metadata(
        {
            "graph_layout": "strategy_prefill_decode_step",
            "in_sample_rate": IN_SAMPLE_RATE,
            "out_sample_rate": OUT_SAMPLE_RATE,
            "stop_token_ids": STOP_TOKEN,
            "max_signal_length": MAX_SIGNAL_LENGTH,
            "use_f16_kv": USE_F16_KV,
            "compute_in_f32": COMPUTE_IN_F32,
            "shared_initializer_model_file": SHARED_MODEL_NAME,
            "shared_initializer_data_file": SHARED_DATA_NAME,
            "model_file_name_reference_preprocess": Path(onnx_model_Reference_Preprocess).name,
            "model_file_name_target_preprocess": Path(onnx_model_Target_Preprocess).name,
            "model_file_name_decoder": Path(onnx_model_Decoder).name,
            "mel_code_size": MEL_CODE_SIZE,
            **{
                f"model_file_name_main_prefill_{strategy}": Path(
                    onnx_model_Main_Prefill[strategy]
                ).name
                for strategy in DECODE_STRATEGIES
            },
            **{
                f"model_file_name_decode_step_{strategy}": Path(
                    onnx_model_Decode_Step[strategy]
                ).name
                for strategy in DECODE_STRATEGIES
            },
        },
    )

    audio = torch.ones((1, 1, 320000), dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()])
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, pad_mode='reflect').eval()
    encoder = IndexTTS_Encoder(indexTTS, custom_stft, NFFT, N_MELS, MODEL_SAMPLE_RATE, MAX_SIGNAL_LENGTH)

    output_names = []
    for i in range(indexTTS.bigvgan.num_upsamples):
        output_names.append(f"save_bigvgan_conds_{i}")
    output_names.append("bigvgan_cond_layer_speaker_embedding")
    output_names.append("conds_latent")

    torch.onnx.export(
        encoder,
        (audio,),
        onnx_model_Reference_Preprocess,
        input_names=['audio'],
        output_names=output_names,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'conds_latent': {1: 'ref_signal_len'},
        },
        do_constant_folding=True,
        dynamo=False,
        opset_version=OPSET)
    del custom_stft
    del encoder
    del audio
    gc.collect()
    print("\nExport ReferencePreprocess Done.\n\nExport TargetPreprocess Start...")

    conds_latent = torch.ones((1, 32, HIDDEN_SIZE), dtype=torch.float32)
    text_ids = torch.ones((1, 10), dtype=torch.int32)
    target_preprocess = IndexTTS_Target_Preprocess(indexTTS)
    torch.onnx.export(
        target_preprocess,
        (conds_latent, text_ids),
        onnx_model_Target_Preprocess,
        input_names=['conds_latent', 'text_ids'],
        output_names=['hidden_states'],
        dynamic_axes={
            'conds_latent': {1: 'ref_signal_len'},
            'text_ids': {1: 'text_ids_len'},
            'hidden_states': {1: 'prefill_len'},
        },
        do_constant_folding=True,
        dynamo=False,
        opset_version=OPSET)
    del target_preprocess, conds_latent, text_ids
    gc.collect()
    print("\nExport TargetPreprocess Done.\n\nExport strategy graphs Start...")

    if PENALTY_VALUE <= 0.0 or PENALTY_RANGE < 1:
        raise ValueError("Penalty-greedy trace controls require PENALTY_VALUE > 0 and PENALTY_RANGE >= 1.")
    if SAMPLING_TEMPERATURE <= 0.0 or not 0.0 < SAMPLING_TOP_P <= 1.0:
        raise ValueError("Sampling export defaults require temperature > 0 and 0 < top_p <= 1.")
    if SAMPLING_REPETITION_PENALTY <= 0.0 or SAMPLING_TOP_K < 1:
        raise ValueError("Sampling export defaults require repetition_penalty > 0 and top_k >= 1.")
    sampling_top_k = min(SAMPLING_TOP_K, MEL_CODE_SIZE)

    main_core = IndexTTS_Main(indexTTS, NUM_LAYERS, MAX_SIGNAL_LENGTH)
    hidden_states = torch.ones((1, 10, HIDDEN_SIZE), dtype=torch.float32)
    hidden_step = torch.ones((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    history_len = torch.tensor([10], dtype=torch.int64)
    save_ids = torch.zeros((1, 10), dtype=torch.int32)
    past_keys = torch.zeros((1, NUM_HEADS, HEAD_DIM, 10), dtype=kv_dtype)
    past_values = torch.zeros((1, NUM_HEADS, 10, HEAD_DIM), dtype=kv_dtype)
    state_inputs = [past_keys] * NUM_LAYERS + [past_values] * NUM_LAYERS
    state_input_names = (
        [f'in_key_{i}' for i in range(NUM_LAYERS)]
        + [f'in_value_{i}' for i in range(NUM_LAYERS)]
    )
    state_output_names = (
        [f'out_key_{i}' for i in range(NUM_LAYERS)]
        + [f'out_value_{i}' for i in range(NUM_LAYERS)]
    )
    state_axes = {}
    for i in range(NUM_LAYERS):
        state_axes[f'in_key_{i}'] = {3: 'history_len'}
        state_axes[f'out_key_{i}'] = {3: 'kv_seq_len'}
        state_axes[f'in_value_{i}'] = {2: 'history_len'}
        state_axes[f'out_value_{i}'] = {2: 'kv_seq_len'}

    control_tensors = {
        'penalty_value': torch.tensor([PENALTY_VALUE], dtype=torch.float32),
        'penalty_range': torch.tensor([PENALTY_RANGE], dtype=torch.int64),
        'temperature': torch.tensor([SAMPLING_TEMPERATURE], dtype=torch.float32),
        'top_k': torch.tensor([sampling_top_k], dtype=torch.int64),
        'top_p': torch.tensor([SAMPLING_TOP_P], dtype=torch.float32),
        'repetition_penalty': torch.tensor([SAMPLING_REPETITION_PENALTY], dtype=torch.float32),
    }

    for strategy in DECODE_STRATEGIES:
        if strategy == 'greedy':
            decode_control_names = []
        elif strategy == 'penalty_greedy':
            decode_control_names = ['penalty_value', 'penalty_range']
        else:
            decode_control_names = ['temperature', 'top_k', 'top_p', 'repetition_penalty']
        prefill_control_names = ['temperature', 'top_k', 'top_p'] if strategy == 'sampling' else []
        prefill_controls = [control_tensors[name] for name in prefill_control_names]
        decode_controls = [control_tensors[name] for name in decode_control_names]

        prefill = INDEXTTS_MAIN_PREFILL_STRATEGY(main_core, strategy, MEL_CODE_SIZE)
        prefill_axes = {
            **{name: axes for name, axes in state_axes.items() if name.startswith('out_')},
            'hidden_states': {1: 'prefill_len'},
        }
        torch.onnx.export(
            prefill,
            (hidden_states, *prefill_controls),
            onnx_model_Main_Prefill[strategy],
            input_names=['hidden_states', *prefill_control_names],
            output_names=[*state_output_names, 'last_hidden_state', 'next_token', 'kv_seq_len'],
            dynamic_axes=prefill_axes,
            do_constant_folding=True,
            dynamo=False,
            opset_version=OPSET,
        )
        del prefill

        main_decode = INDEXTTS_MAIN_DECODE_STRATEGY(main_core, strategy, MEL_CODE_SIZE)
        decode_axes = {
            **state_axes,
            'save_ids_in': {1: 'save_ids_len'},
            'save_ids_out': {1: 'save_ids_len_out'},
        }
        torch.onnx.export(
            main_decode,
            (*state_inputs, hidden_step, save_ids, history_len, *decode_controls),
            onnx_model_Main_Decode[strategy],
            input_names=[*state_input_names, 'hidden_states', 'save_ids_in', 'history_len', *decode_control_names],
            output_names=[*state_output_names, 'last_hidden_state', 'next_token', 'save_ids_out', 'kv_seq_len'],
            dynamic_axes=decode_axes,
            do_constant_folding=True,
            dynamo=False,
            opset_version=OPSET,
        )
        del main_decode

    decode_embed = IndexTTS_Decode_Embed(indexTTS)
    current_token = torch.zeros((1, 1), dtype=torch.int32)
    torch.onnx.export(
        decode_embed,
        (current_token, save_ids),
        onnx_model_Decode_Embed,
        input_names=['current_token', 'save_ids_in'],
        output_names=['hidden_states'],
        dynamic_axes={'save_ids_in': {1: 'save_ids_len'}},
        do_constant_folding=True,
        dynamo=False,
        opset_version=OPSET,
    )
    del main_core, hidden_states, hidden_step, history_len, save_ids
    del past_keys, past_values, state_inputs, control_tensors, decode_embed, current_token
    gc.collect()
    print("\nExport strategy components Done.\n\nExport Decoder Start...")

    all_inputs = []
    input_names = []
    for i in range(indexTTS.bigvgan.num_upsamples):
        input_names.append(f"save_bigvgan_conds_{i}")
        all_inputs.append(torch.ones((1, indexTTS.bigvgan.conds[i].out_channels, 1), dtype=torch.float32))
    input_names.append("bigvgan_cond_layer_speaker_embedding")
    all_inputs.append(torch.ones((1, SPEAKER_EMBED_SIZE, 1), dtype=torch.float32))
    input_names.append("save_hidden_state")
    all_inputs.append(torch.ones((10, HIDDEN_SIZE), dtype=torch.float32))

    decoder = IndexTTS_Decoder(indexTTS)
    torch.onnx.export(
        decoder,
        tuple(all_inputs),
        onnx_model_Decoder,
        input_names=input_names,
        output_names=['generated_wav'],
        dynamic_axes={
            'save_hidden_state': {0: 'kv_seq_len'},
            'generated_wav': {2: 'generated_len'}
        },
        do_constant_folding=True,
        dynamo=False,
        opset_version=OPSET)
    del decoder
    del all_inputs
    del input_names
    gc.collect()
    print("\nExport Decoder Done.")

    # ── Metadata carrier + stamp the metadata onto every exported graph ──
    metadata_marker = torch.zeros((1,), dtype=torch.int64)
    torch.onnx.export(
        METADATA_CARRIER(),
        (metadata_marker,),
        onnx_model_Metadata,
        input_names=['metadata_marker'],
        output_names=['metadata_marker_out'],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False
    )
    del metadata_marker
    component_graphs = (
        [Path(onnx_model_Reference_Preprocess), Path(onnx_model_Target_Preprocess)]
        + [Path(onnx_model_Main_Prefill[strategy]) for strategy in DECODE_STRATEGIES]
        + [Path(onnx_model_Main_Decode[strategy]) for strategy in DECODE_STRATEGIES]
        + [Path(onnx_model_Decode_Embed)]
        + [Path(onnx_model_Decoder), Path(onnx_model_Metadata)]
    )
    del indexTTS
    gc.collect()
    shared_stats = bundle_shared_initializers(
        onnx_folder,
        model_paths=component_graphs,
        metadata=onnx_metadata,
    )
    build_decode_step_graphs(onnx_folder, DECODE_STRATEGIES)
    final_graphs = (
        [Path(onnx_model_Reference_Preprocess), Path(onnx_model_Target_Preprocess)]
        + [Path(onnx_model_Main_Prefill[strategy]) for strategy in DECODE_STRATEGIES]
        + [Path(onnx_model_Decode_Step[strategy]) for strategy in DECODE_STRATEGIES]
        + [Path(onnx_model_Decoder), Path(onnx_model_Metadata)]
    )
    shared_audit = audit_shared_bundle(onnx_folder, final_graphs)
    replace_onnx_metadata(onnx_model_Metadata, onnx_metadata)
    print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {len(final_graphs)} final graph(s).")
    print(
        f"[Shared weights] {shared_stats['initializer_references']} references -> "
        f"{shared_stats['unique_initializers']} unique tensors; deduplicated "
        f"{shared_stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB; final blob "
        f"{shared_audit['external_bytes'] / (1024 * 1024):.2f} MiB."
    )
    print("\nCompact IndexTTS export done.")


if project_path in sys.path:
    sys.path.remove(project_path)

print("\nStart running inference via Inference_IndexTTS_ONNX.py ...")
subprocess.run(
    [sys.executable, str(script_dir / "Inference_IndexTTS_ONNX.py"), "--onnx-folder", str(onnx_folder)],
    check=True,
)
