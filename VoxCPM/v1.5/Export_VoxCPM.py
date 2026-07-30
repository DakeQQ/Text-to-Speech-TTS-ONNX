import gc
import math
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from torch import nn
from torch.nn.utils import weight_norm

import voxcpm.model.voxcpm as voxcpm_model_module
import voxcpm.modules.audiovae as voxcpm_audiovae_pkg
import voxcpm.modules.audiovae.audio_vae as voxcpm_audio_vae


script_dir = Path(__file__).resolve().parent
raw_onnx_folder = script_dir / "VoxCPM_ONNX_Raw"
onnx_folder = script_dir / "VoxCPM_ONNX"
raw_onnx_folder.mkdir(parents=True, exist_ok=True)
if str(script_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent.parent))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
from Rewrite_VoxCPM_ONNX import rewrite_voxcpm_onnx_folder
from Shared_Weights import (
    GraphComponent,
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    bundle_shared_initializers,
    compose_graphs,
)

# User-configurable export options
path_voxcpm = Path.home() / "Downloads" / "VoxCPM1.5"

# Exported sequence and audio capacities
MAX_SEQ_LEN = 1024                       # Maximum exported text/audio sequence length.
IN_SAMPLE_RATE = 44100                   # Public prompt-audio ONNX input rate.
OUT_SAMPLE_RATE = 44100                  # Public generated-waveform ONNX output rate.
IN_AUDIO_DTYPE = "F32"                   # "F16" | "F32" | "INT16".
OUT_AUDIO_DTYPE = "F32"                  # "F16" | "F32" | "INT16".
MODEL_SAMPLE_RATE = 44100                # Native VoxCPM VAE sample rate; do not edit.
MAX_PROMPT_AUDIO_LEN = 30 * IN_SAMPLE_RATE  # Maximum exported prompt-audio capacity in samples.

_AUDIO_DTYPES = {"F16": torch.float16, "F32": torch.float32, "INT16": torch.int16}
# Exported diffusion schedule
FIXED_TIMESTEPS = 10                     # More unrolled diffusion steps improve quality but cost latency.

# Quantization preparation
REORDER_DOWNPROJ_FOR_QUANT = True        # Enable only after same-noise quality validation for the selected KV precision.
REORDER_OPROJ_FOR_QUANT = True           # Optional exact value/o-projection channel reorder.
REORDER_KEY = "absmean"                  # "absmean" | "L4" | "rms" | "std".

# Exported KV-cache precision
USE_F16_KV = True                        # False selects full-f32 KV storage and attention.
COMPUTE_IN_F32 = False                   # With f16 KV storage, upcast K/V only at attention matmuls.


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        variance = x.float().square().mean(dim=axis, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        return g.op(
            "SimplifiedLayerNormalization",
            x,
            scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )


def simplified_layer_norm(x, scale, epsilon, axis=-1):
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


def _channel_score(weight, key, dims):
    weight = weight.float()
    absolute = weight.abs()
    if key == "rms":
        return weight.square().mean(dim=dims).sqrt()
    if key == "L4":
        return absolute.pow(4).mean(dim=dims).pow(0.25)
    if key == "std":
        return weight.std(dim=dims)
    if key == "absmean":
        return absolute.mean(dim=dims)
    pass
def _reorder_transformer_channels(
    layers,
    num_heads,
    num_key_value_heads,
    head_dim,
    qk_heads,
):
    """Permute both sides of quantization-sensitive channel boundaries."""
    with torch.no_grad():
        for layer in layers:
            if REORDER_DOWNPROJ_FOR_QUANT:
                down_weight = layer.mlp.down_proj.weight
                permutation = torch.argsort(_channel_score(down_weight, REORDER_KEY, (0,)))
                intermediate_size = layer.mlp.down_proj.in_features
                gate_up_weight = layer.mlp.gate_up_proj.weight
                reordered_gate_up = torch.cat(
                    [
                        gate_up_weight[:intermediate_size][permutation],
                        gate_up_weight[intermediate_size:][permutation],
                    ],
                    dim=0,
                )
                layer.mlp.gate_up_proj.weight.copy_(reordered_gate_up)
                layer.mlp.down_proj.weight.copy_(down_weight[:, permutation])

            if REORDER_OPROJ_FOR_QUANT:
                heads_per_kv = num_heads // num_key_value_heads
                output_weight = layer.self_attn.o_proj.weight
                output_by_head = output_weight.view(output_weight.shape[0], num_heads, head_dim)
                permutations = []
                for kv_head in range(num_key_value_heads):
                    grouped = output_by_head[
                        :,
                        kv_head * heads_per_kv:(kv_head + 1) * heads_per_kv,
                    ]
                    permutations.append(
                        torch.argsort(_channel_score(grouped, REORDER_KEY, (0, 1)))
                    )

                reordered_output = output_by_head.clone()
                for head in range(num_heads):
                    reordered_output[:, head] = output_by_head[
                        :, head, permutations[head // heads_per_kv]
                    ]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                qkv_weight = layer.self_attn.qkv.weight
                qkv_by_head = qkv_weight.view(-1, head_dim, qkv_weight.shape[1]).clone()
                for kv_head, permutation in enumerate(permutations):
                    qkv_by_head[qk_heads + kv_head] = qkv_by_head[qk_heads + kv_head][permutation]
                qkv_weight.copy_(qkv_by_head.reshape_as(qkv_weight))
                if layer.self_attn.qkv.bias is not None:
                    qkv_bias = layer.self_attn.qkv.bias
                    qkv_bias_by_head = qkv_bias.view(-1, head_dim).clone()
                    for kv_head, permutation in enumerate(permutations):
                        qkv_bias_by_head[qk_heads + kv_head] = qkv_bias_by_head[
                            qk_heads + kv_head
                        ][permutation]
                    qkv_bias.copy_(qkv_bias_by_head.reshape_as(qkv_bias))


# ══════════════════════════════════════════════════════════════════════════════
# Standalone VoxCPM package overrides required for ONNX export
# ══════════════════════════════════════════════════════════════════════════════
def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.register_buffer(
            "pad_zeros",
            torch.zeros([1, self.in_channels, self.__padding * 2], dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x):
        if self.__padding == 0:
            return super().forward(x)
        x_pad = torch.cat([self.pad_zeros, x], dim=-1)
        return super().forward(x_pad)


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]


def WNCausalConv1d(*args, **kwargs):
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args, **kwargs):
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


def snake(x, alpha, alpha_reciprocal):
    return x + alpha_reciprocal * torch.sin(alpha * x).square()


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        self.register_buffer("alpha_reciprocal", (self.alpha + 1e-9).reciprocal())

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.alpha_reciprocal = (self.alpha + 1e-9).reciprocal()

    def forward(self, x):
        return snake(x, self.alpha, self.alpha_reciprocal)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class CausalResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNCausalConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ),
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.block(x)


class CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim: int = 16, input_dim=None, stride: int = 1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNCausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: list = [2, 4, 8, 8],
        depthwise: bool = False,
    ):
        super().__init__()
        self.block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            self.block += [CausalEncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
        groups = d_model if depthwise else 1
        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        hidden_state = self.block(x)
        return self.fc_mu(hidden_state)


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNCausalConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        groups=1,
        use_noise_block: bool = False,
    ):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNCausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if use_noise_block:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                CausalResidualUnit(output_dim, dilation=1, groups=groups),
                CausalResidualUnit(output_dim, dilation=3, groups=groups),
                CausalResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransposeLastTwoDim(torch.nn.Module):
    def forward(self, x):
        return torch.transpose(x, -1, -2)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        depthwise: bool = False,
        d_out: int = 1,
        use_noise_block: bool = False,
    ):
        super().__init__()
        if depthwise:
            layers = [
                WNCausalConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                WNCausalConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers += [
                CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=groups,
                    use_noise_block=use_noise_block,
                )
            ]

        layers += [
            Snake1d(output_dim),
            WNCausalConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AudioVAEConfig(BaseModel):
    encoder_dim: int = 128
    encoder_rates: List[int] = [2, 5, 8, 8]
    latent_dim: int = 64
    decoder_dim: int = 1536
    decoder_rates: List[int] = [8, 8, 5, 2]
    depthwise: bool = True
    sample_rate: int = 16000
    use_noise_block: bool = False


class AudioVAE(nn.Module):
    def __init__(self, config: Optional[AudioVAEConfig] = None):
        if config is None:
            config = AudioVAEConfig()
        super().__init__()

        encoder_dim = config.encoder_dim
        encoder_rates = config.encoder_rates
        latent_dim = config.latent_dim
        decoder_dim = config.decoder_dim
        decoder_rates = config.decoder_rates
        depthwise = config.depthwise
        sample_rate = config.sample_rate
        use_noise_block = config.use_noise_block

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.depthwise = depthwise
        self.use_noise_block = use_noise_block

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = CausalEncoder(encoder_dim, latent_dim, encoder_rates, depthwise=depthwise)
        self.decoder = CausalDecoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            depthwise=depthwise,
            use_noise_block=use_noise_block,
        )
        self.sample_rate = sample_rate
        self.chunk_size = math.prod(encoder_rates)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        pass
        pad_to = self.hop_length
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def encode(self, audio_data: torch.Tensor, sample_rate: int):
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        audio_data = self.preprocess(audio_data, sample_rate)
        return self.encoder(audio_data)


def _patch_voxcpm_audio_vae():
    replacements = {
        "WNConv1d": WNConv1d,
        "WNConvTranspose1d": WNConvTranspose1d,
        "CausalConv1d": CausalConv1d,
        "CausalTransposeConv1d": CausalTransposeConv1d,
        "WNCausalConv1d": WNCausalConv1d,
        "WNCausalTransposeConv1d": WNCausalTransposeConv1d,
        "snake": snake,
        "Snake1d": Snake1d,
        "init_weights": init_weights,
        "CausalResidualUnit": CausalResidualUnit,
        "CausalEncoderBlock": CausalEncoderBlock,
        "CausalEncoder": CausalEncoder,
        "NoiseBlock": NoiseBlock,
        "CausalDecoderBlock": CausalDecoderBlock,
        "TransposeLastTwoDim": TransposeLastTwoDim,
        "CausalDecoder": CausalDecoder,
        "AudioVAEConfig": AudioVAEConfig,
        "AudioVAE": AudioVAE,
    }
    for name, value in replacements.items():
        setattr(voxcpm_audio_vae, name, value)
    voxcpm_audiovae_pkg.AudioVAE = AudioVAE
    voxcpm_audiovae_pkg.AudioVAEConfig = AudioVAEConfig
    voxcpm_model_module.AudioVAE = AudioVAE
    voxcpm_model_module.AudioVAEConfig = AudioVAEConfig


_patch_voxcpm_audio_vae()
VoxCPMModel = voxcpm_model_module.VoxCPMModel
LoRAConfig = voxcpm_model_module.LoRAConfig


class VoxCPM:
    def __init__(
        self,
        voxcpm_model_path: str,
        zipenhancer_model_path: Optional[str] = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        enable_denoiser: bool = True,
        optimize: bool = True,
        device: Optional[str] = None,
        lora_config: Optional[LoRAConfig] = None,
        lora_weights_path: Optional[str] = None,
    ):
        print(f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}")

        if lora_weights_path is not None and lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
            )
            print(f"Auto-created default LoRAConfig for loading weights from: {lora_weights_path}")

        self.tts_model = VoxCPMModel.from_local(
            voxcpm_model_path,
            optimize=optimize,
            device=device,
            lora_config=lora_config,
        )

        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}")
            loaded_keys, skipped_keys = self.tts_model.load_lora_weights(lora_weights_path)
            print(f"Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}")

        self.denoiser = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from voxcpm.zipenhancer import ZipEnhancer
            self.denoiser = ZipEnhancer(zipenhancer_model_path)

    @classmethod
    def from_pretrained(
        cls,
        hf_model_id: str = "openbmb/VoxCPM1.5",
        load_denoiser: bool = True,
        zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
        cache_dir: str = None,
        local_files_only: bool = False,
        optimize: bool = True,
        device: Optional[str] = None,
        lora_config: Optional[LoRAConfig] = None,
        lora_weights_path: Optional[str] = None,
        **kwargs,
    ):
        if os.path.isdir(hf_model_id):
            local_path = hf_model_id
        else:
            local_path = snapshot_download(
                repo_id=hf_model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            device=device,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            **kwargs,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VAE Encoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_VAE_ENCODER(torch.nn.Module):
    def __init__(self, voxcpm, in_sample_rate):
        super(VOXCPM_VAE_ENCODER, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self._remove_weight_norm(self.voxcpm.audio_vae.encoder)
        self.patch_len = self.voxcpm.patch_size * self.voxcpm.chunk_size
        self.register_buffer("pad_zeros", torch.zeros([1, 1, self.patch_len], dtype=torch.float32), persistent=False)
        self.register_buffer("pad_zeros_right", torch.zeros([1, 1, self.patch_len], dtype=torch.float32), persistent=False)
        self.in_sample_rate = in_sample_rate
        self.sr_scale = float(MODEL_SAMPLE_RATE / self.in_sample_rate)

        if "int" in IN_AUDIO_DTYPE.lower():
            with torch.no_grad():
                first_conv = self.voxcpm.audio_vae.encoder.block[0]
                first_conv.weight.mul_(1.0 / 32768.0)

    @staticmethod
    def _remove_weight_norm(module):
        for child in module.modules():
            try:
                torch.nn.utils.remove_weight_norm(child)
            except ValueError:
                pass

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def forward(self, prompt_audio):
        prompt_audio = prompt_audio.float()
        if self.sr_scale != 1.0:
            prompt_audio = torch.nn.functional.interpolate(
                prompt_audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )
        padding_size = self.patch_len - prompt_audio.shape[-1] % self.patch_len
        prompt_audio = torch.cat([self.pad_zeros[..., :padding_size], prompt_audio, self.pad_zeros_right], dim=-1)
        audio_feat = self.voxcpm.audio_vae.encoder(prompt_audio)
        audio_feat = audio_feat.view(self.voxcpm.audio_vae.latent_dim, -1, self.voxcpm.patch_size).permute(1, 2, 0)
        return audio_feat


# ══════════════════════════════════════════════════════════════════════════════
# Fused Feature Encoder + Conditioning Module
# Replaces separate Feat_Encoder and Feat_Cond modules.
# Returns both feat_embed (for LM) and feat_cond (for diffusion) in one call.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_FEAT_ENCODER_COND(torch.nn.Module):
    def __init__(self, voxcpm, max_prompt_audio_len, in_sample_rate):
        super(VOXCPM_FEAT_ENCODER_COND, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)

        # === Feat Encoder geometry ===
        self.head_dim = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.head_dim
        self.num_heads = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_heads
        self.num_key_value_heads = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_key_value_heads
        self.num_key_value_groups = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_key_value_groups
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.register_buffer(
            "rope_permutation",
            torch.cat((torch.arange(self.head_dim // 2, self.head_dim), torch.arange(self.head_dim // 2))).to(torch.int32),
            persistent=False,
        )
        hidden_size = self.voxcpm.feat_encoder.encoder.config.hidden_size
        self.rms_norm_epsilon = float(self.voxcpm.feat_encoder.encoder.config.rms_norm_eps)
        self.register_buffer(
            "rms_scale",
            torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32),
            persistent=False,
        )

        max_prompt_feat_len = (max_prompt_audio_len // in_sample_rate * MODEL_SAMPLE_RATE) // (self.voxcpm.patch_size * self.voxcpm.chunk_size) + 1
        special_tokens = self.voxcpm.feat_encoder.special_token.expand(1, max_prompt_feat_len, 1, -1).squeeze(0).half().float()
        self.register_buffer("special_tokens", special_tokens, persistent=False)
        self.q_len = self.voxcpm.patch_size + 1  # Fixed to 5 for VoxCPM1.5
        position_ids = torch.arange(self.q_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = self.voxcpm.feat_encoder.encoder.rope_emb(position_ids)
        rope_half = self.voxcpm.feat_encoder.encoder.rope_emb.dim // 2
        rope_emb_sin = torch.cat((-rope_emb_sin[:, :rope_half], rope_emb_sin[:, rope_half:]), dim=-1)
        self.register_buffer("rope_emb_cos", rope_emb_cos.unsqueeze(1).unsqueeze(1).unsqueeze(0), persistent=False)
        self.register_buffer("rope_emb_sin", rope_emb_sin.unsqueeze(1).unsqueeze(1).unsqueeze(0), persistent=False)

        norm_factor = self.voxcpm.feat_encoder.encoder.config.hidden_size ** 0.5
        scale_factor = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.head_dim ** -0.25

        with torch.no_grad():
            for layer in self.voxcpm.feat_encoder.encoder.layers:
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                in_features = int(q_proj.in_features)
                out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
                has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
                qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
                if has_bias:
                    z = lambda feat: torch.zeros(feat, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
                    qb = q_proj.bias if q_proj.bias is not None else z(q_proj.out_features)
                    kb = k_proj.bias if k_proj.bias is not None else z(k_proj.out_features)
                    vb = v_proj.bias if v_proj.bias is not None else z(v_proj.out_features)
                    qkv.bias.copy_(torch.cat([qb * scale_factor, kb * scale_factor, vb], dim=0))

                layer.self_attn.q_out_features = int(q_proj.out_features)
                layer.self_attn.k_out_features = int(k_proj.out_features)
                layer.self_attn.v_out_features = int(v_proj.out_features)
                layer.self_attn.qkv = qkv
                del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                del layer.input_layernorm

                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj
                in_feat = gate.in_features
                out_feat = gate.out_features + up.out_features
                gate_up = torch.nn.Linear(in_feat, out_feat, bias=False)
                gate_up.weight.copy_(torch.cat([gate.weight * w, up.weight * w], dim=0))
                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

            w = self.voxcpm.feat_encoder.encoder.norm.weight.unsqueeze(0) * norm_factor
            self.voxcpm.enc_to_lm_proj.weight.mul_(w)
            del self.voxcpm.feat_encoder.encoder.norm

        _reorder_transformer_channels(
            self.voxcpm.feat_encoder.encoder.layers,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.qk_heads,
        )

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        return simplified_layer_norm(x, self.rms_scale, self.rms_norm_epsilon)

    def rotate_half(self, x):
        return x.reshape(-1, 2, self.head_dim // 2).flip(-2).reshape_as(x)

    def forward(self, audio_feat):
        # === Feature Encoder: produces feat_embed for the LM ===
        audio_feat_len = torch._shape_as_tensor(audio_feat)[0].unsqueeze(0)
        hidden_states = self.voxcpm.feat_encoder.in_proj(audio_feat)
        hidden_states = torch.cat([self.special_tokens[:audio_feat_len], hidden_states], dim=-2)
        hidden_states = hidden_states.view(-1, self.q_len, self.voxcpm.feat_encoder.in_proj.out_features)
        for layer in self.voxcpm.feat_encoder.encoder.layers:
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, self.q_len, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * self.rope_emb_cos + self.rotate_half(qk) * self.rope_emb_sin
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.q_len, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)
            attn = torch.softmax(torch.matmul(q, k), dim=-1)
            attn = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(-1, self.q_len, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        feat_embed = hidden_states[:, 0]
        feat_embed = self._rms_norm(feat_embed)
        feat_embed = self.voxcpm.enc_to_lm_proj(feat_embed).unsqueeze(0)

        # === Feature Conditioning: produces feat_cond for diffusion ===
        feat_cond = self.voxcpm.feat_decoder.estimator.cond_proj(audio_feat[[-1]])
        feat_cond = torch.cat([feat_cond, feat_cond], dim=0)

        return feat_embed, feat_cond


# ══════════════════════════════════════════════════════════════════════════════
# Fused Prefill Module
# Replaces: Text_Embed + multiple Concat calls + Rotary_Mask_Prefill
# Produces the full prefill hidden_states, rotary embeddings, and causal mask
# in a single model call.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_PREFILL(torch.nn.Module):
    def __init__(self, voxcpm, max_seq_len):
        super(VOXCPM_PREFILL, self).__init__()
        self.embed_tokens = voxcpm.base_lm.embed_tokens
        self.register_buffer("audio_start_id", torch.tensor([[101]], dtype=torch.int32), persistent=False)

        # Causal attention mask
        attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.float32))) * -128.0
        self.register_buffer("attention_mask", attention_mask, persistent=False)

        # Precompute rotary embeddings
        position_ids = torch.arange(max_seq_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = voxcpm.base_lm.rope_emb(position_ids)
        rope_half = voxcpm.base_lm.rope_emb.dim // 2
        rope_emb_sin = torch.cat((-rope_emb_sin[:, :rope_half], rope_emb_sin[:, rope_half:]), dim=-1)
        cos = rope_emb_cos.unsqueeze(1).unsqueeze(1)
        sin = rope_emb_sin.unsqueeze(1).unsqueeze(1)
        self.register_buffer("cos_rotary_pos_emb", cos.half().float(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half().float(), persistent=False)

    def forward(self, prompt_text_ids, target_text_ids, feat_embed):
        # Build full sequence: [prompt_text | target_text | audio_start | feat_embed]
        text_ids = torch.cat([prompt_text_ids, target_text_ids, self.audio_start_id], dim=1)
        text_embed = self.embed_tokens(text_ids)
        concat_text_len = torch._shape_as_tensor(text_embed)[1].unsqueeze(0)

        hidden_states = torch.cat([text_embed, feat_embed], dim=1)
        ids_len = torch._shape_as_tensor(hidden_states)[1].unsqueeze(0)

        # Compute rotary embeddings and causal mask
        rotary_cos = self.cos_rotary_pos_emb[:ids_len]
        rotary_sin = self.sin_rotary_pos_emb[:ids_len]
        attention_mask = self.attention_mask[..., :ids_len, :ids_len]

        return hidden_states, concat_text_len, rotary_cos, rotary_sin, attention_mask, ids_len


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding (Decode Only)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_ROTARY_MASK_DECODE(torch.nn.Module):
    def __init__(self, voxcpm, max_seq_len):
        super().__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = voxcpm.base_lm.rope_emb(position_ids)
        rope_half = voxcpm.base_lm.rope_emb.dim // 2
        rope_emb_sin = torch.cat((-rope_emb_sin[:, :rope_half], rope_emb_sin[:, rope_half:]), dim=-1)
        cos = rope_emb_cos.unsqueeze(1).unsqueeze(1)
        sin = rope_emb_sin.unsqueeze(1).unsqueeze(1)
        self.register_buffer("cos_rotary_pos_emb", cos.half().float(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half().float(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[kv_seq_len]
        rotary_sin = self.sin_rotary_pos_emb[kv_seq_len]
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Main Transformer Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_MAIN(torch.nn.Module):
    def __init__(self, voxcpm, max_seq_len):
        super(VOXCPM_MAIN, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self.compute_in_f32 = COMPUTE_IN_F32

        self.head_dim = self.voxcpm.base_lm.layers._modules['0'].self_attn.head_dim
        self.num_heads = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_heads
        self.num_key_value_heads = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_key_value_heads
        self.num_key_value_groups = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_key_value_groups
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.register_buffer(
            "rope_permutation",
            torch.cat((torch.arange(self.head_dim // 2, self.head_dim), torch.arange(self.head_dim // 2))).to(torch.int32),
            persistent=False,
        )

        hidden_size = self.voxcpm.base_lm.config.hidden_size
        self.rms_norm_epsilon = float(self.voxcpm.base_lm.config.rms_norm_eps)
        self.register_buffer(
            "rms_scale",
            torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32),
            persistent=False,
        )

        self.total_layers = self.voxcpm.base_lm.config.num_hidden_layers + self.voxcpm.residual_lm.config.num_hidden_layers

        self.norm_factor = self.voxcpm.base_lm.config.hidden_size ** 0.5
        self.scale_factor_base = float(self.voxcpm.base_lm.layers._modules['0'].self_attn.head_dim ** -0.25)
        self._fuse_weights()
        _reorder_transformer_channels(
            [*self.voxcpm.base_lm.layers, *self.voxcpm.residual_lm.layers],
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.qk_heads,
        )
        self._fuse_dit_stop_proj()

    def _fuse_dit_stop_proj(self):
        """Fuse lm_to_dit_proj and stop_proj into a single linear to reduce two matmuls to one."""
        with torch.no_grad():
            dit_proj = self.voxcpm.lm_to_dit_proj
            stop_proj = self.voxcpm.stop_proj
            in_features = dit_proj.in_features
            dit_out = dit_proj.out_features
            stop_out = stop_proj.out_features
            self.dit_out_features = dit_out
            self.stop_out_features = stop_out
            has_bias = (dit_proj.bias is not None) or (stop_proj.bias is not None)
            fused = torch.nn.Linear(in_features, dit_out + stop_out, bias=has_bias)
            fused.weight.copy_(torch.cat([dit_proj.weight, stop_proj.weight], dim=0))
            if has_bias:
                z = lambda feat: torch.zeros(feat, dtype=dit_proj.weight.dtype, device=dit_proj.weight.device)
                db = dit_proj.bias if dit_proj.bias is not None else z(dit_out)
                sb = stop_proj.bias if stop_proj.bias is not None else z(stop_out)
                fused.bias.copy_(torch.cat([db, sb], dim=0))
            self.fused_dit_stop_proj = fused
            del self.voxcpm.lm_to_dit_proj, self.voxcpm.stop_proj

    def _fuse_weights(self):
        with torch.no_grad():
            for layer in self.voxcpm.base_lm.layers:
                self._fuse_qkv_projection(layer)
                self._fuse_gate_up_projection(layer)
            for layer in self.voxcpm.residual_lm.layers:
                self._fuse_qkv_projection(layer)
                self._fuse_gate_up_projection(layer)
            final_norm_weight = self.voxcpm.residual_lm.norm.weight.unsqueeze(0) * self.norm_factor
            self.voxcpm.res_to_dit_proj.weight.mul_(final_norm_weight)
            del self.voxcpm.residual_lm.norm

    def _fuse_qkv_projection(self, layer):
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight * self.scale_factor_base, k_proj.weight * self.scale_factor_base, v_proj.weight], dim=0))
        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj) * self.scale_factor_base, _get_bias(k_proj) * self.scale_factor_base, _get_bias(v_proj)], dim=0))
        layer.self_attn.q_out_features = int(q_proj.out_features)
        layer.self_attn.k_out_features = int(k_proj.out_features)
        layer.self_attn.v_out_features = int(v_proj.out_features)
        layer.self_attn.qkv = qkv
        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * self.norm_factor
        qkv.weight.mul_(input_norm_weight)
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer):
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * self.norm_factor
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                VOXCPM_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        return simplified_layer_norm(x, self.rms_scale, self.rms_norm_epsilon)

    def _rotate_half(self, x):
        return x.reshape(-1, 2, self.head_dim // 2).flip(-2).reshape_as(x)

    def forward(self, *all_inputs):
        feat_embed         = all_inputs[-6]
        concat_text_len    = all_inputs[-5]
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        # Shared f16 mask for the in-f16 attention path: cast once, reused across every layer of both loops.
        # Only the compute precision changes with COMPUTE_IN_F32; the f16 KV cache storage is unaffected.
        attn_mask_f16 = attention_mask.half() if (USE_F16_KV and not self.compute_in_f32) else None
        save_key = []
        save_value = []

        for i, layer in enumerate(self.voxcpm.base_lm.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * rotary_pos_emb_cos + self._rotate_half(qk) * rotary_pos_emb_sin
            if USE_F16_KV and not self.compute_in_f32:
                qk = qk.half()  # earliest clean point (post-RoPE, pre-split): q and k share the f16 cast
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(1, 2, 0, 3)
            if USE_F16_KV:
                if self.compute_in_f32:
                    k = k.half()  # storage-only cast; k is upcast again at the matmul
                v = v.half()
            k = k.permute(2, 1, 3, 0)
            v = v.transpose(0, 2)
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.total_layers], v), dim=-2)
            save_key.append(k)
            save_value.append(v)
            if USE_F16_KV:
                if self.compute_in_f32:
                    attn = torch.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1)
                    attn = torch.matmul(attn, v.float()).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
                else:
                    attn = torch.softmax(torch.matmul(q, k) + attn_mask_f16, dim=-1)
                    attn = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features).float()
            else:
                attn = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
                attn = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        hidden_states = self.voxcpm.base_lm.norm(hidden_states)
        fsq_layer_out = self.voxcpm.fsq_layer(hidden_states[:, concat_text_len:])
        hidden_states = hidden_states[:, :concat_text_len]
        lm_hidden = torch.cat([hidden_states, fsq_layer_out], dim=1)[:, [-1]]
        hidden_states = torch.cat([hidden_states, fsq_layer_out + feat_embed], dim=1)

        i = self.voxcpm.base_lm.config.num_hidden_layers
        for layer in self.voxcpm.residual_lm.layers:
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * rotary_pos_emb_cos + self._rotate_half(qk) * rotary_pos_emb_sin
            if USE_F16_KV and not self.compute_in_f32:
                qk = qk.half()  # earliest clean point (post-RoPE, pre-split): q and k share the f16 cast
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(1, 2, 0, 3)
            if USE_F16_KV:
                if self.compute_in_f32:
                    k = k.half()  # storage-only cast; k is upcast again at the matmul
                v = v.half()
            k = k.permute(2, 1, 3, 0)
            v = v.transpose(0, 2)
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.total_layers], v), dim=-2)
            save_key.append(k)
            save_value.append(v)
            if USE_F16_KV:
                if self.compute_in_f32:
                    attn = torch.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1)
                    attn = torch.matmul(attn, v.float()).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
                else:
                    attn = torch.softmax(torch.matmul(q, k) + attn_mask_f16, dim=-1)
                    attn = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features).float()
            else:
                attn = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
                attn = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            i += 1

        residual_hidden = hidden_states[:, [-1]]
        residual_hidden = self._rms_norm(residual_hidden)
        fused_out = self.fused_dit_stop_proj(lm_hidden)
        dit_hidden_1, stop_intermediate = torch.split(fused_out, [self.dit_out_features, self.stop_out_features], dim=-1)
        dit_hidden_2 = self.voxcpm.res_to_dit_proj(residual_hidden)
        dit_hidden = dit_hidden_1 + dit_hidden_2
        random = torch.randn((1, self.voxcpm.patch_size, self.voxcpm.feat_decoder.in_channels), dtype=torch.float32)
        stop_flag = self.voxcpm.stop_head(self.voxcpm.stop_actn(stop_intermediate)).argmax(dim=-1, keepdims=False).int()
        return *save_key, *save_value, random, dit_hidden, stop_flag


# ══════════════════════════════════════════════════════════════════════════════
# Fused Feature Decoder Module (Full Diffusion Loop)
# All timesteps are unrolled into a single forward pass.
# Reduces timesteps session.run() calls to 1.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_FEAT_DECODER(torch.nn.Module):
    def __init__(self, voxcpm, fixed_timesteps):
        super(VOXCPM_FEAT_DECODER, self).__init__()
        self.voxcpm = voxcpm
        self.head_dim = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.head_dim
        self.num_heads = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.num_heads
        self.num_key_value_heads = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.num_key_value_heads
        self.num_key_value_groups = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.num_key_value_groups
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.register_buffer(
            "rope_permutation",
            torch.cat((torch.arange(self.head_dim // 2, self.head_dim), torch.arange(self.head_dim // 2))).to(torch.int32),
            persistent=False,
        )
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        hidden_size = self.voxcpm.feat_decoder.estimator.config.hidden_size
        self.rms_norm_epsilon = float(self.voxcpm.feat_decoder.estimator.config.rms_norm_eps)
        self.register_buffer(
            "rms_scale",
            torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32),
            persistent=False,
        )

        # Precompute all timestep data
        self.timesteps = fixed_timesteps
        sway_sampling_coef = 1.0
        t_span = torch.linspace(1, 0, fixed_timesteps + 1, dtype=torch.float32)
        t_span = (t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span))[1:]
        t = self.voxcpm.feat_decoder.estimator.time_embeddings(t_span[:-1])
        t = self.voxcpm.feat_decoder.estimator.time_mlp(t)
        dt = (t_span[:-1] - t_span[1:]).view(1, 1, -1)
        if self.voxcpm.feat_decoder.mean_mode:
            dt_in = self.voxcpm.feat_decoder.estimator.delta_time_mlp(self.voxcpm.feat_decoder.estimator.time_embeddings(dt)).unsqueeze(0)
        else:
            dt_in = self.voxcpm.feat_decoder.estimator.delta_time_mlp(self.voxcpm.feat_decoder.estimator.time_embeddings(torch.tensor([0], dtype=torch.float32)))
        self.register_buffer("dt", dt, persistent=False)
        self.register_buffer("t_all", (t + dt_in).unsqueeze(0), persistent=False)  # [1, timesteps-1, hidden]

        self.prefix_plus = self.voxcpm.patch_size + 1
        self.q_len = 9  # Fixed to 9 for VoxCPM1.5 CFM
        position_ids = torch.arange(self.q_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = self.voxcpm.feat_decoder.estimator.decoder.rope_emb(position_ids)
        rope_half = self.voxcpm.feat_decoder.estimator.decoder.rope_emb.dim // 2
        rope_emb_sin = torch.cat((-rope_emb_sin[:, :rope_half], rope_emb_sin[:, rope_half:]), dim=-1)
        self.register_buffer("rope_emb_cos", rope_emb_cos.view(1, self.q_len, 1, 1, -1), persistent=False)
        self.register_buffer("rope_emb_sin", rope_emb_sin.view(1, self.q_len, 1, 1, -1), persistent=False)

        scale_factor = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.head_dim ** -0.25
        norm_factor = self.voxcpm.feat_decoder.estimator.config.hidden_size ** 0.5
        with torch.no_grad():
            for layer in self.voxcpm.feat_decoder.estimator.decoder.layers:
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                in_features = int(q_proj.in_features)
                out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
                has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
                qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
                if has_bias:
                    z = lambda feat: torch.zeros(feat, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
                    qb = q_proj.bias if q_proj.bias is not None else z(q_proj.out_features)
                    kb = k_proj.bias if k_proj.bias is not None else z(k_proj.out_features)
                    vb = v_proj.bias if v_proj.bias is not None else z(v_proj.out_features)
                    qkv.bias.copy_(torch.cat([qb * scale_factor, kb * scale_factor, vb], dim=0))
                layer.self_attn.q_out_features = int(q_proj.out_features)
                layer.self_attn.k_out_features = int(k_proj.out_features)
                layer.self_attn.v_out_features = int(v_proj.out_features)
                layer.self_attn.qkv = qkv
                del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                del layer.input_layernorm

                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj
                in_feat = gate.in_features
                out_feat = gate.out_features + up.out_features
                gate_up = torch.nn.Linear(in_feat, out_feat, bias=False)
                gate_up.weight.copy_(torch.cat([gate.weight * w, up.weight * w], dim=0))
                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

            w = self.voxcpm.feat_decoder.estimator.decoder.norm.weight.unsqueeze(0) * norm_factor
            self.voxcpm.feat_decoder.estimator.out_proj.weight.mul_(w)
            del self.voxcpm.feat_decoder.estimator.decoder.norm

        _reorder_transformer_channels(
            self.voxcpm.feat_decoder.estimator.decoder.layers,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.qk_heads,
        )

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        return simplified_layer_norm(x, self.rms_scale, self.rms_norm_epsilon)

    def rotate_half(self, x):
        return x.reshape(-1, 2, self.head_dim // 2).flip(-2).reshape_as(x)

    def _single_step(self, step, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus):
        t = self.t_all[:, step]
        dt = self.dt[..., step]
        dit_hidden_t = dit_hidden + t
        dit_hidden_t = torch.cat([dit_hidden_t, t], dim=0)
        x = self.voxcpm.feat_decoder.estimator.in_proj(random)
        x = torch.cat([x, x], dim=0)
        hidden_states = torch.cat([dit_hidden_t, feat_cond, x], dim=1)
        for layer in self.voxcpm.feat_decoder.estimator.decoder.layers:
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, self.q_len, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * self.rope_emb_cos + self.rotate_half(qk) * self.rope_emb_sin
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.q_len, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)
            attn = torch.softmax(torch.matmul(q, k), dim=-1)
            attn = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(-1, self.q_len, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        hidden_states = hidden_states[:, self.prefix_plus:]
        hidden_states = self._rms_norm(hidden_states)
        hidden_states = self.voxcpm.feat_decoder.estimator.out_proj(hidden_states)
        dphi_dt, cfg_dphi_dt = hidden_states.split([1, 1], dim=0)
        dot_product = (dphi_dt * cfg_dphi_dt).sum((1, 2), keepdim=True)
        squared_norm = cfg_dphi_dt.square().sum((1, 2), keepdim=True)
        st_star = dot_product / squared_norm
        dphi_dt = cfg_value_minus * cfg_dphi_dt * st_star + cfg_value * dphi_dt
        return random - dt * dphi_dt

    def forward(self, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus):
        # Full diffusion loop unrolled - all timesteps in one call
        for step in range(self.timesteps - 1):
            random = self._single_step([step], random, dit_hidden, feat_cond, cfg_value, cfg_value_minus)
        return random



# ══════════════════════════════════════════════════════════════════════════════
# VAE Decoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_VAE_DECODE(torch.nn.Module):
    def __init__(self, voxcpm, output_sample_rate):
        super(VOXCPM_VAE_DECODE, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self._remove_weight_norm(self.voxcpm.audio_vae.decoder)
        self.scale = float(output_sample_rate / MODEL_SAMPLE_RATE)
        self.single_decode_len = self.voxcpm.patch_size * self.voxcpm.chunk_size

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    @staticmethod
    def _remove_weight_norm(module):
        for child in module.modules():
            try:
                torch.nn.utils.remove_weight_norm(child)
            except ValueError:
                pass

    def forward(self, latent_pred):
        decode_audio = self.voxcpm.audio_vae.decode(latent_pred.transpose(-1, -2))
        if self.scale != 1.0:
            decode_audio = torch.nn.functional.interpolate(
                decode_audio,
                scale_factor=self.scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )
        decode_audio = decode_audio.clamp(-1.0, 1.0)
        audio_out_len = torch._shape_as_tensor(decode_audio)[-1].unsqueeze(0)
        if "int" in OUT_AUDIO_DTYPE.lower():
            decode_audio = (decode_audio * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16)
        elif "32" in OUT_AUDIO_DTYPE:
            decode_audio = decode_audio.float()
        else:
            decode_audio = decode_audio.half()
        return decode_audio, audio_out_len


# ══════════════════════════════════════════════════════════════════════════════
# Compact graph components
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_COMPACT_PREFILL_INPUTS(torch.nn.Module):
    """Build prefill tensors and own all reusable empty/decode constants."""

    def __init__(
        self,
        prefill,
        base_num_kv_heads,
        base_head_dim,
        residual_num_kv_heads,
        residual_head_dim,
        kv_dtype,
    ):
        super().__init__()
        self.prefill = prefill
        self.register_buffer(
            "empty_base_key",
            torch.zeros((base_num_kv_heads, 1, base_head_dim, 0), dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_base_value",
            torch.zeros((base_num_kv_heads, 1, 0, base_head_dim), dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_residual_key",
            torch.zeros((residual_num_kv_heads, 1, residual_head_dim, 0), dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_residual_value",
            torch.zeros((residual_num_kv_heads, 1, 0, residual_head_dim), dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer("zero_history", torch.zeros((1,), dtype=torch.int64), persistent=False)
        self.register_buffer(
            "zero_decode_mask",
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, prompt_text_ids, target_text_ids, feat_embed):
        outputs = self.prefill(prompt_text_ids, target_text_ids, feat_embed)
        zero_dependency = torch._shape_as_tensor(feat_embed)[1].unsqueeze(0) * 0
        return (
            *outputs,
            self.empty_base_key + zero_dependency.to(self.empty_base_key.dtype),
            self.empty_base_value + zero_dependency.to(self.empty_base_value.dtype),
            self.empty_residual_key + zero_dependency.to(self.empty_residual_key.dtype),
            self.empty_residual_value + zero_dependency.to(self.empty_residual_value.dtype),
            self.zero_history + zero_dependency,
            self.zero_decode_mask + zero_dependency.to(self.zero_decode_mask.dtype),
        )


class VOXCPM_COMPACT_DECODE_INPUTS(torch.nn.Module):
    """Internal decode-position RoPE, mask, mode switch, and length increment."""

    def __init__(self, voxcpm, max_seq_len):
        super().__init__()
        self.rotary = VOXCPM_ROTARY_MASK_DECODE(voxcpm, max_seq_len)
        self.register_buffer("zero_concat_text_len", torch.zeros((1,), dtype=torch.int64), persistent=False)
        self.register_buffer(
            "zero_attention_mask",
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, kv_seq_len):
        rotary_cos, rotary_sin, kv_seq_len_out = self.rotary(kv_seq_len)
        zero_dependency = kv_seq_len * 0
        return (
            rotary_cos,
            rotary_sin,
            self.zero_concat_text_len + zero_dependency,
            self.zero_attention_mask + zero_dependency.to(self.zero_attention_mask.dtype),
            kv_seq_len_out,
        )


class VOXCPM_COMPACT_MAIN_CORE(torch.nn.Module):
    """Expose Main state and logits while removing its unused internal RNG output."""

    def __init__(self, main, kv_tensor_count):
        super().__init__()
        self.main = main
        self.kv_tensor_count = kv_tensor_count

    def forward(self, *inputs):
        outputs = self.main(*inputs)
        return (*outputs[:self.kv_tensor_count], outputs[-2], outputs[-1])


class VOXCPM_CONDITION_SELECTOR(torch.nn.Module):
    def __init__(self, no_prompt_feat_cond):
        super().__init__()
        self.register_buffer("no_prompt_feat_cond", no_prompt_feat_cond, persistent=False)

    def forward(self, feat_cond, use_prompt):
        condition = use_prompt.reshape(1, 1, 1).to(torch.bool)
        return torch.where(condition, feat_cond, self.no_prompt_feat_cond)


class VOXCPM_LATENT_ACCUMULATOR(torch.nn.Module):
    def forward(self, generated_latents, current_latent):
        return torch.cat((generated_latents, current_latent), dim=1)


class VOXCPM_VAE_DECODE_STREAM(torch.nn.Module):
    def __init__(self, vae_decoder):
        super().__init__()
        self.vae_decoder = vae_decoder

    def forward(self, previous_latent, current_latent):
        return self.vae_decoder(torch.cat((previous_latent, current_latent), dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# Concat Utility (for streaming VAE decode only)
# ══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Metadata helpers — export the pipeline geometry once so inference never has to
# hand-duplicate the fixed-at-export constants (mirrors the ASR repo).
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
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

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in onnx_model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            onnx_model.metadata_props.add(key=key, value=value)
    onnx.save(onnx_model, onnx_path)


def replace_onnx_metadata(onnx_path, metadata):
    """Replace the metadata carrier contract without loading external weights."""
    import onnx

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    del onnx_model.metadata_props[:]
    for key, value in metadata.items():
        onnx_model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(onnx_model, onnx_path)


class METADATA_CARRIER(torch.nn.Module):
    """Tiny identity graph that carries the static package contract."""

    def forward(self, marker):
        return marker


COMPACT_COMPONENT_FILES = {
    "reference_vae": "VoxCPM_Component_ReferenceVAEEncoder.onnx",
    "feat_encoder": "VoxCPM_Component_FeatEncoderCond.onnx",
    "prefill_inputs": "VoxCPM_Component_PrefillInputs.onnx",
    "condition_selector": "VoxCPM_Component_ConditionSelector.onnx",
    "main_core": "VoxCPM_Component_MainCore.onnx",
    "feat_decoder": "VoxCPM_Component_FeatDecoder.onnx",
    "decode_inputs": "VoxCPM_Component_DecodeInputs.onnx",
    "latent_accumulator": "VoxCPM_Component_LatentAccumulator.onnx",
}
COMPACT_MODEL_FILES = {
    "reference_preprocess": "VoxCPM_ReferencePreprocess.onnx",
    "main_prefill": "VoxCPM_MainPrefill.onnx",
    "decode_step": "VoxCPM_DecodeStep.onnx",
    "vae_decoder": "VoxCPM_VAE_Decoder.onnx",
    "vae_decoder_stream": "VoxCPM_VAE_Decoder_Stream.onnx",
    "metadata": "VoxCPM_Metadata.onnx",
}
VOXCPM_STOP_TOKEN_IDS = (1,)
ONNX_OPSET = 20


def _export_component(
    module,
    inputs,
    path,
    input_names,
    output_names,
    dynamic_axes=None,
):
    module.eval()
    torch.onnx.export(
        module,
        inputs,
        str(path),
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=ONNX_OPSET,
        dynamo=False,
        external_data=True,
    )


def _compact_kv_layout(
    base_key,
    base_value,
    residual_key,
    residual_value,
    base_layers,
    residual_layers,
):
    tensors = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    total_layers = base_layers + residual_layers
    for state_name, sequence_axis in (("key", 3), ("value", 2)):
        for layer_index in range(total_layers):
            is_base = layer_index < base_layers
            if state_name == "key":
                tensor = base_key if is_base else residual_key
            else:
                tensor = base_value if is_base else residual_value
            input_name = f"in_{state_name}_{layer_index}"
            output_name = f"out_{state_name}_{layer_index}"
            tensors.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            dynamic_axes[input_name] = {sequence_axis: "history_len"}
            dynamic_axes[output_name] = {sequence_axis: "kv_seq_len"}
    return tensors, input_names, output_names, dynamic_axes


def _assert_flip_rotation_order(*head_dims):
    for head_dim in head_dims:
        values = torch.arange(head_dim * 3, dtype=torch.float32).view(3, head_dim)
        permutation = torch.cat(
            (torch.arange(head_dim // 2, head_dim), torch.arange(head_dim // 2))
        )
        expected = torch.index_select(values, -1, permutation)
        actual = values.reshape(-1, 2, head_dim // 2).flip(-2).reshape_as(values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _stamp_raw_metadata(metadata):
    graph_paths = sorted(raw_onnx_folder.glob("*.onnx"))
    for graph_path in graph_paths:
        write_onnx_metadata(str(graph_path), metadata)
    print(f"\n[Metadata] Stamped {len(metadata)} keys into {len(graph_paths)} component graph(s).")


def _compose_compact_graphs(component_paths, kv_in_names, kv_out_names, base_layers):
    reference_path = onnx_folder / COMPACT_MODEL_FILES["reference_preprocess"]
    main_prefill_path = onnx_folder / COMPACT_MODEL_FILES["main_prefill"]
    decode_step_path = onnx_folder / COMPACT_MODEL_FILES["decode_step"]

    compose_graphs(
        [
            GraphComponent(component_paths["reference_vae"], "reference_vae/", {}),
            GraphComponent(
                component_paths["feat_encoder"],
                "feat_encoder/",
                {"audio_feat": "audio_feat"},
            ),
        ],
        reference_path,
        ("feat_embed", "feat_cond"),
        graph_name="voxcpm_reference_preprocess",
    )

    empty_connections = {}
    total_layers = len(kv_in_names) // 2
    for layer_index in range(total_layers):
        empty_connections[f"in_key_{layer_index}"] = (
            "empty_base_key" if layer_index < base_layers else "empty_residual_key"
        )
        empty_connections[f"in_value_{layer_index}"] = (
            "empty_base_value" if layer_index < base_layers else "empty_residual_value"
        )
    empty_connections.update(
        {
            "concat_text_len": "concat_text_len",
            "hidden_states": "hidden_states",
            "rotary_cos": "rotary_cos",
            "rotary_sin": "rotary_sin",
            "attention_mask": "attention_mask",
        }
    )
    compose_graphs(
        [
            GraphComponent(component_paths["prefill_inputs"], "prefill/", {}),
            GraphComponent(component_paths["condition_selector"], "condition/", {}),
            GraphComponent(component_paths["main_core"], "main/", empty_connections),
            GraphComponent(
                component_paths["feat_decoder"],
                "feat_decoder/",
                {
                    "dit_hidden": "dit_hidden",
                    "feat_cond": "selected_feat_cond",
                },
            ),
        ],
        main_prefill_path,
        (*kv_out_names, "latent_pred", "stop_flag", "kv_seq_len"),
        graph_name="voxcpm_main_prefill",
    )

    decode_main_connections = {
        "feat_embed": "feat_embed",
        "concat_text_len": "zero_concat_text_len",
        "hidden_states": "feat_embed",
        "rotary_cos": "rotary_cos",
        "rotary_sin": "rotary_sin",
        "attention_mask": "zero_attention_mask",
    }
    compose_graphs(
        [
            GraphComponent(
                component_paths["feat_encoder"],
                "feat_encoder/",
                {},
                {"audio_feat": "previous_latent"},
            ),
            GraphComponent(component_paths["decode_inputs"], "decode_inputs/", {}),
            GraphComponent(component_paths["main_core"], "main/", decode_main_connections),
            GraphComponent(
                component_paths["feat_decoder"],
                "feat_decoder/",
                {
                    "dit_hidden": "dit_hidden",
                    "feat_cond": "feat_cond",
                },
            ),
            GraphComponent(
                component_paths["latent_accumulator"],
                "accumulator/",
                {"current_latent": "latent_pred"},
            ),
        ],
        decode_step_path,
        (*kv_out_names, "latent_pred", "stop_flag", "kv_seq_len_out", "generated_latents"),
        graph_name="voxcpm_decode_step",
        input_names=(
            *kv_in_names,
            "previous_latent",
            "kv_seq_len",
            "noise",
            "cfg_value",
            "cfg_value_minus",
            "generated_latents_in",
        ),
    )

    for path in component_paths.values():
        path.unlink()
        path.with_name(path.name + ".data").unlink(missing_ok=True)


def export_compact_voxcpm():
    print("Compact export start ...")
    if raw_onnx_folder.exists():
        shutil.rmtree(raw_onnx_folder)
    raw_onnx_folder.mkdir(parents=True)
    if onnx_folder.exists():
        shutil.rmtree(onnx_folder)

    with torch.inference_mode():
        model = VoxCPM.from_pretrained(
            path_voxcpm,
            load_denoiser=False,
            optimize=False,
        ).tts_model
        model = model.float().to("cpu").eval()

        base_layers = int(model.base_lm.config.num_hidden_layers)
        residual_layers = int(model.residual_lm.config.num_hidden_layers)
        total_layers = base_layers + residual_layers
        kv_tensor_count = total_layers * 2
        base_head_dim = int(model.base_lm.layers[0].self_attn.head_dim)
        base_num_kv_heads = int(model.base_lm.layers[0].self_attn.num_key_value_heads)
        residual_head_dim = int(model.residual_lm.layers[0].self_attn.head_dim)
        residual_num_kv_heads = int(
            model.residual_lm.layers[0].self_attn.num_key_value_heads
        )
        hidden_size = int(model.base_lm.embed_tokens.embedding_dim)
        feat_hidden_size = int(model.feat_encoder.config.hidden_size)
        patch_size = int(model.patch_size)
        chunk_size = int(model.chunk_size)
        feat_dim = int(model.feat_dim)
        feat_in_channels = int(model.feat_decoder.in_channels)
        cond_proj_out = int(model.feat_decoder.estimator.cond_proj.out_features)
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        samples_per_latent_frame = (
            patch_size * chunk_size * OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE
        )
        samples_per_latent_frame = int(samples_per_latent_frame)

        _assert_flip_rotation_order(base_head_dim, residual_head_dim)

        # Every destructive fusion wrapper is constructed exactly once.
        vae_encoder = VOXCPM_VAE_ENCODER(model, IN_SAMPLE_RATE)
        feat_encoder = VOXCPM_FEAT_ENCODER_COND(
            model,
            MAX_PROMPT_AUDIO_LEN,
            IN_SAMPLE_RATE,
        )
        default_audio_feat = torch.zeros((1, patch_size, feat_dim), dtype=torch.float32)
        _, no_prompt_feat_cond = feat_encoder(default_audio_feat)
        condition_selector = VOXCPM_CONDITION_SELECTOR(no_prompt_feat_cond.detach())
        prefill_inputs = VOXCPM_COMPACT_PREFILL_INPUTS(
            VOXCPM_PREFILL(model, MAX_SEQ_LEN),
            base_num_kv_heads,
            base_head_dim,
            residual_num_kv_heads,
            residual_head_dim,
            kv_dtype,
        )
        decode_inputs = VOXCPM_COMPACT_DECODE_INPUTS(model, MAX_SEQ_LEN)
        main_core = VOXCPM_COMPACT_MAIN_CORE(
            VOXCPM_MAIN(model, MAX_SEQ_LEN),
            kv_tensor_count,
        )
        feat_decoder = VOXCPM_FEAT_DECODER(model, FIXED_TIMESTEPS)
        vae_decoder = VOXCPM_VAE_DECODE(model, OUT_SAMPLE_RATE)
        vae_decoder_stream = VOXCPM_VAE_DECODE_STREAM(vae_decoder)
        latent_accumulator = VOXCPM_LATENT_ACCUMULATOR()

        metadata = build_model_metadata(
            {
                "graph_layout": "compact_prefill_decode_v2",
                "model_file_name_reference_preprocess": COMPACT_MODEL_FILES["reference_preprocess"],
                "model_file_name_main_prefill": COMPACT_MODEL_FILES["main_prefill"],
                "model_file_name_decode_step": COMPACT_MODEL_FILES["decode_step"],
                "model_file_name_vae_decoder": COMPACT_MODEL_FILES["vae_decoder"],
                "model_file_name_vae_decoder_stream": COMPACT_MODEL_FILES["vae_decoder_stream"],
                "shared_initializer_model_file": SHARED_MODEL_NAME,
                "shared_initializer_data_file": SHARED_DATA_NAME,
                "in_sample_rate": IN_SAMPLE_RATE,
                "out_sample_rate": OUT_SAMPLE_RATE,
                "max_seq_len": MAX_SEQ_LEN,
                "stop_token_ids": VOXCPM_STOP_TOKEN_IDS,
                "streaming_crop_samples": samples_per_latent_frame,
                "use_f16_kv": USE_F16_KV,
                "compute_in_f32": COMPUTE_IN_F32,
            },
        )

        component_paths = {
            name: raw_onnx_folder / file_name
            for name, file_name in COMPACT_COMPONENT_FILES.items()
        }

        prompt_audio = torch.zeros(
            (1, 1, MAX_PROMPT_AUDIO_LEN),
            dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()],
        )
        _export_component(
            vae_encoder,
            (prompt_audio,),
            component_paths["reference_vae"],
            ("prompt_audio",),
            ("audio_feat",),
            {
                "prompt_audio": {2: "audio_len"},
                "audio_feat": {0: "audio_feat_len"},
            },
        )
        del prompt_audio

        audio_feat = torch.zeros((20, patch_size, feat_dim), dtype=torch.float32)
        _export_component(
            feat_encoder,
            (audio_feat,),
            component_paths["feat_encoder"],
            ("audio_feat",),
            ("feat_embed", "feat_cond"),
            {
                "audio_feat": {0: "audio_feat_len"},
                "feat_embed": {1: "audio_feat_len"},
            },
        )
        del audio_feat

        prompt_text_ids = torch.zeros((1, 5), dtype=torch.int32)
        target_text_ids = torch.zeros((1, 10), dtype=torch.int32)
        feat_embed_dummy = torch.zeros((1, 20, hidden_size), dtype=torch.float32)
        _export_component(
            prefill_inputs,
            (prompt_text_ids, target_text_ids, feat_embed_dummy),
            component_paths["prefill_inputs"],
            ("prompt_text_ids", "target_text_ids", "feat_embed"),
            (
                "hidden_states",
                "concat_text_len",
                "rotary_cos",
                "rotary_sin",
                "attention_mask",
                "kv_seq_len",
                "empty_base_key",
                "empty_base_value",
                "empty_residual_key",
                "empty_residual_value",
                "zero_history",
                "zero_decode_mask",
            ),
            {
                "prompt_text_ids": {1: "prompt_len"},
                "target_text_ids": {1: "target_len"},
                "feat_embed": {1: "audio_feat_len"},
                "hidden_states": {1: "ids_len"},
                "rotary_cos": {0: "ids_len"},
                "rotary_sin": {0: "ids_len"},
                "attention_mask": {2: "ids_len", 3: "ids_len"},
            },
        )
        del prompt_text_ids, target_text_ids, feat_embed_dummy

        feat_cond_dummy = torch.zeros((2, patch_size, cond_proj_out), dtype=torch.float32)
        use_prompt_dummy = torch.ones((1,), dtype=torch.int32)
        _export_component(
            condition_selector,
            (feat_cond_dummy, use_prompt_dummy),
            component_paths["condition_selector"],
            ("feat_cond", "use_prompt"),
            ("selected_feat_cond",),
        )

        history_length = 0
        base_key = torch.zeros(
            (base_num_kv_heads, 1, base_head_dim, history_length),
            dtype=kv_dtype,
        )
        base_value = torch.zeros(
            (base_num_kv_heads, 1, history_length, base_head_dim),
            dtype=kv_dtype,
        )
        residual_key = torch.zeros(
            (residual_num_kv_heads, 1, residual_head_dim, history_length),
            dtype=kv_dtype,
        )
        residual_value = torch.zeros(
            (residual_num_kv_heads, 1, history_length, residual_head_dim),
            dtype=kv_dtype,
        )
        kv_tensors, kv_in_names, kv_out_names, kv_dynamic_axes = _compact_kv_layout(
            base_key,
            base_value,
            residual_key,
            residual_value,
            base_layers,
            residual_layers,
        )
        ids_length = 25
        concat_text_length = torch.tensor([10], dtype=torch.int64)
        main_feat_embed = torch.zeros(
            (1, ids_length - int(concat_text_length.item()), feat_hidden_size),
            dtype=torch.float32,
        )
        hidden_states = torch.ones((1, ids_length, hidden_size), dtype=torch.float32)
        rotary_cos = torch.zeros((ids_length, 1, 1, base_head_dim), dtype=torch.float32)
        rotary_sin = torch.zeros_like(rotary_cos)
        attention_mask = torch.zeros((1, 1, ids_length, ids_length), dtype=torch.float32)
        main_inputs = kv_tensors + [
            main_feat_embed,
            concat_text_length,
            hidden_states,
            rotary_cos,
            rotary_sin,
            attention_mask,
        ]
        main_input_names = kv_in_names + [
            "feat_embed",
            "concat_text_len",
            "hidden_states",
            "rotary_cos",
            "rotary_sin",
            "attention_mask",
        ]
        main_dynamic_axes = {
            **kv_dynamic_axes,
            "feat_embed": {1: "audio_feat_len"},
            "hidden_states": {1: "ids_len"},
            "rotary_cos": {0: "ids_len"},
            "rotary_sin": {0: "ids_len"},
            "attention_mask": {2: "ids_len", 3: "kv_seq_len"},
        }
        _export_component(
            main_core,
            tuple(main_inputs),
            component_paths["main_core"],
            main_input_names,
            (*kv_out_names, "dit_hidden", "stop_flag"),
            main_dynamic_axes,
        )
        del main_inputs, hidden_states, rotary_cos, rotary_sin, attention_mask

        noise = torch.ones((1, patch_size, feat_in_channels), dtype=torch.float32)
        dit_hidden = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
        cfg_value = torch.ones((1,), dtype=torch.float32)
        cfg_value_minus = torch.zeros((1,), dtype=torch.float32)
        _export_component(
            feat_decoder,
            (noise, dit_hidden, feat_cond_dummy, cfg_value, cfg_value_minus),
            component_paths["feat_decoder"],
            ("noise", "dit_hidden", "feat_cond", "cfg_value", "cfg_value_minus"),
            ("latent_pred",),
        )

        kv_seq_len = torch.tensor([ids_length], dtype=torch.int64)
        _export_component(
            decode_inputs,
            (kv_seq_len,),
            component_paths["decode_inputs"],
            ("kv_seq_len",),
            (
                "rotary_cos",
                "rotary_sin",
                "zero_concat_text_len",
                "zero_attention_mask",
                "kv_seq_len_out",
            ),
        )

        generated_latents = torch.zeros(
            (1, patch_size * 2, feat_in_channels),
            dtype=torch.float32,
        )
        current_latent = torch.zeros(
            (1, patch_size, feat_in_channels),
            dtype=torch.float32,
        )
        _export_component(
            latent_accumulator,
            (generated_latents, current_latent),
            component_paths["latent_accumulator"],
            ("generated_latents_in", "current_latent"),
            ("generated_latents",),
            {
                "generated_latents_in": {1: "generated_latent_len"},
                "generated_latents": {1: "generated_latent_len_out"},
            },
        )

        latent_pred = torch.ones(
            (1, patch_size * 2, feat_in_channels),
            dtype=torch.float32,
        )
        _export_component(
            vae_decoder,
            (latent_pred,),
            raw_onnx_folder / COMPACT_MODEL_FILES["vae_decoder"],
            ("generated_latents",),
            ("audio_out", "audio_out_len"),
            {
                "generated_latents": {1: "generated_latent_len"},
                "audio_out": {2: "audio_out_len"},
            },
        )
        _export_component(
            vae_decoder_stream,
            (current_latent, current_latent),
            raw_onnx_folder / COMPACT_MODEL_FILES["vae_decoder_stream"],
            ("previous_latent", "current_latent"),
            ("audio_out", "audio_out_len"),
        )

        metadata_marker = torch.zeros((1,), dtype=torch.int64)
        _export_component(
            METADATA_CARRIER(),
            (metadata_marker,),
            raw_onnx_folder / COMPACT_MODEL_FILES["metadata"],
            ("metadata_marker",),
            ("metadata_marker_out",),
        )

        _stamp_raw_metadata(metadata)
        del (
            model,
            vae_encoder,
            feat_encoder,
            condition_selector,
            prefill_inputs,
            decode_inputs,
            main_core,
            feat_decoder,
            vae_decoder,
            vae_decoder_stream,
            latent_accumulator,
        )
        gc.collect()

    rewrite_report = rewrite_voxcpm_onnx_folder(raw_onnx_folder, onnx_folder)
    for rewritten_model in rewrite_report["models"]:
        print(
            f"[Rewrite] {rewritten_model['model']}: "
            f"{rewritten_model['raw_nodes']} -> {rewritten_model['final_nodes']} nodes, "
            f"causal Conv={rewritten_model['conv_rewrites']}, "
            f"ConvTranspose={rewritten_model['conv_transpose_rewrites']}"
        )

    component_paths = {
        name: onnx_folder / file_name
        for name, file_name in COMPACT_COMPONENT_FILES.items()
    }
    bundle_targets = sorted(onnx_folder.glob("*.onnx"))
    bundle_stats = bundle_shared_initializers(
        onnx_folder,
        bundle_targets,
        metadata=metadata,
    )
    print(
        f"[Shared weights] {bundle_stats['initializer_references']} references -> "
        f"{bundle_stats['unique_initializers']} exact tensors; "
        f"saved {bundle_stats['deduplicated_bytes'] / (1024 ** 2):.2f} MiB."
    )

    _compose_compact_graphs(
        component_paths,
        kv_in_names,
        kv_out_names,
        base_layers,
    )
    replace_onnx_metadata(
        onnx_folder / COMPACT_MODEL_FILES["metadata"],
        metadata,
    )
    shutil.rmtree(raw_onnx_folder)
    print(f"[Cleanup] Removed temporary export folder: {raw_onnx_folder}")
    print("\nCompact export done!")

    print("\nStart compact inference via Inference_VoxCPM_ONNX.py ...")
if __name__ == "__main__":
    export_compact_voxcpm()
