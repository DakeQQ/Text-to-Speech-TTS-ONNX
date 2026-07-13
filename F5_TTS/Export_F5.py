from __future__ import annotations

import gc
import site
import subprocess
import sys
import math
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
import torchaudio
import yaml
import rjieba
from omegaconf import OmegaConf
from pypinyin import lazy_pinyin, Style
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding
from vocos.feature_extractors import EncodecFeatures, FeatureExtractor
from STFT_Process import STFT_Process  
from f5_tts.model import CFM
from f5_tts.infer.utils_infer import load_checkpoint

script_dir          = Path(__file__).resolve().parent
onnx_folder         = script_dir / "F5_ONNX"
onnx_folder.mkdir(parents=True, exist_ok=True)

use_fp16_transformer = False                                                                                 # Export the F5_Transformer.onnx in float16 format.
F5_safetensors_path  = "/home/iamj/Downloads/F5TTS_v1_Base/model_1250000.safetensors"                      # The F5-TTS model download path.           URL: https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base
vocab_path           = "/home/iamj/Downloads/F5TTS_v1_Base/vocab.txt"                                      # The F5-TTS model vocab download path.     URL: https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base
vocos_model_path     = "/home/iamj/Downloads/vocos-mel-24khz"                                              # The Vocos model download path.            URL: https://huggingface.co/charactr/vocos-mel-24khz/tree/main
onnx_model_Preprocess  = str(onnx_folder / "F5_Preprocess.onnx")                                             # The exported onnx model path.
onnx_model_Transformer = str(onnx_folder / "F5_Transformer.onnx")                                            # The exported onnx model path.
onnx_model_Decode      = str(onnx_folder / "F5_Decode.onnx")                                                 # The exported onnx model path.
onnx_model_Metadata    = str(onnx_folder / "F5_Metadata.onnx")                                               # Tiny metadata carrier graph.

# Model Parameters
DYNAMIC_AXES = True                     # Default dynamic_axes is input audio length. Note, some providers only work for static axes.
NFE_STEP = 32                           # F5-TTS model setting, 0~31, Fixed at the export process.
SAMPLE_RATE = 24000                     # F5-TTS model setting
CFG_STRENGTH = 2.0                      # F5-TTS model setting
SWAY_COEFFICIENT = -1.0                 # F5-TTS model setting
TARGET_RMS = 0.1                        # The root-mean-square value for the audio
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT. It affects the generated audio length and speech speed.
if NFE_STEP < 1:
    raise ValueError("NFE_STEP must be >= 1.")

# STFT/ISTFT Settings
N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 1024                             # Number of FFT components for the STFT process
WINDOW_LENGTH = 1024                    # Length of windowing, edit it carefully.
WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT
MAX_SIGNAL_LENGTH = 4096                # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.

OPSET = 18


python_package_path = site.getsitepackages()[-1]


# Load the vocab.txt
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)

# Export-specific local replacements for the F5 transformer and Vocos decoder.
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    return torch.arange(length, dtype=torch.int64).clamp(max=max_pos - 1).unsqueeze(0)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x / (torch.sqrt(variance + self.eps))
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        processor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: int | None = None,
        context_pre_only: bool = False,
        qk_norm: str | None = None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)])

    def forward(self, x, c=None, mask=None, rope=None, rope_cos=None, rope_sin=None, c_rope=None):
        if c is not None:
            raise NotImplementedError("The standalone F5 export path only supports DiT self-attention.")
        return self.processor(self, x, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)

    def fuse_qkv(self, scale=1.0):
        """Collapse the separate to_q/to_k/to_v projections into one GEMM (to_qkv),
        folding the attention scale (head_dim ** -0.25) into the q & k weight rows.
        Bit-identical: each fused output column is the same dot product as before."""
        fused = nn.Linear(self.dim, 3 * self.inner_dim)
        fused.weight.data = torch.cat((self.to_q.weight.data * scale, self.to_k.weight.data * scale, self.to_v.weight.data), dim=0)
        fused.bias.data = torch.cat((self.to_q.bias.data * scale, self.to_k.bias.data * scale, self.to_v.bias.data), dim=0)
        self.to_qkv = fused
        del self.to_q, self.to_k, self.to_v


def rotate_half(x, heads, head_dim, head_dim_half):
    return x.view(2, heads, -1, head_dim_half, 2).flip(-1).reshape(2, heads, -1, head_dim)


def apply_rotary(x, rope_cos, rope_sin, heads, head_dim, head_dim_half):
    return x * rope_cos + rotate_half(x, heads, head_dim, head_dim_half) * rope_sin


class AttnProcessor:
    def __init__(self, head_dim, hidden_size, heads):
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.hidden_size = hidden_size
        self.heads = heads
        self.heads_2 = heads + heads

    def __call__(self, attn: Attention, x, mask=None, rope_cos=None, rope_sin=None):
        # One fused GEMM, then a single reshape splits the packed q/k stack from v across the head axis.
        qkv = attn.to_qkv(x).view(2, -1, 3 * self.heads, self.head_dim).transpose(1, 2)
        qk, value = torch.split(qkv, [self.heads_2, self.heads], dim=1)
        # Rotate q & k together in one broadcasted op (RoPE table shared across heads), then split and
        # transpose k for the score matmul.
        qk = apply_rotary(qk, rope_cos, rope_sin, self.heads_2, self.head_dim, self.head_dim_half)
        query, key = torch.split(qk, self.heads, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2))
        if use_fp16_transformer:
            weights = torch.softmax(scores.float() * 100.0, dim=-1, dtype=torch.float32).half()
        else:
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        x = torch.matmul(weights, value).transpose(1, 2).reshape(2, -1, self.hidden_size)
        return attn.to_out[0](x)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, qk_norm=None, pe_attn_head=None, attn_backend="torch", attn_mask_enabled=True):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(dim_head, dim, heads),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope_cos=None, rope_sin=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + gate_msa * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm)
        x = x + gate_mlp * ff_output
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len):
        text_mask = text == 0
        text_drop = self.text_embed(torch.zeros_like(text))
        text = self.text_embed(text)
        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1))
        if self.extra_modeling:
            pos_idx = get_pos_embed_indices(0, seq_len, max_pos=self.precompute_max_pos)
            pos_idx = self.freqs_cis[pos_idx]
            text = text + pos_idx
            text = text.masked_fill(text_mask, 0.0)
            for block in self.text_blocks:
                text = block(text)
                text = text.masked_fill(text_mask, 0.0)
            text_drop = text_drop + pos_idx
            text_drop = text_drop.masked_fill(text_mask, 0.0)
            for block in self.text_blocks:
                text_drop = block(text_drop)
                text_drop = text_drop.masked_fill(text_mask, 0.0)
        return text, text_drop


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, drop_audio_cond=False):
        x = self.proj(torch.cat((x, cond), dim=-1))
        return self.conv_pos_embed(x) + x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers)
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.checkpoint_activations = checkpoint_activations
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(self, x, cond, cond_drop, time, rope_cos, rope_sin, mask=None):
        x = torch.cat((x, x), dim=0)
        cond = torch.cat((cond, cond_drop), dim=0)
        x = self.input_embed(x, cond)
        for block in self.transformer_blocks:
            x = block(x, time, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
        return self.proj_out(self.norm_out(x, time))


class VocosConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float, adanorm_num_embeddings: int | None = None):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = VocosAdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6) if adanorm_num_embeddings else nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        if self.adanorm:
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosAdaLayerNorm(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        return x * scale + shift


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: float | None = None,
        adanorm_num_embeddings: int | None = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = VocosAdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6) if adanorm_num_embeddings else nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                VocosConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        return self.final_layer_norm(x.transpose(1, 2))


class UnusedISTFTPlaceholder(nn.Module):
    """State-dict-compatible placeholder; actual ISTFT is handled by STFT_Process."""

    def __init__(self, n_fft: int):
        super().__init__()
        self.register_buffer("window", torch.empty(n_fft, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        raise RuntimeError("Vocos ISTFT is unused in this export path; use the custom STFT_Process ISTFT.")


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.out = nn.Linear(dim, out_dim)
        self.istft = UnusedISTFTPlaceholder(n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        return mag, p


EXPORT_CLASS_OVERRIDES = {
    "vocos.models.VocosBackbone": VocosBackbone,
    "vocos.heads.ISTFTHead": ISTFTHead,
}


def instantiate_export_class(args: Any | tuple[Any, ...], init: dict[str, Any]) -> Any:
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_path = init["class_path"]
    args_class = EXPORT_CLASS_OVERRIDES.get(class_path)
    if args_class is None:
        class_module, class_name = class_path.rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class Vocos(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config_path: str) -> Vocos:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_export_class(args=(), init=config["feature_extractor"])
        backbone = instantiate_export_class(args=(), init=config["backbone"])
        head = instantiate_export_class(args=(), init=config["head"])
        return cls(feature_extractor=feature_extractor, backbone=backbone, head=head)

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str | None = None) -> Vocos:
        config_path = repo_id + "/config.yaml"
        model_path = repo_id + "/pytorch_model.bin"
        model = cls.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        features = self.feature_extractor(audio_input, **kwargs)
        return self.decode(features, **kwargs)

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.backbone(features_input, **kwargs)
        return self.head(x)


class F5Preprocess(torch.nn.Module):
    def __init__(self, f5_model, custom_stft, nfft, n_mels, sample_rate, num_head, head_dim, target_rms, use_fp16):
        super(F5Preprocess, self).__init__()
        self.f5_text_embed = f5_model.transformer.text_embed
        self.custom_stft = custom_stft
        self.num_channels = n_mels
        self.base_rescale_factor = 1.0      # Official setting
        self.interpolation_factor = 1.0     # Official setting
        self.target_rms = target_rms
        base = 10000.0 * self.base_rescale_factor ** (head_dim / (head_dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(MAX_SIGNAL_LENGTH, dtype=torch.float32), inv_freq) / self.interpolation_factor
        freqs_cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
        freqs_sin = torch.stack((-freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
        # head axis kept at 1 so a single table broadcasts across the packed q/k head stack (2 * num_head).
        self.rope_cos = freqs_cos.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1).half()
        self.rope_sin = freqs_sin.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1).half()
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0)
        self.inv_int16 = float(1.0 / 32768.0)
        self.use_fp16 = use_fp16

    def forward(self,
                audio: torch.ShortTensor,
                text_ids: torch.IntTensor,
                max_duration: torch.LongTensor,
                ):
        audio = audio.float() * self.inv_int16
        audio_rms = torch.sqrt(torch.mean(audio * audio))
        rms_scale = torch.where(audio_rms < self.target_rms, audio_rms / self.target_rms, torch.ones_like(audio_rms))
        rms_scale = rms_scale.reshape(1)
        audio_gain = torch.where(audio_rms < self.target_rms, self.target_rms / audio_rms.clamp(min=1e-6), torch.ones_like(audio_rms))
        audio = audio * audio_gain
        mel_signal_real, mel_signal_imag = self.custom_stft(audio)
        mel_signal = torch.matmul(self.fbank, torch.sqrt(mel_signal_real * mel_signal_real + mel_signal_imag * mel_signal_imag)).transpose(1, 2).clamp(min=1e-5).log()
        mel_signal_len = mel_signal.shape[1]
        ref_signal_len = mel_signal_len - 1
        zeros = torch.zeros((1, max_duration, self.num_channels), dtype=torch.float32)
        zeros_split_A = zeros[:, :-mel_signal_len]
        zeros_split_B = zeros[:, :-text_ids.shape[-1], 0]
        mel_signal = torch.cat((mel_signal, zeros_split_A), dim=1)
        noise = torch.randn_like(zeros)
        rope_cos = self.rope_cos[:, :, :max_duration]
        rope_sin = self.rope_sin[:, :, :max_duration]
        text, text_drop = self.f5_text_embed(torch.cat((text_ids + 1, zeros_split_B.to(text_ids.dtype)), dim=-1), max_duration[0])
        cat_mel_text = torch.cat((mel_signal, text), dim=-1)
        cat_mel_text_drop = torch.cat((zeros, text_drop), dim=-1)
        if self.use_fp16:
            return noise.half(), rope_cos, rope_sin, cat_mel_text.half(), cat_mel_text_drop.half(), ref_signal_len, rms_scale
        return noise, rope_cos.float(), rope_sin.float(), cat_mel_text, cat_mel_text_drop, ref_signal_len, rms_scale


def get_epss_timesteps(n, dtype=torch.float32):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, dtype=dtype)
    return dt * torch.tensor(t, dtype=dtype)


class F5Transformer(torch.nn.Module):
    def __init__(self, f5_model, cfg, steps, sway_coef, dtype):
        super(F5Transformer, self).__init__()
        self.f5_transformer = f5_model.transformer
        self.time_mlp = f5_model.transformer.time_embed.time_mlp
        self.freq_embed_dim = 256
        self.time_mlp_dim = 1024
        self.cfg_strength = cfg
        self.cond_scale = 1.0 + cfg
        self.uncond_scale = -cfg
        self.sway_sampling_coef = sway_coef
        t = get_epss_timesteps(steps, dtype=torch.float32)
        time_step = t + self.sway_sampling_coef * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
        delta_t = torch.diff(time_step).to(dtype).view(-1, 1, 1)
        time_expand = torch.zeros((1, len(time_step), self.time_mlp_dim), dtype=torch.float32)
        half_dim = self.freq_embed_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
        for i in range(len(time_step)):
            emb = time_step[i] * emb_factor
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            time_expand[:, [i]] = self.time_mlp(emb)
        self.register_buffer("delta_t", delta_t, persistent=False)
        self.register_buffer("time_expand", time_expand.to(dtype), persistent=False)
        self.num_steps = steps

    def denoise_step(self, noise, cat_mel_text, cat_mel_text_drop, rope_cos, rope_sin, time_embed, delta_t):
        pred = self.f5_transformer(x=noise, cond=cat_mel_text, cond_drop=cat_mel_text_drop, time=time_embed, rope_cos=rope_cos, rope_sin=rope_sin)
        pred, pred_drop = torch.split(pred, [1, 1], dim=0)
        return noise + (pred * self.cond_scale + pred_drop * self.uncond_scale) * delta_t

    def forward(self,
                noise: torch.FloatTensor,
                rope_cos: torch.FloatTensor,
                rope_sin: torch.FloatTensor,
                cat_mel_text: torch.FloatTensor,
                cat_mel_text_drop: torch.FloatTensor,
                time_step: torch.LongTensor
                ):
        # The NFE loop lives in the inference driver; each call runs one denoise step, gathering this
        # step's time embedding / delta_t from the precomputed tables via the time_step index.
        return self.denoise_step(noise, cat_mel_text, cat_mel_text_drop, rope_cos, rope_sin, self.time_expand[:, time_step], self.delta_t[time_step])


class F5Decode(torch.nn.Module):
    def __init__(self, vocos, custom_istft, target_rms, use_fp16):
        super(F5Decode, self).__init__()
        self.vocos = vocos
        self.custom_istft = custom_istft
        self.target_rms = float(target_rms)
        self.use_fp16 = use_fp16

    def forward(self,
                denoised: torch.FloatTensor,
                ref_signal_len: torch.LongTensor,
                rms_scale: torch.FloatTensor,
                ):
        denoised = denoised[:, ref_signal_len:]
        if self.use_fp16:
            denoised = denoised.float()
        denoised = self.vocos.decode(denoised.transpose(1, 2))
        generated_signal = self.custom_istft(*denoised)
        generated_signal = generated_signal * rms_scale.to(generated_signal.dtype)
        return (generated_signal.clamp(min=-1.0, max=1.0) * 32767.0).to(torch.int16)


def load_model(ckpt_path):
    model_cfg = OmegaConf.load(python_package_path + "/f5_tts/configs/F5TTS_v1_Base.yaml")
    model_cls = globals()[model_cfg.model.backbone]
    model = CFM(
        transformer=model_cls(**model_cfg.model.arch, text_num_embeds=vocab_size, mel_dim=N_MELS),
        mel_spec_kwargs=dict(  # Not important here. Use the custom STFT/ISTFT instead.
            target_sample_rate=SAMPLE_RATE,
            n_mel_channels=N_MELS,
            hop_length=HOP_LENGTH,
        ),
        odeint_kwargs=dict(
            method='euler',     # Only the Euler method is implemented for ONNX here.
        ),
        vocab_char_map=vocab_char_map,
    ).to('cpu')
    return load_checkpoint(model, ckpt_path, 'cpu', use_ema=True), model_cfg.model.arch.heads, model_cfg.model.arch.dim


# From the official code
def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        segments = rjieba.cut(text)
            
        for seg in segments:
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)
    return final_text_list


# From the official code
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1
):
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    text = torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


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

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in onnx_model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            onnx_model.metadata_props.add(key=key, value=value)
    onnx.save(onnx_model, onnx_path)


class METADATA_CARRIER(torch.nn.Module):
    """Tiny identity graph whose only purpose is to carry the runtime metadata."""

    def forward(self, marker):
        return marker


# Dummy input shapes used only while tracing the export graphs.
DUMMY_AUDIO_LENGTH = 160000
DUMMY_TEXT_IDS_LENGTH = 60
DUMMY_MAX_GENERATED_LENGTH = 600
DUMMY_TEXT_EMBED_LENGTH = 512 + N_MELS
DUMMY_REFERENCE_SIGNAL_LENGTH = DUMMY_AUDIO_LENGTH // HOP_LENGTH + 1
DUMMY_MAX_DURATION = min(DUMMY_REFERENCE_SIGNAL_LENGTH + DUMMY_MAX_GENERATED_LENGTH, MAX_SIGNAL_LENGTH)


print("\n\nStart to Export the F5-TTS Preprocess Part.")
with torch.inference_mode():
    # Dummy for Export the F5_Preprocess part
    audio = torch.ones((1, 1, DUMMY_AUDIO_LENGTH), dtype=torch.int16)
    text_ids = torch.ones((1, DUMMY_TEXT_IDS_LENGTH), dtype=torch.int32)
    max_duration = torch.tensor([DUMMY_MAX_DURATION], dtype=torch.long)
    f5_model, NUM_HEAD, HIDDEN_SIZE = load_model(F5_safetensors_path)
    HEAD_DIM = HIDDEN_SIZE // NUM_HEAD
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    f5_preprocess = F5Preprocess(f5_model, custom_stft, nfft=NFFT, n_mels=N_MELS, sample_rate=SAMPLE_RATE, num_head=NUM_HEAD, head_dim=HEAD_DIM, target_rms=TARGET_RMS, use_fp16=use_fp16_transformer)
    torch.onnx.export(
        f5_preprocess,
        (audio, text_ids, max_duration),
        onnx_model_Preprocess,
        input_names=['audio', 'text_ids', 'max_duration'],
        output_names=['noise', 'rope_cos', 'rope_sin', 'cat_mel_text', 'cat_mel_text_drop', 'ref_signal_len', 'rms_scale'],
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'text_ids': {1: 'text_ids_len'},
            'noise': {1: 'max_duration'},
            'rope_cos': {2: 'max_duration'},
            'rope_sin': {2: 'max_duration'},
            'cat_mel_text': {1: 'max_duration'},
            'cat_mel_text_drop': {1: 'max_duration'}
        } if DYNAMIC_AXES else None,
        dynamo=False,
        opset_version=OPSET)
    del custom_stft
    del f5_preprocess
    del audio
    del text_ids
    del max_duration
    gc.collect()
print("\nExport Done.")


print("\n\nStart to Export the F5-TTS Transformer Part.")
with torch.inference_mode():
    scale_factor = math.pow(HEAD_DIM, -0.25)
    if use_fp16_transformer:
        print("\nNote: Exporting F5_Transformer.onnx in float16 format will take a long time.")
        scale_factor *= 0.1  # To avoid overflow in float16 format.
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Fuse q/k/v into one GEMM per block, folding the head_dim**-0.25 attention scale into q & k.
    for i in range(len(f5_model.transformer.transformer_blocks)):
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.fuse_qkv(scale_factor)

    noise = torch.ones((1, DUMMY_MAX_DURATION, N_MELS), dtype=dtype)
    rope_cos = torch.ones((2, 1, DUMMY_MAX_DURATION, HEAD_DIM), dtype=dtype)
    rope_sin = torch.ones((2, 1, DUMMY_MAX_DURATION, HEAD_DIM), dtype=dtype)
    cat_mel_text = torch.ones((1, DUMMY_MAX_DURATION, DUMMY_TEXT_EMBED_LENGTH), dtype=dtype)
    cat_mel_text_drop = torch.ones((1, DUMMY_MAX_DURATION, DUMMY_TEXT_EMBED_LENGTH), dtype=dtype)
    time_step = torch.tensor([0], dtype=torch.long)
    print('\nNote: Exporting the Transformer as a single denoise step; the inference driver runs the NFE loop.')
    f5_transformer = F5Transformer(f5_model, cfg=CFG_STRENGTH, steps=NFE_STEP, sway_coef=SWAY_COEFFICIENT, dtype=dtype)
    if use_fp16_transformer:
        f5_transformer = f5_transformer.half()
    transformer_inputs = (noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, time_step)
    transformer_input_names = ['noise', 'rope_cos', 'rope_sin', 'cat_mel_text', 'cat_mel_text_drop', 'time_step']
    transformer_output_names = ['denoised']
    torch.onnx.export(
        f5_transformer,
        transformer_inputs,
        onnx_model_Transformer,
        input_names=transformer_input_names,
        output_names=transformer_output_names,
        dynamic_axes={
            'noise': {1: 'max_duration'},
            'rope_cos': {2: 'max_duration'},
            'rope_sin': {2: 'max_duration'},
            'cat_mel_text': {1: 'max_duration'},
            'cat_mel_text_drop': {1: 'max_duration'},
            'denoised': {1: 'max_duration'}
        } if DYNAMIC_AXES else None,
        dynamo=False,
        opset_version=OPSET)
    del f5_transformer
    del noise
    del rope_cos
    del rope_sin
    del cat_mel_text
    del cat_mel_text_drop
    gc.collect()
    print("\nExport Done.")


print("\n\nStart to Export the F5-TTS Decode Part.")
with torch.inference_mode():
    # Dummy for Export the F5_Decode part
    denoised = torch.ones((1, DUMMY_MAX_DURATION, N_MELS), dtype=dtype)
    ref_signal_len = torch.tensor(DUMMY_REFERENCE_SIGNAL_LENGTH, dtype=torch.long)
    rms_scale = torch.ones((1,), dtype=torch.float32)
    custom_istft = STFT_Process(model_type='istft_A', n_fft=NFFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    # Vocos model preprocess
    vocos = Vocos.from_pretrained(vocos_model_path)
    f5_decode = F5Decode(vocos, custom_istft, target_rms=TARGET_RMS, use_fp16=use_fp16_transformer)
    torch.onnx.export(
        f5_decode,
        (denoised, ref_signal_len, rms_scale),
        onnx_model_Decode,
        input_names=['denoised', 'ref_signal_len', 'rms_scale'],
        output_names=['output_audio'],
        dynamic_axes={
            'denoised': {1: 'max_duration'},
            'output_audio': {2: 'generated_len'},
        } if DYNAMIC_AXES else None,
        dynamo=False,
        opset_version=OPSET)
    del f5_decode
    del denoised
    del ref_signal_len
    del rms_scale
    del vocos
    del custom_istft
    gc.collect()
    print("\nExport Done.")

# ── Metadata carrier + stamp the metadata onto every exported graph ──
onnx_metadata = build_model_metadata(
    {
        "f5_tts_metadata_version": 2,
        "producer": Path(__file__).name,
        "f5_safetensors_path": F5_safetensors_path,
        "vocab_path": vocab_path,
        "vocos_model_path": vocos_model_path,
        "sample_rate": SAMPLE_RATE,
        "nfe_step": NFE_STEP,
        "cfg_strength": CFG_STRENGTH,
        "sway_coefficient": SWAY_COEFFICIENT,
        "target_rms": TARGET_RMS,
        "dynamic_axes": DYNAMIC_AXES,
        "max_signal_length": MAX_SIGNAL_LENGTH,
        "use_fp16_transformer": use_fp16_transformer,
        "activations_fp16": False,
        "opset": 17,
    },
    {
        "num_mels": N_MELS,
        "nfft": NFFT,
        "hop_length": HOP_LENGTH,
        "window_length": WINDOW_LENGTH,
        "window_type": WINDOW_TYPE,
    },
    {
        "num_attention_heads": NUM_HEAD,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": vocab_size,
    },
)
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
_written = 0
for _target in sorted(str(p) for p in onnx_folder.glob("*.onnx")):
    write_onnx_metadata(_target, onnx_metadata)
    _written += 1
print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {_written} ONNX graph(s).")

print("\nExport done!")
print("\nStart running inference via F5-TTS-ONNX-Inference.py ...")
subprocess.run(
    [sys.executable, str(script_dir / "F5-TTS-ONNX-Inference.py"), "--onnx-folder", str(onnx_folder)],
    check=True,
)
