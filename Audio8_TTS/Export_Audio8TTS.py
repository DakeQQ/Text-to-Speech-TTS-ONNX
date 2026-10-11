import gc
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import types
from collections import Counter, defaultdict
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModel, AutoTokenizer  # transformers>=4.55 (README recommends 4.57.x)

from Shared_Weights import (
    DECODE_STEP_PREFIX,
    FAST_FRAME_PREFIX,
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    SLOW_DECODE_PREFIX,
    build_decode_step_graphs,
    bundle_shared_initializers,
)


# ──────────────────────────────────────────────────────────────────────────────
# Export settings
# ──────────────────────────────────────────────────────────────────────────────
download_path = str(Path.home() / 'Downloads' / 'Audio8-TTS-Preview-0.1b')
MAX_SEQ_LEN = 2048                   # Slow-AR context; fixed in rotary tables and masks. Must not exceed config.max_seq_len.
MAX_REF_SECONDS = 30                 # Longest reference clip; sizes the codec-encoder rotary table.
DO_EXPORT = True
STREAM_WINDOW_FRAMES = 32            # Decoder_Stream window (frames); larger is closer to the full decode but slower.
USE_F16_KV = True                    # Store slow-AR KV cache in float16. Mamba conv/ssm states always stay float32.
COMPUTE_IN_F32 = False               # Only affects USE_F16_KV: keep f16 KV storage but run attention matmuls in f32.
USE_F16_CODEC = False                # Trace the codec graphs under float16 autocast.
PREVENT_F16_OVERFLOW = False         # Pure float16 is not supported yet.
OPSET = 20
IN_SAMPLE_RATE = 44100               # Reference audio rate fed to ReferencePreprocess.
OUT_SAMPLE_RATE = 44100              # Generated audio rate from Decoder.
IN_AUDIO_DTYPE = "F32"               # "F16" | "F32" | "INT16"
OUT_AUDIO_DTYPE = "F32"              # "F16" | "F32" | "INT16"
REORDER_DOWNPROJ_FOR_QUANT = True    # Exact channel reorder that makes later weight-only quantization friendlier.
REORDER_OPROJ_FOR_QUANT = True       # Optional; validate audio quality when enabled.
REORDER_KEY = "absmean"              # "absmean" | "L4" | "rms" | "std"


_AUDIO_DTYPES = {"F16": torch.float16, "F32": torch.float32, "INT16": torch.int16}


# ──────────────────────────────────────────────────────────────────────────────
# Derived export paths
# ──────────────────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent
onnx_folder = script_dir / "Audio8TTS_ONNX"
onnx_folder.mkdir(parents=True, exist_ok=True)

onnx_model_Metadata = str(onnx_folder / "Audio8TTS_Metadata.onnx")
onnx_model_Reference_Preprocess = str(onnx_folder / "Audio8TTS_ReferencePreprocess.onnx")
onnx_model_Target_Preprocess = str(onnx_folder / "Audio8TTS_TargetPreprocess.onnx")
onnx_model_Decoder = str(onnx_folder / "Audio8TTS_Decoder.onnx")
onnx_model_Decoder_Stream = str(onnx_folder / "Audio8TTS_Decoder_Stream.onnx")
DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")
onnx_model_Slow_Prefill = {
    strategy: str(onnx_folder / f"Audio8TTS_SlowPrefill_{strategy}.onnx") for strategy in DECODE_STRATEGIES
}
onnx_model_Slow_Decode = {
    strategy: str(onnx_folder / f"{SLOW_DECODE_PREFIX}{strategy}.onnx") for strategy in DECODE_STRATEGIES
}
onnx_model_Fast_Frame = {
    strategy: str(onnx_folder / f"{FAST_FRAME_PREFIX}{strategy}.onnx") for strategy in DECODE_STRATEGIES
}
onnx_model_Decode_Step = {
    strategy: str(onnx_folder / f"{DECODE_STEP_PREFIX}{strategy}.onnx") for strategy in DECODE_STRATEGIES
}


# ──────────────────────────────────────────────────────────────────────────────
# Model and codec loading
# ──────────────────────────────────────────────────────────────────────────────
# The remote code's relative codec import fails for folder names containing a dot ("0.1b"),
# so the codec module is imported straight from the checkpoint folder.
if download_path not in sys.path:
    sys.path.insert(0, download_path)
from modeling_arktts_codec import (  # noqa: E402
    ArkttsCausalConv1d,
    ArkttsCausalConvTranspose1d,
    ArkttsCodec,
    ArkttsCodecWindowTransformer,
    ArkttsConvNeXtBlock,
    ArkttsSnake1d,
)


def load_audio8_model(model_path):
    # No dtype kwarg: transformers<4.56 stores an unknown `dtype=` in the config and fails to log it.
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="eager")
    model = model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if MAX_SEQ_LEN > int(model.config.max_seq_len):
        raise ValueError(f"MAX_SEQ_LEN={MAX_SEQ_LEN} exceeds config.max_seq_len={model.config.max_seq_len}.")
    with torch.no_grad():
        multiplier = float(model.slow.embedding_multiplier)
        if multiplier != 1.0:
            model.slow.embed_tokens.weight.mul_(multiplier)
            model.codebook_embeddings.weight.mul_(multiplier)
            model.slow.embedding_multiplier = 1.0
    return model


def load_audio8_codec(model_path, config):
    """Mirror ArkttsModel.load_codec without its package-relative import."""
    codec = ArkttsCodec(config)
    state = torch.load(Path(model_path) / config.codec_filename, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("generator.") for key in state):
        state = {key[len("generator."):]: value for key, value in state.items() if key.startswith("generator.")}
    state = {key: value for key, value in state.items() if not key.endswith(("freqs_cis", "causal_mask"))}
    codec.load_state_dict(state, strict=True)
    codec = codec.float().eval()
    for parameter in codec.parameters():
        parameter.requires_grad = False
    return codec


# ──────────────────────────────────────────────────────────────────────────────
# Inline export-only codec patches
# ──────────────────────────────────────────────────────────────────────────────
def _static_causal_conv_forward(self, x):
    """Causal Conv1d with a static zero prefix that exports as Concat.

    The original right `_extra_padding` is 0 whenever the input length is a multiple of the stride,
    which holds for every layer once the waveform is padded to a multiple of the codec frame size.
    """
    if self.padding > 0:
        x = torch.cat((self.left_padding, x), dim=-1)
    return self.conv(x)


def _static_causal_transpose_forward(self, x):
    """Causal ConvTranspose1d with a constant right-trim Slice."""
    x = self.conv(x)
    return x[..., :-self.right_trim] if self.right_trim > 0 else x


def _precomputed_snake_forward(self, x):
    return x + self.inv_alpha * torch.sin(self.alpha * x).pow(2)


def _fused_convnext_forward(self, x):
    residual = x
    x = self.dwconv(x).permute(0, 2, 1)
    x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
    return residual + x.permute(0, 2, 1)


def _remove_weight_norms(codec):
    parametrized = legacy = 0
    for module in list(codec.modules()):
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            parametrized += 1
        elif hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            torch.nn.utils.remove_weight_norm(module)
            legacy += 1
    return parametrized, legacy


def _fold_convnext(block):
    """Fold the LayerNorm affine into pwconv1 and the layer scale `gamma` into pwconv2."""
    dim = block.norm.normalized_shape[0]
    with torch.no_grad():
        block.pwconv1.bias.add_(block.pwconv1.weight @ block.norm.bias)
        block.pwconv1.weight.mul_(block.norm.weight.unsqueeze(0))
        block.pwconv2.weight.mul_(block.gamma.unsqueeze(1))
        block.pwconv2.bias.mul_(block.gamma)
    block.norm = torch.nn.LayerNorm(dim, eps=block.norm.eps, elementwise_affine=False)
    del block.gamma
    block.forward = types.MethodType(_fused_convnext_forward, block)


def prepare_codec_for_export(codec):
    """Remove weight norm and swap dynamic-shape codec layers for export-friendly exact equivalents."""
    parametrized, legacy = _remove_weight_norms(codec)
    stats = {
        "weight_norm_parametrized": parametrized,
        "weight_norm_legacy": legacy,
        "static_padding_convs": 0,
        "transpose_convs": 0,
        "snakes": 0,
        "fused_convnext": 0,
    }
    for module in list(codec.modules()):
        if isinstance(module, ArkttsCausalConv1d):
            padding = torch.zeros(1, module.conv.in_channels, module.padding)
            module.register_buffer("left_padding", padding, persistent=False)
            module.forward = types.MethodType(_static_causal_conv_forward, module)
            stats["static_padding_convs"] += int(module.padding > 0)
        elif isinstance(module, ArkttsCausalConvTranspose1d):
            module.right_trim = int(module.kernel_size - module.stride)
            module.forward = types.MethodType(_static_causal_transpose_forward, module)
            stats["transpose_convs"] += 1
        elif isinstance(module, ArkttsSnake1d):
            inv_alpha = (module.alpha.detach() + 1e-9).reciprocal()
            module.register_buffer("inv_alpha", inv_alpha, persistent=False)
            module.forward = types.MethodType(_precomputed_snake_forward, module)
            stats["snakes"] += 1
        elif isinstance(module, ArkttsConvNeXtBlock):
            _fold_convnext(module)
            stats["fused_convnext"] += 1
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Tensor Utility Modules
# ──────────────────────────────────────────────────────────────────────────────
def shape_dim_as_tensor(tensor, dim):
    """Return one dynamic dimension as a rank-1 int64 tensor."""
    return torch._shape_as_tensor(tensor)[dim].unsqueeze(0)


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


def half_split_permutation(num_heads, head_dim):
    """Row order that turns interleaved (x0, x1) RoPE pairs into the half-split layout, per head."""
    per_head = torch.cat((torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)))
    return torch.cat([per_head + head * head_dim for head in range(num_heads)])


def half_split_rope_tables(max_len, head_dim, base, round_to_bf16):
    """cos/sin tables for the flip-based rotate_half: cos=cat(c, c), sin=cat(-s, s)."""
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    phases = torch.outer(torch.arange(max_len).float(), frequencies[: head_dim // 2])
    cos, sin = phases.cos(), phases.sin()
    if round_to_bf16:
        # Audio8 bakes its interleaved tables in bfloat16; match that rounding exactly.
        cos, sin = cos.bfloat16().float(), sin.bfloat16().float()
    return torch.cat((cos, cos), dim=-1), torch.cat((-sin, sin), dim=-1)


def flip_rotate_half(x, head_dim):
    """cat(x2, x1) via flip on the half axis; more efficient than split()+cat() in ONNX Runtime."""
    shape = x.shape
    return x.reshape(*shape[:-1], 2, head_dim // 2).flip(-2).reshape(shape)


# ──────────────────────────────────────────────────────────────────────────────
# Codec Transformer Module
# ──────────────────────────────────────────────────────────────────────────────
class A8_CODEC_TRANSFORMER(torch.nn.Module):
    """Windowed causal codec transformer with norms, layer scales and the attention scale folded into weights.

    Mirrors ArkttsCodecWindowTransformer exactly: interleaved RoPE becomes half-split through a per-head
    q/k row permutation (q·k is invariant), causal/window mask is precomputed as an additive int8 buffer,
    and QK RoPE is pre-unsqueezed in half precision and applied jointly before splitting.
    """

    def __init__(self, window_module, max_len, channels_first_input=True, channels_first_output=None):
        super().__init__()
        for name in ("input_proj", "output_proj", "look_ahead_conv"):
            if not isinstance(getattr(window_module, name), torch.nn.Identity):
                raise ValueError(f"Codec transformer {name} is not Identity; folding is not implemented for it.")
        first = window_module.layers[0]
        attention = first.attention
        self.num_heads = attention.n_head
        self.num_kv_heads = attention.n_local_heads
        self.head_dim = attention.head_dim
        self.groups = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qk_heads = self.num_heads + self.num_kv_heads
        self.window_size = window_module.window_size
        self.channels_first_input = channels_first_input
        self.channels_first_output = (
            window_module.channels_first if channels_first_output is None else channels_first_output
        )
        self.norm_eps = first.attention_norm.eps
        self.final_norm_eps = window_module.norm.eps
        dim = first.attention_norm.weight.shape[0]
        self.register_buffer("unit_scale", torch.ones(dim), persistent=False)
        self.register_buffer("final_scale", window_module.norm.weight.detach().clone(), persistent=False)
        cos, sin = half_split_rope_tables(max_len, self.head_dim, window_module.rope_base, round_to_bf16=True)
        self.register_buffer("rope_cos", cos.unsqueeze(1).half(), persistent=False)
        self.register_buffer("rope_sin", sin.unsqueeze(1).half(), persistent=False)
        # Precompute additive causal + sliding-window int8 mask: eliminates per-call dynamic mask logic
        positions = torch.arange(max_len)
        row, column = positions.unsqueeze(1), positions.unsqueeze(0)
        allowed = column <= row
        if self.window_size is not None:
            allowed = allowed & (column > row - self.window_size)
        mask = ((~allowed).to(torch.int8) * -128).reshape(1, 1, 1, max_len, max_len)
        self.register_buffer("causal_mask", mask, persistent=False)
        self.qkv = torch.nn.ModuleList()
        self.o_proj = torch.nn.ModuleList()
        self.gate_up = torch.nn.ModuleList()
        self.down = torch.nn.ModuleList()
        q_permutation = half_split_permutation(self.num_heads, self.head_dim)
        k_permutation = half_split_permutation(self.num_kv_heads, self.head_dim)
        with torch.no_grad():
            for block in window_module.layers:
                self._fold_block(block, q_permutation, k_permutation)

    def _fold_block(self, block, q_permutation, k_permutation):
        attention, feed_forward = block.attention, block.feed_forward
        dim = attention.wqkv.in_features
        scale = self.head_dim ** -0.25
        weight = attention.wqkv.weight * block.attention_norm.weight.unsqueeze(0)
        q, k, v = weight.split((self.q_size, self.kv_size, self.kv_size), dim=0)
        qkv = torch.nn.Linear(dim, self.q_size + 2 * self.kv_size, bias=False)
        qkv.weight.copy_(torch.cat((q[q_permutation] * scale, k[k_permutation] * scale, v), dim=0))
        o_proj = torch.nn.Linear(self.q_size, dim, bias=False)
        o_proj.weight.copy_(attention.wo.weight * block.attention_layer_scale.gamma.unsqueeze(1))
        inter = feed_forward.w1.out_features
        gate_up = torch.nn.Linear(dim, 2 * inter, bias=False)
        gate_up_weight = torch.cat((feed_forward.w1.weight, feed_forward.w3.weight), dim=0)
        gate_up.weight.copy_(gate_up_weight * block.ffn_norm.weight.unsqueeze(0))
        down = torch.nn.Linear(inter, dim, bias=False)
        down.weight.copy_(feed_forward.w2.weight * block.ffn_layer_scale.gamma.unsqueeze(1))
        self.qkv.append(qkv)
        self.o_proj.append(o_proj)
        self.gate_up.append(gate_up)
        self.down.append(down)

    def forward(self, x):
        if self.channels_first_input:
            x = x.transpose(1, 2)
        length = x.shape[1]
        mask = self.causal_mask[..., :length, :length].to(x.dtype)
        cos = self.rope_cos[:length].to(x.dtype)
        sin = self.rope_sin[:length].to(x.dtype)
        for qkv, o_proj, gate_up, down in zip(self.qkv, self.o_proj, self.gate_up, self.down):
            hidden = simplified_layer_norm(x, self.unit_scale, self.norm_eps)
            qk, v = qkv(hidden).split((self.q_size + self.kv_size, self.kv_size), dim=-1)
            qk = qk.reshape(1, -1, self.qk_heads, self.head_dim)
            qk = qk * cos + flip_rotate_half(qk, self.head_dim) * sin
            q, k = qk.split((self.num_heads, self.num_kv_heads), dim=-2)
            q = q.reshape(1, -1, self.num_kv_heads, self.groups, self.head_dim).permute(0, 2, 3, 1, 4)
            k = k.permute(0, 2, 3, 1).unsqueeze(2)
            v = v.reshape(1, -1, self.num_kv_heads, self.head_dim).transpose(1, 2).unsqueeze(2)
            scores = torch.matmul(q, k) + mask
            attn = torch.matmul(torch.softmax(scores, dim=-1), v)
            attn = attn.permute(0, 3, 1, 2, 4).reshape(1, -1, self.q_size)
            x = x + o_proj(attn)
            hidden = simplified_layer_norm(x, self.unit_scale, self.norm_eps)
            gate, up = gate_up(hidden).chunk(2, dim=-1)
            x = x + down(F.silu(gate) * up)
        x = simplified_layer_norm(x, self.final_scale, self.final_norm_eps)
        return x.transpose(1, 2) if self.channels_first_output else x


# ──────────────────────────────────────────────────────────────────────────────
# Audio Pipeline Modules
# ──────────────────────────────────────────────────────────────────────────────
def fold_codebook_tables(quantizer):
    """Fold each VQ out_proj into its codebook.

    Returns one stacked [sum(sizes), dim] table, the per-codebook row offsets and clamp limits,
    and the summed out_proj bias (semantic codebook first, then the residual codebooks).
    """
    vector_quantizers = list(quantizer.semantic_quantizer.quantizers) + list(quantizer.quantizer.quantizers)
    tables, offsets, limits = [], [], []
    bias = 0.0
    offset = 0
    with torch.no_grad():
        for vq in vector_quantizers:
            projection = vq.out_proj.weight[:, :, 0]
            tables.append(vq.codebook.weight @ projection.t())
            bias = bias + vq.out_proj.bias
            offsets.append(offset)
            limits.append(vq.codebook_size - 1)
            offset += vq.codebook_size
    return torch.cat(tables, dim=0), bias, offsets, limits


class A8_VQ_ENCODE(torch.nn.Module):
    """One ArkttsVectorQuantizer level: cosine nearest-neighbour code, then out_proj of the chosen entry."""

    def __init__(self, vq):
        super().__init__()
        self.codebook = vq.codebook
        with torch.no_grad():
            normalized = F.normalize(vq.codebook.weight)
            self.codebook_normalized_t = torch.nn.Parameter(normalized.t().contiguous(), requires_grad=False)
            self.in_proj_weight_t = torch.nn.Parameter(vq.in_proj.weight[:, :, 0].t().contiguous(), requires_grad=False)
            self.in_proj_bias = torch.nn.Parameter(vq.in_proj.bias.detach().clone(), requires_grad=False)
            self.out_proj_weight_t = torch.nn.Parameter(vq.out_proj.weight[:, :, 0].t().contiguous(), requires_grad=False)
            self.out_proj_bias = torch.nn.Parameter(vq.out_proj.bias.detach().clone(), requires_grad=False)

    def forward(self, residual):
        projected = residual @ self.in_proj_weight_t + self.in_proj_bias
        flattened = F.normalize(projected, dim=-1)
        indices = (flattened @ self.codebook_normalized_t).argmax(-1)
        quantized = self.codebook(indices) @ self.out_proj_weight_t + self.out_proj_bias
        return quantized, indices


class A8_DECODER(torch.nn.Module):
    """Decode flat frame-major codec ids [1, num_codebooks * N] back to a raw audio waveform."""

    def __init__(self, codec, output_sample_rate, model_output_sample_rate, max_frames):
        super().__init__()
        quantizer = codec.quantizer
        table, bias, offsets, limits = fold_codebook_tables(quantizer)
        self.num_codebooks = len(offsets)
        self.codebook_table = torch.nn.Embedding(table.shape[0], table.shape[1])
        with torch.no_grad():
            self.codebook_table.weight.copy_(table)
        self.register_buffer("codebook_bias", bias.detach().reshape(1, 1, -1).clone(), persistent=False)
        self.register_buffer(
            "code_offsets", torch.tensor(offsets, dtype=torch.int32).reshape(1, 1, -1), persistent=False
        )
        self.register_buffer("code_max", torch.tensor(limits, dtype=torch.int32).reshape(1, 1, -1), persistent=False)
        self.register_buffer("code_min", torch.zeros(1, 1, len(limits), dtype=torch.int32), persistent=False)
        self.post_module = A8_CODEC_TRANSFORMER(quantizer.post_module, max_frames, channels_first_input=False)
        self.upsample = quantizer.upsample
        self.decoder = codec.decoder
        self.scale = output_sample_rate / model_output_sample_rate
        self.output_dtype = OUT_AUDIO_DTYPE.upper()
        self.static_conv_fusion_count = sum(
            int(isinstance(module, ArkttsCausalConv1d) and module.padding > 0)
            for module in list(self.upsample.modules()) + list(self.decoder.modules())
        )

    def forward(self, generated_codec):
        codes = generated_codec.reshape(1, -1, self.num_codebooks).int()
        codes = torch.minimum(torch.maximum(codes, self.code_min), self.code_max) + self.code_offsets
        latent = self.codebook_table(codes).sum(dim=2) + self.codebook_bias
        hidden = self.post_module(latent)
        generated_wav = self.decoder(self.upsample(hidden))
        if self.scale != 1.0:
            generated_wav = F.interpolate(
                generated_wav,
                scale_factor=self.scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )
        if self.output_dtype == "INT16":
            # The codec ends in Tanh, so the PCM scale cannot be folded into the last conv as in Qwen.
            generated_wav = (generated_wav * 32767.0).clamp(min=-32768.0, max=32767.0).to(torch.int16)
        elif self.output_dtype == "F32":
            generated_wav = generated_wav.clamp(min=-1.0, max=1.0).float()
        else:
            generated_wav = generated_wav.clamp(min=-1.0, max=1.0).half()
        generated_len = shape_dim_as_tensor(generated_wav, -1)
        return generated_wav, generated_len


class A8_REFERENCE_PREPROCESS(torch.nn.Module):
    """Reference waveform to codec ids: ref_codes [1, num_codebooks, T] (int32) and ref_code_len."""

    def __init__(self, codec, in_sample_rate, max_seconds):
        super().__init__()
        self.frame_length = int(codec.frame_length)
        model_sample_rate = int(codec.sample_rate)
        max_code_frames = math.ceil(max_seconds * model_sample_rate / self.frame_length)
        encoder_frames_per_code = self.frame_length // int(codec.hop_length)
        for name, module in list(codec.encoder.named_modules()):
            if isinstance(module, ArkttsCodecWindowTransformer):
                parent_name, _, child_name = name.rpartition(".")
                parent = codec.encoder.get_submodule(parent_name) if parent_name else codec.encoder
                setattr(parent, child_name, A8_CODEC_TRANSFORMER(module, max_code_frames * encoder_frames_per_code))
        self.encoder = codec.encoder
        self.downsample = codec.quantizer.downsample
        self.pre_module = A8_CODEC_TRANSFORMER(
            codec.quantizer.pre_module, max_code_frames, channels_first_output=False
        )
        vector_quantizers = (
            list(codec.quantizer.semantic_quantizer.quantizers) + list(codec.quantizer.quantizer.quantizers)
        )
        self.vq_layers = torch.nn.ModuleList(A8_VQ_ENCODE(vq) for vq in vector_quantizers)
        self.register_buffer("right_padding", torch.zeros(1, 1, self.frame_length - 1), persistent=False)
        self.sr_scale = float(model_sample_rate / in_sample_rate)
        self.inv_int16 = 1.0 / 32768.0
        self.integer_input = "int" in IN_AUDIO_DTYPE.lower()
        self.input_scale_folded = self.integer_input and self.sr_scale == 1.0
        if self.input_scale_folded:
            first_conv = self.encoder.block[0]
            if not isinstance(first_conv, ArkttsCausalConv1d):
                raise TypeError("Expected the codec encoder to start with ArkttsCausalConv1d.")
            with torch.no_grad():
                first_conv.conv.weight.mul_(self.inv_int16)
        self.static_conv_fusion_count = sum(
            int(isinstance(module, ArkttsCausalConv1d) and module.padding > 0)
            for module in list(self.encoder.modules()) + list(self.downsample.modules())
        )

    def forward(self, prompt_audio):
        prompt_audio = prompt_audio.float()
        if self.sr_scale < 1.0:
            prompt_audio = F.interpolate(
                prompt_audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )
        if self.integer_input and not self.input_scale_folded:
            prompt_audio = prompt_audio * self.inv_int16
        if self.sr_scale > 1.0:
            prompt_audio = F.interpolate(
                prompt_audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )
        audio_len = shape_dim_as_tensor(prompt_audio, -1)
        padded_len = (audio_len + (self.frame_length - 1)) // self.frame_length * self.frame_length
        # HF encode right-pads to whole codec frames, which also makes every causal conv's right padding zero.
        prompt_audio = torch.cat((prompt_audio, self.right_padding), dim=-1)[..., :padded_len]
        residual = self.pre_module(self.downsample(self.encoder(prompt_audio)))
        codes = []
        last_level = len(self.vq_layers) - 1
        for level, vq in enumerate(self.vq_layers):
            quantized, indices = vq(residual)
            codes.append(indices)
            if level < last_level:
                residual = residual - quantized
        ref_codes = torch.stack(codes, dim=1).int()
        ref_code_len = shape_dim_as_tensor(ref_codes, -1)
        return ref_codes, ref_code_len


# ──────────────────────────────────────────────────────────────────────────────
# Slow AR Embedding, Rotary & Mask Modules
# ──────────────────────────────────────────────────────────────────────────────
class A8_TEXT_EMBED(torch.nn.Module):
    """Text token ids [1, L] -> slow-AR input embedding."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.slow.embed_tokens
        multiplier = float(getattr(model.slow, "embedding_multiplier", 1.0))
        if multiplier != 1.0:
            with torch.no_grad():
                self.embed_tokens.weight.mul_(multiplier)
                if hasattr(model, "codebook_embeddings"):
                    model.codebook_embeddings.weight.mul_(multiplier)
                model.slow.embedding_multiplier = 1.0

    def forward(self, ids):
        return self.embed_tokens(ids)


class A8_CODES_EMBED(torch.nn.Module):
    """Codec columns [1, num_codebooks, T] -> slow-AR input embedding.

    Every column is semantic (row 0 = c0 + semantic_begin_id), so HF's semantic gate is always true.
    Reference codes and generated frames share this path: a generated frame has c0 = semantic token.
    """

    def __init__(self, model):
        super().__init__()
        config = model.config
        self.embed_tokens = model.slow.embed_tokens
        self.codebook_embeddings = model.codebook_embeddings
        multiplier = float(getattr(model.slow, "embedding_multiplier", 1.0))
        if multiplier != 1.0:
            with torch.no_grad():
                self.embed_tokens.weight.mul_(multiplier)
                self.codebook_embeddings.weight.mul_(multiplier)
                model.slow.embedding_multiplier = 1.0
        self.num_codebooks = int(config.num_codebooks)
        self.semantic_begin_id = int(config.semantic_begin_id)
        offsets = torch.arange(self.num_codebooks, dtype=torch.int32) * int(config.codebook_size)
        self.register_buffer("codebook_offsets", offsets.reshape(1, 1, -1), persistent=False)

    def forward(self, codes):
        codes = codes.int().transpose(1, 2)
        semantic = self.embed_tokens(codes[..., 0] + self.semantic_begin_id)
        codebook_sum = self.codebook_embeddings(codes + self.codebook_offsets).sum(dim=2)
        return semantic + codebook_sum


class A8_TARGET_PREPROCESS(torch.nn.Module):
    """[prefix text | reference codes | suffix text] -> slow-AR prompt embedding and its length.

    Zero-shot passes T = 0 reference codes and an empty suffix.
    """

    def __init__(self, model):
        super().__init__()
        self.text_embed = A8_TEXT_EMBED(model)
        self.codes_embed = A8_CODES_EMBED(model)

    def forward(self, prefix_ids, ref_codes, suffix_ids):
        prefix_emb = self.text_embed(prefix_ids)
        codes_emb = self.codes_embed(ref_codes)
        suffix_emb = self.text_embed(suffix_ids)
        hidden_states = torch.cat((prefix_emb, codes_emb, suffix_emb), dim=1)
        return hidden_states, shape_dim_as_tensor(hidden_states, 1)


class A8_SLOW_ROTARY_MASK_PREFILL(torch.nn.Module):
    """Half-split RoPE tables (FalconH1RotaryEmbedding, fp32) plus Qwen's int8 causal mask table."""

    def __init__(self, model, max_seq_len):
        super().__init__()
        rotary = model.slow.rotary_emb
        inv_freq = rotary.inv_freq.detach().float()
        scaling = float(rotary.attention_scaling)
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        phases = positions * inv_freq
        cos = (phases.cos() * scaling).reshape(1, max_seq_len, 1, 1, -1)
        sin = (phases.sin() * scaling).reshape(1, max_seq_len, 1, 1, -1)
        # Kept in fp32: HF computes these tables in fp32, and half precision would break greedy parity.
        self.register_buffer("cos_rotary_pos_emb", torch.cat((cos, cos), dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat((-sin, sin), dim=-1).half(), persistent=False)
        self.mask_dtype = torch.float16 if (USE_F16_KV and not COMPUTE_IN_F32) else torch.float32
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].to(self.mask_dtype)
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].to(self.mask_dtype)
        attention_mask = self.attention_mask[..., history_len:kv_seq_len, :kv_seq_len].to(self.mask_dtype)
        return rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, kv_seq_len


def mamba_segment_sum(x, lower_strict, lower):
    """FalconH1 `segment_sum`: out[..., i, j] = sum(x[..., j+1:i+1]) for i >= j, else -inf (stable masked cumsum)."""
    x = torch.where(lower_strict, x.unsqueeze(-1), 0.0).cumsum(dim=-2)
    return x.masked_fill(~lower, float("-inf"))


class A8_MAMBA_SSD(torch.nn.Module):
    """Mamba2 SSD scan shared by every slow layer (n_groups = 1, fp32), matching FalconH1Mixer.torch_forward.

    `prefill` is the naive chunked SSD from a zero state with the broadcast-sum contractions written as
    MatMuls; `decode` is the exact one-token recurrence used when HF has precomputed states.
    """

    def __init__(self, num_heads, head_dim, state_size, chunk_size, max_seq_len):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.state_size = int(state_size)
        self.chunk_size = int(chunk_size)
        chunk_positions = torch.arange(self.chunk_size)
        step_positions = torch.arange(math.ceil(max_seq_len / self.chunk_size) + 1)
        self.register_buffer("chunk_lower_strict", chunk_positions[:, None] > chunk_positions[None, :], persistent=False)
        self.register_buffer("chunk_lower", chunk_positions[:, None] >= chunk_positions[None, :], persistent=False)
        self.register_buffer("step_lower_strict", step_positions[:, None] > step_positions[None, :], persistent=False)
        self.register_buffer("step_lower", step_positions[:, None] >= step_positions[None, :], persistent=False)
        pad = self.chunk_size - 1
        self.register_buffer("x_pad", torch.zeros(1, pad, self.num_heads, self.head_dim), persistent=False)
        self.register_buffer("head_pad", torch.zeros(1, pad, self.num_heads), persistent=False)
        self.register_buffer("state_pad", torch.zeros(1, pad, self.state_size), persistent=False)
        self.register_buffer(
            "zero_state", torch.zeros(1, self.num_heads, 1, self.head_dim * self.state_size), persistent=False
        )

    def prefill(self, x, dt, A, B, C, D):
        """x [1, L, H, d], dt [1, L, H] (after softplus), A/D [H], B/C [1, L, n] -> y [1, L, H*d], state [1, H, d, n]."""
        chunk, heads, head_dim, state_size = self.chunk_size, self.num_heads, self.head_dim, self.state_size
        seq_len = shape_dim_as_tensor(x, 1)
        num_chunks = (seq_len + (chunk - 1)) // chunk
        padded_len = num_chunks * chunk
        d_residual = x * D.unsqueeze(-1)
        # Zero padding keeps the final state exact: padded steps have dt*A = 0 and x*dt = 0.
        xdt = torch.cat((x * dt.unsqueeze(-1), self.x_pad), dim=1)[:, :padded_len]
        adt = torch.cat((dt * A, self.head_pad), dim=1)[:, :padded_len]
        B = torch.cat((B, self.state_pad), dim=1)[:, :padded_len].reshape(1, -1, chunk, state_size)
        C = torch.cat((C, self.state_pad), dim=1)[:, :padded_len].reshape(1, -1, chunk, state_size)
        xdt = xdt.reshape(1, -1, chunk, heads, head_dim).permute(0, 3, 1, 2, 4)  # [1, H, nc, c, d]
        adt = adt.reshape(1, -1, chunk, heads).permute(0, 3, 1, 2)                # [1, H, nc, c]
        a_cumsum = adt.cumsum(dim=-1)

        # Intra-chunk (diagonal blocks): G = C·B^T is head-independent because n_groups = 1.
        intra_decay = torch.exp(mamba_segment_sum(adt, self.chunk_lower_strict, self.chunk_lower))
        y_diag = ((C @ B.transpose(-1, -2)).unsqueeze(1) * intra_decay) @ xdt

        # Per-chunk end states, then the inter-chunk recurrence from a zero initial state.
        decay_states = torch.exp(a_cumsum[..., -1:] - a_cumsum)
        states = (xdt * decay_states.unsqueeze(-1)).transpose(-1, -2) @ B.unsqueeze(1)  # [1, H, nc, d, n]
        states = torch.cat((self.zero_state, states.reshape(1, heads, -1, head_dim * state_size)), dim=2)
        steps = num_chunks + 1
        chunk_ends = F.pad(a_cumsum[..., -1], (1, 0))
        inter_decay = torch.exp(
            mamba_segment_sum(chunk_ends, self.step_lower_strict[:steps, :steps], self.step_lower[:steps, :steps])
        )
        new_states = inter_decay @ states
        final_state = new_states[:, :, -1].reshape(1, heads, head_dim, state_size)
        states = new_states[:, :, :-1].reshape(1, heads, -1, head_dim, state_size)

        # State -> output per chunk (off-diagonal blocks).
        y_off = (C.unsqueeze(1) @ states.transpose(-1, -2)) * torch.exp(a_cumsum).unsqueeze(-1)
        y = (y_diag + y_off).permute(0, 2, 3, 1, 4).reshape(1, -1, heads, head_dim)[:, :seq_len] + d_residual
        return y.reshape(1, -1, heads * head_dim), final_state

    def decode(self, state, x, dt, A, B, C, D):
        """state [1, H, d, n], x [1, 1, H, d], dt [1, 1, H], B/C [1, 1, n] -> y [1, 1, H*d], new state."""
        x = x[:, 0]
        dt = dt[:, 0]
        state = state * torch.exp(dt * A)[..., None, None] + (x * dt.unsqueeze(-1)).unsqueeze(-1) * B.unsqueeze(1)
        # squeeze(-1) exports as an If node whose subgraph captures outer values; reshape keeps the graph flat.
        y = (state @ C.unsqueeze(-1)).reshape(1, self.num_heads, self.head_dim) + x * D.unsqueeze(-1)
        return y.reshape(1, 1, self.num_heads * self.head_dim), state


class A8_SLOW(torch.nn.Module):
    """Falcon-H1 slow AR (parallel attention + Mamba2 per layer) with norms and multipliers folded into weights.

    Both branches read the same normed input, so one fused projection yields [q | k | v | z | xBC | dt].
    """

    def __init__(self, model):
        super().__init__()
        slow = model.slow
        config = slow.config
        first = slow.layers[0]
        mixer = first.mamba
        if mixer.mamba_rms_norm:
            raise ValueError("mamba_rms_norm=True (gated RMSNorm) is not implemented.")
        if tuple(float(v) for v in mixer.time_step_limit) != (0.0, float("inf")):
            raise ValueError(f"time_step_limit={mixer.time_step_limit} needs a dt clamp, which is not implemented.")
        self.num_layers = len(slow.layers)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(first.self_attn.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.ssm_heads = int(mixer.num_heads)
        self.ssm_head_dim = int(mixer.head_dim)
        self.ssm_state_size = int(mixer.ssm_state_size)
        self.ssm_inner = int(mixer.intermediate_size)
        self.conv_dim = int(mixer.conv_dim)
        self.conv_keep = int(mixer.conv_kernel_size) - 1
        self.intermediate_size = int(first.feed_forward.intermediate_size)
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.split_sizes = [self.q_size + self.kv_size, self.kv_size, self.ssm_inner, self.conv_dim, self.ssm_heads]
        self.rms_norm_eps = float(config.rms_norm_eps)
        self.compute_in_f32 = COMPUTE_IN_F32
        self.save_key = [None] * self.num_layers
        self.save_value = [None] * self.num_layers
        self.register_buffer("unit_scale", torch.ones(self.hidden_size), persistent=False)
        self.register_buffer("final_scale", slow.final_layernorm.weight.detach().clone(), persistent=False)
        self.ssd = A8_MAMBA_SSD(self.ssm_heads, self.ssm_head_dim, self.ssm_state_size, mixer.chunk_size, MAX_SEQ_LEN)
        self.in_proj = torch.nn.ModuleList()
        self.o_proj = torch.nn.ModuleList()
        self.out_proj = torch.nn.ModuleList()
        self.gate_up = torch.nn.ModuleList()
        self.down = torch.nn.ModuleList()
        self.conv_weight = torch.nn.ParameterList()
        self.conv_bias = torch.nn.ParameterList()
        self.ssm_A = torch.nn.ParameterList()
        self.ssm_D = torch.nn.ParameterList()
        self.dt_bias = torch.nn.ParameterList()
        with torch.no_grad():
            for layer in slow.layers:
                self._fold_layer(layer)
            if REORDER_DOWNPROJ_FOR_QUANT:
                self._reorder_downproj_for_quant(REORDER_KEY)
            if REORDER_OPROJ_FOR_QUANT:
                self._reorder_oproj_for_quant(REORDER_KEY)
        self.semantic_output = model.semantic_output

    @staticmethod
    def _constant(tensor):
        return torch.nn.Parameter(tensor.detach().float().clone(), requires_grad=False)

    def _fold_layer(self, layer):
        attn, mixer, mlp = layer.self_attn, layer.mamba, layer.feed_forward
        projections = {
            "q_proj": attn.q_proj, "k_proj": attn.k_proj, "v_proj": attn.v_proj, "o_proj": attn.o_proj,
            "in_proj": mixer.in_proj, "out_proj": mixer.out_proj,
            "gate_proj": mlp.gate_proj, "up_proj": mlp.up_proj, "down_proj": mlp.down_proj,
        }
        biased = [name for name, module in projections.items() if module.bias is not None]
        if biased:
            raise ValueError(f"Slow-layer biases {biased} are not supported by the weight folding.")
        qk_scale = self.head_dim ** -0.25
        attn_in = float(layer.attention_in_multiplier)
        q = attn.q_proj.weight * (attn_in * qk_scale)
        k = attn.k_proj.weight * (attn_in * qk_scale * float(attn.key_multiplier))
        v = attn.v_proj.weight * attn_in
        ssm = mixer.in_proj.weight * mixer.mup_vector.reshape(-1, 1) * float(mixer.ssm_in_multiplier)
        in_proj = torch.nn.Linear(self.hidden_size, sum(self.split_sizes), bias=False)
        in_proj.weight.copy_(torch.cat((q, k, v, ssm), dim=0) * layer.input_layernorm.weight.unsqueeze(0))
        o_proj = torch.nn.Linear(self.q_size, self.hidden_size, bias=False)
        o_proj.weight.copy_(attn.o_proj.weight * float(layer.attn_out_multiplier))
        out_proj = torch.nn.Linear(self.ssm_inner, self.hidden_size, bias=False)
        out_proj.weight.copy_(mixer.out_proj.weight * float(layer.ssm_out_multiplier))
        gate_up = torch.nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        gate_up_weight = torch.cat((mlp.gate_proj.weight * float(mlp.gate_multiplier), mlp.up_proj.weight), dim=0)
        gate_up.weight.copy_(gate_up_weight * layer.pre_ff_layernorm.weight.unsqueeze(0))
        down = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        down.weight.copy_(mlp.down_proj.weight * float(mlp.down_multiplier))
        conv_bias = mixer.conv1d.bias if mixer.conv1d.bias is not None else torch.zeros(self.conv_dim)
        self.in_proj.append(in_proj)
        self.o_proj.append(o_proj)
        self.out_proj.append(out_proj)
        self.gate_up.append(gate_up)
        self.down.append(down)
        self.conv_weight.append(self._constant(mixer.conv1d.weight))
        self.conv_bias.append(self._constant(conv_bias))
        self.ssm_A.append(self._constant(-torch.exp(mixer.A_log.float())))
        self.ssm_D.append(self._constant(mixer.D))
        self.dt_bias.append(self._constant(mixer.dt_bias))

    @staticmethod
    def _channel_score(weight, key, dims):
        if key == "rms":
            return weight.square().mean(dim=dims).sqrt()
        if key == "L4":
            return weight.abs().pow(4).mean(dim=dims).pow(0.25)
        if key == "std":
            return weight.movedim(dims, tuple(range(len(dims)))).reshape(-1, weight.shape[-1]).std(0)
        if key == "absmean":
            return weight.abs().mean(dim=dims)
        raise ValueError(f"Unknown REORDER_KEY: {key}")

    def _reorder_downproj_for_quant(self, key):
        for gate_up, down in zip(self.gate_up, self.down):
            permutation = torch.argsort(self._channel_score(down.weight, key, (0,)))
            gate, up = gate_up.weight.split(self.intermediate_size, dim=0)
            gate_up.weight.copy_(torch.cat((gate[permutation], up[permutation]), dim=0))
            down.weight.copy_(down.weight[:, permutation])

    def _reorder_oproj_for_quant(self, key):
        """Permute each KV group's value channels and the matching o_proj columns together (exact)."""
        groups = self.num_key_value_groups
        v_start = self.q_size + self.kv_size
        for in_proj, o_proj in zip(self.in_proj, self.o_proj):
            by_head = o_proj.weight.view(self.hidden_size, self.num_heads, self.head_dim)
            permutations = [
                torch.argsort(self._channel_score(by_head[:, g * groups:(g + 1) * groups], key, (0, 1)))
                for g in range(self.num_key_value_heads)
            ]
            reordered = by_head.clone()
            for head in range(self.num_heads):
                reordered[:, head] = by_head[:, head, permutations[head // groups]]
            o_proj.weight.copy_(reordered.reshape_as(o_proj.weight))
            value = in_proj.weight[v_start:v_start + self.kv_size].view(-1, self.head_dim, self.hidden_size).clone()
            for g, permutation in enumerate(permutations):
                value[g] = value[g][permutation]
            in_proj.weight[v_start:v_start + self.kv_size] = value.reshape(self.kv_size, self.hidden_size)

    def forward(self, *all_inputs, decode=False):
        """Inputs: L keys, L values, L conv states, L ssm states, hidden_states, cos, sin, mask.

        `decode=True` traces the one-token Mamba recurrence; otherwise the chunked SSD from a zero state.
        """
        layers = self.num_layers
        hidden_states, rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask = all_inputs[-4:]
        conv_out, ssm_out = [None] * layers, [None] * layers
        for i in range(layers):
            residual = hidden_states
            normed = simplified_layer_norm(hidden_states, self.unit_scale, self.rms_norm_eps)
            qk, v, gate, xbc, dt = self.in_proj[i](normed).split(self.split_sizes, dim=-1)

            # Attention branch (Qwen KV layout: K [1, kv, 1, d, T], V [1, kv, 1, T, d]).
            qk = qk.reshape(1, -1, 1, self.qk_heads, self.head_dim)
            qk = qk * rotary_pos_emb_cos + flip_rotate_half(qk, self.head_dim) * rotary_pos_emb_sin
            if USE_F16_KV and not self.compute_in_f32:
                qk = qk.half()
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.reshape(1, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim).permute(0, 2, 3, 1, 4)
            v = v.reshape(1, -1, 1, self.num_key_value_heads, self.head_dim)
            if USE_F16_KV:
                if self.compute_in_f32:
                    k = k.half()
                v = v.half()
            k = torch.cat((all_inputs[i], k.permute(0, 3, 2, 4, 1)), dim=-1)
            v = torch.cat((all_inputs[i + layers], v.transpose(1, 3)), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            if USE_F16_KV and self.compute_in_f32:
                attn = torch.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1)
                attn = torch.matmul(attn, v.float())
            else:
                attn = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
                attn = torch.matmul(attn, v)
                if USE_F16_KV:
                    attn = attn.float()
            attn = self.o_proj[i](attn.permute(0, 3, 1, 2, 4).reshape(1, -1, self.q_size))

            # Mamba2 branch: the carried conv state replaces HF's zero left padding / rolling window.
            xbc_full = torch.cat((all_inputs[i + 2 * layers], xbc.transpose(1, 2)), dim=-1)
            conv_out[i] = xbc_full[..., -self.conv_keep:]
            xbc = F.silu(F.conv1d(xbc_full, self.conv_weight[i], self.conv_bias[i], groups=self.conv_dim))
            x, B, C = xbc.transpose(1, 2).split([self.ssm_inner, self.ssm_state_size, self.ssm_state_size], dim=-1)
            x = x.reshape(1, -1, self.ssm_heads, self.ssm_head_dim)
            dt = F.softplus(dt + self.dt_bias[i])
            if decode:
                y, ssm_out[i] = self.ssd.decode(
                    all_inputs[i + 3 * layers], x, dt, self.ssm_A[i], B, C, self.ssm_D[i]
                )
            else:
                y, ssm_out[i] = self.ssd.prefill(x, dt, self.ssm_A[i], B, C, self.ssm_D[i])
            mamba = self.out_proj[i](y * F.silu(gate))

            hidden_states = residual + (mamba + attn)
            residual = hidden_states
            normed = simplified_layer_norm(hidden_states, self.unit_scale, self.rms_norm_eps)
            gate, up = self.gate_up[i](normed).split(self.intermediate_size, dim=-1)
            hidden_states = residual + self.down[i](F.silu(gate) * up)

        last_hidden_state = simplified_layer_norm(hidden_states[:, -1], self.final_scale, self.rms_norm_eps)
        logits = self.semantic_output(last_hidden_state)
        return *self.save_key, *self.save_value, *conv_out, *ssm_out, last_hidden_state.unsqueeze(1), logits


class A8_SLOW_MERGED(torch.nn.Module):
    """Slow AR core plus its rotary/mask tables; appends kv_seq_len (mirrors TTS_MAIN_MERGED)."""

    def __init__(self, model, max_seq_len):
        super().__init__()
        self.rotary_mask = A8_SLOW_ROTARY_MASK_PREFILL(model, max_seq_len)
        self.transformer = A8_SLOW(model)

    def forward(self, *all_inputs, decode=False):
        hidden_states = all_inputs[-2]
        history_len = all_inputs[-1]
        ids_len = shape_dim_as_tensor(hidden_states, 1)
        rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, kv_seq_len = self.rotary_mask(ids_len, history_len)
        outputs = self.transformer(
            *all_inputs[:-2], hidden_states, rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, decode=decode
        )
        return *outputs, kv_seq_len


class A8_FAST(torch.nn.Module):
    """Fast AR (4 pre-norm blocks) over one frame with a growing KV cache; norms and q/k scale folded.

    Interleaved RoPE becomes half-split through a per-head q/k row permutation (q·k is invariant).
    The fast KV only lives inside one frame graph, so it stays float32.
    """

    def __init__(self, model):
        super().__init__()
        if not isinstance(model.fast_project_in, torch.nn.Identity):
            raise ValueError("fast_project_in is not Identity; folding it is not implemented.")
        attention = model.fast_layers[0].attention
        if attention.qk_norm or attention.wqkv.bias is not None or attention.wo.bias is not None:
            raise ValueError("Fast-AR qk_norm / attention biases are not supported by the weight folding.")
        self.num_layers = len(model.fast_layers)
        self.num_heads = int(attention.n_head)
        self.num_key_value_heads = int(attention.n_local_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = int(attention.head_dim)
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.hidden_size = int(attention.wqkv.in_features)
        self.intermediate_size = int(model.fast_layers[0].feed_forward.w1.out_features)
        self.rms_norm_eps = float(model.fast_norm.eps)
        self.save_key = [None] * self.num_layers
        self.save_value = [None] * self.num_layers
        rope = model.fast_freqs_cis.detach().float()  # already bf16-rounded by _precompute_rope
        cos, sin = rope[..., 0], rope[..., 1]
        positions = rope.shape[0]
        self.register_buffer("rope_cos", torch.cat((cos, cos), -1).reshape(1, positions, 1, 1, -1), persistent=False)
        self.register_buffer("rope_sin", torch.cat((-sin, sin), -1).reshape(1, positions, 1, 1, -1), persistent=False)
        causal = (1.0 - torch.tril(torch.ones(positions, positions))) * -128.0
        self.register_buffer("attention_mask", causal.reshape(1, 1, 1, positions, positions), persistent=False)
        self.register_buffer("unit_scale", torch.ones(self.hidden_size), persistent=False)
        self.in_proj = torch.nn.ModuleList()
        self.o_proj = torch.nn.ModuleList()
        self.gate_up = torch.nn.ModuleList()
        self.down = torch.nn.ModuleList()
        q_permutation = half_split_permutation(self.num_heads, self.head_dim)
        k_permutation = half_split_permutation(self.num_key_value_heads, self.head_dim)
        with torch.no_grad():
            for block in model.fast_layers:
                self._fold_block(block, q_permutation, k_permutation)
            if REORDER_DOWNPROJ_FOR_QUANT:
                A8_SLOW._reorder_downproj_for_quant(self, REORDER_KEY)
            if REORDER_OPROJ_FOR_QUANT:
                A8_SLOW._reorder_oproj_for_quant(self, REORDER_KEY)
            self.head = torch.nn.Linear(self.hidden_size, model.fast_output.out_features, bias=False)
            self.head.weight.copy_(model.fast_output.weight * model.fast_norm.weight.unsqueeze(0))

    _channel_score = staticmethod(A8_SLOW._channel_score)

    def _fold_block(self, block, q_permutation, k_permutation):
        attention, feed_forward = block.attention, block.feed_forward
        scale = self.head_dim ** -0.25
        weight = attention.wqkv.weight * block.attention_norm.weight.unsqueeze(0)
        q, k, v = weight.split((self.q_size, self.kv_size, self.kv_size), dim=0)
        in_proj = torch.nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        in_proj.weight.copy_(torch.cat((q[q_permutation] * scale, k[k_permutation] * scale, v), dim=0))
        o_proj = torch.nn.Linear(self.q_size, self.hidden_size, bias=False)
        o_proj.weight.copy_(attention.wo.weight)
        gate_up = torch.nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        gate_up_weight = torch.cat((feed_forward.w1.weight, feed_forward.w3.weight), dim=0)
        gate_up.weight.copy_(gate_up_weight * block.ffn_norm.weight.unsqueeze(0))
        down = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        down.weight.copy_(feed_forward.w2.weight)
        self.in_proj.append(in_proj)
        self.o_proj.append(o_proj)
        self.gate_up.append(gate_up)
        self.down.append(down)

    def forward(self, *all_inputs):
        """Inputs: L keys [1, kv, 1, d, T], L values [1, kv, 1, T, d], hidden_states [1, n, D], history_len [1].

        Returns the updated KV, the logits of the last position and kv_seq_len.
        """
        layers = self.num_layers
        hidden_states, history_len = all_inputs[-2:]
        kv_seq_len = history_len + shape_dim_as_tensor(hidden_states, 1)
        cos = self.rope_cos[:, history_len:kv_seq_len]
        sin = self.rope_sin[:, history_len:kv_seq_len]
        attention_mask = self.attention_mask[..., history_len:kv_seq_len, :kv_seq_len]
        for i in range(layers):
            residual = hidden_states
            normed = simplified_layer_norm(hidden_states, self.unit_scale, self.rms_norm_eps)
            qk, v = self.in_proj[i](normed).split((self.q_size + self.kv_size, self.kv_size), dim=-1)
            qk = qk.reshape(1, -1, 1, self.qk_heads, self.head_dim)
            qk = qk * cos + flip_rotate_half(qk, self.head_dim) * sin
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.reshape(1, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim).permute(0, 2, 3, 1, 4)
            v = v.reshape(1, -1, 1, self.num_key_value_heads, self.head_dim)
            k = torch.cat((all_inputs[i], k.permute(0, 3, 2, 4, 1)), dim=-1)
            v = torch.cat((all_inputs[i + layers], v.transpose(1, 3)), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v)
            hidden_states = residual + self.o_proj[i](attn.permute(0, 3, 1, 2, 4).reshape(1, -1, self.q_size))
            residual = hidden_states
            normed = simplified_layer_norm(hidden_states, self.unit_scale, self.rms_norm_eps)
            gate, up = self.gate_up[i](normed).split(self.intermediate_size, dim=-1)
            hidden_states = residual + self.down[i](F.silu(gate) * up)
        last_hidden_state = simplified_layer_norm(hidden_states[:, -1], self.unit_scale, self.rms_norm_eps)
        return *self.save_key, *self.save_value, self.head(last_hidden_state), kv_seq_len


class APPLY_PENALTY(torch.nn.Module):
    """Repetition penalty over the last `penalty_range` ids (repository extension; HF Audio8 has none)."""

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:]
        penalized = logits.gather(1, target_indices) * penalty_value
        return logits.scatter(1, target_indices, penalized)


class ARKTTS_TOPK_TOPP_SAMPLING(torch.nn.Module):
    """ArkttsLegacyTopKTopP: untempered cumsum over ALL logits, temperature only afterwards, exponential-race draw."""

    def __init__(self):
        super().__init__()
        self.register_buffer("neg_inf", torch.tensor(float("-inf")), persistent=False)
        self.register_buffer("min_temperature", torch.tensor(1e-5), persistent=False)

    def sample(self, logits, temperature, top_k, top_p):
        values, indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        probabilities = torch.exp(values - torch.logsumexp(logits, dim=-1, keepdim=True))
        cumulative = torch.cumsum(probabilities, dim=-1)
        # Rank 0 is always kept: its exclusive cumulative mass is exactly 0.
        keep = (cumulative <= top_p) | ((cumulative - probabilities) <= 0.0)
        temperature = torch.maximum(temperature, self.min_temperature)
        scores = torch.where(keep, values / temperature, self.neg_inf)
        cumulative = torch.cumsum(torch.softmax(scores, dim=-1), dim=-1)
        total = cumulative[:, -1:]
        winner = torch.argmax((cumulative >= torch.rand_like(total) * total).int(), dim=-1, keepdim=True)
        return torch.gather(indices, 1, winner).int()


class ARKTTS_RAS_SAMPLING(torch.nn.Module):
    """Slow-token draw with repetition-aware sampling: a (ras_top_p, ras_temperature) draw replaces a recent repeat."""

    def __init__(self, config):
        super().__init__()
        self.sampling = ARKTTS_TOPK_TOPP_SAMPLING()
        self.window = int(config.ras_window_size)
        self.eos_index = int(config.codebook_size)
        self.register_buffer("ras_top_p", torch.tensor([float(config.ras_top_p)]), persistent=False)
        self.register_buffer("ras_temperature", torch.tensor([float(config.ras_temperature)]), persistent=False)
        self.register_buffer("zero_window", torch.zeros(1, self.window, dtype=torch.int32), persistent=False)

    def forward(self, logits, temperature, top_k, top_p, save_ids):
        normal = self.sampling.sample(logits, temperature, top_k, top_p)
        high = self.sampling.sample(logits, self.ras_temperature, top_k, self.ras_top_p)
        # HF starts the window at zeros and never inserts the step-0 token, so it is cat(zeros, save_ids[1:])[-window:].
        window = torch.cat((self.zero_window, save_ids[:, 1:]), dim=-1)[:, -self.window:]
        repeated = (window == normal).int().amax(dim=-1, keepdim=True) > 0
        token = torch.where(repeated & (normal < self.eos_index), high, normal)
        return token, torch.cat((save_ids, token), dim=-1)


class A8_SLOW_PREFILL_STRATEGY(torch.nn.Module):
    """Slow-AR prefill from empty KV / Mamba states with an integrated first-token strategy."""

    def __init__(self, slow_core, strategy):
        super().__init__()
        self.slow_core = slow_core
        self.strategy = strategy
        transformer = slow_core.transformer
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        self.num_layers = transformer.num_layers
        self.register_buffer(
            "empty_key",
            torch.zeros(1, transformer.num_key_value_heads, 1, transformer.head_dim, 0, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_value",
            torch.zeros(1, transformer.num_key_value_heads, 1, 0, transformer.head_dim, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer("zero_conv", torch.zeros(1, transformer.conv_dim, transformer.conv_keep), persistent=False)
        self.register_buffer(
            "zero_ssm",
            torch.zeros(1, transformer.ssm_heads, transformer.ssm_head_dim, transformer.ssm_state_size),
            persistent=False,
        )
        self.register_buffer("zero_history_len", torch.zeros(1, dtype=torch.int64), persistent=False)
        self.sampling = ARKTTS_TOPK_TOPP_SAMPLING()

    def forward(self, *args):
        hidden_states = args[0]
        layers = self.num_layers
        outputs = self.slow_core(
            *([self.empty_key] * layers),
            *([self.empty_value] * layers),
            *([self.zero_conv] * layers),
            *([self.zero_ssm] * layers),
            hidden_states,
            self.zero_history_len,
        )
        states = outputs[:layers * 4]
        last_hidden_state, logits, kv_seq_len = outputs[-3:]
        if self.strategy == "sampling":
            temperature, top_k, top_p = args[1:]
            # HF has no RAS window at step 0 (previous is None), so the first token is a plain top-k/top-p draw.
            token = self.sampling.sample(logits, temperature, top_k, top_p)
        else:
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return *states, last_hidden_state, token, kv_seq_len


class A8_SLOW_DECODE_STRATEGY(torch.nn.Module):
    """Embed one generated frame, advance the slow-AR KV / Mamba states and select the next semantic token."""

    def __init__(self, model, slow_core, strategy):
        super().__init__()
        self.slow_core = slow_core
        self.strategy = strategy
        self.num_layers = slow_core.transformer.num_layers
        self.codes_embed = A8_CODES_EMBED(model)
        self.penalty = APPLY_PENALTY()
        self.sampling = ARKTTS_RAS_SAMPLING(model.config)

    def forward(self, *args):
        state_count = self.num_layers * 4
        states = args[:state_count]
        frame_codec_ids, history_len = args[state_count:state_count + 2]
        hidden_states = self.codes_embed(frame_codec_ids.unsqueeze(-1))
        outputs = self.slow_core(*states, hidden_states, history_len, decode=True)
        states_out = outputs[:state_count]
        last_hidden_state, logits, kv_seq_len = outputs[-3:]

        if self.strategy == "greedy":
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            return *states_out, last_hidden_state, token, kv_seq_len

        save_ids = args[state_count + 2]
        if self.strategy == "penalty_greedy":
            penalty_value, penalty_range = args[state_count + 3:state_count + 5]
            penalized_logits = self.penalty(logits, save_ids, penalty_value, penalty_range)
            use_penalty = shape_dim_as_tensor(save_ids, 1) >= penalty_range
            logits = torch.where(use_penalty, penalized_logits, logits)
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            save_ids = torch.cat([save_ids, token], dim=-1)
        else:
            temperature, top_k, top_p = args[state_count + 3:state_count + 6]
            token, save_ids = self.sampling(logits, temperature, top_k, top_p, save_ids)
        return *states_out, last_hidden_state, token, save_ids, kv_seq_len


class A8_FAST_FRAME_STRATEGY(torch.nn.Module):
    """Fully unrolled fast AR: one slow hidden state plus its semantic code yield all num_codebooks codes of a frame."""

    def __init__(self, fast_core, model, strategy):
        super().__init__()
        self.fast_core = fast_core
        self.fast_embeddings = model.fast_embeddings
        self.strategy = strategy
        self.num_layers = fast_core.num_layers
        self.num_codebooks = int(model.config.num_codebooks)
        self.max_code = int(model.config.codebook_size) - 1
        # The fast KV only lives inside one frame, so it stays float32 (matches A8_FAST).
        self.register_buffer(
            "empty_key",
            torch.zeros(1, fast_core.num_key_value_heads, 1, fast_core.head_dim, 0),
            persistent=False,
        )
        self.register_buffer(
            "empty_value",
            torch.zeros(1, fast_core.num_key_value_heads, 1, 0, fast_core.head_dim),
            persistent=False,
        )
        self.register_buffer("zero_history_len", torch.zeros(1, dtype=torch.int64), persistent=False)
        self.register_buffer("empty_ids", torch.zeros(1, 0, dtype=torch.int32), persistent=False)
        self.penalty = APPLY_PENALTY()
        self.sampling = ARKTTS_TOPK_TOPP_SAMPLING()

    def forward(self, *args):
        semantic_token_in, last_hidden_state_slow, generated_codec_in = args[:3]
        strategy_args = args[3:]
        layers = self.num_layers
        frame_cb0 = torch.clamp(semantic_token_in, 0, self.max_code)
        tokens = [frame_cb0]
        # HF discards the fast output at position 0, so the first call covers positions 0 and 1 at once.
        hidden_states = torch.cat((last_hidden_state_slow, self.fast_embeddings(frame_cb0)), dim=1)
        kv = [self.empty_key] * layers + [self.empty_value] * layers
        history_len = self.zero_history_len
        save_ids = self.empty_ids

        for position in range(1, self.num_codebooks):
            outputs = self.fast_core(*kv, hidden_states, history_len)
            kv = list(outputs[:layers * 2])
            logits, history_len = outputs[-2:]

            if self.strategy == "greedy":
                token = torch.argmax(logits, dim=-1, keepdim=True).int()
            elif self.strategy == "penalty_greedy":
                penalty_value, penalty_range = strategy_args
                if position > 1:
                    logits = self.penalty(logits, save_ids, penalty_value, penalty_range)
                token = torch.argmax(logits, dim=-1, keepdim=True).int()
                save_ids = token if position == 1 else torch.cat((save_ids, token), dim=-1)
            else:
                temperature, top_k, top_p = strategy_args
                token = self.sampling.sample(logits, temperature, top_k, top_p)

            tokens.append(token)
            if position + 1 < self.num_codebooks:
                hidden_states = self.fast_embeddings(token)

        # Single Concat replaces 9 cascading Concat operations
        frame_codec_ids = torch.cat(tokens, dim=-1)
        generated_codec = torch.cat((generated_codec_in, frame_codec_ids), dim=-1)
        return frame_codec_ids, generated_codec


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
            elif isinstance(value, dict):
                metadata[str(key)] = json.dumps(value, sort_keys=True, separators=(",", ":"))
            elif isinstance(value, (list, tuple)):
                metadata[str(key)] = ",".join(str(item) for item in value)
            else:
                metadata[str(key)] = str(value)
    return metadata


def set_onnx_metadata(model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    """Set or replace metadata_props on an in-memory ModelProto."""
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))


def write_onnx_metadata(onnx_path, metadata):
    """Replace ``metadata_props`` in place, leaving any external-weight sidecar untouched."""
    model = onnx.load(onnx_path, load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, onnx_path)


def replace_onnx_metadata(onnx_path, metadata):
    """Replace the metadata carrier contract without loading external weights."""
    model = onnx.load(onnx_path, load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, onnx_path)


def atomic_save_model(model: onnx.ModelProto, final_path: str | Path) -> None:
    """Atomically persist an in-memory ModelProto with unconditional temporary cleanup."""
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=final_path.parent,
            prefix=f".{final_path.stem}.",
            suffix=".onnx",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        onnx.save(model, str(temp_path))
        temp_path.replace(final_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def export_to_model_proto(module, args, input_names, output_names, dynamic_axes=None) -> onnx.ModelProto:
    """Export a PyTorch module directly to an in-memory ModelProto without disk footprint."""
    buf = io.BytesIO()
    torch.onnx.export(
        module,
        tuple(args),
        buf,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
    )
    buf.seek(0)
    return onnx.load_model_from_string(buf.getvalue())


def compute_graph_metrics(model: onnx.ModelProto) -> dict:
    """Compute operator histogram, node counts, and memory/copy operation metrics."""
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)

    copy_ops = {"Concat", "Pad", "Tile", "Expand", "Transpose", "ScatterND"}
    shape_ops = {"Shape", "Size", "Reshape", "Squeeze", "Unsqueeze", "Flatten"}
    layout_ops = {"Transpose"}
    indexing_ops = {"Gather", "GatherElements", "GatherND", "Slice", "Split", "ScatterND"}
    elementwise_ops = {
        "Add", "Sub", "Mul", "Div", "Pow", "Neg", "Abs", "Exp", "Log", "Sqrt", "Sin", "Cos",
        "Softplus", "Sigmoid", "Tanh", "Erf", "Relu", "Where", "Clip", "Min", "Max",
        "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal", "Not", "And", "Or", "Xor",
    }

    init_bytes = sum(len(i.raw_data) for i in graph.initializer)
    custom_nodes = [n.op_type for n in graph.node if n.domain and n.domain not in ("", "ai.onnx")]

    return {
        "node_count": len(graph.node),
        "op_histogram": dict(op_counts),
        "initializer_count": len(graph.initializer),
        "initializer_bytes": init_bytes,
        "cast_count": op_counts.get("Cast", 0),
        "shape_op_count": sum(op_counts.get(op, 0) for op in shape_ops),
        "layout_op_count": sum(op_counts.get(op, 0) for op in layout_ops),
        "indexing_op_count": sum(op_counts.get(op, 0) for op in indexing_ops),
        "elementwise_op_count": sum(op_counts.get(op, 0) for op in elementwise_ops),
        "copy_op_count": sum(op_counts.get(op, 0) for op in copy_ops),
        "custom_nodes": custom_nodes,
    }


def fuse_static_zero_prefix_convs(model: onnx.ModelProto, expected_match_count: int | None = None) -> tuple[onnx.ModelProto, int]:
    """Fold exact Constant-zero-prefix Concat nodes into standard ONNX Conv pads in-memory."""
    import numpy as np
    from onnx import helper, numpy_helper

    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    matches = []

    for conv in graph.node:
        if conv.op_type != "Conv" or conv.domain or len(conv.input) < 2:
            continue
        concat = producers.get(conv.input[0])
        if concat is None or concat.op_type != "Concat" or concat.domain or len(concat.input) != 2:
            continue
        concat_attributes = {attr.name: helper.get_attribute_value(attr) for attr in concat.attribute}
        if set(concat_attributes) != {"axis"} or concat_attributes["axis"] not in (-1, 2):
            continue

        constant = producers.get(concat.input[0])
        dynamic_input = concat.input[1]
        if constant is None or constant.op_type != "Constant" or constant.domain or not dynamic_input:
            continue
        if len(constant.output) != 1 or len(constant.attribute) != 1 or constant.attribute[0].name != "value":
            continue

        prefix_tensor = constant.attribute[0].t
        prefix = numpy_helper.to_array(prefix_tensor)
        if (
            prefix.ndim != 3
            or prefix.shape[0] != 1
            or prefix.shape[2] <= 0
            or not np.all(prefix == 0)
        ):
            continue

        matches.append((conv, concat, constant, dynamic_input, int(prefix.shape[2])))

    if expected_match_count is not None and len(matches) != expected_match_count:
        raise ValueError(
            f"fuse_static_zero_prefix_convs: expected {expected_match_count} matches, but found {len(matches)}. Failing closed."
        )

    removed_node_ids = set()
    for conv, concat, constant, dynamic_input, padding_left in matches:
        conv.input[0] = dynamic_input
        for attr in list(conv.attribute):
            if attr.name == "pads":
                conv.attribute.remove(attr)
        conv.attribute.append(helper.make_attribute("pads", [padding_left, 0]))
        removed_node_ids.update((id(concat), id(constant)))

    retained_nodes = [node for node in graph.node if id(node) not in removed_node_ids]
    del graph.node[:]
    graph.node.extend(retained_nodes)
    return model, len(matches)


def optimize_onnx_graph(model_path: str | Path) -> dict[str, int]:
    """Fail-closed graph surgery: eliminate duplicate Cast chains and redundant Identity parameter aliases."""
    import onnx

    model_path = Path(model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    graph = model.graph
    node_map = {n.output[0]: n for n in graph.node if len(n.output) > 0}
    inits = {i.name for i in graph.initializer}
    graph_outs = {o.name for o in graph.output}

    remap = {}
    removed_nodes = set()
    cast_pairs_removed = 0

    # 1. Eliminate identical back-to-back Cast nodes (Cast to T -> Cast to T)
    for n in graph.node:
        if n.op_type == "Cast":
            prev = node_map.get(n.input[0])
            if prev and prev.op_type == "Cast":
                to_n = [a.i for a in n.attribute if a.name == "to"][0]
                to_p = [a.i for a in prev.attribute if a.name == "to"][0]
                if to_n == to_p and prev.output[0] not in graph_outs:
                    remap[n.output[0]] = prev.output[0]
                    removed_nodes.add(id(n))
                    cast_pairs_removed += 1

    # 2. Eliminate redundant Identity aliases where input is an existing initializer
    identity_removed = 0
    for n in graph.node:
        if n.op_type == "Identity" and n.input[0] in inits and n.output[0] not in graph_outs:
            remap[n.output[0]] = n.input[0]
            removed_nodes.add(id(n))
            identity_removed += 1

    if not remap:
        return {"cast_pairs": 0, "identity": 0, "total_removed": 0}

    for node in graph.node:
        if id(node) in removed_nodes:
            continue
        for idx, inp in enumerate(node.input):
            curr = inp
            while curr in remap:
                curr = remap[curr]
            node.input[idx] = curr

    retained = [n for n in graph.node if id(n) not in removed_nodes]
    del graph.node[:]
    graph.node.extend(retained)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent,
            prefix=f".{model_path.stem}.opt.",
            suffix=".onnx",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        onnx.save(model, temp_path)
        temp_path.replace(model_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return {
        "cast_pairs": cast_pairs_removed,
        "identity": identity_removed,
        "total_removed": len(removed_nodes),
    }


class METADATA_CARRIER(torch.nn.Module):
    """Tiny identity graph that carries the static package contract."""

    def forward(self, marker):
        return marker


def run_compact_strategy_export():
    """Export the strategy pipeline with one hot DecodeStep per frame."""
    print('Compact strategy export start ...')
    with torch.inference_mode():
        for path in onnx_folder.glob("*.onnx*"):
            if path.is_file():
                path.unlink()

        model = load_audio8_model(download_path)
        config = model.config
        slow_core = A8_SLOW_MERGED(model, MAX_SEQ_LEN)
        fast_core = A8_FAST(model)
        transformer = slow_core.transformer
        num_layers = transformer.num_layers
        hidden_size = transformer.hidden_size
        num_codebooks = int(config.num_codebooks)
        codebook_size = int(config.codebook_size)
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        samples_per_codec_frame = int(config.codec_frame_size * OUT_SAMPLE_RATE / config.codec_sample_rate)

        # Dummy values used only to trace dynamic strategy inputs during export.
        generation_config = model.generation_config
        trace_temperature = float(getattr(generation_config, "temperature", None) or 0.7)
        trace_top_p = float(getattr(generation_config, "top_p", None) or 0.9)
        trace_top_k = int(getattr(generation_config, "top_k", None) or 50)

        # ArkttsProcessor encodes every prompt part separately; only the target and reference text are tokenized at runtime.
        tokenizer = AutoTokenizer.from_pretrained(
            download_path, use_fast=True, trust_remote_code=True, fix_mistral_regex=False
        )

        def encode(*parts):
            return [token_id for part in parts for token_id in tokenizer.encode(part, add_special_tokens=False)]

        zero_shot_prefix_ids = encode(
            "<|im_start|>system\n", "convert the provided text to speech", "<|im_end|>\n", "<|im_start|>user\n"
        )
        zero_shot_suffix_ids = encode("<|im_end|>\n", "<|im_start|>assistant\n<|voice|>")
        clone_prefix_head_ids = encode(
            "<|im_start|>system\n", "convert the provided text to speech reference to the following:\n\nText:\n"
        )
        clone_prefix_tail_ids = encode("\n\nSpeech:\n")
        clone_suffix_head_ids = encode("<|im_end|>\n", "<|im_start|>user\n")
        del tokenizer
        if zero_shot_suffix_ids[-1] != int(config.semantic_begin_id) - 1:
            raise ValueError("<|voice|> is not the single id right before the semantic range.")

        metadata = build_model_metadata(
            {
                "graph_layout": "strategy_prefill_decode_step",
                "in_sample_rate": IN_SAMPLE_RATE,
                "out_sample_rate": OUT_SAMPLE_RATE,
                "max_seq_len": MAX_SEQ_LEN,
                "num_codebooks": num_codebooks,
                "codebook_size": codebook_size,
                "stop_token_ids": [codebook_size],
                "samples_per_codec_frame": samples_per_codec_frame,
                "use_f16_kv": USE_F16_KV,
                "compute_in_f32": COMPUTE_IN_F32,
                "shared_initializer_model_file": SHARED_MODEL_NAME,
                "shared_initializer_data_file": SHARED_DATA_NAME,
                "model_file_name_reference_preprocess": Path(onnx_model_Reference_Preprocess).name,
                "model_file_name_target_preprocess": Path(onnx_model_Target_Preprocess).name,
                "model_file_name_decoder": Path(onnx_model_Decoder).name,
                "model_file_name_decoder_stream": Path(onnx_model_Decoder_Stream).name,
                **{
                    f"model_file_name_slow_prefill_{strategy}": Path(onnx_model_Slow_Prefill[strategy]).name
                    for strategy in DECODE_STRATEGIES
                },
                **{
                    f"model_file_name_decode_step_{strategy}": Path(onnx_model_Decode_Step[strategy]).name
                    for strategy in DECODE_STRATEGIES
                },
                "zero_shot_prefix_ids": zero_shot_prefix_ids,
                "zero_shot_suffix_ids": zero_shot_suffix_ids,
                "clone_prefix_head_ids": clone_prefix_head_ids,
                "clone_prefix_tail_ids": clone_prefix_tail_ids,
                "clone_suffix_head_ids": clone_suffix_head_ids,
            },
        )

        def export(module, args, path, input_names, output_names, dynamic_axes=None):
            torch.onnx.export(
                module,
                tuple(args),
                path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                dynamo=False,
            )

        reference_codec = load_audio8_codec(download_path, config)
        prepare_codec_for_export(reference_codec)
        reference_preprocess = A8_REFERENCE_PREPROCESS(reference_codec, IN_SAMPLE_RATE, MAX_REF_SECONDS).eval()
        prompt_audio = torch.zeros(1, 1, IN_SAMPLE_RATE + 777, dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()])
        with torch.amp.autocast('cpu', dtype=torch.float16, enabled=USE_F16_CODEC):
            ref_proto = export_to_model_proto(
                reference_preprocess,
                (prompt_audio,),
                ['prompt_audio'],
                ['ref_codes', 'ref_code_len'],
                {'prompt_audio': {2: 'audio_len'}, 'ref_codes': {2: 'ref_code_len'}},
            )
        ref_proto, matched_ref = fuse_static_zero_prefix_convs(
            ref_proto,
            reference_preprocess.static_conv_fusion_count,
        )
        set_onnx_metadata(ref_proto, metadata)
        atomic_save_model(ref_proto, onnx_model_Reference_Preprocess)
        print(f"[Conv rewrite in-memory] graph=ReferencePreprocess, matched={matched_ref}, Conv_pads_updated={matched_ref}")
        del reference_preprocess, reference_codec, prompt_audio, ref_proto

        target_preprocess = A8_TARGET_PREPROCESS(model).eval()
        target_proto = export_to_model_proto(
            target_preprocess,
            (
                torch.zeros(1, 10, dtype=torch.int32),
                torch.zeros(1, num_codebooks, 10, dtype=torch.int32),
                torch.zeros(1, 10, dtype=torch.int32),
            ),
            ['prefix_ids', 'ref_codes', 'suffix_ids'],
            ['hidden_states', 'ids_len'],
            {
                'prefix_ids': {1: 'prefix_len'},
                'ref_codes': {2: 'ref_code_len'},
                'suffix_ids': {1: 'suffix_len'},
                'hidden_states': {1: 'ids_len'},
            },
        )
        set_onnx_metadata(target_proto, metadata)
        atomic_save_model(target_proto, onnx_model_Target_Preprocess)
        del target_preprocess, target_proto

        decoder_codec = load_audio8_codec(download_path, config)
        prepare_codec_for_export(decoder_codec)
        decoder = A8_DECODER(decoder_codec, OUT_SAMPLE_RATE, int(config.codec_sample_rate), MAX_SEQ_LEN).eval()
        generated_codec = torch.zeros(1, num_codebooks * 10, dtype=torch.int32)
        with torch.amp.autocast('cpu', dtype=torch.float16, enabled=USE_F16_CODEC):
            dec_proto = export_to_model_proto(
                decoder,
                (generated_codec,),
                ['generated_codec'],
                ['generated_wav', 'generated_len'],
                {'generated_codec': {1: 'generated_codec_len'}, 'generated_wav': {2: 'generated_wav_len'}},
            )
            dec_proto, matched_dec = fuse_static_zero_prefix_convs(dec_proto, decoder.static_conv_fusion_count)
            set_onnx_metadata(dec_proto, metadata)
            atomic_save_model(dec_proto, onnx_model_Decoder)
            print(f"[Conv rewrite in-memory] graph=Decoder, matched={matched_dec}, Conv_pads_updated={matched_dec}")
            del dec_proto

            stream_proto = export_to_model_proto(
                decoder,
                (torch.zeros(1, num_codebooks * STREAM_WINDOW_FRAMES, dtype=torch.int32),),
                ['generated_codec'],
                ['generated_wav', 'generated_len'],
            )
            stream_proto, matched_stream = fuse_static_zero_prefix_convs(
                stream_proto,
                decoder.static_conv_fusion_count,
            )
            set_onnx_metadata(stream_proto, metadata)
            atomic_save_model(stream_proto, onnx_model_Decoder_Stream)
            print(f"[Conv rewrite in-memory] graph=Decoder_Stream, matched={matched_stream}, Conv_pads_updated={matched_stream}")
            del stream_proto
        del decoder, decoder_codec, generated_codec

        meta_proto = export_to_model_proto(
            METADATA_CARRIER(),
            (torch.zeros(1, dtype=torch.int64),),
            ['metadata_marker'],
            ['metadata_marker_out'],
        )
        set_onnx_metadata(meta_proto, metadata)
        atomic_save_model(meta_proto, onnx_model_Metadata)
        del meta_proto

        empty_key = torch.zeros(1, transformer.num_key_value_heads, 1, transformer.head_dim, 0, dtype=kv_dtype)
        empty_value = torch.zeros(1, transformer.num_key_value_heads, 1, 0, transformer.head_dim, dtype=kv_dtype)
        zero_conv = torch.zeros(1, transformer.conv_dim, transformer.conv_keep)
        zero_ssm = torch.zeros(1, transformer.ssm_heads, transformer.ssm_head_dim, transformer.ssm_state_size)
        state_args = (
            [empty_key] * num_layers + [empty_value] * num_layers + [zero_conv] * num_layers + [zero_ssm] * num_layers
        )
        state_in_names = [f'in_{kind}_{index}' for kind in ('key', 'value', 'conv', 'ssm') for index in range(num_layers)]
        state_out_names = [f'out_{kind}_{index}' for kind in ('key', 'value', 'conv', 'ssm') for index in range(num_layers)]
        state_axes = {}
        for index in range(num_layers):
            state_axes[f'in_key_{index}'] = {4: 'history_len'}
            state_axes[f'out_key_{index}'] = {4: 'kv_seq_len'}
            state_axes[f'in_value_{index}'] = {3: 'history_len'}
            state_axes[f'out_value_{index}'] = {3: 'kv_seq_len'}

        history_len = torch.zeros(1, dtype=torch.int64)
        frame_codec_ids = torch.zeros(1, num_codebooks, dtype=torch.int32)
        save_ids = torch.zeros(1, 10, dtype=torch.int32)
        # Representative values used only to trace dynamic strategy inputs.
        penalty_value = torch.tensor([0.8], dtype=torch.float32)
        penalty_range = torch.tensor([5], dtype=torch.int64)
        temperature = torch.tensor([trace_temperature], dtype=torch.float32)
        top_k = torch.tensor([min(trace_top_k, codebook_size)], dtype=torch.int64)
        top_p = torch.tensor([trace_top_p], dtype=torch.float32)

        for strategy in DECODE_STRATEGIES:
            slow_prefill = A8_SLOW_PREFILL_STRATEGY(slow_core, strategy)
            prefill_args = [torch.zeros(1, 10, hidden_size)]
            prefill_inputs = ['hidden_states']
            if strategy == 'sampling':
                prefill_args.extend([temperature, top_k, top_p])
                prefill_inputs.extend(['temperature', 'top_k', 'top_p'])
            export(
                slow_prefill,
                prefill_args,
                onnx_model_Slow_Prefill[strategy],
                prefill_inputs,
                state_out_names + ['last_hidden_state', 'semantic_token', 'kv_seq_len'],
                {
                    **{name: axes for name, axes in state_axes.items() if name.startswith('out_')},
                    'hidden_states': {1: 'ids_len'},
                },
            )
            del slow_prefill

            slow_decode = A8_SLOW_DECODE_STRATEGY(model, slow_core, strategy)
            decode_args = state_args + [frame_codec_ids, history_len]
            decode_inputs = state_in_names + ['frame_codec_ids', 'history_len']
            decode_outputs = state_out_names + ['last_hidden_state', 'semantic_token']
            decode_axes = dict(state_axes)
            if strategy == 'penalty_greedy':
                decode_args.extend([save_ids, penalty_value, penalty_range])
                decode_inputs.extend(['main_save_ids', 'main_penalty_value', 'main_penalty_range'])
            elif strategy == 'sampling':
                decode_args.extend([save_ids, temperature, top_k, top_p])
                decode_inputs.extend(['main_save_ids', 'main_temperature', 'main_top_k', 'main_top_p'])
            if strategy != 'greedy':
                decode_outputs.append('main_save_ids_out')
                decode_axes['main_save_ids'] = {1: 'main_history_len'}
                decode_axes['main_save_ids_out'] = {1: 'main_history_len_out'}
            decode_outputs.append('kv_seq_len')
            export(
                slow_decode,
                decode_args,
                onnx_model_Slow_Decode[strategy],
                decode_inputs,
                decode_outputs,
                decode_axes,
            )
            del slow_decode

            fast_frame = A8_FAST_FRAME_STRATEGY(fast_core, model, strategy)
            fast_args = [
                torch.zeros(1, 1, dtype=torch.int32),
                torch.zeros(1, 1, hidden_size),
                torch.zeros(1, 0, dtype=torch.int32),
            ]
            fast_inputs = ['semantic_token_in', 'last_hidden_state_slow', 'generated_codec_in']
            if strategy == 'penalty_greedy':
                fast_args.extend([penalty_value, penalty_range])
                fast_inputs.extend(['fast_penalty_value', 'fast_penalty_range'])
            elif strategy == 'sampling':
                fast_args.extend([temperature, top_k, top_p])
                fast_inputs.extend(['fast_temperature', 'fast_top_k', 'fast_top_p'])
            export(
                fast_frame,
                fast_args,
                onnx_model_Fast_Frame[strategy],
                fast_inputs,
                ['frame_codec_ids', 'generated_codec'],
                {
                    'generated_codec_in': {1: 'generated_codec_len'},
                    'generated_codec': {1: 'generated_codec_len_out'},
                },
            )
            del fast_frame

        del slow_core, fast_core, model
        gc.collect()

        for target in sorted(str(path) for path in onnx_folder.glob("*.onnx")):
            write_onnx_metadata(target, metadata)

        shared_stats = bundle_shared_initializers(onnx_folder, metadata=metadata)
        decode_steps = build_decode_step_graphs(onnx_folder, DECODE_STRATEGIES)
        replace_onnx_metadata(onnx_model_Metadata, metadata)
        print(
            f"[Shared weights] {shared_stats['initializer_references']} references -> "
            f"{shared_stats['unique_initializers']} unique tensors; "
            f"deduplicated {shared_stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB."
        )
        print(f"[DecodeStep] Built {len(decode_steps)} strategy graphs; component graphs removed.")

        # Post-export fail-closed graph optimization pass
        total_opt_stats = {"cast_pairs": 0, "identity": 0, "total_removed": 0}
        for target in sorted(onnx_folder.glob("*.onnx")):
            if target.name == SHARED_MODEL_NAME:
                continue
            graph_stats = optimize_onnx_graph(target)
            for k in total_opt_stats:
                total_opt_stats[k] += graph_stats[k]
        print(
            f"[Graph Surgery] Eliminated {total_opt_stats['cast_pairs']} redundant duplicate Cast pairs, "
            f"{total_opt_stats['identity']} dead Identity aliases; "
            f"total {total_opt_stats['total_removed']} nodes eliminated."
        )

        # Comprehensive graph metrics reporting (§9.29)
        print("\n[Final Graph Metrics Summary]")
        for target in sorted(onnx_folder.glob("*.onnx")):
            if target.name == SHARED_MODEL_NAME:
                continue
            m = onnx.load(str(target), load_external_data=False)
            gm = compute_graph_metrics(m)
            print(
                f"  {target.name:42s}: nodes={gm['node_count']:4d}, "
                f"copy_ops={gm['copy_op_count']:3d}, elem_ops={gm['elementwise_op_count']:4d}, "
                f"casts={gm['cast_count']:2d}, layout_ops={gm['layout_op_count']:3d}"
            )

    print('Compact strategy export done!')


if __name__ == "__main__" and DO_EXPORT:
    run_compact_strategy_export()
    print('\nStart running the TTS by ONNXRuntime via Inference_Audio8TTS_ONNX.py.\nNow loading . . . it could cost minutes.')
    raise SystemExit(subprocess.call([
        sys.executable,
        str(script_dir / "Inference_Audio8TTS_ONNX.py"),
        "--onnx-folder",
        str(onnx_folder),
    ]))
