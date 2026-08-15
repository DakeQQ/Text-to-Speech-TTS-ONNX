import gc
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from librosa.filters import mel as librosa_mel_fn
# transformers==4.57.6
from transformers import AutoTokenizer
from transformers.models.mimi import modeling_mimi as mimi_mod
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as mod

from STFT_Process import STFT_Process
from Shared_Weights import (
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    build_decode_step_graphs,
    bundle_shared_initializers,
)



# ─────────────────────────────────────────────────────────────────────────────
# Export settings
# ─────────────────────────────────────────────────────────────────────────────
download_path            = str(Path.home() / 'Downloads' / 'Qwen3-TTS-12Hz-0.6B-Base')  # Source model folder: Base, CustomVoice, or VoiceDesign.
MAX_SEQ_LEN              = 1024                     # Maximum prompt + generated sequence length; fixed in rotary tables and masks.
DO_EXPORT                = True                     # Set True to run the export pipeline
STREAM_WINDOW_FRAMES     = 7                        # Streaming sliding window frame count, Lower is faster but affects quality. (at least ≥ 3, recommended ≥ 7, fixed at export time)
USE_F16_KV               = True                     # Use float16 KV cache (saves memory, may reduce quality)
COMPUTE_IN_F32           = False                    # Only affects USE_F16_KV: True → keep f16 KV storage but run the attention matmuls in f32 (more accurate); False (default) → compute in f16 (fastest). No effect when USE_F16_KV is False.
USE_F16_ENCODER          = False                    # Pre-process the encoder in FP16 format for better GPU utilization.
PREVENT_F16_OVERFLOW     = False                    # Prevent float16 overflow. Currently, it didn't support pure float16.
OPSET                    = 20                       # ONNX opset version
IN_SAMPLE_RATE           = 24000                    # Public prompt-audio ONNX input rate.
OUT_SAMPLE_RATE          = 24000                    # Public generated-waveform ONNX output rate.
IN_AUDIO_DTYPE           = "F32"                    # "F16" | "F32" | "INT16".
OUT_AUDIO_DTYPE          = "F32"                    # "F16" | "F32" | "INT16".

_AUDIO_DTYPES = {"F16": torch.float16, "F32": torch.float32, "INT16": torch.int16}
# Exact channel reorders that make later weight-only quantization friendlier.
REORDER_DOWNPROJ_FOR_QUANT = True                   # Recommended: reorder MLP channels before down_proj quantization.
REORDER_OPROJ_FOR_QUANT    = True                   # Optional: reorder value/o_proj channels; validate audio quality when enabled.
REORDER_KEY                = "absmean"              # Channel score: "absmean" | "L4" | "rms" | "std".


# ─────────────────────────────────────────────────────────────────────────────
# Derived export paths
# ─────────────────────────────────────────────────────────────────────────────
script_dir                               = Path(__file__).resolve().parent
onnx_folder                              = script_dir / "QwenTTS_ONNX"
onnx_folder.mkdir(parents=True, exist_ok=True)
onnx_raw_folder                          = onnx_folder / "raw"
onnx_raw_folder.mkdir(parents=True, exist_ok=True)
onnx_model_Metadata                      = str(onnx_folder     / "QwenTTS_Metadata.onnx")            # Tiny metadata carrier graph.
onnx_model_Decoder                       = str(onnx_folder     / "QwenTTS_Decoder.onnx")
onnx_model_Decoder_Raw                   = str(onnx_raw_folder / "QwenTTS_Decoder.onnx")
onnx_model_Decoder_Stream                = str(onnx_folder     / "QwenTTS_Decoder_Stream.onnx")
onnx_model_Decoder_Stream_Raw            = str(onnx_raw_folder / "QwenTTS_Decoder_Stream.onnx")
onnx_model_Reference_Preprocess          = str(onnx_folder     / "QwenTTS_ReferencePreprocess.onnx")
onnx_model_Target_Preprocess             = str(onnx_folder     / "QwenTTS_TargetPreprocess.onnx")

DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")
onnx_model_Main_Prefill = {
    strategy: str(onnx_folder / f"QwenTTS_MainPrefill_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Main_Decode = {
    strategy: str(onnx_folder / f"QwenTTS_MainDecode_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Predictor_Frame = {
    strategy: str(onnx_folder / f"QwenTTS_PredictorFrame_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Decode_Step = {
    strategy: str(onnx_folder / f"QwenTTS_DecodeStep_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}


# ─────────────────────────────────────────────────────────────────────────────
# Inline export-only model patches.
# ─────────────────────────────────────────────────────────────────────────────

def _mimi_embed(self):
    cached = getattr(self, "_embed", None)
    if cached is None:
        cached = self.embed_sum / self.cluster_usage.unsqueeze(-1)
        self._embed = cached
    return cached


def _mimi_embed_T(self):
    cached = getattr(self, "_embed_T", None)
    if cached is None:
        cached = self.embed_sum.t() / self.cluster_usage
        self._embed_T = cached
    return cached


def _mimi_embed_norm(self):
    cached = getattr(self, "_embed_norm", None)
    if cached is None:
        embed_T = self.embed_T
        cached = (embed_T * embed_T).sum(0, keepdim=True)
        self._embed_norm = cached
    return cached


def _mimi_quantize(self, hidden_states):
    dot_product = torch.matmul(hidden_states, self.embed_T)
    dist = self.embed_norm - (dot_product + dot_product)
    return dist.argmin(dim=-1).int()


def _mimi_encode(self, hidden_states):
    return self.quantize(hidden_states)


def _mimi_static_padding_forward(self, hidden_states, padding_cache=None):
    if self.static_padding > 0:
        hidden_states = torch.cat((self.static_left_padding, hidden_states), dim=-1)
    return self.conv(hidden_states)


def _speaker_pooling_forward(self, hidden_states):
    seq_length = shape_dim_as_tensor(hidden_states, -1)
    mean, std = self._compute_statistics(hidden_states, 1.0 / seq_length)
    attention = torch.cat([hidden_states, mean.expand_as(hidden_states), std.expand_as(hidden_states)], dim=1)
    attention = self.conv(self.tanh(self.tdnn(attention)))
    attention = torch.nn.functional.softmax(attention, dim=2)
    mean, std = self._compute_statistics(hidden_states, attention)
    return torch.cat((mean, std), dim=1)


def _install_mimi_codebook_patch():
    codebook_cls = mimi_mod.MimiEuclideanCodebook
    codebook_cls.embed = property(_mimi_embed)
    codebook_cls.embed_T = property(_mimi_embed_T)
    codebook_cls.embed_norm = property(_mimi_embed_norm)
    codebook_cls.quantize = _mimi_quantize
    codebook_cls.encode = _mimi_encode


class Qwen3TTSTokenizerV2CausalConvNet(torch.nn.Module):
    """Causal Conv1d with static zero-prefix padding that exports as Concat."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride
        self.register_buffer(
            "left_padding",
            torch.zeros(1, in_channels, self.padding, dtype=torch.float32),
            persistent=False,
        )

    def _get_extra_padding_for_conv1d(self, hidden_state):
        length = hidden_state.shape[-1]
        n_frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state):
        if self.padding:
            hidden_state = torch.cat((self.left_padding, hidden_state), dim=-1)
        return self.conv(hidden_state).contiguous()


class Qwen3TTSTokenizerV2CausalTransConvNet(torch.nn.Module):
    """Causal ConvTranspose1d with a constant right-trim Slice."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride)
        self.left_pad = 0
        self.right_pad = int(kernel_size - stride)

    def forward(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        if self.right_pad > 0:
            hidden_state = hidden_state[..., :-self.right_pad]
        return hidden_state.contiguous()


class Qwen3TTSTokenizerV2ConvNeXtBlockUnfused(torch.nn.Module):
    """Original ConvNeXt block layout used only while loading checkpoint weights."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = mod.Qwen3TTSTokenizerV2CausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(dim, 4 * dim)
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(4 * dim, dim)
        self.gamma = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        hidden_states = hidden_states.transpose(1, 2)
        return residual + hidden_states


class Qwen3TTSTokenizerV2ConvNeXtBlock(torch.nn.Module):
    """Fused ConvNeXt block with LayerNorm affine and gamma folded into linears."""

    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = mod.Qwen3TTSTokenizerV2CausalConvNet(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=1,
        )
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.pwconv1 = torch.nn.Linear(dim, 4 * dim)
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(4 * dim, dim)

    @staticmethod
    def from_unfused(unfused):
        dim = unfused.dwconv.conv.in_channels
        fused = Qwen3TTSTokenizerV2ConvNeXtBlock(dim)

        fused.dwconv.load_state_dict(unfused.dwconv.state_dict())

        norm_weight = unfused.norm.weight.data
        norm_bias = unfused.norm.bias.data
        old_pw1_weight = unfused.pwconv1.weight.data
        old_pw1_bias = unfused.pwconv1.bias.data
        fused.pwconv1.weight.data = old_pw1_weight * norm_weight.unsqueeze(0)
        fused.pwconv1.bias.data = old_pw1_bias + (old_pw1_weight @ norm_bias)

        gamma = unfused.gamma.data
        old_pw2_weight = unfused.pwconv2.weight.data
        old_pw2_bias = unfused.pwconv2.bias.data
        fused.pwconv2.weight.data = old_pw2_weight * gamma.unsqueeze(1)
        fused.pwconv2.bias.data = old_pw2_bias * gamma

        fused.act = unfused.act
        return fused

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return residual + hidden_states


class SnakeBeta(torch.nn.Module):
    """SnakeBeta activation with export-time precomputation support."""

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features
        self.alpha = torch.nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = torch.nn.Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 0.000000001
        self._alpha_exp = None
        self._beta_inv = None

    def precompute(self):
        alpha_exp = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1)
        beta_exp = torch.exp(self.beta).unsqueeze(0).unsqueeze(-1)
        self._alpha_exp = alpha_exp
        self._beta_inv = 1.0 / (beta_exp + self.no_div_by_zero)

    def forward(self, hidden_states):
        if self._alpha_exp is not None and self._beta_inv is not None:
            alpha_exp = self._alpha_exp
            beta_inv = self._beta_inv
        else:
            alpha_exp = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
            beta_inv = 1.0 / (torch.exp(self.beta.unsqueeze(0).unsqueeze(-1)) + self.no_div_by_zero)

        return hidden_states + beta_inv * torch.pow(torch.sin(hidden_states * alpha_exp), 2)


class EuclideanCodebook(torch.nn.Module):
    def __init__(self, dim: int, codebook_size: int, epsilon: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.cluster_usage = torch.nn.Parameter(torch.ones(codebook_size))
        self.embedding_sum = torch.nn.Parameter(torch.zeros(codebook_size, dim))
        self._embedding = None

    def precompute_embedding(self):
        self._embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]

    def decode(self, codes):
        if self._embedding is None:
            self.precompute_embedding()
        return torch.nn.functional.embedding(codes, self._embedding)


class VectorQuantization(torch.nn.Module):
    def __init__(self, dim: int, codebook_size: int, codebook_dim=None, epsilon: float = 1e-5):
        super().__init__()
        if codebook_dim is None:
            codebook_dim = dim

        self.project_out = torch.nn.Linear(codebook_dim, dim) if codebook_dim != dim else torch.nn.Identity()
        self.epsilon = epsilon
        self._codebook = EuclideanCodebook(dim=codebook_dim, codebook_size=codebook_size, epsilon=epsilon)
        self.codebook_size = codebook_size

    def decode(self, codes):
        return self.project_out(self._codebook.decode(codes))


class ResidualVectorQuantization(torch.nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList([VectorQuantization(**kwargs) for _ in range(num_quantizers)])

    def decode(self, codes):
        quantized = None
        for idx, layer_codes in enumerate(torch.split(codes, 1, dim=0)):
            layer_quantized = self.layers[idx].decode(layer_codes)
            quantized = layer_quantized if quantized is None else quantized + layer_quantized
        return quantized.squeeze(0).transpose(1, 2)


class ResidualVectorQuantizer(torch.nn.Module):
    def __init__(
        self,
        dimension: int = 128,
        input_dimension=None,
        output_dimension=None,
        n_q: int = 8,
        q_dropout: bool = False,
        no_quantization_rate: float = 0.0,
        bins: int = 1024,
        decay: float = 0.99,
        force_projection: bool = False,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.no_quantization_rate = no_quantization_rate
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.bins = bins
        self.decay = decay
        if self.input_dimension == self.dimension and not force_projection:
            self.input_proj = torch.nn.Identity()
        else:
            self.input_proj = torch.nn.Conv1d(self.input_dimension, self.dimension, 1, bias=False)
        if self.output_dimension == self.dimension and not force_projection:
            self.output_proj = torch.nn.Identity()
        else:
            self.output_proj = torch.nn.Conv1d(self.dimension, self.output_dimension, 1, bias=False)
        self.vq = ResidualVectorQuantization(dim=self.dimension, codebook_size=self.bins, num_quantizers=self.n_q)

    def decode(self, codes):
        codes = codes.transpose(0, 1)
        quantized = self.vq.decode(codes)
        return self.output_proj(quantized)


class SplitResidualVectorQuantizer(torch.nn.Module):
    def __init__(self, *, n_q: int = 8, n_q_semantic: int = 1, **kwargs):
        super().__init__()
        self.max_n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        q_dropout = kwargs.pop("q_dropout", False)
        self.rvq_first = ResidualVectorQuantizer(n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs)
        self.rvq_rest = ResidualVectorQuantizer(n_q=n_q - n_q_semantic, force_projection=True, q_dropout=q_dropout, **kwargs)

    def decode(self, codes):
        first_code = codes[:, [self.n_q_semantic]]
        quantized_0 = self.rvq_first.vq.decode(first_code)
        quantized_0 = self.rvq_first.output_proj(quantized_0)
        rest_codes = codes[:, self.n_q_semantic:].transpose(0, 1)
        quantized_1 = self.rvq_rest.vq.decode(rest_codes)
        quantized_1 = self.rvq_rest.output_proj(quantized_1)
        return quantized_0 + quantized_1


def _install_tokenizer_v2_patches(load_unfused_convnext=False):
    mod.Qwen3TTSTokenizerV2CausalConvNet = Qwen3TTSTokenizerV2CausalConvNet
    mod.Qwen3TTSTokenizerV2CausalTransConvNet = Qwen3TTSTokenizerV2CausalTransConvNet
    mod.SnakeBeta = SnakeBeta
    mod.EuclideanCodebook = EuclideanCodebook
    mod.VectorQuantization = VectorQuantization
    mod.ResidualVectorQuantization = ResidualVectorQuantization
    mod.ResidualVectorQuantizer = ResidualVectorQuantizer
    mod.SplitResidualVectorQuantizer = SplitResidualVectorQuantizer
    mod.Qwen3TTSTokenizerV2ConvNeXtBlockUnfused = Qwen3TTSTokenizerV2ConvNeXtBlockUnfused
    if load_unfused_convnext:
        mod.Qwen3TTSTokenizerV2ConvNeXtBlock = Qwen3TTSTokenizerV2ConvNeXtBlockUnfused
    else:
        mod.Qwen3TTSTokenizerV2ConvNeXtBlock = Qwen3TTSTokenizerV2ConvNeXtBlock


_install_mimi_codebook_patch()
_install_tokenizer_v2_patches()


# ─────────────────────────────────────────────────────────────────────────────
# Mode Selection (mutually exclusive)
# ─────────────────────────────────────────────────────────────────────────────
# Options: "voice_clone", "custom_voice", or "voice_design" (one per export run)
if "custom" in download_path.lower():
    MODE = "custom_voice"
elif "design" in download_path.lower():
    MODE = "voice_design"
else:
    MODE = "voice_clone"


LANGUAGE_ID_MAP = {
    'english':    2050,
    'german':     2053,
    'spanish':    2054,
    'chinese':    2055,
    'japanese':   2058,
    'french':     2061,
    'korean':     2064,
    'russian':    2069,
    'italian':    2070,
    'portuguese': 2071,
}
SPEAKER_ID_MAP = {
    'serena':   3066,
    'vivian':   3065,
    'uncle_fu': 3010,
    'ryan':     3061,
    'aiden':    2861,
    'ono_anna': 2873,
    'sohee':    2864,
    'eric':     2875,
    'dylan':    2878,
}
SPEAKER_DIALECT_MAP = {
    'serena':   False,
    'vivian':   False,
    'uncle_fu': False,
    'ryan':     False,
    'aiden':    False,
    'ono_anna': False,
    'sohee':    False,
    'eric':     'sichuan_dialect',
    'dylan':    'beijing_dialect',
}
DIALECT_LANGUAGE_ID_MAP = {
    'beijing_dialect':  2074,
    'sichuan_dialect':  2062,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Tensor Utility Modules
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Modules
# ─────────────────────────────────────────────────────────────────────────────
class TTS_EMBED_A(torch.nn.Module):
    """Text token → projected text embedding (used for both reference and target text)."""

    def __init__(self, tts):
        super().__init__()
        self.tts = tts.model
        self._replace_gelu_with_tanh_approximation(self.tts)
        self.talker_text_embed = self.tts.talker.model.text_embedding
        self.text_projection   = self.tts.talker.text_projection

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def forward(self, text_ids):
        return self.text_projection(self.talker_text_embed(text_ids))


class TTS_EMBED_B(torch.nn.Module):
    """Input token ids → talker codec embedding."""

    def __init__(self, tts):
        super().__init__()
        self.talker_input_embed = tts.model.talker.model.codec_embedding

    def forward(self, codec_ids):
        return self.talker_input_embed(codec_ids)


class TTS_EMBED_C(torch.nn.Module):
    """Codec ids → code-predictor codec embedding for all RVQ layers (fused)."""

    def __init__(self, tts):
        super().__init__()
        self.talker_code_predictor_embed = tts.model.talker.code_predictor.model.codec_embedding
        self.num_code_groups = tts.model.talker.code_predictor.model.config.num_code_groups

    def forward(self, codec_ids, codec_embed, trailing_text_hidden, gather_id):
        codec_ids = codec_ids.reshape(self.num_code_groups, -1)
        codec_id_groups = torch.split(codec_ids, 1, dim=0)
        codec_embed = codec_embed + trailing_text_hidden[:, gather_id]
        for layer, layer_ids in enumerate(codec_id_groups[1:]):
            codec_embed = codec_embed + self.talker_code_predictor_embed._modules[f'{layer}'](layer_ids)
        return codec_embed


class TTS_EMBED_D(torch.nn.Module):
    """Codec ids → code-predictor codec embedding for one RVQ layer (split)."""

    def __init__(self, tts, layer):
        super().__init__()
        self.talker_code_predictor_embed = tts.model.talker.code_predictor.model.codec_embedding._modules[f'{layer}']

    def forward(self, codec_ids):
        return self.talker_code_predictor_embed(codec_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Embedding & Attention Mask Modules
# ─────────────────────────────────────────────────────────────────────────────
class TTS_MAIN_ROTARY_MASK_PREFILL(torch.nn.Module):
    """
    Compute rotary position embeddings and a causal attention mask for the
    main talker prefill (multi-token) phase.
    """

    def __init__(self, tts, max_seq_len):
        super().__init__()
        self.tts           = tts.model.talker.model
        self.mrope_section = tts.model.talker.config.rope_scaling['mrope_section']
        head_dim_half      = tts.model.talker.config.head_dim // 2
        modality_num       = len(self.mrope_section)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).view(1, 1, 1, -1).expand(3, 1, 1, -1)
        inv_freq     = self.tts.rotary_emb.inv_freq.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        idx_theta    = (inv_freq @ position_ids).transpose(2, 3)
        cos          = torch.cat([torch.cos(idx_theta)] * 2, dim=-1)
        sin          = torch.cat([torch.sin(idx_theta)] * 2, dim=-1)

        cos = torch.cat([self.apply_interleaved_rope(cos[..., :head_dim_half], modality_num)] * 2, dim=-1)
        sin = torch.cat([self.apply_interleaved_rope(sin[..., :head_dim_half], modality_num)] * 2, dim=-1)
        sin[..., :head_dim_half] = sin[..., :head_dim_half] * -1.0

        cos = cos.unsqueeze(2).unsqueeze(2).half()
        sin = sin.unsqueeze(2).unsqueeze(2).half()

        # Mask dtype tracks the attention-compute dtype: float16 only for the minimum-cast f16 KV attention
        # in TTS_MAIN (f16 KV computing in f16); float32 otherwise (including compute_in_f32). It is added to
        # the (f16 or f32) attention scores in TTS_MAIN.
        self.mask_dtype = torch.float16 if (USE_F16_KV and not COMPUTE_IN_F32) else torch.float32
        # Causal mask: -128 for masked positions (used with int8 KV cache arithmetic)
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

    def apply_interleaved_rope(self, x, modality_num):
        x_t = x[0].clone()
        index_ranges = []
        for i, n in enumerate(self.mrope_section[1:], 1):
            index_ranges.append((i, n * modality_num))
        for beg_idx, end_idx in index_ranges:
            x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
        return x_t

    def forward(self, ids_len, history_len):
        kv_seq_len         = ids_len + history_len
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask     = self.attention_mask[..., history_len:kv_seq_len, :kv_seq_len].to(self.mask_dtype)
        return rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, kv_seq_len


class TTS_PREDICTOR_ROTARY_MASK_PREFILL(torch.nn.Module):
    """
    Compute rotary position embeddings and a causal attention mask for the
    code-predictor prefill (multi-token) phase.
    """

    def __init__(self, tts, max_seq_len):
        super().__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq     = tts.model.talker.code_predictor.model.rotary_emb.inv_freq
        idx_theta    = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos          = torch.cat([torch.cos(idx_theta)] * 2, dim=-1).half()
        sin          = torch.sin(idx_theta)
        sin          = torch.cat([-sin, sin], dim=-1).half()

        # Mask dtype tracks the attention-compute dtype: float16 only for the minimum-cast f16 KV attention
        # in TTS_PREDICTOR (f16 KV computing in f16); float32 otherwise (including compute_in_f32). It is added
        # to the (f16 or f32) attention scores in TTS_PREDICTOR.
        self.mask_dtype = torch.float16 if (USE_F16_KV and not COMPUTE_IN_F32) else torch.float32
        # Causal mask: -128 for masked positions (used with int8 KV cache arithmetic)
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

    def forward(self, ids_len, history_len):
        kv_seq_len         = ids_len + history_len
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask     = self.attention_mask[..., history_len:kv_seq_len, :kv_seq_len].to(self.mask_dtype)
        return rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, kv_seq_len


# ─────────────────────────────────────────────────────────────────────────────
# Audio Pipeline Modules
# ─────────────────────────────────────────────────────────────────────────────
class TTS_ENCODER(torch.nn.Module):
    """
    Encode a raw audio waveform into:
      - ref_code       : RVQ codec tokens (used as in-context reference)
      - ref_code_len   : number of codec frames
      - speaker_embed  : speaker identity embedding derived from Mel spectrogram
    """

    def __init__(self, tts, in_sample_rate, max_seq_len, stft_model, nfft_stft, n_mels):
        super().__init__()
        self.tts             = tts
        self._replace_gelu_with_tanh_approximation(self.tts.model)
        self.encoder         = self.tts.model.speech_tokenizer.model.encoder.eval()
        self.speaker_encoder = self.tts.model.speaker_encoder.eval()

        for param in self.tts.model.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

        self._prepare_speaker_conv_padding()
        self._prepare_static_mimi_padding()
        self._fuse_encoder_weights()

        # Pre-computed values
        self.stft_model = stft_model
        self.in_sample_rate = in_sample_rate
        self.model_sample_rate = int(getattr(self.tts.model, "speaker_encoder_sample_rate", 24000))
        self.sr_scale   = float(self.model_sample_rate / self.in_sample_rate)
        self.inv_int16 = 1.0 / 32768.0
        self.integer_input = "int" in IN_AUDIO_DTYPE.lower()
        self.input_scale_folded = self.integer_input and self.sr_scale == 1.0
        if self.input_scale_folded:
            self._fuse_pcm_input_scale()
        # Reference mel_spectrogram framing: reflect-pad (n_fft - hop) // 2 on each side, then a
        # center=False STFT (stft_model is built with center_pad=False to avoid a second pad).
        self.stft_pad   = (nfft_stft - stft_model.hop_len) // 2
        self.fbank      = torch.from_numpy(librosa_mel_fn(sr=self.model_sample_rate, n_fft=nfft_stft, n_mels=n_mels, fmin=0, fmax=self.model_sample_rate // 2)).float().unsqueeze(0)

        self.num_heads     = self.encoder.encoder_transformer.layers._modules['0'].self_attn.num_heads
        self.qk_heads      = self.num_heads + self.num_heads
        self.head_dim      = self.encoder.encoder_transformer.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2

        position_ids      = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq          = self.encoder.encoder_transformer.layers._modules['0'].self_attn.rotary_emb.inv_freq
        idx_theta         = (position_ids * inv_freq).unsqueeze(1).unsqueeze(0)
        cos, sin          = torch.cos(idx_theta), torch.sin(idx_theta)
        self.rope_emb_cos = torch.cat([cos,  cos], dim=-1)
        self.rope_emb_sin = torch.cat([-sin, sin], dim=-1)

    # ── Weight Fusion ─────────────────────────────────────────────────────────

    def _prepare_speaker_conv_padding(self):
        """Remove exporter-hostile zero-width reflect padding from kernel-1 convolutions."""
        patched = 0
        for module in self.speaker_encoder.modules():
            if not isinstance(module, torch.nn.Conv1d):
                continue
            if module.kernel_size != (1,) or module.stride != (1,) or module.dilation != (1,):
                continue
            if module.padding != "same":
                continue
            module.padding = (0,)
            module.padding_mode = "zeros"
            module._reversed_padding_repeated_twice = (0, 0)
            patched += 1
        self.static_speaker_padding_count = patched
        self.speaker_encoder.asp.forward = _speaker_pooling_forward.__get__(
            self.speaker_encoder.asp,
            type(self.speaker_encoder.asp),
        )

    def _prepare_static_mimi_padding(self):
        """Specialize stride-1 causal Mimi padding; strided layers remain dynamic."""
        patched = 0
        for module in self.encoder.encoder.modules():
            if not isinstance(module, mimi_mod.MimiConv1d):
                continue
            if not module.causal or module.conv.stride[0] != 1 or module.pad_mode != "constant":
                continue
            padding = int(module.padding_total.item())
            module.static_padding = padding
            module.register_buffer(
                "static_left_padding",
                torch.zeros(1, module.in_channels, padding, dtype=module.conv.weight.dtype),
                persistent=False,
            )
            module.forward = _mimi_static_padding_forward.__get__(module, type(module))
            patched += 1
        self.static_mimi_padding_count = patched
        self.static_mimi_fusion_count = sum(
            module.static_padding > 0
            for module in self.encoder.encoder.modules()
            if isinstance(module, mimi_mod.MimiConv1d) and hasattr(module, "static_padding")
        )

    def _fuse_encoder_weights(self):
        """Fuse QKV projections, layer norms, and layer scales for the encoder transformer."""
        scale_factor = self.encoder.encoder_transformer.layers._modules['0'].self_attn.head_dim ** -0.25
        with torch.no_grad():
            for layer in self.encoder.encoder_transformer.layers:
                self._fuse_qkv_projection(layer, scale_factor)
                self._fuse_input_layernorm_into_qkv(layer)
                self._fuse_post_layernorm_into_mlp(layer)
                self._fuse_layer_scales(layer)

    def _fuse_pcm_input_scale(self):
        """Absorb int16-to-float scaling into both linear waveform consumers."""
        first_encoder_conv = self.encoder.encoder.layers._modules['0'].conv
        with torch.no_grad():
            first_encoder_conv.weight.mul_(self.inv_int16)
            self.stft_model.stft_kernel.mul_(self.inv_int16)

    def _fuse_qkv_projection(self, layer, scale_factor):
        """Fuse Q, K, V projections into a single QKV linear."""
        q_proj, k_proj, v_proj = (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        )
        in_features  = q_proj.in_features
        out_features = q_proj.out_features + k_proj.out_features + v_proj.out_features
        qkv          = torch.nn.Linear(in_features, out_features, bias=(q_proj.bias is not None))
        qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
        if q_proj.bias is not None:
            qkv.bias.copy_(torch.cat([q_proj.bias * scale_factor, k_proj.bias * scale_factor, v_proj.bias], dim=0))
        layer.self_attn.qkv   = qkv
        layer.self_attn.q_dim = q_proj.out_features
        layer.self_attn.k_dim = k_proj.out_features
        layer.self_attn.v_dim = v_proj.out_features
        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

    def _fuse_input_layernorm_into_qkv(self, layer):
        """Absorb input_layernorm affine parameters into the QKV projection."""
        norm   = layer.input_layernorm
        linear = layer.self_attn.qkv
        if linear.bias is not None:
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        else:
            linear.bias = torch.nn.Parameter(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = norm.bias = None

    def _fuse_post_layernorm_into_mlp(self, layer):
        """Absorb post_attention_layernorm affine parameters into MLP fc1."""
        norm   = layer.post_attention_layernorm
        linear = layer.mlp.fc1
        if linear.bias is not None:
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        else:
            linear.bias = torch.nn.Parameter(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = norm.bias = None

    def _fuse_layer_scales(self, layer):
        """Fuse self-attention and MLP layer scales into output projections."""
        scale  = layer.self_attn_layer_scale.scale
        linear = layer.self_attn.o_proj
        if linear.bias is not None:
            linear.bias.data.mul_(scale.data)
        linear.weight.data.mul_(scale.data.unsqueeze(1))

        scale  = layer.mlp_layer_scale.scale
        linear = layer.mlp.fc2
        if linear.bias is not None:
            linear.bias.data.mul_(scale.data)
        linear.weight.data.mul_(scale.data.unsqueeze(1))

    # ── Utility ───────────────────────────────────────────────────────────────

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def rotate_half(self, x):
        """Rotate using flip() — more efficient than split()+cat() in ONNX Runtime."""
        x = x.view(1, -1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(1, -1, self.qk_heads, self.head_dim)

    def forward(self, prompt_audio):
        # Public ONNX input is int16 PCM; convert to the official float waveform domain internally.
        prompt_audio = prompt_audio.float()
        if self.sr_scale < 1.0:
            prompt_audio = torch.nn.functional.interpolate(
                prompt_audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )
        if self.integer_input and not self.input_scale_folded:
            prompt_audio *= self.inv_int16
        if self.sr_scale > 1.0:
            prompt_audio = torch.nn.functional.interpolate(
                prompt_audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )

        # Encode audio through the convolutional encoder
        hidden_states = self.encoder.encoder(prompt_audio).transpose(1, 2)
        ids_len       = hidden_states.shape[1]
        rope_emb_cos  = self.rope_emb_cos[:, :ids_len].float()
        rope_emb_sin  = self.rope_emb_sin[:, :ids_len].float()

        # Transformer layers with fused RoPE attention
        for layer in self.encoder.encoder_transformer.layers:
            residual      = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            qkv           = layer.self_attn.qkv(hidden_states)
            qkv           = qkv.reshape(1, -1, self.qk_heads + self.num_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_heads], dim=-2)
            qk            = qk * rope_emb_cos + self.rotate_half(qk) * rope_emb_sin
            q, k          = torch.split(qk, [self.num_heads, self.num_heads], dim=-2)
            q             = q.transpose(1, 2)
            k             = k.permute(0, 2, 3, 1)
            v             = v.transpose(1, 2)
            attn          = torch.softmax(torch.matmul(q, k), dim=-1)
            attn          = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual      = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp.fc2(layer.mlp.activation_fn(layer.mlp.fc1(hidden_states)))

        # Downsample and quantise to RVQ codes
        embeddings = self.encoder.downsample(hidden_states.transpose(1, 2))
        ref_code   = self.encoder.quantizer.encode(embeddings, self.tts.model.speech_tokenizer.config.encoder_valid_num_quantizers)
        ref_code   = ref_code.squeeze(1)

        # Compute speaker embedding from log-Mel spectrogram (matches reference mel_spectrogram):
        # reflect-pad (n_fft - hop) // 2, center=False STFT, magnitude sqrt(re^2 + im^2 + 1e-9),
        # then dynamic-range compression log(clamp(mel, min=1e-5)).
        stft_audio           = torch.nn.functional.pad(prompt_audio, (self.stft_pad, self.stft_pad), mode='reflect')
        real_part, imag_part = self.stft_model(stft_audio)
        magnitude            = torch.sqrt(real_part * real_part + imag_part * imag_part + 1e-9)
        mel_features         = torch.matmul(self.fbank, magnitude).clamp(min=1e-5).log()
        speaker_embed        = self.speaker_encoder(mel_features)
        ref_code_len         = shape_dim_as_tensor(ref_code, 1)

        return ref_code, ref_code_len, speaker_embed.float()


class TTS_PREPROCESS(torch.nn.Module):
    """
    Build the full talker input embedding from:
      - reference-audio codec embedding
      - speaker embedding
      - language embedding
      - reference text embedding
      - target text embedding

    Returns talker_input_embed, trailing_text_hidden, and the sequence length.

    Modes:
      - voice_clone:  speaker_embed is from speaker encoder (x-vector), codec_embed from ref audio.
      - custom_voice: speaker_embed is from speaker id token, codec_embed is empty.
      - voice_design: no speaker_embed at all, codec_embed is empty. Voice identity comes from instruct.
    """

    def __init__(self, tts, mode="voice_clone"):
        super().__init__()
        self.tts = tts
        self.mode = mode
        self.talker_text_embed  = self.tts.model.talker.model.text_embedding
        self.talker_input_embed = self.tts.model.talker.model.codec_embedding

        config = self.tts.model.config

        # Pre-compute special-token embeddings
        sp_tokens = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]], dtype=torch.int32)
        self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = self.tts.model.talker.text_projection(self.talker_text_embed(sp_tokens)).chunk(3, dim=1)

        # Pre-compute fixed codec prefix / suffix embeddings
        if mode == "voice_design":
            # voice_design: no speaker token → codec prefix is [think, think_bos, language, think_eos, pad, bos]
            # The pad portion aligns with: pad*4 + bos = 5 positions
            self._talker_input_embed = torch.cat([self.tts_pad_embed.expand(-1, 4, -1), self.tts_bos_embed], dim=1)
        else:
            # voice_clone / custom_voice: codec prefix includes speaker → pad*5 + bos = 6 positions
            self._talker_input_embed = torch.cat([self.tts_pad_embed.expand(-1, 5, -1), self.tts_bos_embed], dim=1)
        self.codec_bos_embed   = self.talker_input_embed(torch.tensor([[config.talker_config.codec_bos_id]],                                            dtype=torch.int32))
        self.codec_think_embed = self.talker_input_embed(torch.tensor([[config.talker_config.codec_think_id, config.talker_config.codec_think_bos_id]], dtype=torch.int32))
        self.codec_eos_embed   = self.talker_input_embed(torch.tensor([[config.talker_config.codec_think_eos_id]],                                      dtype=torch.int32))
        self.codec_pad_embed   = self.talker_input_embed(torch.tensor([[config.talker_config.codec_pad_id]],                                            dtype=torch.int32))

        # Role header embedding
        system_head     = "<|im_start|>assistant\n"
        system_head_ids = self.tts.processor(text=system_head, return_tensors="pt", padding=True)["input_ids"].int()
        self._talker_input_embed_role = self.tts.model.talker.text_projection(self.talker_text_embed(system_head_ids))
        self.talker_prefix_len = self._talker_input_embed_role.shape[1] + self._talker_input_embed.shape[1]
        self.register_buffer(
            "voice_design_ids_len",
            torch.tensor([self.talker_prefix_len + 1], dtype=torch.int64),
            persistent=False,
        )

    def forward(self, *args):
        if self.mode == "voice_design":
            # voice_design: forward(language_embed, target_text_embed)
            language_embed, target_text_embed = args
            return self._forward_voice_design(language_embed, target_text_embed)
        else:
            # voice_clone / custom_voice: forward(language_embed, target_text_embed, codec_embed, speaker_embed, ref_prompt_text_embed)
            language_embed, target_text_embed, codec_embed, speaker_embed, ref_prompt_text_embed = args
            return self._forward_default(language_embed, target_text_embed, codec_embed, speaker_embed, ref_prompt_text_embed)

    def _forward_default(self, language_embed, target_text_embed, codec_embed, speaker_embed, ref_prompt_text_embed):
        # Prepend BOS to the codec sequence
        codec_embed = torch.cat([self.codec_bos_embed, codec_embed], dim=1)
        codec_len   = shape_dim_as_tensor(codec_embed, 1)

        # Build text sequence and pad to match codec length
        text_embed = torch.cat([ref_prompt_text_embed, target_text_embed, self.tts_eos_embed], dim=1)
        text_len   = shape_dim_as_tensor(text_embed, 1)
        text_embed = torch.cat([text_embed, self.tts_pad_embed.repeat(1, (codec_len - text_len).clamp(min=0), 1)], dim=1)

        # Build codec conditioning prefix and combine with role header
        codec_input_embed   = torch.cat([self.codec_think_embed, language_embed, self.codec_eos_embed, speaker_embed, self.codec_pad_embed], dim=1)
        _talker_input_embed = self._talker_input_embed + codec_input_embed
        talker_input_embed  = torch.cat([self._talker_input_embed_role, _talker_input_embed], dim=1)

        # Interleave text and codec embeddings
        icl_input_embed      = text_embed[:, :codec_len] + codec_embed
        trailing_text_hidden = torch.cat([text_embed[:, codec_len:], self.tts_pad_embed], dim=1)
        trailing_len_minus   = (text_len - codec_len).clamp(min=0).int()
        hidden_states        = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        ids_len              = codec_len + self.talker_prefix_len
        return hidden_states, ids_len, trailing_text_hidden, trailing_len_minus

    def _forward_voice_design(self, language_embed, target_text_embed):
        # voice_design: no ref audio, no speaker → streaming-style text + codec interleave
        # Build text sequence: target_text + eos
        text_embed = torch.cat([target_text_embed, self.tts_eos_embed], dim=1)

        # Build codec conditioning prefix (no speaker): [think, think_bos, language, think_eos, pad]
        codec_input_embed   = torch.cat([self.codec_think_embed, language_embed, self.codec_eos_embed, self.codec_pad_embed], dim=1)
        _talker_input_embed = self._talker_input_embed + codec_input_embed
        talker_input_embed  = torch.cat([self._talker_input_embed_role, _talker_input_embed], dim=1)

        # For voice_design with no ref_code, the first text token is combined with codec_bos
        first_text_token    = text_embed[:, [0]] + self.codec_bos_embed
        talker_input_embed  = torch.cat([talker_input_embed, first_text_token], dim=1)

        # Remaining text tokens become trailing_text_hidden for streaming
        trailing_text_hidden = torch.cat([text_embed[:, 1:], self.tts_pad_embed], dim=1)
        trailing_len_minus   = shape_dim_as_tensor(target_text_embed, 1).int()
        hidden_states        = talker_input_embed
        ids_len              = self.voice_design_ids_len
        return hidden_states, ids_len, trailing_text_hidden, trailing_len_minus


class TTS_REFERENCE_PREPROCESS(torch.nn.Module):
    """Voice-clone reference audio and prompt text conditioning in one graph."""

    def __init__(self, tts, in_sample_rate, max_seq_len, stft_model, nfft_stft, n_mels):
        super().__init__()
        self.encoder = TTS_ENCODER(tts, in_sample_rate, max_seq_len, stft_model, nfft_stft, n_mels)
        self.embed_a = TTS_EMBED_A(tts)
        self.embed_b = TTS_EMBED_B(tts)
        self.embed_c = TTS_EMBED_C(tts)
        hidden_size = tts.model.talker.config.hidden_size
        self.register_buffer("zero_trailing_hidden", torch.zeros(1, 1, hidden_size), persistent=False)
        self.register_buffer("zero_gather_id", torch.zeros(1, dtype=torch.int32), persistent=False)
        self.static_mimi_fusion_count = self.encoder.static_mimi_fusion_count

    def forward(self, prompt_audio, prompt_text_ids):
        ref_code, _, speaker_embed = self.encoder(prompt_audio)
        codec_embed_0 = self.embed_b(ref_code[[0]])
        codec_embed = self.embed_c(
            ref_code,
            codec_embed_0,
            self.zero_trailing_hidden,
            self.zero_gather_id,
        )
        ref_prompt_text_embed = self.embed_a(prompt_text_ids)
        return codec_embed, speaker_embed, ref_prompt_text_embed


class TTS_TARGET_PREPROCESS(torch.nn.Module):
    """Raw target/instruction token IDs to the complete Main prefill embedding."""

    def __init__(self, tts, mode):
        super().__init__()
        self.mode = mode
        self.embed_a = TTS_EMBED_A(tts)
        self.embed_b = TTS_EMBED_B(tts)
        self.preprocess = TTS_PREPROCESS(tts, mode=mode)
        hidden_size = tts.model.talker.config.hidden_size
        self.register_buffer("empty_embed", torch.zeros(1, 0, hidden_size), persistent=False)

    def _prepend_instruction(self, hidden_states, instruct_text_ids):
        instruct_embed = self.embed_a(instruct_text_ids)
        hidden_states = torch.cat([instruct_embed, hidden_states], dim=1)
        return hidden_states, shape_dim_as_tensor(hidden_states, 1)

    def forward(self, *args):
        if self.mode == "voice_clone":
            (
                language_id,
                target_text_ids,
                instruct_text_ids,
                codec_embed,
                speaker_embed,
                ref_prompt_text_embed,
            ) = args
            outputs = self.preprocess(
                self.embed_b(language_id),
                self.embed_a(target_text_ids),
                codec_embed,
                speaker_embed,
                ref_prompt_text_embed,
            )
        elif self.mode == "custom_voice":
            language_id, speaker_id, target_text_ids, instruct_text_ids = args
            outputs = self.preprocess(
                self.embed_b(language_id),
                self.embed_a(target_text_ids),
                self.empty_embed,
                self.embed_b(speaker_id),
                self.empty_embed,
            )
        else:
            language_id, target_text_ids, instruct_text_ids = args
            outputs = self.preprocess(
                self.embed_b(language_id),
                self.embed_a(target_text_ids),
            )

        hidden_states, _, trailing_text_hidden, trailing_len_minus = outputs
        hidden_states, ids_len = self._prepend_instruction(hidden_states, instruct_text_ids)
        return hidden_states, ids_len, trailing_text_hidden, trailing_len_minus


class TTS_DECODER(torch.nn.Module):
    """
    Decode RVQ codec tokens back to a raw audio waveform.
    - voice_clone: Combines reference codec tokens with the generated sequence.
    - custom_voice: Decodes only the generated sequence (no ref_code).
    """

    def __init__(self, tts, output_sample_rate, model_output_sample_rate, max_seq_len, mode="voice_clone"):
        super().__init__()
        self.tts = tts
        self.mode = mode
        self._replace_gelu_with_tanh_approximation(self.tts.model)
        self.decoder         = self.tts.model.speech_tokenizer.model.decoder.eval()
        self.hidden_size     = self.decoder.config.hidden_size
        self.num_code_groups = self.tts.model.talker.code_predictor.model.config.num_code_groups
        self.scale           = output_sample_rate / model_output_sample_rate
        self.upsample_rate   = self.tts.model.speech_tokenizer.model.decode_upsample_rate
        self.static_conv_fusion_count = sum(
            isinstance(module, Qwen3TTSTokenizerV2CausalConvNet) and module.padding > 0
            for module in self.decoder.modules()
        )
        for param in self.tts.model.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.decoder.pre_transformer.parameters():
            param.requires_grad = False
        
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        self.rms_norm_eps = float(self.decoder.pre_transformer.config.rms_norm_eps)
        eps_hidden = torch.tensor([self.rms_norm_eps * self.hidden_size], dtype=torch.float32)
        if PREVENT_F16_OVERFLOW:
            eps_hidden *= self.overflow_scale.square()
        self.register_buffer("eps_hidden", eps_hidden, persistent=False)
        self.register_buffer(
            "rms_norm_scale",
            torch.full((self.hidden_size,), self.hidden_size ** -0.5, dtype=torch.float32),
            persistent=False,
        )

        self._fuse_decoder_weights()

        self.num_heads     = self.decoder.pre_transformer.layers._modules['0'].self_attn.config.num_attention_heads
        self.qk_heads      = self.num_heads + self.num_heads
        self.head_dim      = self.decoder.pre_transformer.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2

        position_ids      = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq          = self.decoder.pre_transformer.rotary_emb.inv_freq
        idx_theta         = (position_ids * inv_freq).unsqueeze(1).unsqueeze(0)
        cos, sin          = torch.cos(idx_theta), torch.sin(idx_theta)
        self.rope_emb_cos = torch.cat([cos,  cos], dim=-1).half()
        self.rope_emb_sin = torch.cat([-sin, sin], dim=-1).half()

        if "int" in OUT_AUDIO_DTYPE.lower():
            self._fuse_output_scale()

    # ── Output Scale & Activation Fusion ──────────────────────────────────────

    def _fuse_output_scale(self):
        """Fuse the int16 PCM scale into the final decoder convolution."""
        with torch.no_grad():
            # Fuse 32767.0 into the last block that has a conv weight
            for block in reversed(list(self.decoder.decoder)):
                conv = None
                if hasattr(block, 'conv') and hasattr(block.conv, 'weight'):
                    conv = block.conv
                elif hasattr(block, 'weight') and isinstance(block, torch.nn.Conv1d):
                    conv = block
                if conv is not None:
                    conv.weight.mul_(32767.0)
                    if conv.bias is not None:
                        conv.bias.mul_(32767.0)
                    break

    # ── Weight Fusion ─────────────────────────────────────────────────────────

    def _fuse_decoder_weights(self):
        """Fuse QKV projections, layer norms, layer scales, and final norm for the decoder pre-transformer."""
        scale_factor = self.decoder.pre_transformer.layers._modules['0'].self_attn.head_dim ** -0.25
        norm_factor  = self.hidden_size ** 0.5

        with torch.no_grad():
            for layer in self.decoder.pre_transformer.layers:
                self._fuse_qkv_projection(layer, scale_factor)
                self._fuse_input_layernorm_into_qkv(layer, norm_factor)
                self._fuse_gate_up_projection(layer, norm_factor)
                self._fuse_layer_scales(layer)

            # Fuse final pre-transformer norm into output projection
            final_norm_weight = self.decoder.pre_transformer.norm.weight.unsqueeze(0) * norm_factor
            self.decoder.pre_transformer.output_proj.weight.mul_(final_norm_weight)
            del self.decoder.pre_transformer.norm

    def _fuse_qkv_projection(self, layer, scale_factor):
        """Fuse Q, K, V projections into a single QKV linear with scale-factor baked in."""
        q_proj, k_proj, v_proj = (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        )
        in_features  = q_proj.in_features
        out_features = q_proj.out_features + k_proj.out_features + v_proj.out_features
        qkv          = torch.nn.Linear(in_features, out_features, bias=(q_proj.bias is not None))
        qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
        if q_proj.bias is not None:
            qkv.bias.copy_(torch.cat([q_proj.bias * scale_factor, k_proj.bias * scale_factor, v_proj.bias], dim=0))
        layer.self_attn.qkv   = qkv
        layer.self_attn.q_dim = q_proj.out_features
        layer.self_attn.k_dim = k_proj.out_features
        layer.self_attn.v_dim = v_proj.out_features
        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

    def _fuse_input_layernorm_into_qkv(self, layer, norm_factor):
        """Absorb input_layernorm (RMSNorm) into the QKV projection."""
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        layer.self_attn.qkv.weight.mul_(input_norm_weight)
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up         = layer.mlp.gate_proj, layer.mlp.up_proj
        gate_up          = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    def _fuse_layer_scales(self, layer):
        """Fuse self-attention and MLP layer scales into output projections."""
        scale  = layer.self_attn_layer_scale.scale
        linear = layer.self_attn.o_proj
        if linear.bias is not None:
            linear.bias.data.mul_(scale.data)
        linear.weight.data.mul_(scale.data.unsqueeze(1))

        scale  = layer.mlp_layer_scale.scale
        linear = layer.mlp.down_proj
        if linear.bias is not None:
            linear.bias.data.mul_(scale.data)
        linear.weight.data.mul_(scale.data.unsqueeze(1))

    # ── Utility ───────────────────────────────────────────────────────────────

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        return simplified_layer_norm(x, self.rms_norm_scale, self.rms_norm_eps)

    def rotate_half(self, x):
        """Rotate using flip() — more efficient than split()+cat() in ONNX Runtime."""
        x = x.view(1, -1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(1, -1, self.qk_heads, self.head_dim)

    def forward(self, generated_codec):
        concat_codec = generated_codec.reshape(1, -1, self.num_code_groups).transpose(1, 2)

        hidden_states = self.decoder.quantizer.decode(concat_codec)
        hidden_states = self.decoder.pre_conv(hidden_states).transpose(1, 2)
        hidden_states = self.decoder.pre_transformer.input_proj(hidden_states)

        ids_len      = hidden_states.shape[1]
        rope_emb_cos = self.rope_emb_cos[:, :ids_len].float()
        rope_emb_sin = self.rope_emb_sin[:, :ids_len].float()

        # Pre-transformer (full-attention over the entire codec sequence)
        for layer in self.decoder.pre_transformer.layers:
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv           = layer.self_attn.qkv(hidden_states)
            qkv           = qkv.reshape(1, -1, self.qk_heads + self.num_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_heads], dim=-2)
            qk            = qk * rope_emb_cos + self.rotate_half(qk) * rope_emb_sin
            q, k          = torch.split(qk, [self.num_heads, self.num_heads], dim=-2)
            q             = q.transpose(1, 2)
            k             = k.permute(0, 2, 3, 1)
            v             = v.transpose(1, 2)
            attn          = torch.softmax(torch.matmul(q, k), dim=-1)
            attn          = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, [layer.mlp.down_proj.in_features] * 2, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # Final norm, output projection, and convolutional decode
        hidden_states = self._rms_norm(hidden_states)
        hidden_states = self.decoder.pre_transformer.output_proj(hidden_states)
        generated_wav = hidden_states.transpose(1, 2)

        for blocks in self.decoder.upsample:
            for block in blocks:
                generated_wav = block(generated_wav)
        for block in self.decoder.decoder:
            generated_wav = block(generated_wav)

        if self.scale != 1.0:
            generated_wav = torch.nn.functional.interpolate(
                generated_wav,
                scale_factor=self.scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )

        if "int" in OUT_AUDIO_DTYPE.lower():
            generated_wav = generated_wav.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        elif "32" in OUT_AUDIO_DTYPE:
            generated_wav = generated_wav.clamp(min=-1.0, max=1.0).float()
        else:
            generated_wav = generated_wav.clamp(min=-1.0, max=1.0).half()
        generated_len = shape_dim_as_tensor(generated_wav, -1)
        return generated_wav, generated_len


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Model Modules
# ─────────────────────────────────────────────────────────────────────────────
class TTS_MAIN(torch.nn.Module):
    """
    Main talker auto-regressive transformer.
    Returns updated KV caches, the final hidden state, and the greedy codec token.
    """

    def __init__(self, tts):
        super().__init__()
        self.tts = tts.model.talker
        self._replace_gelu_with_tanh_approximation(self.tts.model)

        self.head_dim             = self.tts.config.head_dim
        self.head_dim_half        = self.head_dim // 2
        self.hidden_size          = self.tts.config.hidden_size
        self.num_heads            = self.tts.config.num_attention_heads
        self.num_key_value_heads  = self.tts.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads             = self.num_heads + self.num_key_value_heads
        self.num_layers           = self.tts.config.num_hidden_layers
        self.num_layers_2         = self.num_layers * 2
        self.num_layers_3         = self.num_layers * 3
        self.num_layers_4         = self.num_layers * 4
        self.num_layers_5         = self.num_layers * 5

        suppress_ids = [
            token_id
            for token_id in range(self.tts.config.vocab_size - 1024, self.tts.config.vocab_size)
            if token_id != self.tts.config.codec_eos_token_id
        ]
        suppress_logits_bias = torch.zeros((1, self.tts.config.vocab_size), dtype=torch.float32)
        suppress_logits_bias[..., suppress_ids] = -1e7
        self.register_buffer("suppress_logits_bias", suppress_logits_bias, persistent=False)

        self.save_key   = [None] * self.num_layers
        self.save_value = [None] * self.num_layers

        # When True (and USE_F16_KV): keep the f16 KV storage but run the attention matmuls in f32.
        self.compute_in_f32 = COMPUTE_IN_F32

        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        self.rms_norm_eps = float(self.tts.config.rms_norm_eps)
        eps_hidden = torch.tensor([self.rms_norm_eps * self.hidden_size], dtype=torch.float32)
        eps_head = torch.tensor([self.rms_norm_eps * self.head_dim], dtype=torch.float32)
        if PREVENT_F16_OVERFLOW:
            eps_hidden *= self.overflow_scale.square()
            eps_head *= self.overflow_scale.square()
        self.register_buffer("eps_hidden", eps_hidden, persistent=False)
        self.register_buffer("eps_head", eps_head, persistent=False)
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((self.hidden_size,), self.hidden_size ** -0.5, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "head_norm_scale",
            torch.full((self.head_dim,), self.head_dim ** -0.5, dtype=torch.float32),
            persistent=False,
        )

        self._fuse_weights()
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    # ── Weight Fusion ─────────────────────────────────────────────────────────

    def _fuse_weights(self):
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = self.hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5
        with torch.no_grad():
            for layer in self.tts.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn           = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        in_features    = int(q_proj.in_features)
        out_features   = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias       = any(p.bias is not None for p in (q_proj, k_proj, v_proj))
        qkv            = torch.nn.Linear(in_features, out_features, bias=has_bias)

        attn.q_out_features  = int(q_proj.out_features)
        attn.k_out_features  = int(k_proj.out_features)
        attn.v_out_features  = int(v_proj.out_features)
        attn.qkv_in_features = in_features
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        del attn.q_proj, attn.k_proj, attn.v_proj

        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)
        q_norm_repeated     = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated     = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight   = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up           = layer.mlp.gate_proj, layer.mlp.up_proj
        gate_up            = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    @staticmethod
    def _channel_score(weight, key, dims):
        absolute = weight.abs()
        if key == "rms":
            return weight.square().mean(dim=dims).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(dim=dims).pow(0.25)
        if key == "std":
            flattened = weight.movedim(dims, tuple(range(len(dims)))).reshape(-1, weight.shape[-1])
            return flattened.std(0)
        if key == "absmean":
            return absolute.mean(dim=dims)
        pass
    def _reorder_downproj_for_quant(self, key):
        with torch.no_grad():
            for layer in self.tts.model.layers:
                down_weight = layer.mlp.down_proj.weight
                permutation = torch.argsort(self._channel_score(down_weight, key, (0,)))
                intermediate_size = layer.mlp.down_proj.in_features
                gate_up_weight = layer.mlp.gate_up_proj.weight
                layer.mlp.gate_up_proj.weight.copy_(torch.cat([
                    gate_up_weight[:intermediate_size][permutation],
                    gate_up_weight[intermediate_size:][permutation],
                ], dim=0))
                layer.mlp.down_proj.weight.copy_(down_weight[:, permutation])

    def _reorder_oproj_for_quant(self, key):
        heads_per_kv = self.num_heads // self.num_key_value_heads
        with torch.no_grad():
            for layer in self.tts.model.layers:
                output_weight = layer.self_attn.o_proj.weight
                output_by_head = output_weight.view(output_weight.shape[0], self.num_heads, self.head_dim)
                permutations = []
                for kv_head in range(self.num_key_value_heads):
                    grouped = output_by_head[:, kv_head * heads_per_kv:(kv_head + 1) * heads_per_kv]
                    permutations.append(torch.argsort(self._channel_score(grouped, key, (0, 1))))

                reordered_output = output_by_head.clone()
                for head in range(self.num_heads):
                    reordered_output[:, head] = output_by_head[:, head, permutations[head // heads_per_kv]]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                qkv_weight = layer.self_attn.qkv.weight
                qkv_by_head = qkv_weight.view(-1, self.head_dim, qkv_weight.shape[1]).clone()
                for kv_head, permutation in enumerate(permutations):
                    qkv_by_head[self.qk_heads + kv_head] = qkv_by_head[self.qk_heads + kv_head][permutation]
                qkv_weight.copy_(qkv_by_head.reshape_as(qkv_weight))

    # ── Utility ───────────────────────────────────────────────────────────────

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x, eps, scale):
        return simplified_layer_norm(x, scale, self.rms_norm_eps)

    def rotate_half(self, x):
        x = x.view(1, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(1, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]

        for i, layer in enumerate(self.tts.model.layers):
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.eps_hidden, self.hidden_norm_scale)
            qkv           = layer.self_attn.qkv(hidden_states)
            qkv           = qkv.reshape(1, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk            = self._rms_norm(qk, self.eps_head, self.head_norm_scale) * layer.self_attn.qk_norm_weight
            qk_rot        = qk * rotary_pos_emb_cos + self.rotate_half(qk) * rotary_pos_emb_sin

            if USE_F16_KV and not self.compute_in_f32:
                # Earliest clean cast: q and k share qk_rot, so the split/reshape/permute below run in f16.
                qk_rot = qk_rot.half()

            q, k          = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q             = q.reshape(1, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q             = q.permute(0, 2, 3, 1, 4)

            if USE_F16_KV:
                if self.compute_in_f32:
                    k = k.half()   # store-only cast; q stays f32 and k is upcast back at the matmul
                v = v.half()

            k = torch.cat((all_inputs[i],                   k.permute(0, 3, 2, 4, 1)), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v.transpose(1, 3)),        dim=-2)
            self.save_key[i]   = k
            self.save_value[i] = v

            if USE_F16_KV and self.compute_in_f32:
                attn          = torch.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1)
                attn          = torch.matmul(attn, v.float()).permute(0, 3, 1, 2, 4).reshape(1, -1, layer.self_attn.o_proj.in_features)
            else:
                attn          = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
                attn          = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(1, -1, layer.self_attn.o_proj.in_features)
                if USE_F16_KV:
                    attn = attn.float()
            hidden_states = residual + layer.self_attn.o_proj(attn)

            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.eps_hidden, self.hidden_norm_scale)
            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, [layer.mlp.down_proj.in_features] * 2, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        last_hidden_states = simplified_layer_norm(
            hidden_states[:, -1],
            self.tts.model.norm.weight,
            self.rms_norm_eps,
        )
        logits             = self.tts.codec_head(last_hidden_states) + self.suppress_logits_bias

        return *self.save_key, *self.save_value, last_hidden_states.unsqueeze(1), logits


class TTS_PREDICTOR(torch.nn.Module):
    """
    RVQ code-predictor transformer.
    Accepts KV caches + hidden states and returns updated KV caches
    together with the final hidden state (fed to the LM heads).
    """

    def __init__(self, tts):
        super().__init__()
        self.tts = tts.model.talker.code_predictor
        self._replace_gelu_with_tanh_approximation(self.tts.model)

        self.head_dim             = self.tts.config.head_dim
        self.head_dim_half        = self.head_dim // 2
        self.hidden_size          = self.tts.config.hidden_size
        self.num_heads            = self.tts.config.num_attention_heads
        self.num_key_value_heads  = self.tts.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads             = self.num_heads + self.num_key_value_heads
        self.num_layers           = self.tts.config.num_hidden_layers
        self.num_layers_2         = self.num_layers * 2
        self.num_layers_3         = self.num_layers * 3
        self.num_layers_4         = self.num_layers * 4
        self.num_layers_5         = self.num_layers * 5

        self.save_key   = [None] * self.num_layers
        self.save_value = [None] * self.num_layers

        # When True (and USE_F16_KV): keep the f16 KV storage but run the attention matmuls in f32.
        self.compute_in_f32 = COMPUTE_IN_F32

        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        self.rms_norm_eps = float(self.tts.config.rms_norm_eps)
        eps_hidden = torch.tensor([self.rms_norm_eps * self.hidden_size], dtype=torch.float32)
        eps_head = torch.tensor([self.rms_norm_eps * self.head_dim], dtype=torch.float32)
        if PREVENT_F16_OVERFLOW:
            eps_hidden *= self.overflow_scale.square()
            eps_head *= self.overflow_scale.square()
        self.register_buffer("eps_hidden", eps_hidden, persistent=False)
        self.register_buffer("eps_head", eps_head, persistent=False)
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((self.hidden_size,), self.hidden_size ** -0.5, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "head_norm_scale",
            torch.full((self.head_dim,), self.head_dim ** -0.5, dtype=torch.float32),
            persistent=False,
        )

        self._fuse_weights()
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    # ── Weight Fusion ─────────────────────────────────────────────────────────

    def _fuse_weights(self):
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = self.hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5
        with torch.no_grad():
            for layer in self.tts.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn           = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        in_features    = int(q_proj.in_features)
        out_features   = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias       = any(p.bias is not None for p in (q_proj, k_proj, v_proj))
        qkv            = torch.nn.Linear(in_features, out_features, bias=has_bias)

        attn.q_out_features  = int(q_proj.out_features)
        attn.k_out_features  = int(k_proj.out_features)
        attn.v_out_features  = int(v_proj.out_features)
        attn.qkv_in_features = in_features
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        del attn.q_proj, attn.k_proj, attn.v_proj

        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)
        q_norm_repeated     = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated     = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight   = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up           = layer.mlp.gate_proj, layer.mlp.up_proj
        gate_up            = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    _channel_score = staticmethod(TTS_MAIN._channel_score)

    def _reorder_downproj_for_quant(self, key):
        TTS_MAIN._reorder_downproj_for_quant(self, key)

    def _reorder_oproj_for_quant(self, key):
        TTS_MAIN._reorder_oproj_for_quant(self, key)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x, eps, scale):
        return simplified_layer_norm(x, scale, self.rms_norm_eps)

    def rotate_half(self, x):
        x = x.view(1, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(1, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        hidden_states = self.tts.small_to_mtp_projection(hidden_states)

        for i, layer in enumerate(self.tts.model.layers):
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.eps_hidden, self.hidden_norm_scale)
            qkv           = layer.self_attn.qkv(hidden_states)
            qkv           = qkv.reshape(1, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk            = self._rms_norm(qk, self.eps_head, self.head_norm_scale) * layer.self_attn.qk_norm_weight
            qk_rot        = qk * rotary_pos_emb_cos + self.rotate_half(qk) * rotary_pos_emb_sin
            
            if USE_F16_KV and not self.compute_in_f32:
                # Earliest clean cast: q and k share qk_rot, so the split/reshape/permute below run in f16.
                qk_rot = qk_rot.half()

            q, k          = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q             = q.reshape(1, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q             = q.permute(0, 2, 3, 1, 4)

            if USE_F16_KV:
                if self.compute_in_f32:
                    k = k.half()   # store-only cast; q stays f32 and k is upcast back at the matmul
                v = v.half()

            k = torch.cat((all_inputs[i],                   k.permute(0, 3, 2, 4, 1)), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v.transpose(1, 3)),        dim=-2)
            self.save_key[i]   = k
            self.save_value[i] = v

            if USE_F16_KV and self.compute_in_f32:
                attn          = torch.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1)
                attn          = torch.matmul(attn, v.float()).permute(0, 3, 1, 2, 4).reshape(1, -1, layer.self_attn.o_proj.in_features)
            else:
                attn          = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
                attn          = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(1, -1, layer.self_attn.o_proj.in_features)
                if USE_F16_KV:
                    attn = attn.float()
            hidden_states = residual + layer.self_attn.o_proj(attn)

            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.eps_hidden, self.hidden_norm_scale)
            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, [layer.mlp.down_proj.in_features] * 2, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        hidden_states = self._rms_norm(hidden_states[:, -1], self.eps_hidden, self.hidden_norm_scale)
        return *self.save_key, *self.save_value, hidden_states


class TTS_MAIN_MERGED(torch.nn.Module):
    """Main talker with rotary tables, causal mask, and KV-length update in one graph."""

    def __init__(self, tts, max_seq_len):
        super().__init__()
        self.rotary_mask = TTS_MAIN_ROTARY_MASK_PREFILL(tts, max_seq_len)
        self.transformer = TTS_MAIN(tts)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-2]
        history_len = all_inputs[-1]
        ids_len = shape_dim_as_tensor(hidden_states, 1)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = self.rotary_mask(ids_len, history_len)
        outputs = self.transformer(
            *all_inputs[:-2],
            hidden_states,
            rotary_cos,
            rotary_sin,
            attention_mask,
        )
        return *outputs, kv_seq_len


class TTS_PREDICTOR_MERGED(torch.nn.Module):
    """Code predictor with rotary tables, causal mask, and KV-length update in one graph."""

    def __init__(self, tts, max_seq_len):
        super().__init__()
        self.rotary_mask = TTS_PREDICTOR_ROTARY_MASK_PREFILL(tts, max_seq_len)
        self.transformer = TTS_PREDICTOR(tts)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-2]
        history_len = all_inputs[-1]
        ids_len = shape_dim_as_tensor(hidden_states, 1)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = self.rotary_mask(ids_len, history_len)
        outputs = self.transformer(
            *all_inputs[:-2],
            hidden_states,
            rotary_cos,
            rotary_sin,
            attention_mask,
        )
        return *outputs, kv_seq_len


class TTS_PREDICTOR_LM_HEAD(torch.nn.Module):
    """
    LM head for one RVQ code group.
    Fuses the final layer-norm into the linear projection weight.
    """

    def __init__(self, tts, indices):
        super().__init__()
        self.tts    = tts.model.talker.code_predictor
        hidden_size = self.tts.config.hidden_size
        norm_factor = hidden_size ** 0.5

        with torch.no_grad():
            w                = self.tts.model.norm.weight.unsqueeze(0) * norm_factor
            original_lm_head = self.tts.lm_head._modules[f"{indices}"]
            self.lm_head     = torch.nn.Linear(original_lm_head.in_features, original_lm_head.out_features, bias=False)
            self.lm_head.weight.copy_(original_lm_head.weight * w)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


class TTS_MAIN_PREFILL_STRATEGY(torch.nn.Module):
    """Main prefill with empty KV initialization and an integrated token strategy."""

    def __init__(self, main_core, strategy):
        super().__init__()
        self.main_core = main_core
        self.strategy = strategy
        transformer = main_core.transformer
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
        self.register_buffer("zero_history_len", torch.zeros(1, dtype=torch.int64), persistent=False)
        self.sampling = TOPK_TOPP_SAMPLING()

    def forward(self, *args):
        hidden_states = args[0]
        outputs = self.main_core(
            *([self.empty_key] * self.num_layers),
            *([self.empty_value] * self.num_layers),
            hidden_states,
            self.zero_history_len,
        )
        kv = outputs[:self.num_layers * 2]
        last_hidden_state, logits, kv_seq_len = outputs[-3:]

        if self.strategy == "greedy":
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            return *kv, last_hidden_state, token, kv_seq_len
        if self.strategy == "penalty_greedy":
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            return *kv, last_hidden_state, token, kv_seq_len

        temperature, top_k, top_p = args[1:]
        token = self.sampling.sample(logits, temperature, top_k, top_p)
        return *kv, last_hidden_state, token, kv_seq_len


class TTS_MAIN_DECODE_STRATEGY(torch.nn.Module):
    """Embed one generated frame, advance Main KV, and select the next Main token."""

    def __init__(self, tts, main_core, strategy):
        super().__init__()
        self.main_core = main_core
        self.strategy = strategy
        self.num_layers = main_core.transformer.num_layers
        self.embed_b = TTS_EMBED_B(tts)
        self.embed_c = TTS_EMBED_C(tts)
        self.penalty = APPLY_PENALTY()
        self.sampling = TOPK_TOPP_SAMPLING()

    def forward(self, *args):
        kv_count = self.num_layers * 2
        kv = args[:kv_count]
        frame_codec_ids = args[kv_count]
        trailing_text_hidden = args[kv_count + 1]
        gather_id = args[kv_count + 2]
        history_len = args[kv_count + 3]
        codec_embed_0 = self.embed_b(frame_codec_ids[:, :1])
        hidden_states = self.embed_c(
            frame_codec_ids,
            codec_embed_0,
            trailing_text_hidden,
            gather_id,
        )
        outputs = self.main_core(*kv, hidden_states, history_len)
        kv_out = outputs[:kv_count]
        last_hidden_state, logits, kv_seq_len = outputs[-3:]

        if self.strategy == "greedy":
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            return *kv_out, last_hidden_state, token, kv_seq_len

        save_ids = args[kv_count + 4]
        if self.strategy == "penalty_greedy":
            penalty_value, penalty_range = args[kv_count + 5:kv_count + 7]
            penalized_logits = self.penalty(logits, save_ids, penalty_value, penalty_range)
            use_penalty = shape_dim_as_tensor(save_ids, 1) >= penalty_range
            logits = torch.where(use_penalty, penalized_logits, logits)
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            save_ids = torch.cat([save_ids, token], dim=-1)
        else:
            temperature, top_k, top_p, repetition_penalty = args[kv_count + 5:kv_count + 9]
            token, save_ids = self.sampling(
                logits,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                save_ids,
            )
        return *kv_out, last_hidden_state, token, save_ids, kv_seq_len


class TTS_PREDICTOR_FRAME_STRATEGY(torch.nn.Module):
    """Fully unrolled 15-stage RVQ Predictor for one generated audio frame."""

    def __init__(self, predictor_core, embed_b, lm_heads, embed_layers, strategy):
        super().__init__()
        self.predictor_core = predictor_core
        self.embed_b = embed_b
        self.lm_heads = lm_heads
        self.embed_layers = embed_layers
        self.strategy = strategy
        transformer = predictor_core.transformer
        self.num_layers = transformer.num_layers
        self.num_stages = len(lm_heads)
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
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
        self.register_buffer("zero_history_len", torch.zeros(1, dtype=torch.int64), persistent=False)
        self.register_buffer("empty_ids", torch.zeros(1, 0, dtype=torch.int32), persistent=False)
        self.penalty = APPLY_PENALTY()
        self.sampling = TOPK_TOPP_SAMPLING()

    def forward(self, *args):
        codec_token_main, last_hidden_state_main, generated_codec = args[:3]
        strategy_args = args[3:]
        hidden_states = torch.cat(
            [last_hidden_state_main, self.embed_b(codec_token_main)],
            dim=1,
        )
        kv = [self.empty_key] * self.num_layers + [self.empty_value] * self.num_layers
        history_len = self.zero_history_len
        save_ids = self.empty_ids
        frame_codec_ids = codec_token_main

        for stage in range(self.num_stages):
            outputs = self.predictor_core(*kv, hidden_states, history_len)
            kv = list(outputs[:self.num_layers * 2])
            last_hidden_state = outputs[-2]
            history_len = outputs[-1]
            logits = self.lm_heads[stage](last_hidden_state)

            if self.strategy == "greedy":
                token = torch.argmax(logits, dim=-1, keepdim=True).int()
            elif self.strategy == "penalty_greedy":
                penalty_value, penalty_range = strategy_args
                if stage > 0:
                    logits = self.penalty(logits, save_ids, penalty_value, penalty_range)
                token = torch.argmax(logits, dim=-1, keepdim=True).int()
                save_ids = token if stage == 0 else torch.cat([save_ids, token], dim=-1)
            else:
                temperature, top_k, top_p, repetition_penalty = strategy_args
                if stage == 0:
                    token = self.sampling.sample(logits, temperature, top_k, top_p)
                    save_ids = token
                else:
                    token, save_ids = self.sampling(
                        logits,
                        temperature,
                        top_k,
                        top_p,
                        repetition_penalty,
                        save_ids,
                    )

            frame_codec_ids = torch.cat([frame_codec_ids, token], dim=-1)
            if stage + 1 < self.num_stages:
                hidden_states = self.embed_layers[stage](token)

        generated_codec = torch.cat([generated_codec, frame_codec_ids], dim=-1)
        return frame_codec_ids, generated_codec


class APPLY_PENALTY(torch.nn.Module):
    """Apply a repetition penalty over the most recent `penalty_range` tokens."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:]
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

    def sample(self, scores, temperature, top_k, top_p):
        sorted_scores, sorted_indices = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=True)
        sorted_probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        keep = (cumulative_probabilities - sorted_probabilities) <= top_p

        kept_mass = torch.where(keep, cumulative_probabilities, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax((cumulative_probabilities >= threshold).int(), dim=-1, keepdim=True)
        return torch.gather(sorted_indices, 1, winner).int()

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        scores = SIGN_AWARE_REPETITION_PENALTY.apply(logits, repetition_penalty, previous_ids)
        sampled_id = self.sample(scores, temperature, top_k, top_p)
        save_id = torch.cat([previous_ids, sampled_id], dim=-1)
        return sampled_id, save_id


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


def repair_embed_c_output_shape(raw_path, final_path, hidden_size):
    """Repair one legacy-exporter value-info loss without changing graph computation."""
    import onnx

    raw_path = Path(raw_path)
    final_path = Path(final_path)
    model = onnx.load(raw_path, load_external_data=False)
    graph = model.graph
    outputs = [value for value in graph.output if value.name == "codec_embed_sum"]
    producers = [node for node in graph.node if "codec_embed_sum" in node.output]
    inputs = {value.name: value for value in graph.input}
    def _shape(value):
        tensor_type = value.type.tensor_type
        return tensor_type, tensor_type.shape.dim

    output_type, output_dims = _shape(outputs[0])
    base_type, base_dims = _shape(inputs["codec_embed_0"])
    preserved_nodes = [node.SerializeToString() for node in graph.node]
    preserved_initializers = [initializer.SerializeToString() for initializer in graph.initializer]
    preserved_inputs = [value.SerializeToString() for value in graph.input]
    output_dims[0].ClearField("dim_param")
    output_dims[0].dim_value = 1

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
        onnx.save(model, temp_path)
        temp_path.replace(final_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print("[Embed-C rewrite] matched=1, value_info_batch=1, nodes_changed=0, initializers_changed=0")


def fuse_static_zero_prefix_convs(raw_path, final_path, expected_match_count):
    """Fold exact Constant-zero-prefix Concat nodes into standard ONNX Conv pads."""
    from collections import defaultdict

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    raw_path = Path(raw_path)
    final_path = Path(final_path)
    model = onnx.load(raw_path, load_external_data=False)
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    initializers = {initializer.name: initializer for initializer in graph.initializer}
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

        conv_attributes = {attr.name: helper.get_attribute_value(attr) for attr in conv.attribute}
        weight = initializers.get(conv.input[1])
        group = int(conv_attributes.get("group", 1))
        matches.append((conv, concat, constant, dynamic_input, int(prefix.shape[2])))

    preserved_inputs = [value.SerializeToString() for value in graph.input]
    preserved_outputs = [value.SerializeToString() for value in graph.output]
    preserved_initializers = [initializer.SerializeToString() for initializer in graph.initializer]
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
        onnx.save(model, temp_path)
        temp_path.replace(final_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(
        f"[Conv rewrite] graph={raw_path.name}, matched={len(matches)}, "
        f"Conv_pads_updated={len(matches)}, Concat_deleted={len(matches)}, "
        f"Constant_deleted={len(matches)}, initializers_changed=0"
    )


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
        for path in onnx_raw_folder.glob("*.onnx*"):
            if path.is_file():
                path.unlink()

        _install_tokenizer_v2_patches(load_unfused_convnext=True)
        try:
            model = Qwen3TTSModel.from_pretrained(
                download_path,
                device_map="cpu",
                dtype=torch.float32,
                attn_implementation="eager",
            )
        finally:
            _install_tokenizer_v2_patches(load_unfused_convnext=False)

        for block_index, upsample_block in enumerate(model.model.speech_tokenizer.model.decoder.upsample):
            for module_index, module in enumerate(upsample_block):
                if isinstance(module, Qwen3TTSTokenizerV2ConvNeXtBlockUnfused):
                    model.model.speech_tokenizer.model.decoder.upsample[block_index][module_index] = (
                        Qwen3TTSTokenizerV2ConvNeXtBlock.from_unfused(module)
                    )
        for layer in model.model.speech_tokenizer.model.decoder.quantizer.rvq_first.vq.layers:
            layer._codebook.precompute_embedding()
        for layer in model.model.speech_tokenizer.model.decoder.quantizer.rvq_rest.vq.layers:
            layer._codebook.precompute_embedding()
        for module in model.model.speech_tokenizer.model.decoder.decoder.modules():
            if isinstance(module, SnakeBeta):
                module.precompute()
        model.model = model.model.eval()

        talker_config = model.model.talker.config
        predictor_config = model.model.talker.code_predictor.config
        head_dim = talker_config.head_dim
        hidden_size = talker_config.hidden_size
        num_kv_heads = talker_config.num_key_value_heads
        num_layers_main = talker_config.num_hidden_layers
        num_code_groups = predictor_config.num_code_groups
        num_predictor_stages = num_code_groups - 1
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32

        # Fixed Qwen3-TTS geometry is read from the checkpoint or kept local here.
        speech_tokenizer_config = model.model.speech_tokenizer.config
        model_output_sample_rate = int(speech_tokenizer_config.output_sample_rate)
        n_mels = 128
        nfft_stft = 1024
        window_length = 1024
        hop_length = 256
        window_type = "hann"
        stop_token_ids = [int(talker_config.codec_eos_token_id)]
        model_samples_per_codec_frame = int(speech_tokenizer_config.decode_upsample_rate)
        samples_per_codec_frame = (
            model_samples_per_codec_frame * OUT_SAMPLE_RATE / model_output_sample_rate
        )
        samples_per_codec_frame = int(samples_per_codec_frame)
        tokenizer = AutoTokenizer.from_pretrained(
            download_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        instruction_prefix_token_ids = tokenizer.encode(
            "<|im_start|>system\n",
            add_special_tokens=False,
        )
        instruction_suffix_token_ids = tokenizer.encode(
            "<|im_end|>\n",
            add_special_tokens=False,
        )
        del tokenizer

        metadata = build_model_metadata(
            {
                "graph_layout": "strategy_prefill_decode_step",
                "mode": MODE,
                "language_id_map": LANGUAGE_ID_MAP,
                "out_sample_rate": OUT_SAMPLE_RATE,
                "max_seq_len": MAX_SEQ_LEN,
                "stop_token_ids": stop_token_ids,
                "use_f16_kv": USE_F16_KV,
                "compute_in_f32": COMPUTE_IN_F32,
                "shared_initializer_model_file": SHARED_MODEL_NAME,
                "shared_initializer_data_file": SHARED_DATA_NAME,
                "model_file_name_target_preprocess": Path(onnx_model_Target_Preprocess).name,
                "model_file_name_decoder": Path(onnx_model_Decoder).name,
                "model_file_name_decoder_stream": Path(onnx_model_Decoder_Stream).name,
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
                "vocab_size": predictor_config.vocab_size,
                **(
                    {
                        "in_sample_rate": IN_SAMPLE_RATE,
                        "model_file_name_reference_preprocess": Path(
                            onnx_model_Reference_Preprocess
                        ).name,
                    }
                    if MODE == "voice_clone"
                    else {}
                ),
                **(
                    {
                        "instruction_prefix_token_ids": instruction_prefix_token_ids,
                        "instruction_suffix_token_ids": instruction_suffix_token_ids,
                    }
                    if MODE in {"custom_voice", "voice_design"}
                    else {}
                ),
                **(
                    {
                        "speaker_id_map": SPEAKER_ID_MAP,
                        "speaker_dialect_map": SPEAKER_DIALECT_MAP,
                        "dialect_language_id_map": DIALECT_LANGUAGE_ID_MAP,
                    }
                    if MODE == "custom_voice"
                    else {}
                ),
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

        stft_model = STFT_Process(
            model_type='stft_B',
            n_fft=nfft_stft,
            win_length=window_length,
            hop_len=hop_length,
            max_frames=0,
            window_type=window_type,
            pad_mode='constant',
            center_pad=False,
        ).eval()

        if MODE == "voice_clone":
            reference_preprocess = TTS_REFERENCE_PREPROCESS(
                model,
                IN_SAMPLE_RATE,
                MAX_SEQ_LEN,
                stft_model,
                nfft_stft,
                n_mels,
            )
            reference_preprocess_raw = onnx_raw_folder / "QwenTTS_ReferencePreprocess.onnx"
            prompt_audio = torch.zeros(
                1,
                1,
                IN_SAMPLE_RATE,
                dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()],
            )
            prompt_text_ids = torch.zeros(1, 10, dtype=torch.int32)
            with torch.amp.autocast('cpu', dtype=torch.float16, enabled=USE_F16_ENCODER):
                export(
                    reference_preprocess,
                    (prompt_audio, prompt_text_ids),
                    str(reference_preprocess_raw),
                    ['prompt_audio', 'prompt_text_ids'],
                    ['codec_embed', 'speaker_embed', 'ref_prompt_text_embed'],
                    {
                        'prompt_audio': {2: 'audio_len'},
                        'prompt_text_ids': {1: 'prompt_text_len'},
                        'codec_embed': {1: 'ref_code_len'},
                        'ref_prompt_text_embed': {1: 'prompt_text_len'},
                    },
                )
            write_onnx_metadata(str(reference_preprocess_raw), metadata)
            fuse_static_zero_prefix_convs(
                str(reference_preprocess_raw),
                onnx_model_Reference_Preprocess,
                reference_preprocess.static_mimi_fusion_count,
            )
            del reference_preprocess, prompt_audio, prompt_text_ids, stft_model
        else:
            del stft_model

        target_preprocess = TTS_TARGET_PREPROCESS(model, MODE)
        language_id = torch.zeros(1, 1, dtype=torch.int32)
        target_text_ids = torch.zeros(1, 10, dtype=torch.int32)
        instruct_text_ids = torch.zeros(1, 1, dtype=torch.int32)
        target_outputs = ['hidden_states', 'ids_len', 'trailing_text_hidden', 'trailing_len_minus']
        target_axes = {
            'target_text_ids': {1: 'target_text_len'},
            'instruct_text_ids': {1: 'instruct_text_len'},
            'hidden_states': {1: 'ids_len'},
            'trailing_text_hidden': {1: 'trailing_len'},
        }
        if MODE == "voice_clone":
            target_args = (
                language_id,
                target_text_ids,
                instruct_text_ids,
                torch.zeros(1, 10, hidden_size),
                torch.zeros(1, 1, hidden_size),
                torch.zeros(1, 10, hidden_size),
            )
            target_inputs = [
                'language_id', 'target_text_ids', 'instruct_text_ids',
                'codec_embed', 'speaker_embed', 'ref_prompt_text_embed',
            ]
            target_axes.update({
                'codec_embed': {1: 'ref_code_len'},
                'ref_prompt_text_embed': {1: 'prompt_text_len'},
            })
        elif MODE == "custom_voice":
            target_args = (
                language_id,
                torch.zeros(1, 1, dtype=torch.int32),
                target_text_ids,
                instruct_text_ids,
            )
            target_inputs = ['language_id', 'speaker_id', 'target_text_ids', 'instruct_text_ids']
        else:
            target_args = (language_id, target_text_ids, instruct_text_ids)
            target_inputs = ['language_id', 'target_text_ids', 'instruct_text_ids']
        export(
            target_preprocess,
            target_args,
            onnx_model_Target_Preprocess,
            target_inputs,
            target_outputs,
            target_axes,
        )
        del target_preprocess, target_args

        predictor_core = TTS_PREDICTOR_MERGED(model, MAX_SEQ_LEN)
        predictor_embed_b = TTS_EMBED_B(model)
        predictor_lm_heads = torch.nn.ModuleList([
            TTS_PREDICTOR_LM_HEAD(model, stage)
            for stage in range(num_predictor_stages)
        ])
        predictor_embed_layers = torch.nn.ModuleList([
            TTS_EMBED_D(model, stage)
            for stage in range(num_predictor_stages)
        ])
        main_core = TTS_MAIN_MERGED(model, MAX_SEQ_LEN)

        kv_tensors = {
            'key': torch.zeros(1, num_kv_heads, 1, head_dim, 0, dtype=kv_dtype),
            'value': torch.zeros(1, num_kv_heads, 1, 0, head_dim, dtype=kv_dtype),
        }
        kv_inputs = [kv_tensors['key']] * num_layers_main + [kv_tensors['value']] * num_layers_main
        kv_in_names = [f'in_key_{index}' for index in range(num_layers_main)] + [
            f'in_value_{index}' for index in range(num_layers_main)
        ]
        kv_out_names = [f'out_key_{index}' for index in range(num_layers_main)] + [
            f'out_value_{index}' for index in range(num_layers_main)
        ]
        kv_axes = {}
        for index in range(num_layers_main):
            kv_axes[f'in_key_{index}'] = {4: 'history_len'}
            kv_axes[f'out_key_{index}'] = {4: 'kv_seq_len'}
            kv_axes[f'in_value_{index}'] = {3: 'history_len'}
            kv_axes[f'out_value_{index}'] = {3: 'kv_seq_len'}

        hidden_states = torch.zeros(1, 10, hidden_size)
        frame_codec_ids = torch.zeros(1, num_code_groups, dtype=torch.int32)
        trailing_text_hidden = torch.zeros(1, 10, hidden_size)
        gather_id = torch.zeros(1, dtype=torch.int32)
        history_len = torch.zeros(1, dtype=torch.int64)
        save_ids = torch.zeros(1, 10, dtype=torch.int32)
        # Representative values used only to trace dynamic strategy inputs.
        penalty_value = torch.tensor([0.8], dtype=torch.float32)
        penalty_range = torch.tensor([5], dtype=torch.int64)
        temperature = torch.tensor([0.8], dtype=torch.float32)
        top_k = torch.tensor([min(50, predictor_config.vocab_size)], dtype=torch.int64)
        top_p = torch.tensor([0.95], dtype=torch.float32)
        sampling_penalty = torch.tensor([1.05], dtype=torch.float32)

        for strategy in DECODE_STRATEGIES:
            prefill = TTS_MAIN_PREFILL_STRATEGY(main_core, strategy)
            prefill_args = [hidden_states]
            prefill_inputs = ['hidden_states']
            prefill_outputs = kv_out_names + ['last_hidden_state', 'codec_token_main']
            prefill_axes = {
                **{name: axes for name, axes in kv_axes.items() if name.startswith('out_')},
                'hidden_states': {1: 'ids_len'},
            }
            if strategy == 'sampling':
                prefill_args.extend([temperature, top_k, top_p])
                prefill_inputs.extend(['temperature', 'top_k', 'top_p'])
            prefill_outputs.append('kv_seq_len')
            export(
                prefill,
                prefill_args,
                onnx_model_Main_Prefill[strategy],
                prefill_inputs,
                prefill_outputs,
                prefill_axes,
            )
            del prefill

            main_decode = TTS_MAIN_DECODE_STRATEGY(model, main_core, strategy)
            main_args = kv_inputs + [
                frame_codec_ids,
                trailing_text_hidden,
                gather_id,
                history_len,
            ]
            main_inputs = kv_in_names + [
                'frame_codec_ids', 'trailing_text_hidden', 'gather_id', 'history_len',
            ]
            main_outputs = kv_out_names + ['last_hidden_state', 'codec_token_main']
            main_axes = {
                **kv_axes,
                'trailing_text_hidden': {1: 'trailing_len'},
            }
            if strategy == 'penalty_greedy':
                main_args.extend([save_ids, penalty_value, penalty_range])
                main_inputs.extend(['main_save_ids', 'penalty_value', 'penalty_range'])
            elif strategy == 'sampling':
                main_args.extend([save_ids, temperature, top_k, top_p, sampling_penalty])
                main_inputs.extend([
                    'main_save_ids', 'main_temperature', 'main_top_k',
                    'main_top_p', 'main_repetition_penalty',
                ])
            if strategy != 'greedy':
                main_outputs.append('main_save_ids_out')
                main_axes['main_save_ids'] = {1: 'main_history_len'}
                main_axes['main_save_ids_out'] = {1: 'main_history_len_out'}
            main_outputs.append('kv_seq_len')
            export(
                main_decode,
                main_args,
                onnx_model_Main_Decode[strategy],
                main_inputs,
                main_outputs,
                main_axes,
            )
            del main_decode

            predictor_frame = TTS_PREDICTOR_FRAME_STRATEGY(
                predictor_core,
                predictor_embed_b,
                predictor_lm_heads,
                predictor_embed_layers,
                strategy,
            )
            predictor_args = [
                torch.zeros(1, 1, dtype=torch.int32),
                torch.zeros(1, 1, hidden_size),
                torch.zeros(1, 0, dtype=torch.int32),
            ]
            predictor_inputs = ['codec_token_main_in', 'last_hidden_state_main', 'generated_codec_in']
            predictor_axes = {
                'generated_codec_in': {1: 'generated_codec_len'},
                'generated_codec': {1: 'generated_codec_len_out'},
            }
            if strategy == 'penalty_greedy':
                predictor_args.extend([penalty_value, penalty_range])
                predictor_inputs.extend(['predictor_penalty_value', 'predictor_penalty_range'])
            elif strategy == 'sampling':
                predictor_args.extend([temperature, top_k, top_p, sampling_penalty])
                predictor_inputs.extend([
                    'predictor_temperature', 'predictor_top_k',
                    'predictor_top_p', 'predictor_repetition_penalty',
                ])
            export(
                predictor_frame,
                predictor_args,
                onnx_model_Predictor_Frame[strategy],
                predictor_inputs,
                ['frame_codec_ids', 'generated_codec'],
                predictor_axes,
            )
            del predictor_frame

        del predictor_core, predictor_embed_b, predictor_lm_heads, predictor_embed_layers, main_core

        generated_codec = torch.zeros(1, num_code_groups * 10, dtype=torch.int32)
        decoder = TTS_DECODER(
            model,
            OUT_SAMPLE_RATE,
            model_output_sample_rate,
            MAX_SEQ_LEN,
            mode="voice_clone" if MODE == "voice_clone" else "custom_voice",
        )
        del model
        gc.collect()

        export(
            decoder,
            (generated_codec,),
            onnx_model_Decoder_Raw,
            ['generated_codec'],
            ['generated_wav', 'generated_len'],
            {
                'generated_codec': {1: 'generated_codec_len'},
                'generated_wav': {2: 'generated_wav_len'},
            },
        )
        write_onnx_metadata(onnx_model_Decoder_Raw, metadata)
        fuse_static_zero_prefix_convs(
            onnx_model_Decoder_Raw,
            onnx_model_Decoder,
            decoder.static_conv_fusion_count,
        )
        export(
            decoder,
            (torch.zeros(1, num_code_groups * STREAM_WINDOW_FRAMES, dtype=torch.int32),),
            onnx_model_Decoder_Stream_Raw,
            ['generated_codec'],
            ['generated_wav', 'generated_len'],
            None,
        )
        write_onnx_metadata(onnx_model_Decoder_Stream_Raw, metadata)
        fuse_static_zero_prefix_convs(
            onnx_model_Decoder_Stream_Raw,
            onnx_model_Decoder_Stream,
            decoder.static_conv_fusion_count,
        )
        del decoder, generated_codec

        export(
            METADATA_CARRIER(),
            (torch.zeros(1, dtype=torch.int64),),
            onnx_model_Metadata,
            ['metadata_marker'],
            ['metadata_marker_out'],
            None,
        )
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
        shutil.rmtree(onnx_raw_folder)

    print('Compact strategy export done!')


if DO_EXPORT:
    run_compact_strategy_export()
    print('\nStart running the TTS by ONNXRuntime via Inference_QwenTTS_ONNX.py.\nNow loading . . . it could cost minutes.')
    raise SystemExit(subprocess.call([
        sys.executable,
        str(script_dir / "Inference_QwenTTS_ONNX.py"),
        "--onnx-folder",
        str(onnx_folder),
    ]))
pass
