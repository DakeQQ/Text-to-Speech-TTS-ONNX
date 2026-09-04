from __future__ import annotations

import gc
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as tokenizer_mod
from transformers.models.mimi import modeling_mimi as mimi_mod


SCRIPT_DIR = Path(__file__).resolve().parent
BREEZE_REPOSITORY = SCRIPT_DIR.parents[2] / "breeze-tts-main"
if str(BREEZE_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(BREEZE_REPOSITORY))

from models.breeze import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    BreezeConfig,
    BreezeForConditionalGeneration,
)
from Shared_Weights import (  # noqa: E402
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    build_decode_step_graphs,
    bundle_shared_initializers,
    copy_text_tokenizer,
)


# =============================================================================
# Easy-to-edit export configuration
# =============================================================================
# Change values in this section when using a different checkpoint or runtime.
# Paths
download_path              = str(Path.home() / "Downloads" / "Breeze-TTS-2")  # Local Breeze-TTS-2 checkpoint.
ONNX_FOLDER                = SCRIPT_DIR / "BreezeTTS_ONNX"                    # Exported ONNX files.

# Export behavior
MAX_SEQ_LEN                = 2048       # Maximum sequence length supported by the graphs.
DO_EXPORT                  = True       # Run the export and demo when this file starts.
USE_BATCH                  = False      # Export conditional/unconditional batch graphs.
STREAM_WINDOW_FRAMES       = 7          # Codec frames produced by the streaming decoder.

# Runtime precision
USE_F16_KV                 = True       # Store transformer key/value caches in float16.
COMPUTE_IN_F32             = False      # Compute attention and matmuls in float32.
USE_F16_ENCODER            = False      # Use float16 autocast while tracing the audio encoder.

# Quantization preparation
REORDER_DOWNPROJ_FOR_QUANT = True       # Reorder MLP channels for quantization.
REORDER_OPROJ_FOR_QUANT    = True       # Reorder attention output channels for quantization.
REORDER_KEY                = "absmean"  # Channel score: absmean, L4, rms, or std.

# Audio and ONNX export
OPSET                      = 20         # ONNX opset used for exported graphs.
IN_SAMPLE_RATE             = 24000      # Input prompt-audio sample rate.
OUT_SAMPLE_RATE            = 24000      # Output waveform sample rate.
IN_AUDIO_DTYPE             = "F32"      # Input audio type: F16, F32, or INT16.
OUT_AUDIO_DTYPE            = "F32"      # Output audio type: F16, F32, or INT16.
# =============================================================================

_AUDIO_DTYPES = {"F16": torch.float16, "F32": torch.float32, "INT16": torch.int16}

REORDER_SCORE_KEYS = frozenset(("absmean", "L4", "rms", "std"))

DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")

FINAL_MODEL_NAMES = (
    "BreezeTTS_Metadata.onnx",
    "BreezeTTS_ReferencePreprocess.onnx",
    "BreezeTTS_TargetPreprocess.onnx",
    "BreezeTTS_MainPrefill_greedy.onnx",
    "BreezeTTS_MainPrefill_penalty_greedy.onnx",
    "BreezeTTS_MainPrefill_sampling.onnx",
    "BreezeTTS_DecodeStep_greedy.onnx",
    "BreezeTTS_DecodeStep_penalty_greedy.onnx",
    "BreezeTTS_DecodeStep_sampling.onnx",
    "BreezeTTS_Decoder.onnx",
    "BreezeTTS_Decoder_Stream.onnx",
    SHARED_MODEL_NAME,
    SHARED_DATA_NAME,
)


def _mimi_embed(self):
    cached = getattr(self, "_embed", None)
    if cached is None:
        cached = self.embed_sum / self.cluster_usage.unsqueeze(-1)
        self._embed = cached
    return cached


def _mimi_embed_t(self):
    cached = getattr(self, "_embed_T", None)
    if cached is None:
        cached = self.embed_sum.t() / self.cluster_usage
        self._embed_T = cached
    return cached


def _mimi_embed_norm(self):
    cached = getattr(self, "_embed_norm", None)
    if cached is None:
        embed_t = self.embed_T
        cached = (embed_t * embed_t).sum(0, keepdim=True)
        self._embed_norm = cached
    return cached


def _mimi_quantize(self, hidden_states):
    dot_product = torch.matmul(hidden_states, self.embed_T)
    distance = self.embed_norm - (dot_product + dot_product)
    return distance.argmin(dim=-1).int()


def _mimi_encode(self, hidden_states):
    return self.quantize(hidden_states)


def _mimi_static_padding_forward(self, hidden_states, padding_cache=None):
    if self.static_padding > 0:
        hidden_states = torch.cat((self.static_left_padding, hidden_states), dim=-1)
    return self.conv(hidden_states)


def _install_mimi_codebook_patch():
    codebook_class = mimi_mod.MimiEuclideanCodebook
    codebook_class.embed = property(_mimi_embed)
    codebook_class.embed_T = property(_mimi_embed_t)
    codebook_class.embed_norm = property(_mimi_embed_norm)
    codebook_class.quantize = _mimi_quantize
    codebook_class.encode = _mimi_encode


class Qwen3TTSTokenizerV2CausalConvNet(torch.nn.Module):
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
        frames = (length - self.kernel_size + self.padding) / self.stride + 1
        ideal_length = (math.ceil(frames) - 1) * self.stride + (self.kernel_size - self.padding)
        return ideal_length - length

    def forward(self, hidden_state):
        if self.padding:
            hidden_state = torch.cat((self.left_padding, hidden_state), dim=-1)
        return self.conv(hidden_state).contiguous()


class Qwen3TTSTokenizerV2CausalTransConvNet(torch.nn.Module):
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
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = tokenizer_mod.Qwen3TTSTokenizerV2CausalConvNet(
            dim, dim, kernel_size=7, groups=dim, dilation=1
        )
        self.norm = torch.nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(dim, 4 * dim)
        self.act = torch.nn.GELU()
        self.pwconv2 = torch.nn.Linear(4 * dim, dim)
        self.gamma = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        hidden_states = self.gamma * hidden_states
        return residual + hidden_states.transpose(1, 2)


class Qwen3TTSTokenizerV2ConvNeXtBlock(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = tokenizer_mod.Qwen3TTSTokenizerV2CausalConvNet(
            dim, dim, kernel_size=7, groups=dim, dilation=1
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
        first_weight = unfused.pwconv1.weight.data
        first_bias = unfused.pwconv1.bias.data
        fused.pwconv1.weight.data = first_weight * norm_weight.unsqueeze(0)
        fused.pwconv1.bias.data = first_bias + first_weight @ norm_bias

        gamma = unfused.gamma.data
        second_weight = unfused.pwconv2.weight.data
        second_bias = unfused.pwconv2.bias.data
        fused.pwconv2.weight.data = second_weight * gamma.unsqueeze(1)
        fused.pwconv2.bias.data = second_bias * gamma
        fused.act = unfused.act
        return fused

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dwconv(hidden_states).transpose(1, 2)
        hidden_states = self.norm(hidden_states)
        hidden_states = self.pwconv1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.pwconv2(hidden_states)
        return residual + hidden_states.transpose(1, 2)


class SnakeBeta(torch.nn.Module):
    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(in_features) * alpha)
        self.beta = torch.nn.Parameter(torch.zeros(in_features) * alpha)
        self.no_div_by_zero = 1.0e-9
        self._alpha_exp = None
        self._beta_inv = None

    def precompute(self):
        self._alpha_exp = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1)
        beta_exp = torch.exp(self.beta).unsqueeze(0).unsqueeze(-1)
        self._beta_inv = 1.0 / (beta_exp + self.no_div_by_zero)

    def forward(self, hidden_states):
        if self._alpha_exp is None:
            alpha_exp = torch.exp(self.alpha).unsqueeze(0).unsqueeze(-1)
            beta_inv = 1.0 / (
                torch.exp(self.beta).unsqueeze(0).unsqueeze(-1) + self.no_div_by_zero
            )
        else:
            alpha_exp = self._alpha_exp
            beta_inv = self._beta_inv
        return hidden_states + beta_inv * torch.sin(hidden_states * alpha_exp).square()


class EuclideanCodebook(torch.nn.Module):
    def __init__(self, dim: int, codebook_size: int, epsilon: float = 1e-5):
        super().__init__()
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
        codebook_dim = dim if codebook_dim is None else codebook_dim
        self.project_out = (
            torch.nn.Linear(codebook_dim, dim) if codebook_dim != dim else torch.nn.Identity()
        )
        self._codebook = EuclideanCodebook(codebook_dim, codebook_size, epsilon)

    def decode(self, codes):
        return self.project_out(self._codebook.decode(codes))


class ResidualVectorQuantization(torch.nn.Module):
    def __init__(self, *, num_quantizers: int, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def decode(self, codes):
        quantized = None
        for index, layer_codes in enumerate(torch.split(codes, 1, dim=0)):
            layer_quantized = self.layers[index].decode(layer_codes)
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
        self.dimension = dimension
        self.input_dimension = input_dimension or dimension
        self.output_dimension = output_dimension or dimension
        self.input_proj = (
            torch.nn.Identity()
            if self.input_dimension == dimension and not force_projection
            else torch.nn.Conv1d(self.input_dimension, dimension, 1, bias=False)
        )
        self.output_proj = (
            torch.nn.Identity()
            if self.output_dimension == dimension and not force_projection
            else torch.nn.Conv1d(dimension, self.output_dimension, 1, bias=False)
        )
        self.vq = ResidualVectorQuantization(
            dim=dimension, codebook_size=bins, num_quantizers=n_q
        )

    def decode(self, codes):
        return self.output_proj(self.vq.decode(codes.transpose(0, 1)))


class SplitResidualVectorQuantizer(torch.nn.Module):
    def __init__(self, *, n_q: int = 8, n_q_semantic: int = 1, **kwargs):
        super().__init__()
        q_dropout = kwargs.pop("q_dropout", False)
        self.n_q_semantic = n_q_semantic
        self.rvq_first = ResidualVectorQuantizer(
            n_q=n_q_semantic, force_projection=True, q_dropout=False, **kwargs
        )
        self.rvq_rest = ResidualVectorQuantizer(
            n_q=n_q - n_q_semantic,
            force_projection=True,
            q_dropout=q_dropout,
            **kwargs,
        )

    def decode(self, codes):
        first_code = codes[:, [self.n_q_semantic]]
        quantized_first = self.rvq_first.vq.decode(first_code)
        quantized_first = self.rvq_first.output_proj(quantized_first)
        rest_codes = codes[:, self.n_q_semantic :].transpose(0, 1)
        quantized_rest = self.rvq_rest.vq.decode(rest_codes)
        quantized_rest = self.rvq_rest.output_proj(quantized_rest)
        return quantized_first + quantized_rest


def _install_tokenizer_patches(load_unfused_convnext=False):
    tokenizer_mod.Qwen3TTSTokenizerV2CausalConvNet = Qwen3TTSTokenizerV2CausalConvNet
    tokenizer_mod.Qwen3TTSTokenizerV2CausalTransConvNet = Qwen3TTSTokenizerV2CausalTransConvNet
    tokenizer_mod.SnakeBeta = SnakeBeta
    tokenizer_mod.EuclideanCodebook = EuclideanCodebook
    tokenizer_mod.VectorQuantization = VectorQuantization
    tokenizer_mod.ResidualVectorQuantization = ResidualVectorQuantization
    tokenizer_mod.ResidualVectorQuantizer = ResidualVectorQuantizer
    tokenizer_mod.SplitResidualVectorQuantizer = SplitResidualVectorQuantizer
    tokenizer_mod.Qwen3TTSTokenizerV2ConvNeXtBlockUnfused = Qwen3TTSTokenizerV2ConvNeXtBlockUnfused
    tokenizer_mod.Qwen3TTSTokenizerV2ConvNeXtBlock = (
        Qwen3TTSTokenizerV2ConvNeXtBlockUnfused
        if load_unfused_convnext
        else Qwen3TTSTokenizerV2ConvNeXtBlock
    )


_install_mimi_codebook_patch()
_install_tokenizer_patches()


def shape_dim_as_tensor(tensor, dim):
    return torch._shape_as_tensor(tensor)[dim].unsqueeze(0)


def dynamic_batch_axis():
    return {0: "cfg_batch"} if USE_BATCH else {}


def static_batch_output_dimensions(output_dimensions, output_names):
    dimensions = {name: dict(axes) for name, axes in output_dimensions.items()}
    if not USE_BATCH:
        for name in output_names:
            dimensions.setdefault(name, {})[0] = 1
    return dimensions


def validate_export_controls():
    if type(USE_BATCH) is not bool:
        raise TypeError(f"USE_BATCH must be bool, got {type(USE_BATCH).__name__}")
    for name, value in (
        ("IN_AUDIO_DTYPE", IN_AUDIO_DTYPE),
        ("OUT_AUDIO_DTYPE", OUT_AUDIO_DTYPE),
    ):
        if not isinstance(value, str) or value.upper() not in _AUDIO_DTYPES:
            choices = ", ".join(sorted(_AUDIO_DTYPES))
            raise ValueError(f"Unknown {name} {value!r}; expected one of: {choices}")
    for name, value in (
        ("IN_SAMPLE_RATE", IN_SAMPLE_RATE),
        ("OUT_SAMPLE_RATE", OUT_SAMPLE_RATE),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    for name, value in (
        ("REORDER_DOWNPROJ_FOR_QUANT", REORDER_DOWNPROJ_FOR_QUANT),
        ("REORDER_OPROJ_FOR_QUANT", REORDER_OPROJ_FOR_QUANT),
    ):
        if type(value) is not bool:
            raise TypeError(f"{name} must be bool, got {type(value).__name__}")
    if REORDER_KEY not in REORDER_SCORE_KEYS:
        choices = ", ".join(sorted(REORDER_SCORE_KEYS))
        raise ValueError(f"Unknown REORDER_KEY {REORDER_KEY!r}; expected one of: {choices}")


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, scale, epsilon, axis):
        variance = hidden_states.float().square().mean(dim=axis, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(graph, hidden_states, scale, epsilon, axis):
        output = graph.op(
            "SimplifiedLayerNormalization",
            hidden_states,
            scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )
        output.setType(hidden_states.type())
        return output


def simplified_layer_norm(hidden_states, scale, epsilon, axis=-1):
    return SIMPLIFIED_LAYER_NORM.apply(hidden_states, scale, float(epsilon), axis)


def build_model_metadata(*sections):
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
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, onnx_path)


def set_onnx_static_output_dimensions(onnx_path, output_dimensions):
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    outputs = {value.name: value for value in model.graph.output}
    missing = sorted(set(output_dimensions) - set(outputs))
    if missing:
        raise ValueError(
            f"Cannot set dimensions for missing outputs in {Path(onnx_path).name}: "
            f"{', '.join(missing)}"
        )
    for name, axes in output_dimensions.items():
        dimensions = outputs[name].type.tensor_type.shape.dim
        for axis, value in axes.items():
            if axis >= len(dimensions):
                raise ValueError(
                    f"Output {name!r} rank {len(dimensions)} has no axis {axis}"
                )
            dimensions[axis].Clear()
            dimensions[axis].dim_value = int(value)
    onnx.save(model, onnx_path)


def eliminate_redundant_cast_chains(onnx_path):
    import onnx
    from onnx import helper

    onnx_path = Path(onnx_path)
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    producers = {
        output: node
        for node in graph.node
        for output in node.output
        if output
    }
    public_outputs = {value.name for value in graph.output}

    def cast_destination(node):
        if node.op_type != "Cast" or node.domain or len(node.attribute) != 1:
            return None
        attribute = node.attribute[0]
        if attribute.name != "to":
            return None
        return int(helper.get_attribute_value(attribute))

    remap = {}
    removed = set()
    for child in graph.node:
        child_destination = cast_destination(child)
        if (
            child_destination is None
            or len(child.input) != 1
            or len(child.output) != 1
            or child.output[0] in public_outputs
        ):
            continue
        parent = producers.get(child.input[0])
        if (
            parent is None
            or len(parent.output) != 1
            or cast_destination(parent) != child_destination
        ):
            continue
        remap[child.output[0]] = parent.output[0]
        removed.add(id(child))

    def resolve(name):
        while name in remap:
            name = remap[name]
        return name

    if removed:
        retained_nodes = []
        for node in graph.node:
            if id(node) in removed:
                continue
            for index, name in enumerate(node.input):
                node.input[index] = resolve(name)
            retained_nodes.append(node)
        del graph.node[:]
        graph.node.extend(retained_nodes)
        retained_value_info = [
            value for value in graph.value_info if value.name not in remap
        ]
        del graph.value_info[:]
        graph.value_info.extend(retained_value_info)

        with tempfile.NamedTemporaryFile(
            dir=onnx_path.parent,
            prefix=f".{onnx_path.stem}.",
            suffix=".onnx",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            onnx.save(model, temporary_path)
            temporary_path.replace(onnx_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return len(removed)


def fuse_static_zero_prefix_convs(source_path, destination_path, expected_match_count):
    from collections import defaultdict

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    source_path = Path(source_path)
    destination_path = Path(destination_path)
    model = onnx.load(source_path, load_external_data=False)
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    matches = []
    for conv in graph.node:
        if conv.op_type != "Conv" or conv.domain or len(conv.input) < 2:
            continue
        concat = producers.get(conv.input[0])
        if (
            concat is None
            or concat.op_type != "Concat"
            or concat.domain
            or len(concat.input) != 2
            or len(concat.output) != 1
            or consumers[concat.output[0]] != [conv]
        ):
            continue
        concat_attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in concat.attribute
        }
        if concat_attributes.get("axis") not in (-1, 2):
            continue
        constant = producers.get(concat.input[0])
        if (
            constant is None
            or constant.op_type != "Constant"
            or constant.domain
            or len(constant.output) != 1
            or len(constant.attribute) != 1
            or constant.attribute[0].name != "value"
            or consumers[constant.output[0]] != [concat]
        ):
            continue
        prefix = numpy_helper.to_array(
            constant.attribute[0].t, base_dir=str(source_path.parent)
        )
        if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[2] <= 0:
            continue
        if not np.all(prefix == 0):
            continue
        matches.append((conv, concat, constant, concat.input[1], int(prefix.shape[2])))

    if len(matches) != expected_match_count:
        raise ValueError(
            f"Static zero-prefix Conv rewrite found {len(matches)} matches in "
            f"{source_path.name}; expected {expected_match_count}"
        )

    removed = set()
    for conv, concat, constant, dynamic_input, left_padding in matches:
        conv.input[0] = dynamic_input
        retained_attributes = [attribute for attribute in conv.attribute if attribute.name != "pads"]
        del conv.attribute[:]
        conv.attribute.extend(retained_attributes)
        conv.attribute.append(helper.make_attribute("pads", [left_padding, 0]))
        removed.update((id(concat), id(constant)))
    retained_nodes = [node for node in graph.node if id(node) not in removed]
    del graph.node[:]
    graph.node.extend(retained_nodes)

    with tempfile.NamedTemporaryFile(
        dir=destination_path.parent,
        prefix=f".{destination_path.stem}.",
        suffix=".onnx",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    onnx.save(model, temporary_path)
    temporary_path.replace(destination_path)
    source_path.unlink(missing_ok=True)
    print(f"[Conv rewrite] {destination_path.name}: fused {len(matches)} static prefixes")


class TTS_TEXT_ENCODER(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_encoder = model.text_encoder.eval()
        self.text_encoder_proj = model.text_encoder_proj.eval()

    def forward(self, text_ids):
        position_ids = torch.arange(
            text_ids.shape[1], dtype=torch.int32, device=text_ids.device
        ).unsqueeze(0)
        hidden_states = self.text_encoder(
            input_ids=text_ids,
            attention_mask=None,
            position_ids=position_ids,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
        return self.text_encoder_proj(hidden_states).float()


class TTS_AUDIO_EMBEDDING(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding.embed_audio_tokens
        self.projector = embedding.audio_embeds_projector
        self.register_buffer(
            "offsets", embedding.audio_tokens_offsets.to(torch.int32), persistent=False
        )

    def forward(self, codec_ids):
        hidden_states = self.embedding(codec_ids + self.offsets)
        if self.projector is not None:
            hidden_states = self.projector(hidden_states)
        return hidden_states.sum(dim=2)


class TTS_AUDIO_ENCODER(torch.nn.Module):
    def __init__(self, tokenizer_model, num_codebooks, max_seq_len):
        super().__init__()
        self.encoder = tokenizer_model.encoder.eval()
        self.num_codebooks = num_codebooks
        self.valid_num_quantizers = tokenizer_model.config.encoder_valid_num_quantizers
        self.model_sample_rate = int(tokenizer_model.config.input_sample_rate)
        self.input_resample_scale = float(self.model_sample_rate / IN_SAMPLE_RATE)
        self.integer_input = IN_AUDIO_DTYPE.upper() == "INT16"
        self.inv_int16 = 1.0 / 32768.0
        self._prepare_static_mimi_padding()
        self._fuse_encoder_weights()

        first_attention = self.encoder.encoder_transformer.layers[0].self_attn
        self.num_heads = first_attention.num_heads
        self.qk_heads = self.num_heads * 2
        self.head_dim = first_attention.head_dim
        self.head_dim_half = self.head_dim // 2
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inverse_frequency = first_attention.rotary_emb.inv_freq
        theta = (positions * inverse_frequency).unsqueeze(1).unsqueeze(0)
        cosine, sine = torch.cos(theta), torch.sin(theta)
        self.register_buffer(
            "rope_cos", torch.cat((cosine, cosine), dim=-1).half(), persistent=False
        )
        self.register_buffer(
            "rope_sin", torch.cat((-sine, sine), dim=-1).half(), persistent=False
        )
        causal_mask = (
            ~torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        ).to(torch.int8) * -128
        self.register_buffer(
            "attention_mask",
            causal_mask.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _replace_gelu(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate="tanh"))
            else:
                self._replace_gelu(child)

    def _prepare_static_mimi_padding(self):
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
        self.static_mimi_fusion_count = sum(
            module.static_padding > 0
            for module in self.encoder.encoder.modules()
            if isinstance(module, mimi_mod.MimiConv1d)
            and hasattr(module, "static_padding")
        )
        if self.static_mimi_fusion_count > patched:
            raise RuntimeError("Mimi static-padding fusion count exceeds patched modules")

    def _fuse_encoder_weights(self):
        scale_factor = self.encoder.encoder_transformer.layers[0].self_attn.head_dim ** -0.25
        with torch.no_grad():
            for layer in self.encoder.encoder_transformer.layers:
                attention = layer.self_attn
                q_proj, k_proj, v_proj = attention.q_proj, attention.k_proj, attention.v_proj
                qkv = torch.nn.Linear(
                    q_proj.in_features,
                    q_proj.out_features + k_proj.out_features + v_proj.out_features,
                    bias=q_proj.bias is not None,
                )
                qkv.weight.copy_(
                    torch.cat(
                        (q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight),
                        dim=0,
                    )
                )
                if q_proj.bias is not None:
                    qkv.bias.copy_(
                        torch.cat(
                            (q_proj.bias * scale_factor, k_proj.bias * scale_factor, v_proj.bias),
                            dim=0,
                        )
                    )
                attention.qkv = qkv
                del attention.q_proj, attention.k_proj, attention.v_proj

                input_norm = layer.input_layernorm
                if qkv.bias is None:
                    qkv.bias = torch.nn.Parameter(qkv.weight @ input_norm.bias)
                else:
                    qkv.bias.add_(qkv.weight @ input_norm.bias)
                qkv.weight.mul_(input_norm.weight.unsqueeze(0))
                input_norm.elementwise_affine = False
                input_norm.weight = input_norm.bias = None

                post_norm = layer.post_attention_layernorm
                first_linear = layer.mlp.fc1
                if first_linear.bias is None:
                    first_linear.bias = torch.nn.Parameter(first_linear.weight @ post_norm.bias)
                else:
                    first_linear.bias.add_(first_linear.weight @ post_norm.bias)
                first_linear.weight.mul_(post_norm.weight.unsqueeze(0))
                post_norm.elementwise_affine = False
                post_norm.weight = post_norm.bias = None

                attention_scale = layer.self_attn_layer_scale.scale
                layer.self_attn.o_proj.weight.mul_(attention_scale.unsqueeze(1))
                if layer.self_attn.o_proj.bias is not None:
                    layer.self_attn.o_proj.bias.mul_(attention_scale)
                mlp_scale = layer.mlp_layer_scale.scale
                layer.mlp.fc2.weight.mul_(mlp_scale.unsqueeze(1))
                if layer.mlp.fc2.bias is not None:
                    layer.mlp.fc2.bias.mul_(mlp_scale)

    def rotate_half(self, hidden_states):
        hidden_states = hidden_states.view(
            1, -1, self.qk_heads, 2, self.head_dim_half
        ).flip(-2)
        return hidden_states.view(1, -1, self.qk_heads, self.head_dim)

    def forward(self, prompt_audio):
        prompt_audio = prompt_audio.float()
        if self.integer_input:
            prompt_audio *= self.inv_int16
        if self.input_resample_scale != 1.0:
            prompt_audio = torch.nn.functional.interpolate(
                prompt_audio,
                scale_factor=self.input_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        hidden_states = self.encoder.encoder(prompt_audio).transpose(1, 2)
        sequence_length = hidden_states.shape[1]
        rope_cos = self.rope_cos[:, :sequence_length].to(hidden_states.dtype)
        rope_sin = self.rope_sin[:, :sequence_length].to(hidden_states.dtype)
        attention_mask = self.attention_mask[
            :, :, :sequence_length, :sequence_length
        ].to(hidden_states.dtype)

        for layer in self.encoder.encoder_transformer.layers:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states).reshape(
                1, -1, self.qk_heads + self.num_heads, self.head_dim
            )
            query_key, value = torch.split(qkv, (self.qk_heads, self.num_heads), dim=-2)
            query_key = query_key * rope_cos + self.rotate_half(query_key) * rope_sin
            query, key = torch.split(query_key, (self.num_heads, self.num_heads), dim=-2)
            query = query.transpose(1, 2)
            key = key.permute(0, 2, 3, 1)
            value = value.transpose(1, 2)
            attention = torch.softmax(
                torch.matmul(query, key) + attention_mask, dim=-1
            )
            attention = torch.matmul(attention, value).transpose(1, 2).reshape(
                1, -1, layer.self_attn.o_proj.in_features
            )
            hidden_states = residual + layer.self_attn.o_proj(attention)
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp.fc2(
                layer.mlp.activation_fn(layer.mlp.fc1(hidden_states))
            )

        embeddings = self.encoder.downsample(hidden_states.transpose(1, 2))
        codes = self.encoder.quantizer.encode(embeddings, self.valid_num_quantizers)
        return codes[: self.num_codebooks].permute(1, 2, 0).int()


class TTS_REFERENCE_PREPROCESS(torch.nn.Module):
    def __init__(self, model, tokenizer_model, num_codebooks, max_seq_len):
        super().__init__()
        self.audio_encoder = TTS_AUDIO_ENCODER(tokenizer_model, num_codebooks, max_seq_len)
        self.text_encoder = TTS_TEXT_ENCODER(model)

    def forward(self, prompt_audio, ref_text_ids):
        ref_code = self.audio_encoder(prompt_audio)
        ref_text_embed = self.text_encoder(ref_text_ids)
        return ref_code, ref_text_embed


class TTS_TARGET_PREPROCESS(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_encoder = TTS_TEXT_ENCODER(model)
        self.audio_embedding = TTS_AUDIO_EMBEDDING(model.backbone_model.embed_tokens)
        num_codebooks = model.config.num_codebooks
        with torch.no_grad():
            audio_eos_embed = self.audio_embedding(
                torch.zeros(1, 1, num_codebooks, dtype=torch.int32)
            ).float()
        self.register_buffer("audio_eos_embed", audio_eos_embed, persistent=False)

    def forward(self, branch_text_ids, ref_code, ref_text_embed):
        branch_text_embed = self.text_encoder(branch_text_ids)
        ref_audio_embed = self.audio_embedding(ref_code)
        ref_audio_eos_embed = self.audio_eos_embed[:, : ref_code.shape[1]]
        inputs_embeds = torch.cat(
            (ref_text_embed, ref_audio_embed, ref_audio_eos_embed, branch_text_embed),
            dim=1,
        ).float()
        attention_mask = torch.ones_like(inputs_embeds[..., 0], dtype=torch.int64)
        return inputs_embeds, attention_mask


def _flip_rotate(hidden_states):
    half = hidden_states.shape[-1] // 2
    return hidden_states.reshape(*hidden_states.shape[:-1], 2, half).flip(-2).reshape_as(
        hidden_states
    )


def _build_rope_tables(rotary_embedding, hidden_size, max_seq_len):
    positions = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    hidden_states = torch.zeros(1, 1, hidden_size, dtype=torch.float32)
    cosine, sine = rotary_embedding(hidden_states, positions)
    sine = sine.clone()
    sine[..., : sine.shape[-1] // 2] *= -1.0
    return cosine[0].half(), sine[0].half()


def _module_epsilon(module):
    for attribute in ("variance_epsilon", "eps"):
        if hasattr(module, attribute):
            return float(getattr(module, attribute))
    raise TypeError(f"Cannot derive RMSNorm epsilon from {type(module).__name__}")


def _projection_bias(projection, dtype, device):
    if projection.bias is not None:
        return projection.bias
    return torch.zeros(projection.out_features, dtype=dtype, device=device)


def _fuse_main_layer(layer, num_heads, num_key_value_heads, head_dim, hidden_size):
    attention = layer.self_attn
    if hasattr(attention, "qkv"):
        raise RuntimeError("Main layer QKV projection was already fused")

    q_proj, k_proj, v_proj = attention.q_proj, attention.k_proj, attention.v_proj
    widths = (q_proj.out_features, k_proj.out_features, v_proj.out_features)
    expected_widths = (
        num_heads * head_dim,
        num_key_value_heads * head_dim,
        num_key_value_heads * head_dim,
    )
    if widths != expected_widths:
        raise ValueError(f"Unexpected Main QKV widths {widths}; expected {expected_widths}")
    if any(
        projection.in_features != hidden_size
        for projection in (q_proj, k_proj, v_proj)
    ):
        raise ValueError(
            "Main QKV input widths do not all match the hidden size "
            f"{hidden_size}"
        )

    q_norm, k_norm = attention.q_norm, attention.k_norm
    if q_norm.weight.shape != k_norm.weight.shape or q_norm.weight.numel() != head_dim:
        raise ValueError(
            "Main Q/K RMSNorm weights must have matching head-dimension shapes"
        )
    q_epsilon = _module_epsilon(q_norm)
    k_epsilon = _module_epsilon(k_norm)
    if q_epsilon != k_epsilon:
        raise ValueError(
            f"Main Q/K RMSNorm epsilons differ: {q_epsilon} != {k_epsilon}"
        )

    has_bias = any(projection.bias is not None for projection in (q_proj, k_proj, v_proj))
    qkv = torch.nn.Linear(
        hidden_size,
        sum(widths),
        bias=has_bias,
        device=q_proj.weight.device,
        dtype=q_proj.weight.dtype,
    )
    qkv.weight.copy_(torch.cat((q_proj.weight, k_proj.weight, v_proj.weight), dim=0))
    if has_bias:
        qkv.bias.copy_(
            torch.cat(
                tuple(
                    _projection_bias(projection, qkv.weight.dtype, qkv.weight.device)
                    for projection in (q_proj, k_proj, v_proj)
                ),
                dim=0,
            )
        )

    combined_qk_scale = head_dim**-0.25 * math.sqrt(head_dim)
    q_scale = q_norm.weight * combined_qk_scale
    k_scale = k_norm.weight * combined_qk_scale
    attention.qk_norm_weight = torch.nn.Parameter(
        torch.cat(
            (q_scale.repeat(num_heads), k_scale.repeat(num_key_value_heads)), dim=0
        ).reshape(1, 1, num_heads + num_key_value_heads, head_dim)
    )
    attention.qk_norm_epsilon = q_epsilon
    attention.q_width, attention.k_width, attention.v_width = widths

    input_norm = layer.input_layernorm
    if input_norm.weight.numel() != hidden_size:
        raise ValueError("Main input RMSNorm width does not match hidden size")
    layer.input_norm_epsilon = _module_epsilon(input_norm)
    qkv.weight.mul_(input_norm.weight.unsqueeze(0) * math.sqrt(hidden_size))
    attention.qkv = qkv
    del attention.q_proj, attention.k_proj, attention.v_proj
    del attention.q_norm, attention.k_norm, layer.input_layernorm

    post_norm = layer.post_attention_layernorm
    gate_proj, up_proj = layer.mlp.gate_proj, layer.mlp.up_proj
    if not (
        gate_proj.out_features
        == up_proj.out_features
        == layer.mlp.down_proj.in_features
    ):
        raise ValueError("Main gate/up/down intermediate widths must match")
    if gate_proj.in_features != hidden_size or up_proj.in_features != hidden_size:
        raise ValueError("Main gate/up input widths do not match the hidden size")
    layer.post_norm_epsilon = _module_epsilon(post_norm)
    post_scale = post_norm.weight.unsqueeze(0) * math.sqrt(hidden_size)
    has_mlp_bias = gate_proj.bias is not None or up_proj.bias is not None
    gate_up = torch.nn.Linear(
        gate_proj.in_features,
        gate_proj.out_features + up_proj.out_features,
        bias=has_mlp_bias,
        device=gate_proj.weight.device,
        dtype=gate_proj.weight.dtype,
    )
    gate_up.weight.copy_(
        torch.cat((gate_proj.weight * post_scale, up_proj.weight * post_scale), dim=0)
    )
    if has_mlp_bias:
        gate_up.bias.copy_(
            torch.cat(
                (
                    _projection_bias(gate_proj, gate_up.weight.dtype, gate_up.weight.device),
                    _projection_bias(up_proj, gate_up.weight.dtype, gate_up.weight.device),
                ),
                dim=0,
            )
        )
    layer.mlp.gate_up_proj = gate_up
    del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm


def _fuse_depth_layer(layer, num_heads, num_key_value_heads, head_dim, hidden_size):
    attention = layer.self_attn
    if hasattr(attention, "qkv"):
        raise RuntimeError("Depth layer QKV projection was already fused")

    q_proj, k_proj, v_proj = attention.q_proj, attention.k_proj, attention.v_proj
    widths = (q_proj.out_features, k_proj.out_features, v_proj.out_features)
    expected_widths = (
        num_heads * head_dim,
        num_key_value_heads * head_dim,
        num_key_value_heads * head_dim,
    )
    if widths != expected_widths:
        raise ValueError(f"Unexpected Depth QKV widths {widths}; expected {expected_widths}")
    if any(
        projection.in_features != hidden_size
        for projection in (q_proj, k_proj, v_proj)
    ):
        raise ValueError(
            "Depth QKV input widths do not all match the hidden size "
            f"{hidden_size}"
        )
    if hasattr(attention, "q_norm") or hasattr(attention, "k_norm"):
        raise ValueError("Breeze Depth attention must not have Q/K RMSNorm modules")

    split_scale = head_dim**-0.25
    has_bias = any(projection.bias is not None for projection in (q_proj, k_proj, v_proj))
    qkv = torch.nn.Linear(
        hidden_size,
        sum(widths),
        bias=has_bias,
        device=q_proj.weight.device,
        dtype=q_proj.weight.dtype,
    )
    qkv.weight.copy_(
        torch.cat(
            (q_proj.weight * split_scale, k_proj.weight * split_scale, v_proj.weight),
            dim=0,
        )
    )
    if has_bias:
        qkv.bias.copy_(
            torch.cat(
                (
                    _projection_bias(q_proj, qkv.weight.dtype, qkv.weight.device)
                    * split_scale,
                    _projection_bias(k_proj, qkv.weight.dtype, qkv.weight.device)
                    * split_scale,
                    _projection_bias(v_proj, qkv.weight.dtype, qkv.weight.device),
                ),
                dim=0,
            )
        )
    attention.q_width, attention.k_width, attention.v_width = widths

    input_norm = layer.input_layernorm
    if input_norm.weight.numel() != hidden_size:
        raise ValueError("Depth input RMSNorm width does not match hidden size")
    layer.input_norm_epsilon = _module_epsilon(input_norm)
    qkv.weight.mul_(input_norm.weight.unsqueeze(0) * math.sqrt(hidden_size))
    attention.qkv = qkv
    del attention.q_proj, attention.k_proj, attention.v_proj, layer.input_layernorm

    post_norm = layer.post_attention_layernorm
    gate_proj, up_proj = layer.mlp.gate_proj, layer.mlp.up_proj
    if not (
        gate_proj.out_features
        == up_proj.out_features
        == layer.mlp.down_proj.in_features
    ):
        raise ValueError("Depth gate/up/down intermediate widths must match")
    if gate_proj.in_features != hidden_size or up_proj.in_features != hidden_size:
        raise ValueError("Depth gate/up input widths do not match the hidden size")
    layer.post_norm_epsilon = _module_epsilon(post_norm)
    post_scale = post_norm.weight.unsqueeze(0) * math.sqrt(hidden_size)
    has_mlp_bias = gate_proj.bias is not None or up_proj.bias is not None
    gate_up = torch.nn.Linear(
        gate_proj.in_features,
        gate_proj.out_features + up_proj.out_features,
        bias=has_mlp_bias,
        device=gate_proj.weight.device,
        dtype=gate_proj.weight.dtype,
    )
    gate_up.weight.copy_(
        torch.cat((gate_proj.weight * post_scale, up_proj.weight * post_scale), dim=0)
    )
    if has_mlp_bias:
        gate_up.bias.copy_(
            torch.cat(
                (
                    _projection_bias(gate_proj, gate_up.weight.dtype, gate_up.weight.device),
                    _projection_bias(up_proj, gate_up.weight.dtype, gate_up.weight.device),
                ),
                dim=0,
            )
        )
    layer.mlp.gate_up_proj = gate_up
    del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm


def _channel_score(weight, key, dimensions):
    if key not in REORDER_SCORE_KEYS:
        choices = ", ".join(sorted(REORDER_SCORE_KEYS))
        raise ValueError(f"Unknown reorder score key {key!r}; expected one of: {choices}")
    absolute = weight.abs()
    if key == "rms":
        return weight.square().mean(dim=dimensions).sqrt()
    if key == "L4":
        return absolute.pow(4).mean(dim=dimensions).pow(0.25)
    if key == "std":
        moved = weight.movedim(dimensions, tuple(range(len(dimensions))))
        return moved.reshape(-1, weight.shape[-1]).std(dim=0)
    return absolute.mean(dim=dimensions)


def _reorder_transformer_layer(
    layer,
    num_heads,
    num_key_value_heads,
    head_dim,
    reorder_down,
    reorder_output,
    score_key,
):
    if getattr(layer, "_breeze_quant_reordered", False):
        raise RuntimeError("Transformer channel reorder was already applied")

    with torch.no_grad():
        if reorder_down:
            down_weight = layer.mlp.down_proj.weight
            permutation = torch.argsort(_channel_score(down_weight, score_key, (0,)))
            intermediate_size = layer.mlp.down_proj.in_features
            gate_up_weight = layer.mlp.gate_up_proj.weight
            layer.mlp.gate_up_proj.weight.copy_(
                torch.cat(
                    (
                        gate_up_weight[:intermediate_size][permutation],
                        gate_up_weight[intermediate_size:][permutation],
                    ),
                    dim=0,
                )
            )
            if layer.mlp.gate_up_proj.bias is not None:
                gate_up_bias = layer.mlp.gate_up_proj.bias
                layer.mlp.gate_up_proj.bias.copy_(
                    torch.cat(
                        (
                            gate_up_bias[:intermediate_size][permutation],
                            gate_up_bias[intermediate_size:][permutation],
                        ),
                        dim=0,
                    )
                )
            layer.mlp.down_proj.weight.copy_(down_weight[:, permutation])

        if reorder_output:
            attention = layer.self_attn
            output_weight = attention.o_proj.weight
            output_by_head = output_weight.reshape(
                output_weight.shape[0], num_heads, head_dim
            )
            heads_per_key_value = num_heads // num_key_value_heads
            permutations = []
            for kv_head in range(num_key_value_heads):
                first_head = kv_head * heads_per_key_value
                grouped = output_by_head[
                    :, first_head : first_head + heads_per_key_value, :
                ]
                permutations.append(
                    torch.argsort(_channel_score(grouped, score_key, (0, 1)))
                )

            reordered_output = output_by_head.clone()
            for head in range(num_heads):
                permutation = permutations[head // heads_per_key_value]
                reordered_output[:, head] = output_by_head[:, head, permutation]
            output_weight.copy_(reordered_output.reshape_as(output_weight))

            qk_heads = num_heads + num_key_value_heads
            qkv_weight = attention.qkv.weight
            qkv_by_head = qkv_weight.reshape(-1, head_dim, qkv_weight.shape[1]).clone()
            for kv_head, permutation in enumerate(permutations):
                qkv_by_head[qk_heads + kv_head] = qkv_by_head[
                    qk_heads + kv_head, permutation
                ]
            qkv_weight.copy_(qkv_by_head.reshape_as(qkv_weight))
            if attention.qkv.bias is not None:
                qkv_bias = attention.qkv.bias.reshape(-1, head_dim).clone()
                for kv_head, permutation in enumerate(permutations):
                    qkv_bias[qk_heads + kv_head] = qkv_bias[
                        qk_heads + kv_head, permutation
                    ]
                attention.qkv.bias.copy_(qkv_bias.reshape_as(attention.qkv.bias))

    layer._breeze_quant_reordered = True


class TTS_MAIN_GEOMETRY(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        cosine, sine = _build_rope_tables(
            model.backbone_model.rotary_emb, model.config.hidden_size, max_seq_len
        )
        self.register_buffer("cosine", cosine, persistent=False)
        self.register_buffer("sine", sine, persistent=False)
        self.mask_dtype = torch.float16 if USE_F16_KV and not COMPUTE_IN_F32 else torch.float32
        causal_mask = (~torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))).to(
            torch.int8
        ) * -128
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.register_buffer(
            "key_positions", torch.arange(max_seq_len, dtype=torch.int32), persistent=False
        )

    def prefill(self, inputs_embeds, attention_mask):
        position_ids = torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1
        position_ids = torch.where(
            attention_mask.to(torch.bool), position_ids, torch.ones_like(position_ids)
        )
        cosine = self.cosine[position_ids].unsqueeze(2).float()
        sine = self.sine[position_ids].unsqueeze(2).float()
        sequence_length = inputs_embeds.shape[1]
        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        padding_mask = (
            (~attention_mask.to(torch.bool)).to(torch.int8).unsqueeze(1) * -128
        )
        additive_mask = torch.minimum(
            causal_mask.unsqueeze(0), padding_mask
        ).unsqueeze(1).unsqueeze(1).to(self.mask_dtype)
        return cosine, sine, additive_mask, shape_dim_as_tensor(inputs_embeds, 1)

    def decode(self, pad_lengths, history_len):
        position_ids = (history_len - pad_lengths).to(torch.int32).unsqueeze(1)
        cosine = self.cosine[position_ids].unsqueeze(2).float()
        sine = self.sine[position_ids].unsqueeze(2).float()
        kv_seq_len = history_len + 1
        key_positions = self.key_positions[: kv_seq_len[0]]
        allowed = key_positions.unsqueeze(0) >= pad_lengths.to(torch.int32).unsqueeze(1)
        additive_mask = (~allowed).to(self.mask_dtype).unsqueeze(1).unsqueeze(1).unsqueeze(1) * -128.0
        return cosine, sine, additive_mask, kv_seq_len


class TTS_MAIN(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone_model
        self.lm_head = model.lm_head
        first_attention = self.backbone.layers[0].self_attn
        self.num_layers = len(self.backbone.layers)
        self.num_heads = first_attention.config.num_attention_heads
        self.num_key_value_heads = first_attention.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = first_attention.head_dim
        self.hidden_size = first_attention.q_proj.in_features
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.compute_in_f32 = COMPUTE_IN_F32
        self.use_f16_kv = USE_F16_KV
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((self.hidden_size,), self.hidden_size**-0.5, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "head_norm_scale",
            torch.full((self.head_dim,), self.head_dim**-0.5, dtype=torch.float32),
            persistent=False,
        )
        self.final_norm_epsilon = _module_epsilon(self.backbone.norm)
        self.save_keys = [None] * self.num_layers
        self.save_values = [None] * self.num_layers
        self._fuse_weights()

    def _fuse_weights(self):
        with torch.no_grad():
            for layer in self.backbone.layers:
                _fuse_main_layer(
                    layer,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.hidden_size,
                )
                _reorder_transformer_layer(
                    layer,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    REORDER_DOWNPROJ_FOR_QUANT,
                    REORDER_OPROJ_FOR_QUANT,
                    REORDER_KEY,
                )

    def forward(self, *all_inputs):
        hidden_states, cosine, sine, attention_mask = all_inputs[-4:]
        batch_size, sequence_length = hidden_states.shape[:2]

        for index, layer in enumerate(self.backbone.layers):
            residual = hidden_states
            normalized = simplified_layer_norm(
                hidden_states, self.hidden_norm_scale, layer.input_norm_epsilon
            )
            qkv = layer.self_attn.qkv(normalized)
            query_key, value = torch.split(
                qkv,
                (
                    layer.self_attn.q_width + layer.self_attn.k_width,
                    layer.self_attn.v_width,
                ),
                dim=-1,
            )
            query_key = query_key.reshape(
                batch_size, sequence_length, self.qk_heads, self.head_dim
            )
            value = value.reshape(
                batch_size, sequence_length, self.num_key_value_heads, self.head_dim
            )
            query_key = simplified_layer_norm(
                query_key,
                self.head_norm_scale,
                layer.self_attn.qk_norm_epsilon,
            ) * layer.self_attn.qk_norm_weight
            query_key = query_key * cosine + _flip_rotate(query_key) * sine
            if self.use_f16_kv and not self.compute_in_f32:
                query_key = query_key.half()
            query, key = torch.split(
                query_key, (self.num_heads, self.num_key_value_heads), dim=-2
            )
            query = query.reshape(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            ).permute(0, 2, 3, 1, 4)
            key = key.permute(0, 2, 3, 1).unsqueeze(2)
            value = value.transpose(1, 2).unsqueeze(2)

            if self.use_f16_kv:
                if self.compute_in_f32:
                    key = key.half()
                value = value.half()

            key = torch.cat((all_inputs[index], key), dim=-1)
            value = torch.cat((all_inputs[index + self.num_layers], value), dim=-2)
            self.save_keys[index] = key
            self.save_values[index] = value

            if self.use_f16_kv and self.compute_in_f32:
                scores = torch.matmul(query, key.float()) + attention_mask
                context = torch.matmul(torch.softmax(scores, dim=-1), value.float())
            else:
                scores = torch.matmul(query, key) + attention_mask
                context = torch.matmul(torch.softmax(scores, dim=-1), value)
                if self.use_f16_kv:
                    context = context.float()
            context = context.permute(0, 3, 1, 2, 4).reshape(
                batch_size, sequence_length, layer.self_attn.o_proj.in_features
            )
            hidden_states = residual + layer.self_attn.o_proj(context)
            residual = hidden_states
            normalized = simplified_layer_norm(
                hidden_states, self.hidden_norm_scale, layer.post_norm_epsilon
            )
            gate_up = layer.mlp.gate_up_proj(normalized)
            gate, up = torch.split(
                gate_up, (layer.mlp.down_proj.in_features,) * 2, dim=-1
            )
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        last_hidden_state = simplified_layer_norm(
            hidden_states[:, -1:, :],
            self.backbone.norm.weight,
            self.final_norm_epsilon,
        )
        logits = self.lm_head(last_hidden_state[:, 0, :].float()).float()
        return *self.save_keys, *self.save_values, last_hidden_state.float(), logits


class TTS_DEPTH(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.depth_model = model.depth_decoder.model
        config = model.config.depth_decoder_config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.compute_in_f32 = COMPUTE_IN_F32
        self.use_f16_kv = USE_F16_KV
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((self.hidden_size,), self.hidden_size**-0.5, dtype=torch.float32),
            persistent=False,
        )
        final_norm = self.depth_model.norm
        self.final_norm_epsilon = _module_epsilon(final_norm)
        codebook_weight = model.depth_decoder.codebooks_head.weight
        expected_stages = model.config.num_codebooks - 1
        if tuple(codebook_weight.shape) != (
            expected_stages,
            self.hidden_size,
            config.vocab_size,
        ):
            raise ValueError(
                f"Unexpected Depth codebook-head shape {tuple(codebook_weight.shape)}"
            )
        final_scale = final_norm.weight * math.sqrt(self.hidden_size)
        heads = []
        with torch.no_grad():
            for stage in range(expected_stages):
                head = torch.nn.Linear(
                    self.hidden_size,
                    config.vocab_size,
                    bias=False,
                    device=codebook_weight.device,
                    dtype=codebook_weight.dtype,
                )
                head.weight.copy_((codebook_weight[stage] * final_scale[:, None]).T)
                heads.append(head)
        self.codebook_heads = torch.nn.ModuleList(heads)
        del self.depth_model.norm
        self.save_keys = [None] * self.num_layers
        self.save_values = [None] * self.num_layers
        self._fuse_weights()

    def _fuse_weights(self):
        with torch.no_grad():
            for layer in self.depth_model.layers:
                _fuse_depth_layer(
                    layer,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    self.hidden_size,
                )
                _reorder_transformer_layer(
                    layer,
                    self.num_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    REORDER_DOWNPROJ_FOR_QUANT,
                    REORDER_OPROJ_FOR_QUANT,
                    REORDER_KEY,
                )

    def forward(self, *all_inputs):
        hidden_states, cosine, sine, attention_mask = all_inputs[-4:]
        batch_size, sequence_length = hidden_states.shape[:2]

        for index, layer in enumerate(self.depth_model.layers):
            residual = hidden_states
            normalized = simplified_layer_norm(
                hidden_states, self.hidden_norm_scale, layer.input_norm_epsilon
            )
            qkv = layer.self_attn.qkv(normalized)
            query_key, value = torch.split(
                qkv,
                (
                    layer.self_attn.q_width + layer.self_attn.k_width,
                    layer.self_attn.v_width,
                ),
                dim=-1,
            )
            query_key = query_key.reshape(
                batch_size,
                sequence_length,
                self.num_heads + self.num_key_value_heads,
                self.head_dim,
            )
            value = value.reshape(
                batch_size, sequence_length, self.num_key_value_heads, self.head_dim
            )
            query_key = query_key * cosine + _flip_rotate(query_key) * sine
            if self.use_f16_kv and not self.compute_in_f32:
                query_key = query_key.half()
            query, key = torch.split(
                query_key, (self.num_heads, self.num_key_value_heads), dim=-2
            )
            query = query.reshape(
                batch_size,
                sequence_length,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            ).permute(0, 2, 3, 1, 4)
            key = key.permute(0, 2, 3, 1).unsqueeze(2)
            value = value.transpose(1, 2).unsqueeze(2)

            if self.use_f16_kv:
                if self.compute_in_f32:
                    key = key.half()
                value = value.half()

            key = torch.cat((all_inputs[index], key), dim=-1)
            value = torch.cat((all_inputs[index + self.num_layers], value), dim=-2)
            self.save_keys[index] = key
            self.save_values[index] = value

            if self.use_f16_kv and self.compute_in_f32:
                scores = torch.matmul(query, key.float()) + attention_mask
                context = torch.matmul(torch.softmax(scores, dim=-1), value.float())
            else:
                scores = torch.matmul(query, key) + attention_mask
                context = torch.matmul(torch.softmax(scores, dim=-1), value)
                if self.use_f16_kv:
                    context = context.float()
            context = context.permute(0, 3, 1, 2, 4).reshape(
                batch_size, sequence_length, layer.self_attn.o_proj.in_features
            )
            hidden_states = residual + layer.self_attn.o_proj(context)
            residual = hidden_states
            normalized = simplified_layer_norm(
                hidden_states, self.hidden_norm_scale, layer.post_norm_epsilon
            )
            gate_up = layer.mlp.gate_up_proj(normalized)
            gate, up = torch.split(
                gate_up, (layer.mlp.down_proj.in_features,) * 2, dim=-1
            )
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        hidden_states = simplified_layer_norm(
            hidden_states, self.hidden_norm_scale, self.final_norm_epsilon
        )
        return *self.save_keys, *self.save_values, hidden_states.float()


class TTS_MAIN_CORE(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        self.geometry = TTS_MAIN_GEOMETRY(model, max_seq_len)
        self.transformer = TTS_MAIN(model)
        self.audio_embedding = TTS_AUDIO_EMBEDDING(model.backbone_model.embed_tokens)
        self.num_layers = self.transformer.num_layers
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        self.register_buffer(
            "empty_key",
            torch.zeros(
                1,
                self.transformer.num_key_value_heads,
                1,
                self.transformer.head_dim,
                0,
                dtype=kv_dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "empty_value",
            torch.zeros(
                1,
                self.transformer.num_key_value_heads,
                1,
                0,
                self.transformer.head_dim,
                dtype=kv_dtype,
            ),
            persistent=False,
        )
        suppress = torch.zeros(1, model.config.vocab_size + 1, dtype=torch.float32)
        suppress[:, model.config.vocab_size - 3 : model.config.vocab_size] = -1.0e9
        self.register_buffer("suppress", suppress, persistent=False)

    def prefill(self, inputs_embeds, attention_mask):
        cosine, sine, additive_mask, kv_seq_len = self.geometry.prefill(
            inputs_embeds, attention_mask
        )
        if USE_BATCH:
            batch_size = inputs_embeds.shape[0]
            empty_keys = [
                self.empty_key.expand(batch_size, -1, -1, -1, -1)
            ] * self.num_layers
            empty_values = [
                self.empty_value.expand(batch_size, -1, -1, -1, -1)
            ] * self.num_layers
        else:
            empty_keys = [self.empty_key] * self.num_layers
            empty_values = [self.empty_value] * self.num_layers
        outputs = self.transformer(
            *empty_keys,
            *empty_values,
            inputs_embeds,
            cosine,
            sine,
            additive_mask,
        )
        return *outputs, kv_seq_len

    def decode(self, kv, frame_codec_ids, pad_lengths, history_len):
        if USE_BATCH:
            batch_size = pad_lengths.shape[0]
            frame_batch = frame_codec_ids.unsqueeze(1).expand(batch_size, -1, -1)
        else:
            frame_batch = frame_codec_ids.unsqueeze(1)
        hidden_states = self.audio_embedding(frame_batch).float()
        cosine, sine, additive_mask, kv_seq_len = self.geometry.decode(
            pad_lengths, history_len
        )
        outputs = self.transformer(
            *kv, hidden_states, cosine, sine, additive_mask
        )
        return *outputs, kv_seq_len

    def guided_logits(self, logits, guidance_weights):
        return _guided_logits(logits, guidance_weights) + self.suppress


class TTS_DEPTH_CORE(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = TTS_DEPTH(model)
        self.num_layers = self.transformer.num_layers
        self.num_codebooks = model.config.num_codebooks
        self.num_stages = self.num_codebooks - 1
        self.vocab_size = model.config.depth_decoder_config.vocab_size
        cosine, sine = _build_rope_tables(
            self.depth_model.rotary_emb,
            model.config.depth_decoder_config.hidden_size,
            self.num_codebooks + 1,
        )
        self.register_buffer("cosine", cosine, persistent=False)
        self.register_buffer("sine", sine, persistent=False)
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        self.register_buffer(
            "empty_key",
            torch.zeros(
                1,
                self.transformer.num_key_value_heads,
                1,
                self.transformer.head_dim,
                0,
                dtype=kv_dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "empty_value",
            torch.zeros(
                1,
                self.transformer.num_key_value_heads,
                1,
                0,
                self.transformer.head_dim,
                dtype=kv_dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "prefill_mask",
            torch.tensor([[[[[0, -128], [0, 0]]]]], dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "decode_mask",
            torch.zeros(1, 1, 1, 1, self.num_codebooks + 1, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "empty_ids", torch.zeros(1, 0, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "zero_placeholder", torch.zeros(1, 1, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "stage_offsets",
            torch.arange(self.num_stages, dtype=torch.int32) * self.vocab_size,
            persistent=False,
        )
        suppress = torch.zeros(1, self.vocab_size, dtype=torch.float32)
        suppress[:, model.config.vocab_size - 3 :] = -1.0e9
        self.register_buffer("suppress", suppress, persistent=False)
        self.mask_dtype = torch.float16 if USE_F16_KV and not COMPUTE_IN_F32 else torch.float32

    @property
    def depth_model(self):
        return self.transformer.depth_model

    @property
    def codebook_heads(self):
        return self.transformer.codebook_heads

    def initial_hidden_states(self, codec_token_main, last_hidden_state_main):
        if USE_BATCH:
            batch_size = last_hidden_state_main.shape[0]
            main_token_batch = codec_token_main.expand(batch_size, -1)
            placeholder = self.zero_placeholder.expand(batch_size, -1)
        else:
            main_token_batch = codec_token_main
            placeholder = self.zero_placeholder
        input_ids = torch.cat((placeholder, main_token_batch), dim=1)
        input_embeds = self.depth_model.embed_tokens(input_ids)
        main_hidden = last_hidden_state_main[:, 0, :]
        if self.depth_model.backbone_hidden_state_projector is not None:
            main_hidden = self.depth_model.backbone_hidden_state_projector(main_hidden)
        input_embeds = torch.cat((main_hidden.unsqueeze(1), input_embeds[:, 1:, :]), dim=1)
        return self.depth_model.inputs_embeds_projector(input_embeds)

    def stage_hidden_states(self, token, stage):
        token_batch = token + self.stage_offsets[stage]
        return self.depth_model.inputs_embeds_projector(
            self.depth_model.embed_tokens(token_batch)
        )

    def prefill(self, hidden_states):
        if USE_BATCH:
            batch_size = hidden_states.shape[0]
            keys = [
                self.empty_key.expand(batch_size, -1, -1, -1, -1)
            ] * self.num_layers
            values = [
                self.empty_value.expand(batch_size, -1, -1, -1, -1)
            ] * self.num_layers
        else:
            keys = [self.empty_key] * self.num_layers
            values = [self.empty_value] * self.num_layers
        return self.transformer(
            *keys,
            *values,
            hidden_states,
            self.cosine[:2].unsqueeze(0).unsqueeze(2).float(),
            self.sine[:2].unsqueeze(0).unsqueeze(2).float(),
            self.prefill_mask.to(self.mask_dtype),
        )

    def decode(self, kv, hidden_states, stage):
        history_length = stage + 2
        return self.transformer(
            *kv,
            hidden_states,
            self.cosine[stage + 1 : stage + 2].unsqueeze(0).unsqueeze(2).float(),
            self.sine[stage + 1 : stage + 2].unsqueeze(0).unsqueeze(2).float(),
            self.decode_mask[..., :history_length].to(self.mask_dtype),
        )

    def logits(self, hidden_states, stage, guidance_weights):
        logits = self.codebook_heads[stage](hidden_states[:, -1, :])
        return _guided_logits(logits, guidance_weights) + self.suppress


def _guided_logits(logits, guidance_weights):
    return (logits * guidance_weights.reshape(-1, 1)).sum(dim=0, keepdim=True)


class APPLY_PENALTY(torch.nn.Module):
    def forward(self, logits, save_ids, penalty_value, penalty_range):
        target_indices = save_ids[:, -penalty_range:]
        return INT32_GATHER_SCATTER_PENALTY.apply(
            logits, target_indices, penalty_value
        )


class INT32_GATHER_SCATTER_PENALTY(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target_indices, penalty_value):
        target_indices_long = target_indices.long()
        penalized = logits.gather(1, target_indices_long) * penalty_value
        return logits.scatter(1, target_indices_long, penalized)

    @staticmethod
    def symbolic(graph, logits, target_indices, penalty_value):
        selected = graph.op(
            "GatherElements", logits, target_indices, axis_i=1
        )
        penalized = graph.op("Mul", selected, penalty_value)
        return graph.op(
            "ScatterElements", logits, target_indices, penalized, axis_i=1
        )


class SIGN_AWARE_REPETITION_PENALTY(torch.autograd.Function):
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
    def symbolic(graph, logits, repetition_penalty, previous_ids):
        previous_logits = graph.op("GatherElements", logits, previous_ids, axis_i=1)
        zero = graph.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))
        previous_scores = graph.op(
            "Where",
            graph.op("Less", previous_logits, zero),
            graph.op("Mul", previous_logits, repetition_penalty),
            graph.op("Div", previous_logits, repetition_penalty),
        )
        return graph.op(
            "ScatterElements", logits, previous_ids, previous_scores, axis_i=1
        )


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    def sample(self, scores, temperature, top_k, top_p):
        sorted_scores, sorted_indices = torch.topk(
            scores, k=top_k, dim=-1, largest=True, sorted=True
        )
        probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        keep = (cumulative - probabilities) <= top_p
        kept_mass = torch.where(keep, cumulative, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax((cumulative >= threshold).int(), dim=-1, keepdim=True)
        return torch.gather(sorted_indices, 1, winner).int()

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        scores = SIGN_AWARE_REPETITION_PENALTY.apply(
            logits, repetition_penalty, previous_ids
        )
        sampled_id = self.sample(scores, temperature, top_k, top_p)
        return sampled_id, torch.cat((previous_ids, sampled_id), dim=-1)


class TTS_MAIN_PREFILL_STRATEGY(torch.nn.Module):
    def __init__(self, main_core, strategy):
        super().__init__()
        self.main_core = main_core
        self.strategy = strategy
        self.num_layers = main_core.num_layers
        self.sampling = TOPK_TOPP_SAMPLING()

    def forward(self, *args):
        inputs_embeds, attention_mask, guidance_weights = args[:3]
        outputs = self.main_core.prefill(
            inputs_embeds, attention_mask
        )
        kv = outputs[: self.num_layers * 2]
        last_hidden_state, logits, kv_seq_len = outputs[-3:]
        logits = self.main_core.guided_logits(logits, guidance_weights)
        if self.strategy == "sampling":
            token = self.sampling.sample(logits, *args[3:6])
        else:
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return *kv, last_hidden_state, token, kv_seq_len


class TTS_MAIN_DECODE_STRATEGY(torch.nn.Module):
    def __init__(self, main_core, strategy):
        super().__init__()
        self.main_core = main_core
        self.strategy = strategy
        self.num_layers = main_core.num_layers
        self.penalty = APPLY_PENALTY()
        self.sampling = TOPK_TOPP_SAMPLING()

    def forward(self, *args):
        kv_count = self.num_layers * 2
        kv = args[:kv_count]
        frame_codec_ids = args[kv_count]
        pad_lengths = args[kv_count + 1]
        history_len = args[kv_count + 2]
        guidance_weights = args[kv_count + 3]
        cursor = kv_count + 4

        outputs = self.main_core.decode(
            kv, frame_codec_ids, pad_lengths, history_len
        )
        kv_out = outputs[:kv_count]
        last_hidden_state, logits, kv_seq_len = outputs[-3:]
        logits = self.main_core.guided_logits(logits, guidance_weights)

        if self.strategy == "greedy":
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            return *kv_out, last_hidden_state, token, kv_seq_len

        save_ids = args[cursor]
        cursor += 1
        if self.strategy == "penalty_greedy":
            penalty_value, penalty_range = args[cursor : cursor + 2]
            penalized = self.penalty(logits, save_ids, penalty_value, penalty_range)
            use_penalty = shape_dim_as_tensor(save_ids, 1) >= penalty_range
            logits = torch.where(use_penalty, penalized, logits)
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            save_ids = torch.cat((save_ids, token), dim=-1)
        else:
            token, save_ids = self.sampling(logits, *args[cursor : cursor + 4], save_ids)
        return *kv_out, last_hidden_state, token, save_ids, kv_seq_len


class TTS_PREDICTOR_FRAME_STRATEGY(torch.nn.Module):
    def __init__(self, depth_core, strategy):
        super().__init__()
        self.depth_core = depth_core
        self.strategy = strategy
        self.num_layers = depth_core.num_layers
        self.num_stages = depth_core.num_stages
        self.penalty = APPLY_PENALTY()
        self.sampling = TOPK_TOPP_SAMPLING()

    def _select(self, logits, strategy_args, save_ids, stage):
        if self.strategy == "greedy":
            return torch.argmax(logits, dim=-1, keepdim=True).int(), save_ids
        if self.strategy == "penalty_greedy":
            penalty_value, penalty_range = strategy_args
            if stage > 0:
                logits = self.penalty(logits, save_ids, penalty_value, penalty_range)
            token = torch.argmax(logits, dim=-1, keepdim=True).int()
            save_ids = token if stage == 0 else torch.cat((save_ids, token), dim=-1)
            return token, save_ids
        if stage == 0:
            token = self.sampling.sample(logits, *strategy_args[:3])
            return token, token
        return self.sampling(logits, *strategy_args, save_ids)

    def forward(self, *args):
        codec_token_main, last_hidden_state_main, generated_codec, guidance_weights = args[:4]
        strategy_args = args[4:]
        hidden_states = self.depth_core.initial_hidden_states(
            codec_token_main, last_hidden_state_main
        )
        outputs = self.depth_core.prefill(hidden_states)
        kv = list(outputs[: self.num_layers * 2])
        hidden_states = outputs[-1]
        save_ids = self.depth_core.empty_ids
        frame_codec_ids = codec_token_main

        logits = self.depth_core.logits(
            hidden_states, 0, guidance_weights
        )
        token, save_ids = self._select(
            logits, strategy_args, save_ids, 0
        )
        frame_codec_ids = torch.cat((frame_codec_ids, token), dim=-1)

        for stage in range(1, self.num_stages):
            stage_token = (
                token.expand(last_hidden_state_main.shape[0], -1)
                if USE_BATCH
                else token
            )
            hidden_states = self.depth_core.stage_hidden_states(stage_token, stage)
            outputs = self.depth_core.decode(kv, hidden_states, stage)
            kv = list(outputs[: self.num_layers * 2])
            hidden_states = outputs[-1]
            logits = self.depth_core.logits(
                hidden_states, stage, guidance_weights
            )
            token, save_ids = self._select(
                logits, strategy_args, save_ids, stage
            )
            frame_codec_ids = torch.cat((frame_codec_ids, token), dim=-1)

        generated_codec = torch.cat((generated_codec, frame_codec_ids), dim=-1)
        return frame_codec_ids, generated_codec


class TTS_DECODER(torch.nn.Module):
    def __init__(self, tokenizer_model, num_codebooks, max_seq_len):
        super().__init__()
        self.decoder = tokenizer_model.decoder.eval()
        self.num_codebooks = num_codebooks
        self.hidden_size = self.decoder.config.hidden_size
        self.rms_norm_eps = float(self.decoder.pre_transformer.config.rms_norm_eps)
        self.model_sample_rate = int(tokenizer_model.config.output_sample_rate)
        self.output_resample_scale = float(OUT_SAMPLE_RATE / self.model_sample_rate)
        self.output_dtype = OUT_AUDIO_DTYPE.upper()
        self.register_buffer(
            "rms_norm_scale",
            torch.full((self.hidden_size,), self.hidden_size**-0.5),
            persistent=False,
        )
        self.static_conv_fusion_count = sum(
            isinstance(module, Qwen3TTSTokenizerV2CausalConvNet)
            and module.padding > 0
            for module in self.decoder.modules()
        )
        self._fuse_decoder_weights()

        first_attention = self.decoder.pre_transformer.layers[0].self_attn
        self.num_heads = first_attention.config.num_attention_heads
        self.qk_heads = self.num_heads * 2
        self.head_dim = first_attention.head_dim
        self.head_dim_half = self.head_dim // 2
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inverse_frequency = self.decoder.pre_transformer.rotary_emb.inv_freq
        theta = (positions * inverse_frequency).unsqueeze(1).unsqueeze(0)
        cosine, sine = torch.cos(theta), torch.sin(theta)
        self.register_buffer(
            "rope_cos", torch.cat((cosine, cosine), dim=-1).half(), persistent=False
        )
        self.register_buffer(
            "rope_sin", torch.cat((-sine, sine), dim=-1).half(), persistent=False
        )
        positions = torch.arange(max_seq_len)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        self.sliding_window = int(self.decoder.config.sliding_window)
        if self.sliding_window <= 0:
            raise ValueError(
                f"Codec decoder sliding window must be positive, got {self.sliding_window}"
            )
        allowed = (distance >= 0) & (distance < self.sliding_window)
        causal_mask = (~allowed).to(torch.int8) * -128
        self.register_buffer(
            "attention_mask",
            causal_mask.unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _fuse_decoder_weights(self):
        scale_factor = self.decoder.pre_transformer.layers[0].self_attn.head_dim ** -0.25
        norm_factor = self.hidden_size**0.5
        with torch.no_grad():
            for layer in self.decoder.pre_transformer.layers:
                attention = layer.self_attn
                q_proj, k_proj, v_proj = attention.q_proj, attention.k_proj, attention.v_proj
                qkv = torch.nn.Linear(
                    q_proj.in_features,
                    q_proj.out_features + k_proj.out_features + v_proj.out_features,
                    bias=q_proj.bias is not None,
                )
                qkv.weight.copy_(
                    torch.cat(
                        (q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight),
                        dim=0,
                    )
                )
                if q_proj.bias is not None:
                    qkv.bias.copy_(
                        torch.cat(
                            (q_proj.bias * scale_factor, k_proj.bias * scale_factor, v_proj.bias),
                            dim=0,
                        )
                    )
                attention.qkv = qkv
                del attention.q_proj, attention.k_proj, attention.v_proj

                input_norm = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(input_norm)
                del layer.input_layernorm

                post_norm = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate, up = layer.mlp.gate_proj, layer.mlp.up_proj
                gate_up = torch.nn.Linear(
                    gate.in_features, gate.out_features + up.out_features, bias=False
                )
                gate_up.weight.copy_(
                    torch.cat((gate.weight * post_norm, up.weight * post_norm), dim=0)
                )
                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

                layer.self_attn.o_proj.weight.mul_(
                    layer.self_attn_layer_scale.scale.unsqueeze(1)
                )
                layer.mlp.down_proj.weight.mul_(
                    layer.mlp_layer_scale.scale.unsqueeze(1)
                )

            final_norm = self.decoder.pre_transformer.norm.weight.unsqueeze(0) * norm_factor
            self.decoder.pre_transformer.output_proj.weight.mul_(final_norm)
            del self.decoder.pre_transformer.norm

    def _rms_norm(self, hidden_states):
        return simplified_layer_norm(
            hidden_states, self.rms_norm_scale, self.rms_norm_eps
        )

    def rotate_half(self, hidden_states):
        hidden_states = hidden_states.view(
            1, -1, self.qk_heads, 2, self.head_dim_half
        ).flip(-2)
        return hidden_states.view(1, -1, self.qk_heads, self.head_dim)

    def forward(self, generated_codec):
        codes = generated_codec.reshape(1, -1, self.num_codebooks).transpose(1, 2)
        hidden_states = self.decoder.quantizer.decode(codes)
        hidden_states = self.decoder.pre_conv(hidden_states).transpose(1, 2)
        hidden_states = self.decoder.pre_transformer.input_proj(hidden_states)
        sequence_length = hidden_states.shape[1]
        rope_cos = self.rope_cos[:, :sequence_length].float()
        rope_sin = self.rope_sin[:, :sequence_length].float()
        attention_mask = self.attention_mask[
            :, :, :sequence_length, :sequence_length
        ].float()

        for layer in self.decoder.pre_transformer.layers:
            residual = hidden_states
            normalized = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(normalized).reshape(
                1, -1, self.qk_heads + self.num_heads, self.head_dim
            )
            query_key, value = torch.split(qkv, (self.qk_heads, self.num_heads), dim=-2)
            query_key = query_key * rope_cos + self.rotate_half(query_key) * rope_sin
            query, key = torch.split(query_key, (self.num_heads, self.num_heads), dim=-2)
            query = query.transpose(1, 2)
            key = key.permute(0, 2, 3, 1)
            value = value.transpose(1, 2)
            attention = torch.softmax(
                torch.matmul(query, key) + attention_mask, dim=-1
            )
            attention = torch.matmul(attention, value).transpose(1, 2).reshape(
                1, -1, layer.self_attn.o_proj.in_features
            )
            hidden_states = residual + layer.self_attn.o_proj(attention)
            residual = hidden_states
            normalized = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(normalized)
            gate, up = torch.split(
                gate_up, (layer.mlp.down_proj.in_features,) * 2, dim=-1
            )
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        hidden_states = self._rms_norm(hidden_states)
        generated_wav = self.decoder.pre_transformer.output_proj(hidden_states).transpose(1, 2)
        for blocks in self.decoder.upsample:
            for block in blocks:
                generated_wav = block(generated_wav)
        for block in self.decoder.decoder:
            generated_wav = block(generated_wav)

        generated_wav = generated_wav.clamp(min=-1.0, max=1.0)
        if self.output_resample_scale != 1.0:
            generated_wav = torch.nn.functional.interpolate(
                generated_wav,
                scale_factor=self.output_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        if self.output_dtype == "INT16":
            generated_wav = (generated_wav * 32767.0).clamp(
                min=-32768.0, max=32767.0
            ).to(torch.int16)
        elif self.output_dtype == "F32":
            generated_wav = generated_wav.float()
        else:
            generated_wav = generated_wav.half()
        return generated_wav, shape_dim_as_tensor(generated_wav, -1)


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


def _convert_tokenizer_decoder_blocks(tokenizer_model):
    for block_index, upsample_block in enumerate(tokenizer_model.decoder.upsample):
        for module_index, module in enumerate(upsample_block):
            if isinstance(module, Qwen3TTSTokenizerV2ConvNeXtBlockUnfused):
                tokenizer_model.decoder.upsample[block_index][module_index] = (
                    Qwen3TTSTokenizerV2ConvNeXtBlock.from_unfused(module)
                )
    for layer in tokenizer_model.decoder.quantizer.rvq_first.vq.layers:
        layer._codebook.precompute_embedding()
    for layer in tokenizer_model.decoder.quantizer.rvq_rest.vq.layers:
        layer._codebook.precompute_embedding()
    for module in tokenizer_model.decoder.decoder.modules():
        if isinstance(module, SnakeBeta):
            module.precompute()


def run_compact_strategy_export():
    validate_export_controls()
    print("Breeze TTS 2 compact strategy export start ...")
    with torch.inference_mode(), tempfile.TemporaryDirectory(
        dir=SCRIPT_DIR, prefix=".breezetts_export_"
    ) as staging_name:
        staging = Path(staging_name)

        model_config = BreezeConfig.from_pretrained(download_path)
        model_config.text_encoder_config.preferred_attn_implementation = "eager"
        model = BreezeForConditionalGeneration.from_pretrained(
            download_path,
            config=model_config,
            dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).float().eval()

        _install_tokenizer_patches(load_unfused_convnext=True)
        try:
            audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
                str(Path(download_path) / "audio_tokenizer"),
                device_map="cpu",
                dtype=torch.float32,
                attn_implementation="eager",
            )
        finally:
            _install_tokenizer_patches(load_unfused_convnext=False)
        tokenizer_model = audio_tokenizer.model.eval()
        _convert_tokenizer_decoder_blocks(tokenizer_model)

        depth_config = model.config.depth_decoder_config
        first_main_attention = model.backbone_model.layers[0].self_attn
        hidden_size = model.backbone_model.norm.weight.shape[0]
        num_layers_main = len(model.backbone_model.layers)
        num_kv_heads = first_main_attention.config.num_key_value_heads
        head_dim = first_main_attention.head_dim
        num_codebooks = model.config.num_codebooks
        codec_codebook_size = tokenizer_model.config.decoder_config.codebook_size
        codec_output_sample_rate = int(tokenizer_model.config.output_sample_rate)
        model_samples_per_codec_frame = int(tokenizer_model.config.decode_upsample_rate)
        samples_per_codec_frame = (
            model_samples_per_codec_frame * OUT_SAMPLE_RATE / codec_output_sample_rate
        )
        if not samples_per_codec_frame.is_integer():
            raise ValueError(
                "OUT_SAMPLE_RATE must map each codec frame to a whole number of "
                f"samples for streaming; got {samples_per_codec_frame:g}"
            )
        samples_per_codec_frame = int(samples_per_codec_frame)
        main_logits_size = model.lm_head.out_features
        depth_vocab_size = depth_config.vocab_size
        main_eos_token_id = model.backbone_eos_token_id
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32

        private_models = staging / ".models"
        model_paths = []

        def private_model_path(name):
            model_directory = private_models / Path(name).stem
            model_directory.mkdir(parents=True, exist_ok=True)
            model_path = model_directory / name
            model_paths.append(model_path)
            return model_path

        reference_path = private_model_path("BreezeTTS_ReferencePreprocess.onnx")
        target_path = private_model_path("BreezeTTS_TargetPreprocess.onnx")
        decoder_path = private_model_path("BreezeTTS_Decoder.onnx")
        decoder_stream_path = private_model_path("BreezeTTS_Decoder_Stream.onnx")
        metadata_path = private_model_path("BreezeTTS_Metadata.onnx")
        prefill_paths = {
            strategy: private_model_path(f"BreezeTTS_MainPrefill_{strategy}.onnx")
            for strategy in DECODE_STRATEGIES
        }
        main_decode_paths = {
            strategy: private_model_path(f"BreezeTTS_MainDecode_{strategy}.onnx")
            for strategy in DECODE_STRATEGIES
        }
        predictor_paths = {
            strategy: private_model_path(f"BreezeTTS_PredictorFrame_{strategy}.onnx")
            for strategy in DECODE_STRATEGIES
        }

        metadata = build_model_metadata(
            {
                "graph_layout": "strategy_prefill_decode_step",
                "model_type": "breeze_tts_2",
                "in_sample_rate": IN_SAMPLE_RATE,
                "out_sample_rate": OUT_SAMPLE_RATE,
                "in_audio_dtype": IN_AUDIO_DTYPE.upper(),
                "out_audio_dtype": OUT_AUDIO_DTYPE.upper(),
                "max_seq_len": MAX_SEQ_LEN,
                "num_codebooks": num_codebooks,
                "codec_codebook_size": codec_codebook_size,
                "main_logits_size": main_logits_size,
                "depth_vocab_size": depth_vocab_size,
                "main_eos_token_id": main_eos_token_id,
                "codebook_pad_token_id": model.config.codebook_pad_token_id,
                "codebook_eos_token_id": model.config.codebook_eos_token_id,
                "stop_token_ids": [main_eos_token_id],
                "samples_per_codec_frame": samples_per_codec_frame,
                "stream_window_frames": STREAM_WINDOW_FRAMES,
                "use_f16_kv": USE_F16_KV,
                "compute_in_f32": COMPUTE_IN_F32,
                "use_batch": USE_BATCH,
                "cfg_branch_order": (
                    "conditional,unconditional" if USE_BATCH else "conditional"
                ),
                "shared_initializer_model_file": SHARED_MODEL_NAME,
                "shared_initializer_data_file": SHARED_DATA_NAME,
                "model_file_name_metadata": "BreezeTTS_Metadata.onnx",
                "model_file_name_reference_preprocess": "BreezeTTS_ReferencePreprocess.onnx",
                "model_file_name_target_preprocess": "BreezeTTS_TargetPreprocess.onnx",
                "model_file_name_decoder": "BreezeTTS_Decoder.onnx",
                "model_file_name_decoder_stream": "BreezeTTS_Decoder_Stream.onnx",
                "model_file_name_shared_initializers": SHARED_MODEL_NAME,
                **{
                    f"model_file_name_main_prefill_{strategy}": prefill_paths[strategy].name
                    for strategy in DECODE_STRATEGIES
                },
                **{
                    f"model_file_name_decode_step_{strategy}": f"BreezeTTS_DecodeStep_{strategy}.onnx"
                    for strategy in DECODE_STRATEGIES
                },
            }
        )

        def export(module, args, destination, input_names, output_names, dynamic_axes=None):
            torch.onnx.export(
                module,
                tuple(args),
                str(destination),
                input_names=list(input_names),
                output_names=list(output_names),
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                dynamo=False,
                do_constant_folding=True,
            )
            removed_casts = eliminate_redundant_cast_chains(destination)
            if removed_casts:
                print(
                    f"[Cast rewrite] {Path(destination).name}: removed "
                    f"{removed_casts} redundant casts"
                )

        reference_source = reference_path.parent / ".reference_trace.onnx"
        reference_preprocess = TTS_REFERENCE_PREPROCESS(
            model, tokenizer_model, num_codebooks, MAX_SEQ_LEN
        ).eval()
        with torch.amp.autocast("cpu", dtype=torch.float16, enabled=USE_F16_ENCODER):
            export(
                reference_preprocess,
                (
                    torch.zeros(
                        1,
                        1,
                        IN_SAMPLE_RATE,
                        dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()],
                    ),
                    torch.zeros(1, 8, dtype=torch.int32),
                ),
                reference_source,
                ("prompt_audio", "ref_text_ids"),
                ("ref_code", "ref_text_embed"),
                {
                    "prompt_audio": {2: "audio_len"},
                    "ref_text_ids": {1: "ref_text_len"},
                    "ref_code": {1: "ref_frames"},
                    "ref_text_embed": {1: "ref_text_len"},
                },
            )
        write_onnx_metadata(reference_source, metadata)
        fuse_static_zero_prefix_convs(
            reference_source,
            reference_path,
            reference_preprocess.audio_encoder.static_mimi_fusion_count,
        )
        del reference_preprocess

        target_preprocess = TTS_TARGET_PREPROCESS(model).eval()
        export(
            target_preprocess,
            (
                torch.zeros(1, 8, dtype=torch.int32),
                torch.zeros(1, 3, num_codebooks, dtype=torch.int32),
                torch.zeros(1, 4, hidden_size, dtype=torch.float32),
            ),
            target_path,
            ("branch_text_ids", "ref_code", "ref_text_embed"),
            ("inputs_embeds", "attention_mask"),
            {
                "branch_text_ids": {1: "branch_text_len"},
                "ref_code": {1: "ref_frames"},
                "ref_text_embed": {1: "ref_text_len"},
                "inputs_embeds": {1: "prompt_len"},
                "attention_mask": {1: "prompt_len"},
            },
        )
        del target_preprocess

        main_core = TTS_MAIN_CORE(model, MAX_SEQ_LEN).eval()
        depth_core = TTS_DEPTH_CORE(model).eval()

        trace_batch_size = 2 if USE_BATCH else 1
        kv_inputs = [
            torch.zeros(trace_batch_size, num_kv_heads, 1, head_dim, 4, dtype=kv_dtype)
            for _ in range(num_layers_main)
        ] + [
            torch.zeros(trace_batch_size, num_kv_heads, 1, 4, head_dim, dtype=kv_dtype)
            for _ in range(num_layers_main)
        ]
        kv_input_names = [f"in_key_{index}" for index in range(num_layers_main)] + [
            f"in_value_{index}" for index in range(num_layers_main)
        ]
        kv_output_names = [f"out_key_{index}" for index in range(num_layers_main)] + [
            f"out_value_{index}" for index in range(num_layers_main)
        ]
        kv_axes = {}
        transformer_output_dimensions = {"last_hidden_state": {2: hidden_size}}
        for index in range(num_layers_main):
            kv_axes[f"in_key_{index}"] = {
                **dynamic_batch_axis(),
                4: "history_len",
            }
            kv_axes[f"out_key_{index}"] = {
                **dynamic_batch_axis(),
                4: "kv_seq_len",
            }
            kv_axes[f"in_value_{index}"] = {
                **dynamic_batch_axis(),
                3: "history_len",
            }
            kv_axes[f"out_value_{index}"] = {
                **dynamic_batch_axis(),
                3: "kv_seq_len",
            }
            transformer_output_dimensions[f"out_key_{index}"] = {
                1: num_kv_heads,
                2: 1,
                3: head_dim,
            }
            transformer_output_dimensions[f"out_value_{index}"] = {
                1: num_kv_heads,
                2: 1,
                4: head_dim,
            }

        prefill_embeds = torch.zeros(trace_batch_size, 5, hidden_size, dtype=torch.float32)
        prefill_mask = (
            torch.tensor(
                [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.int64
            )
            if USE_BATCH
            else torch.ones(1, 5, dtype=torch.int64)
        )
        guidance_weights = torch.tensor(
            [4.0, -3.0] if USE_BATCH else [1.0], dtype=torch.float32
        )
        frame_codec_ids = torch.zeros(1, num_codebooks, dtype=torch.int32)
        pad_lengths = torch.tensor([0, 1] if USE_BATCH else [0], dtype=torch.int64)
        history_len = torch.tensor([4], dtype=torch.int64)
        main_save_ids = torch.zeros(1, 3, dtype=torch.int32)
        generated_codec = torch.zeros(1, 0, dtype=torch.int32)
        main_token = torch.zeros(1, 1, dtype=torch.int32)
        last_hidden = torch.zeros(
            trace_batch_size, 1, hidden_size, dtype=torch.float32
        )
        penalty_value = torch.tensor([0.8], dtype=torch.float32)
        penalty_range = torch.tensor([10], dtype=torch.int64)
        temperature = torch.tensor([0.9], dtype=torch.float32)
        top_k = torch.tensor([50], dtype=torch.int64)
        top_p = torch.tensor([1.0], dtype=torch.float32)
        main_repetition = torch.tensor([1.1], dtype=torch.float32)
        depth_repetition = torch.tensor([1.0], dtype=torch.float32)

        for strategy in DECODE_STRATEGIES:
            prefill = TTS_MAIN_PREFILL_STRATEGY(main_core, strategy).eval()
            prefill_args = [prefill_embeds, prefill_mask, guidance_weights]
            prefill_inputs = ["inputs_embeds", "attention_mask", "guidance_weights"]
            if strategy == "sampling":
                prefill_args.extend((temperature, top_k, top_p))
                prefill_inputs.extend(("temperature", "top_k", "top_p"))
            export(
                prefill,
                prefill_args,
                prefill_paths[strategy],
                prefill_inputs,
                kv_output_names + ["last_hidden_state", "codec_token_main", "kv_seq_len"],
                {
                    **{name: axes for name, axes in kv_axes.items() if name.startswith("out_")},
                    "inputs_embeds": {
                        **dynamic_batch_axis(),
                        1: "prompt_len",
                    },
                    "attention_mask": {
                        **dynamic_batch_axis(),
                        1: "prompt_len",
                    },
                    "guidance_weights": dynamic_batch_axis(),
                    "last_hidden_state": dynamic_batch_axis(),
                },
            )
            set_onnx_static_output_dimensions(
                prefill_paths[strategy],
                static_batch_output_dimensions(
                    transformer_output_dimensions,
                    kv_output_names + ["last_hidden_state", "codec_token_main", "kv_seq_len"],
                ),
            )
            del prefill

            main_decode = TTS_MAIN_DECODE_STRATEGY(main_core, strategy).eval()
            main_args = kv_inputs + [
                frame_codec_ids,
                pad_lengths,
                history_len,
                guidance_weights,
            ]
            main_inputs = kv_input_names + [
                "frame_codec_ids",
                "pad_lengths",
                "history_len",
                "guidance_weights",
            ]
            main_outputs = kv_output_names + ["last_hidden_state", "codec_token_main"]
            main_axes = {
                **kv_axes,
                "pad_lengths": dynamic_batch_axis(),
                "guidance_weights": dynamic_batch_axis(),
                "last_hidden_state": dynamic_batch_axis(),
            }
            if strategy == "penalty_greedy":
                main_args.extend((main_save_ids, penalty_value, penalty_range))
                main_inputs.extend(("main_save_ids", "penalty_value", "penalty_range"))
            elif strategy == "sampling":
                main_args.extend(
                    (main_save_ids, temperature, top_k, top_p, main_repetition)
                )
                main_inputs.extend(
                    (
                        "main_save_ids",
                        "main_temperature",
                        "main_top_k",
                        "main_top_p",
                        "main_repetition_penalty",
                    )
                )
            if strategy != "greedy":
                main_outputs.append("main_save_ids_out")
                main_axes["main_save_ids"] = {1: "main_history_len"}
                main_axes["main_save_ids_out"] = {1: "main_history_len_out"}
            main_outputs.append("kv_seq_len")
            export(
                main_decode,
                main_args,
                main_decode_paths[strategy],
                main_inputs,
                main_outputs,
                main_axes,
            )
            set_onnx_static_output_dimensions(
                main_decode_paths[strategy],
                static_batch_output_dimensions(
                    transformer_output_dimensions, main_outputs
                ),
            )
            del main_decode

            predictor = TTS_PREDICTOR_FRAME_STRATEGY(depth_core, strategy).eval()
            predictor_args = [
                main_token,
                last_hidden,
                generated_codec,
                guidance_weights,
            ]
            predictor_inputs = [
                "codec_token_main_in",
                "last_hidden_state_main",
                "generated_codec_in",
                "guidance_weights",
            ]
            if strategy == "penalty_greedy":
                predictor_args.extend((penalty_value, penalty_range))
                predictor_inputs.extend(
                    ("predictor_penalty_value", "predictor_penalty_range")
                )
            elif strategy == "sampling":
                predictor_args.extend(
                    (temperature, top_k, top_p, depth_repetition)
                )
                predictor_inputs.extend(
                    (
                        "predictor_temperature",
                        "predictor_top_k",
                        "predictor_top_p",
                        "predictor_repetition_penalty",
                    )
                )
            export(
                predictor,
                predictor_args,
                predictor_paths[strategy],
                predictor_inputs,
                ("frame_codec_ids", "generated_codec"),
                {
                    "last_hidden_state_main": dynamic_batch_axis(),
                    "guidance_weights": dynamic_batch_axis(),
                    "generated_codec_in": {1: "generated_codec_len"},
                    "generated_codec": {1: "generated_codec_len_out"},
                },
            )
            set_onnx_static_output_dimensions(
                predictor_paths[strategy],
                static_batch_output_dimensions(
                    {}, ("frame_codec_ids", "generated_codec")
                ),
            )
            del predictor

        del depth_core, main_core, model
        gc.collect()

        decoder = TTS_DECODER(tokenizer_model, num_codebooks, MAX_SEQ_LEN).eval()
        decoder_source = decoder_path.parent / ".decoder_trace.onnx"
        export(
            decoder,
            (torch.zeros(1, num_codebooks * 10, dtype=torch.int32),),
            decoder_source,
            ("generated_codec",),
            ("generated_wav", "generated_len"),
            {
                "generated_codec": {1: "generated_codec_len"},
                "generated_wav": {2: "generated_wav_len"},
            },
        )
        write_onnx_metadata(decoder_source, metadata)
        fuse_static_zero_prefix_convs(
            decoder_source, decoder_path, decoder.static_conv_fusion_count
        )

        stream_source = decoder_stream_path.parent / ".decoder_stream_trace.onnx"
        export(
            decoder,
            (
                torch.zeros(
                    1,
                    num_codebooks * STREAM_WINDOW_FRAMES,
                    dtype=torch.int32,
                ),
            ),
            stream_source,
            ("generated_codec",),
            ("generated_wav", "generated_len"),
            None,
        )
        write_onnx_metadata(stream_source, metadata)
        fuse_static_zero_prefix_convs(
            stream_source,
            decoder_stream_path,
            decoder.static_conv_fusion_count,
        )
        del decoder, tokenizer_model, audio_tokenizer
        gc.collect()

        export(
            METADATA_CARRIER(),
            (torch.zeros(1, dtype=torch.int64),),
            metadata_path,
            ("metadata_marker",),
            ("metadata_marker_out",),
            None,
        )
        for model_path in model_paths:
            write_onnx_metadata(model_path, metadata)

        shared_stats = bundle_shared_initializers(
            staging, model_paths=model_paths, metadata=metadata
        )
        decode_steps = build_decode_step_graphs(staging, DECODE_STRATEGIES)
        replace_onnx_metadata(staging / "BreezeTTS_Metadata.onnx", metadata)

        if ONNX_FOLDER.exists():
            shutil.rmtree(ONNX_FOLDER)
        ONNX_FOLDER.mkdir(parents=True)
        for name in FINAL_MODEL_NAMES:
            shutil.move(str(staging / name), str(ONNX_FOLDER / name))
        tokenizer_file_count = copy_text_tokenizer(download_path, ONNX_FOLDER)

        print(
            f"[Shared weights] {shared_stats['initializer_references']} references -> "
            f"{shared_stats['unique_initializers']} unique tensors"
        )
        print(f"[DecodeStep] Built {len(decode_steps)} strategy graphs")
        print(f"[Tokenizer] Copied {tokenizer_file_count} text tokenizer files")
    print(f"Breeze TTS 2 export complete: {ONNX_FOLDER}")


if __name__ == "__main__" and DO_EXPORT:
    run_compact_strategy_export()
    print("\nStart running the Breeze TTS 2 demo via Inference_BreezeTTS_ONNX.py ...")
    raise SystemExit(subprocess.call([
        sys.executable,
        str(SCRIPT_DIR / "Inference_BreezeTTS_ONNX.py"),
        "--onnx-folder",
        str(ONNX_FOLDER),
    ]))