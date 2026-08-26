"""Export every ZipVoice inference stage to ONNX Runtime graphs."""

from __future__ import annotations

import gc
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch import Tensor, nn
from torch.onnx.operators import shape_as_tensor

from STFT_Process import STFT_Process


SCRIPT_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIRECTORY = Path.home() / "Downloads"
OFFICIAL_SOURCE = DOWNLOADS_DIRECTORY / "ZipVoice-master"
MODEL_ROOT = DOWNLOADS_DIRECTORY / "ZipVoice"
VOCOS_DIRECTORY = DOWNLOADS_DIRECTORY / "vocos-mel-24khz"

# ============================== USER CONFIG ==============================

# Available models:
# - Single speaker: zipvoice, zipvoice_libritts, zipvoice_distill, zipvoice_distill_libritts
# - Dialogue:       zipvoice_dialog, zipvoice_dialog_opendialog, zipvoice_dialog_stereo
MODEL_NAME = "zipvoice_distill"
CHECKPOINT_NAME: str | None = None
IN_SAMPLE_RATE = 24000
OUT_SAMPLE_RATE = 24000
OPSET = 20
DYNAMIC_AXES = True

# ============================ FIXED PIPELINE =============================

MODEL_SAMPLE_RATE = 24000
N_FFT = 1024
WINDOW_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 100
TARGET_RMS = 0.1
FEATURE_SCALE = 0.1
DEFAULT_T_SHIFT = 0.5
CROSSFADE_SECONDS = 0.1
PROMPT_TRAILING_SILENCE_MS = 200


@dataclass(frozen=True)
class Variant:
    package_stem: str
    model_class: str
    tokenizer_type: str
    output_channels: int
    default_num_step: int
    default_guidance_scale: float
    chunk_target_seconds: float
    distilled: bool = False
    dialogue: bool = False


VARIANTS = {
    "zipvoice": Variant(
        "ZipVoice", "ZipVoice", "emilia", 1, 16, 1.0, 25.0
    ),
    "zipvoice_libritts": Variant(
        "ZipVoice_LibriTTS", "ZipVoice", "libritts", 1, 16, 1.0, 25.0
    ),
    "zipvoice_distill": Variant(
        "ZipVoice_Distill", "ZipVoiceDistill", "emilia", 1, 8, 3.0, 25.0, True
    ),
    "zipvoice_distill_libritts": Variant(
        "ZipVoice_Distill_LibriTTS",
        "ZipVoiceDistill",
        "libritts",
        1,
        8,
        3.0,
        25.0,
        True,
    ),
    "zipvoice_dialog": Variant(
        "ZipVoice_Dialog", "ZipVoiceDialog", "dialog", 1, 16, 1.5, 40.0, False, True
    ),
    "zipvoice_dialog_opendialog": Variant(
        "ZipVoice_Dialog_OpenDialog",
        "ZipVoiceDialog",
        "dialog",
        1,
        16,
        1.5,
        40.0,
        False,
        True,
    ),
    "zipvoice_dialog_stereo": Variant(
        "ZipVoice_Dialog_Stereo",
        "ZipVoiceDialogStereo",
        "dialog",
        2,
        16,
        1.5,
        40.0,
        False,
        True,
    ),
}

VARIANT = VARIANTS[MODEL_NAME]
MODEL_DIRECTORY = MODEL_ROOT / MODEL_NAME
OUTPUT_FOLDER = SCRIPT_DIR / f"{VARIANT.package_stem}_ONNX"
COMPONENT_FOLDER = SCRIPT_DIR / f"{VARIANT.package_stem}_ONNX_Raw"


class VocosAdaLayerNorm(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.embedding_dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings, embedding_dim)
        self.shift = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.ones_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)

    def forward(self, inputs: Tensor, condition: Tensor) -> Tensor:
        scale = F.embedding(
            condition,
            self.scale.weight,
            self.scale.padding_idx,
            self.scale.max_norm,
            self.scale.norm_type,
            self.scale.scale_grad_by_freq,
            self.scale.sparse,
        )
        shift = F.embedding(
            condition,
            self.shift.weight,
            self.shift.padding_idx,
            self.shift.max_norm,
            self.shift.norm_type,
            self.shift.scale_grad_by_freq,
            self.shift.sparse,
        )
        normalized = F.layer_norm(inputs, (self.embedding_dim,), eps=self.eps)
        return normalized * scale + shift


class VocosConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        adanorm_num_embeddings: int | None = None,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.adaptive_norm = adanorm_num_embeddings is not None
        self.norm = (
            VocosAdaLayerNorm(adanorm_num_embeddings, dim)
            if adanorm_num_embeddings is not None
            else nn.LayerNorm(dim, eps=1e-6)
        )
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(
        self,
        inputs: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        residual = inputs
        hidden = _conv1d(inputs, self.dwconv).transpose(1, 2)
        hidden = _vocos_norm(hidden, self.norm, condition)
        hidden = _linear(hidden, self.pwconv1)
        hidden = F.gelu(hidden, approximate=self.act.approximate)
        hidden = _linear(hidden, self.pwconv2)
        if not getattr(self, "_onnx_gamma_folded", False):
            hidden = self.gamma * hidden
        return residual + hidden.transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: float | None = None,
        adanorm_num_embeddings: int | None = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, 7, padding=3)
        self.adaptive_norm = adanorm_num_embeddings is not None
        self.norm = (
            VocosAdaLayerNorm(adanorm_num_embeddings, dim)
            if adanorm_num_embeddings is not None
            else nn.LayerNorm(dim, eps=1e-6)
        )
        layer_scale = layer_scale_init_value or 1.0 / num_layers
        self.convnext = nn.ModuleList(
            VocosConvNeXtBlock(
                dim,
                intermediate_dim,
                layer_scale,
                adanorm_num_embeddings,
            )
            for _ in range(num_layers)
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = _conv1d(inputs, self.embed).transpose(1, 2)
        hidden = _vocos_norm(hidden, self.norm).transpose(1, 2)
        for block in self.convnext:
            residual = hidden
            block_hidden = _conv1d(hidden, block.dwconv).transpose(1, 2)
            block_hidden = _vocos_norm(block_hidden, block.norm)
            block_hidden = _linear(block_hidden, block.pwconv1)
            block_hidden = F.gelu(
                block_hidden,
                approximate=block.act.approximate,
            )
            block_hidden = _linear(block_hidden, block.pwconv2)
            if not getattr(block, "_onnx_gamma_folded", False):
                block_hidden = block.gamma * block_hidden
            hidden = residual + block_hidden.transpose(1, 2)
        return _vocos_norm(hidden.transpose(1, 2), self.final_layer_norm)


class UnusedISTFTPlaceholder(nn.Module):
    def __init__(self, n_fft: int) -> None:
        super().__init__()
        self.register_buffer("window", torch.empty(n_fft, dtype=torch.float32))


class VocosISTFTHead(nn.Module):
    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "center",
    ) -> None:
        super().__init__()
        del hop_length, padding
        self.num_bins = n_fft // 2 + 1
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = UnusedISTFTPlaceholder(n_fft)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        spectral = _linear(inputs, self.out).transpose(1, 2)
        magnitude, phase = spectral.split(self.num_bins, dim=1)
        return torch.exp(magnitude).clamp(max=1e2), phase


class VocosSpectralDecoder(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return _vocos_forward(self, features)


def _linear(inputs: Tensor, layer: Any) -> Tensor:
    return F.linear(inputs, layer.weight, layer.bias)


def _conv1d(inputs: Tensor, layer: nn.Conv1d) -> Tensor:
    return F.conv1d(
        inputs,
        layer.weight,
        layer.bias,
        layer.stride,
        layer.padding,
        layer.dilation,
        layer.groups,
    )


def _linear_as_conv1d(inputs: Tensor, layer: nn.Linear) -> Tensor:
    return F.conv1d(inputs, layer.weight.unsqueeze(-1), layer.bias)


def _channel_layer_norm(inputs: Tensor, norm: nn.LayerNorm) -> Tensor:
    centered = inputs - inputs.mean(dim=1, keepdim=True)
    normalized = centered * torch.rsqrt(
        centered.square().mean(dim=1, keepdim=True) + norm.eps
    )
    if not getattr(norm, "_onnx_affine_folded", False):
        if norm.weight is not None:
            normalized = normalized * norm.weight.reshape(1, -1, 1)
        if norm.bias is not None:
            normalized = normalized + norm.bias.reshape(1, -1, 1)
    return normalized


def _vocos_norm(
    inputs: Tensor,
    norm: nn.Module,
    condition: Tensor | None = None,
) -> Tensor:
    if isinstance(norm, VocosAdaLayerNorm):
        if condition is None:
            raise ValueError("Adaptive Vocos normalization requires a condition")
        scale = F.embedding(
            condition,
            norm.scale.weight,
            norm.scale.padding_idx,
            norm.scale.max_norm,
            norm.scale.norm_type,
            norm.scale.scale_grad_by_freq,
            norm.scale.sparse,
        )
        shift = F.embedding(
            condition,
            norm.shift.weight,
            norm.shift.padding_idx,
            norm.shift.max_norm,
            norm.shift.norm_type,
            norm.shift.scale_grad_by_freq,
            norm.shift.sparse,
        )
        normalized = F.layer_norm(inputs, (norm.embedding_dim,), eps=norm.eps)
        return normalized * scale + shift

    if not isinstance(norm, nn.LayerNorm):
        raise TypeError(f"Unsupported Vocos normalization: {type(norm).__name__}")
    affine_folded = getattr(norm, "_onnx_affine_folded", False)
    return F.layer_norm(
        inputs,
        norm.normalized_shape,
        None if affine_folded else norm.weight,
        None if affine_folded else norm.bias,
        norm.eps,
    )


def _vocos_forward(
    decoder: VocosSpectralDecoder,
    features: Tensor,
) -> tuple[Tensor, Tensor]:
    backbone = decoder.backbone
    if not isinstance(backbone.norm, nn.LayerNorm):
        raise TypeError("Channel-first Vocos export requires LayerNorm")
    hidden = _conv1d(features, backbone.embed)
    hidden = _channel_layer_norm(hidden, backbone.norm)

    for block in backbone.convnext:
        if not isinstance(block.norm, nn.LayerNorm):
            raise TypeError("Channel-first Vocos export requires LayerNorm")
        residual = hidden
        block_hidden = _conv1d(hidden, block.dwconv)
        block_hidden = _channel_layer_norm(block_hidden, block.norm)
        block_hidden = _linear_as_conv1d(block_hidden, block.pwconv1)
        block_hidden = F.gelu(block_hidden, approximate=block.act.approximate)
        block_hidden = _linear_as_conv1d(block_hidden, block.pwconv2)
        if not getattr(block, "_onnx_gamma_folded", False):
            block_hidden = block.gamma.reshape(1, -1, 1) * block_hidden
        hidden = residual + block_hidden

    hidden = _channel_layer_norm(hidden, backbone.final_layer_norm)
    spectral = _linear_as_conv1d(hidden, decoder.head.out)
    magnitude, phase = spectral.split(decoder.head.num_bins, dim=1)
    return torch.exp(magnitude).clamp(max=1e2), phase


def _swoosh_forward(inputs: Tensor, offset: float, constant: float) -> Tensor:
    inputs = inputs.to(torch.float32)
    shifted = inputs - offset
    return F.softplus(shifted) - 0.08 * inputs - constant


def _compact_rel_positional_encoding(inputs: Tensor, module: Any) -> Tensor:
    sequence_length = shape_as_tensor(inputs)[0]
    positions = torch.arange(
        1 - sequence_length,
        sequence_length,
        dtype=torch.float32,
        device=inputs.device,
    ).unsqueeze(1)
    frequencies = 1 + torch.arange(
        module.embed_dim // 2,
        dtype=torch.float32,
        device=inputs.device,
    )
    compression_length = module.embed_dim**0.5
    compressed = (
        compression_length
        * positions.sign()
        * (
            (positions.abs() + compression_length).log()
            - math.log(compression_length)
        )
    )
    length_scale = module.length_factor * module.embed_dim / (2.0 * math.pi)
    angles = compressed / length_scale
    angles = angles.atan() * frequencies
    positional = torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(1)
    positional = torch.cat(
        (positional[:, :-1], torch.ones_like(positional[:, -1:])),
        dim=1,
    )
    return positional.to(inputs.dtype).unsqueeze(0)


def _relative_attention_weights(
    module: Any,
    inputs: Tensor,
    projected_position: Tensor,
    relative_indices: Tensor,
    key_padding_mask: Tensor | None,
    attention_mask: Tensor | None,
) -> Tensor:
    projected = _linear(inputs, module.in_proj)
    sequence_length, batch_size, _ = projected.shape
    projected = projected.reshape(
        sequence_length,
        batch_size,
        module.num_heads,
        -1,
    ).permute(2, 1, 0, 3)
    query, key, position_query = projected.split(
        (module.query_head_dim, module.query_head_dim, module.pos_head_dim),
        dim=-1,
    )
    key = key.transpose(2, 3)
    attention_scores = torch.matmul(query, key)

    relative_length = 2 * sequence_length - 1
    position_scores = torch.matmul(position_query, projected_position)
    position_scores = torch.gather(
        position_scores.reshape(-1, relative_length),
        dim=1,
        index=relative_indices,
    ).reshape(
        module.num_heads,
        batch_size,
        sequence_length,
        sequence_length,
    )
    attention_scores = attention_scores + position_scores

    if attention_mask is not None:
        attention_scores = attention_scores.masked_fill(attention_mask, -1000.0)
    if key_padding_mask is not None:
        attention_scores = attention_scores.masked_fill(
            key_padding_mask.unsqueeze(1),
            -1000.0,
        )
    return F.softmax(attention_scores, dim=-1)


def _self_attention(
    module: Any,
    inputs: Tensor,
    weights: Tensor,
    num_heads: int,
) -> Tensor:
    sequence_length, batch_size, _ = inputs.shape
    values = _linear(inputs, module.in_proj)
    values = values.reshape(sequence_length, batch_size, num_heads, -1)
    values = values.permute(2, 1, 0, 3)
    values = torch.matmul(weights, values)
    values = values.permute(2, 1, 0, 3).contiguous()
    values = values.view(sequence_length, batch_size, -1)
    return _linear(values, module.out_proj)


def _feed_forward(module: Any, inputs: Tensor) -> Tensor:
    hidden = _linear(inputs, module.in_proj)
    hidden = _swoosh_forward(hidden, offset=4.0, constant=0.035)
    return _linear(hidden, module.out_proj)


def _nonlinear_attention(
    module: Any,
    inputs: Tensor,
    weights: Tensor,
    num_heads: int,
) -> Tensor:
    hidden = _linear(inputs, module.in_proj)
    sequence_length, batch_size, _ = hidden.shape
    scale, values, gate = hidden.chunk(3, dim=2)
    values = values * torch.tanh(scale)
    values = values.reshape(sequence_length, batch_size, num_heads, -1)
    values = values.permute(2, 1, 0, 3)
    values = torch.matmul(weights, values)
    values = values.permute(2, 1, 0, 3).reshape(
        sequence_length,
        batch_size,
        -1,
    )
    return _linear(values * gate, module.out_proj)


def _convolution(
    module: Any,
    inputs: Tensor,
    padding_mask: Tensor | None,
) -> Tensor:
    hidden, gate = _linear(inputs, module.in_proj).chunk(2, dim=2)
    hidden = hidden * torch.sigmoid(gate)
    hidden = hidden.permute(1, 2, 0)
    if padding_mask is not None:
        hidden = hidden.masked_fill(padding_mask.unsqueeze(1), 0.0)
    hidden = _conv1d(hidden, module.depthwise_conv).permute(2, 0, 1)
    hidden = _swoosh_forward(hidden, offset=1.0, constant=0.313261687)
    return _linear(hidden, module.out_proj)


def _bias_norm(module: Any, inputs: Tensor) -> Tensor:
    channel_dim = module.channel_dim
    if channel_dim < 0:
        channel_dim += inputs.ndim
    bias = module.bias
    for _ in range(channel_dim + 1, inputs.ndim):
        bias = bias.unsqueeze(-1)
    output_scale = getattr(module, "_onnx_output_scale", None)
    if output_scale is None:
        output_scale = module.log_scale.exp()
    scale = torch.mean(
        (inputs - bias).square(),
        dim=channel_dim,
        keepdim=True,
    ).pow(-0.5) * output_scale
    return inputs * scale


def _bypass(module: Any, original: Tensor, inputs: Tensor) -> Tensor:
    residual_scale = getattr(module, "_onnx_residual_scale", None)
    if residual_scale is not None:
        return inputs + original * residual_scale
    return original + (inputs - original) * module.bypass_scale


def _zipformer_encoder_layer(
    layer: Any,
    inputs: Tensor,
    projected_position: Tensor,
    relative_indices: Tensor,
    time_embedding: Tensor | None,
    attention_mask: Tensor | None,
    padding_mask: Tensor | None,
) -> Tensor:
    original = inputs
    attention_weights = _relative_attention_weights(
        layer.self_attn_weights,
        inputs,
        projected_position,
        relative_indices,
        padding_mask,
        attention_mask,
    )
    if time_embedding is not None:
        inputs = inputs + time_embedding
    inputs = inputs + _feed_forward(layer.feed_forward1, inputs)
    inputs = inputs + _nonlinear_attention(
        layer.nonlin_attention,
        inputs,
        attention_weights[:1],
        layer.self_attn_weights.num_heads,
    )
    inputs = inputs + _self_attention(
        layer.self_attn1,
        inputs,
        attention_weights,
        layer.self_attn_weights.num_heads,
    )
    if layer.use_conv:
        if time_embedding is not None:
            inputs = inputs + time_embedding
        inputs = inputs + _convolution(layer.conv_module1, inputs, padding_mask)
    inputs = inputs + _feed_forward(layer.feed_forward2, inputs)
    inputs = _bypass(layer.bypass_mid, original, inputs)
    inputs = inputs + _self_attention(
        layer.self_attn2,
        inputs,
        attention_weights,
        layer.self_attn_weights.num_heads,
    )
    if layer.use_conv:
        if time_embedding is not None:
            inputs = inputs + time_embedding
        inputs = inputs + _convolution(layer.conv_module2, inputs, padding_mask)
    inputs = inputs + _feed_forward(layer.feed_forward3, inputs)
    inputs = _bias_norm(layer.norm, inputs)
    return _bypass(layer.bypass, original, inputs)


def _zipformer_encoder(
    encoder: Any,
    inputs: Tensor,
    time_embedding: Tensor | None,
    attention_mask: Tensor | None,
    padding_mask: Tensor | None,
    projected_positions: Tensor | None = None,
    relative_indices: Tensor | None = None,
) -> Tensor:
    positional = None
    if projected_positions is None:
        positional = _compact_rel_positional_encoding(inputs, encoder.encoder_pos)
    sequence_length, batch_size, _ = inputs.shape
    if relative_indices is None:
        num_heads = encoder.layers[0].self_attn_weights.num_heads
        rows = torch.arange(
            start=sequence_length - 1,
            end=-1,
            step=-1,
            dtype=torch.int32,
            device=inputs.device,
        )
        columns = torch.arange(
            sequence_length,
            dtype=torch.int32,
            device=inputs.device,
        )
        relative_indices = (
            rows.repeat(batch_size * num_heads).unsqueeze(-1) + columns
        )
    if encoder.time_emb is not None and time_embedding is None:
        raise ValueError("Zipformer encoder requires a time embedding")

    hidden = inputs
    relative_length = 2 * sequence_length - 1
    for layer_index, layer in enumerate(encoder.layers):
        if projected_positions is None:
            projected_position = _linear(
                positional,
                layer.self_attn_weights.linear_pos,
            ).reshape(
                -1,
                relative_length,
                layer.self_attn_weights.num_heads,
                layer.self_attn_weights.pos_head_dim,
            ).permute(2, 0, 3, 1)
        else:
            projected_position = projected_positions[layer_index]
        hidden = _zipformer_encoder_layer(
            layer,
            hidden,
            projected_position,
            relative_indices,
            time_embedding,
            attention_mask,
            padding_mask,
        )
    return hidden


def _downsample(inputs: Tensor, module: Any) -> Tensor:
    sequence_length, batch_size, channels = inputs.shape
    factor = module.downsample
    downsampled_length = (sequence_length + factor - 1) // factor
    padding = downsampled_length * factor - sequence_length
    trailing = inputs[-1:].expand(padding, batch_size, channels)
    hidden = torch.cat((inputs, trailing), dim=0)
    hidden = hidden.reshape(downsampled_length, factor, batch_size, channels)
    weights = getattr(module, "_onnx_weights", None)
    if weights is None:
        weights = F.softmax(module.bias, dim=0)
    weights = weights.reshape(1, factor, 1, 1)
    return (hidden * weights).sum(dim=1)


def _upsample(inputs: Tensor, factor: int) -> Tensor:
    sequence_length, batch_size, channels = inputs.shape
    hidden = inputs.unsqueeze(1).expand(
        sequence_length,
        factor,
        batch_size,
        channels,
    )
    return hidden.reshape(sequence_length * factor, batch_size, channels)


def _zipformer_stack(
    stack: Any,
    inputs: Tensor,
    time_embedding: Tensor | None,
    attention_mask: Tensor | None,
    padding_mask: Tensor | None,
    projected_positions: Tensor | None = None,
    relative_indices: Tensor | None = None,
) -> Tensor:
    if not hasattr(stack, "downsample_factor"):
        return _zipformer_encoder(
            stack,
            inputs,
            time_embedding,
            attention_mask,
            padding_mask,
            projected_positions,
            relative_indices,
        )

    original = inputs
    factor = stack.downsample_factor
    hidden = _downsample(inputs, stack.downsample)
    if time_embedding is not None and time_embedding.dim() == 3:
        time_embedding = time_embedding[::factor]
    if attention_mask is not None:
        attention_mask = attention_mask[::factor, ::factor]
    if padding_mask is not None:
        padding_mask = padding_mask[..., ::factor]
    hidden = _zipformer_encoder(
        stack.encoder,
        hidden,
        time_embedding,
        attention_mask,
        padding_mask,
        projected_positions,
        relative_indices,
    )
    hidden = _upsample(hidden, factor)[: original.shape[0]]
    return _bypass(stack.out_combiner, original, hidden)


def _timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    frequencies: Tensor,
) -> Tensor:
    if timesteps.dim() == 2:
        timesteps = timesteps.transpose(0, 1)
    angles = timesteps[..., None].to(torch.float32) * frequencies[None]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if embedding_dim % 2:
        embedding = torch.cat(
            (embedding, torch.zeros_like(embedding[..., :1])),
            dim=-1,
        )
    return embedding


def _time_embedding(module: Any, timesteps: Tensor) -> Tensor:
    hidden = _linear(timesteps, module[0])
    shifted = hidden - 1.0
    hidden = F.softplus(shifted) - 0.08 * hidden - 0.313261687
    return _linear(hidden, module[2])


def _flow_projection_index(module: Any, input_width: int) -> int:
    if not isinstance(module.in_proj, nn.ModuleList):
        if input_width != module.in_proj.in_features:
            raise ValueError(
                f"Flow input width {input_width} != {module.in_proj.in_features}"
            )
        return -1
    if input_width not in module.in_dim:
        raise ValueError(f"Flow input width {input_width} not in {module.in_dim}")
    return 0 if input_width == module.in_dim[0] else 1


def _flow_input_projection(module: Any, projection_index: int) -> nn.Linear:
    if projection_index < 0:
        return module.in_proj
    return module.in_proj[projection_index]


def _flow_zipformer_hidden(
    module: Any,
    hidden: Tensor,
    time_embeddings: Tensor,
    relative_geometry: dict[int, tuple[Tensor, Tensor]],
    padding_mask: Tensor | None,
    projection_index: int,
) -> Tensor:
    layer_offsets = {factor: 0 for factor in relative_geometry}
    for stack_index, stack in enumerate(module.encoders):
        factor = getattr(stack, "downsample_factor", 1)
        projected_positions, relative_indices = relative_geometry[factor]
        encoder = stack.encoder if factor != 1 else stack
        layer_offset = layer_offsets[factor]
        layer_count = len(encoder.layers)
        stack_projected_positions = projected_positions[
            layer_offset : layer_offset + layer_count
        ]
        layer_offsets[factor] += layer_count
        hidden = _zipformer_stack(
            stack,
            hidden,
            time_embeddings[stack_index],
            attention_mask=None,
            padding_mask=padding_mask,
            projected_positions=stack_projected_positions,
            relative_indices=relative_indices,
        )

    if projection_index >= 0:
        hidden = _linear(hidden, module.out_proj[projection_index])
    else:
        hidden = _linear(hidden, module.out_proj)
    return hidden.permute(1, 0, 2)


def _text_zipformer_forward(
    module: Any,
    inputs: Tensor,
    padding_mask: Tensor,
) -> Tensor:
    hidden = _linear(inputs.permute(1, 0, 2), module.in_proj)
    for stack in module.encoders:
        hidden = _zipformer_stack(
            stack,
            hidden,
            time_embedding=None,
            attention_mask=None,
            padding_mask=padding_mask,
        )
    return _linear(hidden, module.out_proj).permute(1, 0, 2)


class ZipVoicePreprocess(nn.Module):
    def __init__(self, output_channels: int) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.input_resample_scale = float(MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
        self.spectral = STFT_Process(
            model_type="stft_B",
            n_fft=N_FFT,
            win_length=WINDOW_LENGTH,
            hop_len=HOP_LENGTH,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
        )
        mel_filter = torchaudio.functional.melscale_fbanks(
            n_freqs=N_FFT // 2 + 1,
            f_min=0.0,
            f_max=MODEL_SAMPLE_RATE / 2,
            n_mels=N_MELS,
            sample_rate=MODEL_SAMPLE_RATE,
            norm=None,
            mel_scale="htk",
        ).transpose(0, 1)
        self.register_buffer("mel_filter", mel_filter.unsqueeze(0))

    def forward(self, audio: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.input_resample_scale != 1.0:
            audio = F.interpolate(
                audio,
                scale_factor=self.input_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        prompt_rms = torch.sqrt(torch.mean(audio.square()))
        rms_scale = torch.clamp(prompt_rms / TARGET_RMS, max=1.0).reshape(1)
        audio_gain = torch.clamp(TARGET_RMS / prompt_rms, min=1.0)
        audio = audio * audio_gain
        if self.output_channels == 1:
            audio = audio.mean(dim=1, keepdim=True)

        sample_count = shape_as_tensor(audio)[-1]
        channel_audio = audio.reshape(-1, 1, audio.shape[-1])
        real, imaginary = self.spectral(channel_audio)
        magnitude = torch.sqrt(real.square() + imaginary.square())
        mel = torch.matmul(self.mel_filter, magnitude).clamp(min=1e-7).log()
        features = mel.reshape(1, self.output_channels * N_MELS, mel.shape[-1])
        features = features.transpose(1, 2)

        expected_frames = torch.div(
            sample_count + HOP_LENGTH // 2,
            HOP_LENGTH,
            rounding_mode="floor",
        )
        features = torch.cat((features, features[:, -1:]), dim=1)
        features = features[:, :expected_frames]
        features = features * FEATURE_SCALE
        return features, expected_frames, rms_scale


class ZipVoiceTextEncoder(nn.Module):
    def __init__(self, model: nn.Module, variant: Variant) -> None:
        super().__init__()
        self.embed = model.embed
        self.text_encoder = model.text_encoder
        self.pad_id = model.pad_id
        self.output_channels = variant.output_channels
        self.dialogue = variant.dialogue
        if self.dialogue:
            self.speaker_embedding = model.spk_embed
            self.speaker_a_id = model.spk_a_id
            self.speaker_b_id = model.spk_b_id

    def _encode(self, token_ids: Tensor, padding_mask: Tensor) -> Tensor:
        embedded = F.embedding(
            token_ids,
            self.embed.weight,
            self.embed.padding_idx,
            self.embed.max_norm,
            self.embed.norm_type,
            self.embed.scale_grad_by_freq,
            self.embed.sparse,
        )
        encoded = _text_zipformer_forward(
            self.text_encoder,
            embedded,
            padding_mask,
        )
        if not self.dialogue:
            return encoded

        turn_markers = (
            (token_ids == self.speaker_a_id) | (token_ids == self.speaker_b_id)
        ).to(torch.int64)
        speaker_indices = turn_markers.cumsum(dim=1).remainder(2)
        speaker_addition = F.embedding(
            speaker_indices,
            self.speaker_embedding.weight,
            self.speaker_embedding.padding_idx,
            self.speaker_embedding.max_norm,
            self.speaker_embedding.norm_type,
            self.speaker_embedding.scale_grad_by_freq,
            self.speaker_embedding.sparse,
        )
        valid_tokens = (token_ids != self.pad_id).unsqueeze(-1)
        return encoded + torch.where(
            valid_tokens,
            speaker_addition.to(encoded.dtype),
            torch.zeros_like(encoded),
        )

    def forward(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        target_token_count = shape_as_tensor(tokens)[1]
        prompt_token_count = shape_as_tensor(prompt_tokens)[1]
        combined_token_count = target_token_count + prompt_token_count

        combined_tokens = torch.cat((prompt_tokens, tokens), dim=1)
        combined_tokens = F.pad(combined_tokens, (0, 1), value=self.pad_id)
        padding_mask = (
            torch.arange(
                combined_token_count + 1,
                device=combined_tokens.device,
            )
            == combined_token_count
        ).unsqueeze(0)
        encoded = self._encode(combined_tokens, padding_mask)

        target_frames = torch.ceil(
            prompt_features_len.to(torch.float32)
            / prompt_token_count.to(torch.float32)
            * target_token_count.to(torch.float32)
            / speed
        ).to(torch.int64)
        total_frames = (prompt_features_len + target_frames).reshape(())
        token_duration = torch.div(
            total_frames,
            combined_token_count,
            rounding_mode="floor",
        ).reshape(())

        frame_positions = torch.arange(
            total_frames,
            device=encoded.device,
        )
        repeated_frames = token_duration * combined_token_count
        safe_duration = torch.clamp(token_duration, min=1)
        token_indices = torch.where(
            frame_positions < repeated_frames,
            torch.div(
                frame_positions,
                safe_duration,
                rounding_mode="floor",
            ),
            combined_token_count,
        )
        text_condition = torch.index_select(
            encoded,
            dim=1,
            index=token_indices,
        )

        empty_condition = torch.zeros_like(prompt_features[:, :1]).expand(
            -1,
            target_frames,
            -1,
        )
        speech_condition = torch.cat(
            (prompt_features, empty_condition),
            dim=1,
        )
        initial_noise = torch.randn_like(speech_condition)
        return initial_noise, text_condition, speech_condition


class ZipVoiceFlowCondition(nn.Module):
    def __init__(self, model: nn.Module, variant: Variant) -> None:
        super().__init__()
        self.distilled = variant.distilled
        self.state_width = N_MELS * variant.output_channels
        input_width = self.state_width * 2 + N_MELS
        projection_index = _flow_projection_index(model.fm_decoder, input_width)
        projection = _flow_input_projection(model.fm_decoder, projection_index)
        condition_weight = projection.weight[:, self.state_width :].detach().clone()
        if self.distilled:
            self.register_buffer("condition_weight", condition_weight)
            self.register_buffer(
                "condition_bias",
                projection.bias.detach().clone(),
            )
        else:
            self.register_buffer("text_weight", condition_weight[:, :N_MELS])
            self.register_buffer("speech_weight", condition_weight[:, N_MELS:])

    @staticmethod
    def _project(
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        batch_size, frame_count, _ = inputs.shape
        projected = F.linear(
            inputs.reshape(-1, weight.shape[1]),
            weight,
            bias,
        )
        return projected.reshape(batch_size, frame_count, weight.shape[0])

    def forward(
        self,
        text_condition: Tensor,
        speech_condition: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.distilled:
            condition = torch.cat((text_condition, speech_condition), dim=2)
            return self._project(
                condition,
                self.condition_weight,
                self.condition_bias,
            )
        return (
            self._project(text_condition, self.text_weight),
            self._project(speech_condition, self.speech_weight),
        )


class ZipVoiceFlowGeometry(nn.Module):
    def __init__(self, model: nn.Module, variant: Variant) -> None:
        super().__init__()
        position_specs: set[tuple[int, float, int, int]] = set()
        factors: set[int] = set()
        factor_projections: dict[int, list[nn.Linear]] = {}
        for stack in model.fm_decoder.encoders:
            factor = getattr(stack, "downsample_factor", 1)
            encoder = stack.encoder if factor != 1 else stack
            factors.add(factor)
            position_specs.add(
                (
                    encoder.encoder_pos.embed_dim,
                    encoder.encoder_pos.length_factor,
                    encoder.layers[0].self_attn_weights.num_heads,
                    encoder.layers[0].self_attn_weights.pos_head_dim,
                )
            )
            factor_projections.setdefault(factor, []).extend(
                layer.self_attn_weights.linear_pos for layer in encoder.layers
            )
        if factors != {1, 2, 4}:
            raise ValueError(f"Unsupported flow downsampling factors: {sorted(factors)}")
        if len(position_specs) != 1:
            raise ValueError("Flow stacks must share one positional configuration")

        self.factors = (1, 2, 4)
        (
            self.position_dim,
            length_factor,
            self.num_heads,
            self.pos_head_dim,
        ) = position_specs.pop()
        self.batch_multiplier = 1 if variant.distilled else 2
        self.factor_layer_counts = {
            factor: len(factor_projections[factor]) for factor in self.factors
        }
        self.compression_length = self.position_dim**0.5
        self.log_compression_length = math.log(self.compression_length)
        self.inverse_length_scale = (
            2.0 * math.pi / (length_factor * self.position_dim)
        )
        self.register_buffer(
            "position_frequencies",
            1
            + torch.arange(
                self.position_dim // 2,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        for factor in self.factors:
            projections = factor_projections[factor]
            self.register_buffer(
                f"position_weight_{factor}",
                torch.cat(
                    [projection.weight.detach() for projection in projections],
                    dim=0,
                ),
                persistent=False,
            )
            self.register_buffer(
                f"position_bias_{factor}",
                torch.cat(
                    [
                        projection.bias.detach()
                        if projection.bias is not None
                        else projection.weight.new_zeros(projection.out_features)
                        for projection in projections
                    ],
                    dim=0,
                ),
                persistent=False,
            )

    def forward(self, condition_projection: Tensor) -> tuple[Tensor, ...]:
        shape = shape_as_tensor(condition_projection)
        total_frames = shape[1]
        batch_size = condition_projection.shape[0] * self.batch_multiplier
        outputs: list[Tensor] = []
        for factor in self.factors:
            sequence_length = torch.div(
                total_frames + factor - 1,
                factor,
                rounding_mode="floor",
            )
            positions = torch.arange(
                1 - sequence_length,
                sequence_length,
                dtype=torch.float32,
                device=condition_projection.device,
            ).unsqueeze(1)
            compressed = (
                self.compression_length
                * positions.sign()
                * (
                    (positions.abs() + self.compression_length).log()
                    - self.log_compression_length
                )
            )
            angles = (compressed * self.inverse_length_scale).atan()
            angles = angles * self.position_frequencies
            positional = torch.stack(
                (angles.cos(), angles.sin()),
                dim=-1,
            ).flatten(1)
            positional = torch.cat(
                (
                    positional[:, :-1],
                    torch.ones_like(positional[:, -1:]),
                ),
                dim=1,
            )
            positional = positional.to(condition_projection.dtype).unsqueeze(0)
            projected_positions = F.linear(
                positional,
                getattr(self, f"position_weight_{factor}"),
                getattr(self, f"position_bias_{factor}"),
            ).reshape(
                1,
                -1,
                self.factor_layer_counts[factor],
                self.num_heads,
                self.pos_head_dim,
            ).permute(2, 3, 0, 4, 1)

            rows = torch.arange(
                start=sequence_length - 1,
                end=-1,
                step=-1,
                dtype=torch.int32,
                device=condition_projection.device,
            )
            columns = torch.arange(
                sequence_length,
                dtype=torch.int32,
                device=condition_projection.device,
            )
            relative_indices = (
                rows.repeat(batch_size * self.num_heads).unsqueeze(-1)
                + columns
            )
            outputs.extend((projected_positions, relative_indices))
        return tuple(outputs)


class ZipVoiceTimeEmbedding(nn.Module):
    def __init__(self, model: nn.Module, variant: Variant) -> None:
        super().__init__()
        decoder = model.fm_decoder
        self.time_embed = decoder.time_embed
        self.time_embed_dim = decoder.time_embed_dim
        self.distilled = variant.distilled
        self.register_buffer(
            "time_frequencies",
            torch.exp(
                -math.log(10000.0)
                * torch.arange(
                    self.time_embed_dim // 2,
                    dtype=torch.float32,
                )
                / (self.time_embed_dim // 2)
            ),
            persistent=False,
        )

        if self.distilled:
            if decoder.guidance_scale_embed is None:
                raise ValueError("Distilled flow requires guidance embedding")
            self.guidance_scale_embed = decoder.guidance_scale_embed
            self.guidance_scale_embed_dim = decoder.guidance_scale_embed_dim
            self.register_buffer(
                "guidance_frequencies",
                torch.exp(
                    -math.log(10000.0)
                    * torch.arange(
                        self.guidance_scale_embed_dim // 2,
                        dtype=torch.float32,
                    )
                    / (self.guidance_scale_embed_dim // 2)
                ),
                persistent=False,
            )
        else:
            self.guidance_scale_embed = None
            self.guidance_scale_embed_dim = 0

        stack_projections: list[nn.Linear] = []
        for stack in decoder.encoders:
            encoder = stack.encoder if hasattr(stack, "downsample_factor") else stack
            if encoder.time_emb is None or not isinstance(encoder.time_emb[1], nn.Linear):
                raise TypeError("Flow stack requires a linear time projection")
            stack_projections.append(encoder.time_emb[1])
        output_widths = {projection.out_features for projection in stack_projections}
        if len(output_widths) != 1:
            raise ValueError("Flow stack time projections must have one output width")
        self.stack_count = len(stack_projections)
        self.stack_embedding_dim = output_widths.pop()
        self.register_buffer(
            "stack_time_weight",
            torch.cat(
                [projection.weight.detach() for projection in stack_projections],
                dim=0,
            ),
            persistent=False,
        )
        self.register_buffer(
            "stack_time_bias",
            torch.cat(
                [
                    projection.bias.detach()
                    if projection.bias is not None
                    else projection.weight.new_zeros(projection.out_features)
                    for projection in stack_projections
                ],
                dim=0,
            ),
            persistent=False,
        )

    def forward(
        self,
        timesteps: Tensor,
        guidance_scale: Tensor | None = None,
    ) -> Tensor:
        if timesteps.dim() != 1:
            raise ValueError("Time embedding table requires a timestep vector")
        time_embedding = _timestep_embedding(
            timesteps,
            self.time_embed_dim,
            self.time_frequencies,
        )
        if self.distilled:
            if guidance_scale is None:
                raise ValueError("Distilled flow requires guidance scale")
            guidance_embedding = _timestep_embedding(
                guidance_scale,
                self.guidance_scale_embed_dim,
                self.guidance_frequencies,
            )
            time_embedding = time_embedding + _linear(
                guidance_embedding,
                self.guidance_scale_embed,
            )
        time_embedding = _time_embedding(self.time_embed, time_embedding)
        time_embedding = _swoosh_forward(
            time_embedding,
            offset=1.0,
            constant=0.313261687,
        )
        stack_time_embeddings = F.linear(
            time_embedding,
            self.stack_time_weight,
            self.stack_time_bias,
        )
        return stack_time_embeddings.reshape(
            -1,
            self.stack_count,
            self.stack_embedding_dim,
        )


class ZipVoiceFlowStep(nn.Module):
    def __init__(self, model: nn.Module, variant: Variant) -> None:
        super().__init__()
        self.fm_decoder = model.fm_decoder
        self.distilled = variant.distilled
        self.state_width = N_MELS * variant.output_channels
        input_width = self.state_width * 2 + N_MELS
        self.projection_index = _flow_projection_index(
            self.fm_decoder,
            input_width,
        )
        projection = _flow_input_projection(
            self.fm_decoder,
            self.projection_index,
        )
        state_weight = projection.weight[:, : self.state_width].detach().clone()
        self.hidden_width = projection.out_features
        if self.distilled:
            self.register_buffer(
                "state_weight",
                state_weight.transpose(0, 1).contiguous(),
            )
        else:
            self.register_buffer("state_weight", state_weight)
            self.register_buffer("state_bias", projection.bias.detach().clone())

    def _velocity_from_hidden(
        self,
        hidden: Tensor,
        time_embeddings: Tensor,
        relative_geometry: dict[int, tuple[Tensor, Tensor]],
    ) -> Tensor:
        return _flow_zipformer_hidden(
            self.fm_decoder,
            hidden.permute(1, 0, 2),
            time_embeddings,
            relative_geometry,
            padding_mask=None,
            projection_index=self.projection_index,
        )

    def forward(
        self,
        delta_t: Tensor,
        state: Tensor,
        condition_projection: Tensor,
        time_embeddings: Tensor,
        projected_positions_1: Tensor,
        relative_indices_1: Tensor,
        projected_positions_2: Tensor,
        relative_indices_2: Tensor,
        projected_positions_4: Tensor,
        relative_indices_4: Tensor,
        timestep: Tensor | None = None,
        guidance_scale: Tensor | None = None,
        speech_projection: Tensor | None = None,
    ) -> Tensor:
        relative_geometry = {
            1: (projected_positions_1, relative_indices_1),
            2: (projected_positions_2, relative_indices_2),
            4: (projected_positions_4, relative_indices_4),
        }
        if self.distilled:
            batch_size, frame_count, _ = state.shape
            state_hidden = torch.addmm(
                condition_projection.reshape(-1, self.hidden_width),
                state.reshape(-1, self.state_width),
                self.state_weight,
            ).reshape(batch_size, frame_count, self.hidden_width)
            velocity = self._velocity_from_hidden(
                hidden=state_hidden,
                time_embeddings=time_embeddings,
                relative_geometry=relative_geometry,
            )
        else:
            if timestep is None or guidance_scale is None or speech_projection is None:
                raise ValueError("Non-distilled flow requires all CFG controls")
            state_hidden = F.linear(state, self.state_weight, self.state_bias)
            unconditional = state_hidden + torch.where(
                timestep > 0.5,
                torch.zeros_like(speech_projection),
                speech_projection,
            )
            conditional = state_hidden + condition_projection + speech_projection
            hidden_pair = torch.cat(
                (
                    unconditional,
                    conditional,
                ),
                dim=0,
            )
            active_guidance = torch.where(
                timestep > 0.5,
                guidance_scale,
                guidance_scale * 2.0,
            )
            unconditional, conditional = self._velocity_from_hidden(
                hidden=hidden_pair,
                time_embeddings=time_embeddings,
                relative_geometry=relative_geometry,
            ).chunk(2, dim=0)
            velocity = (
                (1.0 + active_guidance) * conditional
                - active_guidance * unconditional
            )
        return state + velocity * delta_t


class ZipVoiceDecode(nn.Module):
    def __init__(
        self,
        decoder: VocosSpectralDecoder,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.istft = STFT_Process(
            model_type="istft_A",
            n_fft=N_FFT,
            win_length=WINDOW_LENGTH,
            hop_len=HOP_LENGTH,
            window_type="hann",
            center_pad=True,
        )
        self.output_channels = output_channels
        self.output_resample_scale = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)

    def forward(
        self,
        final_features: Tensor,
        prompt_features_len: Tensor,
        rms_scale: Tensor,
    ) -> Tensor:
        generated = final_features[:, prompt_features_len:]
        batch_size, generated_frames, _ = generated.shape
        decoder_input = generated.reshape(
            batch_size,
            generated_frames,
            self.output_channels,
            N_MELS,
        ).permute(0, 2, 3, 1)
        decoder_input = decoder_input.reshape(
            batch_size * self.output_channels,
            N_MELS,
            generated_frames,
        )
        if not getattr(self.decoder, "_onnx_input_scale_folded", False):
            decoder_input = decoder_input / FEATURE_SCALE
        magnitude, phase = _vocos_forward(self.decoder, decoder_input)
        audio = self.istft(magnitude, phase).reshape(
            batch_size,
            self.output_channels,
            -1,
        )
        audio = audio.clamp(min=-1.0, max=1.0)
        audio = audio * rms_scale.reshape(1, 1, 1)
        if self.output_resample_scale != 1.0:
            audio = F.interpolate(
                audio,
                scale_factor=self.output_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        return audio[0]


class MetadataCarrier(nn.Module):
    def forward(self, marker: Tensor) -> Tensor:
        return marker


def _model_path(suffix: str, folder: Path = OUTPUT_FOLDER) -> Path:
    return folder / f"{VARIANT.package_stem}_{suffix}.onnx"


def _checkpoint_path() -> Path:
    if CHECKPOINT_NAME is not None:
        return MODEL_DIRECTORY / CHECKPOINT_NAME
    candidates = (MODEL_DIRECTORY / "model.safetensors", MODEL_DIRECTORY / "model.pt")
    return next(path for path in candidates if path.is_file())


def _load_model() -> nn.Module:
    if str(OFFICIAL_SOURCE) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_SOURCE))

    from zipvoice.models.zipvoice import ZipVoice  # pyright: ignore[reportMissingImports]
    from zipvoice.models.zipvoice_dialog import (  # pyright: ignore[reportMissingImports]
        ZipVoiceDialog,
        ZipVoiceDialogStereo,
    )
    from zipvoice.models.zipvoice_distill import (  # pyright: ignore[reportMissingImports]
        ZipVoiceDistill,
    )
    from zipvoice.tokenizer.tokenizer import (  # pyright: ignore[reportMissingImports]
        DialogTokenizer,
        EmiliaTokenizer,
        LibriTTSTokenizer,
    )
    from zipvoice.utils.checkpoint import (  # pyright: ignore[reportMissingImports]
        load_checkpoint,
    )
    from zipvoice.utils.scaling_converter import (  # pyright: ignore[reportMissingImports]
        convert_scaled_to_non_scaled,
    )

    tokenizer_classes = {
        "emilia": EmiliaTokenizer,
        "libritts": LibriTTSTokenizer,
        "dialog": DialogTokenizer,
    }
    tokenizer = tokenizer_classes[VARIANT.tokenizer_type](
        token_file=str(MODEL_DIRECTORY / "tokens.txt")
    )
    tokenizer_config: dict[str, Any] = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
    }
    if VARIANT.dialogue:
        tokenizer_config.update(
            spk_a_id=tokenizer.spk_a_id,
            spk_b_id=tokenizer.spk_b_id,
        )

    with (MODEL_DIRECTORY / "model.json").open(encoding="utf-8") as file:
        model_config = json.load(file)["model"]
    model_classes = {
        "ZipVoice": ZipVoice,
        "ZipVoiceDistill": ZipVoiceDistill,
        "ZipVoiceDialog": ZipVoiceDialog,
        "ZipVoiceDialogStereo": ZipVoiceDialogStereo,
    }
    model = model_classes[VARIANT.model_class](**model_config, **tokenizer_config)

    checkpoint = _checkpoint_path()
    if checkpoint.suffix == ".safetensors":
        import safetensors.torch

        safetensors.torch.load_model(model, checkpoint)
    elif checkpoint.suffix == ".pt":
        load_checkpoint(filename=checkpoint, model=model, strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.suffix}")

    model.eval()
    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)
    return _prepare_model_for_export(model)


def _prepare_model_for_export(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        for module in model.modules():
            if (
                type(module).__name__ == "RelPositionMultiheadAttentionWeights"
                and not getattr(module, "_onnx_qkp_interleaved", False)
            ):
                num_heads = module.num_heads
                query_head_dim = module.query_head_dim
                pos_head_dim = module.pos_head_dim
                query_dim = num_heads * query_head_dim
                pos_dim = num_heads * pos_head_dim
                query_weight, key_weight, position_weight = (
                    module.in_proj.weight.split(
                        (query_dim, query_dim, pos_dim),
                        dim=0,
                    )
                )
                module.in_proj.weight.copy_(
                    torch.cat(
                        (
                            query_weight.reshape(num_heads, query_head_dim, -1),
                            key_weight.reshape(num_heads, query_head_dim, -1),
                            position_weight.reshape(num_heads, pos_head_dim, -1),
                        ),
                        dim=1,
                    ).reshape_as(module.in_proj.weight)
                )
                if module.in_proj.bias is not None:
                    query_bias, key_bias, position_bias = module.in_proj.bias.split(
                        (query_dim, query_dim, pos_dim),
                        dim=0,
                    )
                    module.in_proj.bias.copy_(
                        torch.cat(
                            (
                                query_bias.reshape(num_heads, query_head_dim),
                                key_bias.reshape(num_heads, query_head_dim),
                                position_bias.reshape(num_heads, pos_head_dim),
                            ),
                            dim=1,
                        ).reshape_as(module.in_proj.bias)
                    )
                module._onnx_qkp_interleaved = True
            if (
                type(module).__name__ == "BiasNorm"
                and "_onnx_output_scale" not in module._buffers
            ):
                module.register_buffer(
                    "_onnx_output_scale",
                    module.log_scale.detach().exp(),
                    persistent=False,
                )
            if (
                type(module).__name__ == "SimpleDownsample"
                and "_onnx_weights" not in module._buffers
            ):
                module.register_buffer(
                    "_onnx_weights",
                    F.softmax(module.bias.detach(), dim=0),
                    persistent=False,
                )
        for module in model.modules():
            if (
                type(module).__name__ == "Zipformer2EncoderLayer"
                and "_onnx_residual_scale" not in module.bypass._buffers
            ):
                bypass_scale = module.bypass.bypass_scale.detach()
                output_scale = module.norm._onnx_output_scale
                module.norm._buffers["_onnx_output_scale"] = (
                    output_scale * bypass_scale
                )
                module.bypass.register_buffer(
                    "_onnx_residual_scale",
                    1.0 - bypass_scale,
                    persistent=False,
                )
    return model


def _load_vocos() -> VocosSpectralDecoder:
    with (VOCOS_DIRECTORY / "config.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    backbone = VocosBackbone(**config["backbone"]["init_args"])
    head = VocosISTFTHead(**config["head"]["init_args"])
    decoder = VocosSpectralDecoder(backbone, head)
    state = torch.load(
        VOCOS_DIRECTORY / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    decoder_state = {
        key: value
        for key, value in state.items()
        if key.startswith(("backbone.", "head."))
    }
    decoder.load_state_dict(decoder_state, strict=True)
    return decoder.eval()


def _prepare_vocos_for_export(
    decoder: VocosSpectralDecoder,
) -> VocosSpectralDecoder:
    if getattr(decoder, "_onnx_input_scale_folded", False):
        return decoder
    with torch.no_grad():
        decoder.backbone.embed.weight.div_(FEATURE_SCALE)
        for block in decoder.backbone.convnext:
            norm = block.norm
            if not isinstance(norm, nn.LayerNorm):
                raise TypeError("Vocos affine folding requires LayerNorm")
            if norm.bias is not None:
                folded_bias = F.linear(
                    norm.bias,
                    block.pwconv1.weight,
                    block.pwconv1.bias,
                )
                if block.pwconv1.bias is None:
                    block.pwconv1.bias = nn.Parameter(folded_bias)
                else:
                    block.pwconv1.bias.copy_(folded_bias)
            if norm.weight is not None:
                block.pwconv1.weight.mul_(norm.weight.unsqueeze(0))
            norm._onnx_affine_folded = True

            block.pwconv2.weight.mul_(block.gamma[:, None])
            block.pwconv2.bias.mul_(block.gamma)
            block._onnx_gamma_folded = True

        final_norm = decoder.backbone.final_layer_norm
        if final_norm.bias is not None:
            folded_bias = F.linear(
                final_norm.bias,
                decoder.head.out.weight,
                decoder.head.out.bias,
            )
            if decoder.head.out.bias is None:
                decoder.head.out.bias = nn.Parameter(folded_bias)
            else:
                decoder.head.out.bias.copy_(folded_bias)
        if final_norm.weight is not None:
            decoder.head.out.weight.mul_(final_norm.weight.unsqueeze(0))
        final_norm._onnx_affine_folded = True
    decoder._onnx_input_scale_folded = True
    return decoder


def _export(
    module: nn.Module,
    arguments: tuple[Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
) -> None:
    module = module.eval()
    if any(isinstance(child, torch.jit.ScriptModule) for child in module.modules()):
        module = torch.jit.trace(module, arguments, check_trace=False)

    torch.onnx.export(
        module,
        arguments,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False,
        external_data=True,
    )


def _write_metadata(path: Path) -> None:
    import onnx

    metadata = {
        "model_name": MODEL_NAME,
        "tokenizer_type": VARIANT.tokenizer_type,
        "output_channels": VARIANT.output_channels,
        "sample_rate": MODEL_SAMPLE_RATE,
        "in_sample_rate": IN_SAMPLE_RATE,
        "out_sample_rate": OUT_SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "center_pad": 1,
        "default_num_step": VARIANT.default_num_step,
        "default_guidance_scale": VARIANT.default_guidance_scale,
        "default_t_shift": DEFAULT_T_SHIFT,
        "chunk_target_seconds": VARIANT.chunk_target_seconds,
        "crossfade_seconds": CROSSFADE_SECONDS,
        "prompt_trailing_silence_ms": PROMPT_TRAILING_SILENCE_MS,
    }
    model = onnx.load(str(path), load_external_data=False)
    for key, value in metadata.items():
        model.metadata_props.add(key=key, value=str(value))
    onnx.save(model, str(path))


def _export_preprocess(output_folder: Path = OUTPUT_FOLDER) -> None:
    audio = torch.ones(
        1,
        VARIANT.output_channels,
        IN_SAMPLE_RATE,
        dtype=torch.float32,
    )
    _export(
        ZipVoicePreprocess(VARIANT.output_channels),
        (audio,),
        _model_path("Preprocess", output_folder),
        ["audio"],
        ["prompt_features", "prompt_features_len", "rms_scale"],
        {
            "audio": {1: "audio_channels", 2: "audio_samples"},
            "prompt_features": {1: "prompt_frames"},
        },
    )


def _export_text_encoder(
    model: nn.Module,
    output_folder: Path = OUTPUT_FOLDER,
) -> None:
    prompt_frames = 94
    acoustic_width = N_MELS * VARIANT.output_channels
    arguments = (
        torch.tensor([[2, 3, 4, 5]], dtype=torch.int64),
        torch.tensor([[6, 7]], dtype=torch.int64),
        torch.ones(1, prompt_frames, acoustic_width, dtype=torch.float32),
        torch.tensor(prompt_frames, dtype=torch.int64),
        torch.tensor(1.0, dtype=torch.float32),
    )
    _export(
        ZipVoiceTextEncoder(model, VARIANT),
        arguments,
        _model_path("TextEncoder", output_folder),
        ["tokens", "prompt_tokens", "prompt_features", "prompt_features_len", "speed"],
        ["initial_noise", "text_condition", "speech_condition"],
        {
            "tokens": {1: "target_tokens"},
            "prompt_tokens": {1: "prompt_tokens"},
            "prompt_features": {1: "prompt_frames"},
            "initial_noise": {1: "total_frames"},
            "text_condition": {1: "total_frames"},
            "speech_condition": {1: "total_frames"},
        },
    )


def _export_flow_condition(
    model: nn.Module,
    output_folder: Path = OUTPUT_FOLDER,
) -> None:
    total_frames = 200
    acoustic_width = N_MELS * VARIANT.output_channels
    arguments = (
        torch.ones(1, total_frames, N_MELS, dtype=torch.float32),
        torch.ones(1, total_frames, acoustic_width, dtype=torch.float32),
    )
    output_names = (
        ["condition_projection"]
        if VARIANT.distilled
        else ["condition_projection", "speech_projection"]
    )
    _export(
        ZipVoiceFlowCondition(model, VARIANT),
        arguments,
        _model_path("FlowCondition", output_folder),
        ["text_condition", "speech_condition"],
        output_names,
        {
            "text_condition": {1: "total_frames"},
            "speech_condition": {1: "total_frames"},
            **{name: {1: "total_frames"} for name in output_names},
        },
    )


def _export_flow_geometry(
    model: nn.Module,
    output_folder: Path = OUTPUT_FOLDER,
) -> None:
    total_frames = 200
    flow_hidden = model.fm_decoder.encoders[0].layers[0].embed_dim
    module = ZipVoiceFlowGeometry(model, VARIANT)
    output_names: list[str] = []
    dynamic_axes = {"condition_projection": {1: "total_frames"}}
    for factor in module.factors:
        positional_name = f"projected_positions_{factor}"
        indices_name = f"relative_indices_{factor}"
        output_names.extend((positional_name, indices_name))
        dynamic_axes[positional_name] = {4: f"relative_frames_{factor}"}
        dynamic_axes[indices_name] = {
            0: f"relative_rows_{factor}",
            1: f"relative_columns_{factor}",
        }
    _export(
        module,
        (torch.ones(1, total_frames, flow_hidden, dtype=torch.float32),),
        _model_path("FlowGeometry", output_folder),
        ["condition_projection"],
        output_names,
        dynamic_axes,
    )


def _export_time_embedding(
    model: nn.Module,
    output_folder: Path = OUTPUT_FOLDER,
) -> None:
    num_steps = VARIANT.default_num_step
    arguments: tuple[Tensor, ...] = (
        torch.arange(num_steps, dtype=torch.float32) / num_steps,
    )
    input_names = ["timesteps"]
    if VARIANT.distilled:
        arguments += (
            torch.tensor(VARIANT.default_guidance_scale, dtype=torch.float32),
        )
        input_names.append("guidance_scale")
    _export(
        ZipVoiceTimeEmbedding(model, VARIANT),
        arguments,
        _model_path("TimeEmbedding", output_folder),
        input_names,
        ["time_embeddings"],
        {
            "timesteps": {0: "num_steps"},
            "time_embeddings": {0: "num_steps"},
        },
    )


def _export_flow_step(
    model: nn.Module,
    output_folder: Path = OUTPUT_FOLDER,
) -> None:
    total_frames = 200
    acoustic_width = N_MELS * VARIANT.output_channels
    flow_hidden = _flow_input_projection(
        model.fm_decoder,
        _flow_projection_index(
            model.fm_decoder,
            acoustic_width * 2 + N_MELS,
        ),
    ).out_features
    arguments: tuple[Tensor, ...] = (
        torch.tensor(0.05, dtype=torch.float32),
        torch.ones(1, total_frames, acoustic_width, dtype=torch.float32),
        torch.ones(1, total_frames, flow_hidden, dtype=torch.float32),
        torch.ones(
            len(model.fm_decoder.encoders),
            flow_hidden,
            dtype=torch.float32,
        ),
    )
    input_names = ["delta_t", "x", "condition_projection", "time_embeddings"]
    dynamic_axes = {
        "x": {1: "total_frames"},
        "condition_projection": {1: "total_frames"},
        "x_next": {1: "total_frames"},
    }
    geometry = ZipVoiceFlowGeometry(model, VARIANT)
    for factor in geometry.factors:
        sequence_length = (total_frames + factor - 1) // factor
        positional_name = f"projected_positions_{factor}"
        indices_name = f"relative_indices_{factor}"
        arguments += (
            torch.ones(
                geometry.factor_layer_counts[factor],
                geometry.num_heads,
                1,
                geometry.pos_head_dim,
                2 * sequence_length - 1,
                dtype=torch.float32,
            ),
            torch.zeros(
                geometry.batch_multiplier
                * geometry.num_heads
                * sequence_length,
                sequence_length,
                dtype=torch.int32,
            ),
        )
        input_names.extend((positional_name, indices_name))
        dynamic_axes[positional_name] = {4: f"relative_frames_{factor}"}
        dynamic_axes[indices_name] = {
            0: f"relative_rows_{factor}",
            1: f"relative_columns_{factor}",
        }
    if not VARIANT.distilled:
        arguments += (
            torch.tensor(0.25, dtype=torch.float32),
            torch.tensor(VARIANT.default_guidance_scale, dtype=torch.float32),
            torch.ones(1, total_frames, flow_hidden, dtype=torch.float32),
        )
        input_names.extend(("t", "guidance_scale", "speech_projection"))
        dynamic_axes["speech_projection"] = {1: "total_frames"}
    _export(
        ZipVoiceFlowStep(model, VARIANT),
        arguments,
        _model_path("FlowStep", output_folder),
        input_names,
        ["x_next"],
        dynamic_axes,
    )


def _export_decode(output_folder: Path = OUTPUT_FOLDER) -> None:
    total_frames = 200
    generated_frames = 150
    acoustic_width = N_MELS * VARIANT.output_channels
    arguments = (
        torch.ones(1, total_frames, acoustic_width, dtype=torch.float32),
        torch.tensor(total_frames - generated_frames, dtype=torch.int64),
        torch.ones(1, dtype=torch.float32),
    )
    _export(
        ZipVoiceDecode(
            _prepare_vocos_for_export(_load_vocos()),
            VARIANT.output_channels,
        ),
        arguments,
        _model_path("Decode", output_folder),
        ["final_features", "prompt_features_len", "rms_scale"],
        ["output_audio"],
        {
            "final_features": {1: "total_frames"},
            "output_audio": {1: "audio_samples"},
        },
    )


def _export_metadata(output_folder: Path = OUTPUT_FOLDER) -> None:
    marker = torch.zeros(1, dtype=torch.float32)
    path = _model_path("Metadata", output_folder)
    _export(MetadataCarrier(), (marker,), path, ["marker"], ["marker_out"], {})
    _write_metadata(path)


def _remove_stale_package_artifacts(
    final_model: Path,
    metadata_model: Path,
) -> None:
    retained_paths = {
        final_model,
        metadata_model,
        final_model.with_name(final_model.name + ".data"),
        metadata_model.with_name(metadata_model.name + ".data"),
    }
    for pattern in (
        f"{VARIANT.package_stem}_*.onnx",
        f"{VARIANT.package_stem}_*.onnx.data",
        f".{VARIANT.package_stem}_*.tmp",
    ):
        for path in OUTPUT_FOLDER.glob(pattern):
            if path not in retained_paths:
                path.unlink(missing_ok=True)
    for name in ("tokens.txt", "model.json"):
        (OUTPUT_FOLDER / name).unlink(missing_ok=True)


def _prepare_component_folder() -> None:
    COMPONENT_FOLDER.mkdir(parents=True, exist_ok=True)
    for pattern in (
        f"{VARIANT.package_stem}_*.onnx",
        f"{VARIANT.package_stem}_*.onnx.data",
        f".{VARIANT.package_stem}_*.tmp",
    ):
        for path in COMPONENT_FOLDER.glob(pattern):
            path.unlink(missing_ok=True)


def _run_post_export_inference(output_folder: Path) -> None:
    task_mode = "dialogue" if VARIANT.dialogue else "single"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "Inference_ZipVoice_ONNX.py"),
            "--onnx-folder",
            str(output_folder),
            "--tokens",
            str(MODEL_DIRECTORY / "tokens.txt"),
            "--tokenizer",
            VARIANT.tokenizer_type,
            "--task-mode",
            task_mode,
        ],
        check=True,
    )


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    _prepare_component_folder()
    final_model = _model_path("Pipeline")
    metadata_model = _model_path("Metadata")
    with torch.inference_mode():
        _export_preprocess(COMPONENT_FOLDER)
        model = _load_model()
        _export_text_encoder(model, COMPONENT_FOLDER)
        _export_flow_condition(model, COMPONENT_FOLDER)
        _export_flow_geometry(model, COMPONENT_FOLDER)
        _export_time_embedding(model, COMPONENT_FOLDER)
        _export_flow_step(model, COMPONENT_FOLDER)
        del model
        gc.collect()
        _export_decode(COMPONENT_FOLDER)
        _export_metadata(COMPONENT_FOLDER)

    from Merge_ONNX import _merge_pipeline

    _merge_pipeline(COMPONENT_FOLDER, final_model)

    _remove_stale_package_artifacts(final_model, metadata_model)
    print(f"Exported {final_model}; starting inference", flush=True)
    _run_post_export_inference(OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
