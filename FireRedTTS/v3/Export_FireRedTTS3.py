"""Export a self-contained FireRedTTS3 ONNX package from local checkpoints.

All tensor forwards used by exported graphs are declared here. The FireRed
source checkout is intentionally never imported by this exporter.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import onnx
import torch
import torch.nn.functional as functional
import torchaudio
from safetensors import safe_open
from torch import nn
from torchaudio.compliance.kaldi import get_mel_banks

from Shared_Weights import (
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    build_metadata,
    inference_metadata,
    promote_directory,
    replace_onnx_metadata,
    write_metadata_carrier,
)
from STFT_Process import STFT_Process


# ---------------------------------------------------------------------------
# Editable export settings
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = SCRIPT_DIR.parents[3]
CHECKPOINT_ROOT = Path(
    os.environ.get("FIREREDTTS3_CHECKPOINT_ROOT", DOWNLOADS_DIR / "FireRedTTS3")
).expanduser()
MODEL_VARIANT = "instruct"  # "base" | "instruct"
DO_EXPORT     = True        # Set True to run the export pipeline.

# Fixed model and runtime configuration.
OPSET = 20
MAX_SEQ_LEN = 1024
MAX_AUDIO_PATCHES = 400
MIN_AUDIO_PATCHES = 6
FLOW_STEPS = 10
IN_SAMPLE_RATE = 24_000
OUT_SAMPLE_RATE = 24_000
DEFAULT_BASE_CFG = 2.0
DEFAULT_INSTRUCT_CLONE_CFG = 2.0
DEFAULT_INSTRUCT_CFG = 1.2
STOP_THRESHOLD_DEFAULT = 0.5
CACHE_STORAGE_DTYPE = torch.float16

TEXT_EOT_ID = 151677
AUDIO_SOS_ID = 151669
LATENT_IN_PAD_ID = 151655
LATENT_OUT_PAD_ID = 151656
REDAE_SCALE = 0.4
QWEN_ROPE_THETA = 1_000_000.0
QWEN_RMS_EPS = 1.0e-6
PATCH_RMS_EPS = 1.0e-6
POST_EXPORT_DEMO_PACKAGE_ENV = "FIREREDTTS3_POST_EXPORT_DEMO_PACKAGE"


@dataclass(frozen=True)
class QwenGeometry:
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    key_value_heads: int
    head_dim: int
    vocab_size: int | None
    rope_theta: float
    rms_eps: float
    sliding_window: int | None = None
    max_sequence_length: int = MAX_SEQ_LEN


@dataclass(frozen=True)
class PackageGeometry:
    variant: str
    redae_dim: int
    patch_size: int
    history_patches: int
    history_length: int
    patch_hidden_size: int
    patch_depth: int
    patch_heads: int
    patch_mlp_ratio: int
    dit_hidden_size: int
    dit_depth: int
    dit_heads: int
    dit_mlp_ratio: int
    speaker_dim: int | None
    audio_patch_size: int
    out_sample_rate: int
    redae_downsample_rate: int
    encoder_qwen: QwenGeometry
    downsample_qwen: QwenGeometry
    decoder_qwen: QwenGeometry
    backbone_qwen: QwenGeometry


def _read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _inspect_qwen_geometry(
    checkpoint: Path,
    prefix: str,
    *,
    include_vocab: bool,
    sliding_window: int | None = None,
    max_sequence_length: int = MAX_SEQ_LEN,
) -> QwenGeometry:
    layer_pattern = re.compile(re.escape(prefix) + r"layers\.(\d+)\.self_attn\.q_proj\.weight$")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        keys = tuple(source.keys())
        layer_indices = sorted(
            int(match.group(1))
            for key in keys
            if (match := layer_pattern.match(key)) is not None
        )
        q_weight = source.get_tensor(prefix + "layers.0.self_attn.q_proj.weight")
        k_weight = source.get_tensor(prefix + "layers.0.self_attn.k_proj.weight")
        gate_weight = source.get_tensor(prefix + "layers.0.mlp.gate_proj.weight")
        q_norm = source.get_tensor(prefix + "layers.0.self_attn.q_norm.weight")
        embedding_key = prefix + "embed_tokens.weight"
        embedding = source.get_tensor(embedding_key) if include_vocab else None

    hidden_size = int(q_weight.shape[1])
    query_projection_size = int(q_weight.shape[0])
    head_dim = int(q_norm.shape[0])
    attention_heads = query_projection_size // head_dim
    key_value_heads = int(k_weight.shape[0]) // head_dim
    return QwenGeometry(
        hidden_size=hidden_size,
        intermediate_size=int(gate_weight.shape[0]),
        layers=len(layer_indices),
        attention_heads=attention_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        vocab_size=int(embedding.shape[0]) if embedding is not None else None,
        rope_theta=QWEN_ROPE_THETA,
        rms_eps=QWEN_RMS_EPS,
        sliding_window=sliding_window,
        max_sequence_length=max_sequence_length,
    )


def _uniform_sliding_window(
    config: Mapping[str, object], prefix: str
) -> int | None:
    if not bool(config[f"{prefix}_use_sliding_window"]):
        return None
    layer_count = int(config[f"{prefix}_num_hidden_layers"])
    first_sliding_layer = int(config[f"{prefix}_max_window_layers"])
    if first_sliding_layer >= layer_count:
        return None
    if first_sliding_layer != 0:
        raise ValueError(
            "The manual Qwen exporter requires one attention policy for all "
            f"{prefix} layers; first sliding layer is {first_sliding_layer}."
        )
    window = int(config[f"{prefix}_sliding_window"])
    if window <= 0:
        raise ValueError(f"{prefix} sliding window must be positive, got {window}.")
    return window


def read_package_geometry(root: Path, variant: str) -> PackageGeometry:
    tts_config = _read_json(root / f"fireredtts3_{variant}" / "config.json")
    redae_config = _read_json(root / "redae" / "config.json")
    tts_checkpoint = root / f"fireredtts3_{variant}" / "model.safetensors"
    redae_checkpoint = root / "redae" / "model.safetensors"
    backbone_prefix = "backbone_llm." if variant == "base" else "backbone_llm.model."
    backbone = _inspect_qwen_geometry(tts_checkpoint, backbone_prefix, include_vocab=True)
    encoder = _inspect_qwen_geometry(
        redae_checkpoint,
        "encoder.qwen3.",
        include_vocab=False,
        sliding_window=_uniform_sliding_window(redae_config, "enc"),
        max_sequence_length=int(redae_config["enc_max_position_embeddings"]),
    )
    downsample = _inspect_qwen_geometry(
        redae_checkpoint, "encoder.downsample.qwen3.", include_vocab=False
    )
    decoder = _inspect_qwen_geometry(
        redae_checkpoint,
        "decoder.qwen3.",
        include_vocab=False,
        sliding_window=_uniform_sliding_window(redae_config, "dec"),
        max_sequence_length=int(redae_config["dec_max_position_embeddings"]),
    )

    patch_size = int(tts_config["patch_size"])
    history_patches = int(tts_config["num_history_patches"])
    redae_dim = int(tts_config["redae_dim"])
    history_length = patch_size * history_patches
    audio_patch_size = int(redae_config["audio_patch_size"])
    downsample_rate = audio_patch_size * int(redae_config["enc_extra_downsample_rate"])
    return PackageGeometry(
        variant=variant,
        redae_dim=redae_dim,
        patch_size=patch_size,
        history_patches=history_patches,
        history_length=history_length,
        patch_hidden_size=int(tts_config["patch_encoder_hidden_size"]),
        patch_depth=int(tts_config["patch_encoder_depth"]),
        patch_heads=int(tts_config["patch_encoder_num_heads"]),
        patch_mlp_ratio=int(tts_config["patch_encoder_mlp_ratio"]),
        dit_hidden_size=int(tts_config["dit_hidden_size"]),
        dit_depth=int(tts_config["dit_depth"]),
        dit_heads=int(tts_config["dit_num_heads"]),
        dit_mlp_ratio=int(tts_config["dit_mlp_ratio"]),
        speaker_dim=int(tts_config["spk_in_dim"]) if variant == "base" else None,
        audio_patch_size=audio_patch_size,
        out_sample_rate=int(redae_config["audio_sample_rate"]),
        redae_downsample_rate=downsample_rate,
        encoder_qwen=encoder,
        downsample_qwen=downsample,
        decoder_qwen=decoder,
        backbone_qwen=backbone,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def load_module_from_safetensors(module: nn.Module, checkpoint: Path, prefix: str) -> None:
    """Copy matching checkpoint tensors into an exporter-local module."""
    expected = module.state_dict()
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        for name, reference in expected.items():
            tensor = source.get_tensor(prefix + name)
            state[name] = tensor.to(dtype=reference.dtype)
    module.load_state_dict(state, strict=False, assign=True)


def _fuse_parallel_linears(*linears: nn.Linear) -> nn.Linear:
    if not linears:
        raise ValueError("At least one linear projection is required for fusion.")
    input_size = linears[0].in_features
    if any(linear.in_features != input_size for linear in linears):
        raise ValueError("Parallel linear projections must share their input size.")
    has_bias = any(linear.bias is not None for linear in linears)
    reference = linears[0].weight
    fused = nn.Linear(
        input_size,
        sum(linear.out_features for linear in linears),
        bias=has_bias,
        device=reference.device,
        dtype=reference.dtype,
    )
    with torch.no_grad():
        fused.weight.copy_(torch.cat(tuple(linear.weight for linear in linears)))
        if has_bias:
            fused.bias.copy_(
                torch.cat(
                    tuple(
                        linear.bias
                        if linear.bias is not None
                        else torch.zeros(
                            linear.out_features,
                            device=reference.device,
                            dtype=reference.dtype,
                        )
                        for linear in linears
                    )
                )
            )
    return fused.eval()


def _compose_linears(first: nn.Linear, second: nn.Linear) -> nn.Linear:
    if first.out_features != second.in_features:
        raise ValueError("Consecutive linear projections have incompatible dimensions.")
    reference = first.weight
    composed = nn.Linear(
        first.in_features,
        second.out_features,
        bias=first.bias is not None or second.bias is not None,
        device=reference.device,
        dtype=reference.dtype,
    )
    with torch.no_grad():
        first_weight = first.weight.to(torch.float64)
        second_weight = second.weight.to(torch.float64)
        composed.weight.copy_((second_weight @ first_weight).to(reference.dtype))
        if composed.bias is not None:
            bias = torch.zeros(
                second.out_features, device=reference.device, dtype=torch.float64
            )
            if first.bias is not None:
                bias.add_(second_weight @ first.bias.to(torch.float64))
            if second.bias is not None:
                bias.add_(second.bias.to(torch.float64))
            composed.bias.copy_(bias.to(reference.dtype))
    return composed.eval()


class ManualRMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension, dtype=torch.float32))
        self.epsilon = float(epsilon)

    def take_weight(self) -> torch.Tensor | None:
        if self.weight is None:
            return None
        weight = self.weight.detach().clone()
        self.register_parameter("weight", None)
        return weight

    def fold_into(self, *linears: nn.Linear) -> None:
        weight = self.take_weight()
        if weight is None:
            return
        with torch.no_grad():
            for linear in linears:
                if linear.in_features != weight.numel():
                    raise ValueError(
                        "RMSNorm and linear dimensions do not match for folding."
                    )
                linear.weight.mul_(weight.unsqueeze(0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.epsilon)
        return normalized if self.weight is None else self.weight * normalized


def _build_rotary_tables(
    sequence_length: int,
    head_dim: int,
    rope_theta: float,
    *,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(sequence_length, dtype=torch.float32)
    inv_frequency = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_frequency = torch.pow(torch.tensor(rope_theta), -inv_frequency / head_dim)
    frequencies = positions.unsqueeze(-1) * inv_frequency.unsqueeze(0)
    if interleaved:
        embeddings = torch.stack((frequencies, frequencies), dim=-1).flatten(start_dim=-2)
    else:
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
    embeddings = embeddings.unsqueeze(0).unsqueeze(0)
    return embeddings.cos(), embeddings.sin()


def _build_rotary_signs(head_dim: int, *, interleaved: bool) -> torch.Tensor:
    if interleaved:
        return torch.tensor((-1.0, 1.0), dtype=torch.float32).repeat(
            head_dim // 2
        )
    half_size = head_dim // 2
    return torch.cat(
        (-torch.ones(half_size, dtype=torch.float32), torch.ones(half_size))
    )


def _build_causal_mask(
    sequence_length: int, sliding_window: int | None = None
) -> torch.Tensor:
    positions = torch.arange(sequence_length, dtype=torch.int64)
    if sliding_window is None:
        allowed = positions.unsqueeze(0) <= positions.unsqueeze(1)
    else:
        offsets = torch.arange(1 - sliding_window, 1, dtype=torch.int64)
        allowed = positions.unsqueeze(-1) + offsets.unsqueeze(0) >= 0
    mask = (~allowed).to(torch.int8) * -128
    return mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)


class ManualQwenAttention(nn.Module):
    def __init__(self, geometry: QwenGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.head_dim_half = geometry.head_dim // 2
        self.qkv_heads = geometry.attention_heads + 2 * geometry.key_value_heads
        query_projection_size = geometry.attention_heads * geometry.head_dim
        key_value_projection_size = geometry.key_value_heads * geometry.head_dim
        self.query_projection_size = query_projection_size
        self.key_value_projection_size = key_value_projection_size
        self.q_proj = nn.Linear(geometry.hidden_size, query_projection_size, bias=False)
        self.k_proj = nn.Linear(
            geometry.hidden_size, key_value_projection_size, bias=False
        )
        self.v_proj = nn.Linear(
            geometry.hidden_size, key_value_projection_size, bias=False
        )
        self.qkv_proj: nn.Linear | None = None
        self.o_proj = nn.Linear(query_projection_size, geometry.hidden_size, bias=False)
        self.q_norm = ManualRMSNorm(geometry.head_dim, geometry.rms_eps)
        self.k_norm = ManualRMSNorm(geometry.head_dim, geometry.rms_eps)
        self.score_scale = geometry.head_dim ** -0.5

    def prepare_for_export(self) -> None:
        if self.qkv_proj is not None:
            return
        self.qkv_proj = _fuse_parallel_linears(
            self.q_proj, self.k_proj, self.v_proj
        )
        with torch.no_grad():
            self.q_norm.weight.mul_(self.score_scale)
        self.score_scale = 1.0
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

    def _rotate_half(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, head_count, sequence_length = tensor.shape[:3]
        tensor = tensor.view(
            batch_size,
            head_count,
            sequence_length,
            2,
            self.head_dim_half,
        ).flip(-2)
        return tensor.view(
            batch_size,
            head_count,
            sequence_length,
            self.geometry.head_dim,
        )

    def _apply_rotary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            query * cosine + self._rotate_half(query) * sine,
            key * cosine + self._rotate_half(key) * sine,
        )

    def _grouped_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.geometry.sliding_window is not None:
            return self._grouped_sliding_attention(
                query, key, value, causal_mask
            )
        batch_size, _, query_length, head_dim = query.shape
        group_count = self.geometry.attention_heads // self.geometry.key_value_heads
        grouped_query = query.view(
            batch_size,
            self.geometry.key_value_heads,
            group_count,
            query_length,
            head_dim,
        )
        scores = torch.matmul(grouped_query, key.unsqueeze(2).transpose(-1, -2))
        if self.score_scale != 1.0:
            scores = scores * self.score_scale
        scores = scores + causal_mask
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probabilities, value.unsqueeze(2))
        attended = attended.reshape(
            batch_size,
            self.geometry.attention_heads,
            query_length,
            head_dim,
        )
        return attended.transpose(1, 2).reshape(batch_size, query_length, -1)

    def _sliding_windows(self, tensor: torch.Tensor) -> torch.Tensor:
        window = self.geometry.sliding_window
        if window is None:
            raise RuntimeError("Sliding windows require a configured window size.")
        batch_size, _, sequence_length, head_dim = tensor.shape
        image = tensor.transpose(2, 3).reshape(
            batch_size,
            self.geometry.key_value_heads * head_dim,
            sequence_length,
            1,
        )
        columns = functional.unfold(
            image,
            kernel_size=(window, 1),
            padding=(window - 1, 0),
        )[:, :, :sequence_length]
        return columns.reshape(
            batch_size,
            self.geometry.key_value_heads,
            head_dim,
            window,
            sequence_length,
        ).permute(0, 1, 4, 3, 2)

    def _grouped_sliding_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        window = self.geometry.sliding_window
        if window is None:
            raise RuntimeError("Sliding attention requires a configured window size.")
        batch_size, _, query_length, head_dim = query.shape
        group_count = self.geometry.attention_heads // self.geometry.key_value_heads
        grouped_query = query.view(
            batch_size,
            self.geometry.key_value_heads,
            group_count,
            query_length,
            head_dim,
        )
        key_windows = self._sliding_windows(key)
        value_windows = self._sliding_windows(value)
        scores = torch.matmul(
            grouped_query.unsqueeze(-2),
            key_windows.unsqueeze(2).transpose(-1, -2),
        ).squeeze(-2)
        if self.score_scale != 1.0:
            scores = scores * self.score_scale
        scores = scores + causal_mask
        probabilities = torch.softmax(scores, dim=-1)
        attended = torch.matmul(
            probabilities.unsqueeze(-2),
            value_windows.unsqueeze(2),
        ).squeeze(-2)
        attended = attended.reshape(
            batch_size,
            self.geometry.attention_heads,
            query_length,
            head_dim,
        )
        return attended.transpose(1, 2).reshape(batch_size, query_length, -1)

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = hidden_states.shape[:2]
        if self.qkv_proj is None:
            qkv_projection = torch.cat(
                (
                    self.q_proj(hidden_states),
                    self.k_proj(hidden_states),
                    self.v_proj(hidden_states),
                ),
                dim=-1,
            )
        else:
            qkv_projection = self.qkv_proj(hidden_states)
        qkv = qkv_projection.view(
            batch_size,
            sequence_length,
            self.qkv_heads,
            self.geometry.head_dim,
        ).transpose(1, 2)
        query, key, value = qkv.split(
            (
                self.geometry.attention_heads,
                self.geometry.key_value_heads,
                self.geometry.key_value_heads,
            ),
            dim=1,
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        return query, key, value

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value = self._project_qkv(hidden_states)
        query, key = self._apply_rotary(query, key, cosine, sine)
        all_key = torch.cat((past_key.to(key.dtype), key), dim=2)
        all_value = torch.cat((past_value.to(value.dtype), value), dim=2)
        return (
            self.o_proj(
                self._grouped_attention(query, all_key, all_value, causal_mask)
            ),
            all_key.to(CACHE_STORAGE_DTYPE),
            all_value.to(CACHE_STORAGE_DTYPE),
        )

    def forward_full(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = self._project_qkv(hidden_states)
        query, key = self._apply_rotary(query, key, cosine, sine)
        return self.o_proj(
            self._grouped_attention(query, key, value, causal_mask)
        )


class ManualQwenMLP(nn.Module):
    def __init__(self, geometry: QwenGeometry) -> None:
        super().__init__()
        self.intermediate_size = geometry.intermediate_size
        self.gate_proj = nn.Linear(geometry.hidden_size, geometry.intermediate_size, bias=False)
        self.up_proj = nn.Linear(geometry.hidden_size, geometry.intermediate_size, bias=False)
        self.gate_up_proj: nn.Linear | None = None
        self.down_proj = nn.Linear(geometry.intermediate_size, geometry.hidden_size, bias=False)

    def prepare_for_export(self) -> None:
        if self.gate_up_proj is not None:
            return
        self.gate_up_proj = _fuse_parallel_linears(self.gate_proj, self.up_proj)
        self.gate_proj = None
        self.up_proj = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gate_up_proj is None:
            gate = self.gate_proj(hidden_states)
            up = self.up_proj(hidden_states)
        else:
            gate, up = self.gate_up_proj(hidden_states).split(
                (self.intermediate_size, self.intermediate_size), dim=-1
            )
        return self.down_proj(functional.silu(gate) * up)


class ManualQwenLayer(nn.Module):
    def __init__(self, geometry: QwenGeometry) -> None:
        super().__init__()
        self.input_layernorm = ManualRMSNorm(geometry.hidden_size, geometry.rms_eps)
        self.self_attn = ManualQwenAttention(geometry)
        self.post_attention_layernorm = ManualRMSNorm(geometry.hidden_size, geometry.rms_eps)
        self.mlp = ManualQwenMLP(geometry)

    def fold_for_export(self) -> None:
        if self.self_attn.qkv_proj is None or self.mlp.gate_up_proj is None:
            raise RuntimeError("Parallel projections must be fused before RMSNorm folding.")
        self.input_layernorm.fold_into(self.self_attn.qkv_proj)
        self.post_attention_layernorm.fold_into(self.mlp.gate_up_proj)

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        attended, key, value = self.self_attn.forward_cached(
            self.input_layernorm(hidden_states),
            past_key,
            past_value,
            cosine,
            sine,
            causal_mask,
        )
        hidden_states = residual + attended
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, key, value

    def forward_full(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn.forward_full(
            self.input_layernorm(hidden_states),
            cosine,
            sine,
            causal_mask,
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class ManualQwen(nn.Module):
    def __init__(self, geometry: QwenGeometry, *, include_embedding: bool) -> None:
        super().__init__()
        self.geometry = geometry
        self.embed_tokens = (
            nn.Embedding(geometry.vocab_size, geometry.hidden_size)
            if include_embedding and geometry.vocab_size is not None
            else None
        )
        self.layers = nn.ModuleList(ManualQwenLayer(geometry) for _ in range(geometry.layers))
        self.norm = ManualRMSNorm(geometry.hidden_size, geometry.rms_eps)
        rotary_cosine, rotary_sine = _build_rotary_tables(
            geometry.max_sequence_length,
            geometry.head_dim,
            geometry.rope_theta,
            interleaved=False,
        )
        rotation_signs = _build_rotary_signs(
            geometry.head_dim, interleaved=False
        )
        rotary_sine = rotary_sine * rotation_signs.reshape(1, 1, 1, -1)
        self.register_buffer(
            "rotary_cosine", rotary_cosine.to(torch.float16), persistent=False
        )
        self.register_buffer(
            "rotary_sine", rotary_sine.to(torch.float16), persistent=False
        )
        causal_mask = _build_causal_mask(
            geometry.max_sequence_length,
            geometry.sliding_window,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(token_ids)

    def forward_full(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        sequence_length = torch._shape_as_tensor(input_embeds)[1]
        cosine = self.rotary_cosine[:, :, :sequence_length].float()
        sine = self.rotary_sine[:, :, :sequence_length].float()
        causal_mask = (
            self.causal_mask[:, :, :, :sequence_length, :sequence_length]
            if self.geometry.sliding_window is None
            else self.causal_mask[:, :, :, :sequence_length]
        ).float()
        for layer in self.layers:
            hidden_states = layer.forward_full(
                hidden_states,
                cosine,
                sine,
                causal_mask,
            )
        return self.norm(hidden_states)

    def forward_cached(
        self,
        input_embeds: torch.Tensor,
        keys: tuple[torch.Tensor, ...],
        values: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if self.geometry.sliding_window is not None:
            raise RuntimeError("Cached sliding-window Qwen is not used by this exporter.")
        hidden_states = input_embeds
        sequence_length = torch._shape_as_tensor(input_embeds)[1]
        sequence_offset = torch._shape_as_tensor(keys[0])[2]
        cosine = self.rotary_cosine[
            :, :, sequence_offset : sequence_offset + sequence_length
        ].float()
        sine = self.rotary_sine[
            :, :, sequence_offset : sequence_offset + sequence_length
        ].float()
        key_length = sequence_offset + sequence_length
        causal_mask = self.causal_mask[
            :, :, :, sequence_offset:key_length, :key_length
        ].float()
        output_keys: list[torch.Tensor] = []
        output_values: list[torch.Tensor] = []
        for layer, key, value in zip(self.layers, keys, values):
            hidden_states, key, value = layer.forward_cached(
                hidden_states,
                key,
                value,
                cosine,
                sine,
                causal_mask,
            )
            output_keys.append(key)
            output_values.append(value)
        return self.norm(hidden_states), tuple(output_keys), tuple(output_values)

class ManualPatchAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.head_dim_half = self.head_dim // 2
        self.score_scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.to_qkv: nn.Linear | None = None
        self.to_out = nn.ModuleList((nn.Linear(hidden_size, hidden_size), nn.Dropout(0.0)))

    def prepare_for_export(self) -> None:
        if self.to_qkv is not None:
            return
        self.to_qkv = _fuse_parallel_linears(self.to_q, self.to_k, self.to_v)
        with torch.no_grad():
            self.to_qkv.weight[: self.hidden_size].mul_(self.score_scale)
            self.to_qkv.bias[: self.hidden_size].mul_(self.score_scale)
        self.score_scale = 1.0
        self.to_q = None
        self.to_k = None
        self.to_v = None

    def _rotate_half(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, head_count, sequence_length = tensor.shape[:3]
        tensor = tensor.view(
            batch_size,
            head_count,
            sequence_length,
            self.head_dim_half,
            2,
        ).flip(-1)
        return tensor.view(
            batch_size,
            head_count,
            sequence_length,
            self.head_dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape[:2]
        if self.to_qkv is None:
            qkv_projection = torch.cat(
                (
                    self.to_q(hidden_states),
                    self.to_k(hidden_states),
                    self.to_v(hidden_states),
                ),
                dim=-1,
            )
        else:
            qkv_projection = self.to_qkv(hidden_states)
        qkv = qkv_projection.view(
            batch_size,
            sequence_length,
            3 * self.heads,
            self.head_dim,
        ).transpose(1, 2)
        query_key, value = qkv.split((2 * self.heads, self.heads), dim=1)
        query_key = query_key * cosine + self._rotate_half(query_key) * sine
        query, key = query_key.split((self.heads, self.heads), dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2))
        if self.score_scale != 1.0:
            scores = scores * self.score_scale
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.to_out[1](self.to_out[0](output))


class ManualPatchFeedForward(nn.Module):
    def __init__(self, hidden_size: int, multiplier: int) -> None:
        super().__init__()
        inner_size = hidden_size * multiplier
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(hidden_size, inner_size), nn.GELU(approximate="tanh")),
            nn.Dropout(0.0),
            nn.Linear(inner_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(hidden_states)


class ManualPatchBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, multiplier: int) -> None:
        super().__init__()
        self.norm1 = ManualRMSNorm(hidden_size, PATCH_RMS_EPS)
        self.attn = ManualPatchAttention(hidden_size, heads)
        self.norm2 = ManualRMSNorm(hidden_size, PATCH_RMS_EPS)
        self.mlp = ManualPatchFeedForward(hidden_size, multiplier)

    def fold_for_export(self) -> None:
        if self.attn.to_qkv is None:
            raise RuntimeError("Patch QKV projection must be fused before RMSNorm folding.")
        self.norm1.fold_into(self.attn.to_qkv)
        self.norm2.fold_into(self.mlp.ff[0][0])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cosine, sine
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class ManualPatchFinal(nn.Module):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.norm_final = ManualRMSNorm(hidden_size, PATCH_RMS_EPS)
        self.linear = nn.Linear(hidden_size, out_size)

    def fold_for_export(self) -> None:
        self.norm_final.fold_into(self.linear)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(hidden_states))


class ManualPatchEncoder(nn.Module):
    def __init__(self, geometry: PackageGeometry) -> None:
        super().__init__()
        self.patch_size = geometry.patch_size
        self.hidden_size = geometry.patch_hidden_size
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, self.hidden_size, dtype=torch.float32))
        self.in_proj = nn.Linear(geometry.redae_dim, self.hidden_size)
        self.blocks = nn.ModuleList(
            ManualPatchBlock(self.hidden_size, geometry.patch_heads, geometry.patch_mlp_ratio)
            for _ in range(geometry.patch_depth)
        )
        self.out_proj = ManualPatchFinal(self.hidden_size, geometry.backbone_qwen.hidden_size)
        rotary_cosine, rotary_sine = _build_rotary_tables(
            self.patch_size + 1,
            self.hidden_size // geometry.patch_heads,
            10_000.0,
            interleaved=True,
        )
        rotation_signs = _build_rotary_signs(
            self.hidden_size // geometry.patch_heads, interleaved=True
        )
        rotary_sine = rotary_sine * rotation_signs.reshape(1, 1, 1, -1)
        self.register_buffer("rotary_cosine", rotary_cosine, persistent=False)
        self.register_buffer("rotary_sine", rotary_sine, persistent=False)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = inputs_embeds.shape[:2]
        projected = self.in_proj(inputs_embeds)
        hidden_states = projected.reshape(-1, self.patch_size, self.hidden_size)
        cls_token = self.cls_tok.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat((cls_token, hidden_states), dim=1)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                self.rotary_cosine,
                self.rotary_sine,
            )
        hidden_states = self.out_proj(hidden_states[:, 0])
        return hidden_states.reshape(batch_size, sequence_length // self.patch_size, -1)


def _modulate(hidden_states: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return hidden_states * (1.0 + scale) + shift


class ManualFlowConv(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states.transpose(1, 2)).transpose(1, 2)


class ManualTimeEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = frequency_size
        self._time_table_ready = False
        self.time_mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        half_size = self.frequency_size // 2
        frequencies = torch.arange(half_size, dtype=torch.float32)
        frequencies = torch.exp(
            frequencies * (-math.log(10_000.0) / (half_size - 1))
        )
        time_values = torch.linspace(
            0.0, 1.0, FLOW_STEPS + 1, dtype=torch.float32
        )
        time_values = 1.0 - torch.cos(time_values * (0.5 * torch.pi))
        scheduled = 1000.0 * time_values[:-1].unsqueeze(1) * frequencies.unsqueeze(0)
        scheduled = torch.cat((scheduled.sin(), scheduled.cos()), dim=-1)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.register_buffer("scheduled", scheduled, persistent=False)
        self.register_buffer(
            "time_mlp_results",
            torch.empty(FLOW_STEPS, hidden_size, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "time_deltas", time_values[1:] - time_values[:-1], persistent=False
        )

    def prepare_for_export(self) -> None:
        if self._time_table_ready:
            return
        with torch.no_grad():
            self.time_mlp_results.copy_(self.time_mlp(self.scheduled))
        self._time_table_ready = True

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if not timestep.is_floating_point():
            if not self._time_table_ready:
                raise RuntimeError("Time MLP results have not been prepared.")
            return torch.index_select(
                self.time_mlp_results,
                0,
                timestep.reshape(-1).to(torch.int64),
            )
        embedding = (
            1000.0
            * timestep.reshape(-1, 1)
            * self.frequencies.unsqueeze(0)
        )
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return self.time_mlp(embedding)


class ManualFlowBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, multiplier: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = ManualRMSNorm(hidden_size, PATCH_RMS_EPS)
        self.attn = ManualPatchAttention(hidden_size, heads)
        self.norm2 = ManualRMSNorm(hidden_size, PATCH_RMS_EPS)
        self.conv = ManualFlowConv(hidden_size)
        self.norm3 = ManualRMSNorm(hidden_size, PATCH_RMS_EPS)
        self.mlp = ManualPatchFeedForward(hidden_size, multiplier)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size))

    def fold_for_export(self) -> None:
        linear = self.adaLN_modulation[1]
        hidden_size = linear.in_features
        for norm, chunk_index in (
            (self.norm1, 1),
            (self.norm3, 4),
            (self.norm2, 7),
        ):
            weight = norm.take_weight()
            if weight is None:
                continue
            start = chunk_index * hidden_size
            end = start + hidden_size
            with torch.no_grad():
                linear.weight[start:end].mul_(weight.unsqueeze(1))
                linear.bias[start:end].mul_(weight).add_(weight - 1.0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = self.adaLN_modulation(condition).split(
            (self.hidden_size,) * 9, dim=-1
        )
        hidden_states = hidden_states + gate_msa * self.attn(
            _modulate(self.norm1(hidden_states), shift_msa, scale_msa),
            cosine,
            sine,
        )
        hidden_states = hidden_states + gate_conv * self.conv(
            _modulate(self.norm2(hidden_states), shift_conv, scale_conv)
        )
        return hidden_states + gate_mlp * self.mlp(
            _modulate(self.norm3(hidden_states), shift_mlp, scale_mlp)
        )


class ManualFlowFinal(nn.Module):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1.0e-6)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, hidden_states: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).split(
            (self.hidden_size, self.hidden_size), dim=-1
        )
        return self.linear(_modulate(self.norm_final(hidden_states), shift, scale))


class ManualFlowDiT(nn.Module):
    def __init__(self, geometry: PackageGeometry, input_channels: int) -> None:
        super().__init__()
        self.trajectory_channels = geometry.redae_dim
        self.condition_channels = geometry.dit_hidden_size
        self.speaker_channels = (
            input_channels - self.trajectory_channels - self.condition_channels
        )
        self.in_proj = nn.Linear(input_channels, geometry.dit_hidden_size)
        self.trajectory_proj: nn.Linear | None = None
        self.condition_proj: nn.Linear | None = None
        self.speaker_proj: nn.Linear | None = None
        self.t_embedder = ManualTimeEmbedder(geometry.dit_hidden_size)
        self.blocks = nn.ModuleList(
            ManualFlowBlock(geometry.dit_hidden_size, geometry.dit_heads, geometry.dit_mlp_ratio)
            for _ in range(geometry.dit_depth)
        )
        self.final_layer = ManualFlowFinal(geometry.dit_hidden_size, geometry.redae_dim)
        rotary_cosine, rotary_sine = _build_rotary_tables(
            geometry.history_length + geometry.patch_size,
            geometry.dit_hidden_size // geometry.dit_heads,
            10_000.0,
            interleaved=True,
        )
        rotation_signs = _build_rotary_signs(
            geometry.dit_hidden_size // geometry.dit_heads, interleaved=True
        )
        rotary_sine = rotary_sine * rotation_signs.reshape(1, 1, 1, -1)
        self.register_buffer("rotary_cosine", rotary_cosine, persistent=False)
        self.register_buffer("rotary_sine", rotary_sine, persistent=False)

    def prepare_for_export(self) -> None:
        if self.trajectory_proj is not None:
            return
        linear = self.in_proj
        hidden_size = linear.out_features
        reference = linear.weight
        self.trajectory_proj = nn.Linear(
            self.trajectory_channels,
            hidden_size,
            bias=True,
            device=reference.device,
            dtype=reference.dtype,
        ).eval()
        self.condition_proj = nn.Linear(
            self.condition_channels,
            hidden_size,
            bias=False,
            device=reference.device,
            dtype=reference.dtype,
        ).eval()
        if self.speaker_channels:
            self.speaker_proj = nn.Linear(
                self.speaker_channels,
                hidden_size,
                bias=False,
                device=reference.device,
                dtype=reference.dtype,
            ).eval()
        with torch.no_grad():
            trajectory_end = self.trajectory_channels
            condition_end = trajectory_end + self.condition_channels
            self.trajectory_proj.weight.copy_(linear.weight[:, :trajectory_end])
            self.trajectory_proj.bias.copy_(linear.bias)
            self.condition_proj.weight.copy_(
                linear.weight[:, trajectory_end:condition_end]
            )
            if self.speaker_proj is not None:
                self.speaker_proj.weight.copy_(linear.weight[:, condition_end:])
        self.in_proj = None

    def project_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if self.trajectory_proj is None:
            raise RuntimeError("DiT trajectory projection has not been prepared.")
        return self.trajectory_proj(trajectory)

    def project_condition(self, condition: torch.Tensor) -> torch.Tensor:
        if self.condition_proj is None:
            raise RuntimeError("DiT condition projection has not been prepared.")
        return self.condition_proj(condition)

    def project_speaker(self, speaker: torch.Tensor) -> torch.Tensor:
        if self.speaker_proj is None:
            raise RuntimeError("This DiT has no prepared speaker projection.")
        return self.speaker_proj(speaker)

    def forward_hidden(
        self, hidden_states: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        condition = self.t_embedder(timestep.reshape(-1)).unsqueeze(1)
        return self.forward_conditioned(hidden_states, condition)

    def forward_conditioned(
        self, hidden_states: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                condition,
                self.rotary_cosine,
                self.rotary_sine,
            )
        return self.final_layer(hidden_states, condition)

    def forward(self, inputs: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        if self.trajectory_proj is None:
            hidden_states = self.in_proj(inputs)
        else:
            sections = (
                self.trajectory_channels,
                self.condition_channels,
                self.speaker_channels,
            )
            trajectory, condition, *speaker = inputs.split(
                tuple(size for size in sections if size), dim=-1
            )
            hidden_states = self.project_trajectory(trajectory)
            hidden_states = hidden_states + self.project_condition(condition)
            if speaker:
                hidden_states = hidden_states + self.project_speaker(speaker[0])
        return self.forward_hidden(hidden_states, timestep)


class ManualQwenClsDownsample(nn.Module):
    def __init__(self, geometry: PackageGeometry) -> None:
        super().__init__()
        self.downsample_rate = int(
            geometry.redae_downsample_rate // geometry.audio_patch_size
        )
        self.hidden_size = geometry.encoder_qwen.hidden_size
        self.cls_tok = nn.Parameter(torch.ones(1, 1, self.hidden_size, dtype=torch.float32))
        self.qwen3 = ManualQwen(geometry.downsample_qwen, include_embedding=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.downsample_rate, self.hidden_size)
        cls_token = self.cls_tok.expand(hidden_states.shape[0], -1, -1)
        hidden_states = self.qwen3.forward_full(torch.cat((hidden_states, cls_token), dim=1))
        hidden_states = hidden_states[:, -1]
        return hidden_states.reshape(batch_size, -1, self.hidden_size)


class ManualRedAEEncoder(nn.Module):
    def __init__(self, geometry: PackageGeometry) -> None:
        super().__init__()
        hidden_size = geometry.encoder_qwen.hidden_size
        self.audio_patch_size = geometry.audio_patch_size
        self.downsample_rate = geometry.redae_downsample_rate
        self.in_proj = nn.Sequential(
            nn.Linear(self.audio_patch_size, hidden_size), nn.Linear(hidden_size, hidden_size)
        )
        self.qwen3 = ManualQwen(geometry.encoder_qwen, include_embedding=False)
        self.downsample = ManualQwenClsDownsample(geometry)
        self.out_proj = nn.Linear(hidden_size, geometry.redae_dim)

    def fold_for_export(self) -> None:
        if isinstance(self.in_proj, nn.Sequential):
            self.in_proj = _compose_linears(self.in_proj[0], self.in_proj[1])
        self.downsample.qwen3.norm.fold_into(self.out_proj)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        hidden_states = audio.reshape(audio.shape[0], -1, self.audio_patch_size)
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.qwen3.forward_full(hidden_states)
        hidden_states = self.downsample(hidden_states)
        return self.out_proj(hidden_states)


class ManualSameISTFT(nn.Module):
    """Real/imag overlap-add ISTFT equivalent to RedAE's same-padding path."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad = (win_length - hop_length) // 2
        window = torch.hann_window(win_length, dtype=torch.float32)
        self.register_buffer("window", window)

        frequency_bins = n_fft // 2 + 1
        frequencies = torch.arange(frequency_bins, dtype=torch.float32).unsqueeze(1)
        samples = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        angles = (2.0 * torch.pi / n_fft) * frequencies * samples
        scale = torch.full((frequency_bins, 1), 2.0, dtype=torch.float32)
        scale[0] = 1.0
        if n_fft % 2 == 0:
            scale[-1] = 1.0
        inverse_real = (scale * torch.cos(angles) / n_fft) * window.unsqueeze(0)
        inverse_imag = (scale * -torch.sin(angles) / n_fft) * window.unsqueeze(0)
        self.register_buffer(
            "inverse_kernel",
            torch.cat((inverse_real, inverse_imag), dim=0).unsqueeze(1),
            persistent=False,
        )
        self.register_buffer(
            "window_square_kernel", window.square().reshape(1, 1, -1), persistent=False
        )

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        spectrum = torch.cat((real, imag), dim=1)
        reconstructed = functional.conv_transpose1d(
            spectrum, self.inverse_kernel, stride=self.hop_length
        )
        frame_count = torch._shape_as_tensor(real)[2]
        envelope = functional.conv_transpose1d(
            torch.ones((1, 1, frame_count), dtype=real.dtype, device=real.device),
            self.window_square_kernel,
            stride=self.hop_length,
        )
        return (
            reconstructed[..., self.pad : -self.pad] / envelope[..., self.pad : -self.pad]
        ).squeeze(1)


class ManualISTFTHead(nn.Module):
    def __init__(self, hidden_size: int, audio_patch_size: int) -> None:
        super().__init__()
        self.out = nn.Linear(hidden_size, audio_patch_size * 4 + 2)
        self.frequency_bins = audio_patch_size * 2 + 1
        self.istft = ManualSameISTFT(
            n_fft=audio_patch_size * 4,
            hop_length=audio_patch_size,
            win_length=audio_patch_size * 4,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        prediction = self.out(hidden_states).transpose(1, 2)
        magnitude, phase = prediction.split(
            (self.frequency_bins, self.frequency_bins), dim=1
        )
        magnitude = torch.clip(torch.exp(magnitude), max=100.0)
        return self.istft(magnitude * torch.cos(phase), magnitude * torch.sin(phase))


class ManualRedAEDecoder(nn.Module):
    def __init__(self, geometry: PackageGeometry) -> None:
        super().__init__()
        hidden_size = geometry.decoder_qwen.hidden_size
        upsample_rate = geometry.redae_downsample_rate // geometry.audio_patch_size
        self.hidden_size = hidden_size
        self.upsample_rate = upsample_rate
        self.in_proj = nn.Linear(geometry.redae_dim, upsample_rate * hidden_size)
        self.qwen3 = ManualQwen(geometry.decoder_qwen, include_embedding=False)
        self.istft_head = ManualISTFTHead(hidden_size, geometry.audio_patch_size)

    def fold_for_export(self) -> None:
        self.qwen3.norm.fold_into(self.istft_head.out)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden_states = self.in_proj(latents).reshape(latents.shape[0], -1, self.hidden_size)
        return self.istft_head(self.qwen3.forward_full(hidden_states))


class ManualRedAE(nn.Module):
    def __init__(self, geometry: PackageGeometry) -> None:
        super().__init__()
        self.encoder = ManualRedAEEncoder(geometry)
        self.decoder = ManualRedAEDecoder(geometry)


def prepare_module_for_export(module: nn.Module) -> None:
    children = tuple(module.modules())
    for child in children:
        prepare = getattr(child, "prepare_for_export", None)
        if prepare is not None:
            prepare()
    for child in children:
        fold = getattr(child, "fold_for_export", None)
        if fold is not None:
            fold()
    for child in children:
        child._non_persistent_buffers_set.clear()


def load_redae(geometry: PackageGeometry, checkpoint: Path) -> ManualRedAE:
    redae = ManualRedAE(geometry).eval()
    load_module_from_safetensors(redae.encoder, checkpoint, "encoder.")
    load_module_from_safetensors(redae.decoder, checkpoint, "decoder.")
    prepare_module_for_export(redae)
    return redae


def _nonlinear(config: str, channels: int) -> nn.Sequential:
    layers: OrderedDict[str, nn.Module] = OrderedDict()
    for name in config.split("-"):
        if name == "relu":
            layers["relu"] = nn.ReLU(inplace=True)
        elif name == "prelu":
            layers["prelu"] = nn.PReLU(channels)
        elif name == "batchnorm":
            layers["batchnorm"] = nn.BatchNorm1d(channels)
        elif name == "batchnorm_":
            layers["batchnorm"] = nn.BatchNorm1d(channels, affine=False)
    return nn.Sequential(layers)


class ManualBasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(output_channels),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = functional.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return functional.relu(output + self.shortcut(inputs))


class ManualFCM(nn.Module):
    def __init__(self, feature_dim: int = 80, channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer1 = nn.Sequential(
            ManualBasicResBlock(channels, channels, stride=2),
            ManualBasicResBlock(channels, channels),
        )
        self.layer2 = nn.Sequential(
            ManualBasicResBlock(channels, channels, stride=2),
            ManualBasicResBlock(channels, channels),
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.out_channels = channels * (feature_dim // 8)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = functional.relu(self.bn1(self.conv1(inputs.unsqueeze(1))))
        output = self.layer2(self.layer1(output))
        output = functional.relu(self.bn2(self.conv2(output)))
        return output.reshape(output.shape[0], output.shape[1] * output.shape[2], output.shape[3])


class ManualTDNNLayer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        padding: int = -1,
        bias: bool = False,
        config: str = "batchnorm-relu",
    ) -> None:
        super().__init__()
        if padding < 0:
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = _nonlinear(config, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.nonlinear(self.linear(inputs))


class ManualCAMLayer(nn.Module):
    def __init__(
        self,
        bottleneck_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        reduction: int = 2,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.linear_local = nn.Conv1d(
            bottleneck_channels,
            output_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.linear1 = nn.Conv1d(bottleneck_channels, bottleneck_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bottleneck_channels // reduction, output_channels, 1)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _segment_pool(inputs: torch.Tensor, segment_length: int = 100) -> torch.Tensor:
        return functional.avg_pool1d(
            inputs, kernel_size=segment_length, stride=segment_length, ceil_mode=True
        )

    @staticmethod
    def _expand_segments(
        values: torch.Tensor, output_length: int, segment_length: int = 100
    ) -> torch.Tensor:
        shape = values.shape
        expanded = values.unsqueeze(-1).expand(*shape, segment_length)
        expanded = expanded.reshape(*shape[:-1], -1)
        return expanded[..., :output_length]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local = self.linear_local(inputs)
        context = inputs.mean(dim=-1, keepdim=True) + self._segment_pool(inputs)
        gate = self.sigmoid(self.linear2(self.relu(self.linear1(context))))
        gate = self._expand_segments(gate, inputs.shape[-1])
        return local * gate


class ManualCAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.nonlinear1 = _nonlinear("batchnorm-relu", input_channels)
        self.linear1 = nn.Conv1d(input_channels, bottleneck_channels, 1, bias=False)
        self.nonlinear2 = _nonlinear("batchnorm-relu", bottleneck_channels)
        self.cam_layer = ManualCAMLayer(
            bottleneck_channels, output_channels, kernel_size, dilation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.cam_layer(self.nonlinear2(self.linear1(self.nonlinear1(inputs))))


class ManualCAMDenseTDNNBlock(nn.ModuleList):
    def __init__(
        self,
        layer_count: int,
        input_channels: int,
        output_channels: int,
        bottleneck_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        for index in range(layer_count):
            self.add_module(
                f"tdnnd{index + 1}",
                ManualCAMDenseTDNNLayer(
                    input_channels + index * output_channels,
                    output_channels,
                    bottleneck_channels,
                    kernel_size,
                    dilation,
                ),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer in self:
            output = torch.cat((output, layer(output)), dim=1)
        return output


class ManualTransitLayer(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.nonlinear = _nonlinear("batchnorm-relu", input_channels)
        self.linear = nn.Conv1d(input_channels, output_channels, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.nonlinear(inputs))


class ManualStatsPool(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat((inputs.mean(dim=-1), inputs.std(dim=-1, unbiased=True)), dim=-1)


class ManualDenseLayer(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.linear = nn.Conv1d(input_channels, output_channels, 1, bias=False)
        self.nonlinear = _nonlinear("batchnorm_", output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2:
            inputs = self.linear(inputs.unsqueeze(-1)).squeeze(-1)
        else:
            inputs = self.linear(inputs)
        return self.nonlinear(inputs)


class ManualCAMPPlus(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = ManualFCM(feature_dim=80)
        channels = self.head.out_channels
        xvector: OrderedDict[str, nn.Module] = OrderedDict()
        xvector["tdnn"] = ManualTDNNLayer(channels, 128, 5, stride=2)
        channels = 128
        for index, (layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2)), start=1
        ):
            xvector[f"block{index}"] = ManualCAMDenseTDNNBlock(
                layers, channels, 32, 128, kernel_size, dilation
            )
            channels += layers * 32
            xvector[f"transit{index}"] = ManualTransitLayer(channels, channels // 2)
            channels //= 2
        xvector["out_nonlinear"] = _nonlinear("batchnorm-relu", channels)
        xvector["stats"] = ManualStatsPool()
        xvector["dense"] = ManualDenseLayer(channels * 2, 512)
        self.xvector = nn.Sequential(xvector)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.xvector(self.head(features.permute(0, 2, 1)))


def load_campp(checkpoint: Path) -> ManualCAMPPlus:
    model = ManualCAMPPlus().eval()
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model


class ManualTTSCore(nn.Module):
    def __init__(self, geometry: PackageGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        self.backbone = ManualQwen(geometry.backbone_qwen, include_embedding=True)
        self.patch_encoder = ManualPatchEncoder(geometry)
        self.dit_head = nn.Linear(geometry.backbone_qwen.hidden_size, geometry.dit_hidden_size)
        input_channels = geometry.redae_dim + geometry.dit_hidden_size
        if geometry.variant == "base":
            self.spk_proj_llm = nn.Linear(geometry.speaker_dim, geometry.backbone_qwen.hidden_size)
            self.spk_proj_dit = nn.Linear(geometry.speaker_dim, geometry.speaker_dim)
            input_channels += geometry.speaker_dim
        else:
            self.spk_proj_llm = None
            self.spk_proj_dit = None
        self.dit = ManualFlowDiT(geometry, input_channels)
        self.stop_head = nn.Linear(geometry.backbone_qwen.hidden_size, 1)


def load_tts_core(geometry: PackageGeometry, checkpoint: Path) -> ManualTTSCore:
    core = ManualTTSCore(geometry).eval()
    backbone_prefix = "backbone_llm." if geometry.variant == "base" else "backbone_llm.model."
    load_module_from_safetensors(core.backbone, checkpoint, backbone_prefix)
    load_module_from_safetensors(core.patch_encoder, checkpoint, "patch_encoder.")
    load_module_from_safetensors(core.dit_head, checkpoint, "dit_head.")
    load_module_from_safetensors(core.dit, checkpoint, "dit.")
    load_module_from_safetensors(core.stop_head, checkpoint, "stop_head.")
    if geometry.variant == "base":
        load_module_from_safetensors(core.spk_proj_llm, checkpoint, "spk_proj_llm.")
        load_module_from_safetensors(core.spk_proj_dit, checkpoint, "spk_proj_dit.")
    prepare_module_for_export(core)
    return core


def _right_aligned_history(values: torch.Tensor, width: int) -> torch.Tensor:
    zeros = values.new_zeros((values.shape[0], width, values.shape[-1]))
    return torch.cat((zeros, values), dim=1)[:, -width:]


def _append_history(history: torch.Tensor, values: torch.Tensor, width: int) -> torch.Tensor:
    retained = history[:, values.shape[1] : width]
    return torch.cat((retained, values), dim=1)


def _empty_cache(geometry: QwenGeometry, reference: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    key = reference.new_zeros((reference.shape[0], geometry.key_value_heads, 0, geometry.head_dim))
    value = torch.zeros_like(key)
    return (
        tuple(key for _ in range(geometry.layers)),
        tuple(value for _ in range(geometry.layers)),
    )


def _split_cache(
    cache_tensors: tuple[torch.Tensor, ...], layer_count: int
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    return cache_tensors[:layer_count], cache_tensors[layer_count:]


class ManualFlowRunner(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.core = core
        self.core.dit.t_embedder.prepare_for_export()
        self.patch_size = core.geometry.patch_size
        self.history_length = core.geometry.history_length
        self.condition_projection = (
            _compose_linears(core.dit_head, core.dit.condition_proj)
            if core.dit.condition_proj is not None
            else None
        )
        self.register_buffer(
            "timestep_indices", torch.arange(FLOW_STEPS, dtype=torch.int64)
        )
        self.register_buffer(
            "time_deltas", core.dit.t_embedder.time_deltas.detach().clone()
        )

    def forward(
        self,
        latent_history: torch.Tensor,
        condition_history: torch.Tensor,
        flow_noise: torch.Tensor,
        cfg: torch.Tensor,
        speaker_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.core.dit.trajectory_proj is not None:
            condition_hidden = self.condition_projection(condition_history)
            condition_hidden = condition_hidden.repeat_interleave(
                self.patch_size, dim=1
            )
            if speaker_condition is not None:
                condition_hidden = condition_hidden + self.core.dit.project_speaker(
                    speaker_condition
                ).unsqueeze(1)
        else:
            projected_condition = self.core.dit_head(condition_history)
            repeated_condition = projected_condition.repeat_interleave(
                self.patch_size, dim=1
            )
            if speaker_condition is not None:
                speaker = speaker_condition.unsqueeze(1).expand(
                    -1, self.history_length + self.patch_size, -1
                )
                conditioning = torch.cat((repeated_condition, speaker), dim=-1)
            else:
                conditioning = repeated_condition
        current = flow_noise
        cfg_value = cfg.reshape(1, 1, 1)
        for index in range(FLOW_STEPS):
            timestep = self.timestep_indices[index : index + 1]
            time_condition = self.core.dit.t_embedder(timestep).unsqueeze(1)
            time_condition = torch.cat((time_condition, time_condition), dim=0)
            dt = self.time_deltas[index]
            trajectory = torch.cat((latent_history, current), dim=1)
            if self.core.dit.trajectory_proj is not None:
                trajectory_hidden = self.core.dit.project_trajectory(trajectory)
                velocity = self.core.dit.forward_conditioned(
                    torch.cat(
                        (
                            trajectory_hidden + condition_hidden,
                            trajectory_hidden,
                        ),
                        dim=0,
                    ),
                    time_condition,
                )
            else:
                conditioned = torch.cat((trajectory, conditioning), dim=-1)
                unconditioned = torch.cat(
                    (trajectory, torch.zeros_like(conditioning)), dim=-1
                )
                velocity = self.core.dit.forward_conditioned(
                    self.core.dit.in_proj(
                        torch.cat((conditioned, unconditioned), dim=0)
                    ),
                    time_condition,
                )
            velocity_conditioned, velocity_unconditioned = velocity.split((1, 1), dim=0)
            velocity = velocity_conditioned + cfg_value * (velocity_conditioned - velocity_unconditioned)
            current = current + dt * velocity[:, -self.patch_size :]
        return current


class TextTokenSelector(nn.Module):
    """Select one text token with graph-owned greedy or top-k/top-p sampling."""

    def forward(
        self,
        logits: torch.Tensor,
        previous_ids: torch.Tensor,
        do_sample: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        repetition_penalty: torch.Tensor,
    ) -> torch.Tensor:
        previous_scores = torch.gather(logits, 1, previous_ids)
        adjusted_scores = torch.where(
            previous_scores < 0.0,
            previous_scores * repetition_penalty,
            previous_scores / repetition_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids, adjusted_scores)
        greedy_token = torch.argmax(scores, dim=-1, keepdim=True)

        sorted_scores, sorted_indices = torch.topk(
            scores, k=top_k, dim=-1, largest=True, sorted=True
        )
        probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        keep = (cumulative - probabilities) <= top_p
        kept_mass = torch.where(keep, cumulative, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax(
            (cumulative >= threshold).to(torch.int32), dim=-1, keepdim=True
        )
        sampled_token = torch.gather(sorted_indices, 1, winner)
        return torch.where(do_sample.reshape(1, 1), sampled_token, greedy_token)


class WaveformAligner(nn.Module):
    def __init__(self, alignment: int) -> None:
        super().__init__()
        self.alignment = alignment
        self.register_buffer(
            "left_padding",
            torch.zeros(1, self.alignment - 1, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, prompt_audio: torch.Tensor) -> torch.Tensor:
        audio_length = torch._shape_as_tensor(prompt_audio)[1]
        padding = torch.remainder(-audio_length, self.alignment)
        start = self.alignment - 1 - padding
        return torch.cat(
            (self.left_padding.expand(prompt_audio.shape[0], -1), prompt_audio),
            dim=1,
        )[:, start:]


class WaveformRateConverter(nn.Module):
    def __init__(self, input_sample_rate: int, output_sample_rate: int) -> None:
        super().__init__()
        if input_sample_rate <= 0 or output_sample_rate <= 0:
            raise ValueError(
                f"Audio sample rates must be positive, got "
                f"{input_sample_rate} -> {output_sample_rate}."
            )
        self.scale = float(output_sample_rate / input_sample_rate)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.scale == 1.0:
            return waveform
        return functional.interpolate(
            waveform.unsqueeze(1),
            scale_factor=self.scale,
            mode="linear",
            align_corners=False,
            recompute_scale_factor=False,
        ).squeeze(1)


class KaldiFbankFrontend(nn.Module):
    """Source-compatible 16 kHz Kaldi fbank implemented with ONNX tensor ops."""

    def __init__(self, input_sample_rate: int) -> None:
        super().__init__()
        self.resampler = torchaudio.transforms.Resample(
            input_sample_rate, 16000, dtype=torch.float32
        )
        frame_size = 400
        padded_frame_size = 512
        frequency_bins = padded_frame_size // 2 + 1
        self.padded_frame_size = padded_frame_size
        self.frequency_bins = frequency_bins
        self.stft = STFT_Process(
            "stft_B",
            n_fft=padded_frame_size,
            win_length=padded_frame_size,
            hop_len=padded_frame_size,
            max_frames=1,
            window_type="rectangular",
            center_pad=False,
        )
        frame_kernel = torch.eye(frame_size, dtype=torch.float32).reshape(
            frame_size, 1, frame_size
        )
        window = torch.hann_window(frame_size, periodic=False).pow(0.85)
        mel_filters, _ = get_mel_banks(
            80,
            padded_frame_size,
            16000.0,
            20.0,
            0.0,
            100.0,
            -500.0,
            1.0,
        )
        mel_filters = functional.pad(mel_filters, (0, 1)).transpose(0, 1)
        self.register_buffer("frame_kernel", frame_kernel, persistent=False)
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("mel_filters", mel_filters, persistent=False)
        self.register_buffer(
            "epsilon", torch.tensor(torch.finfo(torch.float32).eps), persistent=False
        )

    def forward(self, prompt_audio: torch.Tensor) -> torch.Tensor:
        waveform = self.resampler(prompt_audio)
        frames = functional.conv1d(
            waveform.unsqueeze(1), self.frame_kernel, stride=160
        ).transpose(1, 2)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        previous = torch.cat((frames[:, :, :1], frames[:, :, :-1]), dim=-1)
        frames = (frames - 0.97 * previous) * self.window
        frames = functional.pad(frames, (0, 112))
        real, imaginary = self.stft(
            frames.reshape(-1, 1, self.padded_frame_size)
        )
        power = (real.square() + imaginary.square()).reshape(
            frames.shape[0], frames.shape[1], self.frequency_bins
        )
        features = torch.maximum(power @ self.mel_filters, self.epsilon).log()
        return features - features.mean(dim=1, keepdim=True)


class RedAEEncodeGraph(nn.Module):
    def __init__(self, redae: ManualRedAE, geometry: PackageGeometry) -> None:
        super().__init__()
        self.encoder = redae.encoder
        self.input_rate_converter = WaveformRateConverter(
            IN_SAMPLE_RATE, geometry.out_sample_rate
        )
        self.aligner = WaveformAligner(
            geometry.redae_downsample_rate * geometry.patch_size
        )
        self.latent_scale = 1.0 if geometry.variant == "base" else REDAE_SCALE

    def forward(self, prompt_audio: torch.Tensor) -> torch.Tensor:
        prompt_audio = self.input_rate_converter(prompt_audio)
        return self.encoder(self.aligner(prompt_audio)) * self.latent_scale


class RedAEDecodeGraph(nn.Module):
    def __init__(self, redae: ManualRedAE, geometry: PackageGeometry) -> None:
        super().__init__()
        self.decoder = redae.decoder
        self.latent_scale = 1.0 if geometry.variant == "base" else REDAE_SCALE
        self.latent_scale_inv = 1.0 / self.latent_scale
        self.downsample_rate = geometry.redae_downsample_rate
        self.output_rate_converter = WaveformRateConverter(
            geometry.out_sample_rate, OUT_SAMPLE_RATE
        )

    def forward(
        self,
        generated_latents: torch.Tensor,
        prefix_latents: torch.Tensor,
    ) -> torch.Tensor:
        latents = torch.cat((prefix_latents, generated_latents), dim=1)
        waveform = self.decoder(latents * self.latent_scale_inv)
        prefix_samples = torch._shape_as_tensor(prefix_latents)[1] * self.downsample_rate
        waveform = waveform[:, prefix_samples:]
        return self.output_rate_converter(waveform)


class BaseReferencePreprocessGraph(nn.Module):
    def __init__(
        self,
        redae: ManualRedAE,
        campp: ManualCAMPPlus,
        geometry: PackageGeometry,
    ) -> None:
        super().__init__()
        self.encoder = redae.encoder
        self.campp = campp
        self.input_rate_converter = WaveformRateConverter(
            IN_SAMPLE_RATE, geometry.out_sample_rate
        )
        self.aligner = WaveformAligner(
            geometry.redae_downsample_rate * geometry.patch_size
        )
        self.fbank = KaldiFbankFrontend(geometry.out_sample_rate)

    def forward(self, prompt_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_audio = self.input_rate_converter(prompt_audio)
        prompt_audio = self.aligner(prompt_audio)
        return self.encoder(prompt_audio), self.campp(self.fbank(prompt_audio))


class BaseInputPrefillGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        text_ids: torch.Tensor,
        prompt_latents: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        text_embeddings = self.core.backbone.embed(text_ids)
        prompt_embeddings = self.core.patch_encoder(prompt_latents)
        speaker_llm = self.core.spk_proj_llm(speaker_embedding).unsqueeze(1)
        input_embeddings = torch.cat((speaker_llm, text_embeddings, prompt_embeddings), dim=1)
        keys, values = _empty_cache(self.core.geometry.backbone_qwen, input_embeddings)
        hidden_states, keys, values = self.core.backbone.forward_cached(input_embeddings, keys, values)
        prompt_hidden = hidden_states[:, -prompt_embeddings.shape[1] :]
        condition_history = _right_aligned_history(
            prompt_hidden, self.core.geometry.history_patches + 1
        )
        latent_history = _right_aligned_history(
            prompt_latents, self.core.geometry.history_length
        )
        stop_logits = self.core.stop_head(hidden_states[:, -1])
        speaker_condition = self.core.spk_proj_dit(speaker_embedding)
        return (
            *keys,
            *values,
            hidden_states[:, -1:],
            stop_logits,
            condition_history,
            latent_history,
            speaker_condition,
        )


class BaseAudioStartGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.flow = ManualFlowRunner(core)
        self.patch_size = core.geometry.patch_size
        self.history_length = core.geometry.history_length

    def forward(
        self,
        latent_history: torch.Tensor,
        condition_history: torch.Tensor,
        speaker_condition: torch.Tensor,
        cfg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flow_noise = torch.randn_like(latent_history[:, : self.patch_size])
        patch = self.flow(latent_history, condition_history, flow_noise, cfg, speaker_condition)
        return (
            patch,
            _append_history(latent_history, patch, self.history_length),
            patch,
        )


class BaseDecodeCoreGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        prior_patch: torch.Tensor,
        stop_threshold: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        keys, values = _split_cache(cache_tensors, self.core.geometry.backbone_qwen.layers)
        input_embeddings = self.core.patch_encoder(prior_patch)
        hidden_states, keys, values = self.core.backbone.forward_cached(input_embeddings, keys, values)
        stop_score = torch.sigmoid(self.core.stop_head(hidden_states[:, -1]))
        return (
            *keys,
            *values,
            hidden_states[:, -1:],
            stop_score,
            stop_score >= stop_threshold,
        )


class BaseFlowPatchGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.flow = ManualFlowRunner(core)
        self.history_length = core.geometry.history_length
        self.condition_width = core.geometry.history_patches + 1

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        latent_history: torch.Tensor,
        condition_history: torch.Tensor,
        speaker_condition: torch.Tensor,
        flow_noise: torch.Tensor,
        cfg: torch.Tensor,
        generated_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_history = _append_history(
            condition_history, last_hidden_state, self.condition_width
        )
        patch = self.flow(latent_history, condition_history, flow_noise, cfg, speaker_condition)
        return (
            patch,
            _append_history(latent_history, patch, self.history_length),
            condition_history,
            torch.cat((generated_latents, patch), dim=1),
        )


class InstructInputPrefillGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.core = core
        self.selector = TextTokenSelector()
        self.register_buffer(
            "empty_text_history",
            torch.zeros(1, 0, dtype=torch.int64),
            persistent=False,
        )

    def _scatter_latents(
        self,
        embeddings: torch.Tensor,
        latents: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        patch_embeddings = self.core.patch_encoder(latents)
        return embeddings.masked_scatter(
            mask.unsqueeze(-1), patch_embeddings.reshape(-1)
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        latents_in: torch.Tensor,
        latents_out: torch.Tensor,
        text_do_sample: torch.Tensor,
        text_temperature: torch.Tensor,
        text_top_k: torch.Tensor,
        text_top_p: torch.Tensor,
        text_repetition_penalty: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        latents_in_mask = text_ids == LATENT_IN_PAD_ID
        latents_out_mask = text_ids == LATENT_OUT_PAD_ID
        input_embeddings = self.core.backbone.embed(text_ids)
        input_embeddings = self._scatter_latents(input_embeddings, latents_in, latents_in_mask)
        input_embeddings = self._scatter_latents(input_embeddings, latents_out, latents_out_mask)
        keys, values = _empty_cache(self.core.geometry.backbone_qwen, input_embeddings)
        hidden_states, keys, values = self.core.backbone.forward_cached(input_embeddings, keys, values)
        selected_output_hidden = hidden_states.masked_select(
            latents_out_mask.unsqueeze(-1)
        ).reshape(1, -1, hidden_states.shape[-1])
        condition_history = _right_aligned_history(
            selected_output_hidden, self.core.geometry.history_patches + 1
        )
        latent_history = _right_aligned_history(latents_out, self.core.geometry.history_length)
        text_logits = functional.linear(hidden_states[:, -1], self.core.backbone.embed_tokens.weight)
        next_text_id = self.selector(
            text_logits,
            self.empty_text_history,
            text_do_sample,
            text_temperature,
            text_top_k,
            text_top_p,
            text_repetition_penalty,
        )
        return (
            *keys,
            *values,
            hidden_states[:, -1:],
            next_text_id,
            self.empty_text_history,
            condition_history,
            latent_history,
        )


class InstructTextDecodeGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.core = core
        self.selector = TextTokenSelector()

    def forward(
        self,
        text_ids: torch.Tensor,
        text_history: torch.Tensor,
        text_do_sample: torch.Tensor,
        text_temperature: torch.Tensor,
        text_top_k: torch.Tensor,
        text_top_p: torch.Tensor,
        text_repetition_penalty: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        keys, values = _split_cache(cache_tensors, self.core.geometry.backbone_qwen.layers)
        hidden_states, keys, values = self.core.backbone.forward_cached(
            self.core.backbone.embed(text_ids), keys, values
        )
        logits = functional.linear(hidden_states[:, -1], self.core.backbone.embed_tokens.weight)
        text_history = torch.cat((text_history, text_ids), dim=1)
        next_text_id = self.selector(
            logits,
            text_history,
            text_do_sample,
            text_temperature,
            text_top_k,
            text_top_p,
            text_repetition_penalty,
        )
        return (
            *keys,
            *values,
            hidden_states[:, -1:],
            next_text_id,
            text_history,
            text_ids == TEXT_EOT_ID,
        )


class InstructAudioStartGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.flow = ManualFlowRunner(core)
        self.patch_size = core.geometry.patch_size
        self.history_length = core.geometry.history_length
        self.condition_width = core.geometry.history_patches + 1

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        condition_history: torch.Tensor,
        append_last_hidden: torch.Tensor,
        latent_history: torch.Tensor,
        cfg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        appended_history = _append_history(
            condition_history, last_hidden_state, self.condition_width
        )
        selector = append_last_hidden.reshape(1, 1, 1)
        condition_history = condition_history * (1.0 - selector) + appended_history * selector
        flow_noise = torch.randn_like(latent_history[:, : self.patch_size])
        patch = self.flow(latent_history, condition_history, flow_noise, cfg)
        return (
            patch,
            _append_history(latent_history, patch, self.history_length),
            condition_history,
            patch,
        )


class InstructAudioDecodeCoreGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.core = core

    def forward(
        self,
        prior_patch: torch.Tensor,
        stop_threshold: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        keys, values = _split_cache(cache_tensors, self.core.geometry.backbone_qwen.layers)
        hidden_states, keys, values = self.core.backbone.forward_cached(
            self.core.patch_encoder(prior_patch), keys, values
        )
        stop_score = torch.sigmoid(self.core.stop_head(hidden_states[:, -1]))
        return (
            *keys,
            *values,
            hidden_states[:, -1:],
            stop_score,
            stop_score >= stop_threshold,
        )


class InstructAudioFlowGraph(nn.Module):
    def __init__(self, core: ManualTTSCore) -> None:
        super().__init__()
        self.flow = ManualFlowRunner(core)
        self.history_length = core.geometry.history_length
        self.condition_width = core.geometry.history_patches + 1

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        latent_history: torch.Tensor,
        condition_history: torch.Tensor,
        flow_noise: torch.Tensor,
        cfg: torch.Tensor,
        generated_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_history = _append_history(
            condition_history, last_hidden_state, self.condition_width
        )
        patch = self.flow(latent_history, condition_history, flow_noise, cfg)
        return (
            patch,
            _append_history(latent_history, patch, self.history_length),
            condition_history,
            torch.cat((generated_latents, patch), dim=1),
        )


def _package_name(variant: str) -> str:
    return f"FireRedTTS3_{variant.capitalize()}_ONNX"


def _metadata_file_name() -> str:
    return "FireRedTTS3_Metadata.onnx"


def _cache_names(geometry: QwenGeometry, direction: str) -> tuple[list[str], list[str]]:
    keys = [f"{direction}_key_{index}" for index in range(geometry.layers)]
    values = [f"{direction}_value_{index}" for index in range(geometry.layers)]
    return keys, values


def _cache_dynamic_axes(geometry: QwenGeometry) -> dict[str, dict[int, str]]:
    axes: dict[str, dict[int, str]] = {}
    for direction in ("in", "out"):
        keys, values = _cache_names(geometry, direction)
        dimension_name = "cache_length_in" if direction == "in" else "cache_length_out"
        for name in (*keys, *values):
            axes[name] = {2: dimension_name}
    return axes


def _dummy_caches(geometry: QwenGeometry, length: int = 2) -> tuple[torch.Tensor, ...]:
    key = torch.zeros(
        1,
        geometry.key_value_heads,
        length,
        geometry.head_dim,
        dtype=CACHE_STORAGE_DTYPE,
    )
    value = torch.zeros_like(key)
    return tuple(key.clone() for _ in range(geometry.layers)) + tuple(
        value.clone() for _ in range(geometry.layers)
    )


def _rewrite_external_locations(
    model: onnx.ModelProto,
    export_folder: Path,
    output_path: Path,
) -> list[tuple[Path, Path]]:
    """Give every graph-private external data file a collision-free package name."""
    locations: dict[str, Path] = {}
    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        entries = {entry.key: entry for entry in tensor.external_data}
        location_entry = entries.get("location")
        source = (export_folder / location_entry.value).resolve()
        locations.setdefault(location_entry.value, source)

    destinations: dict[str, Path] = {}
    for index, location in enumerate(sorted(locations), start=1):
        destination = output_path.parent / f"{output_path.stem}.external.{index:04d}.bin"
        destinations[location] = destination

    for tensor in model.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = destinations[entry.value].name
                break
    return [(locations[location], destination) for location, destination in destinations.items()]


def _value_contract(value: onnx.ValueInfoProto) -> dict[str, object]:
    tensor_type = value.type.tensor_type
    dimensions: list[int | str] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append("?")
    return {
        "name": value.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": dimensions,
    }


def _typed_shape_bytes(data_type: int, dimensions: Iterable[int]) -> int | None:
    shape = tuple(int(dimension) for dimension in dimensions)
    if any(dimension <= 0 for dimension in shape):
        return None
    try:
        item_size = onnx.helper.tensor_dtype_to_np_dtype(data_type).itemsize
    except (KeyError, TypeError, ValueError):
        return None
    return math.prod(shape) * int(item_size)


def _graph_metrics(model: onnx.ModelProto) -> dict[str, object]:
    try:
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
        shape_inference_succeeded = True
    except (onnx.checker.ValidationError, RuntimeError, TypeError, ValueError):
        inferred = model
        shape_inference_succeeded = False

    value_bytes: dict[str, int] = {}
    initializer_names = {initializer.name for initializer in inferred.graph.initializer}
    initializer_bytes = 0
    for initializer in inferred.graph.initializer:
        size = _typed_shape_bytes(initializer.data_type, initializer.dims)
        if size is not None:
            value_bytes[initializer.name] = size
            initializer_bytes += size
    for value in (
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dimensions = [
            int(dimension.dim_value)
            if dimension.HasField("dim_value")
            else 0
            for dimension in tensor_type.shape.dim
        ]
        size = _typed_shape_bytes(tensor_type.elem_type, dimensions)
        if size is not None:
            value_bytes[value.name] = size

    histogram = Counter(node.op_type for node in inferred.graph.node)
    known_traffic = sum(
        sum(value_bytes.get(name, 0) for name in (*node.input, *node.output))
        for node in inferred.graph.node
    )
    last_consumer: dict[str, int] = {}
    final_index = len(inferred.graph.node)
    for index, node in enumerate(inferred.graph.node):
        for name in node.input:
            last_consumer[name] = index
    for output in inferred.graph.output:
        last_consumer[output.name] = final_index

    active = {
        value.name: value_bytes[value.name]
        for value in inferred.graph.input
        if value.name in value_bytes and value.name not in initializer_names
    }
    peak_live_bytes = sum(active.values())
    for index, node in enumerate(inferred.graph.node):
        for name in node.output:
            if name in value_bytes:
                active[name] = value_bytes[name]
        peak_live_bytes = max(peak_live_bytes, sum(active.values()))
        for name in node.input:
            if last_consumer.get(name) == index:
                active.pop(name, None)
        for name in node.output:
            if name not in last_consumer:
                active.pop(name, None)

    families = {
        "cast": ("Cast",),
        "shape": ("Shape", "Size"),
        "layout": ("Transpose", "Reshape", "Flatten", "Squeeze", "Unsqueeze"),
        "indexing": ("Gather", "GatherElements", "GatherND", "Slice", "Split", "ScatterND"),
        "elementwise": (
            "Add", "Sub", "Mul", "Div", "Neg", "Pow", "Where", "Clip", "Equal",
            "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Not",
        ),
        "materializing_copy_upper_bound": (
            "Concat", "Pad", "Tile", "Expand", "Transpose", "ScatterND", "Reshape",
        ),
    }
    return {
        "node_count": len(inferred.graph.node),
        "initializer_count": len(inferred.graph.initializer),
        "initializer_bytes": initializer_bytes,
        "operator_histogram": dict(sorted(histogram.items())),
        "operator_families": {
            name: sum(histogram[operator] for operator in operators)
            for name, operators in families.items()
        },
        "known_tensor_traffic_bytes_lower_bound": known_traffic,
        "peak_live_activation_bytes_lower_bound": peak_live_bytes,
        "shape_inference_succeeded": shape_inference_succeeded,
        "inputs": [_value_contract(value) for value in inferred.graph.input],
        "outputs": [_value_contract(value) for value in inferred.graph.output],
        "opsets": {
            opset.domain or "ai.onnx": int(opset.version)
            for opset in inferred.opset_import
        },
        "custom_nodes": sorted(
            {
                f"{node.domain}::{node.op_type}"
                for node in inferred.graph.node
                if node.domain not in ("", "ai.onnx")
            }
        ),
    }


def _export_graph(
    module: nn.Module,
    arguments: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: Mapping[str, Mapping[int, str]] | None,
) -> None:
    module.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.export_",
    ) as temp_name:
        export_folder = Path(temp_name)
        export_path = export_folder / output_path.name
        torch.onnx.export(
            module,
            arguments,
            str(export_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dict(dynamic_axes) if dynamic_axes else None,
            opset_version=OPSET,
            dynamo=False,
            external_data=True,
            do_constant_folding=True,
        )
        model = onnx.load(str(export_path), load_external_data=False)
        relocations = _rewrite_external_locations(model, export_folder, output_path)
        onnx.save(model, str(output_path))
        for source, destination in relocations:
            os.replace(source, destination)


def _tokenizer_fingerprint(root: Path) -> str:
    return fingerprint_paths(
        (
            root / "text_tokenizer" / "tokenizer.json",
            root / "text_tokenizer" / "tokenizer_config.json",
            root / "text_tokenizer" / "vocab.json",
        )
    )


def _raw_graph_files(variant: str) -> dict[str, str]:
    files = {
        "model_file_name_redae_decode": "FireRedTTS3_RedAEDecode.onnx",
    }
    if variant == "base":
        files.update(
            {
                "model_file_name_base_reference_preprocess": "FireRedTTS3_BaseReferencePreprocess.onnx",
                "model_file_name_base_input_prefill": "FireRedTTS3_BaseInputPrefill.onnx",
                "model_file_name_base_audio_start": "FireRedTTS3_BaseAudioStart.onnx",
                "model_file_name_base_decode_core": "FireRedTTS3_BaseDecodeCore.onnx",
                "model_file_name_base_flow_patch": "FireRedTTS3_BaseFlowPatch.onnx",
            }
        )
    else:
        files.update(
            {
                "model_file_name_redae_encode": "FireRedTTS3_RedAEEncode.onnx",
                "model_file_name_instruct_input_prefill": "FireRedTTS3_InstructInputPrefill.onnx",
                "model_file_name_instruct_text_decode_step": "FireRedTTS3_InstructTextDecodeStep.onnx",
                "model_file_name_instruct_audio_start": "FireRedTTS3_InstructAudioStart.onnx",
                "model_file_name_instruct_audio_decode_core": "FireRedTTS3_InstructAudioDecodeCore.onnx",
                "model_file_name_instruct_audio_flow_patch": "FireRedTTS3_InstructAudioFlowPatch.onnx",
            }
        )
    return files


def build_raw_metadata(root: Path, geometry: PackageGeometry) -> dict[str, str]:
    return inference_metadata(
        build_metadata(
            {
                "package_schema_version": 2,
                "graph_layout": "raw_prefill_decode_core_flow",
                "runtime_tensor_contract": "waveform_and_token_ids_to_waveform",
                "graph_owned_preprocess": True,
                "graph_owned_sampling": True,
                "graph_owned_postprocess": True,
                "device_resident_decode_state": True,
                "model_variant": geometry.variant,
                "shared_initializer_model_file": SHARED_MODEL_NAME,
                "shared_initializer_data_file": SHARED_DATA_NAME,
                "tokenizer_path_or_fingerprint": _tokenizer_fingerprint(root),
                "out_sample_rate": OUT_SAMPLE_RATE,
                "input_audio_sample_rate": IN_SAMPLE_RATE,
                "model_audio_sample_rate": geometry.out_sample_rate,
                "redae_downsample_rate": geometry.redae_downsample_rate,
                "redae_upsample_rate": (
                    geometry.redae_downsample_rate // geometry.audio_patch_size
                ),
                "redae_max_seq_len": min(
                    geometry.encoder_qwen.max_sequence_length,
                    geometry.decoder_qwen.max_sequence_length,
                ),
                "patch_size": geometry.patch_size,
                "flow_steps": FLOW_STEPS,
                "flow_schedule": "one_minus_cosine_half_pi",
                "default_cfg": (
                    DEFAULT_BASE_CFG
                    if geometry.variant == "base"
                    else DEFAULT_INSTRUCT_CFG
                ),
                "default_clone_cfg": (
                    DEFAULT_BASE_CFG
                    if geometry.variant == "base"
                    else DEFAULT_INSTRUCT_CLONE_CFG
                ),
                "max_seq_len": MAX_SEQ_LEN,
                "vocab_size": geometry.backbone_qwen.vocab_size,
                "max_audio_patches": MAX_AUDIO_PATCHES,
                "min_audio_patches": MIN_AUDIO_PATCHES,
                "stop_threshold_default": STOP_THRESHOLD_DEFAULT,
                "text_eot_id": TEXT_EOT_ID,
                "audio_sos_id": AUDIO_SOS_ID,
                "latent_in_pad_id": LATENT_IN_PAD_ID,
                "latent_out_pad_id": LATENT_OUT_PAD_ID,
                **_raw_graph_files(geometry.variant),
            }
        )
    )


def _write_metadata_to_package(package_folder: Path, metadata: Mapping[str, str]) -> None:
    metadata_path = package_folder / _metadata_file_name()
    write_metadata_carrier(metadata_path, metadata, opset_version=OPSET)
    for graph_path in package_folder.glob("*.onnx"):
        if graph_path != metadata_path:
            replace_onnx_metadata(graph_path, metadata)


def _export_redae_graphs(package_folder: Path, redae: ManualRedAE, geometry: PackageGeometry) -> None:
    if geometry.variant != "base":
        _export_graph(
            RedAEEncodeGraph(redae, geometry),
            (torch.zeros(1, geometry.redae_downsample_rate * geometry.patch_size),),
            package_folder / "FireRedTTS3_RedAEEncode.onnx",
            ["prompt_audio"],
            ["prompt_latents"],
            {
                "prompt_audio": {1: "audio_samples"},
                "prompt_latents": {1: "latent_frames"},
            },
        )
    _export_graph(
        RedAEDecodeGraph(redae, geometry),
        (
            torch.zeros(1, geometry.patch_size, geometry.redae_dim),
            torch.zeros(1, geometry.patch_size, geometry.redae_dim),
        ),
        package_folder / "FireRedTTS3_RedAEDecode.onnx",
        ["generated_latents", "prefix_latents"],
        ["waveform"],
        {
            "generated_latents": {1: "generated_latent_frames"},
            "prefix_latents": {1: "prefix_latent_frames"},
            "waveform": {1: "audio_samples"},
        },
    )


def _export_base_reference_graph(
    package_folder: Path,
    geometry: PackageGeometry,
    redae: ManualRedAE,
    campp: ManualCAMPPlus,
) -> None:
    prompt_audio = torch.zeros(1, geometry.redae_downsample_rate * geometry.patch_size)
    _export_graph(
        BaseReferencePreprocessGraph(redae, campp, geometry),
        (prompt_audio,),
        package_folder / "FireRedTTS3_BaseReferencePreprocess.onnx",
        ["prompt_audio"],
        ["prompt_latents", "speaker_embedding"],
        {
            "prompt_audio": {1: "audio_samples"},
            "prompt_latents": {1: "latent_frames"},
        },
    )


def _export_base_tts_graphs(
    package_folder: Path,
    geometry: PackageGeometry,
    core: ManualTTSCore,
) -> None:
    backbone = geometry.backbone_qwen
    in_keys, in_values = _cache_names(backbone, "in")
    out_keys, out_values = _cache_names(backbone, "out")
    cache_axes = _cache_dynamic_axes(backbone)

    text_ids = torch.zeros(1, 8, dtype=torch.int64)
    prompt_latents = torch.zeros(1, geometry.patch_size, geometry.redae_dim)
    speaker_embedding = torch.zeros(1, geometry.speaker_dim)
    _export_graph(
        BaseInputPrefillGraph(core),
        (text_ids, prompt_latents, speaker_embedding),
        package_folder / "FireRedTTS3_BaseInputPrefill.onnx",
        ["text_ids", "prompt_latents", "speaker_embedding"],
        [
            *out_keys,
            *out_values,
            "last_hidden_state",
            "stop_logits",
            "condition_history",
            "latent_history",
            "speaker_condition",
        ],
        {
            **{name: {1: "text_length"} for name in ("text_ids",)},
            "prompt_latents": {1: "prompt_latent_frames"},
            **{
                name: axes
                for name, axes in cache_axes.items()
                if name.startswith("out_")
            },
        },
    )

    latent_history = torch.zeros(1, geometry.history_length, geometry.redae_dim)
    condition_history = torch.zeros(
        1, geometry.history_patches + 1, backbone.hidden_size
    )
    speaker_condition = torch.zeros(1, geometry.speaker_dim)
    flow_noise = torch.zeros(1, geometry.patch_size, geometry.redae_dim)
    cfg = torch.tensor([DEFAULT_BASE_CFG], dtype=torch.float32)
    _export_graph(
        BaseAudioStartGraph(core),
        (latent_history, condition_history, speaker_condition, cfg),
        package_folder / "FireRedTTS3_BaseAudioStart.onnx",
        ["latent_history", "condition_history", "speaker_condition", "cfg"],
        ["generated_patch", "next_latent_history", "generated_latents"],
        None,
    )

    prior_patch = torch.zeros(1, geometry.patch_size, geometry.redae_dim)
    stop_threshold = torch.tensor([STOP_THRESHOLD_DEFAULT], dtype=torch.float32)
    _export_graph(
        BaseDecodeCoreGraph(core),
        (prior_patch, stop_threshold, *_dummy_caches(backbone)),
        package_folder / "FireRedTTS3_BaseDecodeCore.onnx",
        ["prior_patch", "stop_threshold", *in_keys, *in_values],
        [*out_keys, *out_values, "last_hidden_state", "stop_score", "should_stop"],
        {**cache_axes},
    )
    generated_latents = torch.zeros(1, geometry.patch_size, geometry.redae_dim)
    _export_graph(
        BaseFlowPatchGraph(core),
        (
            torch.zeros(1, 1, backbone.hidden_size),
            latent_history,
            condition_history,
            speaker_condition,
            flow_noise,
            cfg,
            generated_latents,
        ),
        package_folder / "FireRedTTS3_BaseFlowPatch.onnx",
        [
            "last_hidden_state",
            "latent_history",
            "condition_history",
            "speaker_condition",
            "flow_noise",
            "cfg",
            "generated_latents",
        ],
        [
            "generated_patch",
            "next_latent_history",
            "next_condition_history",
            "generated_latents_out",
        ],
        {
            "generated_latents": {1: "generated_latent_frames"},
            "generated_latents_out": {1: "generated_latent_frames_out"},
        },
    )


def _export_instruct_graphs(package_folder: Path, geometry: PackageGeometry, core: ManualTTSCore) -> None:
    backbone = geometry.backbone_qwen
    in_keys, in_values = _cache_names(backbone, "in")
    out_keys, out_values = _cache_names(backbone, "out")
    cache_axes = _cache_dynamic_axes(backbone)
    text_ids = torch.zeros(1, 12, dtype=torch.int64)
    latents_in = torch.zeros(1, geometry.patch_size, geometry.redae_dim)
    latents_out = torch.zeros(1, geometry.patch_size, geometry.redae_dim)
    text_ids[:, 3] = LATENT_IN_PAD_ID
    text_ids[:, 8] = LATENT_OUT_PAD_ID
    text_do_sample = torch.tensor([True])
    text_temperature = torch.tensor([0.7], dtype=torch.float32)
    text_top_k = torch.tensor([20], dtype=torch.int64)
    text_top_p = torch.tensor([0.8], dtype=torch.float32)
    text_repetition_penalty = torch.tensor([1.0], dtype=torch.float32)
    _export_graph(
        InstructInputPrefillGraph(core),
        (
            text_ids,
            latents_in,
            latents_out,
            text_do_sample,
            text_temperature,
            text_top_k,
            text_top_p,
            text_repetition_penalty,
        ),
        package_folder / "FireRedTTS3_InstructInputPrefill.onnx",
        [
            "text_ids",
            "latents_in",
            "latents_out",
            "text_do_sample",
            "text_temperature",
            "text_top_k",
            "text_top_p",
            "text_repetition_penalty",
        ],
        [
            *out_keys,
            *out_values,
            "last_hidden_state",
            "next_text_id",
            "text_history",
            "condition_history",
            "latent_history",
        ],
        {
            "text_ids": {1: "text_length"},
            "latents_in": {1: "input_latent_frames"},
            "latents_out": {1: "output_latent_frames"},
            "text_history": {1: "text_history_length"},
            **{
                name: axes
                for name, axes in cache_axes.items()
                if name.startswith("out_")
            },
        },
    )
    _export_graph(
        InstructTextDecodeGraph(core),
        (
            torch.zeros(1, 1, dtype=torch.int64),
            torch.zeros(1, 2, dtype=torch.int64),
            text_do_sample,
            text_temperature,
            text_top_k,
            text_top_p,
            text_repetition_penalty,
            *_dummy_caches(backbone),
        ),
        package_folder / "FireRedTTS3_InstructTextDecodeStep.onnx",
        [
            "text_ids",
            "text_history",
            "text_do_sample",
            "text_temperature",
            "text_top_k",
            "text_top_p",
            "text_repetition_penalty",
            *in_keys,
            *in_values,
        ],
        [
            *out_keys,
            *out_values,
            "last_hidden_state",
            "next_text_id",
            "text_history_out",
            "is_eot",
        ],
        {
            "text_ids": {1: "text_length"},
            "text_history": {1: "text_history_length"},
            "text_history_out": {1: "text_history_length_out"},
            **cache_axes,
        },
    )

    latent_history = torch.zeros(1, geometry.history_length, geometry.redae_dim)
    condition_history = torch.zeros(
        1, geometry.history_patches + 1, backbone.hidden_size
    )
    flow_noise = torch.zeros(1, geometry.patch_size, geometry.redae_dim)
    cfg = torch.tensor([DEFAULT_INSTRUCT_CFG], dtype=torch.float32)
    _export_graph(
        InstructAudioStartGraph(core),
        (
            torch.zeros(1, 1, backbone.hidden_size),
            condition_history,
            torch.ones(1, dtype=torch.float32),
            latent_history,
            cfg,
        ),
        package_folder / "FireRedTTS3_InstructAudioStart.onnx",
        [
            "last_hidden_state",
            "condition_history",
            "append_last_hidden",
            "latent_history",
            "cfg",
        ],
        [
            "generated_patch",
            "next_latent_history",
            "next_condition_history",
            "generated_latents",
        ],
        None,
    )
    stop_threshold = torch.tensor([STOP_THRESHOLD_DEFAULT], dtype=torch.float32)
    _export_graph(
        InstructAudioDecodeCoreGraph(core),
        (
            torch.zeros(1, geometry.patch_size, geometry.redae_dim),
            stop_threshold,
            *_dummy_caches(backbone),
        ),
        package_folder / "FireRedTTS3_InstructAudioDecodeCore.onnx",
        ["prior_patch", "stop_threshold", *in_keys, *in_values],
        [*out_keys, *out_values, "last_hidden_state", "stop_score", "should_stop"],
        {**cache_axes},
    )
    generated_latents = torch.zeros(1, geometry.patch_size, geometry.redae_dim)
    _export_graph(
        InstructAudioFlowGraph(core),
        (
            torch.zeros(1, 1, backbone.hidden_size),
            latent_history,
            condition_history,
            flow_noise,
            cfg,
            generated_latents,
        ),
        package_folder / "FireRedTTS3_InstructAudioFlowPatch.onnx",
        [
            "last_hidden_state",
            "latent_history",
            "condition_history",
            "flow_noise",
            "cfg",
            "generated_latents",
        ],
        [
            "generated_patch",
            "next_latent_history",
            "next_condition_history",
            "generated_latents_out",
        ],
        {
            "generated_latents": {1: "generated_latent_frames"},
            "generated_latents_out": {1: "generated_latent_frames_out"},
        },
    )


def export_package() -> Path:
    root = CHECKPOINT_ROOT.resolve()
    geometry = read_package_geometry(root, MODEL_VARIANT)
    final_folder = SCRIPT_DIR / _package_name(geometry.variant)
    staging_folder = final_folder.with_name(final_folder.name + ".staging")
    shutil.rmtree(staging_folder, ignore_errors=True)
    staging_folder.mkdir(parents=True)

    try:
        tts_checkpoint = root / f"fireredtts3_{geometry.variant}" / "model.safetensors"
        redae_checkpoint = root / "redae" / "model.safetensors"
        redae = load_redae(geometry, redae_checkpoint)
        _export_redae_graphs(staging_folder, redae, geometry)
        if geometry.variant == "base":
            campp = load_campp(root / "campp" / "campplus_voxceleb.bin")
            _export_base_reference_graph(staging_folder, geometry, redae, campp)
            del campp, redae
            gc.collect()
        else:
            del redae
            gc.collect()
        core = load_tts_core(geometry, tts_checkpoint)
        if geometry.variant == "base":
            _export_base_tts_graphs(staging_folder, geometry, core)
        else:
            _export_instruct_graphs(staging_folder, geometry, core)
        metadata = build_raw_metadata(root, geometry)
        _write_metadata_to_package(staging_folder, metadata)
        promote_directory(staging_folder, final_folder)
        return final_folder
    finally:
        gc.collect()
        shutil.rmtree(staging_folder, ignore_errors=True)


def _run_post_export_demo(package: Path) -> None:
    print("Merging DecodeStep graphs for the post-export demo.")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "Merge_ONNX.py"),
            "--package-folder",
            str(package),
        ],
        check=True,
    )
    demo_modes = (
        ("base_clone",)
        if MODEL_VARIANT == "base"
        else ("instruct_clone", "voice_design", "semantic_edit", "acoustic_edit")
    )
    environment = os.environ.copy()
    environment["FIREREDTTS3_RUN_MODES"] = ",".join(demo_modes)
    print(
        f"Starting FireRedTTS3 demos {demo_modes} via "
        "Inference_FireRedTTS3_ONNX.py with one package load."
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "Inference_FireRedTTS3_ONNX.py"),
            "--package-folder",
            str(package),
        ],
        env=environment,
        check=True,
    )


def _restart_for_post_export_demo(package: Path) -> None:
    environment = os.environ.copy()
    environment[POST_EXPORT_DEMO_PACKAGE_ENV] = str(package.resolve())
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve())],
        environment,
    )


def main() -> None:
    demo_package = os.environ.pop(POST_EXPORT_DEMO_PACKAGE_ENV, None)
    if demo_package is not None:
        _run_post_export_demo(Path(demo_package))
        return
    if not DO_EXPORT:
        print("Set DO_EXPORT = True in Export_FireRedTTS3.py to create a raw package.")
        return
    package = export_package()
    print(
        f"Raw FireRedTTS3 {MODEL_VARIANT} package exported to: {package}",
        flush=True,
    )
    _restart_for_post_export_demo(package)


if __name__ == "__main__":
    main()
