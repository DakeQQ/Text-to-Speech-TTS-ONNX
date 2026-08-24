from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchaudio
import yaml
from torch import nn

from Raon_Config import (
    UNSUPPORTED_TRAILING_VOCAB_TOKENS,
    RaonArchitecture,
    require_architecture,
)
from STFT_Process import STFT_Process


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ASSET_DIR = Path.home() / "Downloads" / "Raon-OpenTTS-0.3B"

# Easy-to-change settings
# Point this folder at either Raon-OpenTTS-0.3B or Raon-OpenTTS-1B.
# Leave CHECKPOINT_PATH as None to discover its single model_*.pt file.
CHECKPOINT_PATH: Path | None = None
CONFIG_PATH             = DEFAULT_ASSET_DIR / "config.yaml"
VOCAB_PATH              = DEFAULT_ASSET_DIR / "vocab.txt"
VOCODER_CHECKPOINT_PATH: Path | None = None
ONNX_FOLDER             = SCRIPT_DIR / "Raon_ONNX"
OPSET                   = 20
USE_FP16_TRANSFORMER    = False

# Sampling controls
# These affect generation speed, guidance strength, and output loudness.
NFE_STEP         = 32
CFG_STRENGTH     = 2.0
SWAY_COEFFICIENT = -1.0
TARGET_RMS       = 0.1

# Shared model settings for both supported checkpoints.
MODEL_SAMPLE_RATE = 16_000
IN_SAMPLE_RATE    = 16_000
OUT_SAMPLE_RATE   = 16_000
N_MELS            = 80
NFFT              = 1024
WINDOW_LENGTH     = 1024
HOP_LENGTH        = 256
WINDOW_TYPE       = "hann"
MAX_SIGNAL_LENGTH = 4096

POSITION_CONV_GROUPS = 16


@dataclass
class PreflightResult:
    checkpoint_path: Path
    config_path: Path
    vocab_path: Path
    vocoder_checkpoint_path: Path
    config: dict[str, Any]
    architecture: RaonArchitecture
    vocab_map: dict[str, int]
    vocab_sha256: str
    checkpoint_state: dict[str, torch.Tensor]


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def discover_checkpoint(asset_dir: Path) -> Path:
    resolved = asset_dir.expanduser().resolve()
    candidates = sorted(path for path in resolved.glob("model_*.pt") if path.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one model_*.pt checkpoint in {resolved}, found {candidates}"
        )
    return candidates[0]


def discover_vocoder_checkpoint(asset_dir: Path) -> Path:
    resolved = asset_dir.expanduser().resolve()
    candidates = (
        resolved / "generator.ckpt",
        resolved.parent / "Raon-OpenTTS-0.3B" / "generator.ckpt",
        resolved.parent / "Raon-OpenTTS-1B" / "generator.ckpt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


def _nested(config: dict[str, Any], dotted_path: str, default: Any = ...) -> Any:
    value: Any = config
    for component in dotted_path.split("."):
        if not isinstance(value, dict) or component not in value:
            if default is not ...:
                return default
            raise ValueError(f"Configuration is missing required key: {dotted_path}")
        value = value[component]
    return value


def architecture_from_config(config: dict[str, Any]) -> RaonArchitecture:
    model_name = _nested(config, "model.name")
    return require_architecture(model_name, "model.name")


def load_and_validate_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")

    architecture = architecture_from_config(config)
    expected = {
        "model.name": architecture.model_name,
        "model.tokenizer": "custom",
        "model.backbone": "DiT",
        "model.arch.dim": architecture.dim,
        "model.arch.depth": architecture.depth,
        "model.arch.heads": architecture.heads,
        "model.arch.ff_mult": architecture.ff_mult,
        "model.arch.text_dim": architecture.text_dim,
        "model.arch.text_mask_padding": True,
        "model.arch.qk_norm": None,
        "model.arch.conv_layers": architecture.text_conv_layers,
        "model.arch.pe_attn_head": None,
        "model.arch.attn_mask_enabled": False,
        "model.arch.checkpoint_activations": False,
        "model.arch.logit_softcapping": None,
        "model.arch.post_norm": False,
        "model.arch.norm_type": "rmsnorm",
        "model.mel_spec.target_sample_rate": MODEL_SAMPLE_RATE,
        "model.mel_spec.n_mel_channels": N_MELS,
        "model.mel_spec.hop_length": HOP_LENGTH,
        "model.mel_spec.win_length": WINDOW_LENGTH,
        "model.mel_spec.n_fft": NFFT,
        "model.mel_spec.mel_spec_type": "sbhifigan16k",
    }
    mismatches = []
    for dotted_path, expected_value in expected.items():
        actual_value = _nested(config, dotted_path)
        if actual_value != expected_value:
            mismatches.append(f"{dotted_path}: expected {expected_value!r}, found {actual_value!r}")

    long_skip = _nested(config, "model.arch.long_skip_connection", False)
    if long_skip is not False:
        mismatches.append(
            f"model.arch.long_skip_connection: expected False or absent, found {long_skip!r}"
        )
    configured_head_dim = _nested(config, "model.arch.dim_head", 64)
    if configured_head_dim != architecture.head_dim:
        mismatches.append(
            "model.arch.dim_head: expected "
            f"{architecture.head_dim}, found {configured_head_dim!r}"
        )
    if architecture.dim % POSITION_CONV_GROUPS != 0:
        mismatches.append(
            f"model.arch.dim must be divisible by {POSITION_CONV_GROUPS}: "
            f"found {architecture.dim}"
        )
    if architecture.head_dim <= 2 or architecture.head_dim % 2 != 0:
        mismatches.append(
            f"model.arch.dim_head must be an even integer greater than 2: "
            f"found {architecture.head_dim}"
        )
    if mismatches:
        raise ValueError("Unsupported Raon configuration:\n  " + "\n  ".join(mismatches))
    return config


def load_vocab(path: Path) -> tuple[dict[str, int], str]:
    tokens: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.endswith("\n"):
                raise ValueError(
                    f"Vocabulary line {index + 1} is not newline-terminated: {path}"
                )
            token = line[:-1]
            if token in seen:
                raise ValueError(
                    f"Duplicate vocabulary token at line {index + 1}: {token!r}"
                )
            seen.add(token)
            tokens.append(token)
    if tuple(tokens[-len(UNSUPPORTED_TRAILING_VOCAB_TOKENS) :]) == (
        UNSUPPORTED_TRAILING_VOCAB_TOKENS
    ):
        del tokens[-len(UNSUPPORTED_TRAILING_VOCAB_TOKENS) :]
    if not tokens:
        raise ValueError(f"Vocabulary is empty: {path}")
    vocab_map = {token: index for index, token in enumerate(tokens)}
    if vocab_map.get(" ") != 0:
        raise ValueError(
            f"Vocabulary must map a literal space to token 0; found {vocab_map.get(' ')!r}: {path}"
        )
    if sorted(vocab_map.values()) != list(range(len(vocab_map))):
        raise ValueError("Vocabulary token IDs must be contiguous from zero")
    return vocab_map, hashlib.sha256(path.read_bytes()).hexdigest()


def load_weights_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)
    except RuntimeError as error:
        if "mmap" not in str(error).casefold():
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def _expected_checkpoint_shapes(
    vocab_rows: int, architecture: RaonArchitecture
) -> dict[str, tuple[int, ...]]:
    dim = architecture.dim
    attention_inner_dim = architecture.attention_inner_dim
    shapes: dict[str, tuple[int, ...]] = {
        "transformer.time_embed.time_mlp.0.weight": (dim, 256),
        "transformer.time_embed.time_mlp.0.bias": (dim,),
        "transformer.time_embed.time_mlp.2.weight": (dim, dim),
        "transformer.time_embed.time_mlp.2.bias": (dim,),
        "transformer.text_embed.text_embed.weight": (vocab_rows, architecture.text_dim),
        "transformer.rotary_embed.inv_freq": (architecture.head_dim // 2,),
        "transformer.input_embed.proj.weight": (
            dim,
            N_MELS * 2 + architecture.text_dim,
        ),
        "transformer.input_embed.proj.bias": (dim,),
        "transformer.input_embed.conv_pos_embed.conv1d.0.weight": (
            dim,
            dim // POSITION_CONV_GROUPS,
            31,
        ),
        "transformer.input_embed.conv_pos_embed.conv1d.0.bias": (dim,),
        "transformer.input_embed.conv_pos_embed.conv1d.2.weight": (
            dim,
            dim // POSITION_CONV_GROUPS,
            31,
        ),
        "transformer.input_embed.conv_pos_embed.conv1d.2.bias": (dim,),
        "transformer.norm_out.linear.weight": (dim * 2, dim),
        "transformer.norm_out.linear.bias": (dim * 2,),
        "transformer.proj_out.weight": (N_MELS, dim),
        "transformer.proj_out.bias": (N_MELS,),
    }
    text_inner = architecture.text_dim * architecture.text_conv_mult
    for index in range(architecture.text_conv_layers):
        prefix = f"transformer.text_embed.text_blocks.{index}"
        shapes.update(
            {
                f"{prefix}.dwconv.weight": (architecture.text_dim, 1, 7),
                f"{prefix}.dwconv.bias": (architecture.text_dim,),
                f"{prefix}.norm.weight": (architecture.text_dim,),
                f"{prefix}.norm.bias": (architecture.text_dim,),
                f"{prefix}.pwconv1.weight": (text_inner, architecture.text_dim),
                f"{prefix}.pwconv1.bias": (text_inner,),
                f"{prefix}.grn.gamma": (1, 1, text_inner),
                f"{prefix}.grn.beta": (1, 1, text_inner),
                f"{prefix}.pwconv2.weight": (architecture.text_dim, text_inner),
                f"{prefix}.pwconv2.bias": (architecture.text_dim,),
            }
        )
    ff_dim = dim * architecture.ff_mult
    for index in range(architecture.depth):
        prefix = f"transformer.transformer_blocks.{index}"
        shapes.update(
            {
                f"{prefix}.attn_norm.linear.weight": (dim * 6, dim),
                f"{prefix}.attn_norm.linear.bias": (dim * 6,),
                f"{prefix}.attn.to_q.weight": (attention_inner_dim, dim),
                f"{prefix}.attn.to_q.bias": (attention_inner_dim,),
                f"{prefix}.attn.to_k.weight": (attention_inner_dim, dim),
                f"{prefix}.attn.to_k.bias": (attention_inner_dim,),
                f"{prefix}.attn.to_v.weight": (attention_inner_dim, dim),
                f"{prefix}.attn.to_v.bias": (attention_inner_dim,),
                f"{prefix}.attn.to_out.0.weight": (dim, attention_inner_dim),
                f"{prefix}.attn.to_out.0.bias": (dim,),
                f"{prefix}.ff.ff.0.0.weight": (ff_dim, dim),
                f"{prefix}.ff.ff.0.0.bias": (ff_dim,),
                f"{prefix}.ff.ff.2.weight": (dim, ff_dim),
                f"{prefix}.ff.ff.2.bias": (dim,),
            }
        )
    return shapes


def load_ema_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = load_weights_file(path)
    if not isinstance(checkpoint, dict) or "ema_model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint must contain ema_model_state_dict: {path}")
    ema_state = checkpoint["ema_model_state_dict"]
    if not isinstance(ema_state, dict):
        raise ValueError(f"ema_model_state_dict must be a mapping: {path}")

    normalized: dict[str, torch.Tensor] = {}
    for original_name, tensor in ema_state.items():
        if not isinstance(original_name, str):
            raise ValueError("Checkpoint contains a non-string state-dict key")
        name = original_name.removeprefix("ema_model.")
        if name in {"initted", "step"}:
            continue
        if name in normalized:
            raise ValueError(f"Checkpoint key collision after EMA prefix removal: {name}")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Checkpoint entry is not a tensor: {original_name}")
        normalized[name] = tensor
    return normalized


def validate_checkpoint_state(
    state: dict[str, torch.Tensor],
    vocab_map: dict[str, int],
    checkpoint_path: Path,
    architecture: RaonArchitecture,
) -> None:
    embedding_name = "transformer.text_embed.text_embed.weight"
    embedding = state.get(embedding_name)
    if embedding is None or embedding.ndim != 2:
        raise ValueError(f"Checkpoint is missing a rank-2 {embedding_name}: {checkpoint_path}")
    required_rows = len(vocab_map) + 1
    checkpoint_rows = int(embedding.shape[0])
    if checkpoint_rows != required_rows:
        difference = required_rows - checkpoint_rows
        raise ValueError(
            "Vocabulary/checkpoint mismatch: "
            f"the vocabulary loads {len(vocab_map)} unique tokens and therefore requires "
            f"{required_rows} text-embedding rows, but {checkpoint_path.name} has "
            f"{checkpoint_rows}. The supplied vocabulary has {difference} more required row(s). "
            "Supply a verified matching vocabulary or an explicitly approved upstream mapping; "
            "do not truncate, reorder, resize, or silently remap tokens."
        )

    expected_shapes = _expected_checkpoint_shapes(required_rows, architecture)
    expected_keys = set(expected_shapes)
    actual_keys = set(state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    wrong_shapes = [
        f"{name}: expected {expected_shapes[name]}, found {tuple(state[name].shape)}"
        for name in sorted(expected_keys & actual_keys)
        if tuple(state[name].shape) != expected_shapes[name]
    ]
    block_indices = sorted(
        {
            int(name.split("transformer.transformer_blocks.", 1)[1].split(".", 1)[0])
            for name in state
            if name.startswith("transformer.transformer_blocks.")
        }
    )
    if block_indices != list(range(architecture.depth)):
        wrong_shapes.append(
            "transformer block indices: expected "
            f"0..{architecture.depth - 1}, found {block_indices}"
        )
    if missing or unexpected or wrong_shapes:
        details = []
        if missing:
            details.append(f"missing keys ({len(missing)}): {missing[:10]}")
        if unexpected:
            details.append(f"unexpected keys ({len(unexpected)}): {unexpected[:10]}")
        details.extend(wrong_shapes[:20])
        raise ValueError("Checkpoint architecture mismatch:\n  " + "\n  ".join(details))


def preflight_assets(
    checkpoint_path: Path,
    config_path: Path,
    vocab_path: Path,
    vocoder_checkpoint_path: Path,
) -> PreflightResult:
    checkpoint_path = _require_file(checkpoint_path, "Raon checkpoint")
    config_path = _require_file(config_path, "Raon configuration")
    vocab_path = _require_file(vocab_path, "Raon vocabulary")
    config = load_and_validate_config(config_path)
    architecture = architecture_from_config(config)
    vocab_map, vocab_sha256 = load_vocab(vocab_path)
    try:
        checkpoint_state = load_ema_checkpoint(checkpoint_path)
    except (OSError, RuntimeError) as error:
        raise ValueError(
            f"Unable to read Raon checkpoint {checkpoint_path}: {error}"
        ) from error
    validate_checkpoint_state(
        checkpoint_state,
        vocab_map,
        checkpoint_path,
        architecture,
    )

    try:
        vocoder_checkpoint_path = _require_file(vocoder_checkpoint_path, "HiFi-GAN checkpoint")
    except FileNotFoundError as error:
        vocoder_dir = vocoder_checkpoint_path.expanduser().resolve().parent
        raise FileNotFoundError(
            f"{error}\nDownload the official SpeechBrain HiFi-GAN checkpoint with:\n"
            "huggingface-cli download speechbrain/tts-hifigan-libritts-16kHz "
            f"generator.ckpt --local-dir {vocoder_dir}"
        ) from error

    return PreflightResult(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        vocab_path=vocab_path,
        vocoder_checkpoint_path=vocoder_checkpoint_path,
        config=config,
        architecture=architecture,
        vocab_map=vocab_map,
        vocab_sha256=vocab_sha256,
        checkpoint_state=checkpoint_state,
    )


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10_000.0, theta_rescale_factor: float = 1.0
) -> torch.Tensor:
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    frequencies = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
    )
    positions = torch.arange(end, dtype=torch.float32)
    phases = torch.outer(positions, frequencies)
    return torch.cat((phases.cos(), phases.sin()), dim=-1)


class GRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        response = torch.norm(tensor, p=2, dim=1, keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (tensor * normalized) + self.beta + tensor


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * 3
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=padding,
            groups=dim,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        residual = tensor
        tensor = self.dwconv(tensor.transpose(1, 2)).transpose(1, 2)
        tensor = self.norm(tensor)
        tensor = self.pwconv1(tensor)
        tensor = self.act(tensor)
        tensor = self.grn(tensor)
        tensor = self.pwconv2(tensor)
        return residual + tensor


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = math.log(10_000.0) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timestep.device, dtype=torch.float32) * -exponent
        )
        embedding = scale * timestep.unsqueeze(1) * frequencies.unsqueeze(0)
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


class ConvPositionEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 31,
        groups: int = POSITION_CONV_GROUPS,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConvPositionEmbedding requires an odd kernel size")
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv1d(tensor.transpose(1, 2)).transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        variance = tensor.float().square().mean(dim=-1, keepdim=True)
        normalized = tensor * torch.rsqrt(variance + self.eps)
        return normalized.to(self.weight.dtype) * self.weight


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        tensor: torch.Tensor,
        embedding: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if modulation is None:
            if embedding is None:
                raise RuntimeError("AdaLayerNorm requires embedding or modulation")
            modulation = self.linear(self.silu(embedding))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.split(
            modulation, self.dim, dim=-1
        )
        tensor = self.norm(tensor) * (1.0 + scale_msa) + shift_msa
        return tensor, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        tensor: torch.Tensor,
        embedding: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if modulation is None:
            if embedding is None:
                raise RuntimeError("AdaLayerNormFinal requires embedding or modulation")
            modulation = self.linear(self.silu(embedding))
        scale, shift = torch.split(modulation, self.dim, dim=-1)
        return self.norm(tensor) * (1.0 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(approximate="tanh")),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.ff(tensor)


class RotaryEmbeddingState(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            10_000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inverse_frequency, persistent=True)


def rotate_half(tensor: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    return torch.index_select(tensor, -1, permutation)


def apply_rotary(
    tensor: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    permutation: torch.Tensor,
) -> torch.Tensor:
    return tensor * rope_cos + rotate_half(tensor, permutation) * rope_sin


class AttnProcessor:
    def __init__(self, head_dim: int, heads: int, fp16: bool = False) -> None:
        self.head_dim = head_dim
        self.heads = heads
        self.inner_dim = heads * head_dim
        self.qk_heads = heads * 2
        self.fp16 = fp16
        assert self.qk_heads + self.heads == self.heads * 3

    def __call__(
        self,
        attention: Attention,
        tensor: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        assert attention.inner_dim == self.inner_dim
        qkv = attention.to_qkv(tensor).view(
            2, -1, self.heads * 3, self.head_dim
        ).transpose(1, 2)
        query_key, value = torch.split(qkv, [self.qk_heads, self.heads], dim=1)
        query_key = apply_rotary(
            query_key,
            rope_cos,
            rope_sin,
            attention.rope_permutation,
        )
        query, key = torch.split(query_key, self.heads, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2))
        if self.fp16:
            weights = torch.softmax(scores.float() * 100.0, dim=-1, dtype=torch.float32).half()
        else:
            weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(2, -1, self.inner_dim)
        return attention.to_out[0](attended)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.inner_dim = heads * dim_head
        permutation = torch.arange(dim_head, dtype=torch.int64).view(-1, 2).flip(-1).reshape(-1)
        self.register_buffer("rope_permutation", permutation, persistent=False)
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.ModuleList((nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)))
        self.processor = AttnProcessor(dim_head, heads)

    def forward(
        self, tensor: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        return self.processor(self, tensor, rope_cos, rope_sin)

    def fuse_qkv(self, scale: float, fp16: bool) -> None:
        fused = nn.Linear(
            self.dim,
            self.inner_dim * 3,
            device=self.to_q.weight.device,
            dtype=self.to_q.weight.dtype,
        )
        with torch.no_grad():
            fused.weight.copy_(
                torch.cat(
                    (self.to_q.weight * scale, self.to_k.weight * scale, self.to_v.weight),
                    dim=0,
                )
            )
            fused.bias.copy_(
                torch.cat(
                    (self.to_q.bias * scale, self.to_k.bias * scale, self.to_v.bias),
                    dim=0,
                )
            )
        self.to_qkv = fused
        self.processor.fp16 = fp16
        del self.to_q, self.to_k, self.to_v

    def refresh_export_buffers(self) -> None:
        permutation = torch.arange(
            self.inner_dim // self.heads, dtype=torch.int64, device=self.to_out[0].weight.device
        ).view(-1, 2).flip(-1).reshape(-1)
        self.rope_permutation = permutation


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        tensor: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        modulation: torch.Tensor,
    ) -> torch.Tensor:
        normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(
            tensor, modulation=modulation
        )
        tensor = tensor + gate_msa * self.attn(normalized, rope_cos, rope_sin)
        normalized = self.ff_norm(tensor) * (1.0 + scale_mlp) + shift_mlp
        return tensor + gate_mlp * self.ff(normalized)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, frequency_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = SinusPositionEmbedding(frequency_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(frequency_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(hidden)


class TextEmbedding(nn.Module):
    def __init__(
        self,
        text_num_embeds: int,
        text_dim: int,
        mask_padding: bool,
        conv_layers: int,
        conv_mult: int = 2,
    ) -> None:
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        self.precompute_max_pos = MAX_SIGNAL_LENGTH
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(text_dim, self.precompute_max_pos).unsqueeze(0),
            persistent=False,
        )
        self.text_blocks = nn.Sequential(
            *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
        )

    def forward(
        self, text_ids: torch.Tensor, max_duration: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_mask = (text_ids == 0).unsqueeze(-1) if self.mask_padding else None
        dropped_ids = torch.zeros_like(text_ids)
        text = self.text_embed(torch.cat((text_ids, dropped_ids), dim=0))
        text = text + self.freqs_cis[:, :max_duration]
        if text_mask is not None:
            text = torch.where(text_mask, 0.0, text)
        for block in self.text_blocks:
            text = block(text)
            if text_mask is not None:
                text = torch.where(text_mask, 0.0, text)
        return torch.split(text, [1, 1], dim=0)

    def refresh_export_buffers(self) -> None:
        self.freqs_cis = precompute_freqs_cis(
            self.text_embed.embedding_dim, self.precompute_max_pos
        ).unsqueeze(0).to(self.text_embed.weight.device)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim: int, text_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, output_dim)
        self.conv_pos_embed = ConvPositionEmbedding(output_dim)

    def forward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        projected = self.proj(torch.cat((tensor, condition), dim=-1))
        return projected + self.conv_pos_embed(projected)


class DiT(nn.Module):
    def __init__(self, vocab_size: int, architecture: RaonArchitecture) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedding(architecture.dim)
        self.text_embed = TextEmbedding(
            vocab_size,
            architecture.text_dim,
            mask_padding=True,
            conv_layers=architecture.text_conv_layers,
            conv_mult=architecture.text_conv_mult,
        )
        self.input_embed = InputEmbedding(N_MELS, architecture.text_dim, architecture.dim)
        self.rotary_embed = RotaryEmbeddingState(architecture.head_dim)
        self.dim = architecture.dim
        self.depth = architecture.depth
        self.heads = architecture.heads
        self.head_dim = architecture.head_dim
        self.text_dim = architecture.text_dim
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=architecture.dim,
                    heads=architecture.heads,
                    dim_head=architecture.head_dim,
                    ff_mult=architecture.ff_mult,
                    dropout=0.1,
                )
                for _ in range(architecture.depth)
            ]
        )
        self.norm_out = AdaLayerNormFinal(architecture.dim)
        self.proj_out = nn.Linear(architecture.dim, N_MELS)
        self.time_activation = nn.SiLU()

    def forward(
        self,
        tensor: torch.Tensor,
        condition: torch.Tensor,
        dropped_condition: torch.Tensor,
        time_embedding: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        tensor = torch.cat((tensor, tensor), dim=0)
        condition = torch.cat((condition, dropped_condition), dim=0)
        tensor = self.input_embed(tensor, condition)
        modulations = torch.split(
            self.time_modulation(self.time_activation(time_embedding)),
            self.time_modulation_split_sizes,
            dim=-1,
        )
        for block, modulation in zip(self.transformer_blocks, modulations[:-1]):
            tensor = block(tensor, rope_cos, rope_sin, modulation)
        tensor = self.norm_out(tensor, modulation=modulations[-1])
        return self.proj_out(tensor)

    def fuse_for_export(self, fp16: bool) -> None:
        scale = self.head_dim**-0.25
        if fp16:
            scale *= 0.1
        for block in self.transformer_blocks:
            block.attn.fuse_qkv(scale, fp16=fp16)

        block_width = self.dim * 6
        final_width = self.dim * 2
        reference = self.transformer_blocks[0].attn_norm.linear.weight
        fused = nn.Linear(
            self.dim,
            block_width * self.depth + final_width,
            device=reference.device,
            dtype=reference.dtype,
        )
        offset = 0
        with torch.no_grad():
            for block in self.transformer_blocks:
                next_offset = offset + block_width
                fused.weight[offset:next_offset].copy_(block.attn_norm.linear.weight)
                fused.bias[offset:next_offset].copy_(block.attn_norm.linear.bias)
                offset = next_offset
            fused.weight[offset:].copy_(self.norm_out.linear.weight)
            fused.bias[offset:].copy_(self.norm_out.linear.bias)
        self.time_modulation = fused
        self.time_modulation_split_sizes = [block_width] * self.depth + [final_width]
        for block in self.transformer_blocks:
            del block.attn_norm.linear, block.attn_norm.silu
        del self.norm_out.linear, self.norm_out.silu

    def refresh_export_buffers(self) -> None:
        self.text_embed.refresh_export_buffers()
        for block in self.transformer_blocks:
            block.attn.refresh_export_buffers()


class RaonModel(nn.Module):
    def __init__(self, vocab_size: int, architecture: RaonArchitecture) -> None:
        super().__init__()
        self.transformer = DiT(vocab_size, architecture)

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.transformer(*args)


def build_model(assets: PreflightResult) -> RaonModel:
    with torch.device("meta"):
        model = RaonModel(len(assets.vocab_map), assets.architecture)
    incompatible = model.load_state_dict(assets.checkpoint_state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.transformer.refresh_export_buffers()
    meta_tensors = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.is_meta
    ]
    if meta_tensors:
        raise RuntimeError(f"Model contains unresolved meta tensors after strict load: {meta_tensors}")
    return model.eval()


class RaonPreprocess(nn.Module):
    def __init__(
        self,
        text_embedding: TextEmbedding,
        architecture: RaonArchitecture,
        use_fp16_transformer: bool,
    ) -> None:
        super().__init__()
        self.text_embedding = text_embedding
        self.custom_stft = STFT_Process(
            model_type="stft_B",
            n_fft=NFFT,
            win_length=WINDOW_LENGTH,
            hop_len=HOP_LENGTH,
            max_frames=0,
            window_type=WINDOW_TYPE,
            center_pad=False,
            pad_mode="reflect",
        ).eval()
        mel_filterbank = torchaudio.functional.melscale_fbanks(
            n_freqs=NFFT // 2 + 1,
            f_min=0.0,
            f_max=MODEL_SAMPLE_RATE / 2,
            n_mels=N_MELS,
            sample_rate=MODEL_SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1).unsqueeze(0)
        self.register_buffer("mel_filterbank", mel_filterbank, persistent=False)
        self.register_buffer(
            "mel_padding",
            torch.zeros((1, MAX_SIGNAL_LENGTH, N_MELS), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "text_padding",
            torch.zeros((1, MAX_SIGNAL_LENGTH), dtype=torch.int32),
            persistent=False,
        )

        inverse_frequency = 1.0 / (
            10_000.0
            ** (
                torch.arange(0, architecture.head_dim, 2, dtype=torch.float32)
                / architecture.head_dim
            )
        )
        phases = torch.outer(
            torch.arange(MAX_SIGNAL_LENGTH, dtype=torch.float32), inverse_frequency
        )
        rope_cos = torch.stack((phases.cos(), phases.cos()), dim=-1).flatten(-2)
        rope_sin = torch.stack((-phases.sin(), phases.sin()), dim=-1).flatten(-2)
        transformer_dtype = torch.float16 if use_fp16_transformer else torch.float32
        self.register_buffer(
            "rope_cos",
            rope_cos.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1).to(transformer_dtype),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            rope_sin.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1).to(transformer_dtype),
            persistent=False,
        )
        self.use_fp16_transformer = use_fp16_transformer

    def forward(
        self,
        audio: torch.Tensor,
        text_ids: torch.Tensor,
        max_duration: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        audio = audio.float()
        peak = audio.abs().amax()
        audio = (audio / torch.clamp(peak, min=1e-3)).clamp(min=-1.0, max=1.0)
        audio_rms = torch.sqrt(torch.mean(audio.square()))
        rms_scale = torch.clamp(audio_rms / TARGET_RMS, max=1.0).reshape(1)
        audio = audio * torch.clamp(TARGET_RMS / audio_rms.clamp(min=1e-6), min=1.0)

        real, imaginary = self.custom_stft(audio)
        magnitude = torch.sqrt(real.square() + imaginary.square())
        mel = torch.matmul(self.mel_filterbank, magnitude)
        mel = mel.transpose(1, 2).clamp(min=1e-5).log()
        ref_signal_len = torch.div(
            torch._shape_as_tensor(audio)[-1], HOP_LENGTH, rounding_mode="floor"
        ).to(torch.int64)

        zeros = self.mel_padding[:, :max_duration]
        mel = torch.cat((mel, self.mel_padding), dim=1)[:, :max_duration]
        text_ids = torch.cat((text_ids + 1, self.text_padding), dim=-1)[:, :max_duration]
        text, dropped_text = self.text_embedding(text_ids, max_duration)
        noise = torch.randn_like(zeros)
        condition = torch.cat((mel, text), dim=-1)
        dropped_condition = torch.cat((zeros, dropped_text), dim=-1)
        rope_cos = self.rope_cos[:, :, :max_duration]
        rope_sin = self.rope_sin[:, :, :max_duration]
        if self.use_fp16_transformer:
            return (
                noise.half(),
                rope_cos,
                rope_sin,
                condition.half(),
                dropped_condition.half(),
                ref_signal_len,
                rms_scale,
            )
        return noise, rope_cos, rope_sin, condition, dropped_condition, ref_signal_len, rms_scale


def get_epss_timesteps(steps: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    predefined = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    selected = predefined.get(steps)
    if selected is None:
        return torch.linspace(0.0, 1.0, steps + 1, dtype=dtype)
    return torch.tensor(selected, dtype=dtype) / 32.0


class RaonTransformer(nn.Module):
    def __init__(self, transformer: DiT, fp16: bool) -> None:
        super().__init__()
        self.transformer = transformer
        timesteps = get_epss_timesteps(NFE_STEP)
        timesteps = timesteps + SWAY_COEFFICIENT * (
            torch.cos(torch.pi * 0.5 * timesteps) - 1.0 + timesteps
        )
        dtype = torch.float16 if fp16 else torch.float32
        delta_t = torch.diff(timesteps).to(dtype).view(-1, 1, 1)
        assert tuple(delta_t.shape) == (NFE_STEP, 1, 1)
        self.register_buffer(
            "delta_t", delta_t, persistent=False
        )

        with torch.no_grad():
            expanded = transformer.time_embed(timesteps).unsqueeze(0)
        assert tuple(expanded.shape) == (1, NFE_STEP + 1, transformer.dim)
        self.register_buffer("time_expand", expanded.to(dtype), persistent=False)

    def forward(
        self,
        noise: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        condition: torch.Tensor,
        dropped_condition: torch.Tensor,
        time_step: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.transformer(
            noise,
            condition,
            dropped_condition,
            self.time_expand[:, time_step],
            rope_cos,
            rope_sin,
        )
        conditional, unconditional = torch.split(prediction, [1, 1], dim=0)
        guided = (1.0 + CFG_STRENGTH) * conditional - CFG_STRENGTH * unconditional
        return noise + guided * self.delta_t[time_step]


LRELU_SLOPE = 0.1


def _same_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size * dilation - dilation) // 2


class WNConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()
        padding = _same_padding(kernel_size, dilation) if stride == 1 else 0
        convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        self.conv = nn.utils.weight_norm(convolution) if weight_norm else convolution

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)

    def remove_weight_norm(self) -> None:
        nn.utils.remove_weight_norm(self.conv)


class WNConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()
        convolution = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv = nn.utils.weight_norm(convolution) if weight_norm else convolution

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(tensor)

    def remove_weight_norm(self) -> None:
        nn.utils.remove_weight_norm(self.conv)


class ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [WNConv1d(channels, channels, kernel_size, dilation=value) for value in dilation]
        )
        self.convs2 = nn.ModuleList(
            [WNConv1d(channels, channels, kernel_size, dilation=1) for _ in dilation]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for first, second in zip(self.convs1, self.convs2):
            residual = first(F.leaky_relu(tensor, LRELU_SLOPE))
            residual = second(F.leaky_relu(residual, LRELU_SLOPE))
            tensor = tensor + residual
        return tensor

    def remove_weight_norm(self) -> None:
        for convolution in self.convs1:
            convolution.remove_weight_norm()
        for convolution in self.convs2:
            convolution.remove_weight_norm()


class ResBlock2(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [WNConv1d(channels, channels, kernel_size, dilation=value) for value in dilation]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        for convolution in self.convs:
            tensor = tensor + convolution(F.leaky_relu(tensor, LRELU_SLOPE))
        return tensor

    def remove_weight_norm(self) -> None:
        for convolution in self.convs:
            convolution.remove_weight_norm()


class HifiganGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        resblock_type: str = "1",
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        upsample_kernel_sizes: tuple[int, ...] = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        upsample_factors: tuple[int, ...] = (8, 8, 2, 2),
        inference_padding: int = 5,
        conv_post_bias: bool = True,
    ) -> None:
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.conv_pre = WNConv1d(in_channels, upsample_initial_channel, 7)
        resblock_class = ResBlock1 if resblock_type == "1" else ResBlock2
        self.ups = nn.ModuleList(
            [
                WNConvTranspose1d(
                    upsample_initial_channel // (2**index),
                    upsample_initial_channel // (2 ** (index + 1)),
                    kernel_size,
                    stride=factor,
                    padding=(kernel_size - factor) // 2,
                )
                for index, (factor, kernel_size) in enumerate(
                    zip(upsample_factors, upsample_kernel_sizes)
                )
            ]
        )
        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = upsample_initial_channel // (2 ** (index + 1))
            for kernel_size, dilation in zip(
                resblock_kernel_sizes, resblock_dilation_sizes
            ):
                self.resblocks.append(resblock_class(channels, kernel_size, dilation))
        self.conv_post = WNConv1d(channels, out_channels, 7, bias=conv_post_bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.conv_pre(tensor)
        for stage in range(self.num_upsamples):
            tensor = self.ups[stage](F.leaky_relu(tensor, LRELU_SLOPE))
            fused = self.resblocks[stage * self.num_kernels](tensor)
            for kernel_index in range(1, self.num_kernels):
                fused = fused + self.resblocks[stage * self.num_kernels + kernel_index](tensor)
            tensor = fused / self.num_kernels
        tensor = self.conv_post(F.leaky_relu(tensor))
        return torch.tanh(tensor)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()


def load_hifigan_vocoder(path: Path) -> tuple[HifiganGenerator, dict[str, float]]:
    vocoder = HifiganGenerator().eval()
    state = load_weights_file(path)
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise ValueError(f"HiFi-GAN checkpoint must be a plain state dict: {path}")
    incompatible = vocoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict HiFi-GAN load failed: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state

    generator = torch.Generator(device="cpu").manual_seed(0)
    probe = torch.randn((1, N_MELS, 7), generator=generator)
    with torch.inference_mode():
        before = vocoder.forward(probe)
        vocoder.remove_weight_norm()
        after = vocoder.forward(probe)
    difference = (before - after).abs()
    maximum = float(difference.max())
    mean = float(difference.mean())
    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
    return vocoder, {"max_error": maximum, "mean_error": mean}


class RaonDecode(nn.Module):
    def __init__(self, vocoder: HifiganGenerator, fp16_transformer: bool) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.fp16_transformer = fp16_transformer

    def forward(
        self,
        denoised: torch.Tensor,
        ref_signal_len: torch.Tensor,
        rms_scale: torch.Tensor,
    ) -> torch.Tensor:
        generated_mel = denoised[:, ref_signal_len:]
        if self.fp16_transformer:
            generated_mel = generated_mel.float()
        waveform = self.vocoder.forward(generated_mel.transpose(1, 2))
        return (waveform * rms_scale).clamp(min=-1.0, max=1.0)


class MetadataCarrier(nn.Module):
    def forward(self, marker: torch.Tensor) -> torch.Tensor:
        return marker


def build_model_metadata(assets: PreflightResult, fp16_transformer: bool) -> dict[str, str]:
    architecture = assets.architecture
    values: dict[str, Any] = {
        "schema_version": 1,
        "architecture": f"Raon-OpenTTS-{architecture.model_name}-DiT-HiFiGAN",
        "model_name": architecture.model_name,
        "backbone": "DiT",
        "vocoder": "sbhifigan16k",
        "vocoder_upsample_factor": 256,
        "model_file_name_preprocess": "Raon_Preprocess.onnx",
        "model_file_name_transformer": "Raon_Transformer.onnx",
        "model_file_name_decode": "Raon_Decode.onnx",
        "model_file_name_metadata": "Raon_Metadata.onnx",
        "checkpoint_file_name": assets.checkpoint_path.name,
        "config_file_name": assets.config_path.name,
        "vocab_file_name": assets.vocab_path.name,
        "vocoder_checkpoint_file_name": assets.vocoder_checkpoint_path.name,
        "sample_rate": MODEL_SAMPLE_RATE,
        "in_sample_rate": IN_SAMPLE_RATE,
        "out_sample_rate": OUT_SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": NFFT,
        "window_length": WINDOW_LENGTH,
        "hop_length": HOP_LENGTH,
        "window_type": WINDOW_TYPE,
        "center_pad": False,
        "mel_norm": "slaney",
        "mel_scale": "slaney",
        "nfe_step": NFE_STEP,
        "cfg_strength": CFG_STRENGTH,
        "sway_coefficient": SWAY_COEFFICIENT,
        "max_signal_length": MAX_SIGNAL_LENGTH,
        "model_dim": architecture.dim,
        "model_depth": architecture.depth,
        "model_heads": architecture.heads,
        "head_dim": architecture.head_dim,
        "ff_mult": architecture.ff_mult,
        "text_dim": architecture.text_dim,
        "text_conv_layers": architecture.text_conv_layers,
        "text_mask_padding": True,
        "qk_norm": "null",
        "pe_attn_head": "null",
        "attn_mask_enabled": False,
        "logit_softcapping": "null",
        "post_norm": False,
        "norm_type": "rmsnorm",
        "vocab_size": len(assets.vocab_map),
        "vocab_sha256": assets.vocab_sha256,
        "audio_input_dtype": "float32",
        "text_ids_dtype": "int32",
        "time_step_dtype": "int32",
        "preprocess_output_dtype": "float16" if fp16_transformer else "float32",
        "transformer_dtype": "float16" if fp16_transformer else "float32",
        "decode_dtype": "float32",
        "opset": OPSET,
    }
    return {
        str(key): "1" if value is True else "0" if value is False else str(value)
        for key, value in values.items()
    }


def write_onnx_metadata(path: Path, metadata: dict[str, str]) -> None:
    import onnx

    model = onnx.load(path, load_external_data=False)
    existing = {entry.key: entry for entry in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.metadata.tmp")
    try:
        onnx.save(model, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_output_folders(output_folder: Path, raw_folder: Path) -> None:
    for path in (output_folder, raw_folder):
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    output_folder.mkdir(parents=True)
    raw_folder.mkdir(parents=True)


def _export_onnx(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> None:
    torch.onnx.export(
        module,
        inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        external_data=True,
        opset_version=OPSET,
    )


def export_package(assets: PreflightResult) -> Path:
    from Rewrite_Raon_ONNX import rewrite_mish_subgraphs, validate_rewrite_parity

    output_folder = ONNX_FOLDER.expanduser().resolve()
    raw_folder = output_folder.with_name(f"{output_folder.name}_Raw")
    _prepare_output_folders(output_folder, raw_folder)

    preprocess_path = output_folder / "Raon_Preprocess.onnx"
    transformer_path = output_folder / "Raon_Transformer.onnx"
    raw_transformer_path = raw_folder / "Raon_Transformer.onnx"
    decode_path = output_folder / "Raon_Decode.onnx"
    metadata_path = output_folder / "Raon_Metadata.onnx"

    model = build_model(assets)
    assets.checkpoint_state.clear()
    gc.collect()

    dummy_audio_length = 16_017
    dummy_duration = 128
    audio = torch.ones((1, 1, dummy_audio_length), dtype=torch.float32)
    text_ids = torch.ones((1, 12), dtype=torch.int32)
    max_duration = torch.tensor([dummy_duration], dtype=torch.int64)
    preprocess = RaonPreprocess(
        model.transformer.text_embed,
        assets.architecture,
        use_fp16_transformer=USE_FP16_TRANSFORMER,
    ).eval()
    with torch.inference_mode():
        _export_onnx(
            preprocess,
            (audio, text_ids, max_duration),
            preprocess_path,
            ["audio", "text_ids", "max_duration"],
            [
                "noise",
                "rope_cos",
                "rope_sin",
                "cat_mel_text",
                "cat_mel_text_drop",
                "ref_signal_len",
                "rms_scale",
            ],
            {
                "audio": {2: "audio_len"},
                "text_ids": {1: "text_len"},
                "noise": {1: "max_duration"},
                "rope_cos": {2: "max_duration"},
                "rope_sin": {2: "max_duration"},
                "cat_mel_text": {1: "max_duration"},
                "cat_mel_text_drop": {1: "max_duration"},
            },
        )
    del preprocess, audio, text_ids, max_duration
    gc.collect()

    model.transformer.fuse_for_export(fp16=USE_FP16_TRANSFORMER)
    transformer = RaonTransformer(
        model.transformer,
        fp16=USE_FP16_TRANSFORMER,
    ).eval()
    if USE_FP16_TRANSFORMER:
        transformer = transformer.half()
    transformer_dtype = torch.float16 if USE_FP16_TRANSFORMER else torch.float32
    state = torch.ones((1, dummy_duration, N_MELS), dtype=transformer_dtype)
    rope_cos = torch.ones(
        (2, 1, dummy_duration, assets.architecture.head_dim),
        dtype=transformer_dtype,
    )
    rope_sin = torch.ones_like(rope_cos)
    condition = torch.ones(
        (1, dummy_duration, N_MELS + assets.architecture.text_dim),
        dtype=transformer_dtype,
    )
    dropped_condition = torch.ones_like(condition)
    time_step = torch.tensor([0], dtype=torch.int32)
    assert state.shape[0] == condition.shape[0] == dropped_condition.shape[0] == 1
    assert time_step.ndim == 1 and time_step.shape[0] == 1
    with torch.inference_mode():
        _export_onnx(
            transformer,
            (state, rope_cos, rope_sin, condition, dropped_condition, time_step),
            raw_transformer_path,
            [
                "noise",
                "rope_cos",
                "rope_sin",
                "cat_mel_text",
                "cat_mel_text_drop",
                "time_step",
            ],
            ["denoised"],
            {
                "noise": {1: "max_duration"},
                "rope_cos": {2: "max_duration"},
                "rope_sin": {2: "max_duration"},
                "cat_mel_text": {1: "max_duration"},
                "cat_mel_text_drop": {1: "max_duration"},
                "denoised": {1: "max_duration"},
            },
        )
    del transformer, model, state, rope_cos, rope_sin, condition, dropped_condition, time_step
    gc.collect()

    vocoder, weight_norm_parity = load_hifigan_vocoder(assets.vocoder_checkpoint_path)
    decode = RaonDecode(vocoder, fp16_transformer=USE_FP16_TRANSFORMER).eval()
    denoised = torch.ones((1, dummy_duration, N_MELS), dtype=transformer_dtype)
    ref_signal_len = torch.tensor(dummy_audio_length // HOP_LENGTH, dtype=torch.int64)
    rms_scale = torch.ones((1,), dtype=torch.float32)
    with torch.inference_mode():
        _export_onnx(
            decode,
            (denoised, ref_signal_len, rms_scale),
            decode_path,
            ["denoised", "ref_signal_len", "rms_scale"],
            ["output_audio"],
            {
                "denoised": {1: "max_duration"},
                "output_audio": {2: "audio_len"},
            },
        )
    del decode, vocoder, denoised, ref_signal_len, rms_scale
    gc.collect()

    marker = torch.zeros((1,), dtype=torch.int64)
    _export_onnx(
        MetadataCarrier(),
        (marker,),
        metadata_path,
        ["metadata_marker"],
        ["metadata_marker_out"],
        None,
    )

    metadata = build_model_metadata(assets, USE_FP16_TRANSFORMER)
    for path in (preprocess_path, raw_transformer_path, decode_path, metadata_path):
        write_onnx_metadata(path, metadata)

    rewrite_report = rewrite_mish_subgraphs(
        raw_transformer_path,
        transformer_path,
        expected_matches=2,
    )
    rewrite_parity = validate_rewrite_parity(
        raw_transformer_path,
        transformer_path,
        duration=8,
        fp16=USE_FP16_TRANSFORMER,
    )
    shutil.rmtree(raw_folder)

    print(
        "HiFi-GAN weight-normalization parity: "
        f"max_error={weight_norm_parity['max_error']:.8g} "
        f"mean_error={weight_norm_parity['mean_error']:.8g}"
    )
    print(
        "Transformer rewrite parity: "
        f"matches={rewrite_report['matched_subgraphs']} "
        f"max_error={rewrite_parity['max_error']:.8g} "
        f"mean_error={rewrite_parity['mean_error']:.8g}"
    )
    for path in (preprocess_path, transformer_path, decode_path, metadata_path):
        print(f"Exported {path.name}: {path.stat().st_size} bytes")
    print(f"Exported Raon package: {output_folder}")
    return output_folder


def main() -> tuple[Path, Path]:
    try:
        checkpoint_path = (
            discover_checkpoint(CONFIG_PATH.parent)
            if CHECKPOINT_PATH is None
            else CHECKPOINT_PATH
        )
        vocoder_checkpoint_path = (
            discover_vocoder_checkpoint(CONFIG_PATH.parent)
            if VOCODER_CHECKPOINT_PATH is None
            else VOCODER_CHECKPOINT_PATH
        )
        assets = preflight_assets(
            checkpoint_path,
            CONFIG_PATH,
            VOCAB_PATH,
            vocoder_checkpoint_path,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Export preflight failed: {error}") from error
    return export_package(assets), assets.vocab_path


if __name__ == "__main__":
    onnx_folder, vocab_path = main()
    print("\nStart running the Raon-OpenTTS demo via Inference_Raon_OpenTTS_ONNX.py ...")
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(SCRIPT_DIR / "Inference_Raon_OpenTTS_ONNX.py"),
                "--onnx-folder",
                str(onnx_folder),
                "--vocab-path",
                str(vocab_path),
            ]
        )
    )