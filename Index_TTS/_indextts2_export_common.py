"""Export the official IndexTTS2 inference modules to a compact ONNX package.

The exporter keeps text normalization, BPE/Qwen tokenization, audio loading,
and loop control in Python. Every neural forward used by IndexTTS2 synthesis,
including optional Qwen emotion-text classification, is exported to ONNX.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import gc
import importlib
import math
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


QWEN_TOKENIZER_FOLDER = "qwen0.6bemo4-merge"
QWEN_TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class ExportProfile:
    """Static configuration for one compatible IndexTTS2 model release."""

    model_version: str
    script_dir: Path
    project_path: Path
    models_path: Path
    output_folder_name: str
    text_tokenizer_file: str
    stft_process: type[nn.Module]
    max_signal_length: int = 2048
    use_f16_kv: bool = True
    compute_in_f32: bool = False
    opset: int = 20
    cfm_steps: int = 25
    in_sample_rate: int = 22050
    out_sample_rate: int = 22050
    in_audio_dtype: str = "F32"
    out_audio_dtype: str = "F32"
    emotion_text_max_seq_length: int = 1024
    emotion_text_reorder_downproj: bool = True
    emotion_text_reorder_key: str = "absmean"
    emotion_text_kv_dtype: str = "F16"

    @property
    def emotion_text_model_path(self) -> Path:
        return self.models_path / QWEN_TOKENIZER_FOLDER

    @property
    def onnx_folder(self) -> Path:
        return self.script_dir / self.output_folder_name


_profile: ExportProfile | None = None
_adapter: "ExportAdapter | None" = None
STFT_Process: type[nn.Module] | None = None

MAX_SIGNAL_LENGTH = 2048
USE_F16_KV = True
COMPUTE_IN_F32 = False
OPSET = 20
CFM_STEPS = 25
IN_SAMPLE_RATE = 22050
OUT_SAMPLE_RATE = 22050
IN_AUDIO_DTYPE = "F32"
OUT_AUDIO_DTYPE = "F32"
EMOTION_TEXT_MAX_SEQ_LENGTH = 1024
EMOTION_TEXT_REORDER_DOWNPROJ = True
EMOTION_TEXT_REORDER_KEY = "absmean"
EMOTION_TEXT_KV_DTYPE = "F16"


_AUDIO_DTYPES = {"F16": torch.float16, "F32": torch.float32, "INT16": torch.int16}
script_dir: Path
project_path: Path
models_path: Path
emotion_text_model_path: Path
TEXT_TOKENIZER_FILE: str
onnx_folder: Path
repo_root: Path

from Index_TTS.v2.Shared_Weights import (  # noqa: E402
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    build_decoder_postprocess_graph,
    build_decode_step_graphs,
    build_reference_preprocess_graph,
    build_synthesis_graph,
    build_target_prefill_graphs,
    bundle_shared_initializers,
)


def model_path(name: str) -> str:
    return str(onnx_folder / f"IndexTTS2_{name}.onnx")


def copy_tokenizer_assets(destination: Path) -> None:
    text_tokenizer = models_path / TEXT_TOKENIZER_FILE
    if not text_tokenizer.is_file():
        raise FileNotFoundError(f"Missing text tokenizer: {text_tokenizer}")
    shutil.copy2(text_tokenizer, destination / TEXT_TOKENIZER_FILE)

    source_folder = emotion_text_model_path
    target_folder = destination / QWEN_TOKENIZER_FOLDER
    target_folder.mkdir(parents=True, exist_ok=True)
    for name in QWEN_TOKENIZER_FILES:
        source = source_folder / name
        if source.is_file():
            shutil.copy2(source, target_folder / name)
    if not (target_folder / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Missing Qwen tokenizer: {source_folder}")


onnx_models: dict[str, str]
onnx_model_reference_preprocess: str
onnx_model_synthesis: str
onnx_model_emotion_text_prefill: str
onnx_model_emotion_text_decode: str
onnx_model_main_prefill: dict[str, str]
onnx_model_target_prefill: dict[str, str]
onnx_model_main_decode: dict[str, str]
onnx_model_decode_step: dict[str, str]


def _configure_graph_paths() -> None:
    global onnx_models
    global onnx_model_reference_preprocess
    global onnx_model_synthesis
    global onnx_model_emotion_text_prefill
    global onnx_model_emotion_text_decode
    global onnx_model_main_prefill
    global onnx_model_target_prefill
    global onnx_model_main_decode
    global onnx_model_decode_step

    onnx_models = {
        "feature_extractor": model_path("FeatureExtractor"),
        "semantic_encoder": model_path("SemanticEncoder"),
        "reference": model_path("Reference"),
        "conditioning": model_path("Conditioning"),
        "target": model_path("TargetPreprocess"),
        "decode_embed": model_path("DecodeEmbed"),
        "latent": model_path("Latent"),
        "acoustic": model_path("Acoustic"),
        "cfm_estimator": model_path("CFMEstimator"),
        "decoder": model_path("Decoder"),
        "metadata": model_path("Metadata"),
    }
    onnx_model_reference_preprocess = model_path("ReferencePreprocess")
    onnx_model_synthesis = model_path("Synthesis")
    onnx_model_emotion_text_prefill = model_path("EmotionTextPrefill")
    onnx_model_emotion_text_decode = model_path("EmotionTextDecode")
    onnx_model_main_prefill = {
        strategy: model_path(f"MainPrefill_{strategy}")
        for strategy in ("greedy", "penalty_greedy", "sampling")
    }
    onnx_model_target_prefill = {
        strategy: model_path(f"TargetPrefill_{strategy}")
        for strategy in onnx_model_main_prefill
    }
    onnx_model_main_decode = {
        strategy: model_path(f"MainDecode_{strategy}")
        for strategy in onnx_model_main_prefill
    }
    onnx_model_decode_step = {
        strategy: model_path(f"DecodeStep_{strategy}")
        for strategy in onnx_model_main_prefill
    }


def active_profile() -> ExportProfile:
    if _profile is None:
        raise RuntimeError("Configure an IndexTTS2 export profile before exporting.")
    return _profile


def active_adapter() -> "ExportAdapter":
    if _adapter is None:
        raise RuntimeError("Configure an IndexTTS2 export adapter before exporting.")
    return _adapter


class ExportAdapter:
    """Version-owned graph adapters; the default preserves the v2.5 contract."""

    def resolve_auxiliary_paths(self, config: Any) -> dict[str, str]:
        return resolve_auxiliary_paths(config)

    def load_gpt(self, config: Any) -> nn.Module:
        return load_gpt(config)

    def load_acoustic_modules(
        self,
        config: Any,
        auxiliary_paths: dict[str, str],
    ) -> tuple[nn.Module, nn.Module, nn.Module]:
        return load_acoustic_modules(config, auxiliary_paths)

    def make_reference(
        self,
        semantic_codec: nn.Module,
        s2mel: nn.Module,
        campplus: nn.Module,
        cfm_projection: nn.Module,
    ) -> nn.Module:
        del semantic_codec
        return IndexTTS2Reference(
            s2mel.models["length_regulator"],
            campplus,
            cfm_projection,
        )

    def make_conditioning(
        self,
        gpt: nn.Module,
        speaker_matrix: torch.Tensor,
        emotion_matrix: torch.Tensor,
    ) -> nn.Module:
        return IndexTTS2Conditioning(gpt, speaker_matrix, emotion_matrix)

    def target_export(
        self,
        gpt: nn.Module,
        speaker_latent: torch.Tensor,
        emotion_vector: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
        language_id = torch.tensor([0], dtype=torch.int64)
        return (
            IndexTTS2TargetPreprocess(gpt),
            (speaker_latent, emotion_vector, text_ids, language_id),
            ["speaker_latent", "emotion_vector", "text_ids", "language_id"],
        )

    def speaker_latent_example(self, hidden_size: int) -> torch.Tensor:
        return torch.zeros(1, 1, hidden_size)

    def make_latent(
        self,
        gpt: nn.Module,
        main_core: nn.Module,
        config: Any,
        semantic_hidden_size: int,
    ) -> nn.Module:
        del gpt, main_core
        return IndexTTS2LatentBypass(
            semantic_hidden_size,
            int(getattr(config.semantic_codec, "downsample_scale", 2)),
        )

    def acoustic_export(
        self,
        semantic_codec: nn.Module,
        s2mel: nn.Module,
        cfm_projection: nn.Module,
        config: Any,
        style_embed_size: int,
        semantic_hidden_size: int,
    ) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
        acoustic = IndexTTS2AcousticConditioning(
            semantic_codec,
            None,
            s2mel.models["length_regulator"],
            cfm_projection,
        )
        mel_codes = torch.zeros(1, 20, dtype=torch.int32)
        gpt_latent = torch.zeros(1, 40, semantic_hidden_size)
        cfg_rate = torch.tensor([0.7], dtype=torch.float32)
        duration_factor = torch.tensor([1.0], dtype=torch.float32)
        style = torch.zeros(1, style_embed_size)
        reference_hidden = torch.zeros(1, 100, int(config.s2mel.DiT.hidden_dim))
        return (
            acoustic,
            (
                mel_codes,
                gpt_latent,
                cfg_rate,
                duration_factor,
                style,
                reference_hidden,
                cfm_projection.null_hidden,
            ),
            [
                "mel_codes",
                "gpt_latent",
                "cfg_rate",
                "duration_factor",
                "style",
                "reference_hidden",
                "null_hidden",
            ],
        )

    def cfm_time_schedule(
        self,
        steps: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_span = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32, device=device)
        euler_deltas = time_span[1:] - time_span[:-1]
        step_times = []
        current_time = time_span[0]
        for euler_delta in euler_deltas:
            step_times.append(current_time)
            current_time = current_time + euler_delta
        return torch.stack(step_times), euler_deltas

    def metadata_fields(self, config: Any) -> dict[str, Any]:
        return {
            "model_version": active_profile().model_version,
            "speaker_conditioning_mode": "campplus",
            "target_language_embedding": True,
            "use_gpt_latent": False,
            "max_text_tokens": int(config.gpt.max_text_tokens),
        }


def configure_export(profile: ExportProfile, adapter: ExportAdapter | None = None) -> None:
    """Bind one release profile before constructing any exporter graph."""
    global _profile, _adapter, STFT_Process
    global script_dir, project_path, models_path, emotion_text_model_path
    global TEXT_TOKENIZER_FILE, onnx_folder, repo_root
    global MAX_SIGNAL_LENGTH, USE_F16_KV, COMPUTE_IN_F32, OPSET, CFM_STEPS
    global IN_SAMPLE_RATE, OUT_SAMPLE_RATE, IN_AUDIO_DTYPE, OUT_AUDIO_DTYPE
    global EMOTION_TEXT_MAX_SEQ_LENGTH, EMOTION_TEXT_REORDER_DOWNPROJ
    global EMOTION_TEXT_REORDER_KEY, EMOTION_TEXT_KV_DTYPE

    _profile = profile
    _adapter = adapter or ExportAdapter()
    script_dir = profile.script_dir.expanduser().resolve()
    project_path = profile.project_path.expanduser().resolve()
    models_path = profile.models_path.expanduser().resolve()
    emotion_text_model_path = profile.emotion_text_model_path
    TEXT_TOKENIZER_FILE = profile.text_tokenizer_file
    onnx_folder = profile.onnx_folder
    repo_root = script_dir.parent.parent
    STFT_Process = profile.stft_process
    MAX_SIGNAL_LENGTH = profile.max_signal_length
    USE_F16_KV = profile.use_f16_kv
    COMPUTE_IN_F32 = profile.compute_in_f32
    OPSET = profile.opset
    CFM_STEPS = profile.cfm_steps
    IN_SAMPLE_RATE = profile.in_sample_rate
    OUT_SAMPLE_RATE = profile.out_sample_rate
    IN_AUDIO_DTYPE = profile.in_audio_dtype
    OUT_AUDIO_DTYPE = profile.out_audio_dtype
    EMOTION_TEXT_MAX_SEQ_LENGTH = profile.emotion_text_max_seq_length
    EMOTION_TEXT_REORDER_DOWNPROJ = profile.emotion_text_reorder_downproj
    EMOTION_TEXT_REORDER_KEY = profile.emotion_text_reorder_key
    EMOTION_TEXT_KV_DTYPE = profile.emotion_text_kv_dtype
    for import_path in (script_dir, project_path, repo_root):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))
    _configure_graph_paths()


def build_model_metadata(*sections: dict[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}
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


class METADATA_CARRIER(nn.Module):
    """Tiny identity graph that carries the static package contract."""

    def forward(self, marker: torch.Tensor) -> torch.Tensor:
        return marker


class ONNXStaticReshape(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        values: torch.Tensor,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        eager_shape = tuple(
            values.shape[index] if dimension == 0 else dimension
            for index, dimension in enumerate(shape)
        )
        return values.reshape(eager_shape)

    @staticmethod
    def symbolic(graph: Any, values: Any, shape: tuple[int, ...]) -> Any:
        shape_constant = graph.op(
            "Constant",
            value_t=torch.tensor(shape, dtype=torch.int64),
        )
        return graph.op("Reshape", values, shape_constant)


def onnx_static_reshape(
    values: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return ONNXStaticReshape.apply(values, shape)


class IndexTTS2EmotionTextCore(nn.Module):
    """Qwen3 core with folded norms, fused projections, and floating-point KV state."""

    def __init__(self, model: nn.Module, max_sequence_length: int) -> None:
        super().__init__()
        self.transformer = model.model.eval()
        self.lm_head = model.lm_head.eval()
        self.num_layers = int(model.config.num_hidden_layers)
        self.num_heads = int(model.config.num_attention_heads)
        self.num_kv_heads = int(model.config.num_key_value_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = int(model.config.head_dim)
        self.hidden_size = int(model.config.hidden_size)
        self.attention_size = self.num_heads * self.head_dim
        self.qk_heads = self.num_heads + self.num_kv_heads
        self.total_qkv_heads = self.qk_heads + self.num_kv_heads
        self.compute_in_f32 = COMPUTE_IN_F32
        self.use_f16_kv = EMOTION_TEXT_KV_DTYPE == "F16"
        hidden_norm = self.transformer.layers[0].input_layernorm
        qk_norm = self.transformer.layers[0].self_attn.q_norm
        self.hidden_norm_epsilon = float(
            getattr(
                hidden_norm,
                "variance_epsilon",
                getattr(hidden_norm, "eps", 1.0e-6),
            )
        )
        self.qk_norm_epsilon = float(
            getattr(
                qk_norm,
                "variance_epsilon",
                getattr(qk_norm, "eps", self.hidden_norm_epsilon),
            )
        )
        final_norm = self.transformer.norm
        self.final_norm_epsilon = float(
            getattr(
                final_norm,
                "variance_epsilon",
                getattr(final_norm, "eps", self.hidden_norm_epsilon),
            )
        )
        self.hidden_norm_epsilon_sum = self.hidden_size * self.hidden_norm_epsilon
        self.qk_norm_epsilon_sum = self.head_dim * self.qk_norm_epsilon
        self.final_norm_epsilon_sum = self.hidden_size * self.final_norm_epsilon
        self.register_buffer(
            "final_norm_scale",
            final_norm.weight.detach().float().clone() * self.hidden_size**0.5,
        )

        with torch.no_grad():
            for layer in self.transformer.layers:
                attention = layer.self_attn
                projections = (
                    attention.q_proj,
                    attention.k_proj,
                    attention.v_proj,
                )
                output_sizes = [projection.out_features for projection in projections]
                expected_output_sizes = [
                    self.attention_size,
                    self.num_kv_heads * self.head_dim,
                    self.num_kv_heads * self.head_dim,
                ]
                fused_qkv = nn.Linear(
                    projections[0].in_features,
                    sum(output_sizes),
                    bias=any(projection.bias is not None for projection in projections),
                    device=projections[0].weight.device,
                    dtype=projections[0].weight.dtype,
                )
                fused_qkv.weight.copy_(
                    torch.cat([projection.weight for projection in projections], dim=0)
                )
                if fused_qkv.bias is not None:
                    fused_qkv.bias.copy_(
                        torch.cat(
                            [
                                projection.bias
                                if projection.bias is not None
                                else torch.zeros(
                                    projection.out_features,
                                    device=projection.weight.device,
                                    dtype=projection.weight.dtype,
                                )
                                for projection in projections
                            ],
                            dim=0,
                        )
                    )
                fused_qkv.weight.mul_(
                    layer.input_layernorm.weight.detach().unsqueeze(0)
                    * self.hidden_size**0.5
                )
                attention.onnx_qkv = fused_qkv
                q_norm_weight = attention.q_norm.weight.detach().repeat(
                    self.num_heads
                )
                k_norm_weight = attention.k_norm.weight.detach().repeat(
                    self.num_kv_heads
                )
                attention.register_buffer(
                    "onnx_qk_norm_weight",
                    torch.cat((q_norm_weight, k_norm_weight), dim=0).view(
                        1,
                        1,
                        self.qk_heads,
                        self.head_dim,
                    )
                    * self.head_dim**0.25,
                )
                del (
                    attention.q_proj,
                    attention.k_proj,
                    attention.v_proj,
                    attention.q_norm,
                    attention.k_norm,
                    layer.input_layernorm,
                )

                mlp = layer.mlp
                fused_gate_up = nn.Linear(
                    mlp.gate_proj.in_features,
                    mlp.gate_proj.out_features + mlp.up_proj.out_features,
                    bias=False,
                    device=mlp.gate_proj.weight.device,
                    dtype=mlp.gate_proj.weight.dtype,
                )
                fused_gate_up.weight.copy_(
                    torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight), dim=0)
                    * (
                        layer.post_attention_layernorm.weight.detach().unsqueeze(0)
                        * self.hidden_size**0.5
                    )
                )
                mlp.onnx_gate_up = fused_gate_up
                mlp.onnx_intermediate_size = int(mlp.down_proj.in_features)
                if EMOTION_TEXT_REORDER_DOWNPROJ:
                    down_weight = mlp.down_proj.weight.detach()
                    absolute = down_weight.abs()
                    if EMOTION_TEXT_REORDER_KEY == "rms":
                        importance = down_weight.square().mean(dim=0).sqrt()
                    elif EMOTION_TEXT_REORDER_KEY == "L4":
                        importance = absolute.pow(4).mean(dim=0).pow(0.25)
                    elif EMOTION_TEXT_REORDER_KEY == "std":
                        importance = down_weight.std(dim=0)
                    else:
                        importance = absolute.mean(dim=0)
                    permutation = torch.argsort(importance)
                    intermediate_size = mlp.onnx_intermediate_size
                    gate_up_weight = fused_gate_up.weight.detach()
                    fused_gate_up.weight.copy_(
                        torch.cat(
                            (
                                gate_up_weight[:intermediate_size][permutation],
                                gate_up_weight[intermediate_size:][permutation],
                            ),
                            dim=0,
                        )
                    )
                    mlp.down_proj.weight.copy_(down_weight[:, permutation])
                del mlp.gate_proj, mlp.up_proj, layer.post_attention_layernorm

            positions = torch.arange(
                max_sequence_length,
                dtype=torch.float32,
                device=self.transformer.rotary_emb.inv_freq.device,
            )
            frequencies = torch.outer(
                positions,
                self.transformer.rotary_emb.inv_freq.detach().float(),
            )
            rotary_cos = torch.cos(frequencies)
            rotary_sin = torch.sin(frequencies)
            self.register_buffer(
                "rotary_cos",
                torch.cat((rotary_cos, rotary_cos), dim=-1)
                .half()
                .view(1, max_sequence_length, 1, self.head_dim),
            )
            self.register_buffer(
                "rotary_sin",
                torch.cat((-rotary_sin, rotary_sin), dim=-1)
                .half()
                .view(1, max_sequence_length, 1, self.head_dim),
            )
            self.register_buffer(
                "mask_row_positions",
                torch.arange(max_sequence_length, dtype=torch.int32).view(
                    max_sequence_length,
                    1,
                ),
                persistent=False,
            )
            self.register_buffer(
                "mask_column_positions",
                torch.arange(max_sequence_length, dtype=torch.int32).view(
                    1,
                    max_sequence_length,
                ),
                persistent=False,
            )
            mask_dtype = (
                torch.float16
                if self.use_f16_kv and not self.compute_in_f32
                else torch.float32
            )
            self.register_buffer(
                "mask_zero",
                torch.tensor(0.0, dtype=mask_dtype),
                persistent=False,
            )
            self.register_buffer(
                "mask_negative",
                torch.tensor(-65504.0, dtype=mask_dtype),
                persistent=False,
            )
            del self.transformer.norm

    def _rms_norm(
        self,
        values: torch.Tensor,
        epsilon_sum: float,
    ) -> torch.Tensor:
        values_float = values.float()
        squared_sum = values_float.square().sum(dim=-1, keepdim=True)
        return values_float * torch.rsqrt(squared_sum + epsilon_sum)

    def _rotate_half(self, values: torch.Tensor) -> torch.Tensor:
        values = onnx_static_reshape(
            values,
            (1, -1, self.qk_heads, 2, self.head_dim // 2),
        )
        return onnx_static_reshape(
            values.flip(-2),
            (1, -1, self.qk_heads, self.head_dim),
        )

    def forward(self, *all_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        input_ids = all_inputs[-2]
        history_length = all_inputs[-1]
        input_length = torch._shape_as_tensor(input_ids)[1:2]
        kv_sequence_length = history_length + input_length
        rotary_cos = self.rotary_cos[:, history_length:kv_sequence_length].float()
        rotary_sin = self.rotary_sin[:, history_length:kv_sequence_length].float()
        mask_rows = self.mask_row_positions[
            history_length:kv_sequence_length
        ]
        mask_columns = self.mask_column_positions[:, :kv_sequence_length]
        attention_mask = torch.where(
            mask_columns <= mask_rows,
            self.mask_zero,
            self.mask_negative,
        ).view(1, 1, input_length, kv_sequence_length)
        hidden_states = self.transformer.embed_tokens(input_ids)
        saved_keys = []
        saved_values = []

        for index, layer in enumerate(self.transformer.layers):
            residual = hidden_states
            normalized = self._rms_norm(
                hidden_states,
                self.hidden_norm_epsilon_sum,
            )
            qkv = layer.self_attn.onnx_qkv(normalized)
            qkv = onnx_static_reshape(
                qkv,
                (1, -1, self.total_qkv_heads, self.head_dim),
            )
            query_key, value = torch.split(
                qkv,
                (self.qk_heads, self.num_kv_heads),
                dim=-2,
            )
            query_key = self._rms_norm(
                query_key,
                self.qk_norm_epsilon_sum,
            ) * layer.self_attn.onnx_qk_norm_weight
            query_key = (
                query_key * rotary_cos
                + self._rotate_half(query_key) * rotary_sin
            )
            if self.use_f16_kv and not self.compute_in_f32:
                query_key = query_key.half()
            query, key = torch.split(
                query_key,
                (self.num_heads, self.num_kv_heads),
                dim=-2,
            )
            query = onnx_static_reshape(
                query,
                (
                    1,
                    -1,
                    self.num_kv_heads,
                    self.num_kv_groups,
                    self.head_dim,
                ),
            ).permute(0, 2, 3, 1, 4)
            key = key.permute(0, 2, 3, 1)
            value = value.permute(0, 2, 1, 3)
            if self.use_f16_kv:
                if self.compute_in_f32:
                    key = key.half()
                value = value.half()
            key = torch.cat((all_inputs[index], key), dim=-1)
            value = torch.cat(
                (all_inputs[index + self.num_layers], value),
                dim=-2,
            )
            saved_keys.append(key)
            saved_values.append(value)

            if self.use_f16_kv and self.compute_in_f32:
                scores = torch.matmul(query, key.unsqueeze(2).float())
                attended = torch.matmul(
                    torch.softmax(scores + attention_mask, dim=-1),
                    value.unsqueeze(2).float(),
                )
            else:
                scores = torch.matmul(query, key.unsqueeze(2))
                attended = torch.matmul(
                    torch.softmax(scores + attention_mask, dim=-1),
                    value.unsqueeze(2),
                )
                if self.use_f16_kv:
                    attended = attended.float()
            attended = onnx_static_reshape(
                attended.permute(0, 3, 1, 2, 4),
                (1, -1, self.attention_size),
            )
            hidden_states = residual + layer.self_attn.o_proj(attended)

            residual = hidden_states
            normalized = self._rms_norm(
                hidden_states,
                self.hidden_norm_epsilon_sum,
            )
            gate, up = torch.split(
                layer.mlp.onnx_gate_up(normalized),
                layer.mlp.onnx_intermediate_size,
                dim=-1,
            )
            hidden_states = residual + layer.mlp.down_proj(
                layer.mlp.act_fn(gate) * up
            )

        hidden_states = self._rms_norm(
            hidden_states[:, -1],
            self.final_norm_epsilon_sum,
        ) * self.final_norm_scale
        logits = self.lm_head(hidden_states)
        next_token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return (
            *saved_keys,
            *saved_values,
            next_token,
            kv_sequence_length,
        )


class IndexTTS2EmotionTextDecode(nn.Module):
    def __init__(self, core: IndexTTS2EmotionTextCore) -> None:
        super().__init__()
        self.core = core

    def forward(self, *all_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.core(*all_inputs)


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def clear_module(*modules: object) -> None:
    del modules
    gc.collect()


def _register_inline_module(
    module_name: str,
    symbols: dict[str, object],
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__dict__.update(symbols)
    module.__package__ = module_name.rpartition(".")[0]
    sys.modules[module_name] = module
    parent_name, _, attribute_name = module_name.rpartition(".")
    parent_module = importlib.import_module(parent_name)
    setattr(parent_module, attribute_name, module)
    return module


def _assert_device_map(device_map: dict[Any, list[int]], num_blocks: int) -> None:
    assigned_blocks = [block for blocks in device_map.values() for block in blocks]
    duplicates = sorted(
        {block for block in assigned_blocks if assigned_blocks.count(block) > 1}
    )
    missing = sorted(set(range(num_blocks)) - set(assigned_blocks))
    extra = sorted(set(assigned_blocks) - set(range(num_blocks)))
def _get_device_map(num_blocks: int, devices: Any) -> dict[Any, list[int]]:
    devices = list(devices)
    if not devices:
        return {"cpu": list(range(num_blocks))}
    blocks_per_device = math.ceil(num_blocks / len(devices))
    return {
        device: list(range(start, min(start + blocks_per_device, num_blocks)))
        for device_index, device in enumerate(devices)
        if (start := device_index * blocks_per_device) < num_blocks
    }


def _install_transformers_compatibility_modules() -> None:
    """Use maintained Transformers GPT-2 classes instead of vendored copies."""
    from transformers.models.gpt2.modeling_gpt2 import (
        GPT2Model,
        GPT2PreTrainedModel,
    )

    model_parallel_module = "transformers.utils.model_parallel_utils"
    try:
        parallel_module = importlib.import_module(model_parallel_module)
    except ModuleNotFoundError as error:
        parallel_module = None
    if parallel_module is None or not all(
        hasattr(parallel_module, name) for name in ("assert_device_map", "get_device_map")
    ):
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


def write_onnx_metadata(onnx_path: str | Path, metadata: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, str(onnx_path))


def replace_onnx_metadata(
    onnx_path: str | Path,
    metadata: dict[str, str],
) -> None:
    """Replace the metadata carrier contract without loading external weights."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, str(onnx_path))


def export_onnx(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_path: str,
    *,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
) -> None:
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
            opset_version=OPSET,
        )


class IndexTTS2SemanticEncoder(nn.Module):
    """Match ``IndexTTS2.get_emb`` from precomputed feature-extractor inputs."""

    def __init__(
        self,
        semantic_model: nn.Module,
        semantic_mean: torch.Tensor,
        semantic_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.semantic_model = semantic_model.eval()
        self.register_buffer("semantic_mean", semantic_mean.detach().float())
        self.register_buffer("semantic_std", semantic_std.detach().float())

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        features = outputs.hidden_states[17]
        return (features - self.semantic_mean) / self.semantic_std


class IndexTTS2FeatureExtractor(nn.Module):
    """Build Wav2Vec2-BERT, CAMPPlus, and reference-mel inputs from PCM."""

    def __init__(self, feature_extractor: Any, config: Any) -> None:
        super().__init__()
        import torchaudio
        from librosa.filters import mel as librosa_mel_fn

        spect_params = config.s2mel.preprocess_params.spect_params
        model_sample_rate = int(config.s2mel.preprocess_params.sr)

        self.input_resample_scale = float(model_sample_rate / IN_SAMPLE_RATE)
        self.resample = torchaudio.transforms.Resample(model_sample_rate, 16000)
        self.semantic_stft = STFT_Process(
            model_type="stft_B",
            n_fft=512,
            win_length=400,
            hop_len=160,
            max_frames=0,
            window_type="povey",
            center_pad=False,
            analysis_length=400,
            remove_dc_offset=True,
            preemphasis=0.97,
        )
        reference_nfft = int(spect_params.n_fft)
        reference_hop = int(spect_params.hop_length)
        reference_window = int(spect_params.win_length)
        self.reference_pad = (reference_nfft - reference_hop) // 2
        self.reference_stft = STFT_Process(
            model_type="stft_B",
            n_fft=reference_nfft,
            win_length=reference_window,
            hop_len=reference_hop,
            max_frames=0,
            window_type="hann",
            center_pad=False,
        )

        semantic_filters = torch.from_numpy(
            feature_extractor.mel_filters.astype("float32", copy=False)
        )
        kaldi_filters = torchaudio.compliance.kaldi.get_mel_banks(
            num_bins=80,
            window_length_padded=512,
            sample_freq=16000.0,
            low_freq=20.0,
            high_freq=0.0,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )[0]
        kaldi_filters = F.pad(kaldi_filters, (0, 1)).transpose(0, 1)
        reference_filters = librosa_mel_fn(
            sr=model_sample_rate,
            n_fft=reference_nfft,
            n_mels=int(spect_params.n_mels),
            fmin=float(spect_params.get("fmin", 0)),
            fmax=(
                None
                if str(spect_params.get("fmax", "None")) == "None"
                else 8000.0
            ),
        )
        self.register_buffer("semantic_filters", semantic_filters.contiguous())
        self.register_buffer("kaldi_filters", kaldi_filters.float().contiguous())
        self.register_buffer(
            "reference_filters",
            torch.from_numpy(reference_filters).float().contiguous(),
        )
        self.register_buffer(
            "semantic_padding",
            torch.full(
                (1, 1, int(feature_extractor.num_mel_bins)),
                float(feature_extractor.padding_value),
            ),
            persistent=False,
        )
        self.register_buffer(
            "attention_padding",
            torch.zeros(1, 1, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_22k = audio.float()
        if "int" in IN_AUDIO_DTYPE.lower():
            audio_22k = audio_22k * (1.0 / 32768.0)
        if self.input_resample_scale != 1.0:
            audio_22k = F.interpolate(
                audio_22k,
                scale_factor=self.input_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        audio_16k = self.resample(audio_22k)

        real_16k, imag_16k = self.semantic_stft(audio_16k)
        power_16k = real_16k.square() + imag_16k.square()
        semantic_power = power_16k * float(32768**2)
        semantic_mel = torch.matmul(
            semantic_power.transpose(1, 2),
            self.semantic_filters,
        )
        semantic_mel = torch.log(
            torch.clamp(semantic_mel, min=1.192092955078125e-7)
        )
        centered = semantic_mel - semantic_mel.mean(dim=1, keepdim=True)
        frame_count = torch._shape_as_tensor(centered)[1:2]
        variance = centered.square().sum(dim=1, keepdim=True)
        variance = variance / (frame_count - 1).to(centered.dtype).view(1, 1, 1)
        normalized = centered / torch.sqrt(variance + 1.0e-7)

        attention_mask = torch.ones_like(
            semantic_mel[:, :, 0],
            dtype=torch.int64,
        )
        padded_count = frame_count + torch.remainder(frame_count, 2)
        normalized = torch.cat((normalized, self.semantic_padding), dim=1)
        normalized = normalized[:, :padded_count]
        attention_mask = torch.cat(
            (attention_mask, self.attention_padding),
            dim=1,
        )
        attention_mask = attention_mask[:, :padded_count]
        input_features = normalized.reshape(1, -1, 160)
        attention_mask = attention_mask.reshape(1, -1, 2)[:, :, 1]

        style_features = torch.matmul(
            power_16k.transpose(1, 2),
            self.kaldi_filters,
        )
        style_features = torch.log(
            torch.clamp(style_features, min=torch.finfo(torch.float32).eps)
        )
        style_features = style_features - style_features.mean(
            dim=1,
            keepdim=True,
        )

        padded_audio = F.pad(
            audio_22k,
            (self.reference_pad, self.reference_pad),
            mode="reflect",
        )
        reference_real, reference_imag = self.reference_stft(padded_audio)
        reference_magnitude = torch.sqrt(
            reference_real.square() + reference_imag.square() + 1.0e-9
        )
        reference_mel = torch.matmul(
            self.reference_filters,
            reference_magnitude,
        )
        reference_mel = torch.log(torch.clamp(reference_mel, min=1.0e-5))
        return input_features, attention_mask, reference_mel, style_features


class IndexTTS2Reference(nn.Module):
    """Build the official continuous reference condition and CAMPPlus style."""

    def __init__(
        self,
        length_regulator: nn.Module,
        campplus: nn.Module,
        cfm_projection: nn.Module,
    ) -> None:
        super().__init__()
        self.length_regulator = length_regulator.eval()
        self.campplus = campplus.eval()
        self.cfm_projection = cfm_projection.eval()

    def forward(
        self,
        semantic_features: torch.Tensor,
        reference_mel: torch.Tensor,
        style_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_lengths = torch._shape_as_tensor(reference_mel)[2:3]
        prompt_condition = self.length_regulator(
            semantic_features,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]
        style = self.campplus(style_features)
        reference_hidden, null_hidden = self.cfm_projection(
            reference_mel,
            style,
            prompt_condition,
        )
        return style, reference_hidden, null_hidden


class IndexTTS2EmotionMatrix(nn.Module):
    """Select and blend the official grouped emotion lookup vectors."""

    group_counts = (3, 17, 2, 8, 4, 5, 10, 24)

    def __init__(
        self,
        speaker_matrix: torch.Tensor,
        emotion_matrix: torch.Tensor,
    ) -> None:
        super().__init__()
        expected_rows = sum(self.group_counts)
        self.register_buffer("speaker_matrix", speaker_matrix.detach().float())
        self.register_buffer("emotion_matrix", emotion_matrix.detach().float())

    def forward(
        self,
        style: torch.Tensor,
        base_vector: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        selected = []
        start = 0
        for count in self.group_counts:
            stop = start + count
            similarities = F.cosine_similarity(
                style.float(),
                self.speaker_matrix[start:stop],
                dim=1,
            )
            index = torch.argmax(similarities)
            selected.append(self.emotion_matrix[start:stop][index])
            start = stop
        selected_matrix = torch.stack(selected)
        matrix_vector = torch.sum(weights.reshape(-1, 1) * selected_matrix, dim=0)
        return matrix_vector.unsqueeze(0) + (1.0 - weights.sum()) * base_vector


class IndexTTS2Conditioning(nn.Module):
    """Run the official speaker and emotion Conformer/Perceiver forwards."""

    def __init__(
        self,
        gpt: nn.Module,
        speaker_matrix: torch.Tensor,
        emotion_matrix: torch.Tensor,
    ) -> None:
        super().__init__()
        self.gpt = gpt.eval()
        self.speaker_projection = gpt.spk_emb_proj.eval()
        self.emotion_matrix = IndexTTS2EmotionMatrix(
            speaker_matrix,
            emotion_matrix,
        )

    def forward(
        self,
        speaker_features: torch.Tensor,
        speaker_lengths: torch.Tensor,
        emotion_features: torch.Tensor,
        emotion_lengths: torch.Tensor,
        emotion_alpha: torch.Tensor,
        style: torch.Tensor,
        emotion_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        speaker_latent = self.speaker_projection(style).unsqueeze(1)
        base_emotion = self.gpt.merge_emovec(
            speaker_features,
            emotion_features,
            speaker_lengths,
            emotion_lengths,
            alpha=emotion_alpha,
        )
        emotion_vector = self.emotion_matrix(
            style,
            base_emotion,
            emotion_weights,
        )
        return speaker_latent, emotion_vector


class IndexTTS2TargetPreprocess(nn.Module):
    """Construct the exact batch-one GPT prefill embedding sequence."""

    def __init__(self, gpt: nn.Module) -> None:
        super().__init__()
        self.gpt = gpt.eval()
        self.use_campplus = gpt.spk_cond_mode == "campplus"
        self.start_text_token = int(gpt.start_text_token)
        self.stop_text_token = int(gpt.stop_text_token)
        self.register_buffer(
            "text_position_table",
            gpt.text_pos_embedding.emb.weight.detach().half(),
        )
        self.register_buffer(
            "mel_position_table",
            gpt.mel_pos_embedding.emb.weight.detach().half(),
        )
        self.register_buffer(
            "start_text_id",
            torch.tensor([[gpt.start_text_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "stop_text_id",
            torch.tensor([[gpt.stop_text_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "start_mel_id",
            torch.tensor([[gpt.start_mel_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "normal_speed_id",
            torch.zeros(1, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "half_speed_id",
            torch.ones(1, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        speaker_latent: torch.Tensor,
        emotion_vector: torch.Tensor,
        text_ids: torch.Tensor,
        language_id: torch.Tensor,
    ) -> torch.Tensor:
        text_with_bounds = torch.cat(
            (self.start_text_id, text_ids, self.stop_text_id),
            dim=1,
        )
        text_length = torch._shape_as_tensor(text_with_bounds)[1:2]
        text_hidden = self.gpt.text_embedding(text_with_bounds)
        text_hidden = text_hidden + self.text_position_table[:text_length].float()
        if self.use_campplus:
            text_hidden = text_hidden + self.gpt.lang_embedding(language_id).unsqueeze(1)

        conditioned_speaker = speaker_latent + emotion_vector.unsqueeze(1)
        if self.use_campplus:
            zero_condition = torch.zeros_like(conditioned_speaker).expand(
                -1,
                2,
                -1,
            )
            prompt_hidden = torch.cat(
                (conditioned_speaker, zero_condition, text_hidden),
                dim=1,
            )
        else:
            half_speed = self.gpt.speed_emb(self.half_speed_id).unsqueeze(1)
            normal_speed = self.gpt.speed_emb(self.normal_speed_id).unsqueeze(1)
            prompt_hidden = torch.cat(
                (conditioned_speaker, half_speed, normal_speed, text_hidden),
                dim=1,
            )

        start_mel_hidden = self.gpt.mel_embedding(self.start_mel_id)
        batch_size = torch._shape_as_tensor(text_ids)[0]
        start_mel_hidden = start_mel_hidden + self.mel_position_table[:batch_size].float()
        return torch.cat((prompt_hidden, start_mel_hidden), dim=1)


class IndexTTS2DecodeEmbed(nn.Module):
    """Embed one generated code at the position used by official generation."""

    def __init__(self, gpt: nn.Module) -> None:
        super().__init__()
        self.gpt = gpt.eval()
        self.register_buffer(
            "mel_position_table",
            gpt.mel_pos_embedding.emb.weight.detach().half(),
        )

    def forward(
        self,
        current_token: torch.Tensor,
        save_ids_in: torch.Tensor,
    ) -> torch.Tensor:
        # cached_mel_emb excludes the start-mel token. The official generation
        # path therefore uses generated_count + 1 for each cached decode call.
        generated_count = torch._shape_as_tensor(save_ids_in)[1]
        hidden_states = self.gpt.mel_embedding(current_token)
        mel_position = self.mel_position_table[
            generated_count + 1 : generated_count + 2
        ].float()
        return hidden_states + mel_position


class IndexTTS2Main(nn.Module):
    """GPT-2 core with fused QKV projections and explicit growing KV state."""

    def __init__(self, gpt: nn.Module, max_sequence_length: int) -> None:
        super().__init__()
        self.gpt = gpt.eval()
        self.inference_model = gpt.inference_model.eval()
        self.num_layers = int(gpt.layers)
        self.num_heads = int(gpt.heads)
        self.hidden_size = int(gpt.model_dim)
        self.head_dim = self.hidden_size // self.num_heads
        self.compute_in_f32 = COMPUTE_IN_F32
        self.save_key: list[torch.Tensor | None] = [None] * self.num_layers
        self.save_value: list[torch.Tensor | None] = [None] * self.num_layers

        self.register_buffer(
            "attention_mask",
            (
                (
                    1
                    - torch.tril(
                        torch.ones(
                            1,
                            1,
                            max_sequence_length,
                            max_sequence_length,
                            dtype=torch.int8,
                        )
                    )
                )
                * -128
            ),
        )

        scaling = float(self.head_dim**-0.25)
        for layer in self.inference_model.transformer.h:
            qkv_weight = layer.attn.c_attn.weight.detach().clone()
            qkv_bias = layer.attn.c_attn.bias.detach().clone()
            qkv_weight[:, : 2 * self.hidden_size] *= scaling
            qkv_bias[: 2 * self.hidden_size] *= scaling
            layer.attn.register_buffer("onnx_qkv_weight", qkv_weight)
            layer.attn.register_buffer("onnx_qkv_bias", qkv_bias)
            layer.attn.register_buffer(
                "onnx_out_proj_weight",
                layer.attn.c_proj.weight.detach()
                .view(self.num_heads, self.head_dim, self.hidden_size)
                .contiguous(),
            )
            layer.attn.register_buffer(
                "onnx_out_proj_bias",
                layer.attn.c_proj.bias.detach().view(1, 1, -1).contiguous(),
            )

        mel_head = self.inference_model.lm_head[1]
        self.register_buffer(
            "mel_head_weight",
            mel_head.weight.detach().transpose(0, 1).contiguous(),
        )
        self.register_buffer("mel_head_bias", mel_head.bias.detach().clone())

    def forward(self, *all_inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        hidden_state = all_inputs[-2]
        history_length = all_inputs[-1]
        input_length = torch._shape_as_tensor(hidden_state)[1:2]
        kv_sequence_length = history_length + input_length
        attention_mask = self.attention_mask[
            :, :, history_length:kv_sequence_length, :kv_sequence_length
        ]
        if USE_F16_KV and not self.compute_in_f32:
            attention_mask = attention_mask.half()
        else:
            attention_mask = attention_mask.float()

        for index, layer in enumerate(self.inference_model.transformer.h):
            hidden_states_norm = layer.ln_1(hidden_state)
            qkv = torch.matmul(hidden_states_norm, layer.attn.onnx_qkv_weight)
            qkv = qkv + layer.attn.onnx_qkv_bias
            qkv = qkv.reshape(1, -1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(0, 3, 1, 2, 4).reshape(
                1,
                self.num_heads,
                -1,
                3 * self.head_dim,
            )

            if USE_F16_KV and not self.compute_in_f32:
                qkv = qkv.half()
            query, key, value = torch.split(qkv, self.head_dim, dim=-1)
            key = key.transpose(-1, -2)
            if USE_F16_KV and self.compute_in_f32:
                key = key.half()
                value = value.half()

            key = torch.cat((all_inputs[index], key), dim=-1)
            value = torch.cat(
                (all_inputs[index + self.num_layers], value),
                dim=-2,
            )
            self.save_key[index] = key
            self.save_value[index] = value

            if USE_F16_KV and self.compute_in_f32:
                attention_output = torch.matmul(
                    query,
                    key.float(),
                )
                attention_output = attention_output + attention_mask
                attention_output = torch.matmul(
                    torch.softmax(attention_output, dim=-1),
                    value.float(),
                )
            else:
                attention_output = torch.matmul(
                    query,
                    key,
                )
                attention_output = attention_output + attention_mask
                attention_output = torch.matmul(
                    torch.softmax(attention_output, dim=-1),
                    value,
                )
                if USE_F16_KV:
                    attention_output = attention_output.float()

            attention_output = torch.matmul(
                attention_output,
                layer.attn.onnx_out_proj_weight,
            ).sum(dim=1)
            attention_output = attention_output + layer.attn.onnx_out_proj_bias
            hidden_state = hidden_state + attention_output

            feed_forward = torch.matmul(
                layer.ln_2(hidden_state),
                layer.mlp.c_fc.weight,
            )
            feed_forward = feed_forward + layer.mlp.c_fc.bias
            feed_forward = layer.mlp.act(feed_forward)
            feed_forward = torch.matmul(feed_forward, layer.mlp.c_proj.weight)
            feed_forward = feed_forward + layer.mlp.c_proj.bias
            hidden_state = hidden_state + feed_forward

        transformer_hidden = self.inference_model.transformer.ln_f(hidden_state)
        logits_hidden = self.inference_model.lm_head[0](transformer_hidden[:, -1])
        logits = torch.matmul(logits_hidden, self.mel_head_weight) + self.mel_head_bias
        return (
            *[value for value in self.save_key if value is not None],
            *[value for value in self.save_value if value is not None],
            transformer_hidden,
            logits,
            kv_sequence_length,
        )


class ApplyRecentPenalty(nn.Module):
    def forward(
        self,
        logits: torch.Tensor,
        previous_ids: torch.Tensor,
        penalty_value: torch.Tensor,
        penalty_range: torch.Tensor,
    ) -> torch.Tensor:
        target_indices = previous_ids[:, -penalty_range:].long()
        penalized = logits.gather(1, target_indices) * penalty_value
        return logits.scatter(1, target_indices, penalized)


class SignAwareRepetitionPenalty(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        repetition_penalty: torch.Tensor,
        previous_ids: torch.Tensor,
    ) -> torch.Tensor:
        previous_ids_long = previous_ids.long()
        previous_logits = torch.gather(logits, 1, previous_ids_long)
        previous_scores = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        return torch.scatter(logits, 1, previous_ids_long, previous_scores)

    @staticmethod
    def symbolic(
        graph: Any,
        logits: Any,
        repetition_penalty: Any,
        previous_ids: Any,
    ) -> Any:
        previous_logits = graph.op("GatherElements", logits, previous_ids, axis_i=1)
        zero = graph.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))
        previous_scores = graph.op(
            "Where",
            graph.op("Less", previous_logits, zero),
            graph.op("Mul", previous_logits, repetition_penalty),
            graph.op("Div", previous_logits, repetition_penalty),
        )
        return graph.op(
            "ScatterElements",
            logits,
            previous_ids,
            previous_scores,
            axis_i=1,
        )


class TopKTopPSampling(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.register_buffer("one", torch.tensor([1], dtype=torch.int64), persistent=False)
        self.register_buffer(
            "vocabulary_size",
            torch.tensor([vocabulary_size], dtype=torch.int64),
            persistent=False,
        )

    def sample(
        self,
        scores: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        greedy_scores: torch.Tensor,
    ) -> torch.Tensor:
        top_k = torch.minimum(torch.maximum(top_k, self.one), self.vocabulary_size)
        sorted_scores, sorted_indices = torch.topk(
            scores,
            k=top_k,
            dim=-1,
            largest=True,
            sorted=True,
        )
        sorted_probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        keep = (cumulative_probabilities - sorted_probabilities) <= top_p
        kept_mass = torch.where(keep, cumulative_probabilities, 0.0).amax(
            dim=-1,
            keepdim=True,
        )
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax(
            (cumulative_probabilities >= threshold).int(),
            dim=-1,
            keepdim=True,
        )
        sampled_id = torch.gather(sorted_indices, 1, winner).int()
        greedy_id = torch.argmax(greedy_scores, dim=-1, keepdim=True).int()
        return torch.where(top_k == self.one, greedy_id, sampled_id)

    def forward(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        repetition_penalty: torch.Tensor,
        previous_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = SignAwareRepetitionPenalty.apply(
            logits,
            repetition_penalty,
            previous_ids,
        )
        sampled_id = self.sample(scores, temperature, top_k, top_p, logits)
        return sampled_id, torch.cat((previous_ids, sampled_id), dim=-1)


class IndexTTS2TokenStrategy(nn.Module):
    def __init__(self, strategy: str, vocabulary_size: int) -> None:
        super().__init__()
        self.strategy = strategy
        self.penalty = ApplyRecentPenalty()
        self.sampling = TopKTopPSampling(vocabulary_size)

    def forward(
        self,
        logits: torch.Tensor,
        previous_ids: torch.Tensor,
        *controls: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.strategy == "penalty_greedy":
            penalty_value, penalty_range = controls
            penalized_logits = self.penalty(
                logits,
                previous_ids,
                penalty_value,
                penalty_range,
            )
            use_penalty = torch._shape_as_tensor(previous_ids)[1:2] >= penalty_range
            logits = torch.where(use_penalty, penalized_logits, logits)
        elif self.strategy == "sampling":
            return self.sampling(logits, *controls, previous_ids)

        next_token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return next_token, torch.cat((previous_ids, next_token), dim=-1)


class IndexTTS2MainPrefillStrategy(nn.Module):
    def __init__(
        self,
        main_core: IndexTTS2Main,
        strategy: str,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.main_core = main_core
        self.strategy_name = strategy
        self.strategy = IndexTTS2TokenStrategy(strategy, vocabulary_size)
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
        self.register_buffer(
            "zero_history_length",
            torch.zeros(1, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *controls: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.main_core(
            *([self.empty_key] * self.num_layers),
            *([self.empty_value] * self.num_layers),
            hidden_states,
            self.zero_history_length,
        )
        state_count = self.num_layers * 2
        logits = outputs[state_count + 1]
        if self.strategy_name == "sampling":
            next_token = self.strategy.sampling.sample(logits, *controls, logits)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return (
            *outputs[:state_count],
            outputs[state_count][:, -1],
            next_token,
            outputs[state_count + 2],
        )


class IndexTTS2MainDecodeStrategy(nn.Module):
    def __init__(
        self,
        main_core: IndexTTS2Main,
        strategy: str,
        vocabulary_size: int,
    ) -> None:
        super().__init__()
        self.main_core = main_core
        self.strategy = IndexTTS2TokenStrategy(strategy, vocabulary_size)
        self.state_count = main_core.num_layers * 2

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        states = args[: self.state_count]
        hidden_states = args[self.state_count]
        previous_ids = args[self.state_count + 1]
        history_length = args[self.state_count + 2]
        controls = args[self.state_count + 3 :]
        outputs = self.main_core(*states, hidden_states, history_length)
        next_token, save_ids_out = self.strategy(
            outputs[self.state_count + 1],
            previous_ids,
            *controls,
        )
        return (
            *outputs[: self.state_count],
            outputs[self.state_count][:, -1],
            next_token,
            save_ids_out,
            outputs[self.state_count + 2],
        )


class IndexTTS2Latent(nn.Module):
    """Recreate the official post-generation GPT latent in one optimized pass."""

    def __init__(self, gpt: nn.Module, main_core: IndexTTS2Main) -> None:
        super().__init__()
        self.gpt = gpt.eval()
        self.main_core = main_core
        self.use_campplus = getattr(gpt, "spk_cond_mode", None) == "campplus"
        self.num_layers = main_core.num_layers
        self.state_count = self.num_layers * 2
        self.start_text_token = int(gpt.start_text_token)
        self.stop_text_token = int(gpt.stop_text_token)
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        self.register_buffer(
            "text_position_table",
            gpt.text_pos_embedding.emb.weight.detach().half(),
        )
        self.register_buffer(
            "mel_position_table",
            gpt.mel_pos_embedding.emb.weight.detach().half(),
        )
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
        self.register_buffer(
            "zero_history_length",
            torch.zeros(1, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "start_text_id",
            torch.tensor([[gpt.start_text_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "stop_text_id",
            torch.tensor([[gpt.stop_text_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "start_mel_id",
            torch.tensor([[gpt.start_mel_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "stop_mel_id",
            torch.tensor([[gpt.stop_mel_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "normal_speed_id",
            torch.zeros(1, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "half_speed_id",
            torch.ones(1, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        speaker_latent: torch.Tensor,
        emotion_vector: torch.Tensor,
        text_ids: torch.Tensor,
        mel_codes: torch.Tensor,
    ) -> torch.Tensor:
        text_with_bounds = torch.cat(
            (self.start_text_id, text_ids, self.stop_text_id),
            dim=1,
        )
        text_length = torch._shape_as_tensor(text_with_bounds)[1:2]
        text_hidden = self.gpt.text_embedding(text_with_bounds)
        text_hidden = text_hidden + self.text_position_table[:text_length].float()

        mel_with_bounds = torch.cat(
            (self.start_mel_id, mel_codes, self.stop_mel_id),
            dim=1,
        )
        mel_length = torch._shape_as_tensor(mel_with_bounds)[1:2]
        mel_hidden = self.gpt.mel_embedding(mel_with_bounds)
        mel_hidden = mel_hidden + self.mel_position_table[:mel_length].float()

        conditioned_speaker = speaker_latent + emotion_vector.unsqueeze(1)
        if self.use_campplus:
            zero_condition = torch.zeros_like(conditioned_speaker).expand(
                -1,
                2,
                -1,
            )
            prefix_hidden = torch.cat(
                (conditioned_speaker, zero_condition, text_hidden),
                dim=1,
            )
        else:
            half_speed = self.gpt.speed_emb(self.half_speed_id).unsqueeze(1)
            normal_speed = self.gpt.speed_emb(self.normal_speed_id).unsqueeze(1)
            prefix_hidden = torch.cat(
                (conditioned_speaker, half_speed, normal_speed, text_hidden),
                dim=1,
            )
        mel_offset = torch._shape_as_tensor(prefix_hidden)[1:2]
        hidden_states = torch.cat((prefix_hidden, mel_hidden), dim=1)
        outputs = self.main_core(
            *([self.empty_key] * self.num_layers),
            *([self.empty_value] * self.num_layers),
            hidden_states,
            self.zero_history_length,
        )
        mel_latent = outputs[self.state_count][:, mel_offset:]
        mel_latent = self.gpt.final_norm(mel_latent)
        return mel_latent[:, :-2]


class IndexTTS2AcousticConditioning(nn.Module):
    """Match semantic-code embedding, GPT fusion, and length regulation."""

    def __init__(
        self,
        semantic_codec: nn.Module,
        gpt_projection: nn.Module,
        length_regulator: nn.Module,
        cfm_projection: nn.Module,
    ) -> None:
        super().__init__()
        self.semantic_codec = semantic_codec.eval()
        self.gpt_projection = (
            None if gpt_projection is None else gpt_projection.eval()
        )
        self.length_regulator = length_regulator.eval()
        self.cfm_projection = cfm_projection.eval()

    def forward(
        self,
        mel_codes: torch.Tensor,
        gpt_latent: torch.Tensor,
        cfg_rate: torch.Tensor,
        duration_factor: torch.Tensor,
        style: torch.Tensor,
        reference_hidden: torch.Tensor,
        null_hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        semantic_hidden = self.semantic_codec.decode(mel_codes)
        if self.gpt_projection is None:
            semantic_hidden = semantic_hidden + gpt_latent * 0.0
        else:
            semantic_hidden = semantic_hidden + self.gpt_projection(gpt_latent)
        semantic_length = torch._shape_as_tensor(semantic_hidden)[1:2]
        target_length = (
            semantic_length.float() * 1.72 * duration_factor
        ).int()
        condition = self.length_regulator(
            semantic_hidden,
            ylens=target_length,
            n_quantizers=3,
            f0=None,
        )[0]
        target_hidden = self.cfm_projection.project_without_prompt(
            style,
            condition,
        )
        conditional_hidden = torch.cat(
            (reference_hidden, target_hidden),
            dim=1,
        )
        static_hidden = torch.cat(
            (conditional_hidden, null_hidden.expand_as(conditional_hidden)),
            dim=0,
        )
        target_mask = torch.cat(
            (
                torch.zeros_like(reference_hidden[:, :, :1]),
                torch.ones_like(target_hidden[:, :, :1]),
            ),
            dim=1,
        )
        cfg_scales = torch.cat((1.0 + cfg_rate, -cfg_rate)).view(2, 1, 1)
        cfg_scale_sum = cfg_scales.sum().reshape(1)
        return static_hidden, target_length, cfg_scales, cfg_scale_sum, target_mask


class IndexTTS2LatentBypass(nn.Module):
    """Keep the v2 merge contract without enabling v2.5's optional GPT latent."""

    def __init__(self, hidden_size: int, semantic_upsample_scale: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.semantic_upsample_scale = semantic_upsample_scale

    def forward(
        self,
        speaker_latent: torch.Tensor,
        emotion_vector: torch.Tensor,
        text_ids: torch.Tensor,
        mel_codes: torch.Tensor,
    ) -> torch.Tensor:
        code_values = mel_codes.float().unsqueeze(-1)
        code_values = torch.cat(
            [code_values] * self.semantic_upsample_scale,
            dim=1,
        )
        anchor = (
            speaker_latent.sum()
            + emotion_vector.sum()
            + text_ids.float().sum()
        ) * 0.0
        return (
            code_values.expand(-1, -1, self.hidden_size) * 0.0
            + anchor.reshape(1, 1, 1)
        )


class ExportSConv1d(nn.Module):
    """Trace-safe equivalent of the stride-one SConv1d used by v2 Wavenet."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.conv = source.conv
        inner_conv = source.conv.conv
        effective_kernel = (inner_conv.kernel_size[0] - 1) * inner_conv.dilation[0] + 1
        padding_total = effective_kernel - 1
        if source.causal:
            self.padding = (padding_total, 0)
        else:
            padding_right = padding_total // 2
            self.padding = (padding_total - padding_right, padding_right)
        self.pad_mode = "constant" if source.pad_mode == "zero" else source.pad_mode

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.padding != (0, 0):
            values = F.pad(values, self.padding, mode=self.pad_mode)
        return self.conv(values)


def replace_wavenet_sconv1d(module: nn.Module) -> int:
    from indextts.s2mel.modules.encodec import SConv1d

    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SConv1d):
            setattr(module, name, ExportSConv1d(child))
            replacements += 1
        else:
            replacements += replace_wavenet_sconv1d(child)
    return replacements


class PrecomputedCFMTimeProjection(nn.Module):
    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("values", values.detach().clone().contiguous())

    def forward(self, step_index: torch.Tensor) -> torch.Tensor:
        return self.values[step_index]


class AllValidSelfAttention(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.wqkv = source.wqkv
        self.wo = source.wo
        self.num_heads = int(source.n_head)
        self.num_local_heads = int(source.n_local_heads)
        self.head_dim = int(source.head_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        frequencies: torch.Tensor,
        _mask: torch.Tensor,
        _input_position: torch.Tensor,
    ) -> torch.Tensor:
        from indextts.s2mel.modules.gpt_fast.model import apply_rotary_emb

        batch_size, sequence_length, _ = hidden.shape
        key_value_size = self.num_local_heads * self.head_dim
        query, key, value = self.wqkv(hidden).split(
            [key_value_size, key_value_size, key_value_size],
            dim=-1,
        )
        query = query.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        key = key.view(
            batch_size,
            sequence_length,
            self.num_local_heads,
            self.head_dim,
        )
        value = value.view(
            batch_size,
            sequence_length,
            self.num_local_heads,
            self.head_dim,
        )
        query = apply_rotary_emb(query, frequencies).transpose(1, 2)
        key = apply_rotary_emb(key, frequencies).transpose(1, 2)
        value = value.transpose(1, 2)
        repeat_count = self.num_heads // self.num_local_heads
        key = key.repeat_interleave(repeat_count, dim=1)
        value = value.repeat_interleave(repeat_count, dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            self.num_heads * self.head_dim,
        )
        return self.wo(attended)


class AllValidWavenet(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.hidden_channels = int(source.hidden_channels)
        self.n_layers = int(source.n_layers)
        self.cond_layer = source.cond_layer
        self.in_layers = source.in_layers
        self.res_skip_layers = source.res_skip_layers
        self.drop = source.drop
        self.register_buffer(
            "channel_count",
            torch.tensor([self.hidden_channels], dtype=torch.int32),
            persistent=False,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        from indextts.s2mel.modules import commons

        output = torch.zeros_like(hidden)
        conditioning = self.cond_layer(conditioning)
        for index in range(self.n_layers):
            hidden_input = self.in_layers[index](hidden)
            condition_start = index * 2 * self.hidden_channels
            condition_end = condition_start + 2 * self.hidden_channels
            activated = commons.fused_add_tanh_sigmoid_multiply(
                hidden_input,
                conditioning[:, condition_start:condition_end],
                self.channel_count,
            )
            residual_skip = self.res_skip_layers[index](self.drop(activated))
            if index < self.n_layers - 1:
                hidden = hidden + residual_skip[:, : self.hidden_channels]
                output = output + residual_skip[:, self.hidden_channels :]
            else:
                output = output + residual_skip
        return output


class IndexTTS2CFMStaticProjection(nn.Module):
    """Project static CFM inputs before the repeated Euler evaluations."""

    def __init__(self, estimator: nn.Module, style_embed_size: int) -> None:
        super().__init__()
        merge = estimator.cond_x_merge_linear
        self.in_channels = int(estimator.in_channels)
        condition_size = int(estimator.cond_projection.out_features)
        expected_static_size = (
            self.in_channels
            + condition_size
            + style_embed_size
        )
        static_weight = merge.weight.detach()[:, self.in_channels :]
        condition_start = self.in_channels
        condition_end = condition_start + condition_size
        condition_merge_weight = static_weight[:, condition_start:condition_end]
        self.register_buffer(
            "prompt_weight",
            static_weight[:, : self.in_channels].clone().contiguous(),
        )
        self.register_buffer(
            "condition_weight",
            torch.matmul(
                condition_merge_weight,
                estimator.cond_projection.weight.detach(),
            ).contiguous(),
        )
        condition_bias = F.linear(
            estimator.cond_projection.bias.detach(),
            condition_merge_weight,
            merge.bias.detach(),
        )
        self.register_buffer("condition_bias", condition_bias.contiguous())
        self.register_buffer(
            "style_weight",
            static_weight[:, condition_end:].clone().contiguous(),
        )
        null_features = torch.cat(
            (
                torch.zeros(
                    2 * self.in_channels,
                    dtype=merge.weight.dtype,
                    device=merge.weight.device,
                ),
                estimator.cond_projection.bias.detach(),
                torch.zeros(
                    style_embed_size,
                    dtype=merge.weight.dtype,
                    device=merge.weight.device,
                ),
            ),
            dim=0,
        )
        null_hidden = F.linear(
            null_features,
            merge.weight.detach(),
            merge.bias.detach(),
        )
        self.register_buffer(
            "null_hidden",
            null_hidden.view(1, 1, -1).clone().contiguous(),
        )

    def forward(
        self,
        prompt_mel: torch.Tensor,
        style: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditional_hidden = F.linear(
            prompt_mel.transpose(1, 2),
            self.prompt_weight,
        )
        conditional_hidden = conditional_hidden + F.linear(
            condition,
            self.condition_weight,
            self.condition_bias,
        )
        conditional_hidden = conditional_hidden + F.linear(
            style,
            self.style_weight,
        ).unsqueeze(1)
        return conditional_hidden, self.null_hidden

    def project_without_prompt(
        self,
        style: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.linear(
            condition,
            self.condition_weight,
            self.condition_bias,
        )
        return hidden + F.linear(style, self.style_weight).unsqueeze(1)


class IndexTTS2CFMStep(nn.Module):
    """Evaluate one precomputed-schedule CFM step and fused Euler update."""

    def __init__(
        self,
        estimator: nn.Module,
        time_schedule: Any,
    ) -> None:
        super().__init__()
        self.estimator = estimator.eval()
        self.estimator.style_as_token = int(self.estimator.style_as_token)
        self.estimator.time_as_token = int(self.estimator.time_as_token)
        with torch.inference_mode():
            step_times, euler_deltas = time_schedule(
                CFM_STEPS,
                self.estimator.input_pos.device,
            )
            timestep_hidden = self.estimator.t_embedder(step_times)
            transformer_condition = timestep_hidden.unsqueeze(1)
            adaptive_norms = [
                norm
                for layer in self.estimator.transformer.layers
                for norm in (layer.attention_norm, layer.ffn_norm)
            ]
            adaptive_norms.append(self.estimator.transformer.norm)
            for layer in self.estimator.transformer.layers:
                layer.attention = AllValidSelfAttention(layer.attention)
            for adaptive_norm in adaptive_norms:
                adaptive_norm.project_layer = PrecomputedCFMTimeProjection(
                    adaptive_norm.project_layer(transformer_condition)
                )
            wavenet_time = self.estimator.t_embedder2(step_times).unsqueeze(2)
            self.estimator.wavenet.cond_layer = PrecomputedCFMTimeProjection(
                self.estimator.wavenet.cond_layer(wavenet_time)
            )
            euler_deltas = euler_deltas.view(CFM_STEPS, 1, 1)
            final_shift, final_scale = self.estimator.final_layer.adaLN_modulation(
                timestep_hidden
            ).chunk(2, dim=1)
            self.register_buffer(
                "final_scale",
                (
                    euler_deltas * (1.0 + final_scale).unsqueeze(1)
                ).clone().contiguous(),
            )
            self.register_buffer(
                "final_shift",
                (euler_deltas * final_shift.unsqueeze(1)).clone().contiguous(),
            )
            self.wavenet = AllValidWavenet(self.estimator.wavenet)
        self.in_channels = int(self.estimator.in_channels)
        self.hidden_size = int(self.estimator.cond_x_merge_linear.out_features)
        self.register_buffer(
            "valid_frame_mask",
            torch.ones(1, 1, 1, dtype=torch.bool),
            persistent=False,
        )
        skip_weight = self.estimator.skip_linear.weight.detach()
        skip_bias = self.estimator.skip_linear.bias.detach()
        projection_weights: dict[str, torch.Tensor] = {}
        for name, projection in (
            ("wavenet", self.estimator.conv1),
            ("residual", self.estimator.res_projection),
        ):
            projection_weight = projection.weight.detach()
            projection_weights[name] = torch.matmul(
                projection_weight,
                skip_weight,
            )
            self.register_buffer(
                f"{name}_bias",
                F.linear(
                    skip_bias,
                    projection_weight,
                    projection.bias.detach(),
                ).contiguous(),
            )
        self.register_buffer(
            "mel_projection_weight",
            torch.cat(
                (
                    self.estimator.cond_x_merge_linear.weight.detach()
                    [:, : self.in_channels],
                    projection_weights["residual"][:, self.hidden_size :],
                    projection_weights["wavenet"][:, self.hidden_size :],
                ),
                dim=0,
            ).contiguous(),
        )
        self.register_buffer(
            "hidden_projection_weight",
            torch.cat(
                (
                    projection_weights["residual"][:, : self.hidden_size],
                    projection_weights["wavenet"][:, : self.hidden_size],
                ),
                dim=0,
            ).contiguous(),
        )
        output_weight = self.estimator.conv2.weight.detach().squeeze(-1)
        self.register_buffer(
            "output_weight",
            torch.matmul(
                output_weight,
                self.estimator.final_layer.linear.weight.detach(),
            ).contiguous(),
        )
        output_bias = F.linear(
            self.estimator.final_layer.linear.bias.detach(),
            output_weight,
            self.estimator.conv2.bias.detach(),
        )
        self.register_buffer(
            "output_bias",
            (euler_deltas * output_bias.view(1, 1, -1)).contiguous(),
        )

    def forward(
        self,
        mel_features: torch.Tensor,
        step_index: torch.Tensor,
        static_hidden: torch.Tensor,
        cfg_scales: torch.Tensor,
        cfg_scale_sum: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        branch_count = torch._shape_as_tensor(cfg_scales)[0]
        static_hidden = static_hidden[:branch_count]
        mel_features = mel_features * target_mask
        mel_hidden, residual_mel, wavenet_mel = F.linear(
            mel_features,
            self.mel_projection_weight,
        ).split(self.hidden_size, dim=-1)
        hidden = static_hidden + mel_hidden
        input_position = self.estimator.input_pos[: hidden.size(1)]
        hidden = self.estimator.transformer(
            hidden,
            step_index,
            input_position,
            self.valid_frame_mask.unsqueeze(1),
        )

        residual_hidden, wavenet_hidden = F.linear(
            hidden,
            self.hidden_projection_weight,
        ).split(self.hidden_size, dim=-1)
        residual_hidden = residual_hidden + residual_mel + self.residual_bias
        hidden = (wavenet_hidden + wavenet_mel + self.wavenet_bias).transpose(1, 2)
        hidden = self.wavenet(
            hidden,
            step_index,
        ).transpose(1, 2) + residual_hidden

        normalized = self.estimator.final_layer.norm_final(hidden)
        guided_normalized = (normalized * cfg_scales).sum(dim=0, keepdim=True)
        hidden = guided_normalized * self.final_scale[step_index]
        hidden = hidden + cfg_scale_sum * self.final_shift[step_index]
        delta_velocity = F.linear(
            hidden,
            self.output_weight,
        )
        delta_velocity = delta_velocity + cfg_scale_sum * self.output_bias[step_index]
        return (mel_features + delta_velocity) * target_mask


class FrozenPeriodicActivation(nn.Module):
    """Freeze Snake/SnakeBeta parameters as broadcast-ready buffers."""

    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        alpha = activation.alpha.detach()
        if bool(activation.alpha_logscale):
            alpha = torch.exp(alpha)
        beta_parameter = getattr(activation, "beta", activation.alpha).detach()
        if bool(activation.alpha_logscale):
            beta_parameter = torch.exp(beta_parameter)
        epsilon = float(activation.no_div_by_zero)
        self.register_buffer("alpha", alpha.view(1, -1, 1).contiguous())
        self.register_buffer(
            "inverse_beta",
            (1.0 / (beta_parameter + epsilon)).view(1, -1, 1).contiguous(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        periodic = torch.sin(values * self.alpha)
        return values + periodic.square() * self.inverse_beta


class OptimizedAliasFreeActivation(nn.Module):
    """Preserve BigVGAN-v2 alias-free math without dynamic filter expansion."""

    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        channels = int(activation.act.alpha.numel())
        upsample = activation.upsample
        downsample = activation.downsample.lowpass
        self.act = FrozenPeriodicActivation(activation.act)
        self.up_stride = int(upsample.stride)
        self.up_pad = int(upsample.pad)
        self.up_crop_left = int(upsample.pad_left)
        self.up_crop_right = int(upsample.pad_right)
        self.down_stride = int(downsample.stride)
        self.down_pad_left = int(downsample.pad_left)
        self.down_pad_right = int(downsample.pad_right)
        self.register_buffer(
            "up_filter",
            (
                upsample.filter.detach().expand(channels, -1, -1)
                * float(upsample.ratio)
            ).contiguous(),
        )
        self.register_buffer(
            "down_filter",
            downsample.filter.detach().expand(channels, -1, -1).contiguous(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = F.pad(values, (self.up_pad, self.up_pad), mode="replicate")
        values = F.conv_transpose1d(
            values,
            self.up_filter,
            stride=self.up_stride,
            groups=self.up_filter.shape[0],
        )
        values = values[..., self.up_crop_left : -self.up_crop_right]
        values = self.act(values)
        values = F.pad(
            values,
            (self.down_pad_left, self.down_pad_right),
            mode="replicate",
        )
        return F.conv1d(
            values,
            self.down_filter,
            stride=self.down_stride,
            groups=self.down_filter.shape[0],
        )


def optimize_bigvgan_alias_free_activations(module: nn.Module) -> int:
    from indextts.s2mel.modules.bigvgan.alias_free_activation.torch.act import (
        Activation1d,
    )

    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Activation1d):
            setattr(module, name, OptimizedAliasFreeActivation(child))
            replacements += 1
        else:
            replacements += optimize_bigvgan_alias_free_activations(child)
    return replacements


class IndexTTS2Decoder(nn.Module):
    def __init__(self, bigvgan: nn.Module, model_sample_rate: int) -> None:
        super().__init__()
        self.bigvgan = bigvgan.eval()
        replacements = optimize_bigvgan_alias_free_activations(self.bigvgan)
        self.output_resample_scale = float(OUT_SAMPLE_RATE / model_sample_rate)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        waveform = self.bigvgan(mel.float())
        waveform = waveform.clamp(-1.0, 1.0)
        if self.output_resample_scale != 1.0:
            waveform = F.interpolate(
                waveform,
                scale_factor=self.output_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            return (32767.0 * waveform).clamp(-32768.0, 32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return waveform.float()
        return waveform.half()


def remove_all_weight_norm(module: nn.Module) -> int:
    from torch.nn.utils import remove_weight_norm

    removed = 0
    for child in module.modules():
        try:
            remove_weight_norm(child)
        except (AttributeError, ValueError):
            continue
        removed += 1
    return removed


def resolve_auxiliary_paths(config: Any) -> dict[str, str]:
    from indextts.utils.model_download import ensure_models_available

    return ensure_models_available(str(models_path))


def load_gpt(config: Any) -> nn.Module:
    _install_transformers_compatibility_modules()
    from indextts.gpt.model_v2 import UnifiedVoice

    gpt_config = OmegaConf.to_container(config.gpt, resolve=True)
    gpt = UnifiedVoice(
        **gpt_config,
        use_accel=False,
        spk_cond_mode="campplus",
    )
    checkpoint = torch.load(
        models_path / str(config.gpt_checkpoint),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    gpt.load_state_dict(state, strict=True)
    del checkpoint, state
    gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)
    return freeze(gpt)


def load_acoustic_modules(
    config: Any,
    auxiliary_paths: dict[str, str],
) -> tuple[nn.Module, nn.Module, nn.Module]:
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
    from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
    from indextts.codec.models import EnhancedCodec

    semantic_codec = EnhancedCodec(
        **OmegaConf.to_container(config.semantic_codec, resolve=True),
        cfg=config.semantic_codec,
    )
    semantic_codec.load_checkpoint(str(models_path / "codec.pth"))
    semantic_codec = freeze(semantic_codec)
    remove_all_weight_norm(semantic_codec)

    s2mel = MyModel(config.s2mel, use_gpt_latent=False)
    s2mel, _, _, _ = load_checkpoint2(
        s2mel,
        None,
        str(models_path / str(config.s2mel_checkpoint)),
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    s2mel = freeze(s2mel)
    remove_all_weight_norm(s2mel)
    s2mel.models["cfm"].estimator.setup_caches(
        max_batch_size=1,
        max_seq_length=8192,
    )

    campplus = CAMPPlus(
        feat_dim=80,
        embedding_size=int(config.s2mel.style_encoder.dim),
    )
    campplus_state = torch.load(
        auxiliary_paths["campplus"],
        map_location="cpu",
        weights_only=True,
    )
    campplus.load_state_dict(campplus_state, strict=True)
    campplus = freeze(campplus)
    return semantic_codec, s2mel, campplus


def build_export_metadata(
    config: Any,
    emotion_text_config: Any,
    emotion_text_prompt_prefix_token_ids: list[int],
    emotion_text_prompt_suffix_token_ids: list[int],
    emotion_text_content_prefix: str,
    emotion_text_think_end_token_id: int,
) -> dict[str, str]:
    emotion_stop_ids = emotion_text_config.eos_token_id
    if not isinstance(emotion_stop_ids, (list, tuple)):
        emotion_stop_ids = [emotion_stop_ids]
    return build_model_metadata(
        {
            "graph_layout": "raw_audio_emotion_text_merged_gpt_cached_cfm_step",
            **active_adapter().metadata_fields(config),
            "cfm_steps": CFM_STEPS,
            "shared_initializer_model_file": SHARED_MODEL_NAME,
            "shared_initializer_data_file": SHARED_DATA_NAME,
        },
        {
            "in_sample_rate": IN_SAMPLE_RATE,
            "out_sample_rate": OUT_SAMPLE_RATE,
            "semantic_input_sample_rate": 16000,
            "semantic_frame_length": 400,
            "semantic_frame_shift": 160,
        },
        {
            "mel_code_size": int(config.gpt.number_mel_codes),
            "stop_mel_token": int(config.gpt.stop_mel_token),
            "max_signal_length": MAX_SIGNAL_LENGTH,
            "use_f16_kv": USE_F16_KV,
            "compute_in_f32": COMPUTE_IN_F32,
        },
        {
            "emotion_text_num_layers": int(emotion_text_config.num_hidden_layers),
            "emotion_text_max_seq_length": EMOTION_TEXT_MAX_SEQ_LENGTH,
            "emotion_text_stop_token_ids": emotion_stop_ids,
            "emotion_text_prompt_prefix_token_ids": (
                emotion_text_prompt_prefix_token_ids
            ),
            "emotion_text_prompt_suffix_token_ids": (
                emotion_text_prompt_suffix_token_ids
            ),
            "emotion_text_content_prefix": emotion_text_content_prefix,
            "emotion_text_think_end_token_id": emotion_text_think_end_token_id,
            "emotion_text_kv_dtype": (
                "float16" if EMOTION_TEXT_KV_DTYPE == "F16" else "float32"
            ),
        },
        {
            "model_file_name_reference_preprocess": Path(
                onnx_model_reference_preprocess
            ).name,
            "model_file_name_conditioning": Path(onnx_models["conditioning"]).name,
            "model_file_name_synthesis": Path(onnx_model_synthesis).name,
            "model_file_name_cfm_estimator": Path(onnx_models["cfm_estimator"]).name,
            "model_file_name_decoder": Path(onnx_models["decoder"]).name,
            "model_file_name_metadata": Path(onnx_models["metadata"]).name,
            "model_file_name_emotion_text_prefill": Path(
                onnx_model_emotion_text_prefill
            ).name,
            "model_file_name_emotion_text_decode": Path(
                onnx_model_emotion_text_decode
            ).name,
            **{
                f"model_file_name_target_prefill_{strategy}": Path(
                    onnx_model_target_prefill[strategy]
                ).name
                for strategy in onnx_model_target_prefill
            },
            **{
                f"model_file_name_decode_step_{strategy}": Path(
                    onnx_model_decode_step[strategy]
                ).name
                for strategy in onnx_model_decode_step
            },
        },
    )


def build_emotion_text_prefill_graph(
    decode_path: str | Path,
    prefill_path: str | Path,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
) -> None:
    import onnx
    from onnx import TensorProto, helper

    decode_path = Path(decode_path)
    prefill_path = Path(prefill_path)
    model = onnx.load(str(decode_path), load_external_data=False)
    remap = {
        **{f"in_key_{index}": "emotion_text_empty_key" for index in range(num_layers)},
        **{
            f"in_value_{index}": "emotion_text_empty_value"
            for index in range(num_layers)
        },
        "history_length": "emotion_text_zero_history_length",
    }
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
    retained_inputs = [
        value for value in model.graph.input if value.name == "input_ids"
    ]
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)
    cache_element_type = (
        TensorProto.FLOAT16
        if EMOTION_TEXT_KV_DTYPE == "F16"
        else TensorProto.FLOAT
    )
    initializers = [
        helper.make_tensor(
            "emotion_text_empty_key",
            cache_element_type,
            [1, num_kv_heads, head_dim, 0],
            [],
        ),
        helper.make_tensor(
            "emotion_text_empty_value",
            cache_element_type,
            [1, num_kv_heads, 0, head_dim],
            [],
        ),
        helper.make_tensor(
            "emotion_text_zero_history_length",
            TensorProto.INT64,
            [1],
            [0],
        ),
    ]
    model.graph.initializer.extend(initializers)
    model.graph.name = "IndexTTS2_EmotionTextPrefill"
    prefill_path.unlink(missing_ok=True)
    prefill_path.with_name(prefill_path.name + ".data").unlink(missing_ok=True)
    onnx.save(model, str(prefill_path))
def export_emotion_text_graphs(emotion_text_config: Any) -> None:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        emotion_text_model_path,
        dtype=torch.float32,
        device_map="cpu",
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    core = IndexTTS2EmotionTextCore(model, EMOTION_TEXT_MAX_SEQ_LENGTH)
    num_layers = int(emotion_text_config.num_hidden_layers)
    num_kv_heads = int(emotion_text_config.num_key_value_heads)
    head_dim = int(emotion_text_config.head_dim)
    state_input_names = [f"in_key_{index}" for index in range(num_layers)] + [
        f"in_value_{index}" for index in range(num_layers)
    ]
    state_output_names = [f"out_key_{index}" for index in range(num_layers)] + [
        f"out_value_{index}" for index in range(num_layers)
    ]
    state_axes: dict[str, dict[int, str]] = {}
    for index in range(num_layers):
        state_axes[f"in_key_{index}"] = {3: "history_length"}
        state_axes[f"out_key_{index}"] = {3: "kv_sequence_length"}
        state_axes[f"in_value_{index}"] = {2: "history_length"}
        state_axes[f"out_value_{index}"] = {2: "kv_sequence_length"}

    history_length = torch.tensor([10], dtype=torch.int64)
    cache_dtype = (
        torch.float16 if EMOTION_TEXT_KV_DTYPE == "F16" else torch.float32
    )
    past_key = torch.zeros(
        1,
        num_kv_heads,
        head_dim,
        10,
        dtype=cache_dtype,
    )
    past_value = torch.zeros(
        1,
        num_kv_heads,
        10,
        head_dim,
        dtype=cache_dtype,
    )
    state_inputs = [past_key] * num_layers + [past_value] * num_layers
    decode_token = torch.ones(1, 1, dtype=torch.int32)
    decode = IndexTTS2EmotionTextDecode(core)
    export_onnx(
        decode,
        (*state_inputs, decode_token, history_length),
        onnx_model_emotion_text_decode,
        input_names=[*state_input_names, "input_ids", "history_length"],
        output_names=[*state_output_names, "next_token", "kv_sequence_length"],
        dynamic_axes={**state_axes, "input_ids": {1: "input_length"}},
    )
    build_emotion_text_prefill_graph(
        onnx_model_emotion_text_decode,
        onnx_model_emotion_text_prefill,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    del decode, core, model
    gc.collect()
    print("Exported Qwen3 EmotionText prefill and decode graphs.")


def export_feature_extractor_graph(
    config: Any,
    auxiliary_paths: dict[str, str],
) -> None:
    from transformers import SeamlessM4TFeatureExtractor

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
        auxiliary_paths["w2v_bert"],
        local_files_only=True,
    )
    wrapper = IndexTTS2FeatureExtractor(feature_extractor, config)
    audio = torch.zeros(
        1,
        1,
        IN_SAMPLE_RATE,
        dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()],
    )
    export_onnx(
        wrapper,
        (audio,),
        onnx_models["feature_extractor"],
        input_names=["audio"],
        output_names=[
            "input_features",
            "attention_mask",
            "reference_mel",
            "style_features",
        ],
        dynamic_axes={
            "audio": {2: "audio_samples"},
            "input_features": {1: "feature_frames"},
            "attention_mask": {1: "feature_frames"},
            "reference_mel": {2: "reference_mel_frames"},
            "style_features": {1: "style_frames"},
        },
    )
    del wrapper, feature_extractor
    gc.collect()
    print("Exported FeatureExtractor.")


def export_semantic_encoder_graph(
    config: Any,
    auxiliary_paths: dict[str, str],
) -> None:
    from transformers import Wav2Vec2BertModel

    semantic_model = Wav2Vec2BertModel.from_pretrained(
        auxiliary_paths["w2v_bert"],
        local_files_only=True,
    )
    semantic_statistics = torch.load(
        models_path / str(config.w2v_stat),
        map_location="cpu",
        weights_only=True,
    )
    semantic_mean = semantic_statistics["mean"]
    semantic_std = torch.sqrt(semantic_statistics["var"])
    del semantic_statistics
    wrapper = IndexTTS2SemanticEncoder(
        freeze(semantic_model),
        semantic_mean,
        semantic_std,
    )
    input_size = int(semantic_model.config.feature_projection_input_dim)
    input_features = torch.zeros(1, 100, input_size, dtype=torch.float32)
    attention_mask = torch.ones(1, 100, dtype=torch.int64)
    export_onnx(
        wrapper,
        (input_features, attention_mask),
        onnx_models["semantic_encoder"],
        input_names=["input_features", "attention_mask"],
        output_names=["semantic_features"],
        dynamic_axes={
            "input_features": {1: "feature_frames"},
            "attention_mask": {1: "feature_frames"},
            "semantic_features": {1: "semantic_frames"},
        },
    )
    del wrapper, semantic_model, semantic_mean, semantic_std
    gc.collect()
    print("Exported SemanticEncoder.")


def _trace_strategy_controls() -> dict[str, torch.Tensor]:
    # These values only exercise dynamic graph inputs during export.
    return {
        "penalty_value": torch.tensor([0.8], dtype=torch.float32),
        "penalty_range": torch.tensor([10], dtype=torch.int64),
        "temperature": torch.tensor([0.8], dtype=torch.float32),
        "top_k": torch.tensor([20], dtype=torch.int64),
        "top_p": torch.tensor([0.9], dtype=torch.float32),
        "repetition_penalty": torch.tensor([1.2], dtype=torch.float32),
    }


def export_gpt_graphs(config: Any) -> None:
    adapter = active_adapter()
    gpt = adapter.load_gpt(config)
    num_layers = int(config.gpt.layers)
    num_heads = int(config.gpt.heads)
    hidden_size = int(config.gpt.model_dim)
    semantic_hidden_size = int(config.semantic_codec.hidden_size)
    style_embed_size = int(config.s2mel.style_encoder.dim)
    head_dim = hidden_size // num_heads
    vocabulary_size = int(config.gpt.number_mel_codes)
    kv_dtype = torch.float16 if USE_F16_KV else torch.float32

    speaker_matrix = torch.load(
        models_path / str(config.spk_matrix),
        map_location="cpu",
        weights_only=True,
    )
    emotion_matrix = torch.load(
        models_path / str(config.emo_matrix),
        map_location="cpu",
        weights_only=True,
    )
    conditioning = adapter.make_conditioning(
        gpt,
        speaker_matrix,
        emotion_matrix,
    )
    speaker_features = torch.zeros(1, 96, semantic_hidden_size)
    emotion_features = torch.zeros(1, 80, semantic_hidden_size)
    speaker_lengths = torch.tensor([96], dtype=torch.int64)
    emotion_lengths = torch.tensor([80], dtype=torch.int64)
    emotion_alpha = torch.tensor([1.0], dtype=torch.float32)
    style = torch.zeros(1, style_embed_size, dtype=torch.float32)
    emotion_weights = torch.zeros(8, dtype=torch.float32)
    export_onnx(
        conditioning,
        (
            speaker_features,
            speaker_lengths,
            emotion_features,
            emotion_lengths,
            emotion_alpha,
            style,
            emotion_weights,
        ),
        onnx_models["conditioning"],
        input_names=[
            "speaker_features",
            "speaker_lengths",
            "emotion_features",
            "emotion_lengths",
            "emotion_alpha",
            "style",
            "emotion_weights",
        ],
        output_names=["speaker_latent", "emotion_vector"],
        dynamic_axes={
            "speaker_features": {1: "speaker_frames"},
            "emotion_features": {1: "emotion_frames"},
        },
    )
    del conditioning, speaker_matrix, emotion_matrix

    speaker_latent = adapter.speaker_latent_example(hidden_size)
    emotion_vector = torch.zeros(1, hidden_size)
    text_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.int32)
    target, target_inputs, target_input_names = adapter.target_export(
        gpt,
        speaker_latent,
        emotion_vector,
        text_ids,
    )
    export_onnx(
        target,
        target_inputs,
        onnx_models["target"],
        input_names=target_input_names,
        output_names=["hidden_states"],
        dynamic_axes={
            "text_ids": {1: "text_length"},
            "hidden_states": {1: "prefill_length"},
        },
    )
    del target

    main_core = IndexTTS2Main(gpt, MAX_SIGNAL_LENGTH)
    state_input_names = [f"in_key_{index}" for index in range(num_layers)] + [
        f"in_value_{index}" for index in range(num_layers)
    ]
    state_output_names = [f"out_key_{index}" for index in range(num_layers)] + [
        f"out_value_{index}" for index in range(num_layers)
    ]
    state_axes: dict[str, dict[int, str]] = {}
    for index in range(num_layers):
        state_axes[f"in_key_{index}"] = {3: "history_length"}
        state_axes[f"out_key_{index}"] = {3: "kv_sequence_length"}
        state_axes[f"in_value_{index}"] = {2: "history_length"}
        state_axes[f"out_value_{index}"] = {2: "kv_sequence_length"}

    hidden_states = torch.zeros(1, 40, hidden_size)
    hidden_step = torch.zeros(1, 1, hidden_size)
    history_length = torch.tensor([40], dtype=torch.int64)
    save_ids = torch.zeros(1, 8, dtype=torch.int32)
    past_key = torch.zeros(1, num_heads, head_dim, 40, dtype=kv_dtype)
    past_value = torch.zeros(1, num_heads, 40, head_dim, dtype=kv_dtype)
    state_inputs = [past_key] * num_layers + [past_value] * num_layers
    controls = _trace_strategy_controls()

    for strategy in onnx_model_main_prefill:
        if strategy == "greedy":
            decode_control_names: list[str] = []
        elif strategy == "penalty_greedy":
            decode_control_names = ["penalty_value", "penalty_range"]
        else:
            decode_control_names = [
                "temperature",
                "top_k",
                "top_p",
                "repetition_penalty",
            ]
        prefill_control_names = (
            ["temperature", "top_k", "top_p"] if strategy == "sampling" else []
        )
        prefill_controls = tuple(controls[name] for name in prefill_control_names)
        decode_controls = tuple(controls[name] for name in decode_control_names)

        prefill = IndexTTS2MainPrefillStrategy(
            main_core,
            strategy,
            vocabulary_size,
        )
        export_onnx(
            prefill,
            (hidden_states, *prefill_controls),
            onnx_model_main_prefill[strategy],
            input_names=["hidden_states", *prefill_control_names],
            output_names=[
                *state_output_names,
                "last_hidden_state",
                "next_token",
                "kv_sequence_length",
            ],
            dynamic_axes={
                **{
                    name: axes
                    for name, axes in state_axes.items()
                    if name.startswith("out_")
                },
                "hidden_states": {1: "prefill_length"},
            },
        )
        del prefill

        decode = IndexTTS2MainDecodeStrategy(
            main_core,
            strategy,
            vocabulary_size,
        )
        export_onnx(
            decode,
            (
                *state_inputs,
                hidden_step,
                save_ids,
                history_length,
                *decode_controls,
            ),
            onnx_model_main_decode[strategy],
            input_names=[
                *state_input_names,
                "hidden_states",
                "save_ids_in",
                "history_length",
                *decode_control_names,
            ],
            output_names=[
                *state_output_names,
                "last_hidden_state",
                "next_token",
                "save_ids_out",
                "kv_sequence_length",
            ],
            dynamic_axes={
                **state_axes,
                "save_ids_in": {1: "save_ids_length"},
                "save_ids_out": {1: "save_ids_length_out"},
            },
        )
        del decode

    decode_embed = IndexTTS2DecodeEmbed(gpt)
    current_token = torch.zeros(1, 1, dtype=torch.int32)
    export_onnx(
        decode_embed,
        (current_token, save_ids),
        onnx_models["decode_embed"],
        input_names=["current_token", "save_ids_in"],
        output_names=["hidden_states"],
        dynamic_axes={"save_ids_in": {1: "save_ids_length"}},
    )
    del decode_embed

    latent = adapter.make_latent(
        gpt,
        main_core,
        config,
        semantic_hidden_size,
    )
    mel_codes = torch.zeros(1, 20, dtype=torch.int32)
    export_onnx(
        latent,
        (speaker_latent, emotion_vector, text_ids, mel_codes),
        onnx_models["latent"],
        input_names=["speaker_latent", "emotion_vector", "text_ids", "mel_codes"],
        output_names=["gpt_latent"],
        dynamic_axes={
            "text_ids": {1: "text_length"},
            "mel_codes": {1: "code_length"},
            "gpt_latent": {1: "code_length"},
        },
    )
    del latent, main_core, gpt
    gc.collect()
    print("Exported Conditioning, TargetPreprocess, GPT strategy, and Latent graphs.")


def export_acoustic_graphs(
    config: Any,
    auxiliary_paths: dict[str, str],
) -> None:
    adapter = active_adapter()
    semantic_hidden_size = int(config.semantic_codec.hidden_size)
    style_embed_size = int(config.s2mel.style_encoder.dim)
    semantic_codec, s2mel, campplus = adapter.load_acoustic_modules(
        config,
        auxiliary_paths,
    )
    cfm_estimator = s2mel.models["cfm"].estimator
    replacements = replace_wavenet_sconv1d(cfm_estimator.wavenet)
    cfm_projection = IndexTTS2CFMStaticProjection(
        cfm_estimator,
        style_embed_size,
    )
    reference = adapter.make_reference(
        semantic_codec,
        s2mel,
        campplus,
        cfm_projection,
    )
    semantic_features = torch.zeros(1, 64, semantic_hidden_size)
    reference_mel = torch.zeros(1, int(config.s2mel.DiT.in_channels), 100)
    style_features = torch.zeros(1, 200, 80)
    export_onnx(
        reference,
        (semantic_features, reference_mel, style_features),
        onnx_models["reference"],
        input_names=["semantic_features", "reference_mel", "style_features"],
        output_names=["style", "reference_hidden", "null_hidden"],
        dynamic_axes={
            "semantic_features": {1: "semantic_frames"},
            "reference_mel": {2: "reference_mel_frames"},
            "style_features": {1: "style_frames"},
            "reference_hidden": {1: "reference_mel_frames"},
        },
    )
    del reference

    acoustic, acoustic_inputs, acoustic_input_names = adapter.acoustic_export(
        semantic_codec,
        s2mel,
        cfm_projection,
        config,
        style_embed_size,
        semantic_hidden_size,
    )
    export_onnx(
        acoustic,
        acoustic_inputs,
        onnx_models["acoustic"],
        input_names=acoustic_input_names,
        output_names=[
            "static_hidden",
            "target_length",
            "cfg_scales",
            "cfg_scale_sum",
            "target_mask",
        ],
        dynamic_axes={
            "mel_codes": {1: "code_length"},
            "gpt_latent": {1: "code_length"},
            "reference_hidden": {1: "reference_mel_frames"},
            "static_hidden": {1: "total_mel_frames"},
            "target_mask": {1: "total_mel_frames"},
        },
    )
    del acoustic

    total_frames = 64
    mel_features = torch.zeros(1, total_frames, int(config.s2mel.DiT.in_channels))
    step_index = torch.zeros(1, dtype=torch.int64)
    static_hidden = torch.zeros(2, total_frames, int(config.s2mel.DiT.hidden_dim))
    target_mask = torch.ones(1, total_frames, 1, dtype=torch.float32)

    cfm_step = IndexTTS2CFMStep(cfm_estimator, adapter.cfm_time_schedule)
    trace_cfg_scales = torch.tensor([1.7, -0.7], dtype=torch.float32).view(2, 1, 1)
    trace_cfg_scale_sum = trace_cfg_scales.sum().reshape(1)
    export_onnx(
        cfm_step,
        (
            mel_features,
            step_index,
            static_hidden,
            trace_cfg_scales,
            trace_cfg_scale_sum,
            target_mask,
        ),
        onnx_models["cfm_estimator"],
        input_names=[
            "mel_features",
            "step_index",
            "static_hidden",
            "cfg_scales",
            "cfg_scale_sum",
            "target_mask",
        ],
        output_names=["next_mel_features"],
        dynamic_axes={
            "mel_features": {1: "total_mel_frames"},
            "static_hidden": {1: "total_mel_frames"},
            "cfg_scales": {0: "cfg_branches"},
            "target_mask": {1: "total_mel_frames"},
            "next_mel_features": {1: "total_mel_frames"},
        },
    )
    del (
        cfm_step,
        cfm_estimator,
        cfm_projection,
        static_hidden,
        semantic_codec,
        s2mel,
        campplus,
    )
    gc.collect()
    print("Exported Reference, Acoustic, and cached CFMEstimator graphs.")


def export_decoder_graph(
    config: Any,
    auxiliary_paths: dict[str, str],
) -> None:
    from indextts.s2mel.modules.bigvgan import bigvgan

    vocoder_path = Path(auxiliary_paths["bigvgan"])
    vocoder = bigvgan.BigVGAN(
        bigvgan.load_hparams_from_json(vocoder_path / "config.json"),
        use_cuda_kernel=False,
    )
    checkpoint = torch.load(
        vocoder_path / "bigvgan_generator.pt",
        map_location="cpu",
        weights_only=True,
    )
    try:
        vocoder.load_state_dict(checkpoint["generator"], strict=True)
    except RuntimeError:
        vocoder.remove_weight_norm()
        vocoder.load_state_dict(checkpoint["generator"], strict=True)
    else:
        vocoder.remove_weight_norm()
    del checkpoint
    decoder = IndexTTS2Decoder(
        freeze(vocoder),
        int(config.s2mel.preprocess_params.sr),
    )
    mel = torch.zeros(1, int(config.s2mel.DiT.in_channels), 10)
    export_onnx(
        decoder,
        (mel,),
        onnx_models["decoder"],
        input_names=["mel"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {2: "target_mel_frames"},
            "waveform": {2: "audio_samples"},
        },
    )
    del decoder, vocoder
    gc.collect()
    print("Exported BigVGAN-v2 Decoder.")


def export_metadata_graph() -> None:
    marker = torch.zeros(1, dtype=torch.int64)
    export_onnx(
        METADATA_CARRIER(),
        (marker,),
        onnx_models["metadata"],
        input_names=["metadata_marker"],
        output_names=["metadata_marker_out"],
        dynamic_axes={},
    )


def assemble_graph_package(
    metadata: dict[str, str],
) -> tuple[dict[str, Any], dict[str, int]]:
    component_graphs = [Path(path) for path in onnx_models.values()]
    component_graphs.extend(Path(path) for path in onnx_model_main_prefill.values())
    component_graphs.extend(Path(path) for path in onnx_model_main_decode.values())
    component_graphs.extend(
        (
            Path(onnx_model_emotion_text_prefill),
            Path(onnx_model_emotion_text_decode),
        )
    )
    shared_stats = bundle_shared_initializers(
        onnx_folder,
        model_paths=component_graphs,
        metadata=metadata,
    )
    build_reference_preprocess_graph(onnx_folder)
    build_target_prefill_graphs(onnx_folder, tuple(onnx_model_target_prefill))
    build_decode_step_graphs(onnx_folder, tuple(onnx_model_decode_step))
    build_synthesis_graph(onnx_folder)
    build_decoder_postprocess_graph(onnx_models["decoder"])

    final_graphs = [
        Path(onnx_model_reference_preprocess),
        Path(onnx_models["conditioning"]),
        Path(onnx_model_synthesis),
        Path(onnx_models["cfm_estimator"]),
        Path(onnx_models["decoder"]),
        Path(onnx_models["metadata"]),
        Path(onnx_model_emotion_text_prefill),
        Path(onnx_model_emotion_text_decode),
        *[Path(path) for path in onnx_model_target_prefill.values()],
        *[Path(path) for path in onnx_model_decode_step.values()],
    ]
    for graph_path in final_graphs:
        write_onnx_metadata(graph_path, metadata)
    replace_onnx_metadata(onnx_models["metadata"], metadata)
    return shared_stats


def run_export(profile: ExportProfile, adapter: ExportAdapter | None = None) -> None:
    configure_export(profile, adapter)
    config_path = models_path / "config.yaml"
    from transformers import AutoConfig, AutoTokenizer

    config = OmegaConf.load(config_path)
    emotion_text_config = AutoConfig.from_pretrained(
        emotion_text_model_path,
        local_files_only=True,
    )
    emotion_text_tokenizer = AutoTokenizer.from_pretrained(
        emotion_text_model_path,
        local_files_only=True,
    )
    content_marker = "INDEXTTS2_EMOTION_CONTENT_MARKER_7F3A9C"
    rendered_prompt = emotion_text_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "文本情感分类"},
            {"role": "user", "content": content_marker},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_prefix, _, prompt_suffix = rendered_prompt.partition(content_marker)
    emotion_text_content_prefix = prompt_prefix[-1:]
    emotion_text_prompt_prefix_token_ids = emotion_text_tokenizer.encode(
        prompt_prefix[:-1],
        add_special_tokens=False,
    )
    emotion_text_prompt_suffix_token_ids = emotion_text_tokenizer.encode(
        prompt_suffix,
        add_special_tokens=False,
    )
    emotion_text_think_end_token_id = emotion_text_tokenizer.convert_tokens_to_ids(
        "</think>"
    )
    del emotion_text_tokenizer
    metadata = build_export_metadata(
        config,
        emotion_text_config,
        emotion_text_prompt_prefix_token_ids,
        emotion_text_prompt_suffix_token_ids,
        emotion_text_content_prefix,
        emotion_text_think_end_token_id,
    )
    print(
        "IndexTTS2 export configuration:",
        f"GPT={config.gpt.layers}x{config.gpt.model_dim}",
        f"DiT={config.s2mel.DiT.depth}x{config.s2mel.DiT.hidden_dim}",
    )
    if onnx_folder.exists():
        shutil.rmtree(onnx_folder)
    onnx_folder.mkdir(parents=True)
    copy_tokenizer_assets(onnx_folder)
    auxiliary_paths = active_adapter().resolve_auxiliary_paths(config)

    print("Starting staged IndexTTS2 export...")
    export_feature_extractor_graph(config, auxiliary_paths)
    export_semantic_encoder_graph(config, auxiliary_paths)
    export_emotion_text_graphs(emotion_text_config)
    export_gpt_graphs(config)
    export_acoustic_graphs(config, auxiliary_paths)
    export_decoder_graph(config, auxiliary_paths)
    export_metadata_graph()
    shared_stats = assemble_graph_package(metadata)
    print(
        f"IndexTTS2 export complete: {shared_stats['initializer_references']} references, "
        f"{shared_stats['unique_initializers']} unique tensors, "
        f"{(onnx_folder / SHARED_DATA_NAME).stat().st_size / (1024 * 1024):.2f} MiB shared blob."
    )