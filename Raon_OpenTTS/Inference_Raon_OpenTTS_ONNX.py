from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import jieba
import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from pydub import AudioSegment, silence
from pypinyin import Style, lazy_pinyin


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Example_Audio import model_reference  # noqa: E402
from Raon_Config import (  # noqa: E402
    UNSUPPORTED_TRAILING_VOCAB_TOKENS,
    require_architecture,
)


# ============================== Configuration ==============================
# Edit demo values directly; CLI options remain available for one-off overrides.
_REFERENCE_AUDIO, REFERENCE_TEXT = model_reference("raon_opentts")
REFERENCE_AUDIO_PATH = Path(_REFERENCE_AUDIO)
TARGET_TEXT = "Some call me Dake, others call me Q Q."
OUTPUT_AUDIO_PATH = SCRIPT_DIR / "generated.wav"
# ===========================================================================

REQUIRED_METADATA_KEYS = {
    "schema_version",
    "architecture",
    "model_name",
    "backbone",
    "vocoder",
    "vocoder_upsample_factor",
    "model_file_name_preprocess",
    "model_file_name_transformer",
    "model_file_name_decode",
    "model_file_name_metadata",
    "sample_rate",
    "in_sample_rate",
    "out_sample_rate",
    "n_mels",
    "n_fft",
    "window_length",
    "hop_length",
    "window_type",
    "center_pad",
    "mel_norm",
    "mel_scale",
    "nfe_step",
    "cfg_strength",
    "sway_coefficient",
    "max_signal_length",
    "model_dim",
    "model_depth",
    "model_heads",
    "head_dim",
    "ff_mult",
    "text_dim",
    "text_conv_layers",
    "text_mask_padding",
    "qk_norm",
    "pe_attn_head",
    "attn_mask_enabled",
    "logit_softcapping",
    "post_norm",
    "norm_type",
    "vocab_size",
    "vocab_sha256",
    "audio_input_dtype",
    "text_ids_dtype",
    "time_step_dtype",
    "preprocess_output_dtype",
    "transformer_dtype",
    "decode_dtype",
    "opset",
}

@dataclass(frozen=True)
class PackageMetadata:
    values: dict[str, str]
    onnx_folder: Path
    preprocess_path: Path
    transformer_path: Path
    decode_path: Path
    metadata_path: Path
    sample_rate: int
    hop_length: int
    n_fft: int
    nfe_step: int
    max_signal_length: int


@dataclass(frozen=True)
class ChunkPlan:
    text_token_ids: tuple[int, ...]
    text_length: int
    mel_frames: int
    ref_signal_len: int
    requested_generated_frames: int
    generated_frames: int
    max_duration: int


def _numpy_dtype(argument: Any) -> np.dtype[Any]:
    match = re.fullmatch(r"tensor\(([^)]+)\)", argument.type)
    if match is None:
        raise ValueError(f"Unsupported ONNX value type: {argument.type!r}")
    try:
        tensor_dtype = onnx.TensorProto.DataType.Value(match.group(1).upper())
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_dtype))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported ONNX tensor type: {argument.type!r}") from error


def _resolved_shape(
    argument: Any, dynamic_dimension: int | None = None
) -> tuple[int, ...]:
    if dynamic_dimension is not None and dynamic_dimension < 0:
        raise ValueError("Dynamic ONNX dimensions must be non-negative")
    shape: list[int] = []
    for axis, dimension in enumerate(argument.shape):
        if isinstance(dimension, int):
            shape.append(dimension)
        elif dynamic_dimension is not None:
            shape.append(dynamic_dimension)
        else:
            raise ValueError(
                f"ONNX value {argument.name!r} has unresolved dimension "
                f"{dimension!r} at axis {axis}"
            )
    return tuple(shape)


def load_vocab(path: Path) -> tuple[dict[str, int], str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Vocabulary does not exist or is not a file: {resolved}")
    tokens: list[str] = []
    seen: set[str] = set()
    with resolved.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.endswith("\n"):
                raise ValueError(
                    f"Vocabulary line {index + 1} is not newline-terminated: {resolved}"
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
        raise ValueError(f"Vocabulary is empty: {resolved}")
    vocab_map = {token: index for index, token in enumerate(tokens)}
    if vocab_map.get(" ") != 0:
        raise ValueError(
            f"Vocabulary must map a literal space to token 0; found {vocab_map.get(' ')!r}"
        )
    if sorted(vocab_map.values()) != list(range(len(vocab_map))):
        raise ValueError("Vocabulary token IDs must be contiguous from zero")
    return vocab_map, hashlib.sha256(resolved.read_bytes()).hexdigest()


def convert_char_to_pinyin(text_list: Sequence[str], polyphone: bool = True) -> list[list[str]]:
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)
        jieba.initialize()
    translations = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})

    def is_chinese(character: str) -> bool:
        return "\u3100" <= character <= "\u9fff"

    converted: list[list[str]] = []
    for original in text_list:
        characters: list[str] = []
        for segment in jieba.cut(original.translate(translations)):
            byte_length = len(segment.encode("utf-8"))
            if byte_length == len(segment):
                if characters and byte_length > 1 and characters[-1] not in " :'\"":
                    characters.append(" ")
                characters.extend(segment)
            elif polyphone and byte_length == 3 * len(segment):
                pinyin = lazy_pinyin(segment, style=Style.TONE3, tone_sandhi=True)
                for index, character in enumerate(segment):
                    if is_chinese(character):
                        characters.append(" ")
                    characters.append(pinyin[index])
            else:
                for character in segment:
                    if ord(character) < 256:
                        characters.extend(character)
                    elif is_chinese(character):
                        characters.append(" ")
                        characters.extend(
                            lazy_pinyin(character, style=Style.TONE3, tone_sandhi=True)
                        )
                    else:
                        characters.append(character)
        converted.append(characters)
    return converted


def chunk_text(text: str, max_chars: int = 135) -> list[str]:
    chunks: list[str] = []
    current = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)
    for sentence in sentences:
        if len(current.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current += (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )
        else:
            if current:
                chunks.append(current.strip())
            current = (
                sentence + " "
                if sentence and len(sentence[-1].encode("utf-8")) == 1
                else sentence
            )
    if current:
        chunks.append(current.strip())
    return chunks


def fix_reference_text_ending(text: str) -> str:
    fixed = (text or "").strip()
    if not fixed:
        return fixed
    if not fixed.endswith(". ") and not fixed.endswith("。"):
        fixed = fixed + " " if fixed.endswith(".") else fixed + ". "
    return fixed


def _remove_silence_edges(audio: AudioSegment, threshold: int) -> AudioSegment:
    leading = silence.detect_leading_silence(audio, silence_threshold=threshold)
    audio = audio[leading:]
    trailing_seconds = audio.duration_seconds
    for millisecond in reversed(audio):
        if millisecond.dBFS > threshold:
            break
        trailing_seconds -= 0.001
    return audio[: int(trailing_seconds * 1000)]


def estimate_reference_seconds(
    audio: AudioSegment,
    base_silence_threshold: int = -42,
    target_dbfs: float = -20.0,
    max_gain_db: float = 30.0,
    threshold_margin_db: float = 18.0,
) -> float:
    if audio.dBFS != float("-inf"):
        gain = max(min(target_dbfs - audio.dBFS, max_gain_db), -max_gain_db)
        normalized = audio.apply_gain(gain)
    else:
        normalized = audio
    if normalized.dBFS != float("-inf"):
        dynamic_threshold = normalized.dBFS - threshold_margin_db
        threshold = min(base_silence_threshold, int(dynamic_threshold))
    else:
        threshold = base_silence_threshold
    trimmed = _remove_silence_edges(normalized, threshold)
    trimmed = trimmed + AudioSegment.silent(duration=50)
    return audio.duration_seconds if len(trimmed) < 80 else trimmed.duration_seconds


def load_reference_audio(path: Path, sample_rate: int, n_fft: int) -> tuple[np.ndarray, AudioSegment]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Reference audio does not exist or is not a file: {resolved}")
    try:
        original = AudioSegment.from_file(resolved)
    except Exception as error:
        raise ValueError(f"Unable to decode reference audio {resolved}: {error}") from error
    mono = original.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
    samples = np.asarray(mono.get_array_of_samples(), dtype=np.int16)
    if samples.size < n_fft:
        raise ValueError(
            f"Reference audio is too short: {samples.size} samples; at least {n_fft} are required"
        )
    audio = np.ascontiguousarray(samples.astype(np.float32) / 32768.0)
    peak = float(np.max(np.abs(audio), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float32))))
    if not math.isfinite(peak) or not math.isfinite(rms) or peak <= 1e-6 or rms <= 1e-7:
        raise ValueError("Reference audio is silent or contains no usable finite signal")
    return audio, original


def plan_chunk(
    *,
    audio_length: int,
    reference_seconds_for_length: float,
    reference_text: str,
    target_text: str,
    vocab_map: dict[str, int],
    sample_rate: int,
    hop_length: int,
    n_fft: int,
    speed: float,
    fix_duration: float | None,
    use_vad_duration: bool,
    max_signal_length: int,
) -> ChunkPlan:
    if audio_length < n_fft:
        raise ValueError(f"Reference audio must contain at least {n_fft} samples")
    if not reference_text:
        raise ValueError("Reference text must not be empty")
    if not target_text:
        raise ValueError("Target text chunk must not be empty")
    if speed <= 0 or not math.isfinite(speed):
        raise ValueError("Speed must be a finite value greater than zero")
    if fix_duration is not None and (fix_duration <= 0 or not math.isfinite(fix_duration)):
        raise ValueError("Fixed duration must be a finite value greater than zero")

    local_speed = 0.3 if len(target_text.encode("utf-8")) < 10 else speed
    ref_signal_len = audio_length // hop_length
    mel_frames = (audio_length - n_fft) // hop_length + 1
    if fix_duration is not None:
        requested_generated_frames = int(fix_duration * sample_rate / hop_length)
    else:
        reference_frames_for_length = int(
            reference_seconds_for_length * sample_rate / hop_length
        )
        reference_seconds = reference_frames_for_length * hop_length / sample_rate
        seconds_per_byte = reference_seconds / max(len(reference_text.encode("utf-8")), 1)
        if use_vad_duration:
            seconds_per_byte = min(seconds_per_byte, 1.0 / 12.0)
        generated_seconds = (
            seconds_per_byte * len(target_text.encode("utf-8")) / max(local_speed, 1e-6)
        )
        requested_generated_frames = int(generated_seconds * sample_rate / hop_length)
    requested_generated_frames = max(requested_generated_frames, 1)

    normalized_text = convert_char_to_pinyin([reference_text + target_text])[0]
    unknown_tokens = sorted({token for token in normalized_text if token not in vocab_map})
    if unknown_tokens:
        preview = ", ".join(repr(token) for token in unknown_tokens[:12])
        if len(unknown_tokens) > 12:
            preview += f", ... ({len(unknown_tokens)} unique tokens total)"
        raise ValueError(
            "Raon-OpenTTS is an English-only model and the normalized text contains "
            f"unsupported tokens: {preview}. Use an English reference transcript and target text."
        )
    token_ids = tuple(vocab_map[token] for token in normalized_text)
    if not token_ids:
        raise ValueError("Text normalization produced an empty token sequence")
    requested_duration = ref_signal_len + requested_generated_frames
    max_duration = max(max(len(token_ids), mel_frames) + 1, requested_duration)
    if max_duration > max_signal_length:
        raise ValueError(
            "Required duration exceeds the exported 4096-frame capacity: "
            f"required {max_duration}, maximum {max_signal_length}. "
            "Use shorter reference/target text or split the target more aggressively."
        )
    generated_frames = max_duration - ref_signal_len
    if generated_frames < 1:
        raise RuntimeError(
            f"Internal duration invariant failed: duration={max_duration}, ref={ref_signal_len}"
        )
    return ChunkPlan(
        text_token_ids=token_ids,
        text_length=len(token_ids),
        mel_frames=mel_frames,
        ref_signal_len=ref_signal_len,
        requested_generated_frames=requested_generated_frames,
        generated_frames=generated_frames,
        max_duration=max_duration,
    )


def crossfade_waveforms(
    waveforms: Sequence[np.ndarray], sample_rate: int, duration_seconds: float
) -> np.ndarray:
    if not waveforms:
        raise ValueError("No generated waveforms are available for crossfade")
    if duration_seconds < 0 or not math.isfinite(duration_seconds):
        raise ValueError("Crossfade duration must be a finite non-negative value")
    combined = np.asarray(waveforms[0]).reshape(-1)
    for next_waveform in waveforms[1:]:
        following = np.asarray(next_waveform).reshape(-1)
        overlap = min(int(duration_seconds * sample_rate), combined.size, following.size)
        if overlap <= 0:
            combined = np.concatenate((combined, following))
            continue
        fade_out = np.linspace(1.0, 0.0, overlap)
        fade_in = np.linspace(0.0, 1.0, overlap)
        blended = combined[-overlap:] * fade_out + following[:overlap] * fade_in
        combined = np.concatenate((combined[:-overlap], blended, following[overlap:]))
    return np.asarray(combined, dtype=np.float32)


def _parse_int(metadata: dict[str, str], key: str, minimum: int = 1) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Metadata key {key!r} must contain an integer") from error
    if value < minimum:
        raise ValueError(f"Metadata key {key!r} must be at least {minimum}, found {value}")
    return value


def _parse_float(metadata: dict[str, str], key: str) -> float:
    try:
        value = float(metadata[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"Metadata key {key!r} must contain a number") from error
    if not math.isfinite(value):
        raise ValueError(f"Metadata key {key!r} must be finite")
    return value


def _resolve_graph_path(folder: Path, value: str, key: str) -> Path:
    if not value or Path(value).name != value or not value.endswith(".onnx"):
        raise ValueError(f"Metadata key {key!r} contains an unsafe graph filename: {value!r}")
    path = (folder / value).resolve()
    if path.parent != folder:
        raise ValueError(f"Metadata graph path escapes package folder: {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Metadata references a missing ONNX graph: {path}")
    return path


def _metadata_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _validate_onnx_container(path: Path) -> None:
    onnx.checker.check_model(str(path))
    model = onnx.load(path, load_external_data=False)
    default_opsets = [
        item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}
    ]
    if default_opsets != [20]:
        raise ValueError(
            f"{path.name} must import default-domain opset 20 exactly once; "
            f"found {default_opsets}"
        )


def load_package_metadata(
    onnx_folder: Path, vocab_path: Path | None
) -> tuple[PackageMetadata, dict[str, int]]:
    folder = onnx_folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"ONNX package folder does not exist: {folder}")
    metadata_path = folder / "Raon_Metadata.onnx"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Required metadata graph is missing: {metadata_path}")
    _validate_onnx_container(metadata_path)
    metadata_model = onnx.load(metadata_path, load_external_data=False)
    metadata_nodes = list(metadata_model.graph.node)
    graph_inputs = list(metadata_model.graph.input)
    graph_outputs = list(metadata_model.graph.output)
    if (
        len(metadata_nodes) != 1
        or len(graph_inputs) != 1
        or len(graph_outputs) != 1
        or metadata_nodes[0].op_type != "Identity"
        or metadata_nodes[0].domain not in {"", "ai.onnx"}
        or metadata_nodes[0].attribute
        or list(metadata_nodes[0].input) != [graph_inputs[0].name]
        or list(metadata_nodes[0].output) != [graph_outputs[0].name]
        or metadata_model.graph.initializer
        or metadata_model.functions
    ):
        raise ValueError(
            "Raon_Metadata.onnx must contain only one standard-domain Identity "
            "connecting its sole input and output"
        )
    session = _metadata_session(metadata_path)
    metadata = dict(session.get_modelmeta().custom_metadata_map)
    marker_inputs = session.get_inputs()
    marker_outputs = session.get_outputs()
    if len(marker_inputs) != 1 or len(marker_outputs) != 1:
        raise ValueError("Raon_Metadata.onnx must expose one input and one output")
    marker_input = marker_inputs[0]
    marker_output_info = marker_outputs[0]
    if (
        marker_input.type != marker_output_info.type
        or marker_input.shape != marker_output_info.shape
    ):
        raise ValueError("Raon_Metadata.onnx Identity input and output must have matching types")
    marker = np.ones(_resolved_shape(marker_input), dtype=_numpy_dtype(marker_input))
    marker_output = session.run(
        [marker_output_info.name], {marker_input.name: marker}
    )[0]
    del session
    if not np.array_equal(marker_output, marker):
        raise ValueError("Raon_Metadata.onnx must be an identity marker graph")

    missing = sorted(REQUIRED_METADATA_KEYS - metadata.keys())
    if missing:
        raise ValueError(f"Raon_Metadata.onnx is missing required metadata keys: {missing}")
    architecture = require_architecture(metadata["model_name"], "package model_name")
    model_name = architecture.model_name
    expected_strings = {
        "schema_version": "1",
        "architecture": f"Raon-OpenTTS-{model_name}-DiT-HiFiGAN",
        "model_name": model_name,
        "backbone": "DiT",
        "vocoder": "sbhifigan16k",
        "window_type": "hann",
        "center_pad": "0",
        "mel_norm": "slaney",
        "mel_scale": "slaney",
        "text_mask_padding": "1",
        "qk_norm": "null",
        "pe_attn_head": "null",
        "attn_mask_enabled": "0",
        "logit_softcapping": "null",
        "post_norm": "0",
        "norm_type": "rmsnorm",
        "opset": "20",
        "model_file_name_metadata": "Raon_Metadata.onnx",
    }
    mismatches = [
        f"{key}: expected {expected!r}, found {metadata.get(key)!r}"
        for key, expected in expected_strings.items()
        if metadata.get(key) != expected
    ]
    expected_integers = {
        "sample_rate": 16_000,
        "in_sample_rate": 16_000,
        "out_sample_rate": 16_000,
        "n_mels": 80,
        "n_fft": 1024,
        "window_length": 1024,
        "hop_length": 256,
        "nfe_step": 32,
        "max_signal_length": 4096,
        "vocoder_upsample_factor": 256,
        "model_dim": architecture.dim,
        "model_depth": architecture.depth,
        "model_heads": architecture.heads,
        "head_dim": architecture.head_dim,
        "ff_mult": architecture.ff_mult,
        "text_dim": architecture.text_dim,
        "text_conv_layers": architecture.text_conv_layers,
    }
    parsed_integers: dict[str, int] = {}
    for key, expected in expected_integers.items():
        value = _parse_int(metadata, key)
        parsed_integers[key] = value
        if value != expected:
            mismatches.append(f"{key}: expected {expected}, found {value}")
    if not math.isclose(_parse_float(metadata, "cfg_strength"), 2.0, rel_tol=0.0, abs_tol=0.0):
        mismatches.append(f"cfg_strength: expected 2.0, found {metadata['cfg_strength']!r}")
    if not math.isclose(_parse_float(metadata, "sway_coefficient"), -1.0, rel_tol=0.0, abs_tol=0.0):
        mismatches.append(f"sway_coefficient: expected -1.0, found {metadata['sway_coefficient']!r}")
    if mismatches:
        raise ValueError("Incompatible Raon package metadata:\n  " + "\n  ".join(mismatches))

    selected_vocab_path = (
        vocab_path
        if vocab_path is not None
        else Path.home() / "Downloads" / f"Raon-OpenTTS-{model_name}" / "vocab.txt"
    )
    vocab_map, vocab_sha256 = load_vocab(selected_vocab_path)
    vocab_size = _parse_int(metadata, "vocab_size")
    if len(vocab_map) != vocab_size or vocab_sha256 != metadata["vocab_sha256"]:
        raise ValueError(
            "Vocabulary does not match the exported package metadata: "
            f"expected size={vocab_size}, sha256={metadata['vocab_sha256']}; "
            f"found size={len(vocab_map)}, sha256={vocab_sha256}"
        )

    preprocess_path = _resolve_graph_path(
        folder, metadata["model_file_name_preprocess"], "model_file_name_preprocess"
    )
    transformer_path = _resolve_graph_path(
        folder, metadata["model_file_name_transformer"], "model_file_name_transformer"
    )
    decode_path = _resolve_graph_path(
        folder, metadata["model_file_name_decode"], "model_file_name_decode"
    )
    for graph_path in (preprocess_path, transformer_path, decode_path):
        _validate_onnx_container(graph_path)
    package = PackageMetadata(
        values=metadata,
        onnx_folder=folder,
        preprocess_path=preprocess_path,
        transformer_path=transformer_path,
        decode_path=decode_path,
        metadata_path=metadata_path,
        sample_rate=parsed_integers["sample_rate"],
        hop_length=parsed_integers["hop_length"],
        n_fft=parsed_integers["n_fft"],
        nfe_step=parsed_integers["nfe_step"],
        max_signal_length=parsed_integers["max_signal_length"],
    )
    return package, vocab_map


def _require_arity(
    arguments: Iterable[Any], expected_count: int, graph_name: str
) -> tuple[Any, ...]:
    values = tuple(arguments)
    if len(values) != expected_count:
        raise ValueError(
            f"{graph_name} must expose {expected_count} values, found {len(values)}"
        )
    return values


def _same_tensor_contract(left: Any, right: Any) -> bool:
    if left.type != right.type or len(left.shape) != len(right.shape):
        return False
    for left_dimension, right_dimension in zip(left.shape, right.shape):
        left_static = isinstance(left_dimension, int)
        right_static = isinstance(right_dimension, int)
        if left_static != right_static:
            return False
        if left_static and left_dimension != right_dimension:
            return False
    return True


def _require_compatible(source: Any, destination: Any) -> None:
    if not _same_tensor_contract(source, destination):
        raise ValueError(
            f"ONNX pipeline mismatch: {source.name!r} ({source.type}, {source.shape}) "
            f"cannot feed {destination.name!r} ({destination.type}, {destination.shape})"
        )


def _provider_device(provider: str) -> str:
    if provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider", "ROCMExecutionProvider"}:
        return "cuda"
    if provider == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def _provider_options(provider: str, device_id: int) -> dict[str, str]:
    if provider == "CUDAExecutionProvider":
        return {
            "device_id": str(device_id),
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": "1",
            "use_tf32": "1",
        }
    if provider in {"TensorrtExecutionProvider", "ROCMExecutionProvider", "DmlExecutionProvider"}:
        return {"device_id": str(device_id)}
    return {}


def _session_options(provider: str, threads: int, verbose: bool) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.log_severity_level = 0 if verbose else 4
    options.log_verbosity_level = 4
    options.inter_op_num_threads = threads
    options.intra_op_num_threads = threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    options.add_session_config_entry("session.inter_op.allow_spinning", "1")
    options.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
    options.add_session_config_entry("session.graph_optimizations_loop_level", "2")
    options.add_session_config_entry("optimization.enable_gelu_approximation", "1")
    options.add_session_config_entry("optimization.enable_cast_chain_elimination", "1")
    if provider != "CPUExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
    return options


def _create_session(
    path: Path, provider: str, device_id: int, threads: int, verbose: bool
) -> ort.InferenceSession:
    options = _session_options(provider, threads, verbose)
    provider_options = _provider_options(provider, device_id)
    kwargs: dict[str, Any] = {"providers": [provider]}
    if provider_options:
        kwargs["provider_options"] = [provider_options]
    session = ort.InferenceSession(str(path), sess_options=options, **kwargs)
    active = session.get_providers()
    if not active or active[0] != provider:
        raise RuntimeError(
            f"ONNX Runtime silently changed providers for {path.name}: "
            f"requested {provider}, active {active}"
        )
    return session


def _shape_matches(argument: Any, shape: Sequence[int]) -> bool:
    return len(argument.shape) == len(shape) and all(
        not isinstance(declared, int) or declared == actual
        for declared, actual in zip(argument.shape, shape)
    )


def _empty_ortvalue(
    argument: Any,
    dynamic_dimension: int | None,
    device: str,
    device_id: int,
) -> ort.OrtValue:
    shape = _resolved_shape(argument, dynamic_dimension)
    return ort.OrtValue.ortvalue_from_shape_and_type(
        shape, _numpy_dtype(argument), device, device_id
    )


def _ortvalue_from_data(
    argument: Any,
    data: Any,
    dynamic_dimension: int | None,
    device: str,
    device_id: int,
) -> ort.OrtValue:
    shape = _resolved_shape(argument, dynamic_dimension)
    array = np.asarray(data, dtype=_numpy_dtype(argument))
    if array.size != math.prod(shape):
        raise ValueError(
            f"Data for ONNX value {argument.name!r} contains {array.size} elements, "
            f"but model shape {shape} requires {math.prod(shape)}"
        )
    array = np.ascontiguousarray(array.reshape(shape))
    return ort.OrtValue.ortvalue_from_numpy(array, device, device_id)


def _bind_values(
    binding: Any, arguments: Sequence[Any], values: Sequence[ort.OrtValue]
) -> None:
    if len(arguments) != len(values):
        raise ValueError(
            f"Cannot bind {len(values)} values to {len(arguments)} ONNX inputs"
        )
    for argument, value in zip(arguments, values, strict=True):
        binding.bind_ortvalue_input(argument.name, value)


class RaonONNXRuntime:
    def __init__(
        self,
        package: PackageMetadata,
        provider: str,
        device_id: int,
        threads: int,
        seed: int,
        verbose: bool,
    ) -> None:
        available = ort.get_available_providers()
        if provider not in available:
            raise RuntimeError(
                f"Requested execution provider {provider!r} is unavailable; available providers: {available}"
            )
        if device_id < 0:
            raise ValueError("device_id must be non-negative")
        if threads < 0:
            raise ValueError("threads must be non-negative")
        ort.set_seed(seed)
        self.package = package
        self.provider = provider
        self.device_id = device_id
        self.device = _provider_device(provider)
        self.run_options = ort.RunOptions()
        self.run_options.log_severity_level = 0 if verbose else 4
        self.preprocess = _create_session(
            package.preprocess_path, provider, device_id, threads, verbose
        )
        self.transformer = _create_session(
            package.transformer_path, provider, device_id, threads, verbose
        )
        self.decode = _create_session(package.decode_path, provider, device_id, threads, verbose)
        self._validate_metadata_consistency()
        self._validate_interfaces()
        self.step_buffers = tuple(
            _ortvalue_from_data(
                self.transformer_inputs[-1],
                [step],
                None,
                self.device,
                self.device_id,
            )
            for step in range(package.nfe_step)
        )

    def _validate_metadata_consistency(self) -> None:
        for session, path in (
            (self.preprocess, self.package.preprocess_path),
            (self.transformer, self.package.transformer_path),
            (self.decode, self.package.decode_path),
        ):
            actual = session.get_modelmeta().custom_metadata_map
            missing = [
                key
                for key in REQUIRED_METADATA_KEYS
                if actual.get(key) != self.package.values.get(key)
            ]
            if missing:
                raise ValueError(
                    f"Graph metadata is missing or inconsistent in {path.name}: {sorted(missing)}"
                )

    def _validate_interfaces(self) -> None:
        self.preprocess_inputs = _require_arity(
            self.preprocess.get_inputs(), 3, f"{self.package.preprocess_path.name} inputs"
        )
        self.preprocess_outputs = _require_arity(
            self.preprocess.get_outputs(), 7, f"{self.package.preprocess_path.name} outputs"
        )
        self.transformer_inputs = _require_arity(
            self.transformer.get_inputs(),
            6,
            f"{self.package.transformer_path.name} inputs",
        )
        self.transformer_outputs = _require_arity(
            self.transformer.get_outputs(),
            1,
            f"{self.package.transformer_path.name} outputs",
        )
        self.decode_inputs = _require_arity(
            self.decode.get_inputs(), 3, f"{self.package.decode_path.name} inputs"
        )
        self.decode_outputs = _require_arity(
            self.decode.get_outputs(), 1, f"{self.package.decode_path.name} outputs"
        )

        for source, destination in zip(
            self.preprocess_outputs[:5], self.transformer_inputs[:5], strict=True
        ):
            _require_compatible(source, destination)
        _require_compatible(self.transformer_inputs[0], self.transformer_outputs[0])
        _require_compatible(self.transformer_outputs[0], self.decode_inputs[0])
        for source, destination in zip(
            self.preprocess_outputs[5:], self.decode_inputs[1:], strict=True
        ):
            _require_compatible(source, destination)

        dtype_metadata = (
            ("audio_input_dtype", self.preprocess_inputs[0]),
            ("text_ids_dtype", self.preprocess_inputs[1]),
            ("preprocess_output_dtype", self.preprocess_outputs[0]),
            ("transformer_dtype", self.transformer_inputs[0]),
            ("time_step_dtype", self.transformer_inputs[-1]),
            ("decode_dtype", self.decode_outputs[0]),
        )
        mismatches = [
            f"{key}: metadata={self.package.values[key]!r}, model={_numpy_dtype(value).name!r}"
            for key, value in dtype_metadata
            if self.package.values[key] != _numpy_dtype(value).name
        ]
        if mismatches:
            raise ValueError(
                "ONNX tensor dtypes disagree with package metadata:\n  "
                + "\n  ".join(mismatches)
            )

    def _run(self, session: ort.InferenceSession, binding: Any) -> None:
        session.run_with_iobinding(binding, self.run_options)

    def synthesize_chunk(
        self, audio: np.ndarray, plan: ChunkPlan, show_progress: bool
    ) -> np.ndarray:
        duration = plan.max_duration
        preprocess_input_values = (
            _ortvalue_from_data(
                self.preprocess_inputs[0],
                audio,
                audio.size,
                "cpu",
                self.device_id,
            ),
            _ortvalue_from_data(
                self.preprocess_inputs[1],
                plan.text_token_ids,
                plan.text_length,
                "cpu",
                self.device_id,
            ),
            _ortvalue_from_data(
                self.preprocess_inputs[2],
                [duration],
                None,
                "cpu",
                self.device_id,
            ),
        )
        preprocess_output_values = tuple(
            _empty_ortvalue(argument, duration, "cpu", self.device_id)
            for argument in self.preprocess_outputs
        )
        preprocess_binding = self.preprocess.io_binding()
        _bind_values(
            preprocess_binding,
            self.preprocess_inputs,
            preprocess_input_values,
        )
        for argument, value in zip(
            self.preprocess_outputs, preprocess_output_values, strict=True
        ):
            preprocess_binding.bind_ortvalue_output(
                argument.name, value
            )
        self._run(self.preprocess, preprocess_binding)

        if self.device == "cpu":
            state_buffers = (
                preprocess_output_values[0],
                _empty_ortvalue(
                    self.transformer_outputs[0],
                    duration,
                    "cpu",
                    self.device_id,
                ),
            )
            constant_values = preprocess_output_values[1:5]
        else:
            state_buffers = (
                _empty_ortvalue(
                    self.transformer_inputs[0],
                    duration,
                    self.device,
                    self.device_id,
                ),
                _empty_ortvalue(
                    self.transformer_outputs[0],
                    duration,
                    self.device,
                    self.device_id,
                ),
            )
            state_buffers[0].update_inplace(preprocess_output_values[0])
            constant_values = tuple(
                _empty_ortvalue(
                    argument, duration, self.device, self.device_id
                )
                for argument in self.transformer_inputs[1:-1]
            )
            for source, destination in zip(
                preprocess_output_values[1:5], constant_values, strict=True
            ):
                destination.update_inplace(source)

        bindings = (self.transformer.io_binding(), self.transformer.io_binding())
        constant_arguments = self.transformer_inputs[1:-1]
        for index, binding in enumerate(bindings):
            _bind_values(binding, constant_arguments, constant_values)
            binding.bind_ortvalue_input(
                self.transformer_inputs[0].name, state_buffers[index]
            )
            binding.bind_ortvalue_output(
                self.transformer_outputs[0].name, state_buffers[1 - index]
            )

        progress_interval = max(1, self.package.nfe_step // 5)
        for step in range(self.package.nfe_step):
            binding = bindings[step & 1]
            binding.bind_ortvalue_input(
                self.transformer_inputs[-1].name, self.step_buffers[step]
            )
            self._run(self.transformer, binding)
            if show_progress and ((step + 1) % progress_interval == 0 or step + 1 == self.package.nfe_step):
                print(f"[Raon-OpenTTS] flow step {step + 1}/{self.package.nfe_step}", flush=True)
        final_state = state_buffers[self.package.nfe_step & 1]

        decode_binding = self.decode.io_binding()
        decode_binding.bind_ortvalue_input(self.decode_inputs[0].name, final_state)
        decode_binding.bind_ortvalue_input(
            self.decode_inputs[1].name, preprocess_output_values[5]
        )
        decode_binding.bind_ortvalue_input(
            self.decode_inputs[2].name, preprocess_output_values[6]
        )
        decode_binding.bind_output(
            self.decode_outputs[0].name, "cpu", self.device_id
        )
        self._run(self.decode, decode_binding)

        decoded_values = decode_binding.get_outputs()
        if len(decoded_values) != 1:
            raise RuntimeError(
                f"Decoder returned {len(decoded_values)} outputs; expected one from the model interface"
            )
        waveform = decoded_values[0].numpy()
        if not _shape_matches(self.decode_outputs[0], waveform.shape):
            raise RuntimeError(
                f"Decoder returned shape {waveform.shape}, which does not match "
                f"model output {self.decode_outputs[0].shape}"
            )
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise RuntimeError("Decoder returned an empty waveform or NaN/Inf")
        return np.asarray(waveform.reshape(-1), dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Raon-OpenTTS using ONNX Runtime only.")
    parser.add_argument(
        "--onnx-folder",
        type=Path,
        default=SCRIPT_DIR / "Raon_Optimized",
        help="Folder containing the four Raon ONNX graphs.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        help="Vocabulary path; defaults to ~/Downloads/Raon-OpenTTS-<model_name>/vocab.txt.",
    )
    parser.add_argument("--reference-audio", type=Path, default=REFERENCE_AUDIO_PATH)
    parser.add_argument("--reference-text", default=REFERENCE_TEXT)
    parser.add_argument("--target-text", default=TARGET_TEXT)
    parser.add_argument("--output-audio", type=Path, default=OUTPUT_AUDIO_PATH)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=9527)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fix-duration", type=float)
    parser.add_argument("--cross-fade-duration", type=float, default=0.15)
    parser.add_argument("--no-vad-duration", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ort-verbose", action="store_true")
    return parser


def run_inference(args: argparse.Namespace) -> Path:
    if not args.reference_text.strip():
        raise ValueError("Reference text must not be empty")
    if not args.target_text.strip():
        raise ValueError("Target text must not be empty")
    if args.speed <= 0 or not math.isfinite(args.speed):
        raise ValueError("--speed must be a finite value greater than zero")
    if args.fix_duration is not None and (
        args.fix_duration <= 0 or not math.isfinite(args.fix_duration)
    ):
        raise ValueError("--fix-duration must be a finite value greater than zero")
    if args.cross_fade_duration < 0 or not math.isfinite(args.cross_fade_duration):
        raise ValueError("--cross-fade-duration must be finite and non-negative")

    package, vocab_map = load_package_metadata(args.onnx_folder, args.vocab_path)
    audio, original_audio = load_reference_audio(
        args.reference_audio, package.sample_rate, package.n_fft
    )
    use_vad_duration = not args.no_vad_duration
    reference_seconds = (
        estimate_reference_seconds(original_audio)
        if use_vad_duration
        else audio.size / package.sample_rate
    )
    if reference_seconds <= 0 or not math.isfinite(reference_seconds):
        raise ValueError("Reference duration estimate is empty or non-finite")

    reference_text_for_chunking = fix_reference_text_ending(args.reference_text.lower())
    if not reference_text_for_chunking:
        raise ValueError("Reference text must not be empty after normalization")
    max_chars = int(
        len(reference_text_for_chunking.encode("utf-8"))
        / max(reference_seconds, 1e-6)
        * (22.0 - reference_seconds)
    )
    target_chunks = chunk_text(args.target_text.lower(), max_chars=max_chars)
    if not target_chunks or any(not chunk for chunk in target_chunks):
        raise ValueError("Target text produced no non-empty synthesis chunks")

    reference_text = reference_text_for_chunking
    if len(reference_text[-1].encode("utf-8")) == 1:
        reference_text += " "
    plans = [
        plan_chunk(
            audio_length=audio.size,
            reference_seconds_for_length=reference_seconds,
            reference_text=reference_text,
            target_text=chunk,
            vocab_map=vocab_map,
            sample_rate=package.sample_rate,
            hop_length=package.hop_length,
            n_fft=package.n_fft,
            speed=args.speed,
            fix_duration=args.fix_duration,
            use_vad_duration=use_vad_duration,
            max_signal_length=package.max_signal_length,
        )
        for chunk in target_chunks
    ]

    runtime = RaonONNXRuntime(
        package,
        provider=args.provider,
        device_id=args.device_id,
        threads=args.threads,
        seed=args.seed,
        verbose=args.ort_verbose,
    )
    started = time.perf_counter()
    waveforms: list[np.ndarray] = []
    for index, (chunk, plan) in enumerate(zip(target_chunks, plans), start=1):
        if not args.quiet:
            print(
                f"[Raon-OpenTTS] chunk {index}/{len(plans)}: "
                f"text_bytes={len(chunk.encode('utf-8'))} duration={plan.max_duration} "
                f"generated_frames={plan.generated_frames}",
                flush=True,
            )
        waveforms.append(runtime.synthesize_chunk(audio, plan, show_progress=not args.quiet))
    waveform = crossfade_waveforms(
        waveforms, package.sample_rate, args.cross_fade_duration
    )
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("Final waveform is empty or contains NaN/Inf")

    output_path = args.output_audio.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_path,
        waveform,
        package.sample_rate,
        subtype="FLOAT",
        format="WAVEX",
    )
    if not args.quiet:
        elapsed = time.perf_counter() - started
        audio_seconds = waveform.size / package.sample_rate
        print(
            f"[Raon-OpenTTS] wrote {output_path} ({audio_seconds:.3f}s) in {elapsed:.3f}s, "
            f"RTF={elapsed / max(audio_seconds, 1e-9):.3f}",
            flush=True,
        )
    return output_path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_inference(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())