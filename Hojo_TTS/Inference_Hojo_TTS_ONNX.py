"""Run Hojo-TTS-Light (40M preset voices and 80M voice cloning) ONNX packages."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from pydub import AudioSegment

from Shared_Weights import SharedInitializerStore
from Streaming_Hojo import StatefulVocoder


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_reference  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
LM_NAME = "Hojo_LM.onnx"
VOCODER_NAME = "Hojo_Vocoder.onnx"
REF_PREPROCESS_NAME = "Hojo_ReferencePreprocess.onnx"
METADATA_NAME = "Hojo_Metadata.onnx"
TOKENIZER_NAME = "tokenizer.json"
_U24 = 1.0 / (1 << 24)

DEFAULT_MODEL_DIR = SCRIPT_DIR / "HojoTTS_Optimized"
FALLBACK_MODEL_DIR = SCRIPT_DIR / "HojoTTS_ONNX"


def resolve_model_dir(folder: str | Path | None = None) -> Path:
    """Return model directory: prefer HojoTTS_Optimized with auto-fallback to HojoTTS_ONNX."""
    if folder is not None:
        target = Path(folder).expanduser()
        if not target.is_absolute() and not (target / METADATA_NAME).is_file():
            if (SCRIPT_DIR / target / METADATA_NAME).is_file() or (SCRIPT_DIR / target).is_dir():
                target = SCRIPT_DIR / target
        target = target.resolve()
        if (target / METADATA_NAME).is_file():
            return target
        if target.name != DEFAULT_MODEL_DIR.name:
            return target
    if (DEFAULT_MODEL_DIR / METADATA_NAME).is_file():
        return DEFAULT_MODEL_DIR
    if (FALLBACK_MODEL_DIR / METADATA_NAME).is_file():
        return FALLBACK_MODEL_DIR
    return DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.is_dir() else FALLBACK_MODEL_DIR


# ============================== Configuration ==============================
# Edit these values directly; CLI flags override these defaults.
ONNX_FOLDER = resolve_model_dir()    # Default model path (HojoTTS_Optimized with auto-fallback to HojoTTS_ONNX).
GENERATED_AUDIO_PATH = str(SCRIPT_DIR / "generated.wav")

TARGET_TTS = [
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology.",
]

# Usable Preset Voice IDs (accepts 0-based int32 index or exact string name):
# ---------------------------------------------------------------------------------
# • Hojo-TTS-Light-40M V1 (15 preset voices):
#     0: "hojo_en_f_01"   1: "hojo_en_f_02"   2: "hojo_en_f_03"   3: "hojo_en_f_04"
#     4: "hojo_en_f_05"   5: "hojo_en_f_06"   6: "hojo_en_f_07"
#     7: "hojo_en_m_01"   8: "hojo_en_m_02"   9: "hojo_en_m_03"  10: "hojo_en_m_04"
#    11: "hojo_en_m_05"
#    12: "hojo_en_u_01"
#    13: "hojo_zh_f_01"  14: "hojo_zh_f_02"
#
# • Hojo-TTS-Light-40M V2 (17 preset voices):
#     0: "hojo_en_f_01"   1: "hojo_en_f_02"   2: "hojo_en_f_03"   3: "hojo_en_f_04"
#     4: "hojo_en_f_05"   5: "hojo_en_f_06"   6: "hojo_en_f_07"   7: "hojo_en_f_08"
#     8: "hojo_en_m_01"   9: "hojo_en_m_02"  10: "hojo_en_m_03"  11: "hojo_en_m_04"
#    12: "hojo_en_m_05"
#    13: "hojo_zh_f_01"  14: "hojo_zh_f_02"  15: "hojo_zh_m_01"  16: "hojo_zh_m_02"
#
# • Hojo-TTS-Light 80M (Voice Cloning / Zero-Shot):
#     None (no preset voices). Uses reference prompt audio + text (PROMPT_AUDIO_PATH, PROMPT_TEXT)
#     or CLI --prompt-audio and --prompt-text via set_reference().
# ---------------------------------------------------------------------------------
VOICE_ID = np.int32(13)              # Preset voice: 0-based int32 row of the package `voices` metadata, or a voice name.
PROMPT_AUDIO_PATH, PROMPT_TEXT = model_reference("hojo_tts")  # Used by 80M voice-cloning packages.

TEMPERATURE = 0.8                    # Sampling temperature (0.0 = greedy decoding).
TOP_P = 0.95                         # Nucleus sampling cumulative probability.
REPETITION_PENALTY = 1.05            # Sign-aware repetition penalty.
RANDOM_SEED = 9527                   # Deterministic 24-bit MT19937 seed.
MAX_NEW_TOKENS = 1024                # Maximum generated audio tokens per sentence chunk (~20.5 s at 50 Hz).
SENTENCE_MAX_CHARS = 80              # Maximum characters per streaming sentence chunk.
TEXT_NORMALIZE = True                # Normalize Chinese digits, percentages, times, years, and ranges.

USE_AUDIO_NORMALIZER = False         # Normalize prompt and generated audio loudness.
ORT_LOG = False                      # Enable ONNX Runtime logging.
ORT_FP16 = False                     # Preserve FP16 runtime paths where supported.
ORT_ACCELERATE_PROVIDERS = []        # Optional providers, e.g. ['CUDAExecutionProvider'].
MAX_THREADS = 0                      # CPU parallel threads; 0 lets ORT choose.
DEVICE_ID = 0                        # Accelerator device index.
SHOW_PROGRESS = True                 # Print pipeline stages and synthesis progress.
# ===========================================================================


def print_progress(message: str) -> None:
    if SHOW_PROGRESS:
        print(f"[Hojo-TTS] {message}", flush=True)


_DIGITS = "零一二三四五六七八九"
_UNITS = ("", "十", "百", "千")
_GROUPS = ("", "万", "亿", "万亿")
_FULLWIDTH = str.maketrans({chr(0xFF10 + i): str(i) for i in range(10)} | {"％": "%", "．": ".", "：": ":"})
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PINYIN_TOKEN_RE = re.compile(r"<zh_[^<>]+>")
_RE_TIME = re.compile(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?(?!\d)")
_RE_YEAR = re.compile(r"(?<!\d)(\d{2,4})(?=年)")
_RE_RANGE = re.compile(r"(?<=\d)[-~～](?=\d)")
_RE_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)%")
_RE_NUMBER = re.compile(r"(?<![\d.])(-?)(\d+)(?:\.(\d+))?(?![\d])")
_SENTENCE_END = "。！？!?；;\n"


def _read_digits(digits: str) -> str:
    return "".join(_DIGITS[int(c)] for c in digits)


def _read_group(n: int) -> str:
    out = []
    zero = False
    for pos in range(3, -1, -1):
        d = n // 10 ** pos % 10
        if d == 0:
            zero = bool(out)
            continue
        if zero:
            out.append("零")
            zero = False
        out.append(_DIGITS[d] + _UNITS[pos])
    return "".join(out)


def read_integer(digits: str) -> str:
    if (len(digits) > 1 and digits[0] == "0") or len(digits) > 16:
        return _read_digits(digits)
    n = int(digits)
    if n == 0:
        return "零"
    parts = []
    group = 0
    need_zero = False
    while n:
        n, chunk = divmod(n, 10000)
        if chunk:
            text = _read_group(chunk) + _GROUPS[group]
            if need_zero:
                text += "零"
            parts.append(text)
            need_zero = chunk < 1000 and n > 0
        else:
            need_zero = bool(parts)
        group += 1
    text = "".join(reversed(parts)).rstrip("零")
    return text[1:] if text.startswith("一十") else text


def _read_number(match: re.Match[str]) -> str:
    sign, integer, frac = match.group(1), match.group(2), match.group(3)
    text = _read_digits(integer) if frac is None and len(integer) >= 11 else read_integer(integer)
    if frac:
        text += "点" + _read_digits(frac)
    return ("负" if sign else "") + text


def _normalize_zh_span(text: str) -> str:
    text = text.translate(_FULLWIDTH)
    text = _RE_TIME.sub(
        lambda m: read_integer(m.group(1))
        + "点"
        + (read_integer(m.group(2)) + "分" if m.group(2) != "00" else "")
        + (read_integer(m.group(3)) + "秒" if m.group(3) else ""),
        text,
    )
    text = _RE_YEAR.sub(lambda m: _read_digits(m.group(1)), text)
    text = _RE_RANGE.sub("到", text)
    text = _RE_PERCENT.sub(lambda m: "百分之" + _read_number(_RE_NUMBER.match(m.group(1))), text)
    return _RE_NUMBER.sub(_read_number, text)


def normalize_zh(text: str) -> str:
    """Spoken-form rewrite for Chinese spans while leaving <zh_*> pinyin override tokens untouched."""
    if not _CJK_RE.search(text):
        return text
    parts = _PINYIN_TOKEN_RE.split(text)
    tokens = _PINYIN_TOKEN_RE.findall(text)
    if not tokens:
        return _normalize_zh_span(text)
    pieces: list[str] = []
    for index, part in enumerate(parts):
        if part:
            pieces.append(_normalize_zh_span(part))
        if index < len(tokens):
            pieces.append(tokens[index])
    return "".join(pieces)


normalize_text = normalize_zh


def split_sentences(text: str, max_chars: int) -> list[str]:
    if max_chars < 1:
        raise ValueError(f"split_sentences: max_chars must be >= 1, got {max_chars}")
    out: list[str] = []
    current = []
    for ch in text:
        current.append(ch)
        if ch in _SENTENCE_END:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    chunks: list[str] = []
    for sentence in out:
        while len(sentence) > max_chars:
            cut = max((sentence.rfind(c, 0, max_chars) for c in "，,、：:"), default=-1)
            cut = cut + 1 if cut > 0 else max_chars
            chunks.append(sentence[:cut])
            sentence = sentence[cut:]
        chunks.append(sentence)
    return [c.strip() for c in chunks if c.strip()]


class BpeTokenizer:
    """Zero-dependency BPE encoder matching Hojo's tokenizer.json or packaged hojo_bpe.txt/hojo_added.txt tables."""

    _SPLIT = re.compile(r"\s+|\w+|[^\w\s]+")
    _DIGIT_SPLIT = re.compile(r"(\d)")
    _WS_RE = re.compile(r"\s+")

    @staticmethod
    def _build_added_matcher(
        tokens: list[str], added_map: dict[str, int],
    ) -> tuple[re.Pattern[str] | None, dict[str, tuple[int, ...]], dict[str, int]]:
        if not tokens:
            return None, {}, {}
        subset: dict[str, int] = {}
        lengths: dict[str, set[int]] = {}
        for tok in tokens:
            if not tok:
                continue
            subset[tok] = added_map[tok]
            lengths.setdefault(tok[0], set()).add(len(tok))
        if not subset:
            return None, {}, {}
        trigger = re.compile("[" + "".join(re.escape(ch) for ch in sorted(lengths)) + "]")
        lengths_desc = {ch: tuple(sorted(lens, reverse=True)) for ch, lens in lengths.items()}
        return trigger, lengths_desc, subset

    def __init__(self, path: Path) -> None:
        path = Path(path)
        if path.is_file() and path.suffix == ".json":
            spec = json.loads(path.read_text(encoding="utf-8"))
            model = spec["model"]
            if model.get("type") != "BPE" or model.get("byte_fallback") or model.get("continuing_subword_prefix") \
                    or model.get("end_of_word_suffix"):
                raise ValueError(f"{path}: unsupported tokenizer model {model.get('type')!r}")
            self.vocab: dict[str, int] = dict(model["vocab"])
            self.added: dict[str, int] = {a["content"]: int(a["id"]) for a in spec["added_tokens"]}
            unk_tok = model.get("unk_token") or "[UNK]"
            self.unk_id = self.vocab.get(unk_tok, self.added.get(unk_tok, 0))
            self.ranks = {tuple(m if isinstance(m, list) else m.split(" ")): i for i, m in enumerate(model["merges"])}
            normalized = [a["content"] for a in spec["added_tokens"] if a.get("normalized")]
            raw = [a["content"] for a in spec["added_tokens"] if not a.get("normalized")]
        else:
            bpe_path = path if path.is_file() and path.name == "hojo_bpe.txt" else path.parent / "hojo_bpe.txt"
            added_path = bpe_path.parent / "hojo_added.txt"
            if not bpe_path.is_file():
                raise FileNotFoundError(f"Neither {path} nor packaged {bpe_path} found")
            lines = bpe_path.read_text(encoding="utf-8").splitlines()
            head = lines[0].split()
            nv, nm, self.unk_id = int(head[1]), int(head[3]), int(head[5])
            self.vocab = {}
            for idx in range(nv):
                tid_s, hex_s = lines[1 + idx].split(" ")
                self.vocab[bytes.fromhex(hex_s).decode("utf-8")] = int(tid_s)
            self.ranks = {}
            for idx in range(nm):
                l_hex, r_hex = lines[1 + nv + idx].split(" ")
                self.ranks.setdefault((bytes.fromhex(l_hex).decode("utf-8"), bytes.fromhex(r_hex).decode("utf-8")), idx)
            self.added = {}
            normalized = []
            raw = []
            if added_path.is_file():
                for line in added_path.read_text(encoding="utf-8").splitlines():
                    if not line:
                        continue
                    tid_s, norm_s, hex_s = line.split(" ")
                    content = bytes.fromhex(hex_s).decode("utf-8")
                    self.added[content] = int(tid_s)
                    (normalized if int(norm_s) != 0 else raw).append(content)
        self._added_trigger, self._added_lens, self._added_subset = self._build_added_matcher(normalized, self.added)
        self._raw_trigger, self._raw_lens, self._raw_subset = self._build_added_matcher(raw, self.added)
        self._bpe = lru_cache(maxsize=65536)(self._bpe_word)

    def token_id(self, token: str) -> int:
        if token in self.added:
            return self.added[token]
        if token in self.vocab:
            return self.vocab[token]
        raise KeyError(f"tokenizer: missing token {token!r}")

    def _bpe_word(self, word: str) -> tuple[int, ...]:
        parts = list(word)
        while len(parts) > 1:
            best = min(range(len(parts) - 1), key=lambda i: self.ranks.get((parts[i], parts[i + 1]), 1 << 30))
            if (parts[best], parts[best + 1]) not in self.ranks:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return tuple(self.vocab.get(p, self.unk_id) for p in parts)

    def _encode_plain(self, text: str, out: list[int]) -> None:
        if not text:
            return
        for piece in self._SPLIT.findall(text):
            if any("0" <= ch <= "9" for ch in piece):
                for sub in self._DIGIT_SPLIT.split(piece):
                    if sub:
                        out.extend(self._bpe(sub))
            else:
                out.extend(self._bpe(piece))

    @staticmethod
    def _scan_added(
        text: str,
        trigger: re.Pattern[str] | None,
        lens_by_first: dict[str, tuple[int, ...]],
        subset: dict[str, int],
        on_plain,
        out: list[int],
    ) -> None:
        if trigger is None or not text:
            on_plain(text, out)
            return
        pos = 0
        scan = 0
        n = len(text)
        while scan < n:
            m = trigger.search(text, scan)
            if m is None:
                break
            idx = m.start()
            matched_id = None
            matched_len = 0
            for length in lens_by_first[text[idx]]:
                if idx + length <= n:
                    tid = subset.get(text[idx:idx + length])
                    if tid is not None:
                        matched_id = tid
                        matched_len = length
                        break
            if matched_id is not None:
                if idx > pos:
                    on_plain(text[pos:idx], out)
                out.append(matched_id)
                pos = scan = idx + matched_len
            else:
                scan = idx + 1
        if pos < n:
            on_plain(text[pos:], out)

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        self._scan_added(text, self._raw_trigger, self._raw_lens, self._raw_subset, self._encode_normalized, out)
        return out

    def _encode_normalized(self, text: str, out: list[int]) -> None:
        if not text:
            return
        text = self._WS_RE.sub(" ", text.lower())
        self._scan_added(text, self._added_trigger, self._added_lens, self._added_subset, self._encode_plain, out)


@dataclass(frozen=True)
class SynthesisStats:
    text: str
    tokens: int
    audio_tokens: int
    samples: int
    sample_rate: int
    prefill_s: float
    decode_s: float
    vocoder_s: float

    @property
    def audio_s(self) -> float:
        return self.samples / self.sample_rate

    @property
    def total_s(self) -> float:
        return self.prefill_s + self.decode_s + self.vocoder_s

    @property
    def rtf(self) -> float:
        return self.total_s / self.audio_s if self.samples else float("inf")


def io_dtype(argument) -> np.dtype:
    match = re.fullmatch(r"tensor\(([^)]+)\)", argument.type)
    element_type = onnx.TensorProto.DataType.Value(match.group(1).upper())
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))


def audio_normalizer(audio: np.ndarray, output_dtype, target_value: float | None = None) -> np.ndarray:
    output_dtype = np.dtype(output_dtype)
    audio = audio.astype(np.float32)
    if target_value is None:
        target_value = 8192.0 if np.issubdtype(output_dtype, np.integer) else 0.25
    rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
    audio *= target_value / (rms + 1.0e-7)
    if np.issubdtype(output_dtype, np.integer):
        limits = np.iinfo(output_dtype)
        np.clip(audio, limits.min, limits.max, out=audio)
    return audio.astype(output_dtype)


def load_prompt_audio(path: str | Path, sample_rate: int, argument) -> np.ndarray:
    samples = np.asarray(
        AudioSegment.from_file(str(path))
        .set_channels(1)
        .set_frame_rate(sample_rate)
        .set_sample_width(2)
        .get_array_of_samples(),
        dtype=np.int16,
    )
    target_dtype = io_dtype(argument)
    if np.issubdtype(target_dtype, np.floating):
        samples = (samples.astype(np.float32) * (1.0 / 32768.0)).astype(target_dtype)
    return np.ascontiguousarray(samples.reshape(1, 1, -1))


def build_session_options(preserve_fp16: bool, max_threads: int = MAX_THREADS, ort_log: bool = ORT_LOG):
    session_options = ort.SessionOptions()
    run_options = ort.RunOptions()
    for options in (session_options, run_options):
        options.log_severity_level = 0 if ort_log else 4
        options.log_verbosity_level = 4
    session_options.inter_op_num_threads = max_threads
    session_options.intra_op_num_threads = max_threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_gelu_approximation": "1",
        "optimization.minimal_build_optimizations": "",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": (
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if preserve_fp16 else ""
        ),
    }.items():
        session_options.add_session_config_entry(key, value)
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    disabled_optimizers = (
        ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if preserve_fp16 else None
    )
    return session_options, run_options, disabled_optimizers


def configure_provider(accelerate_providers: list[str] = ORT_ACCELERATE_PROVIDERS, device_id: int = DEVICE_ID):
    if "OpenVINOExecutionProvider" in accelerate_providers:
        return [{
            "device_type": "CPU",
            "precision": "ACCURACY",
            "num_of_threads": MAX_THREADS if MAX_THREADS else 8,
            "num_streams": 1,
            "enable_opencl_throttling": False,
            "enable_qdq_optimizer": False,
            "disable_dynamic_shapes": False,
        }], "cpu"
    if "CUDAExecutionProvider" in accelerate_providers:
        return [{
            "device_id": device_id,
            "gpu_mem_limit": 24 * 1024**3,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "use_tf32": "1",
            "do_copy_in_default_stream": "0",
            "enable_cuda_graph": "0",
        }], "cuda"
    if "DmlExecutionProvider" in accelerate_providers:
        return [{
            "device_id": device_id,
            "performance_preference": "high_performance",
            "device_filter": "gpu",
            "disable_metacommands": "false",
            "enable_graph_capture": "false",
        }], "dml"
    return None, "cpu"


_META_BASE_INT = (
    "sample_rate", "in_sample_rate", "out_sample_rate", "model_sample_rate", "n_fft",
    "vocab_size", "num_layers", "num_heads", "kv_heads", "head_dim", "hidden_size", "max_positions",
    "text_start_id", "text_end_id", "speech_start_id", "speech_end_id",
    "audio_token_first", "audio_token_last", "hop_length",
)
_META_PRESET_INT = ("spk_start_id", "spk_end_id", "spk_emb0_id", "spk_slots")
_META_CLONE_INT = ("ref_text_start_id", "ref_text_end_id", "ref_speech_start_id", "ref_speech_end_id", "space_id")


def parse_metadata(raw: dict[str, str]) -> dict:
    try:
        mode = raw["tts_mode"]
        if mode not in {"preset_voices", "voice_clone"} or raw["graph_layout"] != "hojo_lm_vocoder_v1":
            raise ValueError("unsupported tts_mode or graph_layout")
        keys = _META_BASE_INT + (_META_PRESET_INT if mode == "preset_voices" else _META_CLONE_INT)
        meta = dict(raw)
        meta.update({key: int(raw[key]) for key in keys})
        meta["rope_theta"] = float(raw["rope_theta"])
        filenames = ["model_file_name_lm", "tokenizer_file"]
        if "model_file_name_vocoder" in raw:
            filenames.append("model_file_name_vocoder")
        if mode == "voice_clone":
            filenames.append("model_file_name_reference_preprocess")
        shared = ("shared_initializer_model_file", "shared_initializer_data_file")
        if any(key in raw for key in shared):
            filenames.extend(shared)
        if "model_file_name_lm_decode" in raw:
            filenames.append("model_file_name_lm_decode")
        streaming = ("model_file_name_fine_stream", "model_file_name_decoder_stream")
        has_streaming = any(key in raw for key in (*streaming, "native_fft_library_file"))
        if has_streaming:
            filenames.extend(streaming)
            if raw.get("advanced_pipeline_version") != "2" or "native_fft_library_file" in raw:
                raise ValueError("advanced packages must use the ORT-only pipeline v2 (re-optimize with --advanced)")
            if mode == "voice_clone" and not raw.get("reference_speaker_projection_output"):
                raise ValueError("stateful cloning requires a projected speaker output")
        if "model_file_name_vocoder" not in raw and not has_streaming:
            raise ValueError("package must specify model_file_name_vocoder or streaming vocoder graphs")
        if "lm_kv_update" in raw and raw["lm_kv_update"] not in {"concat", "inplaced_kv"}:
            raise ValueError(f"unsupported lm_kv_update={raw['lm_kv_update']!r}")
        for key in ("fine_history_frames", "stream_decoder_delay_frames"):
            if key in raw:
                meta[key] = int(raw[key])
                if meta[key] <= 0:
                    raise ValueError(f"invalid {key}")
        for key in filenames:
            value = raw[key]
            if not value or Path(value).name != value:
                raise ValueError(f"{key} must name a file in the package")
        for key in ("in_audio_dtype", "out_audio_dtype"):
            if raw[key] not in {"F16", "F32", "INT16"}:
                raise ValueError(f"unsupported {key}")
        voices = tuple(filter(None, raw["voices"].split(",")))
        if len(set(voices)) != len(voices) or bool(voices) != (mode == "preset_voices"):
            raise ValueError("voices must be unique and nonempty for preset_voices only")
        meta["voices"] = voices
        positive = ("sample_rate", "in_sample_rate", "out_sample_rate", "model_sample_rate", "hop_length", "n_fft",
                    "vocab_size", "num_layers", "num_heads", "kv_heads", "head_dim", "hidden_size", "max_positions")
        if any(meta[key] <= 0 for key in positive) or meta["sample_rate"] != meta["out_sample_rate"]:
            raise ValueError("invalid rates or model dimensions")
        if meta["num_heads"] % meta["kv_heads"] or not 0 <= meta["audio_token_first"] <= meta["audio_token_last"] < meta["vocab_size"]:
            raise ValueError("invalid attention or audio-token range")
        token_keys = [key for key in keys if key.endswith("_id")]
        if any(not 0 <= meta[key] < meta["vocab_size"] for key in token_keys):
            raise ValueError("token id is outside the vocabulary")
        if mode == "preset_voices" and (meta["spk_slots"] <= 0 or meta["spk_emb0_id"] + meta["spk_slots"] > meta["vocab_size"]):
            raise ValueError("invalid speaker placeholder range")
        context_keys = ("vocoder_left_context_frames", "vocoder_right_context_frames")
        if any(key in raw for key in context_keys):
            for key in context_keys:
                meta[key] = int(raw[key])
                if not 0 <= meta[key] <= meta["max_positions"]:
                    raise ValueError(f"invalid {key}")
        profile_keys = ("stream_first_chunk_frames", "stream_chunk_frames")
        if any(key in raw for key in profile_keys):
            for key in profile_keys:
                meta[key] = int(raw[key])
                if not 0 < meta[key] <= meta["max_positions"]:
                    raise ValueError(f"invalid {key}")
        return meta
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid Hojo package metadata: {exc}; re-export the package") from exc


def prompt_parts(meta: dict, tokenizer: BpeTokenizer, reference_text: str | None = None,
                 reference_audio_ids: np.ndarray | None = None) -> tuple[list[int], list[int]]:
    if meta["tts_mode"] == "preset_voices":
        return [meta["text_start_id"]], [
            meta["text_end_id"], meta["spk_start_id"],
            *range(meta["spk_emb0_id"], meta["spk_emb0_id"] + meta["spk_slots"]),
            meta["spk_end_id"], meta["speech_start_id"],
        ]
    if reference_text is None or reference_audio_ids is None:
        raise RuntimeError("Hojo voice cloning requires reference audio and text")
    return [meta["ref_text_start_id"], *tokenizer.encode(reference_text), meta["ref_text_end_id"],
            meta["space_id"], meta["text_start_id"]], [
        meta["text_end_id"], meta["ref_speech_start_id"], *reference_audio_ids.ravel().tolist(),
        meta["ref_speech_end_id"], meta["speech_start_id"],
    ]


def build_prompt(tokenizer: BpeTokenizer, text: str, prefix: list[int], suffix: list[int]) -> np.ndarray:
    return np.asarray([*prefix, *tokenizer.encode(text), *suffix], np.int32)[None]


def _read_folder_metadata(folder: Path) -> dict[str, str]:
    meta_path = folder / METADATA_NAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Hojo metadata carrier not found in {folder}; re-export the package")
    model = onnx.load(str(meta_path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def voice_lookup(voices: tuple[str, ...] | list[str]) -> tuple[dict[str, int], str]:
    """Return the voice name -> int32 row map and the indexed listing shown by voice errors."""
    lookup = {name: index for index, name in enumerate(voices)}
    if len(lookup) != len(voices):
        raise ValueError(f"Hojo voice names must be unique: {list(voices)}")
    return lookup, ", ".join(f"{index}: {name}" for index, name in enumerate(voices))


def resolve_voice_index(target: int | np.integer | np.ndarray | str, lookup: dict[str, int], listing: str) -> int:
    """Validate a 0-based int32 voice row (int, NumPy integer, one-element integer array, digit string) or name."""
    if isinstance(target, np.ndarray) and target.size == 1 and np.issubdtype(target.dtype, np.integer):
        target = target.item()
    elif isinstance(target, str):
        if target in lookup:
            return lookup[target]
        if re.fullmatch(r"\s*[+-]?\d+\s*", target):
            target = int(target)
    if isinstance(target, (int, np.integer)) and not isinstance(target, bool):
        if 0 <= target < len(lookup):
            return int(target)
        raise ValueError(f"Hojo voice index {int(target)} is out of range [0, {len(lookup) - 1}]; voices: {listing}")
    raise ValueError(f"Unknown Hojo voice {target!r}; pass an int32 index in [0, {len(lookup) - 1}] or one of: {listing}")


class HojoTts:
    """Thread-safe Hojo-TTS runtime supporting 40M preset voices and 80M voice cloning."""

    def __init__(
        self,
        folder: str | Path | None = None,
        *,
        voice_id: int | np.integer | np.ndarray | str | None = None,
        voice: int | np.integer | np.ndarray | str | None = None,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        repetition_penalty: float = REPETITION_PENALTY,
        seed: int = RANDOM_SEED,
        max_new_tokens: int = MAX_NEW_TOKENS,
        sentence_max_chars: int = SENTENCE_MAX_CHARS,
        text_normalize: bool = TEXT_NORMALIZE,
        max_threads: int = MAX_THREADS,
        accelerate_providers: list[str] = ORT_ACCELERATE_PROVIDERS,
        device_id: int = DEVICE_ID,
        use_decode: bool = True,
        vocoder_backend: str = "auto",
        cache_reference_prefix: bool = True,
        custom_ops_library: str | Path | None = None,
    ) -> None:
        folder = resolve_model_dir(folder or ONNX_FOLDER)
        if not folder.is_dir():
            raise FileNotFoundError(f"Hojo TTS model folder not found: {folder}")
        self.folder = folder
        raw_meta = self.meta = parse_metadata(_read_folder_metadata(folder))
        self.tts_mode = raw_meta["tts_mode"]
        self.voices = raw_meta["voices"]
        self.in_sample_rate = raw_meta["in_sample_rate"]
        self.sample_rate = raw_meta["out_sample_rate"]
        tok_path = folder / raw_meta["tokenizer_file"]
        self.tokenizer = BpeTokenizer(tok_path if tok_path.is_file() else (folder / "hojo_bpe.txt"))

        # Keeping fp16 weights un-fused is only faster on CPUs with native fp16 GEMM; x86 is ~3.5x slower.
        _, run_options, disabled_optimizers = build_session_options(ORT_FP16, max_threads)
        provider_options, self.device_type = configure_provider(accelerate_providers, device_id)
        providers = accelerate_providers or ["CPUExecutionProvider"]
        if vocoder_backend not in {"auto", "onnx", "stateful"}:
            raise ValueError("vocoder_backend must be auto, onnx, or stateful")
        available_streaming = "model_file_name_fine_stream" in raw_meta
        has_whole_vocoder = "model_file_name_vocoder" in raw_meta and (folder / raw_meta["model_file_name_vocoder"]).is_file()
        cpu_only = providers == ["CPUExecutionProvider"]
        use_stateful = vocoder_backend == "stateful" or (
            vocoder_backend == "auto" and available_streaming and cpu_only
        )
        if not use_stateful and not has_whole_vocoder:
            raise ValueError("This streaming-only package omits Hojo_Vocoder.onnx; use vocoder_backend='stateful' on CPU")
        if use_stateful and (not available_streaming or not cpu_only):
            raise ValueError("Stateful vocoding requires an advanced package and CPUExecutionProvider")
        if use_stateful and ((raw_meta["n_fft"], raw_meta["hop_length"], self.sample_rate) != (1920, 480, 24000)
                             or raw_meta["out_audio_dtype"] != "F32"):
            raise ValueError("Stateful vocoding requires native 24 kHz float32 output")
        for key in ("model_file_name_lm_decode", "model_file_name_fine_stream",
                    "model_file_name_decoder_stream"):
            if key in raw_meta and not (folder / raw_meta[key]).is_file():
                raise FileNotFoundError(folder / raw_meta[key])

        if custom_ops_library is None and os.environ.get("HOJO_KV_OPS_LIBRARY"):
            custom_ops_library = os.environ["HOJO_KV_OPS_LIBRARY"]
        custom_ops_path = Path(custom_ops_library).expanduser().resolve() if custom_ops_library else None
        if custom_ops_path is not None and not custom_ops_path.is_file():
            raise FileNotFoundError(f"Custom ops library not found: {custom_ops_path}")
        needs_kv_custom_ops = raw_meta.get("lm_kv_update") == "inplaced_kv"
        if needs_kv_custom_ops and custom_ops_path is None:
            raise RuntimeError(
                f"{folder}: model uses com.custom.llm::KVCacheUpdate (lm_kv_update='inplaced_kv'); "
                "pass --custom-ops-library / set HOJO_KV_OPS_LIBRARY, or use a --kv-update concat package"
            )

        self._shared_store: SharedInitializerStore | None = None
        shared_file = raw_meta.get("shared_initializer_model_file")
        if shared_file:
            for key in ("shared_initializer_model_file", "shared_initializer_data_file"):
                if not (folder / raw_meta[key]).is_file():
                    raise FileNotFoundError(folder / raw_meta[key])
            if self.device_type == "cpu":
                print_progress("Attaching shared ONNX initializers...")
                self._shared_store = SharedInitializerStore(folder / shared_file, respect_prepack_exclusions=True)

        def _make_sess(name: str, *, with_custom_ops: bool = False) -> tuple[ort.InferenceSession, ort.RunOptions]:
            so, ro, _ = build_session_options(ORT_FP16, max_threads)
            if with_custom_ops and custom_ops_path is not None:
                so.register_custom_ops_library(str(custom_ops_path))
            model_path = folder / name
            if self._shared_store is not None:
                self._shared_store.attach_to(so, model_path=model_path)
            sess = ort.InferenceSession(
                str(model_path),
                sess_options=so,
                providers=providers,
                provider_options=provider_options,
                disabled_optimizers=disabled_optimizers,
            )
            return sess, ro

        self.lm, self._lm_run = _make_sess(raw_meta["model_file_name_lm"], with_custom_ops=needs_kv_custom_ops)
        self._decode = self._decode_run = None
        if use_decode and "model_file_name_lm_decode" in raw_meta:
            self._decode, self._decode_run = _make_sess(raw_meta["model_file_name_lm_decode"], with_custom_ops=needs_kv_custom_ops)
        self._fine = None
        if use_stateful:
            self._fine, _ = _make_sess(raw_meta["model_file_name_fine_stream"])
            self.vocoder, self._voc_run = _make_sess(raw_meta["model_file_name_decoder_stream"])
        else:
            self.vocoder, self._voc_run = _make_sess(raw_meta["model_file_name_vocoder"])
        self.ref_preprocess = None
        self._ref_run = None
        if self.tts_mode == "voice_clone":
            ref_name = raw_meta["model_file_name_reference_preprocess"]
            self.ref_preprocess, self._ref_run = _make_sess(ref_name)

        self._past_names = [f"past_key_values.{layer}.{kind}"
                            for layer in range(self.meta["num_layers"]) for kind in ("key", "value")]
        self._present_names = [f"present.{layer}.{kind}"
                               for layer in range(self.meta["num_layers"]) for kind in ("key", "value")]
        self._output_names = [output.name for output in self.lm.get_outputs()]
        input_names = {argument.name for argument in self.lm.get_inputs()}
        self._inplaced_kv = "history_len" in input_names
        if not set(self._past_names) <= input_names or not set(self._present_names) <= set(self._output_names):
            raise ValueError(f"{folder}: KV inputs/outputs do not match num_layers")
        self._kv_output_indices = [self._output_names.index(name) for name in self._present_names]
        if self._decode is not None:
            if ({value.name: value.type for value in self._decode.get_inputs()}
                    != {value.name: value.type for value in self.lm.get_inputs()}
                    or {value.name: value.type for value in self._decode.get_outputs()}
                    != {value.name: value.type for value in self.lm.get_outputs()}
                    or next(value.shape for value in self._decode.get_inputs() if value.name == "input_ids") != [1, 1]):
                raise ValueError("Single-token decode I/O must match the prefill graph")

        kv_inputs = {i.name: i for i in self.lm.get_inputs() if i.name in self._past_names}
        kv_outputs = {o.name: o for o in self.lm.get_outputs() if o.name in self._present_names}
        if any(arg.type != "tensor(float16)" for arg in (*kv_inputs.values(), *kv_outputs.values())):
            raise ValueError(f"{folder}: Hojo_LM KV cache I/O must be float16; re-export the package")
        kv0_in = kv_inputs[self._past_names[0]]
        self._kv_dtype = np.dtype(np.float16)
        self._k_transposed = (not self._inplaced_kv) and (kv0_in.shape[2] == self.meta["head_dim"])
        self._lm_hidden_dtype = io_dtype(next(o for o in self.lm.get_outputs() if o.name == "hidden"))
        conditioning = self._fine if self._fine is not None else self.vocoder
        self._voc_hidden_dtype = io_dtype(next(i for i in conditioning.get_inputs() if i.name == "hidden_states"))
        self._counter_rng = "rng_seed" in input_names
        if self._counter_rng == ("uniform" in input_names) or (
                self._counter_rng and {"rng_seed": "tensor(uint64)", "rng_step": "tensor(int64)"} != {
                    value.name: value.type for value in self.lm.get_inputs() if value.name.startswith("rng_")}):
            raise ValueError(f"{folder}: Hojo_LM needs either float `uniform` or uint64 rng_seed/int64 rng_step")
        for session in (self.lm, conditioning):
            arg = next((i for i in session.get_inputs() if i.name == "voice"), None)
            if (arg is not None) != bool(self.voices) or arg is not None and (arg.type != "tensor(int32)" or arg.shape):
                raise ValueError(f"{folder}: preset-voice graphs need a scalar int32 'voice' input; re-export the package")

        self._voice_lookup, self._voice_listing = voice_lookup(self.voices)
        # The CPU OrtValue wraps _voice_buf, so writing the buffer re-targets every later binding without a copy.
        self._voice_buf = np.zeros((), np.int32)
        self._voice_value = ort.OrtValue.ortvalue_from_numpy(self._voice_buf)
        self._voice_default = None
        if self.voices and voice_id is None and voice is None:
            voice_id = VOICE_ID
        self._voice_default = self._resolve_voice(voice_id, voice)
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self._max_new = int(max_new_tokens)
        if self._max_new <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.sentence_max_chars = int(sentence_max_chars)
        if (not np.isfinite([self.temperature, self.top_p, self.repetition_penalty]).all()
                or not 0 < self.top_p <= 1 or self.repetition_penalty <= 0 or self.sentence_max_chars <= 0):
            raise ValueError("Sampling controls must be finite, top_p in (0, 1], and repetition_penalty/chunk size positive")
        self.text_normalize = bool(text_normalize)

        m = self.meta
        self._ctrl = np.array([self.temperature, self.top_p, self.repetition_penalty], np.float32)
        # CPU IOBinding reads bound arrays in place, so every bound control buffer is owned by the instance.
        self._greedy_ctrl = np.array([0.0, 1.0, 1.0], np.float32)
        if self._counter_rng:
            self._rng_seed = np.zeros(1, np.uint64)
            self._rng_step = np.zeros(1, np.int64)
        else:
            self._uniform = np.zeros(1, np.float32)
            self._uniform_draws = np.empty(self._max_new, np.float32)
        self._seen = (np.zeros(m["vocab_size"], np.bool_), np.zeros(m["vocab_size"], np.bool_))
        self._seen_values = (
            ort.OrtValue.ortvalue_from_numpy(self._seen[0]),
            ort.OrtValue.ortvalue_from_numpy(self._seen[1]),
        )
        self._step_token_in = np.zeros((1, 1), np.int32)
        self._step_token_in_ov = ort.OrtValue.ortvalue_from_numpy(self._step_token_in)
        self._step_token_out = np.zeros((1, 1), np.int32)
        self._step_token_out_ov = ort.OrtValue.ortvalue_from_numpy(self._step_token_out)
        self._tokens = np.zeros((1, self._max_new), np.int32)
        # Audio frames are compacted as they are written: non-audio steps are overwritten by the next step.
        self._audio_hidden = np.zeros((1, self._max_new, m["hidden_size"]), self._lm_hidden_dtype)
        self._audio_tokens = np.zeros((1, self._max_new), np.int32)
        self._frames = 0
        self._vocoder_hidden_buffer = (
            np.empty((1, self._max_new, m["hidden_size"]), self._voc_hidden_dtype)
            if self._lm_hidden_dtype != self._voc_hidden_dtype else None
        )
        if self._inplaced_kv:
            static_shape = (1, m["kv_heads"], m["max_positions"], m["head_dim"])
            self._history_len = np.zeros(1, np.int64)
            self._static_kv_bufs = {name: np.zeros(static_shape, self._kv_dtype) for name in self._past_names}
            self._static_kv = {name: ort.OrtValue.ortvalue_from_numpy(buf) for name, buf in self._static_kv_bufs.items()}
            self._empty_kv = {}
        else:
            k_shape = (1, m["kv_heads"], m["head_dim"], 0) if self._k_transposed else (1, m["kv_heads"], 0, m["head_dim"])
            v_shape = (1, m["kv_heads"], 0, m["head_dim"])
            self._history_len = None
            self._static_kv_bufs = {}
            self._static_kv = {}
            self._empty_kv = {
                ".key": ort.OrtValue.ortvalue_from_numpy(np.zeros(k_shape, self._kv_dtype)),
                ".value": ort.OrtValue.ortvalue_from_numpy(np.zeros(v_shape, self._kv_dtype)),
            }
        self._prefix: list[int] = []
        self._suffix: list[int] = []
        if self.tts_mode == "preset_voices":
            self._prefix, self._suffix = prompt_parts(m, self.tokenizer)
        self._ref_audio_ids: np.ndarray | None = None
        self._ref_speaker_emb: ort.OrtValue | None = None
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._stateful_vocoder = None
        if self._fine is not None:
            self._stateful_vocoder = StatefulVocoder(
                self._fine, self.vocoder, self._voc_run,
                self._voice_value if self.voices else None,
                decoder_delay_frames=raw_meta.get("stream_decoder_delay_frames", 8),
            )
        self._prefix_cache_enabled = (cache_reference_prefix and cpu_only and not self._inplaced_kv
                                      and self.tts_mode == "voice_clone"
                                      and raw_meta.get("reference_prefix_cache") == "1")
        self._reference_prefix_kv: dict[str, ort.OrtValue] | None = None
        self.reference_prefix_cache_hits = self.reference_prefix_cache_misses = 0
        self._prefix_run = None
        if self._prefix_cache_enabled:
            self._prefix_run = ort.RunOptions()
            self._prefix_run.log_severity_level = run_options.log_severity_level
            self._prefix_run.only_execute_path_to_fetches = True
        self._lm_binding = self._make_lm_binding(self.lm, is_decode=False)
        self._decode_binding = self._make_lm_binding(self._decode, is_decode=True) if self._decode is not None else None

    def _make_lm_binding(self, session: ort.InferenceSession, *, is_decode: bool) -> ort.IOBinding:
        binding = session.io_binding()
        if self.voices:
            binding.bind_ortvalue_input("voice", self._voice_value)
        binding.bind_cpu_input("ctrl", self._ctrl)
        self._bind_random(binding)
        if is_decode:
            binding.bind_ortvalue_input("input_ids", self._step_token_in_ov)
        if self._inplaced_kv:
            binding.bind_cpu_input("history_len", self._history_len)
            for name in self._output_names:
                if name == "token":
                    binding.bind_ortvalue_output("token", self._step_token_out_ov)
                elif name == "seen_out":
                    binding.bind_ortvalue_output("seen_out", self._seen_values[1])
                elif name == "hidden":
                    binding.bind_output("hidden", "cpu", 0, self._lm_hidden_dtype,
                                        [1, 1, self._audio_hidden.shape[2]], self._audio_hidden.ctypes.data)
                else:
                    past_name = name.replace("present.", "past_key_values.", 1)
                    kv_val = self._static_kv[past_name]
                    binding.bind_ortvalue_input(past_name, kv_val)
                    binding.bind_ortvalue_output(name, kv_val)
        else:
            for name in self._output_names:
                if name == "token":
                    binding.bind_ortvalue_output("token", self._step_token_out_ov)
                else:
                    binding.bind_output(name, "cpu")
        return binding

    def set_reference(self, prompt_audio_path: str | Path, prompt_text: str) -> float:
        """Encode reference audio and transcript once for 80M voice cloning; returns preprocess time in seconds."""
        with self._lock:
            return self._set_reference(prompt_audio_path, prompt_text)

    def _set_reference(self, prompt_audio_path: str | Path, prompt_text: str) -> float:
        if self.tts_mode != "voice_clone" or self.ref_preprocess is None:
            return 0.0
        if self._cancel.is_set():
            raise RuntimeError("Hojo TTS cancelled")
        arg = self.ref_preprocess.get_inputs()[0]
        audio = load_prompt_audio(prompt_audio_path, self.in_sample_rate, arg)
        if USE_AUDIO_NORMALIZER:
            audio = audio_normalizer(audio, io_dtype(arg))
        t0 = time.perf_counter()
        output_names = ["prompt_audio_ids", "speaker_embedding"]
        projection_name = self.meta.get("reference_speaker_projection_output")
        if self._stateful_vocoder is not None and projection_name:
            output_names.append(projection_name)
        reference_values = self.ref_preprocess.run(output_names, {arg.name: audio}, run_options=self._ref_run)
        audio_ids, spk_emb = reference_values[:2]
        elapsed = time.perf_counter() - t0
        norm_prompt = normalize_zh(prompt_text) if self.text_normalize else prompt_text
        reference_ids = np.ascontiguousarray(audio_ids, dtype=np.int32).ravel()
        prefix, suffix = prompt_parts(self.meta, self.tokenizer, norm_prompt, reference_ids)
        if self._cancel.is_set():
            raise RuntimeError("Hojo TTS cancelled")
        speaker_dtype = (np.dtype(np.float32) if self._stateful_vocoder is not None else
                         io_dtype(next(i for i in self.vocoder.get_inputs() if i.name == "speaker_embedding")))
        self._ref_speaker_emb = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(spk_emb, dtype=speaker_dtype))
        if self._stateful_vocoder is not None:
            projected = (ort.OrtValue.ortvalue_from_numpy(reference_values[2])
                         if projection_name else self._ref_speaker_emb)
            self._stateful_vocoder.set_speaker(projected)
        self._ref_audio_ids = reference_ids
        self._prefix, self._suffix = prefix, suffix
        self._reference_prefix_kv = None
        return elapsed

    @property
    def voice_id(self) -> int | None:
        """Default 0-based int32 preset voice row; None for voice-cloning packages."""
        return self._voice_default

    @voice_id.setter
    def voice_id(self, value: int | np.integer | np.ndarray | str) -> None:
        self._voice_default = self._resolve_voice(value, None)

    @property
    def voice(self) -> str | None:
        return None if self._voice_default is None else self.voices[self._voice_default]

    @voice.setter
    def voice(self, value: int | np.integer | np.ndarray | str) -> None:
        self._voice_default = self._resolve_voice(None, value)

    def _resolve_voice(self, voice_id, voice) -> int | None:
        """Resolve the `voice_id` / `voice` aliases; both None keeps the default voice."""
        given = [value for value in (voice_id, voice) if value is not None]
        if not given:
            return self._voice_default
        if not self.voices:
            raise ValueError("This Hojo voice-cloning package has no preset voices; use set_reference() instead")
        indices = {resolve_voice_index(value, self._voice_lookup, self._voice_listing) for value in given}
        if len(indices) > 1:
            raise ValueError(f"Conflicting Hojo voice_id={voice_id!r} and voice={voice!r}")
        return indices.pop()

    def prompt_ids(self, text: str) -> np.ndarray:
        m = self.meta
        if self.tts_mode == "voice_clone" and self._ref_audio_ids is None:
            raise RuntimeError("Hojo 80M voice cloning requires set_reference(prompt_audio, prompt_text) first")
        ids = build_prompt(self.tokenizer, text, self._prefix, self._suffix)
        if ids.size + self._max_new > m["max_positions"]:
            raise ValueError(f"Hojo TTS: text too long ({ids.size} prompt tokens, max {m['max_positions']})")
        return ids

    def cancel(self) -> None:
        self._cancel.set()
        self._lm_run.terminate = True
        self._voc_run.terminate = True
        if self._decode_run is not None:
            self._decode_run.terminate = True
        if self._prefix_run is not None:
            self._prefix_run.terminate = True
        if self._ref_run is not None:
            self._ref_run.terminate = True

    def reset_cancel(self) -> None:
        with self._lock:
            self._cancel.clear()
            self._lm_run.terminate = False
            self._voc_run.terminate = False
            if self._decode_run is not None:
                self._decode_run.terminate = False
            if self._prefix_run is not None:
                self._prefix_run.terminate = False
            if self._ref_run is not None:
                self._ref_run.terminate = False

    def _prefill_state(self, prompt: np.ndarray) -> tuple[np.ndarray, dict[str, ort.OrtValue], float]:
        empty = {name: self._empty_kv[name[name.rindex("."):]] for name in self._past_names}
        if not self._prefix_cache_enabled:
            return prompt, empty, 0.0
        if self._cancel.is_set():
            raise RuntimeError("Hojo TTS cancelled")
        length = len(self._prefix)
        if not length or prompt[0, :length].tolist() != self._prefix:
            raise ValueError("Cloning prompt does not begin with its encoded reference-text prefix")
        elapsed = 0.0
        state = self._reference_prefix_kv
        if state is None:
            started = time.perf_counter()
            binding = self.lm.io_binding()
            prefix = np.asarray(self._prefix, np.int32)[None]
            binding.bind_cpu_input("input_ids", prefix)
            binding.bind_cpu_input("ctrl", self._greedy_ctrl)
            self._bind_random(binding)
            binding.bind_cpu_input("seen", self._seen[0])
            for name, value in empty.items():
                binding.bind_ortvalue_input(name, value)
            for name in self._present_names:
                binding.bind_output(name, "cpu")
            self.lm.run_with_iobinding(binding, self._prefix_run)
            state = dict(zip(self._past_names, binding.get_outputs()))
            self._reference_prefix_kv = state
            self.reference_prefix_cache_misses += 1
            elapsed = time.perf_counter() - started
        else:
            self.reference_prefix_cache_hits += 1
        return prompt[:, length:], state, elapsed

    def _bind_random(self, binding: ort.IOBinding) -> None:
        if self._counter_rng:
            binding.bind_cpu_input("rng_seed", self._rng_seed)
            binding.bind_cpu_input("rng_step", self._rng_step)
        else:
            binding.bind_cpu_input("uniform", self._uniform)

    def _generate_steps(self, prompt: np.ndarray) -> Iterator[tuple[int, float, float]]:
        """Yield cumulative LM timings, excluding time suspended for vocoding or playback."""
        self._ctrl[:] = (self.temperature, self.top_p, self.repetition_penalty)
        greedy = self.temperature <= 0.0
        counter_rng = self._counter_rng
        if counter_rng:
            # Every sentence restarts the counter stream, so it matches a freshly constructed instance.
            self._rng_seed[0] = self.seed % (1 << 64)
            self._rng_step[0] = 0
        elif greedy:
            self._uniform[0] = 0.0
        else:
            draws = np.random.RandomState(self.seed).randint(0, 1 << 32, size=self._max_new, dtype=np.uint32)
            np.right_shift(draws, 8, out=draws)
            np.multiply(draws, np.float32(_U24), out=self._uniform_draws, dtype=np.float32)
        audio_first, audio_last = self.meta["audio_token_first"], self.meta["audio_token_last"]
        self._frames = 0
        end_id = self.meta["speech_end_id"]
        self._seen[0].fill(False)
        inplaced_kv = self._inplaced_kv
        if inplaced_kv:
            feed, past, prefix_s = prompt, self._static_kv, 0.0
            self._history_len[0] = 0
        else:
            feed, past, prefix_s = self._prefill_state(prompt)

        session, run_options, binding = self.lm, self._lm_run, self._lm_binding
        decode_binding = self._decode_binding
        if not inplaced_kv:
            for name, value in past.items():
                binding.bind_ortvalue_input(name, value)
            for name in self._present_names:
                binding.bind_output(name, "cpu")
        hidden_shape = [1, 1, self._audio_hidden.shape[2]]
        hidden_step = self._audio_hidden.strides[1]
        hidden_ptr = self._audio_hidden.ctypes.data
        seen_in_ov, seen_out_ov = self._seen_values
        binding.bind_input("input_ids", "cpu", 0, np.int32, list(feed.shape), feed.ctypes.data)
        last_hidden_frame = -1
        prefill_s = decode_s = 0.0
        n = 0
        while n < self._max_new:
            t0 = time.perf_counter()
            if self._cancel.is_set():
                raise RuntimeError("Hojo TTS cancelled")
            if counter_rng:
                self._rng_step[0] = n
            elif not greedy:
                self._uniform[0] = self._uniform_draws[n]
            binding.bind_ortvalue_input("seen", seen_in_ov)
            binding.bind_ortvalue_output("seen_out", seen_out_ov)
            if self._frames != last_hidden_frame:
                binding.bind_output("hidden", "cpu", 0, self._lm_hidden_dtype, hidden_shape,
                                    hidden_ptr + hidden_step * self._frames)
                last_hidden_frame = self._frames
            session.run_with_iobinding(binding, run_options=run_options)
            token = int(self._step_token_out[0, 0])
            self._tokens[0, n] = token
            if audio_first <= token <= audio_last:
                self._audio_tokens[0, self._frames] = token
                self._frames += 1
            n += 1
            finished = token == end_id or n == self._max_new
            if not finished:
                self._step_token_in[0, 0] = token
                if inplaced_kv:
                    self._history_len[0] += feed.shape[1] if n == 1 else 1
                    if n == 1:
                        if decode_binding is not None and self._decode is not None:
                            binding = decode_binding
                            session, run_options = self._decode, self._decode_run
                            last_hidden_frame = -1
                        else:
                            binding.bind_ortvalue_input("input_ids", self._step_token_in_ov)
                else:
                    outputs = binding.get_outputs()
                    if n == 1:
                        if decode_binding is not None and self._decode is not None:
                            binding = decode_binding
                            session, run_options = self._decode, self._decode_run
                            last_hidden_frame = -1
                        else:
                            binding.bind_ortvalue_input("input_ids", self._step_token_in_ov)
                    for name, index in zip(self._past_names, self._kv_output_indices):
                        binding.bind_ortvalue_input(name, outputs[index])
                    # Dynamic Concat outputs must be unbound before their shapes grow on the next step.
                    for name in self._present_names:
                        binding.bind_output(name, "cpu")
                seen_in_ov, seen_out_ov = seen_out_ov, seen_in_ov
            elapsed = time.perf_counter() - t0
            if n == 1:
                prefill_s = elapsed + prefix_s
            else:
                decode_s += elapsed
            yield n, prefill_s, decode_s
            if finished:
                break

    def _generate(self, prompt: np.ndarray) -> tuple[int, float, float]:
        result = None
        for result in self._generate_steps(prompt):
            pass
        if result is None:
            raise RuntimeError("Hojo TTS: generation produced no steps")
        return result

    def _audio_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        """Contiguous views of the audio frames compacted while generating."""
        if not self._frames:
            raise RuntimeError("Hojo TTS: the LM produced no audio tokens")
        return self._audio_hidden[:, :self._frames], self._audio_tokens[:, :self._frames]

    def _vocoder_hidden_input(self, hidden: np.ndarray) -> np.ndarray:
        if hidden.dtype == self._voc_hidden_dtype:
            return np.ascontiguousarray(hidden)
        buffer = self._vocoder_hidden_buffer
        if buffer is None or hidden.shape[1] > buffer.shape[1]:
            raise RuntimeError("Hojo vocoder conversion buffer does not match the hidden-state contract")
        converted = buffer[:, :hidden.shape[1]]
        np.copyto(converted, hidden, casting="same_kind")
        return converted

    def _vocode(self, *, start_frame: int = 0, stop_frame: int | None = None) -> tuple[np.ndarray, int]:
        hidden, audio = self._audio_inputs()
        end_frame = audio.shape[1] if stop_frame is None else stop_frame
        if not 0 <= start_frame < end_frame <= audio.shape[1]:
            raise ValueError(f"Invalid Hojo vocoder frame window [{start_frame}, {end_frame})")
        hidden, audio = hidden[:, start_frame:end_frame], audio[:, start_frame:end_frame]
        hidden = self._vocoder_hidden_input(hidden)
        if self._stateful_vocoder is not None:
            output = np.empty(audio.shape[1] * self.meta["hop_length"], np.float32)
            self._stateful_vocoder.reset()
            self._stateful_vocoder.process(hidden, audio, output, final=True)
            if self._cancel.is_set():
                raise RuntimeError("Hojo TTS cancelled")
            return output, int(audio.shape[1])
        binding = self.vocoder.io_binding()
        binding.bind_cpu_input("hidden_states", hidden)
        binding.bind_cpu_input("audio_tokens", audio)
        if self.voices:
            binding.bind_ortvalue_input("voice", self._voice_value)
        else:
            binding.bind_ortvalue_input("speaker_embedding", self._ref_speaker_emb)
        binding.bind_output("wav", "cpu")
        self.vocoder.run_with_iobinding(binding, run_options=self._voc_run)
        return binding.get_outputs()[0].numpy(), int(audio.shape[1])

    def synthesize_sentence(
        self,
        text: str,
        *,
        voice_id: int | np.integer | np.ndarray | str | None = None,
        voice: int | np.integer | np.ndarray | str | None = None,
    ) -> tuple[np.ndarray, SynthesisStats]:
        return self._synthesize(text, self._resolve_voice(voice_id, voice))

    def _synthesize(self, text: str, voice_idx: int | None) -> tuple[np.ndarray, SynthesisStats]:
        with self._lock:
            if voice_idx is not None:
                self._voice_buf[()] = voice_idx
            prompt = self.prompt_ids(text)
            count, prefill_s, decode_s = self._generate(prompt)
            t0 = time.perf_counter()
            wav, frames = self._vocode()
            vocoder_s = time.perf_counter() - t0
            return wav, SynthesisStats(
                text, count, frames, wav.size, self.sample_rate, prefill_s, decode_s, vocoder_s
            )

    def chunks(self, text: str) -> list[str]:
        if self.text_normalize:
            text = normalize_zh(text)
        return split_sentences(text, self.sentence_max_chars)

    def stream(
        self,
        text: str,
        *,
        voice_id: int | np.integer | np.ndarray | str | None = None,
        voice: int | np.integer | np.ndarray | str | None = None,
    ) -> Iterator[tuple[np.ndarray, SynthesisStats]]:
        voice_idx = self._resolve_voice(voice_id, voice)
        self.reset_cancel()
        for chunk in self.chunks(text):
            yield self._synthesize(chunk, voice_idx)

    def stream_audio(
        self,
        text: str,
        *,
        first_chunk_frames: int | None = None,
        chunk_frames: int | None = None,
        voice_id: int | np.integer | np.ndarray | str | None = None,
        voice: int | np.integer | np.ndarray | str | None = None,
    ) -> Iterator[tuple[np.ndarray, SynthesisStats]]:
        """Stream PCM using the package profile; explicit smaller chunks trade throughput for latency."""
        if first_chunk_frames is None:
            first_chunk_frames = self.meta.get("stream_first_chunk_frames", 32)
        if chunk_frames is None:
            chunk_frames = self.meta.get("stream_chunk_frames", 128)
        for name, value in (("first_chunk_frames", first_chunk_frames), ("chunk_frames", chunk_frames)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        context_keys = ("vocoder_left_context_frames", "vocoder_right_context_frames")
        if not all(key in self.meta for key in context_keys):
            raise ValueError("Audio-frame streaming requires certified vocoder context; re-export the Hojo package")
        if self.sample_rate != self.meta["model_sample_rate"]:
            raise ValueError("Audio-frame streaming currently requires the native model sample rate")
        left, right = (self.meta[key] for key in context_keys)
        hop = self.meta["hop_length"]
        voice_idx = self._resolve_voice(voice_id, voice)
        self.reset_cancel()
        for text_chunk in self.chunks(text):
            with self._lock:
                if voice_idx is not None:
                    self._voice_buf[()] = voice_idx
                prompt = self.prompt_ids(text_chunk)
                if self._stateful_vocoder is not None:
                    yield from self._stream_stateful_sentence(
                        text_chunk, prompt, int(first_chunk_frames), int(chunk_frames), right,
                    )
                    continue
                frames = emitted = previous_count = 0
                previous_prefill = previous_decode = 0.0
                next_emit = int(first_chunk_frames)
                for count, prefill_s, decode_s in self._generate_steps(prompt):
                    token = int(self._tokens[0, count - 1])
                    frames += self.meta["audio_token_first"] <= token <= self.meta["audio_token_last"]
                    finished = token == self.meta["speech_end_id"] or count == self._max_new
                    if finished and not frames:
                        raise RuntimeError("Hojo TTS: the LM produced no audio tokens")
                    stable = frames if finished else max(0, frames - right)
                    if stable <= emitted or stable < next_emit and not finished:
                        continue
                    start_frame = max(0, emitted - left)
                    started = time.perf_counter()
                    waveform, _ = self._vocode(start_frame=start_frame, stop_frame=frames)
                    wav = waveform[(emitted - start_frame) * hop:(stable - start_frame) * hop]
                    if wav.size != (stable - emitted) * hop:
                        raise RuntimeError("Hojo vocoder output does not match the certified streaming frame size")
                    stats = SynthesisStats(
                        text_chunk, count - previous_count, stable - emitted, wav.size, self.sample_rate,
                        prefill_s - previous_prefill, decode_s - previous_decode, time.perf_counter() - started,
                    )
                    emitted, previous_count = stable, count
                    previous_prefill, previous_decode = prefill_s, decode_s
                    next_emit = emitted + int(chunk_frames)
                    yield wav, stats

    def _stream_stateful_sentence(
        self, text: str, prompt: np.ndarray, first_frames: int, chunk_frames: int, right: int,
    ) -> Iterator[tuple[np.ndarray, SynthesisStats]]:
        processor = self._stateful_vocoder
        if processor is None:
            raise RuntimeError("The stateful vocoder is not initialized")
        processor.reset()
        hop = self.meta["hop_length"]
        # Fresh per sentence: callers may still hold views of earlier sentences' PCM.
        storage = np.empty(self._max_new * hop, np.float32)
        processed = emitted = previous_count = 0
        previous_prefill = previous_decode = 0.0
        next_emit = first_frames
        for count, prefill_s, decode_s in self._generate_steps(prompt):
            token = int(self._tokens[0, count - 1])
            frames = self._frames
            finished = token == self.meta["speech_end_id"] or count == self._max_new
            if finished and not frames:
                raise RuntimeError("Hojo TTS: the LM produced no audio tokens")
            stable = frames if finished else max(0, frames - right)
            if stable <= emitted or stable < next_emit and not finished:
                continue
            started = time.perf_counter()
            processor.process(self._vocoder_hidden_input(self._audio_hidden[:, processed:frames]),
                              self._audio_tokens[:, processed:frames],
                              storage[processor.produced_samples:], final=finished)
            if processor.produced_samples < stable * hop:
                raise RuntimeError("The stateful vocoder did not produce the requested stable PCM")
            if self._cancel.is_set():
                raise RuntimeError("Hojo TTS cancelled")
            wav = storage[emitted * hop:stable * hop]
            stats = SynthesisStats(
                text, count - previous_count, stable - emitted, wav.size, self.sample_rate,
                prefill_s - previous_prefill, decode_s - previous_decode, time.perf_counter() - started,
            )
            processed = frames
            emitted, previous_count = stable, count
            previous_prefill, previous_decode = prefill_s, decode_s
            next_emit = emitted + chunk_frames
            yield wav, stats

    def synthesize(
        self,
        text: str,
        *,
        voice_id: int | np.integer | np.ndarray | str | None = None,
        voice: int | np.integer | np.ndarray | str | None = None,
    ) -> np.ndarray:
        parts = [wav for wav, _ in self.stream(text, voice_id=voice_id, voice=voice)]
        if not parts:
            raise ValueError("Hojo TTS: text is empty after normalization")
        return np.concatenate(parts) if len(parts) > 1 else parts[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact Hojo-TTS ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=ONNX_FOLDER,
        help="Optional path to Hojo TTS ONNX package folder (defaults to HojoTTS_Optimized, auto-fallback to HojoTTS_ONNX).",
    )
    parser.add_argument("--text", default=None, help="Optional single target text override.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output WAV path override.")
    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument(
        "--voice-id",
        dest="voice_id",
        type=int,
        default=None,
        help="Optional 0-based int32 preset voice index override (0..N-1 of the package voices).",
    )
    voice_group.add_argument(
        "--voice",
        default=None,
        help="Optional preset voice override (voice name or 0-based int32 index).",
    )
    parser.add_argument("--prompt-audio", default=None, help="Optional 80M reference audio path override.")
    parser.add_argument("--prompt-text", default=None, help="Optional 80M reference transcript override.")
    parser.add_argument("--vocoder-backend", choices=("auto", "onnx", "stateful"), default="auto",
                        help="Choose the portable ONNX renderer or the advanced CPU stateful renderer.")
    parser.add_argument("--no-decode", action="store_true", help="Use the prefill LM for all decoding steps.")
    parser.add_argument("--no-reference-prefix-cache", action="store_true",
                        help="Disable reuse of the invariant cloning reference-text prefix.")
    parser.add_argument("--custom-ops-library", type=Path, default=None,
                        help="Optional KVCacheUpdate custom ops library for lm_kv_update='inplaced_kv' packages (or HOJO_KV_OPS_LIBRARY).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    pipeline_started = time.perf_counter()
    args = parse_args(argv)
    onnx_folder = resolve_model_dir(args.onnx_folder)
    output_path = Path(args.output or GENERATED_AUDIO_PATH).expanduser().resolve()
    targets = [args.text] if args.text else list(TARGET_TTS)

    print_progress(f"Loading Hojo-TTS package: {onnx_folder}")
    tts = HojoTts(
        onnx_folder, voice_id=args.voice_id, voice=args.voice, use_decode=not args.no_decode,
        vocoder_backend=args.vocoder_backend, cache_reference_prefix=not args.no_reference_prefix_cache,
        custom_ops_library=args.custom_ops_library,
    )
    if tts.voices:
        print_progress(f"Selected preset voice_id={tts.voice_id} ({tts.voice}) from {len(tts.voices)} voices")
    encoder_time = 0.0
    if tts.tts_mode == "voice_clone":
        ref_audio = args.prompt_audio or PROMPT_AUDIO_PATH
        ref_text = args.prompt_text if args.prompt_text is not None else PROMPT_TEXT
        print_progress(f"Encoding reference audio: {ref_audio}")
        encoder_time = tts.set_reference(ref_audio, ref_text)
        print_progress(f"Reference audio encoded in {encoder_time:.2f}s ({tts._ref_audio_ids.size} codec tokens).")

    wave_dtype = (np.dtype(np.float32) if tts._stateful_vocoder is not None
                  else io_dtype(tts.vocoder.get_outputs()[0]))
    empty_segment = np.zeros(int(tts.sample_rate * 0.2), dtype=wave_dtype)
    save_generated_wav: list[np.ndarray] = []
    total_samples = 0
    total_gen_s = 0.0
    total_voc_s = 0.0

    for idx, target in enumerate(targets, start=1):
        chunks_wav: list[np.ndarray] = []
        target_tokens = 0
        target_frames = 0
        target_gen_s = 0.0
        target_voc_s = 0.0
        for wav, st in tts.stream(target):
            chunks_wav.append(wav)
            target_tokens += st.tokens
            target_frames += st.audio_tokens
            target_gen_s += st.total_s
            target_voc_s += st.vocoder_s
        if not chunks_wav:
            continue
        merged = np.concatenate(chunks_wav) if len(chunks_wav) > 1 else chunks_wav[0]
        if USE_AUDIO_NORMALIZER and merged.size:
            # One gain per target text: per-chunk RMS gains made adjacent chunks jump in loudness.
            merged = audio_normalizer(merged, wave_dtype)
        audio_s = merged.size / tts.sample_rate
        total_samples += merged.size
        total_gen_s += target_gen_s
        total_voc_s += target_voc_s
        rtf_text = f"{target_gen_s / audio_s:.3f}" if audio_s > 0 else "n/a"
        print(
            f"Target {idx}/{len(targets)}: frames={target_frames}, tokens={target_tokens}, "
            f"audio={audio_s:.2f}s, generation={target_gen_s:.2f}s, RTF={rtf_text}",
            flush=True,
        )
        save_generated_wav.extend((merged, empty_segment))

    if total_samples:
        audio_duration = total_samples / tts.sample_rate
        overall_rtf = (encoder_time + total_gen_s) / audio_duration
        print(
            f"Overall: encoder={encoder_time:.2f}s, vocoder={total_voc_s:.2f}s, "
            f"audio={audio_duration:.2f}s, RTF={overall_rtf:.3f}",
            flush=True,
        )
        print_progress(f"Writing generated audio: {output_path}")
        final_audio = np.concatenate(save_generated_wav)
        if final_audio.dtype == np.float16:
            final_audio = final_audio.astype(np.float32)
        output_subtype = "PCM_16" if np.issubdtype(final_audio.dtype, np.integer) else "FLOAT"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), final_audio, tts.sample_rate, subtype=output_subtype, format="WAVEX")
    else:
        print_progress("No audio was generated.")
    print_progress(f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
