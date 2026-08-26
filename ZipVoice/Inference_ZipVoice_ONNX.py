"""Run the end-to-end ZipVoice ONNX pipeline with minimal NumPy orchestration."""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import time
from functools import reduce
from pathlib import Path
from typing import Any

import cn2an
import inflect
import jieba
import numpy as np
import onnxruntime as ort
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence
from piper_phonemize import phonemize_espeak
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Example_Audio import model_reference  # noqa: E402

# ============================== USER CONFIG ==============================

ORT_PROVIDERS = ["CPUExecutionProvider"]
DEVICE_ID = 0
MAX_THREADS = 0
ORT_LOG = False

PROMPT_WAV, PROMPT_TEXT = model_reference("zipvoice")
_SECOND_PROMPT_WAV, _SECOND_PROMPT_TEXT = model_reference(
    "zipvoice_dialog_speaker_2"
)
SECOND_PROMPT_WAV: str | None = _SECOND_PROMPT_WAV
SECOND_PROMPT_TEXT: str | None = _SECOND_PROMPT_TEXT
TARGET_TEXT = "Hello everyone, I am currently exploring speech synthesis with ZipVoice."
DIALOG_TARGET_TEXT = (
    "[S1]Hello! This voice was cloned from the first example."
    "[S2]你好！这个声音是从第二段示例音频克隆的。"
)
OUTPUT_WAV: Path | None = None
TEST_LIST: str | None = None
OUTPUT_DIRECTORY = SCRIPT_DIR / "zipvoice_results"

RAW_EVALUATION = False
REMOVE_LONG_SILENCE = False
SPEED = 1.0
RANDOM_SEED = 9527
NUM_STEP: int | None = None
GUIDANCE_SCALE: float | None = None
T_SHIFT: float | None = None
MAX_BATCH_DURATION = 100.0
PUNCTUATION = {";", ":", ",", ".", "!", "?", "；", "：", "，", "。", "！", "？"}

jieba.default_logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-folder",
        type=Path,
        default=SCRIPT_DIR / "ZipVoice_Distill_Optimized",
        help="ZipVoice package containing Pipeline and Metadata ONNX models.",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="Optional tokens.txt override.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("emilia", "libritts", "dialog"),
        default=None,
        help="Tokenizer type; must match the exported package metadata.",
    )
    parser.add_argument(
        "--task-mode",
        choices=("single", "dialogue"),
        default=None,
        help="Inference task; must match the exported package metadata.",
    )
    return parser.parse_args()


class EnglishTextNormalizer:
    def __init__(self) -> None:
        abbreviations = (
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
            ("etc", "et cetera"),
            ("btw", "by the way"),
        )
        self.abbreviations = [
            (re.compile(rf"\b{abbreviation}\b", re.IGNORECASE), replacement)
            for abbreviation, replacement in abbreviations
        ]
        self.inflect = inflect.engine()
        self.comma_number = re.compile(r"([0-9][0-9,]+[0-9])")
        self.decimal_number = re.compile(r"([0-9]+\.[0-9]+)")
        self.percent_number = re.compile(r"([0-9.,]*[0-9]+%)")
        self.pounds = re.compile(r"£([0-9,]*[0-9]+)")
        self.dollars = re.compile(r"\$([0-9.,]*[0-9]+)")
        self.fraction = re.compile(r"([0-9]+)/([0-9]+)")
        self.ordinal = re.compile(r"[0-9]+(st|nd|rd|th)")
        self.number = re.compile(r"[0-9]+")

    def normalize(self, text: str) -> str:
        for pattern, replacement in self.abbreviations:
            text = pattern.sub(replacement, text)
        text = self.comma_number.sub(
            lambda match: match.group(1).replace(",", ""),
            text,
        )
        text = self.pounds.sub(r"\1 pounds", text)
        text = self.dollars.sub(self._expand_dollars, text)
        text = self.fraction.sub(self._expand_fraction, text)
        text = self.decimal_number.sub(
            lambda match: match.group(1).replace(".", " point "),
            text,
        )
        text = self.percent_number.sub(
            lambda match: match.group(1).replace("%", " percent "),
            text,
        )
        text = self.ordinal.sub(
            lambda match: f" {self.inflect.number_to_words(match.group(0))} ",
            text,
        )
        return self.number.sub(self._expand_number, text)

    def _expand_dollars(self, match: re.Match[str]) -> str:
        value = match.group(1)
        parts = value.split(".")
        if len(parts) > 2:
            return f" {value} dollars "
        dollars = int(parts[0]) if parts[0] else 0
        cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        if dollars and cents:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            cent_unit = "cent" if cents == 1 else "cents"
            return f" {dollars} {dollar_unit}, {cents} {cent_unit} "
        if dollars:
            dollar_unit = "dollar" if dollars == 1 else "dollars"
            return f" {dollars} {dollar_unit} "
        if cents:
            cent_unit = "cent" if cents == 1 else "cents"
            return f" {cents} {cent_unit} "
        return " zero dollars "

    def _expand_fraction(self, match: re.Match[str]) -> str:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if numerator == 1 and denominator == 2:
            return " one half "
        if numerator == 1 and denominator == 4:
            return " one quarter "
        if denominator == 2:
            return f" {self.inflect.number_to_words(numerator)} halves "
        if denominator == 4:
            return f" {self.inflect.number_to_words(numerator)} quarters "
        numerator_words = self.inflect.number_to_words(numerator)
        denominator_words = self.inflect.number_to_words(denominator)
        return f" {numerator_words} {self.inflect.ordinal(denominator_words)} "

    def _expand_number(self, match: re.Match[str]) -> str:
        number = int(match.group(0))
        if 1000 < number < 3000:
            if number == 2000:
                return " two thousand "
            if 2000 < number < 2010:
                return f" two thousand {self.inflect.number_to_words(number % 100)} "
            if number % 100 == 0:
                return f" {self.inflect.number_to_words(number // 100)} hundred "
            words = self.inflect.number_to_words(
                number,
                andword="",
                zero="oh",
                group=2,
            ).replace(", ", " ")
            return f" {words} "
        return f" {self.inflect.number_to_words(number, andword='')} "


class Tokenizer:
    def __init__(self, token_file: str | Path) -> None:
        self.token2id: dict[str, int] = {}
        with Path(token_file).open(encoding="utf-8") as file:
            for line in file:
                token, token_id = line.rstrip().split("\t")[:2]
                if token in self.token2id:
                    raise ValueError(f"Duplicate token {token!r} in {token_file}.")
                self.token2id[token] = int(token_id)
        self.pad_id = self.token2id["_"]
        self.vocab_size = len(self.token2id)

    def texts_to_token_ids(self, texts: list[str]) -> list[list[int]]:
        return self.tokens_to_token_ids(self.texts_to_tokens(texts))

    def texts_to_tokens(self, _texts: list[str]) -> list[list[str]]:
        del _texts
        raise NotImplementedError

    def tokens_to_token_ids(
        self,
        token_sequences: list[list[str]],
    ) -> list[list[int]]:
        return [
            [self.token2id[token] for token in tokens if token in self.token2id]
            for tokens in token_sequences
        ]


class EmiliaTokenizer(Tokenizer):
    PUNCTUATION_TRANSLATION = str.maketrans(
        {
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "；": ";",
            "：": ":",
            "、": ",",
            "‘": "'",
            "“": '"',
            "”": '"',
            "’": "'",
        }
    )

    def __init__(self, token_file: str | Path) -> None:
        super().__init__(token_file)
        self.english_normalizer = EnglishTextNormalizer()

    def preprocess_text(self, text: str) -> str:
        return self.map_punctuation(text)

    def texts_to_tokens(self, texts: list[str]) -> list[list[str]]:
        token_sequences = []
        for text in texts:
            tokens = []
            for segment, language in self.get_segments(self.preprocess_text(text)):
                if language == "zh":
                    tokens.extend(self.tokenize_chinese(segment))
                elif language == "en":
                    tokens.extend(self.tokenize_english(segment))
                elif language == "pinyin":
                    tokens.extend(self.tokenize_pinyin(segment))
                elif language == "tag":
                    tokens.append(segment)
                else:
                    logging.warning(
                        "No English or Chinese characters found; skipping %r.",
                        segment,
                    )
            token_sequences.append(tokens)
        return token_sequences

    def tokenize_chinese(self, text: str) -> list[str]:
        try:
            normalized = cn2an.transform(text, "an2cn")
            syllables = lazy_pinyin(
                list(jieba.cut(normalized)),
                style=Style.TONE3,
                tone_sandhi=True,
                neutral_tone_with_five=True,
            )
            tokens = []
            for syllable in syllables:
                if syllable[:-1].isalpha() and syllable[-1:] in "12345":
                    tokens.extend(self.separate_pinyin(syllable))
                else:
                    tokens.append(syllable)
            return tokens
        except Exception as error:
            logging.warning("Chinese tokenization failed: %s", error)
            return []

    def tokenize_english(self, text: str) -> list[str]:
        try:
            groups = phonemize_espeak(
                self.english_normalizer.normalize(text),
                "en-us",
            )
            return reduce(lambda left, right: left + right, groups)
        except Exception as error:
            logging.warning("English tokenization failed: %s", error)
            return []

    def tokenize_pinyin(self, text: str) -> list[str]:
        syllable = text.removeprefix("<").removesuffix(">")
        if not (syllable[:-1].isalpha() and syllable[-1:] in "12345"):
            logging.warning("Invalid pinyin token %r; skipping it.", text)
            return []
        return self.separate_pinyin(syllable)

    @staticmethod
    def separate_pinyin(syllable: str) -> list[str]:
        initial = to_initials(syllable, strict=False)
        final = to_finals_tone3(
            syllable,
            strict=False,
            neutral_tone_with_five=True,
        )
        tokens = []
        if initial:
            tokens.append(f"{initial}0")
        if final:
            tokens.append(final)
        return tokens

    @classmethod
    def map_punctuation(cls, text: str) -> str:
        return (
            text.translate(cls.PUNCTUATION_TRANSLATION)
            .replace("⋯", "…")
            .replace("···", "…")
            .replace("・・・", "…")
            .replace("...", "…")
        )

    @classmethod
    def get_segments(cls, text: str) -> list[tuple[str, str]]:
        parts = re.findall(r"[<[].*?[>\]]|.", text)
        languages = []
        for part in parts:
            if cls.is_chinese(part) or cls.is_pinyin(part):
                languages.append("zh")
            elif cls.is_alphabet(part):
                languages.append("en")
            else:
                languages.append("other")

        segments: list[tuple[str, str]] = []
        current = ""
        current_language = ""
        for index, language in enumerate(languages):
            if index == 0:
                current = parts[index]
                current_language = language
            elif current_language == "other":
                current += parts[index]
                current_language = language
            elif language in (current_language, "other"):
                current += parts[index]
            else:
                segments.append((current, current_language))
                current = parts[index]
                current_language = language
        if current:
            segments.append((current, current_language))

        split_segments = []
        for segment, language in segments:
            for part in re.split(r"([<[].*?[>\]])", segment):
                if not part:
                    continue
                if cls.is_pinyin(part):
                    split_segments.append((part, "pinyin"))
                elif cls.is_tag(part):
                    split_segments.append((part, "tag"))
                else:
                    split_segments.append((part, language))
        return split_segments

    @staticmethod
    def is_chinese(part: str) -> bool:
        return "\u4e00" <= part <= "\u9fa5"

    @staticmethod
    def is_alphabet(part: str) -> bool:
        return "A" <= part <= "Z" or "a" <= part <= "z"

    @staticmethod
    def is_pinyin(part: str) -> bool:
        return part.startswith("<") and part.endswith(">")

    @staticmethod
    def is_tag(part: str) -> bool:
        return part.startswith("[") and part.endswith("]")


class DialogTokenizer(EmiliaTokenizer):
    def __init__(self, token_file: str | Path) -> None:
        super().__init__(token_file)
        self.spk_a_id = self.token2id["[S1]"]
        self.spk_b_id = self.token2id["[S2]"]

    def preprocess_text(self, text: str) -> str:
        return self.map_punctuation(re.sub(r"\s*(\[S[12]\])\s*", r"\1", text))


class LibriTTSTokenizer(Tokenizer):
    def __init__(self, token_file: str | Path) -> None:
        super().__init__(token_file)
        try:
            from tacotron_cleaner.cleaners import custom_english_cleaners
        except ImportError as error:
            raise RuntimeError(
                "LibriTTS tokenization requires espnet_tts_frontend."
            ) from error
        self.normalize = custom_english_cleaners

    def texts_to_tokens(self, texts: list[str]) -> list[list[str]]:
        return [list(self.normalize(text)) for text in texts]


def provider_device(provider: str) -> str:
    if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        return "cuda"
    if provider == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def provider_options(providers: list[str]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for provider in providers:
        if provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            accelerator_options: dict[str, Any] = {"device_id": DEVICE_ID}
            if provider == "CUDAExecutionProvider":
                accelerator_options.update(
                    {
                        "arena_extend_strategy": "kSameAsRequested",
                        "cudnn_conv_algo_search": "EXHAUSTIVE",
                        "do_copy_in_default_stream": "1",
                        "use_tf32": "1",
                    }
                )
            options.append(accelerator_options)
        elif provider == "DmlExecutionProvider":
            options.append({"device_id": DEVICE_ID})
        else:
            options.append({})
    return options


def create_session_options() -> tuple[ort.SessionOptions, ort.RunOptions]:
    session_options = ort.SessionOptions()
    run_options = ort.RunOptions()
    severity = 0 if ORT_LOG else 4
    session_options.log_severity_level = severity
    run_options.log_severity_level = severity
    session_options.inter_op_num_threads = MAX_THREADS
    session_options.intra_op_num_threads = MAX_THREADS
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.use_device_allocator_for_initializers": "1",
    }.items():
        session_options.add_session_config_entry(key, value)
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return session_options, run_options


def discover_package(folder: Path) -> tuple[str, Path, Path]:
    pipeline_paths = tuple(folder.glob("*_Pipeline.onnx"))
    if len(pipeline_paths) != 1:
        raise ValueError(
            f"Expected exactly one *_Pipeline.onnx in {folder}, "
            f"found {len(pipeline_paths)}."
        )
    pipeline_path = pipeline_paths[0]
    package_stem = pipeline_path.name.removesuffix("_Pipeline.onnx")
    metadata_path = folder / f"{package_stem}_Metadata.onnx"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing metadata model for {pipeline_path.name}: {metadata_path}"
        )
    return package_stem, pipeline_path, metadata_path


def sinc_resample(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    if source_rate == target_rate:
        return np.ascontiguousarray(waveform)

    common_divisor = math.gcd(source_rate, target_rate)
    source_step = source_rate // common_divisor
    target_step = target_rate // common_divisor
    lowpass_filter_width = 6
    base_frequency = np.float32(min(source_step, target_step) * 0.99)
    width = math.ceil(lowpass_filter_width * source_step / base_frequency)

    offsets = (
        np.arange(-width, width + source_step, dtype=np.float32)
        / np.float32(source_step)
    )[None, :]
    phases = (
        np.arange(0, -target_step, -1, dtype=np.float32)
        / np.float32(target_step)
    )[:, None]
    positions = (phases + offsets) * base_frequency
    np.clip(
        positions,
        np.float32(-lowpass_filter_width),
        np.float32(lowpass_filter_width),
        out=positions,
    )
    window = np.square(
        np.cos(positions * np.float32(math.pi / lowpass_filter_width / 2.0))
    )
    angles = positions * np.float32(math.pi)
    kernels = np.ones_like(angles)
    nonzero = angles != 0
    kernels[nonzero] = np.sin(angles[nonzero]) / angles[nonzero]
    kernels *= window * np.float32(base_frequency / source_step)

    audio = np.asarray(waveform, dtype=np.float32)
    original_shape = audio.shape
    audio = audio.reshape(-1, original_shape[-1])
    padded = np.pad(audio, ((0, 0), (width, width + source_step)))
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        kernels.shape[-1],
        axis=-1,
    )[:, ::source_step]
    resampled = np.ascontiguousarray(windows) @ kernels.T
    target_length = math.ceil(target_step * original_shape[-1] / source_step)
    resampled = resampled.reshape(audio.shape[0], -1)[:, :target_length]
    return np.ascontiguousarray(
        resampled.reshape(original_shape[:-1] + (target_length,))
    )


def load_audio(path: str | Path, sampling_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )
    return sinc_resample(np.ascontiguousarray(audio.T), source_rate, sampling_rate)


def save_audio(path: Path, waveform: np.ndarray, sampling_rate: int) -> None:
    sf.write(
        str(path),
        np.ascontiguousarray(waveform.T),
        sampling_rate,
        subtype="PCM_16",
    )


def audio_to_segment(audio: np.ndarray, sampling_rate: int) -> AudioSegment:
    channels = audio.shape[0] if audio.ndim > 1 else 1
    channel_audio = audio.reshape(channels, -1)
    pcm = (channel_audio * np.float32(32768.0)).clip(-32768, 32767).astype(
        np.int16
    )
    interleaved = pcm.T.reshape(-1) if channels > 1 else pcm[0]
    return AudioSegment(
        data=interleaved.tobytes(),
        sample_width=2,
        frame_rate=sampling_rate,
        channels=channels,
    )


def segment_to_audio(segment: AudioSegment) -> np.ndarray:
    samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
    samples /= np.float32(32768.0)
    if segment.channels == 1:
        return np.ascontiguousarray(samples[None, :])
    return np.ascontiguousarray(samples.reshape(-1, segment.channels).T)


def remove_silence_edges(
    segment: AudioSegment,
    keep_silence: int = 100,
    silence_threshold: float = -50,
) -> AudioSegment:
    start = detect_leading_silence(
        segment,
        silence_threshold=silence_threshold,
    )
    segment = segment[max(0, start - keep_silence) :]
    reversed_segment = segment.reverse()
    start = detect_leading_silence(
        reversed_segment,
        silence_threshold=silence_threshold,
    )
    return reversed_segment[max(0, start - keep_silence) :].reverse()


def remove_silence(
    audio: np.ndarray,
    sampling_rate: int,
    only_edge: bool = False,
    trail_sil: float = 0,
) -> np.ndarray:
    segment = audio_to_segment(audio, sampling_rate)
    if not only_edge:
        non_silent = split_on_silence(
            segment,
            min_silence_len=1000,
            silence_thresh=-50,
            keep_silence=1000,
            seek_step=10,
        )
        segment = AudioSegment.silent(duration=0)
        for part in non_silent:
            segment += part
    segment = remove_silence_edges(segment)
    segment += AudioSegment.silent(duration=trail_sil)
    return segment_to_audio(segment)


def cross_fade_concat(
    chunks: list[np.ndarray],
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> np.ndarray:
    if len(chunks) <= 1:
        return chunks[0] if chunks else np.empty((1, 0), dtype=np.float32)
    fade_samples = int(fade_duration * sample_rate)
    if fade_samples <= 0:
        return np.concatenate(chunks, axis=-1)

    result = chunks[0]
    for next_chunk in chunks[1:]:
        overlap = min(fade_samples, result.shape[-1], next_chunk.shape[-1])
        if overlap <= 0:
            result = np.concatenate((result, next_chunk), axis=-1)
            continue
        fade = np.linspace(1.0, 0.0, overlap, dtype=np.float32)[None, :]
        result = np.concatenate(
            (
                result[..., :-overlap],
                result[..., -overlap:] * fade
                + next_chunk[..., :overlap] * (np.float32(1.0) - fade),
                next_chunk[..., overlap:],
            ),
            axis=-1,
        )
    return np.ascontiguousarray(result)


def add_punctuation(text: str) -> str:
    text = text.strip()
    if text[-1] not in PUNCTUATION:
        text += "."
    return text


def chunk_tokens_punctuation(
    tokens: list[str],
    max_tokens: int = 100,
) -> list[list[str]]:
    sentences: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if not current and sentences and (token in PUNCTUATION or token == " "):
            sentences[-1].append(token)
        else:
            current.append(token)
            if token in PUNCTUATION:
                sentences.append(current)
                current = []
    if current:
        sentences.append(current)

    chunks: list[list[str]] = []
    current = []
    for sentence in sentences:
        if len(current) + len(sentence) <= max_tokens:
            current.extend(sentence)
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def chunk_tokens_dialog(
    tokens: list[str],
    max_tokens: int = 100,
) -> list[list[str]]:
    dialogs: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token == "[S1]":
            if current:
                dialogs.append(current)
            current = []
        current.append(token)
    if current:
        dialogs.append(current)

    chunks: list[list[str]] = []
    current = []
    for dialog in dialogs:
        if len(current) + len(dialog) <= max_tokens:
            current.extend(dialog)
        else:
            if current:
                chunks.append(current)
            current = dialog
    if current:
        chunks.append(current)
    return chunks


def batchify_tokens(
    token_sequences: list[list[int]],
    max_duration: float,
    prompt_duration: float,
    token_duration: float,
) -> tuple[list[list[list[int]]], list[int]]:
    indexed = sorted(enumerate(token_sequences), key=lambda item: len(item[1]))
    indices = [index for index, _ in indexed]
    batches: list[list[list[int]]] = []
    batch: list[list[int]] = []
    batch_tokens = 0
    for tokens in (tokens for _, tokens in indexed):
        if (
            batch_tokens * token_duration
            + len(batch) * prompt_duration
            + len(tokens) * token_duration
            <= max_duration
        ):
            batch.append(tokens)
            batch_tokens += len(tokens)
        else:
            if batch:
                batches.append(batch)
            batch = [tokens]
            batch_tokens = len(tokens)
    if batch:
        batches.append(batch)
    return batches, indices


class ZipVoiceRuntime:
    def __init__(self, pipeline_path: Path, metadata_path: Path) -> None:
        self.device_type = provider_device(ORT_PROVIDERS[0])
        self.device_id = DEVICE_ID if self.device_type != "cpu" else 0
        self.session_options, self.run_options = create_session_options()
        metadata_session = ort.InferenceSession(
            str(metadata_path),
            sess_options=self.session_options,
            providers=["CPUExecutionProvider"],
        )
        metadata = metadata_session.get_modelmeta().custom_metadata_map
        self.metadata = dict(metadata)
        self.session = ort.InferenceSession(
            str(pipeline_path),
            sess_options=self.session_options,
            providers=ORT_PROVIDERS,
            provider_options=provider_options(ORT_PROVIDERS),
        )
        self.model_sample_rate = int(metadata["sample_rate"])
        self.in_sample_rate = int(
            metadata.get("in_sample_rate", self.model_sample_rate)
        )
        self.out_sample_rate = int(
            metadata.get("out_sample_rate", self.model_sample_rate)
        )
        self.dialogue = metadata["tokenizer_type"] == "dialog"
        self.output_channels = int(metadata["output_channels"])
        self.output_name = self.session.get_outputs()[0].name

    def create_context(
        self,
        audio: np.ndarray,
        prompt_ids: list[int],
        speed: float,
        num_step: int,
        guidance_scale: float,
        t_shift: float,
    ) -> dict[str, Any]:
        if not prompt_ids:
            raise ValueError("Prompt tokenization produced no tokens.")
        if speed <= 0.0:
            raise ValueError("SPEED must be greater than zero.")
        if num_step < 1:
            raise ValueError("NUM_STEP must be at least one.")
        if guidance_scale < 0.0:
            raise ValueError("GUIDANCE_SCALE must be non-negative.")
        if not 0.0 < t_shift <= 1.0:
            raise ValueError("T_SHIFT must be in the interval (0, 1].")

        fixed_arrays = {
            "audio": np.ascontiguousarray(audio[None], dtype=np.float32),
            "prompt_tokens": np.ascontiguousarray([prompt_ids], dtype=np.int64),
            "speed": np.asarray(speed, dtype=np.float32),
            "num_step": np.asarray(num_step, dtype=np.int64),
            "guidance_scale": np.asarray(guidance_scale, dtype=np.float32),
            "t_shift": np.asarray(t_shift, dtype=np.float32),
        }
        fixed_values = {
            name: ort.OrtValue.ortvalue_from_numpy(
                array,
                self.device_type,
                self.device_id,
            )
            for name, array in fixed_arrays.items()
        }
        return {"fixed_values": fixed_values, "workspaces": {}}

    def _workspace(
        self,
        context: dict[str, Any],
        target_ids: list[int],
    ) -> tuple[dict[str, Any], np.ndarray]:
        target_array = np.ascontiguousarray([target_ids], dtype=np.int64)
        key = target_array.shape
        workspace = context["workspaces"].get(key)
        if workspace is not None:
            return workspace, target_array

        target_value = ort.OrtValue.ortvalue_from_shape_and_type(
            target_array.shape,
            np.int64,
            self.device_type,
            self.device_id,
        )
        values = {**context["fixed_values"], "tokens": target_value}
        binding = self.session.io_binding()
        for argument in self.session.get_inputs():
            binding.bind_ortvalue_input(argument.name, values[argument.name])
        binding.bind_output(self.output_name, self.device_type, self.device_id)
        workspace = {
            "binding": binding,
            "target_value": target_value,
            "bound_values": tuple(values.values()),
        }
        context["workspaces"][key] = workspace
        return workspace, target_array

    def synthesize_chunk(
        self,
        context: dict[str, Any],
        target_ids: list[int],
    ) -> tuple[np.ndarray, float]:
        if not target_ids:
            raise ValueError("Target tokenization produced no tokens.")
        workspace, target_array = self._workspace(context, target_ids)
        workspace["target_value"].update_inplace(target_array)
        started = time.perf_counter()
        self.session.run_with_iobinding(
            workspace["binding"],
            run_options=self.run_options,
        )
        elapsed = time.perf_counter() - started
        waveform = workspace["binding"].get_outputs()[0].numpy().copy()
        return waveform, elapsed


def load_tools(tokenizer_type: str, tokens_path: Path) -> tuple[Any, dict[str, Any]]:
    tokenizer_classes = {
        "emilia": EmiliaTokenizer,
        "libritts": LibriTTSTokenizer,
        "dialog": DialogTokenizer,
    }
    tokenizer = tokenizer_classes[tokenizer_type](token_file=str(tokens_path))
    tools = {
        "add_punctuation": add_punctuation,
        "batchify_tokens": batchify_tokens,
        "chunk_dialog": chunk_tokens_dialog,
        "chunk_punctuation": chunk_tokens_punctuation,
        "crossfade": cross_fade_concat,
        "load_audio": load_audio,
        "remove_silence": remove_silence,
    }
    return tokenizer, tools


def resolve_input_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def input_rows(
    dialogue: bool,
    default_output_path: Path,
) -> list[tuple[str, str, tuple[Path, ...], str, Path]]:
    if TEST_LIST is None:
        if dialogue and SECOND_PROMPT_WAV is not None:
            if SECOND_PROMPT_TEXT is None:
                raise ValueError("SECOND_PROMPT_TEXT is required with SECOND_PROMPT_WAV.")
            prompt_paths = (
                Path(PROMPT_WAV).expanduser(),
                Path(SECOND_PROMPT_WAV).expanduser(),
            )
            prompt_text = f"[S1]{PROMPT_TEXT}[S2]{SECOND_PROMPT_TEXT}"
            target_text = DIALOG_TARGET_TEXT
        else:
            prompt_paths = (Path(PROMPT_WAV).expanduser(),)
            prompt_text = PROMPT_TEXT
            target_text = DIALOG_TARGET_TEXT if dialogue else TARGET_TEXT
        output_path = OUTPUT_WAV or default_output_path
        return [(output_path.stem, prompt_text, prompt_paths, target_text, output_path)]

    test_list = Path(TEST_LIST).expanduser().resolve()
    rows = []
    for line in test_list.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if dialogue and len(fields) == 6:
            name, first_text, second_text, first_wav, second_wav, target = fields
            prompt_text = f"[S1]{first_text}[S2]{second_text}"
            prompt_paths = (
                resolve_input_path(first_wav, test_list.parent),
                resolve_input_path(second_wav, test_list.parent),
            )
        else:
            name, prompt_text, prompt_wav, target = fields
            prompt_paths = (resolve_input_path(prompt_wav, test_list.parent),)
        output_name = name if Path(name).suffix else f"{name}.wav"
        output_path = OUTPUT_DIRECTORY / output_name
        rows.append((name, prompt_text, prompt_paths, target, output_path))
    return rows


def prepare_prompt_audio(
    paths: tuple[Path, ...],
    runtime: ZipVoiceRuntime,
    tools: dict[str, Any],
    trailing_silence_ms: int,
) -> np.ndarray:
    loaded = [
        tools["load_audio"](str(path), sampling_rate=runtime.in_sample_rate)
        for path in paths
    ]
    if runtime.output_channels == 1 and runtime.dialogue:
        loaded = [
            audio.mean(axis=0, keepdims=True) if audio.shape[0] != 1 else audio
            for audio in loaded
        ]
        prompt_audio = np.concatenate(loaded, axis=1)
    elif runtime.output_channels == 1:
        prompt_audio = np.concatenate(loaded, axis=1)
    elif len(loaded) == 1:
        if loaded[0].shape[0] != 2:
            raise ValueError("A merged stereo prompt must contain two channels.")
        prompt_audio = loaded[0]
    elif loaded[0].shape[0] == 2:
        prompt_audio = np.concatenate(loaded, axis=1)
    else:
        first_length = loaded[0].shape[1]
        total_length = first_length + loaded[1].shape[1]
        prompt_audio = np.zeros((2, total_length), dtype=np.float32)
        prompt_audio[0, :first_length] = loaded[0][0]
        prompt_audio[1, first_length:] = loaded[1][0]

    if not RAW_EVALUATION:
        prompt_audio = tools["remove_silence"](
            prompt_audio,
            runtime.in_sample_rate,
            only_edge=False,
            trail_sil=trailing_silence_ms,
        )
    return prompt_audio


def token_plan(
    tokenizer: Any,
    tools: dict[str, Any],
    prompt_text: str,
    target_text: str,
    prompt_duration: float,
    dialogue: bool,
    chunk_target_seconds: float,
) -> tuple[list[int], list[tuple[int, list[int]]]]:
    if RAW_EVALUATION:
        prompt_ids = tokenizer.texts_to_token_ids([prompt_text])[0]
        target_ids = tokenizer.texts_to_token_ids([target_text])[0]
        return prompt_ids, [(0, target_ids)]

    prompt_text = tools["add_punctuation"](prompt_text)
    target_text = tools["add_punctuation"](target_text)
    prompt_tokens = tokenizer.texts_to_tokens([prompt_text])[0]
    target_tokens = tokenizer.texts_to_tokens([target_text])[0]
    token_duration = prompt_duration / (len(prompt_tokens) * SPEED)
    max_tokens = int((chunk_target_seconds - prompt_duration) / token_duration)
    chunker = tools["chunk_dialog"] if dialogue else tools["chunk_punctuation"]
    token_chunks = chunker(target_tokens, max_tokens=max_tokens)
    target_chunks = tokenizer.tokens_to_token_ids(token_chunks)
    prompt_ids = tokenizer.tokens_to_token_ids([prompt_tokens])[0]
    batches, indices = tools["batchify_tokens"](
        target_chunks,
        MAX_BATCH_DURATION,
        prompt_duration,
        token_duration,
    )
    execution_order = [tokens for batch in batches for tokens in batch]
    return prompt_ids, list(zip(indices, execution_order))


def synthesize_item(
    runtime: ZipVoiceRuntime,
    tokenizer: Any,
    tools: dict[str, Any],
    prompt_text: str,
    prompt_paths: tuple[Path, ...],
    target_text: str,
    output_path: Path,
    num_step: int,
    guidance_scale: float,
    t_shift: float,
) -> tuple[float, float]:
    metadata = runtime.metadata
    prompt_audio = prepare_prompt_audio(
        prompt_paths,
        runtime,
        tools,
        int(metadata["prompt_trailing_silence_ms"]),
    )
    prompt_duration = prompt_audio.shape[-1] / runtime.in_sample_rate
    prompt_ids, chunks = token_plan(
        tokenizer,
        tools,
        prompt_text,
        target_text,
        prompt_duration,
        runtime.dialogue,
        float(metadata["chunk_target_seconds"]),
    )
    context = runtime.create_context(
        prompt_audio,
        prompt_ids,
        SPEED,
        num_step,
        guidance_scale,
        t_shift,
    )

    inference_seconds = 0.0
    chunk_waveforms = []
    for original_index, target_ids in chunks:
        waveform, elapsed = runtime.synthesize_chunk(context, target_ids)
        chunk_waveforms.append((original_index, waveform))
        inference_seconds += elapsed

    ordered = [
        waveform
        for _, waveform in sorted(chunk_waveforms, key=lambda item: item[0])
    ]
    if RAW_EVALUATION:
        final_audio = ordered[0]
    else:
        final_audio = tools["crossfade"](
            ordered,
            fade_duration=float(metadata["crossfade_seconds"]),
            sample_rate=runtime.out_sample_rate,
        )
        final_audio = tools["remove_silence"](
            final_audio,
            runtime.out_sample_rate,
            only_edge=not REMOVE_LONG_SILENCE,
            trail_sil=0,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio(output_path, final_audio, runtime.out_sample_rate)
    audio_seconds = final_audio.shape[-1] / runtime.out_sample_rate
    print(
        f"Saved {output_path} | pipeline={inference_seconds:.3f}s "
        f"RTF={inference_seconds / audio_seconds:.3f}"
    )
    return inference_seconds, audio_seconds


def main() -> None:
    args = parse_args()
    onnx_folder = args.onnx_folder.expanduser().resolve()
    package_stem, pipeline_path, metadata_path = discover_package(onnx_folder)

    ort.set_seed(RANDOM_SEED)
    runtime = ZipVoiceRuntime(pipeline_path, metadata_path)
    metadata = runtime.metadata
    metadata_tokenizer = metadata["tokenizer_type"]
    tokenizer_type = args.tokenizer or metadata_tokenizer
    if tokenizer_type != metadata_tokenizer:
        raise ValueError(
            f"Tokenizer {tokenizer_type!r} does not match package metadata "
            f"{metadata_tokenizer!r}."
        )
    task_mode = args.task_mode or ("dialogue" if runtime.dialogue else "single")
    dialogue = task_mode == "dialogue"
    if dialogue != runtime.dialogue:
        expected_mode = "dialogue" if runtime.dialogue else "single"
        raise ValueError(
            f"Task mode {task_mode!r} does not match package metadata; "
            f"expected {expected_mode!r}."
        )
    tokens_path = args.tokens or (onnx_folder / "tokens.txt")
    if args.tokens is None and not tokens_path.is_file():
        tokens_path = (
            Path.home()
            / "Downloads"
            / "ZipVoice"
            / metadata["model_name"]
            / "tokens.txt"
        )
    num_step = NUM_STEP if NUM_STEP is not None else int(metadata["default_num_step"])
    guidance_scale = (
        GUIDANCE_SCALE
        if GUIDANCE_SCALE is not None
        else float(metadata["default_guidance_scale"])
    )
    t_shift = T_SHIFT if T_SHIFT is not None else float(metadata["default_t_shift"])
    tokenizer, tools = load_tools(tokenizer_type, tokens_path)

    print(
        f"Package={package_stem} model={metadata['model_name']} "
        f"provider={ORT_PROVIDERS[0]} tokenizer={tokenizer_type} "
        f"task={task_mode} channels={runtime.output_channels} "
        f"steps={num_step} guidance={guidance_scale} t_shift={t_shift}"
    )
    total_inference = 0.0
    total_audio = 0.0
    default_output_path = SCRIPT_DIR / f"{metadata['model_name']}_output.wav"
    for _, prompt_text, prompt_paths, target_text, output_path in input_rows(
        dialogue,
        default_output_path,
    ):
        if dialogue and not target_text.startswith("[S1]"):
            raise ValueError("Dialogue target text must start with [S1].")
        inference_seconds, audio_seconds = synthesize_item(
            runtime,
            tokenizer,
            tools,
            prompt_text,
            prompt_paths,
            target_text,
            output_path,
            num_step,
            guidance_scale,
            t_shift,
        )
        total_inference += inference_seconds
        total_audio += audio_seconds
    print(f"Overall RTF={total_inference / total_audio:.3f}")


if __name__ == "__main__":
    main()
