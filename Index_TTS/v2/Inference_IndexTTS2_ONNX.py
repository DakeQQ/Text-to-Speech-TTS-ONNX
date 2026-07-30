"""Run the compact IndexTTS2 ONNX pipeline.

Raw-audio feature extraction, neural forwards, and each fused Euler update run
in ONNX Runtime. Audio I/O, text normalization/tokenization, random diffusion
initialization, and loop control remain in Python to match official inference.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_TTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = INDEX_TTS_DIR.parent
for import_path in (REPO_ROOT, INDEX_TTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Index_TTS.v2.Shared_Weights import (  # noqa: E402
    attach_shared_initializers,
)
from Example_Audio import reference_audio_path  # noqa: E402


DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "IndexTTS2_Optimized",
        help="Folder containing exported or optimized IndexTTS2 graphs.",
    )
    return parser.parse_args()


ARGS = parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()

# =============================================================================
# 用户配置 / User configuration
# =============================================================================
# 可以同时启用多个模式。每个启用的模式分别生成一个 WAV，参考音频只预处理一次。
# Multiple modes may be enabled together. Each enabled mode writes one WAV while
# sharing a single reference-audio preprocessing pass.
#
# RUN_* = True：运行该模式。RUN_* = False：完全跳过该模式，其余配置不生效。
# RUN_* = True runs that mode. RUN_* = False skips it and ignores its settings.
#
# 下文的“忽略/跳过”只针对当前模式的输出。其他设为 True 的模式仍会独立运行。
# “Ignored/skipped” below applies only to that mode's output. Other enabled modes
# still run independently. To run only one mode, set the other three switches False.
#
# 默认配置会运行全部四种模式并生成四个 WAV。
# The default configuration runs all four modes and generates four WAV files.

# 公共路径与声音来源 / Shared paths and voice source ---------------------------
PROJECT_PATH = Path.home() / "Downloads" / "index-tts-main"
MODEL_DIR = Path.home() / "Downloads" / "IndexTTS-2"
TOKENIZER_PATH = MODEL_DIR / "bpe.model"
REFERENCE_AUDIO_PATH = Path(reference_audio_path("indextts"))
MAX_REFERENCE_SECONDS = 15.0

# 模式 1：普通声音克隆 / Mode 1: Normal voice clone ---------------------------
# 只使用 REFERENCE_AUDIO_PATH 的音色和原始情绪。/ Uses only the timbre and original emotion from REFERENCE_AUDIO_PATH.
# 此输出跳过 Qwen，并忽略情绪参考音频、手动情绪向量及其强度。 / This output skips Qwen and ignores emotion-reference audio and manual vectors.
RUN_NORMAL_VOICE_CLONE = True
NORMAL_TARGET_TEXT = "大家好，我现在正在大可奇奇体验 AI 科技。"
NORMAL_OUTPUT_PATH = SCRIPT_DIR / "generated_v2.wav"

# 模式 2：情绪参考音频 / Mode 2: Emotion-reference audio ----------------------
# 音色来自 REFERENCE_AUDIO_PATH，情绪来自 EMOTION_AUDIO_REFERENCE_PATH。
# Timbre comes from REFERENCE_AUDIO_PATH; emotion comes from EMOTION_AUDIO_REFERENCE_PATH.
# 
# 此输出跳过 Qwen，并忽略手动情绪向量。/This output skips Qwen and ignores the manual emotion vector.
#
# 仓库没有附带独立情绪音频，因此默认复用普通参考音频以完整演示执行流程。这样可以验证模式 2，但不会产生独立的情绪迁移效果。实际使用时请替换为真实的情绪音频，例如：Path("/path/to/emo_sad.wav")。
# The repository has no separate emotion clip, so the default reuses the normal reference to exercise the complete Mode 2 path. This validates the workflow but does not provide independent emotion transfer. Replace it with a real emotion clip.
RUN_EMOTION_AUDIO_VOICE_CLONE = True
EMOTION_AUDIO_REFERENCE_PATH: Path | None = REFERENCE_AUDIO_PATH
EMOTION_AUDIO_ALPHA = 1.0
EMOTION_AUDIO_TARGET_TEXT = "酒楼丧尽天良，开始借机竞拍房间，唉，一群蠢货。"
EMOTION_AUDIO_OUTPUT_PATH = SCRIPT_DIR / "generated_v2_emotion_audio.wav"

# 模式 3：Qwen 文本情绪 -------------------------------------------------------
# QWEN_TARGET_TEXT 是最终合成并朗读的文本。
# QWEN_EMOTION_PROMPT 只用于让 Qwen 提取八维情绪向量，其内容不会被朗读：
#   - 设为 None：直接分析 QWEN_TARGET_TEXT 本身的情绪。
#   - 设为非空字符串：分析单独提供的情绪描述，再将得到的情绪作用于 QWEN_TARGET_TEXT。下方默认值演示了这种用法。
# QWEN_EMOTION_ALPHA 控制提取出的情绪向量对合成结果的作用强度。此模式不使用情绪参考音频，并忽略 MANUAL_EMOTION_VECTOR。
# 设为 False 时，不会加载 Qwen tokenizer 和两个 Qwen ONNX Session，本区块中的其他配置也不会生效。
#
# QWEN_TARGET_TEXT is the text that will be synthesized and spoken.
# QWEN_EMOTION_PROMPT is used only by Qwen to extract an eight-value emotion
# vector; its content will not be spoken:
#   - Set it to None to analyze the emotion of QWEN_TARGET_TEXT directly.
#   - Set it to a non-empty string to analyze a separate emotion description and apply the resulting emotion to QWEN_TARGET_TEXT. The default value below demonstrates this usage.
# QWEN_EMOTION_ALPHA controls how strongly the extracted emotion vector affects synthesis. 
# This mode does not use emotion-reference audio and ignores MANUAL_EMOTION_VECTOR.
# When set to False, the Qwen tokenizer and two Qwen ONNX sessions are not loaded, and the other settings in this section have no effect.
RUN_QWEN_TEXT_EMOTION_VOICE_CLONE = True
QWEN_TOKENIZER_PATH = MODEL_DIR / "qwen0.6bemo4-merge"
QWEN_TARGET_TEXT = "快躲起來！是他要來了！他要來抓我們了！"
QWEN_EMOTION_PROMPT: str | None = "极度恐惧、慌张，并带有强烈的惊讶感。"
QWEN_EMOTION_ALPHA = 0.6
QWEN_MAX_NEW_TOKENS = 512
QWEN_OUTPUT_PATH = SCRIPT_DIR / "generated_v2_emotion.wav"

# 模式 4：手动情绪向量 / Mode 4: Manual emotion vector ------------------------
# 向量顺序：[高兴, 愤怒, 悲伤, 恐惧, 反感, 低落, 惊讶, 自然]。
# Vector order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
# 此模式直接使用 MANUAL_EMOTION_VECTOR；跳过 Qwen，也不读取情绪参考音频。
# This mode directly uses MANUAL_EMOTION_VECTOR, skips Qwen, and ignores emotion-reference audio.
RUN_MANUAL_EMOTION_VECTOR_VOICE_CLONE = True
MANUAL_EMOTION_VECTOR = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
MANUAL_EMOTION_ALPHA = 1.0
MANUAL_EMOTION_TARGET_TEXT = "对不起嘛！我的记性真的不太好。"
MANUAL_EMOTION_OUTPUT_PATH = SCRIPT_DIR / "generated_v2_emotion_vector.wav"

# 生成设置 / Generation settings ------------------------------------------------
# 每次只加载一种解码策略。Only one decoding strategy is loaded per process.
# 可选值 / Options: greedy | penalty_greedy | sampling
DECODE_STRATEGY = "sampling"
PENALTY_VALUE = 0.8              # penalty_greedy 的乘法惩罚 / Multiplicative penalty.
PENALTY_RANGE = 20               # penalty_greedy 的近期窗口 / Recent-token window.
TEMPERATURE = 0.8                # 越高变化越大 / Higher values add variation.
TOP_K = 20                       # sampling 候选数量 / Sampling candidate count.
TOP_P = 0.8                      # sampling 核采样阈值 / Nucleus threshold, range (0, 1].
REPETITION_PENALTY = 1.2         # sampling 重复惩罚 / Sampling repetition penalty.
MAX_TOKENS = 1500                # 每段上限；0 使用图容量 / Per-segment limit; 0 uses capacity.
MAX_TEXT_TOKENS_PER_SEGMENT = 120  # 长文本分段长度 / Long-text segment length.

# CFM Euler 步数在导出时固定，并从包元数据读取。/ CFM Euler steps are fixed at export and read from package metadata.
CFG_RATE = 0.7                   # <= 0 使用单 CFM 分支 / <= 0 uses one CFM branch.
DIFFUSION_TEMPERATURE = 1.0      # 声学随机性 / Acoustic variation.
SEED = 9527                      # 整数可复现 / Set an integer for repeatability.
INTERVAL_SILENCE_MS = 200        # 文本段间静音 / Silence between text segments.

# 运行时设置 / Runtime settings -------------------------------------------------
ORT_ACCELERATE_PROVIDERS = []    # e.g. ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider"].
MAX_THREADS = 0                  # CPU/OpenVINO 线程数 / CPU and OpenVINO thread count.
DEVICE_ID = 0                    # 加速器设备编号 / Accelerator device index.
ORT_LOG = False                  # ONNX Runtime 详细日志 / Verbose ONNX Runtime logging.
VERBOSE = False                  # 打印语义 token / Print generated semantic token IDs.
SHOW_PROGRESS = True             # 打印流程进度 / Print pipeline progress.


def print_progress(message: str) -> None:
    if SHOW_PROGRESS:
        print(f"[IndexTTS2] {message}", flush=True)


def enabled_voice_modes() -> tuple[str, ...]:
    modes = []
    if RUN_NORMAL_VOICE_CLONE:
        modes.append("normal")
    if RUN_EMOTION_AUDIO_VOICE_CLONE:
        modes.append("emotion_audio")
    if RUN_QWEN_TEXT_EMOTION_VOICE_CLONE:
        modes.append("qwen_text_emotion")
    if RUN_MANUAL_EMOTION_VECTOR_VOICE_CLONE:
        modes.append("manual_emotion_vector")
    return tuple(modes)


def metadata_from(path: Path) -> dict[str, str]:
    model = onnx.load(str(path), load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    del model
    expected_keys = {
        "graph_layout",
        "cfm_steps",
        "shared_initializer_model_file",
        "shared_initializer_data_file",
        "in_sample_rate",
        "out_sample_rate",
        "semantic_input_sample_rate",
        "semantic_frame_length",
        "semantic_frame_shift",
        "mel_code_size",
        "stop_mel_token",
        "max_signal_length",
        "use_f16_kv",
        "compute_in_f32",
        "emotion_text_num_layers",
        "emotion_text_max_seq_length",
        "emotion_text_stop_token_ids",
        "emotion_text_prompt_prefix_token_ids",
        "emotion_text_prompt_suffix_token_ids",
        "emotion_text_content_prefix",
        "emotion_text_think_end_token_id",
        "emotion_text_kv_dtype",
        "model_file_name_reference_preprocess",
        "model_file_name_conditioning",
        "model_file_name_synthesis",
        "model_file_name_cfm_estimator",
        "model_file_name_decoder",
        "model_file_name_metadata",
        "model_file_name_emotion_text_prefill",
        "model_file_name_emotion_text_decode",
        *{
            f"model_file_name_target_prefill_{strategy}"
            for strategy in DECODE_STRATEGIES
        },
        *{
            f"model_file_name_decode_step_{strategy}"
            for strategy in DECODE_STRATEGIES
        },
    }
    return metadata


def meta_str(metadata: dict[str, str], key: str, default: str | None = None) -> str:
    value = metadata.get(key, default)
    return value


def meta_int(metadata: dict[str, str], key: str, default: int | None = None) -> int:
    value = metadata.get(key)
    if value is None:
        return default
    return int(value)


@dataclass(frozen=True)
class ModelIO:
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]

    @classmethod
    def from_session(cls, session: ort.InferenceSession) -> "ModelIO":
        return cls(tuple(session.get_inputs()), tuple(session.get_outputs()))


def io_dtype(argument: Any) -> np.dtype:
    match = re.fullmatch(r"tensor\(([^)]+)\)", argument.type)
    element_type = onnx.TensorProto.DataType.Value(match.group(1).upper())
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))


def io_shape(
    argument: Any,
    dynamic_dimensions: tuple[int, ...] = (),
) -> tuple[int, ...]:
    dimensions = iter(dynamic_dimensions)
    shape = []
    for dimension in argument.shape:
        if isinstance(dimension, int) and dimension >= 0:
            shape.append(dimension)
        else:
            try:
                shape.append(next(dimensions))
            except StopIteration as error:
                pass
    try:
        next(dimensions)
    except StopIteration:
        return tuple(shape)
    pass
def repeated_dynamic_dimensions(argument: Any, value: int) -> tuple[int, ...]:
    return tuple(
        value
        for dimension in argument.shape
        if not isinstance(dimension, int) or dimension < 0
    )


def matching_prefix_count(
    outputs: tuple[Any, ...],
    inputs: tuple[Any, ...],
) -> int:
    count = 0
    for output, input_value in zip(outputs, inputs):
        if output.type != input_value.type or len(output.shape) != len(input_value.shape):
            break
        dimensions_match = all(
            (
                isinstance(output_dimension, int)
                and isinstance(input_dimension, int)
                and output_dimension == input_dimension
            )
            or (
                not isinstance(output_dimension, int)
                and not isinstance(input_dimension, int)
            )
            for output_dimension, input_dimension in zip(output.shape, input_value.shape)
        )
        if not dimensions_match:
            break
        count += 1
    return count


def numpy_for(
    argument: Any,
    value: Any,
    dynamic_dimensions: tuple[int, ...] = (),
) -> np.ndarray:
    array = np.asarray(value, dtype=io_dtype(argument))
    shape = io_shape(argument, dynamic_dimensions)
    if array.shape != shape:
        array = np.broadcast_to(array, shape)
    return np.array(array, copy=True, order="C")


def ortvalue_for(
    argument: Any,
    value: Any,
    device_type: str,
    dynamic_dimensions: tuple[int, ...] = (),
) -> ort.OrtValue:
    array = numpy_for(argument, value, dynamic_dimensions)
    device_id = DEVICE_ID if device_type != "cpu" else 0
    return ort.OrtValue.ortvalue_from_numpy(array, device_type, device_id)


def empty_ortvalue_for(
    argument: Any,
    device_type: str,
    dynamic_dimensions: tuple[int, ...] = (),
) -> ort.OrtValue:
    device_id = DEVICE_ID if device_type != "cpu" else 0
    return ort.OrtValue.ortvalue_from_shape_and_type(
        io_shape(argument, dynamic_dimensions),
        io_dtype(argument),
        device_type,
        device_id,
    )


@dataclass(frozen=True)
class RuntimePaths:
    folder: Path
    metadata: Path
    reference_preprocess: Path
    conditioning: Path
    target_prefill: Path
    decode: Path
    synthesis: Path
    cfm_estimator: Path
    decoder: Path
    emotion_text_prefill: Path
    emotion_text_decode: Path
    shared_model: Path
    shared_data: Path

    @classmethod
    def from_metadata(
        cls,
        folder: Path,
        metadata: dict[str, str],
        strategy: str,
    ) -> "RuntimePaths":
        def graph(key: str) -> Path:
            return folder / meta_str(metadata, key)

        paths = cls(
            folder=folder,
            metadata=graph("model_file_name_metadata"),
            reference_preprocess=graph("model_file_name_reference_preprocess"),
            conditioning=graph("model_file_name_conditioning"),
            target_prefill=graph(f"model_file_name_target_prefill_{strategy}"),
            decode=graph(f"model_file_name_decode_step_{strategy}"),
            synthesis=graph("model_file_name_synthesis"),
            cfm_estimator=graph("model_file_name_cfm_estimator"),
            decoder=graph("model_file_name_decoder"),
            emotion_text_prefill=graph("model_file_name_emotion_text_prefill"),
            emotion_text_decode=graph("model_file_name_emotion_text_decode"),
            shared_model=folder / meta_str(metadata, "shared_initializer_model_file"),
            shared_data=folder / meta_str(metadata, "shared_initializer_data_file"),
        )
        missing = [str(path) for path in paths.graphs_with_shared if not path.is_file()]
        return paths

    @property
    def runtime_graphs(self) -> tuple[Path, ...]:
        graphs = (
            self.reference_preprocess,
            self.conditioning,
            self.target_prefill,
            self.decode,
            self.synthesis,
            self.cfm_estimator,
            self.decoder,
        )
        if RUN_QWEN_TEXT_EMOTION_VOICE_CLONE:
            return (*graphs, self.emotion_text_prefill, self.emotion_text_decode)
        return graphs

    @property
    def graphs(self) -> tuple[Path, ...]:
        return (*self.runtime_graphs, self.metadata)

    @property
    def graphs_with_shared(self) -> tuple[Path, ...]:
        return (*self.graphs, self.shared_model, self.shared_data)


def provider_configuration() -> tuple[
    list[str],
    list[dict[str, Any]] | None,
    str,
    Any,
]:
    providers = list(ORT_ACCELERATE_PROVIDERS) or ["CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    unavailable = [provider for provider in providers if provider not in available]
    provider_options = []
    for provider in providers:
        if provider == "CUDAExecutionProvider":
            provider_options.append({"device_id": DEVICE_ID, "use_tf32": "1"})
        elif provider == "DmlExecutionProvider":
            provider_options.append({"device_id": DEVICE_ID})
        elif provider == "OpenVINOExecutionProvider":
            provider_options.append({"device_type": "CPU", "num_of_threads": MAX_THREADS})
        elif provider == "CPUExecutionProvider":
            provider_options.append({})
        else:
            pass
    primary_provider = providers[0]
    if primary_provider == "CUDAExecutionProvider":
        device_type = "cuda"
        raw_device = C.OrtDevice.cuda()
    elif primary_provider == "DmlExecutionProvider":
        device_type = "dml"
        raw_device = C.OrtDevice.dml()
    else:
        device_type = "cpu"
        raw_device = C.OrtDevice.cpu()
    device = C.OrtDevice(raw_device, C.OrtDevice.default_memory(), DEVICE_ID)
    return providers, provider_options, device_type, device


def create_session_options(
    metadata: dict[str, str],
) -> tuple[ort.SessionOptions, ort.RunOptions, list[str] | None]:
    options = ort.SessionOptions()
    run_options = ort.RunOptions()
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 0 if ORT_LOG else 3
    for key, value in (
        ("session.set_denormal_as_zero", "1"),
        ("session.intra_op.allow_spinning", "1"),
        ("session.inter_op.allow_spinning", "1"),
        ("session.use_device_allocator_for_initializers", "1"),
    ):
        options.add_session_config_entry(key, value)

    use_f16_kv = meta_str(metadata, "use_f16_kv") == "1"
    compute_in_f32 = meta_str(metadata, "compute_in_f32") == "1"
    disabled_optimizers = None
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if use_f16_kv and not compute_in_f32:
        disabled_optimizers = [
            "CastFloat16Transformer",
            "FuseFp16InitializerToFp32NodeTransformer",
        ]
        options.add_session_config_entry(
            "optimization.disable_specified_optimizers",
            ";".join(disabled_optimizers),
        )

    run_options.log_severity_level = 0 if ORT_LOG else 3
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options, run_options, disabled_optimizers


def bind_inputs(
    binding: ort.IOBinding,
    arguments: tuple[Any, ...],
    values: tuple[ort.OrtValue, ...],
) -> None:
    for argument, value in zip(arguments, values, strict=True):
        binding.bind_ortvalue_input(argument.name, value)


def bind_outputs(
    binding: ort.IOBinding,
    arguments: tuple[Any, ...],
    device: Any,
    buffers: tuple[ort.OrtValue | None, ...] | None = None,
) -> None:
    if buffers is None:
        buffers = (None,) * len(arguments)
    for argument, buffer in zip(arguments, buffers, strict=True):
        if buffer is None:
            binding._iobinding.bind_output(argument.name, device)
        else:
            binding.bind_ortvalue_output(argument.name, buffer)


def rebind_outputs(
    binding: ort.IOBinding,
    arguments: tuple[Any, ...],
    device: Any,
    buffers: tuple[ort.OrtValue | None, ...] | None = None,
) -> None:
    binding.clear_binding_outputs()
    bind_outputs(binding, arguments, device, buffers)


def run_binding(
    session: ort.InferenceSession,
    binding: ort.IOBinding,
    run_options: ort.RunOptions,
) -> list[ort.OrtValue]:
    run_bound(session, binding, run_options)
    return list(binding.get_outputs())


def run_bound(
    session: ort.InferenceSession,
    binding: ort.IOBinding,
    run_options: ort.RunOptions,
) -> None:
    session.run_with_iobinding(binding, run_options=run_options)


class RuntimeSessions:
    def __init__(
        self,
        paths: RuntimePaths,
        metadata: dict[str, str],
    ) -> None:
        options, self.run_options, disabled_optimizers = create_session_options(metadata)
        providers, provider_options, self.device_type, self.device = provider_configuration()
        self.host_device = C.OrtDevice(
            C.OrtDevice.cpu(),
            C.OrtDevice.default_memory(),
            0,
        )
        shared_started = time.perf_counter()
        print_progress("Attaching shared ONNX initializers...")
        self.shared_refs = attach_shared_initializers(options, paths.shared_model)
        print_progress(
            f"Shared ONNX initializers ready in "
            f"{time.perf_counter() - shared_started:.2f}s."
        )

        graph_count = 7 + (2 if RUN_QWEN_TEXT_EMOTION_VOICE_CLONE else 0)
        loaded_graphs = 0

        def load(path: Path) -> ort.InferenceSession:
            nonlocal loaded_graphs
            loaded_graphs += 1
            print_progress(
                f"Loading ONNX graph {loaded_graphs}/{graph_count}: {path.name}"
            )
            return ort.InferenceSession(
                str(path),
                sess_options=options,
                providers=providers,
                provider_options=provider_options,
                disabled_optimizers=disabled_optimizers,
            )

        self.reference = load(paths.reference_preprocess)
        self.conditioning = load(paths.conditioning)
        self.target_prefill = load(paths.target_prefill)
        self.decode = load(paths.decode)
        self.synthesis = load(paths.synthesis)
        self.cfm = load(paths.cfm_estimator)
        self.decoder = load(paths.decoder)
        self.emotion_text_prefill = (
            load(paths.emotion_text_prefill)
            if RUN_QWEN_TEXT_EMOTION_VOICE_CLONE
            else None
        )
        self.emotion_text_decode = (
            load(paths.emotion_text_decode)
            if RUN_QWEN_TEXT_EMOTION_VOICE_CLONE
            else None
        )
        self.reference_io = ModelIO.from_session(self.reference)
        self.conditioning_io = ModelIO.from_session(self.conditioning)
        self.target_prefill_io = ModelIO.from_session(self.target_prefill)
        self.decode_io = ModelIO.from_session(self.decode)
        self.synthesis_io = ModelIO.from_session(self.synthesis)
        self.cfm_io = ModelIO.from_session(self.cfm)
        self.decoder_io = ModelIO.from_session(self.decoder)
        self.emotion_text_prefill_io = (
            ModelIO.from_session(self.emotion_text_prefill)
            if self.emotion_text_prefill is not None
            else None
        )
        self.emotion_text_decode_io = (
            ModelIO.from_session(self.emotion_text_decode)
            if self.emotion_text_decode is not None
            else None
        )
        self.state_count = matching_prefix_count(
            self.target_prefill_io.outputs,
            self.decode_io.inputs,
        )
        self.cfm_steps = meta_int(metadata, "cfm_steps")
        self.reference_bindings = (
            self.reference.io_binding(),
            self.reference.io_binding(),
        )
        self.conditioning_binding = self.conditioning.io_binding()
        self.target_prefill_binding = self.target_prefill.io_binding()
        self.decode_bindings = (self.decode.io_binding(), self.decode.io_binding())
        self.synthesis_binding = self.synthesis.io_binding()
        self.cfm_bindings = (self.cfm.io_binding(), self.cfm.io_binding())
        self.decoder_binding = self.decoder.io_binding()
        self.style_buffer = empty_ortvalue_for(
            self.synthesis_io.inputs[6],
            self.device_type,
        )
        self.null_hidden_buffer = empty_ortvalue_for(
            self.synthesis_io.inputs[8],
            self.device_type,
        )
        self.speaker_latent_buffer = empty_ortvalue_for(
            self.target_prefill_io.inputs[0],
            self.device_type,
        )
        self.emotion_vector_buffer = empty_ortvalue_for(
            self.target_prefill_io.inputs[1],
            self.device_type,
        )
        token_batch = io_shape(self.target_prefill_io.outputs[-1])[0]
        self.token_buffers = (
            empty_ortvalue_for(
                self.target_prefill_io.outputs[-2],
                self.device_type,
                repeated_dynamic_dimensions(
                    self.target_prefill_io.outputs[-2],
                    token_batch,
                ),
            ),
            empty_ortvalue_for(
                self.decode_io.outputs[-3],
                self.device_type,
                repeated_dynamic_dimensions(
                    self.decode_io.outputs[-3],
                    token_batch,
                ),
            ),
        )
        self.history_buffers = (
            empty_ortvalue_for(
                self.target_prefill_io.outputs[-1],
                self.device_type,
            ),
            empty_ortvalue_for(
                self.decode_io.outputs[-1],
                self.device_type,
            ),
        )
        self.target_length_buffer = empty_ortvalue_for(
            self.synthesis_io.outputs[1],
            self.device_type,
        )
        self.cfg_scales_buffer = empty_ortvalue_for(
            self.synthesis_io.outputs[2],
            self.device_type,
        )
        self.cfg_scale_sum_buffer = empty_ortvalue_for(
            self.synthesis_io.outputs[3],
            self.device_type,
        )
        self.no_cfg_scales = ortvalue_for(
            self.cfm_io.inputs[3],
            1,
            self.device_type,
            (1,),
        )
        self.no_cfg_scale_sum = ortvalue_for(
            self.cfm_io.inputs[4],
            1,
            self.device_type,
        )
        self.step_index_array = numpy_for(self.cfm_io.inputs[1], 0)
        self.step_index_value = ortvalue_for(
            self.cfm_io.inputs[1],
            self.step_index_array,
            self.device_type,
        )
        self.accepted_length_array = numpy_for(self.synthesis_io.inputs[4], 0)
        self.accepted_length_value = ortvalue_for(
            self.synthesis_io.inputs[4],
            self.accepted_length_array,
            self.device_type,
        )
        self.control_values: tuple[ort.OrtValue, ...] = ()
        self.cfm_shape: tuple[int, ...] | None = None
        self.cfm_mel_buffers: tuple[ort.OrtValue, ort.OrtValue] | None = None
        self.cfm_noise: np.ndarray | None = None
        self.cfm_initial: np.ndarray | None = None
        self.reference_hidden_value: ort.OrtValue | None = None

        bind_inputs(
            self.conditioning_binding,
            (self.conditioning_io.inputs[5],),
            (self.style_buffer,),
        )
        bind_inputs(
            self.target_prefill_binding,
            self.target_prefill_io.inputs[:2],
            (self.speaker_latent_buffer, self.emotion_vector_buffer),
        )
        bind_inputs(
            self.synthesis_binding,
            (
                *self.synthesis_io.inputs[:2],
                self.synthesis_io.inputs[4],
                self.synthesis_io.inputs[6],
                self.synthesis_io.inputs[8],
            ),
            (
                self.speaker_latent_buffer,
                self.emotion_vector_buffer,
                self.accepted_length_value,
                self.style_buffer,
                self.null_hidden_buffer,
            ),
        )
        for binding in self.cfm_bindings:
            bind_inputs(
                binding,
                (self.cfm_io.inputs[1],),
                (self.step_index_value,),
            )

        bind_outputs(
            self.reference_bindings[0],
            self.reference_io.outputs,
            self.device,
            (None, self.style_buffer, None, self.null_hidden_buffer),
        )
        bind_outputs(
            self.reference_bindings[1],
            self.reference_io.outputs[:1],
            self.device,
        )
        bind_outputs(
            self.conditioning_binding,
            self.conditioning_io.outputs,
            self.device,
            (self.speaker_latent_buffer, self.emotion_vector_buffer),
        )

        self.emotion_text_prefill_binding = None
        self.emotion_text_decode_bindings: tuple[ort.IOBinding, ...] = ()
        self.emotion_token_buffers: tuple[ort.OrtValue, ...] = ()
        self.emotion_history_buffers: tuple[ort.OrtValue, ...] = ()
        self.emotion_state_count = 0
        if (
            self.emotion_text_prefill is not None
            and self.emotion_text_decode is not None
            and self.emotion_text_prefill_io is not None
            and self.emotion_text_decode_io is not None
        ):
            self.emotion_state_count = matching_prefix_count(
                self.emotion_text_prefill_io.outputs,
                self.emotion_text_decode_io.inputs,
            )
            emotion_layers = meta_int(metadata, "emotion_text_num_layers")
            expected_state_count = 2 * emotion_layers
            cache_dtype = meta_str(
                metadata,
                "emotion_text_kv_dtype",
                "float16",
            )
            cache_element_type = {
                "float16": "float16",
                "float32": "float",
            }.get(cache_dtype)
            expected_types = [
                f"tensor({cache_element_type})"
            ] * expected_state_count
            actual_types = [
                output.type
                for output in self.emotion_text_prefill_io.outputs[
                    : self.emotion_state_count
                ]
            ]
            self.emotion_text_prefill_binding = self.emotion_text_prefill.io_binding()
            self.emotion_text_decode_bindings = (
                self.emotion_text_decode.io_binding(),
                self.emotion_text_decode.io_binding(),
            )
            emotion_batch = io_shape(self.emotion_text_prefill_io.outputs[-1])[0]
            self.emotion_token_buffers = (
                empty_ortvalue_for(
                    self.emotion_text_prefill_io.outputs[-2],
                    self.device_type,
                    repeated_dynamic_dimensions(
                        self.emotion_text_prefill_io.outputs[-2],
                        emotion_batch,
                    ),
                ),
                empty_ortvalue_for(
                    self.emotion_text_decode_io.outputs[-2],
                    self.device_type,
                    repeated_dynamic_dimensions(
                        self.emotion_text_decode_io.outputs[-2],
                        emotion_batch,
                    ),
                ),
            )
            self.emotion_history_buffers = (
                empty_ortvalue_for(
                    self.emotion_text_prefill_io.outputs[-1],
                    self.device_type,
                ),
                empty_ortvalue_for(
                    self.emotion_text_decode_io.outputs[-1],
                    self.device_type,
                ),
            )

    def prepare_cfm(
        self,
        total_frames: int,
    ) -> tuple[
        tuple[ort.OrtValue, ort.OrtValue],
        np.ndarray,
        np.ndarray,
    ]:
        shape = io_shape(self.cfm_io.inputs[0], (total_frames,))
        if shape != self.cfm_shape:
            self.cfm_mel_buffers = (
                empty_ortvalue_for(
                    self.cfm_io.inputs[0],
                    self.device_type,
                    (total_frames,),
                ),
                empty_ortvalue_for(
                    self.cfm_io.outputs[0],
                    self.device_type,
                    (total_frames,),
                ),
            )
            dtype = io_dtype(self.cfm_io.inputs[0])
            self.cfm_noise = np.empty(
                (shape[0], shape[2], shape[1]),
                dtype=dtype,
            )
            self.cfm_initial = np.empty(shape, dtype=dtype)
            for index, binding in enumerate(self.cfm_bindings):
                bind_inputs(
                    binding,
                    (self.cfm_io.inputs[0],),
                    (self.cfm_mel_buffers[index],),
                )
                rebind_outputs(
                    binding,
                    self.cfm_io.outputs,
                    self.device,
                    (self.cfm_mel_buffers[1 - index],),
                )
            self.cfm_shape = shape
        pass
        pass
        pass
        return self.cfm_mel_buffers, self.cfm_noise, self.cfm_initial


@dataclass(frozen=True)
class Frontend:
    tokenizer: Any
    emotion_tokenizer: Any | None


@dataclass(frozen=True)
class ReferenceFeatures:
    speaker_features: ort.OrtValue
    speaker_length: ort.OrtValue
    self_emotion_length: ort.OrtValue
    reference_hidden: ort.OrtValue
    prompt_frames: int
    input_sample_rate: int
    minimum_audio_samples: int


@dataclass(frozen=True)
class GenerationControls:
    penalty_value: float
    penalty_range: int
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float


def resolve_frontend() -> Frontend:
    project_path = PROJECT_PATH.expanduser().resolve()
    model_dir = MODEL_DIR.expanduser().resolve()
    tokenizer_path = TOKENIZER_PATH.expanduser().resolve()
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))

    from indextts.utils.front import TextNormalizer, TextTokenizer

    normalizer = TextNormalizer(enable_glossary=True)
    try:
        normalizer.load()
        glossary_path = model_dir / "glossary.yaml"
        if glossary_path.is_file():
            normalizer.load_glossary_from_yaml(str(glossary_path))
    except (ImportError, ModuleNotFoundError) as exc:
        warnings.warn(
            f"Official text normalization is unavailable ({exc}); using SentencePiece without normalization.",
            RuntimeWarning,
        )
        normalizer = None
    tokenizer = TextTokenizer(str(tokenizer_path), normalizer)
    emotion_tokenizer = None
    if RUN_QWEN_TEXT_EMOTION_VOICE_CLONE:
        qwen_tokenizer_oath = (
            QWEN_TOKENIZER_PATH or model_dir / "qwen0.6bemo4-merge"
        ).expanduser().resolve()
        from transformers import AutoTokenizer

        emotion_tokenizer = AutoTokenizer.from_pretrained(
            qwen_tokenizer_oath,
            local_files_only=True,
        )
    return Frontend(
        tokenizer=tokenizer,
        emotion_tokenizer=emotion_tokenizer,
    )


def generation_controls(
    metadata: dict[str, str],
) -> GenerationControls:
    controls = GenerationControls(
        penalty_value=PENALTY_VALUE,
        penalty_range=PENALTY_RANGE,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )
    return GenerationControls(
        **{
            **controls.__dict__,
            "top_k": min(controls.top_k, meta_int(metadata, "mel_code_size")),
        }
    )


def load_audio(
    path: Path,
    sample_rate: int,
    max_seconds: float,
    min_samples: int,
    argument: Any,
) -> np.ndarray:
    path = path.expanduser().resolve()
    segment = (
        AudioSegment.from_file(path)
        .set_channels(1)
        .set_frame_rate(sample_rate)
        .set_sample_width(2)
    )
    audio = np.asarray(
        segment.get_array_of_samples(),
        dtype=np.int16,
    )
    max_samples = int(max_seconds * sample_rate)
    if max_samples > 0:
        audio = audio[:max_samples]
    if np.issubdtype(io_dtype(argument), np.floating):
        audio = (audio.astype(np.float32) * (1.0 / 32768.0)).astype(
            io_dtype(argument)
        )
    else:
        audio = audio.astype(io_dtype(argument), copy=False)
    return np.ascontiguousarray(audio)


def parse_emotion_text_output(content: str, text_input: str) -> dict[str, float]:
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        parsed = {
            match.group(1): float(match.group(2))
            for match in re.finditer(
                r'([^\s":.,]+?)"?\s*:\s*([\d.]+)',
                content,
            )
        }

    melancholic_words = {
        "低落",
        "melancholy",
        "melancholic",
        "depression",
        "depressed",
        "gloomy",
    }
    if any(word in text_input.lower() for word in melancholic_words):
        parsed["悲伤"], parsed["低落"] = (
            parsed.get("低落", 0.0),
            parsed.get("悲伤", 0.0),
        )

    key_map = (
        ("高兴", "happy"),
        ("愤怒", "angry"),
        ("悲伤", "sad"),
        ("恐惧", "afraid"),
        ("反感", "disgusted"),
        ("低落", "melancholic"),
        ("惊讶", "surprised"),
        ("自然", "calm"),
    )
    detected: dict[str, float] = {}
    for source, target in key_map:
        try:
            score = float(parsed.get(source, 0.0))
        except (TypeError, ValueError) as error:
            pass
        detected[target] = min(max(score, 0.0), 1.2)
    if not any(score > 0.0 for score in detected.values()):
        detected["calm"] = 1.0
    return detected


def run_emotion_text_model(
    sessions: RuntimeSessions,
    tokenizer: Any,
    metadata: dict[str, str],
    text_input: str,
) -> str:
    content_ids = tokenizer(
        [meta_str(metadata, "emotion_text_content_prefix") + text_input],
        add_special_tokens=False,
        return_tensors="np",
    )["input_ids"].reshape(-1)
    input_ids = np.concatenate(
        (
            np.asarray(
                [
                    int(token)
                    for token in meta_str(
                        metadata,
                        "emotion_text_prompt_prefix_token_ids",
                    ).split(",")
                    if token
                ]
            ),
            content_ids,
            np.asarray(
                [
                    int(token)
                    for token in meta_str(
                        metadata,
                        "emotion_text_prompt_suffix_token_ids",
                    ).split(",")
                    if token
                ]
            ),
        )
    ).reshape(1, -1)
    max_sequence_length = meta_int(metadata, "emotion_text_max_seq_length")
    generation_limit = min(
        QWEN_MAX_NEW_TOKENS,
        max_sequence_length - input_ids.shape[1],
    )
    print_progress(
        f"Running emotion-text analysis (up to {generation_limit} new tokens)..."
    )

    input_ids_value = ortvalue_for(
        sessions.emotion_text_prefill_io.inputs[0],
        input_ids,
        sessions.device_type,
        (input_ids.shape[1],),
    )
    bind_inputs(
        sessions.emotion_text_prefill_binding,
        sessions.emotion_text_prefill_io.inputs,
        (input_ids_value,),
    )
    rebind_outputs(
        sessions.emotion_text_prefill_binding,
        sessions.emotion_text_prefill_io.outputs,
        sessions.device,
        (
            *((None,) * sessions.emotion_state_count),
            sessions.emotion_token_buffers[0],
            sessions.emotion_history_buffers[0],
        ),
    )
    outputs = run_binding(
        sessions.emotion_text_prefill,
        sessions.emotion_text_prefill_binding,
        sessions.run_options,
    )
    state_count = sessions.emotion_state_count
    states = outputs[:state_count]
    current_token = sessions.emotion_token_buffers[0]
    history_length = sessions.emotion_history_buffers[0]
    stop_tokens = {
        int(token)
        for token in meta_str(metadata, "emotion_text_stop_token_ids").split(",")
        if token
    }
    generated: list[int] = []
    decode_calls = 0
    while len(generated) < generation_limit:
        token = int(current_token.numpy().item())
        if token in stop_tokens:
            break
        generated.append(token)
        if len(generated) % 32 == 0:
            print_progress(
                f"Emotion-text generation: {len(generated)}/{generation_limit} tokens"
            )
        if len(generated) >= generation_limit:
            break

        binding_index = decode_calls & 1
        binding = sessions.emotion_text_decode_bindings[binding_index]
        bind_inputs(
            binding,
            sessions.emotion_text_decode_io.inputs,
            (*states, current_token, history_length),
        )
        rebind_outputs(
            binding,
            sessions.emotion_text_decode_io.outputs,
            sessions.device,
            (
                *((None,) * state_count),
                sessions.emotion_token_buffers[1 - binding_index],
                sessions.emotion_history_buffers[1 - binding_index],
            ),
        )
        outputs = run_binding(
            sessions.emotion_text_decode,
            binding,
            sessions.run_options,
        )
        states = outputs[:state_count]
        current_token = sessions.emotion_token_buffers[1 - binding_index]
        history_length = sessions.emotion_history_buffers[1 - binding_index]
        decode_calls += 1

    think_end_id = meta_int(metadata, "emotion_text_think_end_token_id")
    if think_end_id in generated:
        generated = generated[
            len(generated) - generated[::-1].index(think_end_id) :
        ]
    print_progress(f"Emotion-text analysis complete: {len(generated)} tokens.")
    return tokenizer.decode(generated, skip_special_tokens=True)


def qwen_emotion_vector(
    sessions: RuntimeSessions,
    frontend: Frontend,
    metadata: dict[str, str],
    target_text: str,
) -> list[float]:
    emotion_text = (
        target_text if QWEN_EMOTION_PROMPT is None else QWEN_EMOTION_PROMPT
    )
    content = run_emotion_text_model(
        sessions,
        frontend.emotion_tokenizer,
        metadata,
        emotion_text,
    )
    detected = parse_emotion_text_output(content, emotion_text)
    print(f"Detected emotion vector: {detected}")
    return [float(value) for value in detected.values()]


def prepare_reference(
    sessions: RuntimeSessions,
    metadata: dict[str, str],
) -> ReferenceFeatures:
    input_sample_rate = meta_int(metadata, "in_sample_rate")
    semantic_sample_rate = meta_int(metadata, "semantic_input_sample_rate")
    semantic_frame_length = meta_int(metadata, "semantic_frame_length", 400)
    semantic_frame_shift = meta_int(metadata, "semantic_frame_shift", 160)
    minimum_semantic_samples = semantic_frame_length + semantic_frame_shift
    minimum_audio_samples = (
        (minimum_semantic_samples - 1) * input_sample_rate // semantic_sample_rate
    ) + 1
    speaker_audio = load_audio(
        REFERENCE_AUDIO_PATH,
        input_sample_rate,
        MAX_REFERENCE_SECONDS,
        minimum_audio_samples,
        sessions.reference_io.inputs[0],
    )
    speaker_audio_value = ortvalue_for(
        sessions.reference_io.inputs[0],
        speaker_audio,
        sessions.device_type,
        (speaker_audio.size,),
    )
    bind_inputs(
        sessions.reference_bindings[0],
        sessions.reference_io.inputs,
        (speaker_audio_value,),
    )
    speaker_features, _, reference_hidden, _ = run_binding(
        sessions.reference,
        sessions.reference_bindings[0],
        sessions.run_options,
    )
    speaker_length = ortvalue_for(
        sessions.conditioning_io.inputs[1],
        speaker_features.shape()[1],
        sessions.device_type,
    )
    self_emotion_length = ortvalue_for(
        sessions.conditioning_io.inputs[3],
        speaker_features.shape()[1],
        sessions.device_type,
    )
    bind_inputs(
        sessions.synthesis_binding,
        (sessions.synthesis_io.inputs[7],),
        (reference_hidden,),
    )
    sessions.reference_hidden_value = reference_hidden
    return ReferenceFeatures(
        speaker_features=speaker_features,
        speaker_length=speaker_length,
        self_emotion_length=self_emotion_length,
        reference_hidden=reference_hidden,
        prompt_frames=reference_hidden.shape()[1],
        input_sample_rate=input_sample_rate,
        minimum_audio_samples=minimum_audio_samples,
    )


def prepare_conditioning(
    sessions: RuntimeSessions,
    reference: ReferenceFeatures,
    vector: list[float] | None,
    emotion_audio_path: Path | None,
    emotion_alpha: float,
) -> None:
    if vector is not None:
        vector_scale = min(max(emotion_alpha, 0.0), 1.0)
        vector = [int(value * vector_scale * 10000) / 10000 for value in vector]
    if emotion_audio_path is None:
        emotion_features = reference.speaker_features
        emotion_lengths_value = reference.self_emotion_length
        emotion_alpha = 1.0
    elif (
        emotion_audio_path.expanduser().resolve()
        == REFERENCE_AUDIO_PATH.expanduser().resolve()
    ):
        emotion_features = reference.speaker_features
        emotion_lengths_value = reference.self_emotion_length
    else:
        emotion_audio = load_audio(
            emotion_audio_path,
            reference.input_sample_rate,
            MAX_REFERENCE_SECONDS,
            reference.minimum_audio_samples,
            sessions.reference_io.inputs[0],
        )
        emotion_audio_value = ortvalue_for(
            sessions.reference_io.inputs[0],
            emotion_audio,
            sessions.device_type,
            (emotion_audio.size,),
        )
        bind_inputs(
            sessions.reference_bindings[1],
            sessions.reference_io.inputs,
            (emotion_audio_value,),
        )
        emotion_features = run_binding(
            sessions.reference,
            sessions.reference_bindings[1],
            sessions.run_options,
        )[0]
        emotion_lengths_value = ortvalue_for(
            sessions.conditioning_io.inputs[3],
            emotion_features.shape()[1],
            sessions.device_type,
        )
    emotion_alpha_value = ortvalue_for(
        sessions.conditioning_io.inputs[4],
        emotion_alpha,
        sessions.device_type,
    )
    emotion_weights_value = ortvalue_for(
        sessions.conditioning_io.inputs[6],
        vector if vector is not None else 0,
        sessions.device_type,
    )
    bind_inputs(
        sessions.conditioning_binding,
        (
            *sessions.conditioning_io.inputs[:5],
            sessions.conditioning_io.inputs[6],
        ),
        (
            reference.speaker_features,
            reference.speaker_length,
            emotion_features,
            emotion_lengths_value,
            emotion_alpha_value,
            emotion_weights_value,
        ),
    )
    run_bound(
        sessions.conditioning,
        sessions.conditioning_binding,
        sessions.run_options,
    )


def bind_generation_controls(
    sessions: RuntimeSessions,
    controls: GenerationControls,
) -> None:
    prefill_arguments = sessions.target_prefill_io.inputs[3:]
    if DECODE_STRATEGY == "sampling":
        prefill_data = (controls.temperature, controls.top_k, controls.top_p)
        prefill_values = tuple(
            ortvalue_for(argument, value, sessions.device_type)
            for argument, value in zip(prefill_arguments, prefill_data, strict=True)
        )
        decode_values = (
            *prefill_values,
            ortvalue_for(
                sessions.decode_io.inputs[-1],
                controls.repetition_penalty,
                sessions.device_type,
            ),
        )
    elif DECODE_STRATEGY == "penalty_greedy":
        decode_data = (controls.penalty_value, controls.penalty_range)
        decode_values = tuple(
            ortvalue_for(argument, value, sessions.device_type)
            for argument, value in zip(
                sessions.decode_io.inputs[-2:],
                decode_data,
                strict=True,
            )
        )
        prefill_values = ()
    else:
        prefill_values = ()
        decode_values = ()

    cfg_rate_value = ortvalue_for(
        sessions.synthesis_io.inputs[5],
        CFG_RATE,
        sessions.device_type,
    )
    sessions.control_values = (*prefill_values, *decode_values, cfg_rate_value)
    bind_inputs(
        sessions.target_prefill_binding,
        prefill_arguments,
        prefill_values,
    )
    bind_inputs(
        sessions.synthesis_binding,
        (sessions.synthesis_io.inputs[5],),
        (cfg_rate_value,),
    )
    cfg_scales = (
        sessions.cfg_scales_buffer
        if CFG_RATE > 0.0
        else sessions.no_cfg_scales
    )
    cfg_scale_sum = (
        sessions.cfg_scale_sum_buffer
        if CFG_RATE > 0.0
        else sessions.no_cfg_scale_sum
    )
    for binding in sessions.cfm_bindings:
        bind_inputs(
            binding,
            sessions.cfm_io.inputs[3:5],
            (cfg_scales, cfg_scale_sum),
        )
    decode_arguments = sessions.decode_io.inputs[
        sessions.state_count + 3 :
    ]
    for binding in sessions.decode_bindings:
        bind_inputs(binding, decode_arguments, decode_values)


def generate_codes(
    sessions: RuntimeSessions,
    metadata: dict[str, str],
    text_ids: ort.OrtValue,
) -> tuple[ort.OrtValue, int]:
    bind_inputs(
        sessions.target_prefill_binding,
        (sessions.target_prefill_io.inputs[2],),
        (text_ids,),
    )
    rebind_outputs(
        sessions.target_prefill_binding,
        (
            *sessions.target_prefill_io.outputs[: sessions.state_count],
            *sessions.target_prefill_io.outputs[-2:],
        ),
        sessions.device,
        (
            *((None,) * sessions.state_count),
            sessions.token_buffers[0],
            sessions.history_buffers[0],
        ),
    )
    prefill_outputs = run_binding(
        sessions.target_prefill,
        sessions.target_prefill_binding,
        sessions.run_options,
    )

    states = prefill_outputs[: sessions.state_count]
    current_token = sessions.token_buffers[0]
    save_ids = current_token
    history_length = sessions.history_buffers[0]
    prefill_length = int(history_length.numpy().item())
    max_tokens = MAX_TOKENS if MAX_TOKENS else meta_int(metadata, "max_signal_length")
    max_tokens = min(max_tokens, meta_int(metadata, "max_signal_length") - prefill_length)
    generation_started = time.perf_counter()
    print_progress(f"Generating semantic codes (limit {max_tokens})...")

    stop_token = meta_int(metadata, "stop_mel_token")
    accepted_tokens = 0
    decode_calls = 0
    while accepted_tokens < max_tokens:
        token = int(current_token.numpy().flat[0])
        if token == stop_token:
            break
        accepted_tokens += 1
        if accepted_tokens % 50 == 0 or accepted_tokens == max_tokens:
            print_progress(
                f"Semantic generation: {accepted_tokens}/{max_tokens} codes"
            )
        if accepted_tokens >= max_tokens:
            break

        binding_index = decode_calls & 1
        binding = sessions.decode_bindings[binding_index]
        bind_inputs(
            binding,
            sessions.decode_io.inputs[: sessions.state_count + 3],
            (*states, current_token, save_ids, history_length),
        )
        rebind_outputs(
            binding,
            (
                *sessions.decode_io.outputs[: sessions.state_count],
                *sessions.decode_io.outputs[-3:],
            ),
            sessions.device,
            (
                *((None,) * sessions.state_count),
                sessions.token_buffers[1 - binding_index],
                None,
                sessions.history_buffers[1 - binding_index],
            ),
        )
        outputs = run_binding(sessions.decode, binding, sessions.run_options)
        states = outputs[: sessions.state_count]
        save_ids = outputs[sessions.state_count + 1]
        current_token = sessions.token_buffers[1 - binding_index]
        history_length = sessions.history_buffers[1 - binding_index]
        decode_calls += 1
    print_progress(
        f"Semantic generation complete: {accepted_tokens} codes in "
        f"{time.perf_counter() - generation_started:.2f}s."
    )

    sessions.accepted_length_array[0] = accepted_tokens
    sessions.accepted_length_value.update_inplace(sessions.accepted_length_array)
    if VERBOSE:
        generated = save_ids.numpy().reshape(-1)[:accepted_tokens].tolist()
        print(f"Generated semantic codes: {generated}")
    return save_ids, accepted_tokens


def solve_cfm(
    sessions: RuntimeSessions,
    prompt_frames: int,
    static_hidden: ort.OrtValue,
    target_mask: ort.OrtValue,
    temperature: float,
    rng: np.random.Generator,
) -> ort.OrtValue:
    steps = sessions.cfm_steps
    total_frames = static_hidden.shape()[1]
    target_frames = total_frames - prompt_frames
    diffusion_started = time.perf_counter()
    print_progress(
        f"Running CFM diffusion ({steps} steps, {target_frames} target frames)..."
    )
    mel_buffers, noise, initial = sessions.prepare_cfm(total_frames)
    rng.standard_normal(noise.shape, dtype=noise.dtype, out=noise)
    np.multiply(noise, temperature, out=noise)
    np.copyto(initial, noise.transpose(0, 2, 1))
    mel_buffers[0].update_inplace(initial)
    expected_mask_shape = io_shape(sessions.cfm_io.inputs[5], (total_frames,))
    for binding in sessions.cfm_bindings:
        bind_inputs(
            binding,
            (sessions.cfm_io.inputs[2], sessions.cfm_io.inputs[5]),
            (static_hidden, target_mask),
        )

    progress_interval = max(1, steps // 5)
    for index in range(steps):
        sessions.step_index_array[0] = index
        sessions.step_index_value.update_inplace(sessions.step_index_array)
        binding_index = index & 1
        run_bound(
            sessions.cfm,
            sessions.cfm_bindings[binding_index],
            sessions.run_options,
        )
        completed_steps = index + 1
        if completed_steps % progress_interval == 0 or completed_steps == steps:
            print_progress(
                f"CFM diffusion: {completed_steps}/{steps} steps "
                f"({time.perf_counter() - diffusion_started:.2f}s elapsed)"
            )

    return mel_buffers[steps & 1]


def synthesize_segment(
    sessions: RuntimeSessions,
    metadata: dict[str, str],
    prompt_frames: int,
    text_ids: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    text_ids_value = ortvalue_for(
        sessions.target_prefill_io.inputs[2],
        text_ids,
        sessions.device_type,
        (text_ids.shape[1],),
    )
    save_ids, code_count = generate_codes(
        sessions,
        metadata,
        text_ids_value,
    )
    print_progress("Preparing acoustic conditioning...")
    bind_inputs(
        sessions.synthesis_binding,
        sessions.synthesis_io.inputs[2:4],
        (text_ids_value, save_ids),
    )
    rebind_outputs(
        sessions.synthesis_binding,
        sessions.synthesis_io.outputs,
        sessions.device,
        (
            None,
            sessions.target_length_buffer,
            sessions.cfg_scales_buffer,
            sessions.cfg_scale_sum_buffer,
            None,
        ),
    )
    synthesis_outputs = run_binding(
        sessions.synthesis,
        sessions.synthesis_binding,
        sessions.run_options,
    )
    static_hidden = synthesis_outputs[0]
    target_mask = synthesis_outputs[-1]
    target_frames = int(sessions.target_length_buffer.numpy().item())
    expected_total_frames = prompt_frames + target_frames
    mel = solve_cfm(
        sessions,
        prompt_frames,
        static_hidden,
        target_mask,
        DIFFUSION_TEMPERATURE,
        rng,
    )
    print_progress("Decoding waveform...")
    bind_inputs(
        sessions.decoder_binding,
        sessions.decoder_io.inputs,
        (mel, sessions.target_length_buffer),
    )
    rebind_outputs(
        sessions.decoder_binding,
        sessions.decoder_io.outputs,
        sessions.host_device,
    )
    waveform = run_binding(
        sessions.decoder,
        sessions.decoder_binding,
        sessions.run_options,
    )[0].numpy()
    return (
        waveform.reshape(-1).astype(io_dtype(sessions.decoder_io.outputs[0]), copy=False),
        code_count,
        target_frames,
    )


def tokenize_segments(
    frontend: Frontend,
    text: str,
    argument: Any,
) -> list[tuple[str, np.ndarray]]:
    tokens = frontend.tokenizer.tokenize(text)
    segments = frontend.tokenizer.split_segments(
        tokens,
        MAX_TEXT_TOKENS_PER_SEGMENT,
    )
    output = []
    for segment in segments:
        ids = frontend.tokenizer.convert_tokens_to_ids(segment)
        if frontend.tokenizer.unk_token_id in ids:
            warnings.warn("Text segment contains unknown tokenizer IDs.", RuntimeWarning)
        label = "".join(segment).replace("▁", " ").strip()
        output.append((label, numpy_for(argument, [ids], (len(ids),))))
    return output


def run_demo(
    sessions: RuntimeSessions,
    frontend: Frontend,
    metadata: dict[str, str],
    prompt_frames: int,
    target_text: str,
    output_path: Path,
    demo_name: str,
) -> Path:
    print_progress(f"Tokenizing and splitting {demo_name} text...")
    segments = tokenize_segments(
        frontend,
        target_text,
        sessions.target_prefill_io.inputs[2],
    )
    print_progress(f"Prepared {len(segments)} {demo_name} text segment(s).")
    rng = np.random.default_rng(SEED)
    output_sample_rate = meta_int(metadata, "out_sample_rate")
    silence_samples = int(output_sample_rate * INTERVAL_SILENCE_MS / 1000)
    silence = np.zeros(
        silence_samples,
        dtype=io_dtype(sessions.decoder_io.outputs[0]),
    )
    waveforms = []
    started = time.perf_counter()
    for index, (label, text_ids) in enumerate(segments, start=1):
        segment_started = time.perf_counter()
        print_progress(
            f"Starting {demo_name} segment {index}/{len(segments)}: {label!r}"
        )
        waveform, code_count, mel_frames = synthesize_segment(
            sessions,
            metadata,
            prompt_frames,
            text_ids,
            rng,
        )
        if waveforms and silence_samples:
            waveforms.append(silence)
        waveforms.append(waveform)
        print(
            f"{demo_name.title()} segment {index}/{len(segments)}: {label!r}, "
            f"codes={code_count}, mel_frames={mel_frames}, "
            f"audio={waveform.size / output_sample_rate:.2f}s, "
            f"time={time.perf_counter() - segment_started:.2f}s",
            flush=True,
        )
    generated = np.concatenate(waveforms)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print_progress(f"Writing generated audio: {output_path}")
    if generated.dtype == np.float16:
        generated = generated.astype(np.float32)
    output_subtype = "PCM_16" if np.issubdtype(generated.dtype, np.integer) else "FLOAT"
    sf.write(
        str(output_path),
        generated,
        output_sample_rate,
        subtype=output_subtype,
    )
    elapsed = time.perf_counter() - started
    duration = generated.size / output_sample_rate
    print(
        f"Saved {output_path} ({duration:.2f}s audio, {elapsed:.2f}s inference, "
        f"RTF={elapsed / max(duration, 1.0e-9):.3f}).",
        flush=True,
    )
    return output_path


def run_synthesis(
    sessions: RuntimeSessions,
    metadata: dict[str, str],
) -> tuple[Path, ...]:
    print_progress("Loading the text frontend...")
    frontend = resolve_frontend()
    controls = generation_controls(metadata)
    bind_generation_controls(sessions, controls)

    reference_started = time.perf_counter()
    print_progress("Preprocessing shared speaker reference audio...")
    reference = prepare_reference(sessions, metadata)
    print_progress(
        f"Shared reference features ready: {reference.prompt_frames} frames in "
        f"{time.perf_counter() - reference_started:.2f}s."
    )

    outputs: list[Path] = []
    if RUN_NORMAL_VOICE_CLONE:
        print_progress("Preparing normal voice-clone conditioning...")
        prepare_conditioning(sessions, reference, None, None, 1.0)
        outputs.append(
            run_demo(
                sessions,
                frontend,
                metadata,
                reference.prompt_frames,
                NORMAL_TARGET_TEXT,
                NORMAL_OUTPUT_PATH,
                "normal voice clone",
            )
        )

    if RUN_EMOTION_AUDIO_VOICE_CLONE:
        pass
        print_progress("Preparing emotion-audio voice-clone conditioning...")
        prepare_conditioning(
            sessions,
            reference,
            None,
            EMOTION_AUDIO_REFERENCE_PATH,
            float(EMOTION_AUDIO_ALPHA),
        )
        outputs.append(
            run_demo(
                sessions,
                frontend,
                metadata,
                reference.prompt_frames,
                EMOTION_AUDIO_TARGET_TEXT,
                EMOTION_AUDIO_OUTPUT_PATH,
                "emotion-audio voice clone",
            )
        )

    if RUN_QWEN_TEXT_EMOTION_VOICE_CLONE:
        print_progress("Preparing Qwen text-emotion conditioning...")
        emotion_vector = qwen_emotion_vector(
            sessions,
            frontend,
            metadata,
            QWEN_TARGET_TEXT,
        )
        prepare_conditioning(
            sessions,
            reference,
            emotion_vector,
            None,
            float(QWEN_EMOTION_ALPHA),
        )
        outputs.append(
            run_demo(
                sessions,
                frontend,
                metadata,
                reference.prompt_frames,
                QWEN_TARGET_TEXT,
                QWEN_OUTPUT_PATH,
                "Qwen text-emotion voice clone",
            )
        )

    if RUN_MANUAL_EMOTION_VECTOR_VOICE_CLONE:
        print_progress("Preparing manual emotion-vector conditioning...")
        prepare_conditioning(
            sessions,
            reference,
            MANUAL_EMOTION_VECTOR,
            None,
            float(MANUAL_EMOTION_ALPHA),
        )
        outputs.append(
            run_demo(
                sessions,
                frontend,
                metadata,
                reference.prompt_frames,
                MANUAL_EMOTION_TARGET_TEXT,
                MANUAL_EMOTION_OUTPUT_PATH,
                "manual emotion-vector voice clone",
            )
        )
    return tuple(outputs)


def main() -> None:
    pipeline_started = time.perf_counter()
    print_progress(f"Enabled voice modes: {', '.join(enabled_voice_modes())}")
    metadata_path = ONNX_FOLDER / "IndexTTS2_Metadata.onnx"
    print_progress(f"Reading package metadata: {metadata_path}")
    metadata = metadata_from(metadata_path)
    paths = RuntimePaths.from_metadata(ONNX_FOLDER, metadata, DECODE_STRATEGY)

    sessions_started = time.perf_counter()
    print_progress(f"Loading ONNX Runtime sessions (strategy={DECODE_STRATEGY})...")
    sessions = RuntimeSessions(paths, metadata)
    print_progress(
        f"ONNX Runtime sessions ready in "
        f"{time.perf_counter() - sessions_started:.2f}s."
    )
    run_synthesis(sessions, metadata)
    print_progress(
        f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s."
    )


if __name__ == "__main__":
    main()