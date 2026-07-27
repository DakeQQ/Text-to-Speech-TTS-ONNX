import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import soundfile as sf
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer

from Shared_Weights import attach_shared_initializers


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import reference_audio_path


SCRIPT_DIR = Path(__file__).resolve().parent
DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")


# ============================== Configuration ==============================
# Edit these values directly; the CLI is reserved for selecting the ONNX folder.
DOWNLOAD_PATH = str(Path.home() / "Downloads" / "MOSS-TTS-Nano-100M")
GENERATED_AUDIO_PATH = SCRIPT_DIR / "generated.wav"
TARGET_TTS = [
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology.",
]

MODE = "continuation"             # continuation | voice_clone
PROMPT_TEXT = None                # Continuation requires both text and audio, or neither.
PROMPT_AUDIO_PATH = None          # None uses the bundled reference in voice_clone mode.
DECODE_STRATEGY = "sampling"      # greedy | penalty_greedy | sampling
MAX_FRAMES = 375                  # Maximum generated frames; 0 uses graph capacity.
MIN_FRAMES = 0                    # Minimum frames before accepting a stop decision.

# MOSS keeps text and audio sampling controls separate.
TEXT_TEMPERATURE = 0.8
TEXT_TOP_P = 0.9
TEXT_TOP_K = 10
AUDIO_TEMPERATURE = 0.8
AUDIO_TOP_P = 0.9
AUDIO_TOP_K = 10
AUDIO_REPETITION_PENALTY = 1.05

ORT_LOG = False                   # Enable ONNX Runtime logging.
ORT_FP16 = False                  # Enable FP16 runtime settings where supported.
ORT_ACCELERATE_PROVIDERS = []     # Optional providers, e.g. ['CUDAExecutionProvider'].
MAX_THREADS = 0                   # CPU parallel threads; 0 lets ORT choose.
DEVICE_ID = 0                     # Accelerator device index.
SHOW_PROGRESS = True              # Print pipeline stages and loop progress.
# ===========================================================================


def print_progress(message):
    if SHOW_PROGRESS:
        print(f"[MOSS-TTS-Nano] {message}", flush=True)


if MODE not in {"continuation", "voice_clone"}:
    raise ValueError("MODE must be 'continuation' or 'voice_clone'.")
if DECODE_STRATEGY not in DECODE_STRATEGIES:
    raise ValueError(f"DECODE_STRATEGY must be one of {DECODE_STRATEGIES}.")
if MAX_FRAMES < 0 or MIN_FRAMES < 0:
    raise ValueError("MAX_FRAMES and MIN_FRAMES must be >= 0.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run compact MOSS-TTS-Nano ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=Path(__file__).resolve().parent / "MOSS_TTS_Nano_Optimized",
    )
    return parser.parse_args()


ARGS = parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()
GENERATED_AUDIO_PATH = GENERATED_AUDIO_PATH.expanduser().resolve()
pipeline_started = time.perf_counter()

if MODE == "voice_clone" and PROMPT_TEXT is not None:
    raise ValueError("voice_clone mode does not accept PROMPT_TEXT.")
if MODE == "continuation" and (PROMPT_TEXT is None) != (PROMPT_AUDIO_PATH is None):
    raise ValueError("Prompted continuation requires both PROMPT_TEXT and PROMPT_AUDIO_PATH.")
PROMPT_AUDIO_PATH = (
    Path(PROMPT_AUDIO_PATH).expanduser().resolve()
    if PROMPT_AUDIO_PATH is not None
    else reference_audio_path("moss_tts") if MODE == "voice_clone" else None
)
USE_PROMPT_AUDIO = PROMPT_AUDIO_PATH is not None


def read_metadata(path):
    model = onnx.load(str(path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


METADATA_PATH = ONNX_FOLDER / "MossTTSNano_Metadata.onnx"
print_progress(f"Reading package metadata: {METADATA_PATH}")
METADATA = read_metadata(METADATA_PATH)
EXPECTED_METADATA_KEYS = {
    "graph_layout",
    "in_sample_rate",
    "out_sample_rate",
    "max_seq_len",
    "use_f16_kv",
    "compute_in_f32",
    "shared_initializer_model_file",
    "shared_initializer_data_file",
    "model_file_name_audio_encoder",
    "model_file_name_audio_decoder",
    "audio_vocab_size",
    "audio_start_token_id",
    "audio_end_token_id",
    "audio_user_slot_token_id",
    "audio_assistant_slot_token_id",
    "audio_pad_token_id",
    "continue_decision_id",
    "stop_decision_id",
    "user_prompt_prefix_token_ids",
    "user_prompt_after_reference_token_ids",
    "assistant_prompt_prefix_token_ids",
    "no_reference_token_ids",
    "samples_per_frame_per_channel",
    *{
        f"model_file_name_main_prefill_{strategy}"
        for strategy in DECODE_STRATEGIES
    },
    *{
        f"model_file_name_decode_step_{strategy}"
        for strategy in DECODE_STRATEGIES
    },
}
if set(METADATA) != EXPECTED_METADATA_KEYS:
    raise ValueError(
        "MOSS TTS Nano metadata keys do not match the runtime contract: "
        f"missing={sorted(EXPECTED_METADATA_KEYS - set(METADATA))}, "
        f"extra={sorted(set(METADATA) - EXPECTED_METADATA_KEYS)}."
    )
if METADATA.get("graph_layout") != "strategy_prefill_decode_step":
    raise ValueError(
        "MOSS TTS Nano strategy_prefill_decode_step metadata is required; "
        "re-export before inference."
    )


def meta_str(key):
    try:
        return METADATA[key]
    except KeyError as exc:
        raise KeyError(f"Required metadata key {key!r} is missing from {METADATA_PATH}.") from exc


def meta_int(key):
    try:
        return int(meta_str(key))
    except KeyError as exc:
        raise KeyError(f"Required metadata key {key!r} is missing from {METADATA_PATH}.") from exc


def meta_int_list(key):
    return [int(value) for value in meta_str(key).split(",") if value]


def meta_bool(key):
    value = METADATA.get(key)
    if value not in {"0", "1"}:
        raise ValueError(f"Metadata flag {key!r} must be 0 or 1, got {value!r}.")
    return value == "1"


MAX_SEQ_LEN = meta_int("max_seq_len")
MAX_NEW_FRAMES = MAX_FRAMES if MAX_FRAMES > 0 else MAX_SEQ_LEN
MIN_NEW_FRAMES = MIN_FRAMES
IN_SAMPLE_RATE = meta_int("in_sample_rate")
OUT_SAMPLE_RATE = meta_int("out_sample_rate")
SAMPLES_PER_CODEC_FRAME = meta_int("samples_per_frame_per_channel")
STOP_DECISION = meta_int("stop_decision_id")
TEXT_DECISION_COUNT = max(STOP_DECISION, meta_int("continue_decision_id")) + 1
AUDIO_VOCAB_SIZE = meta_int("audio_vocab_size")
AUDIO_PAD_TOKEN = meta_int("audio_pad_token_id")
ASSISTANT_SLOT = meta_int("audio_assistant_slot_token_id")
USER_SLOT = meta_int("audio_user_slot_token_id")
AUDIO_START_TOKEN = meta_int("audio_start_token_id")
AUDIO_END_TOKEN = meta_int("audio_end_token_id")
PRESERVE_FP16_ATTENTION = meta_bool("use_f16_kv") and not meta_bool("compute_in_f32")

if MIN_NEW_FRAMES > MAX_NEW_FRAMES:
    raise ValueError("MIN_FRAMES cannot exceed the effective MAX_FRAMES cap.")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: np.dtype
    shape: tuple[int | str | None, ...]

    @classmethod
    def from_node_arg(cls, node_arg):
        dtype = {
            "tensor(float)": np.dtype(np.float32),
            "tensor(float16)": np.dtype(np.float16),
            "tensor(int16)": np.dtype(np.int16),
            "tensor(int32)": np.dtype(np.int32),
            "tensor(int64)": np.dtype(np.int64),
        }.get(node_arg.type)
        if dtype is None:
            raise TypeError(f"Unsupported model tensor type: {node_arg.type}")
        return cls(node_arg.name, dtype, tuple(node_arg.shape))

    @property
    def rank(self):
        return len(self.shape)

    @property
    def dynamic_axes(self):
        return tuple(index for index, dim in enumerate(self.shape) if not isinstance(dim, int))

    @property
    def is_static(self):
        return not self.dynamic_axes

    def concrete_shape(self, *dynamic_dimensions):
        if len(dynamic_dimensions) != len(self.dynamic_axes):
            raise ValueError(
                f"{self.name!r} needs {len(self.dynamic_axes)} dynamic dimension(s), "
                f"got {len(dynamic_dimensions)}."
            )
        dynamic_values = iter(dynamic_dimensions)
        return tuple(
            dim if isinstance(dim, int) else int(next(dynamic_values))
            for dim in self.shape
        )

    def array(self, data):
        array = np.ascontiguousarray(data, dtype=self.dtype)
        if array.ndim != self.rank or any(
            isinstance(dim, int) and array.shape[index] != dim
            for index, dim in enumerate(self.shape)
        ):
            raise ValueError(f"{self.name!r} expects {self.shape}, got {array.shape}.")
        return array

    def filled_array(self, value, *dynamic_dimensions):
        return np.full(
            self.concrete_shape(*dynamic_dimensions),
            value,
            dtype=self.dtype,
        )

    def ort_value(self, data, device, device_id):
        return onnxruntime.OrtValue.ortvalue_from_numpy(
            self.array(data),
            device,
            device_id,
        )

    def full(self, value, device, device_id):
        return onnxruntime.OrtValue.ortvalue_from_numpy(
            self.filled_array(value),
            device,
            device_id,
        )

    def zeros(self, device, device_id, *dynamic_dimensions):
        return onnxruntime.OrtValue.ortvalue_from_numpy(
            self.filled_array(0, *dynamic_dimensions),
            device,
            device_id,
        )


@dataclass(frozen=True)
class SessionIO:
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]

    @classmethod
    def from_session(cls, session):
        return cls(
            tuple(TensorSpec.from_node_arg(value) for value in session.get_inputs()),
            tuple(TensorSpec.from_node_arg(value) for value in session.get_outputs()),
        )


def select_one(specs, predicate, role):
    matches = tuple(spec for spec in specs if predicate(spec))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {role}, found {[spec.name for spec in matches]}.")
    return matches[0]


def is_integer(spec):
    return np.issubdtype(spec.dtype, np.integer)


def is_floating(spec):
    return np.issubdtype(spec.dtype, np.floating)


def indexed_specs(specs, predicate):
    return tuple((index, spec) for index, spec in enumerate(specs) if predicate(spec))


def bind_values(binding, specs, values):
    if len(specs) != len(values):
        raise ValueError(f"Cannot bind {len(values)} values to {len(specs)} model inputs.")
    for spec, value in zip(specs, values):
        binding.bind_ortvalue_input(spec.name, value)


def run_binding(session, binding, run_options):
    session.run_with_iobinding(binding, run_options=run_options)
    return binding.get_outputs()


def rank3_axes(spec, sequence_label=None):
    feature_candidates = tuple(
        index
        for index, dim in enumerate(spec.shape)
        if isinstance(dim, int) and dim > 1
    )
    if spec.rank != 3 or len(feature_candidates) != 1:
        raise ValueError(f"Cannot infer batch/sequence/features axes from {spec.name!r}: {spec.shape}")
    feature_axis = feature_candidates[0]
    if sequence_label is None:
        sequence_candidates = tuple(
            index for index in spec.dynamic_axes if index != feature_axis
        )
    else:
        sequence_candidates = tuple(
            index
            for index, dim in enumerate(spec.shape)
            if sequence_label in str(dim).lower()
        )
    if len(sequence_candidates) != 1:
        raise ValueError(f"Cannot infer sequence axis from {spec.name!r}: {spec.shape}")
    sequence_axis = sequence_candidates[0]
    batch_axis = next(index for index in range(spec.rank) if index not in (sequence_axis, feature_axis))
    return batch_axis, sequence_axis, feature_axis


def pack_rank3(spec, rows, fill_value=0):
    batch_axis, sequence_axis, feature_axis = rank3_axes(spec)
    array = spec.filled_array(fill_value, rows.shape[0])
    np.transpose(array, (batch_axis, sequence_axis, feature_axis))[0] = rows
    return array


def unpack_rank3(spec, array, sequence_label=None):
    axes = rank3_axes(spec, sequence_label)
    return np.ascontiguousarray(np.transpose(spec.array(array), axes)[0])


def unpack_waveform(output_spec, decoder_input_spec, array):
    waveform = output_spec.array(array)
    sample_axes = tuple(
        index
        for index, dim in enumerate(output_spec.shape)
        if "sample" in str(dim).lower()
    )
    if len(sample_axes) != 1:
        raise ValueError(f"Cannot infer sample axis from {output_spec.name!r}: {output_spec.shape}")
    sample_axis = sample_axes[0]
    input_batch_axis, _, _ = rank3_axes(decoder_input_spec)
    batch_size = int(decoder_input_spec.shape[input_batch_axis])
    batch_axis = next(
        index
        for index in range(output_spec.rank)
        if index != sample_axis and waveform.shape[index] == batch_size
    )
    channel_axis = next(
        index for index in range(output_spec.rank) if index not in (batch_axis, sample_axis)
    )
    return np.ascontiguousarray(np.transpose(waveform, (batch_axis, sample_axis, channel_axis))[0])


def load_prompt_audio(path, sample_rate, input_spec):
    _, _, channel_axis = rank3_axes(input_spec)
    channels = int(input_spec.shape[channel_axis])
    samples = np.array(
        AudioSegment.from_file(path)
        .set_channels(channels)
        .set_frame_rate(sample_rate)
        .set_sample_width(2)
        .get_array_of_samples(),
        dtype=np.int16,
    )
    if np.issubdtype(input_spec.dtype, np.floating):
        samples = (samples.astype(np.float32) * (1.0 / 32768.0)).astype(
            input_spec.dtype
        )
    else:
        samples = samples.astype(input_spec.dtype, copy=False)
    return pack_rank3(input_spec, samples.reshape(-1, channels))


session_options = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
for options in (session_options, run_options):
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
session_options.inter_op_num_threads = MAX_THREADS
session_options.intra_op_num_threads = MAX_THREADS
session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    if PRESERVE_FP16_ATTENTION
    else onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
)
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
        "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
        if ORT_FP16
        else ""
    ),
}.items():
    session_options.add_session_config_entry(key, value)
run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
disabled_optimizers = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16
    else None
)

if "OpenVINOExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [{
        "device_type": "CPU",
        "precision": "ACCURACY",
        "num_of_threads": MAX_THREADS if MAX_THREADS else 8,
        "num_streams": 1,
        "enable_opencl_throttling": False,
        "enable_qdq_optimizer": False,
        "disable_dynamic_shapes": False,
    }]
    device_type = "cpu"
    ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [{
        "device_id": DEVICE_ID,
        "gpu_mem_limit": 24 * 1024**3,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "use_tf32": "1",
        "do_copy_in_default_stream": "0",
        "enable_cuda_graph": "0",
    }]
    device_type = "cuda"
    ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
    provider_options = [{
        "device_id": DEVICE_ID,
        "performance_preference": "high_performance",
        "device_filter": "gpu",
        "disable_metacommands": "false",
        "enable_graph_capture": "false",
    }]
    device_type = "dml"
    ort_device_type = C.OrtDevice.dml()
else:
    provider_options = None
    device_type = "cpu"
    ort_device_type = C.OrtDevice.cpu()

PROVIDERS = ORT_ACCELERATE_PROVIDERS or ["CPUExecutionProvider"]
ORT_DEVICE = C.OrtDevice(ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
KV_DEVICE_TYPE = "cpu" if device_type == "dml" else device_type
KV_DEVICE = (
    C.OrtDevice(C.OrtDevice.cpu(), C.OrtDevice.default_memory(), DEVICE_ID)
    if device_type == "dml"
    else ORT_DEVICE
)

shared_started = time.perf_counter()
print_progress("Attaching shared ONNX initializers...")
SHARED_REFS = attach_shared_initializers(
    session_options,
    ONNX_FOLDER / meta_str("shared_initializer_model_file"),
)
shared_data_path = ONNX_FOLDER / meta_str("shared_initializer_data_file")
if not shared_data_path.is_file():
    raise FileNotFoundError(f"Missing shared initializer data: {shared_data_path}")
print_progress(
    f"Shared ONNX initializers ready in {time.perf_counter() - shared_started:.2f}s."
)

session_count = 0
session_total = 3 + int(USE_PROMPT_AUDIO)


def create_session(file_name):
    global session_count
    path = ONNX_FOLDER / file_name
    session_count += 1
    print_progress(f"Loading ONNX graph {session_count}/{session_total}: {path.name}")
    return onnxruntime.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=PROVIDERS,
        provider_options=provider_options,
        disabled_optimizers=disabled_optimizers,
    )


startup_start = time.perf_counter()
prefill_session = create_session(
    meta_str(f"model_file_name_main_prefill_{DECODE_STRATEGY}")
)
decode_session = create_session(
    meta_str(f"model_file_name_decode_step_{DECODE_STRATEGY}")
)
decoder_session = create_session(meta_str("model_file_name_audio_decoder"))
encoder_session = (
    create_session(meta_str("model_file_name_audio_encoder"))
    if USE_PROMPT_AUDIO
    else None
)
startup_seconds = time.perf_counter() - startup_start
print_progress(
    f"ONNX Runtime sessions ready in {startup_seconds:.2f}s; "
    f"strategy={DECODE_STRATEGY}; providers={decode_session.get_providers()}."
)

prefill_binding = prefill_session.io_binding()
decode_bindings = (decode_session.io_binding(), decode_session.io_binding())
decoder_binding = decoder_session.io_binding()
encoder_binding = encoder_session.io_binding() if encoder_session else None

prefill_io = SessionIO.from_session(prefill_session)
decode_io = SessionIO.from_session(decode_session)
decoder_io = SessionIO.from_session(decoder_session)
encoder_io = SessionIO.from_session(encoder_session) if encoder_session else None

prompt_input_spec = select_one(
    prefill_io.inputs,
    lambda spec: spec.rank == 3 and is_integer(spec),
    "prefill token input",
)
prefill_control_specs = tuple(spec for spec in prefill_io.inputs if spec != prompt_input_spec)
decode_state_input_specs = tuple(
    spec for spec in decode_io.inputs if spec.rank == 4 and is_floating(spec)
)
decode_generated_input_spec = select_one(
    decode_io.inputs,
    lambda spec: spec.rank == 3 and is_integer(spec),
    "generated-code input",
)
integer_scalar_inputs = tuple(
    spec for spec in decode_io.inputs if spec.rank == 1 and is_integer(spec)
)
length_dtype = max((spec.dtype for spec in integer_scalar_inputs), key=lambda dtype: dtype.itemsize)
decode_length_input_spec = select_one(
    integer_scalar_inputs,
    lambda spec: spec.dtype == length_dtype,
    "sequence-length input",
)
decode_control_specs = tuple(
    spec
    for spec in decode_io.inputs
    if spec not in decode_state_input_specs
    and spec != decode_generated_input_spec
    and spec != decode_length_input_spec
)

prefill_state_outputs = indexed_specs(
    prefill_io.outputs,
    lambda spec: spec.rank == 4 and is_floating(spec),
)
decode_state_outputs = indexed_specs(
    decode_io.outputs,
    lambda spec: spec.rank == 4 and is_floating(spec),
)
if len(prefill_state_outputs) != len(decode_state_input_specs) or len(decode_state_outputs) != len(
    decode_state_input_specs
):
    raise RuntimeError("Prefill and decode recurrent-state counts do not align.")

prefill_length_output = select_one(
    prefill_io.outputs,
    lambda spec: spec.rank == 1 and spec.dtype == length_dtype,
    "prefill sequence-length output",
)
decode_length_output = select_one(
    decode_io.outputs,
    lambda spec: spec.rank == 1 and spec.dtype == length_dtype,
    "decode sequence-length output",
)
prefill_decision_output = select_one(
    prefill_io.outputs,
    lambda spec: spec.rank == 1 and is_integer(spec) and spec.dtype != length_dtype,
    "prefill decision output",
)
decode_decision_output = select_one(
    decode_io.outputs,
    lambda spec: spec.rank == 1 and is_integer(spec) and spec.dtype != length_dtype,
    "decode decision output",
)
decode_generated_output = select_one(
    decode_io.outputs,
    lambda spec: spec.rank == 3 and is_integer(spec),
    "generated-code output",
)

prefill_length_index = prefill_io.outputs.index(prefill_length_output)
prefill_decision_index = prefill_io.outputs.index(prefill_decision_output)
decode_length_index = decode_io.outputs.index(decode_length_output)
decode_decision_index = decode_io.outputs.index(decode_decision_output)
decode_generated_index = decode_io.outputs.index(decode_generated_output)

text_control_values = (
    TEXT_TEMPERATURE,
    min(TEXT_TOP_K, TEXT_DECISION_COUNT),
    TEXT_TOP_P,
)
audio_control_values = (
    AUDIO_TEMPERATURE,
    min(AUDIO_TOP_K, AUDIO_VOCAB_SIZE),
    AUDIO_TOP_P,
    AUDIO_REPETITION_PENALTY,
)
if DECODE_STRATEGY == "sampling":
    prefill_constants = tuple(
        spec.full(value, device_type, DEVICE_ID)
        for spec, value in zip(prefill_control_specs, text_control_values)
    )
    decode_constants = tuple(
        spec.full(value, device_type, DEVICE_ID)
        for spec, value in zip(
            decode_control_specs,
            text_control_values + audio_control_values,
        )
    )
elif DECODE_STRATEGY == "penalty_greedy":
    prefill_constants = ()
    decode_constants = (
        decode_control_specs[0].full(AUDIO_REPETITION_PENALTY, device_type, DEVICE_ID),
    )
else:
    prefill_constants = ()
    decode_constants = ()
bind_values(prefill_binding, prefill_control_specs, prefill_constants)
for binding in decode_bindings:
    bind_values(binding, decode_control_specs, decode_constants)

prefill_static_buffers = []
prefill_static_specs = {
    prefill_length_output: prefill_length_output,
    prefill_decision_output: decode_decision_output,
}
for (_, prefill_spec), (_, decode_spec) in zip(prefill_state_outputs, decode_state_outputs):
    if decode_spec.is_static:
        prefill_static_specs[prefill_spec] = decode_spec
prefill_dynamic_output_specs = tuple(
    spec for spec in prefill_io.outputs if spec not in prefill_static_specs
)
for output_spec in prefill_io.outputs:
    if output_spec in prefill_static_specs:
        output_device = KV_DEVICE_TYPE if output_spec.rank == 4 else device_type
        value = prefill_static_specs[output_spec].zeros(output_device, DEVICE_ID)
        prefill_binding.bind_ortvalue_output(output_spec.name, value)
        prefill_static_buffers.append(value)
    else:
        target = KV_DEVICE if output_spec.rank == 4 else ORT_DEVICE
        prefill_binding._iobinding.bind_output(output_spec.name, target)

decode_static_banks = []
decode_dynamic_output_specs = tuple(spec for spec in decode_io.outputs if not spec.is_static)
for binding in decode_bindings:
    bank = []
    for spec in decode_io.outputs:
        if spec.is_static:
            output_device = KV_DEVICE_TYPE if spec.rank == 4 else device_type
            value = spec.zeros(output_device, DEVICE_ID)
            binding.bind_ortvalue_output(spec.name, value)
            bank.append(value)
        else:
            target = KV_DEVICE if spec.rank == 4 else ORT_DEVICE
            binding._iobinding.bind_output(spec.name, target)
    decode_static_banks.append(bank)

empty_generated_codec = decode_generated_input_spec.zeros(device_type, DEVICE_ID, 0)

print_progress("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(DOWNLOAD_PATH, trust_remote_code=True)


def encode_text(text):
    return list(tokenizer.encode(text, add_special_tokens=False))


USER_PROMPT_PREFIX = meta_int_list("user_prompt_prefix_token_ids")
USER_PROMPT_AFTER_REFERENCE = meta_int_list(
    "user_prompt_after_reference_token_ids"
)
ASSISTANT_PROMPT_PREFIX = meta_int_list("assistant_prompt_prefix_token_ids")
NO_REFERENCE_PREFIX = (
    USER_PROMPT_PREFIX
    + meta_int_list("no_reference_token_ids")
    + USER_PROMPT_AFTER_REFERENCE
)
_, _, packed_feature_axis = rank3_axes(prompt_input_spec)
PACKED_WIDTH = int(prompt_input_spec.shape[packed_feature_axis])


def text_rows(token_ids):
    rows = np.full(
        (len(token_ids), PACKED_WIDTH),
        AUDIO_PAD_TOKEN,
        dtype=prompt_input_spec.dtype,
    )
    if token_ids:
        rows[:, 0] = np.asarray(token_ids, dtype=prompt_input_spec.dtype)
    return rows


def audio_prefix_rows(prompt_audio_codes, slot_token_id):
    rows = np.full(
        (prompt_audio_codes.shape[0], PACKED_WIDTH),
        AUDIO_PAD_TOKEN,
        dtype=prompt_input_spec.dtype,
    )
    rows[:, 0] = slot_token_id
    rows[:, 1:] = prompt_audio_codes
    return rows


def build_prompt_input_ids(text, mode, prompt_text=None, prompt_audio_codes=None):
    text_token_ids = encode_text(text)
    if mode == "voice_clone":
        prompt_token_ids = USER_PROMPT_PREFIX + [AUDIO_START_TOKEN]
        suffix_token_ids = (
            [AUDIO_END_TOKEN]
            + USER_PROMPT_AFTER_REFERENCE
            + text_token_ids
            + ASSISTANT_PROMPT_PREFIX
            + [AUDIO_START_TOKEN]
        )
        rows = np.concatenate(
            [
                text_rows(prompt_token_ids),
                audio_prefix_rows(prompt_audio_codes, USER_SLOT),
                text_rows(suffix_token_ids),
            ],
            axis=0,
        )
    else:
        effective_text = text if prompt_text is None else prompt_text + text
        sections = [
            text_rows(NO_REFERENCE_PREFIX + encode_text(effective_text) + ASSISTANT_PROMPT_PREFIX),
            text_rows([AUDIO_START_TOKEN]),
        ]
        if prompt_audio_codes is not None:
            sections.append(audio_prefix_rows(prompt_audio_codes, ASSISTANT_SLOT))
        rows = np.concatenate(sections, axis=0)
    return pack_rank3(prompt_input_spec, rows, AUDIO_PAD_TOKEN)


prompt_audio_codes = None
encoder_seconds = 0.0
if encoder_session:
    print_progress(f"Encoding reference audio: {PROMPT_AUDIO_PATH}")
    encoder_input_spec = encoder_io.inputs[0]
    encoder_codes_spec = select_one(
        encoder_io.outputs,
        lambda spec: spec.rank == 3 and is_integer(spec),
        "audio-code output",
    )
    encoder_length_spec = select_one(
        encoder_io.outputs,
        lambda spec: spec.rank == 1 and is_integer(spec),
        "audio-code length output",
    )
    audio = load_prompt_audio(PROMPT_AUDIO_PATH, IN_SAMPLE_RATE, encoder_input_spec)
    encoder_input = encoder_input_spec.ort_value(audio, device_type, DEVICE_ID)
    encoder_binding.bind_ortvalue_input(encoder_input_spec.name, encoder_input)
    encoder_length_buffer = encoder_length_spec.zeros(device_type, DEVICE_ID)
    for spec in encoder_io.outputs:
        if spec == encoder_length_spec:
            encoder_binding.bind_ortvalue_output(spec.name, encoder_length_buffer)
        else:
            encoder_binding._iobinding.bind_output(spec.name, ORT_DEVICE)
    encoder_start = time.perf_counter()
    encoder_outputs = run_binding(encoder_session, encoder_binding, run_options)
    encoder_seconds = time.perf_counter() - encoder_start
    code_length = int(
        encoder_outputs[encoder_io.outputs.index(encoder_length_spec)].numpy().reshape(-1)[0]
    )
    prompt_audio_codes = unpack_rank3(
        encoder_codes_spec,
        encoder_outputs[encoder_io.outputs.index(encoder_codes_spec)].numpy(),
        "frame",
    )
    prompt_audio_codes = prompt_audio_codes[:code_length]
    print_progress(
        f"Reference conditioning ready: {prompt_audio_codes.shape[0]} frame(s) x "
        f"{prompt_audio_codes.shape[1]} channels in {encoder_seconds:.2f}s."
    )


def bind_dynamic_outputs(binding, specs):
    for spec in specs:
        target = KV_DEVICE if spec.rank == 4 else ORT_DEVICE
        binding._iobinding.bind_output(spec.name, target)


decoder_input_spec = select_one(
    decoder_io.inputs,
    lambda spec: spec.rank == 3 and is_integer(spec),
    "audio decoder input",
)
decoder_output_spec = select_one(
    decoder_io.outputs,
    lambda spec: spec.rank == 3 and (is_integer(spec) or is_floating(spec)),
    "audio decoder output",
)


def decode_audio(generated_codec):
    decoder_input = generated_codec
    prompt_prefix_samples = 0
    if MODE == "continuation" and prompt_audio_codes is not None:
        generated_codes = unpack_rank3(decode_generated_output, generated_codec.numpy())
        decoder_rows = np.concatenate(
            [prompt_audio_codes, generated_codes],
            axis=0,
        )
        decoder_input = decoder_input_spec.ort_value(
            pack_rank3(decoder_input_spec, decoder_rows),
            device_type,
            DEVICE_ID,
        )
        prompt_prefix_samples = prompt_audio_codes.shape[0] * SAMPLES_PER_CODEC_FRAME

    decoder_binding.bind_ortvalue_input(decoder_input_spec.name, decoder_input)
    bind_dynamic_outputs(decoder_binding, decoder_io.outputs)
    start = time.perf_counter()
    outputs = run_binding(decoder_session, decoder_binding, run_options)
    waveform = unpack_waveform(decoder_output_spec, decoder_input_spec, outputs[0].numpy())
    if prompt_prefix_samples:
        waveform = waveform[prompt_prefix_samples:]
    return waveform, time.perf_counter() - start


generated_waveforms = []
total_frames = 0
total_generation_seconds = 0.0
total_decoder_seconds = 0.0
total_audio_seconds = 0.0
total_prefill_calls = 0
total_decode_calls = 0

print_progress(f"Prepared {len(TARGET_TTS)} text target(s).")
for target_index, target in enumerate(TARGET_TTS, start=1):
    print_progress(f"Starting target {target_index}/{len(TARGET_TTS)}: {target!r}")
    prompt_ids = build_prompt_input_ids(
        target,
        MODE,
        prompt_text=PROMPT_TEXT,
        prompt_audio_codes=prompt_audio_codes,
    )
    _, prompt_sequence_axis, _ = rank3_axes(prompt_input_spec)
    prompt_len = prompt_ids.shape[prompt_sequence_axis]
    if prompt_len >= MAX_SEQ_LEN:
        raise ValueError(f"Prompt length {prompt_len} reaches MAX_SEQ_LEN={MAX_SEQ_LEN}.")
    generation_limit = min(MAX_NEW_FRAMES, MAX_SEQ_LEN - prompt_len - 1)
    print_progress(f"Generating codec frames (limit {generation_limit})...")
    prompt_value = prompt_input_spec.ort_value(prompt_ids, device_type, DEVICE_ID)
    prefill_binding.bind_ortvalue_input(prompt_input_spec.name, prompt_value)
    bind_dynamic_outputs(prefill_binding, prefill_dynamic_output_specs)
    generation_start = time.perf_counter()
    prefill_result = run_binding(prefill_session, prefill_binding, run_options)
    total_prefill_calls += 1

    recurrent_state = tuple(prefill_result[index] for index, _ in prefill_state_outputs)
    decision = int(prefill_result[prefill_decision_index].numpy().reshape(-1)[0])
    kv_seq_len = prefill_result[prefill_length_index]
    generated_codec = empty_generated_codec
    generated_frames = 0
    decode_step = 0

    while (
        generated_frames < generation_limit
        and (decision != STOP_DECISION or generated_frames < MIN_NEW_FRAMES)
    ):
        binding = decode_bindings[decode_step & 1]

        bind_values(binding, decode_state_input_specs, recurrent_state)
        binding.bind_ortvalue_input(decode_length_input_spec.name, kv_seq_len)
        binding.bind_ortvalue_input(decode_generated_input_spec.name, generated_codec)
        bind_dynamic_outputs(binding, decode_dynamic_output_specs)

        decode_result = run_binding(decode_session, binding, run_options)
        total_decode_calls += 1
        recurrent_state = tuple(decode_result[index] for index, _ in decode_state_outputs)
        decision = int(decode_result[decode_decision_index].numpy().reshape(-1)[0])
        kv_seq_len = decode_result[decode_length_index]
        generated_codec = decode_result[decode_generated_index]
        generated_frames += 1
        decode_step += 1
        if generated_frames % 50 == 0 or generated_frames == generation_limit:
            print_progress(
                f"Codec generation: {generated_frames}/{generation_limit} frames"
            )

    generation_seconds = time.perf_counter() - generation_start
    total_frames += generated_frames
    total_generation_seconds += generation_seconds

    decoder_seconds = 0.0
    audio_seconds = generated_frames * SAMPLES_PER_CODEC_FRAME / OUT_SAMPLE_RATE
    if generated_frames:
        print_progress("Decoding waveform...")
        waveform, decoder_seconds = decode_audio(generated_codec)
        generated_waveforms.append(waveform)
        audio_seconds = waveform.shape[0] / OUT_SAMPLE_RATE
    total_decoder_seconds += decoder_seconds
    total_audio_seconds += audio_seconds
    target_rtf = (generation_seconds + decoder_seconds) / audio_seconds if audio_seconds else float("inf")
    print(
        f"Target {target_index}/{len(TARGET_TTS)}: prompt={prompt_len}, frames={generated_frames}, "
        f"decision={decision}, prefill_calls=1, decode_calls={generated_frames}, "
        f"generation={generation_seconds:.3f}s, decoder={decoder_seconds:.3f}s, "
        f"audio={audio_seconds:.3f}s, RTF={target_rtf:.3f}",
        flush=True,
    )

if generated_waveforms:
    silence = np.zeros(
        (int(OUT_SAMPLE_RATE * 0.2), generated_waveforms[0].shape[1]),
        dtype=generated_waveforms[0].dtype,
    )
    segments = []
    for index, waveform in enumerate(generated_waveforms):
        if index:
            segments.append(silence)
        segments.append(waveform)
    final_waveform = np.concatenate(segments, axis=0)
    print_progress(f"Writing generated audio: {GENERATED_AUDIO_PATH}")
    if final_waveform.dtype == np.float16:
        final_waveform = final_waveform.astype(np.float32)
    output_subtype = "PCM_16" if np.issubdtype(final_waveform.dtype, np.integer) else "FLOAT"
    sf.write(
        GENERATED_AUDIO_PATH,
        final_waveform,
        OUT_SAMPLE_RATE,
        subtype=output_subtype,
        format="WAVEX",
    )
    print(
        f"Saved {final_waveform.shape} {final_waveform.dtype} waveform to "
        f"{GENERATED_AUDIO_PATH}",
        flush=True,
    )

frames_per_second = total_frames / total_generation_seconds if total_generation_seconds else 0.0
total_inference_seconds = encoder_seconds + total_generation_seconds + total_decoder_seconds
overall_rtf = total_inference_seconds / total_audio_seconds if total_audio_seconds else float("inf")
print(
    f"Completed targets={len(TARGET_TTS)}, frames={total_frames}, "
    f"MainPrefill calls={total_prefill_calls}, DecodeStep calls={total_decode_calls}, "
    f"frames/s={frames_per_second:.3f}, audio={total_audio_seconds:.3f}s, "
    f"inference={total_inference_seconds:.3f}s, RTF={overall_rtf:.3f}",
    flush=True,
)
print_progress(f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s.")