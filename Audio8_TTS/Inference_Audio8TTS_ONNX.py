import argparse
import concurrent.futures
import re
import sys
import time
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
DOWNLOAD_PATH = str(Path.home() / "Downloads" / "Audio8-TTS-Preview-0.1b")  # Tokenizer files only.
GENERATED_AUDIO_PATH = str(SCRIPT_DIR / "generated.wav")

TARGET_TTS = [
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology.",
]
TTS_MODE = "voice_clone"             # zero_shot | voice_clone
PROMPT_AUDIO_PATH = reference_audio_path("audio8_tts")
PROMPT_TEXT = "对，这就是我，万人敬仰的太乙真人。"  # Transcript of the reference audio; required for voice_clone.
PROMPT_CODES_PATH = ""               # Optional [10, T] .npy of reference codes; skips ReferencePreprocess.

DECODE_STRATEGY = "sampling"         # greedy | penalty_greedy | sampling (sampling = HF generate default).
MAX_NEW_TOKENS = 512                 # Maximum generated frames.
TEMPERATURE = 0.7                    # Sampling temperature.
TOP_P = 0.9                          # Sampling nucleus probability.
TOP_K = 50                           # Sampling candidate count.
MIN_SEQ_LEN = 0                      # Minimum frames before a stop token is accepted; 0 matches HF.
PENALTY_RANGE = 10                   # Recent-token window for penalty-greedy decoding.
REPEAT_PENALTY = 0.8                 # Penalty-greedy repeat score multiplier.
SPEAKER_TAG = "<|speaker:0|>"        # Default speaker tag for voice clone prompt text.

STREAMING = False                    # Decode audio incrementally with stateless windows; meant for accelerators.
USE_AUDIO_NORMALIZER = False         # Normalize prompt and generated audio loudness.
ORT_LOG = False                      # Enable ONNX Runtime logging.
ORT_FP16 = False                     # Enable FP16 runtime settings where supported.
ORT_ACCELERATE_PROVIDERS = []        # Optional providers, e.g. ['CUDAExecutionProvider'].
MAX_THREADS = 0                      # CPU parallel threads; 0 lets ORT choose.
DEVICE_ID = 0                        # Accelerator device index.
SHOW_PROGRESS = True                 # Print pipeline stages and loop progress.
# ===========================================================================


def print_progress(message):
    if SHOW_PROGRESS:
        print(f"[Audio8-TTS] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run compact Audio8-TTS ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=Path(__file__).resolve().parent / "Audio8TTS_Optimized",
    )
    return parser.parse_args()


ARGS = parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()


def audio_normalizer(audio, output_dtype, target_value=None):
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


def load_prompt_audio(path, sample_rate, argument):
    channels = argument.shape[-2]
    samples = np.asarray(
        AudioSegment.from_file(path)
        .set_channels(channels)
        .set_frame_rate(sample_rate)
        .set_sample_width(2)
        .get_array_of_samples()
        , dtype=np.int16
    )
    target_dtype = io_dtype(argument)
    if np.issubdtype(target_dtype, np.floating):
        samples = (samples.astype(np.float32) * (1.0 / 32768.0)).astype(target_dtype)
    return io_array(argument, samples.reshape(-1, channels).T)


def io_dtype(argument):
    match = re.fullmatch(r"tensor\(([^)]+)\)", argument.type)
    element_type = onnx.TensorProto.DataType.Value(match.group(1).upper())
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))


def io_shape(argument, dynamic_dimensions=()):
    dynamic_dimensions = iter(dynamic_dimensions)
    shape = []
    for dimension in argument.shape:
        if isinstance(dimension, int) and dimension >= 0:
            shape.append(dimension)
        else:
            shape.append(next(dynamic_dimensions))
    return tuple(shape)


def static_io_shape(argument):
    if any(
        not isinstance(dimension, int) or dimension < 0
        for dimension in argument.shape
    ):
        return None
    return tuple(argument.shape)


def io_array(argument, data):
    array = np.asarray(data, dtype=io_dtype(argument))
    if array.ndim == len(argument.shape):
        shape = tuple(
            dimension
            if isinstance(dimension, int) and dimension >= 0
            else array.shape[axis]
            for axis, dimension in enumerate(argument.shape)
        )
    else:
        shape = tuple(
            dimension if isinstance(dimension, int) and dimension >= 0 else -1
            for dimension in argument.shape
        )
    return np.ascontiguousarray(array.reshape(shape))


def ort_from_io(argument, data, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        io_array(argument, data),
        device,
        device_id,
    )


def ort_buffer_from_io(argument, device, device_id, dynamic_dimensions=()):
    return onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        io_shape(argument, dynamic_dimensions),
        io_dtype(argument),
        device,
        device_id,
    )


def bind_io_inputs(binding, arguments, values):
    for argument, value in zip(arguments, values):
        binding.bind_ortvalue_input(argument.name, value)


def bind_io_outputs(
    binding,
    arguments,
    ort_device,
    device,
    device_id,
    device_allocate_all=False,
):
    buffers = []
    for argument in arguments:
        if device_allocate_all or static_io_shape(argument) is None:
            binding._iobinding.bind_output(argument.name, ort_device)
            buffers.append(None)
        else:
            buffer = ort_buffer_from_io(argument, device, device_id)
            binding.bind_ortvalue_output(argument.name, buffer)
            buffers.append(buffer)
    return tuple(buffers)


class ConstantOrtValues:
    def __init__(self, device, device_id):
        self.device = device
        self.device_id = device_id
        self.values = {}

    def get(self, argument, data):
        array = io_array(argument, data)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in self.values:
            self.values[key] = onnxruntime.OrtValue.ortvalue_from_numpy(
                array,
                self.device,
                self.device_id,
            )
        return self.values[key]


class IOBindingBank:
    def __init__(self, session, count, ort_device, device, device_id):
        self.session = session
        self.output_arguments = tuple(session.get_outputs())
        self.device_allocate_all = device == "cuda"
        self.outputs_to_rebind = tuple(
            argument
            for argument in self.output_arguments
            if self.device_allocate_all or static_io_shape(argument) is None
        )
        self.bindings = tuple(session.io_binding() for _ in range(count))
        self.output_buffers = tuple(
            bind_io_outputs(
                binding,
                self.output_arguments,
                ort_device,
                device,
                device_id,
                device_allocate_all=self.device_allocate_all,
            )
            for binding in self.bindings
        )
        self.used = [False] * count
        self.ort_device = ort_device

    def select(self, index):
        slot = index % len(self.bindings)
        return slot, self.bindings[slot]

    def run(self, slot, run_options):
        binding = self.bindings[slot]
        if self.used[slot]:
            for argument in self.outputs_to_rebind:
                binding._iobinding.bind_output(argument.name, self.ort_device)
        self.session.run_with_iobinding(binding, run_options=run_options)
        self.used[slot] = True
        return binding.get_outputs()


pipeline_started = time.perf_counter()
metadata_path = ONNX_FOLDER / "Audio8TTS_Metadata.onnx"
print_progress(f"Reading package metadata: {metadata_path}")
metadata_model = onnx.load(metadata_path, load_external_data=False)
metadata = {entry.key: entry.value for entry in metadata_model.metadata_props}
del metadata_model
expected_metadata_keys = {
    "graph_layout",
    "in_sample_rate",
    "out_sample_rate",
    "max_seq_len",
    "num_codebooks",
    "codebook_size",
    "stop_token_ids",
    "samples_per_codec_frame",
    "use_f16_kv",
    "compute_in_f32",
    "shared_initializer_model_file",
    "shared_initializer_data_file",
    "model_file_name_reference_preprocess",
    "model_file_name_target_preprocess",
    "model_file_name_decoder",
    "model_file_name_decoder_stream",
    "zero_shot_prefix_ids",
    "zero_shot_suffix_ids",
    "clone_prefix_head_ids",
    "clone_prefix_tail_ids",
    "clone_suffix_head_ids",
    *{
        f"model_file_name_slow_prefill_{strategy}"
        for strategy in DECODE_STRATEGIES
    },
    *{
        f"model_file_name_decode_step_{strategy}"
        for strategy in DECODE_STRATEGIES
    },
}
missing_metadata = sorted(expected_metadata_keys - metadata.keys())
if missing_metadata:
    raise ValueError(
        f"{metadata_path.name} is missing required metadata key(s): {missing_metadata}."
    )
precision_flags = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
invalid_precision = {
    key: value for key, value in precision_flags.items() if value not in {"0", "1"}
}
if invalid_precision:
    raise ValueError(
        f"{metadata_path.name} has invalid precision metadata: {invalid_precision}."
    )
preserve_fp16_attention = (
    precision_flags["use_f16_kv"] == "1"
    and precision_flags["compute_in_f32"] == "0"
)

session_options = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
for options in (session_options, run_options):
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
session_options.inter_op_num_threads = MAX_THREADS
session_options.intra_op_num_threads = MAX_THREADS
session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
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
        if ORT_FP16 or preserve_fp16_attention
        else ""
    ),
}.items():
    session_options.add_session_config_entry(key, value)
run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
disabled_optimizers = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16 or preserve_fp16_attention
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

ORT_DEVICE = C.OrtDevice(ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)


def meta_str(key):
    try:
        return metadata[key]
    except KeyError as exc:
        raise ValueError(f"Missing required Audio8TTS metadata key: {key!r}.") from exc


def meta_int(key):
    return int(meta_str(key))


def meta_float(key):
    return float(meta_str(key))


def meta_int_list(key):
    return [int(value) for value in meta_str(key).split(",") if value]


shared_model_path = ONNX_FOLDER / meta_str("shared_initializer_model_file")
shared_started = time.perf_counter()
print_progress("Attaching shared ONNX initializers...")
SHARED_REFS = attach_shared_initializers(session_options, shared_model_path)
print_progress(
    f"Shared ONNX initializers ready in {time.perf_counter() - shared_started:.2f}s."
)

session_count = 0
session_total = 0


def create_session(path):
    global session_count
    session_count += 1
    print_progress(f"Loading ONNX graph {session_count}/{session_total}: {path.name}")
    return onnxruntime.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=ORT_ACCELERATE_PROVIDERS or ["CPUExecutionProvider"],
        provider_options=provider_options,
        disabled_optimizers=disabled_optimizers,
    )


IN_SAMPLE_RATE = meta_int("in_sample_rate")
OUT_SAMPLE_RATE = meta_int("out_sample_rate")
MAX_SEQ_LEN = meta_int("max_seq_len")
NUM_CODEBOOKS = meta_int("num_codebooks")
CODE_LIMIT = meta_int("codebook_size")
STOP_TOKEN_SET = set(meta_int_list("stop_token_ids"))

SUPPORTED_MODES = ("zero_shot", "voice_clone")
if TTS_MODE not in SUPPORTED_MODES:
    raise ValueError(f"TTS_MODE must be one of {SUPPORTED_MODES}, got {TTS_MODE!r}.")
if DECODE_STRATEGY not in DECODE_STRATEGIES:
    raise ValueError(f"DECODE_STRATEGY must be one of {DECODE_STRATEGIES}, got {DECODE_STRATEGY!r}.")

max_new_tokens = int(MAX_NEW_TOKENS)
temperature_value = float(TEMPERATURE)
top_p_value = float(TOP_P)
# One top_k feeds both the 4097-wide slow head and the 4096-wide fast head.
top_k_value = min(int(TOP_K), CODE_LIMIT)
if max_new_tokens <= 0:
    raise ValueError("MAX_NEW_TOKENS must be positive.")
if temperature_value <= 0.0:
    raise ValueError("TEMPERATURE must be positive.")
if not 0.0 < top_p_value <= 1.0:
    raise ValueError("TOP_P must be in (0, 1].")
if top_k_value <= 0:
    raise ValueError("TOP_K must be positive.")

print_progress("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    DOWNLOAD_PATH,
    use_fast=True,
    trust_remote_code=True,
    fix_mistral_regex=False,
)

voice_clone = TTS_MODE == "voice_clone"
use_reference_graph = voice_clone and not PROMPT_CODES_PATH
session_started = time.perf_counter()
session_total = 4 + int(use_reference_graph) + int(STREAMING)
reference_session = (
    create_session(ONNX_FOLDER / meta_str("model_file_name_reference_preprocess"))
    if use_reference_graph
    else None
)
target_session = create_session(ONNX_FOLDER / meta_str("model_file_name_target_preprocess"))
prefill_session = create_session(
    ONNX_FOLDER / meta_str(f"model_file_name_slow_prefill_{DECODE_STRATEGY}")
)
decode_session = create_session(
    ONNX_FOLDER / meta_str(f"model_file_name_decode_step_{DECODE_STRATEGY}")
)
decoder_session = create_session(ONNX_FOLDER / meta_str("model_file_name_decoder"))
decoder_stream_session = (
    create_session(ONNX_FOLDER / meta_str("model_file_name_decoder_stream"))
    if STREAMING
    else None
)
print_progress(
    f"ONNX Runtime sessions ready in {time.perf_counter() - session_started:.2f}s; "
    f"strategy={DECODE_STRATEGY}; providers={decode_session.get_providers()}."
)

reference_arguments = tuple(reference_session.get_inputs()) if reference_session else ()
target_arguments = tuple(target_session.get_inputs())
prefill_arguments = tuple(prefill_session.get_inputs())
prefill_output_arguments = tuple(prefill_session.get_outputs())
decode_arguments = tuple(decode_session.get_inputs())
decode_output_arguments = tuple(decode_session.get_outputs())
decoder_arguments = tuple(decoder_session.get_inputs())
decoder_output_arguments = tuple(decoder_session.get_outputs())
decoder_stream_arguments = tuple(decoder_stream_session.get_inputs()) if decoder_stream_session else ()

# Prefill outputs: slow KV / Mamba states, last_hidden_state, semantic_token, kv_seq_len.
num_states = len(prefill_output_arguments) - 3
if num_states <= 0 or num_states % 4 != 0:
    raise ValueError(
        f"Prefill graph exposes {num_states} state tensors; expected a positive multiple of 4."
    )
# The merged DecodeStep takes: states, history_len, [main_save_ids, main controls],
# semantic_token_in, last_hidden_state_slow, generated_codec_in, [fast controls].
decode_state_arguments = decode_arguments[:num_states]
decode_history_argument = decode_arguments[num_states]
decode_cursor = num_states + 1
decode_save_argument = None
if DECODE_STRATEGY != "greedy":
    decode_save_argument = decode_arguments[decode_cursor]
    decode_cursor += 1
control_data = ()
if DECODE_STRATEGY == "penalty_greedy":
    control_data = (REPEAT_PENALTY, PENALTY_RANGE)
elif DECODE_STRATEGY == "sampling":
    control_data = (temperature_value, top_k_value, top_p_value)
decode_main_control_arguments = decode_arguments[decode_cursor:decode_cursor + len(control_data)]
decode_cursor += len(control_data)
decode_fast_arguments = decode_arguments[decode_cursor:decode_cursor + 3]
decode_cursor += 3
decode_fast_control_arguments = decode_arguments[decode_cursor:]
if len(decode_fast_control_arguments) != len(control_data):
    raise ValueError(
        f"DecodeStep graph exposes {len(decode_fast_control_arguments)} fast-AR control input(s); "
        f"expected {len(control_data)}."
    )

decode_last_hidden_pos = num_states
decode_token_pos = num_states + 1
decode_save_id_pos = num_states + 2 if decode_save_argument else None
decode_history_pos = num_states + 3 if decode_save_argument else num_states + 2
decode_generated_codec_pos = len(decode_output_arguments) - 2
decode_frame_pos = len(decode_output_arguments) - 1

reference_bank = (
    IOBindingBank(reference_session, 1, ORT_DEVICE, device_type, DEVICE_ID)
    if reference_session
    else None
)
target_bank = IOBindingBank(target_session, 1, ORT_DEVICE, device_type, DEVICE_ID)
prefill_bank = IOBindingBank(prefill_session, 1, ORT_DEVICE, device_type, DEVICE_ID)
decode_bank = IOBindingBank(decode_session, 2, ORT_DEVICE, device_type, DEVICE_ID)
decoder_bank = IOBindingBank(decoder_session, 1, ORT_DEVICE, device_type, DEVICE_ID)
decoder_stream_bank = (
    IOBindingBank(decoder_stream_session, 1, ORT_DEVICE, device_type, DEVICE_ID)
    if decoder_stream_session
    else None
)
decoder_stream_input = (
    ort_buffer_from_io(decoder_stream_arguments[0], device_type, DEVICE_ID)
    if decoder_stream_session
    else None
)
if decoder_stream_bank:
    decoder_stream_bank.bindings[0].bind_ortvalue_input(
        decoder_stream_arguments[0].name,
        decoder_stream_input,
    )
target_binding = target_bank.bindings[0]
prefill_binding = prefill_bank.bindings[0]
decode_bindings = decode_bank.bindings
constants = ConstantOrtValues(device_type, DEVICE_ID)

ZERO_SHOT_PREFIX_IDS = meta_int_list("zero_shot_prefix_ids")
ZERO_SHOT_SUFFIX_IDS = meta_int_list("zero_shot_suffix_ids")
CLONE_PREFIX_HEAD_IDS = meta_int_list("clone_prefix_head_ids")
CLONE_PREFIX_TAIL_IDS = meta_int_list("clone_prefix_tail_ids")
CLONE_SUFFIX_HEAD_IDS = meta_int_list("clone_suffix_head_ids")
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:\d+\|>")


def clean_text(text):
    return " ".join(str(text).strip().split())


def encode_text(text):
    return tokenizer.encode(text, add_special_tokens=False)


def format_reference_text(text):
    cleaned = clean_text(text)
    if SPEAKER_TAG_PATTERN.search(cleaned):
        return cleaned
    return f"{SPEAKER_TAG}{cleaned}"


def load_reference_codes(path):
    codes = np.load(path).astype(np.int64)
    if codes.ndim != 2 or codes.shape[0] != NUM_CODEBOOKS or codes.shape[1] == 0:
        raise ValueError(
            f"reference codes must have shape [{NUM_CODEBOOKS}, T>0], got {tuple(codes.shape)}"
        )
    if codes.min() < 0 or codes.max() >= CODE_LIMIT:
        raise ValueError(f"reference codes must be in [0, {CODE_LIMIT - 1}]")
    return codes.astype(np.int32)[None]


# Reference conditioning is shared by every target: [prefix_head | reference text | prefix_tail] and the codes.
if voice_clone:
    if not PROMPT_TEXT or not clean_text(PROMPT_TEXT):
        raise ValueError("PROMPT_TEXT is required when TTS_MODE is 'voice_clone'.")
    clone_prefix_ids = CLONE_PREFIX_HEAD_IDS + encode_text(format_reference_text(PROMPT_TEXT)) + CLONE_PREFIX_TAIL_IDS
else:
    clone_prefix_ids = []

encoder_time = 0.0
if not voice_clone:
    reference_codes = np.zeros((1, NUM_CODEBOOKS, 0), dtype=np.int32)
elif PROMPT_CODES_PATH:
    print_progress(f"Loading reference codes: {PROMPT_CODES_PATH}")
    reference_codes = load_reference_codes(PROMPT_CODES_PATH)
else:
    print_progress(f"Encoding reference audio: {PROMPT_AUDIO_PATH}")
    prompt_audio = load_prompt_audio(PROMPT_AUDIO_PATH, IN_SAMPLE_RATE, reference_arguments[0])
    if USE_AUDIO_NORMALIZER:
        prompt_audio = audio_normalizer(prompt_audio, io_dtype(reference_arguments[0]))
    reference_values = (ort_from_io(reference_arguments[0], prompt_audio, device_type, DEVICE_ID),)
    bind_io_inputs(reference_bank.bindings[0], reference_arguments, reference_values)
    encoder_start = time.perf_counter()
    reference_outputs = reference_bank.run(0, run_options)
    encoder_time = time.perf_counter() - encoder_start
    reference_codes = reference_outputs[0].numpy().copy()
    del reference_values, reference_outputs
    print_progress(f"Reference audio encoded into {reference_codes.shape[-1]} codec frames.")
reference_codes_ort = ort_from_io(target_arguments[1], reference_codes, device_type, DEVICE_ID)

# Sampling controls are constants of the run, so they are bound once.
if DECODE_STRATEGY == "sampling":
    bind_io_inputs(
        prefill_binding,
        prefill_arguments[1:],
        tuple(
            constants.get(argument, value)
            for argument, value in zip(prefill_arguments[1:], control_data)
        ),
    )
main_control_values = tuple(
    constants.get(argument, value)
    for argument, value in zip(decode_main_control_arguments, control_data)
)
fast_control_values = tuple(
    constants.get(argument, value)
    for argument, value in zip(decode_fast_control_arguments, control_data)
)
for binding in decode_bindings:
    bind_io_inputs(binding, decode_main_control_arguments, main_control_values)
    bind_io_inputs(binding, decode_fast_control_arguments, fast_control_values)
empty_generated_codec = ort_buffer_from_io(
    decode_fast_arguments[2],
    device_type,
    DEVICE_ID,
    dynamic_dimensions=(0,),
)


def decode_stream_window(window_numpy):
    decoder_stream_input.update_inplace(window_numpy)
    slot, _ = decoder_stream_bank.select(0)
    start = time.perf_counter()
    outputs = decoder_stream_bank.run(slot, run_options)
    elapsed = time.perf_counter() - start
    return outputs[0].numpy().reshape(-1).copy(), elapsed


def decode_full_audio(generated_codec):
    slot, binding = decoder_bank.select(0)
    binding.bind_ortvalue_input(decoder_arguments[0].name, generated_codec)
    start = time.perf_counter()
    outputs = decoder_bank.run(slot, run_options)
    elapsed = time.perf_counter() - start
    return outputs[0].numpy().reshape(-1).copy(), elapsed


save_generated_wav = []
wave_dtype = io_dtype(decoder_output_arguments[0])
empty_segment = np.zeros(int(OUT_SAMPLE_RATE * 0.2), dtype=wave_dtype)
total_audio_samples = 0
total_generation_time = 0.0
total_decoder_time = 0.0
samples_per_codec_frame = meta_int("samples_per_codec_frame")
if decoder_stream_session:
    stream_window_frames = decoder_stream_arguments[0].shape[-1] // NUM_CODEBOOKS

for target_index, target in enumerate(TARGET_TTS, start=1):
    target_text = clean_text(target)
    if not target_text:
        raise ValueError("text must not be empty")
    target_token_ids = encode_text(target_text)
    if voice_clone:
        prefix_ids = clone_prefix_ids
        suffix_ids = CLONE_SUFFIX_HEAD_IDS + target_token_ids + ZERO_SHOT_SUFFIX_IDS
    else:
        prefix_ids = ZERO_SHOT_PREFIX_IDS + target_token_ids + ZERO_SHOT_SUFFIX_IDS
        suffix_ids = []
    target_values = (
        ort_from_io(
            target_arguments[0], np.asarray(prefix_ids, dtype=np.int32).reshape(1, -1), device_type, DEVICE_ID
        ),
        reference_codes_ort,
        ort_from_io(
            target_arguments[2], np.asarray(suffix_ids, dtype=np.int32).reshape(1, -1), device_type, DEVICE_ID
        ),
    )
    bind_io_inputs(target_binding, target_arguments, target_values)
    step_start = time.perf_counter()
    hidden_states, ids_len_value = target_bank.run(0, run_options)
    main_time = time.perf_counter() - step_start
    ids_len = int(ids_len_value.numpy().flat[0])
    if ids_len >= MAX_SEQ_LEN:
        raise ValueError(
            f"Prompt length {ids_len} must be smaller than {MAX_SEQ_LEN}; use a shorter reference clip or text."
        )

    prefill_binding.bind_ortvalue_input(prefill_arguments[0].name, hidden_states)
    step_start = time.perf_counter()
    prefill_outputs = prefill_bank.run(0, run_options)
    main_time += time.perf_counter() - step_start
    cached_state_tensors = prefill_outputs[:num_states]
    last_hidden_state, codec_token_main, history_len = prefill_outputs[num_states:]
    main_save_ids = codec_token_main if DECODE_STRATEGY != "greedy" else None
    generated_codec = empty_generated_codec
    generated_frames = 0
    stream_frames = []
    stream_futures = []
    stream_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if STREAMING else None
    decode_step = 0
    selected_token_id = int(codec_token_main.numpy().flat[0])

    generation_limit = min(max_new_tokens, MAX_SEQ_LEN - ids_len)
    print_progress(f"Generating codec frames (prompt {ids_len} tokens, limit {generation_limit})...")
    # The stop token is sampled, not embedded: it ends generation before a frame is produced, as in HF generate.
    while (
        generated_frames < generation_limit
        and (selected_token_id not in STOP_TOKEN_SET or generated_frames < MIN_SEQ_LEN)
    ):
        binding_index = decode_step & 1
        binding = decode_bindings[binding_index]
        binding.bind_ortvalue_input(decode_history_argument.name, history_len)
        binding.bind_ortvalue_input(decode_fast_arguments[0].name, codec_token_main)
        binding.bind_ortvalue_input(decode_fast_arguments[1].name, last_hidden_state)
        bind_io_inputs(binding, decode_state_arguments, cached_state_tensors)
        binding.bind_ortvalue_input(decode_fast_arguments[2].name, generated_codec)
        if main_save_ids is not None:
            binding.bind_ortvalue_input(decode_save_argument.name, main_save_ids)

        step_start = time.perf_counter()
        decode_outputs = decode_bank.run(binding_index, run_options)
        main_time += time.perf_counter() - step_start

        cached_state_tensors = decode_outputs[:num_states]
        selected_token_id = int(decode_outputs[decode_token_pos].numpy().flat[0])
        if decode_save_id_pos is not None:
            main_save_ids = decode_outputs[decode_save_id_pos]
        generated_codec = decode_outputs[decode_generated_codec_pos]
        generated_frames += 1
        decode_step += 1
        if generated_frames % 50 == 0 or generated_frames == generation_limit:
            print_progress(f"Codec generation: {generated_frames}/{generation_limit} frames")

        last_hidden_state = decode_outputs[decode_last_hidden_pos]
        codec_token_main = decode_outputs[decode_token_pos]
        history_len = decode_outputs[decode_history_pos]

        if STREAMING:
            stream_frames.append(decode_outputs[decode_frame_pos].numpy().copy())
            if len(stream_frames) > stream_window_frames:
                stream_frames.pop(0)
            if len(stream_frames) == stream_window_frames:
                window = np.ascontiguousarray(np.concatenate(stream_frames, axis=1))
                stream_futures.append(stream_executor.submit(decode_stream_window, window))

    print_progress("Finalizing streaming waveform..." if STREAMING else "Decoding waveform...")
    if STREAMING:
        stream_executor.shutdown(wait=True)
    stream_results = [future.result() for future in stream_futures]
    if generated_frames == 0:
        generated_wav, decoder_time = np.zeros(0, dtype=wave_dtype), 0.0
    elif stream_results:
        chunks = []
        decoder_time = 0.0
        for index, (wave, elapsed) in enumerate(stream_results):
            chunks.append(wave if index == 0 else wave[-samples_per_codec_frame:])
            decoder_time += elapsed
        generated_wav = np.concatenate(chunks)
    else:
        generated_wav, decoder_time = decode_full_audio(generated_codec)

    if USE_AUDIO_NORMALIZER and generated_wav.size:
        generated_wav = audio_normalizer(generated_wav, wave_dtype)
    audio_duration = generated_wav.size / OUT_SAMPLE_RATE
    generation_time = main_time + decoder_time
    total_audio_samples += generated_wav.size
    total_generation_time += generation_time
    total_decoder_time += decoder_time
    rtf_text = f"{generation_time / audio_duration:.3f}" if audio_duration > 0 else "n/a"
    print(
        f"Target {target_index}/{len(TARGET_TTS)}: frames={generated_frames}, "
        f"audio={audio_duration:.2f}s, generation={generation_time:.2f}s, RTF={rtf_text}",
        flush=True,
    )
    save_generated_wav.extend((generated_wav, empty_segment))

if total_audio_samples:
    audio_duration = total_audio_samples / OUT_SAMPLE_RATE
    overall_rtf = (encoder_time + total_generation_time) / audio_duration
    print(
        f"Overall: encoder={encoder_time:.2f}s, decoder={total_decoder_time:.2f}s, "
        f"audio={audio_duration:.2f}s, RTF={overall_rtf:.3f}",
        flush=True,
    )
    print_progress(f"Writing generated audio: {GENERATED_AUDIO_PATH}")
    final_audio = np.concatenate(save_generated_wav)
    if final_audio.dtype == np.float16:
        final_audio = final_audio.astype(np.float32)
    output_subtype = "PCM_16" if np.issubdtype(final_audio.dtype, np.integer) else "FLOAT"
    sf.write(
        GENERATED_AUDIO_PATH,
        final_audio,
        OUT_SAMPLE_RATE,
        subtype=output_subtype,
        format="WAVEX",
    )
else:
    print_progress("No audio was generated; the model stopped before the first codec frame.")
print_progress(f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s.")
