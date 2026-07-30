import argparse
import concurrent.futures
import json
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
DOWNLOAD_PATH = str(Path.home() / "Downloads" / "Qwen3-TTS-12Hz-0.6B-Base")
GENERATED_AUDIO_PATH = str(SCRIPT_DIR / "generated.wav")

TARGET_TTS = [
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing iamj's AI technology.",
]
TTS_LANGUAGE = "Chinese"
PROMPT_AUDIO_PATH = reference_audio_path("qwen_tts")
PROMPT_TEXT = "对，这就是我，万人敬仰的太乙真人。"
SPEAKER_NAME = "Vivian"
INSTRUCT_TEXT = "Speak very happily"
VOICE_DESCRIPTION = "A young female with a warm, gentle tone and slight breathiness"

DECODE_STRATEGY = "penalty_greedy"  # greedy | penalty_greedy | sampling
MAX_FRAMES = 0                       # Optional generation cap; 0 uses graph capacity.
MIN_SEQ_LEN = 2                      # Minimum frames before a stop token is accepted.
PENALTY_RANGE = 5                    # Recent-token window for penalty-greedy decoding.
REPEAT_PENALTY = 0.8                 # Penalty-greedy repeat score multiplier.
TOP_K = 10                           # Sampling candidate count.
TOP_P = 0.95                         # Sampling nucleus probability.
TEMPERATURE = 0.8                    # Sampling temperature.
SAMPLING_REPETITION_PENALTY = 1.1    # Sampling repetition penalty.

STREAMING = False                    # Decode audio incrementally.
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
        print(f"[Qwen3-TTS] {message}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run compact Qwen3-TTS ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=Path(__file__).resolve().parent / "QwenTTS_Optimized",
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


def bind_io_outputs(binding, arguments, ort_device, device, device_id):
    buffers = []
    for argument in arguments:
        if static_io_shape(argument) is None:
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
        self.dynamic_outputs = tuple(
            argument
            for argument in self.output_arguments
            if static_io_shape(argument) is None
        )
        self.bindings = tuple(session.io_binding() for _ in range(count))
        self.output_buffers = tuple(
            bind_io_outputs(
                binding,
                self.output_arguments,
                ort_device,
                device,
                device_id,
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
            for argument in self.dynamic_outputs:
                binding._iobinding.bind_output(argument.name, self.ort_device)
        self.session.run_with_iobinding(binding, run_options=run_options)
        self.used[slot] = True
        return binding.get_outputs()


pipeline_started = time.perf_counter()
metadata_path = ONNX_FOLDER / "QwenTTS_Metadata.onnx"
print_progress(f"Reading package metadata: {metadata_path}")
metadata_model = onnx.load(metadata_path, load_external_data=False)
metadata = {entry.key: entry.value for entry in metadata_model.metadata_props}
del metadata_model
expected_metadata_keys = {
    "graph_layout",
    "mode",
    "language_id_map",
    "out_sample_rate",
    "max_seq_len",
    "stop_token_ids",
    "use_f16_kv",
    "compute_in_f32",
    "shared_initializer_model_file",
    "shared_initializer_data_file",
    "model_file_name_target_preprocess",
    "model_file_name_decoder",
    "model_file_name_decoder_stream",
    "vocab_size",
    *{
        f"model_file_name_main_prefill_{strategy}"
        for strategy in DECODE_STRATEGIES
    },
    *{
        f"model_file_name_decode_step_{strategy}"
        for strategy in DECODE_STRATEGIES
    },
}
metadata_mode = metadata.get("mode")
if metadata_mode == "voice_clone":
    expected_metadata_keys.update(
        {"in_sample_rate", "model_file_name_reference_preprocess"}
    )
elif metadata_mode == "custom_voice":
    expected_metadata_keys.update(
        {
            "instruction_prefix_token_ids",
            "instruction_suffix_token_ids",
            "speaker_id_map",
            "speaker_dialect_map",
            "dialect_language_id_map",
        }
    )
elif metadata_mode == "voice_design":
    expected_metadata_keys.update(
        {"instruction_prefix_token_ids", "instruction_suffix_token_ids"}
    )
else:
    pass
precision_flags = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
invalid_precision = {
    key: value for key, value in precision_flags.items() if value not in {"0", "1"}
}
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
        pass
def meta_int(key):
    return int(meta_str(key))


def meta_int_list(key):
    return [int(value) for value in meta_str(key).split(",") if value]


def meta_json(key):
    return json.loads(meta_str(key))


shared_model_path = ONNX_FOLDER / meta_str("shared_initializer_model_file")
shared_data_path = ONNX_FOLDER / meta_str("shared_initializer_data_file")
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


MODE = meta_str("mode")
OUT_SAMPLE_RATE = meta_int("out_sample_rate")
MAX_SEQ_LEN = meta_int("max_seq_len")
STOP_TOKEN_SET = set(meta_int_list("stop_token_ids"))
LANGUAGE_ID_MAP = meta_json("language_id_map")

language_key = TTS_LANGUAGE.lower()
language_id_value = int(LANGUAGE_ID_MAP[language_key])
speaker_id_value = 0
if MODE == "custom_voice":
    SPEAKER_ID_MAP = meta_json("speaker_id_map")
    SPEAKER_DIALECT_MAP = meta_json("speaker_dialect_map")
    DIALECT_LANGUAGE_ID_MAP = meta_json("dialect_language_id_map")
    speaker_key = SPEAKER_NAME.lower()
    speaker_id_value = int(SPEAKER_ID_MAP[speaker_key])
    dialect = SPEAKER_DIALECT_MAP.get(speaker_key, False)
    if dialect and language_key == "chinese":
        language_id_value = int(DIALECT_LANGUAGE_ID_MAP[dialect])

if DECODE_STRATEGY == "sampling":
    TOP_K = min(TOP_K, meta_int("vocab_size"))

print_progress("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    DOWNLOAD_PATH,
    trust_remote_code=True,
    fix_mistral_regex=True,
)

session_started = time.perf_counter()
session_total = 4 + int(MODE == "voice_clone") + int(STREAMING)
reference_session = (
    create_session(ONNX_FOLDER / meta_str("model_file_name_reference_preprocess"))
    if MODE == "voice_clone"
    else None
)
target_session = create_session(ONNX_FOLDER / meta_str("model_file_name_target_preprocess"))
prefill_session = create_session(
    ONNX_FOLDER / meta_str(f"model_file_name_main_prefill_{DECODE_STRATEGY}")
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

num_main_kv = len(prefill_output_arguments) - 3
decode_state_arguments = decode_arguments[:num_main_kv]

decode_cursor = num_main_kv
decode_trailing_argument = decode_arguments[decode_cursor]
decode_gather_argument = decode_arguments[decode_cursor + 1]
decode_history_argument = decode_arguments[decode_cursor + 2]
decode_cursor += 3
decode_save_argument = None
if DECODE_STRATEGY != "greedy":
    decode_save_argument = decode_arguments[decode_cursor]
    decode_cursor += 1
control_data = ()
if DECODE_STRATEGY == "penalty_greedy":
    control_data = (REPEAT_PENALTY, PENALTY_RANGE)
elif DECODE_STRATEGY == "sampling":
    control_data = (TEMPERATURE, TOP_K, TOP_P, SAMPLING_REPETITION_PENALTY)
decode_main_control_arguments = decode_arguments[decode_cursor:decode_cursor + len(control_data)]
decode_cursor += len(control_data)
decode_predictor_arguments = decode_arguments[decode_cursor:decode_cursor + 3]
decode_cursor += 3
decode_predictor_control_arguments = decode_arguments[decode_cursor:]

decode_last_hidden_pos = num_main_kv
decode_token_pos = num_main_kv + 1
decode_save_id_pos = num_main_kv + 2 if decode_save_argument else None
decode_history_pos = num_main_kv + 3 if decode_save_argument else num_main_kv + 2
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

target_language_argument = target_arguments[0]
if MODE == "custom_voice":
    target_speaker_argument = target_arguments[1]
    target_text_argument = target_arguments[2]
    target_instruction_argument = target_arguments[3]
else:
    target_speaker_argument = None
    target_text_argument = target_arguments[1]
    target_instruction_argument = target_arguments[2]

language_id = constants.get(target_language_argument, language_id_value)
if target_speaker_argument:
    speaker_id = constants.get(target_speaker_argument, speaker_id_value)

if MODE == "voice_clone":
    print_progress("Encoding reference audio and prompt text...")
    prompt_audio = load_prompt_audio(
        PROMPT_AUDIO_PATH,
        meta_int("in_sample_rate"),
        reference_arguments[0],
    )
    if USE_AUDIO_NORMALIZER:
        prompt_audio = audio_normalizer(prompt_audio, io_dtype(reference_arguments[0]))
    prompt_ids = tokenizer(
        PROMPT_TEXT,
        add_special_tokens=False,
        return_tensors="np",
    )["input_ids"]
    reference_values = (
        ort_from_io(reference_arguments[0], prompt_audio, device_type, DEVICE_ID),
        ort_from_io(reference_arguments[1], prompt_ids, device_type, DEVICE_ID),
    )
    bind_io_inputs(reference_bank.bindings[0], reference_arguments, reference_values)
    reference_start = time.perf_counter()
    reference_outputs = reference_bank.run(0, run_options)
    encoder_time = time.perf_counter() - reference_start
    print_progress(f"Reference conditioning ready in {encoder_time:.2f}s.")
else:
    reference_outputs = ()
    encoder_time = 0.0

instruction_text = ""
if MODE == "custom_voice" and INSTRUCT_TEXT:
    instruction_text = INSTRUCT_TEXT
elif MODE == "voice_design":
    instruction_text = VOICE_DESCRIPTION
if instruction_text:
    instruction_body_ids = tokenizer(
        instruction_text,
        add_special_tokens=False,
        return_tensors="np",
    )["input_ids"].reshape(-1)
    instruction_token_ids = np.concatenate(
        (
            np.asarray(meta_int_list("instruction_prefix_token_ids")),
            instruction_body_ids,
            np.asarray(meta_int_list("instruction_suffix_token_ids")),
        )
    ).reshape(1, -1)
else:
    instruction_token_ids = np.empty(0)
instruction_ids = constants.get(target_instruction_argument, instruction_token_ids)

target_binding.bind_ortvalue_input(target_language_argument.name, language_id)
target_binding.bind_ortvalue_input(target_instruction_argument.name, instruction_ids)
if MODE == "voice_clone":
    bind_io_inputs(target_binding, target_arguments[3:], reference_outputs)
elif MODE == "custom_voice":
    target_binding.bind_ortvalue_input(target_speaker_argument.name, speaker_id)

if DECODE_STRATEGY == "sampling":
    prefill_control_data = (TEMPERATURE, TOP_K, TOP_P)
    bind_io_inputs(
        prefill_binding,
        prefill_arguments[1:],
        tuple(
            constants.get(argument, value)
            for argument, value in zip(prefill_arguments[1:], prefill_control_data)
        ),
    )

gather_id_numpy = io_array(decode_gather_argument, 0)
gather_id_buffer = onnxruntime.OrtValue.ortvalue_from_numpy(
    gather_id_numpy,
    device_type,
    DEVICE_ID,
)
empty_generated_codec = ort_buffer_from_io(
    decode_predictor_arguments[2],
    device_type,
    DEVICE_ID,
    dynamic_dimensions=(0,),
)
main_control_values = tuple(
    constants.get(argument, value)
    for argument, value in zip(decode_main_control_arguments, control_data)
)
predictor_control_values = tuple(
    constants.get(argument, value)
    for argument, value in zip(decode_predictor_control_arguments, control_data)
)
for binding in decode_bindings:
    binding.bind_ortvalue_input(decode_gather_argument.name, gather_id_buffer)
    bind_io_inputs(binding, decode_main_control_arguments, main_control_values)
    bind_io_inputs(binding, decode_predictor_control_arguments, predictor_control_values)


def decode_stream_window(window_numpy):
    decoder_stream_input.update_inplace(window_numpy)
    slot, _ = decoder_stream_bank.select(0)
    start = time.perf_counter()
    outputs = decoder_stream_bank.run(slot, run_options)
    wave = outputs[0].numpy().reshape(-1).copy()
    return wave, time.perf_counter() - start


def decode_full_audio(generated_codec):
    slot, binding = decoder_bank.select(0)
    binding.bind_ortvalue_input(decoder_arguments[0].name, generated_codec)
    start = time.perf_counter()
    outputs = decoder_bank.run(slot, run_options)
    return outputs[0].numpy().reshape(-1).copy(), time.perf_counter() - start


save_generated_wav = []
wave_dtype = io_dtype(decoder_output_arguments[0])
empty_segment = np.zeros(int(OUT_SAMPLE_RATE * 0.2), dtype=wave_dtype)
total_audio_samples = 0
total_generation_time = 0.0
total_decoder_time = 0.0
if decoder_stream_session:
    stream_window_frames = (
        decoder_stream_arguments[0].shape[-1]
        // decode_output_arguments[decode_frame_pos].shape[-1]
    )
    samples_per_codec_frame = (
        decoder_stream_bank.output_arguments[0].shape[-1]
        // stream_window_frames
    )

print_progress(f"Prepared {len(TARGET_TTS)} text target(s).")
for target_index, target in enumerate(TARGET_TTS, start=1):
    print_progress(f"Starting target {target_index}/{len(TARGET_TTS)}: {target!r}")
    target_token_ids = tokenizer(
        target,
        add_special_tokens=False,
        return_tensors="np",
    )["input_ids"]
    target_ids = ort_from_io(
        target_text_argument,
        target_token_ids,
        device_type,
        DEVICE_ID,
    )
    target_binding.bind_ortvalue_input(target_text_argument.name, target_ids)
    target_outputs = target_bank.run(0, run_options)
    hidden_states, ids_len_value, trailing_text_hidden, trailing_len_value = target_outputs
    ids_len = int(ids_len_value.numpy().flat[0])
    trailing_len = int(trailing_len_value.numpy().flat[0])

    prefill_binding.bind_ortvalue_input(prefill_arguments[0].name, hidden_states)
    main_start = time.perf_counter()
    prefill_outputs = prefill_bank.run(0, run_options)
    main_time = time.perf_counter() - main_start
    cached_state_tensors = prefill_outputs[:num_main_kv]
    last_hidden_state, codec_token_main, history_len = prefill_outputs[num_main_kv:]
    main_save_ids = codec_token_main if DECODE_STRATEGY != "greedy" else None
    generated_codec = empty_generated_codec
    generated_frames = 0
    stream_frames = []
    stream_futures = []
    stream_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if STREAMING else None

    for binding in decode_bindings:
        binding.bind_ortvalue_input(decode_trailing_argument.name, trailing_text_hidden)
    control_rebinds_left = [2, 2]
    decode_step = 0
    selected_token_id = int(codec_token_main.numpy().flat[0])

    generation_limit = MAX_SEQ_LEN - ids_len
    if MAX_FRAMES > 0:
        generation_limit = min(generation_limit, MAX_FRAMES)
    print_progress(f"Generating codec frames (limit {generation_limit})...")
    while (
        generated_frames < generation_limit
        and (selected_token_id not in STOP_TOKEN_SET or generated_frames < MIN_SEQ_LEN)
    ):
        gather_id_numpy[0] = min(generated_frames, trailing_len)
        gather_id_buffer.update_inplace(gather_id_numpy)
        binding_index = decode_step & 1
        binding = decode_bindings[binding_index]
        if control_rebinds_left[binding_index]:
            control_rebinds_left[binding_index] -= 1
            binding.bind_ortvalue_input(decode_history_argument.name, history_len)
            binding.bind_ortvalue_input(decode_predictor_arguments[0].name, codec_token_main)
            binding.bind_ortvalue_input(decode_predictor_arguments[1].name, last_hidden_state)
        bind_io_inputs(binding, decode_state_arguments, cached_state_tensors)
        binding.bind_ortvalue_input(decode_predictor_arguments[2].name, generated_codec)
        if main_save_ids is not None:
            binding.bind_ortvalue_input(decode_save_argument.name, main_save_ids)

        step_start = time.perf_counter()
        decode_outputs = decode_bank.run(binding_index, run_options)
        step_elapsed = time.perf_counter() - step_start
        main_time += step_elapsed

        cached_state_tensors = decode_outputs[:num_main_kv]
        selected_token_id = int(decode_outputs[decode_token_pos].numpy().flat[0])
        if decode_save_id_pos is not None:
            main_save_ids = decode_outputs[decode_save_id_pos]
        generated_codec = decode_outputs[decode_generated_codec_pos]
        generated_frames += 1
        decode_step += 1
        if generated_frames % 50 == 0 or generated_frames == generation_limit:
            print_progress(
                f"Codec generation: {generated_frames}/{generation_limit} frames"
            )

        if any(control_rebinds_left):
            last_hidden_state = decode_outputs[decode_last_hidden_pos]
            codec_token_main = decode_outputs[decode_token_pos]
            history_len = decode_outputs[decode_history_pos]

        if STREAMING:
            frame_numpy = decode_outputs[decode_frame_pos].numpy().copy()
            stream_frames.append(frame_numpy)
            if len(stream_frames) > stream_window_frames:
                stream_frames.pop(0)
            if len(stream_frames) == stream_window_frames:
                window = np.ascontiguousarray(np.concatenate(stream_frames, axis=1))
                stream_futures.append(stream_executor.submit(decode_stream_window, window))

    print_progress(
        "Finalizing streaming waveform..." if STREAMING else "Decoding waveform..."
    )
    if STREAMING:
        stream_executor.shutdown(wait=True)
        stream_results = [future.result() for future in stream_futures]
        if stream_results:
            chunks = []
            decoder_time = 0.0
            for index, (wave, elapsed) in enumerate(stream_results):
                chunks.append(wave if index == 0 else wave[-samples_per_codec_frame:])
                decoder_time += elapsed
            generated_wav = np.concatenate(chunks)
        else:
            generated_wav, decoder_time = decode_full_audio(generated_codec)
    else:
        generated_wav, decoder_time = decode_full_audio(generated_codec)

    if USE_AUDIO_NORMALIZER:
        generated_wav = audio_normalizer(generated_wav, wave_dtype)
    audio_duration = generated_wav.size / OUT_SAMPLE_RATE
    generation_time = main_time + decoder_time
    total_audio_samples += generated_wav.size
    total_generation_time += generation_time
    total_decoder_time += decoder_time
    print(
        f"Target {target_index}/{len(TARGET_TTS)}: frames={generated_frames}, "
        f"audio={audio_duration:.2f}s, generation={generation_time:.2f}s, "
        f"RTF={generation_time / audio_duration:.3f}",
        flush=True,
    )
    save_generated_wav.extend((generated_wav, empty_segment))

if save_generated_wav:
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
print_progress(f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s.")