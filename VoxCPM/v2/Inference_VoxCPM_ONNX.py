"""Run the VoxCPM2 ONNX package."""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import time
from pathlib import Path

import inflect
import numpy as np
import onnx
import onnxruntime
import soundfile as sf
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import LlamaTokenizerFast
from wetext import Normalizer

from Shared_Weights import attach_shared_initializers


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from Example_Audio import reference_audio_path as demo_reference_audio


MODES = ("voice_design", "continuation", "reference_only", "combined")
PROVIDERS = ("cpu", "cuda", "dml", "openvino")


# ============================== Configuration ==============================
# Edit these values directly; the CLI is reserved for selecting the ONNX folder.
MODEL_FOLDER = Path.home() / "Downloads" / "VoxCPM2"  # Tokenizer/model folder.

# Multiple modes may be enabled together. Each enabled mode writes one WAV.
RUN_VOICE_DESIGN = True
VOICE_DESIGN_TARGET_TEXTS = [
    "(用年轻女声说话)大家好，我现在正在大可奇奇体验AI科技。",
]
VOICE_DESIGN_OUTPUT_PATH = SCRIPT_DIR / "generated.wav"

RUN_CONTINUATION = True
CONTINUATION_TARGET_TEXTS = [
    "大家好，我现在正在大可奇奇体验AI科技。",
]
CONTINUATION_OUTPUT_PATH = SCRIPT_DIR / "generated_continuation.wav"

RUN_REFERENCE_ONLY = True
REFERENCE_ONLY_TARGET_TEXTS = [
    "大家好，我现在正在大可奇奇体验AI科技。",
]
REFERENCE_ONLY_OUTPUT_PATH = SCRIPT_DIR / "generated_reference_only.wav"

RUN_COMBINED = True
COMBINED_TARGET_TEXTS = [
    "大家好，我现在正在大可奇奇体验AI科技。",
]
COMBINED_OUTPUT_PATH = SCRIPT_DIR / "generated_combined.wav"

PROMPT_TEXT = "对，这就是我，万人敬仰的太乙真人。"  # Continuation and combined.
REFERENCE_AUDIO_PATH = None  # None uses the bundled reference when needed.
PROMPT_AUDIO_PATH = None     # None uses the bundled prompt when needed.

SEED = 9527                    # Random seed for reproducible generation.
CFG = 2.0                      # Classifier-free guidance strength.
STREAMING = False              # Use the streaming VAE decoder.
MAX_FRAMES = 0                 # Per-sentence frame cap; 0 uses the computed limit.
MIN_SEQ_LEN = 2                # Minimum frames before accepting a stop token.
DECODE_LIMIT_FACTOR = 6        # Text-token multiplier used for the frame limit.

PROVIDER = "cpu"               # cpu | cuda | dml | openvino
DEVICE_ID = 0                  # Accelerator device index.
MAX_THREADS = 0                # CPU parallel threads; 0 lets ONNX Runtime choose.
ORT_LOG = False                # Enable ONNX Runtime logging.
NORMALIZE_AUDIO = False        # Normalize input and generated audio loudness.
USE_TEXT_NORMALIZER = True     # Normalize text before tokenization.
SHOW_PROGRESS = True           # Print pipeline stages and loop progress.
# ===========================================================================


def print_progress(message):
    if SHOW_PROGRESS:
        print(f"[VoxCPM 2] {message}", flush=True)


def _mode_configuration():
    return {
        "voice_design": (
            RUN_VOICE_DESIGN,
            VOICE_DESIGN_TARGET_TEXTS,
            VOICE_DESIGN_OUTPUT_PATH,
        ),
        "continuation": (
            RUN_CONTINUATION,
            CONTINUATION_TARGET_TEXTS,
            CONTINUATION_OUTPUT_PATH,
        ),
        "reference_only": (
            RUN_REFERENCE_ONLY,
            REFERENCE_ONLY_TARGET_TEXTS,
            REFERENCE_ONLY_OUTPUT_PATH,
        ),
        "combined": (
            RUN_COMBINED,
            COMBINED_TARGET_TEXTS,
            COMBINED_OUTPUT_PATH,
        ),
    }


def _enabled_modes(configuration):
    return tuple(
        mode for mode in MODES if configuration[mode][0]
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "VoxCPM_Optimized",
    )
    return parser.parse_args()


ARGS = _parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()


CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def _contains_chinese(text):
    return bool(CHINESE_PATTERN.search(text))


def _replace_corner_mark(text):
    return (
        text.replace("²", "平方")
        .replace("³", "立方")
        .replace("√", "根号")
        .replace("≈", "约等于")
        .replace("<", "小于")
    )


def _remove_bracket(text):
    return (
        text.replace("（", " ")
        .replace("）", " ")
        .replace("【", " ")
        .replace("】", " ")
        .replace("`", "")
        .replace("——", " ")
    )


def _spell_out_number(text, parser):
    output = []
    start = None
    for index, character in enumerate(text):
        if character.isdigit():
            if start is None:
                start = index
        else:
            if start is not None:
                output.append(parser.number_to_words(text[start:index]))
                start = None
            output.append(character)
    if start is not None:
        output.append(parser.number_to_words(text[start:]))
    return "".join(output)


def _replace_blank(text):
    output = []
    for index, character in enumerate(text):
        if character == " " and 0 < index < len(text) - 1:
            if (
                text[index + 1].isascii()
                and text[index + 1] != " "
                and text[index - 1].isascii()
                and text[index - 1] != " "
            ):
                output.append(character)
        elif character != " ":
            output.append(character)
    return "".join(output)


def _clean_markdown(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^(\s*)-\s+", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def _clean_text(text):
    text = _clean_markdown(text)
    text = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]+").sub("", text)
    return text.replace("\n", " ").replace("\t", " ")


class TextNormalizer:
    def __init__(self):
        self.zh = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        self.en = Normalizer(lang="en", operator="tn")
        self.inflect = inflect.engine()

    def normalize(self, text):
        language = "zh" if _contains_chinese(text) else "en"
        text = _clean_text(text)
        if language == "zh":
            text = text.replace("=", "等于")
            if re.search(r"([\d$%^*_+≥≤≠×÷?=])", text):
                text = re.sub(r"(?<=[a-zA-Z0-9])-(?=\d)", " - ", text)
            return _remove_bracket(_replace_corner_mark(_replace_blank(self.zh.normalize(text))))
        return _spell_out_number(self.en.normalize(text), self.inflect)


def _mask_multichar_chinese_tokens(base_tokenizer):
    multichar_tokens = {
        token
        for token in base_tokenizer.get_vocab()
        if len(token.replace("\u2581", "")) >= 2
        and all("\u4e00" <= character <= "\u9fff" for character in token.replace("\u2581", ""))
    }

    class CharacterTokenizer:
        def __call__(self, text):
            tokens = base_tokenizer.tokenize(text)
            processed = []
            for token in tokens:
                clean = token.replace("\u2581", "")
                processed.extend(list(clean) if clean in multichar_tokens else [token])
            return base_tokenizer.convert_tokens_to_ids(processed)

    return CharacterTokenizer()


def _metadata(path):
    model = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}


def _meta_int(metadata, key):
    try:
        return int(metadata[key])
    except KeyError as error:
        raise ValueError(f"Missing required VoxCPM2 metadata key: {key!r}.") from error
def _numpy_dtype(value):
    match = re.fullmatch(r"tensor\(([^)]+)\)", value.type)
    try:
        element_type = onnx.TensorProto.DataType.Value(match.group(1).upper())
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))
    except (AttributeError, ValueError, TypeError) as error:
        raise ValueError(f"Unsupported ONNX tensor type: {value.type!r}.") from error
def _io_shape(value, dynamic_shape=()):
    dynamic_shape = iter(dynamic_shape)
    shape = []
    for dimension in value.shape:
        if isinstance(dimension, int) and dimension >= 0:
            shape.append(dimension)
        else:
            try:
                shape.append(int(next(dynamic_shape)))
            except StopIteration as error:
                raise ValueError(
                    f"Missing dynamic dimension for ONNX value {value.name!r}."
                ) from error
    try:
        next(dynamic_shape)
    except StopIteration:
        return tuple(shape)
    raise ValueError(f"Too many dynamic dimensions supplied for ONNX value {value.name!r}.")
def _device_id(device_type):
    return DEVICE_ID if device_type != "cpu" else 0


def _empty_ort_value(value, device_type, dynamic_shape=()):
    return onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        _io_shape(value, dynamic_shape),
        _numpy_dtype(value),
        device_type,
        _device_id(device_type),
    )


def _data_shape(value, array):
    rank = len(value.shape)
    if array.ndim == rank:
        for actual, declared in zip(array.shape, value.shape):
            if isinstance(declared, int) and declared >= 0 and actual != declared:
                break
        else:
            return tuple(array.shape)

    dynamic_axes = [
        axis
        for axis, dimension in enumerate(value.shape)
        if not isinstance(dimension, int) or dimension < 0
    ]
    static_size = 1
    for dimension in value.shape:
        if isinstance(dimension, int) and dimension >= 0:
            static_size *= dimension
    return _io_shape(value, (array.size // static_size,))


def _ort_value_for(value, data, device_type):
    array = np.asarray(data, dtype=_numpy_dtype(value))
    array = np.ascontiguousarray(array.reshape(_data_shape(value, array)))
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        array,
        device_type,
        _device_id(device_type),
    )


def _filled_ort_value(value, fill_value, device_type):
    array = np.full(_io_shape(value), fill_value, dtype=_numpy_dtype(value))
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        array,
        device_type,
        _device_id(device_type),
    )


def _common_argument(arguments):
    first, *rest = arguments
    contract = first.type, tuple(
        dimension if isinstance(dimension, int) and dimension >= 0 else None
        for dimension in first.shape
    )
    for value in rest:
        value_contract = value.type, tuple(
            dimension if isinstance(dimension, int) and dimension >= 0 else None
            for dimension in value.shape
        )
        if value_contract != contract:
            raise RuntimeError(
                f"Incompatible state ABI: {first.name}={contract!r}, "
                f"{value.name}={value_contract!r}."
            )
    return first


def _matching_prefix(*argument_groups):
    count = 0
    for arguments in zip(*argument_groups):
        try:
            _common_argument(arguments)
        except RuntimeError:
            break
        count += 1
    return count


def _is_static(value):
    return all(isinstance(dimension, int) and dimension >= 0 for dimension in value.shape)


class _IoRunner:
    def __init__(
        self,
        session,
        run_options,
        device_type,
        ort_device_type,
        binding_count=1,
    ):
        self.session = session
        self.run_options = run_options
        self.device_type = device_type
        self.ort_device_type = ort_device_type
        self.inputs = tuple(session.get_inputs())
        self.outputs = tuple(session.get_outputs())
        self.use_fixed_output_buffers = device_type != "cuda"
        self.auto_bound_outputs = tuple(
            value
            for value in self.outputs
            if not self.use_fixed_output_buffers or not _is_static(value)
        )
        self.bindings = []
        self.input_buffers = []
        self.output_buffers = []
        self.has_run = [False] * binding_count
        for _ in range(binding_count):
            binding = session.io_binding()
            buffers = []
            for value in self.outputs:
                if self.use_fixed_output_buffers and _is_static(value):
                    buffer = _empty_ort_value(value, device_type)
                    binding.bind_ortvalue_output(value.name, buffer)
                    buffers.append(buffer)
                else:
                    binding._iobinding.bind_output(value.name, ort_device_type)
            self.bindings.append(binding)
            self.input_buffers.append({})
            self.output_buffers.append(tuple(buffers))

    def bind(self, argument, value, binding_index=0):
        self.bindings[binding_index].bind_ortvalue_input(argument.name, value)
        self.input_buffers[binding_index][argument.name] = value

    def bind_all(self, argument, value):
        for binding_index in range(len(self.bindings)):
            self.bind(argument, value, binding_index)

    def run(self, binding_index=0):
        binding = self.bindings[binding_index]
        if self.has_run[binding_index]:
            for value in self.auto_bound_outputs:
                binding._iobinding.bind_output(value.name, self.ort_device_type)
        self.session.run_with_iobinding(binding, run_options=self.run_options)
        self.has_run[binding_index] = True
        return binding.get_outputs()


def _provider_config():
    if PROVIDER == "cuda":
        return (
            ["CUDAExecutionProvider"],
            [{"device_id": DEVICE_ID}],
            "cuda",
            C.OrtDevice.cuda(),
        )
    if PROVIDER == "dml":
        return (
            ["DmlExecutionProvider"],
            [{"device_id": DEVICE_ID}],
            "dml",
            C.OrtDevice.dml(),
        )
    if PROVIDER == "openvino":
        return (
            ["OpenVINOExecutionProvider"],
            [{"device_type": "CPU", "precision": "ACCURACY"}],
            "cpu",
            C.OrtDevice.cpu(),
        )
    return ["CPUExecutionProvider"], None, "cpu", C.OrtDevice.cpu()


def _session_options():
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    # ORT_ENABLE_ALL changed the same-noise trajectory. Keep runtime
    # rewrites off; offline package optimization is validated separately.
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
    }.items():
        options.add_session_config_entry(key, value)
    return options


def _model_paths(metadata, modes, needs_encoder, needs_stream):
    keys = {
        "decode": "model_file_name_decode_step",
        "vae": "model_file_name_vae_decoder",
    }
    if needs_encoder:
        keys["vae_encoder"] = "model_file_name_vae_encoder"
    if needs_stream:
        keys["stream"] = "model_file_name_vae_decoder_stream"
    paths = {name: ONNX_FOLDER / metadata[key] for name, key in keys.items()}
    for mode in modes:
        paths[f"prefill_{mode}"] = ONNX_FOLDER / metadata[
            f"model_file_name_main_prefill_{mode}"
        ]
    shared_model = ONNX_FOLDER / metadata["shared_initializer_model_file"]
    shared_data = ONNX_FOLDER / metadata["shared_initializer_data_file"]
    return paths, shared_model


def _create_session(path, options, providers, provider_options):
    return onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=providers,
        provider_options=provider_options,
    )


def _audio_normalizer(audio, target=None):
    output_dtype = audio.dtype
    audio = audio.astype(np.float32)
    if target is None:
        target = 8192.0 if np.issubdtype(output_dtype, np.integer) else 0.25
    rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
    audio *= target / (rms + 1e-7)
    if np.issubdtype(output_dtype, np.integer):
        limits = np.iinfo(output_dtype)
        np.clip(audio, limits.min, limits.max, out=audio)
    return audio.astype(output_dtype)


def _load_audio(path, sample_rate, dtype):
    audio = np.asarray(
        AudioSegment.from_file(path)
        .set_channels(1)
        .set_frame_rate(sample_rate)
        .set_sample_width(2)
        .get_array_of_samples(),
        dtype=np.int16,
    )
    if NORMALIZE_AUDIO:
        audio = _audio_normalizer(audio)
    if np.issubdtype(dtype, np.floating):
        return (audio.astype(np.float32) * (1.0 / 32768.0)).astype(dtype)
    return audio.astype(dtype, copy=False)


def _prepare_runtime(modes):
    metadata_path = ONNX_FOLDER / "VoxCPM2_Metadata.onnx"
    print_progress(f"Reading package metadata: {metadata_path}")
    metadata = _metadata(metadata_path)
    expected_metadata_keys = {
        "graph_layout",
        "model_file_name_vae_encoder",
        "model_file_name_decode_step",
        "model_file_name_vae_decoder",
        "model_file_name_vae_decoder_stream",
        "model_file_name_metadata",
        "shared_initializer_model_file",
        "shared_initializer_data_file",
        "in_sample_rate",
        "out_sample_rate",
        "max_seq_len",
        "stop_token_ids",
        "audio_start_token_id",
        *{
            f"model_file_name_main_prefill_{mode}"
            for mode in MODES
        },
    }
    missing_metadata = sorted(expected_metadata_keys - metadata.keys())
    if missing_metadata:
        raise ValueError(
            f"{metadata_path.name} is missing required metadata key(s): {missing_metadata}."
        )
    default_audio = Path(demo_reference_audio("voxcpm"))
    reference_audio = Path(REFERENCE_AUDIO_PATH or default_audio).expanduser().resolve()
    prompt_audio = Path(PROMPT_AUDIO_PATH or default_audio).expanduser().resolve()
    all_condition_paths = {
        "voice_design": (),
        "continuation": (prompt_audio,),
        "reference_only": (reference_audio,),
        "combined": (reference_audio, prompt_audio),
    }
    condition_paths = {mode: all_condition_paths[mode] for mode in modes}
    unique_condition_paths = tuple(
        dict.fromkeys(
            path
            for mode in modes
            for path in condition_paths[mode]
        )
    )
    paths, shared_model = _model_paths(
        metadata,
        modes,
        needs_encoder=bool(unique_condition_paths),
        needs_stream=STREAMING,
    )
    options = _session_options()
    providers, provider_options, device_type, ort_device_type = _provider_config()
    _ort_device_type = C.OrtDevice(
        ort_device_type,
        C.OrtDevice.default_memory(),
        _device_id(device_type),
    )
    shared_started = time.perf_counter()
    print_progress("Attaching shared ONNX initializers...")
    shared_refs = attach_shared_initializers(options, shared_model)
    print_progress(
        f"Shared ONNX initializers ready in "
        f"{time.perf_counter() - shared_started:.2f}s."
    )
    session_started = time.perf_counter()
    sessions = {}
    for index, (name, path) in enumerate(paths.items(), start=1):
        print_progress(f"Loading ONNX graph {index}/{len(paths)}: {path.name}")
        sessions[name] = _create_session(path, options, providers, provider_options)
    print_progress(
        f"ONNX Runtime sessions ready in "
        f"{time.perf_counter() - session_started:.2f}s; "
        f"providers={sessions['decode'].get_providers()}."
    )

    run_options = onnxruntime.RunOptions()
    run_options.log_severity_level = 0 if ORT_LOG else 4
    runners = {
        name: _IoRunner(
            session,
            run_options,
            device_type,
            _ort_device_type,
            binding_count=(
                2
                if name == "decode"
                else len(unique_condition_paths)
                if name == "vae_encoder"
                else 1
            ),
        )
        for name, session in sessions.items()
    }
    decode = runners["decode"]
    mode_contexts = {}
    state_counts = set()
    for mode in modes:
        prefill = runners[f"prefill_{mode}"]
        state_count = _matching_prefix(prefill.outputs, decode.inputs, decode.outputs)
        state_counts.add(state_count)
        text_input, *feature_and_control_inputs = prefill.inputs
        feature_inputs = feature_and_control_inputs[:-3]
        control_inputs = feature_and_control_inputs[-3:]
        mode_contexts[mode] = {
            "prefill": prefill,
            "text_input": text_input,
            "feature_inputs": feature_inputs,
            "control_inputs": control_inputs,
        }
    state_count = state_counts.pop()
    state_inputs = decode.inputs[:state_count]
    (
        previous_latent_input,
        decode_length_input,
        decode_noise_input,
        decode_cfg_input,
        decode_cfg_minus_input,
        generated_latents_input,
    ) = decode.inputs[state_count:]
    prefill_noise_inputs = tuple(
        mode_contexts[mode]["control_inputs"][0] for mode in modes
    )
    prefill_cfg_inputs = tuple(
        mode_contexts[mode]["control_inputs"][1] for mode in modes
    )
    prefill_cfg_minus_inputs = tuple(
        mode_contexts[mode]["control_inputs"][2] for mode in modes
    )
    noise_argument = _common_argument((*prefill_noise_inputs, decode_noise_input))
    cfg_argument = _common_argument((*prefill_cfg_inputs, decode_cfg_input))
    cfg_minus_argument = _common_argument(
        (*prefill_cfg_minus_inputs, decode_cfg_minus_input)
    )
    noise_shape = _io_shape(noise_argument)
    noise_numpy = np.empty(noise_shape, dtype=_numpy_dtype(noise_argument))
    noise_value = _empty_ort_value(noise_argument, device_type)
    cfg_value = _filled_ort_value(cfg_argument, CFG, device_type)
    cfg_minus_value = _filled_ort_value(cfg_minus_argument, 1.0 - CFG, device_type)
    for mode in modes:
        prefill = mode_contexts[mode]["prefill"]
        for prefill_input, value in zip(
            mode_contexts[mode]["control_inputs"],
            (noise_value, cfg_value, cfg_minus_value),
        ):
            prefill.bind(prefill_input, value)
    for decode_input, value in (
        (decode_noise_input, noise_value),
        (decode_cfg_input, cfg_value),
        (decode_cfg_minus_input, cfg_minus_value),
    ):
        decode.bind_all(decode_input, value)

    in_sample_rate = _meta_int(metadata, "in_sample_rate")
    feature_cache = {}
    if unique_condition_paths:
        vae_encoder = runners["vae_encoder"]
        audio_input = vae_encoder.inputs[0]
        audio_dtype = _numpy_dtype(audio_input)
        for index, path in enumerate(unique_condition_paths):
            condition_started = time.perf_counter()
            print_progress(
                f"Encoding conditioning audio {index + 1}/"
                f"{len(unique_condition_paths)}: {path}"
            )
            audio = _load_audio(path, in_sample_rate, audio_dtype)
            audio_value = _ort_value_for(audio_input, audio, device_type)
            vae_encoder.bind(audio_input, audio_value, index)
            feature_value = vae_encoder.run(index)[0]
            feature_cache[path] = feature_value
            print_progress(
                f"Conditioning audio {index + 1}/{len(unique_condition_paths)} ready in "
                f"{time.perf_counter() - condition_started:.2f}s."
            )
        for mode in modes:
            prefill = mode_contexts[mode]["prefill"]
            for feature_input, path in zip(
                mode_contexts[mode]["feature_inputs"],
                condition_paths[mode],
            ):
                prefill.bind(feature_input, feature_cache[path])

    print_progress("Loading the text frontend...")
    tokenizer = _mask_multichar_chinese_tokens(
        LlamaTokenizerFast.from_pretrained(MODEL_FOLDER)
    )
    normalizer = TextNormalizer() if USE_TEXT_NORMALIZER else None
    return {
        "metadata": metadata,
        "runners": runners,
        "mode_contexts": mode_contexts,
        "state_count": state_count,
        "state_inputs": state_inputs,
        "decode_inputs": (
            previous_latent_input,
            decode_length_input,
            generated_latents_input,
        ),
        "noise_shape": noise_shape,
        "noise_numpy": noise_numpy,
        "noise_value": noise_value,
        "device_type": device_type,
        "tokenizer": tokenizer,
        "normalizer": normalizer,
        "shared_refs": shared_refs,
    }


def _run_mode(runtime, mode, target_texts, output_path):
    mode_started = time.perf_counter()
    metadata = runtime["metadata"]
    runners = runtime["runners"]
    mode_context = runtime["mode_contexts"][mode]
    prefill = mode_context["prefill"]
    text_input = mode_context["text_input"]
    decode = runners["decode"]
    vae = runners["vae"]
    state_count = runtime["state_count"]
    state_inputs = runtime["state_inputs"]
    (
        previous_latent_input,
        decode_length_input,
        generated_latents_input,
    ) = runtime["decode_inputs"]
    noise_shape = runtime["noise_shape"]
    noise_numpy = runtime["noise_numpy"]
    noise_value = runtime["noise_value"]
    device_type = runtime["device_type"]
    tokenizer = runtime["tokenizer"]
    normalizer = runtime["normalizer"]
    random_state = np.random.RandomState(SEED)

    def draw_noise():
        noise_numpy[...] = random_state.standard_normal(noise_shape)
        noise_value.update_inplace(noise_numpy)

    seed = SEED
    cfg = CFG
    streaming = STREAMING
    max_frames_override = MAX_FRAMES
    prompt_text = PROMPT_TEXT
    output_audio = []
    sentence_metrics = []
    total_frames = 0
    total_decode_calls = 0
    generation_start = time.perf_counter()
    stop_tokens = {int(item) for item in metadata["stop_token_ids"].split(",") if item}
    audio_start_token = _meta_int(metadata, "audio_start_token_id")
    min_seq_len = MIN_SEQ_LEN
    max_seq_len = _meta_int(metadata, "max_seq_len")
    decode_limit_factor = DECODE_LIMIT_FACTOR
    out_sample_rate = _meta_int(metadata, "out_sample_rate")
    prompt = normalizer.normalize(prompt_text) if normalizer and prompt_text else prompt_text
    if streaming:
        stream = runners["stream"]
        previous_stream_input, current_stream_input = stream.inputs
        stream_audio_output = stream.outputs[0]
        crop_samples = _io_shape(stream_audio_output)[-1] // len(stream.inputs)

        def decode_stream_pair(previous_numpy, current_numpy):
            previous_value = _ort_value_for(
                previous_stream_input,
                previous_numpy,
                device_type,
            )
            current_value = _ort_value_for(
                current_stream_input,
                current_numpy,
                device_type,
            )
            stream.bind(previous_stream_input, previous_value)
            stream.bind(current_stream_input, current_value)
            stream_values = stream.run()
            return stream_values[0].numpy().copy()

    vae_input = vae.inputs[0]
    output_dtype = _numpy_dtype(vae.outputs[0])

    print_progress(f"Prepared {len(target_texts)} text target(s).")
    for sentence_index, sentence in enumerate(target_texts, start=1):
        print_progress(
            f"Starting target {sentence_index}/{len(target_texts)}: {sentence!r}"
        )
        sentence = normalizer.normalize(sentence) if normalizer else sentence
        target_text = re.sub(r"\s+", " ", sentence.replace("\n", " ")).strip()
        target_ids = tokenizer(target_text)
        if mode in {"continuation", "combined"}:
            full_text = re.sub(r"\s+", " ", (prompt + target_text).replace("\n", " ")).strip()
            text_ids = tokenizer(full_text) + [audio_start_token]
        else:
            text_ids = target_ids + [audio_start_token]
        text_value = _ort_value_for(
            text_input,
            text_ids,
            device_type,
        )
        prefill.bind(text_input, text_value)
        draw_noise()
        start = time.perf_counter()
        values = prefill.run()
        state = list(values[:state_count])
        previous_latent, stop_value, kv_seq_len = values[state_count:]
        generated_latents = previous_latent
        stop_flag = stop_value.numpy().flat[0]
        emitted_frames = 1
        decode_calls = 0

        ids_length = int(kv_seq_len.numpy().item())
        max_frames = min(
            len(target_ids) * decode_limit_factor + 10,
            max_seq_len - ids_length,
        )
        if max_frames_override > 0:
            max_frames = min(max_frames, max_frames_override)
        if max_frames <= 0:
            print_progress("Generation skipped: prompt leaves no frame capacity.")
            continue

        stream_futures = []
        stream_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
            if streaming
            else None
        )
        print_progress(f"Generating acoustic frames (limit {max_frames})...")
        while True:
            zero_based_frame = emitted_frames - 1
            if zero_based_frame >= min_seq_len and stop_flag in stop_tokens:
                break
            if emitted_frames >= max_frames:
                break

            binding_index = decode_calls & 1
            for argument, value in zip(state_inputs, state):
                decode.bind(argument, value, binding_index)
            for argument, value in (
                (previous_latent_input, previous_latent),
                (decode_length_input, kv_seq_len),
                (generated_latents_input, generated_latents),
            ):
                decode.bind(argument, value, binding_index)
            draw_noise()
            decoded = decode.run(binding_index)
            state = list(decoded[:state_count])
            current_latent, stop_value, kv_seq_len, generated_latents = decoded[
                state_count:
            ]
            stop_flag = stop_value.numpy().flat[0]

            if streaming:
                previous_numpy = previous_latent.numpy().copy()
                current_numpy = current_latent.numpy().copy()
                stream_futures.append(
                    stream_executor.submit(
                        decode_stream_pair,
                        previous_numpy,
                        current_numpy,
                    )
                )

            previous_latent = current_latent
            emitted_frames += 1
            decode_calls += 1
            if emitted_frames % 25 == 0 or emitted_frames == max_frames:
                print_progress(
                    f"Acoustic generation: {emitted_frames}/{max_frames} frames"
                )

        if streaming:
            print_progress("Finalizing streaming waveform...")
            stream_executor.shutdown(wait=True)
            stream_results = [future.result() for future in stream_futures]
            if stream_results:
                stream_chunks = [
                    chunk if index == 0 else chunk[..., crop_samples:]
                    for index, chunk in enumerate(stream_results)
                ]
                sentence_audio = np.concatenate(stream_chunks, axis=-1)
            else:
                print_progress("Decoding waveform...")
                vae.bind(vae_input, generated_latents)
                vae_values = vae.run()
                sentence_audio = vae_values[0].numpy()
        else:
            print_progress("Decoding waveform...")
            vae.bind(vae_input, generated_latents)
            vae_values = vae.run()
            sentence_audio = vae_values[0].numpy()

        elapsed = time.perf_counter() - start
        output_audio.append(sentence_audio.reshape(-1))
        total_frames += emitted_frames
        total_decode_calls += decode_calls
        print(
            f"Target {sentence_index}/{len(target_texts)} ({mode}): "
            f"{emitted_frames} frames, {decode_calls} DecodeStep calls, "
            f"{emitted_frames / max(elapsed, 1e-9):.3f} frame/s",
            flush=True,
        )
        sentence_metrics.append(
            {
                "text": sentence,
                "emitted_frames": emitted_frames,
                "decode_step_calls": decode_calls,
                "elapsed_seconds": elapsed,
                "frames_per_second": emitted_frames / max(elapsed, 1e-9),
                "audio_samples": int(sentence_audio.size),
            }
        )

    elapsed = time.perf_counter() - generation_start
    audio = np.concatenate(output_audio) if output_audio else np.empty((0,), dtype=output_dtype)
    if NORMALIZE_AUDIO and audio.size:
        audio = _audio_normalizer(audio)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print_progress(f"Writing generated audio: {output_path}")
    if audio.dtype == np.float16:
        audio = audio.astype(np.float32)
    output_subtype = "PCM_16" if np.issubdtype(audio.dtype, np.integer) else "FLOAT"
    sf.write(output_path, audio, out_sample_rate, subtype=output_subtype)
    duration = audio.size / out_sample_rate
    additional_frames = max(total_frames - len(output_audio), 0)
    print(f"Saved: {output_path}", flush=True)
    print(f"Mode: {mode}; seed: {seed}; CFG: {cfg}; streaming: {streaming}", flush=True)
    print(
        f"Frames: {total_frames}; DecodeStep calls: {total_decode_calls}; "
        f"calls/additional-frame: "
        f"{total_decode_calls / max(additional_frames, 1):.3f}",
        flush=True,
    )
    print(f"Generation time: {elapsed:.3f}s", flush=True)
    print(f"RTF: {elapsed / max(duration, 1e-9):.3f}", flush=True)

    print_progress(
        f"Mode {mode} complete in {time.perf_counter() - mode_started:.2f}s."
    )


def main():
    pipeline_started = time.perf_counter()
    configuration = _mode_configuration()
    modes = _enabled_modes(configuration)
    print_progress(f"Enabled modes: {', '.join(modes)}")
    runtime = _prepare_runtime(modes)
    for index, mode in enumerate(modes, start=1):
        _, target_texts, output_path = configuration[mode]
        print_progress(f"Starting mode {index}/{len(modes)}: {mode}")
        _run_mode(runtime, mode, target_texts, output_path)
    print_progress(
        f"All enabled modes complete in "
        f"{time.perf_counter() - pipeline_started:.2f}s."
    )


if __name__ == "__main__":
    main()
