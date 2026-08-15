import argparse
import concurrent.futures
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import inflect
import numpy as np
import onnx
import onnxruntime
import regex
import soundfile as sf
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import LlamaTokenizerFast
from wetext import Normalizer

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import reference_audio_path
from Shared_Weights import attach_shared_initializers


SCRIPT_DIR = Path(__file__).resolve().parent


# ============================== Configuration ==============================
# Edit these values directly; the CLI is reserved for selecting the ONNX folder.
MODEL_FOLDER = Path.home() / "Downloads" / "VoxCPM1.5"  # Tokenizer/model folder.
PROMPT_AUDIO_PATH = reference_audio_path("voxcpm")        # Voice prompt; set None to disable.
PROMPT_TEXT = "对，这就是我，万人敬仰的太乙真人。"              # Transcript for the prompt audio.
TARGET_TTS = [                                             # Sentences synthesized in order.
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology.",
]
GENERATED_AUDIO_PATH = SCRIPT_DIR / "generated.wav"       # Output WAV path.

MAX_FRAMES = 0                 # Per-sentence frame cap; 0 uses the computed limit.
MIN_SEQ_LEN = 2                # Minimum frames before accepting a stop token.
DECODE_LIMIT_FACTOR = 6        # Text-token multiplier used for the frame limit.
SEED = 9527                    # Random seed for reproducible generation.
CFG = 2.5                      # Classifier-free guidance strength.
STREAMING = False              # Use the streaming VAE decoder.

USE_TEXT_NORMALIZER = True     # Normalize text before tokenization.
USE_AUDIO_NORMALIZER = False   # Normalize prompt and generated audio loudness.

MAX_THREADS = 0                # CPU parallel threads; 0 lets ONNX Runtime choose.
DEVICE_ID = 0                  # Accelerator device index.
ORT_LOG = False                # Enable ONNX Runtime logging.
ORT_FP16 = False               # Enable FP16 runtime settings where supported.
ORT_Accelerate_Providers = []  # Optional providers, e.g. ['CUDAExecutionProvider'].
SHOW_PROGRESS = True           # Print pipeline stages and loop progress.
# ===========================================================================


def print_progress(message):
    if SHOW_PROGRESS:
        print(f"[VoxCPM 1.5] {message}", flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run VoxCPM v1.5 ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=Path(__file__).resolve().parent / "VoxCPM_Optimized",
        help="Folder containing ONNX graphs.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()


def _load_compact_metadata():
    metadata_path = onnx_folder / "VoxCPM_Metadata.onnx"
    print_progress(f"Reading package metadata: {metadata_path}")
    model = onnx.load(str(metadata_path), load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    expected_keys = {
        "graph_layout",
        "model_file_name_reference_preprocess",
        "model_file_name_main_prefill",
        "model_file_name_decode_step",
        "model_file_name_vae_decoder",
        "model_file_name_vae_decoder_stream",
        "shared_initializer_model_file",
        "shared_initializer_data_file",
        "in_sample_rate",
        "out_sample_rate",
        "max_seq_len",
        "stop_token_ids",
        "streaming_crop_samples",
        "use_f16_kv",
        "compute_in_f32",
    }
    missing_metadata = sorted(expected_keys - metadata.keys())
    if missing_metadata:
        raise ValueError(
            f"{metadata_path.name} is missing required metadata key(s): {missing_metadata}."
        )
    return metadata


METADATA = _load_compact_metadata()
_precision = {key: METADATA.get(key) for key in ("use_f16_kv", "compute_in_f32")}
_invalid_precision = {
    key: value for key, value in _precision.items() if value not in {"0", "1"}
}
_preserve_fp16_attention = (
    _precision["use_f16_kv"] == "1" and _precision["compute_in_f32"] == "0"
)


chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')


def contains_chinese(text):
    return bool(chinese_char_pattern.search(text))


def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    text = text.replace('√', '根号')
    text = text.replace('≈', '约等于')
    text = text.replace('<', '小于')
    return text


def remove_bracket(text):
    text = text.replace('（', ' ').replace('）', ' ')
    text = text.replace('【', ' ').replace('】', ' ')
    text = text.replace('`', '')
    text = text.replace("——", " ")
    return text


def spell_out_number(text: str, inflect_parser):
    new_text = []
    st = None
    for i, c in enumerate(text):
        if not c.isdigit():
            if st is not None:
                num_str = inflect_parser.number_to_words(text[st: i])
                new_text.append(num_str)
                st = None
            new_text.append(c)
        else:
            if st is None:
                st = i
    if st is not None and st < len(text):
        num_str = inflect_parser.number_to_words(text[st:])
        new_text.append(num_str)
    return ''.join(new_text)


def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def clean_markdown(md_text: str) -> str:
    md_text = re.sub(r"```.*?```", "", md_text, flags=re.DOTALL)
    md_text = re.sub(r"`[^`]*`", "", md_text)
    md_text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", md_text)
    md_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md_text)
    md_text = re.sub(r'^(\s*)-\s+', r'\1', md_text, flags=re.MULTILINE)
    md_text = re.sub(r"<[^>]+>", "", md_text)
    md_text = re.sub(r"^#{1,6}\s*", "", md_text, flags=re.MULTILINE)
    md_text = re.sub(r"\n\s*\n", "\n", md_text)
    return md_text.strip()


def clean_text(text):
    text = clean_markdown(text)
    text = regex.compile(r'\p{Emoji_Presentation}|\p{Emoji}\uFE0F', flags=regex.UNICODE).sub("", text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace('"', "“")
    return text


class TextNormalizer:
    def __init__(self):
        self.zh_tn_model = Normalizer(lang="zh", operator="tn", remove_erhua=True)
        self.en_tn_model = Normalizer(lang="en", operator="tn")
        self.inflect_parser = inflect.engine()

    def normalize(self, text):
        lang = "zh" if contains_chinese(text) else "en"
        text = clean_text(text)
        if lang == "zh":
            text = text.replace("=", "等于")
            if re.search(r'([\d$%^*_+≥≤≠×÷?=])', text):
                text = re.sub(r'(?<=[a-zA-Z0-9])-(?=\d)', ' - ', text)
            text = self.zh_tn_model.normalize(text)
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = remove_bracket(text)
        else:
            text = self.en_tn_model.normalize(text)
            text = spell_out_number(text, self.inflect_parser)
        return text


def audio_normalizer(_audio, target_value=None):
    output_dtype = _audio.dtype
    _audio = _audio.astype(np.float32)
    if target_value is None:
        target_value = 8192.0 if np.issubdtype(output_dtype, np.integer) else 0.25
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    if np.issubdtype(output_dtype, np.integer):
        limits = np.iinfo(output_dtype)
        np.clip(_audio, limits.min, limits.max, out=_audio)
    return _audio.astype(output_dtype)


def mask_multichar_chinese_tokens(tokenizer):
    multichar_tokens = {
        token for token in tokenizer.vocab.keys()
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer) -> None:
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs):
            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []

            for token in tokens:
                clean_token = token.replace("▁", "")

                if clean_token in self.multichar_tokens:
                    chars = list(clean_token)
                    processed.extend(chars)
                else:
                    processed.append(token)

            return processed

        def __call__(self, text: str, **kwargs):
            tokens = self.tokenize(text, **kwargs)
            return self.tokenizer.convert_tokens_to_ids(tokens)

    return CharTokenizerWrapper(tokenizer)


_ONNX_TO_NUMPY = {
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
}


@dataclass(frozen=True, slots=True)
class TensorInfo:
    name: str
    dtype: np.dtype
    shape: tuple

    @classmethod
    def from_node_arg(cls, value):
        try:
            dtype = _ONNX_TO_NUMPY[value.type]
        except KeyError as error:
            raise ValueError(f"Unsupported ONNX tensor type: {value.type!r}.") from error
        return cls(value.name, dtype, tuple(value.shape))

    @property
    def is_static(self):
        return all(isinstance(dimension, int) for dimension in self.shape)

    def resolve_shape(self, *dynamic_dims):
        dynamic_count = sum(
            not isinstance(dimension, int) for dimension in self.shape
        )
        shape = list(self.shape)
        dynamic_index = 0
        for index, dimension in enumerate(shape):
            if not isinstance(dimension, int):
                shape[index] = dynamic_dims[dynamic_index]
                dynamic_index += 1
        return tuple(shape)


def _tensor_infos(session, outputs=False):
    values = session.get_outputs() if outputs else session.get_inputs()
    return tuple(TensorInfo.from_node_arg(value) for value in values)


def _ort_from_array(array):
    return onnxruntime.OrtValue.ortvalue_from_numpy(array, device_type, DEVICE_ID)


def _model_array(info, data, *dynamic_dims):
    return np.asarray(data, dtype=info.dtype).reshape(info.resolve_shape(*dynamic_dims))


def _ort_constant(info, value, *dynamic_dims):
    return _ort_from_array(np.full(info.resolve_shape(*dynamic_dims), value, dtype=info.dtype))


def _ort_constants(infos, value):
    cache = {}
    constants = []
    for info in infos:
        key = (info.dtype.str, info.resolve_shape())
        if key not in cache:
            cache[key] = _ort_constant(info, value)
        constants.append(cache[key])
    return tuple(constants)


def _ort_empty(info):
    return onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        info.resolve_shape(),
        info.dtype,
        device_type,
        DEVICE_ID,
    )


def _bind_inputs(binding, infos, values):
    for info, value in zip(infos, values, strict=True):
        binding.bind_ortvalue_input(info.name, value)


def _bind_static_outputs(binding, infos):
    static_buffers = []
    if device_type == 'cuda':
        return tuple(static_buffers)
    for info in infos:
        if info.is_static:
            value = _ort_empty(info)
            binding.bind_ortvalue_output(info.name, value)
            static_buffers.append(value)
    return tuple(static_buffers)


def _run_binding(session, binding, output_infos):
    auto_bound_infos = (
        tuple(output_infos)
        if device_type == 'cuda'
        else tuple(info for info in output_infos if not info.is_static)
    )
    for info in auto_bound_infos:
        binding._iobinding.bind_output(info.name, _ort_device_type)
    session.run_with_iobinding(binding, run_options=run_options)
    if device_type == 'cuda':
        return tuple(binding.get_outputs())
    dynamic_infos = tuple(info for info in output_infos if not info.is_static)
    bound_infos = (*filter(lambda info: info.is_static, output_infos), *dynamic_infos)
    outputs_by_name = dict(
        zip((info.name for info in bound_infos), binding.get_outputs(), strict=True)
    )
    return tuple(outputs_by_name[info.name] for info in output_infos)


session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer'
        if ORT_FP16 or _preserve_fp16_attention else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = (
    ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer']
    if ORT_FP16 or _preserve_fp16_attention
    else None
)


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',
        'precision':                'ACCURACY',
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False
    }]
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 **3),
        'arena_extend_strategy':              'kNextPowerOfTwo',
        'cudnn_conv_algo_search':             'EXHAUSTIVE',
        'sdpa_kernel':                        '2',
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',
        'device_filter':              'gpu',
        'disable_metacommands':       'false',
        'enable_graph_capture':       'false',
        'enable_graph_serialization': 'false'
    }]
    device_type = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_type = C.OrtDevice(
    _ort_device_type,
    C.OrtDevice.default_memory(),
    DEVICE_ID,
)


def create_session(model_path):
    return onnxruntime.InferenceSession(
        str(model_path),
        sess_options=session_opts,
        providers=ORT_Accelerate_Providers or ['CPUExecutionProvider'],
        provider_options=provider_options,
        disabled_optimizers=disabled_optimizers,
    )


def _run_compact_voxcpm(metadata):
    compact_startup = time.perf_counter()
    streaming = STREAMING
    use_prompt = bool(PROMPT_AUDIO_PATH and PROMPT_TEXT)
    graph_keys = {
        "prefill": "model_file_name_main_prefill",
        "decode": "model_file_name_decode_step",
        "audio": (
            "model_file_name_vae_decoder_stream"
            if streaming
            else "model_file_name_vae_decoder"
        ),
    }
    if use_prompt:
        graph_keys["reference"] = "model_file_name_reference_preprocess"

    shared_started = time.perf_counter()
    print_progress("Attaching shared ONNX initializers...")
    _shared_refs = attach_shared_initializers(
        session_opts,
        onnx_folder / metadata["shared_initializer_model_file"],
    )
    shared_data_path = onnx_folder / metadata["shared_initializer_data_file"]
    print_progress(
        f"Shared ONNX initializers ready in "
        f"{time.perf_counter() - shared_started:.2f}s."
    )
    session_started = time.perf_counter()
    sessions = {}
    for index, (role, key) in enumerate(graph_keys.items(), start=1):
        path = onnx_folder / metadata[key]
        print_progress(f"Loading ONNX graph {index}/{len(graph_keys)}: {path.name}")
        sessions[role] = create_session(path)
    prefill_session = sessions["prefill"]
    decode_session = sessions["decode"]
    audio_session = sessions["audio"]

    prefill_inputs = _tensor_infos(prefill_session)
    (
        prompt_ids_info,
        target_ids_info,
        feat_embed_info,
        feat_cond_info,
        use_prompt_info,
        prefill_noise_info,
        prefill_cfg_info,
        prefill_cfg_minus_info,
    ) = prefill_inputs
    prefill_outputs = _tensor_infos(prefill_session, outputs=True)
    decode_inputs = _tensor_infos(decode_session)
    kv_tensor_count = len(decode_inputs) - 6
    kv_input_infos = decode_inputs[:kv_tensor_count]
    (
        previous_latent_info,
        kv_seq_len_info,
        decode_noise_info,
        decode_cfg_info,
        decode_cfg_minus_info,
        generated_latents_info,
    ) = decode_inputs[kv_tensor_count:]
    decode_outputs = _tensor_infos(decode_session, outputs=True)
    audio_inputs = _tensor_infos(audio_session)
    audio_info = _tensor_infos(audio_session, outputs=True)[0]

    prefill_binding = prefill_session.io_binding()
    decode_bindings = (decode_session.io_binding(), decode_session.io_binding())
    audio_binding = audio_session.io_binding()
    # Retain preallocated outputs for the binding lifetime.
    static_output_buffers = [
        _bind_static_outputs(prefill_binding, prefill_outputs),
        _bind_static_outputs(decode_bindings[0], decode_outputs),
        _bind_static_outputs(decode_bindings[1], decode_outputs),
        _bind_static_outputs(audio_binding, (audio_info,)),
    ]

    if use_prompt:
        reference_session = sessions["reference"]
        reference_inputs = _tensor_infos(reference_session)
        reference_outputs = _tensor_infos(reference_session, outputs=True)
        reference_binding = reference_session.io_binding()
        static_output_buffers.append(
            _bind_static_outputs(reference_binding, reference_outputs)
        )

    print_progress(
        f"ONNX Runtime sessions ready in "
        f"{time.perf_counter() - session_started:.2f}s; "
        f"providers={decode_session.get_providers()}."
    )

    max_seq_len = int(metadata["max_seq_len"])
    min_seq_len = MIN_SEQ_LEN
    decode_limit_factor = DECODE_LIMIT_FACTOR
    in_sample_rate = int(metadata["in_sample_rate"])
    out_sample_rate = int(metadata["out_sample_rate"])
    streaming_crop_samples = int(metadata["streaming_crop_samples"])
    stop_tokens = frozenset(map(int, metadata["stop_token_ids"].split(",")))
    cfg = CFG
    seed = SEED

    noise_buffers = {}
    noise_values = []
    for info in (prefill_noise_info, decode_noise_info):
        key = (info.dtype.str, info.resolve_shape())
        if key not in noise_buffers:
            array = np.empty(key[1], dtype=info.dtype)
            noise_buffers[key] = (array, _ort_from_array(array))
        noise_values.append(noise_buffers[key][1])
    prefill_noise_value, decode_noise_value = noise_values
    prefill_cfg_value, decode_cfg_value = _ort_constants(
        (prefill_cfg_info, decode_cfg_info),
        cfg,
    )
    prefill_cfg_minus_value, decode_cfg_minus_value = _ort_constants(
        (prefill_cfg_minus_info, decode_cfg_minus_info),
        1.0 - cfg,
    )
    use_prompt_value = _ort_constant(use_prompt_info, int(use_prompt))
    random_state = np.random.RandomState(seed)

    def draw_noise(info):
        array, value = noise_buffers[(info.dtype.str, info.resolve_shape())]
        array[...] = random_state.standard_normal(array.shape)
        value.update_inplace(array)

    _bind_inputs(
        prefill_binding,
        (use_prompt_info, prefill_noise_info, prefill_cfg_info, prefill_cfg_minus_info),
        (
            use_prompt_value,
            prefill_noise_value,
            prefill_cfg_value,
            prefill_cfg_minus_value,
        ),
    )
    for binding in decode_bindings:
        _bind_inputs(
            binding,
            (decode_noise_info, decode_cfg_info, decode_cfg_minus_info),
            (decode_noise_value, decode_cfg_value, decode_cfg_minus_value),
        )

    print_progress("Loading the text frontend...")
    tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(MODEL_FOLDER))
    text_normalizer = TextNormalizer() if USE_TEXT_NORMALIZER else None
    if use_prompt:
        reference_started = time.perf_counter()
        print_progress(f"Preparing prompt audio: {PROMPT_AUDIO_PATH}")
        audio = np.asarray(
            AudioSegment.from_file(PROMPT_AUDIO_PATH)
            .set_channels(1)
            .set_frame_rate(in_sample_rate)
            .set_sample_width(2)
            .get_array_of_samples(),
            dtype=np.int16,
        )
        if USE_AUDIO_NORMALIZER:
            audio = audio_normalizer(audio)
        if np.issubdtype(reference_inputs[0].dtype, np.floating):
            audio = (audio.astype(np.float32) * (1.0 / 32768.0)).astype(
                reference_inputs[0].dtype
            )
        else:
            audio = audio.astype(reference_inputs[0].dtype, copy=False)
        audio_value = _ort_from_array(
            _model_array(reference_inputs[0], audio, audio.size)
        )
        _bind_inputs(reference_binding, reference_inputs, (audio_value,))
        feat_embed, feat_cond = _run_binding(
            reference_session,
            reference_binding,
            reference_outputs,
        )
        print_progress(
            f"Prompt conditioning ready in "
            f"{time.perf_counter() - reference_started:.2f}s."
        )
        normalized_prompt_text = (
            text_normalizer.normalize(PROMPT_TEXT)
            if text_normalizer is not None
            else PROMPT_TEXT
        )
        prompt_token_ids = tokenizer(normalized_prompt_text)
    else:
        feat_embed = _ort_constant(feat_embed_info, 0, 0)
        feat_cond = _ort_constant(feat_cond_info, 0)
        prompt_token_ids = ()
        if not PROMPT_AUDIO_PATH:
            print("Info: No prompt audio provided; deterministic seeded voice mode is active.\n")
        else:
            print("Warning: Prompt audio is ignored because prompt text is empty.\n")

    prompt_ids_numpy = _model_array(
        prompt_ids_info,
        prompt_token_ids,
        len(prompt_token_ids),
    )
    prompt_ids = _ort_from_array(prompt_ids_numpy)
    _bind_inputs(
        prefill_binding,
        (prompt_ids_info, feat_embed_info, feat_cond_info),
        (prompt_ids, feat_embed, feat_cond),
    )

    audio_prefix_shape = tuple(audio_info.shape[:-1])
    blank_segment = np.zeros(
        (*audio_prefix_shape, int(out_sample_rate * 0.1)),
        dtype=audio_info.dtype,
    )

    if streaming:
        def decode_stream_pair(previous_numpy, current_numpy):
            previous_value = _ort_from_array(previous_numpy)
            current_value = _ort_from_array(current_numpy)
            _bind_inputs(
                audio_binding,
                audio_inputs,
                (previous_value, current_value),
            )
            return _run_binding(
                audio_session,
                audio_binding,
                (audio_info,),
            )[0].numpy().copy()

    saved_audio = []
    generation_start = time.perf_counter()
    total_decode_calls = 0
    total_emitted_frames = 0
    print_progress(f"Prepared {len(TARGET_TTS)} text target(s).")
    for sentence_index, sentence in enumerate(TARGET_TTS, start=1):
        print_progress(
            f"Starting target {sentence_index}/{len(TARGET_TTS)}: {sentence!r}"
        )
        if text_normalizer is not None:
            sentence = text_normalizer.normalize(sentence)
        target_token_ids = tokenizer(sentence)
        target_ids_numpy = _model_array(
            target_ids_info,
            target_token_ids,
            len(target_token_ids),
        )
        target_ids = _ort_from_array(target_ids_numpy)
        ids_len = int(
            prompt_ids_numpy.shape[1]
            + target_ids_numpy.shape[1]
            + 1
            + feat_embed.shape()[1]
        )
        max_frames = min(
            (target_ids_numpy.shape[1] + 1) * decode_limit_factor + 10,
            max_seq_len - 1 - ids_len,
        )
        if MAX_FRAMES > 0:
            max_frames = min(max_frames, MAX_FRAMES)
        if max_frames <= 0:
            print_progress("Generation skipped: prompt leaves no frame capacity.")
            saved_audio.append(blank_segment)
            continue

        print_progress(f"Generating acoustic frames (limit {max_frames})...")
        _bind_inputs(prefill_binding, (target_ids_info,), (target_ids,))
        draw_noise(prefill_noise_info)
        sentence_start = time.perf_counter()
        prefill_values = _run_binding(prefill_session, prefill_binding, prefill_outputs)
        state_values = list(prefill_values[:kv_tensor_count])
        previous_latent, stop_value, kv_seq_len = prefill_values[-3:]
        generated_latents = previous_latent
        stop_flag = int(stop_value.numpy().flat[0])
        decode_step = 0
        if streaming:
            chunk_shape = audio_info.resolve_shape()
            decode_calls = max(max_frames - 1, 0)
            stream_capacity = (
                chunk_shape[-1]
                + max(decode_calls - 1, 0) * (chunk_shape[-1] - streaming_crop_samples)
                if decode_calls
                else 0
            )
            stream_audio = np.empty(
                (*chunk_shape[:-1], stream_capacity),
                dtype=audio_info.dtype,
            )
            stream_futures = []
            stream_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        while (
            decode_step + 1 < max_frames
            and (decode_step < min_seq_len or stop_flag not in stop_tokens)
        ):

            binding = decode_bindings[decode_step & 1]
            _bind_inputs(binding, kv_input_infos, state_values)
            _bind_inputs(
                binding,
                (previous_latent_info, kv_seq_len_info, generated_latents_info),
                (previous_latent, kv_seq_len, generated_latents),
            )
            draw_noise(decode_noise_info)

            decode_values = _run_binding(decode_session, binding, decode_outputs)
            decode_step += 1
            state_values = list(decode_values[:kv_tensor_count])
            current_latent, stop_value, kv_seq_len, generated_latents = decode_values[-4:]
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
            emitted_frames = decode_step + 1
            if emitted_frames % 25 == 0 or emitted_frames == max_frames:
                print_progress(
                    f"Acoustic generation: {emitted_frames}/{max_frames} frames"
                )

        if streaming:
            print_progress("Finalizing streaming waveform...")
            stream_executor.shutdown(wait=True)
            stream_results = [future.result() for future in stream_futures]
            stream_offset = 0
            for index, chunk in enumerate(stream_results):
                crop = streaming_crop_samples if index > 0 else 0
                chunk = chunk[..., crop:]
                next_offset = stream_offset + chunk.shape[-1]
                stream_audio[..., stream_offset:next_offset] = chunk
                stream_offset = next_offset
            sentence_audio = stream_audio[..., :stream_offset]
        else:
            print_progress("Decoding waveform...")
            _bind_inputs(audio_binding, audio_inputs, (generated_latents,))
            sentence_audio = _run_binding(
                audio_session,
                audio_binding,
                (audio_info,),
            )[0].numpy().copy()

        elapsed = time.perf_counter() - sentence_start
        emitted_frames = decode_step + 1
        total_decode_calls += decode_step
        total_emitted_frames += emitted_frames
        print(
            f"Target {sentence_index}/{len(TARGET_TTS)}: "
            f"{emitted_frames / elapsed:.3f} frame/s "
            f"({decode_step} DecodeStep calls for {emitted_frames} emitted frames)",
            flush=True,
        )
        saved_audio.extend((sentence_audio, blank_segment))

    elapsed_total = time.perf_counter() - generation_start
    if saved_audio:
        audio_out = np.concatenate(saved_audio, axis=-1).reshape(-1)
    else:
        audio_out = np.empty((0,), dtype=audio_info.dtype)
    if USE_AUDIO_NORMALIZER and audio_out.size:
        audio_out = audio_normalizer(audio_out)
    print_progress(f"Writing generated audio: {GENERATED_AUDIO_PATH}")
    if audio_out.dtype == np.float16:
        audio_out = audio_out.astype(np.float32)
    output_subtype = "PCM_16" if np.issubdtype(audio_out.dtype, np.integer) else "FLOAT"
    sf.write(
        GENERATED_AUDIO_PATH,
        audio_out,
        out_sample_rate,
        subtype=output_subtype,
        format="WAVEX",
    )
    blank_samples = blank_segment.shape[-1] * len(TARGET_TTS)
    speech_seconds = max((audio_out.size - blank_samples) / out_sample_rate, 1e-9)
    print(f"Generate complete. Saved {GENERATED_AUDIO_PATH}.", flush=True)
    print(f"Seed: {seed}; CFG: {cfg}; streaming: {streaming}", flush=True)
    print(
        f"Frames: {total_emitted_frames}; DecodeStep calls: {total_decode_calls}; "
        f"calls/additional-frame: "
        f"{total_decode_calls / max(total_emitted_frames - len(TARGET_TTS), 1):.3f}",
        flush=True,
    )
    print(f"Startup time: {generation_start - compact_startup:.3f}s", flush=True)
    print(f"Generation time: {elapsed_total:.3f}s", flush=True)
    print(f"RTF: {elapsed_total / speech_seconds:.3f}", flush=True)
    print_progress(
        f"Pipeline complete in {time.perf_counter() - compact_startup:.2f}s."
    )


if __name__ == "__main__":
    _run_compact_voxcpm(METADATA)
