import argparse
import site
import time
from pathlib import Path
try:
    import rjieba
except ImportError:
    rjieba = None
    import jieba
import torch
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style
python_package_path = site.getsitepackages()[-1]


def _parse_args():
    parser = argparse.ArgumentParser(description="Run F5 TTS ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=Path(__file__).resolve().parent / "F5_ONNX",
        help="Folder containing ONNX graphs.",
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=None,
        help="Override the vocab.txt path stamped into the exported model metadata.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
script_dir = Path(__file__).resolve().parent
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()
vocab_path_override = _ARGS.vocab_path.expanduser().resolve() if _ARGS.vocab_path else None

onnx_model_Preprocess  = str(onnx_folder / "F5_Preprocess.onnx")                                    # The exported onnx model path.
onnx_model_Transformer = str(onnx_folder / "F5_Transformer.onnx")                                   # The exported onnx model path.
onnx_model_Decode      = str(onnx_folder / "F5_Decode.onnx")                                        # The exported onnx model path.
onnx_model_Metadata    = str(onnx_folder / "F5_Metadata.onnx")                                      # Tiny metadata carrier graph.
generated_audio        = str(script_dir / "generated_audio.wav")
test_in_english = False

if test_in_english:
    reference_audio  = python_package_path + "/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text         = "Some call me nature, others call me mother nature."
    gen_text         = "Some call me Dake, others call me QQ."
else:
    reference_audio  = python_package_path + "/f5_tts/infer/examples/basic/basic_ref_zh.wav"        # The reference audio path.
    ref_text         = "对，这就是我，万人敬仰的太乙真人。"                                               # The ASR result of reference audio.
    gen_text         = "对，这就是我，万人敬仰的大可奇奇。"                                               # The target TTS.


ORT_Accelerate_Providers = []                             # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                          # else keep empty.
RANDOM_SEED = 9527                                        # Set seed to reproduce the generated audio
SPEED = 1.0                                               # Set for talking speed. Only works with dynamic_axes=True
MAX_THREADS = 0                                           # Max CPU parallel threads.
DEVICE_ID = 0                                             # The GPU id, default to 0.
ORT_LOG = False                                           # Enable ONNX Runtime logging (disable for best performance)
ORT_FP16 = False                                          # FP16 ORT settings (ARM64-v8.2a or newer required for CPU)
# MODEL_SAMPLE_RATE, HOP_LENGTH, NFE_STEP and MAX_SIGNAL_LENGTH are read from the exported model metadata
# (see the "MODEL METADATA" section below) so they always match the exported graphs.

# From the official code
def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        if rjieba is None:
            if jieba.dt.initialized is False:
                jieba.default_logger.setLevel(50)  # CRITICAL
                jieba.initialize()
            segments = jieba.cut(text)
        else:
            segments = rjieba.cut(text)
        for seg in segments:
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)
    return final_text_list


# From the official code
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1
):
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    text = torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (mirrors the Qwen_TTS inference implementation)
# ─────────────────────────────────────────────────────────────────────────────
def _copy_ort(ortvalue):
    """Materialise an OrtValue onto `device_type` as an independent buffer we own."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(ortvalue.numpy(), device_type, DEVICE_ID)


def ensure_rank1_host_ortvalue(ortvalue):
    """Return a rank-1 CPU OrtValue for scalar values passed into Decode."""
    array = ortvalue.numpy()
    if array.ndim == 1:
        return ortvalue
    if array.ndim == 0:
        return onnxruntime.OrtValue.ortvalue_from_numpy(array.reshape(1), 'cpu', DEVICE_ID)
    raise ValueError(f"Expected scalar or rank-1 OrtValue, got shape {array.shape}.")


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    """Create an ORT InferenceSession with standard options."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def bind_inputs(binding, names, values):
    """Bind a batch of OrtValue inputs by name."""
    for name, value in zip(names, values):
        binding.bind_ortvalue_input(name, value)


def bind_device_outputs(binding, names, device_obj):
    """Bind outputs to a device so ORT auto-allocates them (fresh OrtValue per run)."""
    for name in names:
        binding._iobinding.bind_output(name, device_obj)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.enable_cpu_mem_arena     = True
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
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Only the Transformer denoise graph uses the configured accelerator; the metadata carrier and
# Preprocess / Decode graphs always run on CPU via `packed_settings_cpu`.
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers or "TensorrtExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      8 * (1024 ** 3),    # 8 GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["kNextPowerOfTwo", "kSameAsRequested"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '1',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',                # Set to '0' to avoid potential errors when enabled.
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':              DEVICE_ID,
        'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
        'device_filter':          'gpu'                # [any, npu, gpu]
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)

# packed_settings drives the Transformer graph on the configured accelerator; the metadata carrier and
# Preprocess / Decode graphs stay on CPU via packed_settings_cpu.
packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}
packed_settings_cpu = {
    "_session_opts":        session_opts,
    "_providers":           ['CPUExecutionProvider'],
    "_provider_options":    None,
    "_disabled_optimizers": disabled_optimizers
}


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# MODEL METADATA  (single source of truth — stamped by Export_F5.py)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The exporter stamps every graph (plus a tiny F5_Metadata.onnx carrier) with the pipeline
# geometry, so inference stays locked to the exported graphs.
if not Path(onnx_model_Metadata).exists():
    raise FileNotFoundError(
        f"{onnx_model_Metadata} was not found. Re-export with Export_F5.py "
        "to generate the metadata carrier and stamp the model metadata."
    )
ort_session_Metadata = create_session(onnx_model_Metadata, **packed_settings_cpu)
_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}
if _model_meta.get("f5_tts_metadata_version") != "3":
    raise ValueError(
        f"Required F5_TTS metadata version 3 is missing from {onnx_model_Metadata}. "
        "Re-export with Export_F5.py to stamp the model metadata."
    )


def _meta_int(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            "Re-export with Export_F5.py to stamp the model metadata."
        )
    return int(value)


MODEL_SAMPLE_RATE = _meta_int("sample_rate")
HOP_LENGTH        = _meta_int("hop_length")
NFE_STEP          = _meta_int("nfe_step")
MAX_SIGNAL_LENGTH = _meta_int("max_signal_length")
if NFE_STEP < 1:
    raise ValueError("Exported NFE_STEP must be >= 1.")

if vocab_path_override is None:
    vocab_path_value = _model_meta.get("vocab_path")
    if not vocab_path_value:
        raise KeyError(
            f"Required metadata key 'vocab_path' is missing from {onnx_model_Metadata}. "
            "Re-export with Export_F5.py or pass --vocab-path."
        )
    vocab_path = Path(vocab_path_value).expanduser()
else:
    vocab_path = vocab_path_override
if not vocab_path.exists():
    raise FileNotFoundError(f"Vocab file was not found: {vocab_path}")
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)
print(
    f"  Model metadata: {len(_model_meta)} keys "
    f"(sample_rate={MODEL_SAMPLE_RATE}, hop_length={HOP_LENGTH}, nfe_step={NFE_STEP})."
)
print(f"  Vocab: {vocab_path} ({vocab_size} tokens).")

ort_session_Preprocess = create_session(onnx_model_Preprocess, **packed_settings_cpu)
in_name_Preprocess     = get_in_names(ort_session_Preprocess)
out_name_Preprocess    = get_out_names(ort_session_Preprocess)

ort_session_Transformer = create_session(onnx_model_Transformer, **packed_settings)
print(f"\nUsable Providers: {ort_session_Transformer.get_providers()}")
in_name_Transformer     = get_in_names(ort_session_Transformer)
out_name_Transformer    = get_out_names(ort_session_Transformer)
if 'time_step' not in in_name_Transformer:
    raise ValueError("F5_Transformer.onnx has no time_step input. Re-export with the unfused Export_F5.py.")

ort_session_Decode = create_session(onnx_model_Decode, **packed_settings_cpu)
in_name_Decode     = get_in_names(ort_session_Decode)
out_name_Decode    = get_out_names(ort_session_Decode)

# Load the input audio
print(f"\nReference Audio: {reference_audio}")
audio = np.array(AudioSegment.from_file(reference_audio).set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)

if len(ref_text[-1].encode("utf-8")) == 1:
    ref_text = ref_text + " "
local_speed = SPEED
if len(gen_text.encode("utf-8")) < 10:
    local_speed = 0.3
ref_text_len = len(ref_text.encode('utf-8'))
gen_text_len = len(gen_text.encode('utf-8'))
ref_audio_len = audio_len // HOP_LENGTH
cond_signal_len = ref_audio_len + 1
text = convert_char_to_pinyin([ref_text + gen_text])
text_ids = list_str_to_idx(text, vocab_char_map).numpy()
duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)
max_duration_value = max(max(text_ids.shape[-1], cond_signal_len) + 1, duration)
if max_duration_value > MAX_SIGNAL_LENGTH:
    raise ValueError(
        f"Requested max_duration {max_duration_value} exceeds exported max_signal_length {MAX_SIGNAL_LENGTH}. "
        "Use shorter text/audio or re-export with a larger MAX_SIGNAL_LENGTH."
    )
max_duration = np.array([max_duration_value], dtype=np.int64)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# DEVICE SELECTION FOR IOBINDING
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# The Transformer graph runs on the accelerator when configured; Preprocess and Decode always run on CPU.
# Transformer tensors therefore live on `device_type`, while Preprocess/Decode tensors stay on the host.
# The Preprocess -> Transformer and Transformer -> Decode hand-offs cross that boundary once each.
_ort_host_obj   = C.OrtDevice(C.OrtDevice.cpu(), C.OrtDevice.default_memory(), DEVICE_ID)

# One reusable IOBinding per graph.
binding_Preprocess = ort_session_Preprocess.io_binding()
binding_Decode     = ort_session_Decode.io_binding()

print("\n\nRun F5-TTS by ONNX Runtime.")
start_count = time.time()

# ── Preprocess graph (runs once on CPU). Host inputs → host auto-allocated outputs. ──
bind_inputs(binding_Preprocess, in_name_Preprocess, [
    onnxruntime.OrtValue.ortvalue_from_numpy(audio,        'cpu', DEVICE_ID),
    onnxruntime.OrtValue.ortvalue_from_numpy(text_ids,     'cpu', DEVICE_ID),
    onnxruntime.OrtValue.ortvalue_from_numpy(max_duration, 'cpu', DEVICE_ID),
])
bind_device_outputs(binding_Preprocess, out_name_Preprocess, _ort_host_obj)
run(ort_session_Preprocess, binding_Preprocess)
preprocess_outputs = binding_Preprocess.get_outputs()
# preprocess_outputs order matches out_name_Preprocess: [noise, rope_cos, rope_sin,
#                                                         cat_mel_text, cat_mel_text_drop,
#                                                         ref_signal_len, rms_scale, ref_mel_tail]

# ── Transformer graph. The unfused export runs one denoise step per call; the NFE loop runs here.
# ── The four conditioning tensors from Preprocess never change, so they are bound once.
if device_type == 'cpu':
    # Preprocess and Transformer share the host: bind Preprocess conditioning outputs straight in (zero-copy).
    cond_inputs = list(preprocess_outputs[1:5])
else:
    # Copy the four conditioning tensors onto the accelerator once so the loop stays copy-free.
    cond_inputs = [_copy_ort(preprocess_outputs[k]) for k in range(1, 5)]

noise_host     = preprocess_outputs[0].numpy()                                                  # Preprocess noise (shape + dtype)
# Two noise buffers ping-pong across the NFE steps so each step reads one buffer and writes the other
# (distinct src/dst avoids a read-after-write hazard from aliasing one buffer as both input and output).
noise_buffers = [
    onnxruntime.OrtValue.ortvalue_from_numpy(noise_host.copy(),         device_type, DEVICE_ID),
    onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros_like(noise_host), device_type, DEVICE_ID),
]
# time_step is a host int index; the graph gathers this step's time embedding / delta_t from the
# precomputed tables. Pre-build one CPU OrtValue per step so the loop stays allocation-free.
time_step_buffers = [
    onnxruntime.OrtValue.ortvalue_from_numpy(np.array([step], dtype=np.int64), 'cpu', DEVICE_ID)
    for step in range(NFE_STEP)
]
binding_Transformer = ort_session_Transformer.io_binding()
noise_name     = in_name_Transformer[0]
cond_names     = in_name_Transformer[1:5]
time_step_name = in_name_Transformer[-1]
noise_dst_name = out_name_Transformer[0]                           # denoise output
bind_inputs(binding_Transformer, cond_names, cond_inputs)          # conditioning is identical every step

print("NFE_STEP: 0")
for step in range(NFE_STEP):
    noise_src = noise_buffers[step & 1]
    noise_dst = noise_buffers[(step + 1) & 1]
    binding_Transformer.bind_ortvalue_input(noise_name, noise_src)
    binding_Transformer.bind_ortvalue_input(time_step_name, time_step_buffers[step])
    binding_Transformer.bind_ortvalue_output(noise_dst_name, noise_dst)
    run(ort_session_Transformer, binding_Transformer)
    print(f"NFE_STEP: {step}")
noise_final = noise_buffers[NFE_STEP & 1]


# ── Decode graph (runs once on CPU). Bring the final denoise back to the host for the CPU session;
# ── ref_signal_len / rms_scale are already host tensors from Preprocess. ──
denoised_for_decode = noise_final if device_type == 'cpu' else onnxruntime.OrtValue.ortvalue_from_numpy(noise_final.numpy(), 'cpu', DEVICE_ID)
rms_scale_for_decode = ensure_rank1_host_ortvalue(preprocess_outputs[6])
bind_inputs(binding_Decode, in_name_Decode, [denoised_for_decode, preprocess_outputs[5], rms_scale_for_decode, preprocess_outputs[7]])
bind_device_outputs(binding_Decode, [out_name_Decode[0]], _ort_host_obj)
run(ort_session_Decode, binding_Decode)
generated_signal = binding_Decode.get_outputs()[0].numpy()
end_count = time.time()

# Save to audio
sf.write(generated_audio, generated_signal.reshape(-1), MODEL_SAMPLE_RATE, format='WAVEX')
print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{end_count - start_count:.3f}")
