import time
import numpy as np
import soundfile as sf
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
download_path                            = r'/home/DakeQQ/Downloads/Qwen3-TTS-12Hz-0.6B-Base'                         # Source model folder
onnx_model_Embed_A                       = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Embed_A.onnx'
onnx_model_Embed_B                       = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Embed_B.onnx'
onnx_model_Embed_C                       = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Embed_C.onnx'
onnx_model_Embed_D                       = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Embed_D.onnx'
onnx_model_Preprocess                    = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Preprocess.onnx'
onnx_model_Encoder                       = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Encoder.onnx'
onnx_model_Predictor                     = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Predictor.onnx'
onnx_model_Pred_LmHead                   = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Predictor_LmHead.onnx'
onnx_model_Main                          = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Main.onnx'
onnx_model_Decoder                       = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Decoder.onnx'
onnx_model_Main_Rotary_Mask_Text_Prefill = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Main_Rotary_Mask_Text_Prefill.onnx'
onnx_model_Main_Rotary_Mask_Text_Decode  = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Main_Rotary_Mask_Text_Decode.onnx'
onnx_model_Pred_Rotary_Mask_Text_Prefill = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Predictor_Rotary_Mask_Text_Prefill.onnx'
onnx_model_Pred_Rotary_Mask_Text_Decode  = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/QwenTTS_Predictor_Rotary_Mask_Text_Decode.onnx'
onnx_model_Gather_0                      = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/Gather_0.onnx'
onnx_model_Concat_Embed                  = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/Concat_Embed.onnx'
onnx_model_Concat_Ids                    = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/Concat_Ids.onnx'
onnx_model_Greedy                        = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/Greedy_Search.onnx'
onnx_model_First_Beam                    = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/First_Beam_Search.onnx'
onnx_model_Second_Beam                   = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/Second_Beam_Search.onnx'
onnx_model_Penalty                       = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/Apply_Penalty.onnx'
onnx_model_Argmax                        = r'/home/DakeQQ/Downloads/QwenTTS_Optimized/Argmax.onnx'


# ─────────────────────────────────────────────────────────────────────────────
# Prompts & targets
# ─────────────────────────────────────────────────────────────────────────────
generated_audio_path = r"./generated.wav"                     # Output file
prompt_audio_path    = "./example/basic_ref_zh.wav"           # Reference audio for voice cloning
prompt_text          = "对，这就是我，万人敬仰的太乙真人。"         # Transcription of the reference audio
target_tts           = [                                      # Texts to synthesize
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
]


# ─────────────────────────────────────────────────────────────────────────────
# Language & generation settings
# ─────────────────────────────────────────────────────────────────────────────
TTS_LANGUAGE = "Chinese"          # Options: [English, German, Spanish, Chinese, Japanese, French, Korean, Russian, Italian, Portuguese]
MAX_SEQ_LEN  = 1024               # Maximum decode length (fixed at export time)
MIN_SEQ_LEN  = 2                  # Minimum decode length (editable at runtime)
STOP_TOKEN   = [2150]             # EOS token id for QwenTTS — Do not change
NUM_CODE_GROUPS_MINUS = 15        # Fixed value for the Qwen3-TTS

# ─────────────────────────────────────────────────────────────────────────────
# Audio settings
# ─────────────────────────────────────────────────────────────────────────────
IN_SAMPLE_RATE  = 24000           # Prompt audio sample rate  (fixed at export time)
OUT_SAMPLE_RATE = 24000           # Output audio sample rate  (fixed at export time)


# ─────────────────────────────────────────────────────────────────────────────
# Decoding settings
# ─────────────────────────────────────────────────────────────────────────────
USE_BEAM_SEARCH = False           # False → greedy decoding
BEAM_SIZE       = 3               # Active beam width
TOP_K           = 3               # Top-K sampling parameter
PENALTY_RANGE   = 5               # Recent-token window for repetition penalty
REPEAT_PENALTY  = 0.8             # Repetition penalty coefficient (1.0 = disabled)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime / optimisation flags
# ─────────────────────────────────────────────────────────────────────────────
USE_AUDIO_NORMALIZER     = False  # Normalize output loudness (may alter voice characteristics)
ORT_LOG                  = False  # Enable ONNX Runtime logging (disable for best performance)
ORT_FP16                 = False  # FP16 ORT settings (ARM64-v8.2a or newer required for CPU)
ORT_Accelerate_Providers = []     # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS              = 0      # CPU thread count (0 = auto)
DEVICE_ID                = 0      # Device index


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def audio_normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


def create_ort_with_data(data, dtype, device, device_id):
    """Create an OrtValue from a Python list/scalar."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    """Create a zero-filled OrtValue with the given shape."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


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


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE ID MAPPING
# ══════════════════════════════════════════════════════════════════════════════
LANGUAGE_ID_MAP = {
    'english':    2050,
    'german':     2053,
    'spanish':    2054,
    'chinese':    2055,
    'japanese':   2058,
    'french':     2061,
    'korean':     2064,
    'russian':    2069,
    'italian':    2070,
    'portuguese': 2071,
}
language_id  = LANGUAGE_ID_MAP[TTS_LANGUAGE.lower()]


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
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
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,                 # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 ** 3),   # 24 GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',                # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',                # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',  # ["default", "high_performance", "minimum_power"]
        'device_filter':              'gpu',               # [gpu, npu, any]
        'disable_metacommands':       'false',             # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',             # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'              # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device        = 'cpu' if 'dml' in device_type else device_type

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}


# ══════════════════════════════════════════════════════════════════════════════
# DECODING STRATEGY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = (REPEAT_PENALTY != 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & STOP TOKENS & PROMPT
# ══════════════════════════════════════════════════════════════════════════════
tokenizer      = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
STOP_TOKEN_SET = set(STOP_TOKEN)
prompt_tokens  = tokenizer(prompt_text, return_tensors='np')['input_ids'].astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

# --- Embed A ---
ort_session_Embed_A = create_session(onnx_model_Embed_A, **packed_settings)
in_name_Embed_A     = get_in_names(ort_session_Embed_A)
out_name_Embed_A    = get_out_names(ort_session_Embed_A)

# --- Embed B ---
ort_session_Embed_B = create_session(onnx_model_Embed_B, **packed_settings)
in_name_Embed_B     = get_in_names(ort_session_Embed_B)
out_name_Embed_B    = get_out_names(ort_session_Embed_B)

# --- Embed C ---
ort_session_Embed_C = create_session(onnx_model_Embed_C, **packed_settings)
in_name_Embed_C     = get_in_names(ort_session_Embed_C)
out_name_Embed_C    = get_out_names(ort_session_Embed_C)
in_meta_Embed_C     = ort_session_Embed_C._inputs_meta

# --- Embed D (multi-group) ---
path_name           = onnx_model_Embed_D.split('.')[0]
ort_session_Embed_D = [create_session(f'{path_name}_{i}.onnx', **packed_settings) for i in range(NUM_CODE_GROUPS_MINUS)]
in_name_Embed_D     = get_in_names(ort_session_Embed_D[0])
out_name_Embed_D    = get_out_names(ort_session_Embed_D[0])

# --- Gather / Concat ---
ort_session_Gather_0     = create_session(onnx_model_Gather_0, **packed_settings)
in_name_Gather_0         = get_in_names(ort_session_Gather_0)
out_name_Gather_0        = get_out_names(ort_session_Gather_0)

ort_session_Concat_Embed = create_session(onnx_model_Concat_Embed, **packed_settings)
in_name_Concat_Embed     = get_in_names(ort_session_Concat_Embed)
out_name_Concat_Embed    = get_out_names(ort_session_Concat_Embed)

ort_session_Concat_Ids   = create_session(onnx_model_Concat_Ids, **packed_settings)
in_name_Concat_Ids       = get_in_names(ort_session_Concat_Ids)
out_name_Concat_Ids      = get_out_names(ort_session_Concat_Ids)

# --- Preprocess ---
ort_session_Preprocess = create_session(onnx_model_Preprocess, **packed_settings)
in_name_Preprocess     = get_in_names(ort_session_Preprocess)
out_name_Preprocess    = get_out_names(ort_session_Preprocess)

# --- Encoder ---
ort_session_Encoder = create_session(onnx_model_Encoder, **packed_settings)
in_name_Encoder     = get_in_names(ort_session_Encoder)
out_name_Encoder    = get_out_names(ort_session_Encoder)

# --- LM Head (multi-group) ---
path_name          = onnx_model_Pred_LmHead.split('.')[0]
ort_session_LmHead = [create_session(f'{path_name}_{i}.onnx', **packed_settings) for i in range(NUM_CODE_GROUPS_MINUS)]
in_name_LmHead     = get_in_names(ort_session_LmHead[0])
out_name_LmHead    = get_out_names(ort_session_LmHead[0])
vocab_size_LmHead  = ort_session_LmHead[0]._outputs_meta[0].shape[1]

# --- Decoder ---
ort_session_Decoder = create_session(onnx_model_Decoder, **packed_settings)
in_name_Decoder     = get_in_names(ort_session_Decoder)
out_name_Decoder    = get_out_names(ort_session_Decoder)

# --- Main Rotary ---
ort_session_Main_Rotary_Text_Prefill = create_session(onnx_model_Main_Rotary_Mask_Text_Prefill, **packed_settings)
in_name_Main_Rotary_Text_Prefill     = get_in_names(ort_session_Main_Rotary_Text_Prefill)
out_name_Main_Rotary_Text_Prefill    = get_out_names(ort_session_Main_Rotary_Text_Prefill)

ort_session_Main_Rotary_Text_Decode  = create_session(onnx_model_Main_Rotary_Mask_Text_Decode, **packed_settings)
in_name_Main_Rotary_Text_Decode      = get_in_names(ort_session_Main_Rotary_Text_Decode)
out_name_Main_Rotary_Text_Decode     = get_out_names(ort_session_Main_Rotary_Text_Decode)

# --- Predictor Rotary ---
ort_session_Predictor_Rotary_Text_Prefill = create_session(onnx_model_Pred_Rotary_Mask_Text_Prefill, **packed_settings)
in_name_Predictor_Rotary_Text_Prefill     = get_in_names(ort_session_Predictor_Rotary_Text_Prefill)
out_name_Predictor_Rotary_Text_Prefill    = get_out_names(ort_session_Predictor_Rotary_Text_Prefill)

ort_session_Predictor_Rotary_Text_Decode  = create_session(onnx_model_Main_Rotary_Mask_Text_Decode, **packed_settings)
in_name_Predictor_Rotary_Text_Decode      = get_in_names(ort_session_Predictor_Rotary_Text_Decode)
out_name_Predictor_Rotary_Text_Decode     = get_out_names(ort_session_Predictor_Rotary_Text_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")

in_name_Main  = get_in_names(ort_session_Main)
out_name_Main = get_out_names(ort_session_Main)
in_meta_Main  = ort_session_Main._inputs_meta

# Derived index offsets
num_keys_values_Main        = len(out_name_Main) - 2
num_layers_Main             = num_keys_values_Main // 2
num_keys_values_Main_plus_1 = num_keys_values_Main + 1
num_keys_values_Main_plus_2 = num_keys_values_Main + 2
num_keys_values_Main_plus_3 = num_keys_values_Main + 3
num_keys_values_Main_plus_4 = num_keys_values_Main + 4

# Partitioned name lists
in_name_Main_kv      = in_name_Main[:num_keys_values_Main]
in_name_Main_others  = in_name_Main[num_keys_values_Main:]
out_name_Main_kv     = out_name_Main[:num_keys_values_Main]
out_name_Main_others = out_name_Main[num_keys_values_Main:]

# Dtype introspection
kv_dtype_Main     = np.float16 if 'float16' in in_meta_Main[0].type else np.float32
hidden_dtype_Main = np.float16 if 'float16' in in_meta_Main[num_keys_values_Main].type else np.float32

# Initial KV cache
init_past_keys_Main   = create_ort_with_shape((1, in_meta_Main[0].shape[1],               1, in_meta_Main[0].shape[3],               0), kv_dtype_Main, kv_device, DEVICE_ID)
init_past_values_Main = create_ort_with_shape((1, in_meta_Main[num_layers_Main].shape[1], 1, 0, in_meta_Main[num_layers_Main].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)

# --- Predictor ---
ort_session_Predictor = create_session(onnx_model_Predictor, **packed_settings)

in_name_Predictor         = get_in_names(ort_session_Predictor)
out_name_Predictor        = get_out_names(ort_session_Predictor)
in_meta_Predictor         = ort_session_Predictor._inputs_meta
num_keys_values_Predictor = len(out_name_Predictor) - 1
num_layers_Predictor      = num_keys_values_Predictor // 2

# Derived index offsets
num_keys_values_Predictor_plus_1 = num_keys_values_Predictor + 1
num_keys_values_Predictor_plus_2 = num_keys_values_Predictor + 2
num_keys_values_Predictor_plus_3 = num_keys_values_Predictor + 3
num_keys_values_Predictor_plus_4 = num_keys_values_Predictor + 4

# Partitioned name lists
in_name_Predictor_kv      = in_name_Predictor[:num_keys_values_Predictor]
in_name_Predictor_others  = in_name_Predictor[num_keys_values_Predictor:]
out_name_Predictor_kv     = out_name_Predictor[:num_keys_values_Predictor]
out_name_Predictor_hidden = out_name_Predictor[num_keys_values_Predictor]

# Dtype introspection
kv_dtype_Predictor     = np.float16 if 'float16' in in_meta_Predictor[0].type else np.float32
hidden_dtype_Predictor = np.float16 if 'float16' in in_meta_Predictor[num_keys_values_Predictor].type else np.float32

if hidden_dtype_Predictor != hidden_dtype_Main:
    raise ValueError(f"Hidden state dtype mismatch between Main and Predictor: {hidden_dtype_Main} vs {hidden_dtype_Predictor}")

# Initial KV cache
init_past_keys_Predictor   = create_ort_with_shape((1, in_meta_Predictor[0].shape[1],                    1, in_meta_Predictor[0].shape[3],                    0), kv_dtype_Predictor, kv_device, DEVICE_ID)
init_past_values_Predictor = create_ort_with_shape((1, in_meta_Predictor[num_layers_Predictor].shape[1], 1, 0, in_meta_Predictor[num_layers_Predictor].shape[4]), kv_dtype_Predictor, kv_device, DEVICE_ID)

# --- Greedy ---
ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
in_name_Greedy     = get_in_names(ort_session_Greedy)
out_name_Greedy    = get_out_names(ort_session_Greedy)

# --- Argmax ---
ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
in_name_Argmax     = get_in_names(ort_session_Argmax)
out_name_Argmax    = get_out_names(ort_session_Argmax)

# --- Beam Search (optional) ---
if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")

    ort_session_First_Beam     = create_session(onnx_model_First_Beam, **packed_settings)
    in_name_First_Beam         = get_in_names(ort_session_First_Beam)
    out_name_First_Beam        = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_kv      = in_name_First_Beam[:num_keys_values_Predictor]
    in_name_First_Beam_logits  = in_name_First_Beam[num_keys_values_Predictor]
    out_name_First_Beam_kv     = out_name_First_Beam[:num_keys_values_Predictor]
    out_name_First_Beam_others = out_name_First_Beam[num_keys_values_Predictor_plus_1:]

    ort_session_Second_Beam     = create_session(onnx_model_Second_Beam, **packed_settings)
    in_name_Second_Beam         = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam        = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_kv      = in_name_Second_Beam[:num_keys_values_Predictor]
    in_name_Second_Beam_logits  = in_name_Second_Beam[num_keys_values_Predictor]
    out_name_Second_Beam_kv     = out_name_Second_Beam[:num_keys_values_Predictor]
    out_name_Second_Beam_others = out_name_Second_Beam[num_keys_values_Predictor_plus_1:]

    beam_ids_buf   = create_ort_with_shape([BEAM_SIZE, 1], np.int32,               device_type, DEVICE_ID)
    beam_score_buf = create_ort_with_shape([BEAM_SIZE, 1], hidden_dtype_Predictor, device_type, DEVICE_ID)

# --- Penalty (optional) ---
if USE_PENALTY:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
    in_name_Penalty     = get_in_names(ort_session_Penalty)
    out_name_Penalty    = get_out_names(ort_session_Penalty)

    penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
input_ids_prompt          = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_tokens, device_type, DEVICE_ID)
init_history_len          = create_ort_with_data([0],                               np.int64,    device_type, DEVICE_ID)
init_predictor_ids_len    = create_ort_with_data([2],                               np.int64,    device_type, DEVICE_ID)
init_generated_codec      = create_ort_with_shape([1, 0],                           np.int32,    device_type, DEVICE_ID)
top_k                     = create_ort_with_data([TOP_K],                           np.int64,    device_type, DEVICE_ID)
beam_size                 = create_ort_with_data([BEAM_SIZE],                       np.int64,    device_type, DEVICE_ID)
gather_id_0               = create_ort_with_data([0],                               np.int32,    device_type, DEVICE_ID)
gather_id_cache           = [create_ort_with_data([i],                               np.int32,   device_type, DEVICE_ID) for i in range(MAX_SEQ_LEN)]
init_trailing_text_hidden = create_ort_with_shape([1, 1, in_meta_Embed_C[2].shape[2]],          hidden_dtype_Main, device_type, DEVICE_ID)
init_predictor_save_id    = create_ort_with_shape([BEAM_SIZE if USE_BEAM_SEARCH else 1, 0],     np.int32,          device_type, DEVICE_ID)
init_main_greedy_ids      = create_ort_with_shape([1, 0],                                       np.int32,          device_type, DEVICE_ID)
init_decode_attn_mask     = create_ort_with_shape([1, 1, 1, 1, 1],                              hidden_dtype_Main, device_type, DEVICE_ID)
init_frame_codec_ids      = create_ort_with_shape([1, 0],                                       np.int32,          device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# INPUT FEED DICTIONARIES
# ══════════════════════════════════════════════════════════════════════════════
input_feed_Encoder                       = {}
input_feed_Embed_A                       = {}
input_feed_Embed_B                       = {}
input_feed_Embed_C                       = {}
input_feed_Embed_D                       = {}
input_feed_Gather_0                      = {}
input_feed_Concat_Embed                  = {}
input_feed_Concat_Ids                    = {}
input_feed_Concat_Aux                    = {}
input_feed_Preprocess                    = {}
input_feed_Main_Rotary_Text_Prefill      = {}
input_feed_Main_Rotary_Text_Decode       = {}
input_feed_Predictor_Rotary_Text_Prefill = {}
input_feed_Predictor_Rotary_Text_Decode  = {}
input_feed_Main                          = {}
input_feed_Predictor                     = {}
input_feed_LmHead                        = {}
input_feed_Argmax                        = {}
input_feed_Greedy                        = {}
input_feed_First_Beam                    = {}
input_feed_Second_Beam                   = {}
input_feed_Penalty                       = {}
input_feed_Decoder                       = {}


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT AUDIO ENCODING & REFERENCE EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
prompt_audio = np.array(AudioSegment.from_file(prompt_audio_path).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
if USE_AUDIO_NORMALIZER:
    prompt_audio = audio_normalizer(prompt_audio)
prompt_audio = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_audio.reshape(1, 1, -1), device_type, DEVICE_ID)
save_generated_wav = []
empty_segment = np.zeros(int(OUT_SAMPLE_RATE * 0.2), dtype=np.int16)  # 200ms

input_feed_Encoder[in_name_Encoder[0]] = prompt_audio
encoder_start = time.perf_counter()
ref_code, ref_code_len, speaker_embed = ort_session_Encoder.run_with_ort_values(out_name_Encoder, input_feed_Encoder, run_options=run_options)
encoder_time = time.perf_counter() - encoder_start
print(f'\nEncoder time: {encoder_time:.2f} s')

# Convert speaker_embed from fp16 to fp32 if needed (Preprocess expects fp32)
if speaker_embed.data_type() == 'tensor(float16)':
    speaker_embed = onnxruntime.OrtValue.ortvalue_from_numpy(speaker_embed.numpy().astype(np.float32), device_type, DEVICE_ID)

input_feed_Embed_A[in_name_Embed_A[0]] = input_ids_prompt
ref_prompt_text_embed = ort_session_Embed_A.run_with_ort_values(out_name_Embed_A, input_feed_Embed_A, run_options=run_options)[0]

input_feed_Gather_0[in_name_Gather_0[0]] = ref_code
ref_ids_0 = ort_session_Gather_0.run_with_ort_values(out_name_Gather_0, input_feed_Gather_0, run_options=run_options)[0]

input_feed_Embed_B[in_name_Embed_B[0]] = ref_ids_0
codec_embed_0 = ort_session_Embed_B.run_with_ort_values(out_name_Embed_B, input_feed_Embed_B, run_options=run_options)[0]

input_feed_Embed_B[in_name_Embed_B[0]] = create_ort_with_data([[language_id]], np.int32, device_type, DEVICE_ID)
language_embed = ort_session_Embed_B.run_with_ort_values(out_name_Embed_B, input_feed_Embed_B, run_options=run_options)[0]

input_feed_Embed_C[in_name_Embed_C[0]] = ref_code
input_feed_Embed_C[in_name_Embed_C[1]] = codec_embed_0
input_feed_Embed_C[in_name_Embed_C[2]] = init_trailing_text_hidden
input_feed_Embed_C[in_name_Embed_C[3]] = gather_id_0
codec_embed = ort_session_Embed_C.run_with_ort_values(out_name_Embed_C, input_feed_Embed_C, run_options=run_options)[0]


# ══════════════════════════════════════════════════════════════════════════════
# PRE-POPULATE FIXED INPUT FEED ENTRIES
# ══════════════════════════════════════════════════════════════════════════════

# Preprocess fixed inputs (same for all targets)
input_feed_Preprocess[in_name_Preprocess[0]] = codec_embed
input_feed_Preprocess[in_name_Preprocess[1]] = speaker_embed
input_feed_Preprocess[in_name_Preprocess[2]] = language_embed
input_feed_Preprocess[in_name_Preprocess[3]] = ref_prompt_text_embed

# Decoder fixed inputs (same for all targets)
input_feed_Decoder[in_name_Decoder[0]] = ref_code
input_feed_Decoder[in_name_Decoder[1]] = ref_code_len

# Predictor Rotary Text Prefill fixed inputs
input_feed_Predictor_Rotary_Text_Prefill[in_name_Predictor_Rotary_Text_Prefill[0]] = init_predictor_ids_len
input_feed_Predictor_Rotary_Text_Prefill[in_name_Predictor_Rotary_Text_Prefill[1]] = init_history_len

# Predictor KV cache initial state
for i in range(num_layers_Predictor):
    input_feed_Predictor[in_name_Predictor[i]] = init_past_keys_Predictor
for i in range(num_layers_Predictor, num_keys_values_Predictor):
    input_feed_Predictor[in_name_Predictor[i]] = init_past_values_Predictor

# Penalty fixed inputs
if USE_PENALTY:
    input_feed_Penalty[in_name_Penalty[2]] = penalty_value
    input_feed_Penalty[in_name_Penalty[3]] = penalty_range

if USE_BEAM_SEARCH:
    input_feed_First_Beam[in_name_First_Beam[num_keys_values_Predictor_plus_2]]   = beam_size
    input_feed_Second_Beam[in_name_Second_Beam[num_keys_values_Predictor_plus_3]] = beam_size
    input_feed_Second_Beam[in_name_Second_Beam[num_keys_values_Predictor_plus_4]] = top_k


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTOR STEPS
# ══════════════════════════════════════════════════════════════════════════════
def predictor_steps(codec_token_main, last_hidden_state_Main, gather_id):
    if not USE_BEAM_SEARCH:
        input_feed_Concat_Aux[in_name_Concat_Ids[0]] = init_frame_codec_ids
        input_feed_Concat_Aux[in_name_Concat_Ids[1]] = codec_token_main
        frame_codec_buf = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]

    predictor_save_id = init_predictor_save_id

    input_feed_Embed_B[in_name_Embed_B[0]] = codec_token_main
    codec_embed_main = ort_session_Embed_B.run_with_ort_values(out_name_Embed_B, input_feed_Embed_B, run_options=run_options)[0]

    input_feed_Concat_Embed[in_name_Concat_Embed[0]] = last_hidden_state_Main
    input_feed_Concat_Embed[in_name_Concat_Embed[1]] = codec_embed_main
    hidden_states_predictor = ort_session_Concat_Embed.run_with_ort_values(out_name_Concat_Embed, input_feed_Concat_Embed, run_options=run_options)[0]

    rotary_cos_Predictor, rotary_sin_Predictor, attention_mask_Predictor, kv_seq_len_Predictor = ort_session_Predictor_Rotary_Text_Prefill.run_with_ort_values(out_name_Predictor_Rotary_Text_Prefill, input_feed_Predictor_Rotary_Text_Prefill, run_options=run_options)

    input_feed_Predictor[in_name_Predictor_others[3]] = attention_mask_Predictor

    # Reset Predictor KV cache to empty for this new predictor sequence
    for i in range(num_layers_Predictor):
        input_feed_Predictor[in_name_Predictor[i]] = init_past_keys_Predictor
    for i in range(num_layers_Predictor, num_keys_values_Predictor):
        input_feed_Predictor[in_name_Predictor[i]] = init_past_values_Predictor

    is_prefill_step_Predictor = True

    for num_decode_Predictor in range(NUM_CODE_GROUPS_MINUS):
        input_feed_Predictor[in_name_Predictor_others[0]] = hidden_states_predictor
        input_feed_Predictor[in_name_Predictor_others[1]] = rotary_cos_Predictor
        input_feed_Predictor[in_name_Predictor_others[2]] = rotary_sin_Predictor

        all_outputs_Predictor       = ort_session_Predictor.run_with_ort_values(out_name_Predictor, input_feed_Predictor, run_options=run_options)
        last_hidden_state_Predictor = all_outputs_Predictor[num_keys_values_Predictor]

        input_feed_LmHead[in_name_LmHead[0]] = last_hidden_state_Predictor
        logits_Predictor = ort_session_LmHead[num_decode_Predictor].run_with_ort_values(out_name_LmHead, input_feed_LmHead, run_options=run_options)[0]

        if USE_PENALTY and not is_prefill_step_Predictor:
            input_feed_Penalty[in_name_Penalty[0]] = logits_Predictor
            input_feed_Penalty[in_name_Penalty[1]] = predictor_save_id
            logits_Predictor = ort_session_Penalty.run_with_ort_values(out_name_Penalty, input_feed_Penalty, run_options=run_options)[0]

        if USE_BEAM_SEARCH:
            if is_prefill_step_Predictor:
                input_feed_First_Beam.update(zip(in_name_First_Beam_kv, all_outputs_Predictor))
                input_feed_First_Beam[in_name_First_Beam_logits]                            = logits_Predictor
                input_feed_First_Beam[in_name_First_Beam[num_keys_values_Predictor_plus_1]] = predictor_save_id
                all_outputs_Predictor = ort_session_First_Beam.run_with_ort_values(out_name_First_Beam, input_feed_First_Beam, run_options=run_options)
            else:
                input_feed_Second_Beam.update(zip(in_name_Second_Beam_kv, all_outputs_Predictor))
                input_feed_Second_Beam[in_name_Second_Beam_logits]                            = logits_Predictor
                input_feed_Second_Beam[in_name_Second_Beam[num_keys_values_Predictor_plus_1]] = predictor_save_id
                input_feed_Second_Beam[in_name_Second_Beam[num_keys_values_Predictor_plus_2]] = predictor_beam_score
                all_outputs_Predictor = ort_session_Second_Beam.run_with_ort_values(out_name_Second_Beam, input_feed_Second_Beam, run_options=run_options)

            predictor_save_id      = all_outputs_Predictor[num_keys_values_Predictor]
            predictor_beam_score   = all_outputs_Predictor[num_keys_values_Predictor_plus_1]
            next_codec_ids         = all_outputs_Predictor[num_keys_values_Predictor_plus_2]
            codec_token_predictor  = all_outputs_Predictor[num_keys_values_Predictor_plus_3]
        else:
            if USE_PENALTY:
                input_feed_Greedy[in_name_Greedy[0]] = logits_Predictor
                input_feed_Greedy[in_name_Greedy[1]] = predictor_save_id
                codec_token_predictor, predictor_save_id = ort_session_Greedy.run_with_ort_values(out_name_Greedy, input_feed_Greedy, run_options=run_options)
            else:
                input_feed_Argmax[in_name_Argmax[0]] = logits_Predictor
                codec_token_predictor = ort_session_Argmax.run_with_ort_values(out_name_Argmax, input_feed_Argmax, run_options=run_options)[0]

            next_codec_ids = codec_token_predictor

        input_feed_Predictor.update(zip(in_name_Predictor_kv, all_outputs_Predictor))

        input_feed_Embed_D[in_name_Embed_D[0]] = next_codec_ids
        hidden_states_predictor = ort_session_Embed_D[num_decode_Predictor].run_with_ort_values(out_name_Embed_D, input_feed_Embed_D, run_options=run_options)[0]

        input_feed_Predictor_Rotary_Text_Decode[in_name_Predictor_Rotary_Text_Decode[0]] = kv_seq_len_Predictor
        rotary_cos_Predictor, rotary_sin_Predictor, kv_seq_len_Predictor = ort_session_Predictor_Rotary_Text_Decode.run_with_ort_values(out_name_Predictor_Rotary_Text_Decode, input_feed_Predictor_Rotary_Text_Decode, run_options=run_options)

        if is_prefill_step_Predictor:
            input_feed_Predictor[in_name_Predictor_others[3]] = init_decode_attn_mask
            is_prefill_step_Predictor = False

        if not USE_BEAM_SEARCH:
            input_feed_Concat_Aux[in_name_Concat_Ids[0]] = frame_codec_buf
            input_feed_Concat_Aux[in_name_Concat_Ids[1]] = codec_token_predictor
            frame_codec_buf = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]

    if USE_BEAM_SEARCH:
        input_feed_Gather_0[in_name_Gather_0[0]] = predictor_save_id
        best_predictor_save_id = ort_session_Gather_0.run_with_ort_values(out_name_Gather_0, input_feed_Gather_0, run_options=run_options)[0]

        input_feed_Concat_Aux[in_name_Concat_Ids[0]] = codec_token_main
        input_feed_Concat_Aux[in_name_Concat_Ids[1]] = best_predictor_save_id
        frame_codec_ids = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]
    else:
        if USE_PENALTY:
            input_feed_Concat_Aux[in_name_Concat_Ids[0]] = codec_token_main
            input_feed_Concat_Aux[in_name_Concat_Ids[1]] = predictor_save_id
            frame_codec_ids = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]

        else:
            frame_codec_ids = frame_codec_buf

    input_feed_Embed_C[in_name_Embed_C[0]] = frame_codec_ids
    input_feed_Embed_C[in_name_Embed_C[1]] = codec_embed_main
    input_feed_Embed_C[in_name_Embed_C[3]] = gather_id
    hidden_states_main = ort_session_Embed_C.run_with_ort_values(out_name_Embed_C, input_feed_Embed_C, run_options=run_options)[0]

    input_feed_Concat_Ids[in_name_Concat_Ids[1]] = frame_codec_ids
    generated_codec = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Ids, run_options=run_options)[0]

    return hidden_states_main, generated_codec


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
total_audio_samples = 0
total_generation_time = 0.0
total_decoder_time = 0.0

for target_idx, target in enumerate(target_tts):
    target_tokens    = tokenizer(target, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids_target = onnxruntime.OrtValue.ortvalue_from_numpy(target_tokens, device_type, DEVICE_ID)
    main_greedy_ids  = init_main_greedy_ids
    is_prefill_step_Main = True
    main_time_total = 0.0
    predictor_time_total = 0.0

    input_feed_Embed_A[in_name_Embed_A[0]] = input_ids_target
    target_text_embed = ort_session_Embed_A.run_with_ort_values(out_name_Embed_A, input_feed_Embed_A, run_options=run_options)[0]

    input_feed_Preprocess[in_name_Preprocess[4]] = target_text_embed
    hidden_states, ids_len, trailing_text_hidden, trailing_len_minus = ort_session_Preprocess.run_with_ort_values(out_name_Preprocess, input_feed_Preprocess, run_options=run_options)

    trailing_len_minus_numpy               = trailing_len_minus.numpy()
    input_feed_Embed_C[in_name_Embed_C[2]] = trailing_text_hidden

    input_feed_Main_Rotary_Text_Prefill[in_name_Main_Rotary_Text_Prefill[0]] = ids_len
    input_feed_Main_Rotary_Text_Prefill[in_name_Main_Rotary_Text_Prefill[1]] = init_history_len
    rotary_cos_Main, rotary_sin_Main, attention_mask_Main, kv_seq_len_Main   = ort_session_Main_Rotary_Text_Prefill.run_with_ort_values(out_name_Main_Rotary_Text_Prefill, input_feed_Main_Rotary_Text_Prefill, run_options=run_options)

    input_feed_Main[in_name_Main_others[3]] = attention_mask_Main

    # Reset Main KV cache to empty for this new target
    for i in range(num_layers_Main):
        input_feed_Main[in_name_Main[i]] = init_past_keys_Main
    for i in range(num_layers_Main, num_keys_values_Main):
        input_feed_Main[in_name_Main[i]] = init_past_values_Main

    num_decode_Main = 0
    generate_limit  = MAX_SEQ_LEN - ids_len.numpy()

    while num_decode_Main < generate_limit:
        input_feed_Main[in_name_Main_others[0]] = hidden_states
        input_feed_Main[in_name_Main_others[1]] = rotary_cos_Main
        input_feed_Main[in_name_Main_others[2]] = rotary_sin_Main

        main_step_start        = time.perf_counter()
        all_outputs_Main       = ort_session_Main.run_with_ort_values(out_name_Main, input_feed_Main, run_options=run_options)
        main_time_total       += time.perf_counter() - main_step_start
        last_hidden_state_Main = all_outputs_Main[num_keys_values_Main]
        logits_Main            = all_outputs_Main[num_keys_values_Main_plus_1]

        if USE_PENALTY and num_decode_Main >= PENALTY_RANGE:
            input_feed_Penalty[in_name_Penalty[0]] = logits_Main
            input_feed_Penalty[in_name_Penalty[1]] = main_greedy_ids
            logits_Main = ort_session_Penalty.run_with_ort_values(out_name_Penalty, input_feed_Penalty, run_options=run_options)[0]

        input_feed_Argmax[in_name_Argmax[0]] = logits_Main
        codec_token_main = ort_session_Argmax.run_with_ort_values(out_name_Argmax, input_feed_Argmax, run_options=run_options)[0]

        max_logits_idx = codec_token_main.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET and num_decode_Main >= MIN_SEQ_LEN:
            break

        input_feed_Concat_Aux[in_name_Concat_Ids[0]] = main_greedy_ids
        input_feed_Concat_Aux[in_name_Concat_Ids[1]] = codec_token_main
        main_greedy_ids = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]

        if is_prefill_step_Main:
            input_feed_Main[in_name_Main_others[3]]      = init_decode_attn_mask
            input_feed_Concat_Ids[in_name_Concat_Ids[0]] = init_generated_codec
            is_prefill_step_Main = False

        gather_id = gather_id_cache[num_decode_Main] if num_decode_Main <= trailing_len_minus_numpy else trailing_len_minus
        predictor_step_start = time.perf_counter()
        hidden_states, generated_codec = predictor_steps(codec_token_main, last_hidden_state_Main, gather_id)
        predictor_time_total += time.perf_counter() - predictor_step_start
        input_feed_Concat_Ids[in_name_Concat_Ids[0]] = generated_codec

        input_feed_Main_Rotary_Text_Decode[in_name_Main_Rotary_Text_Decode[0]] = kv_seq_len_Main
        rotary_cos_Main, rotary_sin_Main, kv_seq_len_Main = ort_session_Main_Rotary_Text_Decode.run_with_ort_values(out_name_Main_Rotary_Text_Decode, input_feed_Main_Rotary_Text_Decode, run_options=run_options)
        input_feed_Main.update(zip(in_name_Main_kv, all_outputs_Main))

        num_decode_Main += 1

    input_feed_Decoder[in_name_Decoder[2]] = generated_codec
    decoder_start = time.perf_counter()
    generated_wav = ort_session_Decoder.run_with_ort_values(out_name_Decoder, input_feed_Decoder, run_options=run_options)[0]
    decoder_time = time.perf_counter() - decoder_start
    generated_wav = generated_wav.numpy().reshape(-1)
    if USE_AUDIO_NORMALIZER:
        generated_wav = audio_normalizer(generated_wav)

    # Timing statistics for this target
    main_tokens_per_sec = num_decode_Main / main_time_total if main_time_total > 0 else 0
    predictor_tokens_per_sec = (num_decode_Main * NUM_CODE_GROUPS_MINUS) / predictor_time_total if predictor_time_total > 0 else 0
    audio_duration = len(generated_wav) / OUT_SAMPLE_RATE
    target_gen_time = main_time_total + predictor_time_total + decoder_time
    total_audio_samples += len(generated_wav)
    total_generation_time += target_gen_time
    total_decoder_time += decoder_time

    print(f'\n┌──────────────────┬──────────┬──────────┬──────────────┐')
    print(f'│  Target {target_idx:<9}│          │          │              │')
    print(f'├──────────────────┼──────────┼──────────┼──────────────┤')
    print(f'│ Stage            │   Tokens │  Time(s) │    Tokens/s  │')
    print(f'├──────────────────┼──────────┼──────────┼──────────────┤')
    print(f'│ Main             │ {num_decode_Main:>8d} │ {main_time_total:>8.2f} │ {main_tokens_per_sec:>12.2f} │')
    print(f'│ Predictor        │ {num_decode_Main * NUM_CODE_GROUPS_MINUS:>8d} │ {predictor_time_total:>8.2f} │ {predictor_tokens_per_sec:>12.2f} │')
    print(f'│ Decoder          │        — │ {decoder_time:>8.2f} │            — │')
    print(f'├──────────────────┼──────────┴──────────┴──────────────┤')
    print(f'│ Audio duration   │ {audio_duration:>8.2f} s{" ":>25}│')
    print(f'│ Target RTF       │ {target_gen_time / audio_duration:>8.3f}{" ":>27}│' if audio_duration > 0 else f'│ Target RTF       │{"N/A":>9}{" ":>27}│')
    print(f'└──────────────────┴────────────────────────────────────┘')

    save_generated_wav.append(generated_wav)
    save_generated_wav.append(empty_segment)  # Append silence between target sentence
if save_generated_wav:
    total_audio_duration = total_audio_samples / OUT_SAMPLE_RATE
    overall_rtf = (encoder_time + total_generation_time) / total_audio_duration if total_audio_duration > 0 else float('inf')
    print(f'\n┌─────────────────────────┬─────────────────┐')
    print(f'│    Overall Statistics   │                 │')
    print(f'├─────────────────────────┼─────────────────┤')
    print(f'│ Encoder time            │ {encoder_time:>8.2f} s      │')
    print(f'│ Total decoder time      │ {total_decoder_time:>8.2f} s      │')
    print(f'│ Total generation time   │ {total_generation_time:>8.2f} s      │')
    print(f'│ Total audio duration    │ {total_audio_duration:>8.2f} s      │')
    print(f'│ Overall RTF             │ {overall_rtf:>8.3f}        │')
    print(f'└─────────────────────────┴─────────────────┘')

    save_generated_wav = np.concatenate(save_generated_wav)
    sf.write(generated_audio_path, save_generated_wav, OUT_SAMPLE_RATE, format='WAVEX')
