import time
import soundfile as sf
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from modeling_modified.text_normalize import TextNormalizer
from transformers import LlamaTokenizerFast


path_voxcpm                         = r'/home/DakeQQ/Downloads/VoxCPM1.5'                                 # Set the folder path where the VoxCPM1.5 project downloaded.
onnx_model_Text_Embed               = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Text_Embed.onnx'       # Assign a path where the exported VoxCPM model stored.
onnx_model_VAE_Encoder              = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_VAE_Encoder.onnx'
onnx_model_Feat_Encoder             = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Feat_Encoder.onnx'
onnx_model_Feat_Cond                = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Feat_Cond.onnx'
onnx_model_Concat                   = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Concat.onnx'
onnx_model_Rotary_Mask_Text_Prefill = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Rotary_Mask_Text_Prefill.onnx'
onnx_model_Rotary_Mask_Text_Decode  = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Rotary_Mask_Text_Decode.onnx'
onnx_model_Main                     = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Main.onnx'
onnx_model_Feat_Decoder             = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Feat_Decoder.onnx'
onnx_model_VAE_Decoder              = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_VAE_Decoder.onnx'

prompt_audio_path = "./example/basic_ref_zh.wav"                                    # optional: path to a prompt speech for voice cloning else None.
prompt_text = "对，这就是我，万人敬仰的太乙真人。"                                        # The reference text for the prompt speech.
target_tts = ["大家好，我现在正在大可奇奇体验AI科技。", "Hello everyone, I'm currently experiencing DakeQQ's AI technology."]  # The test query after the export process.
generated_audio_path = r"./generated.wav"                                           # The generated audio path.

# === Decoding limits & tokens ===
STOP_TOKEN = [1]                         # The stop_id in VoxCPM is "1"
MAX_SEQ_LEN = 1024                       # The max decode length; keep the same as exported model.
MIN_SEQ_LEN = 2                          # The min decode length
DECODE_LIMIT_FACTOR = 6                  # Decode length limit factor, integer >= 1

# === Audio configuration ===
IN_SAMPLE_RATE = 44100                   # Input prompt audio sample rate; keep the same as exported model.
OUT_SAMPLE_RATE = 44100                  # Output audio sample rate; keep the same as exported model.

# === Guidance, diffusion & randomness ===
FIXED_TIMESTEPS = 10                     # Fixed timesteps; keep the same as exported model.
CFG_VALUE = 2.5                          # Lower values result in more natural speech for long text, while higher values stay closer to the original sound features.
RANDOM_SEED = 1                          # Global random seed

# === Feature flags ===
STREAMING = False                        # Enable streaming synthesis. Unlike the official implementation, this version processes a single latent at a time for faster performance, albeit with potential discontinuities during piece-by-piece decoding.
USE_TEXT_NORMALIZER = True               # Use text normalizer
USE_AUDIO_NORMALIZER = False             # Use an audio normalizer to stabilize loudness, though this may result in a loss of original audio characteristics.

# === ONNX / runtime configuration ===
MAX_THREADS = 0                          # Parallel CPU threads, 0 for auto
DEVICE_ID = 0                            # Device id, default 0
ORT_LOG = False                          # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16 = False                         # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers = []            # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def audio_normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


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
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

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
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)


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
        'gpu_mem_limit':                      24 * (1024 **3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',   # ["default", "high_performance", "minimum_power"] ; Default (Gpus first), HighPerformance (GPUs first), LowPower (NPUs first)
        'device_filter':              'gpu',                # [gpu, npu, any],
        'disable_metacommands':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'               # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
# --- Text Embed ---
ort_session_Text_Embed = create_session(onnx_model_Text_Embed, **packed_settings)
in_name_Text_Embed     = get_in_names(ort_session_Text_Embed)[0]
out_name_Text_Embed    = [get_out_names(ort_session_Text_Embed)[0]]

# --- VAE Encoder ---
ort_session_VAE_Encoder = create_session(onnx_model_VAE_Encoder, **packed_settings)
in_name_VAE_Encoder     = get_in_names(ort_session_VAE_Encoder)[0]
out_name_VAE_Encoder    = [get_out_names(ort_session_VAE_Encoder)[0]]

# --- Feat Encoder ---
ort_session_Feat_Encoder = create_session(onnx_model_Feat_Encoder, **packed_settings)
in_name_Feat_Encoder     = get_in_names(ort_session_Feat_Encoder)[0]
out_name_Feat_Encoder    = [get_out_names(ort_session_Feat_Encoder)[0]]

# --- Feat Cond ---
ort_session_Feat_Cond    = create_session(onnx_model_Feat_Cond, **packed_settings)
model_dtype_Feat_Cond    = np.float16 if 'float16' in ort_session_Feat_Cond._inputs_meta[0].type else np.float32
in_name_Feat_Cond        = get_in_names(ort_session_Feat_Cond)[0]
out_name_Feat_Cond       = [get_out_names(ort_session_Feat_Cond)[0]]

# --- Concat ---
ort_session_Concat = create_session(onnx_model_Concat, **packed_settings)
in_name_Concat     = get_in_names(ort_session_Concat)
out_name_Concat    = get_out_names(ort_session_Concat)

# --- Rotary + Mask (Text Prefill) ---
ort_session_Rotary_Mask_Text_Prefill = create_session(onnx_model_Rotary_Mask_Text_Prefill, **packed_settings)
in_name_Rotary_Mask_Text_Prefill     = get_in_names(ort_session_Rotary_Mask_Text_Prefill)
out_name_Rotary_Mask_Text_Prefill    = get_out_names(ort_session_Rotary_Mask_Text_Prefill)

# --- Rotary + Mask (Text Decode) ---
ort_session_Rotary_Mask_Text_Decode = create_session(onnx_model_Rotary_Mask_Text_Decode, **packed_settings)
in_name_Rotary_Mask_Text_Decode     = get_in_names(ort_session_Rotary_Mask_Text_Decode)
out_name_Rotary_Mask_Text_Decode    = get_out_names(ort_session_Rotary_Mask_Text_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
print(f"\nUsable Providers: {ort_session_Main.get_providers()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
model_dtype_Main       = np.float16 if 'float16' in ort_session_Main._inputs_meta[0].type else np.float32
in_name_Main           = get_in_names(ort_session_Main)
out_name_Main          = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)

# --- Feat Decoder ---
ort_session_Feat_Decoder = create_session(onnx_model_Feat_Decoder, **packed_settings)
model_dtype_Feat_Decoder = np.float16 if 'float16' in ort_session_Feat_Decoder._inputs_meta[2].type else np.float32
in_name_Feat_Decoder     = get_in_names(ort_session_Feat_Decoder)
out_name_Feat_Decoder    = get_out_names(ort_session_Feat_Decoder)

# --- VAE Decoder ---
ort_session_VAE_Decoder    = create_session(onnx_model_VAE_Decoder, **packed_settings)
model_dtype_VAE_Decoder    = np.float16 if 'float16' in ort_session_VAE_Decoder._inputs_meta[0].type else np.float32
DYNAMIC_SHAPE_VAE_DECODE   = isinstance(ort_session_VAE_Decoder._inputs_meta[0].shape[1], str)
in_name_VAE_Decoder        = get_in_names(ort_session_VAE_Decoder)[0]
out_name_VAE_Decoder       = get_out_names(ort_session_VAE_Decoder)
half_decode_len            = 7056  # Fixed for VoxCPM1.5


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
generate_limit  = MAX_SEQ_LEN - 1
num_keys_values = amount_of_outputs_Main - 3
num_layers      = num_keys_values // 2

# Derived index offsets
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5

_meta = ort_session_Main._inputs_meta


# ══════════════════════════════════════════════════════════════════════════════
# STATIC ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
# --- Scalars & Lengths ---
init_history_len       = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_concat_text_len   = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)

# --- Special Tokens ---
init_audio_start_ids   = create_ort_with_data([[101]], np.int32, device_type, DEVICE_ID)

# --- Masks ---
init_mask_prefill          = create_ort_with_data([1], np.int8, device_type, DEVICE_ID)
init_decode_attention_mask = create_ort_with_shape((1, 1, 1, 1), model_dtype_Main, device_type, DEVICE_ID)

# --- KV Cache & Embedding Shapes ---
shape_keys   = (_meta[0].shape[0],          1, _meta[0].shape[2],          0)
shape_vals   = (_meta[num_layers].shape[0],  1, 0, _meta[num_layers].shape[3])
shape_embed  = (1, 0, _meta[num_keys_values].shape[2])
shape_latent = (ort_session_VAE_Decoder._inputs_meta[0].shape[0], 0, ort_session_VAE_Decoder._inputs_meta[0].shape[2])

init_past_keys_Main   = create_ort_with_shape(shape_keys,    model_dtype_Main,        device_type, DEVICE_ID)
init_past_values_Main = create_ort_with_shape(shape_vals,    model_dtype_Main,        device_type, DEVICE_ID)
init_feat_embed       = create_ort_with_shape(shape_embed,   model_dtype_Main,        device_type, DEVICE_ID)
init_latent_pred      = create_ort_with_shape(shape_latent,  model_dtype_VAE_Decoder, device_type, DEVICE_ID)

# --- CFG Values ---
cfg_value       = create_ort_with_data([CFG_VALUE],       model_dtype_Feat_Decoder, device_type, DEVICE_ID)
cfg_value_minus = create_ort_with_data([1.0 - CFG_VALUE], model_dtype_Feat_Decoder, device_type, DEVICE_ID)

# --- Time Steps ---
timesteps      = FIXED_TIMESTEPS - 1
init_cfm_steps = create_ort_with_data([0], np.int32, device_type, DEVICE_ID)

# --- Audio Post-processing ---
blank_segment = np.zeros((1, 1, int(OUT_SAMPLE_RATE * 0.1)), dtype=np.int16)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION SETUP & IO FEEDS
# ══════════════════════════════════════════════════════════════════════════════
input_feed_Text_Embed               = {}
input_feed_VAE_Encoder              = {}
input_feed_Feat_Encoder             = {}
input_feed_Feat_Cond                = {}
input_feed_Concat                   = {}
input_feed_Rotary_Mask_Text_Prefill = {}
input_feed_Rotary_Mask_Text_Decode  = {}
input_feed_Main                     = {}
input_feed_Feat_Decoder             = {}
input_feed_VAE_Decoder              = {}

# Audio Start Embedding
input_feed_Text_Embed[in_name_Text_Embed] = init_audio_start_ids
audio_start_embed = ort_session_Text_Embed.run_with_ort_values(out_name_Text_Embed, input_feed_Text_Embed, run_options=run_options)[0]

# Feat Cond Initialization
input_feed_Feat_Cond[in_name_Feat_Cond] = create_ort_with_shape((1, ort_session_Feat_Cond._inputs_meta[0].shape[1], ort_session_Feat_Cond._inputs_meta[0].shape[2]), model_dtype_Feat_Cond, device_type, DEVICE_ID)
init_feat_cond_0 = ort_session_Feat_Cond.run_with_ort_values(out_name_Feat_Cond, input_feed_Feat_Cond, run_options=run_options)[0]

# Feat Decoder: Fixed Inputs
input_feed_Feat_Decoder[in_name_Feat_Decoder[4]] = cfg_value
input_feed_Feat_Decoder[in_name_Feat_Decoder[5]] = cfg_value_minus

# Rotary Mask Prefill: Fixed Inputs
input_feed_Rotary_Mask_Text_Prefill[in_name_Rotary_Mask_Text_Prefill[1]] = init_history_len
input_feed_Rotary_Mask_Text_Prefill[in_name_Rotary_Mask_Text_Prefill[2]] = init_mask_prefill


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & PROMPT HANDLING
# ══════════════════════════════════════════════════════════════════════════════
tokenizer       = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(path_voxcpm))
text_normalizer = TextNormalizer()

if prompt_audio_path:
    if prompt_text:
        use_prompt_audio = True
        audio = np.array(AudioSegment.from_file(prompt_audio_path).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
        if USE_AUDIO_NORMALIZER:
            audio = audio_normalizer(audio)
        audio = onnxruntime.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), device_type, DEVICE_ID)
    else:
        use_prompt_audio = False
        print("Warning: No prompt text provided, so the prompt audio will be ignored.\n")
else:
    use_prompt_audio = False
    print("Info: No prompt audio provided, using ransom seed to generate voice.\n")

count_time = time.time()
if use_prompt_audio:
    # VAE Encoder
    input_feed_VAE_Encoder[in_name_VAE_Encoder] = audio
    audio_feat = ort_session_VAE_Encoder.run_with_ort_values(out_name_VAE_Encoder, input_feed_VAE_Encoder, run_options=run_options)[0]

    # Feat Cond
    input_feed_Feat_Cond[in_name_Feat_Cond] = audio_feat
    init_feat_cond = ort_session_Feat_Cond.run_with_ort_values(out_name_Feat_Cond, input_feed_Feat_Cond, run_options=run_options)[0]

    # Text Processing
    if USE_TEXT_NORMALIZER:
        prompt_text = text_normalizer.normalize(prompt_text)
    prompt_ids      = np.array([tokenizer(prompt_text)], dtype=np.int32)
    prompt_text_len = prompt_ids.shape[-1]

    # Text Embed
    input_feed_Text_Embed[in_name_Text_Embed] = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_ids, device_type, DEVICE_ID)
    prompt_embed = ort_session_Text_Embed.run_with_ort_values(out_name_Text_Embed, input_feed_Text_Embed, run_options=run_options)[0]
else:
    init_feat_cond  = init_feat_cond_0
    prompt_text_len = 0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
save_audio_out = []

for sentence in target_tts:
    print(f"Convert to Speech: {sentence}")
    if USE_TEXT_NORMALIZER:
        sentence = text_normalizer.normalize(sentence)

    # --- Encode Target Text ---
    target_ids = np.array([tokenizer(sentence)], dtype=np.int32)
    input_feed_Text_Embed[in_name_Text_Embed] = onnxruntime.OrtValue.ortvalue_from_numpy(target_ids, device_type, DEVICE_ID)
    target_embed = ort_session_Text_Embed.run_with_ort_values(out_name_Text_Embed, input_feed_Text_Embed, run_options=run_options)[0]

    # --- Combine Embeddings ---
    if use_prompt_audio:
        input_feed_Concat[in_name_Concat[0]] = prompt_embed
        input_feed_Concat[in_name_Concat[1]] = target_embed
        target_embed, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)

    input_feed_Concat[in_name_Concat[0]] = target_embed
    input_feed_Concat[in_name_Concat[1]] = audio_start_embed
    concat_embed, concat_text_len = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)

    # --- Calculate Max Length & Initial Features ---
    if use_prompt_audio:
        input_feed_Feat_Encoder[in_name_Feat_Encoder] = audio_feat
        feat_embed = ort_session_Feat_Encoder.run_with_ort_values(out_name_Feat_Encoder, input_feed_Feat_Encoder, run_options=run_options)[0]

        input_feed_Concat[in_name_Concat[0]] = concat_embed
        input_feed_Concat[in_name_Concat[1]] = feat_embed
        concat_embed, ids_len = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
    else:
        feat_embed = init_feat_embed
        ids_len    = concat_text_len

    max_len = min((concat_text_len.numpy() - prompt_text_len) * DECODE_LIMIT_FACTOR + 10, generate_limit - ids_len.numpy())

    # --- Rotary Embeddings & Causal Mask (Prefill) ---
    input_feed_Rotary_Mask_Text_Prefill[in_name_Rotary_Mask_Text_Prefill[0]] = ids_len
    rotary_cos, rotary_sin, attention_mask, kv_seq_len = ort_session_Rotary_Mask_Text_Prefill.run_with_ort_values(
        out_name_Rotary_Mask_Text_Prefill, input_feed_Rotary_Mask_Text_Prefill, run_options=run_options)

    # --- Prepare Main Decoder Inputs ---
    input_feed_Main[in_name_Main[num_keys_values]]          = feat_embed
    input_feed_Main[in_name_Main[num_keys_values_plus_1]]   = concat_text_len
    input_feed_Main[in_name_Main[num_keys_values_plus_2]]   = concat_embed
    input_feed_Main[in_name_Main[num_keys_values_plus_3]]   = rotary_cos
    input_feed_Main[in_name_Main[num_keys_values_plus_4]]   = rotary_sin
    input_feed_Main[in_name_Main[num_keys_values_plus_5]]   = attention_mask

    # Reset KV Cache
    for i in range(num_layers):
        input_feed_Main[in_name_Main[i]] = init_past_keys_Main
    for i in range(num_layers, num_keys_values):
        input_feed_Main[in_name_Main[i]] = init_past_values_Main

    feat_cond = init_feat_cond

    if not STREAMING:
        save_latent = init_latent_pred if DYNAMIC_SHAPE_VAE_DECODE else []

    # ──────────────────────────────────────────────────────────────────────────
    # AUTO-REGRESSIVE DECODING
    # ──────────────────────────────────────────────────────────────────────────
    num_decode   = 0
    start_decode = time.time()

    while num_decode < max_len:
        # --- Transformer ---
        all_outputs_Main = ort_session_Main.run_with_ort_values(out_name_Main, input_feed_Main, run_options=run_options)

        # --- Flow Matching / Diffusion ---
        input_feed_Feat_Decoder[in_name_Feat_Decoder[0]] = init_cfm_steps
        input_feed_Feat_Decoder[in_name_Feat_Decoder[1]] = all_outputs_Main[num_keys_values]
        input_feed_Feat_Decoder[in_name_Feat_Decoder[2]] = all_outputs_Main[num_keys_values_plus_1]
        input_feed_Feat_Decoder[in_name_Feat_Decoder[3]] = feat_cond

        for i in range(timesteps):
            all_outputs_Feat_Decoder = ort_session_Feat_Decoder.run_with_ort_values(out_name_Feat_Decoder, input_feed_Feat_Decoder, run_options=run_options)
            input_feed_Feat_Decoder[in_name_Feat_Decoder[0]] = all_outputs_Feat_Decoder[0]
            input_feed_Feat_Decoder[in_name_Feat_Decoder[1]] = all_outputs_Feat_Decoder[1]

        latent_pred = all_outputs_Feat_Decoder[1]

        # --- Handle Output ---
        if STREAMING:
            if num_decode < 1:
                pre_latent_pred = latent_pred
            else:
                input_feed_Concat[in_name_Concat[0]] = pre_latent_pred
                input_feed_Concat[in_name_Concat[1]] = latent_pred
                save_latent, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
                input_feed_VAE_Decoder[in_name_VAE_Decoder] = save_latent
                audio_out, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
                pre_latent_pred = latent_pred
                audio_out = audio_out.numpy()
                if num_decode > 1:
                    audio_out = audio_out[..., half_decode_len:]
                save_audio_out.append(audio_out)
        else:
            if DYNAMIC_SHAPE_VAE_DECODE:
                input_feed_Concat[in_name_Concat[0]] = save_latent
                input_feed_Concat[in_name_Concat[1]] = latent_pred
                save_latent, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
            else:
                save_latent.append(latent_pred)

        # --- Check Stop Token ---
        if num_decode >= MIN_SEQ_LEN and all_outputs_Main[num_keys_values_plus_2].numpy() in STOP_TOKEN:
            break

        # --- Update Inputs for Next Iteration ---
        input_feed_Feat_Encoder[in_name_Feat_Encoder] = latent_pred
        feat_embed = ort_session_Feat_Encoder.run_with_ort_values(out_name_Feat_Encoder, input_feed_Feat_Encoder, run_options=run_options)[0]

        input_feed_Feat_Cond[in_name_Feat_Cond] = latent_pred
        feat_cond = ort_session_Feat_Cond.run_with_ort_values(out_name_Feat_Cond, input_feed_Feat_Cond, run_options=run_options)[0]

        input_feed_Main.update(zip(in_name_Main[:num_keys_values], all_outputs_Main))
        input_feed_Main[in_name_Main[num_keys_values]]        = feat_embed
        input_feed_Main[in_name_Main[num_keys_values_plus_2]] = feat_embed

        # Rotary embeddings for next decode step
        input_feed_Rotary_Mask_Text_Decode[in_name_Rotary_Mask_Text_Decode[0]] = kv_seq_len
        rotary_cos, rotary_sin, kv_seq_len = ort_session_Rotary_Mask_Text_Decode.run_with_ort_values(
            out_name_Rotary_Mask_Text_Decode, input_feed_Rotary_Mask_Text_Decode, run_options=run_options)
        input_feed_Main[in_name_Main[num_keys_values_plus_3]] = rotary_cos
        input_feed_Main[in_name_Main[num_keys_values_plus_4]] = rotary_sin

        if num_decode < 1:
            input_feed_Main[in_name_Main[num_keys_values_plus_1]] = init_concat_text_len
            input_feed_Main[in_name_Main[num_keys_values_plus_5]] = init_decode_attention_mask

        num_decode += 1
        print(f"    Decode: {num_decode}")

    print(f"\nDecode Speed: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s\n")

    # ──────────────────────────────────────────────────────────────────────────
    # FINALIZE SENTENCE AUDIO (NON-STREAMING)
    # ──────────────────────────────────────────────────────────────────────────
    if not STREAMING:
        if DYNAMIC_SHAPE_VAE_DECODE:
            input_feed_VAE_Decoder[in_name_VAE_Decoder] = save_latent
            audio_out, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
            save_audio_out.append(audio_out.numpy())
        else:
            input_feed_Concat[in_name_Concat[0]] = save_latent[0]
            input_feed_Concat[in_name_Concat[1]] = save_latent[1]
            concat_latent, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
            input_feed_VAE_Decoder[in_name_VAE_Decoder] = concat_latent
            audio_out, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
            save_audio_out.append(audio_out.numpy())
            for i in range(2, len(save_latent)):
                input_feed_Concat[in_name_Concat[0]] = save_latent[i - 1]
                input_feed_Concat[in_name_Concat[1]] = save_latent[i]
                concat_latent, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
                input_feed_VAE_Decoder[in_name_VAE_Decoder] = concat_latent
                audio_out, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
                audio_out = audio_out.numpy()[..., half_decode_len:]
                save_audio_out.append(audio_out)

    save_audio_out.append(blank_segment)


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING & STATS
# ══════════════════════════════════════════════════════════════════════════════
cost_time = time.time() - count_time
audio_out = np.concatenate(save_audio_out, axis=-1).reshape(-1)
if USE_AUDIO_NORMALIZER:
    audio_out = audio_normalizer(audio_out)
sf.write(generated_audio_path, audio_out, OUT_SAMPLE_RATE, format='WAVEX')

total_audio_duration = (audio_out.shape[-1] - blank_segment.shape[-1] * len(target_tts)) / OUT_SAMPLE_RATE
rtf = cost_time / total_audio_duration

print(f"\nGenerate Complete.")
print(f"Saving to: {generated_audio_path}")
print(f"Time Cost: {cost_time:.3f} Seconds")
print(f"RTF: {rtf:.3f}")
