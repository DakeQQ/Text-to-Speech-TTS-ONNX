import time
import soundfile as sf
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from modeling_modified.text_normalize import TextNormalizer
from transformers import LlamaTokenizerFast


path_voxcpm                         = r'/home/DakeQQ/Downloads/VoxCPM1.5'                                           # Set the folder path where the VoxCPM1.5 project downloaded.
onnx_model_VAE_Encoder              = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_VAE_Encoder.onnx'            # Assign a path where the optimized VoxCPM model stored.
onnx_model_Feat_Encoder_Cond        = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Feat_Encoder_Cond.onnx'
onnx_model_Prefill                  = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Prefill.onnx'
onnx_model_Rotary_Mask_Text_Decode  = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Rotary_Mask_Text_Decode.onnx'
onnx_model_Main                     = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Main.onnx'
onnx_model_Feat_Decoder             = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Feat_Decoder.onnx'
onnx_model_VAE_Decoder              = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_VAE_Decoder.onnx'
onnx_model_Concat                   = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM_Concat.onnx'                 # Only used for streaming mode

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
CFG_VALUE = 2.5                          # Lower values result in more natural speech for long text, while higher values stay closer to the original sound features.
RANDOM_SEED = 1                          # Global random seed

# === Feature flags ===
STREAMING = True                        # Enable streaming synthesis. Unlike the official implementation, this version processes two latents at a time for faster performance, albeit with potential discontinuities during piece-by-piece decoding.
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
        'device_type':              'CPU',
        'precision':                'ACCURACY',
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
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
    device_type      = 'cuda'
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
# --- VAE Encoder ---
ort_session_VAE_Encoder = create_session(onnx_model_VAE_Encoder, **packed_settings)
in_name_VAE_Encoder     = get_in_names(ort_session_VAE_Encoder)[0]
out_name_VAE_Encoder    = [get_out_names(ort_session_VAE_Encoder)[0]]

# --- Fused Feat Encoder + Cond ---
ort_session_Feat_Encoder_Cond = create_session(onnx_model_Feat_Encoder_Cond, **packed_settings)
in_name_Feat_Encoder_Cond     = get_in_names(ort_session_Feat_Encoder_Cond)[0]
out_name_Feat_Encoder_Cond    = get_out_names(ort_session_Feat_Encoder_Cond)

# --- Fused Prefill ---
ort_session_Prefill = create_session(onnx_model_Prefill, **packed_settings)
in_name_Prefill     = get_in_names(ort_session_Prefill)
out_name_Prefill    = get_out_names(ort_session_Prefill)

# --- Rotary Mask Decode ---
ort_session_Rotary_Mask_Text_Decode = create_session(onnx_model_Rotary_Mask_Text_Decode, **packed_settings)
in_name_Rotary_Mask_Text_Decode     = get_in_names(ort_session_Rotary_Mask_Text_Decode)
out_name_Rotary_Mask_Text_Decode    = get_out_names(ort_session_Rotary_Mask_Text_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
print(f"\nUsable Providers: {ort_session_Main.get_providers()}\n")

# --- Feat Decoder (full diffusion loop in one call) ---
ort_session_Feat_Decoder = create_session(onnx_model_Feat_Decoder, **packed_settings)
model_dtype_Feat_Decoder = np.float16 if 'float16' in ort_session_Feat_Decoder._inputs_meta[1].type else np.float32
in_name_Feat_Decoder     = get_in_names(ort_session_Feat_Decoder)
out_name_Feat_Decoder    = get_out_names(ort_session_Feat_Decoder)

# --- VAE Decoder ---
ort_session_VAE_Decoder    = create_session(onnx_model_VAE_Decoder, **packed_settings)
model_dtype_VAE_Decoder    = np.float16 if 'float16' in ort_session_VAE_Decoder._inputs_meta[0].type else np.float32
DYNAMIC_SHAPE_VAE_DECODE   = isinstance(ort_session_VAE_Decoder._inputs_meta[0].shape[1], str)
in_name_VAE_Decoder        = get_in_names(ort_session_VAE_Decoder)[0]
out_name_VAE_Decoder       = get_out_names(ort_session_VAE_Decoder)
half_decode_len            = 7056  # Fixed for VoxCPM1.5

# --- Concat (streaming only) ---
if STREAMING:
    ort_session_Concat = create_session(onnx_model_Concat, **packed_settings)
    in_name_Concat     = get_in_names(ort_session_Concat)
    out_name_Concat    = get_out_names(ort_session_Concat)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
model_dtype_Main       = np.float16 if 'float16' in ort_session_Main._inputs_meta[0].type else np.float32
in_name_Main           = get_in_names(ort_session_Main)
out_name_Main          = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)

num_keys_values = amount_of_outputs_Main - 3
num_layers      = num_keys_values // 2

num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5

_meta = ort_session_Main._inputs_meta


# ══════════════════════════════════════════════════════════════════════════════
# STATIC ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
generate_limit = MAX_SEQ_LEN - 1

# --- Scalars & Lengths ---
init_concat_text_len   = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)

# --- Masks ---
init_decode_attention_mask = create_ort_with_shape((1, 1, 1, 1), model_dtype_Main, device_type, DEVICE_ID)

# --- KV Cache & Embedding Shapes ---
shape_keys   = (_meta[0].shape[0],          1, _meta[0].shape[2],          0)
shape_vals   = (_meta[num_layers].shape[0],  1, 0, _meta[num_layers].shape[3])
shape_embed  = (1, 0, _meta[num_keys_values].shape[2])

init_past_keys_Main   = create_ort_with_shape(shape_keys, model_dtype_Main, device_type, DEVICE_ID)
init_past_values_Main = create_ort_with_shape(shape_vals, model_dtype_Main, device_type, DEVICE_ID)
init_feat_embed       = create_ort_with_shape(shape_embed, model_dtype_Main, device_type, DEVICE_ID)

# --- CFG Values ---
cfg_value       = create_ort_with_data([CFG_VALUE],       model_dtype_Feat_Decoder, device_type, DEVICE_ID)
cfg_value_minus = create_ort_with_data([1.0 - CFG_VALUE], model_dtype_Feat_Decoder, device_type, DEVICE_ID)

# --- Audio Post-processing ---
blank_segment = np.zeros((1, 1, int(OUT_SAMPLE_RATE * 0.1)), dtype=np.int16)

# --- Empty prompt IDs (for no-prompt case) ---
empty_prompt_ids = create_ort_with_data([[]], np.int32, device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION SETUP & IO FEEDS
# ══════════════════════════════════════════════════════════════════════════════
input_feed_VAE_Encoder              = {}
input_feed_Feat_Encoder_Cond        = {}
input_feed_Prefill                  = {}
input_feed_Rotary_Mask_Text_Decode  = {}
input_feed_Main                     = {}
input_feed_Feat_Decoder             = {}
input_feed_VAE_Decoder              = {}

# Feat Decoder: Fixed CFG Inputs
input_feed_Feat_Decoder[in_name_Feat_Decoder[3]] = cfg_value
input_feed_Feat_Decoder[in_name_Feat_Decoder[4]] = cfg_value_minus

# Compute init_feat_cond (zero-input conditioning for no-prompt case)
_meta_fec = ort_session_Feat_Encoder_Cond._inputs_meta[0]
_zero_feat_shape = (1, _meta_fec.shape[1], _meta_fec.shape[2])
_zero_feat_dtype = np.float16 if 'float16' in _meta_fec.type else np.float32
input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = create_ort_with_shape(_zero_feat_shape, _zero_feat_dtype, device_type, DEVICE_ID)
_init_results = ort_session_Feat_Encoder_Cond.run_with_ort_values(out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)
init_feat_cond_0 = _init_results[1]


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & PROMPT HANDLING
# ══════════════════════════════════════════════════════════════════════════════
tokenizer       = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(path_voxcpm))
text_normalizer = TextNormalizer()

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE AUDIO ENCODING (cached - computed once, reused for all sentences)
# ══════════════════════════════════════════════════════════════════════════════
if prompt_audio_path and prompt_text:
    use_prompt_audio = True

    # Load and encode audio
    audio = np.array(
        AudioSegment.from_file(prompt_audio_path).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(),
        dtype=np.int16
    )
    if USE_AUDIO_NORMALIZER:
        audio = audio_normalizer(audio)
    audio_ort = onnxruntime.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), device_type, DEVICE_ID)

    # VAE Encode (once)
    input_feed_VAE_Encoder[in_name_VAE_Encoder] = audio_ort
    audio_feat = ort_session_VAE_Encoder.run_with_ort_values(out_name_VAE_Encoder, input_feed_VAE_Encoder, run_options=run_options)[0]

    # Feat Encoder + Cond (once) → cached feat_embed_full & feat_cond_init
    input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = audio_feat
    feat_embed_full, feat_cond_init = ort_session_Feat_Encoder_Cond.run_with_ort_values(
        out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)

    # Tokenize prompt text (once)
    if USE_TEXT_NORMALIZER:
        prompt_text = text_normalizer.normalize(prompt_text)
    prompt_ids_np      = np.array([tokenizer(prompt_text)], dtype=np.int32)
    prompt_text_len    = prompt_ids_np.shape[-1]
    prompt_ids_ort     = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_ids_np, device_type, DEVICE_ID)

    del audio, audio_ort
else:
    use_prompt_audio = False
    feat_embed_full  = init_feat_embed
    feat_cond_init   = init_feat_cond_0
    prompt_text_len  = 0
    prompt_ids_ort   = empty_prompt_ids

    if not prompt_audio_path:
        print("Info: No prompt audio provided, using random seed to generate voice.\n")
    else:
        print("Warning: No prompt text provided, so the prompt audio will be ignored.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
save_audio_out = []
count_time = time.time()

for sentence in target_tts:
    print(f"Convert to Speech: {sentence}")
    if USE_TEXT_NORMALIZER:
        sentence = text_normalizer.normalize(sentence)

    # --- Tokenize target text ---
    target_ids_np = np.array([tokenizer(sentence)], dtype=np.int32)
    target_ids_ort = onnxruntime.OrtValue.ortvalue_from_numpy(target_ids_np, device_type, DEVICE_ID)

    # ──────────────────────────────────────────────────────────────────────────
    # PREFILL: Single fused call replaces Text_Embed + 3x Concat + Rotary_Mask
    # Input: prompt_text_ids, target_text_ids, feat_embed
    # Output: hidden_states, concat_text_len, rotary_cos, rotary_sin, attention_mask, ids_len
    # ──────────────────────────────────────────────────────────────────────────
    input_feed_Prefill[in_name_Prefill[0]] = prompt_ids_ort
    input_feed_Prefill[in_name_Prefill[1]] = target_ids_ort
    input_feed_Prefill[in_name_Prefill[2]] = feat_embed_full

    prefill_out      = ort_session_Prefill.run_with_ort_values(out_name_Prefill, input_feed_Prefill, run_options=run_options)
    hidden_states    = prefill_out[0]
    concat_text_len  = prefill_out[1]
    rotary_cos       = prefill_out[2]
    rotary_sin       = prefill_out[3]
    attention_mask   = prefill_out[4]
    ids_len_ort      = prefill_out[5]

    # Get scalar values for max_len calculation (one-time read, outside hot loop)
    concat_text_len_val = int(concat_text_len.numpy().item())
    ids_len_val         = int(ids_len_ort.numpy().item())
    max_len = min((concat_text_len_val - prompt_text_len) * DECODE_LIMIT_FACTOR + 10, generate_limit - ids_len_val)

    # --- Prepare Main Decoder Inputs (prefill step) ---
    input_feed_Main[in_name_Main[num_keys_values]]        = feat_embed_full
    input_feed_Main[in_name_Main[num_keys_values_plus_1]] = concat_text_len
    input_feed_Main[in_name_Main[num_keys_values_plus_2]] = hidden_states
    input_feed_Main[in_name_Main[num_keys_values_plus_3]] = rotary_cos
    input_feed_Main[in_name_Main[num_keys_values_plus_4]] = rotary_sin
    input_feed_Main[in_name_Main[num_keys_values_plus_5]] = attention_mask

    # Reset KV Cache
    for i in range(num_layers):
        input_feed_Main[in_name_Main[i]] = init_past_keys_Main
    for i in range(num_layers, num_keys_values):
        input_feed_Main[in_name_Main[i]] = init_past_values_Main

    feat_cond = feat_cond_init
    kv_seq_len = ids_len_ort

    # Latent accumulation (no Concat in loop for non-streaming)
    save_latent_list = []

    if STREAMING:
        pre_latent_pred = None
        input_feed_Concat = {}

    # ──────────────────────────────────────────────────────────────────────────
    # AUTO-REGRESSIVE DECODING
    # Hot loop: only 4 session.run() calls per step (was 14 in unfused version)
    #   1. Main (transformer)
    #   2. Feat_Decoder (full diffusion loop in one call)
    #   3. Feat_Encoder_Cond (fused feat encoding + conditioning)
    #   4. Rotary_Mask_Decode (next position)
    # ──────────────────────────────────────────────────────────────────────────
    num_decode   = 0
    start_decode = time.time()

    while num_decode < max_len:
        # --- 1. Main Transformer ---
        all_outputs_Main = ort_session_Main.run_with_ort_values(out_name_Main, input_feed_Main, run_options=run_options)

        # --- 2. Feat Decoder (ALL timesteps in one call) ---
        input_feed_Feat_Decoder[in_name_Feat_Decoder[0]] = all_outputs_Main[num_keys_values]         # random
        input_feed_Feat_Decoder[in_name_Feat_Decoder[1]] = all_outputs_Main[num_keys_values_plus_1]  # dit_hidden
        input_feed_Feat_Decoder[in_name_Feat_Decoder[2]] = feat_cond

        latent_pred = ort_session_Feat_Decoder.run_with_ort_values(out_name_Feat_Decoder, input_feed_Feat_Decoder, run_options=run_options)[0]

        # --- Accumulate latent ---
        if STREAMING:
            if pre_latent_pred is None:
                pre_latent_pred = latent_pred
            else:
                input_feed_Concat[in_name_Concat[0]] = pre_latent_pred
                input_feed_Concat[in_name_Concat[1]] = latent_pred
                save_latent_ort = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)[0]
                input_feed_VAE_Decoder[in_name_VAE_Decoder] = save_latent_ort
                audio_out_ort, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
                pre_latent_pred = latent_pred
                audio_out_np = audio_out_ort.numpy()
                if num_decode > 1:
                    audio_out_np = audio_out_np[..., half_decode_len:]
                save_audio_out.append(audio_out_np)
        else:
            save_latent_list.append(latent_pred)

        # --- Check Stop Token ---
        if num_decode >= MIN_SEQ_LEN and all_outputs_Main[num_keys_values_plus_2].numpy() in STOP_TOKEN:
            break

        # --- 3. Fused Feat_Encoder_Cond (one call instead of two) ---
        input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = latent_pred
        feat_embed_new, feat_cond = ort_session_Feat_Encoder_Cond.run_with_ort_values(out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)

        # --- Update Main inputs for next decode step ---
        input_feed_Main.update(zip(in_name_Main[:num_keys_values], all_outputs_Main))
        input_feed_Main[in_name_Main[num_keys_values]]        = feat_embed_new
        input_feed_Main[in_name_Main[num_keys_values_plus_2]] = feat_embed_new

        # --- 4. Rotary for next position ---
        input_feed_Rotary_Mask_Text_Decode[in_name_Rotary_Mask_Text_Decode[0]] = kv_seq_len
        rotary_cos, rotary_sin, kv_seq_len = ort_session_Rotary_Mask_Text_Decode.run_with_ort_values(out_name_Rotary_Mask_Text_Decode, input_feed_Rotary_Mask_Text_Decode, run_options=run_options)
        input_feed_Main[in_name_Main[num_keys_values_plus_3]] = rotary_cos
        input_feed_Main[in_name_Main[num_keys_values_plus_4]] = rotary_sin

        # First decode step: switch to decode-mode inputs
        if num_decode < 1:
            input_feed_Main[in_name_Main[num_keys_values_plus_1]] = init_concat_text_len
            input_feed_Main[in_name_Main[num_keys_values_plus_5]] = init_decode_attention_mask

        num_decode += 1
        print(f"    Decode: {num_decode}")

    print(f"\nDecode Speed: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s\n")

    # ──────────────────────────────────────────────────────────────────────────
    # FINALIZE SENTENCE AUDIO (NON-STREAMING)
    # One-time numpy conversion after decode loop completes
    # ──────────────────────────────────────────────────────────────────────────
    if not STREAMING:
        if DYNAMIC_SHAPE_VAE_DECODE:
            # Concatenate all latents at once (single numpy call, outside hot loop)
            all_latents = np.concatenate([lp.numpy() for lp in save_latent_list], axis=1)
            vae_input = onnxruntime.OrtValue.ortvalue_from_numpy(all_latents.astype(model_dtype_VAE_Decoder), device_type, DEVICE_ID)
            input_feed_VAE_Decoder[in_name_VAE_Decoder] = vae_input
            audio_out_ort, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
            save_audio_out.append(audio_out_ort.numpy())
        else:
            # Paired decode without Concat model (numpy on small tensors, outside loop)
            for i in range(len(save_latent_list) - 1):
                paired = np.concatenate([save_latent_list[i].numpy(), save_latent_list[i + 1].numpy()], axis=1)
                vae_input = onnxruntime.OrtValue.ortvalue_from_numpy(paired.astype(model_dtype_VAE_Decoder), device_type, DEVICE_ID)
                input_feed_VAE_Decoder[in_name_VAE_Decoder] = vae_input
                audio_out_ort, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
                audio_out_np = audio_out_ort.numpy()
                if i > 0:
                    audio_out_np = audio_out_np[..., half_decode_len:]
                save_audio_out.append(audio_out_np)

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
