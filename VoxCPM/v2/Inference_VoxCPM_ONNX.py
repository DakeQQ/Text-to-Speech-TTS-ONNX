import re
import time
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import LlamaTokenizerFast


# ══════════════════════════════════════════════════════════════════════════════
# USER-CONFIGURABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
path_voxcpm2                  = r'/home/DakeQQ/Downloads/VoxCPM2'

onnx_model_VAE_Encoder        = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_AudioVAE_Encode.onnx'
onnx_model_Feat_Encoder_Cond  = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Feat_Encoder_Cond.onnx'
onnx_model_Assemble           = {
									"voice_design":   r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Assemble_VoiceDesign.onnx',
									"continuation":   r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Assemble_Continuation.onnx',
									"reference_only": r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Assemble_ReferenceOnly.onnx',
									"combined":       r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Assemble_Combined.onnx',
}
onnx_model_Prefill            = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Prefill.onnx'
onnx_model_Rotary_Mask_Decode = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Rotary_Mask_Decode.onnx'
onnx_model_Main               = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Main.onnx'
onnx_model_Feat_Decoder       = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Feat_Decoder.onnx'
onnx_model_VAE_Decoder        = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_AudioVAE_Decode.onnx'
onnx_model_Concat             = r'/home/DakeQQ/Downloads/VoxCPM_Optimized/VoxCPM2_Concat.onnx'               # Only used for streaming mode

# === Prompt / target ===
reference_audio_path = "../example/basic_ref_zh.wav"                 # Speaker identity reference. Used by reference_only / combined.
prompt_audio_path    = "../example/basic_ref_zh.wav"                 # Continuation prompt audio. Used by continuation / combined.
prompt_text          = "对，这就是我，万人敬仰的太乙真人。"                # Transcript for prompt_audio_path. Used by continuation / combined.


DEMO_CONFIGS = [
    {
        "mode": "voice_design",             # voice_design — 你对模型说「用年轻女声说话」，它凭空造一个声音。没有任何参考音频输入。/ You tell the model to "speak in a young female voice," and it creates a voice out of thin air. There is no reference audio input.
        "reference_audio_path": None,
        "prompt_audio_path": None,
        "prompt_text": None,
        "target_texts": [
            "(用年轻女声说话)大家好，我现在正在大可奇奇体验AI科技。",
            "(speak in a young female voice)Hello everyone, I'm currently experiencing DakeQQ's AI technology.",
        ],
    },
    {
        "mode": "reference_only",           # reference_only — 你给一段参考音频，模型学它的「音色」（谁的声音），但语气语速由模型自己决定。/ You provide a reference audio clip, and the model learns its "timbre" (whose voice it is), but the tone and pace are determined by the model itself.
        "reference_audio_path": reference_audio_path,
        "prompt_audio_path": None,
        "prompt_text": None,
        "target_texts": [
            "大家好，我现在正在大可奇奇体验AI科技。",
            "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
        ],
    },
    {
        "mode": "continuation",             # continuation — 你给一段「上文音频 + 上文文字」，模型会像在接着说一样延续那段话的风格和韵律。音色也由 prompt 音频决定。/ ou provide "the preceding audio + the preceding text," and the model continues the style and rhythm of that passage as if it were speaking again. The timbre is also determined by the prompt audio.
        "reference_audio_path": None,
        "prompt_audio_path": prompt_audio_path,
        "prompt_text": "对，这就是我，万人敬仰的太乙真人。",
        "target_texts": [
            "大家好，我现在正在大可奇奇体验AI科技。",
            "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
        ],
    },
    {
        "mode": "combined",                 # combined — reference 管「音色是谁」，prompt 管「怎么说」。两者可以是不同音频。 / `reference` determines "whose voice it is," and `prompt` determines "how it is spoken." The two can be different audio clips.
        "reference_audio_path": reference_audio_path,
        "prompt_audio_path": prompt_audio_path,
        "prompt_text": "对，这就是我，万人敬仰的太乙真人。",
        "target_texts": [
            "大家好，我现在正在大可奇奇体验AI科技。",
            "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
        ],
    },
]


# === Decoding limits & tokens ===
STOP_TOKEN = [1]                         # The stop_id for VoxCPM2.
MAX_SEQ_LEN = 1024                       # Max decode length; keep the same as exported model.
MIN_SEQ_LEN = 2                          # Min decode length before checking stop token.
DECODE_LIMIT_FACTOR = 6                  # Decode length limit factor, integer >= 1.

# === Audio configuration ===
IN_SAMPLE_RATE = 16000                   # Input audio sample rate (VoxCPM2 encodes at 16kHz).
OUT_SAMPLE_RATE = 48000                  # Output audio sample rate (VoxCPM2 decodes at 48kHz).

# === Guidance ===
CFG_VALUE = 2.0                          # Classifier-free guidance scale.

# === Feature flags ===
STREAMING = False                        # Enable streaming synthesis. Processes two latents at a time for faster streaming, albeit with potential discontinuities during piece-by-piece decoding.
USE_TEXT_NORMALIZER = True               # Use text normalizer.
USE_AUDIO_NORMALIZER = False             # Use audio normalizer to stabilize loudness.

# === ONNX / runtime configuration ===
ORT_LOG = False                          # Enable ONNX Runtime logging for debugging.
ORT_FP16 = False                         # Set to True for FP16 ONNX Runtime settings.
ORT_Accelerate_Providers = []            # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS = 0                          # Parallel CPU threads, 0 for auto.
DEVICE_ID = 0                            # Device id.


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
	return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
	return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
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


def mask_multichar_chinese_tokens(base_tokenizer):
	"""Mask multi-char CJK tokens so the tokenizer splits them into single chars."""
	multichar_tokens = {
		token for token in base_tokenizer.get_vocab().keys()
		if len(token.replace("\u2581", "")) >= 2
		and all("\u4e00" <= c <= "\u9fff" for c in token.replace("\u2581", ""))
	}

	class CharTokenizerWrapper:
		def __init__(self, tokenizer):
			self.tokenizer = tokenizer
			self.multichar_tokens = multichar_tokens

		def __call__(self, text, **kwargs):
			tokens = self.tokenizer.tokenize(text, **kwargs)
			processed = []
			for token in tokens:
				clean = token.replace("\u2581", "")
				if clean in self.multichar_tokens:
					processed.extend(list(clean))
				else:
					processed.append(token)
			return self.tokenizer.convert_tokens_to_ids(processed)

	return CharTokenizerWrapper(base_tokenizer)


def encode_audio_feat_ort(wav_path):
	"""Load audio, VAE encode, and return ORT value [1, seq, patch_size, latent_dim]."""
	audio = np.array(
		AudioSegment.from_file(wav_path).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(),
		dtype=np.int16
	)
	if USE_AUDIO_NORMALIZER:
		audio = audio_normalizer(audio)
	audio_ort = onnxruntime.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), device_type, DEVICE_ID)
	input_feed_VAE_Encoder[in_name_VAE_Encoder] = audio_ort
	return ort_session_VAE_Encoder.run_with_ort_values(out_name_VAE_Encoder, input_feed_VAE_Encoder, run_options=run_options)[0]


def validate_mode_inputs(mode, reference_path, prompt_path, prompt_text_value):
	"""Validate user-configured inputs for the selected mode."""
	if mode == "voice_design":
		return
	if mode == "reference_only":
		if not reference_path:
			raise ValueError("reference_only mode requires reference_audio_path.")
		return
	if mode == "continuation":
		if not prompt_path:
			raise ValueError("continuation mode requires prompt_audio_path.")
		if not prompt_text_value:
			raise ValueError("continuation mode requires prompt_text.")
		return
	if mode == "combined":
		if not reference_path:
			raise ValueError("combined mode requires reference_audio_path.")
		if not prompt_path:
			raise ValueError("combined mode requires prompt_audio_path.")
		if not prompt_text_value:
			raise ValueError("combined mode requires prompt_text.")
		return
	raise ValueError(f"Unsupported MODE: {mode}")


def prepare_mode_audio_features(mode, reference_path, prompt_path):
	"""Encode only the audio inputs required by the selected mode."""
	ref_audio_feat = empty_audio_feat_ort
	prompt_audio_feat = empty_audio_feat_ort

	if mode in ("reference_only", "combined"):
		ref_audio_feat = encode_audio_feat_ort(reference_path)
	if mode in ("continuation", "combined"):
		prompt_audio_feat = encode_audio_feat_ort(prompt_path)

	return ref_audio_feat, prompt_audio_feat


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
	opt.log_severity_level = 0 if ORT_LOG else 4
	opt.log_verbosity_level = 4

session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
	'session.set_denormal_as_zero': '1',
	'session.intra_op.allow_spinning': '1',
	'session.inter_op.allow_spinning': '1',
	'session.enable_quant_qdq_cleanup': '1',
	'session.qdq_matmulnbits_accuracy_level': '2' if ORT_FP16 else '4',
	'session.use_device_allocator_for_initializers': '1',
	'session.graph_optimizations_loop_level': '2',
	'optimization.enable_gelu_approximation': '1',
	'optimization.minimal_build_optimizations': '',
	'optimization.enable_cast_chain_elimination': '1',
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
		'device_type': 'CPU',
		'precision': 'ACCURACY',
		'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,
		'num_streams': 1,
		'enable_opencl_throttling': False,
		'enable_qdq_optimizer': False,
		'disable_dynamic_shapes': False
	}]
	device_type = 'cpu'
	_ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
	provider_options = [{
		'device_id': DEVICE_ID,
		'gpu_mem_limit': 24 * (1024 ** 3),
		'arena_extend_strategy': 'kNextPowerOfTwo',
		'cudnn_conv_algo_search': 'EXHAUSTIVE',
		'sdpa_kernel': '2',
		'use_tf32': '1',
		'fuse_conv_bias': '0',
		'cudnn_conv_use_max_workspace': '1',
		'cudnn_conv1d_pad_to_nc1d': '0',
		'tunable_op_enable': '0',
		'tunable_op_tuning_enable': '0',
		'tunable_op_max_tuning_duration_ms': 10,
		'do_copy_in_default_stream': '0',
		'enable_cuda_graph': '0',
		'prefer_nhwc': '0',
		'enable_skip_layer_norm_strict_mode': '0',
		'use_ep_level_unified_stream': '0'
	}]
	device_type = 'cuda'
	_ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
	provider_options = [{
		'device_id': DEVICE_ID,
		'performance_preference': 'high_performance',
		'device_filter': 'gpu',
		'disable_metacommands': 'false',
		'enable_graph_capture': 'false',
		'enable_graph_serialization': 'false'
	}]
	device_type = 'dml'
	_ort_device_type = C.OrtDevice.dml()

else:
	provider_options = None
	device_type = 'cpu'
	_ort_device_type = C.OrtDevice.cpu()

packed_settings = {
	"_session_opts": session_opts,
	"_providers": ORT_Accelerate_Providers if ORT_Accelerate_Providers else None,
	"_provider_options": provider_options,
	"_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
print("Loading ONNX sessions...")

ort_session_VAE_Encoder = create_session(onnx_model_VAE_Encoder, **packed_settings)
in_name_VAE_Encoder = get_in_names(ort_session_VAE_Encoder)[0]
out_name_VAE_Encoder = [get_out_names(ort_session_VAE_Encoder)[0]]

ort_session_Feat_Encoder_Cond = create_session(onnx_model_Feat_Encoder_Cond, **packed_settings)
in_name_Feat_Encoder_Cond = get_in_names(ort_session_Feat_Encoder_Cond)[0]
out_name_Feat_Encoder_Cond = get_out_names(ort_session_Feat_Encoder_Cond)

ort_sessions_Assemble = {}
for mode_key in onnx_model_Assemble:
	session = create_session(onnx_model_Assemble[mode_key], **packed_settings)
	ort_sessions_Assemble[mode_key] = {
		"session": session,
		"in_names": get_in_names(session),
		"out_names": get_out_names(session),
	}

ort_session_Prefill = create_session(onnx_model_Prefill, **packed_settings)
in_name_Prefill = get_in_names(ort_session_Prefill)
out_name_Prefill = get_out_names(ort_session_Prefill)

ort_session_Rotary_Mask_Decode = create_session(onnx_model_Rotary_Mask_Decode, **packed_settings)
in_name_Rotary_Mask_Decode = get_in_names(ort_session_Rotary_Mask_Decode)
out_name_Rotary_Mask_Decode = get_out_names(ort_session_Rotary_Mask_Decode)

ort_session_Main = create_session(onnx_model_Main, **packed_settings)
print(f"\nUsable Providers: {ort_session_Main.get_providers()}\n")

ort_session_Feat_Decoder = create_session(onnx_model_Feat_Decoder, **packed_settings)
ort_session_VAE_Decoder = create_session(onnx_model_VAE_Decoder, **packed_settings)
half_decode_len = 7680  # Fixed for VoxCPM2

# --- Concat (streaming only) ---
if STREAMING:
	ort_session_Concat = create_session(onnx_model_Concat, **packed_settings)
	in_name_Concat = get_in_names(ort_session_Concat)
	out_name_Concat = get_out_names(ort_session_Concat)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main = get_in_names(ort_session_Main)
out_name_Main = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)

num_keys_values = amount_of_outputs_Main - 3  # last 3: random, dit_hidden, stop_flag
num_layers = num_keys_values // 2

num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
num_keys_values_plus_7 = num_keys_values + 7

kv_dtype_Main = np.float16 if 'float16' in ort_session_Main._inputs_meta[0].type else np.float32
hidden_dtype_Main = np.float16 if 'float16' in ort_session_Main._inputs_meta[num_keys_values_plus_4].type else np.float32

model_dtype_VAE_Decoder = np.float16 if 'float16' in ort_session_VAE_Decoder._inputs_meta[0].type else np.float32

in_name_Feat_Decoder = get_in_names(ort_session_Feat_Decoder)
out_name_Feat_Decoder = get_out_names(ort_session_Feat_Decoder)

in_name_VAE_Decoder = get_in_names(ort_session_VAE_Decoder)
out_name_VAE_Decoder = get_out_names(ort_session_VAE_Decoder)

_meta = ort_session_Main._inputs_meta


# ══════════════════════════════════════════════════════════════════════════════
# STATIC ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
init_concat_text_len = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_audio_seg1_start = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_audio_seg1_end = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_history_len = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_decode_attention_mask = create_ort_with_shape((1, 1, 1, 1), hidden_dtype_Main, device_type, DEVICE_ID)

shape_keys = (_meta[0].shape[0], 1, _meta[0].shape[2], 0)
shape_vals = (_meta[num_layers].shape[0], 1, 0, _meta[num_layers].shape[3])
shape_embed = (1, 0, _meta[num_keys_values].shape[2])

init_past_keys_Main = create_ort_with_shape(shape_keys, kv_dtype_Main, device_type, DEVICE_ID)
init_past_values_Main = create_ort_with_shape(shape_vals, kv_dtype_Main, device_type, DEVICE_ID)

cfg_value_ort = create_ort_with_data([CFG_VALUE], hidden_dtype_Main, device_type, DEVICE_ID)
cfg_value_minus_ort = create_ort_with_data([1.0 - CFG_VALUE], hidden_dtype_Main, device_type, DEVICE_ID)
sr_cond_ort = create_ort_with_data([OUT_SAMPLE_RATE], np.int32, device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT INPUT FEED DICTIONARIES
# ══════════════════════════════════════════════════════════════════════════════
input_feed_VAE_Encoder = {}
input_feed_Feat_Encoder_Cond = {}
input_feed_Prefill = {}
input_feed_Rotary_Mask_Decode = {}
input_feed_Main = {}
input_feed_Feat_Decoder = {}
input_feed_VAE_Decoder = {}

input_feed_Feat_Decoder[in_name_Feat_Decoder[3]] = cfg_value_ort
input_feed_Feat_Decoder[in_name_Feat_Decoder[4]] = cfg_value_minus_ort
input_feed_VAE_Decoder[in_name_VAE_Decoder[1]] = sr_cond_ort


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & SPECIAL TOKENS
# ══════════════════════════════════════════════════════════════════════════════
AUDIO_START_TOKEN = 101

tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(path_voxcpm2))

_vae_enc_out_shape = ort_session_VAE_Encoder._outputs_meta[0].shape
_patch_size = _vae_enc_out_shape[2]
_latent_dim = _vae_enc_out_shape[3]
empty_audio_feat_ort = create_ort_with_shape((1, 0, _patch_size, _latent_dim), hidden_dtype_Main, device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE DEMO
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("Running inference demos for all modes...")
print("═" * 70)

for demo_config in DEMO_CONFIGS:
	demo_mode = demo_config["mode"]
	demo_reference_audio_path = demo_config["reference_audio_path"]
	demo_prompt_audio_path = demo_config["prompt_audio_path"]
	demo_prompt_text = demo_config["prompt_text"]
	demo_targets = demo_config["target_texts"]

	print(f"\n{'─' * 50}")
	print(f"Demo Mode: {demo_mode}")
	print(f"Reference audio: {demo_reference_audio_path or 'None'}")
	print(f"Prompt audio: {demo_prompt_audio_path or 'None'}")
	print(f"Prompt text: {demo_prompt_text or 'None'}")
	print(f"{'─' * 50}")

	demo_ort_session_Assemble = ort_sessions_Assemble[demo_mode]["session"]
	demo_in_name_Assemble = ort_sessions_Assemble[demo_mode]["in_names"]
	demo_out_name_Assemble = ort_sessions_Assemble[demo_mode]["out_names"]

	validate_mode_inputs(demo_mode, demo_reference_audio_path, demo_prompt_audio_path, demo_prompt_text)
	demo_ref_audio_feat_ort, demo_prompt_audio_feat_ort = prepare_mode_audio_features(
		demo_mode,
		demo_reference_audio_path,
		demo_prompt_audio_path,
	)

	demo_input_feed_Assemble = {}
	if demo_mode == "continuation":
		demo_input_feed_Assemble[demo_in_name_Assemble[1]] = demo_prompt_audio_feat_ort
	elif demo_mode == "reference_only":
		demo_input_feed_Assemble[demo_in_name_Assemble[1]] = demo_ref_audio_feat_ort
	elif demo_mode == "combined":
		demo_input_feed_Assemble[demo_in_name_Assemble[1]] = demo_ref_audio_feat_ort
		demo_input_feed_Assemble[demo_in_name_Assemble[2]] = demo_prompt_audio_feat_ort

	demo_audio_out = []
	demo_start = time.time()

	for sentence in demo_targets:
		print(f"\n  Convert to Speech: {sentence}")

		if USE_TEXT_NORMALIZER:
			from voxcpm.utils.text_normalize import TextNormalizer
			sentence = TextNormalizer().normalize(sentence)

		target_text = re.sub(r"\s+", " ", sentence.replace("\n", " ")).strip()
		target_ids = tokenizer(target_text)
		if demo_mode in ("continuation", "combined"):
			full_text = re.sub(r"\s+", " ", ((demo_prompt_text or "") + target_text).replace("\n", " ")).strip()
			text_ids = tokenizer(full_text) + [AUDIO_START_TOKEN]
		else:
			text_ids = target_ids + [AUDIO_START_TOKEN]
		text_ids_np = np.array([text_ids], dtype=np.int32)
		text_ids_ort = onnxruntime.OrtValue.ortvalue_from_numpy(text_ids_np, device_type, DEVICE_ID)

		demo_input_feed_Assemble[demo_in_name_Assemble[0]] = text_ids_ort
		asm_out = demo_ort_session_Assemble.run_with_ort_values(demo_out_name_Assemble, demo_input_feed_Assemble, run_options=run_options)
		text_token_ort = asm_out[0]
		audio_feat_ort = asm_out[1]
		audio_seg1_start_ort = asm_out[2]
		audio_seg1_end_ort = asm_out[3]
		concat_text_len_ort = asm_out[4]
		ids_len_ort = asm_out[5]

		input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = audio_feat_ort
		feat_embed_ort, feat_cond_init = ort_session_Feat_Encoder_Cond.run_with_ort_values(out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)

		input_feed_Prefill[in_name_Prefill[0]] = text_token_ort
		input_feed_Prefill[in_name_Prefill[1]] = ids_len_ort
		input_feed_Prefill[in_name_Prefill[2]] = feat_embed_ort
		input_feed_Prefill[in_name_Prefill[3]] = audio_seg1_start_ort
		input_feed_Prefill[in_name_Prefill[4]] = audio_seg1_end_ort
		input_feed_Prefill[in_name_Prefill[5]] = concat_text_len_ort
		input_feed_Prefill[in_name_Prefill[6]] = init_history_len

		prefill_out = ort_session_Prefill.run_with_ort_values(out_name_Prefill, input_feed_Prefill, run_options=run_options)
		combined_embed = prefill_out[0]
		feat_embed_audio = prefill_out[1]
		rotary_cos = prefill_out[2]
		rotary_sin = prefill_out[3]
		attention_mask = prefill_out[4]
		kv_seq_len = prefill_out[5]

		input_feed_Main[in_name_Main[num_keys_values]] = feat_embed_audio
		input_feed_Main[in_name_Main[num_keys_values_plus_1]] = audio_seg1_start_ort
		input_feed_Main[in_name_Main[num_keys_values_plus_2]] = audio_seg1_end_ort
		input_feed_Main[in_name_Main[num_keys_values_plus_3]] = concat_text_len_ort
		input_feed_Main[in_name_Main[num_keys_values_plus_4]] = combined_embed
		input_feed_Main[in_name_Main[num_keys_values_plus_5]] = rotary_cos
		input_feed_Main[in_name_Main[num_keys_values_plus_6]] = rotary_sin
		input_feed_Main[in_name_Main[num_keys_values_plus_7]] = attention_mask

		for idx in range(num_layers):
			input_feed_Main[in_name_Main[idx]] = init_past_keys_Main
		for idx in range(num_layers, num_keys_values):
			input_feed_Main[in_name_Main[idx]] = init_past_values_Main

		feat_cond = feat_cond_init
		total_seq_len = int(ids_len_ort.numpy().item())
		max_len = min(int(len(target_ids) * DECODE_LIMIT_FACTOR + 10), MAX_SEQ_LEN - total_seq_len)

		save_latent_list = []
		num_decode = 0
		start_decode = time.time()

		if STREAMING:
			pre_latent_pred = None
			input_feed_Concat = {}

		while num_decode < max_len:
			all_outputs_Main = ort_session_Main.run_with_ort_values(out_name_Main, input_feed_Main, run_options=run_options)

			input_feed_Feat_Decoder[in_name_Feat_Decoder[0]] = all_outputs_Main[num_keys_values]
			input_feed_Feat_Decoder[in_name_Feat_Decoder[1]] = all_outputs_Main[num_keys_values_plus_1]
			input_feed_Feat_Decoder[in_name_Feat_Decoder[2]] = feat_cond
			latent_pred = ort_session_Feat_Decoder.run_with_ort_values(out_name_Feat_Decoder, input_feed_Feat_Decoder, run_options=run_options)[0]

			if STREAMING:
				if pre_latent_pred is None:
					pre_latent_pred = latent_pred
				else:
					input_feed_Concat[in_name_Concat[0]] = pre_latent_pred
					input_feed_Concat[in_name_Concat[1]] = latent_pred
					save_latent_ort = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)[0]
					input_feed_VAE_Decoder[in_name_VAE_Decoder[0]] = save_latent_ort
					audio_out_ort = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)[0]
					pre_latent_pred = latent_pred
					audio_out_np = audio_out_ort.numpy()
					if num_decode > 1:
						audio_out_np = audio_out_np[..., half_decode_len:]
					demo_audio_out.append(audio_out_np)
			else:
				save_latent_list.append(latent_pred.numpy())

			if num_decode >= MIN_SEQ_LEN:
				if int(all_outputs_Main[num_keys_values_plus_2].numpy().item()) in STOP_TOKEN:
					break

			input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = latent_pred
			feat_embed_decode, feat_cond = ort_session_Feat_Encoder_Cond.run_with_ort_values(out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)

			for idx in range(num_keys_values):
				input_feed_Main[in_name_Main[idx]] = all_outputs_Main[idx]
			input_feed_Main[in_name_Main[num_keys_values]] = feat_embed_decode
			input_feed_Main[in_name_Main[num_keys_values_plus_4]] = feat_embed_decode

			input_feed_Rotary_Mask_Decode[in_name_Rotary_Mask_Decode[0]] = kv_seq_len
			rotary_cos, rotary_sin, kv_seq_len = ort_session_Rotary_Mask_Decode.run_with_ort_values(out_name_Rotary_Mask_Decode, input_feed_Rotary_Mask_Decode, run_options=run_options)
			input_feed_Main[in_name_Main[num_keys_values_plus_5]] = rotary_cos
			input_feed_Main[in_name_Main[num_keys_values_plus_6]] = rotary_sin

			if num_decode < 1:
				input_feed_Main[in_name_Main[num_keys_values_plus_1]] = init_audio_seg1_start
				input_feed_Main[in_name_Main[num_keys_values_plus_2]] = init_audio_seg1_end
				input_feed_Main[in_name_Main[num_keys_values_plus_3]] = init_concat_text_len
				input_feed_Main[in_name_Main[num_keys_values_plus_7]] = init_decode_attention_mask

			num_decode += 1

		print(f"    Decoded {num_decode} tokens ({((num_decode + 1) / max(time.time() - start_decode, 1e-6)):.1f} tok/s)")

		if not STREAMING:
			if save_latent_list:
				stacked = np.concatenate(save_latent_list, axis=1)
				input_feed_VAE_Decoder[in_name_VAE_Decoder[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(stacked.astype(model_dtype_VAE_Decoder, copy=False), device_type, DEVICE_ID)
				audio_out = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)[0]
				demo_audio_out.append(audio_out.numpy())

	if demo_audio_out:
		demo_audio_all = np.concatenate([output.reshape(-1) for output in demo_audio_out], axis=-1)
		demo_out_path = f"./generated_demo_{demo_mode}.wav"
		Path(demo_out_path).parent.mkdir(parents=True, exist_ok=True)
		sf.write(demo_out_path, demo_audio_all.reshape(-1), OUT_SAMPLE_RATE, subtype='PCM_16')
		demo_duration = demo_audio_all.shape[-1] / OUT_SAMPLE_RATE
		demo_cost = time.time() - demo_start
		print(f"  Saved: {demo_out_path} ({demo_duration:.2f}s audio, RTF={demo_cost / max(demo_duration, 1e-6):.3f})")

print("\n" + "═" * 70)
print("All mode demos complete.")
print("═" * 70)
