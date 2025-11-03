import time
import soundfile as sf
import numpy as np
import onnxruntime
from transformers import AutoTokenizer


tokenizer_path = r'/home/DakeQQ/Downloads/kani-tts-370m'                                # Set the folder path where the KaniTTS tokenizer.
onnx_model_A = r'/home/DakeQQ/Downloads/KaniTTS_Optimized/KaniTTS_Embed.onnx'           # Assign a path where the exported KaniTTS model stored.
onnx_model_B = r'/home/DakeQQ/Downloads/KaniTTS_Optimized/KaniTTS_Main.onnx'            # Assign a path where the exported KaniTTS model stored.
onnx_model_C = r'/home/DakeQQ/Downloads/KaniTTS_Optimized/Greedy_Search.onnx'           # Assign a path where the exported KaniTTS model stored.
onnx_model_D = r'/home/DakeQQ/Downloads/KaniTTS_Optimized/First_Beam_Search.onnx'       # Assign a path where the exported KaniTTS model stored.
onnx_model_E = r'/home/DakeQQ/Downloads/KaniTTS_Optimized/Second_Beam_Search.onnx'      # Assign a path where the exported KaniTTS model stored.
onnx_model_F = r'/home/DakeQQ/Downloads/KaniTTS_Optimized/Reset_Penality.onnx'          # Assign a path where the exported KaniTTS model stored.
onnx_model_G = r'/home/DakeQQ/Downloads/KaniTTS_Optimized/KaniTTS_Codec.onnx'           # Assign a path where the exported KaniTTS model stored.
generated_audio_path = r"./generated.wav"                                               # The generated audio path.

target_tts = ["大家好，我现在正在大可奇奇体验AI科技。"]                                      # The test query after the export process.
speaker = 'ming'                                                                       # The Speaker type.

"""
kani-tts-370m multilingual:
    Speaker List:
        david — David, English (British)
        puck — Puck, English (Gemini)
        kore — Kore, English (Gemini)
        andrew — Andrew, English
        jenny — Jenny, English (Irish)
        simon — Simon, English
        katie — Katie, English
        seulgi — Seulgi, Korean
        bert — Bert, German
        thorsten — Thorsten, German (Hessisch)
        maria — Maria, Spanish
        mei — Mei, Chinese (Cantonese)
        ming — Ming, Chinese (Shanghai OpenAI)
        karim — Karim, Arabic
        nur — Nur, Arabic
"""

STOP_TOKEN = [64402]                   # The stop_id in KaniTTS is "64402"
MAX_SEQ_LEN = 1024                     # The max decode length.
REPEAT_PENALITY = 0.9                  # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                    # Penalizes the most recent output. "15" means the last 15 tokens.
USE_BEAM_SEARCH = False                # Use beam search or greedy search.
TOP_K = 3                              # The top k candidate in decoding.
BEAM_SIZE = 3                          # Number of beams in searching.
MAX_BEAM_SIZE = 10                     # Max beams for exported model.
SAMPLE_RATE = 22050                    # The sample rate of output audio. Keep the same with exported model.
MAX_THREADS = 0                        # The CPU parallel threads. Set 0 for auto.
DEVICE_ID = 0                          # The GPU id.

ORT_Accelerate_Providers = []          # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                       # else keep empty.

# ONNX Runtime settings
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,  # The default value is 8. Edit freely.
            'num_streams': 1,
            'enable_opencl_throttling': False,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
    device_type = 'cpu'
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '1',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    provider_options = None


# Run the exported model by ONNX Runtime
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4                  # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = in_name_A[0].name
out_name_A = [out_name_A[0].name]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_B.get_providers()}")
model_dtype = ort_session_B._inputs_meta[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
amount_of_outputs_B = len(out_name_B)
in_name_B = [in_name_B[i].name for i in range(len(in_name_B))]
out_name_B = [out_name_B[i].name for i in range(amount_of_outputs_B)]


generate_limit = MAX_SEQ_LEN - 5  # 5 = length of initial input_ids
num_keys_values = 0
for i in ort_session_B._outputs_meta:
    if len(i.shape) == 5:
        num_keys_values += 1
num_keys_values_convs = amount_of_outputs_B - 2
num_conv_layers = num_keys_values_convs - num_keys_values
num_layers = num_keys_values // 2
num_keys_values_convs_plus_1 = num_keys_values_convs + 1
num_keys_values_convs_plus_2 = num_keys_values_convs + 2
num_keys_values_convs_plus_3 = num_keys_values_convs + 3
num_keys_values_convs_plus_4 = num_keys_values_convs + 4
num_keys_values_convs_plus_5 = num_keys_values_convs + 5
num_keys_values_convs_plus_6 = num_keys_values_convs + 6
num_keys_values_convs_plus_7 = num_keys_values_convs + 7
vocab_size = ort_session_B._outputs_meta[num_keys_values_convs].shape[1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(REPEAT_PENALITY, dtype=model_dtype), device_type, DEVICE_ID)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

head_ids = np.array([[64403]], dtype=np.int32)       # For non-Python tokenizer = [64403, 1]
tail_ids = np.array([[2, 64404]], dtype=np.int32)


# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE


if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")


if USE_BEAM_SEARCH:
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]
    
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]
    
    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(len(out_name_F))]

    input_feed_D = {
        in_name_D[num_keys_values_convs_plus_3]: penality_value,
        in_name_D[num_keys_values_convs_plus_4]: beam_size
    }

    input_feed_E = {
        in_name_E[num_keys_values_convs_plus_5]: penality_value,
        in_name_E[num_keys_values_convs_plus_6]: beam_size,
        in_name_E[num_keys_values_convs_plus_7]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
    out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]
    input_feed_C = {in_name_C[2]: penality_value}


ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=["CPUExecutionProvider"], provider_options=None)  # It is recommended to use CPU and Float32 format instead of GPU.
in_name_G = ort_session_G.get_inputs()
out_name_G = ort_session_G.get_outputs()
in_name_G = [in_name_G[i].name for i in range(len(in_name_G))]
out_name_G = [out_name_G[i].name for i in range(len(out_name_G))]


if USE_BEAM_SEARCH:
    penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)
else:
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False


init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
if device_type == 'dml':
    init_past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[0].shape[1], 1, ort_session_B._inputs_meta[0].shape[3], 0), dtype=model_dtype), device_type, DEVICE_ID)
    init_past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_layers].shape[1], 1, 0, ort_session_B._inputs_meta[num_layers].shape[4]), dtype=model_dtype), device_type, DEVICE_ID)
    init_conv_states_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_keys_values].shape[1], 0), dtype=model_dtype), device_type, DEVICE_ID)
else:
    init_past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[0].shape[1], 1, ort_session_B._inputs_meta[0].shape[3], 0), dtype=model_dtype), 'cpu', 0)
    init_past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_layers].shape[1], 1, 0, ort_session_B._inputs_meta[num_layers].shape[4]), dtype=model_dtype), 'cpu', 0)
    init_conv_states_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_keys_values].shape[1], 0), dtype=model_dtype), 'cpu', 0)
init_save_id = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=model_dtype), device_type, DEVICE_ID)
init_batch_size_greedy = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)


# Start to run
for sentence in target_tts:
    sentence = f"{speaker}: {sentence}"
    print(f"\n{sentence}")
    input_ids = tokenizer(sentence, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = np.concatenate([head_ids, input_ids, tail_ids], axis=1)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[1]], dtype=np.int64), device_type, DEVICE_ID)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
    ids_len_1 = init_ids_len_1
    history_len = init_history_len
    past_keys_B = init_past_keys_B
    past_values_B = init_past_values_B
    conv_states_B = init_conv_states_B
    save_id = init_save_id
    repeat_penality = init_repeat_penality

    start_time = time.time()

    input_feed_A = {in_name_A: input_ids}
    hidden_states = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]

    input_feed_B = {
            in_name_B[num_keys_values_convs]: hidden_states,
            in_name_B[num_keys_values_convs_plus_1]: history_len,
            in_name_B[num_keys_values_convs_plus_2]: ids_len
        }

    for i in range(num_layers):
        input_feed_B[in_name_B[i]] = past_keys_B
    for i in range(num_layers, num_keys_values):
        input_feed_B[in_name_B[i]] = past_values_B
    for i in range(num_keys_values, num_keys_values_convs):
        input_feed_B[in_name_B[i]] = conv_states_B

    if USE_BEAM_SEARCH:
        save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
        input_feed_D[in_name_D[num_keys_values_convs_plus_1]] = save_id_beam
        input_feed_D[in_name_D[num_keys_values_convs_plus_2]] = repeat_penality
    else:
        input_feed_C[in_name_C[1]] = repeat_penality
        input_feed_C[in_name_C[3]] = init_batch_size_greedy

    if do_repeat_penality:
        if USE_BEAM_SEARCH:
            input_feed_F = {in_name_F[2]: penality_reset_count_beam_init}
        else:
            penality_reset_count_greedy = 0

    num_decode = 0
    start_decode = time.time()
    while num_decode < generate_limit:
        all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
        if USE_BEAM_SEARCH:
            if num_decode < 1:
                input_feed_D.update(zip(in_name_D[:num_keys_values_convs_plus_1], all_outputs_B))
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                max_logits_idx = all_outputs_D[num_keys_values_convs_plus_5].numpy()
                input_feed_E[in_name_E[num_keys_values_convs_plus_4]] = all_outputs_D[num_keys_values_convs_plus_4]
                if do_repeat_penality:
                    input_feed_F[in_name_F[3]] = all_outputs_D[num_keys_values_convs_plus_4]
            else:
                input_feed_E.update(zip(in_name_E[:num_keys_values_convs_plus_1], all_outputs_B))
                all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                max_logits_idx = all_outputs_E[num_keys_values_convs_plus_4].numpy()
            if max_logits_idx in STOP_TOKEN:
                break
            if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                input_feed_F[in_name_F[0]] = all_outputs_E[num_keys_values_convs_plus_1]
                input_feed_F[in_name_F[1]] = all_outputs_E[num_keys_values_convs_plus_2]
                all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
                input_feed_F[in_name_F[2]] = all_outputs_F[2]
                input_feed_E[in_name_E[num_keys_values_convs_plus_1]] = all_outputs_F[0]
                input_feed_E[in_name_E[num_keys_values_convs_plus_2]] = all_outputs_F[1]
            if num_decode < 1:
                input_feed_B.update(zip(in_name_B[:num_keys_values_convs], all_outputs_D))
                input_feed_A[in_name_A] = all_outputs_D[num_keys_values_convs]
                input_feed_E[in_name_E[num_keys_values_convs_plus_1]] = all_outputs_D[num_keys_values_convs_plus_1]
                input_feed_E[in_name_E[num_keys_values_convs_plus_2]] = all_outputs_D[num_keys_values_convs_plus_2]
                input_feed_E[in_name_E[num_keys_values_convs_plus_3]] = all_outputs_D[num_keys_values_convs_plus_3]
            else:
                input_feed_B.update(zip(in_name_B[:num_keys_values_convs], all_outputs_E))
                input_feed_A[in_name_A] = all_outputs_E[num_keys_values_convs]
                input_feed_E[in_name_E[num_keys_values_convs_plus_1]] = all_outputs_E[num_keys_values_convs_plus_1]
                input_feed_E[in_name_E[num_keys_values_convs_plus_2]] = all_outputs_E[num_keys_values_convs_plus_2]
                input_feed_E[in_name_E[num_keys_values_convs_plus_3]] = all_outputs_E[num_keys_values_convs_plus_3]
            input_feed_B[in_name_B[num_keys_values_convs]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
        else:
            input_feed_C[in_name_C[0]] = all_outputs_B[num_keys_values_convs]
            all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
            max_logits_idx = all_outputs_C[0].numpy()[0, 0]
            if max_logits_idx in STOP_TOKEN:
                break
            if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                reset_ids = save_id_greedy[penality_reset_count_greedy]
                if reset_ids != max_logits_idx:
                    repeat_penality = all_outputs_C[1].numpy()
                    repeat_penality[:, reset_ids] = 1.0
                    input_feed_C[in_name_C[1]].update_inplace(repeat_penality)
                penality_reset_count_greedy += 1
            else:
                input_feed_C[in_name_C[1]] = all_outputs_C[1]
            input_feed_C[in_name_C[0]] = all_outputs_C[0]
            input_feed_B.update(zip(in_name_B[:num_keys_values_convs_plus_1], all_outputs_B))
            input_feed_A[in_name_A] = all_outputs_C[0]
            input_feed_B[in_name_B[num_keys_values_convs]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
            save_id_greedy[num_decode] = max_logits_idx
        input_feed_B[in_name_B[num_keys_values_convs_plus_1]] = all_outputs_B[num_keys_values_convs_plus_1]
        if num_decode < 1:
            input_feed_B[in_name_B[num_keys_values_convs_plus_2]] = ids_len_1
        num_decode += 1
    if num_decode > 0:
        print(f"\n\nDecode: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s")
        if USE_BEAM_SEARCH:
            input_feed_G = {in_name_G[0]: all_outputs_E[num_keys_values_convs_plus_1]}
        else:
            input_feed_G = {in_name_G[0]: onnxruntime.OrtValue.ortvalue_from_numpy(save_id_greedy.reshape(1, -1), 'cpu', 0)}
        input_feed_G[in_name_G[1]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([num_decode], dtype=np.int64), 'cpu', 0)
        audio_out = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)[0]
        print(f"\nGenerate Complete.\n\nSaving to: {generated_audio_path}.\n\nTime Cost: {time.time() - start_time:.3f} Seconds")
        audio_out = audio_out.numpy().reshape(-1)
        sf.write(generated_audio_path, audio_out, SAMPLE_RATE, format='WAVEX')
    else:
        print("\n Generate Failed")
        
