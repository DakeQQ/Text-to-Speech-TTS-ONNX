import gc
import sys
import time
import torch
import shutil
import onnxruntime
import numpy as np

model_path          = r"/home/DakeQQ/Downloads/bigvgan_v2_24khz_100band_256x"      # The BigVGAN project path.    URL: https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x / https://github.com/NVIDIA/BigVGAN
onnx_model_A        = r"/home/DakeQQ/Downloads/BigVGAN_ONNX/BigVGAN.onnx"          # The exported onnx model path.


# ONNX Runtime Settings
ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
MAX_THREADS = 8                         # Max CPU parallel threads.
DEVICE_ID = 0                           # The GPU id, default to 0.

# Model Parameters
DYNAMIC_AXIS = True                     # The default dynamic axis is mel feature length.
USE_TANH = True                         # Set for using tanh(x) at the final output or not.
N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
MAX_SIGNAL_LENGTH = 512                 # Max frames for audio length after STFT processed. For static axis setting.

shutil.copy("./modeling_modified/bigvgan.py", model_path + "/bigvgan.py")
shutil.copy("./modeling_modified/resample.py", model_path + "/alias_free_activation/torch/resample.py")
shutil.copy("./modeling_modified/filter.py", model_path + "/alias_free_activation/torch/filter.py")
shutil.copy("./modeling_modified/act.py", model_path + "/alias_free_activation/torch/act.py")


if model_path not in sys.path:
    sys.path.insert(0, model_path)


from bigvgan import BigVGAN


class BIGVGAN(torch.nn.Module):
    def __init__(self, bigvgan, use_tanh):
        super(BIGVGAN, self).__init__()
        self.bigvgan = bigvgan
        self.bigvgan.use_tanh_at_final = use_tanh
        self.use_tanh = use_tanh

    def forward(self, mel_features):
        generated_wav = self.bigvgan(mel_features)
        generated_wav = (generated_wav * 32767.0)
        if self.use_tanh:
            generated_wav = generated_wav.clamp(min=-32768.0, max=32767.0)
        return generated_wav.to(torch.int16)


with torch.inference_mode():
    model = BigVGAN.from_pretrained(model_path, use_cuda_kernel=False)
    model.remove_weight_norm()  # remove weight norm in the model and set to eval mode
    model = model.eval().to('cpu').float()
    model = BIGVGAN(model, USE_TANH)

    print("\nStart to Export the BigVGAN...\n")
    mel_features = torch.ones((1, N_MELS, MAX_SIGNAL_LENGTH), dtype=torch.float32)
    torch.onnx.export(
        model,
        (mel_features,),
        onnx_model_A,
        input_names=['mel_features'],
        output_names=['generated_wav'],
        dynamic_axes={
            'mel_features': {2: 'mel_features_len'},
            'generated_wav': {2: 'generated_len'}
        } if DYNAMIC_AXIS else None,
        do_constant_folding=True,
        opset_version=17)
    del model
    del mel_features
    gc.collect()
    print("\nExport Done.")


if model_path in sys.path:
    sys.path.remove(model_path)


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4                  # Fatal level = 4, it an adjustable value.
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

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
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
            'fuse_conv_bias': '0',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '1',
            'tunable_op_tuning_enable': '1',
            'tunable_op_max_tuning_duration_ms': 10000,
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

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_A.get_providers()[0]}")
model_A_dtype = ort_session_A._inputs_meta[0].type
if 'float16' in model_A_dtype:
    model_A_dtype = np.float16
else:
    model_A_dtype = np.float32
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name

test_dummy = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((1, ort_session_A._inputs_meta[0].shape[1], 100), dtype=model_A_dtype), device_type', DEVICE_ID)

try:
    print("\nStart to Run the BigVGAN by ONNX Runtime\n")
    start_time = time.time()
    output = ort_session_A.run_with_ort_values(
        [out_name_A0],
        {
            in_name_A0: test_dummy
        })
    print(f"\nBigVGAN ran successfully on ONNX Runtime.\n\nONNXRuntime Time Cost in Seconds: {time.time() - start_time:.3f}")
except:
    print("\nBigVGAN encountered errors when running on ONNX Runtime.")
