import os
import gc
import glob
import onnx
import subprocess
import onnx.version_converter
from onnxslim import slim
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.optimizer import optimize_model


# Path Setting
original_folder_path = r"/home/DakeQQ/Downloads/IndexTTS_ONNX"                     # The original folder.
quanted_folder_path = r"/home/DakeQQ/Downloads/IndexTTS_Optimized"                 # The optimized folder.

model_path = os.path.join(original_folder_path, "IndexTTS_A.onnx")               # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "IndexTTS_A.onnx")        # The optimized model stored path.

# model_path = os.path.join(original_folder_path, "IndexTTS_B.onnx")               # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "IndexTTS_B.onnx")        # The optimized model stored path.
#
# model_path = os.path.join(original_folder_path, "IndexTTS_C.onnx")               # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "IndexTTS_C.onnx")        # The optimized model stored path.
#
# model_path = os.path.join(original_folder_path, "IndexTTS_D.onnx")               # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "IndexTTS_D.onnx")        # The optimized model stored path.
#
# model_path = os.path.join(original_folder_path, "IndexTTS_E.onnx")               # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "IndexTTS_E.onnx")        # The optimized model stored path.

# model_path = os.path.join(original_folder_path, "IndexTTS_F.onnx")               # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "IndexTTS_F.onnx")        # The optimized model stored path.

int8_quant = False
fp16_quant = False
use_gpu = False                                                                  # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                # ['CPUExecutionProvider', 'CUDAExecutionProvider']
target_platform = "amd64"                                                        # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.
use_low_memory_mode_in_Android = False                                           # If you need to use low memory mode on Android, please set it to True.
upgrade_opset = 17                                                               # Optional process. Set 0 for close.

# Start Quantize
if int8_quant and (("IndexTTS_A" not in model_path) or ("IndexTTS_F" not in model_path)):  # A & F are not recommend
    quantize_dynamic(
        model_input=model_path,
        model_output=quanted_model_path,
        per_channel=True,                                        # True for model accuracy but cost a lot of time during quanting process.
        reduce_range=False,                                      # True for some x86_64 platform.
        weight_type=QuantType.QUInt8,                            # It is recommended using uint8 + Symmetric False
        extra_options={'ActivationSymmetric': False,             # True for inference speed. False may keep more accuracy.
                       'WeightSymmetric': False,                 # True for inference speed. False may keep more accuracy.
                       'EnableSubgraph': True,                   # True for more quant.
                       'ForceQuantizeNoInputCheck': False,       # True for more quant.
                       'MatMulConstBOnly': True                  # False for more quant. Sometime, the inference speed may get worse.
                       },
        nodes_to_exclude=None,                                   # Specify the node names to exclude quant process. Example: nodes_to_exclude={'/Gather'}
        use_external_data_format=True                            # Save the model into two parts.
    )
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=True,                 # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=use_low_memory_mode_in_Android,
        verbose=False
    )
else:
    slim(
        model=model_path,
        output_model=quanted_model_path,
        no_shape_infer=True,                 # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=use_low_memory_mode_in_Android,
        verbose=False,
    )


# transformers.optimizer
model = optimize_model(quanted_model_path,
                       use_gpu=use_gpu,
                       opt_level=1 if (("IndexTTS_A" in model_path) or ("IndexTTS_F" in model_path)) and (fp16_quant or use_gpu) else 2,
                       num_heads=8 if (("IndexTTS_A" in model_path) or ("IndexTTS_E" in model_path)) else 0,
                       hidden_size=1280 if (("IndexTTS_A" in model_path) or ("IndexTTS_E" in model_path)) else 0,
                       provider=provider,
                       verbose=False,
                       model_type='bert')
if fp16_quant:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=True if ("IndexTTS_F" in model_path) else False,       # True for more optimize but may get errors.
        max_finite_val=65504.0,
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
    )
model.save_model_to_file(quanted_model_path, use_external_data_format=use_low_memory_mode_in_Android)
del model
gc.collect()


# onnxslim 2nd
if "IndexTTS_F" not in model_path:
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=True if ("IndexTTS_F" in model_path) else False,                 # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=use_low_memory_mode_in_Android,
        verbose=False
    )


# Upgrade the Opset version. (optional process)
model = onnx.load(quanted_model_path)
if upgrade_opset > 0:
    try:
        model = onnx.version_converter.convert_version(model, upgrade_opset)
        onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
        del model
        gc.collect()
    except:
        onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
        del model
        gc.collect()
else:
    onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
    del model
    gc.collect()


pattern = os.path.join(quanted_folder_path, '*.onnx.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
        
        
if not use_low_memory_mode_in_Android:
    # Convert the simplified model to ORT format.
    if provider == 'CPUExecutionProvider':
        optimization_style = "Fixed"
    else:
        optimization_style = "Runtime"      # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
    # Call subprocess may get permission failed on Windows system.
    subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {quanted_folder_path}'], shell=True)
