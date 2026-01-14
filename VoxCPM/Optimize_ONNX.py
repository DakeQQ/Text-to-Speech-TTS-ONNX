import os
import gc
import glob
import torch
import subprocess
import onnx
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils
)

# Path Setting
original_folder_path = r"/home/DakeQQ/Downloads/VoxCPM_ONNX"                    # The original folder.
quanted_folder_path = r"/home/DakeQQ/Downloads/VoxCPM_Optimized"                # The optimized folder.

# Create the output directory if it doesn't exist
os.makedirs(quanted_folder_path, exist_ok=True)

# Set both False for manually setting.
lazy_setting_CPU = True                     # Set true to auto-select CPU settings.
lazy_setting_GPU = False                    # Set true to auto-select GPU settings.

use_openvino = False                        # Set true for OpenVINO optimization.
use_low_memory_mode_in_Android = False      # If True, save the model into 2 parts.
upgrade_opset = 0                           # Optional process. Set 0 for close.


#------------------------------------------------------------------------------ 
# Manual Settings
#------------------------------------------------------------------------------ 
# List of models to process
model_names = [             # Recommended quantize dtype. The int8 is best for CPU.
    "VoxCPM_Text_Embed",    # [int8, int4, float32, float16]
    "VoxCPM_VAE_Encoder",   # [float32, float16]
    "VoxCPM_Feat_Encoder",  # [int8, float32, float16]
    "VoxCPM_Feat_Cond",     # [float32, float16]
    "VoxCPM_Concat",        # [float32, float16]
    "VoxCPM_Main",          # [int8, float32, float16]
    "VoxCPM_Feat_Decoder",  # [float32, int8, float16]
    "VoxCPM_VAE_Decoder"    # [float32, float16]
]

# Manual Settings
quant_int4 = False                       # Quant the model to int4 format (not used by auto settings below).
quant_int8 = False                       # Global default, overridden per model.
quant_float16 = False                    # Global default, overridden per model.
keep_io_dtype = True                     # Will be overridden when needed; must be True for mixed-precision.

# Int4 matmul_nbits_quantizer Settings
algorithm = "k_quant"                    # ["DEFAULT", "RTN", "HQQ", "k_quant"]
bits = 4                                 # [4, 8]; It is not recommended to use 8.
block_size = 32                          # [32, 64, 128, 256]; Smaller block_size => more accuracy, more time and size.
accuracy_level = 4                       # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric = False                  # False may get more accuracy.
nodes_to_exclude = None                  # Example: ["/layers.0/mlp/down_proj/MatMul"]

# Per-model target dtype mapping:
# "int8", "float16", "float32"
if lazy_setting_CPU:
    if use_openvino:
        CPU_MODEL_DTYPE = {
            "VoxCPM_Text_Embed": "int8",
            "VoxCPM_VAE_Encoder": "float32",
            "VoxCPM_Feat_Encoder": "float32",
            "VoxCPM_Feat_Cond": "float32",
            "VoxCPM_Concat": "float32",
            "VoxCPM_Main": "int8",
            "VoxCPM_Feat_Decoder": "float32", # int8 can also work, but it will affect the quality.
            "VoxCPM_VAE_Decoder": "float32",
        }
    else:
        CPU_MODEL_DTYPE = {
            "VoxCPM_Text_Embed": "int8",
            "VoxCPM_VAE_Encoder": "float32",
            "VoxCPM_Feat_Encoder": "int8",
            "VoxCPM_Feat_Cond": "float32",
            "VoxCPM_Concat": "float32",
            "VoxCPM_Main": "int8",
            "VoxCPM_Feat_Decoder": "float32",  # int8 can also work, but it will affect the quality.
            "VoxCPM_VAE_Decoder": "float32",
        }
elif lazy_setting_GPU:
    GPU_MODEL_DTYPE = {
        "VoxCPM_Text_Embed": "float16",
        "VoxCPM_VAE_Encoder": "float16",
        "VoxCPM_Feat_Encoder": "float16",
        "VoxCPM_Feat_Cond": "float16",
        "VoxCPM_Concat": "float16",
        "VoxCPM_Main": "float16",
        "VoxCPM_Feat_Decoder": "float16",
        "VoxCPM_VAE_Decoder": "float16",
    }

# Validate lazy settings (one of them should be True)
if lazy_setting_CPU and lazy_setting_GPU:
    raise ValueError("Only one of lazy_setting_CPU or lazy_setting_GPU can be True.")

# --- Main Processing Loop ---
algorithm_copy = algorithm
for model_name in model_names:
    print(f"--- Processing model: {model_name} ---")
    be_optimized = False

    # Dynamically set model paths for the current iteration
    model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
    quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")

    # Check if the original model file exists before processing
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping.")
        continue

    # --- Auto-select dtype and flags per model ---
    if lazy_setting_GPU:
        target_dtype = GPU_MODEL_DTYPE.get(model_name, "float16")
        keep_io_dtype = False
    elif lazy_setting_CPU:
        target_dtype = CPU_MODEL_DTYPE.get(model_name, "float32")
        keep_io_dtype = False
    else:
        target_dtype = None

    # Reset per-iteration quantization flags
    if target_dtype:
        quant_int4 = False
        quant_int8 = (target_dtype == "int8")
        quant_float16 = (target_dtype == "float16")

    print(f"Selected target dtype for {model_name}: {target_dtype}")
    print(f"quant_int8={quant_int8}, quant_float16={quant_float16}, keep_io_dtype={keep_io_dtype}")

    # Start Quantize / Optimize according to target dtype
    if quant_int4 and ("Embed" in model_path or "Main" in model_path or "Encoder" in model_path or "Decoder" in model_path):
        # Int4 path (not used by current auto rules, but kept for completeness)
        if "Embed" in model_path:
            op_types = ["Gather"]
            quant_axes = [1]
            algorithm = "DEFAULT"  # Fallback to DEFAULT
        else:
            op_types = ["MatMul"]
            quant_axes = [0]
            algorithm = algorithm_copy

        # Start Weight-Only Quantize
        model = quant_utils.load_model_with_shape_infer(Path(model_path))

        if algorithm == "RTN":
            quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types)
            )
        elif algorithm == "HQQ":
            quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
                bits=bits,
                block_size=block_size,
                axis=quant_axes[0],
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
            )
        elif algorithm == "k_quant":
            quant_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types)
            )
        else:
            quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=block_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
            )
        quant_config.bits = bits
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types))),
            algo_config=quant_config,
            nodes_to_exclude=nodes_to_exclude
        )
        quant.process()
        quant.model.save_model_to_file(
            quanted_model_path,
            True  # save_as_external_data
        )

    elif quant_int8:
        # INT8 dynamic weight quantization
        print("Applying INT8 (dynamic) quantization...")
        quantize_dynamic(
            model_input=quant_utils.load_model_with_shape_infer(Path(model_path)),
            model_output=quanted_model_path,
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            extra_options={
                'ActivationSymmetric': False,
                'WeightSymmetric': False,
                'EnableSubgraph': True,
                'ForceQuantizeNoInputCheck': False,
                'MatMulConstBOnly': True
            },
            nodes_to_exclude=None,
            use_external_data_format=True
        )

    elif quant_float16:
        # Float16 path: optimize then convert to fp16
        print("Optimizing model before Float16 conversion...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=1 if (use_openvino or "VAE_Decoder" in model_path) else 2,
            num_heads=16 if ("Main" in model_path or "Encoder" in model_path or "Decoder" in model_path) else 0,
            hidden_size=1024 if ("Main" in model_path or "Encoder" in model_path or "Decoder" in model_path) else 0,
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino
        )
        print("Converting model to Float16...")
        model.convert_float_to_float16(
            keep_io_types=keep_io_dtype,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=65504.0,
            op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=use_low_memory_mode_in_Android)

    else:
        # Float32 path (no quantization) -> just optimize
        print("Target dtype is float32: optimizing without quantization...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=1 if (use_openvino or "VAE_Decoder" in model_path) else 2,
            num_heads=16 if ("Main" in model_path or "Encoder" in model_path or "Decoder" in model_path) else 0,
            hidden_size=1024 if ("Main" in model_path or "Encoder" in model_path or "Decoder" in model_path) else 0,
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=use_low_memory_mode_in_Android)

    # Extra optimization pass if quantization branch didn't already optimize
    if not be_optimized:
        print("Running additional ONNX Runtime optimization on quantized model...")
        model = optimize_model(
            quanted_model_path,
            use_gpu=False,
            opt_level=1 if (use_openvino or "VAE_Decoder" in model_path) else 2,
            num_heads=16 if ("Main" in model_path or "Encoder" in model_path or "Decoder" in model_path) else 0,
            hidden_size=1024 if ("Main" in model_path or "Encoder" in model_path or "Decoder" in model_path) else 0,
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=use_low_memory_mode_in_Android)

    # Slim the model
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=False,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=use_low_memory_mode_in_Android,
        verbose=False
    )

    # Upgrade the Opset version. (optional process)
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            model = onnx.load(quanted_model_path)
            converted_model = onnx.version_converter.convert_version(model, upgrade_opset)
            onnx.save(converted_model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
            del model, converted_model
            gc.collect()
        except Exception as e:
            print(f"Could not upgrade opset due to an error: {e}. Saving model with original opset.")
            model = onnx.load(quanted_model_path)
            onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
            del model
            gc.collect()
    else:
        model = onnx.load(quanted_model_path)
        onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
        del model
        gc.collect()

# Clean up external data files at the very end
print("Cleaning up temporary \*.onnx.data files...")
pattern = os.path.join(quanted_folder_path, '*.onnx.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("--- All models processed successfully! ---")
