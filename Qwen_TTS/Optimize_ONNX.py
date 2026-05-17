import os
import gc
import glob
import onnx
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils
)


# ==============================================================================
# Path Settings
# ==============================================================================
model_type           = "0.6B-Base"                                   # [0.6B-Base, 1.7B-Base, 1.7B-CustomVoice, 1.7B-VoiceDesign]
original_folder_path = r"/home/DakeQQ/Downloads/QwenTTS_ONNX"        # The original exported folder.
quanted_folder_path  = r"/home/DakeQQ/Downloads/QwenTTS_Optimized"   # The optimized folder.

os.makedirs(quanted_folder_path, exist_ok=True)


# ==============================================================================
# Lazy Settings (set one True for auto-select, or both False for manual)
# ==============================================================================
lazy_setting_CPU = True                  # Set True to auto-select CPU settings.
lazy_setting_GPU = False                 # Set True to auto-select GPU settings.
use_openvino     = False                 # Set True for OpenVINO optimization.
SAVE_TWO_PARTS   = False                 # If True, save the model into 2 parts.
upgrade_opset    = 0                     # Optional opset upgrade. Set 0 to disable.
NUM_CODE_GROUPS_MINUS = 15               # Number of Embed_D / LmHead sub-models.


# ==============================================================================
# Model List
# ==============================================================================
model_names = [                                             # Recommended dtype:
    "QwenTTS_Embed_A",                                      # [int4, float32, float16]
    "QwenTTS_Embed_B",                                      # [float32, float16, int4]
    "QwenTTS_Embed_C",                                      # [float32, float16, int4]
    "QwenTTS_Embed_D",                                      # [float32, float16, int4]
    "QwenTTS_Preprocess",                                   # [float32, float16]
    "QwenTTS_Encoder",                                      # [int8, float32, float16]
    "QwenTTS_Predictor",                                    # [int8, float32, float16]
    "QwenTTS_Predictor_LmHead",                             # [int8, float32, float16]
    "QwenTTS_Main",                                         # [int8, float32, float16]
    "QwenTTS_Decoder",                                      # [int8, float32, float16]
    "QwenTTS_Main_Rotary_Mask_Text_Prefill",                # [float32, float16]
    "QwenTTS_Main_Rotary_Mask_Text_Decode",                 # [float32, float16]
    "QwenTTS_Predictor_Rotary_Mask_Text_Prefill",           # [float32, float16]
    "QwenTTS_Predictor_Rotary_Mask_Text_Decode",            # [float32, float16]
    "Gather_0",                                             # [float32, float16]
    "Concat_Embed",                                         # [float32, float16]
    "Concat_Ids",                                           # [float32, float16]
    "Greedy_Search",                                        # [float32, float16]
    "First_Beam_Search",                                    # [float32, float16]
    "Second_Beam_Search",                                   # [float32, float16]
    "Apply_Penalty",                                        # [float32, float16]
    "Argmax",                                               # [float32, float16]
]


# ==============================================================================
# Manual Quantization Settings
# ==============================================================================
quant_int4       = False                 # Quant to int4 (not used by auto settings).
quant_int8       = False                 # Global default, overridden per model.
quant_float16    = False                 # Global default, overridden per model.
keep_io_dtype    = True                  # Must be True for mixed-precision.
fp16_op_block_list = [
    'DynamicQuantizeLinear',
    'DequantizeLinear',
    'DynamicQuantizeMatMul',
    'Range',
    'MatMulIntegerToFloat',
]


# ==============================================================================
# Int4 matmul_nbits_quantizer Settings
# ==============================================================================
algorithm        = "k_quant"             # ["DEFAULT", "RTN", "HQQ", "k_quant"]
bits             = 4                     # [4, 8]; 8 is not recommended.
block_size       = 16                    # [16, 32, 64, 128, 256]; smaller => more accuracy.
accuracy_level   = 4                     # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric  = False                 # False may yield more accuracy.
nodes_to_exclude = None                  # Example: ["/layers.0/mlp/down_proj/MatMul"]


# ==============================================================================
# Per-Model Target Dtype Mapping (CPU)
# ==============================================================================
CPU_MODEL_DTYPE = {
    "QwenTTS_Embed_A":                            "int4",
    "QwenTTS_Embed_B":                            "float32",
    "QwenTTS_Embed_C":                            "float32",
    "QwenTTS_Embed_D":                            "float32",
    "QwenTTS_Preprocess":                         "float32",
    "QwenTTS_Encoder":                            "int8",
    "QwenTTS_Predictor":                          "int8",
    "QwenTTS_Predictor_LmHead":                   "int8",
    "QwenTTS_Main":                               "int8",
    "QwenTTS_Decoder":                            "int8",
    "QwenTTS_Main_Rotary_Mask_Text_Prefill":      "float32",
    "QwenTTS_Main_Rotary_Mask_Text_Decode":       "float32",
    "QwenTTS_Predictor_Rotary_Mask_Text_Prefill": "float32",
    "QwenTTS_Predictor_Rotary_Mask_Text_Decode":  "float32",
    "Gather_0":                                   "float32",
    "Concat_Embed":                               "float32",
    "Concat_Ids":                                 "float32",
    "Greedy_Search":                              "float32",
    "First_Beam_Search":                          "float32",
    "Second_Beam_Search":                         "float32",
    "Apply_Penalty":                              "float32",
    "Argmax":                                     "float32",
}


# ==============================================================================
# Per-Model Target Dtype Mapping (GPU)
# ==============================================================================
GPU_MODEL_DTYPE = {
    "QwenTTS_Embed_A":                            "int4",
    "QwenTTS_Embed_B":                            "float16",
    "QwenTTS_Embed_C":                            "float32",
    "QwenTTS_Embed_D":                            "float16",
    "QwenTTS_Preprocess":                         "float32" if "1.7B-Base" in model_type else "float16",
    "QwenTTS_Encoder":                            "float16",
    "QwenTTS_Predictor":                          "int8",
    "QwenTTS_Predictor_LmHead":                   "float16",
    "QwenTTS_Main":                               "int8",
    "QwenTTS_Decoder":                            "float16",
    "QwenTTS_Main_Rotary_Mask_Text_Prefill":      "float16",
    "QwenTTS_Main_Rotary_Mask_Text_Decode":       "float16",
    "QwenTTS_Predictor_Rotary_Mask_Text_Prefill": "float16",
    "QwenTTS_Predictor_Rotary_Mask_Text_Decode":  "float16",
    "Gather_0":                                   "float16",
    "Concat_Embed":                               "float16",
    "Concat_Ids":                                 "float16",
    "Greedy_Search":                              "float16",
    "First_Beam_Search":                          "float16",
    "Second_Beam_Search":                         "float16",
    "Apply_Penalty":                              "float16",
    "Argmax":                                     "float16",
}

# Validate lazy settings
if lazy_setting_CPU and lazy_setting_GPU:
    raise ValueError("Only one of lazy_setting_CPU or lazy_setting_GPU can be True.")


# ==============================================================================
# Helper Functions
# ==============================================================================
def _is_transformer_block(name):
    return any(k in name for k in ("Main", "Predictor", "Encoder", "Decoder"))


def _is_embed_block(name):
    return "Embed" in name


def _is_decoder_block(name):
    return "Decoder" in name


def _opt_level(name):
    if ("Predictor" in name) and not ("Rotary" in name or "LmHead" in name):
        return 1
    return 1 if use_openvino else 2


def _num_heads(name):
    if "Main" in name or "Predictor" in name:
        return 16
    return 0


def _hidden_size(name, model_size):
    if "Main" in name and not "Rotary" in name:
        if "0.6B" in model_size:
            return 1024
        else:
            return 2048
    elif ("Predictor" in name) and not ("Rotary" in name or "LmHead" in name):
        return 1024
    else:
        return 0


# ==============================================================================
# Core Processing Function
# ==============================================================================
def process_single_model(
    model_path,
    quanted_model_path,
    model_name,
    bits,
    block_size,
    quant_int4_flag,
    quant_int8_flag,
    quant_float16_flag,
    keep_io_flag,
    op_block_list,
):
    """Process a single ONNX file: quantize / optimize / slim."""
    be_optimized = False

    if not lazy_setting_CPU and not lazy_setting_GPU:
        if quant_int4_flag:
            bits = 4
            block_size = 16
        elif quant_int8_flag:
            bits = 8
            block_size = 32

    # ------------------------------------------------------------------
    # Branch 1: Integer quantization (int4 / int8)
    # ------------------------------------------------------------------
    if (quant_int4_flag or quant_int8_flag) and (_is_embed_block(model_name) or _is_transformer_block(model_name)):
        if _is_embed_block(model_name):
            op_types = ["Gather"]
            quant_axes = [1]
            algo = "DEFAULT"
        else:
            op_types = ["MatMul"]
            quant_axes = [0]
            algo = algorithm_copy

        model = quant_utils.load_model_with_shape_infer(Path(model_path))
        axes_tuple = tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))

        if algo == "RTN":
            quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
            )
        elif algo == "HQQ":
            quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
                bits=bits,
                block_size=block_size,
                axis=quant_axes[0],
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=axes_tuple,
            )
        elif algo == "k_quant":
            quant_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
            )
        else:
            quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=block_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=axes_tuple,
            )

        quant_config.bits = bits
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=axes_tuple,
            algo_config=quant_config,
            nodes_to_exclude=nodes_to_exclude,
        )
        quant.process()
        quant.model.save_model_to_file(quanted_model_path, True)

    # ------------------------------------------------------------------
    # Branch 2: Float16 conversion
    # ------------------------------------------------------------------
    elif quant_float16_flag:
        print("Optimizing model before Float16 conversion...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name, model_type),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        print("Converting model to Float16...")
        if _is_decoder_block(model_name):
            op_block_list += ['Mul', 'ReduceSum', 'Sqrt']
        model.convert_float_to_float16(
            keep_io_types=keep_io_flag,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=32767.0,
            op_block_list=op_block_list,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Branch 3: Float32 (optimize only, no quantization)
    # ------------------------------------------------------------------
    else:
        print("Target dtype is float32: optimizing without quantization...")
        be_optimized = True
        model = optimize_model(
            model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name, model_type),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Post-quantization optimization pass
    # ------------------------------------------------------------------
    if not be_optimized:
        print("Running additional ONNX Runtime optimization on quantized model...")
        model = optimize_model(
            quanted_model_path,
            use_gpu=False,
            opt_level=_opt_level(model_name),
            num_heads=_num_heads(model_name),
            hidden_size=_hidden_size(model_name, model_type),
            verbose=False,
            model_type='bert',
            only_onnxruntime=use_openvino,
        )
        model.save_model_to_file(quanted_model_path, use_external_data_format=SAVE_TWO_PARTS)

    # ------------------------------------------------------------------
    # Slim pass
    # ------------------------------------------------------------------
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=False,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=SAVE_TWO_PARTS,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Optional opset upgrade
    # ------------------------------------------------------------------
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            m = onnx.load(quanted_model_path)
            converted_model = onnx.version_converter.convert_version(m, upgrade_opset)
            onnx.save(converted_model, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m, converted_model
            gc.collect()
        except Exception as e:
            print(f"Could not upgrade opset due to an error: {e}. Saving model with original opset.")
            m = onnx.load(quanted_model_path)
            onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
            del m
            gc.collect()
    else:
        m = onnx.load(quanted_model_path)
        onnx.save(m, quanted_model_path, save_as_external_data=SAVE_TWO_PARTS)
        del m
        gc.collect()


# ==============================================================================
# Main Processing Loop
# ==============================================================================
MULTI_FILE_MODELS = {"QwenTTS_Embed_D", "QwenTTS_Predictor_LmHead"}
algorithm_copy = algorithm

for model_name in model_names:
    print(f"\n--- Processing model: {model_name} ---")

    # Auto-select dtype and flags per model
    if lazy_setting_GPU:
        target_dtype = GPU_MODEL_DTYPE.get(model_name, "float16")
        keep_io_dtype = True
    elif lazy_setting_CPU:
        target_dtype = CPU_MODEL_DTYPE.get(model_name, "float32")
        keep_io_dtype = True
    else:
        target_dtype = None

    # Reset per-iteration quantization flags
    if target_dtype:
        quant_int4 = (target_dtype == "int4")
        quant_int8 = (target_dtype == "int8")
        quant_float16 = (target_dtype == "float16")

    print(f"Selected target dtype for {model_name}: {target_dtype}")
    print(f"quant_int8={quant_int8}, quant_float16={quant_float16}, keep_io_dtype={keep_io_dtype}")

    if model_name in MULTI_FILE_MODELS:
        for i in range(NUM_CODE_GROUPS_MINUS):
            sub_name = f"{model_name}_{i}"
            model_path = os.path.join(original_folder_path, f"{sub_name}.onnx")
            quanted_model_path = os.path.join(quanted_folder_path, f"{sub_name}.onnx")

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}. Skipping.")
                continue

            print(f"  Processing sub-model: {sub_name}")
            process_single_model(
                model_path, quanted_model_path, model_name,
                bits, block_size, quant_int4, quant_int8,
                quant_float16, keep_io_dtype, fp16_op_block_list,
            )
    else:
        model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
        quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Skipping.")
            continue

        process_single_model(
            model_path, quanted_model_path, model_name,
            bits, block_size, quant_int4, quant_int8,
            quant_float16, keep_io_dtype, fp16_op_block_list,
        )


# ==============================================================================
# Cleanup
# ==============================================================================
print("Cleaning up temporary *.onnx.data files...")
pattern = os.path.join(quanted_folder_path, '*.onnx.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("--- All models processed successfully! ---")
