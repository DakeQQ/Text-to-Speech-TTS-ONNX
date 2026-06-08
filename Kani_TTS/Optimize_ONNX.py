import os
import gc
import glob
import torch
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForCausalLM
from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils
)

# Path Setting
download_path = r'/home/DakeQQ/Downloads/kani-tts-370m'                          # Set the folder path where the whole project downloaded, otherwise set "NONE".
original_folder_path = r"/home/DakeQQ/Downloads/KaniTTS_ONNX"                    # The original folder.
quanted_folder_path = r"/home/DakeQQ/Downloads/KaniTTS_Optimized"                # The optimized folder.

# Create the output directory if it doesn't exist
os.makedirs(quanted_folder_path, exist_ok=True)

# List of models to process
model_names = [
    "KaniTTS_Embed",
    "KaniTTS_Main",
    "Greedy_Search",
    "First_Beam_Search",
    "Second_Beam_Search",
    "Apply_Penalty",
    "Argmax",
    "KaniTTS_Codec"                      # It is recommended to use CPU and Float32 format instead of low-end GPU.
]

# Settings
quant_int4 = True                        # Quant the model to int4 format.
quant_int8 = False                       # Quant the model to int8 format.
use_matmul_nbits_for_int8 = True         # If True, use matmul_nbits_quantizer for int8 instead of quantize_dynamic.
quant_float16 = False                    # Quant the model to float16 format.
use_openvino = False                     # Set true for OpenVINO optimization.
save_two_part = False                    # If True, save the model into 2 parts.
upgrade_opset = 0                        # Optional process. Set 0 for close.

# Int4 matmul_nbits_quantizer Settings
algorithm = "k_quant"                    # ["DEFAULT", "RTN", "HQQ", "k_quant"]
bits = 4                                 # [4, 8]; It is not recommended to use 8.
block_size = 16                          # [32, 64, 128, 256]; A smaller block_size yields greater accuracy but increases quantization time and model size.
accuracy_level = 4                       # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric = False                  # False may get more accuracy.
nodes_to_exclude = None                  # Set the node names here. Such as: ["/layers.0/mlp/down_proj/MatMul"]


# --- Helper Functions ---

def create_quant_config(algo, op_types, quant_axes, bit):
    """Create a weight-only quantization config based on the selected algorithm."""
    ops_tuple = tuple(op_types)
    axes_tuple = tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))

    if algo == "RTN":
        config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=ops_tuple
        )
    elif algo == "HQQ":
        config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=bit,
            block_size=block_size,
            axis=quant_axes[0],
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=ops_tuple,
            quant_axes=axes_tuple
        )
    elif algo == "k_quant":
        config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=ops_tuple
        )
    else:
        config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=block_size,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=ops_tuple,
            quant_axes=axes_tuple
        )
    config.bits = bit
    return config


def run_nbits_quantization(input_path, output_path, op_types, quant_axes, algo, bit, blk_size=32):
    """Run MatMulNBits weight-only quantization and save the result."""
    print(f"Applying matmul_nbits_quantizer INT{bit} quantization...")
    model = quant_utils.load_model_with_shape_infer(Path(input_path))
    config = create_quant_config(algo, op_types, quant_axes, bit)
    ops_tuple = tuple(op_types)
    axes_tuple = tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))

    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        block_size=blk_size,
        is_symmetric=quant_symmetric,
        accuracy_level=accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=ops_tuple,
        quant_axes=axes_tuple,
        algo_config=config,
        nodes_to_exclude=nodes_to_exclude
    )
    quant.process()
    quant.model.save_model_to_file(output_path, True)


def run_slim(input_model, output_path, no_shape_infer=True, dtype=None):
    """Run onnxslim optimization. Only pass dtype when explicitly set (e.g. 'fp16')."""
    print("Running onnxslim...")
    kwargs = dict(
        model=input_model,
        output_model=output_path,
        no_shape_infer=no_shape_infer,
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=save_two_part,
        verbose=False,
    )
    if dtype is not None:
        kwargs['dtype'] = dtype
    slim(**kwargs)


def run_optimizer(input_path, output_path, num_heads=0, hidden_size=0, apply_fp16=False):
    """Run ORT transformers optimizer with optional FP16 conversion."""
    print("Applying transformers.optimizer...")
    model = optimize_model(
        input_path,
        use_gpu=False,
        opt_level=1 if use_openvino else 2,
        num_heads=num_heads,
        hidden_size=hidden_size,
        verbose=False,
        model_type='bert',
        only_onnxruntime=use_openvino
    )
    if apply_fp16:
        print("Converting model to Float16...")
        model.convert_float_to_float16(
            keep_io_types=False,
            force_fp16_initializers=True,
            use_symbolic_shape_infer=True,
            max_finite_val=32767.0,
            op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
        )
    model.save_model_to_file(output_path, use_external_data_format=save_two_part)
    del model
    gc.collect()


def get_model_config():
    """Load num_heads and hidden_size from the pretrained model config."""
    if download_path is None or download_path.lower() == "none":
        return 0, 0
    model_for_config = AutoModelForCausalLM.from_pretrained(
        download_path, torch_dtype=torch.float16, device_map='cpu',
        trust_remote_code=True, low_cpu_mem_usage=True
    ).eval()
    try:
        num_heads = model_for_config.config.num_attention_heads
        hidden_size = model_for_config.config.hidden_size
    except Exception:
        num_heads, hidden_size = 0, 0
    del model_for_config
    gc.collect()
    return num_heads, hidden_size


def finalize_model(output_path):
    """Apply opset upgrade or re-save to consolidate external data."""
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            model = onnx.load(output_path)
            converted_model = onnx.version_converter.convert_version(model, upgrade_opset)
            onnx.save(converted_model, output_path, save_as_external_data=save_two_part)
            del model, converted_model
        except Exception as e:
            print(f"Could not upgrade opset: {e}. Re-saving with original opset.")
            model = onnx.load(output_path)
            onnx.save(model, output_path, save_as_external_data=save_two_part)
            del model
    else:
        model = onnx.load(output_path)
        onnx.save(model, output_path, save_as_external_data=save_two_part)
        del model
    gc.collect()


# --- Main Processing Loop ---
for model_name in model_names:
    print(f"\n--- Processing model: {model_name} ---")

    model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
    quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")

    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping.")
        continue

    is_embed = "KaniTTS_Embed" in model_name
    is_main = "KaniTTS_Main" in model_name
    is_codec = "KaniTTS_Codec" in model_name
    is_reset = "Reset_Penality" in model_name
    is_first_beam = "First_Beam_Search" in model_name
    skip_full_optimizer = is_reset or is_first_beam or is_codec

    # --- Step 1: Quantization / Initial Optimization ---
    if quant_int4 and (is_embed or is_main):
        # Weight-only INT4 quantization for Embed and Main models
        if is_embed:
            run_nbits_quantization(model_path, quanted_model_path,
                                   op_types=["Gather"], quant_axes=[1], algo="DEFAULT", bit=4, blk_size=16)
        else:
            run_nbits_quantization(model_path, quanted_model_path,
                                   op_types=["MatMul"], quant_axes=[0], algo=algorithm, bit=4, blk_size=block_size)

    elif quant_int8 and not is_reset and not is_codec:
        # INT8 quantization
        if use_matmul_nbits_for_int8:
            if is_embed:
                run_nbits_quantization(model_path, quanted_model_path,
                                       op_types=["Gather"], quant_axes=[1], algo="DEFAULT", bit=4, blk_size=16)
            else:
                run_nbits_quantization(model_path, quanted_model_path,
                                       op_types=["MatMul"], quant_axes=[0], algo=algorithm, bit=8, blk_size=block_size)
        else:
            print("Applying UINT8 dynamic quantization...")
            quantize_dynamic(
                model_input=quant_utils.load_model_with_shape_infer(Path(model_path)),
                model_output=quanted_model_path,
                per_channel=False,
                reduce_range=False,
                weight_type=QuantType.QUInt8,
                extra_options={'ActivationSymmetric': False,
                               'WeightSymmetric': False,
                               'EnableSubgraph': True,
                               'ForceQuantizeNoInputCheck': False,
                               'MatMulConstBOnly': True},
                nodes_to_exclude=None,
                use_external_data_format=True
            )
        run_slim(quanted_model_path, quanted_model_path)

    elif is_codec:
        # Codec: slim then optimize
        run_slim(model_path, quanted_model_path, no_shape_infer=False)
        run_optimizer(quanted_model_path, quanted_model_path)

    elif is_reset:
        # Reset_Penality: optimize directly, optional FP16
        run_optimizer(model_path, quanted_model_path, apply_fp16=quant_float16)

    else:
        # Other models: slim with optional FP16 for First_Beam_Search
        dtype = 'fp16' if (quant_float16 and is_first_beam) else None
        run_slim(quant_utils.load_model_with_shape_infer(Path(model_path)), quanted_model_path, dtype=dtype)

    # --- Step 2: Full Transformer Optimizer Pass (skipped for Reset, First_Beam, Codec) ---
    if not skip_full_optimizer:
        num_heads, hidden_size = get_model_config()
        run_optimizer(quanted_model_path, quanted_model_path,
                      num_heads=num_heads, hidden_size=hidden_size, apply_fp16=quant_float16)
        run_slim(quanted_model_path, quanted_model_path)

    # --- Step 3: Finalize (opset upgrade / re-save) ---
    finalize_model(quanted_model_path)


# --- Cleanup external data files ---
print("\nCleaning up temporary *.onnx.data files...")
for file_path in glob.glob(os.path.join(quanted_folder_path, '*.onnx.data')):
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("--- All models processed successfully! ---")
