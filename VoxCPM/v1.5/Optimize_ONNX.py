"""Quantize and optimize compact VoxCPM graphs, then rebuild one shared bundle.

Each graph has an independent top-level plan. Compatible graph subsets may
reuse shared weight packing as an optional fast path.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (candidate / "Optimize_ONNX_Common.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    _retarget_external_location,
    process_model,
    quantize_weight_only,
    quantize_weight_only_shared,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
)
from Shared_Weights import (  # noqa: E402
    bundle_shared_initializers,
)

# User configuration
SOURCE_FOLDER = SCRIPT_DIR / "VoxCPM_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "VoxCPM_Optimized"
METADATA_MODEL = "VoxCPM_Metadata"
CORE_TEMPLATE = "VoxCPM_DecodeStep"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}
QUANTIZABLE_PREFIXES = ("main/", "feat_encoder/")
SENSITIVE_PREFIXES = ("feat_decoder/", "reference_vae/")

# Quantization and optimization defaults
MATMUL_ALGORITHM = "AFFINE_REFINE_V2"
BLOCK_SIZE = 32
ACCURACY_LEVEL = 4
MAIN_NUM_HEADS = 16
MAIN_HIDDEN_SIZE = 1024


def _selected_constant_nodes(model_path, prefixes, op_type):
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name for tensor in model.graph.initializer}
    weight_index = 1 if op_type == "MatMul" else 0
    return [
        node.name
        for node in model.graph.node
        if node.op_type == op_type
        and len(node.input) == 2
        and node.name.startswith(prefixes)
        and node.input[weight_index] in initializers
    ]


def select_core_nodes(model_path):
    return _selected_constant_nodes(model_path, QUANTIZABLE_PREFIXES, "MatMul")


def select_text_embedding(model_path):
    return _selected_constant_nodes(model_path, ("prefill/",), "Gather")


def select_main_prefill_nodes(model_path):
    return select_core_nodes(model_path) + select_text_embedding(model_path)


# Per-graph quantization and optimization plan
MODEL_PLANS: dict[str, Plan] = {
    "VoxCPM_ReferencePreprocess": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_include=select_core_nodes,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
        optimize=True,
        transformer=False,
        opt_level=2,
    ),
    "VoxCPM_MainPrefill": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_include=select_main_prefill_nodes,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
        optimize=True,
        transformer=True,
        opt_level=2,
    ),
    "VoxCPM_DecodeStep": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_include=select_core_nodes,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
        optimize=True,
        transformer=True,
        opt_level=2,
    ),
    "VoxCPM_VAE_Decoder": Plan(
        method="F32",
        external=True,
        optimize=True,
        transformer=False,
        opt_level=1,
    ),
    "VoxCPM_VAE_Decoder_Stream": Plan(
        method="F32",
        external=True,
        optimize=True,
        transformer=False,
        opt_level=1,
    ),
    "VoxCPM_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}

FUNCTIONAL_MODELS = tuple(name for name in MODEL_PLANS if name != METADATA_MODEL)
CORE_MODELS = (
    "VoxCPM_ReferencePreprocess",
    "VoxCPM_MainPrefill",
    "VoxCPM_DecodeStep",
)

CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    block_size=BLOCK_SIZE,
    accuracy_level=ACCURACY_LEVEL,
    dynamic_default_tensor_type=onnx.TensorProto.FLOAT,
    shape_infer=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def _candidate_keys(model_path, selector, op_type):
    model = onnx.load(str(model_path), load_external_data=False)
    selected = set(selector(model_path))
    initializers = {tensor.name for tensor in model.graph.initializer}
    weight_index = 1 if op_type == "MatMul" else 0
    return {
        (node.op_type, node.input[weight_index])
        for node in model.graph.node
        if node.name in selected
        and node.op_type == op_type
        and len(node.input) == 2
        and node.input[weight_index] in initializers
    }


def configure_precision(source_folder):
    metadata = read_onnx_metadata(str(source_folder / f"{METADATA_MODEL}.onnx"))
    precision = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
    invalid = {key: value for key, value in precision.items() if value not in {"0", "1"}}
    preserve = precision == {"use_f16_kv": "1", "compute_in_f32": "0"}
    if preserve:
        print(
            "[Precision] f16 KV/f16 attention: MainPrefill and DecodeStep use "
            "offline opt_level=0 to prevent CastFloat16Transformer promotion."
        )
    return metadata, preserve


def supports_transformer_fusions(source_folder):
    model = onnx.load(
        str(source_folder / "VoxCPM_DecodeStep.onnx"),
        load_external_data=False,
    )
    return any(
        node.op_type == "SimplifiedLayerNormalization"
        for node in model.graph.node
    )


def resolve_plans(
    preserve_fp16_attention,
    allow_transformer_fusions,
):
    resolved_plans = {}
    for name, plan_config in MODEL_PLANS.items():
        plan = resolve_plan(plan_config, CONFIG)
        transformer = plan.transformer and allow_transformer_fusions
        plan = replace(
            plan,
            transformer=transformer,
            opt_level=0 if preserve_fp16_attention and transformer else plan.opt_level,
        )
        resolved_plans[name] = plan
    return resolved_plans


def _selector_signature(selector):
    if selector is None:
        return None
    if callable(selector):
        return (selector.__module__, selector.__qualname__)
    return tuple(selector)


def _quantization_signature(plan):
    return (
        plan.method,
        plan.algo,
        plan.op_types,
        plan.axes,
        plan.block_size,
        plan.accuracy_level,
        plan.symmetric,
        plan.quant_format,
        _selector_signature(plan.nodes_to_include),
        _selector_signature(plan.nodes_to_exclude),
        plan.external,
    )


def quantize_configured_graphs(
    source_folder,
    output_folder,
    resolved_plans,
):
    prequantized = set()
    transient_paths = []
    stats = {"shared_quantization": []}

    try:
        template_plan = resolved_plans[CORE_TEMPLATE]
        core_plan = replace(
            template_plan,
            op_types=("MatMul",),
            axes=(0,),
            nodes_to_include=select_core_nodes,
        )
        core_names = tuple(
            name
            for name in CORE_MODELS
            if resolved_plans[name].method in WEIGHT_ONLY_BITS
        )
        if CORE_TEMPLATE not in core_names:
            raise ValueError(f"{CORE_TEMPLATE} is not configured for shared weight-only packing.")

        cache_path = output_folder / (
            f".VoxCPM_core_{core_plan.method}_QuantizedWeights.onnx"
        )
        transient_paths.extend((cache_path, Path(str(cache_path) + ".data")))
        print(
            f"[Shared quantization] core MatMul: {core_plan.method}, "
            f"block={core_plan.block_size}, graphs={core_names}."
        )
        pass_stats = quantize_weight_only_shared(
            str(source_folder / f"{CORE_TEMPLATE}.onnx"),
            [
                (
                    str(source_folder / f"{name}.onnx"),
                    str(output_folder / f"{name}.onnx"),
                )
                for name in core_names
            ],
            str(cache_path),
            core_plan,
            bits=WEIGHT_ONLY_BITS[core_plan.method],
            external=core_plan.external,
        )
        prequantized.update(core_names)
        stats["shared_quantization"].append(
            {
                "group": "core_matmul",
                "graphs": core_names,
                "method": core_plan.method,
                "bits": WEIGHT_ONLY_BITS[core_plan.method],
                **pass_stats,
            }
        )

        main_name = "VoxCPM_MainPrefill"
        main_path = output_folder / f"{main_name}.onnx"
        main_temp_path = output_folder / f".{main_name}_Gather.onnx"
        main_gather_plan = replace(
            resolved_plans[main_name],
            op_types=("Gather",),
            axes=(1,),
            nodes_to_include=select_text_embedding,
        )
        print(
            "[Shared quantization] MainPrefill core reused; quantizing its "
            "prefill embedding Gather with portable Q4."
        )
        quantize_weight_only(
            str(main_path),
            str(main_temp_path),
            main_gather_plan,
            4,
            external=True,
        )
        main_temp_data = Path(str(main_temp_path) + ".data")
        main_data = Path(str(main_path) + ".data")
        main_path.unlink(missing_ok=True)
        main_data.unlink(missing_ok=True)
        main_temp_path.replace(main_path)
        if main_temp_data.is_file():
            main_temp_data.replace(main_data)
            _retarget_external_location(
                str(main_path),
                main_temp_data.name,
                main_data.name,
            )

    except Exception as error:
        print(
            "[Shared quantization] Compact core replay was not applicable; "
            f"processing configured graphs independently ({error})."
        )
        prequantized.clear()
        for name in CORE_MODELS:
            output_path = output_folder / f"{name}.onnx"
            output_path.unlink(missing_ok=True)
            output_path.with_name(output_path.name + ".data").unlink(missing_ok=True)
        for path in transient_paths:
            path.unlink(missing_ok=True)
    return {
        "stats": stats,
        "prequantized": prequantized,
        "transient_paths": transient_paths,
    }


def process_graphs(config, resolved_plans, prequantized):
    mixed_precision = uses_mixed_precision(
        resolved_plans[name] for name in FUNCTIONAL_MODELS
    )
    if mixed_precision and config.f16_keep_io_types is None:
        print(
            "[Precision] Not all graphs use F16; enabling keep_io_types for "
            "float16 conversions."
        )
    for name in FUNCTIONAL_MODELS:
        plan = resolved_plans[name]
        is_prequantized = name in prequantized
        print(
            f"\n[Process] {name}: {resolved_plans[name].method}, "
            f"transformer={plan.transformer}, opt_level={plan.opt_level}, "
            f"prequantized={is_prequantized}"
        )
        process_model(
            name,
            plan,
            config,
            mixed_precision=mixed_precision,
            prequantized=is_prequantized,
        )


def rebuild_bundle(source_folder, output_folder, metadata):
    model_paths = [source_folder / f"{METADATA_MODEL}.onnx"]
    model_paths.extend(output_folder / f"{name}.onnx" for name in FUNCTIONAL_MODELS)
    stats = bundle_shared_initializers(
        output_folder,
        model_paths=model_paths,
        metadata=metadata,
    )
    replace_onnx_metadata(
        str(output_folder / f"{METADATA_MODEL}.onnx"),
        metadata,
    )
    print(
        f"[Shared bundle] {stats['initializer_references']} references -> "
        f"{stats['unique_initializers']} tensors; "
        f"deduplicated {stats['deduplicated_bytes'] / (1024 ** 2):.2f} MiB."
    )
    return stats


def main():
    args = parse_args()
    resolved_plans = resolve_plans(False, True)
    metadata, preserve_fp16_attention = configure_precision(SOURCE_FOLDER)
    allow_transformer_fusions = supports_transformer_fusions(SOURCE_FOLDER)
    resolved_plans = resolve_plans(
        preserve_fp16_attention,
        allow_transformer_fusions,
    )
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True)
    transient_paths = []
    try:
        quantization = quantize_configured_graphs(
            SOURCE_FOLDER,
            OUTPUT_FOLDER,
            resolved_plans,
        )
        transient_paths.extend(quantization["transient_paths"])
        process_graphs(
            CONFIG,
            resolved_plans,
            quantization["prequantized"],
        )
        output_metadata = dict(metadata)
        rebuild_bundle(
            SOURCE_FOLDER,
            OUTPUT_FOLDER,
            output_metadata,
        )
    finally:
        for path in transient_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
