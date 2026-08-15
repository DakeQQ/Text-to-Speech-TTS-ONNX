"""Quantize and optimize Qwen3-TTS graphs, then rebuild one shared bundle.

Each graph has an independent top-level plan. Compatible strategy graphs may
reuse shared weight packing as an optional fast path.
"""

from __future__ import annotations

import argparse
import os
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
from Optimize_ONNX_Common import (  # noqa: E402 - imports follow script path setup
    OptimizerConfig,
    Plan,
    process_model,
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
MODEL_TYPE = "0.6B-Base"
STRATEGIES = ("greedy", "penalty_greedy", "sampling")
SOURCE_FOLDER = SCRIPT_DIR / "QwenTTS_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "QwenTTS_Optimized"
QUANTIZATION_TEMPLATE = "QwenTTS_DecodeStep_greedy"
QUANTIZATION_CACHE_NAME = ".QwenTTS_QuantizedWeights.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

# Quantization and optimization defaults
MATMUL_ALGORITHM = "AFFINE_REFINE_V2"
BLOCK_SIZE = 32
ACCURACY_LEVEL = 4
MAIN_NUM_HEADS = 16
MAIN_HIDDEN_SIZE = 1024 if "0.6B" in MODEL_TYPE else 2048

def optional_instruction_gather_nodes(source_path: str) -> list[str]:
    model = onnx.load(source_path, load_external_data=False)
    return [
        node.name
        for node in model.graph.node
        if (
            node.name
            and node.op_type == "Gather"
            and len(node.input) >= 2
            and node.input[1] == "instruct_text_ids"
        )
    ]

# Per-graph quantization and optimization plan
MODEL_PLANS: dict[str, Plan] = {
    "QwenTTS_ReferencePreprocess": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        optimize=True,
        transformer=False,
        external=True,
    ),
    "QwenTTS_TargetPreprocess": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=optional_instruction_gather_nodes,
        optimize=True,
        transformer=False,
        external=True,
    ),
    "QwenTTS_Decoder": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        optimize=True,
        transformer=False,
        external=True,
    ),
    "QwenTTS_Decoder_Stream": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        optimize=True,
        transformer=False,
        external=True,
    ),
    "QwenTTS_MainPrefill_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "QwenTTS_MainPrefill_penalty_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "QwenTTS_MainPrefill_sampling": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "QwenTTS_DecodeStep_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "QwenTTS_DecodeStep_penalty_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "QwenTTS_DecodeStep_sampling": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "QwenTTS_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}

STRATEGY_GRAPH_NAMES = tuple(
    f"QwenTTS_{stage}_{strategy}"
    for stage in ("MainPrefill", "DecodeStep")
    for strategy in STRATEGIES
)
PROCESS_SUPPORT_GRAPH_NAMES = (
    "QwenTTS_ReferencePreprocess",
    "QwenTTS_TargetPreprocess",
    "QwenTTS_Decoder",
    "QwenTTS_Decoder_Stream",
)
PASSTHROUGH_GRAPH_NAMES = ("QwenTTS_Metadata",)


CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    block_size=BLOCK_SIZE,
    accuracy_level=ACCURACY_LEVEL,
    dynamic_default_tensor_type=onnx.TensorProto.FLOAT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def configure_attention_precision():
    metadata = read_onnx_metadata(str(SOURCE_FOLDER / "QwenTTS_Metadata.onnx"))
    precision_flags = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
    invalid_flags = {key: value for key, value in precision_flags.items() if value not in {"0", "1"}}
    preserve_fp16_attention = precision_flags["use_f16_kv"] == "1" and precision_flags["compute_in_f32"] == "0"
    if preserve_fp16_attention:
        print(
            "[Precision] FP16 KV attention requested; skipping native ORT optimization so "
            "CastFloat16Transformer cannot promote attention to FP32."
        )
    return metadata, preserve_fp16_attention


def validate_source_package(metadata):
    model_files = {
        key: SOURCE_FOLDER / value
        for key, value in metadata.items()
        if key.startswith("model_file_name_")
    }
    missing = [
        f"{key} -> {path.name}"
        for key, path in sorted(model_files.items())
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "QwenTTS_ONNX is incomplete; metadata declares missing runtime graph(s):\n  "
            + "\n  ".join(missing)
            + "\nRe-run Export_QwenTTS.py after updating Shared_Weights.py."
        )


def available_strategy_graph_names() -> tuple[str, ...]:
    available = tuple(
        name
        for name in STRATEGY_GRAPH_NAMES
        if (SOURCE_FOLDER / f"{name}.onnx").is_file()
    )
    missing = tuple(name for name in STRATEGY_GRAPH_NAMES if name not in available)
    if missing:
        print(
            "[Graph selection] Skipping unexported strategy graphs: "
            f"{', '.join(missing)}."
        )
    return available


def resolve_plans(preserve_fp16_attention):
    resolved_plans = {}
    for name, plan in MODEL_PLANS.items():
        resolved = resolve_plan(plan, CONFIG)
        if preserve_fp16_attention and name in STRATEGY_GRAPH_NAMES:
            resolved = replace(resolved, opt_level=0)
        resolved_plans[name] = resolved
    return resolved_plans


def weight_quantization_signature(plan):
    def selector_signature(selector):
        return tuple(selector) if isinstance(selector, list) else selector

    return (
        plan.method,
        plan.algo,
        plan.op_types,
        plan.axes,
        plan.block_size,
        plan.accuracy_level,
        plan.symmetric,
        plan.quant_format,
        selector_signature(plan.nodes_to_exclude),
        selector_signature(plan.nodes_to_include),
    )


def shared_weight_plan(resolved_plans, strategy_graph_names):
    if not strategy_graph_names:
        print("[Shared quantization] No strategy graphs were exported.")
        return None
    template_name = (
        QUANTIZATION_TEMPLATE
        if QUANTIZATION_TEMPLATE in strategy_graph_names
        else strategy_graph_names[0]
    )
    template_plan = resolved_plans[template_name]
    if template_plan.method not in WEIGHT_ONLY_BITS:
        print(
            f"[Shared quantization] {template_name} uses "
            f"{template_plan.method}; processing strategy graphs independently."
        )
        return None
    template_signature = weight_quantization_signature(template_plan)
    incompatible = [
        name
        for name in strategy_graph_names
        if weight_quantization_signature(resolved_plans[name]) != template_signature
    ]
    if incompatible:
        print(
            "[Shared quantization] Strategy graphs have independent plans; "
            f"processing them separately (different plans: {incompatible})."
        )
        return None
    return template_name, template_plan


def quantize_shared_transformer_weights(cache_path, resolved_plans, strategy_graph_names):
    shared_plan = shared_weight_plan(resolved_plans, strategy_graph_names)
    if shared_plan is None:
        return set()
    template_name, template_plan = shared_plan
    try:
        quant_stats = quantize_weight_only_shared(
            str(SOURCE_FOLDER / f"{template_name}.onnx"),
            [
                (
                    str(SOURCE_FOLDER / f"{name}.onnx"),
                    str(OUTPUT_FOLDER / f"{name}.onnx"),
                )
                for name in strategy_graph_names
            ],
            str(cache_path),
            template_plan,
            bits=WEIGHT_ONLY_BITS[template_plan.method],
            external=True,
        )
    except Exception as error:
        for name in strategy_graph_names:
            output_path = OUTPUT_FOLDER / f"{name}.onnx"
            output_path.unlink(missing_ok=True)
            output_path.with_name(output_path.name + ".data").unlink(missing_ok=True)
        print(
            "[Shared quantization] Shared packing was not applicable; "
            f"processing strategy graphs independently ({error})."
        )
        return set()
    print(
        f"[Shared quantization] Packed {quant_stats['unique_weights']} unique weights once; "
        f"reused them across {quant_stats['total_rewrites']} weight-only nodes in "
        f"{quant_stats['graph_count']} graphs."
    )
    return set(strategy_graph_names)


def stage_external_data_dependencies(model_path, source_folder, stage_folder):
    model = onnx.load(str(model_path), load_external_data=False)
    locations = {
        {entry.key: entry.value for entry in tensor.external_data}.get("location")
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
    }
    for location in locations - {None}:
        source = source_folder / location
        staged = stage_folder / location
        if staged.exists() or not source.is_file():
            continue
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staged)


def process_graph_in_staging(name, plan, mixed_precision, prequantized=False):
    stage_folder = OUTPUT_FOLDER / f".{name}.stage"
    if stage_folder.exists():
        shutil.rmtree(stage_folder)
    stage_folder.mkdir()

    final_path = OUTPUT_FOLDER / f"{name}.onnx"
    final_data_path = final_path.with_name(final_path.name + ".data")
    staged_path = stage_folder / final_path.name
    staged_data_path = staged_path.with_name(staged_path.name + ".data")
    staged_config = replace(CONFIG, optimized_folder_path=str(stage_folder))

    try:
        if prequantized:
            if not final_path.is_file():
                raise FileNotFoundError(
                    f"Shared quantization did not produce {final_path}."
                )
            final_path.replace(staged_path)
            if final_data_path.is_file():
                final_data_path.replace(staged_data_path)
            stage_external_data_dependencies(
                staged_path,
                final_path.parent,
                stage_folder,
            )

        process_model(
            name,
            plan,
            staged_config,
            mixed_precision=mixed_precision,
            prequantized=prequantized,
        )
        if not staged_path.is_file():
            raise FileNotFoundError(
                f"Optimization did not produce expected graph: {staged_path}."
            )

        final_path.unlink(missing_ok=True)
        final_data_path.unlink(missing_ok=True)
        staged_path.replace(final_path)
        if staged_data_path.is_file():
            staged_data_path.replace(final_data_path)
    finally:
        shutil.rmtree(stage_folder, ignore_errors=True)


def process_graphs(
    resolved_plans,
    prequantized_graphs,
    preserve_fp16_attention,
    strategy_graph_names,
):
    processed_names = (*PROCESS_SUPPORT_GRAPH_NAMES, *strategy_graph_names)
    mixed_precision = uses_mixed_precision(
        resolved_plans[name] for name in processed_names
    )
    if mixed_precision and CONFIG.f16_keep_io_types is None:
        print(
            "[Precision] Not all graphs use F16; enabling keep_io_types for "
            "float16 conversions."
        )
    for name in PROCESS_SUPPORT_GRAPH_NAMES:
        plan = resolved_plans[name]
        print(f"\n{'=' * 60}\nOptimizing support: {name}  [{plan.method}]\n{'=' * 60}")
        process_graph_in_staging(name, plan, mixed_precision)
    for name in strategy_graph_names:
        plan = resolved_plans[name]
        shared = name in prequantized_graphs
        shared_note = ", shared weights" if shared else ""
        print(f"\n{'=' * 60}\nOptimizing: {name}  [{plan.method}{shared_note}]\n{'=' * 60}")
        process_graph_in_staging(
            name,
            plan,
            mixed_precision,
            prequantized=shared,
        )

    expected_names = (*PROCESS_SUPPORT_GRAPH_NAMES, *strategy_graph_names)
    missing = [
        str(OUTPUT_FOLDER / f"{name}.onnx")
        for name in expected_names
        if not (OUTPUT_FOLDER / f"{name}.onnx").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Optimization did not retain required graph(s):\n  " + "\n  ".join(missing)
        )


def rebuild_shared_bundle(metadata, strategy_graph_names):
    model_paths = (
        [SOURCE_FOLDER / f"{name}.onnx" for name in PASSTHROUGH_GRAPH_NAMES]
        + [OUTPUT_FOLDER / f"{name}.onnx" for name in PROCESS_SUPPORT_GRAPH_NAMES]
        + [OUTPUT_FOLDER / f"{name}.onnx" for name in strategy_graph_names]
    )
    stats = bundle_shared_initializers(
        OUTPUT_FOLDER,
        model_paths=model_paths,
        metadata=metadata,
    )
    replace_onnx_metadata(
        str(OUTPUT_FOLDER / "QwenTTS_Metadata.onnx"),
        metadata,
    )
    print(
        f"[Shared weights] {stats['initializer_references']} references -> "
        f"{stats['unique_initializers']} unique tensors; "
        f"deduplicated {stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB."
    )


def _portable_embedding_gather_count(path, plan):
    model = onnx.load(str(path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    included = (
        plan.nodes_to_include(str(path))
        if callable(plan.nodes_to_include)
        else plan.nodes_to_include
    )
    excluded = (
        plan.nodes_to_exclude(str(path))
        if callable(plan.nodes_to_exclude)
        else plan.nodes_to_exclude
    )
    included = None if included is None else set(included)
    excluded = set(excluded or ())
    count = 0
    for node in model.graph.node:
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        if (included is not None and node.name not in included) or node.name in excluded:
            continue
        weight = initializers.get(node.input[0])
        if weight is None or len(weight.dims) != 2:
            continue
        if weight.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
            continue
        gather_axis = next(
            (
                int(onnx.helper.get_attribute_value(attribute))
                for attribute in node.attribute
                if attribute.name == "axis"
            ),
            0,
        )
        if gather_axis == 0 and int(weight.dims[1]) % BLOCK_SIZE == 0:
            count += 1
    return count


def validate_optimized_package(metadata, strategy_graph_names):
    declared_files = {
        value
        for key, value in metadata.items()
        if key.startswith("model_file_name_")
    }
    missing = sorted(
        file_name
        for file_name in declared_files
        if not (OUTPUT_FOLDER / file_name).is_file()
    )
    if missing:
        raise FileNotFoundError(
            "Optimized QwenTTS package is missing metadata-declared graph(s):\n  "
            + "\n  ".join(missing)
        )

    quantized_graph_names = (*PROCESS_SUPPORT_GRAPH_NAMES, *strategy_graph_names)
    for name in quantized_graph_names:
        if "Gather" not in MODEL_PLANS[name].op_types:
            continue
        source_path = SOURCE_FOLDER / f"{name}.onnx"
        output_path = OUTPUT_FOLDER / f"{name}.onnx"
        expected_gathers = _portable_embedding_gather_count(
            source_path,
            MODEL_PLANS[name],
        )
        if not expected_gathers:
            continue
        output = onnx.load(str(output_path), load_external_data=False)
        actual_gathers = sum(
            node.op_type == "GatherBlockQuantized"
            for node in output.graph.node
        )
        if actual_gathers != expected_gathers:
            raise RuntimeError(
                f"{output_path.name} quantized {actual_gathers}/{expected_gathers} "
                "portable embedding Gather node(s)."
            )


def promote_output_folder(staged_output_folder, final_output_folder):
    backup_folder = final_output_folder.with_name(final_output_folder.name + ".previous")
    shutil.rmtree(backup_folder, ignore_errors=True)
    if final_output_folder.exists():
        os.replace(final_output_folder, backup_folder)
    try:
        os.replace(staged_output_folder, final_output_folder)
    except Exception:
        if backup_folder.exists() and not final_output_folder.exists():
            os.replace(backup_folder, final_output_folder)
        raise
    shutil.rmtree(backup_folder, ignore_errors=True)


def main():
    global OUTPUT_FOLDER, CONFIG

    args = parse_args()
    metadata, preserve_fp16_attention = configure_attention_precision()
    validate_source_package(metadata)
    strategy_graph_names = available_strategy_graph_names()
    resolved_plans = resolve_plans(False)
    shared_weight_plan(resolved_plans, strategy_graph_names)
    resolved_plans = resolve_plans(preserve_fp16_attention)
    final_output_folder = OUTPUT_FOLDER
    staged_output_folder = final_output_folder.with_name(final_output_folder.name + ".staging")
    shutil.rmtree(staged_output_folder, ignore_errors=True)
    staged_output_folder.mkdir(parents=True)
    original_output_folder = OUTPUT_FOLDER
    original_config = CONFIG
    OUTPUT_FOLDER = staged_output_folder
    CONFIG = replace(CONFIG, optimized_folder_path=str(staged_output_folder))

    cache_path = staged_output_folder / QUANTIZATION_CACHE_NAME
    cache_artifacts = (cache_path, Path(str(cache_path) + ".data"))
    try:
        prequantized_graphs = quantize_shared_transformer_weights(
            cache_path,
            resolved_plans,
            strategy_graph_names,
        )

        process_graphs(
            resolved_plans,
            prequantized_graphs,
            preserve_fp16_attention,
            strategy_graph_names,
        )
        rebuild_shared_bundle(metadata, strategy_graph_names)
        for artifact in cache_artifacts:
            artifact.unlink(missing_ok=True)
        validate_optimized_package(metadata, strategy_graph_names)
        promote_output_folder(staged_output_folder, final_output_folder)
    finally:
        for artifact in cache_artifacts:
            artifact.unlink(missing_ok=True)
        OUTPUT_FOLDER = original_output_folder
        CONFIG = original_config
        shutil.rmtree(staged_output_folder, ignore_errors=True)


if __name__ == "__main__":
    main()