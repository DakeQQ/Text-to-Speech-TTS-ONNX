"""Quantize and optimize Breeze TTS 2 graphs, then rebuild one shared bundle.

Each graph has an independent top-level plan. Compatible strategy graphs may
reuse shared weight packing as an optional fast path.
"""

from __future__ import annotations

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
    _eliminate_initializer_identity_aliases,
    _materialize_constant_tensors_as_initializers,
    process_model,
    quantize_weight_only_shared,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
)
from Shared_Weights import (  # noqa: E402
    bundle_shared_initializers,
    copy_text_tokenizer,
)


# User configuration
STRATEGIES = ("greedy", "penalty_greedy", "sampling")
SOURCE_FOLDER = SCRIPT_DIR / "BreezeTTS_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "BreezeTTS_Optimized"
QUANTIZATION_TEMPLATE = "BreezeTTS_DecodeStep_greedy"
QUANTIZATION_CACHE_NAME = ".BreezeTTS_QuantizedWeights.onnx"
QUANTIZATION_COVER_NAME = ".BreezeTTS_QuantizationCover.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

# Quantization and optimization defaults
MATMUL_ALGORITHM = "DEFAULT"
BLOCK_SIZE = 64
ACCURACY_LEVEL = 4
MAIN_NUM_HEADS = 16
MAIN_HIDDEN_SIZE = 2048

# Per-graph quantization and optimization plan
MODEL_PLANS: dict[str, Plan] = {
    "BreezeTTS_ReferencePreprocess": Plan(
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
    "BreezeTTS_TargetPreprocess": Plan(
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
    "BreezeTTS_Decoder": Plan(
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
    "BreezeTTS_Decoder_Stream": Plan(
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
    "BreezeTTS_MainPrefill_greedy": Plan(
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
    "BreezeTTS_MainPrefill_penalty_greedy": Plan(
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
    "BreezeTTS_MainPrefill_sampling": Plan(
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
    "BreezeTTS_DecodeStep_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=False,
        external=True,
    ),
    "BreezeTTS_DecodeStep_penalty_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=False,
        external=True,
    ),
    "BreezeTTS_DecodeStep_sampling": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        optimize=True,
        transformer=False,
        external=True,
    ),
    "BreezeTTS_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}

STRATEGY_GRAPH_NAMES = tuple(
    f"BreezeTTS_{stage}_{strategy}"
    for stage in ("MainPrefill", "DecodeStep")
    for strategy in STRATEGIES
)
PROCESS_SUPPORT_GRAPH_NAMES = (
    "BreezeTTS_ReferencePreprocess",
    "BreezeTTS_TargetPreprocess",
    "BreezeTTS_Decoder",
    "BreezeTTS_Decoder_Stream",
)
PASSTHROUGH_GRAPH_NAMES = ("BreezeTTS_Metadata",)
PROCESS_GRAPH_NAMES = (*PROCESS_SUPPORT_GRAPH_NAMES, *STRATEGY_GRAPH_NAMES)


CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    block_size=BLOCK_SIZE,
    accuracy_level=ACCURACY_LEVEL,
    dynamic_default_tensor_type=onnx.TensorProto.FLOAT,
)


def configure_attention_precision():
    metadata = read_onnx_metadata(str(SOURCE_FOLDER / "BreezeTTS_Metadata.onnx"))
    precision_flags = {
        key: metadata[key] for key in ("use_f16_kv", "compute_in_f32")
    }
    preserve_fp16_attention = precision_flags["use_f16_kv"] == "1" and precision_flags["compute_in_f32"] == "0"
    if preserve_fp16_attention:
        print(
            "[Precision] FP16 KV attention requested; skipping native ORT optimization so "
            "CastFloat16Transformer cannot promote attention to FP32."
        )
    return metadata, preserve_fp16_attention


def available_strategy_graph_names() -> tuple[str, ...]:
    return STRATEGY_GRAPH_NAMES


def resolve_plans(preserve_fp16_attention):
    resolved_plans = {}
    for name, plan in MODEL_PLANS.items():
        resolved = resolve_plan(plan, CONFIG)
        if preserve_fp16_attention and name in STRATEGY_GRAPH_NAMES:
            resolved = replace(resolved, opt_level=0)
        resolved_plans[name] = resolved
    return resolved_plans


def graph_local_gather_nodes(source_path: str) -> list[str]:
    """Keep RoPE lookup tables in float while selecting canonical model embeddings."""
    model = onnx.load(source_path, load_external_data=False)
    excluded = [
        node.name
        for node in model.graph.node
        if (
            node.op_type == "Gather"
            and len(node.input) == 2
            and not node.input[0].startswith("breezetts_shared_")
        )
    ]
    del model
    return excluded


def build_quantization_cover(graph_names, cover_path):
    """Build one disconnected MatMul/Gather per unique selected source weight."""
    selected_weights = {}
    initializers = {}
    external_locations = set()
    ir_version = 10
    opset_version = 20

    for name in graph_names:
        source_path = SOURCE_FOLDER / f"{name}.onnx"
        model = onnx.load(source_path, load_external_data=False)
        _materialize_constant_tensors_as_initializers(model.graph)
        _eliminate_initializer_identity_aliases(model.graph)
        ir_version = max(ir_version, model.ir_version)
        opset_version = max(
            opset_version,
            *(opset.version for opset in model.opset_import if not opset.domain),
        )
        initializer_map = {
            initializer.name: initializer for initializer in model.graph.initializer
        }
        for node in model.graph.node:
            if node.op_type not in ("MatMul", "Gather") or len(node.input) != 2:
                continue
            weight_index = 1 if node.op_type == "MatMul" else 0
            weight = initializer_map.get(node.input[weight_index])
            if weight is None:
                continue
            if node.op_type == "Gather" and not weight.name.startswith(
                "breezetts_shared_"
            ):
                continue
            key = (node.op_type, weight.name)
            if key not in selected_weights:
                selected_weights[key] = weight
                initializers.setdefault(weight.name, weight)
                external_locations.update(
                    entry.value
                    for entry in weight.external_data
                    if entry.key == "location"
                )
        del model

    nodes = []
    inputs = []
    outputs = []
    for index, ((op_type, weight_name), weight) in enumerate(
        sorted(selected_weights.items())
    ):
        input_name = f"cover_input_{index:04d}"
        output_name = f"cover_output_{index:04d}"
        if op_type == "MatMul":
            inputs.append(
                onnx.helper.make_tensor_value_info(
                    input_name,
                    weight.data_type,
                    [1, int(weight.dims[0])],
                )
            )
            outputs.append(
                onnx.helper.make_tensor_value_info(
                    output_name,
                    weight.data_type,
                    [1, int(weight.dims[1])],
                )
            )
            nodes.append(
                onnx.helper.make_node(
                    "MatMul",
                    [input_name, weight_name],
                    [output_name],
                    name=f"cover_matmul_{index:04d}",
                )
            )
        else:
            inputs.append(
                onnx.helper.make_tensor_value_info(
                    input_name,
                    onnx.TensorProto.INT32,
                    [1],
                )
            )
            outputs.append(
                onnx.helper.make_tensor_value_info(
                    output_name,
                    weight.data_type,
                    [1, *[int(dimension) for dimension in weight.dims[1:]]],
                )
            )
            nodes.append(
                onnx.helper.make_node(
                    "Gather",
                    [weight_name, input_name],
                    [output_name],
                    name=f"cover_gather_{index:04d}",
                    axis=0,
                )
            )

    graph = onnx.helper.make_graph(
        nodes,
        "breeze_tts_quantization_cover",
        inputs,
        outputs,
        initializer=list(initializers.values()),
    )
    cover = onnx.helper.make_model(
        graph,
        producer_name=Path(__file__).name,
        opset_imports=[onnx.helper.make_opsetid("", opset_version)],
    )
    cover.ir_version = ir_version
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    for location in external_locations:
        source = (SOURCE_FOLDER / location).resolve()
        link = cover_path.parent / location
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.link(source, link)
    onnx.save(cover, cover_path)
    return len(selected_weights)


def quantize_all_shared_weights(cache_path, resolved_plans, graph_names):
    cover_path = SOURCE_FOLDER / QUANTIZATION_COVER_NAME
    cover_path.unlink(missing_ok=True)
    unique_weight_count = build_quantization_cover(graph_names, cover_path)
    template_plan = replace(
        resolved_plans[QUANTIZATION_TEMPLATE],
        nodes_to_exclude=graph_local_gather_nodes,
    )
    operation = "Refining" if template_plan.algo == "AFFINE_REFINE_V2" else "Packing"
    print(
        f"[Shared quantization] {operation} {unique_weight_count} unique operator/weight "
        f"pairs once, then replaying them into {len(graph_names)} graphs."
    )
    try:
        quant_stats = quantize_weight_only_shared(
            str(cover_path),
            [
                (
                    str(SOURCE_FOLDER / f"{name}.onnx"),
                    str(OUTPUT_FOLDER / f"{name}.onnx"),
                )
                for name in graph_names
            ],
            str(cache_path),
            template_plan,
            bits=WEIGHT_ONLY_BITS[template_plan.method],
            external=True,
        )
    finally:
        cover_path.unlink(missing_ok=True)
    print(
        f"[Shared quantization] Packed {quant_stats['unique_weights']} unique weights once; "
        f"reused them across {quant_stats['total_rewrites']} weight-only nodes in "
        f"{quant_stats['graph_count']} graphs."
    )
    return set(graph_names)


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
        shared = name in prequantized_graphs
        shared_note = ", shared weights" if shared else ""
        print(
            f"\n{'=' * 60}\nOptimizing support: {name}  "
            f"[{plan.method}{shared_note}]\n{'=' * 60}"
        )
        process_graph_in_staging(
            name,
            plan,
            mixed_precision,
            prequantized=shared,
        )
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
        str(OUTPUT_FOLDER / "BreezeTTS_Metadata.onnx"),
        metadata,
    )
    print(
        f"[Shared weights] {stats['initializer_references']} references -> "
        f"{stats['unique_initializers']} unique tensors; "
        f"deduplicated {stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB."
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

    metadata, preserve_fp16_attention = configure_attention_precision()
    strategy_graph_names = available_strategy_graph_names()
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
        prequantized_graphs = quantize_all_shared_weights(
            cache_path,
            resolved_plans,
            PROCESS_GRAPH_NAMES,
        )

        process_graphs(
            resolved_plans,
            prequantized_graphs,
            preserve_fp16_attention,
            strategy_graph_names,
        )
        rebuild_shared_bundle(metadata, strategy_graph_names)
        tokenizer_file_count = copy_text_tokenizer(
            SOURCE_FOLDER, staged_output_folder
        )
        print(f"[Tokenizer] Copied {tokenizer_file_count} text tokenizer files")
        for artifact in cache_artifacts:
            artifact.unlink(missing_ok=True)
        promote_output_folder(staged_output_folder, final_output_folder)
    finally:
        for artifact in cache_artifacts:
            artifact.unlink(missing_ok=True)
        OUTPUT_FOLDER = original_output_folder
        CONFIG = original_config
        shutil.rmtree(staged_output_folder, ignore_errors=True)


if __name__ == "__main__":
    main()