"""Quantize and optimize every IndexTTS2 compute graph, then rebuild its package.

Each graph has an independent top-level plan. Compatible TTS autoregressive
graphs and emotion Qwen graphs may share weight-packing passes as an optional
fast path; every other plan is processed independently. The rebuilt bundle
deduplicates all identical packed tensors.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_TTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = INDEX_TTS_DIR.parent
for import_path in (REPO_ROOT, INDEX_TTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Index_TTS.v2.Shared_Weights import (  # noqa: E402
    audit_shared_bundle,
    bundle_shared_initializers,
)
from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    process_model,
    quantize_weight_only_shared,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
    validate_plan,
)


def exclude_non_matrix_weights(model_path: str) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    excluded = []
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            weight = initializers.get(node.input[1])
        elif node.op_type == "Gather" and node.input:
            weight = initializers.get(node.input[0])
        else:
            continue
        if weight is not None and len(weight.dims) != 2:
            excluded.append(node.name)
    del model
    gc.collect()
    return excluded


def matrix_nodes(model_path: str) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    selected = [
        node.name
        for node in model.graph.node
        if node.op_type in {"MatMul", "Gather"}
        and node.name
    ]
    del model
    gc.collect()
    return selected


def convolution_nodes(model_path: str) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    selected = [
        node.name
        for node in model.graph.node
        if node.op_type == "Conv" and node.name
    ]
    del model
    gc.collect()
    return selected


STRATEGIES = ("greedy", "penalty_greedy", "sampling")
SOURCE_FOLDER = SCRIPT_DIR / "IndexTTS2_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "IndexTTS2_Optimized"
QUANTIZATION_TEMPLATE = "IndexTTS2_TargetPrefill_greedy"
QUANTIZATION_CACHE_NAME = ".IndexTTS2_QuantizedWeights.onnx"
QUANTIZATION_COVER_NAME = ".IndexTTS2_QuantizationCover.onnx"
EMOTION_QUANTIZATION_TEMPLATE = "IndexTTS2_EmotionTextPrefill"
EMOTION_QUANTIZATION_CACHE_NAME = ".IndexTTS2_EmotionQuantizedWeights.onnx"
EMOTION_QUANTIZATION_COVER_NAME = ".IndexTTS2_EmotionQuantizationCover.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

MATMUL_ALGORITHM    = "k_quant"
BLOCK_SIZE          = 32
ACCURACY_LEVEL      = 4
MAIN_NUM_HEADS      = 20
MAIN_HIDDEN_SIZE    = 1280
EMOTION_NUM_HEADS   = 16
EMOTION_HIDDEN_SIZE = 1024
DYNAMIC_WEIGHT_TYPE = "QInt8"
DYNAMIC_PER_CHANNEL = False


MODEL_PLANS: dict[str, Plan] = {
    "IndexTTS2_ReferencePreprocess": Plan(
        method="F32",
        optimize=True,
        transformer=True,
        external=True,
    ),
    "IndexTTS2_Conditioning": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "IndexTTS2_EmotionTextPrefill": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=False,
        num_heads=EMOTION_NUM_HEADS,
        hidden_size=EMOTION_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_EmotionTextDecode": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=False,
        num_heads=EMOTION_NUM_HEADS,
        hidden_size=EMOTION_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_TargetPrefill_greedy": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_TargetPrefill_penalty_greedy": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_TargetPrefill_sampling": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_DecodeStep_greedy": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_DecodeStep_penalty_greedy": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_DecodeStep_sampling": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_Synthesis": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "IndexTTS2_CFMEstimator": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "IndexTTS2_Decoder": Plan(
        method="F32",
        optimize=True,
        transformer=True,
        external=True,
    ),
    # This manifest carrier has no weights or quantizable operators.
    "IndexTTS2_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}

STRATEGY_NAMES = tuple(
    f"IndexTTS2_{stage}_{strategy}"
    for stage in ("TargetPrefill", "DecodeStep")
    for strategy in STRATEGIES
)
SHARED_WEIGHT_GRAPH_NAMES = (*STRATEGY_NAMES, "IndexTTS2_Synthesis")
EMOTION_SHARED_WEIGHT_GRAPH_NAMES = (
    "IndexTTS2_EmotionTextPrefill",
    "IndexTTS2_EmotionTextDecode",
)
CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    block_size=BLOCK_SIZE,
    accuracy_level=ACCURACY_LEVEL,
    dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=DYNAMIC_PER_CHANNEL,
    dynamic_default_tensor_type=onnx.TensorProto.FLOAT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def configure_attention_precision() -> dict[str, str]:
    metadata = read_onnx_metadata(str(SOURCE_FOLDER / "IndexTTS2_Metadata.onnx"))
    if (
        metadata.get("graph_layout")
        != "raw_audio_emotion_text_merged_gpt_cached_cfm_step"
    ):
        raise RuntimeError(
            "IndexTTS2 raw_audio_emotion_text_merged_gpt_cached_cfm_step "
            "graphs are required."
        )
    flags = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
    invalid = {key: value for key, value in flags.items() if value not in {"0", "1"}}
    if invalid:
        raise RuntimeError(f"Invalid or missing IndexTTS2 precision metadata: {invalid}")
    if flags["use_f16_kv"] == "1" and flags["compute_in_f32"] == "0":
        print(
            "[Precision] FP16 KV attention is requested. Optimization plans remain "
            "unrestricted; validate any precision changes introduced by ORT."
        )
    return metadata


def resolve_initializer_alias(name: str, aliases: dict[str, str]) -> str:
    seen = set()
    while name in aliases:
        if name in seen:
            raise RuntimeError(f"Initializer Identity alias cycle at {name!r}.")
        seen.add(name)
        name = aliases[name]
    return name


def weight_quantization_signature(plan: Any) -> tuple[Any, ...]:
    def selector_signature(selector: Any) -> Any:
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


def shared_weight_plan(
    resolved_plans: dict[str, Any],
    graph_names: tuple[str, ...],
    template_name: str,
) -> Any | None:
    template_plan = resolved_plans[template_name]
    template_signature = weight_quantization_signature(template_plan)
    if template_plan.method not in WEIGHT_ONLY_BITS:
        print(
            f"[Shared quantization] {template_name} uses {template_plan.method}; "
            "processing this group independently."
        )
        return None
    incompatible = [
        name
        for name in graph_names
        if weight_quantization_signature(resolved_plans[name]) != template_signature
    ]
    if incompatible:
        print(
            "[Shared quantization] This group has independent quantization plans; "
            f"processing every graph separately (different plans: {incompatible})."
        )
        return None
    return template_plan


def collect_constant_weight_entries(
    model_path: Path,
    plan: Any,
) -> dict[tuple[str, str], tuple[Any, Any, Path]]:
    model = onnx.load(str(model_path), load_external_data=False)
    included = (
        plan.nodes_to_include(str(model_path))
        if callable(plan.nodes_to_include)
        else plan.nodes_to_include
    )
    excluded = (
        plan.nodes_to_exclude(str(model_path))
        if callable(plan.nodes_to_exclude)
        else plan.nodes_to_exclude
    )
    included = None if included is None else set(included)
    excluded = set(excluded or ())
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    aliases = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.op_type == "Identity" and len(node.input) == len(node.output) == 1
    }
    entries = {}
    for node in model.graph.node:
        if (
            node.op_type not in plan.op_types
            or (included is not None and node.name not in included)
            or node.name in excluded
        ):
            continue
        if node.op_type == "MatMul" and len(node.input) == 2:
            weight_name = resolve_initializer_alias(node.input[1], aliases)
        elif node.op_type == "Gather" and node.input:
            weight_name = resolve_initializer_alias(node.input[0], aliases)
        else:
            continue
        weight = initializers.get(weight_name)
        if weight is None or len(weight.dims) != 2:
            continue
        node_copy = onnx.NodeProto()
        node_copy.CopyFrom(node)
        weight_copy = onnx.TensorProto()
        weight_copy.CopyFrom(weight)
        entries.setdefault(
            (node.op_type, weight_name),
            (node_copy, weight_copy, model_path),
        )
    del model
    gc.collect()
    return entries


def collect_shared_weight_entries(
    resolved_plans: dict[str, Any],
    graph_names: tuple[str, ...],
) -> tuple[
    dict[tuple[str, str], tuple[Any, Any, Path]],
    dict[str, set[tuple[str, str]]],
]:
    all_entries = {}
    weight_sets = {}
    for name in graph_names:
        entries = collect_constant_weight_entries(
            SOURCE_FOLDER / f"{name}.onnx",
            resolved_plans[name],
        )
        weight_sets[name] = set(entries)
        for signature, entry in entries.items():
            prior = all_entries.get(signature)
            if prior is not None:
                prior_node, prior_weight, _ = prior
                node_attributes = [attr.SerializeToString() for attr in entry[0].attribute]
                prior_attributes = [attr.SerializeToString() for attr in prior_node.attribute]
                if (
                    entry[1].SerializeToString() != prior_weight.SerializeToString()
                    or node_attributes != prior_attributes
                ):
                    raise RuntimeError(
                        f"Shared weight {signature} differs across source graphs."
                    )
            else:
                all_entries[signature] = entry

    if not all(weight_sets.values()):
        raise RuntimeError("A shared quantization graph has no selected matrix weights.")
    shared_weights = set.intersection(*(weight_sets[name] for name in graph_names))
    if not shared_weights:
        raise RuntimeError("Shared quantization graphs have no common matrix weights.")
    print(
        f"[Coverage] One pass owns {len(all_entries)} unique matrix weights across "
        f"{len(graph_names)} graphs; {len(shared_weights)} weights are common to all."
    )
    return all_entries, weight_sets


def build_quantization_cover(
    resolved_plans: dict[str, Any],
    cover_path: Path,
    graph_names: tuple[str, ...],
    template_name: str,
) -> int:
    entries, _ = collect_shared_weight_entries(resolved_plans, graph_names)
    nodes = []
    inputs = []
    outputs = []
    initializers = {}

    for index, (signature, entry) in enumerate(sorted(entries.items())):
        op_type, weight_name = signature
        source_node, source_weight, source_path = entry
        weight = onnx.TensorProto()
        weight.CopyFrom(source_weight)
        for external_entry in weight.external_data:
            if external_entry.key == "location":
                source_data = (source_path.parent / external_entry.value).resolve()
                external_entry.value = os.path.relpath(source_data, cover_path.parent)
        prior_weight = initializers.get(weight_name)
        if prior_weight is not None and prior_weight.SerializeToString() != weight.SerializeToString():
            raise RuntimeError(f"Cover initializer collision for {weight_name!r}.")
        initializers.setdefault(weight_name, weight)

        input_name = f"quant_cover_input_{index}"
        output_name = f"quant_cover_output_{index}"
        if op_type == "MatMul":
            input_type = weight.data_type
            input_shape = [1, weight.dims[0]]
            output_shape = [1, weight.dims[1]]
            node_inputs = [input_name, weight_name]
        else:
            axis = next(
                (
                    onnx.helper.get_attribute_value(attribute)
                    for attribute in source_node.attribute
                    if attribute.name == "axis"
                ),
                0,
            )
            axis = axis if axis >= 0 else axis + len(weight.dims)
            if axis < 0 or axis >= len(weight.dims):
                raise RuntimeError(f"Invalid Gather axis {axis} for {weight_name!r}.")
            input_type = onnx.TensorProto.INT64
            input_shape = [1]
            output_shape = list(weight.dims)
            output_shape[axis : axis + 1] = [1]
            node_inputs = [weight_name, input_name]

        inputs.append(
            onnx.helper.make_tensor_value_info(input_name, input_type, input_shape)
        )
        outputs.append(
            onnx.helper.make_tensor_value_info(output_name, weight.data_type, output_shape)
        )
        cover_node = onnx.helper.make_node(
            op_type,
            node_inputs,
            [output_name],
            name=f"quant_cover/{index}/{op_type}",
        )
        for attribute in source_node.attribute:
            cover_node.attribute.add().CopyFrom(attribute)
        nodes.append(cover_node)

    template = onnx.load(
        str(SOURCE_FOLDER / f"{template_name}.onnx"),
        load_external_data=False,
    )
    graph = onnx.helper.make_graph(
        nodes,
        "IndexTTS2_QuantizationCover",
        inputs,
        outputs,
        initializer=list(initializers.values()),
    )
    cover = onnx.helper.make_model(
        graph,
        producer_name="IndexTTS2 shared quantization cover",
        opset_imports=[
            onnx.helper.make_opsetid(opset.domain, opset.version)
            for opset in template.opset_import
        ],
    )
    cover.ir_version = template.ir_version
    onnx.save(cover, str(cover_path))
    onnx.checker.check_model(str(cover_path))
    print(f"[Quantization cover] Built {len(entries)} unique operator/weight recipes.")
    del cover, graph, template
    gc.collect()
    return len(entries)


def resolve_plans() -> dict[str, Any]:
    resolved_plans = {}
    for name, plan in MODEL_PLANS.items():
        resolved = resolve_plan(plan, CONFIG)
        validate_plan(name, resolved)
        resolved_plans[name] = resolved
    return resolved_plans


def validate_sources() -> None:
    missing = [
        SOURCE_FOLDER / f"{name}.onnx"
        for name in MODEL_PLANS
        if not (SOURCE_FOLDER / f"{name}.onnx").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing IndexTTS2 graph(s): {missing}")
    for artifact in ("IndexTTS_SharedInitializers.onnx", "IndexTTS_SharedInitializers.onnx.data"):
        if not (SOURCE_FOLDER / artifact).is_file():
            raise FileNotFoundError(f"Missing IndexTTS2 shared artifact: {SOURCE_FOLDER / artifact}")


def quantize_shared_weights(
    resolved_plans: dict[str, Any],
    cache_path: Path,
    cover_path: Path,
    graph_names: tuple[str, ...],
    template_name: str,
) -> set[str]:
    template_plan = shared_weight_plan(
        resolved_plans,
        graph_names,
        template_name,
    )
    if template_plan is None:
        return set()
    try:
        unique_weights = build_quantization_cover(
            resolved_plans,
            cover_path,
            graph_names,
            template_name,
        )
        stats = quantize_weight_only_shared(
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
        if stats["unique_weights"] != unique_weights:
            raise RuntimeError(
                f"Shared quantizer produced {stats['unique_weights']} recipes; "
                f"expected {unique_weights}."
            )
    except Exception as error:
        print(
            "[Shared quantization] Shared packing was not applicable; "
            f"processing this group independently ({error})."
        )
        return set()
    print(
        f"[Shared quantization] Packed {stats['unique_weights']} weights once; "
        f"reused them at {stats['total_rewrites']} nodes across {stats['graph_count']} graphs."
    )
    return set(graph_names)


def validate_quantized_graph(name: str, plan: Any) -> None:
    if plan.method not in WEIGHT_ONLY_BITS and plan.method != "DYNAMIC":
        return
    model = onnx.load(str(OUTPUT_FOLDER / f"{name}.onnx"), load_external_data=False)
    quantized_ops = {
        op_type: sum(node.op_type == op_type for node in model.graph.node)
        for op_type in ("MatMulNBits", "GatherBlockQuantized", "MatMulInteger", "ConvInteger")
    }
    del model
    gc.collect()
    quantized_ops = {op_type: count for op_type, count in quantized_ops.items() if count}
    if not quantized_ops:
        print(
            f"  Quantization audit: {name} [{plan.method}] produced no quantized "
            "operators; keeping the processed graph."
        )
        return
    summary = ", ".join(f"{op_type}={count}" for op_type, count in quantized_ops.items())
    print(f"  Quantization audit: {summary}")


def process_graphs(
    resolved_plans: dict[str, Any],
    prequantized_graphs: set[str],
) -> None:
    mixed_precision = uses_mixed_precision(resolved_plans.values())
    if mixed_precision and CONFIG.f16_keep_io_types is None:
        print(
            "[Precision] Not all graphs use F16; enabling keep_io_types for "
            "float16 conversions."
        )
    for name, plan in resolved_plans.items():
        shared = name in prequantized_graphs
        detail = ", shared weights" if shared else ""
        print(f"\nProcessing graph: {name} [{plan.method}{detail}, optimize={plan.optimize}]")
        process_model(
            name,
            plan,
            CONFIG,
            mixed_precision=mixed_precision,
            prequantized=shared,
        )
        validate_quantized_graph(name, plan)


def rebuild_shared_bundle(
    metadata: dict[str, str],
    cache_path: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    model_paths = [OUTPUT_FOLDER / f"{name}.onnx" for name in MODEL_PLANS]
    stats = bundle_shared_initializers(
        OUTPUT_FOLDER,
        model_paths=model_paths,
        metadata=metadata,
    )
    cache_path.unlink(missing_ok=True)
    Path(str(cache_path) + ".data").unlink(missing_ok=True)
    audit = audit_shared_bundle(OUTPUT_FOLDER, model_paths)
    replace_onnx_metadata(
        str(OUTPUT_FOLDER / "IndexTTS2_Metadata.onnx"),
        metadata,
    )
    print(
        f"[Shared bundle] {stats['initializer_references']} references -> "
        f"{stats['unique_initializers']} tensors, "
        f"{audit['external_bytes'] / (1024**3):.2f} GiB shared data."
    )
    return stats, audit


def main() -> None:
    args = parse_args()
    resolved_plans = resolve_plans()
    shared_weight_plan(
        resolved_plans,
        SHARED_WEIGHT_GRAPH_NAMES,
        QUANTIZATION_TEMPLATE,
    )
    shared_weight_plan(
        resolved_plans,
        EMOTION_SHARED_WEIGHT_GRAPH_NAMES,
        EMOTION_QUANTIZATION_TEMPLATE,
    )
    if args.check_only:
        quantized_count = sum(
            plan.method in WEIGHT_ONLY_BITS or plan.method == "DYNAMIC"
            for plan in resolved_plans.values()
        )
        print(
            f"IndexTTS2 optimizer plan is valid: {quantized_count} quantized compute graphs, "
            f"{len(resolved_plans)} optimized graphs total, "
            "with independent per-graph quantization plans. Compatible plans may "
            "reuse optional shared packing passes."
        )
        return

    validate_sources()
    metadata = configure_attention_precision()
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True)
    cache_path = OUTPUT_FOLDER / QUANTIZATION_CACHE_NAME
    cover_path = SOURCE_FOLDER / QUANTIZATION_COVER_NAME
    emotion_cache_path = OUTPUT_FOLDER / EMOTION_QUANTIZATION_CACHE_NAME
    emotion_cover_path = SOURCE_FOLDER / EMOTION_QUANTIZATION_COVER_NAME
    try:
        prequantized_graphs = quantize_shared_weights(
            resolved_plans,
            cache_path,
            cover_path,
            SHARED_WEIGHT_GRAPH_NAMES,
            QUANTIZATION_TEMPLATE,
        )
        prequantized_graphs.update(
            quantize_shared_weights(
                resolved_plans,
                emotion_cache_path,
                emotion_cover_path,
                EMOTION_SHARED_WEIGHT_GRAPH_NAMES,
                EMOTION_QUANTIZATION_TEMPLATE,
            )
        )
        process_graphs(resolved_plans, prequantized_graphs)
        rebuild_shared_bundle(metadata, cache_path)
    finally:
        for temporary_path in (
            cache_path,
            cover_path,
            emotion_cache_path,
            emotion_cover_path,
        ):
            temporary_path.unlink(missing_ok=True)
            Path(str(temporary_path) + ".data").unlink(missing_ok=True)


if __name__ == "__main__":
    main()