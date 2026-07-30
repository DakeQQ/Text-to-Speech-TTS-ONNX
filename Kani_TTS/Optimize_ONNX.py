"""Quantize compact KaniTTS graphs and rebuild one shared bundle.

Each graph has an independent top-level plan. Compatible strategy graphs may
reuse shared weight packing as an optional fast path.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (candidate / "Optimize_ONNX_Common.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    pass
from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    process_model,
    quantize_dynamic_int8_shared,
    quantize_weight_only_shared,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
)
from Shared_Weights import bundle_shared_initializers  # noqa: E402


SOURCE_FOLDER = SCRIPT_DIR / "KaniTTS_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "KaniTTS_Optimized"
QUANTIZATION_TEMPLATE = "KaniTTS_DecodeStep_greedy"
QUANTIZATION_CACHE_NAME = ".KaniTTS_QuantizedWeights.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

DYNAMIC_WEIGHT_TYPE = "QInt8"    # QInt8 (Int8) | QUInt8 (Uint8)
DYNAMIC_PER_CHANNEL = True
MATMUL_ALGORITHM = "k_quant"
BLOCK_SIZE = 32
MAIN_NUM_HEADS = 16
MAIN_HIDDEN_SIZE = 1024
ACCURACY_LEVEL = 4              # 0=default. 1=fp32, 2=fp16, 3=bf16, 4=int8

MODEL_PLANS: dict[str, Plan] = {
    "KaniTTS_MainPrefill_greedy": Plan(
        method="Q4",
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
        first_slim_no_shape_infer=True,
    ),
    "KaniTTS_MainPrefill_penalty_greedy": Plan(
        method="Q4",
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
        first_slim_no_shape_infer=True,
    ),
    "KaniTTS_MainPrefill_sampling": Plan(
        method="Q4",
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
        first_slim_no_shape_infer=True,
    ),
    "KaniTTS_DecodeStep_greedy": Plan(
        method="Q4",
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
        first_slim_no_shape_infer=True,
    ),
    "KaniTTS_DecodeStep_penalty_greedy": Plan(
        method="Q4",
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
        first_slim_no_shape_infer=True,
    ),
    "KaniTTS_DecodeStep_sampling": Plan(
        method="Q4",
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
        first_slim_no_shape_infer=True,
    ),
    "KaniTTS_Codec": Plan(
        method="F32",
        optimize=True,
        transformer=False,
        external=True,
        first_slim_no_shape_infer=True,
    ),
    "KaniTTS_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}

STRATEGIES = ("greedy", "penalty_greedy", "sampling")
STRATEGY_GRAPH_NAMES = tuple(
    f"KaniTTS_{stage}_{strategy}"
    for stage in ("MainPrefill", "DecodeStep")
    for strategy in STRATEGIES
)
SUPPORT_GRAPH_NAMES = ("KaniTTS_Codec", "KaniTTS_Metadata")

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
    return parser.parse_args()


def configure_attention_precision():
    metadata = read_onnx_metadata(str(SOURCE_FOLDER / "KaniTTS_Metadata.onnx"))
    flags = {key: metadata.get(key) for key in ("use_float16_kv", "compute_in_f32")}
    invalid = {key: value for key, value in flags.items() if value not in {"0", "1"}}
    preserve = flags["use_float16_kv"] == "1" and flags["compute_in_f32"] == "0"
    if preserve:
        print(
            "[Precision] FP16 KV attention compute requested; native ORT optimization "
            "is disabled to prevent attention promotion to FP32."
        )
    return metadata, preserve


def resolve_initializer_alias(name, aliases):
    seen = set()
    while name in aliases:
        seen.add(name)
        name = aliases[name]
    return name


def collect_constant_weight_signatures(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    initializer_names = {tensor.name for tensor in model.graph.initializer}
    aliases = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1
    }
    signatures = set()
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            weight_name = resolve_initializer_alias(node.input[1], aliases)
        elif node.op_type == "Gather" and node.input:
            weight_name = resolve_initializer_alias(node.input[0], aliases)
        else:
            continue
        if weight_name in initializer_names:
            signatures.add((node.op_type, weight_name))
    return signatures


def prove_covering_graph():
    template_path = SOURCE_FOLDER / f"{QUANTIZATION_TEMPLATE}.onnx"
    template_weights = collect_constant_weight_signatures(template_path)
    union = set()
    for name in STRATEGY_GRAPH_NAMES:
        graph_weights = collect_constant_weight_signatures(SOURCE_FOLDER / f"{name}.onnx")
        union.update(graph_weights)
        missing = sorted(graph_weights - template_weights)
    if union != template_weights:
        extra = sorted(template_weights - union)
        pass
    print(
        f"[Coverage] {QUANTIZATION_TEMPLATE} covers all {len(union)} unique MatMul/Gather "
        f"weights across {len(STRATEGY_GRAPH_NAMES)} strategy graphs."
    )


def resolve_plans(preserve_fp16_attention):
    resolved_plans = {}
    for name, plan in MODEL_PLANS.items():
        resolved = resolve_plan(plan, CONFIG)
        if preserve_fp16_attention and name in STRATEGY_GRAPH_NAMES:
            resolved = replace(resolved, opt_level=0)
        resolved_plans[name] = resolved
    return resolved_plans


def weight_quantization_signature(plan):
    return (
        plan.method,
        plan.algo,
        plan.op_types,
        plan.axes,
        plan.block_size,
        plan.accuracy_level,
        plan.symmetric,
        plan.quant_format,
        plan.dynamic_weight_type,
        plan.per_channel,
        plan.reduce_range,
        plan.default_tensor_type,
        plan.nodes_to_exclude,
        plan.nodes_to_include,
    )


def shared_weight_plan(resolved_plans):
    template_plan = resolved_plans[QUANTIZATION_TEMPLATE]
    if template_plan.method != "DYNAMIC" and template_plan.method not in WEIGHT_ONLY_BITS:
        print(
            f"[Shared quantization] {QUANTIZATION_TEMPLATE} uses "
            f"{template_plan.method}; processing strategy graphs independently."
        )
        return None
    template_signature = weight_quantization_signature(template_plan)
    incompatible = [
        name
        for name in STRATEGY_GRAPH_NAMES
        if weight_quantization_signature(resolved_plans[name]) != template_signature
    ]
    if incompatible:
        print(
            "[Shared quantization] Strategy graphs have independent plans; "
            f"processing them separately (different plans: {incompatible})."
        )
        return None
    return template_plan


def quantize_shared_strategy_weights(resolved_plans, cache_path):
    template_plan = shared_weight_plan(resolved_plans)
    if template_plan is None:
        return set(), None
    model_paths = [
        (
            str(SOURCE_FOLDER / f"{name}.onnx"),
            str(OUTPUT_FOLDER / f"{name}.onnx"),
        )
        for name in STRATEGY_GRAPH_NAMES
    ]
    try:
        prove_covering_graph()
        if template_plan.method == "DYNAMIC":
            stats = quantize_dynamic_int8_shared(
                str(SOURCE_FOLDER / f"{QUANTIZATION_TEMPLATE}.onnx"),
                model_paths,
                str(cache_path),
                template_plan,
                external=True,
            )
        else:
            stats = quantize_weight_only_shared(
                str(SOURCE_FOLDER / f"{QUANTIZATION_TEMPLATE}.onnx"),
                model_paths,
                str(cache_path),
                template_plan,
                bits=WEIGHT_ONLY_BITS[template_plan.method],
                external=True,
            )
    except Exception as error:
        print(
            "[Shared quantization] Shared packing was not applicable; "
            f"processing strategy graphs independently ({error})."
        )
        return set(), None
    print(
        f"[Shared quantization] Quantized {stats['unique_weights']} unique weights once; "
        f"reused them at {stats['total_rewrites']} sites across {stats['graph_count']} graphs."
    )
    return set(STRATEGY_GRAPH_NAMES), stats


def process_graphs(resolved_plans, prequantized_graphs, preserve_fp16_attention):
    mixed_precision = uses_mixed_precision(resolved_plans.values())
    if mixed_precision and CONFIG.f16_keep_io_types is None:
        print(
            "[Precision] Not all graphs use F16; enabling keep_io_types for "
            "float16 conversions."
        )
    for name in SUPPORT_GRAPH_NAMES:
        plan = resolved_plans[name]
        print(f"\nOptimizing support graph: {name} [{plan.method}]")
        process_model(name, plan, CONFIG, mixed_precision=mixed_precision)
    for name in STRATEGY_GRAPH_NAMES:
        plan = resolved_plans[name]
        shared = name in prequantized_graphs
        shared_note = ", shared weights" if shared else ""
        print(f"\nOptimizing strategy graph: {name} [{plan.method}{shared_note}]")
        process_model(
            name,
            plan,
            CONFIG,
            mixed_precision=mixed_precision,
            prequantized=shared,
        )
def rebuild_shared_bundle(metadata, cache_path):
    model_paths = [OUTPUT_FOLDER / f"{name}.onnx" for name in MODEL_PLANS]
    stats = bundle_shared_initializers(
        OUTPUT_FOLDER,
        model_paths=model_paths,
        metadata=metadata,
    )
    cache_path.unlink(missing_ok=True)
    Path(str(cache_path) + ".data").unlink(missing_ok=True)
    replace_onnx_metadata(
        str(OUTPUT_FOLDER / "KaniTTS_Metadata.onnx"),
        metadata,
    )
    print(
        f"[Shared bundle] {stats['initializer_references']} references -> "
        f"{stats['unique_initializers']} tensors."
    )
    return stats


def report_package(shared_quantization_stats):
    expected_files = {
        *(f"{name}.onnx" for name in MODEL_PLANS),
        "KaniTTS_SharedInitializers.onnx",
        "KaniTTS_SharedInitializers.onnx.data",
    }
    actual_files = {path.name for path in OUTPUT_FOLDER.iterdir() if path.is_file()}
    total_bytes = sum(path.stat().st_size for path in OUTPUT_FOLDER.iterdir() if path.is_file())
    print(f"[Package] {len(expected_files)} files, {total_bytes / (1024 * 1024):.2f} MiB total.")
    for name in MODEL_PLANS:
        path = OUTPUT_FOLDER / f"{name}.onnx"
        model = onnx.load(str(path), load_external_data=False)
        operators = Counter(node.op_type for node in model.graph.node)
        quantized_nodes = (
            operators["MatMulNBits"]
            + operators["GatherBlockQuantized"]
            + operators["MatMulInteger"]
            + operators["DequantizeLinear"]
        )
        print(
            f"  {path.name}: nodes={len(model.graph.node)}, quantized_ops={quantized_nodes}, "
            f"graph={path.stat().st_size / (1024 * 1024):.2f} MiB"
        )
def main():
    args = parse_args()
    resolved_plans = resolve_plans(False)
    shared_weight_plan(resolved_plans)
    metadata, preserve_fp16_attention = configure_attention_precision()
    resolved_plans = resolve_plans(preserve_fp16_attention)
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True)

    cache_path = OUTPUT_FOLDER / QUANTIZATION_CACHE_NAME
    try:
        prequantized_graphs, shared_stats = quantize_shared_strategy_weights(
            resolved_plans,
            cache_path,
        )
        process_graphs(resolved_plans, prequantized_graphs, preserve_fp16_attention)
        rebuild_shared_bundle(metadata, cache_path)
        report_package(shared_stats)
    finally:
        cache_path.unlink(missing_ok=True)
        Path(str(cache_path) + ".data").unlink(missing_ok=True)


if __name__ == "__main__":
    main()