"""Quantize and optimize compact MOSS TTS graphs, then rebuild one shared bundle.

Each graph has an independent top-level plan. Compatible strategy graphs may
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
    process_model,
    quantize_dynamic_int8_shared,
    quantize_weight_only_shared,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
)
from Shared_Weights import bundle_shared_initializers  # noqa: E402


# User configuration
STRATEGIES = ("greedy", "penalty_greedy", "sampling")
SOURCE_FOLDER = SCRIPT_DIR / "MOSS_TTS_Nano_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "MOSS_TTS_Nano_Optimized"
QUANTIZATION_TEMPLATE = "MossTTSNano_DecodeStep_greedy"
QUANTIZATION_CACHE_NAME = ".MossTTSNano_QuantizedWeights.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

# Quantization and optimization defaults
MATMUL_ALGORITHM = "AFFINE_REFINE_V2"
BLOCK_SIZE = 32
ACCURACY_LEVEL = 4
DYNAMIC_WEIGHT_TYPE = "QInt8"  # QInt8 | QUInt8
DYNAMIC_PER_CHANNEL = True
MAIN_NUM_HEADS = 12
MAIN_HIDDEN_SIZE = 768
CODEC_NUM_HEADS = 4
CODEC_HIDDEN_SIZE = 256

# Per-graph quantization and optimization plan
MODEL_PLANS: dict[str, Plan] = {
    "MossTTSNano_MainPrefill_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        opt_level=2,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "MossTTSNano_MainPrefill_penalty_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        opt_level=2,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "MossTTSNano_MainPrefill_sampling": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        opt_level=2,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "MossTTSNano_DecodeStep_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        opt_level=2,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "MossTTSNano_DecodeStep_penalty_greedy": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        opt_level=2,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "MossTTSNano_DecodeStep_sampling": Plan(
        method="Q8",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        opt_level=2,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "MossAudioTokenizer_Encoder": Plan(
        method="F32",
        num_heads=CODEC_NUM_HEADS,
        hidden_size=CODEC_HIDDEN_SIZE,
        opt_level=1,
        transformer=False,
        external=True,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
    ),
    "MossAudioTokenizer_Decoder": Plan(
        method="F32",
        num_heads=CODEC_NUM_HEADS,
        hidden_size=CODEC_HIDDEN_SIZE,
        opt_level=1,
        transformer=False,
        external=True,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
    ),
    "MossTTSNano_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}

STRATEGY_GRAPH_NAMES = tuple(
    f"MossTTSNano_{stage}_{strategy}"
    for stage in ("MainPrefill", "DecodeStep")
    for strategy in STRATEGIES
)
CODEC_GRAPH_NAMES = ("MossAudioTokenizer_Encoder", "MossAudioTokenizer_Decoder")
PASSTHROUGH_GRAPH_NAMES = ("MossTTSNano_Metadata",)

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
    metadata = read_onnx_metadata(str(SOURCE_FOLDER / "MossTTSNano_Metadata.onnx"))
    flags = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
    invalid = {key: value for key, value in flags.items() if value not in {"0", "1"}}
    preserve = flags["use_f16_kv"] == "1" and flags["compute_in_f32"] == "0"
    if preserve:
        print(
            "[Precision] FP16 KV attention compute requested; native ORT optimization "
            "is disabled to prevent CastFloat16Transformer promotion."
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
        elif node.op_type == "Gather" and len(node.input) >= 1:
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
    per_graph = {}
    for name in STRATEGY_GRAPH_NAMES:
        path = SOURCE_FOLDER / f"{name}.onnx"
        weights = collect_constant_weight_signatures(path)
        per_graph[name] = weights
        union.update(weights)
        missing = sorted(weights - template_weights)
    if union != template_weights:
        extra = sorted(template_weights - union)
        pass
    print(
        f"[Coverage] {QUANTIZATION_TEMPLATE} covers all {len(union)} unique MatMul/Gather "
        f"weights across {len(per_graph)} strategy graphs."
    )
    return len(union)


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
        return set()
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
        return set()
    print(
        f"[Shared quantization] Quantized {stats['unique_weights']} unique weights once; "
        f"reused them at {stats['total_rewrites']} sites across {stats['graph_count']} graphs."
    )
    return set(STRATEGY_GRAPH_NAMES)


def process_graphs(
    resolved_plans,
    preserve_fp16_attention,
    prequantized_graphs,
):
    mixed_precision = uses_mixed_precision(
        resolved_plans[name]
        for name in (*CODEC_GRAPH_NAMES, *STRATEGY_GRAPH_NAMES)
    )
    if mixed_precision and CONFIG.f16_keep_io_types is None:
        print(
            "[Precision] Not all graphs use F16; enabling keep_io_types for "
            "float16 conversions."
        )
    for name in CODEC_GRAPH_NAMES:
        plan = resolved_plans[name]
        print(f"\nOptimizing float32 codec graph: {name}")
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
def rebuild_bundle(metadata, cache_path):
    model_paths = (
        [SOURCE_FOLDER / f"{name}.onnx" for name in PASSTHROUGH_GRAPH_NAMES]
        + [OUTPUT_FOLDER / f"{name}.onnx" for name in CODEC_GRAPH_NAMES]
        + [OUTPUT_FOLDER / f"{name}.onnx" for name in STRATEGY_GRAPH_NAMES]
    )
    stats = bundle_shared_initializers(
        OUTPUT_FOLDER,
        model_paths=model_paths,
        metadata=metadata,
        transient_sidecars=(
            (Path(str(cache_path) + ".data").name,)
            if cache_path is not None
            else ()
        ),
    )
    if cache_path is not None:
        cache_path.unlink(missing_ok=True)
        Path(str(cache_path) + ".data").unlink(missing_ok=True)
    replace_onnx_metadata(
        str(OUTPUT_FOLDER / "MossTTSNano_Metadata.onnx"),
        metadata,
    )
    print(
        f"[Shared bundle] {stats['initializer_references']} references -> "
        f"{stats['unique_initializers']} tensors."
    )
    return stats


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
    cache_artifacts = (cache_path, Path(str(cache_path) + ".data"))
    try:
        prequantized_graphs = quantize_shared_strategy_weights(
            resolved_plans,
            cache_path,
        )
        process_graphs(
            resolved_plans,
            preserve_fp16_attention,
            prequantized_graphs,
        )
        rebuild_bundle(metadata, cache_path)
    finally:
        for artifact in cache_artifacts:
            artifact.unlink(missing_ok=True)


if __name__ == "__main__":
    main()