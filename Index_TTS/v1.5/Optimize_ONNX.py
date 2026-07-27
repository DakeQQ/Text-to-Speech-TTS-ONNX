"""Optimize compact IndexTTS graphs and rebuild one shared-weight bundle.

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
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

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
    validate_plan,
)
from Shared_Weights import audit_shared_bundle, bundle_shared_initializers  # noqa: E402


STRATEGIES = ("greedy", "penalty_greedy", "sampling")
SOURCE_FOLDER = SCRIPT_DIR / "IndexTTS_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "IndexTTS_Optimized"
QUANTIZATION_TEMPLATE = "IndexTTS_DecodeStep_greedy"
QUANTIZATION_CACHE_NAME = ".IndexTTS_QuantizedWeights.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

DYNAMIC_WEIGHT_TYPE = "QInt8"  # QInt8 | QUInt8
DYNAMIC_PER_CHANNEL = False
MATMUL_ALGORITHM = "DEFAULT"
BLOCK_SIZE = 32
ACCURACY_LEVEL = 4
MAIN_NUM_HEADS = 8
MAIN_HIDDEN_SIZE = 1280


def exclude_non_matrix_weights(model_path):
    """Keep constant-weight MatMul/Gather nodes outside ORT packing unless their weight is rank 2."""
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
    return excluded


MODEL_PLANS: dict[str, Plan] = {
    "IndexTTS_ReferencePreprocess": Plan(
        method="F32",
        optimize=True,
        transformer=True,
        opt_level=2,
        external=True,
        first_slim_no_shape_infer=False,
        second_slim_no_shape_infer=False,
    ),
    "IndexTTS_TargetPreprocess": Plan(
        method="F32",
        optimize=True,
        transformer=True,
        opt_level=2,
        external=True,
        first_slim_no_shape_infer=False,
        second_slim_no_shape_infer=False,
    ),
    "IndexTTS_MainPrefill_greedy": Plan(
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
        opt_level=2,
    ),
    "IndexTTS_MainPrefill_penalty_greedy": Plan(
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
        opt_level=2,
    ),
    "IndexTTS_MainPrefill_sampling": Plan(
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
        opt_level=2,
    ),
    "IndexTTS_DecodeStep_greedy": Plan(
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
        opt_level=2,
    ),
    "IndexTTS_DecodeStep_penalty_greedy": Plan(
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
        opt_level=2,
    ),
    "IndexTTS_DecodeStep_sampling": Plan(
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
        opt_level=2,
    ),
    "IndexTTS_Decoder": Plan(
        method="F32",
        optimize=True,
        transformer=True,
        opt_level=2,
        external=True,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
    ),
    "IndexTTS_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=True,
        opt_level=2,
        external=True,
        first_slim_no_shape_infer=False,
        second_slim_no_shape_infer=False,
    ),
}

STRATEGY_GRAPH_NAMES = tuple(
    f"IndexTTS_{stage}_{strategy}"
    for stage in ("MainPrefill", "DecodeStep")
    for strategy in STRATEGIES
)
SUPPORT_GRAPH_NAMES = (
    "IndexTTS_ReferencePreprocess",
    "IndexTTS_TargetPreprocess",
    "IndexTTS_Decoder",
    "IndexTTS_Metadata",
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


def configure_attention_precision():
    metadata = read_onnx_metadata(str(SOURCE_FOLDER / "IndexTTS_Metadata.onnx"))
    if metadata.get("graph_layout") != "strategy_prefill_decode_step":
        raise RuntimeError("IndexTTS compact strategy graphs are required.")
    flags = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
    invalid = {key: value for key, value in flags.items() if value not in {"0", "1"}}
    if invalid:
        raise RuntimeError(f"Invalid or missing IndexTTS precision metadata: {invalid}")
    preserve = flags["use_f16_kv"] == "1" and flags["compute_in_f32"] == "0"
    if preserve:
        print(
            "[Precision] FP16 KV attention compute requested; native ORT optimization "
            "is disabled to prevent attention promotion to FP32."
        )
    return metadata, preserve


def validate_no_inserted_precision_casts(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    inserted = [
        node.name
        for node in model.graph.node
        if node.op_type == "Cast" and "InsertedPrecisionFreeCast_" in node.name
    ]
    if inserted:
        raise RuntimeError(
            f"{Path(model_path).name} contains {len(inserted)} unexpected precision-free casts."
        )


def resolve_initializer_alias(name, aliases):
    seen = set()
    while name in aliases:
        if name in seen:
            raise RuntimeError(f"Initializer Identity alias cycle at {name!r}.")
        seen.add(name)
        name = aliases[name]
    return name


def collect_constant_weight_signatures(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
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
        if weight_name in initializers and len(initializers[weight_name].dims) == 2:
            signatures.add((node.op_type, weight_name))
    return signatures


def prove_covering_graph():
    template_path = SOURCE_FOLDER / f"{QUANTIZATION_TEMPLATE}.onnx"
    template_weights = collect_constant_weight_signatures(template_path)
    if not template_weights:
        raise RuntimeError(f"Covering graph {template_path.name} has no constant MatMul/Gather weights.")

    union = set()
    for name in STRATEGY_GRAPH_NAMES:
        graph_weights = collect_constant_weight_signatures(SOURCE_FOLDER / f"{name}.onnx")
        union.update(graph_weights)
        missing = sorted(graph_weights - template_weights)
        if missing:
            raise RuntimeError(
                f"{name} uses {len(missing)} weights absent from {QUANTIZATION_TEMPLATE}: {missing[:8]}"
            )
    if union != template_weights:
        extra = sorted(template_weights - union)
        raise RuntimeError(f"Covering graph contains unexplained weight signatures: {extra[:8]}")
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
        validate_plan(name, resolved)
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


def validate_sources():
    missing = [
        SOURCE_FOLDER / f"{name}.onnx"
        for name in MODEL_PLANS
        if not (SOURCE_FOLDER / f"{name}.onnx").is_file()
    ]
    for artifact in (
        "IndexTTS_SharedInitializers.onnx",
        "IndexTTS_SharedInitializers.onnx.data",
    ):
        path = SOURCE_FOLDER / artifact
        if not path.is_file():
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing compact IndexTTS artifact(s): {missing}")


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
        if preserve_fp16_attention:
            validate_no_inserted_precision_casts(OUTPUT_FOLDER / f"{name}.onnx")


def rebuild_shared_bundle(metadata, cache_path):
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
        str(OUTPUT_FOLDER / "IndexTTS_Metadata.onnx"),
        metadata,
    )
    print(
        f"[Shared bundle] {stats['initializer_references']} references -> "
        f"{stats['unique_initializers']} tensors in one "
        f"{audit['external_bytes'] / (1024 * 1024):.2f} MiB blob."
    )
    return stats, audit


def main():
    args = parse_args()
    resolved_plans = resolve_plans(False)
    shared_weight_plan(resolved_plans)
    if args.check_only:
        quantized_count = sum(
            plan.method in {*WEIGHT_ONLY_BITS, "DYNAMIC"}
            for plan in resolved_plans.values()
        )
        print(
            f"IndexTTS optimizer plan is valid: {quantized_count} quantized graphs, "
            f"{len(resolved_plans)} graphs total."
        )
        return

    validate_sources()
    metadata, preserve_fp16_attention = configure_attention_precision()
    resolved_plans = resolve_plans(preserve_fp16_attention)
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True)

    cache_path = OUTPUT_FOLDER / QUANTIZATION_CACHE_NAME
    try:
        prequantized_graphs = quantize_shared_strategy_weights(
            resolved_plans,
            cache_path,
        )
        process_graphs(resolved_plans, prequantized_graphs, preserve_fp16_attention)
        rebuild_shared_bundle(metadata, cache_path)
    finally:
        cache_path.unlink(missing_ok=True)
        Path(str(cache_path) + ".data").unlink(missing_ok=True)


if __name__ == "__main__":
    main()