"""Quantize and optimize Qwen3-TTS graphs, then rebuild one shared bundle.

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
    pass
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


MODEL_TYPE = "0.6B-Base"
STRATEGIES = ("greedy", "penalty_greedy", "sampling")
SOURCE_FOLDER = SCRIPT_DIR / "QwenTTS_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "QwenTTS_Optimized"
QUANTIZATION_TEMPLATE = "QwenTTS_DecodeStep_greedy"
QUANTIZATION_CACHE_NAME = ".QwenTTS_QuantizedWeights.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

MATMUL_ALGORITHM = "DEFAULT"
BLOCK_SIZE = 32
ACCURACY_LEVEL = 4
MAIN_NUM_HEADS = 16
MAIN_HIDDEN_SIZE = 1024 if "0.6B" in MODEL_TYPE else 2048

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


def shared_weight_plan(resolved_plans):
    template_plan = resolved_plans[QUANTIZATION_TEMPLATE]
    if template_plan.method not in WEIGHT_ONLY_BITS:
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


def quantize_shared_transformer_weights(cache_path, resolved_plans):
    template_plan = shared_weight_plan(resolved_plans)
    if template_plan is None:
        return set()
    try:
        quant_stats = quantize_weight_only_shared(
            str(SOURCE_FOLDER / f"{QUANTIZATION_TEMPLATE}.onnx"),
            [
                (
                    str(SOURCE_FOLDER / f"{name}.onnx"),
                    str(OUTPUT_FOLDER / f"{name}.onnx"),
                )
                for name in STRATEGY_GRAPH_NAMES
            ],
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
        f"[Shared quantization] Packed {quant_stats['unique_weights']} unique weights once; "
        f"reused them across {quant_stats['total_rewrites']} weight-only nodes in "
        f"{quant_stats['graph_count']} graphs."
    )
    return set(STRATEGY_GRAPH_NAMES)


def process_graphs(
    resolved_plans,
    prequantized_graphs,
    preserve_fp16_attention,
):
    processed_names = (*PROCESS_SUPPORT_GRAPH_NAMES, *STRATEGY_GRAPH_NAMES)
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
        process_model(name, plan, CONFIG, mixed_precision=mixed_precision)
    for name in STRATEGY_GRAPH_NAMES:
        plan = resolved_plans[name]
        shared = name in prequantized_graphs
        shared_note = ", shared weights" if shared else ""
        print(f"\n{'=' * 60}\nOptimizing: {name}  [{plan.method}{shared_note}]\n{'=' * 60}")
        process_model(
            name,
            plan,
            CONFIG,
            mixed_precision=mixed_precision,
            prequantized=shared,
        )
def rebuild_shared_bundle(metadata):
    model_paths = (
        [SOURCE_FOLDER / f"{name}.onnx" for name in PASSTHROUGH_GRAPH_NAMES]
        + [OUTPUT_FOLDER / f"{name}.onnx" for name in PROCESS_SUPPORT_GRAPH_NAMES]
        + [OUTPUT_FOLDER / f"{name}.onnx" for name in STRATEGY_GRAPH_NAMES]
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
        prequantized_graphs = quantize_shared_transformer_weights(
            cache_path,
            resolved_plans,
        )

        process_graphs(
            resolved_plans,
            prequantized_graphs,
            preserve_fp16_attention,
        )
        rebuild_shared_bundle(metadata)
    finally:
        for artifact in cache_artifacts:
            artifact.unlink(missing_ok=True)


if __name__ == "__main__":
    main()