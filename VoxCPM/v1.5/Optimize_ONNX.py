"""Quantize and optimize compact VoxCPM graphs, then rebuild one shared bundle.

Each graph has an independent top-level plan. Compatible graph subsets may
reuse shared weight packing as an optional fast path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import onnx
import onnxruntime


SCRIPT_DIR = Path(__file__).resolve().parent
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
    quantize_weight_only_shared,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
    validate_plan,
)
from Shared_Weights import (  # noqa: E402
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    attach_shared_initializers,
    bundle_shared_initializers,
    check_model_allowing_runtime_extensions,
    validate_external_data_bounds,
)

SOURCE_FOLDER = SCRIPT_DIR / "VoxCPM_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "VoxCPM_Optimized"
METADATA_MODEL = "VoxCPM_Metadata"
CORE_TEMPLATE = "VoxCPM_DecodeStep"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}
QUANTIZABLE_PREFIXES = ("main/", "feat_encoder/")
SENSITIVE_PREFIXES = ("feat_decoder/", "reference_vae/")

MATMUL_ALGORITHM = "k_quant"
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
    parser.add_argument("--check-only", action="store_true")
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


def audit_recipe_coverage(source_folder):
    template_path = source_folder / f"{CORE_TEMPLATE}.onnx"
    template_keys = _candidate_keys(template_path, select_core_nodes, "MatMul")
    if not template_keys:
        raise RuntimeError("DecodeStep has no selected Main/FeatureEncoder weights.")
    coverage = {}
    for name in CORE_MODELS:
        path = source_folder / f"{name}.onnx"
        keys = _candidate_keys(path, select_core_nodes, "MatMul")
        missing = sorted(keys - template_keys)
        if missing:
            raise RuntimeError(
                f"{name} has {len(missing)} weights absent from {CORE_TEMPLATE}: {missing[:8]}"
            )
        coverage[name] = len(keys)
        selected = set(select_core_nodes(path))
        model = onnx.load(str(path), load_external_data=False)
        sensitive = sorted(
            node.name
            for node in model.graph.node
            if node.name in selected and node.name.startswith(SENSITIVE_PREFIXES)
        )
        if sensitive:
            raise RuntimeError(f"Quality-sensitive nodes entered the core plan: {sensitive[:8]}")
    text_path = source_folder / "VoxCPM_MainPrefill.onnx"
    text_keys = _candidate_keys(text_path, select_text_embedding, "Gather")
    if len(text_keys) != 1:
        raise RuntimeError(f"Expected one text embedding Gather, found {len(text_keys)}.")
    return {
        "core_template": CORE_TEMPLATE,
        "core_template_unique_weights": len(template_keys),
        "core_selected_nodes": coverage,
        "text_embedding_unique_weights": len(text_keys),
        "preserved_families": ["FeatureDecoder", "ReferenceVAE", "VAE_Decoder"],
    }


def configure_precision(source_folder):
    metadata = read_onnx_metadata(str(source_folder / f"{METADATA_MODEL}.onnx"))
    if metadata.get("graph_layout") != "compact_prefill_decode_v2":
        raise RuntimeError(
            "Optimizer requires the VoxCPM compact_prefill_decode_v2 package."
        )
    precision = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
    invalid = {key: value for key, value in precision.items() if value not in {"0", "1"}}
    if invalid:
        raise RuntimeError(f"Invalid compact precision metadata: {invalid}")
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
        validate_plan(name, plan)
        resolved_plans[name] = plan
    return resolved_plans


def validate_sources():
    missing = [
        SOURCE_FOLDER / f"{name}.onnx"
        for name in MODEL_PLANS
        if not (SOURCE_FOLDER / f"{name}.onnx").is_file()
    ]
    for artifact in (SHARED_MODEL_NAME, SHARED_DATA_NAME):
        path = SOURCE_FOLDER / artifact
        if not path.is_file():
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing compact VoxCPM artifact(s): {missing}")


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
        grouped_plans = {}
        for name in CORE_MODELS:
            plan = resolved_plans[name]
            if plan.method in WEIGHT_ONLY_BITS:
                grouped_plans.setdefault(_quantization_signature(plan), []).append(name)

        for pass_index, names in enumerate(grouped_plans.values(), start=1):
            if len(names) < 2 or CORE_TEMPLATE not in names:
                continue
            plan = resolved_plans[names[0]]
            bits = WEIGHT_ONLY_BITS[plan.method]
            cache_path = output_folder / (
                f".VoxCPM_core_{plan.method}_{pass_index}_QuantizedWeights.onnx"
            )
            transient_paths.extend((cache_path, Path(str(cache_path) + ".data")))
            print(
                f"[Shared quantization] core: {plan.method}, "
                f"block={plan.block_size}, graphs={names}."
            )
            try:
                pass_stats = quantize_weight_only_shared(
                    str(source_folder / f"{CORE_TEMPLATE}.onnx"),
                    [
                        (
                            str(source_folder / f"{name}.onnx"),
                            str(output_folder / f"{name}.onnx"),
                        )
                        for name in names
                    ],
                    str(cache_path),
                    plan,
                    bits=bits,
                    external=plan.external,
                )
            except Exception as error:
                print(
                    "[Shared quantization] Shared packing was not applicable; "
                    f"processing {names} independently ({error})."
                )
                continue
            prequantized.update(names)
            stats["shared_quantization"].append(
                {
                    "group": "core",
                    "graphs": names,
                    "method": plan.method,
                    "bits": bits,
                    **pass_stats,
                }
            )

    except Exception:
        for path in transient_paths:
            path.unlink(missing_ok=True)
        raise

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


def validate_no_inserted_precision_casts(model_path):
    model = onnx.load(str(model_path), load_external_data=False)
    inserted = [
        node.name
        for node in model.graph.node
        if node.op_type == "Cast" and "InsertedPrecisionFreeCast_" in node.name
    ]
    if inserted:
        raise RuntimeError(f"{model_path.name} has {len(inserted)} precision promotion casts.")


def audit_quantized_graphs(
    output_folder,
    resolved_plans,
    coverage,
    allow_fused_rms,
):
    report = {}
    for name in FUNCTIONAL_MODELS:
        plan = resolved_plans[name]
        model = onnx.load(str(output_folder / f"{name}.onnx"), load_external_data=False)
        histogram = {}
        for node in model.graph.node:
            histogram[node.op_type] = histogram.get(node.op_type, 0) + 1
        sensitive = [
            node.name
            for node in model.graph.node
            if node.op_type in {"MatMulNBits", "GatherBlockQuantized"}
            and node.name.startswith(SENSITIVE_PREFIXES)
        ]
        if sensitive:
            raise RuntimeError(f"{name} quantized quality-sensitive nodes: {sensitive[:8]}")
        if not allow_fused_rms:
            forbidden_rms = {
                "SimplifiedLayerNormalization",
                "SkipSimplifiedLayerNormalization",
            } & set(histogram)
            if forbidden_rms:
                raise RuntimeError(
                    f"{name} introduced metadata-forbidden fused RMS operators: "
                    f"{sorted(forbidden_rms)}"
                )
        report[name] = {
            "nodes": len(model.graph.node),
            "operator_histogram": dict(sorted(histogram.items())),
            "matmul_nbits": histogram.get("MatMulNBits", 0),
            "gather_block_quantized": histogram.get("GatherBlockQuantized", 0),
        }
        if name in CORE_MODELS and plan.method in WEIGHT_ONLY_BITS:
            expected = coverage["core_selected_nodes"][name]
            if report[name]["matmul_nbits"] < expected:
                raise RuntimeError(
                    f"{name} retained {report[name]['matmul_nbits']} of at least "
                    f"{expected} configured MatMul recipes."
                )
        expected_gathers = int(
            plan.method in WEIGHT_ONLY_BITS and "Gather" in plan.op_types
        )
        if report[name]["gather_block_quantized"] != expected_gathers:
            raise RuntimeError(
                f"{name} has {report[name]['gather_block_quantized']} quantized Gather "
                f"nodes; configured {expected_gathers}."
            )
    return report


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


def audit_final_package(output_folder, metadata, preserve_fp16_attention):
    expected = {
        *(f"{name}.onnx" for name in FUNCTIONAL_MODELS),
        f"{METADATA_MODEL}.onnx",
        SHARED_MODEL_NAME,
        SHARED_DATA_NAME,
    }
    actual = {path.name for path in output_folder.iterdir() if path.is_file()}
    missing = sorted(expected - actual)
    sidecars = sorted(
        name for name in actual if name.endswith(".onnx.data") and name != SHARED_DATA_NAME
    )
    transient = sorted(name for name in actual if name.startswith(".VoxCPM_"))
    if missing or sidecars or transient:
        raise RuntimeError(
            f"Package audit failed: missing={missing}, sidecars={sidecars}, transient={transient}"
        )
    options = onnxruntime.SessionOptions()
    shared_refs = attach_shared_initializers(options, output_folder / SHARED_MODEL_NAME)
    load_seconds = {}
    for file_name in sorted(expected):
        if not file_name.endswith(".onnx"):
            continue
        path = output_folder / file_name
        check_model_allowing_runtime_extensions(path)
        validate_external_data_bounds(path)
        graph_metadata = read_onnx_metadata(str(path))
        mismatches = {
            key: (value, graph_metadata.get(key))
            for key, value in metadata.items()
            if graph_metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Metadata mismatch in {file_name}: {mismatches}")
        if file_name == SHARED_MODEL_NAME:
            continue
        start = time.perf_counter()
        runtime_session = onnxruntime.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        load_seconds[file_name] = time.perf_counter() - start
        del runtime_session
    if preserve_fp16_attention:
        for name in ("VoxCPM_MainPrefill", "VoxCPM_DecodeStep"):
            validate_no_inserted_precision_casts(output_folder / f"{name}.onnx")
    return {
        "artifact_count": len(expected),
        "shared_initializer_ortvalues": len(shared_refs[1]),
        "session_load_seconds": load_seconds,
        "package_bytes": sum((output_folder / name).stat().st_size for name in expected),
    }


def main():
    args = parse_args()
    resolved_plans = resolve_plans(False, True)
    if args.check_only:
        quantized_count = sum(
            plan.method in WEIGHT_ONLY_BITS or plan.method == "DYNAMIC"
            for plan in resolved_plans.values()
        )
        print(
            f"VoxCPM optimizer plan is valid: {quantized_count} quantized graphs, "
            f"{len(MODEL_PLANS)} graphs total."
        )
        return

    validate_sources()
    metadata, preserve_fp16_attention = configure_precision(SOURCE_FOLDER)
    allow_transformer_fusions = supports_transformer_fusions(SOURCE_FOLDER)
    coverage = audit_recipe_coverage(SOURCE_FOLDER)
    print(f"[Coverage] {json.dumps(coverage, sort_keys=True)}")
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
        audit_quantized_graphs(
            OUTPUT_FOLDER,
            resolved_plans,
            coverage,
            allow_transformer_fusions,
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
    audit_final_package(
        OUTPUT_FOLDER,
        output_metadata,
        preserve_fp16_attention,
    )


if __name__ == "__main__":
    main()
