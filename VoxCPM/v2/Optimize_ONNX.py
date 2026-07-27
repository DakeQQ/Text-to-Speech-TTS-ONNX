"""Build an experimental Q4 package for VoxCPM2 ONNX graphs.

Only constant MatMul/Gather weights under main/, feat_encoder/, and prefill/
are quantized. VAE, feature-decoder, assembly, and state-management branches
remain F32. The result remains experimental until autoregressive validation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import onnx
import onnxruntime


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    quantize_weight_only_shared,
    resolve_plan,
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
GRAPH_LAYOUT = "voxcpm2_prefill_decode_v1"
METADATA_MODEL = "VoxCPM2_Metadata"
PREFILL_MODELS = (
    "VoxCPM2_MainPrefill_VoiceDesign",
    "VoxCPM2_MainPrefill_Continuation",
    "VoxCPM2_MainPrefill_ReferenceOnly",
    "VoxCPM2_MainPrefill_Combined",
)
QUANTIZED_MODELS = (*PREFILL_MODELS, "VoxCPM2_DecodeStep")
QUANTIZATION_TEMPLATE = "VoxCPM2_MainPrefill_Combined"
QUANTIZATION_CACHE_NAME = ".VoxCPM2_Q4_QuantizedWeights.onnx"
CORE_PREFIXES = ("main/", "feat_encoder/", "prefill/")
EXCLUDED_PREFIXES = (
    "feat_decoder/",
    "reference_vae/",
    "vae_encoder/",
    "vae_decoder/",
    "assemble/",
    "decode_inputs/",
    "accumulator/",
)


def select_quantizable_nodes(model_path):
    selected, _ = _constant_weight_nodes(model_path)
    return [
        node_name
        for prefix in CORE_PREFIXES
        for _, node_name, _ in selected[prefix]
    ]


MODEL_PLANS: dict[str, Plan] = {
    "VoxCPM2_AudioVAE_Encode": Plan(
        method="F32",
        optimize=True,
        transformer=True,
        external=True,
    ),
    "VoxCPM2_MainPrefill_VoiceDesign": Plan(
        method="Q4",
        algo="k_quant",
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=32,
        accuracy_level=4,
        nodes_to_include=select_quantizable_nodes,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "VoxCPM2_MainPrefill_Continuation": Plan(
        method="Q4",
        algo="k_quant",
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=32,
        accuracy_level=4,
        nodes_to_include=select_quantizable_nodes,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "VoxCPM2_MainPrefill_ReferenceOnly": Plan(
        method="Q4",
        algo="k_quant",
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=32,
        accuracy_level=4,
        nodes_to_include=select_quantizable_nodes,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "VoxCPM2_MainPrefill_Combined": Plan(
        method="Q4",
        algo="k_quant",
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=32,
        accuracy_level=4,
        nodes_to_include=select_quantizable_nodes,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "VoxCPM2_DecodeStep": Plan(
        method="Q4",
        algo="k_quant",
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=32,
        accuracy_level=4,
        nodes_to_include=select_quantizable_nodes,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "VoxCPM2_AudioVAE_Decode": Plan(
        method="F32",
        optimize=True,
        transformer=True,
        external=True,
    ),
    "VoxCPM2_AudioVAE_Decode_Stream": Plan(
        method="F32",
        optimize=True,
        transformer=True,
        external=True,
    ),
    "VoxCPM2_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def resolve_plans():
    resolved_plans = {}
    for name, plan in MODEL_PLANS.items():
        resolved = resolve_plan(plan, CONFIG)
        validate_plan(name, resolved)
        expected_method = "Q4" if name in QUANTIZED_MODELS else "F32"
        if resolved.method != expected_method or resolved.optimize or resolved.fp16:
            raise RuntimeError(
                f"[{name}] expected {expected_method} without graph optimization or F16."
            )
        if name in QUANTIZED_MODELS and (
            resolved.op_types != ("MatMul", "Gather")
            or resolved.axes != (0, 1)
            or resolved.nodes_to_include is not select_quantizable_nodes
        ):
            raise RuntimeError(f"[{name}] has an unsupported Q4 selection policy.")
        resolved_plans[name] = resolved
    return resolved_plans


def _metadata(path):
    model = onnx.load(str(path), load_external_data=False)
    return {item.key: item.value for item in model.metadata_props}


def _set_metadata(path, metadata):
    model = onnx.load(str(path), load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, str(path))


def _functional_files(metadata):
    names = {
        metadata["model_file_name_vae_encoder"],
        metadata["model_file_name_decode_step"],
        metadata["model_file_name_vae_decoder"],
        metadata["model_file_name_vae_decoder_stream"],
        metadata["model_file_name_metadata"],
        *(
            metadata[f"model_file_name_main_prefill_{mode}"]
            for mode in ("voice_design", "continuation", "reference_only", "combined")
        ),
    }
    return tuple(sorted(names))


def _constant_weight_nodes(path):
    model = onnx.load(str(path), load_external_data=False)
    initializers = {tensor.name for tensor in model.graph.initializer}
    selected = {prefix: [] for prefix in CORE_PREFIXES}
    excluded = []
    for node in model.graph.node:
        if node.op_type not in {"MatMul", "Gather"} or len(node.input) != 2:
            continue
        weight_index = 1 if node.op_type == "MatMul" else 0
        if node.input[weight_index] not in initializers:
            continue
        matching = [prefix for prefix in CORE_PREFIXES if node.name.startswith(prefix)]
        if matching:
            selected[matching[0]].append((node.op_type, node.name, node.input[weight_index]))
        if node.name.startswith(EXCLUDED_PREFIXES):
            excluded.append((node.op_type, node.name, node.input[weight_index]))
    return selected, excluded


def _audit_prefix_coverage(functional_files):
    coverage = {}
    preserved = {}
    decode_weight_keys = set()
    prefill_weight_keys = {}
    for file_name in functional_files:
        path = SOURCE_FOLDER / file_name
        selected, excluded = _constant_weight_nodes(path)
        selected_names = {
            node_name
            for nodes in selected.values()
            for _, node_name, _ in nodes
        }
        overlap = [item for item in excluded if item[1] in selected_names]
        if overlap:
            raise RuntimeError(
                f"Excluded nodes entered the selected families in {file_name}: {overlap[:8]}"
            )
        counts = {prefix: len(nodes) for prefix, nodes in selected.items()}
        coverage[file_name] = counts
        preserved[file_name] = len(excluded)
        keys = {
            (op_type, weight_name)
            for nodes in selected.values()
            for op_type, _, weight_name in nodes
        }
        if file_name == "VoxCPM2_DecodeStep.onnx":
            decode_weight_keys = keys
        elif file_name.startswith("VoxCPM2_MainPrefill_"):
            prefill_weight_keys[file_name] = keys

    if not decode_weight_keys:
        raise RuntimeError("DecodeStep has no selected main/feat_encoder weights.")
    for file_name, keys in prefill_weight_keys.items():
        missing = sorted(decode_weight_keys - keys)
        if missing:
            raise RuntimeError(
                f"{file_name} lacks {len(missing)} DecodeStep shared recipes: {missing[:8]}"
            )
    template_file = f"{QUANTIZATION_TEMPLATE}.onnx"
    template_keys = prefill_weight_keys.get(template_file)
    if not template_keys:
        raise RuntimeError(f"Q4 template {template_file} has no selected weights.")
    target_weight_keys = {
        **prefill_weight_keys,
        "VoxCPM2_DecodeStep.onnx": decode_weight_keys,
    }
    for file_name, keys in target_weight_keys.items():
        missing = sorted(keys - template_keys)
        if missing:
            raise RuntimeError(
                f"{file_name} uses {len(missing)} weights absent from the Q4 template: "
                f"{missing[:8]}"
            )
    return {
        "prefixes": list(CORE_PREFIXES),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "per_graph_selected_nodes": coverage,
        "per_graph_preserved_excluded_nodes": preserved,
        "shared_recipe_template": template_file,
        "shared_recipe_targets": [
            *sorted(prefill_weight_keys),
            "VoxCPM2_DecodeStep.onnx",
        ],
        "shared_recipe_unique_weights": len(template_keys),
    }


def _copy_graph_without_materializing_weights(source, destination):
    model = onnx.load(str(source), load_external_data=False)
    # Preserve initializer references exactly. The output shared bank is rebuilt
    # after all graphs are copied, so no per-model sidecar is needed.
    onnx.save(model, str(destination))


def _finalize_metadata(metadata, functional_files):
    for path in sorted(OUTPUT_FOLDER.glob("*.onnx")):
        _set_metadata(path, metadata)
    return dict(metadata)


def _audit_output(metadata, functional_files):
    expected = {*functional_files, SHARED_MODEL_NAME, SHARED_DATA_NAME}
    actual = {path.name for path in OUTPUT_FOLDER.iterdir() if path.is_file()}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    sidecars = sorted(
        name for name in actual if name.endswith(".onnx.data") and name != SHARED_DATA_NAME
    )
    if missing or unexpected or sidecars:
        raise RuntimeError(
            f"Optimized package audit failed: missing={missing}, unexpected={unexpected}, "
            f"sidecars={sidecars}."
        )

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    shared_refs = attach_shared_initializers(options, OUTPUT_FOLDER / SHARED_MODEL_NAME)
    load_times = {}
    for file_name in functional_files:
        path = OUTPUT_FOLDER / file_name
        check_model_allowing_runtime_extensions(path)
        validate_external_data_bounds(path)
        if _metadata(path) != metadata:
            raise RuntimeError(f"Metadata mismatch in {file_name}.")
        start = time.perf_counter()
        session = onnxruntime.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        load_times[file_name] = time.perf_counter() - start
        del session
    return {
        "artifact_count": len(expected),
        "shared_initializer_count": len(shared_refs[1]),
        "package_bytes": sum((OUTPUT_FOLDER / name).stat().st_size for name in expected),
        "session_load_seconds": load_times,
    }


def _audit_quantized_operators(coverage, functional_files):
    quantized_op_types = {"MatMulNBits", "GatherBlockQuantized"}
    per_graph = {}
    for file_name in functional_files:
        model = onnx.load(str(OUTPUT_FOLDER / file_name), load_external_data=False)
        quantized_nodes = [
            node for node in model.graph.node if node.op_type in quantized_op_types
        ]
        name = Path(file_name).stem
        expected = coverage["per_graph_selected_nodes"][file_name]
        actual = {
            prefix: sum(node.name.startswith(prefix) for node in quantized_nodes)
            for prefix in CORE_PREFIXES
        }
        outside = sorted(
            node.name
            for node in quantized_nodes
            if not node.name.startswith(CORE_PREFIXES)
        )
        if name in QUANTIZED_MODELS:
            if actual != expected or outside:
                raise RuntimeError(
                    f"Q4 operator audit failed for {file_name}: "
                    f"expected={expected}, actual={actual}, outside={outside[:8]}."
                )
        elif quantized_nodes:
            raise RuntimeError(
                f"F32 graph {file_name} unexpectedly contains "
                f"{len(quantized_nodes)} Q4 operators."
            )
        per_graph[file_name] = {
            "MatMulNBits": sum(
                node.op_type == "MatMulNBits" for node in quantized_nodes
            ),
            "GatherBlockQuantized": sum(
                node.op_type == "GatherBlockQuantized" for node in quantized_nodes
            ),
        }
    return per_graph


def _quantize_configured_graphs(resolved_plans):
    cache_path = OUTPUT_FOLDER / QUANTIZATION_CACHE_NAME
    plan = resolved_plans[QUANTIZATION_TEMPLATE]
    stats = quantize_weight_only_shared(
        str(SOURCE_FOLDER / f"{QUANTIZATION_TEMPLATE}.onnx"),
        [
            (
                str(SOURCE_FOLDER / f"{name}.onnx"),
                str(OUTPUT_FOLDER / f"{name}.onnx"),
            )
            for name in QUANTIZED_MODELS
        ],
        str(cache_path),
        plan,
        bits=4,
        external=True,
    )
    return stats, (cache_path, Path(str(cache_path) + ".data"))


def _stage_source_shared_data():
    source = SOURCE_FOLDER / SHARED_DATA_NAME
    destination = OUTPUT_FOLDER / SHARED_DATA_NAME
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        subprocess.run(
            ["cp", "--reflink=always", "--preserve=mode,timestamps", str(source), str(destination)],
            check=True,
            capture_output=True,
            text=True,
        )
        return "reflink"
    except (OSError, subprocess.CalledProcessError):
        shutil.copy2(source, destination)
        return "copy"


def configure_package():
    metadata_path = SOURCE_FOLDER / f"{METADATA_MODEL}.onnx"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = _metadata(metadata_path)
    if metadata.get("graph_layout") != GRAPH_LAYOUT:
        raise RuntimeError("Optimizer requires the VoxCPM2 package.")
    return metadata


def validate_sources(metadata):
    functional_files = _functional_files(metadata)
    expected_files = {f"{name}.onnx" for name in MODEL_PLANS}
    if set(functional_files) != expected_files:
        raise RuntimeError(
            "VoxCPM2 metadata graph set differs from MODEL_PLANS: "
            f"missing={sorted(expected_files - set(functional_files))}, "
            f"unexpected={sorted(set(functional_files) - expected_files)}"
        )
    missing = [
        SOURCE_FOLDER / file_name
        for file_name in functional_files
        if not (SOURCE_FOLDER / file_name).is_file()
    ]
    for artifact in (SHARED_MODEL_NAME, SHARED_DATA_NAME):
        path = SOURCE_FOLDER / artifact
        if not path.is_file():
            missing.append(path)
    if missing:
        raise FileNotFoundError(f"Missing VoxCPM2 artifact(s): {missing}")
    return functional_files


def main():
    args = parse_args()
    resolved_plans = resolve_plans()
    if args.check_only:
        print(
            f"VoxCPM2 optimizer plan is valid: {len(QUANTIZED_MODELS)} Q4 graphs, "
            f"{len(resolved_plans) - len(QUANTIZED_MODELS)} F32 graphs."
        )
        return

    metadata = configure_package()
    functional_files = validate_sources(metadata)
    coverage = _audit_prefix_coverage(functional_files)
    print(f"[Coverage] {json.dumps(coverage, sort_keys=True)}")

    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True)
    _stage_source_shared_data()
    quantization, transient_paths = _quantize_configured_graphs(resolved_plans)
    for file_name in functional_files:
        if Path(file_name).stem in QUANTIZED_MODELS:
            continue
        _copy_graph_without_materializing_weights(
            SOURCE_FOLDER / file_name,
            OUTPUT_FOLDER / file_name,
        )

    bundle_stats = bundle_shared_initializers(
        OUTPUT_FOLDER,
        [OUTPUT_FOLDER / file_name for file_name in functional_files],
        metadata=metadata,
    )
    for path in transient_paths:
        path.unlink(missing_ok=True)
    output_metadata = _finalize_metadata(metadata, functional_files)
    _audit_quantized_operators(coverage, functional_files)
    audit = _audit_output(output_metadata, functional_files)
    print(
        f"[Q4] {quantization['unique_weights']} unique weights reused at "
        f"{quantization['total_rewrites']} nodes across "
        f"{quantization['graph_count']} graphs."
    )
    print(
        f"[Bundle] {bundle_stats['initializer_references']} references -> "
        f"{bundle_stats['unique_initializers']} exact tensors; "
        f"deduplicated {bundle_stats['deduplicated_bytes'] / (1024 ** 2):.2f} MiB."
    )
    print(
        f"[Validate] Loaded {len(functional_files)} functional graphs with "
        f"{audit['shared_initializer_count']} mmap initializers."
    )


if __name__ == "__main__":
    main()
