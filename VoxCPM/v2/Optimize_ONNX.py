"""Build an experimental Q4 package for VoxCPM2 ONNX graphs.

Only constant MatMul/Gather weights under main/, feat_encoder/, and prefill/
are quantized. VAE, feature-decoder, assembly, and state-management branches
remain F32. The result remains experimental until autoregressive validation.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    quantize_weight_only_shared,
    resolve_plan,
)
from Shared_Weights import (  # noqa: E402
    SHARED_DATA_NAME,
    bundle_shared_initializers,
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
    return parser.parse_args()


def resolve_plans():
    resolved_plans = {}
    for name, plan in MODEL_PLANS.items():
        resolved = resolve_plan(plan, CONFIG)
        expected_method = "Q4" if name in QUANTIZED_MODELS else "F32"
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


def _copy_graph_without_materializing_weights(source, destination):
    model = onnx.load(str(source), load_external_data=False)
    # Preserve initializer references exactly. The output shared bank is rebuilt
    # after all graphs are copied, so no per-model sidecar is needed.
    onnx.save(model, str(destination))


def _finalize_metadata(metadata, functional_files):
    for path in sorted(OUTPUT_FOLDER.glob("*.onnx")):
        _set_metadata(path, metadata)
    return dict(metadata)


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
    shutil.copy2(source, destination)
    return "copy"


def configure_package():
    metadata_path = SOURCE_FOLDER / f"{METADATA_MODEL}.onnx"
    metadata = _metadata(metadata_path)
    return metadata


def main():
    args = parse_args()
    resolved_plans = resolve_plans()
    metadata = configure_package()
    functional_files = _functional_files(metadata)

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


if __name__ == "__main__":
    main()
