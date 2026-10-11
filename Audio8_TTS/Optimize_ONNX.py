"""Quantize and optimize Audio8-TTS graphs, then rebuild one shared bundle.

Every graph has its own plan. Strategy graphs share one weight packing and the
TargetPreprocess embeddings use the same packing so they deduplicate in the bundle.
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
    process_model,
    quantize_weight_only_shared,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
)
from Shared_Weights import bundle_shared_initializers  # noqa: E402


# User configuration
STRATEGIES = ("greedy", "penalty_greedy", "sampling")
SOURCE_FOLDER = SCRIPT_DIR / "Audio8TTS_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "Audio8TTS_Optimized"
QUANTIZATION_TEMPLATE = "Audio8TTS_DecodeStep_greedy"
QUANTIZATION_CACHE_NAME = ".Audio8TTS_QuantizedWeights.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

# Quantization defaults
MATMUL_ALGORITHM = "AFFINE_REFINE_V2"
BLOCK_SIZE = 32
ACCURACY_LEVEL = 4
MAIN_HIDDEN_SIZE = 896
MAIN_HEAD_DIM = 64
MAIN_NUM_HEADS = MAIN_HIDDEN_SIZE // MAIN_HEAD_DIM

STRATEGY_GRAPH_NAMES = tuple(
    f"Audio8TTS_{stage}_{strategy}"
    for stage in ("SlowPrefill", "DecodeStep")
    for strategy in STRATEGIES
)
PROCESS_SUPPORT_GRAPH_NAMES = (
    "Audio8TTS_ReferencePreprocess",
    "Audio8TTS_TargetPreprocess",
    "Audio8TTS_Decoder",
    "Audio8TTS_Decoder_Stream",
)
PASSTHROUGH_GRAPH_NAMES = ("Audio8TTS_Metadata",)


def build_model_plans(metadata: dict[str, str]) -> dict[str, Plan]:
    hidden_size = int(metadata.get("hidden_size", MAIN_HIDDEN_SIZE))
    head_dim = int(metadata.get("head_dim", MAIN_HEAD_DIM))
    num_heads = hidden_size // head_dim

    def weight_plan(**overrides) -> Plan:
        settings = dict(
            method="Q8",
            algo=MATMUL_ALGORITHM,
            op_types=("MatMul", "Gather"),
            axes=(0, 1),
            block_size=BLOCK_SIZE,
            accuracy_level=ACCURACY_LEVEL,
            optimize=True,
            transformer=False,
            external=True,
        )
        settings.update(overrides)
        return Plan(**settings)

    # The codec graphs stay F32: the reference VQ search and the decoder have not had an audio A/B test.
    codec_plan = Plan(method="F32", optimize=True, transformer=False, external=True)
    plans = {
        "Audio8TTS_ReferencePreprocess": codec_plan,
        "Audio8TTS_Decoder": codec_plan,
        "Audio8TTS_Decoder_Stream": codec_plan,
        # Same packing as the strategy graphs so the embedding tables share bytes in the bundle.
        "Audio8TTS_TargetPreprocess": weight_plan(),
        "Audio8TTS_Metadata": Plan(method="F32", optimize=False, transformer=False, external=False),
    }
    for name in STRATEGY_GRAPH_NAMES:
        plans[name] = weight_plan(transformer=True, num_heads=num_heads, hidden_size=hidden_size)
    return plans


def read_package_metadata():
    metadata = read_onnx_metadata(str(SOURCE_FOLDER / "Audio8TTS_Metadata.onnx"))
    flags = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
    invalid = {key: value for key, value in flags.items() if value not in {"0", "1"}}
    if invalid:
        raise ValueError(f"Audio8TTS_Metadata.onnx has invalid precision metadata: {invalid}.")
    preserve_fp16_attention = flags["use_f16_kv"] == "1" and flags["compute_in_f32"] == "0"
    if preserve_fp16_attention:
        print(
            "[Precision] FP16 KV attention requested; skipping native ORT optimization so "
            "CastFloat16Transformer cannot promote attention to FP32."
        )
    return metadata, preserve_fp16_attention


def validate_source_package(metadata):
    missing = [
        f"{key} -> {value}"
        for key, value in sorted(metadata.items())
        if key.startswith("model_file_name_") and not (SOURCE_FOLDER / value).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Audio8TTS_ONNX is incomplete; metadata declares missing runtime graph(s):\n  "
            + "\n  ".join(missing)
            + "\nRe-run Export_Audio8TTS.py."
        )


def available_strategy_graph_names() -> tuple[str, ...]:
    available = tuple(name for name in STRATEGY_GRAPH_NAMES if (SOURCE_FOLDER / f"{name}.onnx").is_file())
    skipped = [name for name in STRATEGY_GRAPH_NAMES if name not in available]
    if skipped:
        print(f"[Graph selection] Skipping unexported strategy graphs: {', '.join(skipped)}.")
    return available


def resolve_plans(config, preserve_fp16_attention):
    resolved_plans = {}
    for name, plan in config.model_plans.items():
        resolved = resolve_plan(plan, config)
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
        repr(plan.nodes_to_exclude),
        repr(plan.nodes_to_include),
    )


def shared_weight_plan(resolved_plans, strategy_graph_names):
    if not strategy_graph_names:
        print("[Shared quantization] No strategy graphs were exported.")
        return None
    template_name = (
        QUANTIZATION_TEMPLATE if QUANTIZATION_TEMPLATE in strategy_graph_names else strategy_graph_names[0]
    )
    template_plan = resolved_plans[template_name]
    if template_plan.method not in WEIGHT_ONLY_BITS:
        print(f"[Shared quantization] {template_name} uses {template_plan.method}; processing graphs independently.")
        return None
    signature = weight_quantization_signature(template_plan)
    incompatible = [
        name for name in strategy_graph_names if weight_quantization_signature(resolved_plans[name]) != signature
    ]
    if incompatible:
        print(f"[Shared quantization] Strategy graphs have different plans ({incompatible}); processing separately.")
        return None
    return template_name, template_plan


def quantize_shared_weights(config, cache_path, resolved_plans, strategy_graph_names):
    """Pack the strategy weights once; return the graph names that are already quantized."""
    selected = shared_weight_plan(resolved_plans, strategy_graph_names)
    if selected is None:
        return set()
    template_name, template_plan = selected
    source, output = Path(config.original_folder_path), Path(config.optimized_folder_path)
    try:
        stats = quantize_weight_only_shared(
            str(source / f"{template_name}.onnx"),
            [(str(source / f"{name}.onnx"), str(output / f"{name}.onnx")) for name in strategy_graph_names],
            str(cache_path),
            template_plan,
            bits=WEIGHT_ONLY_BITS[template_plan.method],
            external=True,
        )
    except Exception as error:
        for name in strategy_graph_names:
            (output / f"{name}.onnx").unlink(missing_ok=True)
            (output / f"{name}.onnx.data").unlink(missing_ok=True)
        print(f"[Shared quantization] Shared packing was not applicable; processing independently ({error}).")
        return set()
    print(
        f"[Shared quantization] Packed {stats['unique_weights']} unique weights once; reused them across "
        f"{stats['total_rewrites']} weight-only nodes in {stats['graph_count']} graphs."
    )
    return set(strategy_graph_names)


def _data_sibling(path: Path) -> Path:
    return path.with_name(path.name + ".data")


def stage_external_data_dependencies(model_path, source_folder, stage_folder):
    # Shared-quantized graphs reference the cache sidecar, so it must sit beside the staged graph.
    model = onnx.load(str(model_path), load_external_data=False)
    locations = {
        entry.value
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }
    for location in sorted(locations):
        source, staged = Path(source_folder) / location, Path(stage_folder) / location
        if staged.exists() or not source.is_file():
            continue
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, staged)


def process_graph_in_staging(config, name, plan, mixed_precision, prequantized=False):
    output_folder = Path(config.optimized_folder_path)
    stage_folder = output_folder / f".{name}.stage"
    final_path = output_folder / f"{name}.onnx"
    staged_path = stage_folder / f"{name}.onnx"
    shutil.rmtree(stage_folder, ignore_errors=True)
    stage_folder.mkdir(parents=True)
    try:
        if prequantized:
            if not final_path.is_file():
                raise FileNotFoundError(f"Shared quantization did not produce {final_path}.")
            final_path.replace(staged_path)
            if _data_sibling(final_path).is_file():
                _data_sibling(final_path).replace(_data_sibling(staged_path))
            stage_external_data_dependencies(staged_path, output_folder, stage_folder)
        process_model(
            name,
            plan,
            replace(config, optimized_folder_path=str(stage_folder)),
            mixed_precision=mixed_precision,
            prequantized=prequantized,
        )
        if not staged_path.is_file():
            raise FileNotFoundError(f"Optimization did not produce expected graph: {staged_path}.")
        final_path.unlink(missing_ok=True)
        _data_sibling(final_path).unlink(missing_ok=True)
        staged_path.replace(final_path)
        if _data_sibling(staged_path).is_file():
            _data_sibling(staged_path).replace(_data_sibling(final_path))
    finally:
        shutil.rmtree(stage_folder, ignore_errors=True)


def process_graphs(config, resolved_plans, prequantized_graphs, strategy_graph_names):
    names = (*PROCESS_SUPPORT_GRAPH_NAMES, *strategy_graph_names)
    mixed_precision = uses_mixed_precision(resolved_plans[name] for name in names)
    if mixed_precision and config.f16_keep_io_types is None:
        print("[Precision] Not all graphs use F16; enabling keep_io_types for float16 conversions.")
    for name in PROCESS_SUPPORT_GRAPH_NAMES:
        print(f"\nOptimizing support: {name}  [{resolved_plans[name].method}]")
        process_graph_in_staging(config, name, resolved_plans[name], mixed_precision)
    for name in strategy_graph_names:
        shared = name in prequantized_graphs
        suffix = ", shared weights" if shared else ""
        print(f"\nOptimizing strategy: {name}  [{resolved_plans[name].method}{suffix}]")
        process_graph_in_staging(config, name, resolved_plans[name], mixed_precision, prequantized=shared)
    output_folder = Path(config.optimized_folder_path)
    missing = [name for name in names if not (output_folder / f"{name}.onnx").is_file()]
    if missing:
        raise FileNotFoundError(f"Optimization did not retain required graph(s): {', '.join(missing)}.")


def rebuild_shared_bundle(config, metadata, strategy_graph_names):
    source, output = Path(config.original_folder_path), Path(config.optimized_folder_path)
    model_paths = [
        *(source / f"{name}.onnx" for name in PASSTHROUGH_GRAPH_NAMES),
        *(output / f"{name}.onnx" for name in PROCESS_SUPPORT_GRAPH_NAMES),
        *(output / f"{name}.onnx" for name in strategy_graph_names),
    ]
    stats = bundle_shared_initializers(output, model_paths=model_paths, metadata=metadata)
    replace_onnx_metadata(str(output / "Audio8TTS_Metadata.onnx"), metadata)
    print(
        f"[Shared weights] {stats['initializer_references']} references -> {stats['unique_initializers']} "
        f"unique tensors; deduplicated {stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB."
    )


def portable_embedding_gather_count(path, plan):
    """Count Gather nodes that weight-only quantization can pack as GatherBlockQuantized."""
    if "Gather" not in plan.op_types:
        return 0
    model = onnx.load(str(path), load_external_data=False)
    tables = {
        tensor.name: tensor.dims
        for tensor in model.graph.initializer
        if tensor.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16) and len(tensor.dims) == 2
    }
    count = 0
    for node in model.graph.node:
        if node.op_type != "Gather" or not node.input or node.input[0] not in tables:
            continue
        axis = next((attribute.i for attribute in node.attribute if attribute.name == "axis"), 0)
        if axis == 0 and tables[node.input[0]][1] % plan.block_size == 0:
            count += 1
    return count


def io_signature(path):
    """Name, dtype, rank and static dimensions of every graph input and output."""
    graph = onnx.load(str(path), load_external_data=False).graph

    def describe(value):
        tensor_type = value.type.tensor_type
        dims = tuple(dim.dim_value if dim.HasField("dim_value") else None for dim in tensor_type.shape.dim)
        return value.name, tensor_type.elem_type, dims

    return [describe(value) for value in graph.input], [describe(value) for value in graph.output]


def io_signature_differences(source_io, output_io):
    """List input/output changes; shape inference may only fill in dimensions the source left unknown."""
    differences = []
    for kind, source_entries, output_entries in zip(("input", "output"), source_io, output_io):
        if len(source_entries) != len(output_entries):
            differences.append(f"{kind} count {len(source_entries)} -> {len(output_entries)}")
            continue
        for index, (before, after) in enumerate(zip(source_entries, output_entries)):
            if kind == "input":
                compatible = before == after
            else:
                compatible = (
                    before[:2] == after[:2]
                    and len(before[2]) == len(after[2])
                    and all(old is None or old == new for old, new in zip(before[2], after[2]))
                )
            if not compatible:
                differences.append(f"{kind}[{index}] {before} -> {after}")
    return differences


def validate_optimized_package(config, metadata, resolved_plans, strategy_graph_names):
    source, output = Path(config.original_folder_path), Path(config.optimized_folder_path)
    declared = [value for key, value in sorted(metadata.items()) if key.startswith("model_file_name_")]
    bundle = [metadata["shared_initializer_model_file"], metadata["shared_initializer_data_file"]]
    missing = [name for name in (*declared, *bundle) if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Optimized Audio8TTS package is missing graph(s): {', '.join(missing)}.")
    for name in (*PROCESS_SUPPORT_GRAPH_NAMES, *strategy_graph_names):
        plan = resolved_plans[name]
        expected = portable_embedding_gather_count(source / f"{name}.onnx", plan) if plan.method in WEIGHT_ONLY_BITS else 0
        if expected:
            model = onnx.load(str(output / f"{name}.onnx"), load_external_data=False)
            actual = sum(node.op_type == "GatherBlockQuantized" for node in model.graph.node)
            if actual != expected:
                raise RuntimeError(f"{name}.onnx quantized {actual}/{expected} portable embedding Gather node(s).")
        differences = io_signature_differences(
            io_signature(source / f"{name}.onnx"), io_signature(output / f"{name}.onnx")
        )
        if differences:
            raise RuntimeError(
                f"{name}.onnx changed its input/output signature during optimization: {'; '.join(differences[:3])}"
            )


def promote_output_folder(staged: Path, final: Path) -> None:
    backup = final.with_name(final.name + ".previous")
    shutil.rmtree(backup, ignore_errors=True)
    if final.exists():
        os.replace(final, backup)
    try:
        os.replace(staged, final)
    except Exception:
        if backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    metadata, preserve_fp16_attention = read_package_metadata()
    validate_source_package(metadata)
    strategy_graph_names = available_strategy_graph_names()
    staging_folder = OUTPUT_FOLDER.with_name(OUTPUT_FOLDER.name + ".staging")
    shutil.rmtree(staging_folder, ignore_errors=True)
    staging_folder.mkdir(parents=True)
    config = OptimizerConfig(
        original_folder_path=str(SOURCE_FOLDER),
        optimized_folder_path=str(staging_folder),
        model_plans=build_model_plans(metadata),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        dynamic_default_tensor_type=onnx.TensorProto.FLOAT,
    )
    resolved_plans = resolve_plans(config, preserve_fp16_attention)
    cache_path = staging_folder / QUANTIZATION_CACHE_NAME
    cache_artifacts = (cache_path, _data_sibling(cache_path))
    try:
        prequantized = quantize_shared_weights(config, cache_path, resolved_plans, strategy_graph_names)
        process_graphs(config, resolved_plans, prequantized, strategy_graph_names)
        rebuild_shared_bundle(config, metadata, strategy_graph_names)
        for artifact in cache_artifacts:
            artifact.unlink(missing_ok=True)
        validate_optimized_package(config, metadata, resolved_plans, strategy_graph_names)
        promote_output_folder(staging_folder, OUTPUT_FOLDER)
    finally:
        for artifact in cache_artifacts:
            artifact.unlink(missing_ok=True)
        shutil.rmtree(staging_folder, ignore_errors=True)
    print(f"\nOptimized package written to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
