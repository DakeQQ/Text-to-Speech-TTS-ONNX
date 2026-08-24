"""Stage and optionally weight-only optimize a merged FireRedTTS3 ONNX package."""

from __future__ import annotations

import gc
import fcntl
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx
import onnxruntime


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
    ResolvedPlan,
    process_model,
    quantize_weight_only_shared,
    resolve_plan,
)
from Shared_Weights import (
    attach_shared_initializers,
    audit_shared_initializer_storage,
    bundle_shared_initializers,
    declared_model_files,
    inference_metadata,
    promote_directory,
    read_onnx_metadata,
    validate_package_contract,
    write_metadata_carrier,
)


SOURCE_FOLDER = SCRIPT_DIR / "FireRedTTS3_Instruct_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "FireRedTTS3_Instruct_Optimized"
METADATA_FILE_NAME = "FireRedTTS3_Metadata.onnx"

# Quantization and optimization defaults
MATMUL_ALGORITHM = "AFFINE_REFINE_V2"
BLOCK_SIZE = 128
ACCURACY_LEVEL = 4
DYNAMIC_WEIGHT_TYPE = "QInt8"
DYNAMIC_PER_CHANNEL = True
FLOAT_ATOL = 2.0e-5
FLOAT_RTOL = 2.0e-5
Q8_MAX_NORMALIZED_RMSE = 5.0e-2
Q8_MIN_COSINE_SIMILARITY = 9.99e-1
WEIGHT_ONLY_BITS = {"Q4": 4, "Q8": 8}


def _quant_plan(
    method: str,
    *,
    algorithm: str = MATMUL_ALGORITHM,
    accuracy_level: int = ACCURACY_LEVEL,
    block_size: int = BLOCK_SIZE
) -> Plan:
    method = method.upper()
    if method not in {"Q4", "Q8", "DYNAMIC"}:
        raise ValueError(f"Unsupported quantization method: {method!r}")
    op_types = ("MatMul",) if method == "DYNAMIC" else ("MatMul", "Gather")
    axes = (0,) if method == "DYNAMIC" else (0, 1)
    return Plan(
        method=method,
        algo=algorithm,
        op_types=op_types,
        axes=axes,
        block_size=block_size,
        accuracy_level=accuracy_level,
        optimize=True,
        transformer=False,
        external=True,
    )


# Per-graph quantization and optimization plan. Each runtime graph is quantized
# independently; the final package bundle only deduplicates identical payloads.
MODEL_PLANS: dict[str, Plan] = {
    "FireRedTTS3_RedAEEncode": _quant_plan("Q8"),
    "FireRedTTS3_RedAEDecode": _quant_plan("Q8"),
    "FireRedTTS3_BaseReferencePreprocess": _quant_plan("Q8"),
    "FireRedTTS3_BaseInputPrefill": _quant_plan("Q8"),
    "FireRedTTS3_BaseReferencePrefill": _quant_plan("Q8"),
    "FireRedTTS3_BaseAudioStart": _quant_plan("Q8"),
    "FireRedTTS3_BaseDecodeStep": _quant_plan("Q8"),
    "FireRedTTS3_InstructInputPrefill": _quant_plan("Q8"),
    "FireRedTTS3_InstructInputAudioPrefill": _quant_plan("Q8"),
    "FireRedTTS3_InstructOutputAudioPrefill": _quant_plan("Q8"),
    "FireRedTTS3_InstructTextDecodeStep": _quant_plan("Q8"),
    "FireRedTTS3_InstructAudioStart": _quant_plan("Q8"),
    "FireRedTTS3_InstructAudioDecodeStep": _quant_plan("Q8"),
}


def build_graph_plans(metadata: Mapping[str, str]) -> dict[str, Plan]:
    """Select exactly one explicit plan for every metadata-runtime graph."""
    graph_names = {
        Path(file_name).stem
        for file_name in declared_model_files(metadata).values()
    }
    missing = sorted(graph_names - MODEL_PLANS.keys())
    if missing:
        raise RuntimeError(f"No optimizer plan is declared for graph(s): {missing}")
    return {
        name: MODEL_PLANS[name]
        for name in MODEL_PLANS
        if name in graph_names
    }


def _weight_quantization_signature(plan: ResolvedPlan) -> tuple[object, ...] | None:
    if plan.method == "DYNAMIC":
        raise ValueError(
            "AFFINE_REFINE_V2 DYNAMIC cannot satisfy FireRedTTS3's "
            "quantize-once contract; use Q4 or Q8."
        )
    if plan.method not in WEIGHT_ONLY_BITS:
        return None
    if plan.nodes_to_include is not None or plan.nodes_to_exclude is not None:
        raise ValueError(
            "Shared FireRedTTS3 quantization requires whole-plan MatMul/Gather selection."
        )
    return (
        plan.method,
        plan.algo,
        plan.op_types,
        plan.axes,
        plan.block_size,
        plan.accuracy_level,
        plan.symmetric,
        plan.quant_format,
        plan.affine_v2_settings,
    )


def _initializer_signature(tensor: onnx.TensorProto) -> bytes:
    comparable = onnx.TensorProto()
    comparable.CopyFrom(tensor)
    comparable.name = ""
    return comparable.SerializeToString(deterministic=True)


def _resolve_initializer_alias(name: str, aliases: Mapping[str, str]) -> str:
    visited: set[str] = set()
    while name in aliases:
        if name in visited:
            raise ValueError(f"Initializer Identity alias cycle detected at {name!r}.")
        visited.add(name)
        name = aliases[name]
    return name


def _build_quantization_cover(
    source: Path,
    cover_path: Path,
    graph_names: tuple[str, ...],
    plan: ResolvedPlan,
) -> int:
    entries: dict[
        tuple[str, str],
        tuple[onnx.NodeProto, onnx.TensorProto, Path, bytes],
    ] = {}
    for graph_name in graph_names:
        model_path = source / f"{graph_name}.onnx"
        model = onnx.load(str(model_path), load_external_data=False)
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        aliases = {
            node.output[0]: node.input[0]
            for node in model.graph.node
            if node.op_type == "Identity"
            and len(node.input) == 1
            and len(node.output) == 1
        }
        for node in model.graph.node:
            if node.op_type == "MatMul" and node.op_type in plan.op_types and len(node.input) == 2:
                weight_name = _resolve_initializer_alias(node.input[1], aliases)
            elif node.op_type == "Gather" and node.op_type in plan.op_types and len(node.input) == 2:
                weight_name = _resolve_initializer_alias(node.input[0], aliases)
            else:
                continue
            weight = initializers.get(weight_name)
            if weight is None or len(weight.dims) != 2:
                continue

            key = (node.op_type, weight_name)
            signature = _initializer_signature(weight)
            prior = entries.get(key)
            if prior is not None:
                prior_node, _, prior_path, prior_signature = prior
                attributes = [item.SerializeToString() for item in node.attribute]
                prior_attributes = [item.SerializeToString() for item in prior_node.attribute]
                if signature != prior_signature or attributes != prior_attributes:
                    raise RuntimeError(
                        f"{graph_name} has an incompatible duplicate quantization identity "
                        f"for {node.op_type}:{weight_name}; first seen in {prior_path.name}."
                    )
                continue

            node_copy = onnx.NodeProto()
            node_copy.CopyFrom(node)
            weight_copy = onnx.TensorProto()
            weight_copy.CopyFrom(weight)
            entries[key] = (node_copy, weight_copy, model_path, signature)
        del model
        gc.collect()

    if not entries:
        raise RuntimeError(
            f"Shared quantization group {graph_names} contains no selected matrix weights."
        )

    nodes: list[onnx.NodeProto] = []
    inputs: list[onnx.ValueInfoProto] = []
    outputs: list[onnx.ValueInfoProto] = []
    initializers: dict[str, onnx.TensorProto] = {}
    for index, ((op_type, weight_name), (source_node, source_weight, source_path, _)) in enumerate(
        sorted(entries.items())
    ):
        weight = onnx.TensorProto()
        weight.CopyFrom(source_weight)
        for external_entry in weight.external_data:
            if external_entry.key == "location":
                data_path = (source_path.parent / external_entry.value).resolve()
                external_entry.value = os.path.relpath(data_path, cover_path.parent)
        prior_weight = initializers.get(weight_name)
        if prior_weight is not None and _initializer_signature(prior_weight) != _initializer_signature(weight):
            raise RuntimeError(
                f"Quantization cover initializer name collision for {weight_name!r}."
            )
        initializers.setdefault(weight_name, weight)

        input_name = f"quant_cover_input_{index}"
        output_name = f"quant_cover_output_{index}"
        if op_type == "MatMul":
            input_type = weight.data_type
            input_shape = [1, int(weight.dims[0])]
            output_shape = [1, int(weight.dims[1])]
            node_inputs = [input_name, weight_name]
        else:
            axis = next(
                (
                    int(onnx.helper.get_attribute_value(attribute))
                    for attribute in source_node.attribute
                    if attribute.name == "axis"
                ),
                0,
            )
            axis %= len(weight.dims)
            input_type = onnx.TensorProto.INT64
            input_shape = [1]
            output_shape = [int(dimension) for dimension in weight.dims]
            output_shape[axis : axis + 1] = [1]
            node_inputs = [weight_name, input_name]

        inputs.append(
            onnx.helper.make_tensor_value_info(input_name, input_type, input_shape)
        )
        outputs.append(
            onnx.helper.make_tensor_value_info(
                output_name,
                weight.data_type,
                output_shape,
            )
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
        str(source / f"{graph_names[0]}.onnx"),
        load_external_data=False,
    )
    graph = onnx.helper.make_graph(
        nodes,
        "FireRedTTS3_QuantizationCover",
        inputs,
        outputs,
        initializer=list(initializers.values()),
    )
    cover = onnx.helper.make_model(
        graph,
        producer_name="FireRedTTS3 shared quantization cover",
        opset_imports=[
            onnx.helper.make_opsetid(opset.domain, opset.version)
            for opset in template.opset_import
        ],
    )
    cover.ir_version = template.ir_version
    onnx.save(cover, str(cover_path))
    print(
        f"[Quantization cover] {len(entries)} unique operator/weight recipes "
        f"across {len(graph_names)} graphs."
    )
    del cover, graph, template
    gc.collect()
    return len(entries)


def _quantize_shared_weights(
    source: Path,
    staging: Path,
    resolved_plans: Mapping[str, ResolvedPlan],
) -> tuple[set[str], tuple[Path, ...]]:
    groups: dict[tuple[object, ...], list[str]] = {}
    for graph_name, plan in resolved_plans.items():
        signature = _weight_quantization_signature(plan)
        if signature is not None:
            groups.setdefault(signature, []).append(graph_name)

    prequantized: set[str] = set()
    temporary_artifacts: list[Path] = []
    for index, graph_names_list in enumerate(groups.values()):
        graph_names = tuple(graph_names_list)
        plan = resolved_plans[graph_names[0]]
        cover_path = staging / f".FireRedTTS3_{plan.method}_{index}_QuantizationCover.onnx"
        cache_path = staging / f".FireRedTTS3_{plan.method}_{index}_QuantizedWeights.onnx"
        recipe_count = _build_quantization_cover(
            source,
            cover_path,
            graph_names,
            plan,
        )
        stats = quantize_weight_only_shared(
            str(cover_path),
            [
                (
                    str(source / f"{graph_name}.onnx"),
                    str(staging / f"{graph_name}.onnx"),
                )
                for graph_name in graph_names
            ],
            str(cache_path),
            plan,
            bits=WEIGHT_ONLY_BITS[plan.method],
            external=True,
        )
        if (
            stats["unique_weights"] != recipe_count
            or stats["template_rewrites"] != recipe_count
            or stats["graph_count"] != len(graph_names)
            or stats["total_rewrites"] < recipe_count
        ):
            raise RuntimeError(
                f"Shared {plan.method} quantization statistics do not match the cover."
            )
        print(
            f"[Shared quantization:{plan.method}] Packed {recipe_count} recipes once; "
            f"reused them at {stats['total_rewrites']} nodes across "
            f"{stats['graph_count']} graphs."
        )
        prequantized.update(graph_names)
        temporary_artifacts.extend(
            (
                cover_path,
                Path(str(cover_path) + ".data"),
                cache_path,
                Path(str(cache_path) + ".data"),
            )
        )
    return prequantized, tuple(temporary_artifacts)


def _make_config(source: Path, staging: Path, graph_plans: Mapping[str, Plan]):
    return OptimizerConfig(
        original_folder_path=str(source),
        optimized_folder_path=str(staging),
        model_plans=dict(graph_plans),
        weight_only_algorithm=MATMUL_ALGORITHM,
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
        dynamic_per_channel=DYNAMIC_PER_CHANNEL,
        force_external_data=True,
        dynamic_default_tensor_type=onnx.TensorProto.FLOAT,
    )


def _run_graph_plans(
    staging: Path,
    resolved_plans: Mapping[str, ResolvedPlan],
    config: OptimizerConfig,
    prequantized: set[str],
) -> None:
    for name, plan in resolved_plans.items():
        output_path = staging / f"{name}.onnx"
        shared = name in prequantized
        shared_note = ", shared weights" if shared else ""
        print(
            f"\n{'=' * 60}\nOptimizing: {name}  "
            f"[{plan.method}, {plan.algo}{shared_note}]\n{'=' * 60}"
        )
        process_model(
            name,
            plan,
            config,
            mixed_precision=False,
            prequantized=shared,
        )
        if not output_path.is_file():
            raise FileNotFoundError(f"Optimizer did not produce expected graph: {output_path}")


def _open_shared_session(path: Path, shared_model: Path):
    options = onnxruntime.SessionOptions()
    arrays, values = attach_shared_initializers(options, shared_model)
    session = onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session, arrays, values


def _compare_outputs(
    reference: list[np.ndarray],
    actual: list[np.ndarray],
    graph_name: str,
    method: str,
) -> None:
    if len(reference) != len(actual):
        raise RuntimeError(f"{graph_name} produced a different output count after optimization.")
    for index, (before, after) in enumerate(zip(reference, actual)):
        if before.shape != after.shape:
            raise RuntimeError(
                f"{graph_name} output {index} changed shape from {before.shape} to {after.shape}."
            )
        if method == "F32":
            np.testing.assert_allclose(
                before,
                after,
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                err_msg=f"{graph_name} output {index}",
            )
            continue
        if method != "Q8":
            raise RuntimeError(
                f"No deterministic RedAE comparison policy is defined for {method}."
            )

        before_64 = before.astype(np.float64, copy=False).ravel()
        after_64 = after.astype(np.float64, copy=False).ravel()
        if not np.isfinite(before_64).all() or not np.isfinite(after_64).all():
            raise AssertionError(f"{graph_name} output {index} contains non-finite values.")
        difference_norm = float(np.linalg.norm(after_64 - before_64))
        reference_norm = float(np.linalg.norm(before_64))
        actual_norm = float(np.linalg.norm(after_64))
        normalized_rmse = difference_norm / max(reference_norm, np.finfo(np.float64).tiny)
        cosine = (
            float(np.dot(before_64, after_64) / (reference_norm * actual_norm))
            if reference_norm > 0.0 and actual_norm > 0.0
            else float(reference_norm == actual_norm)
        )
        if (
            normalized_rmse > Q8_MAX_NORMALIZED_RMSE
            or cosine < Q8_MIN_COSINE_SIMILARITY
        ):
            raise AssertionError(
                f"{graph_name} output {index} Q8 fidelity failed: "
                f"normalized_rmse={normalized_rmse:.9g} "
                f"(maximum {Q8_MAX_NORMALIZED_RMSE:.9g}), "
                f"cosine={cosine:.12g} "
                f"(minimum {Q8_MIN_COSINE_SIMILARITY:.12g})."
            )
        print(
            f"[Q8 fidelity] {graph_name} output {index}: "
            f"normalized_rmse={normalized_rmse:.9g}, cosine={cosine:.12g}"
        )


def _audit_package_weights(
    folder: Path,
    metadata: Mapping[str, str],
    label: str,
) -> dict[str, int]:
    stats = audit_shared_initializer_storage(folder, metadata)
    print(
        f"[Weight audit:{label}] {stats['initializer_references']} references -> "
        f"{stats['logical_initializers']} logical initializers, "
        f"{stats['physical_payloads']} unique payloads, "
        f"{stats['aliased_initializers']} storage aliases."
    )
    return stats


def _external_tensor_data(tensor: onnx.TensorProto) -> dict[str, str]:
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        raise ValueError(f"Tensor {tensor.name!r} is not external data.")
    external = {entry.key: entry.value for entry in tensor.external_data}
    if "location" not in external or "length" not in external:
        raise ValueError(f"Tensor {tensor.name!r} has no complete external-data range.")
    return external


def _external_storage_signature(tensor: onnx.TensorProto) -> tuple[object, ...]:
    external = _external_tensor_data(tensor)
    return (
        int(tensor.data_type),
        tuple(int(dimension) for dimension in tensor.dims),
        external["location"],
        int(external.get("offset", "0")),
        int(external["length"]),
    )


def _exact_external_transpose(
    embedding: onnx.TensorProto,
    lm_head: onnx.TensorProto,
    model_path: Path,
) -> bool:
    embedding_shape = tuple(int(dimension) for dimension in embedding.dims)
    head_shape = tuple(int(dimension) for dimension in lm_head.dims)
    if (
        embedding.data_type != lm_head.data_type
        or len(embedding_shape) != 2
        or embedding_shape != tuple(reversed(head_shape))
    ):
        return False

    embedding_data = _external_tensor_data(embedding)
    head_data = _external_tensor_data(lm_head)
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(embedding.data_type))
    expected_bytes = int(np.prod(embedding_shape, dtype=np.int64)) * dtype.itemsize
    if (
        int(embedding_data["length"]) != expected_bytes
        or int(head_data["length"]) != expected_bytes
    ):
        return False

    embedding_view = np.memmap(
        model_path.parent / embedding_data["location"],
        mode="r",
        dtype=dtype,
        offset=int(embedding_data.get("offset", "0")),
        shape=embedding_shape,
    )
    head_view = np.memmap(
        model_path.parent / head_data["location"],
        mode="r",
        dtype=dtype,
        offset=int(head_data.get("offset", "0")),
        shape=head_shape,
    )
    rows_per_chunk = max(
        1,
        (8 * 1024 * 1024) // (embedding_shape[1] * dtype.itemsize),
    )
    try:
        for start in range(0, embedding_shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, embedding_shape[0])
            if not np.array_equal(
                embedding_view[start:end],
                head_view[:, start:end].T,
            ):
                return False
    finally:
        del embedding_view, head_view
    return True


def _tied_source_weights(
    model_path: Path,
) -> tuple[onnx.NodeProto, onnx.TensorProto, onnx.NodeProto, onnx.TensorProto]:
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    embedding_matches = [
        (node, initializers.get(node.input[0]))
        for node in model.graph.node
        if node.op_type == "Gather"
        and node.input
        and "embed_tokens" in node.name
        and initializers.get(node.input[0]) is not None
    ]
    if len(embedding_matches) != 1:
        raise ValueError(
            f"Expected one embedding Gather in {model_path.name}, found {len(embedding_matches)}."
        )
    embedding_node, embedding = embedding_matches[0]
    if embedding is None:
        raise ValueError(f"Embedding initializer is missing in {model_path.name}.")
    embedding_shape = tuple(int(dimension) for dimension in embedding.dims)
    head_matches = [
        (node, initializers.get(node.input[1]))
        for node in model.graph.node
        if node.op_type == "MatMul"
        and len(node.input) >= 2
        and initializers.get(node.input[1]) is not None
        and tuple(int(dimension) for dimension in initializers[node.input[1]].dims)
        == tuple(reversed(embedding_shape))
    ]
    if len(head_matches) != 1:
        raise ValueError(
            f"Expected one embedding-shaped LM head in {model_path.name}, found {len(head_matches)}."
        )
    head_node, lm_head = head_matches[0]
    if lm_head is None or not _exact_external_transpose(embedding, lm_head, model_path):
        raise ValueError(f"{model_path.name} embedding and LM head are not exactly tied.")
    return embedding_node, embedding, head_node, lm_head


def _quantized_tied_components(
    model_path: Path,
    embedding_node_name: str,
    head_node_name: str,
) -> tuple[tuple[onnx.TensorProto, ...], tuple[onnx.TensorProto, ...]]:
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    embedding_nodes = [
        node
        for node in model.graph.node
        if node.op_type == "GatherBlockQuantized"
        and node.name.startswith(embedding_node_name)
    ]
    head_nodes = [
        node
        for node in model.graph.node
        if node.op_type == "MatMulNBits"
        and node.name.startswith(head_node_name)
    ]
    if len(embedding_nodes) != 1 or len(head_nodes) != 1:
        raise ValueError(
            f"Could not identify one quantized embedding/head pair in {model_path.name}."
        )
    embedding_node = embedding_nodes[0]
    head_node = head_nodes[0]
    if len(embedding_node.input) != 4 or len(head_node.input) != 4:
        raise ValueError(f"Unexpected quantized embedding/head inputs in {model_path.name}.")
    return (
        tuple(initializers[name] for name in (embedding_node.input[0], *embedding_node.input[2:])),
        tuple(initializers[name] for name in head_node.input[1:]),
    )


def _validate_tied_q4_storage(
    source: Path,
    optimized: Path,
    metadata: Mapping[str, str],
    resolved_plans: Mapping[str, ResolvedPlan],
) -> None:
    role_keys = (
        "model_file_name_instruct_input_prefill",
        "model_file_name_instruct_text_decode_step",
    )
    file_names = [metadata.get(key) for key in role_keys]
    if not all(file_names):
        print("[Tied Q4] No Instruct embedding/LM-head graph pair is present.")
        return
    graph_names = [Path(file_name).stem for file_name in file_names if file_name]
    if any(
        resolved_plans[name].method != "Q4"
        or resolved_plans[name].algo != MATMUL_ALGORITHM
        or "Gather" not in resolved_plans[name].op_types
        for name in graph_names
    ):
        print("[Tied Q4] Storage sharing skipped because both Instruct graphs are not AFFINE_REFINE_V2 Q4.")
        return

    leader_ranges: tuple[tuple[object, ...], ...] | None = None
    saved_bytes = 0
    source_signatures: tuple[tuple[object, ...], tuple[object, ...]] | None = None
    for file_name in file_names:
        if file_name is None:
            continue
        source_path = source / file_name
        embedding_node, embedding, head_node, lm_head = _tied_source_weights(source_path)
        current_source_signatures = (
            _external_storage_signature(embedding),
            _external_storage_signature(lm_head),
        )
        if source_signatures is None:
            source_signatures = current_source_signatures
        elif current_source_signatures != source_signatures:
            raise RuntimeError("Instruct prefill and decode do not use the same tied source table.")

        quantized_embedding, quantized_head = _quantized_tied_components(
            optimized / file_name,
            embedding_node.name,
            head_node.name,
        )
        embedding_ranges = tuple(
            _external_storage_signature(tensor)[2:]
            for tensor in quantized_embedding
        )
        head_ranges = tuple(
            _external_storage_signature(tensor)[2:]
            for tensor in quantized_head
        )
        if embedding_ranges != head_ranges:
            raise RuntimeError(
                f"{file_name} Q4 embedding and LM head do not share physical storage."
            )
        if leader_ranges is None:
            leader_ranges = embedding_ranges
            saved_bytes = sum(
                int(length)
                for embedding_tensor, head_tensor, (_, _, length) in zip(
                    quantized_embedding,
                    quantized_head,
                    embedding_ranges,
                )
                if embedding_tensor.name != head_tensor.name
            )
        elif embedding_ranges != leader_ranges:
            raise RuntimeError("Instruct prefill and decode use different tied Q4 payloads.")

    print(
        "[Tied Q4] Instruct lm_head shares the embed_tokens AFFINE_REFINE_V2 "
        f"payload; saved {saved_bytes / (1024 * 1024):.2f} MiB."
    )


def _deterministic_redae_comparison(
    source: Path,
    optimized: Path,
    metadata: Mapping[str, str],
    resolved_plans: Mapping[str, ResolvedPlan],
) -> None:
    """Compare deterministic latent and waveform graph outputs after staging."""
    source_shared = source / metadata["shared_initializer_model_file"]
    optimized_shared = optimized / metadata["shared_initializer_model_file"]
    for metadata_key in (
        "model_file_name_redae_encode",
        "model_file_name_redae_decode",
    ):
        file_name = metadata.get(metadata_key)
        if not file_name:
            continue
        source_session, source_arrays, source_values = _open_shared_session(source / file_name, source_shared)
        output_session, output_arrays, output_values = _open_shared_session(optimized / file_name, optimized_shared)
        if metadata_key == "model_file_name_redae_encode":
            inputs = {
                "prompt_audio": np.linspace(
                    -0.1, 0.1, 160, dtype=np.float32
                ).reshape(1, -1)
            }
        else:
            latent_width = source_session.get_inputs()[0].shape[-1]
            if not isinstance(latent_width, int) or latent_width <= 0:
                raise RuntimeError(
                    f"Cannot infer RedAE latent width from {file_name} model I/O."
                )
            inputs = {
                "generated_latents": np.zeros(
                    (1, 1, latent_width), dtype=np.float32
                ),
                "prefix_latents": np.empty(
                    (1, 0, latent_width), dtype=np.float32
                ),
            }
        before = source_session.run(None, inputs)
        after = output_session.run(None, inputs)
        graph_name = Path(file_name).stem
        _compare_outputs(before, after, file_name, resolved_plans[graph_name].method)
        del source_session, source_arrays, source_values, output_session, output_arrays, output_values
        print(f"[Deterministic parity] {file_name}: passed")


@contextmanager
def _exclusive_package_lock(output: Path):
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a+", encoding="ascii") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.seek(0)
            owner = lock_file.read().strip() or "another process"
            raise RuntimeError(
                f"Optimization is already active for {output.name} ({owner})."
            ) from error
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def optimize_package(source_folder: Path, output_folder: Path) -> Path:
    source = source_folder.expanduser().resolve()
    output = output_folder.expanduser().resolve()
    with _exclusive_package_lock(output):
        return _optimize_package_locked(source, output)


def _optimize_package_locked(source: Path, output: Path) -> Path:
    metadata = validate_package_contract(
        source,
        METADATA_FILE_NAME,
        required_keys=(
            "package_schema_version",
            "graph_layout",
            "model_variant",
            "shared_initializer_model_file",
            "shared_initializer_data_file",
        ),
        require_shared_bundle=True,
    )
    metadata = inference_metadata(metadata)
    if metadata["graph_layout"] != "merged_decode_step":
        raise RuntimeError(
            "Optimization requires a merged_decode_step package; run Merge_ONNX.py first."
        )
    if metadata["package_schema_version"] != "2":
        raise RuntimeError("Optimization requires a schema-v2 FireRedTTS3 package.")
    _audit_package_weights(source, metadata, "source")
    graph_plans = build_graph_plans(metadata)
    declared_files = declared_model_files(metadata)
    declared_graph_names = {Path(file_name).stem for file_name in declared_files.values()}
    if set(graph_plans) != declared_graph_names:
        raise RuntimeError("Optimizer plan coverage does not match metadata-declared graph coverage.")
    staging = output.with_name(output.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        config = _make_config(source, staging, graph_plans)
        resolved_plans = {
            name: resolve_plan(plan, config)
            for name, plan in graph_plans.items()
        }
        prequantized, temporary_artifacts = _quantize_shared_weights(
            source,
            staging,
            resolved_plans,
        )
        _run_graph_plans(
            staging,
            resolved_plans,
            config,
            prequantized,
        )
        for artifact in temporary_artifacts:
            artifact.unlink(missing_ok=True)
        bundle_stats = bundle_shared_initializers(
            staging,
            model_paths=[staging / file_name for file_name in declared_files.values()],
            metadata=metadata,
        )

        write_metadata_carrier(staging / METADATA_FILE_NAME, metadata)
        validate_package_contract(
            staging,
            METADATA_FILE_NAME,
            required_keys=(
                "package_schema_version",
                "graph_layout",
                "model_variant",
                "shared_initializer_model_file",
                "shared_initializer_data_file",
            ),
            require_shared_bundle=True,
        )
        _validate_tied_q4_storage(source, staging, metadata, resolved_plans)
        audit = _audit_package_weights(staging, metadata, "optimized")
        if (
            audit["logical_initializers"] != bundle_stats["unique_initializers"]
            or audit["physical_bytes"] != bundle_stats["unique_bytes"]
        ):
            raise RuntimeError("Final shared-initializer audit disagrees with bundle statistics.")
        _deterministic_redae_comparison(source, staging, metadata, resolved_plans)
        promote_directory(staging, output)
        return output
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    optimized = optimize_package(SOURCE_FOLDER, OUTPUT_FOLDER)
    print(f"Optimized FireRedTTS3 package promoted to: {optimized}")


if __name__ == "__main__":
    main()