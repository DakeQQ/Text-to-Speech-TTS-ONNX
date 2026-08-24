from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


STANDARD_DOMAINS = {"", "ai.onnx"}
FLOAT_TYPES = {
    TensorProto.FLOAT16,
    TensorProto.FLOAT,
    TensorProto.DOUBLE,
    TensorProto.BFLOAT16,
}


def _iter_sparse_tensor_parts(
    sparse_tensor: onnx.SparseTensorProto,
) -> Iterator[onnx.TensorProto]:
    yield sparse_tensor.values
    yield sparse_tensor.indices


def _iter_node_tensors(
    nodes: Iterable[onnx.NodeProto],
) -> Iterator[onnx.TensorProto]:
    for node in nodes:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("sparse_tensor"):
                yield from _iter_sparse_tensor_parts(attribute.sparse_tensor)
            for sparse_tensor in attribute.sparse_tensors:
                yield from _iter_sparse_tensor_parts(sparse_tensor)
            if attribute.HasField("g"):
                yield from _iter_graph_tensors(attribute.g)
            for graph in attribute.graphs:
                yield from _iter_graph_tensors(graph)


def _iter_graph_tensors(graph: onnx.GraphProto) -> Iterator[onnx.TensorProto]:
    yield from graph.initializer
    for sparse_tensor in graph.sparse_initializer:
        yield from _iter_sparse_tensor_parts(sparse_tensor)
    yield from _iter_node_tensors(graph.node)


def _iter_model_tensors(model: onnx.ModelProto) -> Iterator[onnx.TensorProto]:
    yield from _iter_graph_tensors(model.graph)
    for function in model.functions:
        yield from _iter_node_tensors(function.node)
    for training_info in model.training_info:
        if training_info.HasField("initialization"):
            yield from _iter_graph_tensors(training_info.initialization)
        if training_info.HasField("algorithm"):
            yield from _iter_graph_tensors(training_info.algorithm)


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _is_position_conv(
    node: onnx.NodeProto, initializers: dict[str, onnx.TensorProto]
) -> bool:
    if (
        node.op_type != "Conv"
        or node.domain not in STANDARD_DOMAINS
        or len(node.input) not in (2, 3)
        or len(node.output) != 1
    ):
        return False
    weight = initializers.get(node.input[1])
    if weight is None or len(weight.dims) != 3 or weight.data_type not in FLOAT_TYPES:
        return False
    if len(node.input) == 3:
        bias = initializers.get(node.input[2])
        if (
            bias is None
            or len(bias.dims) != 1
            or bias.dims[0] != weight.dims[0]
            or bias.data_type != weight.data_type
        ):
            return False
    return (
        weight.dims[0] > 0
        and weight.dims[1] > 0
        and weight.dims[-1] == 31
        and _attribute(node, "auto_pad", b"NOTSET") == b"NOTSET"
        and _attribute(node, "group", 1) == 16
        and list(_attribute(node, "kernel_shape", [])) == [31]
        and list(_attribute(node, "pads", [])) == [15, 15]
        and list(_attribute(node, "strides", [1])) == [1]
        and list(_attribute(node, "dilations", [1])) == [1]
    )


def _unique_node_name(existing_names: set[str], base: str) -> str:
    candidate = base
    suffix = 1
    while candidate in existing_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    existing_names.add(candidate)
    return candidate


def _external_locations(model: onnx.ModelProto) -> set[str]:
    locations: set[str] = set()
    for tensor in _iter_model_tensors(model):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        entries = {entry.key: entry.value for entry in tensor.external_data}
        location = entries.get("location")
        if not location:
            raise RuntimeError(f"External tensor has no location: {tensor.name}")
        locations.add(location)
    return locations


def _copy_external_data(model: onnx.ModelProto, source_dir: Path, destination_dir: Path) -> None:
    for location in _external_locations(model):
        source = (source_dir / location).resolve()
        destination = (destination_dir / location).resolve()
        if source_dir.resolve() not in source.parents or destination_dir.resolve() not in destination.parents:
            raise RuntimeError(f"Unsafe external-data location in ONNX graph: {location}")
        if not source.is_file():
            raise FileNotFoundError(f"Missing ONNX external data file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def rewrite_mish_subgraphs(
    raw_model_path: str | Path,
    final_model_path: str | Path,
    *,
    expected_matches: int = 2,
) -> dict[str, Any]:
    if expected_matches < 1:
        raise ValueError("expected_matches must be at least one")
    raw_path = Path(raw_model_path).expanduser().resolve()
    final_path = Path(final_model_path).expanduser().resolve()
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw ONNX model does not exist: {raw_path}")

    model = onnx.load(raw_path, load_external_data=False)
    interface_before = [
        (value.name, value.type.SerializeToString())
        for value in (*model.graph.input, *model.graph.output)
    ]
    metadata_before = [(item.key, item.value) for item in model.metadata_props]
    initializers_before = [item.SerializeToString() for item in model.graph.initializer]
    opsets_before = [item.SerializeToString() for item in model.opset_import]
    nodes = list(model.graph.node)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    producers = {
        output_name: node
        for node in nodes
        for output_name in node.output
        if output_name
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in nodes:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    node_indices = {id(node): index for index, node in enumerate(nodes)}

    matches: list[tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, str]] = []
    for softplus in nodes:
        if (
            softplus.op_type != "Softplus"
            or softplus.domain not in STANDARD_DOMAINS
            or softplus.attribute
            or len(softplus.input) != 1
            or len(softplus.output) != 1
        ):
            continue
        source = softplus.input[0]
        source_producer = producers.get(source)
        if source_producer is None or not _is_position_conv(source_producer, initializers):
            continue
        softplus_consumers = consumers.get(softplus.output[0], [])
        if len(softplus_consumers) != 1:
            continue
        tanh = softplus_consumers[0]
        if (
            tanh.op_type != "Tanh"
            or tanh.domain not in STANDARD_DOMAINS
            or tanh.attribute
            or list(tanh.input) != [softplus.output[0]]
            or len(tanh.output) != 1
        ):
            continue
        tanh_consumers = consumers.get(tanh.output[0], [])
        if len(tanh_consumers) != 1:
            continue
        multiply = tanh_consumers[0]
        if (
            multiply.op_type != "Mul"
            or multiply.domain not in STANDARD_DOMAINS
            or multiply.attribute
            or len(multiply.input) != 2
            or len(multiply.output) != 1
            or sorted(multiply.input) != sorted((source, tanh.output[0]))
        ):
            continue
        matches.append((softplus, tanh, multiply, source))

    if len(matches) != expected_matches:
        raise RuntimeError(
            "Refusing to mutate transformer graph: expected exactly "
            f"{expected_matches} ConvPosition Mish decompositions, found {len(matches)}"
        )
    assert len(matches) == expected_matches

    matched_indices: set[int] = set()
    replacement_by_index: dict[int, onnx.NodeProto] = {}
    dead_value_info: set[str] = set()
    existing_names = {node.name for node in nodes if node.name}
    for match_index, (softplus, tanh, multiply, source) in enumerate(matches):
        indices = {
            node_indices[id(softplus)],
            node_indices[id(tanh)],
            node_indices[id(multiply)],
        }
        matched_indices.update(indices)
        replacement_by_index[node_indices[id(softplus)]] = helper.make_node(
            "Mish",
            inputs=[source],
            outputs=[multiply.output[0]],
            name=_unique_node_name(existing_names, f"Raon_ConvPosition_Mish_{match_index}"),
        )
        dead_value_info.update((softplus.output[0], tanh.output[0]))

    retained_before = [
        node.SerializeToString()
        for index, node in enumerate(nodes)
        if index not in matched_indices
    ]
    rewritten_nodes: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        replacement = replacement_by_index.get(index)
        if replacement is not None:
            rewritten_nodes.append(replacement)
        if index not in matched_indices:
            rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    retained_value_info = [
        value for value in model.graph.value_info if value.name not in dead_value_info
    ]
    removed_value_info = len(model.graph.value_info) - len(retained_value_info)
    if removed_value_info:
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_info)

    interface_after = [
        (value.name, value.type.SerializeToString())
        for value in (*model.graph.input, *model.graph.output)
    ]
    retained_after = [
        node.SerializeToString()
        for node in model.graph.node
        if not (node.op_type == "Mish" and node.name.startswith("Raon_ConvPosition_Mish_"))
    ]
    if interface_after != interface_before:
        raise RuntimeError("Mish rewrite changed the graph input/output interface")
    if [(item.key, item.value) for item in model.metadata_props] != metadata_before:
        raise RuntimeError("Mish rewrite changed graph metadata")
    if [item.SerializeToString() for item in model.graph.initializer] != initializers_before:
        raise RuntimeError("Mish rewrite changed graph initializers")
    if [item.SerializeToString() for item in model.opset_import] != opsets_before:
        raise RuntimeError("Mish rewrite changed graph opset imports")
    if retained_after != retained_before:
        raise RuntimeError("Mish rewrite changed nodes outside the two matched decompositions")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    _copy_external_data(model, raw_path.parent, final_path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.", suffix=final_path.suffix, dir=final_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        onnx.save(model, temporary_path)
        onnx.checker.check_model(str(temporary_path))
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "raw_model": str(raw_path),
        "final_model": str(final_path),
        "matched_subgraphs": len(matches),
        "inserted_nodes": len(matches),
        "deleted_nodes": len(matches) * 3,
        "net_node_reduction": len(matches) * 2,
        "removed_value_info": removed_value_info,
        "operator": "ai.onnx::Mish-18",
        "interfaces_preserved": True,
        "metadata_preserved": True,
        "initializers_preserved": True,
        "opsets_preserved": True,
    }


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def validate_rewrite_parity(
    raw_model_path: str | Path,
    final_model_path: str | Path,
    *,
    duration: int,
    fp16: bool,
) -> dict[str, float]:
    if duration < 1:
        raise ValueError("duration must be at least one")
    expected_type = "tensor(float16)" if fp16 else "tensor(float)"
    random = np.random.default_rng(0)
    raw_session = _session(Path(raw_model_path).expanduser().resolve())
    arguments = {argument.name: argument for argument in raw_session.get_inputs()}
    expected_names = {
        "noise",
        "rope_cos",
        "rope_sin",
        "cat_mel_text",
        "cat_mel_text_drop",
        "time_step",
    }
    if set(arguments) != expected_names:
        raise RuntimeError(
            "Unexpected transformer interface during rewrite parity: "
            f"expected {sorted(expected_names)}, found {sorted(arguments)}"
        )

    inputs: dict[str, np.ndarray] = {}
    for name, argument in arguments.items():
        if name == "time_step":
            if argument.type != "tensor(int32)" or len(argument.shape) != 1:
                raise RuntimeError(
                    f"time_step must be rank-1 INT32, found {argument.type}/{argument.shape}"
                )
            shape = tuple(
                dimension if isinstance(dimension, int) else 1
                for dimension in argument.shape
            )
            inputs[name] = np.zeros(shape, dtype=np.int32)
            continue
        if argument.type != expected_type:
            raise RuntimeError(
                f"Transformer value {name!r} must be {expected_type}, found {argument.type}"
            )
        if len(argument.shape) not in (3, 4):
            raise RuntimeError(
                f"Transformer value {name!r} has unexpected rank: {argument.shape}"
            )
        sequence_axis = 2 if name in {"rope_cos", "rope_sin"} else 1
        shape = tuple(
            duration
            if index == sequence_axis and not isinstance(dimension, int)
            else dimension
            if isinstance(dimension, int)
            else 1
            for index, dimension in enumerate(argument.shape)
        )
        dtype = np.float16 if argument.type == "tensor(float16)" else np.float32
        inputs[name] = random.standard_normal(shape).astype(dtype)

    raw_output = raw_session.run(None, inputs)[0]
    del raw_session
    final_session = _session(Path(final_model_path).expanduser().resolve())
    final_output = final_session.run(None, inputs)[0]
    del final_session

    if raw_output.shape != final_output.shape:
        raise RuntimeError(
            f"Rewrite changed transformer output shape: {raw_output.shape} vs {final_output.shape}"
        )
    if not np.isfinite(raw_output).all() or not np.isfinite(final_output).all():
        raise RuntimeError("Rewrite parity produced NaN or Inf")
    difference = np.abs(raw_output.astype(np.float32) - final_output.astype(np.float32))
    maximum = float(difference.max(initial=0.0))
    mean = float(difference.mean())
    rtol, atol = ((2e-3, 2e-3) if fp16 else (1e-5, 1e-5))
    np.testing.assert_allclose(final_output, raw_output, rtol=rtol, atol=atol)
    return {"max_error": maximum, "mean_error": mean, "rtol": rtol, "atol": atol}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace exactly two Raon ConvPosition Mish decompositions."
    )
    parser.add_argument("raw_model", type=Path)
    parser.add_argument("final_model", type=Path)
    parser.add_argument("--expected-matches", type=int, default=2)
    parser.add_argument("--validate-parity", action="store_true")
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = rewrite_mish_subgraphs(
        args.raw_model,
        args.final_model,
        expected_matches=args.expected_matches,
    )
    if args.validate_parity:
        report["parity"] = validate_rewrite_parity(
            args.raw_model,
            args.final_model,
            duration=args.duration,
            fp16=args.fp16,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())