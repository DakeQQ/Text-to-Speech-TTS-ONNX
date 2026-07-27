"""Apply narrow, audited causal-padding rewrites to raw VoxCPM2 ONNX graphs."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper


EXPECTED_PATTERNS = {
    "VoxCPM2_AudioVAE_Encode.onnx": {
        "conv_prefix_histogram": Counter({2: 2, 5: 1, 6: 5, 8: 2, 18: 4, 54: 4}),
        "multi_input_concat_arities": Counter({4: 1}),
        "conv_transpose_crop_histogram": Counter(),
    },
    "VoxCPM2_AudioVAE_Decode.onnx": {
        "conv_prefix_histogram": Counter({6: 8, 18: 6, 54: 6}),
        "multi_input_concat_arities": Counter(),
        "conv_transpose_crop_histogram": Counter({2: 3, 5: 1, 6: 1, 8: 1}),
    },
    "VoxCPM2_AudioVAE_Decode_Stream.onnx": {
        "conv_prefix_histogram": Counter({6: 8, 18: 6, 54: 6}),
        "multi_input_concat_arities": Counter(),
        "conv_transpose_crop_histogram": Counter({2: 3, 5: 1, 6: 1, 8: 1}),
    },
}


class RewriteError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise RewriteError(message)


def _attribute(node, name, default=None):
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _set_ints_attribute(node, name, values):
    for attribute in node.attribute:
        if attribute.name == name:
            _require(attribute.type == AttributeProto.INTS, f"{node.op_type}.{name} is not INTS")
            del attribute.ints[:]
            attribute.ints.extend(values)
            return
    node.attribute.append(helper.make_attribute(name, values))


def _is_standard_node(node, op_type):
    return node.op_type == op_type and node.domain in ("", "ai.onnx")


def _build_maps(model, nodes=None):
    producers = {}
    consumers = defaultdict(list)
    for node in model.graph.node if nodes is None else nodes:
        for output in node.output:
            _require(output not in producers, f"Tensor {output!r} has multiple producers")
            producers[output] = node
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    return producers, consumers, initializers


def _constant_tensor(name, producers, initializers):
    initializer = initializers.get(name)
    if initializer is not None:
        return initializer, None

    producer = producers.get(name)
    _require(producer is not None and _is_standard_node(producer, "Constant"), f"{name!r} is not a Constant")
    value_attributes = [
        attribute
        for attribute in producer.attribute
        if attribute.name == "value" and attribute.type == AttributeProto.TENSOR
    ]
    _require(len(value_attributes) == 1, f"Constant producer for {name!r} does not have one tensor value")
    return value_attributes[0].t, producer


def _constant_array(name, producers, initializers):
    tensor, producer = _constant_tensor(name, producers, initializers)
    _require(tensor.data_location != TensorProto.EXTERNAL, f"Small control tensor {name!r} unexpectedly uses external data")
    return numpy_helper.to_array(tensor), producer


def _interface_signature(model):
    return {
        "inputs": [value.SerializeToString() for value in model.graph.input],
        "outputs": [value.SerializeToString() for value in model.graph.output],
        "metadata": [value.SerializeToString() for value in model.metadata_props],
        "opsets": [(entry.domain, entry.version) for entry in model.opset_import],
        "ir_version": model.ir_version,
    }


def _operator_histogram(model):
    return Counter(node.op_type for node in model.graph.node)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _folder_snapshot(folder):
    return {
        str(path.relative_to(folder)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _validate_model(model):
    onnx.checker.check_model(model)
    try:
        inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=False)
    except Exception as error:
        raise RewriteError(f"Strict ONNX shape inference failed: {error}") from error
    onnx.checker.check_model(inferred)


def _rewrite_model(raw_path, final_path, expected):
    raw_path = Path(raw_path).resolve()
    final_path = Path(final_path).resolve()
    _require(raw_path != final_path, "Raw and final ONNX paths must be different")
    _require(raw_path.is_file(), f"Raw ONNX model does not exist: {raw_path}")
    _require(raw_path.stat().st_size < 2_000_000_000, "Targeted VAE rewrite requires an inline ONNX model smaller than 2 GB")

    onnx.checker.check_model(str(raw_path))
    model = onnx.load(raw_path, load_external_data=False)
    _require(
        not any(tensor.data_location == TensorProto.EXTERNAL or tensor.external_data for tensor in model.graph.initializer),
        "Targeted VAE rewrite does not accept external initializers",
    )
    before_contract = _interface_signature(model)
    before_histogram = _operator_histogram(model)
    nodes = list(model.graph.node)
    before_nodes = {id(node): node.SerializeToString() for node in nodes}
    producers, consumers, initializers = _build_maps(model, nodes)
    graph_output_names = {value.name for value in model.graph.output}

    conv_matches = []
    for conv in nodes:
        if not _is_standard_node(conv, "Conv"):
            continue
        _require(len(conv.input) in (2, 3) and len(conv.output) == 1, "A candidate Conv has an unexpected interface")
        concat = producers.get(conv.input[0])
        if concat is None or not _is_standard_node(concat, "Concat"):
            continue
        _require(len(concat.input) >= 2 and len(concat.output) == 1, "Causal Conv Concat must have at least two inputs")
        _require(_attribute(concat, "axis") in (-1, 2), "Causal Conv Concat axis must be the last rank-3 axis")
        _require(consumers[concat.output[0]] == [conv], "Causal Conv Concat output is shared")
        _require(concat.output[0] not in graph_output_names, "Causal Conv Concat unexpectedly feeds a graph output")
        if list(_attribute(conv, "pads", [])) != [0, 0]:
            continue

        zero_name = concat.input[0]
        zero_array, zero_producer = _constant_array(zero_name, producers, initializers)
        _require(zero_array.dtype == np.float32, f"Causal Conv prefix {zero_name!r} must be float32")
        _require(
            zero_array.ndim == 3 and zero_array.shape[0] == 1 and zero_array.shape[2] > 0,
            f"Causal Conv prefix {zero_name!r} must be non-empty rank-3 batch-1 data",
        )
        _require(np.count_nonzero(zero_array) == 0, f"Causal Conv prefix {zero_name!r} is not all zero")
        _require(consumers[zero_name] == [concat], f"Causal Conv prefix {zero_name!r} is shared")

        weight = initializers.get(conv.input[1])
        _require(weight is not None and len(weight.dims) == 3, "Causal Conv weight must be a rank-3 initializer")
        _require(weight.data_type == TensorProto.FLOAT, "Causal Conv weight must be float32")
        group = int(_attribute(conv, "group", 1))
        _require(group > 0 and zero_array.shape[1] == int(weight.dims[1]) * group, "Causal Conv prefix channels do not match its weight")
        _require(list(_attribute(conv, "kernel_shape", [])) == [int(weight.dims[2])], "Causal Conv kernel shape does not match its weight")
        _require(list(_attribute(conv, "pads", [])) == [0, 0], "Causal Conv must start with pads=[0, 0]")
        _require(_attribute(conv, "auto_pad", b"NOTSET") in (b"NOTSET", "NOTSET"), "Causal Conv auto_pad must be NOTSET")
        _require(len(_attribute(conv, "dilations", [1])) == 1, "Causal Conv must be one-dimensional")
        _require(len(_attribute(conv, "strides", [1])) == 1, "Causal Conv must have one stride")
        _require(all(input_name for input_name in concat.input[1:]), "Causal Conv data input is empty")
        conv_matches.append((conv, concat, zero_name, zero_producer, int(zero_array.shape[2])))

    conv_transpose_matches = []
    for conv_transpose in nodes:
        if not _is_standard_node(conv_transpose, "ConvTranspose"):
            continue
        _require(len(conv_transpose.input) in (2, 3) and len(conv_transpose.output) == 1, "A candidate ConvTranspose has an unexpected interface")
        output_name = conv_transpose.output[0]
        output_consumers = consumers[output_name]
        if len(output_consumers) != 1 or not _is_standard_node(output_consumers[0], "Slice"):
            continue
        slice_node = output_consumers[0]
        _require(output_name not in graph_output_names, "ConvTranspose pre-crop output unexpectedly is a graph output")
        _require(len(slice_node.input) == 5 and len(slice_node.output) == 1, "ConvTranspose crop Slice must have five inputs and one output")
        _require(slice_node.input[0] == output_name, "ConvTranspose crop Slice data edge is inconsistent")
        _require(list(_attribute(conv_transpose, "pads", [])) == [0, 0], "ConvTranspose must start with pads=[0, 0]")
        _require(_attribute(conv_transpose, "auto_pad", b"NOTSET") in (b"NOTSET", "NOTSET"), "ConvTranspose auto_pad must be NOTSET")
        _require(_attribute(conv_transpose, "output_shape") is None, "ConvTranspose output_shape must be absent")
        _require(list(_attribute(conv_transpose, "output_padding", [0])) == [0], "ConvTranspose output_padding must be absent or [0]")
        _require(list(_attribute(conv_transpose, "dilations", [1])) == [1], "ConvTranspose dilation must be one")

        controls = [_constant_array(name, producers, initializers) for name in slice_node.input[1:]]
        arrays = [item[0] for item in controls]
        _require(all(array.dtype == np.int64 and array.size == 1 for array in arrays), "ConvTranspose Slice controls must be scalar int64 tensors")
        start, end, axis, step = (int(array.reshape(-1)[0]) for array in arrays)
        _require(start == 0 and end < 0 and axis in (2, -1) and step == 1, "ConvTranspose Slice is not a simple trailing crop")
        crop = -end
        for control_name in slice_node.input[1:]:
            _require(consumers[control_name] == [slice_node], f"ConvTranspose Slice control {control_name!r} is shared")

        weight = initializers.get(conv_transpose.input[1])
        _require(weight is not None and len(weight.dims) == 3, "ConvTranspose weight must be a rank-3 initializer")
        _require(weight.data_type == TensorProto.FLOAT, "ConvTranspose weight must be float32")
        kernel = int(weight.dims[2])
        stride = list(_attribute(conv_transpose, "strides", [1]))
        _require(list(_attribute(conv_transpose, "kernel_shape", [])) == [kernel], "ConvTranspose kernel shape does not match its weight")
        _require(len(stride) == 1 and kernel == 2 * int(stride[0]), "ConvTranspose must use kernel_size=2*stride")
        _require(crop == int(stride[0]), "ConvTranspose crop must equal its stride")
        conv_transpose_matches.append(
            (conv_transpose, slice_node, crop, [item[1] for item in controls], list(slice_node.input[1:]))
        )

    prefix_histogram = Counter(match[4] for match in conv_matches)
    multi_input_concat_arities = Counter(len(match[1].input) for match in conv_matches if len(match[1].input) > 2)
    crop_histogram = Counter(match[2] for match in conv_transpose_matches)
    _require(
        prefix_histogram == expected["conv_prefix_histogram"],
        f"{raw_path.name}: causal Conv prefix histogram {dict(prefix_histogram)} does not match expected {dict(expected['conv_prefix_histogram'])}; model may already be rewritten or incompatible",
    )
    _require(
        multi_input_concat_arities == expected["multi_input_concat_arities"],
        f"{raw_path.name}: multi-input Concat arities {dict(multi_input_concat_arities)} do not match expected {dict(expected['multi_input_concat_arities'])}",
    )
    _require(
        crop_histogram == expected["conv_transpose_crop_histogram"],
        f"{raw_path.name}: ConvTranspose crop histogram {dict(crop_histogram)} does not match expected {dict(expected['conv_transpose_crop_histogram'])}; model may already be rewritten or incompatible",
    )
    removed_node_ids = set()
    mutated_node_ids = set()
    removed_tensor_names = set()
    removed_initializer_names = set()
    rewired_edges = 0

    for conv, concat, zero_name, zero_producer, prefix in conv_matches:
        if len(concat.input) == 2:
            conv.input[0] = concat.input[1]
            removed_node_ids.add(id(concat))
            removed_tensor_names.add(concat.output[0])
            rewired_edges += 1
        else:
            del concat.input[0]
            mutated_node_ids.add(id(concat))
            rewired_edges += 1
        _set_ints_attribute(conv, "pads", [prefix, 0])
        mutated_node_ids.add(id(conv))
        if zero_producer is None:
            removed_initializer_names.add(zero_name)
        else:
            removed_node_ids.add(id(zero_producer))
            removed_tensor_names.update(zero_producer.output)

    for conv_transpose, slice_node, crop, control_producers, control_names in conv_transpose_matches:
        old_output = conv_transpose.output[0]
        conv_transpose.output[0] = slice_node.output[0]
        _set_ints_attribute(conv_transpose, "pads", [0, crop])
        mutated_node_ids.add(id(conv_transpose))
        removed_node_ids.add(id(slice_node))
        removed_tensor_names.add(old_output)
        rewired_edges += len(consumers[slice_node.output[0]])
        for control_name, control_producer in zip(control_names, control_producers):
            if control_producer is None:
                removed_initializer_names.add(control_name)
            else:
                removed_node_ids.add(id(control_producer))
                removed_tensor_names.update(control_producer.output)

    for node in nodes:
        if id(node) not in removed_node_ids and id(node) not in mutated_node_ids:
            _require(node.SerializeToString() == before_nodes[id(node)], f"Unrelated node changed: {node.name or node.op_type}")

    kept_nodes = [node for node in nodes if id(node) not in removed_node_ids]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    kept_initializers = [tensor for tensor in model.graph.initializer if tensor.name not in removed_initializer_names]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    kept_value_info = [value for value in model.graph.value_info if value.name not in removed_tensor_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    _require(_interface_signature(model) == before_contract, f"{raw_path.name}: graph interface, metadata, opset, or IR version changed")
    _validate_model(model)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{final_path.stem}.", suffix=".onnx", dir=final_path.parent, delete=False) as handle:
            temporary_path = Path(handle.name)
        onnx.save_model(model, temporary_path)
        onnx.checker.check_model(str(temporary_path))
        os.replace(temporary_path, final_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    after_histogram = _operator_histogram(model)
    return {
        "model": raw_path.name,
        "raw_nodes": sum(before_histogram.values()),
        "final_nodes": sum(after_histogram.values()),
        "raw_initializers": len(initializers),
        "final_initializers": len(model.graph.initializer),
        "conv_rewrites": len(conv_matches),
        "conv_transpose_rewrites": len(conv_transpose_matches),
        "removed_nodes": len(removed_node_ids),
        "rewired_edges": rewired_edges,
        "transformed_initializers": 0,
        "removed_initializers": len(removed_initializer_names),
        "prefix_histogram": dict(sorted(prefix_histogram.items())),
        "crop_histogram": dict(sorted(crop_histogram.items())),
        "raw_operator_histogram": dict(sorted(before_histogram.items())),
        "final_operator_histogram": dict(sorted(after_histogram.items())),
    }


def _install_staged_folder(stage_folder, final_folder):
    backup_folder = None
    if final_folder.exists():
        _require(final_folder.is_dir(), f"Final path is not a directory: {final_folder}")
        backup_folder = final_folder.parent / f".{final_folder.name}.backup-{uuid.uuid4().hex}"
        os.replace(final_folder, backup_folder)
    try:
        os.replace(stage_folder, final_folder)
    except Exception:
        if backup_folder is not None and backup_folder.exists():
            os.replace(backup_folder, final_folder)
        raise
    if backup_folder is not None:
        shutil.rmtree(backup_folder)


def rewrite_voxcpm2_onnx_folder(raw_folder, final_folder, require_stream_decoder=True):
    raw_folder = Path(raw_folder).resolve()
    final_folder = Path(final_folder).resolve()
    _require(raw_folder.is_dir(), f"Raw ONNX folder does not exist: {raw_folder}")
    _require(raw_folder != final_folder, "Raw and final ONNX folders must be different")
    _require(raw_folder not in final_folder.parents, "Final ONNX folder must not be inside the raw folder")
    expected_patterns = dict(EXPECTED_PATTERNS)
    if not require_stream_decoder:
        expected_patterns.pop("VoxCPM2_AudioVAE_Decode_Stream.onnx")
    for file_name in expected_patterns:
        _require((raw_folder / file_name).is_file(), f"Required raw model is missing: {raw_folder / file_name}")

    raw_snapshot = _folder_snapshot(raw_folder)
    raw_hashes = {file_name: _sha256(raw_folder / file_name) for file_name in expected_patterns}
    final_folder.parent.mkdir(parents=True, exist_ok=True)
    stage_folder = Path(tempfile.mkdtemp(prefix=f".{final_folder.name}.stage-", dir=final_folder.parent))
    installed = False
    try:
        for source in raw_folder.iterdir():
            destination = stage_folder / source.name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

        reports = []
        for file_name, expected in expected_patterns.items():
            reports.append(_rewrite_model(raw_folder / file_name, stage_folder / file_name, expected))

        _require(_folder_snapshot(raw_folder) == raw_snapshot, "Raw ONNX folder changed while rewriting")
        for file_name, digest in raw_hashes.items():
            _require(_sha256(raw_folder / file_name) == digest, f"Raw model changed while rewriting: {file_name}")

        _install_staged_folder(stage_folder, final_folder)
        installed = True
    finally:
        if not installed and stage_folder.exists():
            shutil.rmtree(stage_folder)

    return {
        "raw_folder": str(raw_folder),
        "final_folder": str(final_folder),
        "raw_models_preserved": True,
        "custom_domains_added": [],
        "opset_changes": [],
        "models": reports,
    }


def _parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-folder", type=Path, default=script_dir / "VoxCPM_ONNX_Raw")
    parser.add_argument("--final-folder", type=Path, default=script_dir / "VoxCPM_ONNX")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    report = rewrite_voxcpm2_onnx_folder(arguments.raw_folder, arguments.final_folder)
    print(json.dumps(report, indent=2, sort_keys=True))
