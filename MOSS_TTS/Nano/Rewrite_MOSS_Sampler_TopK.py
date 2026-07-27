"""Targeted ONNX rewrite for the MOSS sampler's runtime-controlled TopK.

The legacy PyTorch exporter cannot pass a tensor-valued ``top_k`` to ``torch.topk``. The source graph therefore
sorts the complete vocabulary and masks positions beyond ``top_k``. This rewrite keeps the raw export untouched,
feeds ``min(top_k, vocab_size)`` to the existing standard ONNX TopK, and removes only the now-dead tail-mask path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REWRITE_METADATA_KEY = "moss_sampler_dynamic_topk_rewrite"
REWRITE_METADATA_VALUE = "2"


class SamplerRewriteError(RuntimeError):
    """Raised when the raw graph does not satisfy every rewrite precondition."""


def _attribute(node, name, default=None):
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _constant_array(node):
    if node is None or node.domain not in ("", "ai.onnx") or node.op_type != "Constant":
        raise SamplerRewriteError("Expected a standard Constant node.")
    value_attributes = [attribute for attribute in node.attribute if attribute.name == "value"]
    if len(value_attributes) != 1:
        raise SamplerRewriteError("Expected Constant to contain exactly one tensor-valued 'value' attribute.")
    return numpy_helper.to_array(helper.get_attribute_value(value_attributes[0]))


def _tensor_spec(value_info):
    if not value_info.type.HasField("tensor_type"):
        raise SamplerRewriteError(f"{value_info.name!r} is not a tensor value.")
    tensor_type = value_info.type.tensor_type
    return tensor_type.elem_type, len(tensor_type.shape.dim), tensor_type.shape.dim


def _unique_name(existing, base):
    if base not in existing:
        existing.add(base)
        return base
    suffix = 1
    while f"{base}_{suffix}" in existing:
        suffix += 1
    name = f"{base}_{suffix}"
    existing.add(name)
    return name


def _build_maps(nodes):
    producers = {}
    consumers = {}
    for index, node in enumerate(nodes):
        for output in node.output:
            if output in producers:
                raise SamplerRewriteError(f"Tensor {output!r} has multiple producers.")
            producers[output] = index
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(index)
    return producers, consumers


def _require_constant(producers, nodes, tensor_name, expected, dtype):
    producer_index = producers.get(tensor_name)
    if producer_index is None:
        raise SamplerRewriteError(f"Expected {tensor_name!r} to be produced by Constant.")
    value = _constant_array(nodes[producer_index])
    expected_array = np.asarray(expected, dtype=dtype)
    if value.dtype != expected_array.dtype or value.shape != expected_array.shape or not np.array_equal(value, expected_array):
        raise SamplerRewriteError(
            f"Constant {tensor_name!r} mismatch: expected {expected_array!r}, got dtype={value.dtype}, "
            f"shape={value.shape}, value={value!r}."
        )
    return producer_index


def _validate_slice_path(nodes, producers, topk_index, slice_index):
    slice_node = nodes[slice_index]
    if slice_node.domain in ("", "ai.onnx") and slice_node.op_type == "Constant":
        positions = _constant_array(slice_node)
        if positions.dtype != np.int32 or positions.ndim != 2 or positions.shape[0] != 1:
            raise SamplerRewriteError("Folded top-k position table must be int32 with shape [1, vocab_size].")
        expected_positions = np.arange(positions.shape[1], dtype=np.int32).reshape(1, -1)
        if not np.array_equal(positions, expected_positions):
            raise SamplerRewriteError("Folded top-k position table is not the exact contiguous range.")
        return {slice_index}
    if slice_node.domain not in ("", "ai.onnx") or slice_node.op_type != "Slice" or len(slice_node.input) != 5:
        raise SamplerRewriteError("Top-k position mask must use an exact Constant or five-input standard Slice.")

    positions_index = producers.get(slice_node.input[0])
    positions = _constant_array(nodes[positions_index] if positions_index is not None else None)
    if positions.dtype != np.int32 or positions.ndim != 2 or positions.shape[0] != 1:
        raise SamplerRewriteError("Top-k position table must be int32 with shape [1, max_vocab_size].")
    expected_positions = np.arange(positions.shape[1], dtype=np.int32).reshape(1, -1)
    if not np.array_equal(positions, expected_positions):
        raise SamplerRewriteError("Top-k position table is not the exact contiguous range [0, max_vocab_size).")

    constant_indices = {
        _require_constant(producers, nodes, slice_node.input[1], [0], np.int64),
        _require_constant(producers, nodes, slice_node.input[3], [1], np.int64),
        _require_constant(producers, nodes, slice_node.input[4], [1], np.int64),
    }
    end_index = producers.get(slice_node.input[2])
    if end_index is None or nodes[end_index].op_type != "Unsqueeze" or len(nodes[end_index].input) != 2:
        raise SamplerRewriteError("Top-k position Slice end must be Unsqueeze(Gather(Shape(TopK values), 1)).")
    unsqueeze = nodes[end_index]
    constant_indices.add(_require_constant(producers, nodes, unsqueeze.input[1], [0], np.int64))
    gather_index = producers.get(unsqueeze.input[0])
    if gather_index is None or nodes[gather_index].op_type != "Gather":
        raise SamplerRewriteError("Top-k position Slice end must gather the TopK values width.")
    gather = nodes[gather_index]
    if _attribute(gather, "axis", 0) != 0:
        raise SamplerRewriteError("Top-k width Gather must use axis 0.")
    constant_indices.add(_require_constant(producers, nodes, gather.input[1], 1, np.int64))
    shape_index = producers.get(gather.input[0])
    if shape_index is None or nodes[shape_index].op_type != "Shape":
        raise SamplerRewriteError("Top-k width Gather must consume Shape(TopK values).")
    if nodes[shape_index].input != [nodes[topk_index].output[0]]:
        raise SamplerRewriteError("Top-k width Shape does not consume the matched TopK values output.")
    return {slice_index, positions_index, end_index, gather_index, shape_index, *constant_indices}


def rewrite_sampler_dynamic_topk(raw_path, final_path, expected_match_count=1):
    """Rewrite every expected full-sort/tail-mask sampler in an arbitrary compact component."""
    raw_path = Path(raw_path).resolve()
    final_path = Path(final_path).resolve()
    if raw_path == final_path:
        raise SamplerRewriteError("Raw and final sampler paths must be distinct.")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if expected_match_count <= 0:
        raise ValueError(f"expected_match_count must be positive, got {expected_match_count}.")

    model = onnx.load(str(raw_path), load_external_data=True)
    onnx.checker.check_model(model)
    if any(prop.key == REWRITE_METADATA_KEY for prop in model.metadata_props):
        raise SamplerRewriteError("Input model is already marked as dynamically rewritten.")
    inferred_model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    value_specs = {
        value.name: _tensor_spec(value)
        for value in [
            *inferred_model.graph.input,
            *inferred_model.graph.output,
            *inferred_model.graph.value_info,
        ]
        if value.name
    }

    standard_opset = next(
        (entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")),
        None,
    )

    input_map = {value.name: value for value in model.graph.input}
    output_names_before = [value.name for value in model.graph.output]
    interface_before = [value.SerializeToString() for value in [*model.graph.input, *model.graph.output]]

    nodes = list(model.graph.node)
    producers, consumers = _build_maps(nodes)
    candidates = []
    for mask_index, mask in enumerate(nodes):
        if mask.domain not in ("", "ai.onnx") or mask.op_type != "Where" or len(mask.input) != 3:
            continue
        less_index = producers.get(mask.input[0])
        topk_index = producers.get(mask.input[1])
        false_index = producers.get(mask.input[2])
        if less_index is None or topk_index is None or false_index is None:
            continue
        less = nodes[less_index]
        topk = nodes[topk_index]
        if less.domain not in ("", "ai.onnx") or less.op_type != "Less" or len(less.input) != 2:
            continue
        top_k_name = less.input[1]
        top_k_value = input_map.get(top_k_name)
        if top_k_value is None:
            continue
        if topk.op_type != "TopK" or not topk.output or mask.input[1] != topk.output[0]:
            continue
        false_value = _constant_array(nodes[false_index])
        if false_value.dtype != np.float32 or false_value.shape != () or not np.isneginf(false_value):
            continue
        candidates.append({
            "mask": mask_index,
            "less": less_index,
            "topk": topk_index,
            "false": false_index,
            "top_k_name": top_k_name,
        })
    if len(candidates) != expected_match_count:
        raise SamplerRewriteError(
            f"Expected exactly {expected_match_count} full-sort/tail-mask matches, found {len(candidates)}."
        )

    topk_indices = [candidate["topk"] for candidate in candidates]
    mask_indices = [candidate["mask"] for candidate in candidates]
    if len(set(topk_indices)) != len(topk_indices) or len(set(mask_indices)) != len(mask_indices):
        raise SamplerRewriteError("Sampler matches overlap on TopK or tail-mask nodes.")

    local_region = set()
    expected_less_by_input = {}
    original_k_inputs = {}
    for candidate in candidates:
        mask_index = candidate["mask"]
        less_index = candidate["less"]
        topk_index = candidate["topk"]
        false_index = candidate["false"]
        top_k_name = candidate["top_k_name"]
        mask = nodes[mask_index]
        less = nodes[less_index]
        topk = nodes[topk_index]

        top_k_dtype, top_k_rank, top_k_dims = _tensor_spec(input_map[top_k_name])
        if top_k_dtype != TensorProto.INT32 or top_k_rank != 1 or top_k_dims[0].dim_value != 1:
            raise SamplerRewriteError(f"{top_k_name} must be int32 with static shape [1].")
        if topk.domain not in ("", "ai.onnx") or len(topk.input) != 2 or len(topk.output) != 2:
            raise SamplerRewriteError("Matched TopK must be a standard two-input/two-output node.")
        if _attribute(topk, "axis", -1) != -1 or _attribute(topk, "largest", 1) != 1:
            raise SamplerRewriteError("Matched TopK must sort largest values along the final axis.")
        if _attribute(topk, "sorted", 1) != 1:
            raise SamplerRewriteError("Matched TopK must produce sorted outputs.")
        expected_specs = {
            topk.input[0]: (TensorProto.FLOAT, 2),
            topk.input[1]: (TensorProto.INT64, 1),
            topk.output[0]: (TensorProto.FLOAT, 2),
            topk.output[1]: (TensorProto.INT64, 2),
            mask.output[0]: (TensorProto.FLOAT, 2),
        }
        for tensor_name, expected_spec in expected_specs.items():
            actual_spec = value_specs.get(tensor_name)
            if actual_spec is None or actual_spec[:2] != expected_spec:
                raise SamplerRewriteError(
                    f"Tensor {tensor_name!r} must have dtype/rank {expected_spec}, "
                    f"got {None if actual_spec is None else actual_spec[:2]}."
                )

        original_k_inputs[topk_index] = topk.input[1]
        k_gather_index = producers.get(topk.input[1])
        if k_gather_index is None or nodes[k_gather_index].op_type != "Gather":
            raise SamplerRewriteError("Full-sort TopK K must be Gather(Shape(scores), [-1]).")
        k_gather = nodes[k_gather_index]
        if _attribute(k_gather, "axis", 0) != 0:
            raise SamplerRewriteError("Full-sort K Gather must use axis 0.")
        _require_constant(producers, nodes, k_gather.input[1], [-1], np.int64)
        k_shape_index = producers.get(k_gather.input[0])
        if k_shape_index is None or nodes[k_shape_index].op_type != "Shape":
            raise SamplerRewriteError("Full-sort K Gather must consume Shape(scores).")
        if nodes[k_shape_index].input != [topk.input[0]]:
            raise SamplerRewriteError("Full-sort K Shape does not consume the matched TopK scores input.")

        slice_index = producers.get(less.input[0])
        if slice_index is None:
            raise SamplerRewriteError("Top-k Less input is missing its position Slice producer.")
        local_region.update(_validate_slice_path(nodes, producers, topk_index, slice_index))
        local_region.update({less_index, false_index})

        mask_consumers = consumers.get(mask.output[0], [])
        if Counter(nodes[index].op_type for index in mask_consumers) != Counter({"Div": 1}):
            raise SamplerRewriteError("Each tail-mask output must feed exactly one post-TopK temperature Div.")
        if mask.output[0] in output_names_before:
            raise SamplerRewriteError("Tail-mask tensor unexpectedly appears in the public graph outputs.")
        expected_less_by_input.setdefault(top_k_name, set()).add(less_index)

    for top_k_name, expected_consumers in expected_less_by_input.items():
        if set(consumers.get(top_k_name, [])) != expected_consumers:
            raise SamplerRewriteError(f"{top_k_name} has consumers outside the matched tail-mask conditions.")

    rewired_consumers = 0
    for candidate in candidates:
        mask_output = nodes[candidate["mask"]].output[0]
        topk_output = nodes[candidate["topk"]].output[0]
        for node in nodes:
            for input_index, input_name in enumerate(node.input):
                if input_name == mask_output:
                    node.input[input_index] = topk_output
                    rewired_consumers += 1
    expected_rewires = expected_match_count
    if rewired_consumers != expected_rewires:
        raise SamplerRewriteError(
            f"Expected to rewire {expected_rewires} tail-mask consumers, rewired {rewired_consumers}."
        )

    existing_names = {
        name
        for value in [*model.graph.input, *model.graph.output, *model.graph.value_info, *model.graph.initializer]
        for name in [value.name]
    }
    existing_names.update(node.name for node in nodes if node.name)
    existing_names.update(output for node in nodes for output in node.output)
    cast_nodes = {}
    cast_outputs = {}
    insertion_nodes = {}
    effective_k_by_topk = {}
    for candidate in sorted(candidates, key=lambda item: item["topk"]):
        top_k_name = candidate["top_k_name"]
        topk_index = candidate["topk"]
        if top_k_name not in cast_nodes:
            cast_output = _unique_name(existing_names, f"{top_k_name}_int64")
            cast_nodes[top_k_name] = helper.make_node(
                "Cast",
                [top_k_name],
                [cast_output],
                name=_unique_name(existing_names, f"MossTopKCast_{top_k_name}"),
                to=TensorProto.INT64,
            )
            cast_outputs[top_k_name] = cast_output
        effective_k = _unique_name(existing_names, f"{top_k_name}_effective_k")
        min_node = helper.make_node(
            "Min",
            [cast_outputs[top_k_name], original_k_inputs[topk_index]],
            [effective_k],
            name=_unique_name(existing_names, f"MossTopKClamp_{topk_index}"),
        )
        insertion_nodes[topk_index] = [min_node]
        effective_k_by_topk[topk_index] = effective_k
        nodes[topk_index].input[1] = effective_k
    for top_k_name, cast_node in cast_nodes.items():
        first_topk = min(
            candidate["topk"] for candidate in candidates if candidate["top_k_name"] == top_k_name
        )
        insertion_nodes[first_topk].insert(0, cast_node)

    # Remove only the matched mask and ancestors in its local position/mask branch that become dead.
    active = set(range(len(nodes))) - set(mask_indices)
    graph_outputs = set(output_names_before)
    changed = True
    while changed:
        changed = False
        active_consumers = {}
        for index in active:
            for input_name in nodes[index].input:
                active_consumers.setdefault(input_name, []).append(index)
        for index in sorted(local_region & active):
            if all(output not in graph_outputs and output not in active_consumers for output in nodes[index].output):
                active.remove(index)
                changed = True

    deleted_indices = (set(range(len(nodes))) - active)
    deleted_histogram = Counter(nodes[index].op_type for index in deleted_indices)
    rewritten_nodes = []
    for index, node in enumerate(nodes):
        rewritten_nodes.extend(insertion_nodes.get(index, ()))
        if index in active:
            rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    metadata = {prop.key: prop for prop in model.metadata_props}
    metadata[REWRITE_METADATA_KEY] = model.metadata_props.add(key=REWRITE_METADATA_KEY)
    metadata[REWRITE_METADATA_KEY].value = REWRITE_METADATA_VALUE
    model.metadata_props.add(
        key="moss_sampler_dynamic_topk_raw",
        value=os.path.relpath(raw_path, start=final_path.parent),
    )

    if interface_before != [value.SerializeToString() for value in [*model.graph.input, *model.graph.output]]:
        raise SamplerRewriteError("Graph input/output interface changed during rewrite.")
    onnx.checker.check_model(model)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)

    final_nodes_by_name = {node.name: node for node in model.graph.node if node.name}
    for topk_index, effective_k in effective_k_by_topk.items():
        original = nodes[topk_index]
        final_topk = final_nodes_by_name.get(original.name)
        if final_topk is None or final_topk.op_type != "TopK" or final_topk.input[1] != effective_k:
            raise SamplerRewriteError("A matched TopK was not correctly rewired in the final graph.")
    removed_mask_outputs = {nodes[index].output[0] for index in mask_indices}
    if any(
        name in removed_mask_outputs
        for node in model.graph.node
        for name in (*node.input, *node.output)
    ):
        raise SamplerRewriteError("A removed tail-mask tensor remains in the final graph.")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.", suffix=".tmp.onnx", dir=final_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        onnx.save(model, str(temporary_path))
        saved = onnx.load(str(temporary_path), load_external_data=True)
        onnx.checker.check_model(saved)
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    raw_histogram = Counter(node.op_type for node in nodes)
    final_histogram = Counter(node.op_type for node in model.graph.node)
    return {
        "raw_model": str(raw_path),
        "final_model": str(final_path),
        "matched": expected_match_count,
        "inserted": {"Cast": len(cast_nodes), "Min": expected_match_count},
        "rewired_consumers": rewired_consumers,
        "deleted": dict(sorted(deleted_histogram.items())),
        "raw_nodes": len(nodes),
        "final_nodes": len(model.graph.node),
        "raw_topk_tail_nodes": {
            op: raw_histogram[op] for op in ("TopK", "Less", "Where", "Slice", "RandomUniformLike")
        },
        "final_topk_tail_nodes": {
            op: final_histogram[op] for op in ("TopK", "Less", "Where", "Slice", "RandomUniformLike")
        },
        "interface_preserved": True,
        "standard_domain": "ai.onnx",
        "opset": standard_opset,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Rewrite expected MOSS sampler full-sort graphs to dynamic TopK.")
    parser.add_argument("raw_model", type=Path)
    parser.add_argument("final_model", type=Path)
    parser.add_argument("--expected-match-count", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    print(json.dumps(rewrite_sampler_dynamic_topk(
        arguments.raw_model,
        arguments.final_model,
        expected_match_count=arguments.expected_match_count,
    ), indent=2))