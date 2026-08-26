"""Internal composition helpers for the ZipVoice ONNX export pipeline."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_REPRESENTATIVE_DIMS = {
    "audio_channels": 1,
    "audio_samples": 38144,
    "prompt_frames": 94,
    "prompt_tokens": 2,
    "target_tokens": 4,
    "total_frames": 200,
}
_SHAPE_OPS = {"Shape", "Size", "Range", "ConstantOfShape"}
_LAYOUT_OPS = {
    "Transpose",
    "Reshape",
    "Flatten",
    "Squeeze",
    "Unsqueeze",
}
_INDEXING_OPS = {
    "Gather",
    "GatherElements",
    "GatherND",
    "ScatterElements",
    "ScatterND",
    "Slice",
    "Split",
}
_ELEMENTWISE_OPS = {
    "Abs",
    "Add",
    "And",
    "Cast",
    "Clip",
    "Div",
    "Equal",
    "Erf",
    "Exp",
    "Greater",
    "Less",
    "Log",
    "Mul",
    "Neg",
    "Not",
    "Or",
    "Pow",
    "Sigmoid",
    "Softplus",
    "Sqrt",
    "Sub",
    "Tanh",
    "Where",
}
_COPY_CANDIDATES = {
    "Concat",
    "Expand",
    "Pad",
    "Reshape",
    "ScatterND",
    "Tile",
    "Transpose",
}


def _nested_graphs(graph: onnx.GraphProto) -> list[onnx.GraphProto]:
    graphs = [graph]
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                graphs.extend(_nested_graphs(attribute.g))
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    graphs.extend(_nested_graphs(nested))
    return graphs


def _value_nbytes(value: onnx.ValueInfoProto) -> int | None:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape") or tensor_type.elem_type == 0:
        return None
    elements = 1
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            elements *= dimension.dim_value
        elif dimension.dim_param in _REPRESENTATIVE_DIMS:
            elements *= _REPRESENTATIVE_DIMS[dimension.dim_param]
        else:
            return None
    try:
        itemsize = np.dtype(
            helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
        ).itemsize
    except (KeyError, TypeError):
        return None
    return elements * itemsize


def _initializer_nbytes(initializer: onnx.TensorProto) -> int:
    if initializer.data_type == TensorProto.STRING:
        return sum(len(value) for value in initializer.string_data)
    try:
        itemsize = np.dtype(
            helper.tensor_dtype_to_np_dtype(initializer.data_type)
        ).itemsize
    except (KeyError, TypeError):
        return len(initializer.raw_data)
    return math.prod(initializer.dims) * itemsize


def _value_contract(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    shape: list[int | str] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(dimension.dim_value)
        else:
            shape.append(dimension.dim_param or "?")
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": shape,
    }


def graph_metrics(model: onnx.ModelProto) -> dict[str, Any]:
    graphs = _nested_graphs(model.graph)
    nodes = [node for graph in graphs for node in graph.node]
    initializers = [
        initializer for graph in graphs for initializer in graph.initializer
    ]
    histogram = Counter(node.op_type for node in nodes)
    known_bytes: dict[str, int] = {}
    for graph in graphs:
        for value in (*graph.input, *graph.output, *graph.value_info):
            nbytes = _value_nbytes(value)
            if nbytes is not None:
                known_bytes[value.name] = nbytes
        for initializer in graph.initializer:
            known_bytes[initializer.name] = _initializer_nbytes(initializer)
    tensor_edges = [name for node in nodes for name in (*node.input, *node.output)]
    known_tensor_edges = [name for name in tensor_edges if name in known_bytes]
    custom_nodes = sorted(
        {
            f"{node.domain or 'ai.onnx'}::{node.op_type}"
            for node in nodes
            if node.domain not in ("", "ai.onnx")
        }
    )
    return {
        "nodes_recursive": len(nodes),
        "initializers_recursive": len(initializers),
        "initializer_bytes": sum(
            _initializer_nbytes(initializer) for initializer in initializers
        ),
        "operator_histogram": dict(sorted(histogram.items())),
        "cast_ops": histogram["Cast"],
        "shape_ops": sum(histogram[name] for name in _SHAPE_OPS),
        "layout_ops": sum(histogram[name] for name in _LAYOUT_OPS),
        "indexing_ops": sum(histogram[name] for name in _INDEXING_OPS),
        "elementwise_ops": sum(histogram[name] for name in _ELEMENTWISE_OPS),
        "materializing_copy_candidates": sum(
            histogram[name] for name in _COPY_CANDIDATES
        ),
        "known_tensor_edge_bytes": sum(
            known_bytes[name] for name in known_tensor_edges
        ),
        "known_tensor_edge_coverage": (
            len(known_tensor_edges) / len(tensor_edges) if tensor_edges else 1.0
        ),
        "inputs": [_value_contract(value) for value in model.graph.input],
        "outputs": [_value_contract(value) for value in model.graph.output],
        "opsets": {
            opset.domain or "ai.onnx": int(opset.version)
            for opset in model.opset_import
        },
        "custom_or_fallback_nodes": custom_nodes,
    }


def _print_graph_metrics(label: str, model: onnx.ModelProto) -> None:
    print(f"GRAPH_METRICS {label} {json.dumps(graph_metrics(model), sort_keys=True)}")


def _prefix_graph(
    model: onnx.ModelProto,
    prefix: str,
    public_names: set[str],
    remap: dict[str, str] | None = None,
) -> onnx.ModelProto:
    model = copy.deepcopy(model)
    remap = remap or {}
    rename: dict[str, str] = {}

    def mapped(name: str) -> str:
        if name in remap:
            return remap[name]
        if not name or name in public_names:
            return name
        return rename.setdefault(name, prefix + name)

    def prefix_nested_graph(graph: onnx.GraphProto) -> None:
        graph.name = prefix + graph.name
        for value in (*graph.input, *graph.output, *graph.value_info):
            value.name = mapped(value.name)
        for initializer in graph.initializer:
            initializer.name = mapped(initializer.name)
        for node in graph.node:
            if node.name:
                node.name = prefix + node.name
            for index, name in enumerate(node.input):
                node.input[index] = mapped(name)
            for index, name in enumerate(node.output):
                node.output[index] = mapped(name)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    prefix_nested_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        prefix_nested_graph(nested)

    prefix_nested_graph(model.graph)
    return model


def _merge_opsets(destination: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    versions: dict[str, int] = {}
    for source in sources:
        for opset in source.opset_import:
            versions[opset.domain] = max(
                versions.get(opset.domain, 0),
                int(opset.version),
            )
    for domain, version in sorted(versions.items()):
        destination.opset_import.append(helper.make_opsetid(domain, version))


def _value_by_name(values: list[onnx.ValueInfoProto], name: str) -> onnx.ValueInfoProto:
    return next(value for value in values if value.name == name)


def _remove_redundant_casts(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, int]:
    if not any(
        node.op_type == "Cast"
        for graph in _nested_graphs(model.graph)
        for node in graph.node
    ):
        return model, 0

    inferred = onnx.shape_inference.infer_shapes(
        model,
        strict_mode=False,
        data_prop=False,
    )
    removed = 0
    inferred_types: dict[tuple[int, ...], dict[str, int]] = {}

    def collect_types(
        graph: onnx.GraphProto,
        path: tuple[int, ...],
        outer_types: dict[str, int],
    ) -> None:
        element_types = dict(outer_types)
        for value in (*graph.input, *graph.output, *graph.value_info):
            if value.type.HasField("tensor_type"):
                element_types[value.name] = value.type.tensor_type.elem_type
        element_types.update(
            (initializer.name, initializer.data_type)
            for initializer in graph.initializer
        )
        inferred_types[path] = element_types
        for node_index, node in enumerate(graph.node):
            for attribute_index, attribute in enumerate(node.attribute):
                if attribute.type == onnx.AttributeProto.GRAPH:
                    collect_types(
                        attribute.g,
                        (*path, node_index, attribute_index, -1),
                        element_types,
                    )
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph_index, nested in enumerate(attribute.graphs):
                        collect_types(
                            nested,
                            (*path, node_index, attribute_index, graph_index),
                            element_types,
                        )

    collect_types(inferred.graph, (), {})

    def replace_captured_inputs(
        graph: onnx.GraphProto,
        replacements: dict[str, str],
    ) -> None:
        local_names = {value.name for value in graph.input}
        local_names.update(initializer.name for initializer in graph.initializer)
        local_names.update(output for node in graph.node for output in node.output)
        active = {
            source: replacement
            for source, replacement in replacements.items()
            if source not in local_names
        }
        if not active:
            return
        for output in graph.output:
            output.name = active.get(output.name, output.name)
        retained_value_info = [
            value for value in graph.value_info if value.name not in active
        ]
        if len(retained_value_info) != len(graph.value_info):
            del graph.value_info[:]
            graph.value_info.extend(retained_value_info)
        for node in graph.node:
            for index, name in enumerate(node.input):
                node.input[index] = active.get(name, name)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    replace_captured_inputs(attribute.g, active)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        replace_captured_inputs(nested, active)

    def rewrite(graph: onnx.GraphProto, path: tuple[int, ...]) -> None:
        nonlocal removed
        element_types = inferred_types[path]

        for node_index, node in enumerate(graph.node):
            for attribute_index, attribute in enumerate(node.attribute):
                if attribute.type == onnx.AttributeProto.GRAPH:
                    rewrite(
                        attribute.g,
                        (*path, node_index, attribute_index, -1),
                    )
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for graph_index, nested in enumerate(attribute.graphs):
                        rewrite(
                            nested,
                            (*path, node_index, attribute_index, graph_index),
                        )

        graph_outputs = {value.name for value in graph.output}
        replacements: dict[str, str] = {}
        retained_nodes: list[onnx.NodeProto] = []

        def resolve(name: str) -> str:
            while name in replacements:
                name = replacements[name]
            return name

        for node in graph.node:
            target_type = next(
                (
                    attribute.i
                    for attribute in node.attribute
                    if attribute.name == "to"
                ),
                None,
            )
            if (
                node.domain in ("", "ai.onnx")
                and node.op_type == "Cast"
                and len(node.input) == 1
                and len(node.output) == 1
                and target_type is not None
                and element_types.get(resolve(node.input[0])) == target_type
            ):
                if node.output[0] in graph_outputs:
                    node.op_type = "Identity"
                    del node.attribute[:]
                    retained_nodes.append(node)
                else:
                    replacements[node.output[0]] = resolve(node.input[0])
                removed += 1
                continue
            retained_nodes.append(node)

        if replacements:
            for node in retained_nodes:
                for index, name in enumerate(node.input):
                    node.input[index] = resolve(name)
                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.GRAPH:
                        replace_captured_inputs(attribute.g, replacements)
                    elif attribute.type == onnx.AttributeProto.GRAPHS:
                        for nested in attribute.graphs:
                            replace_captured_inputs(nested, replacements)
            retained_value_info = [
                value
                for value in graph.value_info
                if value.name not in replacements
            ]
            del graph.value_info[:]
            graph.value_info.extend(retained_value_info)
            del graph.node[:]
            graph.node.extend(retained_nodes)

    rewrite(model.graph, ())
    return model, removed


def _loop_body(flow: onnx.ModelProto) -> onnx.GraphProto:
    flow_inputs = list(flow.graph.input)
    flow_input_names = {value.name for value in flow_inputs}
    flow_output = flow.graph.output[0]
    state_input = copy.deepcopy(_value_by_name(flow_inputs, "loop_state_in"))
    iteration = helper.make_tensor_value_info("loop_iteration", TensorProto.INT64, [])
    condition_in = helper.make_tensor_value_info("loop_condition_in", TensorProto.BOOL, [])
    condition_out = helper.make_tensor_value_info(
        "loop_condition_out",
        TensorProto.BOOL,
        [],
    )

    control_nodes = []
    if "loop_t" in flow_input_names:
        control_nodes.append(
            helper.make_node(
                "Gather",
                ["pipeline_timesteps", "loop_iteration"],
                ["loop_t"],
                name="loop/GatherTimestep",
            )
        )
    control_nodes.extend(
        [
            helper.make_node(
                "Gather",
                ["pipeline_deltas", "loop_iteration"],
                ["loop_delta"],
                name="loop/GatherDelta",
            ),
            helper.make_node(
                "Gather",
                ["pipeline_time_embeddings", "loop_iteration"],
                ["loop_time_embeddings"],
                name="loop/GatherTimeEmbedding",
            ),
            helper.make_node(
                "Identity",
                ["loop_condition_in"],
                ["loop_condition_out"],
                name="loop/Condition",
            ),
        ]
    )
    return helper.make_graph(
        [*control_nodes, *copy.deepcopy(flow.graph.node)],
        "ZipVoiceFlowLoopBody",
        [iteration, condition_in, state_input],
        [condition_out, copy.deepcopy(flow_output)],
        value_info=copy.deepcopy(flow.graph.value_info),
    )


def _merge_pipeline(
    onnx_folder: Path,
    destination: Path | None = None,
    verbose: bool = False,
) -> Path:
    metadata_path, = tuple(onnx_folder.glob("*_Metadata.onnx"))
    package_stem = metadata_path.name.removesuffix("_Metadata.onnx")
    preprocess_path = onnx_folder / f"{package_stem}_Preprocess.onnx"
    text_encoder_path = onnx_folder / f"{package_stem}_TextEncoder.onnx"
    flow_condition_path = onnx_folder / f"{package_stem}_FlowCondition.onnx"
    flow_geometry_path = onnx_folder / f"{package_stem}_FlowGeometry.onnx"
    time_embedding_path = onnx_folder / f"{package_stem}_TimeEmbedding.onnx"
    flow_path = onnx_folder / f"{package_stem}_FlowStep.onnx"
    decode_path = onnx_folder / f"{package_stem}_Decode.onnx"
    if destination is None:
        destination = onnx_folder / f"{package_stem}_Pipeline.onnx"

    raw_preprocess, preprocess_casts = _remove_redundant_casts(
        onnx.load(str(preprocess_path))
    )
    raw_text_encoder, text_encoder_casts = _remove_redundant_casts(
        onnx.load(str(text_encoder_path))
    )
    raw_flow_condition, flow_condition_casts = _remove_redundant_casts(
        onnx.load(str(flow_condition_path))
    )
    raw_flow_geometry, flow_geometry_casts = _remove_redundant_casts(
        onnx.load(str(flow_geometry_path))
    )
    raw_time_embedding, time_embedding_casts = _remove_redundant_casts(
        onnx.load(str(time_embedding_path))
    )
    raw_flow, flow_casts = _remove_redundant_casts(onnx.load(str(flow_path)))
    raw_decode, decode_casts = _remove_redundant_casts(
        onnx.load(str(decode_path))
    )
    if verbose:
        print(
            "Removed redundant Cast nodes: "
            f"Preprocess={preprocess_casts}, "
            f"TextEncoder={text_encoder_casts}, "
            f"FlowCondition={flow_condition_casts}, "
            f"FlowGeometry={flow_geometry_casts}, "
            f"TimeEmbedding={time_embedding_casts}, "
            f"FlowStep={flow_casts}, "
            f"Decode={decode_casts}"
        )
        for label, model in (
            ("Preprocess", raw_preprocess),
            ("TextEncoder", raw_text_encoder),
            ("FlowCondition", raw_flow_condition),
            ("FlowGeometry", raw_flow_geometry),
            ("TimeEmbedding", raw_time_embedding),
            ("FlowStep", raw_flow),
            ("Decode", raw_decode),
        ):
            _print_graph_metrics(label, model)

    preprocess_public = {
        value.name
        for value in (*raw_preprocess.graph.input, *raw_preprocess.graph.output)
    }
    text_encoder_public = {
        value.name
        for value in (*raw_text_encoder.graph.input, *raw_text_encoder.graph.output)
    }
    flow_condition_public = {
        value.name
        for value in (
            *raw_flow_condition.graph.input,
            *raw_flow_condition.graph.output,
        )
    }
    flow_geometry_public = {
        value.name
        for value in (
            *raw_flow_geometry.graph.input,
            *raw_flow_geometry.graph.output,
        )
    }
    decode_public = {
        value.name for value in (*raw_decode.graph.input, *raw_decode.graph.output)
    }
    flow_remap = {
        "delta_t": "loop_delta",
        "x": "loop_state_in",
        "condition_projection": "condition_projection",
        "time_embeddings": "loop_time_embeddings",
        "projected_positions_1": "projected_positions_1",
        "relative_indices_1": "relative_indices_1",
        "projected_positions_2": "projected_positions_2",
        "relative_indices_2": "relative_indices_2",
        "projected_positions_4": "projected_positions_4",
        "relative_indices_4": "relative_indices_4",
        "t": "loop_t",
        "speech_projection": "speech_projection",
        "guidance_scale": "guidance_scale",
        "x_next": "loop_state_out",
    }
    time_embedding_remap = {
        "timesteps": "pipeline_timesteps",
        "guidance_scale": "guidance_scale",
        "time_embeddings": "pipeline_time_embeddings",
    }
    preprocess = _prefix_graph(
        raw_preprocess,
        "preprocess/",
        preprocess_public,
    )
    text_encoder = _prefix_graph(
        raw_text_encoder,
        "text_encoder/",
        text_encoder_public,
    )
    flow_condition = _prefix_graph(
        raw_flow_condition,
        "flow_condition/",
        flow_condition_public,
    )
    flow_geometry = _prefix_graph(
        raw_flow_geometry,
        "flow_geometry/",
        flow_geometry_public,
    )
    time_embedding = _prefix_graph(
        raw_time_embedding,
        "time_embedding/",
        set(),
        time_embedding_remap,
    )
    flow = _prefix_graph(raw_flow, "flow/", set(), flow_remap)
    decode = _prefix_graph(raw_decode, "decode/", decode_public)

    merged = onnx.ModelProto()
    merged.ir_version = max(
        preprocess.ir_version,
        text_encoder.ir_version,
        flow_condition.ir_version,
        flow_geometry.ir_version,
        time_embedding.ir_version,
        flow.ir_version,
        decode.ir_version,
    )
    merged.producer_name = Path(__file__).name
    merged.graph.name = f"{package_stem}_Pipeline"
    _merge_opsets(
        merged,
        preprocess,
        text_encoder,
        flow_condition,
        flow_geometry,
        time_embedding,
        flow,
        decode,
    )
    guidance_component = (
        time_embedding
        if any(
            value.name == "guidance_scale"
            for value in time_embedding.graph.input
        )
        else flow
    )

    merged.graph.input.extend(
        copy.deepcopy(value) for value in preprocess.graph.input
    )
    merged.graph.input.extend(
        copy.deepcopy(value)
        for value in text_encoder.graph.input
        if value.name not in {"prompt_features", "prompt_features_len"}
    )
    merged.graph.input.extend(
        [
            helper.make_tensor_value_info("num_step", TensorProto.INT64, []),
            copy.deepcopy(
                _value_by_name(
                    list(guidance_component.graph.input),
                    "guidance_scale",
                )
            ),
            helper.make_tensor_value_info("t_shift", TensorProto.FLOAT, []),
        ]
    )
    merged.graph.output.extend(copy.deepcopy(value) for value in decode.graph.output)
    merged.graph.initializer.extend(
        copy.deepcopy(initializer)
        for initializer in (
            *preprocess.graph.initializer,
            *text_encoder.graph.initializer,
            *flow_condition.graph.initializer,
            *flow_geometry.graph.initializer,
            *time_embedding.graph.initializer,
            *flow.graph.initializer,
            *decode.graph.initializer,
        )
    )
    merged.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array(True), name="loop_condition"),
            numpy_helper.from_array(np.array(0, dtype=np.int64), name="loop_zero_i64"),
            numpy_helper.from_array(np.array(1, dtype=np.int64), name="loop_one_i64"),
            numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="loop_one_f32"),
        ]
    )
    merged.graph.node.extend(copy.deepcopy(node) for node in preprocess.graph.node)
    merged.graph.node.extend(copy.deepcopy(node) for node in text_encoder.graph.node)
    merged.graph.node.extend(
        copy.deepcopy(node) for node in flow_condition.graph.node
    )
    merged.graph.node.extend(
        copy.deepcopy(node) for node in flow_geometry.graph.node
    )
    merged.graph.node.extend(
        [
            helper.make_node(
                "Cast",
                ["num_step"],
                ["pipeline_num_step_f32"],
                to=TensorProto.FLOAT,
                name="pipeline/CastNumStep",
            ),
            helper.make_node(
                "Div",
                ["loop_one_f32", "pipeline_num_step_f32"],
                ["pipeline_linear_step"],
                name="pipeline/LinearStep",
            ),
            helper.make_node(
                "Sub",
                ["t_shift", "loop_one_f32"],
                ["pipeline_shift_minus_one"],
                name="pipeline/ShiftMinusOne",
            ),
            helper.make_node(
                "Range",
                ["loop_zero_i64", "num_step", "loop_one_i64"],
                ["pipeline_iterations"],
                name="pipeline/Iterations",
            ),
            helper.make_node(
                "Add",
                ["pipeline_iterations", "loop_one_i64"],
                ["pipeline_next_iterations"],
                name="pipeline/NextIterations",
            ),
            helper.make_node(
                "Cast",
                ["pipeline_iterations"],
                ["pipeline_iterations_f32"],
                to=TensorProto.FLOAT,
                name="pipeline/CastIterations",
            ),
            helper.make_node(
                "Cast",
                ["pipeline_next_iterations"],
                ["pipeline_next_iterations_f32"],
                to=TensorProto.FLOAT,
                name="pipeline/CastNextIterations",
            ),
            helper.make_node(
                "Mul",
                ["pipeline_iterations_f32", "pipeline_linear_step"],
                ["pipeline_linear_timesteps"],
                name="pipeline/LinearTimesteps",
            ),
            helper.make_node(
                "Mul",
                ["pipeline_next_iterations_f32", "pipeline_linear_step"],
                ["pipeline_linear_timesteps_next"],
                name="pipeline/LinearTimestepsNext",
            ),
        ]
    )
    for suffix, linear_input, output in (
        ("", "pipeline_linear_timesteps", "pipeline_timesteps"),
        ("Next", "pipeline_linear_timesteps_next", "pipeline_timesteps_next"),
    ):
        merged.graph.node.extend(
            [
                helper.make_node(
                    "Mul",
                    ["pipeline_shift_minus_one", linear_input],
                    [f"pipeline_shift_products{suffix}"],
                    name=f"pipeline/ShiftProducts{suffix}",
                ),
                helper.make_node(
                    "Add",
                    ["loop_one_f32", f"pipeline_shift_products{suffix}"],
                    [f"pipeline_shift_denominators{suffix}"],
                    name=f"pipeline/ShiftDenominators{suffix}",
                ),
                helper.make_node(
                    "Mul",
                    ["t_shift", linear_input],
                    [f"pipeline_shift_numerators{suffix}"],
                    name=f"pipeline/ShiftNumerators{suffix}",
                ),
                helper.make_node(
                    "Div",
                    [
                        f"pipeline_shift_numerators{suffix}",
                        f"pipeline_shift_denominators{suffix}",
                    ],
                    [output],
                    name=f"pipeline/ShiftTimesteps{suffix}",
                ),
            ]
        )
    merged.graph.node.append(
        helper.make_node(
            "Sub",
            ["pipeline_timesteps_next", "pipeline_timesteps"],
            ["pipeline_deltas"],
            name="pipeline/Deltas",
        )
    )
    merged.graph.node.extend(
        copy.deepcopy(node) for node in time_embedding.graph.node
    )
    merged.graph.node.append(
        helper.make_node(
            "Loop",
            ["num_step", "loop_condition", "initial_noise"],
            ["final_features"],
            body=_loop_body(flow),
            name="FlowMatchingLoop",
        )
    )
    merged.graph.node.extend(copy.deepcopy(node) for node in decode.graph.node)
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in preprocess.graph.value_info
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in preprocess.graph.output
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in text_encoder.graph.value_info
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in text_encoder.graph.output
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in flow_condition.graph.value_info
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in flow_condition.graph.output
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in flow_geometry.graph.value_info
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in flow_geometry.graph.output
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in time_embedding.graph.value_info
    )
    merged.graph.value_info.extend(
        copy.deepcopy(value) for value in time_embedding.graph.output
    )
    final_features = copy.deepcopy(flow.graph.output[0])
    final_features.name = "final_features"
    merged.graph.value_info.append(final_features)
    merged.graph.value_info.extend(copy.deepcopy(value) for value in decode.graph.value_info)
    if verbose:
        _print_graph_metrics("Pipeline", merged)

    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_name(destination.name + ".data")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        onnx.save_model(merged, str(temporary_path))
        onnx.checker.check_model(str(temporary_path))
        os.replace(temporary_path, destination)
        temporary_path = None
        sidecar.unlink(missing_ok=True)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    metadata_destination = destination.with_name(
        f"{package_stem}_Metadata.onnx"
    )
    if metadata_path.resolve() != metadata_destination.resolve():
        shutil.copy2(metadata_path, metadata_destination)
        metadata_sidecar = metadata_path.with_name(metadata_path.name + ".data")
        destination_sidecar = metadata_destination.with_name(
            metadata_destination.name + ".data"
        )
        if metadata_sidecar.is_file():
            shutil.copy2(metadata_sidecar, destination_sidecar)
        else:
            destination_sidecar.unlink(missing_ok=True)
    if verbose:
        print(
            "Composed end-to-end package -> "
            f"{destination.name}, {metadata_destination.name}"
        )
    return destination
