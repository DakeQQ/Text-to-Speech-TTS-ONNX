"""Shared Qwen-style ONNX optimization pipeline for the TTS export scripts."""

from __future__ import annotations

import gc
import hashlib
import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import onnx
import onnx.version_converter
from onnx import TensorProto, helper
from onnxruntime.quantization import QuantType, matmul_nbits_quantizer, quant_utils, quantize_dynamic
from onnxslim import slim


NodeSelector = list[str] | Callable[[str], list[str] | None] | None
IntValue = int | Callable[[str], int]

_WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}
_QUANT_FORMATS = {
    "QOPERATOR": quant_utils.QuantFormat.QOperator,
    "QDQ": quant_utils.QuantFormat.QDQ,
}
_DYNAMIC_WEIGHT_TYPES = {"QUINT8": QuantType.QUInt8, "QINT8": QuantType.QInt8}
_VALID_ALGOS = {"DEFAULT", "RTN", "HQQ", "k_quant"}


@dataclass
class Plan:
    """Per-module optimization recipe; ``None`` fields inherit OptimizerConfig defaults."""

    method: str = "DYNAMIC"  # Q2 | Q4 | Q8 | DYNAMIC | F16 | F32
    # weight-only (Q2/Q4/Q8)
    algo: str | None = None
    op_types: tuple[str, ...] | None = None
    axes: tuple[int, ...] | None = None
    block_size: int | None = None
    accuracy_level: int | None = None
    symmetric: bool | None = None
    quant_format: str | None = None
    # dynamic INT8
    dynamic_weight_type: str | None = None
    per_channel: bool | None = None
    reduce_range: bool | None = None
    default_tensor_type: int | None = None
    # node selection
    nodes_to_exclude: NodeSelector = None
    nodes_to_include: NodeSelector = None
    # optimize / precision
    optimize: bool = True
    transformer: bool = True
    opt_level: int | None = None
    fp16: bool = False
    f16_op_block_list: list[str] | None = None
    num_heads: IntValue = 0
    hidden_size: IntValue = 0
    # storage
    external: bool | None = None
    # onnxslim shape inference knobs
    first_slim_no_shape_infer: bool = True
    second_slim_no_shape_infer: bool | None = None


@dataclass
class OptimizerConfig:
    """Global defaults shared by every module in one Optimize_ONNX.py script."""

    original_folder_path: str
    optimized_folder_path: str
    model_plans: dict[str, Plan]
    # weight-only defaults
    weight_only_algorithm: str = "DEFAULT"
    block_size: int = 32
    accuracy_level: int = 4
    quant_symmetric: bool = False
    quant_format: str = "QOperator"
    # dynamic INT8 defaults
    dynamic_weight_type: str = "QInt8"
    dynamic_per_channel: bool = True
    dynamic_reduce_range: bool = False
    dynamic_default_tensor_type: int | None = None
    # node selection defaults
    nodes_to_exclude: NodeSelector = None
    nodes_to_include: NodeSelector = None
    # storage / opset
    force_external_data: bool = False
    upgrade_opset: int = 0
    # graph optimizer
    optimizer_level: int = 2
    optimizer_model_type: str = "bert"
    optimizer_only_onnxruntime: bool = False
    optimizer_fusion_options: dict | None = None
    optimizer_use_gpu: bool = False
    optimizer_provider: str | None = None
    shape_infer: bool = True
    # onnxslim
    slim_skip_fusion_patterns: list[str] | None = None
    slim_skip_optimizations: list[str] | None = None
    slim_size_threshold: int | None = None
    second_slim_no_shape_infer: bool | None = None
    # float16
    f16_keep_io_types: bool | None = None
    f16_force_initializers: bool = True
    f16_min_positive_val: float = 1e-7
    f16_max_finite_val: float = 32767.0
    f16_node_block_list: list[str] | None = None
    f16_op_block_list: list[str] | None = None
    # convert every optimized *.onnx to ORT format (legacy vocoder / preprocess scripts)
    convert_to_ort: bool = False
    ort_optimization_style: str = "Fixed"
    ort_target_platform: str = "amd64"
    ort_enable_type_reduction: bool = True
    # optional side artifacts copied after all models are processed
    copy_artifacts: tuple[str, ...] = ()


@dataclass
class ResolvedPlan:
    method: str
    algo: str
    op_types: tuple[str, ...]
    axes: tuple[int, ...]
    block_size: int
    accuracy_level: int
    symmetric: bool
    quant_format: str
    dynamic_weight_type: str
    per_channel: bool
    reduce_range: bool
    default_tensor_type: int | None
    nodes_to_exclude: NodeSelector
    nodes_to_include: NodeSelector
    optimize: bool
    transformer: bool
    opt_level: int | None
    fp16: bool
    f16_op_block_list: list[str] | None
    num_heads: IntValue
    hidden_size: IntValue
    external: bool
    first_slim_no_shape_infer: bool
    second_slim_no_shape_infer: bool | None


def _pick(value, default):
    return default if value is None else value


def _uses_fp16(plan: Plan | ResolvedPlan) -> bool:
    return plan.fp16 or plan.method.upper() == "F16"


def uses_mixed_precision(plans: Iterable[Plan | ResolvedPlan]) -> bool:
    fp16_plans = tuple(_uses_fp16(plan) for plan in plans)
    return any(fp16_plans) and not all(fp16_plans)


def _fallback_unsupported_k_quant(rp: ResolvedPlan) -> ResolvedPlan:
    if rp.method not in _WEIGHT_ONLY_BITS or rp.algo != "k_quant":
        return rp

    bits = _WEIGHT_ONLY_BITS[rp.method]
    if bits != 4:
        reason = f"{bits}-bit weights are unsupported"
    elif not hasattr(matmul_nbits_quantizer, "KQuantWeightOnlyQuantConfig"):
        reason = "this ONNX Runtime build does not provide KQuantWeightOnlyQuantConfig"
    else:
        return rp

    print(f"  k_quant fallback: {reason}; using DEFAULT.")
    return replace(rp, algo="DEFAULT")


def resolve_plan(plan: Plan, config: OptimizerConfig) -> ResolvedPlan:
    resolved = ResolvedPlan(
        method=plan.method.upper(),
        algo=_pick(plan.algo, config.weight_only_algorithm),
        op_types=_pick(plan.op_types, ("MatMul",)),
        axes=_pick(plan.axes, (0,)),
        block_size=_pick(plan.block_size, config.block_size),
        accuracy_level=_pick(plan.accuracy_level, config.accuracy_level),
        symmetric=_pick(plan.symmetric, config.quant_symmetric),
        quant_format=_pick(plan.quant_format, config.quant_format).upper(),
        dynamic_weight_type=_pick(plan.dynamic_weight_type, config.dynamic_weight_type).upper(),
        per_channel=_pick(plan.per_channel, config.dynamic_per_channel),
        reduce_range=_pick(plan.reduce_range, config.dynamic_reduce_range),
        default_tensor_type=_pick(plan.default_tensor_type, config.dynamic_default_tensor_type),
        nodes_to_exclude=_pick(plan.nodes_to_exclude, config.nodes_to_exclude),
        nodes_to_include=_pick(plan.nodes_to_include, config.nodes_to_include),
        optimize=plan.optimize,
        transformer=plan.transformer,
        opt_level=plan.opt_level,
        fp16=plan.fp16,
        f16_op_block_list=_pick(plan.f16_op_block_list, config.f16_op_block_list),
        num_heads=plan.num_heads,
        hidden_size=plan.hidden_size,
        external=_pick(plan.external, config.force_external_data),
        first_slim_no_shape_infer=plan.first_slim_no_shape_infer,
        second_slim_no_shape_infer=_pick(plan.second_slim_no_shape_infer, config.second_slim_no_shape_infer),
    )
    return _fallback_unsupported_k_quant(resolved)


def model_exceeds_2gb(model_path: str) -> bool:
    total = os.path.getsize(model_path)
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total > 2 * 1024**3


def model_size_mb(model_path: str) -> float:
    total = os.path.getsize(model_path)
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total / (1024 * 1024)


def _remove_external_files(model_path: str) -> None:
    for path in (model_path, model_path + ".data"):
        if os.path.exists(path):
            os.remove(path)


def _save_model(model, model_path: str, external: bool) -> None:
    _remove_external_files(model_path)
    if external:
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(model_path) + ".data",
            size_threshold=1024,
            convert_attribute=True,
        )
    else:
        onnx.save(model, model_path)


def _iter_all_data_tensors(graph):
    yield from graph.initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors
            if attr.HasField("g"):
                yield from _iter_all_data_tensors(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_all_data_tensors(subgraph)


def _retarget_external_location(model_path: str, old_location: str, new_location: str) -> None:
    model = onnx.load(model_path, load_external_data=False)
    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location == TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location" and entry.value == old_location:
                    entry.value = new_location
    onnx.save(model, model_path)
    del model
    gc.collect()


def _materialize_constant_tensors_as_initializers(graph) -> int:
    existing_initializers = {initializer.name for initializer in graph.initializer}
    nodes_to_remove = []
    converted = 0

    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                converted += _materialize_constant_tensors_as_initializers(attr.g)
            for subgraph in attr.graphs:
                converted += _materialize_constant_tensors_as_initializers(subgraph)

        if node.op_type != "Constant" or len(node.output) != 1:
            continue

        tensor = None
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                tensor = TensorProto()
                tensor.CopyFrom(attr.t)
                break
        if tensor is None:
            continue

        output_name = node.output[0]
        if output_name in existing_initializers:
            nodes_to_remove.append(node)
            continue

        tensor.name = output_name
        graph.initializer.append(tensor)
        existing_initializers.add(output_name)
        nodes_to_remove.append(node)
        converted += 1

    for node in nodes_to_remove:
        graph.node.remove(node)

    return converted


def resave(src_path: str, dst_path: str, external: bool) -> None:
    model = onnx.load(src_path)
    converted_constants = _materialize_constant_tensors_as_initializers(model.graph)
    if converted_constants:
        print(f"  Materialized {converted_constants} Constant tensor nodes as initializers before save.")
    _save_model(model, dst_path, external)
    del model
    gc.collect()


def read_onnx_metadata(model_path: str) -> dict[str, str]:
    """Return a model's ``metadata_props`` as a plain dict (external weights left on disk)."""
    model = onnx.load(model_path, load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    del model
    gc.collect()
    return metadata


def write_onnx_metadata(model_path: str, metadata: dict[str, str]) -> None:
    """Add/overwrite ``metadata_props`` on an ONNX file in place, preserving external-weight sidecars.

    ``load_external_data=False`` keeps any ``*.data`` sidecar untouched (only the graph proto + metadata
    are rewritten), so restamping is safe for both inline and external-data models. A no-op when the
    source model carried no metadata.
    """
    if not metadata:
        return
    model = onnx.load(model_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, model_path)
    del model
    gc.collect()


def replace_onnx_metadata(model_path: str, metadata: dict[str, str]) -> None:
    """Replace all ``metadata_props`` on one ONNX file without loading sidecars."""
    model = onnx.load(model_path, load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, model_path)
    del model
    gc.collect()


def _capture_graph_output_origins(model_path: str) -> dict[str, tuple[str, str, int]]:
    model = onnx.load(model_path, load_external_data=False)
    producers = {
        output: (node, index)
        for node in model.graph.node
        for index, output in enumerate(node.output)
        if output
    }
    origins = {}
    for graph_output in model.graph.output:
        value_name = graph_output.name
        producer = producers.get(value_name)
        visited = set()
        while producer is not None and producer[0].op_type == "Identity" and producer[0].input:
            if value_name in visited:
                break
            visited.add(value_name)
            value_name = producer[0].input[0]
            producer = producers.get(value_name)
        if producer is not None and producer[0].name:
            node, output_index = producer
            origins[graph_output.name] = (node.name, node.op_type, output_index)
    del model
    gc.collect()
    return origins


def _restore_missing_graph_outputs(
    model_path: str,
    origins: dict[str, tuple[str, str, int]],
) -> None:
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph
    available_values = {value.name for value in graph.input}
    available_values.update(tensor.name for tensor in graph.initializer)
    available_values.update(
        sparse_tensor.values.name for sparse_tensor in graph.sparse_initializer
    )
    available_values.update(output for node in graph.node for output in node.output if output)
    missing_outputs = [output.name for output in graph.output if output.name not in available_values]
    if not missing_outputs:
        del model
        gc.collect()
        return

    nodes_by_identity = {}
    for node in graph.node:
        nodes_by_identity.setdefault((node.name, node.op_type), []).append(node)

    restored = []
    for output_name in missing_outputs:
        origin = origins.get(output_name)
        candidates = nodes_by_identity.get(origin[:2], []) if origin is not None else []
        if origin is None or len(candidates) != 1 or origin[2] >= len(candidates[0].output):
            del model
            gc.collect()
            pass
        source_name = candidates[0].output[origin[2]]
        if not source_name:
            del model
            gc.collect()
            pass
        graph.node.append(
            helper.make_node(
                "Identity",
                inputs=[source_name],
                outputs=[output_name],
                name=f"onnxslim_restore_output_{len(graph.node)}",
            )
        )
        restored.append(output_name)

    onnx.save(model, model_path)
    print(f"  Restored graph output aliases removed by onnxslim: {', '.join(restored)}")
    del model
    gc.collect()


def run_onnxslim(model_path: str, external: bool, config: OptimizerConfig, no_shape_infer: bool) -> None:
    output_origins = _capture_graph_output_origins(model_path)

    def _slim() -> None:
        slim(
            model=model_path,
            output_model=model_path,
            no_shape_infer=no_shape_infer,
            skip_fusion_patterns=config.slim_skip_fusion_patterns,
            skip_optimizations=config.slim_skip_optimizations,
            size_threshold=config.slim_size_threshold,
            save_as_external_data=external,
            verbose=False,
        )
        _restore_missing_graph_outputs(model_path, output_origins)

    data_path = model_path + ".data"
    if not external or not os.path.exists(data_path):
        _slim()
        return

    stash_path = model_path + ".stash.data"
    if os.path.exists(stash_path):
        os.remove(stash_path)
    os.replace(data_path, stash_path)
    _retarget_external_location(
        model_path,
        os.path.basename(data_path),
        os.path.basename(stash_path),
    )
    try:
        _slim()
    except BaseException:
        if not os.path.exists(data_path):
            os.replace(stash_path, data_path)
            _retarget_external_location(
                model_path,
                os.path.basename(stash_path),
                os.path.basename(data_path),
            )
        pass
    finally:
        if os.path.exists(stash_path):
            os.remove(stash_path)


def build_fusion_options(config: OptimizerConfig):
    if not config.optimizer_fusion_options:
        return None
    from onnxruntime.transformers.fusion_options import FusionOptions

    options = FusionOptions(config.optimizer_model_type)
    for key, value in config.optimizer_fusion_options.items():
        setattr(options, key, value)
    return options


def _deduplicate_node_names(graph) -> int:
    used_names, next_name_suffix, used_values, next_value_suffix, remap, renamed = set(), {}, set(), {}, {}, 0
    used_values.update(i.name for i in graph.input)
    used_values.update(i.name for i in graph.initializer)
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in remap:
                node.input[i] = remap[name]

        name = node.name
        if name:
            if name not in used_names:
                used_names.add(name)
            else:
                suffix = next_name_suffix.get(name, 1)
                while f"{name}_{suffix}" in used_names:
                    suffix += 1
                node.name = f"{name}_{suffix}"
                used_names.add(node.name)
                next_name_suffix[name] = suffix + 1
                renamed += 1

        for i, output in enumerate(node.output):
            if not output:
                continue
            if output not in used_values:
                used_values.add(output)
                continue
            suffix = next_value_suffix.get(output, 1)
            while f"{output}_{suffix}" in used_values:
                suffix += 1
            new_output = f"{output}_{suffix}"
            node.output[i] = new_output
            used_values.add(new_output)
            next_value_suffix[output] = suffix + 1
            remap[output] = new_output
            renamed += 1
    return renamed


def _resolve_int(value: IntValue, src_path: str) -> int:
    return int(value(src_path)) if callable(value) else int(value)


def _resolve_nodes(selector: NodeSelector, src_path: str) -> list[str] | None:
    nodes = selector(src_path) if callable(selector) else selector
    return nodes or None


def _iter_graph_nodes(graph):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from _iter_graph_nodes(attribute.g)
            for subgraph in attribute.graphs:
                yield from _iter_graph_nodes(subgraph)


def _resolve_weight_only_node_filters(graph, rp: ResolvedPlan, src_path: str):
    included = _resolve_nodes(rp.nodes_to_include, src_path)
    excluded = set(_resolve_nodes(rp.nodes_to_exclude, src_path) or ())
    if included is None:
        return None, sorted(excluded)

    included = set(included)
    configured = {
        node.name
        for node in _iter_graph_nodes(graph)
        if node.op_type in rp.op_types
    }
    excluded.update(configured - included)
    return sorted(configured & included), sorted(excluded)


def _is_ort_simplified_layer_norm_bug(error: TypeError) -> bool:
    if str(error) != "'NodeProto' object is not subscriptable":
        return False
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if (
            Path(frame.f_code.co_filename).name == "fusion_simplified_layernorm.py"
            and frame.f_code.co_name == "fuse"
        ):
            return True
        traceback = traceback.tb_next
    return False


def optimize_onnx_model(model_path: str, rp: ResolvedPlan, config: OptimizerConfig, src_path: str,
                        use_fp16: bool, external: bool, keep_io_types: bool) -> None:
    from onnxruntime.transformers.optimizer import optimize_model

    optimize_kwargs = dict(
        use_gpu=config.optimizer_use_gpu,
        opt_level=config.optimizer_level if rp.opt_level is None else rp.opt_level,
        num_heads=_resolve_int(rp.num_heads, src_path),
        hidden_size=_resolve_int(rp.hidden_size, src_path),
        optimization_options=build_fusion_options(config),
        model_type=config.optimizer_model_type,
        only_onnxruntime=config.optimizer_only_onnxruntime,
        verbose=False,
    )
    if config.optimizer_provider is not None:
        optimize_kwargs["provider"] = config.optimizer_provider
    try:
        model = optimize_model(model_path, **optimize_kwargs)
    except TypeError as error:
        from onnxruntime.transformers.fusion_options import FusionOptions

        fusion_options = build_fusion_options(config)
        if fusion_options is None:
            fusion_options = FusionOptions(config.optimizer_model_type)
        fusion_options.enable_layer_norm = False
        optimize_kwargs["optimization_options"] = fusion_options
        print(
            "  ORT SimplifiedLayerNorm fusion failed on a Mul square; "
            "retrying with LayerNorm fusion disabled."
        )
        model = optimize_model(model_path, **optimize_kwargs)
    if use_fp16:
        model.convert_float_to_float16(
            keep_io_types=keep_io_types,
            force_fp16_initializers=config.f16_force_initializers,
            use_symbolic_shape_infer=config.shape_infer,
            max_finite_val=config.f16_max_finite_val,
            min_positive_val=config.f16_min_positive_val,
            op_block_list=rp.f16_op_block_list,
            node_block_list=config.f16_node_block_list,
        )
        renamed = _deduplicate_node_names(model.model.graph)
        if renamed:
            print(f"  Renamed {renamed} duplicate node names after float16 conversion.")
    model.save_model_to_file(model_path, use_external_data_format=external, convert_attribute=True)
    del model
    gc.collect()


def upgrade_opset_version(model_path: str, version: int, external: bool) -> None:
    print(f"  Upgrading opset to {version}...")
    try:
        model = onnx.version_converter.convert_version(onnx.load(model_path), version)
        _save_model(model, model_path, external)
        del model
        gc.collect()
    except Exception as exc:
        print(f"  Opset upgrade failed: {exc}. Keeping current version.")
        resave(model_path, model_path, external)


def build_weight_only_config(rp: ResolvedPlan, bits: int):
    algo = rp.algo
    _ALGO_CONFIG_CLASSES = {
        "RTN": "RTNWeightOnlyQuantConfig",
        "k_quant": "KQuantWeightOnlyQuantConfig",
        "HQQ": "HQQWeightOnlyQuantConfig",
    }
    if algo in _ALGO_CONFIG_CLASSES:
        if not hasattr(matmul_nbits_quantizer, _ALGO_CONFIG_CLASSES[algo]):
            print(f"  {algo} weight-only quantizer is unavailable in this ONNX Runtime build; using DEFAULT.")
            algo = "DEFAULT"

    op_types, axes = list(rp.op_types), list(rp.axes)
    quant_axes = tuple(zip(op_types, axes))
    quant_format = _QUANT_FORMATS[rp.quant_format]
    common = {
        "quant_format": quant_format,
        "op_types_to_quantize": tuple(op_types),
    }
    if algo == "RTN":
        cfg = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(**common)
    elif algo == "HQQ":
        cfg = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=bits, block_size=rp.block_size, axis=axes[0], quant_axes=quant_axes, **common,
        )
    elif algo == "k_quant":
        cfg = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(**common)
    else:
        cfg = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=rp.block_size,
            is_symmetric=rp.symmetric,
            accuracy_level=rp.accuracy_level,
            quant_axes=quant_axes,
            **common,
        )
    cfg.bits = bits
    return cfg, quant_axes, algo


def _eliminate_initializer_identity_aliases(graph) -> int:
    """Expose shared constant weights hidden by exporter-generated Identity nodes."""
    initializer_names = {initializer.name for initializer in graph.initializer}
    graph_outputs = {output.name for output in graph.output}
    aliases: dict[str, str] = {}
    removable_outputs: set[str] = set()

    for node in graph.node:
        if node.op_type != "Identity" or len(node.input) != 1 or len(node.output) != 1:
            continue
        source = aliases.get(node.input[0], node.input[0])
        target = node.output[0]
        if source not in initializer_names or target in graph_outputs:
            continue
        aliases[target] = source
        removable_outputs.add(target)

    if not aliases:
        return 0

    retained = []
    for node in graph.node:
        if node.op_type == "Identity" and len(node.output) == 1 and node.output[0] in removable_outputs:
            continue
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
        retained.append(node)
    graph.ClearField("node")
    graph.node.extend(retained)
    return len(removable_outputs)


def _k_quant_odd_block_matmuls(graph, rp: ResolvedPlan, src_path: str) -> list[str]:
    """Find selected MatMuls that trigger ORT's asymmetric INT4 packing bug."""
    if rp.algo != "k_quant" or "MatMul" not in rp.op_types:
        return []

    included = _resolve_nodes(rp.nodes_to_include, src_path)
    included = None if included is None else set(included)
    excluded = set(_resolve_nodes(rp.nodes_to_exclude, src_path) or ())
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    affected = []
    for node in graph.node:
        if (
            node.op_type != "MatMul"
            or len(node.input) != 2
            or (included is not None and node.name not in included)
            or node.name in excluded
        ):
            continue
        weight = initializers.get(node.input[1])
        if (
            weight is not None
            and len(weight.dims) == 2
            and ((weight.dims[0] + rp.block_size - 1) // rp.block_size) % 2 == 1
        ):
            affected.append(node.name)
    return affected


def quantize_weight_only(src_path: str, dst_path: str, rp: ResolvedPlan, bits: int, external: bool) -> None:
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    converted_constants = _materialize_constant_tensors_as_initializers(model.graph)
    if converted_constants:
        print(f"  Materialized {converted_constants} Constant tensor nodes as initializers for weight quantization.")
    exposed_aliases = _eliminate_initializer_identity_aliases(model.graph)
    if exposed_aliases:
        print(f"  Eliminated {exposed_aliases} initializer Identity aliases before weight quantization.")

    quantization_passes = [(rp, bits)]
    if "Gather" in rp.op_types and (bits != 4 or rp.algo != "DEFAULT"):
        operator_axes = tuple(zip(rp.op_types, rp.axes))
        non_gather = tuple(
            (op_type, axis)
            for op_type, axis in operator_axes
            if op_type != "Gather"
        )
        gather = tuple(pair for pair in operator_axes if pair[0] == "Gather")
        quantization_passes = []
        if non_gather:
            quantization_passes.append(
                (
                    replace(
                        rp,
                        op_types=tuple(op_type for op_type, _ in non_gather),
                        axes=tuple(axis for _, axis in non_gather),
                    ),
                    bits,
                )
            )
        # ORT's GatherBlockQuantized path is INT4-only, independent of the other operators' width.
        quantization_passes.append(
            (
                replace(
                    rp,
                    algo="DEFAULT",
                    op_types=tuple(op_type for op_type, _ in gather),
                    axes=tuple(axis for _, axis in gather),
                ),
                4,
            )
        )
        print(
            "  Gather does not support the requested weight-only configuration; "
            "using DEFAULT Q4 for Gather."
        )

    compatible_passes = []
    for pass_plan, pass_bits in quantization_passes:
        affected = _k_quant_odd_block_matmuls(model.graph, pass_plan, src_path)
        if not affected:
            compatible_passes.append((pass_plan, pass_bits))
            continue
        excluded = set(_resolve_nodes(pass_plan.nodes_to_exclude, src_path) or ())
        compatible_passes.append(
            (replace(pass_plan, nodes_to_exclude=sorted(excluded | set(affected))), pass_bits)
        )
        matmul_axis = pass_plan.axes[pass_plan.op_types.index("MatMul")]
        compatible_passes.append(
            (
                replace(
                    pass_plan,
                    algo="DEFAULT",
                    op_types=("MatMul",),
                    axes=(matmul_axis,),
                    nodes_to_exclude=None,
                    nodes_to_include=affected,
                ),
                pass_bits,
            )
        )
        print(
            f"  Routing {len(affected)} MatMul node(s) with an odd K-block count "
            "through DEFAULT Q4 packing."
        )
    quantization_passes = compatible_passes

    quant = None
    for pass_plan, pass_bits in quantization_passes:
        cfg, quant_axes, algo = build_weight_only_config(pass_plan, pass_bits)
        nodes_to_include, nodes_to_exclude = _resolve_weight_only_node_filters(
            model.graph,
            pass_plan,
            src_path,
        )
        print(
            f"  Quantizing weights ({algo}, {pass_bits}-bit, block={pass_plan.block_size}, "
            f"format={pass_plan.quant_format}, ops={list(pass_plan.op_types)})..."
        )
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            block_size=pass_plan.block_size,
            is_symmetric=pass_plan.symmetric,
            accuracy_level=pass_plan.accuracy_level,
            quant_format=_QUANT_FORMATS[pass_plan.quant_format],
            op_types_to_quantize=tuple(pass_plan.op_types),
            quant_axes=quant_axes,
            algo_config=cfg,
            nodes_to_exclude=nodes_to_exclude,
            nodes_to_include=nodes_to_include,
        )
        quant.process()
        quant.model.topological_sort()
        model = quant.model.model

    _save_model(model, dst_path, external)
    del model, quant
    gc.collect()


def quantize_weight_only_shared(
    template_src_path: str,
    model_paths: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    cache_path: str,
    rp: ResolvedPlan,
    bits: int,
    external: bool,
) -> dict[str, int]:
    """Quantize a covering graph once and apply selected packed weights to peers."""
    shared_layouts = {
        "MatMul": (1, 0),
        "Gather": (0, 1),
    }
    unsupported_op_types = sorted(set(rp.op_types) - set(shared_layouts))
    template_src_path = str(Path(template_src_path).resolve())
    cache_path = str(Path(cache_path).resolve())
    targets = [(str(Path(src).resolve()), str(Path(dst).resolve())) for src, dst in model_paths]
    missing_sources = [src for src, _ in targets if not os.path.isfile(src)]
    if not os.path.isfile(template_src_path):
        missing_sources.insert(0, template_src_path)
    cache_folder = Path(cache_path).parent
    quantize_weight_only(template_src_path, cache_path, rp, bits, external)

    template_includes = _resolve_nodes(rp.nodes_to_include, template_src_path)
    template_includes = None if template_includes is None else set(template_includes)
    template_excludes = set(_resolve_nodes(rp.nodes_to_exclude, template_src_path) or ())

    template = onnx.load(template_src_path, load_external_data=False)
    _materialize_constant_tensors_as_initializers(template.graph)
    _eliminate_initializer_identity_aliases(template.graph)
    template_initializers = {initializer.name for initializer in template.graph.initializer}

    quantized_template = onnx.load(cache_path, load_external_data=False)
    quantized_by_output = {
        output: node
        for node in quantized_template.graph.node
        for output in node.output
        if output
    }
    quantized_initializers = {
        initializer.name: initializer for initializer in quantized_template.graph.initializer
    }

    recipes: dict[tuple[str, str], tuple[onnx.NodeProto, str, tuple[str | None, ...]]] = {}
    template_rewrites = 0
    for node in template.graph.node:
        if (
            (template_includes is not None and node.name not in template_includes)
            or node.name in template_excludes
        ):
            continue
        layout = shared_layouts.get(node.op_type)
        if layout is None or len(node.input) != 2:
            continue
        weight_index, dynamic_index = layout
        weight_name = node.input[weight_index]
        if weight_name not in template_initializers:
            continue
        quantized_node = quantized_by_output.get(node.output[0])
        quantized_inputs = tuple(
            None if index == dynamic_index else name
            for index, name in enumerate(quantized_node.input)
        )
        missing_inputs = [
            name for name in quantized_inputs
            if name and name not in quantized_initializers
        ]
        name_suffix = (
            quantized_node.name[len(node.name):]
            if node.name and quantized_node.name.startswith(node.name)
            else f"_{quantized_node.op_type}"
        )
        recipe_key = (node.op_type, weight_name)
        prior = recipes.get(recipe_key)
        if prior is not None:
            prior_node, prior_suffix, prior_inputs = prior
            same_attributes = [attr.SerializeToString() for attr in prior_node.attribute] == [
                attr.SerializeToString() for attr in quantized_node.attribute
            ]
        else:
            recipes[recipe_key] = (quantized_node, name_suffix, quantized_inputs)
        template_rewrites += 1

    total_rewrites = 0
    total_removed_initializers = 0
    for src_path, dst_path in targets:
        target_includes = _resolve_nodes(rp.nodes_to_include, src_path)
        target_includes = None if target_includes is None else set(target_includes)
        target_excludes = set(_resolve_nodes(rp.nodes_to_exclude, src_path) or ())
        model = onnx.load(src_path, load_external_data=True)
        _materialize_constant_tensors_as_initializers(model.graph)
        _eliminate_initializer_identity_aliases(model.graph)
        initializer_names = {initializer.name for initializer in model.graph.initializer}
        required_quantized_inputs: set[str] = set()
        missing_weights: set[tuple[str, str]] = set()
        graph_rewrites = 0

        for node in model.graph.node:
            if (
                (target_includes is not None and node.name not in target_includes)
                or node.name in target_excludes
            ):
                continue
            layout = shared_layouts.get(node.op_type)
            if layout is None or len(node.input) != 2:
                continue
            weight_index, dynamic_index = layout
            weight_name = node.input[weight_index]
            if weight_name not in initializer_names:
                continue
            recipe_key = (node.op_type, weight_name)
            recipe = recipes.get(recipe_key)
            if recipe is None:
                missing_weights.add(recipe_key)
                continue
            quantized_node, name_suffix, quantized_inputs = recipe
            dynamic_input = node.input[dynamic_index]
            node.ClearField("input")
            node.input.extend(
                dynamic_input if name is None else name
                for name in quantized_inputs
            )
            node.op_type = quantized_node.op_type
            node.domain = quantized_node.domain
            node.name = f"{node.name}{name_suffix}" if node.name else quantized_node.op_type
            node.ClearField("attribute")
            for attribute in quantized_node.attribute:
                node.attribute.add().CopyFrom(attribute)
            required_quantized_inputs.update(name for name in quantized_inputs if name)
            graph_rewrites += 1

        if missing_weights:
            names = [f"{op_type}:{weight_name}" for op_type, weight_name in sorted(missing_weights)]
            pass
        referenced_values = {value.name for value in model.graph.input}
        referenced_values.update(value.name for value in model.graph.output)

        def collect_node_inputs(graph) -> None:
            for graph_node in graph.node:
                referenced_values.update(name for name in graph_node.input if name)
                for attribute in graph_node.attribute:
                    if attribute.HasField("g"):
                        collect_node_inputs(attribute.g)
                    for subgraph in attribute.graphs:
                        collect_node_inputs(subgraph)

        collect_node_inputs(model.graph)
        removed_initializers = 0
        for index in range(len(model.graph.initializer) - 1, -1, -1):
            if model.graph.initializer[index].name not in referenced_values:
                del model.graph.initializer[index]
                removed_initializers += 1

        retained_names = {initializer.name for initializer in model.graph.initializer}
        collisions = sorted(required_quantized_inputs & retained_names)
        _save_model(model, dst_path, external)
        del model
        model = onnx.load(dst_path, load_external_data=False)
        for initializer in quantized_template.graph.initializer:
            if initializer.name in required_quantized_inputs:
                model.graph.initializer.add().CopyFrom(initializer)
        appended_names = {initializer.name for initializer in model.graph.initializer}
        absent_inputs = sorted(required_quantized_inputs - appended_names)
        opsets = {opset.domain: opset for opset in model.opset_import}
        for template_opset in quantized_template.opset_import:
            if template_opset.domain in {"", "ai.onnx"}:
                continue
            existing = opsets.get(template_opset.domain)
            if existing is None:
                existing = model.opset_import.add(domain=template_opset.domain, version=template_opset.version)
                opsets[template_opset.domain] = existing
            elif existing.version < template_opset.version:
                existing.version = template_opset.version

        onnx.save(model, dst_path)
        print(
            f"  {Path(dst_path).name}: reused {len(required_quantized_inputs)} cached tensors "
            f"across {graph_rewrites} weight-only nodes."
        )
        total_rewrites += graph_rewrites
        total_removed_initializers += removed_initializers
        del model
        gc.collect()

    del template, quantized_template
    gc.collect()
    return {
        "graph_count": len(targets),
        "unique_weights": len(recipes),
        "template_rewrites": template_rewrites,
        "total_rewrites": total_rewrites,
        "removed_initializers": total_removed_initializers,
    }


def quantize_dynamic_int8(src_path: str, dst_path: str, rp: ResolvedPlan, external: bool) -> None:
    weight_type = _DYNAMIC_WEIGHT_TYPES[rp.dynamic_weight_type]
    extra_options = {
        "ActivationSymmetric": rp.symmetric,
        "WeightSymmetric": rp.symmetric,
        "EnableSubgraph": True,
        "ForceQuantizeNoInputCheck": False,
        "MatMulConstBOnly": True,
    }
    if rp.default_tensor_type is not None:
        extra_options["DefaultTensorType"] = rp.default_tensor_type
    print(
        f"  Quantizing weights (dynamic INT8, {rp.dynamic_weight_type}, "
        f"per_channel={rp.per_channel}, reduce_range={rp.reduce_range})..."
    )
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    converted_constants = _materialize_constant_tensors_as_initializers(model.graph)
    if converted_constants:
        print(f"  Materialized {converted_constants} Constant tensor nodes as initializers for weight quantization.")
    exposed_aliases = _eliminate_initializer_identity_aliases(model.graph)
    if exposed_aliases:
        print(f"  Eliminated {exposed_aliases} initializer Identity aliases before dynamic quantization.")
    quantize_dynamic(
        model_input=model,
        model_output=dst_path,
        per_channel=rp.per_channel,
        reduce_range=rp.reduce_range,
        weight_type=weight_type,
        extra_options=extra_options,
        nodes_to_quantize=_resolve_nodes(rp.nodes_to_include, src_path),
        nodes_to_exclude=_resolve_nodes(rp.nodes_to_exclude, src_path),
        use_external_data_format=external,
    )
    del model
    gc.collect()


def _source_tensor_signature(tensor: TensorProto, model_path: str) -> tuple:
    """Identify an initializer without loading a shared external-data blob."""
    if tensor.data_location == TensorProto.EXTERNAL:
        external_data = {item.key: item.value for item in tensor.external_data}
        location = external_data.get("location")
        resolved_location = str((Path(model_path).parent / location).resolve())
        return (
            tensor.data_type,
            tuple(tensor.dims),
            "external",
            resolved_location,
            external_data.get("offset", "0"),
            external_data.get("length", ""),
            external_data.get("checksum", ""),
        )

    value = TensorProto()
    value.CopyFrom(tensor)
    value.name = ""
    return (
        tensor.data_type,
        tuple(tensor.dims),
        "inline",
        hashlib.sha256(value.SerializeToString()).digest(),
    )


def quantize_dynamic_int8_shared(
    template_src_path: str,
    model_paths: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    cache_path: str,
    rp: ResolvedPlan,
    external: bool,
) -> dict[str, int]:
    """Quantize one covering graph and replay its dynamic MatMul recipes on peers."""
    template_src_path = str(Path(template_src_path).resolve())
    cache_path = str(Path(cache_path).resolve())
    targets = [(str(Path(src).resolve()), str(Path(dst).resolve())) for src, dst in model_paths]
    missing_sources = [src for src, _ in targets if not os.path.isfile(src)]
    if not os.path.isfile(template_src_path):
        missing_sources.insert(0, template_src_path)
    cache_folder = Path(cache_path).parent
    quantize_dynamic_int8(template_src_path, cache_path, rp, external)

    template = onnx.load(template_src_path, load_external_data=False)
    _materialize_constant_tensors_as_initializers(template.graph)
    _eliminate_initializer_identity_aliases(template.graph)
    template_initializers = {
        initializer.name: initializer for initializer in template.graph.initializer
    }
    template_signatures = {
        name: _source_tensor_signature(initializer, template_src_path)
        for name, initializer in template_initializers.items()
    }

    quantized_template = onnx.load(cache_path, load_external_data=False)
    quantized_initializers = {
        initializer.name: initializer for initializer in quantized_template.graph.initializer
    }
    producer_by_output = {
        output: node
        for node in quantized_template.graph.node
        for output in node.output
        if output
    }

    weight_recipes = {}
    gather_weight_recipes = {}
    static_activation_recipes = {}
    dynamic_quantize_prototype = None
    dequantize_prototype = None
    template_rewrites = 0

    for node in template.graph.node:
        if (
            node.op_type != "MatMul"
            or len(node.input) != 2
            or len(node.output) != 1
            or node.input[1] not in template_initializers
        ):
            continue

        output_mul = producer_by_output.get(node.output[0])
        if output_mul is None or output_mul.op_type != "Mul" or len(output_mul.input) != 2:
            actual = None if output_mul is None else output_mul.op_type
            pass
        output_inputs = [(name, producer_by_output.get(name)) for name in output_mul.input]
        cast_candidates = [(name, producer) for name, producer in output_inputs if producer and producer.op_type == "Cast"]
        scale_candidates = [(name, producer) for name, producer in output_inputs if producer and producer.op_type == "Mul"]
        _, cast_node = cast_candidates[0]
        _, scale_mul = scale_candidates[0]
        integer_matmul = producer_by_output.get(cast_node.input[0])
        activation_quantized, weight_quantized, activation_zero_point, weight_zero_point = integer_matmul.input
        weight_name = node.input[1]
        expected_weight_quantized = f"{weight_name}_quantized"
        expected_weight_scale = f"{weight_name}_scale"
        expected_weight_zero_point = f"{weight_name}_zero_point"
        required_weight_tensors = (
            weight_quantized,
            expected_weight_scale,
            weight_zero_point,
        )
        missing_weight_tensors = [
            name for name in required_weight_tensors if name not in quantized_initializers
        ]
        activation_scale = next(name for name in scale_mul.input if name != expected_weight_scale)
        activation_name = node.input[0]
        expected_activation_quantized = f"{activation_name}_quantized"
        expected_activation_scale = f"{activation_name}_scale"
        expected_activation_zero_point = f"{activation_name}_zero_point"
        if activation_name in template_initializers:
            required_activation_tensors = (
                activation_quantized,
                activation_scale,
                activation_zero_point,
            )
            missing_activation_tensors = [
                name for name in required_activation_tensors if name not in quantized_initializers
            ]
            activation_signature = template_signatures[activation_name]
            prior_activation = static_activation_recipes.get(activation_signature)
            static_activation_recipes[activation_signature] = required_activation_tensors
        else:
            dynamic_quantize = producer_by_output.get(activation_quantized)
            if dynamic_quantize_prototype is None:
                dynamic_quantize_prototype = dynamic_quantize
        recipe = {
            "signature": template_signatures[weight_name],
            "weight_quantized": weight_quantized,
            "weight_scale": expected_weight_scale,
            "weight_zero_point": weight_zero_point,
            "integer_matmul": integer_matmul,
            "cast": cast_node,
            "scale_mul": scale_mul,
            "output_mul": output_mul,
        }
        prior = weight_recipes.get(weight_name)
        if prior is not None:
            comparable_keys = (
                "signature",
                "weight_quantized",
                "weight_scale",
                "weight_zero_point",
            )
        else:
            weight_recipes[weight_name] = recipe
        template_rewrites += 1

    for node in template.graph.node:
        if (
            node.op_type != "Gather"
            or len(node.input) < 2
            or len(node.output) != 1
            or node.input[0] not in template_initializers
        ):
            continue

        dequantize = producer_by_output.get(node.output[0])
        if dequantize is None or dequantize.op_type != "DequantizeLinear" or len(dequantize.input) != 3:
            actual = None if dequantize is None else dequantize.op_type
            pass
        quantized_gather = producer_by_output.get(dequantize.input[0])
        weight_name = node.input[0]
        expected_weight_tensors = (
            f"{weight_name}_quantized",
            f"{weight_name}_scale",
            f"{weight_name}_zero_point",
        )
        missing_weight_tensors = [
            name for name in expected_weight_tensors if name not in quantized_initializers
        ]
        if dequantize_prototype is None:
            dequantize_prototype = dequantize
        recipe = {
            "signature": template_signatures[weight_name],
            "weight_quantized": expected_weight_tensors[0],
            "weight_scale": expected_weight_tensors[1],
            "weight_zero_point": expected_weight_tensors[2],
            "dequantize": dequantize,
        }
        prior = gather_weight_recipes.get(weight_name)
        gather_weight_recipes[weight_name] = recipe
        template_rewrites += 1

    def clone_node(prototype, name, inputs, outputs):
        cloned = onnx.NodeProto()
        cloned.CopyFrom(prototype)
        cloned.name = name
        cloned.ClearField("input")
        cloned.input.extend(inputs)
        cloned.ClearField("output")
        cloned.output.extend(outputs)
        return cloned

    total_rewrites = 0
    total_matmul_rewrites = 0
    total_gather_rewrites = 0
    total_removed_initializers = 0
    for src_path, dst_path in targets:
        source_header = onnx.load(src_path, load_external_data=False)
        _materialize_constant_tensors_as_initializers(source_header.graph)
        _eliminate_initializer_identity_aliases(source_header.graph)
        source_signatures = {
            initializer.name: _source_tensor_signature(initializer, src_path)
            for initializer in source_header.graph.initializer
        }
        del source_header

        model = onnx.load(src_path, load_external_data=True)
        _materialize_constant_tensors_as_initializers(model.graph)
        _eliminate_initializer_identity_aliases(model.graph)
        initializer_names = {initializer.name for initializer in model.graph.initializer}
        reserved_values = {value.name for value in model.graph.input}
        reserved_values.update(value.name for value in model.graph.output)
        reserved_values.update(initializer_names)
        reserved_values.update(
            output for graph_node in model.graph.node for output in graph_node.output if output
        )
        generated_values = set()
        dynamic_activations = {}
        required_quantized_inputs = set()
        rewritten_nodes = []
        graph_rewrites = 0
        graph_matmul_rewrites = 0
        graph_gather_rewrites = 0

        def reserve_generated(names):
            collisions = sorted(set(names) & reserved_values)
            reserved_values.update(names)
            generated_values.update(names)

        for node in model.graph.node:
            if (
                node.op_type == "Gather"
                and len(node.input) >= 2
                and len(node.output) == 1
                and node.input[0] in initializer_names
            ):
                weight_name = node.input[0]
                recipe = gather_weight_recipes.get(weight_name)
                output_name = node.output[0]
                quantized_output = f"{output_name}_quantized"
                reserve_generated((quantized_output,))
                weight_quantized = recipe["weight_quantized"]
                weight_scale = recipe["weight_scale"]
                weight_zero_point = recipe["weight_zero_point"]
                required_quantized_inputs.update(
                    (weight_quantized, weight_scale, weight_zero_point)
                )
                rewritten_nodes.extend((
                    clone_node(
                        node,
                        node.name,
                        (weight_quantized, *node.input[1:]),
                        (quantized_output,),
                    ),
                    clone_node(
                        recipe["dequantize"],
                        f"{output_name}_DequantizeLinear",
                        (quantized_output, weight_scale, weight_zero_point),
                        (output_name,),
                    ),
                ))
                graph_rewrites += 1
                graph_gather_rewrites += 1
                continue

            if (
                node.op_type != "MatMul"
                or len(node.input) != 2
                or len(node.output) != 1
                or node.input[1] not in initializer_names
            ):
                rewritten_nodes.append(node)
                continue

            activation_name, weight_name = node.input
            recipe = weight_recipes.get(weight_name)
            if activation_name in initializer_names:
                activation_recipe = static_activation_recipes.get(source_signatures[activation_name])
                activation_quantized, activation_scale, activation_zero_point = activation_recipe
                required_quantized_inputs.update(activation_recipe)
            else:
                activation_recipe = dynamic_activations.get(activation_name)
                if activation_recipe is None:
                    activation_recipe = (
                        f"{activation_name}_quantized",
                        f"{activation_name}_scale",
                        f"{activation_name}_zero_point",
                    )
                    reserve_generated(activation_recipe)
                    dynamic_activations[activation_name] = activation_recipe
                    rewritten_nodes.append(
                        clone_node(
                            dynamic_quantize_prototype,
                            f"{activation_name}_QuantizeLinear",
                            [activation_name],
                            activation_recipe,
                        )
                    )
                activation_quantized, activation_scale, activation_zero_point = activation_recipe

            output_name = node.output[0]
            stem = node.name or output_name
            integer_output = f"{output_name}_output_quantized"
            cast_output = f"{integer_output}_cast_output"
            scale_output = f"{output_name}_quant_scales_mul"
            reserve_generated((integer_output, cast_output, scale_output))

            weight_quantized = recipe["weight_quantized"]
            weight_scale = recipe["weight_scale"]
            weight_zero_point = recipe["weight_zero_point"]
            required_quantized_inputs.update(
                (weight_quantized, weight_scale, weight_zero_point)
            )
            rewritten_nodes.extend((
                clone_node(
                    recipe["integer_matmul"],
                    f"{stem}_quant",
                    (
                        activation_quantized,
                        weight_quantized,
                        activation_zero_point,
                        weight_zero_point,
                    ),
                    (integer_output,),
                ),
                clone_node(
                    recipe["cast"],
                    f"{integer_output}_cast",
                    (integer_output,),
                    (cast_output,),
                ),
                clone_node(
                    recipe["scale_mul"],
                    f"{stem}_quant_scales_mul",
                    (activation_scale, weight_scale),
                    (scale_output,),
                ),
                clone_node(
                    recipe["output_mul"],
                    f"{stem}_quant_output_scale_mul",
                    (cast_output, scale_output),
                    (output_name,),
                ),
            ))
            graph_rewrites += 1
            graph_matmul_rewrites += 1

        model.graph.ClearField("node")
        model.graph.node.extend(rewritten_nodes)

        referenced_values = {value.name for value in model.graph.input}
        referenced_values.update(value.name for value in model.graph.output)

        def collect_node_inputs(graph) -> None:
            for graph_node in graph.node:
                referenced_values.update(name for name in graph_node.input if name)
                for attribute in graph_node.attribute:
                    if attribute.HasField("g"):
                        collect_node_inputs(attribute.g)
                    for subgraph in attribute.graphs:
                        collect_node_inputs(subgraph)

        collect_node_inputs(model.graph)
        removed_initializers = 0
        for index in range(len(model.graph.initializer) - 1, -1, -1):
            if model.graph.initializer[index].name not in referenced_values:
                del model.graph.initializer[index]
                removed_initializers += 1

        retained_names = {initializer.name for initializer in model.graph.initializer}
        collisions = sorted(required_quantized_inputs & (retained_names | generated_values))
        missing_cached = sorted(required_quantized_inputs - set(quantized_initializers))
        _save_model(model, dst_path, external)
        del model
        model = onnx.load(dst_path, load_external_data=False)
        for name in sorted(required_quantized_inputs):
            model.graph.initializer.add().CopyFrom(quantized_initializers[name])

        opsets = {opset.domain: opset for opset in model.opset_import}
        for template_opset in quantized_template.opset_import:
            existing = opsets.get(template_opset.domain)
            if existing is None:
                existing = model.opset_import.add(
                    domain=template_opset.domain,
                    version=template_opset.version,
                )
                opsets[template_opset.domain] = existing
            elif existing.version < template_opset.version:
                existing.version = template_opset.version

        onnx.save(model, dst_path)
        print(
            f"  {Path(dst_path).name}: reused {len(required_quantized_inputs)} cached tensors "
            f"across {graph_matmul_rewrites} MatMul and {graph_gather_rewrites} Gather nodes."
        )
        total_rewrites += graph_rewrites
        total_matmul_rewrites += graph_matmul_rewrites
        total_gather_rewrites += graph_gather_rewrites
        total_removed_initializers += removed_initializers
        del model
        gc.collect()

    del template, quantized_template
    gc.collect()
    return {
        "graph_count": len(targets),
        "unique_weights": len(weight_recipes) + len(gather_weight_recipes),
        "template_rewrites": template_rewrites,
        "total_rewrites": total_rewrites,
        "matmul_rewrites": total_matmul_rewrites,
        "gather_rewrites": total_gather_rewrites,
        "removed_initializers": total_removed_initializers,
    }


def collect_quant_unsafe_nodes(model_path: str) -> list[str]:
    """Collect MatMul/Gemm/Gather nodes that dynamic quantization should skip.

    Skips MatMul/Gemm fed by float16 or rank>2 constant weights and Gather fed by float16
    weights. This covers frontend filterbanks, relative-position tables, and fp16 embeddings.
    """
    model = onnx.load(model_path)
    fp16_weights: set[str] = set()
    high_rank_weights: set[str] = set()

    def _register(name: str, data_type: int, dims) -> None:
        if data_type == TensorProto.FLOAT16:
            fp16_weights.add(name)
        if len(dims) > 2:
            high_rank_weights.add(name)

    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.name:
            _register(tensor.name, tensor.data_type, tensor.dims)

    for node in model.graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                if attr.HasField("t"):
                    _register(node.output[0], attr.t.data_type, attr.t.dims)
                for tensor in attr.tensors:
                    _register(node.output[0], tensor.data_type, tensor.dims)

    nodes_to_exclude = []
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            if any(name in fp16_weights or name in high_rank_weights for name in node.input):
                nodes_to_exclude.append(node.name)
        elif node.op_type == "Gather":
            if any(name in fp16_weights for name in node.input):
                nodes_to_exclude.append(node.name)
    del model
    gc.collect()
    return nodes_to_exclude


def get_model_paths(config: OptimizerConfig, name: str) -> tuple[str, str]:
    return (
        os.path.join(config.original_folder_path, f"{name}.onnx"),
        os.path.join(config.optimized_folder_path, f"{name}.onnx"),
    )


def process_model(
    name: str,
    rp: ResolvedPlan,
    config: OptimizerConfig,
    mixed_precision: bool,
    *,
    prequantized: bool = False,
) -> None:
    src_path, dst_path = get_model_paths(config, name)
    if not os.path.exists(src_path):
        print(f"  Skipping - file not found: {src_path}")
        return

    source_metadata = read_onnx_metadata(src_path)
    if not prequantized:
        _remove_external_files(dst_path)

    external = rp.external or model_exceeds_2gb(src_path)
    use_fp16 = rp.fp16 or rp.method == "F16"
    keep_io_types = mixed_precision if config.f16_keep_io_types is None else config.f16_keep_io_types

    if prequantized:
        print("  Reusing shared prequantized graph...")
    elif rp.method in _WEIGHT_ONLY_BITS:
        quantize_weight_only(src_path, dst_path, rp, _WEIGHT_ONLY_BITS[rp.method], external)
    elif rp.method == "DYNAMIC":
        quantize_dynamic_int8(src_path, dst_path, rp, external)
    else:
        resave(src_path, dst_path, external)

    if rp.optimize or use_fp16:
        print("  Optimizing (onnxslim -> transformers optimizer -> onnxslim)...")
        run_onnxslim(dst_path, external, config, no_shape_infer=rp.first_slim_no_shape_infer)
        if rp.transformer or use_fp16:
            optimize_onnx_model(dst_path, rp, config, src_path, use_fp16, external, keep_io_types)
            second_no_shape = not config.shape_infer if rp.second_slim_no_shape_infer is None else rp.second_slim_no_shape_infer
            run_onnxslim(dst_path, external, config, no_shape_infer=second_no_shape)

    if config.upgrade_opset > 0:
        upgrade_opset_version(dst_path, config.upgrade_opset, external)

    if not external and os.path.exists(dst_path + ".data"):
        os.remove(dst_path + ".data")

    # Restamp the source model's metadata_props onto the optimized output. Quantization / onnxslim /
    # the transformers optimizer can drop custom metadata; the geometry / token / max_seq_len facts are
    # invariant through those passes, so copying them across keeps the runtime's metadata reads working.
    # Only the activation dtype is allowed to change here, and only for plans that actually run fp16 conversion.
    output_metadata = dict(source_metadata)
    if use_fp16:
        output_metadata["activations_fp16"] = "1"
    write_onnx_metadata(dst_path, output_metadata)


def copy_artifacts(config: OptimizerConfig) -> None:
    for artifact in config.copy_artifacts:
        src_path = os.path.join(config.original_folder_path, artifact)
        dst_path = os.path.join(config.optimized_folder_path, artifact)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            print(f"Copied {artifact} -> {dst_path}")


def convert_to_ort_format(config: OptimizerConfig) -> None:
    """Optionally convert every optimized *.onnx into ORT format (XNNPACK/NNAPI/QNN/CoreML/CPU).

    Off by default; the modern export scripts ship plain *.onnx. The legacy vocoder / preprocess
    scripts opt in via ``convert_to_ort=True``. Uses an argv list (no shell) so folder paths cannot
    be misinterpreted by a shell.
    """
    if not config.convert_to_ort:
        return
    import subprocess
    import sys

    command = [
        sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort",
        "--output_dir", config.optimized_folder_path,
        "--optimization_style", config.ort_optimization_style,
        "--target_platform", config.ort_target_platform,
    ]
    if config.ort_enable_type_reduction:
        command.append("--enable_type_reduction")
    command.append(config.optimized_folder_path)
    print(f"Converting optimized models to ORT format ({config.ort_optimization_style})...")
    subprocess.run(command, check=False)


def run_optimizer(config: OptimizerConfig) -> None:
    os.makedirs(config.optimized_folder_path, exist_ok=True)

    resolved = {name: resolve_plan(plan, config) for name, plan in config.model_plans.items()}
    for name in resolved:
        _, dst_path = get_model_paths(config, name)
        _remove_external_files(dst_path)

    mixed_precision = uses_mixed_precision(config.model_plans.values())
    if mixed_precision and config.f16_keep_io_types is None:
        print(
            "TIP: mixed float16/float32 modules detected - forcing keep_io_types=True on "
            "float16 conversions so shared graph I/O stays float32-compatible."
        )

    for name, rp in resolved.items():
        print(f"\n{'=' * 60}\nProcessing: {name}  [{rp.method}]\n{'=' * 60}")
        process_model(name, rp, config, mixed_precision)

    copy_artifacts(config)
    convert_to_ort_format(config)
    print("\n--- All models processed successfully! ---")
