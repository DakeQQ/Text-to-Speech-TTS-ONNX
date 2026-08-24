"""Shared Qwen-style ONNX optimization pipeline for the TTS export scripts."""

from __future__ import annotations

import gc
import hashlib
import os
import shutil
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnx
import onnx.version_converter
from onnx import TensorProto, helper, numpy_helper
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
_WEIGHT_ONLY_ALGO_BITS = {
    "DEFAULT": frozenset(_WEIGHT_ONLY_BITS.values()),
    "HQQ": frozenset(_WEIGHT_ONLY_BITS.values()),
    "AFFINE_REFINE_V2": frozenset({4, 8}),
    "RTN": frozenset({4}),
    "k_quant": frozenset({4}),
}
_VALID_ALGOS = set(_WEIGHT_ONLY_ALGO_BITS)


@dataclass
class Plan:
    weight_only_algorithm: str = "DEFAULT"

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
    # quantized KV attention / rope-shift rewrites
    kv_surgery: bool | str | None = None  # "auto" | True | False
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
    weight_only_algorithm: str = "AFFINE_REFINE_V2"
    block_size: int = 32
    accuracy_level: int = 4
    quant_symmetric: bool = False
    quant_format: str = "QOperator"
    # dynamic INT8 defaults
    dynamic_weight_type: str = "QInt8"
    dynamic_per_channel: bool = True
    dynamic_reduce_range: bool = False
    dynamic_default_tensor_type: int | None = None
    # AFFINE_REFINE_V2 fitting settings. These are deliberately per-script
    # configuration rather than process globals so model families remain independent.
    affine_v2_seed_iterations: int = 4
    affine_v2_seed_zp_radius: int = 2
    affine_v2_seed_chunk_blocks: int = 65536
    affine_v2_seed_blocks_per_job: int = 1024
    affine_v2_seed_workers: int = 4
    affine_v2_numba_threads: int = 4
    affine_v2_iterations: int = 6
    affine_v2_clip_ratios: tuple[float, ...] = (1.0, 0.94, 0.82, 0.70, 0.55)
    affine_v2_chunk_blocks: int = 8192
    affine_v2_weighted_tolerance: float = 0.15
    affine_v2_asym_zp_sweep_limit: int = 32
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
    # ORT 1.27 CUDA does not execute blocked QuantizeLinear/DequantizeLinear.
    # Keep the arithmetic KV write tails by default; CPU-only callers may opt in.
    kv_attention_surgery: bool | str = "auto"
    kv_blocked_qdq_surgery: bool = False
    # convert every optimized *.onnx to ORT format (legacy vocoder / preprocess scripts)
    convert_to_ort: bool = False
    ort_optimization_style: str = "Fixed"
    ort_target_platform: str = "amd64"
    ort_enable_type_reduction: bool = True
    # optional side artifacts copied after all models are processed
    copy_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class AffineV2Settings:
    """Resolved AFFINE_REFINE_V2 controls retained by shared quantization plans."""

    seed_iterations: int = 4
    seed_zp_radius: int = 2
    seed_chunk_blocks: int = 65536
    seed_blocks_per_job: int = 1024
    seed_workers: int = 4
    numba_threads: int = 4
    iterations: int = 6
    clip_ratios: tuple[float, ...] = (1.0, 0.94, 0.82, 0.70, 0.55)
    chunk_blocks: int = 8192
    weighted_tolerance: float = 0.15
    asym_zp_sweep_limit: int = 32


def _resolve_affine_v2_settings(config: OptimizerConfig) -> AffineV2Settings:
    return AffineV2Settings(
        seed_iterations=config.affine_v2_seed_iterations,
        seed_zp_radius=config.affine_v2_seed_zp_radius,
        seed_chunk_blocks=config.affine_v2_seed_chunk_blocks,
        seed_blocks_per_job=config.affine_v2_seed_blocks_per_job,
        seed_workers=config.affine_v2_seed_workers,
        numba_threads=config.affine_v2_numba_threads,
        iterations=config.affine_v2_iterations,
        clip_ratios=tuple(config.affine_v2_clip_ratios),
        chunk_blocks=config.affine_v2_chunk_blocks,
        weighted_tolerance=config.affine_v2_weighted_tolerance,
        asym_zp_sweep_limit=config.affine_v2_asym_zp_sweep_limit,
    )


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
    kv_surgery: bool | str
    external: bool
    first_slim_no_shape_infer: bool
    second_slim_no_shape_infer: bool | None
    affine_v2_settings: AffineV2Settings = field(default_factory=AffineV2Settings)


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
        print(f"  k_quant fallback: {bits}-bit weights are unsupported; using DEFAULT.")
        return replace(rp, algo="DEFAULT")
    # k_quant is implemented by the common CPU helper, not an ORT opaque config.
    return rp


def resolve_plan(plan: Plan, config: OptimizerConfig) -> ResolvedPlan:
    raw_algorithm = str(_pick(plan.algo, config.weight_only_algorithm))
    algorithm = "k_quant" if raw_algorithm.upper() == "K_QUANT" else raw_algorithm.upper()
    method = plan.method.upper()
    # AFFINE_REFINE_V2 intentionally covers Q4/Q8 and dynamic INT8, not Q2.
    # Preserve Q2's long-standing DEFAULT behavior when it inherits the global default.
    if method == "Q2" and plan.algo is None and algorithm == "AFFINE_REFINE_V2":
        algorithm = "DEFAULT"
    resolved = ResolvedPlan(
        method=method,
        algo=algorithm,
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
        kv_surgery=_pick(plan.kv_surgery, config.kv_attention_surgery),
        external=_pick(plan.external, config.force_external_data),
        first_slim_no_shape_infer=plan.first_slim_no_shape_infer,
        second_slim_no_shape_infer=_pick(plan.second_slim_no_shape_infer, config.second_slim_no_shape_infer),
        affine_v2_settings=_resolve_affine_v2_settings(config),
    )
    return _fallback_unsupported_k_quant(resolved)


def validate_plan(name: str, rp: ResolvedPlan) -> None:
    valid_methods = set(_WEIGHT_ONLY_BITS) | {"DYNAMIC", "F16", "F32"}
    if rp.method not in valid_methods:
        raise ValueError(f"[{name}] unknown method {rp.method!r}; choose one of {sorted(valid_methods)}.")
    if rp.kv_surgery not in ("auto", True, False):
        raise ValueError(
            f"[{name}] kv_surgery must be 'auto', True, or False (got {rp.kv_surgery!r})."
        )
    if rp.method in _WEIGHT_ONLY_BITS:
        bits = _WEIGHT_ONLY_BITS[rp.method]
        supported_bits = _WEIGHT_ONLY_ALGO_BITS.get(rp.algo)
        if supported_bits is None:
            raise ValueError(f"[{name}] unknown algo {rp.algo!r}; choose one of {sorted(_VALID_ALGOS)}.")
        if bits not in supported_bits:
            compatible = sorted(
                algo for algo, supported in _WEIGHT_ONLY_ALGO_BITS.items() if bits in supported
            )
            raise ValueError(
                f"[{name}] algo={rp.algo!r} cannot produce {bits}-bit weights; "
                f"use one of {compatible}."
            )
        if rp.quant_format not in _QUANT_FORMATS:
            raise ValueError(f"[{name}] quant_format must be QOperator or QDQ.")
        if len(rp.op_types) != len(rp.axes):
            raise ValueError(
                f"[{name}] op_types {rp.op_types} and axes {rp.axes} must have equal length."
            )
        if "Gather" in rp.op_types and rp.algo not in ("DEFAULT", "AFFINE_REFINE_V2"):
            raise ValueError(
                f"[{name}] Gather quantization requires DEFAULT or AFFINE_REFINE_V2, got {rp.algo!r}."
            )
        if rp.quant_format == "QDQ" and (rp.algo != "DEFAULT" or bits != 4):
            raise ValueError(
                f"[{name}] QDQ supports only DEFAULT Q4, got {rp.algo!r} Q{bits}."
            )
    if rp.method == "DYNAMIC":
        if rp.dynamic_weight_type not in _DYNAMIC_WEIGHT_TYPES:
            raise ValueError(
                f"[{name}] unknown dynamic_weight_type {rp.dynamic_weight_type!r}."
            )
        if rp.algo == "AFFINE_REFINE_V2" and any(op_type != "MatMul" for op_type in rp.op_types):
            raise ValueError(
                f"[{name}] AFFINE_REFINE_V2 dynamic quantization supports MatMul only, "
                f"got {rp.op_types}."
            )


def _model_size_bytes(model_path: str) -> int:
    model_file = Path(model_path).resolve()
    data_files = {model_file}
    conventional_data_file = Path(model_path + ".data").resolve()
    if conventional_data_file.is_file():
        data_files.add(conventional_data_file)

    known_size = sum(path.stat().st_size for path in data_files)
    if model_file.stat().st_size > 2 * 1024**3:
        return known_size

    model = onnx.load(model_path, load_external_data=False)
    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        location = next(
            (entry.value for entry in tensor.external_data if entry.key == "location"),
            None,
        )
        if location:
            data_file = (model_file.parent / location).resolve()
            if data_file.is_file():
                data_files.add(data_file)
    del model
    gc.collect()
    return sum(path.stat().st_size for path in data_files)


def model_exceeds_2gb(model_path: str) -> bool:
    return _model_size_bytes(model_path) > 2 * 1024**3


def model_size_mb(model_path: str) -> float:
    return _model_size_bytes(model_path) / (1024 * 1024)


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


def _external_tensor_view(tensor: TensorProto, model_path: str) -> np.ndarray:
    """Expose one external initializer without materializing its sibling tensors."""
    if tensor.data_location != TensorProto.EXTERNAL:
        return numpy_helper.to_array(tensor)
    entries = {entry.key: entry.value for entry in tensor.external_data}
    location = entries.get("location")
    if not location:
        raise ValueError(f"External tensor {tensor.name!r} has no data location.")
    dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type))
    shape = tuple(int(dimension) for dimension in tensor.dims)
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    length = int(entries.get("length", expected_bytes))
    if length != expected_bytes:
        raise ValueError(
            f"External tensor {tensor.name!r} has {length} bytes; expected {expected_bytes}."
        )
    return np.memmap(
        Path(model_path).parent / location,
        mode="r",
        dtype=dtype,
        offset=int(entries.get("offset", "0")),
        shape=shape,
    )


def _stage_external_data_dependencies(
    model: onnx.ModelProto,
    source_path: str,
    destination_path: str,
) -> None:
    """Copy source sidecars once when a data-light graph is rewritten."""
    source_folder = Path(source_path).parent
    destination_folder = Path(destination_path).parent
    locations = {
        {entry.key: entry.value for entry in tensor.external_data}.get("location")
        for tensor in _iter_all_data_tensors(model.graph)
        if tensor.data_location == TensorProto.EXTERNAL
    }
    for location in locations - {None}:
        source = source_folder / location
        destination = destination_folder / location
        if source.resolve() == destination.resolve() or destination.exists() or not source.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


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


def _src_through_casts(name: str, producer: dict) -> str:
    while name in producer and producer[name].op_type == "Cast":
        name = producer[name].input[0]
    return name


def _dead_code_elimination(graph) -> None:
    graph_outputs = {output.name for output in graph.output}
    changed = True
    while changed:
        changed = False
        used = set(graph_outputs)
        for node in graph.node:
            used.update(node.input)
        keep = [node for node in graph.node if not node.output or any(output in used for output in node.output)]
        if len(keep) != len(graph.node):
            graph.ClearField("node")
            graph.node.extend(keep)
            changed = True


def _ensure_default_opset21(model: onnx.ModelProto) -> None:
    has_default = False
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            has_default = True
            if opset.version < 21:
                opset.version = 21
    if not has_default:
        model.opset_import.append(helper.make_opsetid("", 21))


def _read_int_list(name: str, producer: dict, init_map: dict[str, TensorProto]):
    initializer = init_map.get(name)
    if initializer is not None:
        try:
            return numpy_helper.to_array(initializer).reshape(-1).tolist()
        except Exception:
            return None
    node = producer.get(name)
    if node is not None and node.op_type == "Constant":
        for attribute in node.attribute:
            if attribute.name == "value":
                try:
                    return numpy_helper.to_array(attribute.t).reshape(-1).tolist()
                except Exception:
                    return None
    return None


def _reduce_single_axis(reduce_node, producer: dict, init_map: dict[str, TensorProto]):
    axes = None
    if len(reduce_node.input) > 1 and reduce_node.input[1]:
        axes = _read_int_list(reduce_node.input[1], producer, init_map)
    else:
        for attribute in reduce_node.attribute:
            if attribute.name == "axes":
                axes = list(attribute.ints)
    if axes is None or len(axes) != 1:
        return None
    return int(axes[0])


def _is_value_scale_tensor(name: str, producer: dict) -> bool:
    source = _src_through_casts(name, producer)
    for prefix in ("in_value_scale_", "out_value_scale_"):
        if source.startswith(prefix) and source[len(prefix):].isdigit():
            return True
    return False


def _split_value_scale_mul(mul, producer: dict) -> tuple[str, str] | None:
    if mul.op_type != "Mul" or len(mul.input) != 2:
        return None
    left, right = mul.input[0], mul.input[1]
    left_is_scale = _is_value_scale_tensor(left, producer)
    right_is_scale = _is_value_scale_tensor(right, producer)
    if left_is_scale == right_is_scale:
        return None
    return (right, left) if left_is_scale else (left, right)


def inspect_kv_surgery(graph) -> tuple[bool, str]:
    inputs = {value.name: value for value in graph.input}
    keys = [
        name
        for name in inputs
        if name.startswith("in_key_")
        and not name.startswith("in_key_scale")
        and not name.startswith("in_key_bias")
    ]
    if not keys:
        return False, "no KV cache inputs (not an attention module) - skipped"
    element_type = inputs[keys[0]].type.tensor_type.elem_type
    if element_type not in (TensorProto.INT8, TensorProto.UINT8, TensorProto.INT32):
        return (
            False,
            f"KV is not int8/uint8/int32 (elem_type={element_type}); "
            "surgery targets Q8/ROTARY_Q8/Q8_CUDA - skipped",
        )
    scale = next((inputs[name] for name in inputs if name.startswith("in_key_scale_")), None)
    grouped = False
    if scale is not None:
        rank = len(scale.type.tensor_type.shape.dim)
        if rank == 6:
            grouped = True
        elif rank != 5:
            return False, f"unexpected key_scale rank {rank} (per-head=5, grouped=6) - skipped"
    if grouped and element_type == TensorProto.INT32:
        return False, "grouped Q8_CUDA (int32-packed) KV - grouped surgery is non-CUDA only - skipped"
    initializers = {initializer.name for initializer in graph.initializer}
    if not any(node.op_type == "MatMul" and node.input[1] not in initializers for node in graph.node):
        return False, "no activation@activation matmuls to rewrite - skipped"
    asymmetric = any(name.startswith("in_key_bias_") for name in inputs)
    scheme = "asymmetric" if asymmetric else "symmetric"
    layout = "grouped" if grouped else "per-head"
    kind = (
        f"Q8_CUDA int32-packed ({scheme})"
        if element_type == TensorProto.INT32
        else f"{scheme} {'uint8' if element_type == TensorProto.UINT8 else 'int8'}"
    )
    family = "Q8/ROTARY_Q8/ROTARY_Q4" if grouped else "Q8/ROTARY_Q8"
    return True, f"{kind} KV ({family}), {layout}"


def inspect_rope_shift_surgery(graph) -> tuple[bool, str]:
    inputs = {value.name: value for value in graph.input}
    keys = [
        name
        for name in inputs
        if name.startswith("in_key_")
        and not name.startswith("in_key_scale")
        and not name.startswith("in_key_bias")
    ]
    if not keys:
        return False, "no in_key_* inputs (not a rope-shift module) - skipped"
    if any(node.op_type == "MatMul" for node in graph.node):
        return False, "has MatMul (attention module, not rope-shift) - skipped"
    element_type = inputs[keys[0]].type.tensor_type.elem_type
    if element_type in (TensorProto.FLOAT, TensorProto.FLOAT16):
        return False, "float (F16/F32) rope-shift has no quant/dequant to convert - skipped"
    if element_type not in (TensorProto.INT8, TensorProto.UINT8):
        return (
            False,
            f"non-int8/uint8 KV (elem_type={element_type}); "
            "rope-shift Q/DQ surgery targets Q8/ROTARY_Q8 - skipped",
        )
    asymmetric = any(name.startswith("in_key_bias_") for name in inputs)
    if (element_type == TensorProto.UINT8) != asymmetric:
        return False, "KV dtype/bias mismatch (int8 must be symmetric, uint8 must carry bias) - skipped"
    dimensions = inputs[keys[0]].type.tensor_type.shape.dim
    if len(dimensions) != 5 or dimensions[3].dim_value <= 0:
        return False, "unexpected key layout (need static per-head axis-3 head_dim) - skipped"
    scale = next((inputs[name] for name in inputs if name.startswith("in_key_scale_")), None)
    if scale is None or len(scale.type.tensor_type.shape.dim) != 5:
        return False, "grouped/absent key_scale (rank != 5) - rope-shift Q/DQ supports per-head only - skipped"
    operations = {node.op_type for node in graph.node}
    if not ({"Div", "Round", "ReduceMax"} <= operations):
        return False, "no quantize tail (Div/Round) - not a quantized rope-shift - skipped"
    scheme = "asymmetric uint8+bias" if asymmetric else "symmetric int8"
    return True, f"{scheme} rope-shift (Q8/ROTARY_Q8), per-head"


def inspect_kv_quantize_surgery(graph) -> tuple[bool, str]:
    inputs = {value.name: value for value in graph.input}
    keys = [name for name in inputs if name.startswith("in_key_") and "scale" not in name and "bias" not in name]
    if not keys:
        return False, "no KV cache inputs (not an attention module) - skipped"
    element_type = inputs[keys[0]].type.tensor_type.elem_type
    asymmetric = any(name.startswith("in_key_bias_") for name in inputs)
    if asymmetric and element_type != TensorProto.UINT8:
        return False, "asymmetric KV is not uint8 (Q8_CUDA int32 write tail unsupported) - skipped"
    if not asymmetric and element_type != TensorProto.INT8:
        return False, "symmetric KV is not int8 (Q8_CUDA int32 / f16 write tail unsupported) - skipped"
    scale = next((inputs[name] for name in inputs if name.startswith("in_key_scale_")), None)
    if scale is None or len(scale.type.tensor_type.shape.dim) != 5:
        return False, "grouped/absent key_scale (rank != 5) - per-head write tail only - skipped"
    key_dimensions = inputs[keys[0]].type.tensor_type.shape.dim
    if len(key_dimensions) != 5 or not key_dimensions[3].HasField("dim_value") or key_dimensions[3].dim_value <= 0:
        return False, "no static per-head head_dim on the key cache - skipped"
    scheme = "asymmetric uint8+bias" if asymmetric else "symmetric int8"
    return True, f"per-head {scheme} write tail"


def plan_kv_surgery(src_path: str, config: OptimizerConfig) -> tuple[bool, str]:
    metadata = onnx.load(src_path, load_external_data=False)
    try:
        applicable, reason = inspect_kv_surgery(metadata.graph)
        if applicable:
            tail_note = (
                " + blocked Q/DQ write tails"
                if config.kv_blocked_qdq_surgery
                else "; arithmetic write tails retained for CUDA"
            )
            return True, f"applying ({reason}) -> DynamicQuantizeMatMul{tail_note}, in-memory"
        rope_ok, rope_reason = inspect_rope_shift_surgery(metadata.graph)
        if rope_ok:
            if not config.kv_blocked_qdq_surgery:
                return False, f"{rope_reason}; blocked Q/DQ disabled for CUDA compatibility"
            return True, f"applying ({rope_reason}) -> DequantizeLinear/QuantizeLinear, in-memory"
        for rejected_reason in (reason, rope_reason):
            if "not an attention module" not in rejected_reason and "not a rope-shift module" not in rejected_reason:
                return False, rejected_reason
        return False, reason
    finally:
        del metadata


def rewire_attention_to_dynamic_quantize_matmul(model: onnx.ModelProto) -> tuple[int, int]:
    graph = model.graph
    initializers = {initializer.name for initializer in graph.initializer}
    producer = {output: node for node in graph.node for output in node.output}
    element_type: dict[str, int] = {}
    for collection in (graph.input, graph.output, graph.value_info):
        for value_info in collection:
            element_type[value_info.name] = value_info.type.tensor_type.elem_type
    for initializer in graph.initializer:
        element_type[initializer.name] = initializer.data_type

    key_inputs = [
        value
        for value in graph.input
        if value.name.startswith("in_key_") and "scale" not in value.name and "bias" not in value.name
    ]
    kv_element_type = key_inputs[0].type.tensor_type.elem_type if key_inputs else TensorProto.INT8
    is_cuda = kv_element_type == TensorProto.INT32
    asymmetric = any(value.name.startswith("in_key_bias_") for value in graph.input)
    target_type = TensorProto.UINT8 if asymmetric else TensorProto.INT8

    bzp_i8, bzp_u8 = "kvsurg_bzp_i8", "kvsurg_bzp_u8"
    for name, array in ((bzp_i8, np.array(0, np.int8)), (bzp_u8, np.array(0, np.uint8))):
        if name not in initializers:
            graph.initializer.append(numpy_helper.from_array(array, name=name))
            initializers.add(name)
    target_bzp = bzp_u8 if asymmetric else bzp_i8
    if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    def one_f32(name: str) -> str:
        if name not in initializers:
            graph.initializer.append(numpy_helper.from_array(np.array(1.0, np.float32), name=name))
            initializers.add(name)
        return name

    def prep_b(traced: str, prefix: str, tag: str) -> tuple[str, list]:
        if not is_cuda:
            return traced, []
        cast_output = f"{prefix}_{tag}_bcast"
        return cast_output, [
            helper.make_node("Cast", [traced], [cast_output], to=target_type, name=cast_output)
        ]

    new_nodes, qk_count, pv_count = [], 0, 0
    for index, node in enumerate(graph.node):
        if node.op_type != "MatMul" or node.input[1] in initializers:
            new_nodes.append(node)
            continue
        activation, right_input, output = node.input[0], node.input[1], node.output[0]
        prefix = (node.name.replace("/", "_") or "kvsurg") + f"_{index}"
        is_pv = activation in producer and producer[activation].op_type == "Softmax"
        if not is_pv:
            right_producer = producer.get(right_input)
            if right_producer is not None and right_producer.op_type == "Reshape":
                key_source = _src_through_casts(right_producer.input[0], producer)
                key_element_type = element_type.get(key_source)
                if is_cuda:
                    new_nodes.append(node)
                    continue
                key_input = f"{prefix}_qk_kre"
                if key_element_type == TensorProto.INT16:
                    key_i8 = f"{prefix}_qk_ki8"
                    casts = [
                        helper.make_node("Cast", [key_source], [key_i8], to=TensorProto.INT8, name=f"{prefix}_qk_kcast"),
                        helper.make_node("Reshape", [key_i8, right_producer.input[1]], [key_input], name=f"{prefix}_qk_reshape"),
                    ]
                    qk_bzp = bzp_i8
                elif key_element_type in (TensorProto.INT8, TensorProto.UINT8):
                    casts = [
                        helper.make_node("Reshape", [key_source, right_producer.input[1]], [key_input], name=f"{prefix}_qk_reshape")
                    ]
                    qk_bzp = bzp_u8 if key_element_type == TensorProto.UINT8 else bzp_i8
                else:
                    new_nodes.append(node)
                    continue
            else:
                key_input, casts = prep_b(_src_through_casts(right_input, producer), prefix, "qk")
                qk_bzp = target_bzp
            new_nodes.extend(casts)
            new_nodes.append(
                helper.make_node(
                    "DynamicQuantizeMatMul",
                    [activation, key_input, one_f32(f"{prefix}_qk_one_f32"), qk_bzp],
                    [output],
                    name=f"{prefix}_qk_dqmm",
                    domain="com.microsoft",
                )
            )
            qk_count += 1
            continue

        value_producer = producer.get(right_input)
        if value_producer is None:
            new_nodes.append(node)
            continue
        if value_producer.op_type == "Add":
            left_mul = producer.get(value_producer.input[0])
            right_mul = producer.get(value_producer.input[1])
            left_split = _split_value_scale_mul(left_mul, producer) if left_mul is not None else None
            right_split = _split_value_scale_mul(right_mul, producer) if right_mul is not None else None
            if left_split is not None and right_split is None:
                value_traced, value_scale_f = left_split
                value_bias = value_producer.input[1]
            elif right_split is not None and left_split is None:
                value_traced, value_scale_f = right_split
                value_bias = value_producer.input[0]
            else:
                new_nodes.append(node)
                continue
        else:
            split = _split_value_scale_mul(value_producer, producer)
            if split is None:
                new_nodes.append(node)
                continue
            value_traced, value_scale_f = split
            value_bias = None
        value_input, casts = prep_b(_src_through_casts(value_traced, producer), prefix, "pv")
        value_scale_transposed, probability_scale = f"{prefix}_pv_vst", f"{prefix}_pv_ps"
        main_output = output if value_bias is None else f"{prefix}_pv_main"
        new_nodes.extend(casts)
        new_nodes.extend([
            helper.make_node("Transpose", [value_scale_f], [value_scale_transposed], perm=[0, 1, 2, 4, 3], name=f"{prefix}_pv_tr"),
            helper.make_node("Mul", [activation, value_scale_transposed], [probability_scale], name=f"{prefix}_pv_mul"),
            helper.make_node(
                "DynamicQuantizeMatMul",
                [probability_scale, value_input, one_f32(f"{prefix}_pv_one_f32"), target_bzp],
                [main_output],
                name=f"{prefix}_pv_dqmm",
                domain="com.microsoft",
            ),
        ])
        if value_bias is not None:
            bias_matmul = f"{prefix}_pv_biasmm"
            new_nodes.extend([
                helper.make_node("MatMul", [activation, value_bias], [bias_matmul], name=f"{prefix}_pv_biasmm"),
                helper.make_node("Add", [main_output, bias_matmul], [output], name=f"{prefix}_pv_biasadd"),
            ])
        pv_count += 1

    graph.ClearField("node")
    graph.node.extend(new_nodes)
    _dead_code_elimination(graph)
    return qk_count, pv_count


def rewire_rope_shift_to_qdq(model: onnx.ModelProto) -> int:
    graph = model.graph
    inputs = {value.name: value for value in graph.input}
    producer = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    key_inputs = [
        value.name
        for value in graph.input
        if value.name.startswith("in_key_")
        and not value.name.startswith("in_key_scale")
        and not value.name.startswith("in_key_bias")
    ]
    asymmetric = any(value.name.startswith("in_key_bias_") for value in graph.input)
    zero_point_type = TensorProto.UINT8 if asymmetric else TensorProto.INT8
    kv_axis = 3
    head_dim = inputs[key_inputs[0]].type.tensor_type.shape.dim[kv_axis].dim_value

    def single_consumer(name: str) -> bool:
        return len(consumers.get(name, [])) == 1

    to_delete, replace, count = set(), {}, 0
    for key_input in key_inputs:
        index = key_input.rsplit("_", 1)[1]
        scale_input, key_output = f"in_key_scale_{index}", f"out_key_{index}"

        cast_chain, current = [], key_input
        while True:
            next_casts = [node for node in consumers.get(current, []) if node.op_type == "Cast"]
            if len(next_casts) != 1:
                break
            cast_chain.append(next_casts[0])
            current = next_casts[0].output[0]
        if not cast_chain:
            continue
        multiplications = [node for node in consumers.get(cast_chain[-1].output[0], []) if node.op_type == "Mul"]
        if len(multiplications) != 1:
            continue
        multiplication = multiplications[0]
        scale_operands = [name for name in multiplication.input if name != cast_chain[-1].output[0]]
        if len(scale_operands) != 1:
            continue
        scale_f32 = scale_operands[0]
        scale_producer = producer.get(scale_f32)
        if not (
            scale_f32 == scale_input
            or (
                scale_producer is not None
                and scale_producer.op_type == "Cast"
                and scale_producer.input
                and scale_producer.input[0] == scale_input
            )
        ):
            continue

        node, tail = producer.get(key_output), []
        while node is not None and node.op_type in ("Cast", "Clip", "Round"):
            tail.append(node)
            node = producer.get(node.input[0])
        if not tail or node is None or node.op_type != "Div":
            continue
        division = node
        if not all(single_consumer(node.output[0]) for node in tail[1:]) or not single_consumer(division.output[0]):
            continue
        quantized_input, scale_new = division.input[0], division.input[1]

        dequantize = helper.make_node(
            "DequantizeLinear",
            [key_input, scale_f32],
            [multiplication.output[0]],
            axis=kv_axis,
            block_size=head_dim,
            name=f"ropeq_dql_{index}",
        )
        scale_shape, zero_point = f"ropeq_scale_shape_{index}", f"ropeq_zero_point_{index}"
        shape_node = helper.make_node("Shape", [scale_new], [scale_shape], name=f"ropeq_shape_{index}")
        zero_node = helper.make_node(
            "ConstantOfShape",
            [scale_shape],
            [zero_point],
            value=helper.make_tensor(f"ropeq_zero_val_{index}", zero_point_type, [1], [0]),
            name=f"ropeq_zero_{index}",
        )
        quantize = helper.make_node(
            "QuantizeLinear",
            [quantized_input, scale_new, zero_point],
            [key_output],
            axis=kv_axis,
            block_size=head_dim,
            name=f"ropeq_ql_{index}",
        )

        to_delete.update(id(node) for node in cast_chain)
        to_delete.update(id(node) for node in tail[1:])
        to_delete.add(id(division))
        replace[id(multiplication)] = [dequantize]
        replace[id(tail[0])] = [shape_node, zero_node, quantize]
        count += 1

    if count == 0:
        return 0
    new_nodes = []
    for node in graph.node:
        if id(node) in to_delete:
            continue
        new_nodes.extend(replace.get(id(node), [node]))
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    _dead_code_elimination(graph)
    _ensure_default_opset21(model)
    return count


def rewire_kv_quantize_to_quantizelinear(model: onnx.ModelProto) -> int:
    graph = model.graph
    applicable, _ = inspect_kv_quantize_surgery(graph)
    if not applicable:
        return 0
    producer = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    initializers = {initializer.name: initializer for initializer in graph.initializer}

    key_input = next(
        value
        for value in graph.input
        if value.name.startswith("in_key_") and "scale" not in value.name and "bias" not in value.name
    )
    head_dim = key_input.type.tensor_type.shape.dim[3].dim_value
    asymmetric = any(value.name.startswith("in_key_bias_") for value in graph.input)
    zero_point_type = TensorProto.UINT8 if asymmetric else TensorProto.INT8

    def find_reduce(name: str):
        node = producer.get(name)
        if node is None:
            return None
        if node.op_type in ("ReduceMax", "ReduceMin"):
            return node
        if node.op_type == "Sub":
            for input_name in node.input:
                reduce_node = find_reduce(input_name)
                if reduce_node is not None:
                    return reduce_node
        return None

    to_delete, replace, count = set(), {}, 0
    for division in graph.node:
        if division.op_type != "Div":
            continue
        round_consumers = consumers.get(division.output[0], [])
        if len(round_consumers) != 1 or round_consumers[0].op_type != "Round":
            continue
        round_node = round_consumers[0]
        current, clip_nodes = round_node.output[0], []
        current_consumers = consumers.get(current, [])
        if len(current_consumers) == 1 and current_consumers[0].op_type == "Clip":
            clip_nodes = [current_consumers[0]]
            current = current_consumers[0].output[0]
        cast_chain = []
        while True:
            next_nodes = consumers.get(current, [])
            if len(next_nodes) == 1 and next_nodes[0].op_type == "Cast":
                cast_chain.append(next_nodes[0])
                current = next_nodes[0].output[0]
            else:
                break
        if not cast_chain:
            continue
        packed = current
        concat = next((node for node in consumers.get(packed, []) if node.op_type == "Concat"), None)
        if concat is None or not (
            concat.output[0].startswith("out_key_") or concat.output[0].startswith("out_value_")
        ):
            continue
        activation, scale = division.input[0], division.input[1]
        scale_mul = producer.get(scale)
        if scale_mul is None or scale_mul.op_type != "Mul":
            continue
        reduce_node = None
        for input_name in scale_mul.input:
            reduce_node = find_reduce(input_name)
            if reduce_node is not None:
                break
        if reduce_node is None:
            continue
        axis = _reduce_single_axis(reduce_node, producer, initializers)
        if axis is None:
            continue
        if axis < 0:
            axis += 5

        scale_shape, zero_point = f"kvq_scale_shape_{count}", f"kvq_zero_point_{count}"
        shape_node = helper.make_node("Shape", [scale], [scale_shape], name=f"kvq_shape_{count}")
        zero_node = helper.make_node(
            "ConstantOfShape",
            [scale_shape],
            [zero_point],
            value=helper.make_tensor(f"kvq_zero_val_{count}", zero_point_type, [1], [0]),
            name=f"kvq_zero_{count}",
        )
        quantize = helper.make_node(
            "QuantizeLinear",
            [activation, scale, zero_point],
            [packed],
            axis=axis,
            block_size=head_dim,
            name=f"kvq_ql_{count}",
        )

        replace[id(division)] = [shape_node, zero_node, quantize]
        to_delete.add(id(round_node))
        to_delete.update(id(node) for node in clip_nodes)
        to_delete.update(id(node) for node in cast_chain)
        count += 1

    if count == 0:
        return 0
    new_nodes = []
    for node in graph.node:
        if id(node) in to_delete:
            continue
        new_nodes.extend(replace.get(id(node), [node]))
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    _dead_code_elimination(graph)
    _ensure_default_opset21(model)
    return count


def apply_kv_surgery(model: onnx.ModelProto, config: OptimizerConfig) -> None:
    applicable, _ = inspect_kv_surgery(model.graph)
    if applicable:
        qk_count, pv_count = rewire_attention_to_dynamic_quantize_matmul(model)
        write_tail_count = (
            rewire_kv_quantize_to_quantizelinear(model) if config.kv_blocked_qdq_surgery else 0
        )
        message = f"    surgery: {qk_count} Q@K + {pv_count} attn@V -> DynamicQuantizeMatMul"
        if write_tail_count:
            message += f"; {write_tail_count} KV write tails -> QuantizeLinear (blocked int8)"
        elif not config.kv_blocked_qdq_surgery:
            message += "; preserved arithmetic KV write tails (CUDA-compatible)"
        print(message)
        return
    if not config.kv_blocked_qdq_surgery:
        return
    applicable, _ = inspect_rope_shift_surgery(model.graph)
    if applicable:
        count = rewire_rope_shift_to_qdq(model)
        print(f"    surgery: {count} rope-shift layers -> DequantizeLinear/QuantizeLinear (blocked int8)")


def _apply_kv_surgery_if_requested(
    model: onnx.ModelProto,
    do_surgery: bool,
    config: OptimizerConfig | None,
) -> None:
    if not do_surgery:
        return
    if config is None:
        raise ValueError("KV surgery requires an OptimizerConfig.")
    apply_kv_surgery(model, config)


def resave(
    src_path: str,
    dst_path: str,
    external: bool,
    *,
    do_surgery: bool = False,
    config: OptimizerConfig | None = None,
) -> None:
    model = onnx.load(src_path)
    _apply_kv_surgery_if_requested(model, do_surgery, config)
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


@dataclass
class Q4RefineStats:
    blocks: int = 0
    improved_blocks: int = 0
    seed_error: float = 0.0
    refined_error: float = 0.0

    def add(self, other: "Q4RefineStats") -> None:
        self.blocks += other.blocks
        self.improved_blocks += other.improved_blocks
        self.seed_error += other.seed_error
        self.refined_error += other.refined_error


_K_QUANT_SEARCH_OFFSETS = np.asarray(
    tuple(-1.0 + 0.1 * index for index in range(20)), dtype=np.float32
)
_K_QUANT_FINAL_CHUNK_VALUES = 262144


def quant_tensor_k_quant_cpu(
    data: np.ndarray,
    num_bits: int = 4,
    group_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize rows with ORT's k-quant objective using bounded CPU buffers."""
    if num_bits < 1:
        raise ValueError(f"num_bits must be positive, got {num_bits}.")
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    values = np.ascontiguousarray(np.asarray(data).reshape(-1, group_size), dtype=np.float32)
    block_count = values.shape[0]
    maxq = (1 << num_bits) - 1
    maxq_float = np.float32(maxq)
    quantized = np.empty_like(values)
    scratch = np.empty_like(values)
    weighted_quantized = np.empty_like(values)

    np.multiply(values, values, out=scratch)
    rms = np.sqrt(np.sum(scratch, axis=1, dtype=np.float32) / np.float32(group_size))
    weights = np.empty_like(values)
    np.abs(values, out=weights)
    np.add(weights, rms[:, None], out=weights)

    minimum = np.min(values, axis=1)
    maximum = np.max(values, axis=1)
    span = maximum - minimum
    sum_weights = np.sum(weights, axis=1, dtype=np.float32)
    np.multiply(weights, values, out=scratch)
    sum_weighted_values = np.sum(scratch, axis=1, dtype=np.float32)

    inverse_scale = np.ones(block_count, dtype=np.float32)
    varying = span != 0.0
    np.divide(maxq_float, span, out=inverse_scale, where=varying)
    best_scale = np.reciprocal(inverse_scale)
    best_minimum = minimum.copy()

    np.subtract(values, best_minimum[:, None], out=scratch)
    np.multiply(scratch, inverse_scale[:, None], out=scratch)
    np.rint(scratch, out=quantized)
    np.clip(quantized, 0.0, maxq_float, out=quantized)
    np.multiply(quantized, best_scale[:, None], out=scratch)
    np.add(scratch, best_minimum[:, None], out=scratch)
    np.subtract(scratch, values, out=scratch)
    np.square(scratch, out=scratch)
    np.multiply(scratch, weights, out=scratch)
    best_error = np.sum(scratch, axis=1, dtype=np.float32)

    candidate_inverse_scale = np.empty(block_count, dtype=np.float32)
    sum_l = np.empty(block_count, dtype=np.float32)
    sum_l2 = np.empty(block_count, dtype=np.float32)
    sum_xl = np.empty(block_count, dtype=np.float32)
    determinant = np.empty(block_count, dtype=np.float32)
    numerator = np.empty(block_count, dtype=np.float32)
    row_scratch = np.empty(block_count, dtype=np.float32)
    candidate_scale = np.empty(block_count, dtype=np.float32)
    candidate_minimum = np.empty(block_count, dtype=np.float32)
    candidate_error = np.empty(block_count, dtype=np.float32)
    valid = np.empty(block_count, dtype=bool)
    improved = np.empty(block_count, dtype=bool)

    for offset in _K_QUANT_SEARCH_OFFSETS:
        np.subtract(maximum, best_minimum, out=span)
        np.not_equal(span, 0.0, out=valid)
        candidate_inverse_scale.fill(1.0)
        np.divide(maxq_float + offset, span, out=candidate_inverse_scale, where=valid)

        np.subtract(values, best_minimum[:, None], out=scratch)
        np.multiply(scratch, candidate_inverse_scale[:, None], out=scratch)
        np.rint(scratch, out=quantized)
        np.clip(quantized, 0.0, maxq_float, out=quantized)

        np.multiply(weights, quantized, out=weighted_quantized)
        np.sum(weighted_quantized, axis=1, dtype=np.float32, out=sum_l)
        np.multiply(weighted_quantized, quantized, out=scratch)
        np.sum(scratch, axis=1, dtype=np.float32, out=sum_l2)
        np.multiply(weighted_quantized, values, out=scratch)
        np.sum(scratch, axis=1, dtype=np.float32, out=sum_xl)

        np.multiply(sum_weights, sum_l2, out=determinant)
        np.multiply(sum_l, sum_l, out=row_scratch)
        np.subtract(determinant, row_scratch, out=determinant)
        np.not_equal(determinant, 0.0, out=valid)
        np.logical_and(valid, np.isfinite(determinant), out=valid)

        np.multiply(sum_weights, sum_xl, out=numerator)
        np.multiply(sum_weighted_values, sum_l, out=row_scratch)
        np.subtract(numerator, row_scratch, out=numerator)
        candidate_scale.fill(0.0)
        np.divide(numerator, determinant, out=candidate_scale, where=valid)

        np.multiply(sum_l2, sum_weighted_values, out=numerator)
        np.multiply(sum_l, sum_xl, out=row_scratch)
        np.subtract(numerator, row_scratch, out=numerator)
        candidate_minimum.fill(0.0)
        np.divide(numerator, determinant, out=candidate_minimum, where=valid)
        np.logical_and(valid, np.isfinite(candidate_scale), out=valid)
        np.logical_and(valid, candidate_scale > 0.0, out=valid)
        np.logical_and(valid, np.isfinite(candidate_minimum), out=valid)

        np.multiply(quantized, candidate_scale[:, None], out=scratch)
        np.add(scratch, candidate_minimum[:, None], out=scratch)
        np.subtract(scratch, values, out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, weights, out=scratch)
        np.sum(scratch, axis=1, dtype=np.float32, out=candidate_error)
        np.less(candidate_error, best_error, out=improved)
        np.logical_and(improved, valid, out=improved)
        np.copyto(best_error, candidate_error, where=improved)
        np.copyto(best_scale, candidate_scale, where=improved)
        np.copyto(best_minimum, candidate_minimum, where=improved)

    zero_point_float = np.empty(block_count, dtype=np.float32)
    np.negative(best_minimum, out=zero_point_float)
    np.divide(zero_point_float, best_scale, out=zero_point_float)
    np.rint(zero_point_float, out=zero_point_float)
    np.clip(zero_point_float, 0.0, maxq_float, out=zero_point_float)
    zero_point = zero_point_float.astype(np.uint8).reshape(-1, 1)

    rows_per_chunk = max(1, _K_QUANT_FINAL_CHUNK_VALUES // group_size)
    final_buffer = np.empty((min(block_count, rows_per_chunk), group_size), dtype=np.float64)
    for start in range(0, block_count, rows_per_chunk):
        end = min(start + rows_per_chunk, block_count)
        final = final_buffer[:end - start]
        np.divide(values[start:end], best_scale[start:end, None], out=final)
        np.add(final, zero_point[start:end], out=final)
        np.rint(final, out=final)
        np.clip(final, 0.0, float(maxq), out=final)
        quantized[start:end] = final
    return quantized, best_scale.reshape(-1, 1), zero_point


def _validate_affine_v2_settings(settings: AffineV2Settings) -> None:
    positive_fields = {
        "seed_iterations": settings.seed_iterations,
        "seed_chunk_blocks": settings.seed_chunk_blocks,
        "seed_blocks_per_job": settings.seed_blocks_per_job,
        "seed_workers": settings.seed_workers,
        "numba_threads": settings.numba_threads,
        "iterations": settings.iterations,
        "chunk_blocks": settings.chunk_blocks,
    }
    invalid = [name for name, value in positive_fields.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"AFFINE_REFINE_V2 requires positive values for {', '.join(invalid)}.")
    if settings.seed_zp_radius < 0:
        raise ValueError("AFFINE_REFINE_V2 seed zero-point radius must be nonnegative.")
    if settings.weighted_tolerance < 0.0:
        raise ValueError("AFFINE_REFINE_V2 weighted tolerance must be nonnegative.")
    if settings.asym_zp_sweep_limit < 16:
        raise ValueError("AFFINE_REFINE_V2 asymmetric zero-point sweep limit must be at least 16.")
    ratios = np.asarray(settings.clip_ratios, dtype=np.float32)
    if ratios.ndim != 1 or not ratios.size or np.any((ratios <= 0.0) | (ratios > 1.0)):
        raise ValueError("AFFINE_REFINE_V2 clip ratios must be a non-empty sequence in (0, 1].")


def _iter_affine_v2_row_chunks(values: np.ndarray, block_size: int, max_blocks: int):
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    padded_columns = block_count * block_size
    rows_per_chunk = max(1, max_blocks // block_count)
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = np.ascontiguousarray(values[row_start:row_end], dtype=np.float32)
        if padded_columns != columns:
            chunk = np.pad(chunk, ((0, 0), (0, padded_columns - columns)), mode="constant")
        yield row_start * block_count, row_end * block_count, chunk.reshape(-1, block_size)


@lru_cache(maxsize=None)
def _affine_v2_seed_executor(worker_count: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="affine-v2-seed")


def _quantize_affine_v2_seed_partition(weight: np.ndarray, block_size: int, bits: int):
    with np.errstate(divide="ignore", invalid="ignore"):
        return quant_tensor_k_quant_cpu(weight, bits, block_size)


def _quantize_affine_v2_seed_blocks(
    weight: np.ndarray,
    block_size: int,
    bits: int,
    settings: AffineV2Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maxq = float((1 << bits) - 1)
    max_workers = max(1, min(settings.seed_workers, os.cpu_count() or 1))
    worker_count = min(max_workers, max(1, weight.shape[0] // settings.seed_blocks_per_job))
    if worker_count == 1:
        varying_q, varying_scale, varying_zp = _quantize_affine_v2_seed_partition(
            weight, block_size, bits
        )
        return (
            np.clip(np.asarray(varying_q, dtype=np.float32), 0.0, maxq),
            np.asarray(varying_scale, dtype=np.float32).reshape(-1, 1),
            np.clip(np.asarray(varying_zp, dtype=np.int16).reshape(-1, 1), 0, int(maxq)).astype(np.uint8),
        )

    partitions = np.array_split(weight, worker_count, axis=0)
    futures = [
        _affine_v2_seed_executor(max_workers).submit(
            _quantize_affine_v2_seed_partition, partition, block_size, bits
        )
        for partition in partitions
    ]
    seed_q = np.empty(weight.shape, dtype=np.float32)
    seed_scale = np.empty((weight.shape[0], 1), dtype=np.float32)
    seed_zp = np.empty((weight.shape[0], 1), dtype=np.uint8)
    offset = 0
    for partition, future in zip(partitions, futures):
        varying_q, varying_scale, varying_zp = future.result()
        end = offset + partition.shape[0]
        seed_q[offset:end] = np.clip(np.asarray(varying_q, dtype=np.float32), 0.0, maxq)
        seed_scale[offset:end] = np.asarray(varying_scale, dtype=np.float32).reshape(-1, 1)
        seed_zp[offset:end] = np.clip(
            np.asarray(varying_zp, dtype=np.int16).reshape(-1, 1), 0, int(maxq)
        ).astype(np.uint8)
        offset = end
    return seed_q, seed_scale, seed_zp


def _affine_v2_seed_blocks(
    weight: np.ndarray,
    block_size: int,
    symmetric: bool,
    bits: int,
    settings: AffineV2Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the raw k-quant seed, including deterministic constant-block handling."""
    maxq = float((1 << bits) - 1)
    midpoint = float(1 << (bits - 1))
    if symmetric:
        tiny = np.finfo(np.float32).tiny
        positive_max = np.maximum(weight.max(axis=1, keepdims=True), np.float32(0.0))
        negative_max = np.maximum(-weight.min(axis=1, keepdims=True), np.float32(0.0))
        seed_scale = np.maximum(
            positive_max / np.float32(maxq - midpoint),
            negative_max / np.float32(midpoint),
        )
        seed_scale = np.where(seed_scale > tiny, seed_scale, np.float32(1.0)).astype(np.float32)
        seed_q = np.clip(np.rint(weight / seed_scale + np.float32(midpoint)), 0.0, maxq)
        seed_zp = np.full((weight.shape[0], 1), int(midpoint), dtype=np.uint8)
        return seed_q, seed_scale, seed_zp

    constant = np.ptp(weight, axis=1) == 0.0
    if not np.any(constant):
        seed_q, seed_scale, seed_zp = _quantize_affine_v2_seed_blocks(
            weight, block_size, bits, settings
        )
    else:
        seed_q = np.empty(weight.shape, dtype=np.float32)
        seed_scale = np.empty((weight.shape[0], 1), dtype=np.float32)
        seed_zp = np.empty((weight.shape[0], 1), dtype=np.uint8)
        varying = ~constant
        if np.any(varying):
            varying_q, varying_scale, varying_zp = _quantize_affine_v2_seed_blocks(
                weight[varying], block_size, bits, settings
            )
            seed_q[varying] = varying_q
            seed_scale[varying] = varying_scale
            seed_zp[varying] = varying_zp
        constant_value = weight[constant, :1]
        positive = constant_value > 0.0
        negative = constant_value < 0.0
        seed_q[constant] = np.where(positive, np.float32(maxq), np.float32(0.0))
        seed_scale[constant] = np.where(
            positive,
            constant_value / np.float32(maxq),
            np.where(negative, -constant_value / np.float32(maxq), np.float32(1.0)),
        )
        seed_zp[constant] = np.where(negative, np.uint8(int(maxq)), np.uint8(0))

    tiny = np.finfo(np.float32).tiny
    valid_scale = np.isfinite(seed_scale) & (seed_scale > tiny)
    if not np.all(valid_scale):
        fallback_scale = (
            weight.max(axis=1, keepdims=True) - weight.min(axis=1, keepdims=True)
        ) / np.float32(maxq)
        fallback_scale = np.where(fallback_scale > tiny, fallback_scale, np.float32(1.0))
        seed_scale = np.where(valid_scale, seed_scale, fallback_scale)
    return seed_q, seed_scale, seed_zp


def _affine_v2_seed_refine_rows(
    data: np.ndarray,
    block_size: int,
    symmetric: bool,
    bits: int,
    settings: AffineV2Settings,
    allow_arbitrary_block_size: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Q4RefineStats]:
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"AFFINE_REFINE_V2 seed expects a 2-D row matrix, got shape {values.shape}.")
    if not allow_arbitrary_block_size and (
        block_size < 16 or block_size > 256 or block_size & (block_size - 1)
    ):
        raise ValueError(
            f"AFFINE_REFINE_V2 seed block_size must be a power of two in [16, 256], got {block_size}."
        )
    if not np.isfinite(values).all():
        raise ValueError("AFFINE_REFINE_V2 seed refuses weights containing NaN or Inf.")

    maxq = float((1 << bits) - 1)
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    total_blocks = rows * block_count
    quantized = np.empty((total_blocks, block_size), dtype=np.uint8)
    scales = np.empty((total_blocks, 1), dtype=np.float32)
    zero_points = np.empty((total_blocks, 1), dtype=np.uint8)
    stats = Q4RefineStats(blocks=total_blocks)
    tiny = np.finfo(np.float32).tiny

    for start, end, weight in _iter_affine_v2_row_chunks(
        values, block_size, settings.seed_chunk_blocks
    ):
        seed_q, seed_scale, seed_zp = _affine_v2_seed_blocks(
            weight, block_size, symmetric, bits, settings
        )
        importance = np.sqrt(np.mean(weight * weight, axis=1, keepdims=True)) + np.abs(weight)
        importance = np.where(
            importance.sum(axis=1, keepdims=True) > 0.0, importance, np.float32(1.0)
        )
        seed_dequant = seed_scale * (seed_q - seed_zp.astype(np.float32))
        seed_error = np.sum(importance * (weight - seed_dequant) ** 2, axis=1, keepdims=True)

        best_q = seed_q.copy()
        best_scale = seed_scale.copy()
        best_zp = seed_zp.copy()
        best_error = seed_error.copy()
        zp_deltas = (0,) if symmetric else range(-settings.seed_zp_radius, settings.seed_zp_radius + 1)
        for delta in zp_deltas:
            candidate_zp = np.clip(seed_zp.astype(np.int16) + delta, 0, int(maxq)).astype(np.float32)
            candidate_scale = seed_scale.copy()
            for _ in range(settings.seed_iterations):
                candidate_q = np.clip(np.rint(weight / candidate_scale + candidate_zp), 0.0, maxq)
                centered_q = candidate_q - candidate_zp
                denominator = np.sum(importance * centered_q * centered_q, axis=1, keepdims=True)
                numerator = np.sum(importance * centered_q * weight, axis=1, keepdims=True)
                fitted_scale = np.divide(
                    numerator, denominator, out=candidate_scale.copy(), where=denominator > tiny
                )
                candidate_scale = np.where(
                    np.isfinite(fitted_scale) & (fitted_scale > tiny), fitted_scale, candidate_scale
                )

            candidate_q = np.clip(np.rint(weight / candidate_scale + candidate_zp), 0.0, maxq)
            candidate_error = np.sum(
                importance * (weight - candidate_scale * (candidate_q - candidate_zp)) ** 2,
                axis=1,
                keepdims=True,
            )
            take = candidate_error[:, 0] < best_error[:, 0]
            best_q[take] = candidate_q[take]
            best_scale[take] = candidate_scale[take]
            best_zp[take] = candidate_zp[take].astype(np.uint8)
            best_error[take] = candidate_error[take]

        quantized[start:end] = best_q.astype(np.uint8)
        scales[start:end] = best_scale
        zero_points[start:end] = best_zp
        stats.improved_blocks += int(np.count_nonzero(best_error[:, 0] < seed_error[:, 0]))
        stats.seed_error += float(seed_error.sum(dtype=np.float64))
        stats.refined_error += float(best_error.sum(dtype=np.float64))

    return (
        quantized.reshape(rows, block_count, block_size),
        scales.reshape(rows, block_count),
        zero_points.reshape(rows, block_count),
        stats,
    )


@lru_cache(maxsize=1)
def _affine_v2_numba_kernel():
    """Build the optional fused CPU kernel lazily so standard ORT paths stay light."""
    try:
        from numba import njit, prange, set_num_threads
    except ImportError:
        return None

    # This helper is sometimes imported through a temporary module while scripts
    # inspect it. Disk caching would then require that transient module name on a
    # later normal import, so keep the optional JIT cache in-process only.
    @njit(parallel=True, nogil=True, cache=False)
    def refine_blocks(
        weight,
        quantized,
        scales,
        zero_points,
        clip_ratios,
        seed_iterations,
        seed_zp_radius,
        affine_iterations,
        tolerance,
        tiny,
        symmetric,
        maxq,
        midpoint,
        zp_sweep_limit,
    ):
        max_code = np.float32(maxq)
        max_code_int = int(maxq)
        block_count, width = weight.shape
        baseline_errors = np.empty(block_count, dtype=np.float32)
        refined_errors = np.empty(block_count, dtype=np.float32)
        improved = np.zeros(block_count, dtype=np.bool_)
        candidate_codes = np.empty((block_count, width), dtype=np.uint8)

        for block_index in prange(block_count):
            sum_squares = np.float32(0.0)
            positive_max = np.float32(0.0)
            negative_max = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block_index, column])
                sum_squares += value * value
                if value > positive_max:
                    positive_max = value
                if -value > negative_max:
                    negative_max = -value
            rms = np.float32(np.sqrt(sum_squares / np.float32(width)))

            raw_scale = np.float32(scales[block_index])
            raw_zero_point_int = int(zero_points[block_index])
            raw_zero_point = np.float32(raw_zero_point_int)
            best_seed_error = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block_index, column])
                centered = np.float32(quantized[block_index, column]) - raw_zero_point
                residual = value - raw_scale * centered
                importance = np.float32(1.0) if sum_squares == 0.0 else rms + np.abs(value)
                best_seed_error += importance * residual * residual

            for delta in range(0 if symmetric else -seed_zp_radius, 1 if symmetric else seed_zp_radius + 1):
                candidate_zero_point_int = min(max_code_int, max(0, raw_zero_point_int + delta))
                candidate_zero_point = np.float32(candidate_zero_point_int)
                candidate_scale = raw_scale
                for _ in range(seed_iterations):
                    denominator = np.float32(0.0)
                    numerator = np.float32(0.0)
                    for column in range(width):
                        value = np.float32(weight[block_index, column])
                        candidate_q = np.rint(value / candidate_scale + candidate_zero_point)
                        candidate_q = min(max_code, max(np.float32(0.0), candidate_q))
                        centered = candidate_q - candidate_zero_point
                        importance = np.float32(1.0) if sum_squares == 0.0 else rms + np.abs(value)
                        denominator += importance * centered * centered
                        numerator += importance * centered * value
                    if denominator <= tiny:
                        break
                    fitted_scale = numerator / denominator
                    if not np.isfinite(fitted_scale) or fitted_scale <= tiny:
                        break
                    if fitted_scale == candidate_scale:
                        break
                    candidate_scale = fitted_scale

                candidate_seed_error = np.float32(0.0)
                for column in range(width):
                    value = np.float32(weight[block_index, column])
                    candidate_q = np.rint(value / candidate_scale + candidate_zero_point)
                    candidate_q = min(max_code, max(np.float32(0.0), candidate_q))
                    candidate_codes[block_index, column] = np.uint8(candidate_q)
                    centered = candidate_q - candidate_zero_point
                    residual = value - candidate_scale * centered
                    importance = np.float32(1.0) if sum_squares == 0.0 else rms + np.abs(value)
                    candidate_seed_error += importance * residual * residual
                if candidate_seed_error < best_seed_error:
                    best_seed_error = candidate_seed_error
                    scales[block_index] = candidate_scale
                    zero_points[block_index] = np.uint8(candidate_zero_point_int)
                    for column in range(width):
                        quantized[block_index, column] = candidate_codes[block_index, column]

            seed_scale = np.float32(scales[block_index])
            seed_zero_point = np.float32(zero_points[block_index])
            seed_zero_point_int = int(zero_points[block_index])
            baseline_plain = np.float32(0.0)
            baseline_weighted = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block_index, column])
                centered = np.float32(quantized[block_index, column]) - seed_zero_point
                residual = value - seed_scale * centered
                squared = residual * residual
                baseline_plain += squared
                baseline_weighted += (rms + np.abs(value)) * squared

            local_plain = baseline_plain
            weighted_bound = tolerance * baseline_weighted
            if symmetric:
                zp_lo = midpoint
                zp_hi = midpoint
            elif max_code_int + 1 <= zp_sweep_limit:
                zp_lo = 0
                zp_hi = max_code_int
            else:
                zp_lo = seed_zero_point_int - zp_sweep_limit // 2
                if zp_lo < 0:
                    zp_lo = 0
                zp_hi = zp_lo + zp_sweep_limit - 1
                if zp_hi > max_code_int:
                    zp_hi = max_code_int
                    zp_lo = zp_hi - zp_sweep_limit + 1
                    if zp_lo < 0:
                        zp_lo = 0

            for zero_point_int in range(zp_lo, zp_hi + 1):
                zero_point = np.float32(zero_point_int)
                positive_scale = np.float32(0.0)
                negative_scale = np.float32(0.0)
                if zero_point_int < max_code_int:
                    positive_scale = positive_max / np.float32(max_code_int - zero_point_int)
                if zero_point_int > 0:
                    negative_scale = negative_max / np.float32(zero_point_int)
                coverage_scale = max(positive_scale, negative_scale)
                if coverage_scale <= tiny:
                    coverage_scale = np.float32(1.0)

                for start_index in range(clip_ratios.size + 1):
                    if start_index == 0:
                        candidate_scale = seed_scale
                    else:
                        candidate_scale = coverage_scale * clip_ratios[start_index - 1]
                    for _ in range(affine_iterations):
                        denominator = np.float32(0.0)
                        numerator = np.float32(0.0)
                        for column in range(width):
                            value = np.float32(weight[block_index, column])
                            candidate_q = np.rint(value / candidate_scale + zero_point)
                            candidate_q = min(max_code, max(np.float32(0.0), candidate_q))
                            centered = candidate_q - zero_point
                            denominator += centered * centered
                            numerator += centered * value
                        if denominator <= tiny:
                            break
                        fitted_scale = numerator / denominator
                        if not np.isfinite(fitted_scale) or fitted_scale <= tiny:
                            break
                        if fitted_scale == candidate_scale:
                            break
                        candidate_scale = fitted_scale

                    candidate_plain = np.float32(0.0)
                    candidate_weighted = np.float32(0.0)
                    for column in range(width):
                        value = np.float32(weight[block_index, column])
                        candidate_q = np.rint(value / candidate_scale + zero_point)
                        candidate_q = min(max_code, max(np.float32(0.0), candidate_q))
                        candidate_codes[block_index, column] = np.uint8(candidate_q)
                        centered = candidate_q - zero_point
                        residual = value - candidate_scale * centered
                        squared = residual * residual
                        candidate_plain += squared
                        candidate_weighted += (rms + np.abs(value)) * squared
                    if candidate_plain < local_plain and candidate_weighted <= weighted_bound:
                        local_plain = candidate_plain
                        scales[block_index] = candidate_scale
                        zero_points[block_index] = np.uint8(zero_point_int)
                        for column in range(width):
                            quantized[block_index, column] = candidate_codes[block_index, column]

            baseline_errors[block_index] = baseline_plain
            refined_errors[block_index] = local_plain
            improved[block_index] = local_plain < baseline_plain

        return baseline_errors, refined_errors, improved

    return refine_blocks, set_num_threads


def _affine_refine_v2_rows(
    data: np.ndarray,
    block_size: int,
    symmetric: bool,
    bits: int,
    settings: AffineV2Settings,
    allow_arbitrary_block_size: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Q4RefineStats]:
    """Minimize plain block MSE while bounding weighted degradation from the k-quant seed."""
    _validate_affine_v2_settings(settings)
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"AFFINE_REFINE_V2 expects a 2-D row matrix, got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("AFFINE_REFINE_V2 refuses weights containing NaN or Inf.")
    if not allow_arbitrary_block_size and (
        block_size < 16 or block_size > 256 or block_size & (block_size - 1)
    ):
        raise ValueError(
            f"AFFINE_REFINE_V2 block_size must be a power of two in [16, 256], got {block_size}."
        )
    if bits not in (4, 7, 8):
        raise ValueError(f"AFFINE_REFINE_V2 supports 4-, 7-, or 8-bit weights, got {bits}-bit.")

    maxq = float((1 << bits) - 1)
    midpoint = int(1 << (bits - 1))
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    total_blocks = rows * block_count
    tiny = np.finfo(np.float32).tiny
    tolerance = np.float32(1.0 + settings.weighted_tolerance)
    ratios = np.asarray(settings.clip_ratios, dtype=np.float32)
    stats = Q4RefineStats(blocks=total_blocks)

    numba_kernel = _affine_v2_numba_kernel()
    if numba_kernel is not None:
        refine_blocks, set_num_threads = numba_kernel
        set_num_threads(settings.numba_threads)
        best_q = np.empty((total_blocks, block_size), dtype=np.uint8)
        best_scales = np.empty(total_blocks, dtype=np.float32)
        best_zero_points = np.empty(total_blocks, dtype=np.uint8)
        for start, end, weight in _iter_affine_v2_row_chunks(values, block_size, settings.chunk_blocks):
            seed_q, seed_scales, seed_zero_points = _affine_v2_seed_blocks(
                weight, block_size, symmetric, bits, settings
            )
            local_q = best_q[start:end]
            local_scales = best_scales[start:end]
            local_zero_points = best_zero_points[start:end]
            local_q[:] = seed_q
            local_scales[:] = seed_scales[:, 0]
            local_zero_points[:] = seed_zero_points[:, 0]
            baseline_plain, local_plain, local_improved = refine_blocks(
                weight,
                local_q,
                local_scales,
                local_zero_points,
                ratios,
                settings.seed_iterations,
                settings.seed_zp_radius,
                settings.iterations,
                tolerance,
                tiny,
                symmetric,
                np.float32(maxq),
                np.int64(midpoint),
                np.int64(settings.asym_zp_sweep_limit),
            )
            stats.improved_blocks += int(np.count_nonzero(local_improved))
            stats.seed_error += float(baseline_plain.sum(dtype=np.float64))
            stats.refined_error += float(local_plain.sum(dtype=np.float64))
        return (
            best_q.reshape(rows, block_count, block_size),
            best_scales.reshape(rows, block_count),
            best_zero_points.reshape(rows, block_count),
            stats,
        )

    best_q, best_scales, best_zero_points, _ = _affine_v2_seed_refine_rows(
        values, block_size, symmetric, bits, settings, allow_arbitrary_block_size
    )
    best_q = best_q.reshape(-1, block_size)
    best_scales = best_scales.reshape(-1)
    best_zero_points = best_zero_points.reshape(-1)

    for start, end, weight in _iter_affine_v2_row_chunks(values, block_size, settings.chunk_blocks):
        local_q = best_q[start:end]
        local_scales = best_scales[start:end]
        local_zero_points = best_zero_points[start:end]
        importance = np.sqrt(np.mean(weight * weight, axis=1, keepdims=True)) + np.abs(weight)
        importance = np.where(
            importance.sum(axis=1, keepdims=True) > 0.0, importance, np.float32(1.0)
        )
        seed_scales = local_scales.copy().reshape(-1, 1)
        seed_q = local_q.astype(np.float32)
        seed_zero_points = local_zero_points.astype(np.float32).reshape(-1, 1)
        seed_residual = weight - seed_scales * (seed_q - seed_zero_points)
        baseline_plain = np.sum(seed_residual * seed_residual, axis=1)
        baseline_weighted = np.sum(importance * seed_residual * seed_residual, axis=1)
        weighted_bound = tolerance * baseline_weighted
        local_plain = baseline_plain.copy()
        local_improved = np.zeros(end - start, dtype=bool)
        positive_max = np.maximum(weight.max(axis=1, keepdims=True), np.float32(0.0))
        negative_max = np.maximum(-weight.min(axis=1, keepdims=True), np.float32(0.0))
        maxq_int = int(maxq)

        if symmetric or maxq_int + 1 <= settings.asym_zp_sweep_limit:
            zero_point_candidates = (midpoint,) if symmetric else range(maxq_int + 1)
            for zero_point_int in zero_point_candidates:
                zero_point = np.float32(zero_point_int)
                positive_scale = (
                    positive_max / np.float32(maxq_int - zero_point_int)
                    if zero_point_int < maxq_int else np.zeros_like(positive_max)
                )
                negative_scale = (
                    negative_max / np.float32(zero_point_int)
                    if zero_point_int > 0 else np.zeros_like(negative_max)
                )
                coverage_scale = np.maximum(positive_scale, negative_scale)
                coverage_scale = np.where(coverage_scale > tiny, coverage_scale, np.float32(1.0))
                starts = [seed_scales, *(coverage_scale * ratio for ratio in ratios)]
                for initial_scale in starts:
                    candidate_scale = initial_scale.copy()
                    for _ in range(settings.iterations):
                        candidate_q = np.clip(
                            np.rint(weight / candidate_scale + zero_point), 0.0, maxq
                        )
                        centered_q = candidate_q - zero_point
                        denominator = np.sum(centered_q * centered_q, axis=1, keepdims=True)
                        numerator = np.sum(centered_q * weight, axis=1, keepdims=True)
                        fitted_scale = np.divide(
                            numerator, denominator, out=candidate_scale.copy(), where=denominator > tiny
                        )
                        candidate_scale = np.where(
                            np.isfinite(fitted_scale) & (fitted_scale > tiny),
                            fitted_scale,
                            candidate_scale,
                        )
                    candidate_q = np.clip(np.rint(weight / candidate_scale + zero_point), 0.0, maxq)
                    residual = weight - candidate_scale * (candidate_q - zero_point)
                    candidate_plain = np.sum(residual * residual, axis=1)
                    candidate_weighted = np.sum(importance * residual * residual, axis=1)
                    take = (candidate_plain < local_plain) & (candidate_weighted <= weighted_bound)
                    local_q[take] = candidate_q[take].astype(np.uint8)
                    local_scales[take] = candidate_scale[take, 0]
                    local_zero_points[take] = np.uint8(zero_point_int)
                    local_plain[take] = candidate_plain[take]
                    local_improved[take] = True
        else:
            half = settings.asym_zp_sweep_limit // 2
            seed_zp = local_zero_points.astype(np.int64).reshape(-1, 1)
            window_lo = np.clip(seed_zp - half, 0, maxq_int)
            window_lo = np.clip(
                window_lo - np.maximum(window_lo + settings.asym_zp_sweep_limit - 1 - maxq_int, 0),
                0,
                maxq_int,
            )
            for offset in range(settings.asym_zp_sweep_limit):
                zp_int = np.clip(window_lo + offset, 0, maxq_int)
                zero_point = zp_int.astype(np.float32)
                positive_denominator = np.float32(maxq_int) - zero_point
                positive_scale = np.where(
                    positive_denominator > 0.0,
                    positive_max / np.where(
                        positive_denominator > 0.0, positive_denominator, np.float32(1.0)
                    ),
                    np.float32(0.0),
                )
                negative_scale = np.where(
                    zero_point > 0.0,
                    negative_max / np.where(zero_point > 0.0, zero_point, np.float32(1.0)),
                    np.float32(0.0),
                )
                coverage_scale = np.maximum(positive_scale, negative_scale)
                coverage_scale = np.where(coverage_scale > tiny, coverage_scale, np.float32(1.0))
                starts = [seed_scales, *(coverage_scale * ratio for ratio in ratios)]
                for initial_scale in starts:
                    candidate_scale = initial_scale.copy()
                    for _ in range(settings.iterations):
                        candidate_q = np.clip(
                            np.rint(weight / candidate_scale + zero_point), 0.0, maxq
                        )
                        centered_q = candidate_q - zero_point
                        denominator = np.sum(centered_q * centered_q, axis=1, keepdims=True)
                        numerator = np.sum(centered_q * weight, axis=1, keepdims=True)
                        fitted_scale = np.divide(
                            numerator, denominator, out=candidate_scale.copy(), where=denominator > tiny
                        )
                        candidate_scale = np.where(
                            np.isfinite(fitted_scale) & (fitted_scale > tiny),
                            fitted_scale,
                            candidate_scale,
                        )
                    candidate_q = np.clip(np.rint(weight / candidate_scale + zero_point), 0.0, maxq)
                    residual = weight - candidate_scale * (candidate_q - zero_point)
                    candidate_plain = np.sum(residual * residual, axis=1)
                    candidate_weighted = np.sum(importance * residual * residual, axis=1)
                    take = (candidate_plain < local_plain) & (candidate_weighted <= weighted_bound)
                    local_q[take] = candidate_q[take].astype(np.uint8)
                    local_scales[take] = candidate_scale[take, 0]
                    local_zero_points[take] = zp_int[take, 0].astype(np.uint8)
                    local_plain[take] = candidate_plain[take]
                    local_improved[take] = True

        stats.improved_blocks += int(np.count_nonzero(local_improved))
        stats.seed_error += float(baseline_plain.sum(dtype=np.float64))
        stats.refined_error += float(local_plain.sum(dtype=np.float64))

    return (
        best_q.reshape(rows, block_count, block_size),
        best_scales.reshape(rows, block_count),
        best_zero_points.reshape(rows, block_count),
        stats,
    )


def _pack_q4_last_axis(values: np.ndarray, pad_value: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if values.shape[-1] & 1:
        values = np.pad(
            values,
            [(0, 0)] * (values.ndim - 1) + [(0, 1)],
            constant_values=pad_value,
        )
    return (values[..., 0::2] | (values[..., 1::2] << 4)).astype(np.uint8)


def _pack_codes_last_axis(values: np.ndarray, bits: int, pad_value: int = 0) -> np.ndarray:
    """Pack MatMulNBits codes: nibbles for Q4 and raw bytes for Q8."""
    if bits == 8:
        return np.ascontiguousarray(np.asarray(values, dtype=np.uint8))
    if bits == 4:
        return _pack_q4_last_axis(values, pad_value)
    raise ValueError(f"unsupported MatMulNBits bit width {bits}; expected 4 or 8.")


def _make_uint4_initializer(name: str, values: np.ndarray) -> TensorProto:
    values = np.asarray(values, dtype=np.uint8)
    flat = values.reshape(-1)
    if flat.size & 1:
        flat = np.pad(flat, (0, 1))
    packed = (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)
    return helper.make_tensor(name, TensorProto.UINT4, values.shape, packed.tobytes(), raw=True)


def _make_uintn_initializer(name: str, values: np.ndarray, bits: int) -> TensorProto:
    """Make logical UINT4/UINT8 GatherBlockQuantized data and zero-point tensors."""
    if bits == 8:
        return numpy_helper.from_array(np.ascontiguousarray(np.asarray(values, dtype=np.uint8)), name=name)
    if bits == 4:
        return _make_uint4_initializer(name, values)
    raise ValueError(f"unsupported GatherBlockQuantized bit width {bits}; expected 4 or 8.")


def _make_quant_initializer(name: str, values: np.ndarray) -> TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(values), name=name)


def _init_map(graph) -> dict[str, TensorProto]:
    return {initializer.name: initializer for initializer in graph.initializer}


def _graph_used_names(graph) -> set[str]:
    used = {value.name for value in graph.input}
    used.update(value.name for value in graph.output)
    used.update(value.name for value in graph.value_info)
    used.update(initializer.name for initializer in graph.initializer)
    for node in graph.node:
        if node.name:
            used.add(node.name)
        used.update(name for name in node.input if name)
        used.update(name for name in node.output if name)
    return used


def _make_name_factory(graph, prefix: str):
    used = _graph_used_names(graph)

    def make(suffix: str) -> str:
        base = f"{prefix}{suffix}"
        if base not in used:
            used.add(base)
            return base
        index = 1
        while f"{base}_{index}" in used:
            index += 1
        name = f"{base}_{index}"
        used.add(name)
        return name

    return make


def _drop_unused_initializers(graph) -> int:
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    unused = {initializer.name for initializer in graph.initializer if initializer.name not in used}
    if unused:
        retained = [initializer for initializer in graph.initializer if initializer.name not in unused]
        graph.ClearField("initializer")
        graph.initializer.extend(retained)
    return len(unused)


def _strip_stale_matmul_nbits_attributes_graph(graph) -> int:
    removed = 0
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g"):
                removed += _strip_stale_matmul_nbits_attributes_graph(attribute.g)
            for subgraph in attribute.graphs:
                removed += _strip_stale_matmul_nbits_attributes_graph(subgraph)
        if node.op_type != "MatMulNBits":
            continue
        retained = [attribute for attribute in node.attribute if attribute.name != "weight_prepacked"]
        if len(retained) == len(node.attribute):
            continue
        node.ClearField("attribute")
        node.attribute.extend(retained)
        removed += 1
    return removed


def strip_stale_matmul_nbits_attributes(model_path: str | Path) -> int:
    """Remove ORT-private MatMulNBits attributes that the public schema rejects."""
    path = str(model_path)
    model = onnx.load(path, load_external_data=False)
    removed = _strip_stale_matmul_nbits_attributes_graph(model.graph)
    if removed:
        onnx.save(model, path)
    del model
    gc.collect()
    return removed


def _ensure_ms_domain_opset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == "com.microsoft":
            opset.version = max(opset.version, 1)
            return
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def _resolve_custom_quant_plan(rp: ResolvedPlan, src_path: str) -> ResolvedPlan:
    return replace(
        rp,
        nodes_to_include=_resolve_nodes(rp.nodes_to_include, src_path),
        nodes_to_exclude=_resolve_nodes(rp.nodes_to_exclude, src_path),
    )


def _k_quant_q4_rows(
    data: np.ndarray,
    block_size: int,
    settings: AffineV2Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"k_quant expects a 2-D row matrix, got shape {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("k_quant refuses weights containing NaN or Inf.")
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    quantized = np.empty((rows * block_count, block_size), dtype=np.uint8)
    scales = np.empty(rows * block_count, dtype=np.float32)
    zero_points = np.empty(rows * block_count, dtype=np.uint8)
    for start, end, weight in _iter_affine_v2_row_chunks(values, block_size, settings.seed_chunk_blocks):
        chunk_q, chunk_scales, chunk_zero_points = quant_tensor_k_quant_cpu(weight, 4, block_size)
        quantized[start:end] = np.clip(chunk_q, 0.0, 15.0).astype(np.uint8)
        scales[start:end] = np.asarray(chunk_scales, dtype=np.float32).reshape(-1)
        zero_points[start:end] = np.clip(
            np.asarray(chunk_zero_points, dtype=np.int16).reshape(-1), 0, 15
        ).astype(np.uint8)
    return (
        quantized.reshape(rows, block_count, block_size),
        scales.reshape(rows, block_count),
        zero_points.reshape(rows, block_count),
    )


def _quantize_k_quant_matmul(graph, node, weight: TensorProto, rp: ResolvedPlan, make_name):
    weight_array = numpy_helper.to_array(weight)
    if weight_array.ndim != 2 or weight_array.dtype.kind != "f":
        print(f"  k_quant: skipping {node.name or weight.name!r}; MatMul weight must be a floating-point matrix.")
        return None
    input_features, output_features = weight_array.shape
    quantized, scales, zero_points = _k_quant_q4_rows(
        weight_array.T, rp.block_size, rp.affine_v2_settings
    )
    weight_name = make_name("weight")
    scale_name = make_name("scales")
    zero_point_name = make_name("zero_points")
    graph.initializer.extend([
        _make_quant_initializer(weight_name, _pack_q4_last_axis(quantized)),
        _make_quant_initializer(scale_name, scales.astype(weight_array.dtype, copy=False)),
        _make_quant_initializer(zero_point_name, _pack_q4_last_axis(zero_points, pad_value=8)),
    ])
    attributes = {"K": input_features, "N": output_features, "bits": 4, "block_size": rp.block_size}
    if rp.accuracy_level:
        attributes["accuracy_level"] = rp.accuracy_level
    return helper.make_node(
        "MatMulNBits",
        [node.input[0], weight_name, scale_name, zero_point_name],
        list(node.output),
        name=f"{node.name}_K_QUANT_Q4" if node.name else make_name("matmul"),
        domain="com.microsoft",
        **attributes,
    )


def quantize_k_quant_model(model: onnx.ModelProto, rp: ResolvedPlan) -> int:
    """Replace selected constant MatMuls with chunked CPU k-quant Q4 ops."""
    quantized_matmuls = 0

    def rewrite_graph(graph) -> None:
        nonlocal quantized_matmuls
        init_map = _init_map(graph)
        make_name = _make_name_factory(graph, "k_quant_q4_")
        replaced_initializers: set[str] = set()
        new_nodes = []
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite_graph(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite_graph(subgraph)
            selected = node.op_type == "MatMul" and "MatMul" in rp.op_types
            if rp.nodes_to_include is not None:
                selected = selected and node.name in rp.nodes_to_include
            if rp.nodes_to_exclude is not None and node.name in rp.nodes_to_exclude:
                selected = False
            replacement = None
            if selected and len(node.input) >= 2:
                weight = init_map.get(node.input[1])
                if weight is not None:
                    replacement = _quantize_k_quant_matmul(graph, node, weight, rp, make_name)
                    if replacement is not None:
                        quantized_matmuls += 1
                        replaced_initializers.add(weight.name)
            new_nodes.append(replacement or node)
        graph.ClearField("node")
        graph.node.extend(new_nodes)
        _drop_unused_initializers(graph)
        obsolete_inputs = replaced_initializers - {initializer.name for initializer in graph.initializer}
        if obsolete_inputs:
            retained_inputs = [value for value in graph.input if value.name not in obsolete_inputs]
            graph.ClearField("input")
            graph.input.extend(retained_inputs)

    rewrite_graph(model.graph)
    if quantized_matmuls:
        _ensure_ms_domain_opset(model)
        _deduplicate_node_names(model.graph)
    print(f"  k_quant CPU surgery: {quantized_matmuls} MatMul -> MatMulNBits.")
    return quantized_matmuls


def _quantize_affine_v2_matmul(
    graph,
    node,
    weight: TensorProto,
    rp: ResolvedPlan,
    bits: int,
    make_name,
    weight_array: np.ndarray | None = None,
):
    if weight_array is None:
        weight_array = numpy_helper.to_array(weight)
    if weight_array.ndim != 2 or weight_array.dtype.kind != "f":
        print(f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; MatMul weight must be a floating-point matrix.")
        return None, None
    input_features, output_features = weight_array.shape
    quantized, scales, zero_points, stats = _affine_refine_v2_rows(
        weight_array.T, rp.block_size, rp.symmetric, bits, rp.affine_v2_settings
    )
    weight_name = make_name("weight")
    scale_name = make_name("scales")
    zero_point_name = make_name("zero_points")
    graph.initializer.extend([
        _make_quant_initializer(weight_name, _pack_codes_last_axis(quantized, bits)),
        _make_quant_initializer(scale_name, scales.astype(weight_array.dtype, copy=False)),
        _make_quant_initializer(
            zero_point_name,
            _pack_codes_last_axis(zero_points, bits, pad_value=1 << (bits - 1)),
        ),
    ])
    attributes = {"K": input_features, "N": output_features, "bits": bits, "block_size": rp.block_size}
    if rp.accuracy_level:
        attributes["accuracy_level"] = rp.accuracy_level
    return helper.make_node(
        "MatMulNBits",
        [node.input[0], weight_name, scale_name, zero_point_name],
        list(node.output),
        name=f"{node.name}_AFFINE_REFINE_V2_Q{bits}" if node.name else make_name("matmul"),
        domain="com.microsoft",
        **attributes,
    ), stats


def _gather_axis(node) -> int:
    for attribute in node.attribute:
        if attribute.name == "axis":
            return int(helper.get_attribute_value(attribute))
    return 0


def _quantize_affine_v2_gather(
    graph,
    node,
    weight: TensorProto,
    rp: ResolvedPlan,
    bits: int,
    quantize_axis: int,
    make_name,
    weight_array: np.ndarray | None = None,
):
    if weight_array is None:
        weight_array = numpy_helper.to_array(weight)
    rank = weight_array.ndim
    if not rank or weight_array.dtype.kind != "f":
        print(f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; Gather data must be floating point.")
        return None, None
    quantize_axis = (quantize_axis + rank) % rank
    gather_axis = (_gather_axis(node) + rank) % rank
    if gather_axis != 0 or quantize_axis != rank - 1:
        print(
            f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; GatherBlockQuantized "
            "requires gather_axis=0 and quantize_axis=last for CPU/CUDA portability."
        )
        return None, None
    logical_width = weight_array.shape[-1]
    if logical_width % rp.block_size:
        print(
            f"  AFFINE_REFINE_V2: skipping {node.name or weight.name!r}; Gather width {logical_width} "
            f"is not divisible by block_size={rp.block_size}, which CUDA does not handle portably."
        )
        return None, None
    outer_shape = weight_array.shape[:-1]
    rows = int(np.prod(outer_shape, dtype=np.int64))
    quantized, scales, zero_points, stats = _affine_refine_v2_rows(
        weight_array.reshape(rows, logical_width),
        rp.block_size,
        rp.symmetric,
        bits,
        rp.affine_v2_settings,
    )
    logical_quantized = quantized.reshape(rows, -1)[:, :logical_width].reshape(weight_array.shape)
    block_count = scales.shape[-1]
    scales = scales.reshape(*outer_shape, block_count).astype(weight_array.dtype, copy=False)
    zero_points = zero_points.reshape(*outer_shape, block_count)
    weight_name = make_name("weight")
    scale_name = make_name("scales")
    zero_point_name = make_name("zero_points")
    graph.initializer.extend([
        _make_uintn_initializer(weight_name, logical_quantized, bits),
        _make_quant_initializer(scale_name, scales),
        _make_uintn_initializer(zero_point_name, zero_points, bits),
    ])
    return helper.make_node(
        "GatherBlockQuantized",
        [weight_name, node.input[1], scale_name, zero_point_name],
        list(node.output),
        name=f"{node.name}_AFFINE_REFINE_V2_Q{bits}" if node.name else make_name("gather"),
        domain="com.microsoft",
        gather_axis=gather_axis,
        quantize_axis=quantize_axis,
        block_size=rp.block_size,
        bits=bits,
    ), stats


def quantize_affine_v2_model(
    model: onnx.ModelProto,
    rp: ResolvedPlan,
    bits: int,
    weight_loader: Callable[[TensorProto], np.ndarray] | None = None,
) -> Q4RefineStats:
    """Replace selected constant MatMul/Gather nodes with AFFINE_REFINE_V2 Q4/Q8 ops."""
    if rp.quant_format != "QOPERATOR":
        raise ValueError("AFFINE_REFINE_V2 supports QOperator format only.")
    if bits not in (4, 8):
        raise ValueError(f"AFFINE_REFINE_V2 supports 4- or 8-bit weights, got {bits}-bit.")
    quant_axes = dict(zip(rp.op_types, rp.axes))
    total = Q4RefineStats()
    quantized_matmuls = 0
    quantized_gathers = 0

    def rewrite_graph(graph) -> None:
        nonlocal quantized_matmuls, quantized_gathers
        init_map = _init_map(graph)
        make_name = _make_name_factory(graph, f"affine_refine_v2_q{bits}_")
        replaced_initializers: set[str] = set()
        new_nodes = []
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite_graph(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite_graph(subgraph)
            selected = node.op_type in rp.op_types
            if rp.nodes_to_include is not None:
                selected = selected and node.name in rp.nodes_to_include
            if rp.nodes_to_exclude is not None and node.name in rp.nodes_to_exclude:
                selected = False
            replacement = None
            stats = None
            if selected and node.op_type == "MatMul" and len(node.input) >= 2:
                weight = init_map.get(node.input[1])
                if weight is not None:
                    weight_array = weight_loader(weight) if weight_loader else None
                    replacement, stats = _quantize_affine_v2_matmul(
                        graph, node, weight, rp, bits, make_name, weight_array
                    )
                    if replacement is not None:
                        quantized_matmuls += 1
                        replaced_initializers.add(weight.name)
            elif selected and node.op_type == "Gather" and len(node.input) >= 2:
                weight = init_map.get(node.input[0])
                if weight is not None:
                    weight_array = weight_loader(weight) if weight_loader else None
                    replacement, stats = _quantize_affine_v2_gather(
                        graph,
                        node,
                        weight,
                        rp,
                        bits,
                        quant_axes.get("Gather", 1),
                        make_name,
                        weight_array,
                    )
                    if replacement is not None:
                        quantized_gathers += 1
                        replaced_initializers.add(weight.name)
            new_nodes.append(replacement or node)
            if stats is not None:
                total.add(stats)
        graph.ClearField("node")
        graph.node.extend(new_nodes)
        _drop_unused_initializers(graph)
        obsolete_inputs = replaced_initializers - {initializer.name for initializer in graph.initializer}
        if obsolete_inputs:
            retained_inputs = [value for value in graph.input if value.name not in obsolete_inputs]
            graph.ClearField("input")
            graph.input.extend(retained_inputs)

    rewrite_graph(model.graph)
    if quantized_matmuls or quantized_gathers:
        _ensure_ms_domain_opset(model)
        _deduplicate_node_names(model.graph)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  AFFINE_REFINE_V2 surgery: {quantized_matmuls} MatMul -> MatMulNBits, "
        f"{quantized_gathers} Gather -> GatherBlockQuantized; improved "
        f"{total.improved_blocks}/{total.blocks} blocks over its internal seed, plain MSE ratio={ratio:.6f}."
    )
    return total


def _quantize_affine_v2_dynamic_matmul(graph, node, weight: TensorProto, rp: ResolvedPlan, make_name):
    weight_array = numpy_helper.to_array(weight)
    if weight_array.ndim != 2 or weight_array.dtype.kind != "f":
        print(
            f"  AFFINE_REFINE_V2 dynamic: skipping {node.name or weight.name!r}; "
            "MatMul weight must be a floating-point matrix."
        )
        return None, None

    bits = 7 if rp.reduce_range else 8
    if rp.per_channel:
        rows = np.ascontiguousarray(weight_array.T, dtype=np.float32)
        block_size = weight_array.shape[0]
    else:
        rows = np.ascontiguousarray(weight_array.reshape(1, -1), dtype=np.float32)
        block_size = weight_array.size
    quantized, scales, zero_points, stats = _affine_refine_v2_rows(
        rows,
        block_size,
        rp.symmetric,
        bits,
        rp.affine_v2_settings,
        allow_arbitrary_block_size=True,
    )
    quantized = quantized.reshape(rows.shape)
    scales = scales.reshape(-1)
    zero_points = zero_points.reshape(-1)
    quantized = quantized.T if rp.per_channel else quantized.reshape(weight_array.shape)
    if rp.dynamic_weight_type == "QINT8":
        offset = 1 << (bits - 1)
        quantized = (quantized.astype(np.int16) - offset).astype(np.int8)
        zero_points = (zero_points.astype(np.int16) - offset).astype(np.int8)
    else:
        quantized = quantized.astype(np.uint8)
        zero_points = zero_points.astype(np.uint8)
    if not rp.per_channel:
        scales = scales[0]
        zero_points = zero_points[0]

    weight_name = make_name(f"{weight.name}_quantized")
    scale_name = make_name(f"{weight.name}_scale")
    zero_point_name = make_name(f"{weight.name}_zero_point")
    graph.initializer.extend([
        _make_quant_initializer(weight_name, quantized),
        _make_quant_initializer(scale_name, scales.astype(np.float32, copy=False)),
        _make_quant_initializer(zero_point_name, zero_points),
    ])
    return helper.make_node(
        "DynamicQuantizeMatMul",
        [node.input[0], weight_name, scale_name, zero_point_name],
        list(node.output),
        name=make_name(f"{node.name or 'matmul'}_dynamic_quantize_matmul"),
        domain="com.microsoft",
    ), stats


def quantize_affine_v2_dynamic_model(model: onnx.ModelProto, rp: ResolvedPlan) -> Q4RefineStats:
    """Replace selected constant MatMuls with V2-refined dynamic INT8/UINT8 ops."""
    total = Q4RefineStats()
    quantized_matmuls = 0

    def rewrite_graph(graph) -> None:
        nonlocal quantized_matmuls
        init_map = _init_map(graph)
        make_name = _make_name_factory(graph, "affine_refine_v2_dynamic_")
        replaced_initializers: set[str] = set()
        new_nodes = []
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite_graph(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite_graph(subgraph)
            selected = node.op_type == "MatMul" and "MatMul" in rp.op_types
            if rp.nodes_to_include is not None:
                selected = selected and node.name in rp.nodes_to_include
            if rp.nodes_to_exclude is not None and node.name in rp.nodes_to_exclude:
                selected = False
            replacement = None
            stats = None
            if selected and len(node.input) >= 2:
                weight = init_map.get(node.input[1])
                if weight is not None:
                    replacement, stats = _quantize_affine_v2_dynamic_matmul(
                        graph, node, weight, rp, make_name
                    )
                    if replacement is not None:
                        quantized_matmuls += 1
                        replaced_initializers.add(weight.name)
            new_nodes.append(replacement or node)
            if stats is not None:
                total.add(stats)
        graph.ClearField("node")
        graph.node.extend(new_nodes)
        _drop_unused_initializers(graph)
        obsolete_inputs = replaced_initializers - {initializer.name for initializer in graph.initializer}
        if obsolete_inputs:
            retained_inputs = [value for value in graph.input if value.name not in obsolete_inputs]
            graph.ClearField("input")
            graph.input.extend(retained_inputs)

    rewrite_graph(model.graph)
    if quantized_matmuls:
        _ensure_ms_domain_opset(model)
    _deduplicate_node_names(model.graph)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  AFFINE_REFINE_V2 dynamic surgery: {quantized_matmuls} MatMul -> DynamicQuantizeMatMul; "
        f"improved {total.improved_blocks}/{total.blocks} channels/tensors over its internal seed, "
        f"plain MSE ratio={ratio:.6f}."
    )
    return total


def build_weight_only_config(rp: ResolvedPlan, bits: int):
    algo = rp.algo
    if algo == "AFFINE_REFINE_V2":
        raise ValueError("AFFINE_REFINE_V2 is handled by the custom common quantization path.")
    if algo == "k_quant":
        raise ValueError("k_quant is handled by the custom CPU quantization path.")
    _ALGO_CONFIG_CLASSES = {
        "RTN": "RTNWeightOnlyQuantConfig",
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


def quantize_weight_only(
    src_path: str,
    dst_path: str,
    rp: ResolvedPlan,
    bits: int,
    external: bool,
    *,
    do_surgery: bool = False,
    config: OptimizerConfig | None = None,
) -> None:
    if rp.algo in {"AFFINE_REFINE_V2", "k_quant"}:
        header_model = onnx.load(src_path, load_external_data=False)
        lazy_external = (
            rp.algo == "AFFINE_REFINE_V2"
            and not do_surgery
            and any(
                tensor.data_location == TensorProto.EXTERNAL
                for tensor in _iter_all_data_tensors(header_model.graph)
            )
        )
        model = header_model if lazy_external else quant_utils.load_model_with_shape_infer(Path(src_path))
        _apply_kv_surgery_if_requested(model, do_surgery, config)
        converted_constants = _materialize_constant_tensors_as_initializers(model.graph)
        if converted_constants:
            print(f"  Materialized {converted_constants} Constant tensor nodes as initializers for weight quantization.")
        exposed_aliases = _eliminate_initializer_identity_aliases(model.graph)
        if exposed_aliases:
            print(f"  Eliminated {exposed_aliases} initializer Identity aliases before weight quantization.")
        custom_plan = _resolve_custom_quant_plan(rp, src_path)
        if rp.algo == "AFFINE_REFINE_V2":
            if bits not in (4, 8):
                raise ValueError(f"{rp.algo} supports Q4/Q8 only, got {bits}-bit.")
            quantization_passes = [(custom_plan, bits)]
            if bits != 4 and "Gather" in custom_plan.op_types:
                operator_axes = tuple(zip(custom_plan.op_types, custom_plan.axes))
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
                                custom_plan,
                                op_types=tuple(op_type for op_type, _ in non_gather),
                                axes=tuple(axis for _, axis in non_gather),
                            ),
                            bits,
                        )
                    )
                if gather:
                    quantization_passes.append(
                        (
                            replace(
                                custom_plan,
                                op_types=tuple(op_type for op_type, _ in gather),
                                axes=tuple(axis for _, axis in gather),
                            ),
                            4,
                        )
                    )
                print(
                    "  GatherBlockQuantized uses portable Q4 packing; keeping "
                    f"{bits}-bit packing for the remaining operators."
                )
            for pass_plan, pass_bits in quantization_passes:
                print(
                    f"  Quantizing weights ({rp.algo}, {pass_bits}-bit, block={rp.block_size}, "
                    f"symmetric={rp.symmetric}, format={rp.quant_format}, ops={list(pass_plan.op_types)})..."
                )
                quantize_affine_v2_model(
                    model,
                    pass_plan,
                    pass_bits,
                    (
                        lambda tensor: _external_tensor_view(tensor, src_path)
                        if lazy_external
                        else None
                    ),
                )
        else:
            if bits != 4:
                raise ValueError(f"{rp.algo} supports Q4 only, got {bits}-bit.")
            print(
                f"  Quantizing weights ({rp.algo}, 4-bit, block={rp.block_size}, "
                f"format={rp.quant_format}, ops={list(rp.op_types)}, CPU-only)..."
            )
            quantize_k_quant_model(model, custom_plan)
        if lazy_external:
            _stage_external_data_dependencies(model, src_path, dst_path)
        _save_model(model, dst_path, external)
        del model
        gc.collect()
        return

    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    _apply_kv_surgery_if_requested(model, do_surgery, config)
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
    if unsupported_op_types:
        raise ValueError(
            "shared weight-only quantization supports MatMul/Gather only, got "
            f"{unsupported_op_types}."
        )
    template_src_path = str(Path(template_src_path).resolve())
    cache_path = str(Path(cache_path).resolve())
    targets = [(str(Path(src).resolve()), str(Path(dst).resolve())) for src, dst in model_paths]
    missing_sources = [src for src, _ in targets if not os.path.isfile(src)]
    if not os.path.isfile(template_src_path):
        missing_sources.insert(0, template_src_path)
    if missing_sources:
        raise FileNotFoundError(
            "shared weight-only quantization is missing source graph(s): "
            + ", ".join(missing_sources)
        )
    quantize_weight_only(template_src_path, cache_path, rp, bits, external)

    template_includes = _resolve_nodes(rp.nodes_to_include, template_src_path)
    template_includes = None if template_includes is None else set(template_includes)
    template_excludes = set(_resolve_nodes(rp.nodes_to_exclude, template_src_path) or ())

    template = onnx.load(template_src_path, load_external_data=False)
    _materialize_constant_tensors_as_initializers(template.graph)
    _eliminate_initializer_identity_aliases(template.graph)
    template_initializers = {initializer.name: initializer for initializer in template.graph.initializer}

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

    template_signatures = {
        name: _source_tensor_signature(initializer, template_src_path)
        for name, initializer in template_initializers.items()
    }
    content_signature_cache: dict[tuple[str, str], tuple] = {}

    def content_signature(tensor: TensorProto, model_path: str) -> tuple:
        key = (str(Path(model_path).resolve()), tensor.name)
        cached = content_signature_cache.get(key)
        if cached is None:
            cached = _source_tensor_content_signature(tensor, model_path)
            content_signature_cache[key] = cached
        return cached

    recipes: dict[tuple[str, str], tuple[onnx.NodeProto, str, tuple[str | None, ...], tuple]] = {}
    skipped_recipes: dict[tuple[str, str], tuple] = {}
    template_rewrites = 0
    for node in template.graph.node:
        if (
            node.op_type not in rp.op_types
            or
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
        if quantized_node is None:
            raise RuntimeError(
                f"shared quantization did not rewrite selected {node.op_type} "
                f"node {node.name or node.output[0]!r} in the template graph."
            )
        recipe_key = (node.op_type, weight_name)
        expected_op_type = {
            "MatMul": "MatMulNBits",
            "Gather": "GatherBlockQuantized",
        }[node.op_type]
        if quantized_node.op_type != expected_op_type:
            skipped_recipes[recipe_key] = template_signatures[weight_name]
            continue
        quantized_inputs = tuple(
            None if index == dynamic_index else name
            for index, name in enumerate(quantized_node.input)
        )
        missing_inputs = [
            name for name in quantized_inputs
            if name and name not in quantized_initializers
        ]
        if missing_inputs:
            raise RuntimeError(
                f"shared quantization cache is missing initializer(s) for "
                f"{node.name or node.output[0]!r}: {missing_inputs}."
            )
        name_suffix = (
            quantized_node.name[len(node.name):]
            if node.name and quantized_node.name.startswith(node.name)
            else f"_{quantized_node.op_type}"
        )
        prior = recipes.get(recipe_key)
        if prior is not None:
            prior_node, _, prior_inputs, _ = prior
            same_attributes = [attr.SerializeToString() for attr in prior_node.attribute] == [
                attr.SerializeToString() for attr in quantized_node.attribute
            ]
            same_layout = (
                prior_node.op_type == quantized_node.op_type
                and prior_node.domain == quantized_node.domain
                and len(prior_inputs) == len(quantized_inputs)
                and tuple(name is None for name in prior_inputs)
                == tuple(name is None for name in quantized_inputs)
            )
            if not (same_attributes and same_layout):
                raise RuntimeError(
                    f"template uses incompatible shared packing recipes for "
                    f"{node.op_type}:{weight_name}."
                )
        else:
            recipes[recipe_key] = (
                quantized_node,
                name_suffix,
                quantized_inputs,
                template_signatures[weight_name],
            )
        template_rewrites += 1

    if not recipes:
        raise RuntimeError("shared quantization template contains no selected constant weights.")

    total_rewrites = 0
    total_removed_initializers = 0
    for src_path, dst_path in targets:
        target_includes = _resolve_nodes(rp.nodes_to_include, src_path)
        target_includes = None if target_includes is None else set(target_includes)
        target_excludes = set(_resolve_nodes(rp.nodes_to_exclude, src_path) or ())
        source_header = onnx.load(src_path, load_external_data=False)
        _materialize_constant_tensors_as_initializers(source_header.graph)
        _eliminate_initializer_identity_aliases(source_header.graph)
        source_initializers = {
            initializer.name: initializer for initializer in source_header.graph.initializer
        }
        source_signatures = {
            initializer.name: _source_tensor_signature(initializer, src_path)
            for initializer in source_initializers.values()
        }
        model = onnx.load(src_path, load_external_data=False)
        _materialize_constant_tensors_as_initializers(model.graph)
        _eliminate_initializer_identity_aliases(model.graph)
        initializer_names = {initializer.name for initializer in model.graph.initializer}
        required_quantized_inputs: set[str] = set()
        missing_weights: set[tuple[str, str]] = set()
        graph_rewrites = 0

        for node in model.graph.node:
            if (
                node.op_type not in rp.op_types
                or
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
                skipped_signature = skipped_recipes.get(recipe_key)
                if skipped_signature is not None:
                    if (
                        source_signatures[weight_name] != skipped_signature
                        and content_signature(source_initializers[weight_name], src_path)
                        != content_signature(template_initializers[weight_name], template_src_path)
                    ):
                        raise RuntimeError(
                            f"shared quantization source weight {node.op_type}:{weight_name} "
                            f"in {Path(src_path).name} does not byte-match the skipped template payload."
                        )
                    continue
                missing_weights.add(recipe_key)
                continue
            quantized_node, name_suffix, quantized_inputs, template_signature = recipe
            if (
                source_signatures[weight_name] != template_signature
                and content_signature(source_initializers[weight_name], src_path)
                != content_signature(template_initializers[weight_name], template_src_path)
            ):
                raise RuntimeError(
                    f"shared quantization source weight {node.op_type}:{weight_name} "
                    f"in {Path(src_path).name} does not byte-match the template payload."
                )
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
            raise RuntimeError(
                f"shared quantization cache has no recipe for selected weight(s) in "
                f"{Path(src_path).name}: {names}."
            )
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
        reserved_values = {value.name for value in model.graph.input}
        reserved_values.update(value.name for value in model.graph.output)
        reserved_values.update(retained_names)
        reserved_values.update(
            output for graph_node in model.graph.node for output in graph_node.output if output
        )
        collisions = sorted(required_quantized_inputs & reserved_values)
        if collisions:
            raise RuntimeError(
                f"shared quantization initializer name collision(s) in "
                f"{Path(src_path).name}: {collisions}."
            )
        missing_cached = sorted(required_quantized_inputs - set(quantized_initializers))
        if missing_cached:
            raise RuntimeError(
                f"shared quantization cache lacks initializer(s): {missing_cached}."
            )
        _stage_external_data_dependencies(model, src_path, dst_path)
        _save_model(model, dst_path, external)
        del model
        model = onnx.load(dst_path, load_external_data=False)
        for initializer in quantized_template.graph.initializer:
            if initializer.name in required_quantized_inputs:
                model.graph.initializer.add().CopyFrom(initializer)
        appended_names = {initializer.name for initializer in model.graph.initializer}
        absent_inputs = sorted(required_quantized_inputs - appended_names)
        if absent_inputs:
            raise RuntimeError(
                f"failed to append shared quantization initializer(s): {absent_inputs}."
            )
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
        del source_header
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


def quantize_dynamic_int8(
    src_path: str,
    dst_path: str,
    rp: ResolvedPlan,
    external: bool,
    *,
    do_surgery: bool = False,
    config: OptimizerConfig | None = None,
) -> None:
    if rp.algo == "AFFINE_REFINE_V2":
        print(
            f"  Quantizing weights ({rp.algo}, dynamic {rp.dynamic_weight_type}, "
            f"per_channel={rp.per_channel}, reduce_range={rp.reduce_range}, symmetric={rp.symmetric})..."
        )
        model = quant_utils.load_model_with_shape_infer(Path(src_path))
        _apply_kv_surgery_if_requested(model, do_surgery, config)
        converted_constants = _materialize_constant_tensors_as_initializers(model.graph)
        if converted_constants:
            print(f"  Materialized {converted_constants} Constant tensor nodes as initializers for weight quantization.")
        exposed_aliases = _eliminate_initializer_identity_aliases(model.graph)
        if exposed_aliases:
            print(f"  Eliminated {exposed_aliases} initializer Identity aliases before dynamic quantization.")
        quantize_affine_v2_dynamic_model(model, _resolve_custom_quant_plan(rp, src_path))
        _save_model(model, dst_path, external)
        del model
        gc.collect()
        return

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
    _apply_kv_surgery_if_requested(model, do_surgery, config)
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


def _source_tensor_content_signature(tensor: TensorProto, model_path: str) -> tuple:
    """Hash one tensor payload when external-reference identity is insufficient."""
    if tensor.data_location == TensorProto.EXTERNAL:
        external_data = {item.key: item.value for item in tensor.external_data}
        location = external_data.get("location")
        if not location:
            raise ValueError(f"External tensor {tensor.name!r} has no data location.")
        length = int(external_data.get("length", "0"))
        if length <= 0:
            raise ValueError(f"External tensor {tensor.name!r} has no positive byte length.")
        digest = hashlib.sha256()
        with (Path(model_path).parent / location).open("rb") as data_file:
            data_file.seek(int(external_data.get("offset", "0")))
            remaining = length
            while remaining:
                chunk = data_file.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise ValueError(f"External tensor {tensor.name!r} ends before its declared length.")
                digest.update(chunk)
                remaining -= len(chunk)
        return tensor.data_type, tuple(tensor.dims), length, digest.digest()

    raw = tensor.raw_data
    if not raw:
        raw = numpy_helper.to_array(tensor).tobytes(order="C")
    return tensor.data_type, tuple(tensor.dims), len(raw), hashlib.sha256(raw).digest()


def quantize_dynamic_int8_shared(
    template_src_path: str,
    model_paths: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    cache_path: str,
    rp: ResolvedPlan,
    external: bool,
) -> dict[str, int]:
    """Quantize one covering graph and replay its dynamic MatMul recipes on peers."""
    if rp.algo == "AFFINE_REFINE_V2":
        raise ValueError(
            "AFFINE_REFINE_V2 dynamic quantization uses DynamicQuantizeMatMul and "
            "cannot replay the legacy QDQ recipe cache; process strategy graphs "
            "independently so the final shared initializer bundle can content-deduplicate them."
        )
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
        target_includes = _resolve_nodes(rp.nodes_to_include, src_path)
        target_includes = None if target_includes is None else set(target_includes)
        target_excludes = set(_resolve_nodes(rp.nodes_to_exclude, src_path) or ())
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

        def is_selected(node):
            return (
                node.op_type in rp.op_types
                and
                (target_includes is None or node.name in target_includes)
                and node.name not in target_excludes
            )

        for node in model.graph.node:
            if (
                node.op_type == "Gather"
                and len(node.input) >= 2
                and len(node.output) == 1
                and node.input[0] in initializer_names
            ):
                if not is_selected(node):
                    rewritten_nodes.append(node)
                    continue
                weight_name = node.input[0]
                recipe = gather_weight_recipes.get(weight_name)
                if recipe is None:
                    raise RuntimeError(
                        "shared quantization cache has no recipe for selected Gather "
                        f"weight {weight_name!r} in {Path(src_path).name}."
                    )
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

            if not is_selected(node):
                rewritten_nodes.append(node)
                continue

            activation_name, weight_name = node.input
            recipe = weight_recipes.get(weight_name)
            if recipe is None:
                raise RuntimeError(
                    "shared quantization cache has no recipe for selected MatMul "
                    f"weight {weight_name!r} in {Path(src_path).name}."
                )
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

    do_surgery = False
    if rp.kv_surgery is not False:
        do_surgery, message = plan_kv_surgery(src_path, config)
        print(f"  KV/rope-shift surgery: {message}")
    if do_surgery and use_fp16:
        print(
            "  Surgery: disabled for float16 conversion; ORT's fp16 converter can otherwise leave the "
            "quantized island with invalid mixed f32/f16 types."
        )
        do_surgery = False

    if prequantized:
        print("  Reusing shared prequantized graph...")
    elif rp.method in _WEIGHT_ONLY_BITS:
        quantize_weight_only(
            src_path,
            dst_path,
            rp,
            _WEIGHT_ONLY_BITS[rp.method],
            external,
            do_surgery=do_surgery,
            config=config,
        )
    elif rp.method == "DYNAMIC":
        quantize_dynamic_int8(src_path, dst_path, rp, external, do_surgery=do_surgery, config=config)
    else:
        resave(src_path, dst_path, external, do_surgery=do_surgery, config=config)

    if rp.optimize or use_fp16:
        print("  Optimizing (onnxslim -> transformers optimizer -> onnxslim)...")
        run_onnxslim(dst_path, external, config, no_shape_infer=rp.first_slim_no_shape_infer)
        if rp.transformer or use_fp16:
            optimize_onnx_model(dst_path, rp, config, src_path, use_fp16, external, keep_io_types)
            second_no_shape = not config.shape_infer if rp.second_slim_no_shape_infer is None else rp.second_slim_no_shape_infer
            run_onnxslim(dst_path, external, config, no_shape_infer=second_no_shape)

    if config.upgrade_opset > 0:
        upgrade_opset_version(dst_path, config.upgrade_opset, external)

    removed_stale_attributes = strip_stale_matmul_nbits_attributes(dst_path)
    if removed_stale_attributes:
        print(
            f"  Removed {removed_stale_attributes} stale MatMulNBits weight_prepacked attribute(s)."
        )

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
    for name, rp in resolved.items():
        validate_plan(name, rp)
        if rp.algo in {"AFFINE_REFINE_V2", "k_quant"}:
            _validate_affine_v2_settings(rp.affine_v2_settings)

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
