"""Merge compact MOSS graphs and pack their initializers into one external-data blob."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SHARED_MODEL_NAME = "MossTTSNano_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
SHARED_INITIALIZER_PREFIX = "mossttsnano_shared_"
MIN_SHARED_INITIALIZER_ELEMENTS = 1024
DEFAULT_DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)
_TensorIdentity = tuple[int, tuple[int, ...], int, bytes]
_SharedTensor = tuple[int, int, str]


def _num_elements(tensor: TensorProto) -> int:
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    return count


def _tensor_bytes(tensor: TensorProto) -> bytes:
    if tensor.raw_data:
        return tensor.raw_data
    return numpy_helper.to_array(tensor).tobytes(order="C")


def _external_ref(
    tensor: TensorProto,
    location: str,
    offset: int,
    length: int,
    name: str | None = None,
) -> TensorProto:
    ref = TensorProto()
    ref.name = tensor.name if name is None else name
    ref.data_type = tensor.data_type
    ref.dims.extend(tensor.dims)
    ref.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = ref.external_data.add()
        entry.key = key
        entry.value = value
    return ref


def _external_data_map(tensor: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _inline_tensor(tensor: TensorProto) -> TensorProto:
    if tensor.data_location != TensorProto.EXTERNAL:
        return tensor
    inline = TensorProto()
    inline.name = tensor.name
    inline.data_type = tensor.data_type
    inline.dims.extend(tensor.dims)
    inline.raw_data = _tensor_bytes(tensor)
    return inline


def _set_metadata(model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        text = str(value)
        if key in existing:
            existing[key].value = text
        else:
            existing[key] = model.metadata_props.add(key=key, value=text)


def _remap_node_inputs(graph: onnx.GraphProto, remap: dict[str, str]) -> None:
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
        for attribute in node.attribute:
            if attribute.HasField("g"):
                _remap_node_inputs(attribute.g, remap)
            for subgraph in attribute.graphs:
                _remap_node_inputs(subgraph, remap)


def _resolve_bundle_targets(
    folder: Path,
    model_paths: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    if model_paths is None:
        targets = sorted(path for path in folder.glob("*.onnx") if path.name != SHARED_MODEL_NAME)
    else:
        targets = sorted(Path(path).resolve() for path in model_paths)
    if not targets:
        raise RuntimeError(f"No ONNX graphs found to bundle in {folder}.")
    names = [path.name for path in targets]
    if len(set(names)) != len(names):
        raise ValueError("Bundled ONNX graph file names must be unique.")
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ONNX graph(s): {missing}")
    return targets


def _rewrite_model_initializers(
    model: onnx.ModelProto,
    data_file: BinaryIO,
    unique: dict[_TensorIdentity, _SharedTensor],
    carrier_initializers: list[TensorProto],
    min_elements: int,
) -> tuple[int, int, int]:
    rewritten: list[TensorProto] = []
    remap: dict[str, str] = {}
    graph_shared_names: set[str] = set()
    reference_count = 0
    source_bytes = 0
    unique_bytes = 0

    for tensor in model.graph.initializer:
        if _num_elements(tensor) < min_elements:
            rewritten.append(_inline_tensor(tensor))
            continue

        raw = _tensor_bytes(tensor)
        identity = (
            tensor.data_type,
            tuple(int(dim) for dim in tensor.dims),
            len(raw),
            hashlib.sha256(raw).digest(),
        )
        source_bytes += len(raw)
        shared = unique.get(identity)
        if shared is None:
            offset = data_file.tell()
            data_file.write(raw)
            canonical_name = f"{SHARED_INITIALIZER_PREFIX}{len(unique):06d}"
            shared = (offset, len(raw), canonical_name)
            unique[identity] = shared
            carrier_initializers.append(
                _external_ref(tensor, SHARED_DATA_NAME, offset, len(raw), name=canonical_name)
            )
            unique_bytes += len(raw)

        offset, length, canonical_name = shared
        remap[tensor.name] = canonical_name
        if canonical_name not in graph_shared_names:
            rewritten.append(
                _external_ref(tensor, SHARED_DATA_NAME, offset, length, name=canonical_name)
            )
            graph_shared_names.add(canonical_name)
        reference_count += 1

    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    _remap_node_inputs(model.graph, remap)
    return reference_count, source_bytes, unique_bytes


def _make_shared_carrier(
    carrier_initializers: list[TensorProto],
    metadata: dict[str, str],
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=[],
        name="moss_tts_nano_shared_initializers",
        inputs=[],
        outputs=[],
        initializer=carrier_initializers,
    )
    carrier = helper.make_model(graph, producer_name=Path(__file__).name)
    carrier.ir_version = 10
    _set_metadata(carrier, metadata)
    return carrier


def bundle_shared_initializers(
    folder: str | Path,
    model_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    metadata: dict[str, str] | None = None,
    transient_sidecars: Iterable[str] = (),
) -> dict[str, int | str]:
    """Exact-deduplicate large initializers and atomically replace checked graph files."""
    folder = Path(folder).resolve()
    if min_elements <= 0:
        raise ValueError(f"min_elements must be positive, got {min_elements}.")
    targets = _resolve_bundle_targets(folder, model_paths)

    shared_metadata = {
        "moss_tts_nano_shared_initializers": "1",
        "shared_initializer_model_file": SHARED_MODEL_NAME,
        "shared_initializer_data_file": SHARED_DATA_NAME,
    }
    if metadata:
        shared_metadata.update({str(key): str(value) for key, value in metadata.items()})

    unique: dict[_TensorIdentity, _SharedTensor] = {}
    carrier_initializers: list[TensorProto] = []
    staged_models: list[tuple[Path, Path]] = []
    graph_reference_count = 0
    source_bytes = 0
    unique_bytes = 0

    with tempfile.TemporaryDirectory(dir=folder, prefix=".mossttsnano_shared_") as temp_name:
        temp_dir = Path(temp_name)
        staged_data = temp_dir / SHARED_DATA_NAME
        with staged_data.open("wb") as data_file:
            for target in targets:
                model = onnx.load(str(target), load_external_data=True)
                reference_count, model_source_bytes, model_unique_bytes = _rewrite_model_initializers(
                    model,
                    data_file,
                    unique,
                    carrier_initializers,
                    min_elements,
                )
                graph_reference_count += reference_count
                source_bytes += model_source_bytes
                unique_bytes += model_unique_bytes
                _set_metadata(model, shared_metadata)
                staged_model = temp_dir / target.name
                onnx.save(model, str(staged_model))
                staged_models.append((staged_model, folder / target.name))

        shared_metadata.update({
            "shared_initializer_count": str(len(carrier_initializers)),
            "shared_initializer_reference_count": str(graph_reference_count),
            "shared_initializer_source_bytes": str(source_bytes),
            "shared_initializer_unique_bytes": str(unique_bytes),
        })
        carrier = _make_shared_carrier(carrier_initializers, shared_metadata)
        staged_carrier = temp_dir / SHARED_MODEL_NAME
        onnx.save(carrier, str(staged_carrier))

        for staged_model, _ in staged_models:
            onnx.checker.check_model(str(staged_model), full_check=False)
        onnx.checker.check_model(str(staged_carrier), full_check=False)

        os.replace(staged_data, folder / SHARED_DATA_NAME)
        for staged_model, target in staged_models:
            os.replace(staged_model, target)
            target.with_name(target.name + ".data").unlink(missing_ok=True)
        os.replace(staged_carrier, folder / SHARED_MODEL_NAME)

    audit = audit_shared_bundle(
        folder,
        model_paths=[folder / target.name for target in targets],
        allowed_sidecars=transient_sidecars,
    )
    return {
        "graph_count": len(targets),
        "initializer_references": graph_reference_count,
        "unique_initializers": len(unique),
        "source_bytes": source_bytes,
        "unique_bytes": unique_bytes,
        "deduplicated_bytes": source_bytes - unique_bytes,
        "shared_model": str(folder / SHARED_MODEL_NAME),
        "shared_data": str(folder / SHARED_DATA_NAME),
        "audited_references": audit["initializer_references"],
    }


def _prefixed_component(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    """Prefix graph-local names while preserving public I/O names."""
    model = copy.deepcopy(model)
    public_names = {value.name for value in model.graph.input}
    public_names.update(value.name for value in model.graph.output)
    remap: dict[str, str] = {}

    for tensor in model.graph.initializer:
        remap[tensor.name] = prefix + tensor.name
        tensor.name = remap[tensor.name]
    for node in model.graph.node:
        if node.name:
            node.name = prefix + node.name
        for output in node.output:
            if output and output not in public_names:
                remap[output] = prefix + output
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = remap.get(name, name)
    for value in model.graph.value_info:
        value.name = remap.get(value.name, value.name)
    return model


def _copy_metadata(dst: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    metadata: dict[str, str] = {}
    for source in sources:
        metadata.update({item.key: item.value for item in source.metadata_props})
    _set_metadata(dst, metadata)


def _merge_opset_imports(dst: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    opsets: dict[str, int] = {}
    for model in sources:
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    for domain, version in sorted(opsets.items()):
        dst.opset_import.add(domain=domain, version=version)


def _merge_initializers(dst: onnx.ModelProto, *sources: onnx.ModelProto) -> set[str]:
    initializers: dict[str, TensorProto] = {}
    for model in sources:
        for tensor in model.graph.initializer:
            existing = initializers.get(tensor.name)
            if existing is None:
                initializers[tensor.name] = tensor
            elif existing.SerializeToString() != tensor.SerializeToString():
                raise RuntimeError(f"Conflicting initializer during DecodeStep merge: {tensor.name}")
    dst.graph.initializer.extend(initializers.values())
    return set(initializers)


def _save_checked_model(model: onnx.ModelProto, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        onnx.save(model, str(temp_path))
        onnx.checker.check_model(str(temp_path), full_check=False)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _delete_model_artifacts(*model_paths: Path) -> None:
    for model_path in model_paths:
        model_path.unlink()
        model_path.with_name(model_path.name + ".data").unlink(missing_ok=True)


def merge_decode_step_graph(
    predictor_path: str | Path,
    main_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Merge PredictorFrame.frame_codec_ids into MainDecodeDecision.frame_codec_ids."""
    predictor_path = Path(predictor_path)
    main_path = Path(main_path)
    output_path = Path(output_path)
    predictor = _prefixed_component(
        onnx.load(str(predictor_path), load_external_data=True),
        "predictor_frame/",
    )
    main = _prefixed_component(
        onnx.load(str(main_path), load_external_data=True),
        "main_decode/",
    )

    predictor_outputs = {value.name: value for value in predictor.graph.output}
    main_inputs = {value.name: value for value in main.graph.input}
    if "frame_codec_ids" not in predictor_outputs or "frame_codec_ids" not in main_inputs:
        raise RuntimeError("DecodeStep merge requires PredictorFrame -> MainDecodeDecision frame_codec_ids.")

    merged = onnx.ModelProto()
    merged.ir_version = max(predictor.ir_version, main.ir_version)
    merged.producer_name = Path(__file__).name
    merged.graph.name = f"{predictor.graph.name}_{main.graph.name}_decode_step"
    _merge_opset_imports(merged, predictor, main)

    seen_inputs: set[str] = set()
    ordered_inputs = list(main.graph.input) + [
        value for value in predictor.graph.input if value.name != "frame_codec_ids"
    ]
    for value in ordered_inputs:
        if value.name == "frame_codec_ids" or value.name in seen_inputs:
            continue
        merged.graph.input.append(value)
        seen_inputs.add(value.name)

    initializer_names = _merge_initializers(merged, main, predictor)
    merged.graph.node.extend(predictor.graph.node)
    merged.graph.node.extend(main.graph.node)

    seen_values = seen_inputs | initializer_names
    for value in list(predictor.graph.value_info) + list(main.graph.value_info):
        if value.name not in seen_values:
            merged.graph.value_info.append(value)
            seen_values.add(value.name)

    desired_outputs = list(main.graph.output)
    desired_outputs.extend(predictor_outputs[name] for name in ("generated_codec", "frame_codec_ids"))
    seen_outputs: set[str] = set()
    for value in desired_outputs:
        if value.name not in seen_outputs:
            merged.graph.output.append(value)
            seen_outputs.add(value.name)

    _copy_metadata(merged, predictor, main)
    _save_checked_model(merged, output_path)
    return output_path


def build_decode_step_graphs(
    folder: str | Path,
    strategies: Iterable[str] = DEFAULT_DECODE_STRATEGIES,
    *,
    delete_components: bool = True,
) -> dict[str, Path]:
    folder = Path(folder)
    outputs: dict[str, Path] = {}
    for strategy in strategies:
        predictor_path = folder / f"MossTTSNano_PredictorFrame_{strategy}.onnx"
        main_path = folder / f"MossTTSNano_MainDecodeDecision_{strategy}.onnx"
        output_path = folder / f"MossTTSNano_DecodeStep_{strategy}.onnx"
        outputs[strategy] = merge_decode_step_graph(predictor_path, main_path, output_path)
        if delete_components:
            _delete_model_artifacts(predictor_path, main_path)
    return outputs


def _collect_node_inputs(graph: onnx.GraphProto, names: set[str]) -> None:
    for node in graph.node:
        names.update(name for name in node.input if name)
        for attribute in node.attribute:
            if attribute.HasField("g"):
                _collect_node_inputs(attribute.g, names)
            for subgraph in attribute.graphs:
                _collect_node_inputs(subgraph, names)


def audit_shared_bundle(
    folder: str | Path,
    model_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    *,
    allowed_sidecars: Iterable[str] = (),
) -> dict[str, int]:
    """Validate external offsets, canonical references, sidecars, metadata, and orphan tensors."""
    folder = Path(folder).resolve()
    carrier_path = folder / SHARED_MODEL_NAME
    data_path = folder / SHARED_DATA_NAME
    if not carrier_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"Missing shared initializer artifacts in {folder}.")
    carrier = onnx.load(str(carrier_path), load_external_data=False)
    data_size = data_path.stat().st_size
    carrier_refs: dict[str, tuple[int, int, int, tuple[int, ...]]] = {}
    ranges = []
    for tensor in carrier.graph.initializer:
        external = _external_data_map(tensor)
        if external.get("location") != SHARED_DATA_NAME:
            raise RuntimeError(f"Carrier tensor {tensor.name!r} has an unexpected data location.")
        offset = int(external.get("offset", "-1"))
        length = int(external.get("length", "-1"))
        if offset < 0 or length < 0 or offset + length > data_size:
            raise RuntimeError(f"Carrier tensor {tensor.name!r} has an out-of-bounds external range.")
        carrier_refs[tensor.name] = (offset, length, tensor.data_type, tuple(tensor.dims))
        ranges.append((offset, offset + length, tensor.name))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if previous[1] > current[0]:
            raise RuntimeError(f"Shared external ranges overlap: {previous[2]!r} and {current[2]!r}.")

    targets = _resolve_bundle_targets(folder, model_paths)
    referenced: set[str] = set()
    initializer_references = 0
    for target in targets:
        model = onnx.load(str(target), load_external_data=False)
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        external_tensors = [
            tensor
            for tensor in model.graph.initializer
            if tensor.data_location == TensorProto.EXTERNAL
        ]
        if (
            external_tensors
            and metadata.get("moss_tts_nano_shared_initializers") != "1"
        ):
            raise RuntimeError(f"{target.name} is missing shared-initializer metadata.")
        node_inputs: set[str] = set()
        _collect_node_inputs(model.graph, node_inputs)
        graph_outputs = {value.name for value in model.graph.output}
        for tensor in external_tensors:
            if not tensor.name.startswith(SHARED_INITIALIZER_PREFIX):
                raise RuntimeError(f"{target.name} has a noncanonical external tensor {tensor.name!r}.")
            expected = carrier_refs.get(tensor.name)
            external = _external_data_map(tensor)
            actual = (
                int(external.get("offset", "-1")),
                int(external.get("length", "-1")),
                tensor.data_type,
                tuple(tensor.dims),
            )
            if external.get("location") != SHARED_DATA_NAME or actual != expected:
                raise RuntimeError(f"{target.name} has an invalid external reference for {tensor.name!r}.")
            if tensor.name not in node_inputs and tensor.name not in graph_outputs:
                raise RuntimeError(f"{target.name} contains orphan initializer {tensor.name!r}.")
            referenced.add(tensor.name)
            initializer_references += 1
    orphan_carrier = sorted(set(carrier_refs) - referenced)
    if orphan_carrier:
        raise RuntimeError(f"Shared carrier contains orphan tensors: {orphan_carrier[:8]}")
    allowed_sidecars = set(allowed_sidecars)
    sidecars = sorted(
        path.name
        for path in folder.glob("*.onnx.data")
        if path.name != SHARED_DATA_NAME and path.name not in allowed_sidecars
    )
    if sidecars:
        raise RuntimeError(f"Unexpected per-graph external-data sidecars: {sidecars}")
    return {
        "graph_count": len(targets),
        "initializer_references": initializer_references,
        "unique_initializers": len(carrier_refs),
        "external_bytes": data_size,
    }


def attach_shared_initializers(session_options, shared_model_path: str | Path):
    """Mmap canonical tensors and inject OrtValues; the returned objects must remain alive."""
    import onnxruntime

    shared_model_path = Path(shared_model_path).resolve()
    shared_model = onnx.load(str(shared_model_path), load_external_data=False)
    arrays: dict[str, np.ndarray] = {}
    ort_values = []
    for tensor in shared_model.graph.initializer:
        if tensor.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(tensor)
        location = external.get("location")
        if not location:
            raise RuntimeError(f"Shared initializer {tensor.name!r} has no external-data location.")
        data_path = shared_model_path.parent / location
        offset = int(external.get("offset", "0"))
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
        shape = tuple(int(dim) for dim in tensor.dims)
        array = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        arrays[tensor.name] = array
        ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(tensor.name, ort_value)
    return arrays, ort_values