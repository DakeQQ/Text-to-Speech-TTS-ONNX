"""Pack KaniTTS ONNX initializers into one exact-deduplicated data blob."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SHARED_MODEL_NAME = "KaniTTS_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
MIN_SHARED_INITIALIZER_ELEMENTS = 1024
_CANONICAL_PREFIX = "kanitts_shared_"
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
    offset: int,
    length: int,
    *,
    name: str,
) -> TensorProto:
    ref = TensorProto()
    ref.name = name
    ref.data_type = tensor.data_type
    ref.dims.extend(tensor.dims)
    ref.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", SHARED_DATA_NAME),
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


def _resolve_bundle_targets(folder: Path, model_paths) -> list[Path]:
    if not model_paths:
        raise ValueError("KaniTTS bundling requires an explicit non-empty model_paths list.")
    targets = [Path(path).expanduser().resolve() for path in model_paths]
    names = [path.name for path in targets]
    if len(set(names)) != len(names):
        raise ValueError("Bundled ONNX graph file names must be unique.")
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ONNX graph(s): {missing}")
    if SHARED_MODEL_NAME in names:
        raise ValueError("The shared-initializer carrier cannot be one of its own bundle targets.")
    folder.mkdir(parents=True, exist_ok=True)
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
        digest = hashlib.sha256(raw).digest()
        identity = (tensor.data_type, tuple(int(dim) for dim in tensor.dims), len(raw), digest)
        source_bytes += len(raw)
        shared = unique.get(identity)
        if shared is None:
            offset = data_file.tell()
            data_file.write(raw)
            canonical_name = f"{_CANONICAL_PREFIX}{len(unique):06d}"
            shared = (offset, len(raw), canonical_name)
            unique[identity] = shared
            carrier_initializers.append(
                _external_ref(tensor, offset, len(raw), name=canonical_name)
            )
            unique_bytes += len(raw)

        offset, length, canonical_name = shared
        remap[tensor.name] = canonical_name
        if canonical_name not in graph_shared_names:
            rewritten.append(_external_ref(tensor, offset, length, name=canonical_name))
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
    marker_input = helper.make_tensor_value_info("shared_marker", TensorProto.INT64, [1])
    marker_output = helper.make_tensor_value_info("shared_marker_out", TensorProto.INT64, [1])
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["shared_marker"], ["shared_marker_out"])],
        name="kani_tts_shared_initializers",
        inputs=[marker_input],
        outputs=[marker_output],
        initializer=carrier_initializers,
    )
    carrier = helper.make_model(
        graph,
        producer_name=Path(__file__).name,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    carrier.ir_version = 10
    _set_metadata(carrier, metadata)
    return carrier


def _validate_external_ranges(model_path: Path, data_path: Path) -> None:
    model = onnx.load(str(model_path), load_external_data=False)
    data_size = data_path.stat().st_size
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        external = _external_data_map(tensor)
        if external.get("location") != SHARED_DATA_NAME:
            raise RuntimeError(
                f"{model_path.name}:{tensor.name} references {external.get('location')!r}, "
                f"expected {SHARED_DATA_NAME!r}."
            )
        offset = int(external.get("offset", "0"))
        length = int(external.get("length", "0"))
        if offset < 0 or length <= 0 or offset + length > data_size:
            raise RuntimeError(
                f"{model_path.name}:{tensor.name} has invalid shared range "
                f"offset={offset}, length={length}, blob_size={data_size}."
            )


def bundle_shared_initializers(
    folder: str | Path,
    model_paths,
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    metadata: dict[str, str] | None = None,
) -> dict[str, int | str]:
    """Rewrite an explicit graph set to one exact-deduplicated external blob."""
    folder = Path(folder).expanduser().resolve()
    if min_elements <= 0:
        raise ValueError(f"min_elements must be positive, got {min_elements}.")
    targets = _resolve_bundle_targets(folder, model_paths)

    shared_metadata = {
        "kani_tts_shared_initializers": "1",
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

    with tempfile.TemporaryDirectory(dir=folder, prefix=".kanitts_shared_") as temp_name:
        temp_dir = Path(temp_name)
        staged_data = temp_dir / SHARED_DATA_NAME
        with staged_data.open("wb") as data_file:
            for source in targets:
                model = onnx.load(str(source), load_external_data=True)
                references, model_source_bytes, model_unique_bytes = _rewrite_model_initializers(
                    model,
                    data_file,
                    unique,
                    carrier_initializers,
                    min_elements,
                )
                graph_reference_count += references
                source_bytes += model_source_bytes
                unique_bytes += model_unique_bytes
                _set_metadata(model, shared_metadata)
                staged_model = temp_dir / source.name
                onnx.save(model, str(staged_model))
                staged_models.append((staged_model, folder / source.name))

        shared_metadata.update({
            "shared_initializer_count": str(len(carrier_initializers)),
            "shared_initializer_reference_count": str(graph_reference_count),
            "shared_initializer_source_bytes": str(source_bytes),
            "shared_initializer_unique_bytes": str(unique_bytes),
        })
        staged_carrier = temp_dir / SHARED_MODEL_NAME
        onnx.save(_make_shared_carrier(carrier_initializers, shared_metadata), str(staged_carrier))

        for staged_model, _ in staged_models:
            onnx.checker.check_model(str(staged_model), full_check=False)
            _validate_external_ranges(staged_model, staged_data)
        onnx.checker.check_model(str(staged_carrier), full_check=False)
        _validate_external_ranges(staged_carrier, staged_data)

        os.replace(staged_data, folder / SHARED_DATA_NAME)
        for staged_model, destination in staged_models:
            os.replace(staged_model, destination)
            destination.with_name(destination.name + ".data").unlink(missing_ok=True)
        os.replace(staged_carrier, folder / SHARED_MODEL_NAME)

    return {
        "graph_count": len(targets),
        "initializer_references": graph_reference_count,
        "unique_initializers": len(unique),
        "source_bytes": source_bytes,
        "unique_bytes": unique_bytes,
        "deduplicated_bytes": source_bytes - unique_bytes,
        "shared_model": str(folder / SHARED_MODEL_NAME),
        "shared_data": str(folder / SHARED_DATA_NAME),
    }


def audit_shared_bundle(folder: str | Path, model_paths) -> dict[str, int]:
    """Validate that every large initializer in the explicit graph set uses one blob."""
    folder = Path(folder).expanduser().resolve()
    targets = _resolve_bundle_targets(folder, model_paths)
    data_path = folder / SHARED_DATA_NAME
    carrier_path = folder / SHARED_MODEL_NAME
    if not data_path.is_file() or not carrier_path.is_file():
        raise FileNotFoundError("KaniTTS shared initializer carrier or data blob is missing.")

    external_references = 0
    for path in [*targets, carrier_path]:
        onnx.checker.check_model(str(path), full_check=False)
        _validate_external_ranges(path, data_path)
        model = onnx.load(str(path), load_external_data=False)
        for tensor in model.graph.initializer:
            if tensor.data_location == TensorProto.EXTERNAL:
                external_references += 1
                if not tensor.name.startswith(_CANONICAL_PREFIX):
                    raise RuntimeError(f"Non-canonical shared initializer name: {tensor.name!r}")
            elif _num_elements(tensor) >= MIN_SHARED_INITIALIZER_ELEMENTS:
                raise RuntimeError(f"Large initializer {path.name}:{tensor.name} was not shared.")
    return {
        "graph_count": len(targets),
        "external_references": external_references,
        "external_bytes": data_path.stat().st_size,
    }


def attach_shared_initializers(session_options, shared_model_path: str | Path):
    """Mmap the shared blob and register OrtValues before graph sessions are built.

    The returned arrays and OrtValues must remain alive for every session that uses
    ``session_options``.
    """
    import onnxruntime

    shared_model_path = Path(shared_model_path).expanduser().resolve()
    shared_model = onnx.load(str(shared_model_path), load_external_data=False)
    arrays: dict[str, np.ndarray] = {}
    ort_values = []
    for tensor in shared_model.graph.initializer:
        if tensor.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(tensor)
        location = external.get("location")
        if location != SHARED_DATA_NAME:
            raise RuntimeError(
                f"Shared initializer {tensor.name!r} references {location!r}, "
                f"expected {SHARED_DATA_NAME!r}."
            )
        data_path = shared_model_path.parent / location
        offset = int(external.get("offset", "0"))
        dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
        shape = tuple(int(dim) for dim in tensor.dims)
        array = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        arrays[tensor.name] = array
        ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(tensor.name, ort_value)
    return arrays, ort_values