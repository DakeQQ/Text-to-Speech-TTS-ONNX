"""Pack Inflect ONNX initializers into one exact-deduplicated mmap blob."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SHARED_MODEL_NAME = "Inflect_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
MIN_SHARED_INITIALIZER_ELEMENTS = 256
_CANONICAL_PREFIX = "inflect_shared_"


def _num_elements(tensor: TensorProto) -> int:
    count = 1
    for dimension in tensor.dims:
        count *= int(dimension)
    return count


def _tensor_bytes(tensor: TensorProto) -> bytes:
    if tensor.raw_data:
        return tensor.raw_data
    return numpy_helper.to_array(tensor).tobytes(order="C")


def _external_reference(
    tensor: TensorProto,
    offset: int,
    length: int,
    name: str,
) -> TensorProto:
    reference = TensorProto()
    reference.name = name
    reference.data_type = tensor.data_type
    reference.dims.extend(tensor.dims)
    reference.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", SHARED_DATA_NAME),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = reference.external_data.add()
        entry.key = key
        entry.value = value
    return reference


def _external_data(tensor: TensorProto) -> dict[str, str]:
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
    existing = {property_.key: property_ for property_ in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = str(value)
        else:
            model.metadata_props.add(key=str(key), value=str(value))


def _remap_inputs(graph: onnx.GraphProto, names: dict[str, str]) -> None:
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = names.get(name, name)
        for attribute in node.attribute:
            if attribute.HasField("g"):
                _remap_inputs(attribute.g, names)
            for subgraph in attribute.graphs:
                _remap_inputs(subgraph, names)


def _rewrite_initializers(
    model: onnx.ModelProto,
    data_file: BinaryIO,
    unique: dict[tuple[int, tuple[int, ...], int, bytes], tuple[int, int, str]],
    carrier_initializers: list[TensorProto],
) -> tuple[int, int, int]:
    rewritten: list[TensorProto] = []
    remap: dict[str, str] = {}
    graph_names: set[str] = set()
    references = 0
    source_bytes = 0
    unique_bytes = 0

    for tensor in model.graph.initializer:
        if _num_elements(tensor) < MIN_SHARED_INITIALIZER_ELEMENTS:
            rewritten.append(_inline_tensor(tensor))
            continue
        raw = _tensor_bytes(tensor)
        identity = (
            int(tensor.data_type),
            tuple(int(dimension) for dimension in tensor.dims),
            len(raw),
            hashlib.sha256(raw).digest(),
        )
        source_bytes += len(raw)
        shared = unique.get(identity)
        if shared is None:
            offset = data_file.tell()
            data_file.write(raw)
            canonical_name = f"{_CANONICAL_PREFIX}{len(unique):06d}"
            shared = (offset, len(raw), canonical_name)
            unique[identity] = shared
            carrier_initializers.append(
                _external_reference(tensor, offset, len(raw), canonical_name)
            )
            unique_bytes += len(raw)
        offset, length, canonical_name = shared
        remap[tensor.name] = canonical_name
        if canonical_name not in graph_names:
            rewritten.append(
                _external_reference(tensor, offset, length, canonical_name)
            )
            graph_names.add(canonical_name)
        references += 1

    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    _remap_inputs(model.graph, remap)
    return references, source_bytes, unique_bytes


def _carrier(initializers: list[TensorProto], metadata: dict[str, str]) -> onnx.ModelProto:
    marker = helper.make_tensor_value_info("marker", TensorProto.INT32, [1])
    marker_out = helper.make_tensor_value_info("marker_out", TensorProto.INT32, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["marker"], ["marker_out"])],
        "inflect_shared_initializers",
        [marker],
        [marker_out],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name=Path(__file__).name,
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 10
    _set_metadata(model, metadata)
    return model


def bundle_shared_initializers(
    folder: str | Path,
    model_paths: list[str | Path],
    metadata: dict[str, str],
) -> dict[str, int | str]:
    """Atomically rewrite graph weights to one exact-deduplicated external blob."""
    folder = Path(folder).expanduser().resolve()
    targets = [Path(path).expanduser().resolve() for path in model_paths]
    shared_metadata = {
        **{str(key): str(value) for key, value in metadata.items()},
        "shared_initializer_model_file": SHARED_MODEL_NAME,
        "shared_initializer_data_file": SHARED_DATA_NAME,
    }
    unique: dict[tuple[int, tuple[int, ...], int, bytes], tuple[int, int, str]] = {}
    carrier_initializers: list[TensorProto] = []
    staged_models: list[tuple[Path, Path]] = []
    reference_count = 0
    source_bytes = 0
    unique_bytes = 0

    with tempfile.TemporaryDirectory(dir=folder, prefix=".inflect_shared_") as temp:
        temporary = Path(temp)
        staged_data = temporary / SHARED_DATA_NAME
        with staged_data.open("wb") as data_file:
            for source in targets:
                model = onnx.load(str(source), load_external_data=True)
                references, source_size, new_size = _rewrite_initializers(
                    model,
                    data_file,
                    unique,
                    carrier_initializers,
                )
                reference_count += references
                source_bytes += source_size
                unique_bytes += new_size
                _set_metadata(model, shared_metadata)
                staged = temporary / source.name
                onnx.save(model, str(staged))
                staged_models.append((staged, folder / source.name))

        shared_metadata.update(
            {
                "shared_initializer_count": str(len(unique)),
                "shared_initializer_reference_count": str(reference_count),
                "shared_initializer_source_bytes": str(source_bytes),
                "shared_initializer_unique_bytes": str(unique_bytes),
            }
        )
        staged_carrier = temporary / SHARED_MODEL_NAME
        onnx.save(_carrier(carrier_initializers, shared_metadata), str(staged_carrier))
        os.replace(staged_data, folder / SHARED_DATA_NAME)
        for staged, destination in staged_models:
            os.replace(staged, destination)
            destination.with_name(destination.name + ".data").unlink(missing_ok=True)
        os.replace(staged_carrier, folder / SHARED_MODEL_NAME)

    return {
        "initializer_references": reference_count,
        "unique_initializers": len(unique),
        "source_bytes": source_bytes,
        "unique_bytes": unique_bytes,
        "deduplicated_bytes": source_bytes - unique_bytes,
    }


def attach_shared_initializers(
    session_options,
    shared_model_path: str | Path,
) -> tuple[dict[str, np.ndarray], list]:
    """Register mmap-backed initializer OrtValues before creating graph sessions."""
    import onnxruntime

    shared_model_path = Path(shared_model_path).expanduser().resolve()
    model = onnx.load(str(shared_model_path), load_external_data=False)
    arrays: dict[str, np.ndarray] = {}
    ort_values = []
    for tensor in model.graph.initializer:
        external = _external_data(tensor)
        data_path = shared_model_path.parent / external["location"]
        array = np.memmap(
            data_path,
            dtype=helper.tensor_dtype_to_np_dtype(tensor.data_type),
            mode="r",
            offset=int(external["offset"]),
            shape=tuple(int(dimension) for dimension in tensor.dims),
        )
        arrays[tensor.name] = array
        ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(tensor.name, ort_value)
    return arrays, ort_values