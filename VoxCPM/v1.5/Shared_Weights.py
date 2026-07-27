"""Compose compact VoxCPM graphs and pack their shared ONNX initializers."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SHARED_MODEL_NAME = "VoxCPM_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
SHARED_PREFIX = "voxcpm_shared_"
MIN_SHARED_INITIALIZER_ELEMENTS = 1024
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)
_TensorIdentity = tuple[int, tuple[int, ...], int, bytes]
_SharedTensor = tuple[int, int, str]


@dataclass(frozen=True)
class GraphComponent:
    """One graph participating in a composition.

    ``connections`` maps this component's public input names to tensor names
    produced by an earlier component. Connected inputs are removed from the
    composed graph interface.
    """

    path: Path
    prefix: str
    connections: Mapping[str, str]
    input_renames: Mapping[str, str] = field(default_factory=dict)


def _num_elements(tensor: TensorProto) -> int:
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    return count


def _external_data_map(tensor: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _tensor_bytes(tensor: TensorProto) -> bytes:
    if tensor.raw_data:
        return tensor.raw_data
    return numpy_helper.to_array(tensor).tobytes(order="C")


def _inline_tensor(tensor: TensorProto) -> TensorProto:
    if tensor.data_location != TensorProto.EXTERNAL:
        return tensor
    inline = TensorProto()
    inline.name = tensor.name
    inline.data_type = tensor.data_type
    inline.dims.extend(tensor.dims)
    inline.raw_data = _tensor_bytes(tensor)
    return inline


def _external_ref(
    tensor: TensorProto,
    location: str,
    offset: int,
    length: int,
    *,
    name: str,
) -> TensorProto:
    reference = TensorProto()
    reference.name = name
    reference.data_type = tensor.data_type
    reference.dims.extend(tensor.dims)
    reference.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = reference.external_data.add()
        entry.key = key
        entry.value = value
    return reference


def _set_metadata(model: onnx.ModelProto, metadata: Mapping[str, str]) -> None:
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        text = str(value)
        if key in existing:
            existing[key].value = text
        else:
            existing[key] = model.metadata_props.add(key=key, value=text)


def _copy_metadata(destination: onnx.ModelProto, sources: Iterable[onnx.ModelProto]) -> None:
    merged: dict[str, str] = {}
    for source in sources:
        for item in source.metadata_props:
            previous = merged.get(item.key)
            if previous is not None and previous != item.value:
                raise RuntimeError(
                    f"Conflicting metadata value for {item.key!r}: {previous!r} != {item.value!r}"
                )
            merged[item.key] = item.value
    _set_metadata(destination, merged)


def _remap_subgraph_inputs(graph: onnx.GraphProto, remap: Mapping[str, str]) -> None:
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
        for attribute in node.attribute:
            if attribute.HasField("g"):
                _remap_subgraph_inputs(attribute.g, remap)
            for subgraph in attribute.graphs:
                _remap_subgraph_inputs(subgraph, remap)


def _prefixed_component(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    """Prefix graph-local names while preserving public I/O and shared names."""
    model = copy.deepcopy(model)
    public_names = {value.name for value in model.graph.input}
    public_names.update(value.name for value in model.graph.output)
    remap: dict[str, str] = {}

    for tensor in model.graph.initializer:
        if not tensor.name.startswith(SHARED_PREFIX):
            remap[tensor.name] = prefix + tensor.name
            tensor.name = remap[tensor.name]
    for sparse in model.graph.sparse_initializer:
        if not sparse.values.name.startswith(SHARED_PREFIX):
            remap[sparse.values.name] = prefix + sparse.values.name
            sparse.values.name = remap[sparse.values.name]
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
        for attribute in node.attribute:
            if attribute.HasField("g"):
                _remap_subgraph_inputs(attribute.g, remap)
            for subgraph in attribute.graphs:
                _remap_subgraph_inputs(subgraph, remap)
    for value in model.graph.value_info:
        value.name = remap.get(value.name, value.name)
    return model


def _merge_opsets(destination: onnx.ModelProto, sources: Iterable[onnx.ModelProto]) -> None:
    opsets: dict[str, int] = {}
    for source in sources:
        for opset in source.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), int(opset.version))
    for domain, version in sorted(opsets.items()):
        destination.opset_import.add(domain=domain, version=version)


def _merge_initializers(
    destination: onnx.ModelProto,
    sources: Iterable[onnx.ModelProto],
) -> set[str]:
    initializers: dict[str, TensorProto] = {}
    for source in sources:
        for tensor in source.graph.initializer:
            existing = initializers.get(tensor.name)
            if existing is None:
                initializers[tensor.name] = tensor
            elif existing.SerializeToString() != tensor.SerializeToString():
                raise RuntimeError(f"Conflicting initializer during VoxCPM graph merge: {tensor.name}")
    destination.graph.initializer.extend(initializers.values())
    return set(initializers)


def check_model_allowing_runtime_extensions(path: str | Path) -> None:
    """Check a model while narrowly validating ORT's fused RMSNorm operator."""
    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    fused_nodes = [
        node
        for node in model.graph.node
        if not node.domain and node.op_type == "SimplifiedLayerNormalization"
    ]
    if not fused_nodes:
        onnx.checker.check_model(str(path), full_check=False)
        return

    known_values = {value.name for value in model.graph.input}
    known_values.update(tensor.name for tensor in model.graph.initializer)
    known_values.update(output for node in model.graph.node for output in node.output)
    fused_ids = {id(node) for node in fused_nodes}
    replacements = []
    for node in fused_nodes:
        attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        if (
            len(node.input) != 2
            or len(node.output) != 1
            or any(name not in known_values for name in node.input)
            or attributes.get("axis") != -1
            or attributes.get("stash_type") != 1
            or "epsilon" not in attributes
        ):
            raise RuntimeError(f"Invalid SimplifiedLayerNormalization node in {path.name}: {node}")
    for node in model.graph.node:
        if id(node) in fused_ids:
            replacements.append(
                helper.make_node(
                    "Identity",
                    [node.input[0]],
                    list(node.output),
                    name=(node.name + "_checker_identity") if node.name else "checker_identity",
                )
            )
        else:
            replacements.append(node)
    del model.graph.node[:]
    model.graph.node.extend(replacements)
    checker_path = path.with_name(path.name + ".checker.onnx")
    try:
        onnx.save(model, str(checker_path))
        onnx.checker.check_model(str(checker_path), full_check=False)
    finally:
        checker_path.unlink(missing_ok=True)


def validate_external_data_bounds(path: str | Path) -> None:
    """Fail when any initializer points outside its declared external-data file."""
    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    for tensor in model.graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        external = _external_data_map(tensor)
        location = external.get("location")
        if not location:
            raise RuntimeError(f"External initializer {tensor.name!r} has no location in {path.name}.")
        data_path = path.parent / location
        if not data_path.is_file():
            raise FileNotFoundError(f"External data for {tensor.name!r} is missing: {data_path}")
        offset = int(external.get("offset", "0"))
        length = int(external.get("length", "0"))
        if offset < 0 or length <= 0 or offset + length > data_path.stat().st_size:
            raise RuntimeError(
                f"External range for {tensor.name!r} is invalid: "
                f"offset={offset}, length={length}, file_size={data_path.stat().st_size}."
            )


def _save_checked_model(model: onnx.ModelProto, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        onnx.save(model, str(temporary_path))
        check_model_allowing_runtime_extensions(temporary_path)
        validate_external_data_bounds(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def compose_graphs(
    components: Sequence[GraphComponent],
    output_path: str | Path,
    output_names: Sequence[str],
    *,
    graph_name: str,
    input_names: Sequence[str] | None = None,
    delete_components: bool = False,
) -> Path:
    """Compose acyclic components by connecting named public tensors."""
    if not components:
        raise ValueError("At least one graph component is required.")
    output_path = Path(output_path)
    loaded = [
        _prefixed_component(
            onnx.load(str(component.path), load_external_data=False),
            component.prefix,
        )
        for component in components
    ]
    models_by_component = list(zip(components, loaded))
    available: dict[str, onnx.ValueInfoProto] = {}
    for _, model in models_by_component:
        for value in model.graph.output:
            if value.name in available:
                raise RuntimeError(f"Multiple components produce public tensor {value.name!r}.")
            available[value.name] = value

    merged = onnx.ModelProto()
    merged.ir_version = max(model.ir_version for model in loaded)
    merged.producer_name = Path(__file__).name
    merged.graph.name = graph_name
    _merge_opsets(merged, loaded)
    _copy_metadata(merged, loaded)

    public_inputs: dict[str, onnx.ValueInfoProto] = {}
    seen_inputs: dict[str, bytes] = {}
    for component, model in models_by_component:
        for value in model.graph.input:
            connected_name = component.connections.get(value.name)
            if connected_name is not None:
                if connected_name not in available:
                    raise RuntimeError(
                        f"{component.path.name} input {value.name!r} connects to missing "
                        f"tensor {connected_name!r}."
                    )
                for node in model.graph.node:
                    for index, name in enumerate(node.input):
                        if name == value.name:
                            node.input[index] = connected_name
                continue
            public_name = component.input_renames.get(value.name, value.name)
            if public_name != value.name:
                old_name = value.name
                value.name = public_name
                for node in model.graph.node:
                    for index, name in enumerate(node.input):
                        if name == old_name:
                            node.input[index] = public_name
            signature = value.type.SerializeToString()
            previous = seen_inputs.get(public_name)
            if previous is None:
                public_inputs[public_name] = value
                seen_inputs[public_name] = signature
            elif previous != signature:
                raise RuntimeError(f"Conflicting public input type for {public_name!r}.")

    if input_names is None:
        ordered_input_names = tuple(public_inputs)
    else:
        ordered_input_names = tuple(input_names)
        if len(set(ordered_input_names)) != len(ordered_input_names):
            raise ValueError("Composed input names must be unique.")
        requested_inputs = set(ordered_input_names)
        missing_inputs = [name for name in ordered_input_names if name not in public_inputs]
        unexpected_inputs = [name for name in public_inputs if name not in requested_inputs]
        if missing_inputs or unexpected_inputs:
            raise RuntimeError(
                "Explicit composed input order does not match public inputs: "
                f"missing={missing_inputs}, unexpected={unexpected_inputs}."
            )
    merged.graph.input.extend(public_inputs[name] for name in ordered_input_names)

    initializer_names = _merge_initializers(merged, loaded)
    for model in loaded:
        merged.graph.node.extend(model.graph.node)

    known_values = set(seen_inputs) | initializer_names
    for model in loaded:
        for value in model.graph.value_info:
            if value.name not in known_values:
                merged.graph.value_info.append(value)
                known_values.add(value.name)

    for name in output_names:
        value = available.get(name)
        if value is None:
            raise RuntimeError(f"Requested composed output {name!r} is not produced by any component.")
        merged.graph.output.append(value)

    _save_checked_model(merged, output_path)
    if delete_components:
        for component in components:
            component.path.unlink()
            component.path.with_name(component.path.name + ".data").unlink(missing_ok=True)
    return output_path


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
        if _num_elements(tensor) < min_elements or tensor.data_type in _UNSHAREABLE_INIT_TYPES:
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
            canonical_name = f"{SHARED_PREFIX}{len(unique):06d}"
            shared = (offset, len(raw), canonical_name)
            unique[identity] = shared
            carrier_initializers.append(
                _external_ref(
                    tensor,
                    SHARED_DATA_NAME,
                    offset,
                    len(raw),
                    name=canonical_name,
                )
            )
            unique_bytes += len(raw)
        offset, length, canonical_name = shared
        remap[tensor.name] = canonical_name
        if canonical_name not in graph_shared_names:
            rewritten.append(
                _external_ref(
                    tensor,
                    SHARED_DATA_NAME,
                    offset,
                    length,
                    name=canonical_name,
                )
            )
            graph_shared_names.add(canonical_name)
        reference_count += 1

    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    _remap_subgraph_inputs(model.graph, remap)
    return reference_count, source_bytes, unique_bytes


def _make_shared_carrier(
    carrier_initializers: Sequence[TensorProto],
    metadata: Mapping[str, str],
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=[],
        name="voxcpm_shared_initializers",
        inputs=[],
        outputs=[],
        initializer=list(carrier_initializers),
    )
    carrier = helper.make_model(graph, producer_name=Path(__file__).name)
    carrier.ir_version = 10
    _set_metadata(carrier, metadata)
    return carrier


def bundle_shared_initializers(
    folder: str | Path,
    model_paths: Sequence[str | Path] | None = None,
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, int | str]:
    """Pack exact-deduplicated weights into the canonical VoxCPM mmap blob."""
    folder = Path(folder).resolve()
    if min_elements <= 0:
        raise ValueError(f"min_elements must be positive, got {min_elements}.")
    if model_paths is None:
        targets = sorted(path for path in folder.glob("*.onnx") if path.name != SHARED_MODEL_NAME)
    else:
        targets = sorted(Path(path).resolve() for path in model_paths)
    if not targets:
        raise RuntimeError(f"No ONNX graphs found to bundle in {folder}.")
    if len({path.name for path in targets}) != len(targets):
        raise ValueError("Bundled ONNX graph file names must be unique.")
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ONNX graph(s): {missing}")

    shared_metadata = {
        "voxcpm_shared_initializers": "1",
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

    with tempfile.TemporaryDirectory(dir=folder, prefix=".voxcpm_shared_") as temp_name:
        temp_dir = Path(temp_name)
        staged_data = temp_dir / SHARED_DATA_NAME
        with staged_data.open("wb") as data_file:
            for target in targets:
                model = onnx.load(str(target), load_external_data=True)
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
                staged_model = temp_dir / target.name
                onnx.save(model, str(staged_model))
                staged_models.append((staged_model, folder / target.name))

        shared_metadata.update(
            {
                "shared_initializer_count": str(len(carrier_initializers)),
                "shared_initializer_reference_count": str(graph_reference_count),
                "shared_initializer_source_bytes": str(source_bytes),
                "shared_initializer_unique_bytes": str(unique_bytes),
            }
        )
        carrier = _make_shared_carrier(carrier_initializers, shared_metadata)
        staged_carrier = temp_dir / SHARED_MODEL_NAME
        onnx.save(carrier, str(staged_carrier))

        for staged_model, _ in staged_models:
            check_model_allowing_runtime_extensions(staged_model)
            validate_external_data_bounds(staged_model)
        onnx.checker.check_model(str(staged_carrier), full_check=False)
        validate_external_data_bounds(staged_carrier)

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


def attach_shared_initializers(session_options, shared_model_path: str | Path):
    """Mmap canonical weights and keep NumPy arrays plus OrtValues alive."""
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
        dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
        shape = tuple(int(dim) for dim in tensor.dims)
        array = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        arrays[tensor.name] = array
        ort_value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(tensor.name, ort_value)
    return arrays, ort_values