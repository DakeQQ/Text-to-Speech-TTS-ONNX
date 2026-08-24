"""Shared package metadata and exact-deduplicated ONNX initializer support."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import BinaryIO

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SHARED_MODEL_NAME = "FireRedTTS3_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
MIN_SHARED_INITIALIZER_ELEMENTS = 1024
SHARED_NAME_PREFIX = "fireredtts3_shared_"
_PACKED_4BIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)
_TensorIdentity = tuple[int, tuple[int, ...], int, bytes]
_SharedTensor = tuple[int, int, str]
INFERENCE_METADATA_KEYS = frozenset(
    {
        "package_schema_version",
        "graph_layout",
        "runtime_tensor_contract",
        "graph_owned_preprocess",
        "graph_owned_sampling",
        "graph_owned_postprocess",
        "device_resident_decode_state",
        "model_variant",
        "shared_initializer_model_file",
        "shared_initializer_data_file",
        "tokenizer_path_or_fingerprint",
        "out_sample_rate",
        "input_audio_sample_rate",
        "model_audio_sample_rate",
        "redae_downsample_rate",
        "redae_upsample_rate",
        "redae_max_seq_len",
        "patch_size",
        "flow_steps",
        "flow_schedule",
        "default_cfg",
        "default_clone_cfg",
        "max_seq_len",
        "vocab_size",
        "max_audio_patches",
        "min_audio_patches",
        "stop_threshold_default",
        "text_eot_id",
        "audio_sos_id",
        "latent_in_pad_id",
        "latent_out_pad_id",
    }
)


class PackageContractError(RuntimeError):
    """Raised when a FireRedTTS3 ONNX package is incomplete or inconsistent."""


def build_metadata(*sections: Mapping[str, object]) -> dict[str, str]:
    """Flatten structured settings to stable ONNX metadata string values."""
    metadata: dict[str, str] = {}
    for section in sections:
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                metadata[str(key)] = "1" if value else "0"
            elif isinstance(value, (dict, list, tuple)):
                metadata[str(key)] = json.dumps(
                    value, sort_keys=True, separators=(",", ":")
                )
            else:
                metadata[str(key)] = str(value)
    return metadata


def inference_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    """Keep only static values consumed by the inference package."""
    return {
        key: value
        for key, value in metadata.items()
        if key in INFERENCE_METADATA_KEYS
        or (
            key.startswith("model_file_name_")
            and key != "model_file_name_metadata"
        )
    }


def write_metadata_carrier(
    model_path: str | Path,
    metadata: Mapping[str, str],
    *,
    opset_version: int = 20,
) -> None:
    """Write an empty ONNX container holding only inference metadata constants."""
    graph = helper.make_graph([], "fireredtts3_metadata", [], [])
    model = helper.make_model(
        graph,
        producer_name=Path(__file__).name,
        opset_imports=[helper.make_opsetid("", opset_version)],
    )
    model.ir_version = 10
    for key, value in inference_metadata(metadata).items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.checker.check_model(model)
    onnx.save(model, str(model_path))


def write_onnx_metadata(model_path: str | Path, metadata: Mapping[str, str]) -> None:
    """Add or update metadata without materializing external initializers."""
    model = onnx.load(str(model_path), load_external_data=False)
    existing = {entry.key: entry for entry in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = str(value)
        else:
            model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, str(model_path))


def replace_onnx_metadata(model_path: str | Path, metadata: Mapping[str, str]) -> None:
    """Replace a graph's metadata with the package contract exactly."""
    model = onnx.load(str(model_path), load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, str(model_path))


def read_onnx_metadata(model_path: str | Path) -> dict[str, str]:
    """Read metadata without loading potentially large external data files."""
    model = onnx.load(str(model_path), load_external_data=False)
    return {entry.key: entry.value for entry in model.metadata_props}


def declared_model_files(metadata: Mapping[str, str]) -> dict[str, str]:
    """Return the metadata-declared runtime graph files keyed by metadata name."""
    return {
        key: value
        for key, value in metadata.items()
        if key.startswith("model_file_name_")
        and key != "model_file_name_metadata"
        and value
    }


def _tensor_element_count(tensor: TensorProto) -> int:
    count = 1
    for dimension in tensor.dims:
        count *= int(dimension)
    return count


def _expected_tensor_bytes(tensor: TensorProto) -> int:
    element_count = _tensor_element_count(tensor)
    if tensor.data_type in _PACKED_4BIT_TYPES:
        return (element_count + 1) // 2
    try:
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type))
    except (KeyError, TypeError, ValueError) as error:
        raise PackageContractError(
            f"Cannot determine a fixed byte width for initializer {tensor.name!r}."
        ) from error
    return element_count * dtype.itemsize


def _tensor_bytes(tensor: TensorProto) -> bytes:
    payload = tensor.raw_data if tensor.raw_data else numpy_helper.to_array(tensor).tobytes(order="C")
    expected = _expected_tensor_bytes(tensor)
    if len(payload) != expected:
        raise PackageContractError(
            f"Initializer {tensor.name!r} has {len(payload)} raw bytes but its "
            f"dtype/shape require exactly {expected}."
        )
    return payload


def _make_external_reference(
    tensor: TensorProto,
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
        ("location", SHARED_DATA_NAME),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        item = reference.external_data.add()
        item.key = key
        item.value = value
    return reference


def _inline_tensor(tensor: TensorProto) -> TensorProto:
    if tensor.data_location != TensorProto.EXTERNAL:
        return copy.deepcopy(tensor)
    inline = TensorProto()
    inline.name = tensor.name
    inline.data_type = tensor.data_type
    inline.dims.extend(tensor.dims)
    inline.raw_data = _tensor_bytes(tensor)
    return inline


def _remap_node_inputs(graph: onnx.GraphProto, remap: Mapping[str, str]) -> None:
    for node in graph.node:
        for index, value in enumerate(node.input):
            node.input[index] = remap.get(value, value)
        for attribute in node.attribute:
            if attribute.HasField("g"):
                _remap_node_inputs(attribute.g, remap)
            for nested_graph in attribute.graphs:
                _remap_node_inputs(nested_graph, remap)


def _rewrite_initializers(
    model: onnx.ModelProto,
    data_file: BinaryIO,
    unique: dict[_TensorIdentity, _SharedTensor],
    physical_data: dict[tuple[int, bytes], tuple[int, int]],
    carrier_initializers: list[TensorProto],
    min_elements: int,
) -> tuple[int, int, int]:
    """Move large initializers into a SHA-256 identity-addressed shared blob."""
    rewritten: list[TensorProto] = []
    remap: dict[str, str] = {}
    emitted_names: set[str] = set()
    references = 0
    source_bytes = 0
    unique_bytes = 0

    for tensor in model.graph.initializer:
        if _tensor_element_count(tensor) < min_elements:
            rewritten.append(_inline_tensor(tensor))
            continue

        payload = _tensor_bytes(tensor)
        digest = hashlib.sha256(payload).digest()
        identity = (
            int(tensor.data_type),
            tuple(int(dimension) for dimension in tensor.dims),
            len(payload),
            digest,
        )
        shared = unique.get(identity)
        if shared is None:
            physical_key = (len(payload), digest)
            physical = physical_data.get(physical_key)
            if physical is None:
                offset = data_file.tell()
                data_file.write(payload)
                physical = (offset, len(payload))
                physical_data[physical_key] = physical
                unique_bytes += len(payload)
            offset, length = physical
            canonical_name = f"{SHARED_NAME_PREFIX}{len(unique):06d}"
            shared = (offset, length, canonical_name)
            unique[identity] = shared
            carrier_initializers.append(
                _make_external_reference(
                    tensor, offset, length, name=canonical_name
                )
            )

        offset, length, canonical_name = shared
        remap[tensor.name] = canonical_name
        if canonical_name not in emitted_names:
            rewritten.append(
                _make_external_reference(
                    tensor, offset, length, name=canonical_name
                )
            )
            emitted_names.add(canonical_name)
        references += 1
        source_bytes += len(payload)

    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    _remap_node_inputs(model.graph, remap)
    return references, source_bytes, unique_bytes


def _make_shared_carrier(
    initializers: Iterable[TensorProto], metadata: Mapping[str, str]
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=[],
        name="fireredtts3_shared_initializers",
        inputs=[],
        outputs=[],
        initializer=list(initializers),
    )
    model = helper.make_model(graph, producer_name=Path(__file__).name)
    model.ir_version = 10
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    return model


def _target_paths(
    folder: Path, model_paths: Iterable[str | Path] | None
) -> list[Path]:
    if model_paths is None:
        targets = [
            path
            for path in folder.glob("*.onnx")
            if path.name != SHARED_MODEL_NAME
        ]
    else:
        targets = [Path(path).resolve() for path in model_paths]

    targets = sorted(set(targets))
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot bundle shared initializers; graph file(s) are missing:\n  "
            + "\n  ".join(missing)
        )
    outside_folder = [str(path) for path in targets if path.parent != folder]
    if outside_folder:
        raise ValueError(
            "Shared initializer targets must be direct children of the package folder:\n  "
            + "\n  ".join(outside_folder)
        )
    return targets


def _bundle_targets_to_folder(
    source_targets: list[Path],
    destination_folder: Path,
    *,
    min_elements: int,
    metadata: Mapping[str, str] | None,
) -> dict[str, int | str]:
    """Read source graphs and write a validated bundle into a destination folder."""
    if not source_targets:
        raise PackageContractError("Cannot bundle shared initializers for an empty package.")
    if not destination_folder.is_dir():
        raise NotADirectoryError(f"Bundle destination does not exist: {destination_folder}")
    if min_elements < 1:
        raise ValueError("min_elements must be positive")
    package_metadata = {str(key): str(value) for key, value in (metadata or {}).items()}
    unique: dict[_TensorIdentity, _SharedTensor] = {}
    physical_data: dict[tuple[int, bytes], tuple[int, int]] = {}
    carrier_initializers: list[TensorProto] = []
    staged: list[tuple[Path, Path]] = []
    reference_count = 0
    source_bytes = 0
    unique_bytes = 0

    with tempfile.TemporaryDirectory(
        dir=destination_folder, prefix=".fireredtts3_shared_"
    ) as temp_name:
        temp_dir = Path(temp_name)
        staged_data = temp_dir / SHARED_DATA_NAME
        with staged_data.open("wb") as data_file:
            for target in source_targets:
                model = onnx.load(str(target), load_external_data=True)
                references, model_source_bytes, model_unique_bytes = _rewrite_initializers(
                    model,
                    data_file,
                    unique,
                    physical_data,
                    carrier_initializers,
                    min_elements,
                )
                del model.metadata_props[:]
                for key, value in package_metadata.items():
                    model.metadata_props.add(key=key, value=value)
                staged_model = temp_dir / target.name
                onnx.save(model, str(staged_model))
                staged.append((staged_model, destination_folder / target.name))
                reference_count += references
                source_bytes += model_source_bytes
                unique_bytes += model_unique_bytes

        staged_carrier = temp_dir / SHARED_MODEL_NAME
        carrier = _make_shared_carrier(carrier_initializers, package_metadata)
        onnx.save(carrier, str(staged_carrier))

        final_data = destination_folder / SHARED_DATA_NAME
        final_carrier = destination_folder / SHARED_MODEL_NAME
        os.replace(staged_data, final_data)
        for staged_model, target in staged:
            os.replace(staged_model, target)
            target.with_name(target.name + ".data").unlink(missing_ok=True)
        os.replace(staged_carrier, final_carrier)

    return {
        "graph_count": len(source_targets),
        "initializer_references": reference_count,
        "unique_initializers": len(unique),
        "source_bytes": source_bytes,
        "unique_bytes": unique_bytes,
        "deduplicated_bytes": source_bytes - unique_bytes,
        "shared_model": str(destination_folder / SHARED_MODEL_NAME),
        "shared_data": str(destination_folder / SHARED_DATA_NAME),
    }


def bundle_shared_initializers(
    folder: str | Path,
    model_paths: Iterable[str | Path] | None = None,
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, int | str]:
    """Rewrite package graphs in place to use one exact-deduplicated bundle.

    Call this only inside a package staging directory. It validates every staged
    graph and carrier before replacement; callers atomically promote the entire
    package directory after this function succeeds.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Package folder does not exist: {folder}")
    targets = _target_paths(folder, model_paths)
    return _bundle_targets_to_folder(
        targets,
        folder,
        min_elements=min_elements,
        metadata=metadata,
    )


def bundle_shared_initializers_from_source(
    source_folder: str | Path,
    destination_folder: str | Path,
    model_paths: Iterable[str | Path] | None = None,
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, int | str]:
    """Build a bundle in an empty staging folder without linking source sidecars.

    This is for large raw exports whose external data cannot safely be copied by
    hard link: ONNX rejects multiple hard links as a potential integrity attack.
    Source graph files and source external data remain read-only throughout the
    operation.
    """
    source = Path(source_folder).resolve()
    destination = Path(destination_folder).resolve()
    if source == destination:
        return bundle_shared_initializers(
            source,
            model_paths=model_paths,
            min_elements=min_elements,
            metadata=metadata,
        )
    if not source.is_dir():
        raise NotADirectoryError(f"Bundle source does not exist: {source}")
    if not destination.is_dir():
        raise NotADirectoryError(f"Bundle destination does not exist: {destination}")
    if any(destination.iterdir()):
        raise PackageContractError(
            f"Source-to-staging bundle destination must be empty: {destination}"
        )
    targets = _target_paths(source, model_paths)
    return _bundle_targets_to_folder(
        targets,
        destination,
        min_elements=min_elements,
        metadata=metadata,
    )


def _external_data_map(tensor: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def audit_shared_initializer_storage(
    folder: str | Path,
    metadata: Mapping[str, str],
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
) -> dict[str, int]:
    """Verify that learned initializers occupy one content-addressed payload each."""
    folder = Path(folder).resolve()
    shared_model_name = metadata.get("shared_initializer_model_file")
    shared_data_name = metadata.get("shared_initializer_data_file")
    if not shared_model_name or not shared_data_name:
        raise PackageContractError("Package metadata is missing shared initializer file names.")

    carrier_path = folder / shared_model_name
    shared_data_path = folder / shared_data_name
    if not carrier_path.is_file() or not shared_data_path.is_file():
        raise FileNotFoundError(
            f"Shared initializer package is incomplete: {carrier_path}, {shared_data_path}"
        )
    carrier = onnx.load(str(carrier_path), load_external_data=False)
    carrier_by_name: dict[str, TensorProto] = {}
    ranges: dict[tuple[int, int], list[str]] = {}
    data_size = shared_data_path.stat().st_size
    for tensor in carrier.graph.initializer:
        if tensor.name in carrier_by_name:
            raise PackageContractError(
                f"Shared initializer carrier repeats tensor name {tensor.name!r}."
            )
        if tensor.data_location != TensorProto.EXTERNAL:
            raise PackageContractError(
                f"Shared initializer {tensor.name!r} is not external data."
            )
        external = _external_data_map(tensor)
        if external.get("location") != shared_data_name:
            raise PackageContractError(
                f"Shared initializer {tensor.name!r} points outside {shared_data_name!r}."
            )
        offset = int(external.get("offset", "0"))
        length = int(external.get("length", "0"))
        expected = _expected_tensor_bytes(tensor)
        if length != expected or offset < 0 or offset + length > data_size:
            raise PackageContractError(
                f"Shared initializer {tensor.name!r} has an invalid external-data range."
            )
        carrier_by_name[tensor.name] = tensor
        ranges.setdefault((offset, length), []).append(tensor.name)

    referenced_names: set[str] = set()
    reference_count = 0
    for file_name in declared_model_files(metadata).values():
        model = onnx.load(str(folder / file_name), load_external_data=False)
        for tensor in model.graph.initializer:
            if _tensor_element_count(tensor) < min_elements:
                continue
            reference_count += 1
            carrier_tensor = carrier_by_name.get(tensor.name)
            if carrier_tensor is None:
                raise PackageContractError(
                    f"{file_name} references unregistered shared initializer {tensor.name!r}."
                )
            tensor_external = _external_data_map(tensor)
            carrier_external = _external_data_map(carrier_tensor)
            tensor_signature = (
                int(tensor.data_type),
                tuple(int(dimension) for dimension in tensor.dims),
                tensor_external.get("location"),
                tensor_external.get("offset", "0"),
                tensor_external.get("length"),
            )
            carrier_signature = (
                int(carrier_tensor.data_type),
                tuple(int(dimension) for dimension in carrier_tensor.dims),
                carrier_external.get("location"),
                carrier_external.get("offset", "0"),
                carrier_external.get("length"),
            )
            if tensor.data_location != TensorProto.EXTERNAL or tensor_signature != carrier_signature:
                raise PackageContractError(
                    f"{file_name}:{tensor.name} does not match its shared carrier entry."
                )
            referenced_names.add(tensor.name)

    orphaned = sorted(carrier_by_name.keys() - referenced_names)
    if orphaned:
        raise PackageContractError(
            f"Shared initializer carrier contains {len(orphaned)} orphaned tensor(s): "
            + ", ".join(orphaned[:8])
        )

    ordered_ranges = sorted(ranges)
    for (offset, length), (next_offset, next_length) in zip(
        ordered_ranges,
        ordered_ranges[1:],
    ):
        if next_offset < offset + length:
            raise PackageContractError(
                "Shared initializer ranges overlap without being exact aliases: "
                f"({offset}, {length}) and ({next_offset}, {next_length})."
            )

    payload_ranges: dict[tuple[int, bytes], tuple[int, int]] = {}
    with shared_data_path.open("rb") as data_file:
        for offset, length in ordered_ranges:
            digest = hashlib.sha256()
            data_file.seek(offset)
            remaining = length
            while remaining:
                chunk = data_file.read(min(remaining, 8 * 1024 * 1024))
                if not chunk:
                    raise PackageContractError(
                        f"Shared initializer payload at offset {offset} is truncated."
                    )
                digest.update(chunk)
                remaining -= len(chunk)
            payload_key = (length, digest.digest())
            prior_range = payload_ranges.get(payload_key)
            if prior_range is not None and prior_range != (offset, length):
                raise PackageContractError(
                    "Duplicate initializer payload is stored in separate ranges: "
                    f"{prior_range} and {(offset, length)}."
                )
            payload_ranges[payload_key] = (offset, length)

    return {
        "initializer_references": reference_count,
        "logical_initializers": len(carrier_by_name),
        "physical_payloads": len(ranges),
        "aliased_initializers": len(carrier_by_name) - len(ranges),
        "physical_bytes": sum(length for _, length in ranges),
    }


def attach_shared_initializers(
    session_options,
    shared_model_path: str | Path,
    *,
    initializer_names: Iterable[str] | None = None,
):
    """Attach memory-mapped shared initializers and return objects to retain.

    The caller must retain the returned arrays and OrtValues for the lifetime of
    every ONNX Runtime session that uses ``session_options``. When
    ``initializer_names`` is provided, only that subset is registered.
    """
    import onnxruntime

    shared_model_path = Path(shared_model_path).resolve()
    if not shared_model_path.is_file():
        raise FileNotFoundError(f"Missing shared initializer carrier: {shared_model_path}")
    carrier = onnx.load(str(shared_model_path), load_external_data=False)
    requested_names = (
        None if initializer_names is None else frozenset(initializer_names)
    )
    carrier_names = {tensor.name for tensor in carrier.graph.initializer}
    if requested_names is not None:
        missing_names = sorted(requested_names - carrier_names)
        if missing_names:
            raise PackageContractError(
                "Requested shared initializers are absent from the carrier: "
                + ", ".join(missing_names[:8])
            )
    arrays: dict[str, np.ndarray] = {}
    ort_values = []
    for tensor in carrier.graph.initializer:
        if requested_names is not None and tensor.name not in requested_names:
            continue
        if tensor.data_type in _PACKED_4BIT_TYPES:
            continue
        if tensor.data_location != TensorProto.EXTERNAL:
            raise PackageContractError(
                f"Shared initializer {tensor.name!r} is not an external-data reference."
            )
        external = _external_data_map(tensor)
        location = external.get("location")
        if not location:
            raise PackageContractError(
                f"Shared initializer {tensor.name!r} has no external data location."
            )
        data_path = shared_model_path.parent / location
        if not data_path.is_file():
            raise FileNotFoundError(
                f"Shared initializer data is missing for {tensor.name!r}: {data_path}"
            )
        dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type))
        shape = tuple(int(dimension) for dimension in tensor.dims)
        offset = int(external.get("offset", "0"))
        expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        length = int(external.get("length", expected_size))
        if length != expected_size or offset < 0 or offset + length > data_path.stat().st_size:
            raise PackageContractError(
                f"Shared initializer {tensor.name!r} has an invalid external-data range."
            )
        array = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        arrays[tensor.name] = array
        value = onnxruntime.OrtValue.ortvalue_from_numpy(array)
        session_options.add_initializer(tensor.name, value)
        ort_values.append(value)
    return arrays, ort_values


def _remap_graph_names(graph: onnx.GraphProto, remap: Mapping[str, str]) -> None:
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = remap.get(name, name)
        for attribute in node.attribute:
            if attribute.HasField("g"):
                _remap_graph_names(attribute.g, remap)
            for nested_graph in attribute.graphs:
                _remap_graph_names(nested_graph, remap)


def prefix_graph_local_names(
    model: onnx.ModelProto,
    prefix: str,
    *,
    preserved_names: Iterable[str] = (),
) -> onnx.ModelProto:
    """Prefix local graph names while preserving public I/O and bridge values."""
    prefixed = copy.deepcopy(model)
    graph = prefixed.graph
    public_names = {value.name for value in graph.input}
    public_names.update(value.name for value in graph.output)
    public_names.update(preserved_names)
    remap: dict[str, str] = {}

    for initializer in graph.initializer:
        if initializer.name.startswith(SHARED_NAME_PREFIX):
            continue
        if initializer.name not in public_names:
            remap[initializer.name] = prefix + initializer.name
    for node in graph.node:
        if node.name:
            node.name = prefix + node.name
        for output in node.output:
            if output and output not in public_names:
                remap.setdefault(output, prefix + output)

    for initializer in graph.initializer:
        initializer.name = remap.get(initializer.name, initializer.name)
    for value in graph.value_info:
        value.name = remap.get(value.name, value.name)
    _remap_graph_names(graph, remap)
    return prefixed


def validate_package_contract(
    folder: str | Path,
    metadata_file_name: str,
    *,
    required_keys: Iterable[str] = (),
    require_shared_bundle: bool = True,
) -> dict[str, str]:
    """Validate declared graph files, package metadata, and shared bundle files."""
    folder = Path(folder).resolve()
    metadata_path = folder / metadata_file_name
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Package metadata graph is missing: {metadata_path}")
    metadata = read_onnx_metadata(metadata_path)
    missing_keys = sorted(set(required_keys) - metadata.keys())
    if missing_keys:
        raise PackageContractError(
            f"{metadata_path.name} is missing required metadata key(s): {missing_keys}"
        )

    declared = declared_model_files(metadata)
    if not declared:
        raise PackageContractError("Package metadata declares no runtime graph files.")
    missing_graphs = [
        f"{key} -> {value}"
        for key, value in sorted(declared.items())
        if not (folder / value).is_file()
    ]
    if missing_graphs:
        raise FileNotFoundError(
            "Package metadata declares missing graph file(s):\n  "
            + "\n  ".join(missing_graphs)
        )

    if require_shared_bundle:
        shared_model_name = metadata.get("shared_initializer_model_file")
        shared_data_name = metadata.get("shared_initializer_data_file")
        if not shared_model_name or not shared_data_name:
            raise PackageContractError("Package metadata is missing shared initializer file names.")
        for file_name in (shared_model_name, shared_data_name):
            if not (folder / file_name).is_file():
                raise FileNotFoundError(
                    f"Package metadata declares missing shared initializer artifact: {file_name}"
                )

    for graph_name in set(declared.values()):
        graph_metadata = read_onnx_metadata(folder / graph_name)
        mismatched = {
            key: value
            for key, value in metadata.items()
            if graph_metadata.get(key) != value
        }
        if mismatched:
            raise PackageContractError(
                f"{graph_name} does not carry the package metadata contract: "
                + ", ".join(sorted(mismatched))
            )
    return metadata


def promote_directory(staging_folder: str | Path, final_folder: str | Path) -> None:
    """Atomically promote a fully validated package directory with rollback."""
    staging = Path(staging_folder).resolve()
    final = Path(final_folder).resolve()
    if not staging.is_dir():
        raise NotADirectoryError(f"Staging package does not exist: {staging}")
    if staging.parent != final.parent:
        raise ValueError("Staging and final package folders must share a parent directory.")
    backup = final.with_name(final.name + ".previous")
    shutil.rmtree(backup, ignore_errors=True)
    if final.exists():
        os.replace(final, backup)
    try:
        os.replace(staging, final)
    except BaseException:
        if backup.exists() and not final.exists():
            os.replace(backup, final)
        raise
    shutil.rmtree(backup, ignore_errors=True)