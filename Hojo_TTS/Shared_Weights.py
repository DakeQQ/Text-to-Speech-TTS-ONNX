"""Pack Hojo-TTS ONNX initializers into one deduplicated external-data blob."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SHARED_MODEL_NAME = "Hojo_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
SHARED_TENSOR_PREFIX = "hojo_tts_shared_"
SHARED_CARRIER_FLAG = "hojo_tts_shared_initializers"
MIN_SHARED_INITIALIZER_ELEMENTS = 1024
MIN_SHARED_FLOAT_ELEMENTS = 2
NORM_OPS = (
    "SimplifiedLayerNormalization",
    "SkipSimplifiedLayerNormalization",
    "LayerNormalization",
    "SkipLayerNormalization",
    "RMSNormalization",
)
FUSED_BIAS_NORM_OPS = ("SkipLayerNormalization", "SkipSimplifiedLayerNormalization")
_FLOAT_SHARE_TYPES = frozenset((TensorProto.FLOAT, TensorProto.FLOAT16))
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)
_TensorIdentity = tuple[int, tuple[int, ...], int, bytes]
_UniqueEntry = tuple[int, tuple[int, ...], int, bytes, str]
_AliasSpec = tuple[_TensorIdentity, int, int]


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
    data_type: int,
    dims: tuple[int, ...] | list[int],
    location: str,
    offset: int,
    length: int,
    name: str,
) -> TensorProto:
    ref = TensorProto()
    ref.name = name
    ref.data_type = data_type
    ref.dims.extend(dims)
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
    mapping: dict[str, str] = {}
    for entry in tensor.external_data:
        if entry.key in mapping:
            raise ValueError(f"Duplicate external_data key {entry.key!r} on {tensor.name}")
        mapping[entry.key] = entry.value
    return mapping


def _inline_tensor(tensor: TensorProto) -> TensorProto:
    if tensor.data_location != TensorProto.EXTERNAL:
        return tensor
    inline = TensorProto()
    inline.name = tensor.name
    inline.data_type = tensor.data_type
    inline.dims.extend(tensor.dims)
    inline.raw_data = _tensor_bytes(tensor)
    return inline


def inline_model_initializers(model_or_path: onnx.ModelProto | str | Path) -> onnx.ModelProto:
    """Load an ONNX model (with external data if from path) and inline all external initializers."""
    if isinstance(model_or_path, (str, Path)):
        model = onnx.load(str(model_or_path), load_external_data=True)
    else:
        model = model_or_path
    inlined = [_inline_tensor(tensor) for tensor in model.graph.initializer]
    del model.graph.initializer[:]
    model.graph.initializer.extend(inlined)
    return model


def iter_subgraphs(node: onnx.NodeProto):
    """Yield every ONNX GraphProto attribute attached to `node`."""
    for attribute in node.attribute:
        if attribute.HasField("g"):
            yield attribute.g
        for subgraph in attribute.graphs:
            yield subgraph


def set_sorted_metadata(model: onnx.ModelProto, metadata: dict[str, object]) -> None:
    """Replace model metadata_props with deterministically sorted string key/value pairs."""
    del model.metadata_props[:]
    for key in sorted(metadata):
        model.metadata_props.add(key=str(key), value=str(metadata[key]))


def _remap_node_inputs(graph: onnx.GraphProto, remap: dict[str, str], *, is_root: bool = True) -> None:
    if not remap:
        return
    if not is_root:
        local = {value.name for value in graph.input} | {tensor.name for tensor in graph.initializer}
        remap = {k: v for k, v in remap.items() if k not in local}
        if not remap:
            return
    active = dict(remap)
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = active.get(name, name)
        for subgraph in iter_subgraphs(node):
            _remap_node_inputs(subgraph, active, is_root=False)
        if not is_root:
            for out in node.output:
                active.pop(out, None)
    if not is_root:
        for out in graph.output:
            out.name = active.get(out.name, out.name)


def collect_live_external_names(models: dict[str, onnx.ModelProto], carrier_name: str = SHARED_MODEL_NAME) -> set[str]:
    """Recursively collect external initializer names referenced across active graphs and control-flow subgraphs."""
    live: set[str] = set()

    def scan_graph(graph: onnx.GraphProto, ext_map: dict[str, TensorProto]) -> None:
        local_ext = dict(ext_map)
        init_names = {t.name for t in graph.initializer}
        for tensor in graph.initializer:
            if tensor.external_data:
                local_ext[tensor.name] = tensor
            else:
                local_ext.pop(tensor.name, None)
        for inp in graph.input:
            if inp.name not in init_names:
                local_ext.pop(inp.name, None)
        for node in graph.node:
            for inp in node.input:
                if inp in local_ext:
                    live.add(inp)
            for out in node.output:
                local_ext.pop(out, None)
            for sg in iter_subgraphs(node):
                scan_graph(sg, local_ext)

    for name, model in models.items():
        if name != carrier_name:
            scan_graph(model.graph, {})
    return live


def collect_normalization_exclusions(
    models: dict[str, onnx.ModelProto],
    carrier_name: str = SHARED_MODEL_NAME,
    shared_names: set[str] | None = None,
) -> set[str]:
    """Collect shared initializers prepacked by ORT LayerNorm/SkipLayerNorm kernels (including BiasSkipLayerNormFusion)."""
    if shared_names is None:
        carrier = models.get(carrier_name)
        if carrier is None:
            return set()
        shared_names = {tensor.name for tensor in carrier.graph.initializer}
    excluded: set[str] = set()

    def scan(graph: onnx.GraphProto) -> None:
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for inp in node.input:
                consumers.setdefault(inp, []).append(node)
        for node in graph.node:
            if node.op_type in NORM_OPS:
                excluded.update(name for name in node.input if name in shared_names)
            elif node.op_type == "Add" and node.output:
                down = consumers.get(node.output[0], ())
                if any(child.op_type in FUSED_BIAS_NORM_OPS for child in down):
                    excluded.update(name for name in node.input if name in shared_names)
            for sg in iter_subgraphs(node):
                scan(sg)

    for name, model in models.items():
        if name != carrier_name:
            scan(model.graph)
    return excluded


def _resolve_bundle_targets(
    folder: Path,
    model_paths: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    if model_paths is None:
        targets = sorted(path for path in folder.glob("*.onnx") if path.name != SHARED_MODEL_NAME)
    else:
        targets = sorted(Path(path).resolve() for path in model_paths)
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot bundle missing graph(s): {', '.join(missing)}.")
    names = [path.name for path in targets]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Cannot bundle graphs that share a file name: {', '.join(duplicates)}.")
    return targets


def _collect_model_initializers(
    model: onnx.ModelProto,
    all_canonical: dict[_TensorIdentity, str],
    unique_order: list[_TensorIdentity],
    unique_meta: dict[_TensorIdentity, _UniqueEntry],
    aliases: dict[_TensorIdentity, _AliasSpec],
    used_aliases: dict[_TensorIdentity, str],
    min_elements: int,
    *,
    min_float_elements: int = MIN_SHARED_FLOAT_ELEMENTS,
) -> tuple[list[tuple[_TensorIdentity, int, tuple[int, ...], int, str]], int, int]:
    """Detach shareable initializers from `model`, deduplicate by content, and remap node inputs."""
    kept_inline: list[TensorProto] = []
    shared_refs: list[tuple[_TensorIdentity, int, tuple[int, ...], int, str]] = []
    remap: dict[str, str] = {}
    graph_shared_names: set[str] = set()
    reference_count = 0
    source_bytes = 0

    for tensor in model.graph.initializer:
        elements = _num_elements(tensor)
        threshold = min_float_elements if tensor.data_type in _FLOAT_SHARE_TYPES else min_elements
        if elements < threshold or tensor.data_type in _UNSHAREABLE_INIT_TYPES:
            kept_inline.append(_inline_tensor(tensor))
            continue

        raw = _tensor_bytes(tensor)
        dims = tuple(int(dim) for dim in tensor.dims)
        identity = (tensor.data_type, dims, len(raw), hashlib.sha256(raw).digest())
        canonical_name = all_canonical.get(identity)
        if canonical_name is None:
            canonical_name = f"{SHARED_TENSOR_PREFIX}{len(all_canonical):06d}"
            all_canonical[identity] = canonical_name
            if identity in aliases:
                if aliases[identity][2] != len(raw):
                    raise ValueError(f"Invalid shared sub-tensor length for {tensor.name}")
                used_aliases[identity] = canonical_name
            else:
                alignment = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type)).alignment
                unique_meta[identity] = (tensor.data_type, dims, alignment, raw, canonical_name)
                unique_order.append(identity)

        remap[tensor.name] = canonical_name
        if canonical_name not in graph_shared_names:
            shared_refs.append((identity, tensor.data_type, dims, len(raw), canonical_name))
            graph_shared_names.add(canonical_name)
            reference_count += 1
            source_bytes += len(raw)

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inline)
    _remap_node_inputs(model.graph, remap)
    return shared_refs, reference_count, source_bytes


def _audio_embedding_alias(
    model: onnx.ModelProto, metadata: dict[str, str],
) -> tuple[_TensorIdentity, _TensorIdentity, int] | None:
    if "audio_token_first" not in metadata or "audio_token_last" not in metadata:
        return None
    names = {"hj_token_embedding"}
    names.update(node.input[0] for node in model.graph.node if node.op_type == "Gather"
                 and (node.input[1] == "input_ids" or node.output[0] in ("inputs_embeds", "hj_emb")))
    candidates = [tensor for tensor in model.graph.initializer if tensor.name in names
                  and tensor.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16) and len(tensor.dims) == 2]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("Shared audio-row alias requires one LM token embedding table")
    tensor = candidates[0]
    first, last = int(metadata["audio_token_first"]), int(metadata["audio_token_last"])
    if not 0 <= first <= last < tensor.dims[0]:
        raise ValueError("Audio token bounds exceed the LM embedding table")
    raw = _tensor_bytes(tensor)
    row_bytes = len(raw) // tensor.dims[0]
    start, stop = first * row_bytes, (last + 1) * row_bytes
    parent = (tensor.data_type, tuple(tensor.dims), len(raw), hashlib.sha256(raw).digest())
    alias = (tensor.data_type, (last - first + 1, tensor.dims[1]), stop - start,
             hashlib.sha256(memoryview(raw)[start:stop]).digest())
    return parent, alias, start


def _make_shared_carrier(
    carrier_initializers: list[TensorProto],
    metadata: dict[str, str],
) -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=[helper.make_node("Identity", ["metadata_marker"], ["metadata_marker_out"])],
        name=SHARED_CARRIER_FLAG,
        inputs=[helper.make_tensor_value_info("metadata_marker", TensorProto.INT32, [1])],
        outputs=[helper.make_tensor_value_info("metadata_marker_out", TensorProto.INT32, [1])],
        initializer=carrier_initializers,
    )
    carrier = helper.make_model(
        graph, producer_name=Path(__file__).name, opset_imports=[helper.make_opsetid("", 20)],
    )
    carrier.ir_version = 8
    set_sorted_metadata(carrier, metadata)
    return carrier


def bundle_shared_initializers(
    folder: str | Path,
    model_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    min_float_elements: int = MIN_SHARED_FLOAT_ELEMENTS,
    metadata: dict[str, str] | None = None,
) -> dict[str, int | str]:
    """Redirect multi-element FP and large initializers to one alignment-compacted external blob."""
    folder = Path(folder).resolve()
    targets = _resolve_bundle_targets(folder, model_paths)

    shared_metadata = {
        SHARED_CARRIER_FLAG: "1",
        "shared_initializer_model_file": SHARED_MODEL_NAME,
        "shared_initializer_data_file": SHARED_DATA_NAME,
    }
    if metadata:
        shared_metadata.update({str(key): str(value) for key, value in metadata.items()})
    lm_name = shared_metadata.get("model_file_name_lm", "Hojo_LM.onnx")
    targets.sort(key=lambda path: (path.name != lm_name, path.name))

    all_canonical: dict[_TensorIdentity, str] = {}
    unique_order: list[_TensorIdentity] = []
    unique_meta: dict[_TensorIdentity, _UniqueEntry] = {}
    aliases: dict[_TensorIdentity, _AliasSpec] = {}
    used_aliases: dict[_TensorIdentity, str] = {}
    rewritten_models: list[tuple[Path, onnx.ModelProto, dict[str, str], list[tuple[_TensorIdentity, int, tuple[int, ...], int, str]]]] = []
    graph_reference_count = 0
    source_bytes = 0

    for target in targets:
        model = onnx.load(str(target), load_external_data=True)
        existing_props = {entry.key: entry.value for entry in model.metadata_props}
        audio_alias = _audio_embedding_alias(model, shared_metadata) if target.name == lm_name else None
        shared_refs, reference_count, model_source_bytes = _collect_model_initializers(
            model,
            all_canonical,
            unique_order,
            unique_meta,
            aliases,
            used_aliases,
            min_elements,
            min_float_elements=min_float_elements,
        )
        if audio_alias is not None:
            parent_id, child_id, relative_offset = audio_alias
            if parent_id in unique_meta:
                aliases[child_id] = (parent_id, relative_offset, child_id[2])
        graph_reference_count += reference_count
        source_bytes += model_source_bytes
        rewritten_models.append((target, model, existing_props, shared_refs))

    # Stable sort by descending element alignment eliminates inter-tensor alignment padding for all power-of-2 dtypes.
    packed_order = sorted(unique_order, key=lambda ident: -unique_meta[ident][2])
    offsets: dict[_TensorIdentity, tuple[int, int]] = {}
    carrier_initializers: list[TensorProto] = []

    with tempfile.TemporaryDirectory(dir=folder, prefix=f".{SHARED_TENSOR_PREFIX}") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staged_data = temp_dir / SHARED_DATA_NAME

        with staged_data.open("wb") as data_file:
            for ident in packed_order:
                data_type, dims, alignment, raw, canonical_name = unique_meta[ident]
                padding = -data_file.tell() % alignment
                if padding:
                    data_file.write(bytes(padding))
                offset = data_file.tell()
                data_file.write(raw)
                offsets[ident] = (offset, len(raw))
                carrier_initializers.append(
                    _external_ref(data_type, dims, SHARED_DATA_NAME, offset, len(raw), canonical_name)
                )

        # Free raw bytes now that the external blob is written to disk.
        unique_meta.clear()

        for child_id, canonical_name in used_aliases.items():
            parent_id, relative_offset, length = aliases[child_id]
            parent_offset, _ = offsets[parent_id]
            alias_offset = parent_offset + relative_offset
            offsets[child_id] = (alias_offset, length)
            carrier_initializers.append(
                _external_ref(child_id[0], child_id[1], SHARED_DATA_NAME, alias_offset, length, canonical_name)
            )

        for _, model, _, shared_refs in rewritten_models:
            for ident, data_type, dims, length, canonical_name in shared_refs:
                offset, _ = offsets[ident]
                model.graph.initializer.append(
                    _external_ref(data_type, dims, SHARED_DATA_NAME, offset, length, canonical_name)
                )

        unique_bytes = staged_data.stat().st_size
        shared_names = {tensor.name for tensor in carrier_initializers}
        loaded_map = {target.name: model for target, model, _, _ in rewritten_models}
        normalization_exclusions = collect_normalization_exclusions(loaded_map, SHARED_MODEL_NAME, shared_names)

        shared_metadata.update({
            "shared_initializer_count": str(len(carrier_initializers)),
            "shared_initializer_references": str(graph_reference_count),
            "shared_initializer_reference_count": str(graph_reference_count),
            "shared_initializer_source_bytes": str(source_bytes),
            "shared_initializer_unique_bytes": str(unique_bytes),
            "shared_initializer_deduplicated_bytes": str(source_bytes - unique_bytes),
            "shared_initializer_subtensor_aliases": str(len(used_aliases)),
            "shared_initializer_subtensor_bytes": str(sum(identity[2] for identity in used_aliases)),
            "shared_initializer_prepack_excluded": ",".join(sorted(normalization_exclusions)),
        })
        if metadata is not None:
            metadata.update(shared_metadata)

        staged_models: list[tuple[Path, Path]] = []
        for target, model, existing_props, _ in rewritten_models:
            per_model_props = dict(existing_props)
            per_model_props.update(shared_metadata)
            if "model_file_name_vocoder" not in shared_metadata:
                per_model_props.pop("model_file_name_vocoder", None)
            set_sorted_metadata(model, per_model_props)
            staged_model = temp_dir / target.name
            onnx.save(model, str(staged_model))
            staged_models.append((staged_model, folder / target.name))

        carrier = _make_shared_carrier(carrier_initializers, shared_metadata)
        staged_carrier = temp_dir / SHARED_MODEL_NAME
        onnx.save(carrier, str(staged_carrier))

        final_data = folder / SHARED_DATA_NAME
        final_carrier = folder / SHARED_MODEL_NAME
        os.replace(staged_data, final_data)
        for staged_model, target in staged_models:
            os.replace(staged_model, target)
            target.with_name(target.name + ".data").unlink(missing_ok=True)
        os.replace(staged_carrier, final_carrier)

    return {
        "graph_count": len(targets),
        "initializer_references": graph_reference_count,
        "unique_initializers": len(all_canonical),
        "source_bytes": source_bytes,
        "unique_bytes": unique_bytes,
        "deduplicated_bytes": source_bytes - unique_bytes,
        "subtensor_aliases": len(used_aliases),
        "subtensor_alias_bytes": sum(identity[2] for identity in used_aliases),
        "prepack_excluded_count": len(normalization_exclusions),
        "shared_model": str(folder / SHARED_MODEL_NAME),
        "shared_data": str(folder / SHARED_DATA_NAME),
    }


def _declared_external_initializer_names(model_path: str | Path) -> set[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    names: set[str] = set()

    def scan(graph: onnx.GraphProto) -> None:
        for tensor in graph.initializer:
            if tensor.external_data:
                names.add(tensor.name)
        for node in graph.node:
            for subgraph in iter_subgraphs(node):
                scan(subgraph)

    scan(model.graph)
    return names


class SharedInitializerStore:
    """Single-mmap shared initializer cache with prepack-exclusion and per-model filtering."""

    def __init__(self, shared_model_path: str | Path, *, respect_prepack_exclusions: bool = True) -> None:
        import onnxruntime

        self.shared_model_path = Path(shared_model_path).resolve()
        shared_model = onnx.load(str(self.shared_model_path), load_external_data=False)
        props = {entry.key: entry.value for entry in shared_model.metadata_props}
        self.excluded: frozenset[str] = (
            frozenset(filter(None, props.get("shared_initializer_prepack_excluded", "").split(",")))
            if respect_prepack_exclusions else frozenset()
        )
        self.arrays: dict[str, np.ndarray] = {}
        self.ort_values: dict[str, object] = {}
        self._blob: np.memmap | None = None
        self._declared_cache: dict[Path, frozenset[str]] = {}

        tensors = list(shared_model.graph.initializer)
        if not tensors:
            return
        locations = {_external_data_map(tensor)["location"] for tensor in tensors}
        if len(locations) != 1:
            raise ValueError("Shared initializers must reference one data blob")
        location = locations.pop()
        if not location or Path(location).name != location:
            raise ValueError("Shared data must be a file in the package")
        blob = np.memmap(self.shared_model_path.parent / location, dtype=np.uint8, mode="r")
        self._blob = blob
        seen_names: set[str] = set()
        for tensor in tensors:
            if tensor.data_type in _UNSHAREABLE_INIT_TYPES:
                raise ValueError(f"Unsupported shared initializer type: {tensor.data_type}")
            external = _external_data_map(tensor)
            offset, length = int(external["offset"]), int(external["length"])
            dtype = np.dtype(helper.tensor_dtype_to_np_dtype(tensor.data_type))
            shape = tuple(int(dim) for dim in tensor.dims)
            expected = _num_elements(tensor) * dtype.itemsize
            if offset < 0 or length != expected or offset + length > blob.size or offset % dtype.alignment != 0:
                raise ValueError(f"Invalid shared initializer byte span: {tensor.name}")
            if tensor.name in seen_names:
                raise ValueError(f"Duplicate shared initializer: {tensor.name}")
            seen_names.add(tensor.name)
            if tensor.name in self.excluded:
                continue
            array = np.ndarray(shape, dtype=dtype, buffer=blob, offset=offset)
            if array.ctypes.data % dtype.alignment != 0:
                continue
            self.arrays[tensor.name] = array
            self.ort_values[tensor.name] = onnxruntime.OrtValue.ortvalue_from_numpy(array)

    def attach_to(self, session_options, model_path: str | Path | None = None) -> int:
        """Bind non-excluded shared OrtValues declared by `model_path` (or all if omitted)."""
        declared: frozenset[str] | None = None
        if model_path is not None:
            resolved = Path(model_path).resolve()
            declared = self._declared_cache.get(resolved)
            if declared is None:
                declared = frozenset(_declared_external_initializer_names(resolved))
                self._declared_cache[resolved] = declared
        attached = 0
        for name, ort_value in self.ort_values.items():
            if declared is not None and name not in declared:
                continue
            session_options.add_initializer(name, ort_value)
            attached += 1
        return attached


def attach_shared_initializers(
    session_options,
    shared_model_path: str | Path,
    *,
    model_path: str | Path | None = None,
    respect_prepack_exclusions: bool = True,
):
    """Mmap the shared blob and inject non-excluded canonical OrtValues into SessionOptions."""
    store = SharedInitializerStore(shared_model_path, respect_prepack_exclusions=respect_prepack_exclusions)
    store.attach_to(session_options, model_path=model_path)
    return store.arrays, list(store.ort_values.values())
