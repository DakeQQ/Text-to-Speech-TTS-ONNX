"""Pack Qwen3-TTS ONNX initializers into one deduplicated external-data blob."""

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


SHARED_MODEL_NAME = "QwenTTS_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
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

    target_names = [path.name for path in targets]
    if len(set(target_names)) != len(target_names):
        raise ValueError("Bundled ONNX graph file names must be unique in the target folder.")
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
        digest = hashlib.sha256(raw).digest()
        identity = (tensor.data_type, tuple(int(dim) for dim in tensor.dims), len(raw), digest)
        source_bytes += len(raw)
        shared = unique.get(identity)
        if shared is None:
            offset = data_file.tell()
            data_file.write(raw)
            canonical_name = f"qwentts_shared_{len(unique):06d}"
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
        name="qwen_tts_shared_initializers",
        inputs=[],
        outputs=[],
        initializer=carrier_initializers,
    )
    carrier = helper.make_model(graph, producer_name=Path(__file__).name)
    carrier.ir_version = 10
    _set_metadata(carrier, metadata)
    return carrier


def check_model_allowing_runtime_extensions(path: str | Path) -> None:
    """Run ONNX checker, substituting only ORT's checker-unknown fused RMS op."""
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
    replacements = []
    fused_ids = {id(node) for node in fused_nodes}
    for node in fused_nodes:
        attributes = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
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


def bundle_shared_initializers(
    folder: str | Path,
    model_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    metadata: dict[str, str] | None = None,
) -> dict[str, int | str]:
    """Redirect all large initializers to one exact-deduplicated external blob.

    Models and the shared blob are staged and validated before atomically replacing
    the original graph files. Tensor identity is keyed by dtype, shape, byte length,
    and SHA-256 digest; equal values in different graphs reuse the same byte range.
    """
    folder = Path(folder).resolve()
    if min_elements <= 0:
        raise ValueError(f"min_elements must be positive, got {min_elements}.")

    targets = _resolve_bundle_targets(folder, model_paths)

    shared_metadata = {
        "qwen_tts_shared_initializers": "1",
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

    with tempfile.TemporaryDirectory(dir=folder, prefix=".qwentts_shared_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
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
            check_model_allowing_runtime_extensions(staged_model)
        onnx.checker.check_model(str(staged_carrier), full_check=False)

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
        "unique_initializers": len(unique),
        "source_bytes": source_bytes,
        "unique_bytes": unique_bytes,
        "deduplicated_bytes": source_bytes - unique_bytes,
        "shared_model": str(folder / SHARED_MODEL_NAME),
        "shared_data": str(folder / SHARED_DATA_NAME),
    }


def _prefixed_component(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    """Prefix graph-local names while preserving public I/O and canonical shared weights."""
    model = copy.deepcopy(model)
    public_names = {value.name for value in model.graph.input}
    public_names.update(value.name for value in model.graph.output)
    remap: dict[str, str] = {}

    for tensor in model.graph.initializer:
        if not tensor.name.startswith("qwentts_shared_"):
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


def _merge_initializers(
    dst: onnx.ModelProto,
    *sources: onnx.ModelProto,
) -> set[str]:
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
        check_model_allowing_runtime_extensions(temp_path)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _delete_model_artifacts(*model_paths: Path) -> None:
    for model_path in model_paths:
        model_path.unlink()
    for model_path in model_paths:
        model_path.with_name(model_path.name + ".data").unlink(missing_ok=True)


def merge_decode_step_graph(
    predictor_path: str | Path,
    main_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Merge one strategy's PredictorFrame output into its MainDecode input."""
    predictor_path = Path(predictor_path)
    main_path = Path(main_path)
    output_path = Path(output_path)
    predictor = _prefixed_component(
        onnx.load(str(predictor_path), load_external_data=False),
        "predictor/",
    )
    main = _prefixed_component(
        onnx.load(str(main_path), load_external_data=False),
        "main/",
    )

    predictor_outputs = {value.name: value for value in predictor.graph.output}
    main_inputs = {value.name: value for value in main.graph.input}
    if "frame_codec_ids" not in predictor_outputs or "frame_codec_ids" not in main_inputs:
        raise RuntimeError("DecodeStep merge requires PredictorFrame -> MainDecode frame_codec_ids.")

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

    desired_outputs = [value for value in main.graph.output]
    desired_outputs.extend(
        predictor_outputs[name]
        for name in ("generated_codec", "frame_codec_ids")
    )
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
        predictor_path = folder / f"QwenTTS_PredictorFrame_{strategy}.onnx"
        main_path = folder / f"QwenTTS_MainDecode_{strategy}.onnx"
        output_path = folder / f"QwenTTS_DecodeStep_{strategy}.onnx"
        outputs[strategy] = merge_decode_step_graph(predictor_path, main_path, output_path)
        if delete_components:
            _delete_model_artifacts(predictor_path, main_path)
    return outputs


def attach_shared_initializers(session_options, shared_model_path: str | Path):
    """Mmap the shared blob and inject canonical OrtValues into SessionOptions.

    The returned arrays and OrtValues must stay alive for every session created
    from ``session_options``.
    """
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