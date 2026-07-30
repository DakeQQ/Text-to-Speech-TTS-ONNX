"""Merge compact IndexTTS2 decode graphs and pack their shared ONNX weights."""

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


SHARED_MODEL_NAME = "IndexTTS_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"
MIN_SHARED_INITIALIZER_ELEMENTS = 1024
DEFAULT_DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")
_CANONICAL_PREFIX = "indextts_shared_"
PACKAGE_STAT_METADATA_KEYS = frozenset({
    "package_graph_count",
    "runtime_graph_count",
    "package_external_reference_count",
    "shared_initializer_count",
    "shared_initializer_reference_count",
    "shared_initializer_source_bytes",
    "shared_initializer_unique_bytes",
    "shared_initializer_deduplicated_bytes",
    "shared_initializer_data_bytes",
})
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
    targets = [Path(path).expanduser().resolve() for path in model_paths]
    names = [path.name for path in targets]
    missing = [str(path) for path in targets if not path.is_file()]
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
        name="index_tts2_shared_initializers",
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


def bundle_shared_initializers(
    folder: str | Path,
    model_paths,
    *,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    metadata: dict[str, str] | None = None,
) -> dict[str, int | str]:
    """Rewrite an explicit IndexTTS2 graph set to one deduplicated external blob."""
    folder = Path(folder).expanduser().resolve()
    targets = _resolve_bundle_targets(folder, model_paths)
    shared_metadata = {
        "index_tts_shared_initializers": "1",
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

    with tempfile.TemporaryDirectory(dir=folder, prefix=".indextts2_shared_") as temp_name:
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


def build_package_metadata(
    metadata: dict[str, str],
    bundle_stats: dict[str, int | str],
    audit: dict[str, int],
    *,
    runtime_graph_count: int,
) -> dict[str, str]:
    """Add deployment-oriented graph and shared-storage facts to a manifest."""
    enriched = {str(key): str(value) for key, value in metadata.items()}
    enriched.update({
        "package_graph_count": str(audit["graph_count"]),
        "runtime_graph_count": str(runtime_graph_count),
        "package_external_reference_count": str(audit["external_references"]),
        "shared_initializer_count": str(bundle_stats["unique_initializers"]),
        "shared_initializer_reference_count": str(
            bundle_stats["initializer_references"]
        ),
        "shared_initializer_source_bytes": str(bundle_stats["source_bytes"]),
        "shared_initializer_unique_bytes": str(bundle_stats["unique_bytes"]),
        "shared_initializer_deduplicated_bytes": str(
            bundle_stats["deduplicated_bytes"]
        ),
        "shared_initializer_data_bytes": str(audit["external_bytes"]),
    })
    return enriched


def static_package_metadata(metadata: dict[str, str]) -> dict[str, str]:
    """Drop bundle-specific facts before creating a derived package."""
    return {
        str(key): str(value)
        for key, value in metadata.items()
        if key not in PACKAGE_STAT_METADATA_KEYS
    }


def _prefixed_component(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    """Prefix graph-local names while preserving public I/O names."""
    model = copy.deepcopy(model)
    public_names = {value.name for value in model.graph.input}
    public_names.update(value.name for value in model.graph.output)
    remap: dict[str, str] = {}

    for tensor in model.graph.initializer:
        if not tensor.name.startswith(_CANONICAL_PREFIX):
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
    dst.graph.initializer.extend(initializers.values())
    return set(initializers)


def _delete_model_artifacts(*model_paths: Path) -> None:
    for model_path in model_paths:
        model_path.unlink(missing_ok=True)
        model_path.with_name(model_path.name + ".data").unlink(missing_ok=True)


def _value_signature(value: onnx.ValueInfoProto) -> tuple[int, int]:
    tensor_type = value.type.tensor_type
    return tensor_type.elem_type, len(tensor_type.shape.dim)


def merge_reference_preprocess_graph(
    feature_path: str | Path,
    semantic_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Connect raw-audio features, Wav2Vec2-BERT, and reference conditioning."""
    feature_path = Path(feature_path)
    semantic_path = Path(semantic_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    feature = _prefixed_component(
        onnx.load(str(feature_path), load_external_data=False),
        "feature/",
    )
    semantic = _prefixed_component(
        onnx.load(str(semantic_path), load_external_data=False),
        "semantic/",
    )
    reference = _prefixed_component(
        onnx.load(str(reference_path), load_external_data=False),
        "reference/",
    )

    feature_outputs = {value.name: value for value in feature.graph.output}
    semantic_inputs = {value.name: value for value in semantic.graph.input}
    semantic_outputs = {value.name: value for value in semantic.graph.output}
    reference_inputs = {value.name: value for value in reference.graph.input}
    connectors = {
        "input_features": (feature_outputs, semantic_inputs),
        "attention_mask": (feature_outputs, semantic_inputs),
        "semantic_features": (semantic_outputs, reference_inputs),
        "reference_mel": (feature_outputs, reference_inputs),
        "style_features": (feature_outputs, reference_inputs),
    }
    reference_outputs = {value.name: value for value in reference.graph.output}
    final_output_names = (
        "semantic_features",
        "style",
        "reference_hidden",
        "null_hidden",
    )
    merged = onnx.ModelProto()
    merged.ir_version = max(
        feature.ir_version,
        semantic.ir_version,
        reference.ir_version,
    )
    merged.producer_name = Path(__file__).name
    merged.graph.name = "IndexTTS2_ReferencePreprocess"
    _merge_opset_imports(merged, feature, semantic, reference)
    merged.graph.input.extend(feature.graph.input)
    initializer_names = _merge_initializers(
        merged,
        feature,
        semantic,
        reference,
    )
    merged.graph.node.extend(feature.graph.node)
    merged.graph.node.extend(semantic.graph.node)
    merged.graph.node.extend(reference.graph.node)

    seen_values = {"audio", *initializer_names}
    for value in (
        list(feature.graph.value_info)
        + list(semantic.graph.value_info)
        + list(reference.graph.value_info)
        + [producers[name] for name, (producers, _) in connectors.items()]
    ):
        if value.name not in seen_values:
            merged.graph.value_info.append(value)
            seen_values.add(value.name)
    merged.graph.output.append(semantic_outputs["semantic_features"])
    merged.graph.output.extend(
        reference_outputs[name] for name in final_output_names[1:]
    )
    _copy_metadata(merged, feature, semantic, reference)
    return output_path


def build_reference_preprocess_graph(
    folder: str | Path,
    *,
    delete_components: bool = True,
) -> Path:
    """Build the v1.5-style raw-audio ReferencePreprocess graph."""
    folder = Path(folder)
    feature_path = folder / "IndexTTS2_FeatureExtractor.onnx"
    semantic_path = folder / "IndexTTS2_SemanticEncoder.onnx"
    reference_path = folder / "IndexTTS2_Reference.onnx"
    output_path = folder / "IndexTTS2_ReferencePreprocess.onnx"
    result = merge_reference_preprocess_graph(
        feature_path,
        semantic_path,
        reference_path,
        output_path,
    )
    if delete_components:
        _delete_model_artifacts(feature_path, semantic_path, reference_path)
    return result


def merge_target_prefill_graph(
    target_path: str | Path,
    prefill_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Connect TargetPreprocess.hidden_states to one MainPrefill graph."""
    target_path = Path(target_path)
    prefill_path = Path(prefill_path)
    output_path = Path(output_path)
    target = _prefixed_component(
        onnx.load(str(target_path), load_external_data=False),
        "target/",
    )
    prefill = _prefixed_component(
        onnx.load(str(prefill_path), load_external_data=False),
        "prefill/",
    )

    target_outputs = {value.name: value for value in target.graph.output}
    prefill_inputs = {value.name: value for value in prefill.graph.input}
    connector = "hidden_states"
    merged = onnx.ModelProto()
    merged.ir_version = max(target.ir_version, prefill.ir_version)
    merged.producer_name = Path(__file__).name
    merged.graph.name = f"{target.graph.name}_{prefill.graph.name}_target_prefill"
    _merge_opset_imports(merged, target, prefill)
    merged.graph.input.extend(target.graph.input)
    merged.graph.input.extend(
        value for value in prefill.graph.input if value.name != connector
    )
    initializer_names = _merge_initializers(merged, target, prefill)
    merged.graph.node.extend(target.graph.node)
    merged.graph.node.extend(prefill.graph.node)

    seen_values = {
        *(value.name for value in merged.graph.input),
        *initializer_names,
        connector,
    }
    for value in list(target.graph.value_info) + list(prefill.graph.value_info):
        if value.name not in seen_values:
            merged.graph.value_info.append(value)
            seen_values.add(value.name)
    merged.graph.value_info.append(target_outputs[connector])
    merged.graph.output.extend(prefill.graph.output)
    _copy_metadata(merged, target, prefill)
    return output_path


def build_target_prefill_graphs(
    folder: str | Path,
    strategies: Iterable[str] = DEFAULT_DECODE_STRATEGIES,
    *,
    delete_components: bool = True,
) -> dict[str, Path]:
    """Build merged TargetPrefill strategy graphs."""
    folder = Path(folder)
    target_path = folder / "IndexTTS2_TargetPreprocess.onnx"
    outputs: dict[str, Path] = {}
    prefill_paths = []
    for strategy in strategies:
        prefill_path = folder / f"IndexTTS2_MainPrefill_{strategy}.onnx"
        output_path = folder / f"IndexTTS2_TargetPrefill_{strategy}.onnx"
        outputs[strategy] = merge_target_prefill_graph(
            target_path,
            prefill_path,
            output_path,
        )
        prefill_paths.append(prefill_path)
    if delete_components:
        _delete_model_artifacts(target_path, *prefill_paths)
    return outputs


def merge_synthesis_graph(
    latent_path: str | Path,
    acoustic_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Slice accepted codes and connect Latent directly to Acoustic."""
    latent_path = Path(latent_path)
    acoustic_path = Path(acoustic_path)
    output_path = Path(output_path)
    latent = _prefixed_component(
        onnx.load(str(latent_path), load_external_data=False),
        "latent/",
    )
    acoustic = _prefixed_component(
        onnx.load(str(acoustic_path), load_external_data=False),
        "acoustic/",
    )

    latent_inputs = {value.name: value for value in latent.graph.input}
    latent_outputs = {value.name: value for value in latent.graph.output}
    acoustic_inputs = {value.name: value for value in acoustic.graph.input}
    merged = onnx.ModelProto()
    merged.ir_version = max(latent.ir_version, acoustic.ir_version)
    merged.producer_name = Path(__file__).name
    merged.graph.name = "IndexTTS2_Synthesis"
    _merge_opset_imports(merged, latent, acoustic)
    merged.graph.input.extend(
        value for value in latent.graph.input if value.name != "mel_codes"
    )
    save_ids = copy.deepcopy(latent_inputs["mel_codes"])
    save_ids.name = "save_ids"
    merged.graph.input.append(save_ids)
    merged.graph.input.append(
        helper.make_tensor_value_info("accepted_length", TensorProto.INT64, [1])
    )
    merged.graph.input.extend(
        value
        for value in acoustic.graph.input
        if value.name not in {"mel_codes", "gpt_latent"}
    )
    initializer_names = _merge_initializers(merged, latent, acoustic)
    slice_initializers = (
        helper.make_tensor("synthesis/slice_starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor("synthesis/slice_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("synthesis/slice_steps", TensorProto.INT64, [1], [1]),
    )
    merged.graph.initializer.extend(slice_initializers)
    initializer_names.update(tensor.name for tensor in slice_initializers)
    merged.graph.node.append(
        helper.make_node(
            "Slice",
            [
                "save_ids",
                "synthesis/slice_starts",
                "accepted_length",
                "synthesis/slice_axes",
                "synthesis/slice_steps",
            ],
            ["mel_codes"],
            name="synthesis/SliceAcceptedCodes",
        )
    )
    merged.graph.node.extend(latent.graph.node)
    merged.graph.node.extend(acoustic.graph.node)

    seen_values = {
        *(value.name for value in merged.graph.input),
        *initializer_names,
        "mel_codes",
        "gpt_latent",
    }
    for value in list(latent.graph.value_info) + list(acoustic.graph.value_info):
        if value.name not in seen_values:
            merged.graph.value_info.append(value)
            seen_values.add(value.name)
    merged.graph.value_info.append(latent_inputs["mel_codes"])
    merged.graph.value_info.append(latent_outputs["gpt_latent"])
    merged.graph.output.extend(acoustic.graph.output)
    _copy_metadata(merged, latent, acoustic)
    return output_path


def build_synthesis_graph(
    folder: str | Path,
    *,
    delete_components: bool = True,
) -> Path:
    """Build the merged accepted-code -> CFM-conditioning graph."""
    folder = Path(folder)
    latent_path = folder / "IndexTTS2_Latent.onnx"
    acoustic_path = folder / "IndexTTS2_Acoustic.onnx"
    output_path = folder / "IndexTTS2_Synthesis.onnx"
    result = merge_synthesis_graph(latent_path, acoustic_path, output_path)
    if delete_components:
        _delete_model_artifacts(latent_path, acoustic_path)
    return result


def build_decoder_postprocess_graph(decoder_path: str | Path) -> Path:
    """Make Decoder consume time-major Euler state and slice target frames."""
    decoder_path = Path(decoder_path)
    model = onnx.load(str(decoder_path), load_external_data=False)
    inputs = {value.name: value for value in model.graph.input}
    mel = inputs["mel"]
    tensor_type = mel.type.tensor_type
    mel_bin_dim = tensor_type.shape.dim[1]
    mel_bins: int | str = (
        int(mel_bin_dim.dim_value)
        if mel_bin_dim.HasField("dim_value")
        else "mel_bins"
    )

    del model.graph.input[:]
    model.graph.input.append(
        helper.make_tensor_value_info(
            "mel_features",
            TensorProto.FLOAT,
            [1, "total_mel_frames", mel_bins],
        )
    )
    model.graph.input.append(
        helper.make_tensor_value_info("target_length", TensorProto.INT32, [1])
    )
    model.graph.value_info.append(mel)
    slice_initializers = (
        helper.make_tensor(
            "decoder/slice_ends",
            TensorProto.INT64,
            [1],
            [np.iinfo(np.int64).max],
        ),
        helper.make_tensor("decoder/slice_axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("decoder/slice_steps", TensorProto.INT64, [1], [1]),
    )
    model.graph.initializer.extend(slice_initializers)
    preprocess_nodes = [
        helper.make_node(
            "Cast",
            ["target_length"],
            ["decoder/target_length_i64"],
            name="decoder/CastTargetLength",
            to=TensorProto.INT64,
        ),
        helper.make_node(
            "Neg",
            ["decoder/target_length_i64"],
            ["decoder/slice_starts"],
            name="decoder/NegTargetLength",
        ),
        helper.make_node(
            "Slice",
            [
                "mel_features",
                "decoder/slice_starts",
                "decoder/slice_ends",
                "decoder/slice_axes",
                "decoder/slice_steps",
            ],
            ["decoder/target_features"],
            name="decoder/SliceTargetMel",
        ),
        helper.make_node(
            "Transpose",
            ["decoder/target_features"],
            ["mel"],
            name="decoder/TransposeTargetMel",
            perm=[0, 2, 1],
        ),
    ]
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(preprocess_nodes)
    model.graph.node.extend(original_nodes)
    model.graph.name = "IndexTTS2_DecoderPostprocess"
    return decoder_path


def merge_decode_step_graph(
    embed_path: str | Path,
    main_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Connect IndexTTS2 DecodeEmbed.hidden_states to one MainDecode graph."""
    embed_path = Path(embed_path)
    main_path = Path(main_path)
    output_path = Path(output_path)
    embed = _prefixed_component(
        onnx.load(str(embed_path), load_external_data=False),
        "embed/",
    )
    main = _prefixed_component(
        onnx.load(str(main_path), load_external_data=False),
        "main/",
    )

    embed_outputs = {value.name: value for value in embed.graph.output}
    main_inputs = {value.name: value for value in main.graph.input}
    connector = "hidden_states"
    common_inputs = set(main_inputs).intersection(value.name for value in embed.graph.input)
    embed_input_map = {value.name: value for value in embed.graph.input}
    merged = onnx.ModelProto()
    merged.ir_version = max(embed.ir_version, main.ir_version)
    merged.producer_name = Path(__file__).name
    merged.graph.name = f"{embed.graph.name}_{main.graph.name}_decode_step"
    _merge_opset_imports(merged, embed, main)

    main_input_list = list(main.graph.input)
    connector_index = next(
        index for index, value in enumerate(main_input_list) if value.name == connector
    )
    ordered_inputs = main_input_list[:connector_index]
    ordered_inputs.extend(
        value for value in embed.graph.input if value.name not in main_inputs
    )
    ordered_inputs.extend(main_input_list[connector_index + 1:])
    seen_inputs: set[str] = set()
    for value in ordered_inputs:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)

    initializer_names = _merge_initializers(merged, embed, main)
    merged.graph.node.extend(embed.graph.node)
    merged.graph.node.extend(main.graph.node)

    seen_values = seen_inputs | initializer_names | {connector}
    for value in list(embed.graph.value_info) + list(main.graph.value_info):
        if value.name not in seen_values:
            merged.graph.value_info.append(value)
            seen_values.add(value.name)
    merged.graph.value_info.append(embed_outputs[connector])
    merged.graph.output.extend(main.graph.output)
    _copy_metadata(merged, embed, main)
    return output_path


def build_decode_step_graphs(
    folder: str | Path,
    strategies: Iterable[str] = DEFAULT_DECODE_STRATEGIES,
    *,
    delete_components: bool = True,
) -> dict[str, Path]:
    """Build merged IndexTTS2 DecodeStep graphs from the shared embed component."""
    folder = Path(folder)
    embed_path = folder / "IndexTTS2_DecodeEmbed.onnx"
    outputs: dict[str, Path] = {}
    main_paths = []
    for strategy in strategies:
        main_path = folder / f"IndexTTS2_MainDecode_{strategy}.onnx"
        output_path = folder / f"IndexTTS2_DecodeStep_{strategy}.onnx"
        outputs[strategy] = merge_decode_step_graph(embed_path, main_path, output_path)
        main_paths.append(main_path)
    if delete_components:
        _delete_model_artifacts(embed_path, *main_paths)
    return outputs


def attach_shared_initializers(session_options, shared_model_path: str | Path):
    """Mmap the shared blob and register its OrtValues before sessions are built."""
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
