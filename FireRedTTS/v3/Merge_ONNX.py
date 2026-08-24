"""Merge FireRedTTS3 raw decode components into hot DecodeStep ONNX graphs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper

from Shared_Weights import (
    PackageContractError,
    bundle_shared_initializers,
    bundle_shared_initializers_from_source,
    inference_metadata,
    prefix_graph_local_names,
    promote_directory,
    read_onnx_metadata,
    replace_onnx_metadata,
    validate_package_contract,
    write_metadata_carrier,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PACKAGE_FOLDER = SCRIPT_DIR / "FireRedTTS3_Base_ONNX"
METADATA_FILE_NAME = "FireRedTTS3_Metadata.onnx"
BRIDGE_NAME = "last_hidden_state"
BASE_REFERENCE_BRIDGES = ("prompt_latents", "speaker_embedding")
INSTRUCT_LATENT_INPUTS = ("latents_in", "latents_out")
FLOAT_ATOL = 2.0e-5
FLOAT_RTOL = 2.0e-5


class MergeContractError(RuntimeError):
    """Raised when raw FireRedTTS3 graph components cannot be safely merged."""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-folder",
        type=Path,
        default=DEFAULT_PACKAGE_FOLDER,
        help="Raw FireRedTTS3 package folder created by Export_FireRedTTS3.py.",
    )
    return parser.parse_args()


def _tensor_bytes(tensor: TensorProto) -> bytes:
    if tensor.raw_data:
        return tensor.raw_data
    return numpy_helper.to_array(tensor).tobytes(order="C")


def _external_identity(tensor: TensorProto) -> tuple[object, ...] | None:
    if tensor.data_location != TensorProto.EXTERNAL:
        return None
    attributes = {item.key: item.value for item in tensor.external_data}
    return (
        "external",
        int(tensor.data_type),
        tuple(int(dimension) for dimension in tensor.dims),
        attributes.get("location"),
        attributes.get("offset", "0"),
        attributes.get("length"),
    )


def _set_metadata(model: onnx.ModelProto, metadata: Mapping[str, str]) -> None:
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))


def _merge_opsets(destination: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    versions: dict[str, int] = {}
    for source in sources:
        for opset in source.opset_import:
            versions[opset.domain] = max(versions.get(opset.domain, 0), int(opset.version))
    for domain, version in sorted(versions.items()):
        destination.opset_import.append(helper.make_opsetid(domain, version))


def _deduplicate_initializers(model: onnx.ModelProto) -> int:
    """Deduplicate only tensors with matching dtype, shape, length, and SHA-256."""
    unique: dict[tuple[object, ...], TensorProto] = {}
    remap: dict[str, str] = {}
    retained: list[TensorProto] = []
    duplicates = 0
    for initializer in model.graph.initializer:
        identity = _external_identity(initializer)
        if identity is None:
            payload = _tensor_bytes(initializer)
            identity = (
                "inline",
                int(initializer.data_type),
                tuple(int(dimension) for dimension in initializer.dims),
                len(payload),
                hashlib.sha256(payload).digest(),
            )
        canonical = unique.get(identity)
        if canonical is None:
            unique[identity] = initializer
            retained.append(initializer)
            continue
        remap[initializer.name] = canonical.name
        duplicates += 1

    if remap:
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                node.input[index] = remap.get(name, name)
        del model.graph.initializer[:]
        model.graph.initializer.extend(retained)
    return duplicates


def merge_decode_components(
    core_path: Path,
    flow_path: Path,
    destination_path: Path,
    metadata: Mapping[str, str],
) -> Path:
    """Compose one raw cached-decode core and flow graph across their hidden bridge."""
    core = prefix_graph_local_names(
        onnx.load(str(core_path), load_external_data=False),
        "core/",
        preserved_names=(BRIDGE_NAME,),
    )
    flow = prefix_graph_local_names(
        onnx.load(str(flow_path), load_external_data=False),
        "flow/",
        preserved_names=(BRIDGE_NAME,),
    )
    core_outputs = {value.name for value in core.graph.output}
    flow_inputs = {value.name for value in flow.graph.input}
    if BRIDGE_NAME not in core_outputs or BRIDGE_NAME not in flow_inputs:
        raise MergeContractError(
            f"Raw components must expose {BRIDGE_NAME!r} as core output and flow input."
        )

    merged = onnx.ModelProto()
    merged.ir_version = max(core.ir_version, flow.ir_version)
    merged.producer_name = Path(__file__).name
    merged.graph.name = f"{core.graph.name}_{flow.graph.name}_decode_step"
    _merge_opsets(merged, core, flow)
    _set_metadata(merged, metadata)

    seen_inputs: set[str] = set()
    for value in (*core.graph.input, *flow.graph.input):
        if value.name == BRIDGE_NAME or value.name in seen_inputs:
            continue
        merged.graph.input.append(copy.deepcopy(value))
        seen_inputs.add(value.name)
    merged.graph.initializer.extend(copy.deepcopy(tensor) for tensor in core.graph.initializer)
    merged.graph.initializer.extend(copy.deepcopy(tensor) for tensor in flow.graph.initializer)
    merged.graph.node.extend(copy.deepcopy(node) for node in core.graph.node)
    merged.graph.node.extend(copy.deepcopy(node) for node in flow.graph.node)

    seen_values = seen_inputs | {tensor.name for tensor in merged.graph.initializer}
    for value in (*core.graph.value_info, *flow.graph.value_info):
        if value.name and value.name not in seen_values:
            merged.graph.value_info.append(copy.deepcopy(value))
            seen_values.add(value.name)
    seen_outputs: set[str] = set()
    for value in (*core.graph.output, *flow.graph.output):
        if value.name and value.name not in seen_outputs:
            merged.graph.output.append(copy.deepcopy(value))
            seen_outputs.add(value.name)

    duplicates = _deduplicate_initializers(merged)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(merged, str(destination_path))
    print(
        f"[Merge] {core_path.name} + {flow_path.name} -> {destination_path.name}; "
        f"byte-identical initializers removed={duplicates}."
    )
    return destination_path


def merge_base_reference_prefill(
    reference_path: Path,
    prefill_path: Path,
    destination_path: Path,
    metadata: Mapping[str, str],
) -> Path:
    """Compose Base waveform preprocessing directly into cached LLM prefill."""
    reference = prefix_graph_local_names(
        onnx.load(str(reference_path), load_external_data=False),
        "reference/",
        preserved_names=BASE_REFERENCE_BRIDGES,
    )
    prefill = prefix_graph_local_names(
        onnx.load(str(prefill_path), load_external_data=False),
        "prefill/",
        preserved_names=BASE_REFERENCE_BRIDGES,
    )
    reference_outputs = {value.name for value in reference.graph.output}
    prefill_inputs = {value.name for value in prefill.graph.input}
    for bridge in BASE_REFERENCE_BRIDGES:
        if bridge not in reference_outputs or bridge not in prefill_inputs:
            raise MergeContractError(
                f"Base reference/prefill components must share bridge {bridge!r}."
            )

    merged = onnx.ModelProto()
    merged.ir_version = max(reference.ir_version, prefill.ir_version)
    merged.producer_name = Path(__file__).name
    merged.graph.name = "fireredtts3_base_reference_prefill"
    _merge_opsets(merged, reference, prefill)
    _set_metadata(merged, metadata)

    for value in prefill.graph.input:
        if value.name not in BASE_REFERENCE_BRIDGES:
            merged.graph.input.append(copy.deepcopy(value))
    merged.graph.input.extend(copy.deepcopy(value) for value in reference.graph.input)
    merged.graph.initializer.extend(
        copy.deepcopy(tensor) for tensor in reference.graph.initializer
    )
    merged.graph.initializer.extend(
        copy.deepcopy(tensor) for tensor in prefill.graph.initializer
    )
    merged.graph.node.extend(copy.deepcopy(node) for node in reference.graph.node)
    merged.graph.node.extend(copy.deepcopy(node) for node in prefill.graph.node)

    seen_values = {
        value.name for value in merged.graph.input
    } | {tensor.name for tensor in merged.graph.initializer}
    for value in (*reference.graph.value_info, *prefill.graph.value_info):
        if value.name and value.name not in seen_values:
            merged.graph.value_info.append(copy.deepcopy(value))
            seen_values.add(value.name)

    merged.graph.output.extend(copy.deepcopy(value) for value in prefill.graph.output)
    prompt_output = next(
        value for value in reference.graph.output if value.name == "prompt_latents"
    )
    merged.graph.output.append(copy.deepcopy(prompt_output))

    duplicates = _deduplicate_initializers(merged)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(merged, str(destination_path))
    onnx.checker.check_model(str(destination_path))
    print(
        f"[Merge] {reference_path.name} + {prefill_path.name} -> "
        f"{destination_path.name}; byte-identical initializers removed={duplicates}."
    )
    return destination_path


def merge_instruct_audio_prefill(
    encode_path: Path,
    prefill_path: Path,
    destination_path: Path,
    metadata: Mapping[str, str],
    *,
    audio_role: str,
) -> Path:
    """Compose Instruct RedAE encode into input- or output-audio prefill."""
    if audio_role not in {"input", "output"}:
        raise ValueError(f"Unsupported Instruct audio role: {audio_role!r}.")
    encode = prefix_graph_local_names(
        onnx.load(str(encode_path), load_external_data=False),
        "encode/",
        preserved_names=("prompt_latents",),
    )
    prefill = prefix_graph_local_names(
        onnx.load(str(prefill_path), load_external_data=False),
        "prefill/",
        preserved_names=INSTRUCT_LATENT_INPUTS,
    )
    if "prompt_latents" not in {value.name for value in encode.graph.output}:
        raise MergeContractError("Instruct RedAE encode must output 'prompt_latents'.")
    prefill_inputs = {value.name: value for value in prefill.graph.input}
    if any(name not in prefill_inputs for name in INSTRUCT_LATENT_INPUTS):
        raise MergeContractError(
            "Instruct prefill must expose latents_in and latents_out inputs."
        )

    active_input = "latents_out" if audio_role == "output" else "latents_in"
    inactive_input = "latents_in" if audio_role == "output" else "latents_out"
    inactive_type = prefill_inputs[inactive_input].type.tensor_type
    inactive_shape = inactive_type.shape.dim
    if (
        len(inactive_shape) != 3
        or not inactive_shape[0].HasField("dim_value")
        or inactive_shape[0].dim_value != 1
        or not inactive_shape[2].HasField("dim_value")
        or inactive_shape[2].dim_value <= 0
    ):
        raise MergeContractError(
            f"Instruct prefill {inactive_input} must have shape [1, frames, width]."
        )
    inactive_dimensions = (
        1,
        int(metadata["patch_size"]),
        int(inactive_shape[2].dim_value),
    )
    inactive_latents = helper.make_tensor(
        "instruct_audio_prefill/inactive_latents",
        inactive_type.elem_type,
        inactive_dimensions,
        [0.0] * math.prod(inactive_dimensions),
    )
    remap = {
        active_input: "prompt_latents",
        inactive_input: inactive_latents.name,
    }
    _remap_graph_inputs(prefill.graph, remap)

    merged = onnx.ModelProto()
    merged.ir_version = max(encode.ir_version, prefill.ir_version)
    merged.producer_name = Path(__file__).name
    merged.graph.name = f"fireredtts3_instruct_{audio_role}_audio_prefill"
    _merge_opsets(merged, encode, prefill)
    _set_metadata(merged, metadata)

    ordered_inputs = ["text_ids", "prompt_audio"]
    ordered_inputs.extend(
        value.name
        for value in prefill.graph.input
        if value.name not in INSTRUCT_LATENT_INPUTS and value.name != "text_ids"
    )
    public_values = {
        value.name: value for value in (*prefill.graph.input, *encode.graph.input)
    }
    merged.graph.input.extend(copy.deepcopy(public_values[name]) for name in ordered_inputs)
    merged.graph.initializer.extend(
        copy.deepcopy(tensor) for tensor in encode.graph.initializer
    )
    merged.graph.initializer.extend(
        copy.deepcopy(tensor) for tensor in prefill.graph.initializer
    )
    merged.graph.initializer.append(inactive_latents)
    merged.graph.node.extend(copy.deepcopy(node) for node in encode.graph.node)
    merged.graph.node.extend(copy.deepcopy(node) for node in prefill.graph.node)

    seen_values = {
        value.name for value in merged.graph.input
    } | {tensor.name for tensor in merged.graph.initializer}
    for value in (*encode.graph.value_info, *prefill.graph.value_info):
        if value.name and value.name not in seen_values:
            merged.graph.value_info.append(copy.deepcopy(value))
            seen_values.add(value.name)
    merged.graph.output.extend(copy.deepcopy(value) for value in prefill.graph.output)
    if audio_role == "output":
        prompt_output = next(
            value for value in encode.graph.output if value.name == "prompt_latents"
        )
        merged.graph.output.append(copy.deepcopy(prompt_output))

    duplicates = _deduplicate_initializers(merged)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(merged, str(destination_path))
    onnx.checker.check_model(str(destination_path))
    print(
        f"[Merge] {encode_path.name} + {prefill_path.name} "
        f"({audio_role} audio) -> {destination_path.name}; "
        f"byte-identical initializers removed={duplicates}."
    )
    return destination_path


def _remap_graph_inputs(graph: onnx.GraphProto, remap: Mapping[str, str]) -> None:
    """Remap graph consumers while leaving public input declarations untouched."""
    for node in graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)


def _onnx_dtype(type_name: str) -> np.dtype:
    if not type_name.startswith("tensor(") or not type_name.endswith(")"):
        raise MergeContractError(f"Unsupported ORT input type: {type_name!r}.")
    name = type_name[7:-1].upper()
    enum = onnx.TensorProto.DataType.Value(name)
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(enum))


def _sample_input(argument: onnxruntime.NodeArg, seed: int) -> np.ndarray:
    dtype = _onnx_dtype(argument.type)
    shape: list[int] = []
    for axis, dimension in enumerate(argument.shape):
        if isinstance(dimension, int) and dimension > 0:
            shape.append(dimension)
        elif "key_" in argument.name or "value_" in argument.name:
            shape.append(2 if axis == 2 else 1)
        elif "text" in argument.name or "mask" in argument.name:
            shape.append(4)
        elif "latent" in argument.name or "patch" in argument.name:
            shape.append(4)
        else:
            shape.append(1)
    generator = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.bool_):
        return np.zeros(shape, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        return np.zeros(shape, dtype=dtype)
    return generator.standard_normal(shape, dtype=np.float32).astype(dtype)


def compare_unmerged_and_merged(
    core_path: Path,
    flow_path: Path,
    merged_path: Path,
) -> None:
    """Compare all merged outputs against executing raw core then raw flow."""
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    core_session = onnxruntime.InferenceSession(
        str(core_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    flow_session = onnxruntime.InferenceSession(
        str(flow_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    merged_session = onnxruntime.InferenceSession(
        str(merged_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    global_inputs: dict[str, np.ndarray] = {}
    for index, argument in enumerate((*core_session.get_inputs(), *flow_session.get_inputs())):
        if argument.name != BRIDGE_NAME and argument.name not in global_inputs:
            global_inputs[argument.name] = _sample_input(argument, index + 31)
    core_outputs = dict(
        zip(
            (argument.name for argument in core_session.get_outputs()),
            core_session.run(None, {argument.name: global_inputs[argument.name] for argument in core_session.get_inputs()}),
        )
    )
    flow_feed = {
        argument.name: core_outputs[BRIDGE_NAME]
        if argument.name == BRIDGE_NAME
        else global_inputs[argument.name]
        for argument in flow_session.get_inputs()
    }
    flow_outputs = dict(zip((argument.name for argument in flow_session.get_outputs()), flow_session.run(None, flow_feed)))
    merged_outputs = dict(
        zip(
            (argument.name for argument in merged_session.get_outputs()),
            merged_session.run(
                None,
                {argument.name: global_inputs[argument.name] for argument in merged_session.get_inputs()},
            ),
        )
    )
    expected = {**core_outputs, **flow_outputs}
    for name, actual in merged_outputs.items():
        reference = expected.get(name)
        if reference is None:
            raise MergeContractError(f"Merged graph produced unexpected output {name!r}.")
        if np.issubdtype(actual.dtype, np.floating):
            np.testing.assert_allclose(actual, reference, rtol=FLOAT_RTOL, atol=FLOAT_ATOL, err_msg=name)
        elif not np.array_equal(actual, reference):
            raise MergeContractError(f"Merged non-floating output differs from raw sequence: {name}.")
    print(f"[Merge] ORT parity passed for {len(merged_outputs)} merged output tensors.")


def compare_base_reference_prefill(
    reference_path: Path,
    prefill_path: Path,
    merged_path: Path,
    metadata: Mapping[str, str],
) -> None:
    """Compare Base reference->prefill execution with its composed graph."""
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    reference_session = onnxruntime.InferenceSession(
        str(reference_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    prefill_session = onnxruntime.InferenceSession(
        str(prefill_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    merged_session = onnxruntime.InferenceSession(
        str(merged_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    native_alignment = (
        int(metadata["redae_downsample_rate"])
        * int(metadata["patch_size"])
        * 3
    )
    public_samples = max(
        1,
        math.ceil(
            native_alignment
            * int(metadata["input_audio_sample_rate"])
            / int(metadata.get("model_audio_sample_rate", metadata["input_audio_sample_rate"]))
        ),
    )
    generator = np.random.default_rng(211)
    reference_feed = {
        reference_session.get_inputs()[0].name: (
            generator.standard_normal((1, public_samples), dtype=np.float32) * 0.01
        )
    }
    reference_values = reference_session.run(None, reference_feed)
    reference_outputs = dict(
        zip(
            (argument.name for argument in reference_session.get_outputs()),
            reference_values,
        )
    )
    prefill_feed = {
        argument.name: (
            reference_outputs[argument.name]
            if argument.name in BASE_REFERENCE_BRIDGES
            else np.zeros((1, 8), dtype=_onnx_dtype(argument.type))
        )
        for argument in prefill_session.get_inputs()
    }
    prefill_outputs = dict(
        zip(
            (argument.name for argument in prefill_session.get_outputs()),
            prefill_session.run(None, prefill_feed),
        )
    )
    merged_feed = {
        **reference_feed,
        "text_ids": prefill_feed["text_ids"],
    }
    merged_outputs = dict(
        zip(
            (argument.name for argument in merged_session.get_outputs()),
            merged_session.run(None, merged_feed),
        )
    )
    expected = {**prefill_outputs, "prompt_latents": reference_outputs["prompt_latents"]}
    for name, actual in merged_outputs.items():
        reference_value = expected[name]
        if np.issubdtype(actual.dtype, np.floating):
            np.testing.assert_allclose(
                actual,
                reference_value,
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                err_msg=name,
            )
        elif not np.array_equal(actual, reference_value):
            raise MergeContractError(
                f"Merged Base reference-prefill output differs: {name}."
            )
    print(
        f"[Merge] Base reference-prefill ORT parity passed for "
        f"{len(merged_outputs)} output tensors."
    )


def compare_instruct_audio_prefill(
    encode_path: Path,
    prefill_path: Path,
    merged_path: Path,
    metadata: Mapping[str, str],
    *,
    audio_role: str,
) -> None:
    """Compare Instruct encode->prefill execution with one composed graph."""
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    encode_session = onnxruntime.InferenceSession(
        str(encode_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    prefill_session = onnxruntime.InferenceSession(
        str(prefill_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    merged_session = onnxruntime.InferenceSession(
        str(merged_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    native_alignment = (
        int(metadata["redae_downsample_rate"])
        * int(metadata["patch_size"])
        * 3
    )
    public_samples = max(
        1,
        math.ceil(
            native_alignment
            * int(metadata["input_audio_sample_rate"])
            / int(metadata.get("model_audio_sample_rate", metadata["input_audio_sample_rate"]))
        ),
    )
    generator = np.random.default_rng(307 if audio_role == "input" else 311)
    audio = generator.standard_normal((1, public_samples), dtype=np.float32) * 0.01
    encoded = encode_session.run(None, {encode_session.get_inputs()[0].name: audio})[0]
    patch_count = encoded.shape[1] // int(metadata["patch_size"])
    text = np.zeros((1, max(12, patch_count + 2)), dtype=np.int64)
    placeholder_id = int(
        metadata[
            "latent_out_pad_id" if audio_role == "output" else "latent_in_pad_id"
        ]
    )
    text[:, 1 : 1 + patch_count] = placeholder_id
    inactive = np.zeros(
        (
            1,
            int(metadata["patch_size"]),
            encoded.shape[-1],
        ),
        dtype=encoded.dtype,
    )
    prefill_feed: dict[str, np.ndarray] = {}
    for argument in prefill_session.get_inputs():
        if argument.name == "text_ids":
            value = text
        elif argument.name == "latents_in":
            value = encoded if audio_role == "input" else inactive
        elif argument.name == "latents_out":
            value = encoded if audio_role == "output" else inactive
        elif argument.name == "text_do_sample":
            value = np.asarray([False], dtype=np.bool_)
        elif argument.name == "text_top_k":
            value = np.asarray([20], dtype=np.int64)
        elif argument.name == "text_temperature":
            value = np.asarray([0.7], dtype=np.float32)
        elif argument.name == "text_top_p":
            value = np.asarray([0.8], dtype=np.float32)
        elif argument.name == "text_repetition_penalty":
            value = np.asarray([1.0], dtype=np.float32)
        else:
            raise MergeContractError(
                f"Unexpected Instruct prefill input in parity test: {argument.name}."
            )
        prefill_feed[argument.name] = value
    expected = dict(
        zip(
            (argument.name for argument in prefill_session.get_outputs()),
            prefill_session.run(None, prefill_feed),
        )
    )
    if audio_role == "output":
        expected["prompt_latents"] = encoded

    merged_feed = {
        argument.name: (
            audio
            if argument.name == "prompt_audio"
            else prefill_feed[argument.name]
        )
        for argument in merged_session.get_inputs()
    }
    actual_outputs = dict(
        zip(
            (argument.name for argument in merged_session.get_outputs()),
            merged_session.run(None, merged_feed),
        )
    )
    for name, actual in actual_outputs.items():
        reference_value = expected[name]
        if np.issubdtype(actual.dtype, np.floating):
            np.testing.assert_allclose(
                actual,
                reference_value,
                rtol=FLOAT_RTOL,
                atol=FLOAT_ATOL,
                err_msg=name,
            )
        elif not np.array_equal(actual, reference_value):
            raise MergeContractError(
                f"Merged Instruct {audio_role}-audio prefill differs: {name}."
            )
    print(
        f"[Merge] Instruct {audio_role}-audio prefill ORT parity passed for "
        f"{len(actual_outputs)} output tensors."
    )


def internalize_flow_noise(model_path: Path) -> None:
    """Replace the merged graph's explicit flow-noise input with ONNX randomness."""
    model = onnx.load(str(model_path), load_external_data=False)
    inputs = {value.name: value for value in model.graph.input}
    if "flow_noise" not in inputs:
        raise MergeContractError(
            f"Merged graph {model_path.name} has no flow_noise input to internalize."
        )
    if "prior_patch" not in inputs:
        raise MergeContractError(
            f"Merged graph {model_path.name} has no prior_patch shape template."
        )
    retained_inputs = [
        value for value in model.graph.input if value.name != "flow_noise"
    ]
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)
    random_node = helper.make_node(
        "RandomNormalLike",
        ["prior_patch"],
        ["flow_noise"],
        name="runtime/flow_noise",
        mean=0.0,
        scale=1.0,
    )
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.append(random_node)
    model.graph.node.extend(original_nodes)
    onnx.save(model, str(model_path))
    onnx.checker.check_model(str(model_path))
    print(f"[Merge] Internalized flow_noise in {model_path.name}.")


def smoke_internalized_decode_step(model_path: Path) -> None:
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if any(argument.name == "flow_noise" for argument in session.get_inputs()):
        raise MergeContractError(
            f"Final merged graph {model_path.name} still exposes flow_noise."
        )
    feeds = {
        argument.name: _sample_input(argument, index + 101)
        for index, argument in enumerate(session.get_inputs())
    }
    outputs = session.run(None, feeds)
    if len(outputs) != len(session.get_outputs()):
        raise MergeContractError(
            f"Final merged graph {model_path.name} returned an incomplete output set."
        )
    for argument, value in zip(session.get_outputs(), outputs):
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise MergeContractError(
                f"Final merged graph produced non-finite output {argument.name!r}."
            )
    print(
        f"[Merge] Internalized-noise ORT smoke passed for {len(outputs)} output tensors."
    )


def _merge_spec(variant: str) -> tuple[str, str, str, str]:
    if variant == "base":
        return (
            "model_file_name_base_decode_core",
            "model_file_name_base_flow_patch",
            "model_file_name_base_decode_step",
            "FireRedTTS3_BaseDecodeStep.onnx",
        )
    if variant == "instruct":
        return (
            "model_file_name_instruct_audio_decode_core",
            "model_file_name_instruct_audio_flow_patch",
            "model_file_name_instruct_audio_decode_step",
            "FireRedTTS3_InstructAudioDecodeStep.onnx",
        )
    raise MergeContractError(f"Unsupported package model_variant: {variant!r}.")


def _rewrite_metadata(package_folder: Path, metadata: dict[str, str]) -> None:
    for path in package_folder.glob("*.onnx"):
        replace_onnx_metadata(path, metadata)


def _link_or_copy(source: str, destination: str) -> str:
    """Hard-link immutable package artifacts when staging stays on one filesystem."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _external_locations(model_paths: list[Path]) -> set[str]:
    locations: set[str] = set()
    for path in model_paths:
        model = onnx.load(str(path), load_external_data=False)
        for tensor in model.graph.initializer:
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            location = {item.key: item.value for item in tensor.external_data}.get("location")
            if location:
                locations.add(location)
    return locations


def _prune_original_external_data(folder: Path, locations: set[str]) -> int:
    """Remove only prebundle generated sidecars that no graph can still reference."""
    referenced = _external_locations(sorted(folder.glob("*.onnx")))
    removed = 0
    for location in locations - referenced:
        candidate = (folder / location).resolve()
        if candidate.parent != folder.resolve() or not candidate.is_file():
            continue
        candidate.unlink()
        removed += 1
    return removed


def merge_package(package_folder: Path) -> Path:
    source = package_folder.expanduser().resolve()
    metadata_path = source / METADATA_FILE_NAME
    metadata = inference_metadata(read_onnx_metadata(metadata_path))
    if metadata.get("graph_layout") != "raw_prefill_decode_core_flow":
        raise MergeContractError(
            f"{source} is not a raw export package; graph_layout={metadata.get('graph_layout')!r}."
        )
    core_key, flow_key, final_key, final_name = _merge_spec(metadata.get("model_variant", ""))
    for key in (core_key, flow_key):
        if key not in metadata:
            raise MergeContractError(f"Raw package metadata is missing {key!r}.")
    staging = source.with_name(source.name + ".merge.staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        bundle_shared_initializers_from_source(source, staging, metadata=metadata)
        core_path = staging / metadata[core_key]
        flow_path = staging / metadata[flow_key]
        merged_path = staging / final_name
        merge_decode_components(core_path, flow_path, merged_path, metadata)
        compare_unmerged_and_merged(
            source / metadata[core_key],
            source / metadata[flow_key],
            merged_path,
        )
        internalize_flow_noise(merged_path)
        smoke_internalized_decode_step(merged_path)

        consumed_component_paths: list[Path] = []
        base_final_key: str | None = None
        base_final_name: str | None = None
        instruct_final_graphs: dict[str, str] = {}
        if metadata.get("model_variant") == "base":
            reference_key = "model_file_name_base_reference_preprocess"
            prefill_key = "model_file_name_base_input_prefill"
            for key in (reference_key, prefill_key):
                if key not in metadata:
                    raise MergeContractError(f"Raw Base package metadata is missing {key!r}.")
            reference_path = staging / metadata[reference_key]
            prefill_path = staging / metadata[prefill_key]
            base_final_key = "model_file_name_base_reference_prefill"
            base_final_name = "FireRedTTS3_BaseReferencePrefill.onnx"
            merged_prefill_path = staging / base_final_name
            merge_base_reference_prefill(
                reference_path,
                prefill_path,
                merged_prefill_path,
                metadata,
            )
            compare_base_reference_prefill(
                source / metadata[reference_key],
                source / metadata[prefill_key],
                merged_prefill_path,
                metadata,
            )
            consumed_component_paths.extend((reference_path, prefill_path))
        else:
            encode_key = "model_file_name_redae_encode"
            prefill_key = "model_file_name_instruct_input_prefill"
            for key in (encode_key, prefill_key):
                if key not in metadata:
                    raise MergeContractError(
                        f"Raw Instruct package metadata is missing {key!r}."
                    )
            encode_path = staging / metadata[encode_key]
            prefill_path = staging / metadata[prefill_key]
            for audio_role in ("input", "output"):
                final_role = f"instruct_{audio_role}_audio_prefill"
                final_graph_key = f"model_file_name_{final_role}"
                final_graph_name = (
                    f"FireRedTTS3_Instruct{audio_role.capitalize()}AudioPrefill.onnx"
                )
                final_graph_path = staging / final_graph_name
                merge_instruct_audio_prefill(
                    encode_path,
                    prefill_path,
                    final_graph_path,
                    metadata,
                    audio_role=audio_role,
                )
                compare_instruct_audio_prefill(
                    source / metadata[encode_key],
                    source / metadata[prefill_key],
                    final_graph_path,
                    metadata,
                    audio_role=audio_role,
                )
                instruct_final_graphs[final_graph_key] = final_graph_name
            consumed_component_paths.append(encode_path)

        for path in (core_path, flow_path, *consumed_component_paths):
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".data").unlink(missing_ok=True)
        metadata = dict(metadata)
        metadata["graph_layout"] = "merged_decode_step"
        del metadata[core_key]
        del metadata[flow_key]
        metadata[final_key] = final_name
        if base_final_key is not None and base_final_name is not None:
            del metadata["model_file_name_base_reference_preprocess"]
            del metadata["model_file_name_base_input_prefill"]
            metadata[base_final_key] = base_final_name
        if instruct_final_graphs:
            del metadata["model_file_name_redae_encode"]
            metadata.update(instruct_final_graphs)
        _rewrite_metadata(staging, metadata)
        write_metadata_carrier(staging / METADATA_FILE_NAME, metadata)
        final_required_keys = [final_key]
        if base_final_key is not None:
            final_required_keys.append(base_final_key)
        final_required_keys.extend(instruct_final_graphs)
        validate_package_contract(
            staging,
            METADATA_FILE_NAME,
            required_keys=(
                "package_schema_version",
                "graph_layout",
                "model_variant",
                "shared_initializer_model_file",
                "shared_initializer_data_file",
                *final_required_keys,
            ),
            require_shared_bundle=True,
        )
        promote_directory(staging, source)
        return source
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    arguments = parse_arguments()
    merged = merge_package(arguments.package_folder)
    print(f"Merged FireRedTTS3 package promoted to: {merged}")


if __name__ == "__main__":
    main()