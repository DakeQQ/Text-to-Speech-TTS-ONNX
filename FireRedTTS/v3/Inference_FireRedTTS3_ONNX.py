"""Run a merged FireRedTTS3 ONNX Runtime package without source-model forwards."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import onnx
import onnxruntime
import soundfile as sound_file
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer

from Shared_Weights import (
    INFERENCE_METADATA_KEYS,
    attach_shared_initializers,
    inference_metadata,
    validate_package_contract,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Example_Audio import model_reference  # noqa: E402


DOWNLOADS_DIR = SCRIPT_DIR.parents[3]

# ---------------------------------------------------------------------------
# Editable runtime settings. The CLI is intentionally limited to package path.
# ---------------------------------------------------------------------------
DEFAULT_INSTRUCT_RUN_MODES = (
    "instruct_clone",
    "voice_design",
    "semantic_edit",
    "acoustic_edit",
)
RUN_MODE = os.environ.get("FIREREDTTS3_RUN_MODE", "")
RUN_MODES = tuple(
    dict.fromkeys(
        mode.strip()
        for mode in os.environ.get(
            "FIREREDTTS3_RUN_MODES",
            RUN_MODE or ",".join(DEFAULT_INSTRUCT_RUN_MODES),
        ).split(",")
        if mode.strip()
    )
)
EXAMPLE_AUDIO_PATH, PROMPT_TEXT = model_reference("fireredtts3")
PROMPT_AUDIO_PATH = Path(EXAMPLE_AUDIO_PATH)
INPUT_AUDIO_PATH = PROMPT_AUDIO_PATH
CLONE_TEXT = "大家好，我现在正在大可奇奇体验AI科技。"
LANGUAGE = "Chinese"
VOICE_DESIGN_TEXT = "太棒了！这是一个完全由文字描述创造的新声音，听起来是不是很有活力？"
VOICE_DESCRIPTION = "一个活泼的年轻女性声音，音调较高，语速较快，情绪兴奋且俏皮。"
SEMANTIC_EDIT_INSTRUCTION = "Replace '太乙真人' with '大可奇奇'."
ACOUSTIC_EDIT_INSTRUCTION = "adjust the speed to 0.5x"

FLOW_STEPS_REQUEST: int | None = None
CFG_OVERRIDE: float | None = None
STOP_THRESHOLD_OVERRIDE: float | None = None
MAX_AUDIO_PATCHES_OVERRIDE: int | None = None
MIN_AUDIO_PATCHES_OVERRIDE: int | None = None
MAX_TEXT_TOKENS = 200
TEXT_TEMPERATURE = 0.7
TEXT_TOP_P = 0.8
TEXT_TOP_K = 20
TEXT_REPETITION_PENALTY = 1.0
SEED = 9527

ORT_ACCELERATE_PROVIDERS: list[str] = []  # e.g. ["CUDAExecutionProvider"]
DEVICE_ID = 0
MAX_THREADS = 0
ORT_LOG = False
SHOW_PROGRESS = True  # Print pipeline stages and loop progress.
PREPACK_ROLES = frozenset(
    {
        "base_audio_start",
        "base_decode_step",
        "instruct_audio_start",
        "instruct_text_decode_step",
        "instruct_audio_decode_step",
    }
)
WEIGHT_ONLY_OP_TYPES = frozenset({"MatMulNBits", "GatherBlockQuantized"})
TOKENIZER_FOLDER = Path(
    os.environ.get("FIREREDTTS3_TOKENIZER_FOLDER", DOWNLOADS_DIR / "FireRedTTS3" / "text_tokenizer")
).expanduser()


METADATA_FILE_NAME = "FireRedTTS3_Metadata.onnx"

MULTI_LANG_TAGS = (
    "Chinese", "English", "Cantonese", "Japanese", "Korean", "Spanish", "French", "Russian",
    "Arabic", "Turkish", "Indonesian", "Portuguese", "Italian", "Dutch", "Vietnamese", "German",
    "Ukrainian", "Thai", "Polish", "Romanian", "Greek", "Czech", "Finnish", "Hindi",
)
MULTI_DIALECT_TAGS = (
    "ZH_Anhui", "ZH_Fujian", "ZH_Gansu", "ZH_Guizhou", "ZH_Hebei", "ZH_Henan", "ZH_Hubei",
    "ZH_Hunan", "ZH_Jiangxi", "ZH_Liaoning", "ZH_Minnan", "ZH_Ningxia", "ZH_Shaanxi",
    "ZH_Shandong", "ZH_Shanghai", "ZH_Shanxi", "ZH_Sichuan", "ZH_Tianjin", "ZH_Wenzhou",
    "ZH_Wu", "ZH_Yunnan",
)

INV_INT16 = float(1.0 / 32768.0)


class RuntimeContractError(RuntimeError):
    """Raised when package metadata or a user request violates the runtime contract."""


TensorValue: TypeAlias = np.ndarray | onnxruntime.OrtValue


def _argument_dtype(argument: onnxruntime.NodeArg) -> np.dtype:
    type_name = argument.type
    if not type_name.startswith("tensor(") or not type_name.endswith(")"):
        raise RuntimeContractError(
            f"Model I/O {argument.name!r} has unsupported ORT type {type_name!r}."
        )
    try:
        data_type = onnx.TensorProto.DataType.Value(type_name[7:-1].upper())
        return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(data_type))
    except (TypeError, ValueError) as error:
        raise RuntimeContractError(
            f"Model I/O {argument.name!r} has unsupported tensor type {type_name!r}."
        ) from error


def _argument_shape(
    argument: onnxruntime.NodeArg,
    dynamic_dimensions: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    dimensions = dict(dynamic_dimensions or {})
    shape: list[int] = []
    for axis, model_dimension in enumerate(argument.shape):
        if isinstance(model_dimension, int):
            dimension = model_dimension
            if axis in dimensions and dimensions[axis] != dimension:
                raise RuntimeContractError(
                    f"Model I/O {argument.name!r} axis {axis} is fixed at {dimension}, "
                    f"not {dimensions[axis]}."
                )
        else:
            try:
                dimension = dimensions[axis]
            except KeyError as error:
                raise RuntimeContractError(
                    f"Model I/O {argument.name!r} axis {axis} ({model_dimension!r}) "
                    "requires a runtime dimension."
                ) from error
        if dimension < 0:
            raise RuntimeContractError(
                f"Model I/O {argument.name!r} axis {axis} has invalid size {dimension}."
            )
        shape.append(dimension)
    unexpected = sorted(set(dimensions) - set(range(len(argument.shape))))
    if unexpected:
        raise RuntimeContractError(
            f"Model I/O {argument.name!r} received dimensions for invalid axes {unexpected}."
        )
    return tuple(shape)


def _model_array(argument: onnxruntime.NodeArg, values: object) -> np.ndarray:
    array = np.asarray(values, dtype=_argument_dtype(argument))
    dynamic_axes = [
        axis
        for axis, dimension in enumerate(argument.shape)
        if not isinstance(dimension, int)
    ]
    dynamic_dimensions: dict[int, int] = {}
    if dynamic_axes:
        if array.ndim == len(argument.shape):
            dynamic_dimensions = {axis: array.shape[axis] for axis in dynamic_axes}
        elif len(dynamic_axes) == 1:
            fixed_elements = 1
            for dimension in argument.shape:
                if isinstance(dimension, int):
                    fixed_elements *= dimension
            if fixed_elements == 0 or array.size % fixed_elements:
                raise RuntimeContractError(
                    f"Values for model I/O {argument.name!r} cannot fill its shape {argument.shape}."
                )
            dynamic_dimensions[dynamic_axes[0]] = array.size // fixed_elements
        else:
            raise RuntimeContractError(
                f"Values for model I/O {argument.name!r} must expose its dynamic axes {dynamic_axes}."
            )
    shape = _argument_shape(argument, dynamic_dimensions)
    expected_elements = 1
    for dimension in shape:
        expected_elements *= dimension
    if array.size != expected_elements:
        raise RuntimeContractError(
            f"Values for model I/O {argument.name!r} contain {array.size} elements; "
            f"the model shape {shape} requires {expected_elements}."
        )
    return np.ascontiguousarray(array.reshape(shape))


def _zeros_for_argument(argument: onnxruntime.NodeArg) -> np.ndarray:
    return np.zeros(_argument_shape(argument), dtype=_argument_dtype(argument))


def _fixed_dimension(argument: onnxruntime.NodeArg, axis: int) -> int:
    try:
        dimension = argument.shape[axis]
    except IndexError as error:
        raise RuntimeContractError(
            f"Model I/O {argument.name!r} has no axis {axis}; shape={argument.shape}."
        ) from error
    if not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeContractError(
            f"Model I/O {argument.name!r} axis {axis} must be a fixed positive dimension; "
            f"shape={argument.shape}."
        )
    return dimension


def _dynamic_axis(argument: onnxruntime.NodeArg) -> int:
    axes = tuple(
        axis
        for axis, dimension in enumerate(argument.shape)
        if not isinstance(dimension, int)
    )
    if len(axes) != 1:
        raise RuntimeContractError(
            f"Model I/O {argument.name!r} must have exactly one dynamic axis; "
            f"shape={argument.shape}."
        )
    return axes[0]


def _validate_ort_value(
    argument: onnxruntime.NodeArg,
    value: onnxruntime.OrtValue,
) -> None:
    if value.data_type() != argument.type:
        raise RuntimeContractError(
            f"Model I/O {argument.name!r} requires {argument.type}, got {value.data_type()}."
        )
    actual_shape = tuple(value.shape())
    if len(actual_shape) != len(argument.shape):
        raise RuntimeContractError(
            f"Model I/O {argument.name!r} requires rank {len(argument.shape)}, "
            f"got shape {actual_shape}."
        )
    for axis, (model_dimension, actual_dimension) in enumerate(
        zip(argument.shape, actual_shape)
    ):
        if isinstance(model_dimension, int) and actual_dimension != model_dimension:
            raise RuntimeContractError(
                f"Model I/O {argument.name!r} axis {axis} requires {model_dimension}, "
                f"got {actual_dimension}."
            )


def _compatible_arguments(
    source: onnxruntime.NodeArg,
    target: onnxruntime.NodeArg,
) -> bool:
    if source.type != target.type or len(source.shape) != len(target.shape):
        return False
    return all(
        source_dimension == target_dimension
        for source_dimension, target_dimension in zip(source.shape, target.shape)
        if isinstance(source_dimension, int) and isinstance(target_dimension, int)
    )


@dataclass(frozen=True)
class SessionContract:
    inputs: tuple[onnxruntime.NodeArg, ...]
    outputs: tuple[onnxruntime.NodeArg, ...]
    cache_input_indices: tuple[int, ...] = ()
    cache_output_indices: tuple[int, ...] = ()

    @classmethod
    def from_session(cls, session: onnxruntime.InferenceSession) -> SessionContract:
        return cls(tuple(session.get_inputs()), tuple(session.get_outputs()))

    @property
    def value_inputs(self) -> tuple[onnxruntime.NodeArg, ...]:
        cache_indices = set(self.cache_input_indices)
        return tuple(
            argument
            for index, argument in enumerate(self.inputs)
            if index not in cache_indices
        )

    @property
    def cache_inputs(self) -> tuple[onnxruntime.NodeArg, ...]:
        return tuple(self.inputs[index] for index in self.cache_input_indices)

    @property
    def value_outputs(self) -> tuple[onnxruntime.NodeArg, ...]:
        cache_indices = set(self.cache_output_indices)
        return tuple(
            argument
            for index, argument in enumerate(self.outputs)
            if index not in cache_indices
        )

    @property
    def cache_outputs(self) -> tuple[onnxruntime.NodeArg, ...]:
        return tuple(self.outputs[index] for index in self.cache_output_indices)


@dataclass(frozen=True)
class RuntimeOutputs:
    values: tuple[onnxruntime.OrtValue, ...]
    cache: tuple[onnxruntime.OrtValue, ...]


def print_progress(message: str) -> None:
    if SHOW_PROGRESS:
        print(f"[FireRedTTS3 ONNX] {message}", flush=True)


def _print_loop_progress(
    label: str,
    current: int,
    maximum: int,
    *,
    interval: int,
) -> None:
    if current % interval == 0 or current == maximum:
        print_progress(f"{label}: {current}/{maximum}")


def _default_package_folder(variant: str) -> Path:
    optimized = SCRIPT_DIR / f"FireRedTTS3_{variant.capitalize()}_Optimized"
    exported = SCRIPT_DIR / f"FireRedTTS3_{variant.capitalize()}_ONNX"
    metadata_path = optimized / METADATA_FILE_NAME
    if metadata_path.is_file():
        try:
            model = onnx.load(metadata_path, load_external_data=False)
            metadata = {entry.key: entry.value for entry in model.metadata_props}
        except (OSError, onnx.checker.ValidationError, ValueError):
            metadata = {}
        if "model_audio_sample_rate" in metadata:
            return optimized
    return exported


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_variant = _run_modes_variant(RUN_MODES)
    parser.add_argument(
        "--package-folder",
        "--onnx-folder",
        dest="package_folder",
        type=Path,
        default=_default_package_folder(default_variant),
        help="Merged or optimized FireRedTTS3 package folder.",
    )
    return parser.parse_args()


def _metadata_int(metadata: Mapping[str, str], key: str) -> int:
    try:
        return int(metadata[key])
    except KeyError as error:
        raise RuntimeContractError(f"Package metadata is missing {key!r}.") from error
    except ValueError as error:
        raise RuntimeContractError(f"Package metadata has a non-integer {key!r}: {metadata[key]!r}.") from error


def _metadata_float(metadata: Mapping[str, str], key: str) -> float:
    try:
        return float(metadata[key])
    except KeyError as error:
        raise RuntimeContractError(f"Package metadata is missing {key!r}.") from error
    except ValueError as error:
        raise RuntimeContractError(f"Package metadata has a non-float {key!r}: {metadata[key]!r}.") from error


def _metadata_string(metadata: Mapping[str, str], key: str) -> str:
    try:
        return metadata[key]
    except KeyError as error:
        raise RuntimeContractError(f"Package metadata is missing {key!r}.") from error


def _run_mode_variant(mode: str) -> str:
    if mode == "base_clone":
        return "base"
    if mode in {"instruct_clone", "voice_design", "semantic_edit", "acoustic_edit"}:
        return "instruct"
    raise RuntimeContractError(f"Unsupported RUN_MODE: {mode!r}.")


def _run_modes_variant(modes: Sequence[str]) -> str:
    if not modes:
        raise RuntimeContractError("At least one RUN_MODE is required.")
    variants = {_run_mode_variant(mode) for mode in modes}
    if len(variants) != 1:
        raise RuntimeContractError(
            "One inference invocation can only run modes from the same package variant."
        )
    return variants.pop()


class PackageRuntime:
    def __init__(self, folder: Path, run_modes: Sequence[str]) -> None:
        self.folder = folder.expanduser().resolve()
        self.run_modes = tuple(run_modes)
        expected_variant = _run_modes_variant(self.run_modes)
        print_progress(f"Reading package metadata: {self.folder / METADATA_FILE_NAME}")
        self.metadata = inference_metadata(
            validate_package_contract(
                self.folder,
                METADATA_FILE_NAME,
                required_keys=INFERENCE_METADATA_KEYS
                - {
                    "default_clone_cfg",
                    "model_audio_sample_rate",
                    "redae_upsample_rate",
                    "redae_max_seq_len",
                },
                require_shared_bundle=True,
            )
        )
        if self.metadata["graph_layout"] != "merged_decode_step":
            raise RuntimeContractError(
                "Inference requires a merged package; run Merge_ONNX.py before this runner."
            )
        if _metadata_int(self.metadata, "package_schema_version") != 2:
            raise RuntimeContractError(
                "Inference requires FireRedTTS3 package schema version 2; re-export the package."
            )
        expected_contract = {
            "runtime_tensor_contract": "waveform_and_token_ids_to_waveform",
            "graph_owned_preprocess": "1",
            "graph_owned_sampling": "1",
            "graph_owned_postprocess": "1",
            "device_resident_decode_state": "1",
            "flow_schedule": "one_minus_cosine_half_pi",
        }
        mismatched_contract = {
            key: self.metadata.get(key)
            for key, expected in expected_contract.items()
            if self.metadata.get(key) != expected
        }
        if mismatched_contract:
            raise RuntimeContractError(
                f"Package does not declare the supported end-to-end contract: {mismatched_contract}."
            )
        self.variant = _metadata_string(self.metadata, "model_variant")
        if self.variant != expected_variant:
            raise RuntimeContractError(
                f"RUN_MODES={self.run_modes!r} require the {expected_variant!r} package, "
                f"but metadata declares {self.variant!r}."
            )
        self.sample_rate = _metadata_int(self.metadata, "out_sample_rate")
        self.input_sample_rate = _metadata_int(self.metadata, "input_audio_sample_rate")
        self.model_sample_rate = (
            _metadata_int(self.metadata, "model_audio_sample_rate")
            if "model_audio_sample_rate" in self.metadata
            else self.input_sample_rate
        )
        self.redae_downsample_rate = _metadata_int(
            self.metadata, "redae_downsample_rate"
        )
        self.redae_upsample_rate = (
            _metadata_int(self.metadata, "redae_upsample_rate")
            if "redae_upsample_rate" in self.metadata
            else 2
        )
        self.patch_size = _metadata_int(self.metadata, "patch_size")
        self.flow_steps = _metadata_int(self.metadata, "flow_steps")
        self.max_seq_len = _metadata_int(self.metadata, "max_seq_len")
        self.redae_max_seq_len = (
            _metadata_int(self.metadata, "redae_max_seq_len")
            if "redae_max_seq_len" in self.metadata
            else self.max_seq_len
        )
        self.vocab_size = _metadata_int(self.metadata, "vocab_size")
        self.default_cfg = _metadata_float(self.metadata, "default_cfg")
        self.default_clone_cfg = (
            _metadata_float(self.metadata, "default_clone_cfg")
            if "default_clone_cfg" in self.metadata
            else 2.0
        )
        self.max_audio_patches = _metadata_int(self.metadata, "max_audio_patches")
        self.min_audio_patches = _metadata_int(self.metadata, "min_audio_patches")
        self.stop_threshold = _metadata_float(self.metadata, "stop_threshold_default")
        self.text_eot_id = _metadata_int(self.metadata, "text_eot_id")
        self.audio_sos_id = _metadata_int(self.metadata, "audio_sos_id")
        self.latent_in_pad_id = _metadata_int(self.metadata, "latent_in_pad_id")
        self.latent_out_pad_id = _metadata_int(self.metadata, "latent_out_pad_id")
        self._validate_runtime_settings()
        self.graph_files = {
            key.removeprefix("model_file_name_"): self.folder / value
            for key, value in self.metadata.items()
            if key.startswith("model_file_name_")
        }
        self._validate_graph_roles()
        self.active_roles = tuple(
            sorted(
                {
                    role
                    for mode in self.run_modes
                    for role in self._runtime_roles(mode)
                }
            )
        )
        self.prepacked_roles = tuple(
            role
            for role in self.active_roles
            if role in PREPACK_ROLES and self._uses_weight_only_ops(role)
        )
        self.options = {
            role: self._session_options(prepack=role in self.prepacked_roles)
            for role in self.active_roles
        }
        self.providers = self._providers()
        self.provider_options = self._provider_options()
        self.device_type, self.ort_device = self._runtime_device()
        self.constant_values: dict[tuple[str, tuple[int, ...], bytes], onnxruntime.OrtValue] = {}
        onnxruntime.set_seed(SEED)
        shared_path = self.folder / self.metadata["shared_initializer_model_file"]
        shared_initializer_names = self._shared_initializer_names()
        shared_started = time.perf_counter()
        print_progress("Attaching shared ONNX initializers...")
        self.shared_arrays, self.shared_ort_values = attach_shared_initializers(
            self.options[self.active_roles[0]],
            shared_path,
            initializer_names=shared_initializer_names,
        )
        for role in self.active_roles[1:]:
            for name, value in zip(self.shared_arrays, self.shared_ort_values):
                self.options[role].add_initializer(name, value)
        print_progress(
            f"Shared ONNX initializers ready in "
            f"{time.perf_counter() - shared_started:.2f}s."
        )
        if self.prepacked_roles:
            print_progress(
                "Prepacking hot weight-only graphs: "
                + ", ".join(self.prepacked_roles)
            )
        print_progress(
            f"Loading {len(self.active_roles)}/{len(self.graph_files)} package graphs "
            f"required by {', '.join(self.run_modes)}."
        )
        session_started = time.perf_counter()
        self.sessions = {
            role: self._create_session(
                role, self.graph_files[role], index, len(self.active_roles)
            )
            for index, role in enumerate(self.active_roles, start=1)
        }
        print_progress(
            f"ONNX Runtime sessions ready in "
            f"{time.perf_counter() - session_started:.2f}s; "
            f"providers={self.providers}."
        )
        self.contracts = {
            role: SessionContract.from_session(session)
            for role, session in self.sessions.items()
        }
        for mode in self.run_modes:
            self._configure_cache_contracts(mode)
        prefill_role = self.prefill_role(self.run_modes[0])
        self.cache_sequence_axis = _dynamic_axis(
            self.contract(prefill_role).cache_outputs[0]
        )
        audio_start_role = "base_audio_start" if self.variant == "base" else "instruct_audio_start"
        latent_argument = self.input_argument("redae_decode", 0)
        patch_argument = self.output_argument(audio_start_role, 0)
        if not _compatible_arguments(patch_argument, latent_argument):
            raise RuntimeContractError(
                f"Generated patch I/O {patch_argument.type}/{patch_argument.shape} is "
                f"incompatible with RedAE latent I/O {latent_argument.type}/{latent_argument.shape}."
            )
        self.latent_frame_axis = _dynamic_axis(latent_argument)
        self.patch_frames = _fixed_dimension(patch_argument, self.latent_frame_axis)
        if self.patch_frames != self.patch_size:
            raise RuntimeContractError(
                f"Metadata patch_size={self.patch_size} does not match graph width "
                f"{self.patch_frames}."
            )
        for mode in self.run_modes:
            self._validate_cache_contracts(mode)

    def _validate_runtime_settings(self) -> None:
        if FLOW_STEPS_REQUEST is not None and FLOW_STEPS_REQUEST != self.flow_steps:
            raise RuntimeContractError(
                f"Requested flow_steps={FLOW_STEPS_REQUEST}, but package requires {self.flow_steps}."
            )
        if self.max_seq_len < 1:
            raise RuntimeContractError(
                f"Package max_seq_len must be positive, got {self.max_seq_len}."
            )
        if self.vocab_size < 1:
            raise RuntimeContractError(
                f"Package vocab_size must be positive, got {self.vocab_size}."
            )
        if (
            self.redae_downsample_rate < 1
            or self.input_sample_rate < 1
            or self.model_sample_rate < 1
            or self.sample_rate < 1
            or self.redae_upsample_rate < 1
            or self.redae_max_seq_len < 1
            or self.patch_size < 1
            or self.redae_downsample_rate % self.redae_upsample_rate
        ):
            raise RuntimeContractError("Package declares invalid RedAE sequence geometry.")

    def cache_length(self, cache: Sequence[onnxruntime.OrtValue]) -> int:
        if not cache:
            raise RuntimeContractError("A recurrent cache is required to measure context length.")
        return int(cache[0].shape()[self.cache_sequence_axis])

    def require_context_capacity(
        self,
        current_length: int,
        added_length: int,
        operation: str,
    ) -> None:
        requested_length = current_length + added_length
        if current_length < 0 or added_length < 0 or requested_length > self.max_seq_len:
            raise RuntimeContractError(
                f"{operation} requires backbone context length {requested_length}, "
                f"but this package supports at most {self.max_seq_len}."
            )

    def encoded_latent_frames(self, sample_count: int, operation: str) -> int:
        if sample_count < 1:
            raise RuntimeContractError(f"{operation} requires non-empty audio.")
        scale = float(self.model_sample_rate / self.input_sample_rate)
        model_sample_count = math.floor(sample_count * scale)
        if model_sample_count < 1:
            raise RuntimeContractError(
                f"{operation} becomes empty after public-to-model rate conversion."
            )
        alignment = self.redae_downsample_rate * self.patch_size
        aligned_samples = math.ceil(model_sample_count / alignment) * alignment
        return aligned_samples // self.redae_downsample_rate

    def require_audio_capacity(self, sample_count: int, operation: str) -> None:
        latent_frames = self.encoded_latent_frames(sample_count, operation)
        audio_patch_size = self.redae_downsample_rate // self.redae_upsample_rate
        sequence_length = latent_frames * self.redae_downsample_rate // audio_patch_size
        if sequence_length > self.redae_max_seq_len:
            raise RuntimeContractError(
                f"{operation} requires {sequence_length} RedAE encoder positions, "
                f"but this package supports at most {self.redae_max_seq_len}."
            )

    def require_decode_capacity(
        self,
        generated_latent_frames: int,
        prefix_latent_frames: int,
    ) -> None:
        sequence_length = (
            generated_latent_frames + prefix_latent_frames
        ) * self.redae_upsample_rate
        if sequence_length > self.redae_max_seq_len:
            raise RuntimeContractError(
                f"RedAE decode requires {sequence_length} positions, but this "
                f"package supports at most {self.redae_max_seq_len}."
            )

    def _validate_graph_roles(self) -> None:
        common = {"redae_decode"}
        if self.variant == "base":
            if "base_reference_prefill" in self.graph_files:
                expected = common | {
                    "base_reference_prefill",
                    "base_audio_start",
                    "base_decode_step",
                }
            else:
                expected = common | {
                    "base_reference_preprocess",
                    "base_input_prefill",
                    "base_audio_start",
                    "base_decode_step",
                }
        else:
            expected = common | {
                "instruct_input_prefill",
                "instruct_text_decode_step",
                "instruct_audio_start",
                "instruct_audio_decode_step",
            }
            if self.instruct_audio_prefill_roles():
                expected.update(self.instruct_audio_prefill_roles())
            else:
                expected.add("redae_encode")
        missing = sorted(expected - self.graph_files.keys())
        if missing:
            raise RuntimeContractError(f"Package metadata is missing runtime graph roles: {missing}.")

    def prefill_role(self, mode: str) -> str:
        if self.variant == "base":
            return (
                "base_reference_prefill"
                if "base_reference_prefill" in self.graph_files
                else "base_input_prefill"
            )
        if self.instruct_audio_prefill_roles():
            if mode == "instruct_clone":
                return "instruct_output_audio_prefill"
            if mode in {"semantic_edit", "acoustic_edit"}:
                return "instruct_input_audio_prefill"
        return "instruct_input_prefill"

    def _decode_roles(self, mode: str) -> tuple[str, ...]:
        if self.variant == "base":
            return ("base_decode_step",)
        roles: list[str] = []
        if mode in {"voice_design", "semantic_edit"}:
            roles.append("instruct_text_decode_step")
        roles.append("instruct_audio_decode_step")
        return tuple(roles)

    def _runtime_roles(self, mode: str) -> tuple[str, ...]:
        prefill_role = self.prefill_role(mode)
        roles = {
            "redae_decode",
            prefill_role,
            *self._decode_roles(mode),
        }
        if self.variant == "base":
            roles.add("base_audio_start")
            if prefill_role == "base_input_prefill":
                roles.add("base_reference_preprocess")
        else:
            roles.add("instruct_audio_start")
            if prefill_role == "instruct_input_prefill" and mode != "voice_design":
                roles.add("redae_encode")
        return tuple(sorted(roles))

    def _shared_initializer_names(self) -> frozenset[str]:
        shared_data_name = self.metadata["shared_initializer_data_file"]
        names: set[str] = set()
        for role in self.active_roles:
            model = onnx.load(self.graph_files[role], load_external_data=False)
            for tensor in model.graph.initializer:
                if tensor.data_location != onnx.TensorProto.EXTERNAL:
                    continue
                external = {entry.key: entry.value for entry in tensor.external_data}
                if external.get("location") == shared_data_name:
                    names.add(tensor.name)
        return frozenset(names)

    def _uses_weight_only_ops(self, role: str) -> bool:
        model = onnx.load(self.graph_files[role], load_external_data=False)
        return any(node.op_type in WEIGHT_ONLY_OP_TYPES for node in model.graph.node)

    def instruct_audio_prefill_roles(self) -> tuple[str, ...]:
        roles = (
            "instruct_input_audio_prefill",
            "instruct_output_audio_prefill",
        )
        present = tuple(role for role in roles if role in self.graph_files)
        if present and len(present) != len(roles):
            raise RuntimeContractError(
                "Instruct package must declare both input- and output-audio "
                "prefill graphs together."
            )
        return present

    @staticmethod
    def _compatible_block_starts(
        arguments: Sequence[onnxruntime.NodeArg],
        reference: Sequence[onnxruntime.NodeArg],
    ) -> tuple[int, ...]:
        width = len(reference)
        return tuple(
            start
            for start in range(len(arguments) - width + 1)
            if all(
                _compatible_arguments(source, target)
                for source, target in zip(reference, arguments[start : start + width])
            )
        )

    def _configure_cache_contracts(self, mode: str) -> None:
        prefill_role = self.prefill_role(mode)
        decode_roles = self._decode_roles(mode)
        prefill = self.contracts[prefill_role]
        maximum = min(
            len(prefill.outputs),
            *(len(self.contracts[role].outputs) for role in decode_roles),
        )
        for width in range(maximum - maximum % 2, 1, -2):
            reference = prefill.outputs[:width]
            input_starts: dict[str, int] = {}
            for role in decode_roles:
                contract = self.contracts[role]
                if not all(
                    _compatible_arguments(source, target)
                    for source, target in zip(reference, contract.outputs[:width])
                ):
                    break
                starts = self._compatible_block_starts(contract.inputs, reference)
                if len(starts) != 1:
                    break
                input_starts[role] = starts[0]
            else:
                cache_output_indices = tuple(range(width))
                self.contracts[prefill_role] = SessionContract(
                    prefill.inputs,
                    prefill.outputs,
                    cache_output_indices=cache_output_indices,
                )
                for role, start in input_starts.items():
                    contract = self.contracts[role]
                    self.contracts[role] = SessionContract(
                        contract.inputs,
                        contract.outputs,
                        cache_input_indices=tuple(range(start, start + width)),
                        cache_output_indices=cache_output_indices,
                    )
                return
        raise RuntimeContractError(
            f"The ONNX graph I/O for {mode} does not expose one unambiguous "
            "paired recurrent block."
        )

    def _validate_cache_contracts(self, mode: str) -> None:
        prefill_role = self.prefill_role(mode)
        decode_roles = self._decode_roles(mode)
        canonical = self.contracts[prefill_role].cache_outputs
        if not canonical or len(canonical) % 2:
            raise RuntimeContractError(
                f"{prefill_role} must expose a non-empty paired recurrent tensor block."
            )
        for role in decode_roles:
            contract = self.contracts[role]
            if len(contract.cache_inputs) != len(canonical):
                raise RuntimeContractError(
                    f"{role} declares {len(contract.cache_inputs)} recurrent inputs; "
                    f"{prefill_role} produces {len(canonical)}."
                )
            if len(contract.cache_outputs) != len(canonical):
                raise RuntimeContractError(
                    f"{role} declares {len(contract.cache_outputs)} recurrent outputs; "
                    f"{prefill_role} produces {len(canonical)}."
                )
            for direction, arguments in (
                ("inputs", contract.cache_inputs),
                ("outputs", contract.cache_outputs),
            ):
                for source, target in zip(canonical, arguments):
                    if not _compatible_arguments(source, target):
                        raise RuntimeContractError(
                            f"Recurrent model I/O is incompatible between {prefill_role} "
                            f"and {role} {direction}: {source.type}/{source.shape} versus "
                            f"{target.type}/{target.shape}."
                        )

    @staticmethod
    def _session_options(*, prepack: bool) -> onnxruntime.SessionOptions:
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = MAX_THREADS
        options.intra_op_num_threads = MAX_THREADS
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        # With shared initializer overrides, ORT rewrites transpose optimized flow Conv outputs.
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.log_severity_level = 0 if ORT_LOG else 4
        for key, value in {
            "session.set_denormal_as_zero": "1",
            "session.intra_op.allow_spinning": "1",
            "session.inter_op.allow_spinning": "1",
            "session.use_device_allocator_for_initializers": "1",
        }.items():
            options.add_session_config_entry(key, value)
        if not prepack:
            options.add_session_config_entry("session.disable_prepacking", "1")
        return options

    def _create_session(
        self,
        role: str,
        path: Path,
        index: int,
        total: int,
    ) -> onnxruntime.InferenceSession:
        print_progress(f"Loading ONNX graph {index}/{total}: {path.name}")
        return onnxruntime.InferenceSession(
            str(path),
            sess_options=self.options[role],
            providers=self.providers,
            provider_options=self.provider_options,
        )

    @staticmethod
    def _providers() -> list[str]:
        requested = ORT_ACCELERATE_PROVIDERS or ["CPUExecutionProvider"]
        available = set(onnxruntime.get_available_providers())
        missing = [provider for provider in requested if provider not in available]
        if missing:
            raise RuntimeContractError(
                f"Requested ONNX Runtime provider(s) are unavailable: {missing}; "
                f"available={sorted(available)}."
            )
        return requested

    def _provider_options(self) -> list[dict[str, str]]:
        options: list[dict[str, str]] = []
        for provider in self.providers:
            if provider in {"CUDAExecutionProvider", "DmlExecutionProvider"}:
                options.append({"device_id": str(DEVICE_ID)})
            else:
                options.append({})
        return options

    def _runtime_device(self) -> tuple[str, C.OrtDevice]:
        if "CUDAExecutionProvider" in self.providers:
            device_type = "cuda"
            ort_device_type = C.OrtDevice.cuda()
        elif "DmlExecutionProvider" in self.providers:
            device_type = "dml"
            ort_device_type = C.OrtDevice.dml()
        else:
            device_type = "cpu"
            ort_device_type = C.OrtDevice.cpu()
        return device_type, C.OrtDevice(
            ort_device_type,
            C.OrtDevice.default_memory(),
            DEVICE_ID,
        )

    def ort_value(self, value: np.ndarray) -> onnxruntime.OrtValue:
        array = np.ascontiguousarray(value)
        return onnxruntime.OrtValue.ortvalue_from_numpy(
            array,
            self.device_type,
            DEVICE_ID,
        )

    def constant(self, value: np.ndarray) -> onnxruntime.OrtValue:
        array = np.ascontiguousarray(value)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in self.constant_values:
            self.constant_values[key] = self.ort_value(array)
        return self.constant_values[key]

    def contract(self, role: str) -> SessionContract:
        try:
            return self.contracts[role]
        except KeyError as error:
            raise RuntimeContractError(f"Package has no session for graph role {role!r}.") from error

    def input_argument(self, role: str, index: int) -> onnxruntime.NodeArg:
        try:
            return self.contract(role).value_inputs[index]
        except IndexError as error:
            raise RuntimeContractError(
                f"{role} has no non-recurrent input at index {index}."
            ) from error

    def output_argument(self, role: str, index: int) -> onnxruntime.NodeArg:
        try:
            return self.contract(role).value_outputs[index]
        except IndexError as error:
            raise RuntimeContractError(
                f"{role} has no non-recurrent output at index {index}."
            ) from error

    def input_array(self, role: str, index: int, values: object) -> np.ndarray:
        return _model_array(self.input_argument(role, index), values)

    def input_constant(
        self,
        role: str,
        index: int,
        values: object,
    ) -> onnxruntime.OrtValue:
        return self.constant(self.input_array(role, index, values))

    def run(
        self,
        role: str,
        values: Sequence[TensorValue],
        *,
        cache: Sequence[onnxruntime.OrtValue] = (),
    ) -> RuntimeOutputs:
        try:
            session = self.sessions[role]
        except KeyError as error:
            raise RuntimeContractError(f"Package has no session for graph role {role!r}.") from error
        contract = self.contract(role)
        if len(values) != len(contract.value_inputs):
            raise RuntimeContractError(
                f"{role} requires {len(contract.value_inputs)} non-recurrent inputs, "
                f"received {len(values)}."
            )
        if len(cache) != len(contract.cache_inputs):
            raise RuntimeContractError(
                f"{role} requires {len(contract.cache_inputs)} recurrent inputs, "
                f"received {len(cache)}."
            )
        binding = session.io_binding()
        input_values: list[onnxruntime.OrtValue] = []
        arguments_and_values = (
            *zip(contract.value_inputs, values),
            *zip(contract.cache_inputs, cache),
        )
        for argument, value in arguments_and_values:
            if isinstance(value, onnxruntime.OrtValue):
                _validate_ort_value(argument, value)
                ort_value = value
            else:
                ort_value = self.ort_value(_model_array(argument, value))
            binding.bind_ortvalue_input(argument.name, ort_value)
            input_values.append(ort_value)
        for output in contract.outputs:
            binding._iobinding.bind_output(output.name, self.ort_device)
        session.run_with_iobinding(binding)
        output_values = tuple(binding.get_outputs())
        for argument, value in zip(contract.outputs, output_values):
            _validate_ort_value(argument, value)
        return RuntimeOutputs(
            values=tuple(
                value
                for index, value in enumerate(output_values)
                if index not in set(contract.cache_output_indices)
            ),
            cache=tuple(output_values[index] for index in contract.cache_output_indices),
        )


def load_tokenizer():
    tokenizer_folder = TOKENIZER_FOLDER
    if not tokenizer_folder.is_dir():
        raise FileNotFoundError(
            "Tokenizer folder is unavailable. Set package metadata/checkpoint layout so "
            f"{tokenizer_folder} exists."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_folder, local_files_only=True)
    special_tokens = [
        "<|sosp|>", "<|eosp|>", "<|empty|>", "<|Human|>", "<|SpeechLM|>", "<|sostm|>",
        "<|eostm|>", "<|sot|>", "<|eot|>", "<|TEXT_ONLY|>", "<|AUDIO_ONLY|>", "<|ASR|>",
        "<|TTS|>", "<|INTERLEAVE|>", "<|UNDERSTANDING|>",
        *[f"<|placeholder_{index:03d}|>" for index in range(1, 193)],
        *[f"<|{tag}|>" for tag in MULTI_LANG_TAGS],
        *[f"<|{tag}|>" for tag in MULTI_DIALECT_TAGS],
        "<|edit|>", "<|frame_patch|>", "<|end_edit|>",
    ]
    vocabulary = tokenizer.get_vocab()
    for token in special_tokens:
        if token not in vocabulary:
            tokenizer.add_tokens([token], special_tokens=True)
            vocabulary = tokenizer.get_vocab()
    return tokenizer


def _tokenizer_fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (
            folder / "tokenizer.json",
            folder / "tokenizer_config.json",
            folder / "vocab.json",
        )
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer contract file is missing: {path}")
        file_digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(path.name.encode("utf-8"))
        digest.update(file_digest.digest())
    return digest.hexdigest()


def token_ids(
    package: PackageRuntime,
    role: str,
    input_index: int,
    tokenizer,
    text: str,
) -> np.ndarray:
    values = tokenizer(text, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
    array = package.input_array(role, input_index, values)
    if array.size and (int(array.min()) < 0 or int(array.max()) >= package.vocab_size):
        raise RuntimeContractError(
            f"Tokenizer produced an ID outside model vocabulary [0, {package.vocab_size})."
        )
    return array


def _read_audio(
    path: Path,
    target_rate: int,
    argument: onnxruntime.NodeArg,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    dtype = _argument_dtype(argument)
    if dtype.kind != "f":
        raise RuntimeContractError(
            f"Audio model I/O {argument.name!r} must be floating-point, got {argument.type}."
        )
    samples = np.asarray(
        AudioSegment.from_file(path)
        .set_channels(1)
        .set_frame_rate(target_rate)
        .set_sample_width(2)
        .get_array_of_samples(),
        dtype=np.int16,
    )
    waveform = (samples.astype(np.float32) * INV_INT16).astype(
        dtype, copy=False
    )
    return _model_array(argument, waveform)


def _base_text(language: str, prompt_text: str, text: str) -> str:
    language_tag = f"<|{language}|>"
    valid = {f"<|{tag}|>" for tag in (*MULTI_LANG_TAGS, *MULTI_DIALECT_TAGS)}
    if language_tag not in valid:
        raise RuntimeContractError(f"Unsupported Base language or dialect: {language!r}.")
    return f"{language_tag}<|sot|>{prompt_text}{text}<|eot|>"


def _convert_to_chatml(
    text_in: str,
    *,
    latent_in_length: int = 0,
    text_out: str = "",
    latent_out_length: int = 0,
) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": []},
        {"role": "assistant", "content": []},
    ]
    if latent_in_length > 0:
        messages[1]["content"].append(("audio", "<|image_pad|>" * latent_in_length))
    messages[1]["content"].append(("text", text_in + " /no_think"))
    messages[2]["content"].append(("text", "<think>\n\n</think>\n\n" + text_out))
    if latent_out_length > 0:
        messages[2]["content"].append(("audio", "<|video_pad|>" * latent_out_length))
    parts: list[str] = [f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"]
    for message in messages[1:]:
        parts.append(f"<|im_start|>{message['role']}\n")
        for kind, content in message["content"]:
            parts.append(content if kind == "text" else f"<|sosp|>{content}<|eosp|>\n")
        parts.append("<|im_end|>\n")
    return "".join(parts)


def _chatml_clone(prompt_patches: int, prompt_text: str, text: str) -> str:
    result = _convert_to_chatml(
        f"Convert text to speech.\n{prompt_text}{text}", latent_out_length=prompt_patches
    )
    return result.removesuffix("<|eosp|>\n<|im_end|>\n")


def _chatml_voice_design(instruction: str, text: str) -> str:
    result = _convert_to_chatml(
        f"{instruction}\n\n根据上述音色描述，首先整理成语音属性，再合成以下文本对应的音频：\n{text}",
        text_out="<|sot|>",
    )
    return result.removesuffix("<|im_end|>\n")


def _chatml_semantic_edit(instruction: str, input_patches: int) -> str:
    result = _convert_to_chatml(
        f"Identify the content of the audio. {instruction.strip()}",
        latent_in_length=input_patches,
        text_out="<|sot|>",
    )
    return result.removesuffix("<|im_end|>\n")


def _chatml_acoustic_edit(instruction: str, input_patches: int) -> str:
    result = _convert_to_chatml(
        instruction, latent_in_length=input_patches, latent_out_length=1
    )
    return result.removesuffix("<|video_pad|><|eosp|>\n<|im_end|>\n")


def _validate_placeholder_count(
    mask: np.ndarray,
    latent_frames: int,
    patch_frames: int,
    name: str,
) -> None:
    if latent_frames % patch_frames:
        raise RuntimeContractError(
            f"{name} latent frames ({latent_frames}) are not divisible by the "
            f"model patch width ({patch_frames})."
        )
    expected = latent_frames // patch_frames
    actual = int(np.count_nonzero(mask))
    if actual != expected:
        raise RuntimeContractError(
            f"{name} placeholder count mismatch: expected {expected}, found {actual}."
        )


def _cfg_value(
    package: PackageRuntime,
    role: str,
    input_index: int,
    mode: str,
) -> onnxruntime.OrtValue:
    if CFG_OVERRIDE is None:
        value = (
            package.default_clone_cfg
            if mode in {"base_clone", "instruct_clone"}
            else package.default_cfg
        )
    else:
        value = CFG_OVERRIDE
    if not math.isfinite(value):
        raise RuntimeContractError(f"CFG must be finite, got {value}.")
    value = max(value, 0.0)
    return package.input_constant(role, input_index, value)


def _stop_threshold(package: PackageRuntime) -> float:
    return package.stop_threshold if STOP_THRESHOLD_OVERRIDE is None else STOP_THRESHOLD_OVERRIDE


def _stop_threshold_value(
    package: PackageRuntime,
    role: str,
    input_index: int,
) -> onnxruntime.OrtValue:
    return package.input_constant(role, input_index, _stop_threshold(package))


def _text_control_values(
    package: PackageRuntime,
    role: str,
    first_input_index: int,
    *,
    do_sample: bool,
) -> tuple[onnxruntime.OrtValue, ...]:
    if (
        TEXT_TEMPERATURE <= 0.0
        or TEXT_TOP_K <= 0
        or TEXT_TOP_K > package.vocab_size
        or not 0.0 < TEXT_TOP_P <= 1.0
        or TEXT_REPETITION_PENALTY <= 0.0
    ):
        raise RuntimeContractError("Invalid text sampling settings.")
    values = (
        do_sample,
        TEXT_TEMPERATURE,
        TEXT_TOP_K,
        TEXT_TOP_P,
        TEXT_REPETITION_PENALTY,
    )
    return tuple(
        package.input_constant(role, first_input_index + offset, value)
        for offset, value in enumerate(values)
    )


def _limits(
    package: PackageRuntime,
    cache: Sequence[onnxruntime.OrtValue],
    prefix_latent_frames: int = 0,
) -> tuple[int, int]:
    maximum = package.max_audio_patches if MAX_AUDIO_PATCHES_OVERRIDE is None else MAX_AUDIO_PATCHES_OVERRIDE
    minimum = package.min_audio_patches if MIN_AUDIO_PATCHES_OVERRIDE is None else MIN_AUDIO_PATCHES_OVERRIDE
    if minimum < 1 or maximum < minimum:
        raise RuntimeContractError(f"Invalid audio patch limits: min={minimum}, max={maximum}.")
    context_maximum = package.max_seq_len - package.cache_length(cache) + 1
    decode_latent_capacity = (
        package.redae_max_seq_len // package.redae_upsample_rate
        - prefix_latent_frames
    )
    decode_maximum = decode_latent_capacity // package.patch_frames
    maximum = min(
        maximum,
        package.max_audio_patches,
        context_maximum,
        decode_maximum,
    )
    if maximum < minimum:
        raise RuntimeContractError(
            f"Combined backbone/RedAE capacity has room for {maximum} audio "
            f"patches, below the required minimum of {minimum}."
        )
    return minimum, maximum


def _tensor_shape(value: TensorValue) -> tuple[int, ...]:
    return tuple(value.shape() if isinstance(value, onnxruntime.OrtValue) else value.shape)


def _empty_latents(package: PackageRuntime) -> onnxruntime.OrtValue:
    return package.input_constant("redae_decode", 1, ())


def _encode_audio(package: PackageRuntime, audio_path: Path) -> onnxruntime.OrtValue:
    role = "redae_encode"
    audio = _read_audio(
        audio_path,
        package.input_sample_rate,
        package.input_argument(role, 0),
    )
    audio_samples = _tensor_shape(audio)[
        _dynamic_axis(package.input_argument(role, 0))
    ]
    package.require_audio_capacity(audio_samples, "Audio encode")
    return package.run(role, (audio,)).values[0]


def _decode_and_write(
    package: PackageRuntime,
    generated_latents: onnxruntime.OrtValue,
    *,
    mode: str,
    prefix_latents: onnxruntime.OrtValue | None,
    started_at: float,
    generated_patches: int,
) -> np.ndarray:
    generated_frames = _tensor_shape(generated_latents)[package.latent_frame_axis]
    prefix_frames = (
        0
        if prefix_latents is None
        else _tensor_shape(prefix_latents)[package.latent_frame_axis]
    )
    package.require_decode_capacity(generated_frames, prefix_frames)
    decode_started = time.perf_counter()
    print_progress("Decoding waveform...")
    waveform_value = package.run(
        "redae_decode",
        (
            generated_latents,
            _empty_latents(package) if prefix_latents is None else prefix_latents,
        ),
    ).values[0]
    waveform = waveform_value.numpy().reshape(-1)
    print_progress(
        f"Waveform decoded in {time.perf_counter() - decode_started:.2f}s."
    )
    output_wav_path = SCRIPT_DIR / f"fireredtts3_onnx_{mode}.wav"
    print_progress(f"Writing generated audio: {output_wav_path}")
    sound_file.write(output_wav_path, waveform, package.sample_rate, subtype="FLOAT", format="WAV")
    elapsed = time.perf_counter() - started_at
    duration = waveform.size / package.sample_rate
    rtf = elapsed / duration if duration else float("inf")
    print(
        f"Generated patches={generated_patches}, audio={duration:.2f}s, elapsed={elapsed:.2f}s, RTF={rtf:.3f}",
        flush=True,
    )
    print_progress(f"Wrote float WAV: {output_wav_path}")
    return waveform


def _base_clone(package: PackageRuntime, tokenizer) -> np.ndarray:
    started_at = time.perf_counter()
    conditioning_started = time.perf_counter()
    print_progress(f"Preparing Base clone conditioning from: {PROMPT_AUDIO_PATH}")
    combined_prefill = "base_reference_prefill" in package.graph_files
    reference_role = (
        "base_reference_prefill" if combined_prefill else "base_reference_preprocess"
    )
    audio_input_index = 1 if combined_prefill else 0
    prompt_audio = _read_audio(
        PROMPT_AUDIO_PATH,
        package.input_sample_rate,
        package.input_argument(reference_role, audio_input_index),
    )
    prompt_samples = _tensor_shape(prompt_audio)[
        _dynamic_axis(package.input_argument(reference_role, audio_input_index))
    ]
    package.require_audio_capacity(prompt_samples, "Base reference preprocessing")
    prefill_role = package.prefill_role("base_clone")
    text = token_ids(
        package,
        prefill_role,
        0,
        tokenizer,
        _base_text(LANGUAGE, PROMPT_TEXT, CLONE_TEXT),
    )
    text_length = _tensor_shape(text)[_dynamic_axis(package.input_argument(prefill_role, 0))]
    expected_prompt_frames = package.encoded_latent_frames(
        prompt_samples, "Base reference preprocessing"
    )
    package.require_context_capacity(
        0,
        1 + text_length + expected_prompt_frames // package.patch_frames,
        "Base prefill",
    )
    if combined_prefill:
        prefill = package.run(prefill_role, (text, prompt_audio))
        *prefill_state, prompt_latents = prefill.values
    else:
        reference = package.run(reference_role, (prompt_audio,))
        prompt_latents, speaker_embedding = reference.values
        prefill = package.run(
            prefill_role,
            (text, prompt_latents, speaker_embedding),
        )
        prefill_state = list(prefill.values)
    prompt_frames = _tensor_shape(prompt_latents)[package.latent_frame_axis]
    if prompt_frames != expected_prompt_frames:
        raise RuntimeContractError(
            f"Base preprocessing produced {prompt_frames} latent frames; "
            f"the public-rate input contract predicted {expected_prompt_frames}."
        )
    print_progress(
        f"Base clone conditioning ready in "
        f"{time.perf_counter() - conditioning_started:.2f}s; "
        f"prompt_latent_frames={prompt_frames}."
    )
    cache = prefill.cache
    _, _, condition_history, latent_history, speaker_condition = prefill_state
    start_role = "base_audio_start"
    start_cfg = _cfg_value(package, start_role, 3, "base_clone")
    start = package.run(
        start_role,
        (latent_history, condition_history, speaker_condition, start_cfg),
    )
    prior_patch, latent_history, generated_latents = start.values
    minimum, maximum = _limits(package, cache, prompt_frames)
    generated_patches = 1
    print_progress(
        f"Generating Base audio patches (minimum {minimum}, limit {maximum})..."
    )
    _print_loop_progress(
        "Base audio generation", generated_patches, maximum, interval=10
    )
    decode_role = "base_decode_step"
    stop_threshold = _stop_threshold_value(package, decode_role, 1)
    decode_cfg = _cfg_value(package, decode_role, 5, "base_clone")
    while generated_patches < maximum:
        outputs = package.run(
            decode_role,
            (
                prior_patch,
                stop_threshold,
                latent_history,
                condition_history,
                speaker_condition,
                decode_cfg,
                generated_latents,
            ),
            cache=cache,
        )
        should_stop = bool(outputs.values[2].numpy().item())
        if generated_patches >= minimum and should_stop:
            stop_score = float(outputs.values[1].numpy().item())
            print_progress(
                f"Stopping Base audio loop at patch {generated_patches} "
                f"(score={stop_score:.4f})."
            )
            break
        cache = outputs.cache
        prior_patch = outputs.values[3]
        latent_history = outputs.values[4]
        condition_history = outputs.values[5]
        generated_latents = outputs.values[6]
        generated_patches += 1
        _print_loop_progress(
            "Base audio generation", generated_patches, maximum, interval=10
        )
    return _decode_and_write(
        package,
        generated_latents,
        mode="base_clone",
        prefix_latents=prompt_latents,
        started_at=started_at,
        generated_patches=generated_patches,
    )


def _dummy_latents(
    package: PackageRuntime,
    input_index: int,
) -> onnxruntime.OrtValue:
    template = _zeros_for_argument(
        package.output_argument("instruct_audio_start", 0)
    )
    return package.input_constant("instruct_input_prefill", input_index, template)


def _instruct_prefill(
    package: PackageRuntime,
    tokenizer,
    chatml: str,
    *,
    latents_in: onnxruntime.OrtValue | None,
    latents_out: onnxruntime.OrtValue | None,
    inject_out: bool,
    do_sample: bool,
) -> tuple[
    RuntimeOutputs,
    tuple[onnxruntime.OrtValue, ...],
    onnxruntime.OrtValue,
    onnxruntime.OrtValue,
    onnxruntime.OrtValue,
]:
    role = "instruct_input_prefill"
    text = token_ids(package, role, 0, tokenizer, chatml)
    text_length = _tensor_shape(text)[_dynamic_axis(package.input_argument(role, 0))]
    package.require_context_capacity(0, text_length, "Instruct prefill")
    input_mask = text == package.latent_in_pad_id
    output_mask = text == package.latent_out_pad_id
    actual_in = (
        0 if latents_in is None else _tensor_shape(latents_in)[package.latent_frame_axis]
    )
    actual_out = (
        0 if latents_out is None else _tensor_shape(latents_out)[package.latent_frame_axis]
    )
    _validate_placeholder_count(input_mask, actual_in, package.patch_frames, "input")
    _validate_placeholder_count(output_mask, actual_out, package.patch_frames, "output")
    if not inject_out and np.any(output_mask):
        raise RuntimeContractError("Unexpected output-audio placeholders for this mode.")
    graph_in = _dummy_latents(package, 1) if latents_in is None else latents_in
    graph_out = _dummy_latents(package, 2) if latents_out is None else latents_out
    controls = _text_control_values(package, role, 3, do_sample=do_sample)
    outputs = package.run(
        role,
        (text, graph_in, graph_out, *controls),
    )
    return (
        outputs,
        outputs.cache,
        outputs.values[0],
        outputs.values[3],
        outputs.values[4],
    )


def _instruct_audio_prefill(
    package: PackageRuntime,
    tokenizer,
    audio_path: Path,
    chatml,
    *,
    audio_role: str,
    do_sample: bool,
) -> tuple[
    RuntimeOutputs,
    tuple[onnxruntime.OrtValue, ...],
    onnxruntime.OrtValue,
    onnxruntime.OrtValue,
    onnxruntime.OrtValue,
    onnxruntime.OrtValue | None,
    int,
]:
    if audio_role not in {"input", "output"}:
        raise RuntimeContractError(f"Unsupported Instruct audio role: {audio_role!r}.")
    role = f"instruct_{audio_role}_audio_prefill"
    audio_argument = package.input_argument(role, 1)
    audio = _read_audio(audio_path, package.input_sample_rate, audio_argument)
    audio_samples = _tensor_shape(audio)[_dynamic_axis(audio_argument)]
    package.require_audio_capacity(audio_samples, "Instruct audio preprocessing")
    latent_frames = package.encoded_latent_frames(
        audio_samples, "Instruct audio preprocessing"
    )
    text = token_ids(package, role, 0, tokenizer, chatml(latent_frames // package.patch_frames))
    text_length = _tensor_shape(text)[_dynamic_axis(package.input_argument(role, 0))]
    package.require_context_capacity(0, text_length, "Instruct audio prefill")
    _validate_placeholder_count(
        text == package.latent_in_pad_id,
        latent_frames if audio_role == "input" else 0,
        package.patch_frames,
        "input",
    )
    _validate_placeholder_count(
        text == package.latent_out_pad_id,
        latent_frames if audio_role == "output" else 0,
        package.patch_frames,
        "output",
    )
    controls = _text_control_values(package, role, 2, do_sample=do_sample)
    outputs = package.run(role, (text, audio, *controls))
    prompt_latents = outputs.values[5] if audio_role == "output" else None
    return (
        outputs,
        outputs.cache,
        outputs.values[0],
        outputs.values[3],
        outputs.values[4],
        prompt_latents,
        latent_frames,
    )


def _run_text_phase(
    package: PackageRuntime,
    cache: tuple[onnxruntime.OrtValue, ...],
    next_text_id: onnxruntime.OrtValue,
    text_history: onnxruntime.OrtValue,
    *,
    do_sample: bool,
) -> tuple[tuple[onnxruntime.OrtValue, ...], onnxruntime.OrtValue, int]:
    role = "instruct_text_decode_step"
    controls = _text_control_values(package, role, 2, do_sample=do_sample)
    generated_tokens = 0
    text_started = time.perf_counter()
    print_progress(
        f"Generating Instruct text plan (limit {MAX_TEXT_TOKENS} tokens)..."
    )
    for _ in range(MAX_TEXT_TOKENS):
        package.require_context_capacity(
            package.cache_length(cache), 1, "Instruct text decode"
        )
        outputs = package.run(
            role,
            (next_text_id, text_history, *controls),
            cache=cache,
        )
        cache = outputs.cache
        text_history = outputs.values[2]
        if bool(outputs.values[3].numpy().item()):
            package.require_context_capacity(
                package.cache_length(cache), 2, "Instruct SOS transition"
            )
            audio_sos = package.input_constant(role, 0, package.audio_sos_id)
            sos_outputs = package.run(
                role,
                (
                    audio_sos,
                    text_history,
                    *controls,
                ),
                cache=cache,
            )
            audio_outputs = package.run(
                role,
                (
                    audio_sos,
                    sos_outputs.values[2],
                    *controls,
                ),
                cache=sos_outputs.cache,
            )
            print_progress(
                f"Text plan and SOS transition ready in "
                f"{time.perf_counter() - text_started:.2f}s."
            )
            return (
                audio_outputs.cache,
                audio_outputs.values[0],
                generated_tokens,
            )
        generated_tokens += 1
        next_text_id = outputs.values[1]
        _print_loop_progress(
            "Text plan generation",
            generated_tokens,
            MAX_TEXT_TOKENS,
            interval=25,
        )
    raise RuntimeContractError(
        f"Instruct text generation exceeded MAX_TEXT_TOKENS={MAX_TEXT_TOKENS} "
        f"without {package.text_eot_id}."
    )


def _instruct_audio_loop(
    package: PackageRuntime,
    cache: tuple[onnxruntime.OrtValue, ...],
    last_hidden_state: onnxruntime.OrtValue,
    condition_history: onnxruntime.OrtValue,
    latent_history: onnxruntime.OrtValue,
    *,
    mode: str,
    append_last_hidden: bool,
    prefix_latent_frames: int = 0,
) -> tuple[onnxruntime.OrtValue, int]:
    start_role = "instruct_audio_start"
    start_cfg = _cfg_value(package, start_role, 4, mode)
    start = package.run(
        start_role,
        (
            last_hidden_state,
            condition_history,
            package.input_constant(start_role, 2, float(append_last_hidden)),
            latent_history,
            start_cfg,
        ),
    )
    prior_patch, latent_history, condition_history, generated_latents = start.values
    minimum, maximum = _limits(package, cache, prefix_latent_frames)
    generated_patches = 1
    print_progress(
        f"Generating Instruct audio patches (minimum {minimum}, limit {maximum})..."
    )
    _print_loop_progress(
        "Instruct audio generation", generated_patches, maximum, interval=10
    )
    decode_role = "instruct_audio_decode_step"
    stop_threshold = _stop_threshold_value(package, decode_role, 1)
    decode_cfg = _cfg_value(package, decode_role, 4, mode)
    while generated_patches < maximum:
        outputs = package.run(
            decode_role,
            (
                prior_patch,
                stop_threshold,
                latent_history,
                condition_history,
                decode_cfg,
                generated_latents,
            ),
            cache=cache,
        )
        should_stop = bool(outputs.values[2].numpy().item())
        if generated_patches >= minimum and should_stop:
            stop_score = float(outputs.values[1].numpy().item())
            print_progress(
                f"Stopping Instruct audio loop at patch {generated_patches} "
                f"(score={stop_score:.4f})."
            )
            break
        cache = outputs.cache
        prior_patch = outputs.values[3]
        latent_history = outputs.values[4]
        condition_history = outputs.values[5]
        generated_latents = outputs.values[6]
        generated_patches += 1
        _print_loop_progress(
            "Instruct audio generation", generated_patches, maximum, interval=10
        )
    return generated_latents, generated_patches


def _instruct_clone(package: PackageRuntime, tokenizer) -> np.ndarray:
    started_at = time.perf_counter()
    conditioning_started = time.perf_counter()
    print_progress(f"Preparing Instruct clone conditioning from: {PROMPT_AUDIO_PATH}")
    if package.instruct_audio_prefill_roles():
        (
            outputs,
            cache,
            last_hidden,
            condition_history,
            latent_history,
            prompt_latents,
            prompt_frames,
        ) = _instruct_audio_prefill(
            package,
            tokenizer,
            PROMPT_AUDIO_PATH,
            lambda patches: _chatml_clone(
                patches, PROMPT_TEXT, CLONE_TEXT
            ),
            audio_role="output",
            do_sample=False,
        )
        if prompt_latents is None:
            raise RuntimeContractError("Output-audio prefill did not return prompt latents.")
    else:
        prompt_latents = _encode_audio(package, PROMPT_AUDIO_PATH)
        prompt_frames = _tensor_shape(prompt_latents)[package.latent_frame_axis]
        outputs, cache, last_hidden, condition_history, latent_history = _instruct_prefill(
            package,
            tokenizer,
            _chatml_clone(
                prompt_frames // package.patch_frames,
                PROMPT_TEXT,
                CLONE_TEXT,
            ),
            latents_in=None,
            latents_out=prompt_latents,
            inject_out=True,
            do_sample=False,
        )
    print_progress(
        f"Instruct clone conditioning ready in "
        f"{time.perf_counter() - conditioning_started:.2f}s; "
        f"prompt_latent_frames={prompt_frames}."
    )
    del outputs
    generated, generated_patches = _instruct_audio_loop(
        package,
        cache,
        last_hidden,
        condition_history,
        latent_history,
        mode="instruct_clone",
        append_last_hidden=False,
        prefix_latent_frames=prompt_frames,
    )
    return _decode_and_write(
        package,
        generated,
        mode="instruct_clone",
        prefix_latents=prompt_latents,
        started_at=started_at,
        generated_patches=generated_patches,
    )


def _voice_design(package: PackageRuntime, tokenizer) -> np.ndarray:
    started_at = time.perf_counter()
    conditioning_started = time.perf_counter()
    print_progress("Preparing voice-design conditioning...")
    outputs, cache, _, condition_history, latent_history = _instruct_prefill(
        package,
        tokenizer,
        _chatml_voice_design(VOICE_DESCRIPTION, VOICE_DESIGN_TEXT),
        latents_in=None,
        latents_out=None,
        inject_out=False,
        do_sample=True,
    )
    print_progress(
        f"Voice-design conditioning ready in "
        f"{time.perf_counter() - conditioning_started:.2f}s."
    )
    cache, audio_hidden, generated_text_tokens = _run_text_phase(
        package,
        cache,
        outputs.values[1],
        outputs.values[2],
        do_sample=True,
    )
    print_progress(f"Voice design text plan token count: {generated_text_tokens}")
    generated, generated_patches = _instruct_audio_loop(
        package,
        cache,
        audio_hidden,
        condition_history,
        latent_history,
        mode="voice_design",
        append_last_hidden=True,
    )
    return _decode_and_write(
        package,
        generated,
        mode="voice_design",
        prefix_latents=None,
        started_at=started_at,
        generated_patches=generated_patches,
    )


def _semantic_edit(package: PackageRuntime, tokenizer) -> np.ndarray:
    started_at = time.perf_counter()
    conditioning_started = time.perf_counter()
    print_progress(f"Preparing semantic-edit conditioning from: {INPUT_AUDIO_PATH}")
    if package.instruct_audio_prefill_roles():
        (
            outputs,
            cache,
            _,
            condition_history,
            latent_history,
            _,
            _,
        ) = _instruct_audio_prefill(
            package,
            tokenizer,
            INPUT_AUDIO_PATH,
            lambda patches: _chatml_semantic_edit(
                SEMANTIC_EDIT_INSTRUCTION, patches
            ),
            audio_role="input",
            do_sample=False,
        )
    else:
        input_latents = _encode_audio(package, INPUT_AUDIO_PATH)
        input_patches = (
            _tensor_shape(input_latents)[package.latent_frame_axis]
            // package.patch_frames
        )
        outputs, cache, _, condition_history, latent_history = _instruct_prefill(
            package,
            tokenizer,
            _chatml_semantic_edit(SEMANTIC_EDIT_INSTRUCTION, input_patches),
            latents_in=input_latents,
            latents_out=None,
            inject_out=False,
            do_sample=False,
        )
    print_progress(
        f"Semantic-edit conditioning ready in "
        f"{time.perf_counter() - conditioning_started:.2f}s."
    )
    cache, audio_hidden, generated_text_tokens = _run_text_phase(
        package,
        cache,
        outputs.values[1],
        outputs.values[2],
        do_sample=False,
    )
    print_progress(f"Semantic edit text token count: {generated_text_tokens}")
    generated, generated_patches = _instruct_audio_loop(
        package,
        cache,
        audio_hidden,
        condition_history,
        latent_history,
        mode="semantic_edit",
        append_last_hidden=True,
    )
    return _decode_and_write(
        package,
        generated,
        mode="semantic_edit",
        prefix_latents=None,
        started_at=started_at,
        generated_patches=generated_patches,
    )


def _acoustic_edit(package: PackageRuntime, tokenizer) -> np.ndarray:
    started_at = time.perf_counter()
    conditioning_started = time.perf_counter()
    print_progress(f"Preparing acoustic-edit conditioning from: {INPUT_AUDIO_PATH}")
    if package.instruct_audio_prefill_roles():
        (
            _,
            cache,
            last_hidden,
            condition_history,
            latent_history,
            _,
            _,
        ) = _instruct_audio_prefill(
            package,
            tokenizer,
            INPUT_AUDIO_PATH,
            lambda patches: _chatml_acoustic_edit(
                ACOUSTIC_EDIT_INSTRUCTION, patches
            ),
            audio_role="input",
            do_sample=False,
        )
    else:
        input_latents = _encode_audio(package, INPUT_AUDIO_PATH)
        input_patches = (
            _tensor_shape(input_latents)[package.latent_frame_axis]
            // package.patch_frames
        )
        _, cache, last_hidden, condition_history, latent_history = _instruct_prefill(
            package,
            tokenizer,
            _chatml_acoustic_edit(ACOUSTIC_EDIT_INSTRUCTION, input_patches),
            latents_in=input_latents,
            latents_out=None,
            inject_out=False,
            do_sample=False,
        )
    print_progress(
        f"Acoustic-edit conditioning ready in "
        f"{time.perf_counter() - conditioning_started:.2f}s."
    )
    generated, generated_patches = _instruct_audio_loop(
        package,
        cache,
        last_hidden,
        condition_history,
        latent_history,
        mode="acoustic_edit",
        append_last_hidden=True,
    )
    return _decode_and_write(
        package,
        generated,
        mode="acoustic_edit",
        prefix_latents=None,
        started_at=started_at,
        generated_patches=generated_patches,
    )


def _validate_tokenizer_contract(tokenizer, package: PackageRuntime) -> None:
    if len(tokenizer) > package.vocab_size:
        raise RuntimeContractError(
            f"Tokenizer has {len(tokenizer)} entries, but the model vocabulary has "
            f"only {package.vocab_size}."
        )
    expected = {
        "<|eot|>": package.text_eot_id,
        "<|sosp|>": package.audio_sos_id,
        "<|image_pad|>": package.latent_in_pad_id,
        "<|video_pad|>": package.latent_out_pad_id,
    }
    for token, token_id in expected.items():
        actual = int(tokenizer.convert_tokens_to_ids(token))
        if actual != token_id:
            raise RuntimeContractError(
                f"Tokenizer ID mismatch for {token}: expected {token_id}, got {actual}."
            )
    actual_fingerprint = _tokenizer_fingerprint(TOKENIZER_FOLDER)
    expected_fingerprint = package.metadata["tokenizer_path_or_fingerprint"]
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeContractError(
            "Tokenizer files do not match the fingerprint recorded in the ONNX package."
        )


def _run_mode(package: PackageRuntime, tokenizer, mode: str) -> None:
    print_progress(f"Starting demo: {mode}.")
    if mode == "base_clone":
        _base_clone(package, tokenizer)
    elif mode == "instruct_clone":
        _instruct_clone(package, tokenizer)
    elif mode == "voice_design":
        _voice_design(package, tokenizer)
    elif mode == "semantic_edit":
        _semantic_edit(package, tokenizer)
    elif mode == "acoustic_edit":
        _acoustic_edit(package, tokenizer)
    else:
        raise RuntimeContractError(f"Unsupported RUN_MODE: {mode!r}.")
    print_progress(f"Demo complete: {mode}.")


def main() -> None:
    pipeline_started = time.perf_counter()
    arguments = parse_arguments()
    package = PackageRuntime(arguments.package_folder, RUN_MODES)
    tokenizer_started = time.perf_counter()
    print_progress(f"Loading tokenizer: {TOKENIZER_FOLDER}")
    tokenizer = load_tokenizer()
    _validate_tokenizer_contract(tokenizer, package)
    print_progress(
        f"Tokenizer ready in {time.perf_counter() - tokenizer_started:.2f}s."
    )
    print_progress(
        f"variant={package.variant}, modes={package.run_modes}, "
        f"providers={package.providers}, device={package.device_type}"
    )
    for mode in package.run_modes:
        _run_mode(package, tokenizer, mode)
    print_progress(
        f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s."
    )


if __name__ == "__main__":
    main()