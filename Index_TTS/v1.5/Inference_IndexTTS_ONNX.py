"""Run the compact IndexTTS v1.5 ONNX pipeline."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Example_Audio import reference_audio_path  # noqa: E402
from Shared_Weights import attach_shared_initializers  # noqa: E402


DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "IndexTTS_Optimized",
        help="Folder containing exported or optimized IndexTTS graphs.",
    )
    return parser.parse_args()


# User configuration: edit these values directly; CLI only selects the ONNX folder.
PROJECT_PATH = Path.home() / "Downloads" / "index-tts-main"                  # Official IndexTTS project.
TOKENIZER_PATH = Path.home() / "Downloads" / "IndexTTS-1.5" / "bpe.model"    # SentencePiece model.
REFERENCE_AUDIO_PATH = Path(reference_audio_path("indextts"))                 # Voice to clone.
TARGET_TEXT = "大家好，我现在正在大可奇奇体验 ai 科技。"                            # Text to synthesize.
GENERATED_AUDIO_PATH = SCRIPT_DIR / "generated.wav"                           # Output WAV path.

# Exactly one strategy is loaded. Options: greedy | penalty_greedy | sampling.
DECODE_STRATEGY = "sampling"
PENALTY_VALUE = 0.8                     # Multiplicative penalty; penalty_greedy only.
PENALTY_RANGE = 10                      # Recent-token window; penalty_greedy only.
SAMPLING_TEMPERATURE = 0.8              # Higher values produce more variation.
SAMPLING_TOP_K = 20                     # Candidate count; sampling only.
SAMPLING_TOP_P = 0.95                   # Nucleus threshold; sampling only, range (0, 1].
SAMPLING_REPETITION_PENALTY = 1.05      # Repetition control; sampling only.
MAX_TOKENS = 600                        # Per-segment limit; 0 uses full graph capacity.
MAX_TEXT_TOKENS_PER_SEGMENT = 120       # Long text is split at this token count.

# ONNX Runtime settings
ORT_LOG = False                         # True enables verbose ONNX Runtime logging.
ORT_FP16 = False                        # True preserves FP16 graph transforms where supported.
ORT_ACCELERATE_PROVIDERS = []           # [] uses CPU; e.g. ["CUDAExecutionProvider"].
MAX_THREADS = 0                         # CPU and OpenVINO thread count.
DEVICE_ID = 0                           # Accelerator device index.
SHOW_PROGRESS = True                    # Print pipeline stages and loop progress.


def print_progress(message: str) -> None:
    if SHOW_PROGRESS:
        print(f"[IndexTTS] {message}", flush=True)


def metadata_from(path: Path) -> dict[str, str]:
    model = onnx.load(str(path), load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    del model
    return metadata


def provider_configuration():
    providers = list(ORT_ACCELERATE_PROVIDERS) or ["CPUExecutionProvider"]
    provider_options = []
    for provider in providers:
        if provider == "CUDAExecutionProvider":
            provider_options.append(
                {
                    "device_id": DEVICE_ID,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "use_tf32": "1",
                }
            )
        elif provider == "DmlExecutionProvider":
            provider_options.append({"device_id": DEVICE_ID})
        elif provider == "OpenVINOExecutionProvider":
            provider_options.append(
                {
                    "device_type": "CPU",
                    "precision": "ACCURACY",
                    "num_of_threads": MAX_THREADS if MAX_THREADS else 8,
                    "num_streams": 1,
                }
            )
        else:
            provider_options.append({})

    primary = providers[0]
    if primary in {
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
    }:
        device_type = "cuda"
        raw_device = C.OrtDevice.cuda()
    elif primary == "DmlExecutionProvider":
        device_type = "dml"
        raw_device = C.OrtDevice.dml()
    else:
        device_type = "cpu"
        raw_device = C.OrtDevice.cpu()
    device_id = DEVICE_ID if device_type != "cpu" else 0
    device = C.OrtDevice(raw_device, C.OrtDevice.default_memory(), device_id)
    return providers, provider_options, device_type, device


def session_configuration(metadata: dict[str, str]):
    options = ort.SessionOptions()
    run_options = ort.RunOptions()
    for value in (options, run_options):
        value.log_severity_level = 0 if ORT_LOG else 4
        value.log_verbosity_level = 4
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in (
        ("session.set_denormal_as_zero", "1"),
        ("session.intra_op.allow_spinning", "1"),
        ("session.inter_op.allow_spinning", "1"),
        ("session.use_device_allocator_for_initializers", "1"),
        ("session.qdq_matmulnbits_accuracy_level", "2" if ORT_FP16 else "4"),
    ):
        options.add_session_config_entry(key, value)

    preserve_fp16_attention = (
        metadata["use_f16_kv"] == "1" and metadata["compute_in_f32"] == "0"
    )
    disabled_optimizers = None
    if preserve_fp16_attention or ORT_FP16:
        disabled_optimizers = [
            "CastFloat16Transformer",
            "FuseFp16InitializerToFp32NodeTransformer",
        ]
        options.add_session_config_entry(
            "optimization.disable_specified_optimizers",
            ";".join(disabled_optimizers),
        )
    if preserve_fp16_attention:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options, run_options, disabled_optimizers


def numpy_dtype(argument):
    value_type = argument.type
    try:
        tensor_type = onnx.TensorProto.DataType.Value(value_type[7:-1].upper())
        return onnx.helper.tensor_dtype_to_np_dtype(tensor_type)
    except ValueError as exc:
        raise ValueError(f"Unsupported ONNX tensor type: {value_type!r}.") from exc
def static_shape(argument):
    shape = tuple(argument.shape)
    return shape if all(isinstance(dim, int) for dim in shape) else None


def model_array(argument, data):
    array = np.asarray(data, dtype=numpy_dtype(argument))
    declared = tuple(argument.shape)
    shape_matches = array.ndim == len(declared) and all(
        not isinstance(dim, int) or array.shape[axis] == dim
        for axis, dim in enumerate(declared)
    )
    if not shape_matches:
        shape = tuple(dim if isinstance(dim, int) else -1 for dim in declared)
        array = array.reshape(shape)
    return np.ascontiguousarray(array)


def ortvalue(argument, data, device_type):
    device_id = DEVICE_ID if device_type != "cpu" else 0
    return ort.OrtValue.ortvalue_from_numpy(
        model_array(argument, data),
        device_type,
        device_id,
    )


def empty_ortvalue(argument, device_type):
    device_id = DEVICE_ID if device_type != "cpu" else 0
    return ort.OrtValue.ortvalue_from_shape_and_type(
        static_shape(argument),
        numpy_dtype(argument),
        device_type,
        device_id,
    )


def constant_ortvalue(argument, data, device_type, cache):
    array = model_array(argument, data)
    key = (array.dtype.str, array.shape, array.tobytes())
    value = cache.get(key)
    if value is None:
        device_id = DEVICE_ID if device_type != "cpu" else 0
        value = ort.OrtValue.ortvalue_from_numpy(array, device_type, device_id)
        cache[key] = value
    return value


class RuntimeBinding:
    def __init__(
        self,
        session,
        run_options,
        device_type,
        device,
        output_buffers=None,
    ):
        self.session = session
        self.run_options = run_options
        self.device = device
        self.inputs = tuple(session.get_inputs())
        self.outputs = tuple(session.get_outputs())
        self.output_positions = {
            argument.name: index for index, argument in enumerate(self.outputs)
        }
        self.binding = session.io_binding()
        self.output_buffers = {} if device_type == "cuda" else dict(output_buffers or {})
        self.auto_bound_outputs = []
        for argument in self.outputs:
            value = self.output_buffers.get(argument.name)
            if value is None and device_type != "cuda" and static_shape(argument) is not None:
                value = empty_ortvalue(argument, device_type)
                self.output_buffers[argument.name] = value
            if value is None:
                self.binding._iobinding.bind_output(argument.name, device)
                self.auto_bound_outputs.append(argument)
            else:
                self.binding.bind_ortvalue_output(argument.name, value)
        self.has_run = False

    def bind(self, arguments, values):
        for argument, value in zip(arguments, values, strict=True):
            self.binding.bind_ortvalue_input(argument.name, value)

    def run(self):
        # Auto-bound outputs become shape-owned after a run and must be refreshed.
        if self.has_run:
            for argument in self.auto_bound_outputs:
                self.binding._iobinding.bind_output(argument.name, self.device)
        self.session.run_with_iobinding(self.binding, run_options=self.run_options)
        self.has_run = True
        return self.binding.get_outputs()

    def value(self, values, argument):
        return values[self.output_positions[argument.name]]


def reusable_buffer(source, source_argument, target_argument):
    if (
        source_argument.type == target_argument.type
        and static_shape(source_argument) == static_shape(target_argument)
    ):
        return source.output_buffers.get(source_argument.name)
    return None


def create_tokenizer():
    project_path = PROJECT_PATH.expanduser().resolve()
    if str(project_path) not in sys.path:
        sys.path.insert(0, str(project_path))
    from indextts.utils.front import TextNormalizer, TextTokenizer

    try:
        return TextTokenizer(
            str(TOKENIZER_PATH.expanduser().resolve()),
            TextNormalizer(),
        )
    except ImportError as exc:
        warnings.warn(
            f"Text normalization is unavailable ({exc}); using SentencePiece only.",
            RuntimeWarning,
        )
        return TextTokenizer(str(TOKENIZER_PATH.expanduser().resolve()), None)


def load_reference_audio(argument, sample_rate, device_type):
    input_shape = tuple(argument.shape)
    batch = int(input_shape[0])
    channels = int(input_shape[1])
    segment = (
        AudioSegment.from_file(REFERENCE_AUDIO_PATH.expanduser().resolve())
        .set_channels(channels)
        .set_frame_rate(sample_rate)
        .set_sample_width(2)
    )
    samples = np.asarray(segment.get_array_of_samples())
    samples = samples.reshape(-1, channels).T
    samples = np.broadcast_to(samples, (batch, *samples.shape)).copy()
    if np.issubdtype(numpy_dtype(argument), np.floating):
        samples = (samples.astype(np.float32) * (1.0 / 32768.0)).astype(
            numpy_dtype(argument)
        )
    return ortvalue(argument, samples, device_type)


def main() -> None:
    pipeline_started = time.perf_counter()
    args = parse_args()
    onnx_folder = args.onnx_folder.expanduser().resolve()
    metadata_path = onnx_folder / "IndexTTS_Metadata.onnx"
    print_progress(f"Reading package metadata: {metadata_path}")
    metadata = metadata_from(metadata_path)
    expected_metadata_keys = {
        "graph_layout",
        "in_sample_rate",
        "out_sample_rate",
        "stop_token_ids",
        "max_signal_length",
        "mel_code_size",
        "use_f16_kv",
        "compute_in_f32",
        "shared_initializer_model_file",
        "shared_initializer_data_file",
        "model_file_name_reference_preprocess",
        "model_file_name_target_preprocess",
        "model_file_name_decoder",
        *{
            f"model_file_name_main_prefill_{strategy}"
            for strategy in DECODE_STRATEGIES
        },
        *{
            f"model_file_name_decode_step_{strategy}"
            for strategy in DECODE_STRATEGIES
        },
    }
    missing_metadata = sorted(expected_metadata_keys - metadata.keys())
    if missing_metadata:
        raise ValueError(
            f"{metadata_path.name} is missing required metadata key(s): {missing_metadata}."
        )
    in_sample_rate = int(metadata["in_sample_rate"])
    out_sample_rate = int(metadata["out_sample_rate"])
    stop_tokens = {int(token) for token in metadata["stop_token_ids"].split(",")}
    max_signal_length = int(metadata["max_signal_length"])
    sampling_top_k = min(SAMPLING_TOP_K, int(metadata["mel_code_size"]))

    paths = {
        "reference": onnx_folder / metadata["model_file_name_reference_preprocess"],
        "target": onnx_folder / metadata["model_file_name_target_preprocess"],
        "prefill": onnx_folder / metadata[
            f"model_file_name_main_prefill_{DECODE_STRATEGY}"
        ],
        "decode": onnx_folder / metadata[
            f"model_file_name_decode_step_{DECODE_STRATEGY}"
        ],
        "decoder": onnx_folder / metadata["model_file_name_decoder"],
    }

    shared_data_path = onnx_folder / metadata["shared_initializer_data_file"]
    providers, provider_options, device_type, device = provider_configuration()
    options, run_options, disabled_optimizers = session_configuration(metadata)
    shared_started = time.perf_counter()
    if device_type == "cpu":
        print_progress("Attaching shared ONNX initializers...")
        _shared_refs = attach_shared_initializers(
            options,
            onnx_folder / metadata["shared_initializer_model_file"],
        )
        print_progress(
            f"Shared ONNX initializers ready in "
            f"{time.perf_counter() - shared_started:.2f}s."
        )
    else:
        _shared_refs = ()
        print_progress(
            "Using native ONNX external shared initializers in "
            f"{time.perf_counter() - shared_started:.2f}s."
        )

    def load(path):
        return ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=providers,
            provider_options=provider_options,
            disabled_optimizers=disabled_optimizers,
        )

    session_started = time.perf_counter()
    sessions = {}
    for index, (name, path) in enumerate(paths.items(), start=1):
        print_progress(f"Loading ONNX graph {index}/{len(paths)}: {path.name}")
        sessions[name] = load(path)
    print_progress(
        f"ONNX Runtime sessions ready in "
        f"{time.perf_counter() - session_started:.2f}s; "
        f"provider={sessions['decode'].get_providers()[0]}."
    )

    reference = RuntimeBinding(
        sessions["reference"], run_options, device_type, device
    )
    target = RuntimeBinding(sessions["target"], run_options, device_type, device)
    prefill = RuntimeBinding(sessions["prefill"], run_options, device_type, device)
    decoder = RuntimeBinding(sessions["decoder"], run_options, device_type, device)

    (target_hidden_output,) = target.outputs
    prefill_hidden_input = next(
        argument for argument in prefill.inputs
        if argument.name == target_hidden_output.name
    )
    prefill_control_inputs = tuple(
        argument for argument in prefill.inputs
        if argument is not prefill_hidden_input
    )

    decode_inputs = tuple(sessions["decode"].get_inputs())
    decode_outputs = tuple(sessions["decode"].get_outputs())
    decode_state_inputs = tuple(
        argument for argument in decode_inputs if len(argument.shape) == 4
    )
    decode_state_outputs = tuple(
        argument for argument in decode_outputs if len(argument.shape) == 4
    )
    decode_flow_inputs = tuple(
        argument for argument in decode_inputs if len(argument.shape) != 4
    )
    decode_flow_outputs = tuple(
        argument for argument in decode_outputs if len(argument.shape) != 4
    )
    (
        decode_token_input,
        decode_saved_input,
        decode_history_input,
        *decode_control_inputs,
    ) = decode_flow_inputs
    (
        decode_last_hidden_output,
        decode_token_output,
        decode_saved_output,
        decode_history_output,
    ) = decode_flow_outputs

    prefill_state_outputs = tuple(
        argument for argument in prefill.outputs if len(argument.shape) == 4
    )
    (
        prefill_last_hidden_output,
        prefill_token_output,
        prefill_history_output,
    ) = tuple(
        argument for argument in prefill.outputs if len(argument.shape) != 4
    )

    shared_last_hidden = reusable_buffer(
        prefill,
        prefill_last_hidden_output,
        decode_last_hidden_output,
    )
    decode_first_overrides = {}
    if shared_last_hidden is not None:
        decode_first_overrides[decode_last_hidden_output.name] = shared_last_hidden
    decode_first = RuntimeBinding(
        sessions["decode"],
        run_options,
        device_type,
        device,
        decode_first_overrides,
    )

    decode_second_overrides = dict(decode_first_overrides)
    for source_argument, target_argument in (
        (prefill_token_output, decode_token_output),
        (prefill_history_output, decode_history_output),
    ):
        value = reusable_buffer(prefill, source_argument, target_argument)
        if value is not None:
            decode_second_overrides[target_argument.name] = value
    decode_second = RuntimeBinding(
        sessions["decode"],
        run_options,
        device_type,
        device,
        decode_second_overrides,
    )
    decode_bindings = (decode_first, decode_second)

    if DECODE_STRATEGY == "greedy":
        decode_control_data = ()
    elif DECODE_STRATEGY == "penalty_greedy":
        decode_control_data = (PENALTY_VALUE, PENALTY_RANGE)
    else:
        decode_control_data = (
            SAMPLING_TEMPERATURE,
            sampling_top_k,
            SAMPLING_TOP_P,
            SAMPLING_REPETITION_PENALTY,
        )

    constant_cache = {}
    decode_control_values = tuple(
        constant_ortvalue(argument, value, device_type, constant_cache)
        for argument, value in zip(
            decode_control_inputs,
            decode_control_data,
            strict=True,
        )
    )
    for binding in decode_bindings:
        binding.bind(decode_control_inputs, decode_control_values)

    prefill_control_data = decode_control_data[:len(prefill_control_inputs)]
    prefill_control_values = tuple(
        constant_ortvalue(argument, value, device_type, constant_cache)
        for argument, value in zip(
            prefill_control_inputs,
            prefill_control_data,
            strict=True,
        )
    )
    prefill.bind(prefill_control_inputs, prefill_control_values)

    (reference_audio_input,) = reference.inputs
    reference_started = time.perf_counter()
    print_progress(f"Preparing reference audio: {REFERENCE_AUDIO_PATH}")
    audio = load_reference_audio(reference_audio_input, in_sample_rate, device_type)
    reference.bind(reference.inputs, (audio,))
    start_time = time.time()
    reference_results = reference.run()
    reference_values = {
        argument.name: reference.value(reference_results, argument)
        for argument in reference.outputs
    }
    print_progress(
        f"Reference conditioning ready in "
        f"{time.perf_counter() - reference_started:.2f}s."
    )

    (target_condition_input,) = tuple(
        argument for argument in target.inputs
        if argument.name in reference_values
    )
    (target_text_input,) = tuple(
        argument for argument in target.inputs
        if argument is not target_condition_input
    )
    target.bind(
        (target_condition_input,),
        (reference_values[target_condition_input.name],),
    )

    decoder_condition_inputs = tuple(
        argument for argument in decoder.inputs
        if argument.name in reference_values
    )
    (decoder_hidden_input,) = tuple(
        argument for argument in decoder.inputs
        if argument.name not in reference_values
    )
    decoder.bind(
        decoder_condition_inputs,
        tuple(reference_values[argument.name] for argument in decoder_condition_inputs),
    )

    print_progress("Loading the text frontend...")
    tokenizer = create_tokenizer()
    text_tokens = tokenizer.tokenize(TARGET_TEXT)
    segments = tokenizer.split_segments(
        text_tokens,
        MAX_TEXT_TOKENS_PER_SEGMENT,
    )
    print_progress(f"Prepared {len(segments)} text segment(s).")

    generate_limit = MAX_TOKENS or max_signal_length
    decoder_hidden_shape = tuple(decoder_hidden_input.shape)
    (decoder_sequence_axis,) = tuple(
        axis
        for axis, dim in enumerate(decoder_hidden_shape)
        if not isinstance(dim, int)
    )
    latent_buffer_shape = tuple(
        generate_limit if axis == decoder_sequence_axis else int(dim)
        for axis, dim in enumerate(decoder_hidden_shape)
    )
    latent_step_shape = tuple(
        dim
        for axis, dim in enumerate(latent_buffer_shape)
        if axis != decoder_sequence_axis
    )
    latent_step_index = [slice(None)] * len(latent_buffer_shape)
    latent_sequence_slice = [slice(None)] * len(latent_buffer_shape)
    latent_buffer = np.empty(
        latent_buffer_shape,
        dtype=numpy_dtype(decoder_hidden_input),
    )
    generated_segments = []

    for segment_index, segment in enumerate(segments):
        split_text = "".join(segment).replace("▁", " ")
        print_progress(
            f"Starting segment {segment_index + 1}/{len(segments)}: {split_text!r}"
        )

        token_ids = tokenizer.convert_tokens_to_ids(segment)
        text_ids = ortvalue(target_text_input, token_ids, device_type)
        target.bind((target_text_input,), (text_ids,))
        target_results = target.run()
        hidden_states = target.value(target_results, target_hidden_output)

        prefill.bind((prefill_hidden_input,), (hidden_states,))
        decode_start = time.time()
        prefill_results = prefill.run()
        states = tuple(
            prefill.value(prefill_results, argument)
            for argument in prefill_state_outputs
        )
        last_hidden = prefill.value(prefill_results, prefill_last_hidden_output)
        current_token = prefill.value(prefill_results, prefill_token_output)
        saved_tokens = current_token
        history_length = prefill.value(prefill_results, prefill_history_output)

        prompt_length = int(history_length.numpy().item())
        segment_limit = min(generate_limit, max_signal_length - prompt_length)
        if segment_limit <= 0:
            warnings.warn(
                f"Segment {segment_index} consumed all attention capacity; skipping it.",
                RuntimeWarning,
            )
            continue
        print_progress(f"Generating semantic codes (limit {segment_limit})...")

        accepted_tokens = 0
        decode_calls = 0
        selected_token = int(current_token.numpy().item())
        stopped = selected_token in stop_tokens
        if not stopped:
            latent_step_index[decoder_sequence_axis] = 0
            latent_buffer[tuple(latent_step_index)] = last_hidden.numpy().reshape(
                latent_step_shape
            )
            accepted_tokens = 1

        while not stopped and accepted_tokens < segment_limit:
            binding = decode_bindings[decode_calls & 1]
            binding.bind(decode_state_inputs, states)
            binding.bind(
                (decode_token_input, decode_saved_input, decode_history_input),
                (current_token, saved_tokens, history_length),
            )
            decode_results = binding.run()
            states = tuple(
                binding.value(decode_results, argument)
                for argument in decode_state_outputs
            )
            last_hidden = binding.value(
                decode_results,
                decode_last_hidden_output,
            )
            current_token = binding.value(decode_results, decode_token_output)
            saved_tokens = binding.value(decode_results, decode_saved_output)
            history_length = binding.value(decode_results, decode_history_output)
            decode_calls += 1
            selected_token = int(current_token.numpy().item())
            stopped = selected_token in stop_tokens
            if not stopped:
                latent_step_index[decoder_sequence_axis] = accepted_tokens
                latent_buffer[tuple(latent_step_index)] = last_hidden.numpy().reshape(
                    latent_step_shape
                )
                accepted_tokens += 1
                if accepted_tokens % 50 == 0 or accepted_tokens == segment_limit:
                    print_progress(
                        f"Semantic generation: {accepted_tokens}/{segment_limit} codes"
                    )

        elapsed = max(time.time() - decode_start, 1.0e-9)
        print_progress(
            f"Semantic generation complete: {accepted_tokens} codes in {elapsed:.2f}s."
        )
        print(
            f"Decode speed: {accepted_tokens / elapsed:.3f} tokens/s",
            flush=True,
        )
        if not accepted_tokens:
            warnings.warn(
                f"No mel tokens were generated for segment {segment_index}; skipping it.",
                RuntimeWarning,
            )
            continue

        latent_sequence_slice[decoder_sequence_axis] = slice(accepted_tokens)
        hidden_value = ortvalue(
            decoder_hidden_input,
            latent_buffer[tuple(latent_sequence_slice)],
            device_type,
        )
        decoder.bind((decoder_hidden_input,), (hidden_value,))
        print_progress("Decoding waveform...")
        decoder_results = decoder.run()
        generated_segments.append(
            np.array(decoder.value(decoder_results, decoder.outputs[0]).numpy(), copy=True)
        )

    generated_audio = (
        generated_segments[0]
        if len(generated_segments) == 1
        else np.concatenate(generated_segments, axis=-1)
    )
    output_path = GENERATED_AUDIO_PATH.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print_progress(f"Writing generated audio: {output_path}")
    if generated_audio.dtype == np.float16:
        generated_audio = generated_audio.astype(np.float32)
    output_subtype = "PCM_16" if np.issubdtype(generated_audio.dtype, np.integer) else "FLOAT"
    sf.write(
        str(output_path),
        generated_audio.reshape(-1),
        out_sample_rate,
        subtype=output_subtype,
        format="WAVEX",
    )
    audio_duration = generated_audio.size / out_sample_rate
    rtf = (time.time() - start_time) / max(audio_duration, 1.0e-9)
    print(
        f"Audio generation is complete: {output_path}; ONNX Runtime RTF={rtf:.3f}.",
        flush=True,
    )
    print_progress(
        f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s."
    )


if __name__ == "__main__":
    main()