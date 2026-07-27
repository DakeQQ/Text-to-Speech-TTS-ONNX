"""Run the compact strategy-based KaniTTS ONNX package."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import soundfile as sf
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer

from Shared_Weights import attach_shared_initializers


SCRIPT_DIR = Path(__file__).resolve().parent
DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")


# ============================== Configuration ==============================
# Edit these values directly; the CLI is reserved for selecting the ONNX folder.
TARGET_TTS = [
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology.",
]
SPEAKER = "jenny"                                      # Speaker name from the model vocabulary.
TOKENIZER_PATH = Path.home() / "Downloads" / "kani-tts-370m"
OUTPUT_PATH = SCRIPT_DIR / "generated.wav"             # Output WAV path.

DECODE_STRATEGY = "sampling"                           # greedy | penalty_greedy | sampling
PENALTY_VALUE = 0.8                                    # Penalty-greedy score multiplier.
PENALTY_RANGE = 10                                     # Recent-token penalty window.
TEMPERATURE = 0.8                                      # Sampling temperature.
TOP_K = 20                                             # Sampling candidate count.
TOP_P = 0.95                                           # Sampling nucleus probability.
REPETITION_PENALTY = 1.1                               # Sampling repetition penalty.
MAX_TOKENS = 0                                         # 0 uses the exported sequence capacity.

ORT_ACCELERATE_PROVIDERS = []                          # [] uses CPU; e.g. ['CUDAExecutionProvider'].
MAX_THREADS = 0                                        # CPU parallel threads; 0 lets ORT choose.
DEVICE_ID = 0                                          # Accelerator device index.
ORT_LOG = False                                        # Enable ONNX Runtime logging.
SHOW_PROGRESS = True                                   # Print pipeline stages and loop progress.
# ===========================================================================


def print_progress(message):
    if SHOW_PROGRESS:
        print(f"[KaniTTS] {message}", flush=True)


ORT_TYPE_TO_DTYPE = {
    "tensor(bool)": np.bool_,
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run compact KaniTTS ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "KaniTTS_Optimized",
        help="Folder containing the compact ONNX package.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing KaniTTS metadata graph: {path}")
    model = onnx.load(str(path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


def require_metadata(metadata: dict[str, str], key: str) -> str:
    try:
        return metadata[key]
    except KeyError as exc:
        raise KeyError(f"Compact KaniTTS metadata is missing required key {key!r}.") from exc


def meta_int(metadata: dict[str, str], key: str) -> int:
    return int(require_metadata(metadata, key))


def meta_bool(metadata: dict[str, str], key: str) -> bool:
    value = require_metadata(metadata, key)
    if value not in {"0", "1"}:
        raise ValueError(f"Metadata key {key!r} must be 0 or 1, got {value!r}.")
    return value == "1"


def meta_int_list(metadata: dict[str, str], key: str) -> list[int]:
    return [int(value) for value in require_metadata(metadata, key).split(",") if value]


def io_dtype(argument):
    try:
        return ORT_TYPE_TO_DTYPE[argument.type]
    except KeyError as exc:
        raise TypeError(f"Unsupported ONNX Runtime tensor type {argument.type!r}.") from exc


def io_shape(argument):
    if any(not isinstance(dimension, int) for dimension in argument.shape):
        raise ValueError(f"{argument.name} does not have a static shape: {argument.shape}.")
    return tuple(argument.shape)


def numpy_for(argument, data):
    return np.ascontiguousarray(np.asarray(data, dtype=io_dtype(argument)))


def ortvalue_for(argument, data, device_type):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        numpy_for(argument, data),
        device_type,
        DEVICE_ID,
    )


def empty_ortvalue_for(argument, device_type, shape_argument=None):
    if shape_argument is None:
        shape_argument = argument
    return onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        io_shape(shape_argument),
        io_dtype(argument),
        device_type,
        DEVICE_ID,
    )


def constant_ortvalue(argument, value, device_type):
    array = np.full(io_shape(argument), value, dtype=io_dtype(argument))
    return onnxruntime.OrtValue.ortvalue_from_numpy(array, device_type, DEVICE_ID)


def bind_inputs(binding, arguments, values):
    for argument, value in zip(arguments, values, strict=True):
        binding.bind_ortvalue_input(argument.name, value)


def bind_outputs(binding, arguments, device, buffers=None):
    buffers = buffers or {}
    for index, argument in enumerate(arguments):
        if index in buffers:
            binding.bind_ortvalue_output(argument.name, buffers[index])
        else:
            binding._iobinding.bind_output(argument.name, device)


def rebind_outputs(binding, arguments, device, buffers=None):
    binding.clear_binding_outputs()
    bind_outputs(binding, arguments, device, buffers)


def run_binding(session, binding, run_options):
    session.run_with_iobinding(binding, run_options=run_options)
    return binding.get_outputs()


def sequence_fragment(argument, values):
    batch_size = argument.shape[0]
    if not isinstance(batch_size, int):
        raise ValueError(f"{argument.name} must have a static batch dimension.")
    row = np.asarray(values, dtype=io_dtype(argument))
    return np.broadcast_to(row, (batch_size, row.size)).copy()


def build_session_options(preserve_fp16_attention):
    session_options = onnxruntime.SessionOptions()
    run_options = onnxruntime.RunOptions()
    for options in (session_options, run_options):
        options.log_severity_level = 0 if ORT_LOG else 4
        options.log_verbosity_level = 4
    session_options.inter_op_num_threads = MAX_THREADS
    session_options.intra_op_num_threads = MAX_THREADS
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    config_entries = {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.minimal_build_optimizations": "",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": (
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
            if preserve_fp16_attention
            else ""
        ),
    }
    for key, value in config_entries.items():
        session_options.add_session_config_entry(key, value)
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    disabled_optimizers = (
        ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
        if preserve_fp16_attention
        else None
    )
    return session_options, run_options, disabled_optimizers


def configure_provider():
    if "OpenVINOExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
        provider_options = [{
            "device_type": "CPU",
            "precision": "ACCURACY",
            "num_of_threads": MAX_THREADS if MAX_THREADS else 8,
            "num_streams": 1,
            "enable_opencl_throttling": False,
            "enable_qdq_optimizer": False,
            "disable_dynamic_shapes": False,
        }]
        return provider_options, "cpu", C.OrtDevice.cpu()
    if "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
        provider_options = [{
            "device_id": DEVICE_ID,
            "gpu_mem_limit": 24 * 1024**3,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "use_tf32": "1",
            "do_copy_in_default_stream": "0",
            "enable_cuda_graph": "0",
        }]
        return provider_options, "cuda", C.OrtDevice.cuda()
    if "DmlExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
        provider_options = [{
            "device_id": DEVICE_ID,
            "performance_preference": "high_performance",
            "device_filter": "gpu",
            "disable_metacommands": "false",
            "enable_graph_capture": "false",
        }]
        return provider_options, "dml", C.OrtDevice.dml()
    return None, "cpu", C.OrtDevice.cpu()


def validate_strategy_settings(vocab_size):
    if DECODE_STRATEGY not in DECODE_STRATEGIES:
        raise ValueError(
            f"Unsupported DECODE_STRATEGY {DECODE_STRATEGY!r}; choose one of {DECODE_STRATEGIES}."
        )
    if MAX_TOKENS < 0:
        raise ValueError("MAX_TOKENS must be zero or positive.")
    if DECODE_STRATEGY == "penalty_greedy" and (
        PENALTY_VALUE <= 0.0 or PENALTY_RANGE < 1
    ):
        raise ValueError("Penalty-greedy requires PENALTY_VALUE > 0 and PENALTY_RANGE >= 1.")
    if DECODE_STRATEGY == "sampling":
        if TEMPERATURE <= 0.0:
            raise ValueError("Sampling requires TEMPERATURE > 0.")
        if TOP_K < 1:
            raise ValueError("Sampling requires TOP_K >= 1.")
        if not 0.0 < TOP_P <= 1.0:
            raise ValueError("Sampling requires 0 < TOP_P <= 1.")
        if REPETITION_PENALTY <= 0.0:
            raise ValueError("Sampling requires REPETITION_PENALTY > 0.")
    return min(TOP_K, vocab_size)


def strategy_control_data(top_k):
    if DECODE_STRATEGY == "greedy":
        return ()
    if DECODE_STRATEGY == "penalty_greedy":
        return PENALTY_VALUE, PENALTY_RANGE
    return TEMPERATURE, top_k, TOP_P, REPETITION_PENALTY


def main():
    pipeline_started = time.perf_counter()
    args = parse_args()
    onnx_folder = args.onnx_folder.expanduser().resolve()
    metadata_path = onnx_folder / "KaniTTS_Metadata.onnx"
    print_progress(f"Reading package metadata: {metadata_path}")
    metadata = read_metadata(metadata_path)
    expected_metadata_keys = {
        "graph_layout",
        "max_seq_len",
        "stop_token_ids",
        "prompt_prefix_token_ids",
        "prompt_suffix_token_ids",
        "shared_initializer_model_file",
        "shared_initializer_data_file",
        "model_file_name_codec",
        "vocab_size",
        "use_float16_kv",
        "compute_in_f32",
        "out_sample_rate",
        "codec_prefix_token_count",
        "codec_token_alignment",
        *{
            f"model_file_name_main_prefill_{strategy}"
            for strategy in DECODE_STRATEGIES
        },
        *{
            f"model_file_name_decode_step_{strategy}"
            for strategy in DECODE_STRATEGIES
        },
    }
    if set(metadata) != expected_metadata_keys:
        raise ValueError(
            "KaniTTS metadata keys do not match the runtime contract: "
            f"missing={sorted(expected_metadata_keys - set(metadata))}, "
            f"extra={sorted(set(metadata) - expected_metadata_keys)}."
        )
    if metadata.get("graph_layout") != "strategy_prefill_decode_step":
        raise ValueError(
            "KaniTTS strategy_prefill_decode_step metadata is required; "
            "re-export before inference."
        )

    top_k = validate_strategy_settings(meta_int(metadata, "vocab_size"))
    preserve_fp16_attention = (
        meta_bool(metadata, "use_float16_kv") and not meta_bool(metadata, "compute_in_f32")
    )
    prefill_path = onnx_folder / require_metadata(
        metadata,
        f"model_file_name_main_prefill_{DECODE_STRATEGY}",
    )
    decode_path = onnx_folder / require_metadata(
        metadata,
        f"model_file_name_decode_step_{DECODE_STRATEGY}",
    )
    session_options, run_options, disabled_optimizers = build_session_options(
        preserve_fp16_attention
    )
    provider_options, device_type, ort_device_type = configure_provider()
    ort_device = C.OrtDevice(ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)

    shared_model_path = onnx_folder / require_metadata(metadata, "shared_initializer_model_file")
    shared_data_path = onnx_folder / require_metadata(metadata, "shared_initializer_data_file")
    if not shared_data_path.is_file():
        raise FileNotFoundError(f"Missing shared initializer data: {shared_data_path}")
    # Shared mmap arrays and OrtValues must live as long as the graph sessions.
    shared_started = time.perf_counter()
    print_progress("Attaching shared ONNX initializers...")
    shared_refs = attach_shared_initializers(session_options, shared_model_path)
    print_progress(
        f"Shared ONNX initializers ready in "
        f"{time.perf_counter() - shared_started:.2f}s."
    )

    def create_session(path):
        return onnxruntime.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=ORT_ACCELERATE_PROVIDERS,
            provider_options=provider_options,
            disabled_optimizers=disabled_optimizers,
        )

    codec_path = onnx_folder / require_metadata(metadata, "model_file_name_codec")
    session_started = time.perf_counter()
    sessions_to_load = (prefill_path, decode_path, codec_path)
    loaded_sessions = []
    for index, path in enumerate(sessions_to_load, start=1):
        print_progress(f"Loading ONNX graph {index}/3: {path.name}")
        loaded_sessions.append(create_session(path))
    prefill_session, decode_session, codec_session = loaded_sessions
    print_progress(
        f"ONNX Runtime sessions ready in "
        f"{time.perf_counter() - session_started:.2f}s."
    )

    prefill_inputs = tuple(prefill_session.get_inputs())
    prefill_output_args = tuple(prefill_session.get_outputs())
    decode_inputs = tuple(decode_session.get_inputs())
    decode_output_args = tuple(decode_session.get_outputs())
    codec_inputs = tuple(codec_session.get_inputs())
    codec_output_args = tuple(codec_session.get_outputs())

    state_count = len(prefill_output_args) - 2
    prefill_input = prefill_inputs[0]
    decode_state_inputs = decode_inputs[:state_count]
    decode_token_input, decode_save_input, decode_length_input = decode_inputs[
        state_count:state_count + 3
    ]
    decode_control_inputs = decode_inputs[state_count + 3:]
    decode_state_outputs = decode_output_args[:state_count]
    decode_token_output = decode_output_args[state_count]
    decode_length_output = decode_output_args[-1]
    codec_decode_input, codec_count_input = codec_inputs
    codec_audio_output = codec_output_args[0]

    control_values = {
        argument.name: constant_ortvalue(argument, value, device_type)
        for argument, value in zip(
            decode_control_inputs,
            strategy_control_data(top_k),
            strict=True,
        )
    }
    static_state_indexes = tuple(
        index
        for index, argument in enumerate(decode_state_inputs)
        if all(isinstance(dimension, int) for dimension in argument.shape)
    )
    dynamic_state_indexes = tuple(
        index for index in range(state_count) if index not in static_state_indexes
    )
    state_buffers = tuple(
        {
            index: empty_ortvalue_for(
                decode_state_outputs[index],
                device_type,
                decode_state_inputs[index],
            )
            for index in static_state_indexes
        }
        for _ in range(2)
    )
    token_buffers = tuple(
        empty_ortvalue_for(decode_token_output, device_type, decode_token_input)
        for _ in range(2)
    )
    length_buffers = tuple(
        empty_ortvalue_for(decode_length_output, device_type, decode_length_input)
        for _ in range(2)
    )

    num_decode_array = np.zeros(
        io_shape(codec_count_input),
        dtype=io_dtype(codec_count_input),
    )
    num_decode_value = ortvalue_for(codec_count_input, num_decode_array, device_type)

    prefill_binding = prefill_session.io_binding()
    decode_bindings = (decode_session.io_binding(), decode_session.io_binding())
    codec_binding = codec_session.io_binding()

    bind_inputs(
        prefill_binding,
        prefill_inputs[1:],
        (control_values[argument.name] for argument in prefill_inputs[1:]),
    )
    prefill_output_buffers = {
        **{index: state_buffers[0][index] for index in static_state_indexes},
        state_count: token_buffers[0],
        state_count + 1: length_buffers[0],
    }
    decode_output_buffers = []
    for binding_index, binding in enumerate(decode_bindings):
        for state_index in static_state_indexes:
            binding.bind_ortvalue_input(
                decode_state_inputs[state_index].name,
                state_buffers[binding_index][state_index],
            )
        binding.bind_ortvalue_input(
            decode_token_input.name,
            token_buffers[binding_index],
        )
        binding.bind_ortvalue_input(
            decode_length_input.name,
            length_buffers[binding_index],
        )
        bind_inputs(
            binding,
            decode_control_inputs,
            (control_values[argument.name] for argument in decode_control_inputs),
        )
        output_index = 1 - binding_index
        decode_output_buffers.append(
            {
                **{
                    index: state_buffers[output_index][index]
                    for index in static_state_indexes
                },
                state_count: token_buffers[output_index],
                state_count + 2: length_buffers[output_index],
            }
        )
    codec_binding.bind_ortvalue_input(codec_count_input.name, num_decode_value)

    print_progress("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH.expanduser().resolve())
    head_ids = sequence_fragment(
        prefill_input,
        meta_int_list(metadata, "prompt_prefix_token_ids"),
    )
    tail_ids = sequence_fragment(
        prefill_input,
        meta_int_list(metadata, "prompt_suffix_token_ids"),
    )
    stop_token_set = set(meta_int_list(metadata, "stop_token_ids"))
    max_seq_len = meta_int(metadata, "max_seq_len")
    sample_rate = meta_int(metadata, "out_sample_rate")
    codec_prefix = meta_int(metadata, "codec_prefix_token_count")
    codec_alignment = meta_int(metadata, "codec_token_alignment")
    blank_shape = (
        codec_decode_input.shape[0],
        *codec_audio_output.shape[1:-1],
        int(sample_rate * 0.3),
    )
    if any(not isinstance(dimension, int) for dimension in blank_shape):
        raise ValueError("Codec audio batch and channel dimensions must be static.")
    blank_segment = np.zeros(blank_shape, dtype=io_dtype(codec_audio_output))

    generated_audio = []
    total_start = time.perf_counter()
    print_progress(f"Prepared {len(TARGET_TTS)} text target(s).")
    for sentence_index, sentence in enumerate(TARGET_TTS, start=1):
        prompt = f"{SPEAKER}: {sentence}"
        print_progress(f"Starting target {sentence_index}/{len(TARGET_TTS)}: {prompt!r}")
        token_ids = numpy_for(
            prefill_input,
            tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="np",
            )["input_ids"],
        )
        token_ids = np.concatenate((head_ids, token_ids, tail_ids), axis=-1)
        input_ids = ortvalue_for(prefill_input, token_ids, device_type)
        prefill_binding.bind_ortvalue_input(prefill_input.name, input_ids)
        rebind_outputs(
            prefill_binding,
            prefill_output_args,
            ort_device,
            prefill_output_buffers,
        )

        generation_start = time.perf_counter()
        prefill_values = run_binding(prefill_session, prefill_binding, run_options)

        states = list(prefill_values[:state_count])
        save_ids = token_buffers[0]
        active_buffer = 0
        prefill_kv_seq_len = int(length_buffers[0].numpy().flat[0])

        generation_capacity = max_seq_len - prefill_kv_seq_len
        if MAX_TOKENS:
            generation_capacity = min(generation_capacity, MAX_TOKENS)
        if generation_capacity <= 0:
            print_progress("Generation skipped: prompt leaves no generation capacity.")
            continue
        print_progress(f"Generating semantic codes (limit {generation_capacity})...")

        selected_token_id = int(token_buffers[active_buffer].numpy().flat[0])
        accepted_tokens = 0
        decode_calls = 0
        while accepted_tokens < generation_capacity:
            if selected_token_id in stop_token_set:
                break
            accepted_tokens += 1
            if accepted_tokens % 50 == 0 or accepted_tokens == generation_capacity:
                print_progress(
                    f"Semantic generation: {accepted_tokens}/{generation_capacity} codes"
                )
            if accepted_tokens >= generation_capacity:
                break

            binding = decode_bindings[active_buffer]
            for state_index in dynamic_state_indexes:
                binding.bind_ortvalue_input(
                    decode_state_inputs[state_index].name,
                    states[state_index],
                )
            binding.bind_ortvalue_input(
                decode_save_input.name,
                save_ids,
            )
            rebind_outputs(
                binding,
                decode_output_args,
                ort_device,
                decode_output_buffers[active_buffer],
            )
            decode_values = run_binding(decode_session, binding, run_options)
            states = list(decode_values[:state_count])
            save_ids = decode_values[state_count + 1]
            active_buffer = 1 - active_buffer
            decode_calls += 1
            selected_token_id = int(token_buffers[active_buffer].numpy().flat[0])

        elapsed = time.perf_counter() - generation_start
        if accepted_tokens <= codec_prefix:
            print_progress(
                f"Generation skipped: only {accepted_tokens} accepted token(s)."
            )
            continue
        print_progress(
            f"Semantic generation complete: {accepted_tokens} codes in {elapsed:.2f}s."
        )
        payload_tokens = accepted_tokens - codec_prefix
        if payload_tokens % codec_alignment:
            raise RuntimeError(
                f"Codec token stream is misaligned: accepted={accepted_tokens}, prefix={codec_prefix}, "
                f"payload={payload_tokens}, alignment={codec_alignment}. The count is not truncated."
            )

        num_decode_array.fill(accepted_tokens)
        num_decode_value.update_inplace(num_decode_array)
        codec_binding.bind_ortvalue_input(codec_decode_input.name, save_ids)
        rebind_outputs(codec_binding, (codec_audio_output,), ort_device)
        print_progress("Decoding waveform...")
        codec_values = run_binding(codec_session, codec_binding, run_options)
        audio = codec_values[0].numpy()
        generated_audio.extend([audio, blank_segment])
        transformer_calls = 1 + decode_calls
        print(
            f"  Decode: {accepted_tokens / elapsed:.3f} token/s; "
            f"accepted={accepted_tokens}; transformer calls/token="
            f"{transformer_calls / accepted_tokens:.3f}",
            flush=True,
        )

    if not generated_audio:
        raise RuntimeError("KaniTTS did not generate any aligned audio stream.")
    output_path = OUTPUT_PATH.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(generated_audio, axis=-1).reshape(-1)
    print_progress(f"Writing generated audio: {output_path}")
    if audio.dtype == np.float16:
        audio = audio.astype(np.float32)
    output_subtype = "PCM_16" if np.issubdtype(audio.dtype, np.integer) else "FLOAT"
    sf.write(
        output_path,
        audio,
        sample_rate,
        subtype=output_subtype,
        format="WAVEX",
    )
    elapsed = time.perf_counter() - total_start
    rtf = elapsed / (audio.size / sample_rate)
    print(
        f"Generate complete. Saved {output_path}. RTF: {rtf:.3f}.",
        flush=True,
    )
    print_progress(
        f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s."
    )
    _ = shared_refs


if __name__ == "__main__":
    main()
