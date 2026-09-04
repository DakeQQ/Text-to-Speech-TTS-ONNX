from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import onnx
import onnxruntime
import soundfile as sf
from onnxruntime.capi import _pybind_state as C
from transformers import GemmaTokenizerFast

from Shared_Weights import attach_shared_initializers


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Example_Audio import model_reference  # noqa: E402


DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Breeze TTS 2 ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "BreezeTTS_Optimized",
        help="Folder containing the exported or optimized Breeze TTS 2 graphs.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional folder containing text tokenizer files; defaults to --onnx-folder.",
    )
    return parser.parse_args()


ARGS = parse_args()


# Request and runtime configuration
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()
DOWNLOAD_PATH = (
    ARGS.tokenizer_path.expanduser().resolve()
    if ARGS.tokenizer_path is not None
    else ONNX_FOLDER
)
REFERENCE_AUDIO_PATH, REFERENCE_TEXT = model_reference("breeze_tts_zh")

# Demo 1: English voice clone with the bundled Chinese voice reference.
RUN_VOICE_CLONE_EN = True
VOICE_CLONE_EN_TEXT = "(sigh) It is good to hear your voice again after all this time."
VOICE_CLONE_EN_INSTRUCTION = "Speak clearly and naturally."
VOICE_CLONE_EN_CFG_SCALE = 1.0
VOICE_CLONE_EN_OUTPUT_PATH = SCRIPT_DIR / "generated_voice_clone_en.wav"

# Demo 2: Chinese voice clone.
RUN_VOICE_CLONE_ZH = True
VOICE_CLONE_ZH_TEXT = "[叹气] 没想到过了这么久，你还记得我的声音。"
VOICE_CLONE_ZH_INSTRUCTION = "清晰自然地说话。"
VOICE_CLONE_ZH_CFG_SCALE = 1.0
VOICE_CLONE_ZH_OUTPUT_PATH = SCRIPT_DIR / "generated_voice_clone_zh.wav"

# Demo 3: English voice design without reference audio.
RUN_VOICE_DESIGN_EN = True
VOICE_DESIGN_EN_TEXT = "(sigh) Welcome aboard. Your journey begins now."
VOICE_DESIGN_EN_INSTRUCTION = (
    "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."
)
VOICE_DESIGN_EN_CFG_SCALE = 4.0
VOICE_DESIGN_EN_OUTPUT_PATH = SCRIPT_DIR / "generated_voice_design_en.wav"

# Demo 4: Chinese voice design without reference audio.
RUN_VOICE_DESIGN_ZH = True
VOICE_DESIGN_ZH_TEXT = "[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。"
VOICE_DESIGN_ZH_INSTRUCTION = (
    "一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。"
)
VOICE_DESIGN_ZH_CFG_SCALE = 4.0
VOICE_DESIGN_ZH_OUTPUT_PATH = SCRIPT_DIR / "generated_voice_design_zh.wav"

# Demo 5: Reference-guided voice direction.
RUN_VOICE_DIRECTION = True
VOICE_DIRECTION_TEXT = "(clears throat) We need to discuss what happened last night."
VOICE_DIRECTION_INSTRUCTION = "Speak slowly with a restrained, serious tone."
VOICE_DIRECTION_CFG_SCALE = 4.0
VOICE_DIRECTION_OUTPUT_PATH = SCRIPT_DIR / "generated_voice_direction.wav"


@dataclass(frozen=True)
class Demo:
    name: str
    text: str
    instruction: str
    cfg_scale: float
    output_path: Path
    prompt_audio_path: str = ""
    ref_text: str = ""


def enabled_demos():
    demos = []
    if RUN_VOICE_CLONE_EN:
        demos.append(
            Demo(
                "English voice clone",
                VOICE_CLONE_EN_TEXT,
                VOICE_CLONE_EN_INSTRUCTION,
                VOICE_CLONE_EN_CFG_SCALE,
                VOICE_CLONE_EN_OUTPUT_PATH,
                REFERENCE_AUDIO_PATH,
                REFERENCE_TEXT,
            )
        )
    if RUN_VOICE_CLONE_ZH:
        demos.append(
            Demo(
                "Chinese voice clone",
                VOICE_CLONE_ZH_TEXT,
                VOICE_CLONE_ZH_INSTRUCTION,
                VOICE_CLONE_ZH_CFG_SCALE,
                VOICE_CLONE_ZH_OUTPUT_PATH,
                REFERENCE_AUDIO_PATH,
                REFERENCE_TEXT,
            )
        )
    if RUN_VOICE_DESIGN_EN:
        demos.append(
            Demo(
                "English voice design",
                VOICE_DESIGN_EN_TEXT,
                VOICE_DESIGN_EN_INSTRUCTION,
                VOICE_DESIGN_EN_CFG_SCALE,
                VOICE_DESIGN_EN_OUTPUT_PATH,
            )
        )
    if RUN_VOICE_DESIGN_ZH:
        demos.append(
            Demo(
                "Chinese voice design",
                VOICE_DESIGN_ZH_TEXT,
                VOICE_DESIGN_ZH_INSTRUCTION,
                VOICE_DESIGN_ZH_CFG_SCALE,
                VOICE_DESIGN_ZH_OUTPUT_PATH,
            )
        )
    if RUN_VOICE_DIRECTION:
        demos.append(
            Demo(
                "voice direction",
                VOICE_DIRECTION_TEXT,
                VOICE_DIRECTION_INSTRUCTION,
                VOICE_DIRECTION_CFG_SCALE,
                VOICE_DIRECTION_OUTPUT_PATH,
                REFERENCE_AUDIO_PATH,
                REFERENCE_TEXT,
            )
        )
    return tuple(demos)


DECODE_STRATEGY = "sampling"
TEMPERATURE = 0.8
TOP_K = 20
TOP_P = 0.9
MAIN_REPETITION_PENALTY = 1.1
DEPTH_REPETITION_PENALTY = 1.1
PENALTY_VALUE = 0.8
PENALTY_RANGE = 20
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
SEED = 9527

STREAMING = False
ORT_LOG = False
ORT_FP16 = False
ORT_ACCELERATE_PROVIDERS = []
MAX_THREADS = 0
DEVICE_ID = 0
SHOW_PROGRESS = True


def print_progress(message):
    if SHOW_PROGRESS:
        print(f"[Breeze TTS 2] {message}", flush=True)


def io_dtype(argument):
    match = re.fullmatch(r"tensor\(([^)]+)\)", argument.type)
    element_type = onnx.TensorProto.DataType.Value(match.group(1).upper())
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))


def io_shape(argument, dynamic_dimensions=()):
    dynamic_dimensions = iter(dynamic_dimensions)
    shape = []
    for dimension in argument.shape:
        if isinstance(dimension, int) and dimension >= 0:
            shape.append(dimension)
        else:
            shape.append(next(dynamic_dimensions))
    return tuple(shape)


def static_io_shape(argument):
    if any(
        not isinstance(dimension, int) or dimension < 0
        for dimension in argument.shape
    ):
        return None
    return tuple(argument.shape)


def io_array(argument, data):
    array = np.asarray(data, dtype=io_dtype(argument))
    if array.ndim == len(argument.shape):
        shape = tuple(
            dimension
            if isinstance(dimension, int) and dimension >= 0
            else array.shape[axis]
            for axis, dimension in enumerate(argument.shape)
        )
    else:
        shape = tuple(
            dimension if isinstance(dimension, int) and dimension >= 0 else -1
            for dimension in argument.shape
        )
    return np.ascontiguousarray(array.reshape(shape))


def ort_from_io(argument, data, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        io_array(argument, data), device, device_id
    )


def ort_buffer_from_io(argument, device, device_id, dynamic_dimensions=()):
    return onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        io_shape(argument, dynamic_dimensions), io_dtype(argument), device, device_id
    )


def bind_io_inputs(binding, arguments, values):
    for argument, value in zip(arguments, values):
        binding.bind_ortvalue_input(argument.name, value)


def bind_io_outputs(
    binding,
    arguments,
    ort_device,
    device,
    device_id,
    device_allocate_all=False,
):
    buffers = []
    for argument in arguments:
        if device_allocate_all or static_io_shape(argument) is None:
            binding._iobinding.bind_output(argument.name, ort_device)
            buffers.append(None)
        else:
            buffer = ort_buffer_from_io(argument, device, device_id)
            binding.bind_ortvalue_output(argument.name, buffer)
            buffers.append(buffer)
    return tuple(buffers)


class ConstantOrtValues:
    def __init__(self, device, device_id):
        self.device = device
        self.device_id = device_id
        self.values = {}

    def get(self, argument, data):
        array = io_array(argument, data)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key not in self.values:
            self.values[key] = onnxruntime.OrtValue.ortvalue_from_numpy(
                array, self.device, self.device_id
            )
        return self.values[key]


class IOBindingBank:
    def __init__(self, session, count, ort_device, device, device_id):
        self.session = session
        self.output_arguments = tuple(session.get_outputs())
        self.device_allocate_all = device == "cuda"
        self.outputs_to_rebind = tuple(
            argument
            for argument in self.output_arguments
            if self.device_allocate_all or static_io_shape(argument) is None
        )
        self.bindings = tuple(session.io_binding() for _ in range(count))
        self.output_buffers = tuple(
            bind_io_outputs(
                binding,
                self.output_arguments,
                ort_device,
                device,
                device_id,
                device_allocate_all=self.device_allocate_all,
            )
            for binding in self.bindings
        )
        self.used = [False] * count
        self.ort_device = ort_device

    def select(self, index):
        slot = index % len(self.bindings)
        return slot, self.bindings[slot]

    def run(self, slot, run_options):
        binding = self.bindings[slot]
        if self.used[slot]:
            for argument in self.outputs_to_rebind:
                binding._iobinding.bind_output(argument.name, self.ort_device)
        self.session.run_with_iobinding(binding, run_options=run_options)
        self.used[slot] = True
        return binding.get_outputs()


def canonical_segment_ids(tokenizer, segment_text):
    encoded = tokenizer(segment_text, add_special_tokens=True)
    rendered = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    canonical = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    return np.asarray(canonical, dtype=np.int32).reshape(1, -1)


def load_reference_audio(path, target_sample_rate, target_dtype):
    waveform, source_sample_rate = sf.read(
        path, always_2d=True, dtype="float32"
    )
    waveform = np.mean(waveform, axis=1, dtype=np.float32)
    if int(source_sample_rate) != int(target_sample_rate):
        waveform = librosa.resample(
            y=waveform,
            orig_sr=int(source_sample_rate),
            target_sr=int(target_sample_rate),
        )
    target_dtype = np.dtype(target_dtype)
    if np.issubdtype(target_dtype, np.integer):
        limits = np.iinfo(target_dtype)
        waveform = np.clip(waveform * 32768.0, limits.min, limits.max)
    return np.ascontiguousarray(waveform, dtype=target_dtype).reshape(1, 1, -1)


def left_pad_branches(branch_outputs):
    embeddings = [output[0].numpy() for output in branch_outputs]
    masks = [output[1].numpy() for output in branch_outputs]
    maximum_length = max(embedding.shape[1] for embedding in embeddings)
    padded_embeddings = []
    padded_masks = []
    pad_lengths = []
    for embedding, mask in zip(embeddings, masks):
        pad_length = maximum_length - embedding.shape[1]
        pad_lengths.append(pad_length)
        padded_embeddings.append(
            np.pad(embedding, ((0, 0), (pad_length, 0), (0, 0)))
        )
        padded_masks.append(np.pad(mask, ((0, 0), (pad_length, 0))))
    return (
        np.ascontiguousarray(np.concatenate(padded_embeddings, axis=0)),
        np.ascontiguousarray(np.concatenate(padded_masks, axis=0)),
        np.asarray(pad_lengths, dtype=np.int64),
    )


def run_inference():
    random.seed(SEED)
    np.random.seed(SEED)
    onnxruntime.set_seed(SEED)

    pipeline_started = time.perf_counter()
    metadata_path = ONNX_FOLDER / "BreezeTTS_Metadata.onnx"
    metadata_model = onnx.load(metadata_path, load_external_data=False)
    metadata = {entry.key: entry.value for entry in metadata_model.metadata_props}
    del metadata_model

    output_sample_rate = int(metadata["out_sample_rate"])
    input_sample_rate = int(metadata["in_sample_rate"])
    graph_max_seq_len = int(metadata["max_seq_len"])
    main_eos_token_id = int(metadata["main_eos_token_id"])
    codebook_pad_token_id = int(metadata["codebook_pad_token_id"])
    num_codebooks = int(metadata["num_codebooks"])
    samples_per_codec_frame = int(metadata["samples_per_codec_frame"])
    stream_window_frames = int(metadata["stream_window_frames"])
    use_batch = metadata.get("use_batch", "1") == "1"
    max_sequence_length = min(MAX_SEQ_LEN, graph_max_seq_len)
    preserve_fp16_attention = (
        metadata["use_f16_kv"] == "1"
        and metadata["compute_in_f32"] == "0"
    )

    print_progress("Loading tokenizer")
    tokenizer = GemmaTokenizerFast.from_pretrained(DOWNLOAD_PATH)

    session_options = onnxruntime.SessionOptions()
    run_options = onnxruntime.RunOptions()
    for options in (session_options, run_options):
        options.log_severity_level = 0 if ORT_LOG else 4
        options.log_verbosity_level = 4
    session_options.inter_op_num_threads = MAX_THREADS
    session_options.intra_op_num_threads = MAX_THREADS
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_gelu_approximation": "1",
        "optimization.minimal_build_optimizations": "",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": (
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
            if ORT_FP16 or preserve_fp16_attention
            else ""
        ),
    }.items():
        session_options.add_session_config_entry(key, value)
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    disabled_optimizers = (
        ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
        if ORT_FP16 or preserve_fp16_attention
        else None
    )

    if "OpenVINOExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
        provider_options = [
            {
                "device_type": "CPU",
                "precision": "ACCURACY",
                "num_of_threads": MAX_THREADS if MAX_THREADS else 8,
                "num_streams": 1,
                "enable_opencl_throttling": False,
                "enable_qdq_optimizer": False,
                "disable_dynamic_shapes": False,
            }
        ]
        device_type = "cpu"
        ort_device_type = C.OrtDevice.cpu()
    elif "CUDAExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
        provider_options = [
            {
                "device_id": DEVICE_ID,
                "gpu_mem_limit": 24 * 1024**3,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "use_tf32": "1",
                "do_copy_in_default_stream": "0",
            }
        ]
        device_type = "cuda"
        ort_device_type = C.OrtDevice.cuda()
    elif "DmlExecutionProvider" in ORT_ACCELERATE_PROVIDERS:
        provider_options = [
            {
                "device_id": DEVICE_ID,
                "performance_preference": "high_performance",
                "device_filter": "gpu",
                "disable_metacommands": "false",
            }
        ]
        device_type = "dml"
        ort_device_type = C.OrtDevice.dml()
    else:
        provider_options = None
        device_type = "cpu"
        ort_device_type = C.OrtDevice.cpu()

    ort_device = C.OrtDevice(
        ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID
    )
    shared_model_path = ONNX_FOLDER / metadata["shared_initializer_model_file"]
    shared_references = attach_shared_initializers(
        session_options, shared_model_path
    )

    def create_session(metadata_key):
        model_path = ONNX_FOLDER / metadata[metadata_key]
        print_progress(f"Loading {model_path.name}")
        return onnxruntime.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=ORT_ACCELERATE_PROVIDERS or ["CPUExecutionProvider"],
            provider_options=provider_options,
            disabled_optimizers=disabled_optimizers,
        )

    reference_session = create_session("model_file_name_reference_preprocess")
    target_session = create_session("model_file_name_target_preprocess")
    prefill_session = create_session(
        f"model_file_name_main_prefill_{DECODE_STRATEGY}"
    )
    decode_session = create_session(
        f"model_file_name_decode_step_{DECODE_STRATEGY}"
    )
    decoder_session = create_session("model_file_name_decoder")
    decoder_stream_session = (
        create_session("model_file_name_decoder_stream") if STREAMING else None
    )

    reference_arguments = tuple(reference_session.get_inputs())
    target_arguments = tuple(target_session.get_inputs())
    prefill_arguments = tuple(prefill_session.get_inputs())
    prefill_output_arguments = tuple(prefill_session.get_outputs())
    decode_arguments = tuple(decode_session.get_inputs())
    decode_output_arguments = tuple(decode_session.get_outputs())
    decoder_arguments = tuple(decoder_session.get_inputs())
    decoder_output_arguments = tuple(decoder_session.get_outputs())
    decoder_stream_arguments = (
        tuple(decoder_stream_session.get_inputs()) if decoder_stream_session else ()
    )

    num_main_kv = len(prefill_output_arguments) - 3
    decode_state_arguments = decode_arguments[:num_main_kv]
    decode_cursor = num_main_kv
    decode_pad_lengths_argument = decode_arguments[decode_cursor]
    decode_history_argument = decode_arguments[decode_cursor + 1]
    decode_guidance_argument = decode_arguments[decode_cursor + 2]
    decode_cursor += 3

    decode_save_argument = None
    if DECODE_STRATEGY != "greedy":
        decode_save_argument = decode_arguments[decode_cursor]
        decode_cursor += 1

    if DECODE_STRATEGY == "penalty_greedy":
        main_control_data = (PENALTY_VALUE, PENALTY_RANGE)
        predictor_control_data = (PENALTY_VALUE, PENALTY_RANGE)
    elif DECODE_STRATEGY == "sampling":
        main_control_data = (
            TEMPERATURE,
            TOP_K,
            TOP_P,
            MAIN_REPETITION_PENALTY,
        )
        predictor_control_data = (
            TEMPERATURE,
            TOP_K,
            TOP_P,
            DEPTH_REPETITION_PENALTY,
        )
    else:
        main_control_data = ()
        predictor_control_data = ()

    decode_main_control_arguments = decode_arguments[
        decode_cursor : decode_cursor + len(main_control_data)
    ]
    decode_cursor += len(main_control_data)
    decode_predictor_arguments = decode_arguments[decode_cursor : decode_cursor + 3]
    decode_cursor += 3
    decode_predictor_control_arguments = decode_arguments[decode_cursor:]

    decode_last_hidden_position = num_main_kv
    decode_token_position = num_main_kv + 1
    decode_save_position = num_main_kv + 2 if decode_save_argument else None
    decode_history_position = (
        num_main_kv + 3 if decode_save_argument else num_main_kv + 2
    )
    decode_generated_position = len(decode_output_arguments) - 2
    decode_frame_position = len(decode_output_arguments) - 1

    reference_bank = IOBindingBank(
        reference_session, 1, ort_device, device_type, DEVICE_ID
    )
    target_bank = IOBindingBank(
        target_session, 2 if use_batch else 1, ort_device, device_type, DEVICE_ID
    )
    prefill_bank = IOBindingBank(
        prefill_session, 1, ort_device, device_type, DEVICE_ID
    )
    decode_bank = IOBindingBank(
        decode_session, 2, ort_device, device_type, DEVICE_ID
    )
    decoder_bank = IOBindingBank(
        decoder_session, 1, ort_device, device_type, DEVICE_ID
    )
    decoder_stream_bank = (
        IOBindingBank(
            decoder_stream_session, 1, ort_device, device_type, DEVICE_ID
        )
        if decoder_stream_session
        else None
    )
    constants = ConstantOrtValues(device_type, DEVICE_ID)
    empty_ref_code = ort_from_io(
        target_arguments[1],
        np.empty((1, 0, num_codebooks), dtype=np.int32),
        device_type,
        DEVICE_ID,
    )
    hidden_size = int(target_arguments[2].shape[-1])
    empty_ref_text_embed = ort_from_io(
        target_arguments[2],
        np.empty((1, 0, hidden_size), dtype=np.float32),
        device_type,
        DEVICE_ID,
    )
    reference_cache = {}

    main_control_values = tuple(
        constants.get(argument, value)
        for argument, value in zip(
            decode_main_control_arguments, main_control_data
        )
    )
    predictor_control_values = tuple(
        constants.get(argument, value)
        for argument, value in zip(
            decode_predictor_control_arguments, predictor_control_data
        )
    )

    for binding in decode_bank.bindings:
        bind_io_inputs(
            binding, decode_main_control_arguments, main_control_values
        )
        bind_io_inputs(
            binding,
            decode_predictor_control_arguments,
            predictor_control_values,
        )

    stream_input_buffer = (
        ort_buffer_from_io(
            decoder_stream_arguments[0], device_type, DEVICE_ID
        )
        if decoder_stream_session
        else None
    )
    if decoder_stream_bank:
        decoder_stream_bank.bindings[0].bind_ortvalue_input(
            decoder_stream_arguments[0].name, stream_input_buffer
        )

    def reference_conditioning(demo):
        has_audio = bool(demo.prompt_audio_path)
        has_text = bool(demo.ref_text)
        if has_audio != has_text:
            raise ValueError(
                f"{demo.name} must provide reference audio and text together"
            )
        if not has_audio:
            return empty_ref_code, empty_ref_text_embed

        cache_key = (demo.prompt_audio_path, demo.ref_text)
        if cache_key not in reference_cache:
            print_progress(f"Preprocessing reference audio for {demo.name}")
            prompt_audio = load_reference_audio(
                demo.prompt_audio_path,
                input_sample_rate,
                io_dtype(reference_arguments[0]),
            )
            ref_text_ids = canonical_segment_ids(tokenizer, f"[S0]{demo.ref_text}")
            reference_values = (
                ort_from_io(
                    reference_arguments[0], prompt_audio, device_type, DEVICE_ID
                ),
                ort_from_io(
                    reference_arguments[1], ref_text_ids, device_type, DEVICE_ID
                ),
            )
            bind_io_inputs(
                reference_bank.bindings[0], reference_arguments, reference_values
            )
            reference_outputs = reference_bank.run(0, run_options)
            reference_cache[cache_key] = (
                reference_outputs[0],
                reference_outputs[1],
            )
        return reference_cache[cache_key]

    def run_demo(demo, demo_index, demo_count):
        demo_started = time.perf_counter()
        print_progress(f"Starting demo {demo_index}/{demo_count}: {demo.name}")
        ref_code, ref_text_embed = reference_conditioning(demo)

        cfg_scale = float(demo.cfg_scale)
        if not math.isfinite(cfg_scale) or cfg_scale < 0.0:
            raise ValueError(f"{demo.name} CFG scale must be finite and non-negative")
        positive_text = (
            f"[S0]<ins_bos>{demo.instruction}<ins_eos>{demo.text}"
        )
        negative_text = f"[S0]{demo.text}"
        if not use_batch:
            branch_texts = (negative_text,) if cfg_scale == 0.0 else (positive_text,)
            guidance_values = np.asarray([1.0], dtype=np.float32)
        elif cfg_scale == 1.0:
            branch_texts = (positive_text,)
            guidance_values = np.asarray([1.0], dtype=np.float32)
        elif cfg_scale == 0.0:
            branch_texts = (negative_text,)
            guidance_values = np.asarray([1.0], dtype=np.float32)
        else:
            branch_texts = (positive_text, negative_text)
            guidance_values = np.asarray(
                [cfg_scale, 1.0 - cfg_scale], dtype=np.float32
            )

        branch_outputs = []
        for branch_index, branch_text in enumerate(branch_texts):
            branch_ids = canonical_segment_ids(tokenizer, branch_text)
            branch_id_value = ort_from_io(
                target_arguments[0], branch_ids, device_type, DEVICE_ID
            )
            slot, binding = target_bank.select(branch_index)
            bind_io_inputs(
                binding,
                target_arguments,
                (branch_id_value, ref_code, ref_text_embed),
            )
            branch_outputs.append(target_bank.run(slot, run_options))

        inputs_embeds, attention_mask, pad_lengths = left_pad_branches(
            branch_outputs
        )
        prefill_len = inputs_embeds.shape[1]
        guidance_weights = constants.get(prefill_arguments[2], guidance_values)
        prefill_values = [
            ort_from_io(
                prefill_arguments[0], inputs_embeds, device_type, DEVICE_ID
            ),
            ort_from_io(
                prefill_arguments[1], attention_mask, device_type, DEVICE_ID
            ),
            guidance_weights,
        ]
        if DECODE_STRATEGY == "sampling":
            prefill_values.extend(
                constants.get(argument, value)
                for argument, value in zip(
                    prefill_arguments[3:], (TEMPERATURE, TOP_K, TOP_P)
                )
            )
        bind_io_inputs(
            prefill_bank.bindings[0], prefill_arguments, prefill_values
        )
        prefill_outputs = prefill_bank.run(0, run_options)

        cached_state = prefill_outputs[:num_main_kv]
        last_hidden_state, codec_token_main, history_len = prefill_outputs[
            num_main_kv:
        ]
        main_save_ids = codec_token_main if decode_save_argument else None
        generated_codec = ort_buffer_from_io(
            decode_predictor_arguments[2],
            device_type,
            DEVICE_ID,
            dynamic_dimensions=(0,),
        )

        decode_pad_lengths = constants.get(
            decode_pad_lengths_argument, pad_lengths
        )
        decode_guidance = constants.get(
            decode_guidance_argument, guidance_values
        )
        for binding in decode_bank.bindings:
            binding.bind_ortvalue_input(
                decode_pad_lengths_argument.name, decode_pad_lengths
            )
            binding.bind_ortvalue_input(
                decode_guidance_argument.name, decode_guidance
            )

        generation_limit = min(
            MAX_NEW_TOKENS, max(0, max_sequence_length - prefill_len - 1)
        )
        valid_frames = []
        stream_window = []
        stream_windows_decoded = 0
        generated_frames = 0
        selected_token = int(codec_token_main.numpy().flat[0])
        output_path = demo.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print_progress(
            f"Generating {demo.name} with {len(branch_texts)} CFG branch(es), "
            f"limit={generation_limit}"
        )
        with sf.SoundFile(
            output_path,
            mode="w",
            samplerate=output_sample_rate,
            channels=1,
            subtype=(
                "PCM_16"
                if np.issubdtype(io_dtype(decoder_output_arguments[0]), np.integer)
                else "FLOAT"
            ),
            format="WAV",
        ) as output_file:
            for step_index in range(generation_limit):
                if selected_token == main_eos_token_id:
                    break

                slot, binding = decode_bank.select(step_index)
                bind_io_inputs(binding, decode_state_arguments, cached_state)
                binding.bind_ortvalue_input(
                    decode_history_argument.name, history_len
                )
                binding.bind_ortvalue_input(
                    decode_predictor_arguments[0].name, codec_token_main
                )
                binding.bind_ortvalue_input(
                    decode_predictor_arguments[1].name, last_hidden_state
                )
                binding.bind_ortvalue_input(
                    decode_predictor_arguments[2].name, generated_codec
                )
                if decode_save_argument:
                    binding.bind_ortvalue_input(
                        decode_save_argument.name, main_save_ids
                    )

                decode_outputs = decode_bank.run(slot, run_options)
                cached_state = decode_outputs[:num_main_kv]
                last_hidden_state = decode_outputs[decode_last_hidden_position]
                codec_token_main = decode_outputs[decode_token_position]
                if decode_save_position is not None:
                    main_save_ids = decode_outputs[decode_save_position]
                history_len = decode_outputs[decode_history_position]
                generated_codec = decode_outputs[decode_generated_position]
                frame = decode_outputs[decode_frame_position].numpy().astype(
                    np.int32, copy=True
                )
                selected_token = int(codec_token_main.numpy().flat[0])
                generated_frames += 1

                if not np.all(frame == codebook_pad_token_id):
                    valid_frames.append(frame)
                    if decoder_stream_session:
                        stream_window.append(frame)
                        if len(stream_window) > stream_window_frames:
                            stream_window.pop(0)
                        if len(stream_window) == stream_window_frames:
                            window = np.ascontiguousarray(
                                np.concatenate(stream_window, axis=1)
                            )
                            stream_input_buffer.update_inplace(
                                io_array(decoder_stream_arguments[0], window)
                            )
                            stream_outputs = decoder_stream_bank.run(
                                0, run_options
                            )
                            wave = stream_outputs[0].numpy().reshape(-1)
                            if np.issubdtype(wave.dtype, np.floating):
                                wave = wave.astype(np.float32, copy=False)
                            output_file.write(
                                wave
                                if stream_windows_decoded == 0
                                else wave[-samples_per_codec_frame:]
                            )
                            stream_windows_decoded += 1

            if valid_frames and (
                not decoder_stream_session or stream_windows_decoded == 0
            ):
                flattened_codes = np.ascontiguousarray(
                    np.concatenate(valid_frames, axis=1)
                )
                decoder_input = ort_from_io(
                    decoder_arguments[0],
                    flattened_codes,
                    device_type,
                    DEVICE_ID,
                )
                slot, binding = decoder_bank.select(0)
                binding.bind_ortvalue_input(
                    decoder_arguments[0].name, decoder_input
                )
                decoder_outputs = decoder_bank.run(slot, run_options)
                wave = decoder_outputs[0].numpy().reshape(-1)
                if np.issubdtype(wave.dtype, np.floating):
                    wave = wave.astype(np.float32, copy=False)
                output_file.write(wave)

        if not valid_frames:
            raise RuntimeError(f"{demo.name} generated no valid codec frames")
        print_progress(
            f"Wrote {output_path}; frames={generated_frames}, "
            f"valid_frames={len(valid_frames)}, "
            f"elapsed={time.perf_counter() - demo_started:.2f}s"
        )
        return output_path

    demos = enabled_demos()
    if not demos:
        raise ValueError("Enable at least one Breeze TTS demo")
    print_progress(f"Enabled demos: {', '.join(demo.name for demo in demos)}")
    for demo_index, demo in enumerate(demos, start=1):
        run_demo(demo, demo_index, len(demos))

    print_progress(
        f"Completed {len(demos)} demo(s) in "
        f"{time.perf_counter() - pipeline_started:.2f}s"
    )
    return shared_references


if __name__ == "__main__":
    run_inference()