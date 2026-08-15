"""Run the standalone BigVGAN ONNX vocoder package."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent


# ============================== Configuration ==============================
# Edit runtime values directly; the CLI is reserved for selecting the ONNX folder.
MEL_PATH = None                         # Optional .npy mel tensor; rank [M,T] or [1,M,T].
MEL_FRAMES = 256                        # Used only when MEL_PATH is None.
OUTPUT_PATH = SCRIPT_DIR / "generated.wav"

ORT_ACCELERATE_PROVIDERS = []           # [] uses CPU; e.g. ["CUDAExecutionProvider"].
MAX_THREADS = 0                         # CPU thread count; 0 lets ONNX Runtime choose.
DEVICE_ID = 0                           # Accelerator device index.
ORT_LOG = False                         # Enable ONNX Runtime logging.
SHOW_PROGRESS = True                    # Print pipeline stages.
# ===========================================================================


def print_progress(message: str) -> None:
    if SHOW_PROGRESS:
        print(f"[BigVGAN] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "BigVGAN_Optimized",
        help="Folder containing BigVGAN.onnx and BigVGAN_Metadata.onnx.",
    )
    return parser.parse_args()


def io_dtype(argument: object) -> np.dtype:
    match = re.fullmatch(r"tensor\(([^)]+)\)", argument.type)
    element_type = onnx.TensorProto.DataType.Value(match.group(1).upper())
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(element_type))


def load_mel(argument: object) -> np.ndarray:
    batch, channels, frames = argument.shape
    if MEL_PATH is None:
        mel = np.zeros((batch, channels, MEL_FRAMES), dtype=io_dtype(argument))
    else:
        mel = np.load(Path(MEL_PATH).expanduser().resolve(), allow_pickle=False)
        if mel.ndim == 2:
            mel = mel[None, :, :]
        mel = np.asarray(mel, dtype=io_dtype(argument))
    return np.ascontiguousarray(mel)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    folder = args.onnx_folder.expanduser().resolve()
    metadata_path = folder / "BigVGAN_Metadata.onnx"
    print_progress(f"Reading package metadata: {metadata_path}")
    model = onnx.load(metadata_path, load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    expected_keys = {
        "graph_layout",
        "out_sample_rate",
        "model_file_name_vocoder",
    }
    missing_metadata = sorted(expected_keys - metadata.keys())
    if missing_metadata:
        raise ValueError(
            f"{metadata_path.name} is missing required metadata key(s): {missing_metadata}."
        )
    vocoder_path = folder / metadata["model_file_name_vocoder"]
    options = ort.SessionOptions()
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.log_severity_level = 0 if ORT_LOG else 4
    providers = ORT_ACCELERATE_PROVIDERS or ["CPUExecutionProvider"]
    provider_options = [
        {"device_id": DEVICE_ID} if provider != "CPUExecutionProvider" else {}
        for provider in providers
    ]
    print_progress(f"Loading ONNX graph: {vocoder_path.name}")
    session = ort.InferenceSession(
        str(vocoder_path),
        sess_options=options,
        providers=providers,
        provider_options=provider_options,
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    mel = load_mel(inputs[0])
    print_progress(f"Decoding mel tensor: {mel.shape}")
    waveform = session.run([outputs[0].name], {inputs[0].name: mel})[0].reshape(-1)
    output_dtype = io_dtype(outputs[0])
    waveform = waveform.astype(output_dtype, copy=False)
    output_path = Path(OUTPUT_PATH).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_16" if np.issubdtype(output_dtype, np.integer) else "FLOAT"
    sf.write(
        output_path,
        waveform,
        int(metadata["out_sample_rate"]),
        subtype=subtype,
    )
    print(
        f"Saved {waveform.size} samples to {output_path}; "
        f"pipeline={time.perf_counter() - started:.2f}s.",
        flush=True,
    )


if __name__ == "__main__":
    main()