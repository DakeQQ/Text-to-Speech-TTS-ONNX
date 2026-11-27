"""Stateful fine-local, ConvNeXt, and in-graph iSTFT execution with CPU OrtValue hand-offs.

Hojo_Decoder_Stream.onnx keeps the inverse FFT, overlap-add and crop in ORT and writes PCM straight into caller storage.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

HOP = 480
_CROP = 720
_FIRST_NONFINAL_DELAY = 8


def _float_dtype(value: ort.NodeArg) -> np.dtype:
    types = {"tensor(float)": np.dtype(np.float32), "tensor(float16)": np.dtype(np.float16)}
    if value.type not in types:
        raise ValueError(f"Hojo streaming tensor {value.name} must be float32 or float16, got {value.type}")
    return types[value.type]


def stream_pcm_end(audio_frames: int, *, final: bool, delay_frames: int = _FIRST_NONFINAL_DELAY) -> int:
    """Cumulative PCM samples the decoder has produced after `audio_frames` frames."""
    return HOP * audio_frames if final else max(0, HOP * (audio_frames - delay_frames) - _CROP)


class StatefulVocoder:
    def __init__(
        self,
        fine: ort.InferenceSession,
        decoder: ort.InferenceSession,
        run_options: ort.RunOptions,
        voice: ort.OrtValue | None,
        *,
        decoder_delay_frames: int = _FIRST_NONFINAL_DELAY,
    ) -> None:
        if any(session.get_providers() != ["CPUExecutionProvider"] for session in (fine, decoder)):
            raise ValueError("The stateful vocoder requires CPUExecutionProvider")
        self.fine, self.decoder = fine, decoder
        self._run = run_options
        self._delay_frames = int(decoder_delay_frames)
        if self._delay_frames <= 0:
            raise ValueError(f"decoder_delay_frames must be positive, got {self._delay_frames}")
        fine_inputs = {value.name: value for value in fine.get_inputs()}
        fine_outputs = {value.name: value for value in fine.get_outputs()}
        decoder_inputs = {value.name: value for value in decoder.get_inputs()}
        decoder_outputs = {value.name: value for value in decoder.get_outputs()}
        self._decoder_outputs = list(decoder_outputs)
        if ({"pcm", "pcm_finite"} - set(self._decoder_outputs)
                or decoder_inputs.get("ola_past") is None
                or decoder_inputs.get("frames_past") is None
                or decoder_inputs["frames_past"].type != "tensor(int64)"):
            raise ValueError("Hojo_Decoder_Stream.onnx must render PCM in-graph; re-optimize the package with --advanced")
        self._hidden_dtype = _float_dtype(fine_inputs["hidden_states"])
        _float_dtype(fine_outputs["bits"])
        self._bits_dtype = _float_dtype(decoder_inputs["bits"])
        self._bits_ort_type = "tensor(float16)" if self._bits_dtype == np.float16 else "tensor(float)"
        self._pcm_dtype = _float_dtype(decoder_outputs["pcm"])
        self._pcm_buffer = np.empty(0, self._pcm_dtype)
        self._pcm_finite = np.zeros((), np.bool_)
        self._pcm_finite_ov = ort.OrtValue.ortvalue_from_numpy(self._pcm_finite)
        self._fine_binding, self._decoder_binding = fine.io_binding(), decoder.io_binding()
        self._fine_outputs = list(fine_outputs)
        self._fine_bits_idx = self._fine_outputs.index("bits")
        self._fine_past = [value for value in fine.get_inputs() if value.name.startswith("fine_past.")]
        self._conv_past = [value for value in decoder.get_inputs() if value.name.startswith("conv_past.")]
        if len(self._fine_past) != 8 or len(self._conv_past) != 8:
            raise ValueError("Unexpected Hojo streaming state count")
        for value, outputs, present in (
            *((value, fine_outputs, value.name.replace("fine_past.", "fine_present.")) for value in self._fine_past),
            *((value, decoder_outputs, value.name.replace("conv_past.", "conv_present.")) for value in self._conv_past),
            (decoder_inputs["ola_past"], decoder_outputs, "ola_present"),
            (decoder_inputs["frames_past"], decoder_outputs, "frames_present"),
        ):
            if present not in outputs or outputs[present].type != value.type:
                raise ValueError(f"Hojo streaming state {value.name}/{present} must have matching I/O types")
        self._fine_past_to_out_idx = [
            (value.name, self._fine_outputs.index(value.name.replace("fine_past.", "fine_present.")))
            for value in self._fine_past
        ]
        self._empty_fine = {
            value.name: ort.OrtValue.ortvalue_from_numpy(
                np.zeros([dim if isinstance(dim, int) else 0 for dim in value.shape], _float_dtype(value)),
            )
            for value in self._fine_past
        }
        # Pre-allocate two static ping-pong state banks for the decoder to avoid per-chunk state allocations.
        ola_dtype = _float_dtype(decoder_inputs["ola_past"])
        self._dec_state_pairs: list[tuple[str, str]] = [
            *((value.name, value.name.replace("conv_past.", "conv_present.")) for value in self._conv_past),
            ("ola_past", "ola_present"),
            ("frames_past", "frames_present"),
        ]
        self._conv_specs = [(value.name, tuple(value.shape), _float_dtype(value)) for value in self._conv_past]
        self._dec_bank_bufs: tuple[dict[str, np.ndarray], dict[str, np.ndarray]] = (
            {
                **{name: np.zeros(shape, dtype) for name, shape, dtype in self._conv_specs},
                "ola_past": np.zeros((1, 3 * HOP), ola_dtype),
                "frames_past": np.zeros(1, np.int64),
            },
            {
                **{name: np.zeros(shape, dtype) for name, shape, dtype in self._conv_specs},
                "ola_past": np.zeros((1, 3 * HOP), ola_dtype),
                "frames_past": np.zeros(1, np.int64),
            },
        )
        self._dec_banks: tuple[dict[str, ort.OrtValue], dict[str, ort.OrtValue]] = (
            {name: ort.OrtValue.ortvalue_from_numpy(buf) for name, buf in self._dec_bank_bufs[0].items()},
            {name: ort.OrtValue.ortvalue_from_numpy(buf) for name, buf in self._dec_bank_bufs[1].items()},
        )
        bits_shape = decoder_inputs["bits"].shape
        self._empty_bits = ort.OrtValue.ortvalue_from_numpy(
            np.empty([dim if isinstance(dim, int) else 0 for dim in bits_shape], self._bits_dtype),
        )
        names = set(fine_inputs)
        self._speaker_name = next((name for name in ("speaker_projection", "speaker_embedding") if name in names), None)
        self._speaker_dtype = _float_dtype(fine_inputs[self._speaker_name]) if self._speaker_name is not None else None
        self._speaker_ort_type = (
            ("tensor(float16)" if self._speaker_dtype == np.float16 else "tensor(float)")
            if self._speaker_dtype is not None else None
        )
        self._speaker_value = None
        self._speaker_ready = self._speaker_name is None
        if "voice" in names:
            if voice is None:
                raise ValueError("Preset streaming requires an int32 voice selector")
            self._fine_binding.bind_ortvalue_input("voice", voice)
        # Bound in place: writing these flags/outputs re-targets the next decoder run without rebinding.
        self._first, self._final = np.array(True), np.array(False)
        self._decoder_binding.bind_cpu_input("is_first", self._first)
        self._decoder_binding.bind_cpu_input("is_final", self._final)
        self._decoder_binding.bind_ortvalue_output("pcm_finite", self._pcm_finite_ov)
        self.reset()

    def set_speaker(self, value: ort.OrtValue) -> None:
        if self._speaker_name is None:
            raise ValueError("Preset streaming does not accept a cloning speaker embedding")
        if value.data_type() != self._speaker_ort_type:
            value = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(value.numpy(), dtype=self._speaker_dtype))
        self._speaker_value = value
        self._fine_binding.bind_ortvalue_input(self._speaker_name, self._speaker_value)
        self._speaker_ready = True

    def reset(self) -> None:
        self._fine_state = dict(self._empty_fine)
        for buf in self._dec_bank_bufs[0].values():
            buf.fill(0)
        self._dec_bank = 0
        self._first[()] = True
        self._final[()] = False
        self._finished = False
        self._audio_frames = self.produced_samples = 0

    def process(
        self, hidden: np.ndarray, tokens: np.ndarray, output: np.ndarray, *, final: bool,
    ) -> int:
        if self._finished:
            raise RuntimeError("Reset the stateful vocoder before starting another utterance")
        if not self._speaker_ready:
            raise RuntimeError("Set the cloning reference before running the stateful vocoder")
        if (hidden.dtype not in (np.float32, np.float16) or hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[2] != 512 or tokens.dtype != np.int32 or tokens.shape != hidden.shape[:2]
                or not hidden.flags.c_contiguous or not tokens.flags.c_contiguous):
            raise ValueError("Stateful vocoding requires contiguous [1, frames, 512] float32/float16 hidden states and int32 IDs")
        frames = tokens.shape[1]
        if not frames and (not final or self._first[()]):
            raise ValueError("An empty chunk is only valid when flushing a nonempty utterance")
        if self._first[()] and not final and frames < self._delay_frames + 3:
            raise ValueError(f"The first non-final chunk needs at least {self._delay_frames + 3} audio frames")
        pcm_size = stream_pcm_end(
            self._audio_frames + frames, final=final, delay_frames=self._delay_frames,
        ) - self.produced_samples
        if (output.dtype != np.float32 or output.ndim != 1 or not output.flags.c_contiguous
                or not output.flags.writeable or output.size < pcm_size):
            raise ValueError("Stateful PCM output must be a writable contiguous float32 vector with room for the chunk")
        if frames:
            binding = self._fine_binding
            if hidden.dtype != self._hidden_dtype:
                hidden = np.ascontiguousarray(hidden, dtype=self._hidden_dtype)
            binding.bind_cpu_input("hidden_states", hidden)
            binding.bind_cpu_input("audio_tokens", tokens)
            for name, value in self._fine_state.items():
                binding.bind_ortvalue_input(name, value)
            for name in self._fine_outputs:
                binding.bind_output(name, "cpu")
            self.fine.run_with_iobinding(binding, self._run)
            fine_out_vals = binding.get_outputs()
            self._fine_state = {name: fine_out_vals[idx] for name, idx in self._fine_past_to_out_idx}
            bits = fine_out_vals[self._fine_bits_idx]
        else:
            bits = self._empty_bits
        self.last_bits = bits
        self._final[()] = final
        binding = self._decoder_binding
        if bits.data_type() != self._bits_ort_type:
            bits = ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(bits.numpy(), dtype=self._bits_dtype))
        binding.bind_ortvalue_input("bits", bits)
        in_bank = self._dec_banks[self._dec_bank]
        out_bank = self._dec_banks[1 - self._dec_bank]
        for past_name, present_name in self._dec_state_pairs:
            binding.bind_ortvalue_input(past_name, in_bank[past_name])
            binding.bind_ortvalue_output(present_name, out_bank[past_name])
        pcm = output
        if self._pcm_dtype != output.dtype:
            if self._pcm_buffer.size < pcm_size:
                self._pcm_buffer = np.empty(pcm_size, self._pcm_dtype)
            pcm = self._pcm_buffer[:pcm_size]
        # An empty flush still binds a zero-length view of the caller's storage; ORT validates [pcm_size].
        binding.bind_output("pcm", "cpu", 0, self._pcm_dtype, [pcm_size], pcm.ctypes.data)
        self.decoder.run_with_iobinding(binding, self._run)
        if not self._pcm_finite[()]:
            raise RuntimeError("Hojo decoder produced non-finite spectral or overlap-add values")
        if pcm is not output:
            np.copyto(output[:pcm_size], pcm)
        self._dec_bank = 1 - self._dec_bank
        self._audio_frames += frames
        self.produced_samples += pcm_size
        self._first[()] = False
        self._finished = final
        if final and self.produced_samples != self._audio_frames * HOP:
            raise RuntimeError("The stateful vocoder did not flush the complete waveform")
        return pcm_size

    def close(self) -> None:
        pass
