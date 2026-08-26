"""Convolutional spectral transforms used by the ZipVoice ONNX exporter."""

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.onnx.operators import shape_as_tensor


_WINDOW_FACTORIES: dict[str, Callable[[int], Tensor]] = {
    "hann": lambda length: torch.hann_window(length, periodic=True),
    "hamming": lambda length: torch.hamming_window(length, periodic=True),
    "blackman": lambda length: torch.blackman_window(length, periodic=True),
}


def _padded_window(win_length: int, n_fft: int, window_type: str) -> Tensor:
    window = _WINDOW_FACTORIES[window_type](win_length).float()
    if win_length == n_fft:
        return window

    padding = n_fft - win_length
    return F.pad(window, (padding // 2, padding - padding // 2))


class STFT_Process(nn.Module):
    """ONNX-exportable STFT/ISTFT implemented with one-dimensional convolutions."""

    def __init__(
        self,
        model_type: str,
        n_fft: int,
        win_length: int,
        hop_len: int,
        window_type: str = "hann",
        center_pad: bool = True,
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.half_n_fft = n_fft // 2
        self.center_pad = center_pad
        self.pad_mode = pad_mode

        window = _padded_window(win_length, n_fft, window_type)
        frequencies = torch.arange(self.half_n_fft + 1, dtype=torch.float32).unsqueeze(1)
        samples = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        angles = 2.0 * torch.pi * frequencies * samples / n_fft

        if model_type == "stft_B":
            real_kernel = (torch.cos(angles) * window).unsqueeze(1)
            imag_kernel = (-torch.sin(angles) * window).unsqueeze(1)
            self.register_buffer("stft_kernel", torch.cat((real_kernel, imag_kernel)))
        elif model_type == "istft_A":
            scale = torch.full((self.half_n_fft + 1, 1), 2.0 / n_fft)
            scale[0] = 1.0 / n_fft
            if n_fft % 2 == 0:
                scale[-1] = 1.0 / n_fft
            real_kernel = scale * torch.cos(angles) * window
            imag_kernel = -scale * torch.sin(angles) * window
            self.register_buffer(
                "inverse_kernel",
                torch.cat((real_kernel, imag_kernel)).unsqueeze(1),
            )
            self.precomputed_window_envelope = center_pad and n_fft == 4 * hop_len
            if self.precomputed_window_envelope:
                window_chunks = window.square().reshape(4, hop_len)
                self.register_buffer(
                    "inverse_window_chunks",
                    torch.stack(
                        (
                            window_chunks[:3].sum(dim=0).reciprocal(),
                            window_chunks.sum(dim=0).reciprocal(),
                            window_chunks[1:].sum(dim=0).reciprocal(),
                            window_chunks[1:3].sum(dim=0).reciprocal(),
                        )
                    ),
                )
            else:
                self.register_buffer(
                    "window_square",
                    window.square().reshape(1, 1, -1),
                )
        else:
            raise ValueError(f"Unsupported spectral transform: {model_type}")

    def _center(self, audio: Tensor) -> Tensor:
        if not self.center_pad:
            return audio
        if self.pad_mode == "reflect":
            left = audio[..., 1 : self.half_n_fft + 1].flip(-1)
            right = audio[..., -(self.half_n_fft + 1) : -1].flip(-1)
            return torch.cat((left, audio, right), dim=-1)
        return F.pad(audio, (self.half_n_fft, self.half_n_fft))

    def _stft(self, audio: Tensor) -> tuple[Tensor, Tensor]:
        spectrum = F.conv1d(
            self._center(audio),
            self.stft_kernel,
            stride=self.hop_len,
        )
        return spectrum.split(self.half_n_fft + 1, dim=1)

    def _istft(self, magnitude: Tensor, phase: Tensor) -> Tensor:
        rectangular = torch.cat(
            (magnitude * torch.cos(phase), magnitude * torch.sin(phase)),
            dim=1,
        )
        audio = F.conv_transpose1d(
            rectangular,
            self.inverse_kernel,
            stride=self.hop_len,
        )
        if self.precomputed_window_envelope:
            audio = audio[..., self.half_n_fft : -self.half_n_fft]
            chunk_count = torch.div(
                shape_as_tensor(audio)[-1],
                self.hop_len,
                rounding_mode="floor",
            )
            chunk_indices = torch.arange(
                chunk_count,
                dtype=torch.int64,
                device=audio.device,
            )
            categories = torch.ones_like(chunk_indices)
            categories = torch.where(
                chunk_indices == 0,
                torch.zeros_like(categories),
                categories,
            )
            categories = torch.where(
                chunk_indices == chunk_count - 1,
                torch.full_like(categories, 2),
                categories,
            )
            categories = torch.where(
                chunk_count == 1,
                torch.full_like(categories, 3),
                categories,
            )
            inverse_envelope = F.embedding(
                categories,
                self.inverse_window_chunks,
            ).reshape(1, 1, -1, self.hop_len)
            chunked_audio = audio.reshape(
                audio.shape[0],
                audio.shape[1],
                -1,
                self.hop_len,
            )
            return (chunked_audio * inverse_envelope).reshape_as(audio)

        frame_weights = torch.ones(
            (1, 1, magnitude.shape[-1]),
            dtype=magnitude.dtype,
            device=magnitude.device,
        )
        window_sum = F.conv_transpose1d(
            frame_weights,
            self.window_square,
            stride=self.hop_len,
        )
        if self.center_pad:
            output_slice = slice(self.half_n_fft, -self.half_n_fft)
            return audio[..., output_slice] / window_sum[..., output_slice]
        return audio / window_sum

    def forward(
        self,
        first: Tensor,
        second: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self.model_type == "stft_B":
            return self._stft(first)
        return self._istft(first, second)
