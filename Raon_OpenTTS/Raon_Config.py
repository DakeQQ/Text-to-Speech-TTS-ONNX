from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RaonArchitecture:
    model_name: str
    dim: int
    depth: int
    heads: int
    head_dim: int
    ff_mult: int
    text_dim: int = 512
    text_conv_layers: int = 4
    text_conv_mult: int = 2

    @property
    def attention_inner_dim(self) -> int:
        return self.heads * self.head_dim


SUPPORTED_ARCHITECTURES = {
    "0.3B": RaonArchitecture("0.3B", 1024, 22, 16, 64, 2),
    "1B": RaonArchitecture("1B", 1408, 28, 24, 64, 4),
}

UNSUPPORTED_TRAILING_VOCAB_TOKENS = (
    "\ufdfa",
    "\ufdfb",
    "\ufffd",
    "\U0001f3b5",
)


def require_architecture(model_name: Any, label: str = "model_name") -> RaonArchitecture:
    if isinstance(model_name, str):
        architecture = SUPPORTED_ARCHITECTURES.get(model_name)
        if architecture is not None:
            return architecture
    supported = ", ".join(sorted(SUPPORTED_ARCHITECTURES))
    raise ValueError(
        f"Unsupported Raon {label} {model_name!r}; supported models: {supported}"
    )