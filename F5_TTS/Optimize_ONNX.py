"""Optimize and quantize every exported F5-TTS ONNX graph.

Edit ``MODEL_PLANS`` and ``CONFIG`` below to choose each graph's precision,
quantization, optimization, and storage policy. The processing pipeline remains
centralized in ``Optimize_ONNX_Common.py``.

    Method       Backend                   Result
    "Q2/Q4/Q8"   matmul_nbits_quantizer    2/4/8-bit weight-only (MatMulNBits)
    "DYNAMIC"    quantize_dynamic          INT8 dynamic (DynamicQuantizeLinear)
    "F16"        convert_float_to_float16  float16 weights & activations
    "F32"        -                         keep float32 (optimize only)

For the DirectML-tuned optimization pass, use ``Optimize_ONNX_DML.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    resolve_plan,
    run_optimizer,
)


SOURCE_FOLDER = SCRIPT_DIR / "F5_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "F5_Optimized"

# The DiT transformer uses 16 heads / hidden_size 1024; the same values feed the attention-fusion
# optimizer for every module (harmless for the non-attention Preprocess / Decode graphs).
# F5_Preprocess carries dynamic STFT shapes, so its onnxslim passes skip shape inference.
MODEL_PLANS: dict[str, Plan] = {
    "F5_Metadata": Plan(
        method="F32",
        transformer=False,
    ),
    "F5_Preprocess": Plan(
        method="F32",
        num_heads=0,
        hidden_size=0,
        opt_level=2,
    ),
    "F5_Transformer": Plan(
        method="F32",
        num_heads=16,
        hidden_size=1024,
        transformer=True,
        opt_level=2,
    ),
    "F5_Decode": Plan(
        method="F32",
        num_heads=0,
        hidden_size=0,
        opt_level=2,
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    optimizer_level=2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def resolve_plans():
    resolved_plans = {}
    for name, plan in MODEL_PLANS.items():
        resolved = resolve_plan(plan, CONFIG)
        resolved_plans[name] = resolved
    return resolved_plans


def main() -> None:
    args = parse_args()
    resolved_plans = resolve_plans()
    run_optimizer(CONFIG)


if __name__ == "__main__":
    main()
