"""Optimize and quantize the exported BigVGAN ONNX vocoder.

Edit ``MODEL_PLANS`` and ``CONFIG`` below to choose the graph's precision,
optimization, and storage policy. The processing pipeline remains centralized in
``Optimize_ONNX_Common.py``.

Each module in MODEL_PLANS picks a method; a Plan field left ``None`` inherits the matching
OptimizerConfig default.

    Method       Backend                   Result
    "Q2/Q4/Q8"   matmul_nbits_quantizer    2/4/8-bit weight-only (MatMulNBits)
    "DYNAMIC"    quantize_dynamic          INT8 dynamic (DynamicQuantizeLinear)
    "F16"        convert_float_to_float16  float16 weights & activations
    "F32"        -                         keep float32 (optimize only)
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
    run_optimizer,
)


# ============================== USER CONFIG ==============================

SOURCE_FOLDER = SCRIPT_DIR / "BigVGAN_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "BigVGAN_Optimized"

UPGRADE_OPSET   = 0        # Target ONNX opset (0 = keep current).
OPTIMIZER_LEVEL = 2        # 0 = no optimization, 1 = basic, 2 = extended


F16_OP_BLOCK_LIST = [      # op types kept out of any float16 conversion
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]


# ============================== MODEL PLANS =============================

# BigVGAN is a single convolutional vocoder graph (no attention), so num_heads/hidden_size stay 0.
MODEL_PLANS: dict[str, Plan] = {
    "BigVGAN": Plan(
        method="F32",
        num_heads=0,
        hidden_size=0,
        opt_level=2,
    ),
    "BigVGAN_Metadata": Plan(
        method="F32",
        num_heads=0,
        hidden_size=0,
        optimize=False,
        external=False,
    ),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    f16_max_finite_val=32767.0,
    f16_op_block_list=F16_OP_BLOCK_LIST,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    parse_args()
    run_optimizer(CONFIG)


if __name__ == "__main__":
    main()
