"""Optimize & quantize the exported F5-TTS ONNX modules (CPU / default providers).

Config-only front-end: this script only defines the per-module Plans and the shared
OptimizerConfig, then delegates the whole quantize/optimize pipeline to
``Optimize_ONNX_Common.py`` (the same structure every TTS export script now uses).

    Method       Backend                   Result
    "Q2/Q4/Q8"   matmul_nbits_quantizer    2/4/8-bit weight-only (MatMulNBits)
    "DYNAMIC"    quantize_dynamic          INT8 dynamic (DynamicQuantizeLinear)
    "F16"        convert_float_to_float16  float16 weights & activations
    "F32"        -                         keep float32 (optimize only)

For the DirectML-tuned optimization pass, use ``Optimize_ONNX_DML.py`` instead.
"""

from pathlib import Path
import sys


# ============================== SHARED PIPELINE =========================

# Reuse the shared optimizer pipeline: walk up to the repo root that holds it.
_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


# ============================== USER CONFIG ==============================

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "F5_ONNX")        # Source *.onnx modules.
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "F5_Optimized")   # Destination folder.

# ============================== MODEL PLANS =============================

# The DiT transformer uses 16 heads / hidden_size 1024; the same values feed the attention-fusion
# optimizer for every module (harmless for the non-attention Preprocess / Decode graphs).
# F5_Preprocess carries dynamic STFT shapes, so its onnxslim passes skip shape inference.
MODEL_PLANS: dict[str, Plan] = {
    "F5_Metadata":    Plan(method="F32", transformer=False),
    "F5_Preprocess":  Plan(method="F32", num_heads=0, hidden_size=0, first_slim_no_shape_infer=True, second_slim_no_shape_infer=True),
    "F5_Transformer": Plan(method="F32", num_heads=16, hidden_size=1024),
    "F5_Decode":      Plan(method="F32", num_heads=0, hidden_size=0),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    optimizer_level=2,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
