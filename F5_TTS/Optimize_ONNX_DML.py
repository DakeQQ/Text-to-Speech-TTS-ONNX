"""Optimize the exported F5-TTS ONNX modules for the DirectML (DML) execution provider.

Config-only front-end: this script only defines the per-module Plans and the shared
OptimizerConfig, then delegates the whole optimize pipeline to ``Optimize_ONNX_Common.py``
(the same structure every TTS export script now uses).

The ORT graph optimizer is designed for CUDA / ORT-GPU; a few of its fusions hurt DirectML,
so this profile disables them (GroupNorm, NHWC Conv, QOrdered MatMul) and runs the optimizer
with ``use_gpu=True`` at opt_level 0. The graphs stay float32.
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

# ORT-DirectML fusion tweaks: GroupNorm has a very negative effect on VRAM/CPU use; NHWC Conv and
# QOrdered MatMul cause performance issues / have no effect on DML.
DML_FUSION_OPTIONS = {
    "enable_group_norm": False,
    "enable_nhwc_conv": False,
    "enable_qordered_matmul": False,
}


# ============================== MODEL PLANS =============================

# F5_Preprocess carries dynamic STFT shapes, so its onnxslim passes skip shape inference.
MODEL_PLANS: dict[str, Plan] = {
    "F5_Metadata":    Plan(method="F32", transformer=False),
    "F5_Preprocess":  Plan(method="F32", num_heads=16, hidden_size=1024,
                           first_slim_no_shape_infer=True, second_slim_no_shape_infer=True),
    "F5_Transformer": Plan(method="F32", num_heads=16, hidden_size=1024),
    "F5_Decode":      Plan(method="F32", num_heads=16, hidden_size=1024),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    optimizer_level=0,
    optimizer_use_gpu=True,
    optimizer_provider="CPUExecutionProvider",
    optimizer_only_onnxruntime=False,
    optimizer_fusion_options=DML_FUSION_OPTIONS,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
