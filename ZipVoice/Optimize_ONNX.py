"""Optimize ZipVoice components and compose a quantized ONNX package.

Set each transformer method to ``F32``, ``F16``, ``DYNAMIC``, ``Q4``, or
``Q8``. Processing remains centralized in ``Optimize_ONNX_Common.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from Export_ZipVoice import COMPONENT_FOLDER as EXPORTED_COMPONENT_FOLDER
from Export_ZipVoice import OUTPUT_FOLDER as EXPORTED_PACKAGE_FOLDER
from Export_ZipVoice import VARIANT


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer  # noqa: E402


# ============================== USER CONFIG ==============================

SOURCE_FOLDER = EXPORTED_COMPONENT_FOLDER
OUTPUT_FOLDER = SCRIPT_DIR / f"{VARIANT.package_stem}_Optimized"


# ============================== MODEL PLANS ==============================

MODEL_PLANS: dict[str, Plan] = {
    f"{VARIANT.package_stem}_Metadata": Plan(
        method="F32",
        transformer=False,
    ),
    f"{VARIANT.package_stem}_Preprocess": Plan(
        method="F32",
        transformer=False,
        optimize=False,
    ),
    f"{VARIANT.package_stem}_TextEncoder": Plan(
        method="F32",
        num_heads=4,
        hidden_size=192,
        transformer=True,
        opt_level=2,
    ),
    f"{VARIANT.package_stem}_FlowCondition": Plan(
        method="F32",
        transformer=True,
        optimize=True,
    ),
    f"{VARIANT.package_stem}_FlowGeometry": Plan(
        method="F32",
        transformer=True,
        optimize=True,
    ),
    f"{VARIANT.package_stem}_TimeEmbedding": Plan(
        method="F32",
        transformer=True,
        optimize=True,
    ),
    f"{VARIANT.package_stem}_FlowStep": Plan(
        method="Q8",
        block_size=64,
        num_heads=4,
        hidden_size=512,
        transformer=True,
        optimize=True,  
        opt_level=2,
    ),
    f"{VARIANT.package_stem}_Decode": Plan(
        method="Q8",
        block_size=64,
        transformer=True,
        optimize=True,
        opt_level=2,
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    optimizer_level=2,
    copy_artifacts=("tokens.txt", "model.json"),
)


def parse_args() -> argparse.Namespace:
    return argparse.ArgumentParser(description=__doc__).parse_args()


def _validate_source_models() -> None:
    missing = [
        SOURCE_FOLDER / f"{model_name}.onnx"
        for model_name in MODEL_PLANS
        if not (SOURCE_FOLDER / f"{model_name}.onnx").is_file()
    ]
    if missing:
        missing_list = "\n".join(f"  - {path.name}" for path in missing)
        raise FileNotFoundError(
            f"Cannot optimize {VARIANT.package_stem}: exported component models are "
            f"missing from {SOURCE_FOLDER}:\n{missing_list}\n"
            f"Run {sys.executable} {SCRIPT_DIR / 'Export_ZipVoice.py'} first."
        )


def main() -> None:
    parse_args()
    _validate_source_models()
    for suffix in ("Prepare", "Pipeline"):
        path = OUTPUT_FOLDER / f"{VARIANT.package_stem}_{suffix}.onnx"
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".data").unlink(missing_ok=True)
    run_optimizer(CONFIG)
    from Merge_ONNX import _merge_pipeline

    _merge_pipeline(OUTPUT_FOLDER)
    for model_name in MODEL_PLANS:
        if model_name == f"{VARIANT.package_stem}_Metadata":
            continue
        path = OUTPUT_FOLDER / f"{model_name}.onnx"
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".data").unlink(missing_ok=True)
    shutil.rmtree(SOURCE_FOLDER)
    for suffix in ("Metadata", "Pipeline"):
        raw_model = EXPORTED_PACKAGE_FOLDER / f"{VARIANT.package_stem}_{suffix}.onnx"
        raw_model.unlink(missing_ok=True)
        raw_model.with_name(raw_model.name + ".data").unlink(missing_ok=True)
    try:
        EXPORTED_PACKAGE_FOLDER.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
