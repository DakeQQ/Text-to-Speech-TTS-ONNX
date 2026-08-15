"""Optimize Inflect ONNX graphs for float16 or float32 execution.

Set ``MODEL_PRECISION`` below to ``"F16"`` for float16 weights and compute or
``"F32"`` for float32 optimization. Graph interfaces remain in their exported
dtypes so the duration and decode stages stay directly compatible.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path



SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from Optimize_ONNX_Common import (  # noqa: E402
	OptimizerConfig,
	Plan,
	process_model,
	read_onnx_metadata,
	replace_onnx_metadata,
	resolve_plan,
)
from Shared_Weights import (  # noqa: E402
	bundle_shared_initializers,
)


# ============================== USER CONFIG ==============================

SOURCE_FOLDER = SCRIPT_DIR / "Inflect_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "Inflect_Optimized"
MODEL_PRECISION = "F32"  # F16 | F32
OPTIMIZER_LEVEL = 2      # 0 = disabled, 1 = basic, 2 = extended

F16_OP_BLOCK_LIST = []

# ========================================================================

METADATA_MODEL_NAME = "Inflect_Metadata.onnx"
FUNCTIONAL_MODELS = ("Inflect_Duration", "Inflect_Decode")
SUPPORTED_PRECISIONS = {"F16", "F32"}


# ============================== MODEL PLANS ==============================

def model_plans(precision: str) -> dict[str, Plan]:
	return {
		name: Plan(
			method=precision,
			optimize=True,
			transformer=True,
			opt_level=OPTIMIZER_LEVEL,
			external=False,
			first_slim_no_shape_infer=True,
		)
		for name in FUNCTIONAL_MODELS
	}


# ============================== PIPELINE ================================

def optimizer_config(plans: dict[str, Plan]) -> OptimizerConfig:
	return OptimizerConfig(
		original_folder_path=str(SOURCE_FOLDER),
		optimized_folder_path=str(OUTPUT_FOLDER),
		model_plans=plans,
		optimizer_level=OPTIMIZER_LEVEL,
		f16_keep_io_types=False,
		f16_force_initializers=True,
		f16_max_finite_val=32767.0,
		f16_op_block_list=F16_OP_BLOCK_LIST,
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	return parser.parse_args()


def resolve_plans(config: OptimizerConfig):
	resolved = {}
	for name, plan in config.model_plans.items():
		resolved_plan = resolve_plan(plan, config)
		resolved[name] = resolved_plan
	return resolved


def rebuild_package(metadata: dict[str, str], precision: str) -> None:
	optimized_metadata = {
		**metadata,
		"graph_precision": precision,
	}
	model_paths = [OUTPUT_FOLDER / f"{name}.onnx" for name in FUNCTIONAL_MODELS]
	stats = bundle_shared_initializers(
		OUTPUT_FOLDER,
		model_paths=model_paths,
		metadata=optimized_metadata,
	)
	metadata_path = OUTPUT_FOLDER / METADATA_MODEL_NAME
	shutil.copy2(SOURCE_FOLDER / METADATA_MODEL_NAME, metadata_path)
	replace_onnx_metadata(str(metadata_path), optimized_metadata)
	print(
		f"[Shared weights] {stats['initializer_references']} references -> "
		f"{stats['unique_initializers']} tensors, "
		f"{stats['unique_bytes'] / (1024 * 1024):.2f} MiB"
	)


def process_graphs(config: OptimizerConfig, resolved: dict) -> None:
	previous_directory = Path.cwd()
	with tempfile.TemporaryDirectory(prefix="inflect_onnx_optimize_") as scratch_dir:
		try:
			os.chdir(scratch_dir)
			for name, plan in resolved.items():
				print(f"\nOptimizing {name} [{plan.method}]")
				process_model(name, plan, config, mixed_precision=False)
		finally:
			os.chdir(previous_directory)


def main() -> None:
	parse_args()
	precision = MODEL_PRECISION.upper()
	plans = model_plans(precision)
	config = optimizer_config(plans)
	resolved = resolve_plans(config)
	metadata = read_onnx_metadata(SOURCE_FOLDER / METADATA_MODEL_NAME)
	if OUTPUT_FOLDER.exists():
		shutil.rmtree(OUTPUT_FOLDER)
	OUTPUT_FOLDER.mkdir(parents=True)
	process_graphs(config, resolved)
	rebuild_package(metadata, precision)
	print(f"Inflect optimized package written to {OUTPUT_FOLDER}")


if __name__ == "__main__":
	main()
