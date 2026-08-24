"""Optimize or quantize the four exported Raon-OpenTTS ONNX graphs.

Edit ``MODEL_PLANS`` to select F32, F16, DYNAMIC, Q2, Q4, or Q8 for each
graph. The implementation remains centralized in ``Optimize_ONNX_Common.py``.
Use ``Optimize_ONNX_DML.py`` for the DirectML-specific fusion profile.
"""

from __future__ import annotations

import argparse
import gc
import sys
from functools import lru_cache
from pathlib import Path

import onnx
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer  # noqa: E402
from Raon_Config import require_architecture  # noqa: E402


SOURCE_FOLDER = SCRIPT_DIR / "Raon_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "Raon_Optimized"
QUANT_METHOD  = "F32"  # Change to F16, DYNAMIC, Q2, Q4, or Q8 for quantization
OPTIMIZER_LEVEL = 2   # Change to 1 for CUDA, DML, OpenVINO


@lru_cache(maxsize=None)
def _source_transformer_architecture(source_path: str) -> tuple[int, int, int]:
    metadata_path = Path(source_path).resolve().with_name("Raon_Metadata.onnx")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Required Raon metadata graph is missing: {metadata_path}")
    model = onnx.load(metadata_path, load_external_data=False)
    metadata = {item.key: item.value for item in model.metadata_props}
    architecture = require_architecture(
        metadata.get("model_name"), f"model_name in {metadata_path}"
    )
    expected = (architecture.heads, architecture.head_dim, architecture.dim)
    try:
        actual = (
            int(metadata["model_heads"]),
            int(metadata["head_dim"]),
            int(metadata["model_dim"]),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Raon metadata must contain integer model_heads, head_dim, and model_dim: "
            f"{metadata_path}"
        ) from error
    if actual != expected:
        raise ValueError(
            f"Raon metadata architecture mismatch for {architecture.model_name}: "
            f"expected heads/head_dim/model_dim={expected}, found {actual}"
        )
    return actual


def _source_model_heads(source_path: str) -> int:
    return _source_transformer_architecture(source_path)[0]


def _source_attention_hidden_size(source_path: str) -> int:
    heads, head_dim, _ = _source_transformer_architecture(source_path)
    return heads * head_dim


MODEL_PLANS: dict[str, Plan] = {
    "Raon_Metadata": Plan(
        method="F32",
        num_heads=0,
        hidden_size=0,
        transformer=False,
    ),
    "Raon_Preprocess": Plan(
        method=QUANT_METHOD,
        num_heads=0,
        hidden_size=0,
        transformer=False,
        opt_level=OPTIMIZER_LEVEL,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
    ),
    "Raon_Transformer": Plan(
        method=QUANT_METHOD ,
        num_heads=_source_model_heads,
        hidden_size=_source_attention_hidden_size,
        transformer=True,
        opt_level=OPTIMIZER_LEVEL,
    ),
    "Raon_Decode": Plan(
        method=QUANT_METHOD,
        num_heads=0,
        hidden_size=0,
        transformer=False,
        opt_level=OPTIMIZER_LEVEL,
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    optimizer_level=OPTIMIZER_LEVEL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def require_source_graphs() -> None:
    missing = [
        SOURCE_FOLDER / f"{name}.onnx"
        for name in MODEL_PLANS
        if not (SOURCE_FOLDER / f"{name}.onnx").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required Raon ONNX source graph(s): {missing}")


def _tensor_interface(
    value: onnx.ValueInfoProto,
) -> tuple[str, int, tuple[int | str | None, ...]]:
    tensor_type = value.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return value.name, tensor_type.elem_type, tuple(dimensions)


def _interfaces_match(source: onnx.ModelProto, output: onnx.ModelProto) -> bool:
    source_inputs = [_tensor_interface(value) for value in source.graph.input]
    output_inputs = [_tensor_interface(value) for value in output.graph.input]
    if source_inputs != output_inputs:
        return False

    source_outputs = [_tensor_interface(value) for value in source.graph.output]
    output_outputs = [_tensor_interface(value) for value in output.graph.output]
    if len(source_outputs) != len(output_outputs):
        return False

    fixed_input_dimensions = {
        (len(shape), axis, dimension)
        for _, _, shape in source_inputs
        for axis, dimension in enumerate(shape)
        if isinstance(dimension, int)
    }
    for source_value, output_value in zip(source_outputs, output_outputs, strict=True):
        source_name, source_type, source_shape = source_value
        output_name, output_type, output_shape = output_value
        if (
            source_name != output_name
            or source_type != output_type
            or len(source_shape) != len(output_shape)
        ):
            return False
        for axis, (source_dimension, output_dimension) in enumerate(
            zip(source_shape, output_shape, strict=True)
        ):
            if source_dimension == output_dimension:
                continue
            if (
                isinstance(source_dimension, str)
                and isinstance(output_dimension, int)
                and (len(source_shape), axis, output_dimension) in fixed_input_dimensions
            ):
                continue
            return False
    return True


def _default_opsets(model: onnx.ModelProto) -> list[int]:
    return [item.version for item in model.opset_import if item.domain in {"", "ai.onnx"}]


def validate_optimized_package(
    source_folder: Path = SOURCE_FOLDER,
    output_folder: Path = OUTPUT_FOLDER,
    model_plans: dict[str, Plan] = MODEL_PLANS,
) -> None:
    for name in model_plans:
        source_path = source_folder / f"{name}.onnx"
        output_path = output_folder / f"{name}.onnx"
        if not output_path.is_file():
            raise FileNotFoundError(f"Optimizer did not produce required graph: {output_path}")
        onnx.checker.check_model(str(output_path))
        source_model = onnx.load(source_path, load_external_data=False)
        output_model = onnx.load(output_path, load_external_data=False)
        if not _interfaces_match(source_model, output_model):
            raise RuntimeError(f"Optimizer changed the graph interface: {name}")
        if _default_opsets(output_model) != _default_opsets(source_model):
            raise RuntimeError(f"Optimizer changed the default-domain opset: {name}")
        source_metadata = {item.key: item.value for item in source_model.metadata_props}
        output_metadata = {item.key: item.value for item in output_model.metadata_props}
        inconsistent_metadata = sorted(
            key for key, value in source_metadata.items() if output_metadata.get(key) != value
        )
        if inconsistent_metadata:
            raise RuntimeError(
                f"Optimizer changed required metadata in {name}: {inconsistent_metadata}"
            )
        forbidden = sorted(
            {
                node.op_type
                for node in output_model.graph.node
                if node.op_type in {"ATen", "PythonOp", "prim"}
                or node.domain.lower() in {"org.pytorch.aten", "prim"}
            }
        )
        if forbidden:
            raise RuntimeError(f"Optimized graph {name} contains forbidden operators: {forbidden}")
        unexpected_domains = sorted(
            {
                node.domain
                for node in output_model.graph.node
                if node.domain not in {"", "ai.onnx", "com.microsoft"}
            }
        )
        if unexpected_domains:
            raise RuntimeError(
                f"Optimized graph {name} contains unexpected domains: {unexpected_domains}"
            )
        del source_model, output_model
        gc.collect()
        session = ort.InferenceSession(
            str(output_path), providers=["CPUExecutionProvider"]
        )
        if session.get_providers()[0] != "CPUExecutionProvider":
            raise RuntimeError(f"CPU validation session fell back for {output_path}")
        del session
        gc.collect()
        print(f"Validated optimized graph: {output_path.name}")


def main() -> int:
    parse_args()
    require_source_graphs()
    run_optimizer(CONFIG)
    validate_optimized_package()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())