"""Quantize and optimize IndexTTS2 v2 or v2.5 graphs, then rebuild its package.

Each graph has an independent top-level plan. Compatible TTS autoregressive
graphs and emotion Qwen graphs may share weight-packing passes as an optional
fast path; every other plan is processed independently. The rebuilt bundle
deduplicates all identical packed tensors.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
INDEX_TTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = INDEX_TTS_DIR.parent
for import_path in (REPO_ROOT, INDEX_TTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Index_TTS.v2.Shared_Weights import (  # noqa: E402
    bundle_shared_initializers,
)
from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    process_model,
    quantize_weight_only_shared,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
)


STRATEGIES = ("greedy", "penalty_greedy", "sampling")
# User configuration
#
# This one optimizer supports the package exported by the matching selector in
# Export_IndexTTS2.py. The retained v2.5 package is selected by default; set
# this to ``"2"`` for IndexTTS2 v2. v2.5 additionally enables its verified
# tied-Q4 Qwen embedding/LM-head storage alias; v2 never enters that path.
MODEL_VERSION = "2.5"  # "2" | "2.5"
_SOURCE_FOLDER_NAMES = {
    "2": "IndexTTS2_ONNX",
    "2.5": "IndexTTS2_5_ONNX",
}
_OUTPUT_FOLDER_NAMES = {
    "2": "IndexTTS2_Optimized",
    "2.5": "IndexTTS2_5_Optimized",
}
_TEXT_TOKENIZER_FILES = {
    "2": "bpe.model",
    "2.5": "multilingual_zh_ja_yue_char_del.tiktoken",
}
if MODEL_VERSION not in _SOURCE_FOLDER_NAMES:
    raise ValueError(f"Unsupported MODEL_VERSION: {MODEL_VERSION!r}")
IS_V25 = MODEL_VERSION == "2.5"
SOURCE_FOLDER = SCRIPT_DIR / _SOURCE_FOLDER_NAMES[MODEL_VERSION]
OUTPUT_FOLDER = SCRIPT_DIR / _OUTPUT_FOLDER_NAMES[MODEL_VERSION]
TEXT_TOKENIZER_FILE = _TEXT_TOKENIZER_FILES[MODEL_VERSION]
QWEN_TOKENIZER_FOLDER = "qwen0.6bemo4-merge"
QWEN_TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
QUANTIZATION_TEMPLATE = "IndexTTS2_TargetPrefill_greedy"
QUANTIZATION_CACHE_NAME = ".IndexTTS2_QuantizedWeights.onnx"
QUANTIZATION_COVER_NAME = ".IndexTTS2_QuantizationCover.onnx"
EMOTION_QUANTIZATION_TEMPLATE = "IndexTTS2_EmotionTextPrefill"
EMOTION_QUANTIZATION_CACHE_NAME = ".IndexTTS2_EmotionQuantizedWeights.onnx"
EMOTION_QUANTIZATION_COVER_NAME = ".IndexTTS2_EmotionQuantizationCover.onnx"
WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}

# Quantization and optimization defaults
MATMUL_ALGORITHM    = "AFFINE_REFINE_V2"
BLOCK_SIZE          = 32
ACCURACY_LEVEL      = 4
MAIN_NUM_HEADS      = 20
MAIN_HIDDEN_SIZE    = 1280
EMOTION_NUM_HEADS   = 16
EMOTION_HIDDEN_SIZE = 1024
DYNAMIC_WEIGHT_TYPE = "QInt8"
DYNAMIC_PER_CHANNEL = False


def exclude_non_matrix_weights(model_path: str) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    constants = {
        node.output[0]: attribute.t
        for node in model.graph.node
        if node.op_type == "Constant" and len(node.output) == 1
        for attribute in node.attribute
        if attribute.HasField("t")
    }
    excluded = []
    for node in model.graph.node:
        if node.op_type == "MatMul" and len(node.input) == 2:
            weight = initializers.get(node.input[1], constants.get(node.input[1]))
        elif node.op_type == "Gather" and node.input:
            weight = initializers.get(node.input[0], constants.get(node.input[0]))
        else:
            continue
        if weight is not None and (
            len(weight.dims) != 2
            or (
                node.op_type == "Gather"
                and (
                    int(weight.dims[1]) < BLOCK_SIZE
                    or int(weight.dims[1]) % BLOCK_SIZE
                )
            )
        ):
            excluded.append(node.name)
    del model
    gc.collect()
    return excluded


def matrix_nodes(model_path: str) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    selected = [
        node.name
        for node in model.graph.node
        if node.op_type in {"MatMul", "Gather"}
        and node.name
    ]
    del model
    gc.collect()
    return selected


def convolution_nodes(model_path: str) -> list[str]:
    model = onnx.load(str(model_path), load_external_data=False)
    selected = [
        node.name
        for node in model.graph.node
        if node.op_type == "Conv" and node.name
    ]
    del model
    gc.collect()
    return selected


def copy_tokenizer_assets(destination: Path) -> None:
    text_tokenizer = SOURCE_FOLDER / TEXT_TOKENIZER_FILE
    if not text_tokenizer.is_file():
        raise FileNotFoundError(f"Missing exported text tokenizer: {text_tokenizer}")
    shutil.copy2(text_tokenizer, destination / TEXT_TOKENIZER_FILE)

    source_folder = SOURCE_FOLDER / QWEN_TOKENIZER_FOLDER
    target_folder = destination / QWEN_TOKENIZER_FOLDER
    target_folder.mkdir(parents=True, exist_ok=True)
    for name in QWEN_TOKENIZER_FILES:
        source = source_folder / name
        if source.is_file():
            shutil.copy2(source, target_folder / name)
    if not (target_folder / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Missing exported Qwen tokenizer: {source_folder}")


# Per-graph quantization and optimization plan
MODEL_PLANS: dict[str, Plan] = {
    "IndexTTS2_ReferencePreprocess": Plan(
        method="F32",
        optimize=True,
        transformer=False,
        external=True,
    ),
    "IndexTTS2_Conditioning": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "IndexTTS2_EmotionTextPrefill": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=False,
        num_heads=EMOTION_NUM_HEADS,
        hidden_size=EMOTION_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_EmotionTextDecode": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=False,
        num_heads=EMOTION_NUM_HEADS,
        hidden_size=EMOTION_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_TargetPrefill_greedy": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_TargetPrefill_penalty_greedy": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_TargetPrefill_sampling": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_DecodeStep_greedy": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_DecodeStep_penalty_greedy": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_DecodeStep_sampling": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        num_heads=MAIN_NUM_HEADS,
        hidden_size=MAIN_HIDDEN_SIZE,
        external=True,
    ),
    "IndexTTS2_Synthesis": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        nodes_to_include=matrix_nodes,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "IndexTTS2_CFMEstimator": Plan(
        method="Q4",
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul", "Gather"),
        axes=(0, 1),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        nodes_to_exclude=exclude_non_matrix_weights,
        optimize=True,
        transformer=True,
        external=True,
    ),
    "IndexTTS2_Decoder": Plan(
        method="F32",
        optimize=True,
        transformer=not IS_V25,
        external=True,
    ),
    # This manifest carrier has no weights or quantizable operators.
    "IndexTTS2_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}

STRATEGY_NAMES = tuple(
    f"IndexTTS2_{stage}_{strategy}"
    for stage in ("TargetPrefill", "DecodeStep")
    for strategy in STRATEGIES
)
SHARED_WEIGHT_GRAPH_NAMES = (*STRATEGY_NAMES, "IndexTTS2_Synthesis")
EMOTION_SHARED_WEIGHT_GRAPH_NAMES = (
    "IndexTTS2_EmotionTextPrefill",
    "IndexTTS2_EmotionTextDecode",
)
CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    block_size=BLOCK_SIZE,
    accuracy_level=ACCURACY_LEVEL,
    dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=DYNAMIC_PER_CHANNEL,
    dynamic_default_tensor_type=onnx.TensorProto.FLOAT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def configure_attention_precision() -> dict[str, str]:
    metadata = read_onnx_metadata(str(SOURCE_FOLDER / "IndexTTS2_Metadata.onnx"))
    package_version = metadata.get("model_version")
    if package_version is not None and package_version != MODEL_VERSION:
        raise ValueError(
            f"MODEL_VERSION={MODEL_VERSION!r} cannot optimize an IndexTTS2 "
            f"v{package_version} package."
        )
    if IS_V25 and package_version != "2.5":
        raise ValueError(
            "IndexTTS2.5 packages must declare model_version=2.5. "
            "Re-export the package with the unified exporter."
        )
    flags = {key: metadata.get(key) for key in ("use_f16_kv", "compute_in_f32")}
    invalid = {key: value for key, value in flags.items() if value not in {"0", "1"}}
    if flags["use_f16_kv"] == "1" and flags["compute_in_f32"] == "0":
        print(
            "[Precision] FP16 KV attention is requested. Optimization plans remain "
            "unrestricted; validate any precision changes introduced by ORT."
        )
    return metadata


def _single_node(
    model: onnx.ModelProto,
    op_type: str,
    name_fragment: str,
) -> onnx.NodeProto:
    matches = [
        node
        for node in model.graph.node
        if node.op_type == op_type and name_fragment in node.name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {op_type} node containing {name_fragment!r}, found {len(matches)}."
        )
    return matches[0]


def _tensor_external_data(tensor: onnx.TensorProto) -> dict[str, str]:
    if tensor.data_location != onnx.TensorProto.EXTERNAL:
        raise ValueError(f"Tensor {tensor.name!r} is not external data.")
    entries = {entry.key: entry.value for entry in tensor.external_data}
    if "location" not in entries or "length" not in entries:
        raise ValueError(f"Tensor {tensor.name!r} has no complete external range.")
    return entries


def _tensor_byte_length(tensor: onnx.TensorProto) -> int:
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        return int(_tensor_external_data(tensor)["length"])
    if tensor.raw_data:
        return len(tensor.raw_data)
    return len(onnx.numpy_helper.to_array(tensor).tobytes(order="C"))


def _tensor_storage_signature(tensor: onnx.TensorProto) -> tuple[Any, ...]:
    external = _tensor_external_data(tensor)
    return (
        tensor.data_type,
        tuple(int(dim) for dim in tensor.dims),
        external["location"],
        external.get("offset", "0"),
        external["length"],
    )


def _external_tensor_bytes_match(
    left: onnx.TensorProto,
    left_path: Path,
    right: onnx.TensorProto,
    right_path: Path,
) -> bool:
    left_data = _tensor_external_data(left)
    right_data = _tensor_external_data(right)
    length = int(left_data["length"])
    if length != int(right_data["length"]):
        return False
    with (left_path.parent / left_data["location"]).open("rb") as left_file, (
        right_path.parent / right_data["location"]
    ).open("rb") as right_file:
        left_file.seek(int(left_data.get("offset", "0")))
        right_file.seek(int(right_data.get("offset", "0")))
        remaining = length
        while remaining:
            size = min(remaining, 8 * 1024 * 1024)
            if left_file.read(size) != right_file.read(size):
                return False
            remaining -= size
    return True


def _exact_external_transpose(
    embedding: onnx.TensorProto,
    lm_head: onnx.TensorProto,
    model_path: Path,
) -> bool:
    embedding_shape = tuple(int(dim) for dim in embedding.dims)
    head_shape = tuple(int(dim) for dim in lm_head.dims)
    if (
        embedding.data_type != lm_head.data_type
        or len(embedding_shape) != 2
        or embedding_shape != tuple(reversed(head_shape))
    ):
        return False

    embedding_data = _tensor_external_data(embedding)
    head_data = _tensor_external_data(lm_head)
    dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(embedding.data_type))
    expected_bytes = int(np.prod(embedding_shape)) * dtype.itemsize
    if (
        int(embedding_data["length"]) != expected_bytes
        or int(head_data["length"]) != expected_bytes
    ):
        return False

    embedding_view = np.memmap(
        model_path.parent / embedding_data["location"],
        mode="r",
        dtype=dtype,
        offset=int(embedding_data.get("offset", "0")),
        shape=embedding_shape,
    )
    head_view = np.memmap(
        model_path.parent / head_data["location"],
        mode="r",
        dtype=dtype,
        offset=int(head_data.get("offset", "0")),
        shape=head_shape,
    )
    rows_per_chunk = max(1, (8 * 1024 * 1024) // (embedding_shape[1] * dtype.itemsize))
    try:
        for start in range(0, embedding_shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, embedding_shape[0])
            if not np.array_equal(embedding_view[start:end], head_view[:, start:end].T):
                return False
    finally:
        del embedding_view, head_view
    return True


def _tied_q4_components(
    model_path: Path,
) -> tuple[tuple[onnx.TensorProto, ...], tuple[onnx.TensorProto, ...]]:
    model = onnx.load(str(model_path), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    embed_node = _single_node(
        model,
        "GatherBlockQuantized",
        "/core/embed_tokens/",
    )
    head_node = _single_node(model, "MatMulNBits", "/core/lm_head/")
    if len(embed_node.input) != 4 or len(head_node.input) != 4:
        raise ValueError("Unexpected Q4 embedding or LM-head input count.")

    embed = tuple(initializers[name] for name in (embed_node.input[0], embed_node.input[2], embed_node.input[3]))
    head = tuple(initializers[name] for name in head_node.input[1:])
    gather_attrs = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in embed_node.attribute
    }
    head_attrs = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in head_node.attribute
    }
    vocab_size, hidden_size = (int(dim) for dim in embed[0].dims)
    if (
        tuple(embed[0].dims) != (vocab_size, hidden_size)
        or embed[0].data_type != onnx.TensorProto.UINT4
        or embed[1].data_type != onnx.TensorProto.FLOAT
        or embed[2].data_type != onnx.TensorProto.UINT4
        or gather_attrs.get("block_size") != BLOCK_SIZE
        or gather_attrs.get("gather_axis") != 0
        or gather_attrs.get("quantize_axis") != 1
        or head[0].data_type != onnx.TensorProto.UINT8
        or head[1].data_type != onnx.TensorProto.FLOAT
        or head[2].data_type != onnx.TensorProto.UINT8
        or head_attrs.get("bits") != 4
        or head_attrs.get("block_size") != BLOCK_SIZE
        or head_attrs.get("K") != hidden_size
        or head_attrs.get("N") != vocab_size
        or hidden_size % BLOCK_SIZE
        or tuple(head[0].dims) != (vocab_size, hidden_size // BLOCK_SIZE, BLOCK_SIZE // 2)
        or tuple(embed[1].dims) != (vocab_size, hidden_size // BLOCK_SIZE)
        or tuple(head[1].dims) != tuple(embed[1].dims)
        or tuple(embed[2].dims) != (vocab_size, hidden_size // BLOCK_SIZE)
        or tuple(head[2].dims) != (vocab_size, hidden_size // (2 * BLOCK_SIZE))
        or any(_tensor_byte_length(left) != _tensor_byte_length(right) for left, right in zip(embed, head))
    ):
        raise ValueError("The Q4 embedding/head layouts cannot share AFFINE_REFINE_V2 storage.")
    return embed, head


def tied_emotion_q4_data_aliases() -> dict[tuple[str, str], tuple[str, str]]:
    """Expose the tied Qwen LM head as a MatMulNBits view of embed_tokens."""
    source_prefill_path = SOURCE_FOLDER / "IndexTTS2_EmotionTextPrefill.onnx"
    source_decode_path = SOURCE_FOLDER / "IndexTTS2_EmotionTextDecode.onnx"
    output_paths = tuple(
        OUTPUT_FOLDER / f"{name}.onnx"
        for name in EMOTION_SHARED_WEIGHT_GRAPH_NAMES
    )
    try:
        source_prefill = onnx.load(str(source_prefill_path), load_external_data=False)
        source_decode = onnx.load(str(source_decode_path), load_external_data=False)
        raw_prefill = {tensor.name: tensor for tensor in source_prefill.graph.initializer}
        raw_decode = {tensor.name: tensor for tensor in source_decode.graph.initializer}
        raw_embed_node = _single_node(source_prefill, "Gather", "/core/embed_tokens/")
        raw_head_node = _single_node(source_prefill, "MatMul", "/core/lm_head/")
        raw_decode_embed_node = _single_node(source_decode, "Gather", "/core/embed_tokens/")
        raw_decode_head_node = _single_node(source_decode, "MatMul", "/core/lm_head/")
        raw_embed = raw_prefill[raw_embed_node.input[0]]
        raw_head = raw_prefill[raw_head_node.input[1]]
        if (
            not _exact_external_transpose(raw_embed, raw_head, source_prefill_path)
            or _tensor_storage_signature(raw_embed)
            != _tensor_storage_signature(raw_decode[raw_decode_embed_node.input[0]])
            or _tensor_storage_signature(raw_head)
            != _tensor_storage_signature(raw_decode[raw_decode_head_node.input[1]])
        ):
            raise ValueError("The source Qwen embedding and lm_head are not one tied table.")

        components = {
            path.name: _tied_q4_components(path)
            for path in output_paths
        }
        leader_path = output_paths[0]
        leader_embed, leader_head = components[leader_path.name]
        for path in output_paths[1:]:
            embed, head = components[path.name]
            for label, leader, current in (
                ("embedding", leader_embed, embed),
                ("lm_head", leader_head, head),
            ):
                if not all(
                    _external_tensor_bytes_match(left, leader_path, right, path)
                    for left, right in zip(leader, current)
                ):
                    raise ValueError(
                        f"Emotion prefill and decode use different {label} Q4 packs."
                    )

        aliases: dict[tuple[str, str], tuple[str, str]] = {}
        for path in output_paths:
            embed, head = components[path.name]
            for embed_tensor, head_tensor, leader_tensor in zip(embed, head, leader_embed):
                leader = (leader_path.name, leader_tensor.name)
                aliases[(path.name, embed_tensor.name)] = leader
                aliases[(path.name, head_tensor.name)] = leader
        saved_bytes = sum(_tensor_byte_length(tensor) for tensor in leader_embed)
        print(
            "[Tied Q4] Emotion Qwen lm_head shares the embed_tokens AFFINE_REFINE_V2 Q4 "
            f"payload; recovering {saved_bytes / (1024 * 1024):.2f} MiB."
        )
        return aliases
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"[Tied Q4] Emotion Qwen storage alias skipped: {error}")
        return {}


def resolve_initializer_alias(name: str, aliases: dict[str, str]) -> str:
    seen = set()
    while name in aliases:
        seen.add(name)
        name = aliases[name]
    return name


def weight_quantization_signature(plan: Any) -> tuple[Any, ...]:
    def selector_signature(selector: Any) -> Any:
        return tuple(selector) if isinstance(selector, list) else selector

    return (
        plan.method,
        plan.algo,
        plan.op_types,
        plan.axes,
        plan.block_size,
        plan.accuracy_level,
        plan.symmetric,
        plan.quant_format,
        selector_signature(plan.nodes_to_exclude),
        selector_signature(plan.nodes_to_include),
    )


def shared_weight_plan(
    resolved_plans: dict[str, Any],
    graph_names: tuple[str, ...],
    template_name: str,
) -> Any | None:
    template_plan = resolved_plans[template_name]
    template_signature = weight_quantization_signature(template_plan)
    if template_plan.method not in WEIGHT_ONLY_BITS:
        print(
            f"[Shared quantization] {template_name} uses {template_plan.method}; "
            "processing this group independently."
        )
        return None
    incompatible = [
        name
        for name in graph_names
        if weight_quantization_signature(resolved_plans[name]) != template_signature
    ]
    if incompatible:
        print(
            "[Shared quantization] This group has independent quantization plans; "
            f"processing every graph separately (different plans: {incompatible})."
        )
        return None
    return template_plan


def collect_constant_weight_entries(
    model_path: Path,
    plan: Any,
) -> dict[tuple[str, str], tuple[Any, Any, Path]]:
    model = onnx.load(str(model_path), load_external_data=False)
    included = (
        plan.nodes_to_include(str(model_path))
        if callable(plan.nodes_to_include)
        else plan.nodes_to_include
    )
    excluded = (
        plan.nodes_to_exclude(str(model_path))
        if callable(plan.nodes_to_exclude)
        else plan.nodes_to_exclude
    )
    included = None if included is None else set(included)
    excluded = set(excluded or ())
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    aliases = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.op_type == "Identity" and len(node.input) == len(node.output) == 1
    }
    entries = {}
    for node in model.graph.node:
        if (
            node.op_type not in plan.op_types
            or (included is not None and node.name not in included)
            or node.name in excluded
        ):
            continue
        if node.op_type == "MatMul" and len(node.input) == 2:
            weight_name = resolve_initializer_alias(node.input[1], aliases)
        elif node.op_type == "Gather" and node.input:
            weight_name = resolve_initializer_alias(node.input[0], aliases)
        else:
            continue
        weight = initializers.get(weight_name)
        if weight is None or len(weight.dims) != 2:
            continue
        node_copy = onnx.NodeProto()
        node_copy.CopyFrom(node)
        weight_copy = onnx.TensorProto()
        weight_copy.CopyFrom(weight)
        entries.setdefault(
            (node.op_type, weight_name),
            (node_copy, weight_copy, model_path),
        )
    del model
    gc.collect()
    return entries


def collect_shared_weight_entries(
    resolved_plans: dict[str, Any],
    graph_names: tuple[str, ...],
) -> tuple[
    dict[tuple[str, str], tuple[Any, Any, Path]],
    dict[str, set[tuple[str, str]]],
]:
    all_entries = {}
    weight_sets = {}
    for name in graph_names:
        entries = collect_constant_weight_entries(
            SOURCE_FOLDER / f"{name}.onnx",
            resolved_plans[name],
        )
        weight_sets[name] = set(entries)
        for signature, entry in entries.items():
            prior = all_entries.get(signature)
            if prior is not None:
                prior_node, prior_weight, _ = prior
                node_attributes = [attr.SerializeToString() for attr in entry[0].attribute]
                prior_attributes = [attr.SerializeToString() for attr in prior_node.attribute]
            else:
                all_entries[signature] = entry

    shared_weights = set.intersection(*(weight_sets[name] for name in graph_names))
    print(
        f"[Coverage] One pass owns {len(all_entries)} unique matrix weights across "
        f"{len(graph_names)} graphs; {len(shared_weights)} weights are common to all."
    )
    return all_entries, weight_sets


def build_quantization_cover(
    resolved_plans: dict[str, Any],
    cover_path: Path,
    graph_names: tuple[str, ...],
    template_name: str,
) -> int:
    entries, _ = collect_shared_weight_entries(resolved_plans, graph_names)
    nodes = []
    inputs = []
    outputs = []
    initializers = {}

    for index, (signature, entry) in enumerate(sorted(entries.items())):
        op_type, weight_name = signature
        source_node, source_weight, source_path = entry
        weight = onnx.TensorProto()
        weight.CopyFrom(source_weight)
        for external_entry in weight.external_data:
            if external_entry.key == "location":
                source_data = (source_path.parent / external_entry.value).resolve()
                external_entry.value = os.path.relpath(source_data, cover_path.parent)
        prior_weight = initializers.get(weight_name)
        initializers.setdefault(weight_name, weight)

        input_name = f"quant_cover_input_{index}"
        output_name = f"quant_cover_output_{index}"
        if op_type == "MatMul":
            input_type = weight.data_type
            input_shape = [1, weight.dims[0]]
            output_shape = [1, weight.dims[1]]
            node_inputs = [input_name, weight_name]
        else:
            axis = next(
                (
                    onnx.helper.get_attribute_value(attribute)
                    for attribute in source_node.attribute
                    if attribute.name == "axis"
                ),
                0,
            )
            axis = axis if axis >= 0 else axis + len(weight.dims)
            input_type = onnx.TensorProto.INT64
            input_shape = [1]
            output_shape = list(weight.dims)
            output_shape[axis : axis + 1] = [1]
            node_inputs = [weight_name, input_name]

        inputs.append(
            onnx.helper.make_tensor_value_info(input_name, input_type, input_shape)
        )
        outputs.append(
            onnx.helper.make_tensor_value_info(output_name, weight.data_type, output_shape)
        )
        cover_node = onnx.helper.make_node(
            op_type,
            node_inputs,
            [output_name],
            name=f"quant_cover/{index}/{op_type}",
        )
        for attribute in source_node.attribute:
            cover_node.attribute.add().CopyFrom(attribute)
        nodes.append(cover_node)

    template = onnx.load(
        str(SOURCE_FOLDER / f"{template_name}.onnx"),
        load_external_data=False,
    )
    graph = onnx.helper.make_graph(
        nodes,
        "IndexTTS2_QuantizationCover",
        inputs,
        outputs,
        initializer=list(initializers.values()),
    )
    cover = onnx.helper.make_model(
        graph,
        producer_name="IndexTTS2 shared quantization cover",
        opset_imports=[
            onnx.helper.make_opsetid(opset.domain, opset.version)
            for opset in template.opset_import
        ],
    )
    cover.ir_version = template.ir_version
    onnx.save(cover, str(cover_path))
    print(f"[Quantization cover] Built {len(entries)} unique operator/weight recipes.")
    del cover, graph, template
    gc.collect()
    return len(entries)


def resolve_plans() -> dict[str, Any]:
    resolved_plans = {}
    for name, plan in MODEL_PLANS.items():
        resolved = resolve_plan(plan, CONFIG)
        resolved_plans[name] = resolved
    return resolved_plans


def quantize_shared_weights(
    resolved_plans: dict[str, Any],
    cache_path: Path,
    cover_path: Path,
    graph_names: tuple[str, ...],
    template_name: str,
) -> set[str]:
    template_plan = shared_weight_plan(
        resolved_plans,
        graph_names,
        template_name,
    )
    if template_plan is None:
        return set()
    try:
        unique_weights = build_quantization_cover(
            resolved_plans,
            cover_path,
            graph_names,
            template_name,
        )
        stats = quantize_weight_only_shared(
            str(cover_path),
            [
                (
                    str(SOURCE_FOLDER / f"{name}.onnx"),
                    str(OUTPUT_FOLDER / f"{name}.onnx"),
                )
                for name in graph_names
            ],
            str(cache_path),
            template_plan,
            bits=WEIGHT_ONLY_BITS[template_plan.method],
            external=True,
        )
    except Exception as error:
        print(
            "[Shared quantization] Shared packing was not applicable; "
            f"processing this group independently ({error})."
        )
        return set()
    print(
        f"[Shared quantization] Packed {stats['unique_weights']} weights once; "
        f"reused them at {stats['total_rewrites']} nodes across {stats['graph_count']} graphs."
    )
    return set(graph_names)


def process_graphs(
    resolved_plans: dict[str, Any],
    prequantized_graphs: set[str],
) -> None:
    mixed_precision = uses_mixed_precision(resolved_plans.values())
    if mixed_precision and CONFIG.f16_keep_io_types is None:
        print(
            "[Precision] Not all graphs use F16; enabling keep_io_types for "
            "float16 conversions."
        )
    for name, plan in resolved_plans.items():
        shared = name in prequantized_graphs
        detail = ", shared weights" if shared else ""
        print(f"\nProcessing graph: {name} [{plan.method}{detail}, optimize={plan.optimize}]")
        process_model(
            name,
            plan,
            CONFIG,
            mixed_precision=mixed_precision,
            prequantized=shared,
        )
def rebuild_shared_bundle(
    metadata: dict[str, str],
    cache_path: Path,
    data_aliases: dict[tuple[str, str], tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    model_paths = [OUTPUT_FOLDER / f"{name}.onnx" for name in MODEL_PLANS]
    stats = bundle_shared_initializers(
        OUTPUT_FOLDER,
        model_paths=model_paths,
        metadata=metadata,
        data_aliases=data_aliases,
    )
    cache_path.unlink(missing_ok=True)
    Path(str(cache_path) + ".data").unlink(missing_ok=True)
    replace_onnx_metadata(
        str(OUTPUT_FOLDER / "IndexTTS2_Metadata.onnx"),
        metadata,
    )
    print(
        f"[Shared bundle] {stats['initializer_references']} references -> "
        f"{stats['unique_initializers']} tensors."
    )
    return stats


def main() -> None:
    args = parse_args()
    resolved_plans = resolve_plans()
    shared_weight_plan(
        resolved_plans,
        SHARED_WEIGHT_GRAPH_NAMES,
        QUANTIZATION_TEMPLATE,
    )
    shared_weight_plan(
        resolved_plans,
        EMOTION_SHARED_WEIGHT_GRAPH_NAMES,
        EMOTION_QUANTIZATION_TEMPLATE,
    )
    metadata = configure_attention_precision()
    if OUTPUT_FOLDER.exists():
        shutil.rmtree(OUTPUT_FOLDER)
    OUTPUT_FOLDER.mkdir(parents=True)
    copy_tokenizer_assets(OUTPUT_FOLDER)
    cache_path = OUTPUT_FOLDER / QUANTIZATION_CACHE_NAME
    cover_path = SOURCE_FOLDER / QUANTIZATION_COVER_NAME
    emotion_cache_path = OUTPUT_FOLDER / EMOTION_QUANTIZATION_CACHE_NAME
    emotion_cover_path = SOURCE_FOLDER / EMOTION_QUANTIZATION_COVER_NAME
    try:
        prequantized_graphs = quantize_shared_weights(
            resolved_plans,
            cache_path,
            cover_path,
            SHARED_WEIGHT_GRAPH_NAMES,
            QUANTIZATION_TEMPLATE,
        )
        prequantized_graphs.update(
            quantize_shared_weights(
                resolved_plans,
                emotion_cache_path,
                emotion_cover_path,
                EMOTION_SHARED_WEIGHT_GRAPH_NAMES,
                EMOTION_QUANTIZATION_TEMPLATE,
            )
        )
        process_graphs(resolved_plans, prequantized_graphs)
        rebuild_shared_bundle(
            metadata,
            cache_path,
            tied_emotion_q4_data_aliases() if IS_V25 else None,
        )
    finally:
        for temporary_path in (
            cache_path,
            cover_path,
            emotion_cache_path,
            emotion_cover_path,
        ):
            temporary_path.unlink(missing_ok=True)
            Path(str(temporary_path) + ".data").unlink(missing_ok=True)


if __name__ == "__main__":
    main()