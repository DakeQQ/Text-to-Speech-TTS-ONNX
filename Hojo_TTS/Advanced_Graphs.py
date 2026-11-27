"""Hojo graph specialization, hardware FP32 stabilization, streaming vocoder adaptation, and KV-cache transforms."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
from Export_Hojo_TTS import (
    const,
    dce,
    node_inputs,
    remap_node_inputs,
    remove_redundant_casts,
)
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper as NH
from Shared_Weights import NORM_OPS, iter_subgraphs, set_sorted_metadata

DECODE_NAME = "Hojo_LM_Decode.onnx"
FINE_NAME = "Hojo_Fine_Stream.onnx"
DECODER_NAME = "Hojo_Decoder_Stream.onnx"
KV_CUSTOM_DOMAIN = "com.custom.llm"
KV_CUSTOM_OP = "KVCacheUpdate"
GOLDEN_PROMPTS: tuple[str, ...] = (
    "你好！欢迎使用离线语音合成。",
    "今天天气不错，我们去公园散步吧。",
    "语音合成模型在手机上实时运行，延迟很低。",
    "他说：“明天见！”然后转身离开了。",
    "二零二四年十月二十七日，下午三点十五分。",
    "百分之九十五的用户表示满意；剩下的还在观望。",
    "这是一个《测试》句子（包含括号）、顿号和问号？",
    "北京欢迎你",
    "人工智能正在改变世界……",
    "请问，ONNX Runtime 在安卓上快吗？",
)


def _bits_info(source: onnx.ModelProto) -> onnx.ValueInfoProto:
    for value in [*source.graph.output, *source.graph.value_info]:
        if value.name != "bits":
            continue
        dims = value.type.tensor_type.shape.dim
        if value.type.tensor_type.elem_type != TP.FLOAT:
            raise ValueError("Unsupported fine-local binary-code shape")
        if not dims:
            # Standard inference loses shapes across contrib skip norms; recover only the certified batch squeeze.
            producers = {name: node for node in source.graph.node for name in node.output}
            weights = {tensor.name: tensor for tensor in source.graph.initializer}
            bits = producers.get("bits")
            sign = producers.get(bits.input[0]) if bits is not None and bits.op_type == "Where" else None
            squeeze = producers.get(sign.input[0]) if sign is not None and sign.op_type == "Greater" else None
            logits = producers.get(squeeze.input[0]) if squeeze is not None and squeeze.op_type == "Squeeze" else None
            hidden = next((item for item in source.graph.input if item.name == "hidden_states"), None)
            axes = weights.get(squeeze.input[1]) if squeeze is not None and len(squeeze.input) == 2 else None
            projections = [producers[name] for name in logits.input if name in producers
                           and producers[name].op_type == "MatMul"] if logits is not None and logits.op_type == "Add" else []
            projection = projections[0] if len(projections) == 1 else None
            weight = weights.get(projection.input[1]) if projection is not None else None
            if (hidden is None or len(hidden.type.tensor_type.shape.dim) != 3
                    or hidden.type.tensor_type.shape.dim[0].dim_value != 1
                    or axes is None or not np.array_equal(NH.to_array(axes), [0])
                    or weight is None or list(weight.dims) != [512, 128]):
                raise ValueError("Unsupported fine-local binary-code shape")
            frame = hidden.type.tensor_type.shape.dim[1]
            return H.make_tensor_value_info("bits", TP.FLOAT, [frame.dim_value or frame.dim_param or "frames", 128])
        if len(dims) not in (2, 3) or dims[-1].dim_value != 128:
            raise ValueError("Unsupported fine-local binary-code shape")
        result = onnx.ValueInfoProto()
        result.CopyFrom(value)
        return result
    raise ValueError("Infer fine-local binary-code shape before preparing streaming graphs")


def extract_model(
    source: onnx.ModelProto,
    inputs: Sequence[onnx.ValueInfoProto],
    outputs: Sequence[onnx.ValueInfoProto],
    name: str,
) -> onnx.ModelProto:
    producers = {value: node for node in source.graph.node for value in node.output}
    boundaries = {value.name for value in inputs}
    frontier, needed = [value.name for value in outputs], set()
    while frontier:
        value = frontier.pop()
        if value in needed:
            continue
        needed.add(value)
        if value not in boundaries and value in producers:
            frontier.extend(node_inputs(producers[value]))
    internal = needed - boundaries
    nodes = [node for node in source.graph.node if any(value in internal for value in node.output)]
    initializers = [tensor for tensor in source.graph.initializer if tensor.name in internal]
    actual_inputs = [value for value in inputs if value.name in needed]
    graph = H.make_graph(nodes, name, actual_inputs, list(outputs), initializer=initializers)
    model = H.make_model(graph, opset_imports=source.opset_import, producer_name="Hojo_Advanced")
    model.ir_version = source.ir_version
    H.set_model_props(model, {item.key: item.value for item in source.metadata_props})
    dce(model)
    onnx.checker.check_model(model)
    return model


def make_fine_reference(source: onnx.ModelProto) -> onnx.ModelProto:
    inferred = onnx.shape_inference.infer_shapes(source)
    return extract_model(inferred, list(inferred.graph.input), [_bits_info(inferred)], "hojo_fine_reference")


def store_weights_fp16(
    model: onnx.ModelProto, *, min_elements: int, min_positive: float, max_finite: float,
) -> int:
    """Store large FP32 initializers as FP16 widened by a leading Cast; graph I/O and compute stay FP32."""
    from onnxruntime.transformers.float16 import convert_np_to_float16

    graph = model.graph
    taken = {tensor.name for tensor in graph.initializer} | {value.name for value in graph.input}
    taken |= {output for node in graph.node for output in node.output}
    tensors, casts = [], []
    for tensor in graph.initializer:
        if tensor.data_type != TP.FLOAT or int(np.prod(tensor.dims)) < min_elements:
            tensors.append(tensor)
            continue
        stored = f"{tensor.name}_f16"
        if stored in taken:
            raise ValueError(f"FP16 weight name collides with an existing value: {stored}")
        # Same rounding as the F16 vocoder plan, so shared-initializer bundling can deduplicate both copies.
        half = convert_np_to_float16(NH.to_array(tensor), min_positive, max_finite).astype(np.float16)
        tensors.append(NH.from_array(half, stored))
        casts.append(H.make_node("Cast", [stored], [tensor.name], to=TP.FLOAT, name=f"{stored}_widen"))
    del graph.initializer[:]
    graph.initializer.extend(tensors)
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    return len(casts)


def specialize_decode(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(source)
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get("graph_layout") != "hojo_lm_vocoder_v1":
        raise ValueError("Decode specialization requires the Hojo LM package contract")
    graph = model.graph
    allowed = {"input_ids", "voice", "ctrl", "uniform", "rng_seed", "rng_step", "seen"}
    if any(value.name not in allowed and not value.name.startswith("past_key_values.") for value in graph.input):
        raise ValueError("Decode specialization cannot discard an external attention/padding mask")
    ids = next(value for value in graph.input if value.name == "input_ids")
    ids.type.tensor_type.shape.dim[1].Clear()
    ids.type.tensor_type.shape.dim[1].dim_value = 1
    producers = {value: node for node in graph.node for value in node.output}
    removed = 0
    for softmax in graph.node:
        if softmax.op_type != "Softmax" or "/self_attn/" not in softmax.name:
            continue
        add = producers.get(softmax.input[0])
        if add is None or add.op_type != "Add":
            raise ValueError(f"Unrecognized decode attention mask at {softmax.name}")
        scores = []
        for value in add.input:
            node = producers.get(value)
            if node is None:
                continue
            if node.op_type in ("MatMul", "FusedMatMul") or node.op_type == "Mul" and any(
                    producers.get(item) is not None and producers[item].op_type in ("MatMul", "FusedMatMul")
                    for item in node.input):
                scores.append(value)
        if len(scores) != 1:
            raise ValueError(f"Ambiguous decode scores at {softmax.name}")
        softmax.input[0] = scores[0]
        removed += 1
    if removed != int(metadata["num_layers"]):
        raise ValueError(f"Decode specialization found {removed} attention layers")
    del graph.value_info[:]
    dce(model)
    model = onnxslim.slim(model)
    H.set_model_props(model, {**metadata, "lm_sequence_length": "1", "lm_kv_update": "concat"})
    onnx.checker.check_model(model)
    return model


def flatten_decode_mqa(source: onnx.ModelProto) -> onnx.ModelProto:
    """Use one four-row matrix product instead of four broadcast single-row products."""
    model = onnx.ModelProto()
    model.CopyFrom(source)
    graph = model.graph
    metadata = {item.key: item.value for item in model.metadata_props}
    ids = next(value for value in graph.input if value.name == "input_ids")
    if ids.type.tensor_type.shape.dim[1].dim_value != 1 or int(metadata["kv_heads"]) != 1:
        raise ValueError("Flattened MQA requires single-token decoding and one KV head")
    heads, width = int(metadata["num_heads"]), int(metadata["head_dim"])
    producers = {value: node for node in graph.node for value in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    key_input = next(value for value in graph.input if value.name == "past_key_values.0.key")
    transposed_cache = key_input.type.tensor_type.shape.dim[2].dim_value == width
    graph.initializer.extend([
        const("r2_mqa_q_shape", [1, heads, width], np.int64),
        const("r2_mqa_k_shape", [1, width, -1], np.int64),
        const("r2_mqa_v_shape", [1, -1, width], np.int64),
        const("r2_mqa_score_shape", [1, heads, 1, -1], np.int64),
        const("r2_mqa_prob_shape", [1, heads, -1], np.int64),
        const("r2_mqa_context_shape", [1, heads, 1, width], np.int64),
        const("r2_mqa_flat_shape", [1, 1, heads * width], np.int64),
    ])
    before: dict[str, list[onnx.NodeProto]] = {}
    after: dict[str, list[onnx.NodeProto]] = {}
    softmaxes = [node for node in graph.node if node.op_type == "Softmax" and "/self_attn/" in node.name]
    if len(softmaxes) != int(metadata["num_layers"]):
        raise ValueError("Flattened decode attention does not match the layer count")
    for layer, softmax in enumerate(softmaxes):
        scale = producers[softmax.input[0]]
        if scale.op_type == "Mul":
            products = [producers[value] for value in scale.input if value in producers
                        and producers[value].op_type in ("MatMul", "FusedMatMul")]
            if len(products) != 1:
                raise ValueError("Unexpected decode score scaling")
            score = products[0]
        elif scale.op_type in ("MatMul", "FusedMatMul"):
            score, scale = scale, None
        else:
            raise ValueError("Decode masks must be removed before flattening MQA")
        values = [node for node in graph.node if node.op_type in ("MatMul", "FusedMatMul")
                  and softmax.output[0] in node.input]
        if len(values) != 1:
            raise ValueError("Unexpected decode value product")
        value = values[0]
        tag = f"r2_mqa_{layer}_"
        # With static seq_len == 1, [1, 1, heads, width] -> Transpose([0,2,1,3]) -> Reshape([1, heads, width])
        # has identical contiguous layout to reshaping [1, 1, heads, width] directly.
        q_source = score.input[0]
        q_prod = producers.get(q_source)
        if q_prod is not None and q_prod.op_type == "Transpose":
            perm = next((list(a.ints) for a in q_prod.attribute if a.name == "perm"), [])
            if perm == [0, 2, 1, 3]:
                q_source = q_prod.input[0]
        preparation = [H.make_node("Reshape", [q_source, "r2_mqa_q_shape"], [tag + "q"])]
        if transposed_cache:
            preparation.append(H.make_node("Reshape", [score.input[1], "r2_mqa_k_shape"], [tag + "kt"]))
        else:
            transpose = producers.get(score.input[1])
            if transpose is None or transpose.op_type != "Transpose":
                raise ValueError("Expected standard K-cache transpose")
            preparation.extend([
                H.make_node("Reshape", [transpose.input[0], "r2_mqa_v_shape"], [tag + "k"]),
                H.make_node("Transpose", [tag + "k"], [tag + "kt"], perm=[0, 2, 1]),
            ])
        before[score.name] = preparation
        score.input[:] = [tag + "q", tag + "kt"]
        old_score = score.output[0]
        score.output[0] = tag + "scores"
        if scale is not None:
            scale.input[:] = [tag + "scores" if name == old_score else name for name in scale.input]
            old_scaled = scale.output[0]
            scale.output[0] = tag + "scaled"
            after[scale.name] = [
                H.make_node("Reshape", [tag + "scaled", "r2_mqa_score_shape"], [old_scaled]),
            ]
        else:
            after[score.name] = [
                H.make_node("Reshape", [tag + "scores", "r2_mqa_score_shape"], [old_score]),
            ]
        before[value.name] = [
            H.make_node("Reshape", [value.input[0], "r2_mqa_prob_shape"], [tag + "probabilities"]),
            H.make_node("Reshape", [value.input[1], "r2_mqa_v_shape"], [tag + "v"]),
        ]
        value.input[:] = [tag + "probabilities", tag + "v"]
        old_value = value.output[0]
        value.output[0] = tag + "context"
        # Collapse [1, heads, width] -> [1, heads, 1, width] -> [Transpose([0,2,1,3])] -> [1, 1, heads*width]
        # into one flat Reshape while keeping the two MQA Reshape barriers around Softmax intact.
        value_readers = consumers.get(old_value, [])
        flat_reshape = None
        if len(value_readers) == 1 and value_readers[0].op_type == "Reshape":
            flat_reshape = value_readers[0]
        elif len(value_readers) == 1 and value_readers[0].op_type == "Transpose":
            perm = next((list(a.ints) for a in value_readers[0].attribute if a.name == "perm"), [])
            t_readers = consumers.get(value_readers[0].output[0], [])
            if perm == [0, 2, 1, 3] and len(t_readers) == 1 and t_readers[0].op_type == "Reshape":
                flat_reshape = t_readers[0]
        if flat_reshape is not None:
            flat_reshape.input[:] = [tag + "context", "r2_mqa_flat_shape"]
            after[value.name] = []
        else:
            after[value.name] = [H.make_node("Reshape", [tag + "context", "r2_mqa_context_shape"], [old_value])]
    nodes = []
    for node in graph.node:
        nodes.extend(before.get(node.name, []))
        nodes.append(node)
        nodes.extend(after.get(node.name, []))
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.value_info[:]
    dce(model)
    H.set_model_props(model, {**metadata, "lm_attention_layout": "flattened_mqa"})
    onnx.checker.check_model(model)
    return model


def restrict_audio_head(source: onnx.ModelProto) -> onnx.ModelProto:
    """Explicitly experimental: preserve external IDs but change the sampling vocabulary."""
    model = onnx.ModelProto()
    model.CopyFrom(source)
    graph = model.graph
    metadata = {item.key: item.value for item in model.metadata_props}
    vocabulary = int(metadata["vocab_size"])
    allowed = np.unique(np.concatenate([
        np.arange(int(metadata["audio_token_first"]), int(metadata["audio_token_last"]) + 1, dtype=np.int32),
        np.array([int(metadata["speech_end_id"])], np.int32),
    ]))
    size = int(allowed.size)
    producers = {output: node for node in graph.node for output in node.output}
    head = producers.get("hj_logits3")
    if head is None or head.op_type not in ("MatMul", "MatMulNBits"):
        raise ValueError("Audio-only sampling requires the original weighted LM head")
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    for index, name in enumerate(list(head.input)[1:], 1):
        tensor = initializers.get(name)
        if tensor is None:
            continue
        array = NH.to_array(tensor)
        if head.op_type == "MatMul" and index == 1 and array.shape[-1] == vocabulary:
            sliced = array[:, allowed]
        elif head.op_type == "MatMulNBits" and array.ndim and array.shape[0] == vocabulary:
            sliced = array[allowed]
        else:
            continue
        replacement = f"r2_audio_head_weight_{index}"
        graph.initializer.append(const(replacement, np.ascontiguousarray(sliced)))
        head.input[index] = replacement
    if head.op_type == "MatMulNBits":
        next(attribute for attribute in head.attribute if attribute.name == "N").i = size
    graph.initializer.append(const("r2_audio_token_ids", allowed))
    penalties = [node for node in graph.node if node.op_type == "Where" and node.input[0] == "seen"]
    if len(penalties) != 1:
        raise ValueError("Expected one full-vocabulary repetition mask")
    penalties[0].input[0] = "r2_audio_seen"

    def narrow(scope: onnx.GraphProto, inherited: dict[str, onnx.TensorProto]) -> None:
        values = {**inherited, **{tensor.name: tensor for tensor in scope.initializer}}
        for node in scope.node:
            if node.op_type in ("Compress", "ScatterElements") and node.input[0] in values:
                tensor = values[node.input[0]]
                array = NH.to_array(tensor)
                if array.shape == (vocabulary,):
                    if node.op_type == "Compress" and np.array_equal(array, np.arange(vocabulary)):
                        name, replacement = "r2_local_vocab_ids", np.arange(size, dtype=array.dtype)
                    elif node.op_type == "ScatterElements" and array.dtype == np.bool_ and not array.any():
                        name, replacement = "r2_local_false", np.zeros(size, np.bool_)
                    else:
                        raise ValueError("Unrecognized sampler vocabulary constant")
                    scope.initializer.append(const(name, replacement))
                    node.input[0] = name
            if "hj_thr" in node.output and node.op_type == "Mul":
                index = next((index for index, name in enumerate(node.input)
                              if name in values and NH.to_array(values[name]).size == 1), None)
                if index is None:
                    raise ValueError("Nucleus threshold is not a constant vocabulary bound")
                scope.initializer.append(const("r2_audio_half_inverse", 0.5 / size, np.float32))
                node.input[index] = "r2_audio_half_inverse"
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    narrow(attribute.g, values)
        del scope.value_info[:]

    narrow(graph, {})
    choice = next((node for node in graph.node if node.op_type == "If" and node.name == "hj_choose_token"), None)
    if choice is None:
        raise ValueError("Audio-only head requires the conditional Hojo sampler")
    local_token = choice.output[0]
    nodes = [H.make_node("Gather", ["seen", "r2_audio_token_ids"], ["r2_audio_seen"])]
    for node in graph.node:
        if node.name != choice.name:
            remap_node_inputs(node, {local_token: "r2_original_audio_token"})
        nodes.append(node)
        if node.name == choice.name:
            nodes.append(H.make_node("Gather", ["r2_audio_token_ids", local_token], ["r2_original_audio_token"]))
    del graph.node[:]
    graph.node.extend(nodes)
    H.set_model_props(model, {**metadata, "audio_only_head": "1", "sampling_vocab_size": str(size)})
    dce(model)
    remove_redundant_casts(model)
    onnx.checker.check_model(model)
    return model


def _session(model: onnx.ModelProto) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def fold_coarse_projection(model: onnx.ModelProto) -> int:
    graph = model.graph
    bits_info = _bits_info(model)
    producers = {value: node for node in graph.node for value in node.output}
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    candidates = [node for node in graph.node if node.op_type == "MatMul" and node.input[0] == "coarse_embeddings"]
    if len(candidates) != 1:
        raise ValueError("Expected one independent coarse-embedding projection")
    projection = candidates[0]
    additions = [node for node in graph.node if node.op_type == "Add" and projection.output[0] in node.input]
    if len(additions) != 1:
        raise ValueError("Coarse projection must have one bias addition")
    bias = additions[0]
    gather = producers.get("coarse_embeddings")
    if gather is None or gather.op_type != "Gather" or gather.input[0] not in initializers:
        raise ValueError("Coarse embedding lookup is not a constant Gather")
    embedding = NH.to_array(initializers[gather.input[0]])
    width = embedding.shape[1]
    evaluator = extract_model(
        model, [H.make_tensor_value_info("coarse_embeddings", TP.FLOAT, [1, "frames", width])],
        [H.make_tensor_value_info(bias.output[0], TP.FLOAT, [1, "frames", width])], "coarse_projection",
    )
    projected = _session(evaluator).run(None, {"coarse_embeddings": embedding[None]})[0][0]
    graph.initializer.append(const("r2_projected_audio_embedding", np.ascontiguousarray(projected)))
    gather.input[0] = "r2_projected_audio_embedding"
    gather.output[0] = bias.output[0]
    drop = {projection.name, bias.name}
    keep = [node for node in graph.node if node.name not in drop]
    del graph.node[:]
    graph.node.extend(keep)
    del graph.value_info[:]
    graph.value_info.append(bits_info)
    dce(model)
    return int(embedding.shape[0])


def fold_speaker_projection(model: onnx.ModelProto, reference: onnx.ModelProto | None) -> str:
    graph = model.graph
    candidates = [node for node in graph.node if node.op_type == "Gemm" and "fine/" in node.name
                  and "speaker_proj" in node.name]
    if len(candidates) != 1:
        raise ValueError("Expected one fine-local speaker projection")
    projection = candidates[0]
    if next((attribute.i for attribute in projection.attribute if attribute.name == "transB"), 0) != 1:
        raise ValueError("Speaker precomputation requires the original transposed Gemm weights")
    weights = {tensor.name: tensor for tensor in graph.initializer}
    width = int(weights[projection.input[1]].dims[1])
    output_width = int(weights[projection.input[1]].dims[0])
    if "voice" in {value.name for value in graph.input}:
        gather = next((node for node in graph.node if node.output[0] == "speaker_embedding"), None)
        if gather is None or gather.op_type != "Gather":
            raise ValueError("Preset speaker projection requires a constant speaker bank")
        bank = NH.to_array(weights[gather.input[0]])
        evaluator = extract_model(
            model, [H.make_tensor_value_info(projection.input[0], TP.FLOAT, [1, width])],
            [H.make_tensor_value_info(projection.output[0], TP.FLOAT, [1, output_width])], "speaker_projection",
        )
        runtime = _session(evaluator)
        rows = np.stack([runtime.run(None, {projection.input[0]: row})[0] for row in bank])
        graph.initializer.append(const("r2_preset_speaker_projection", rows))
        projection.op_type = "Gather"
        projection.input[:] = ["r2_preset_speaker_projection", "voice"]
        del projection.attribute[:]
        dce(model)
        return "preset_table"
    if reference is None:
        raise ValueError("Cloning speaker precomputation requires the reference-preprocess graph")
    new_weight, new_bias = "r2_reference_speaker_weight", "r2_reference_speaker_bias"
    for old, new in ((projection.input[1], new_weight), (projection.input[2], new_bias)):
        tensor = onnx.TensorProto()
        tensor.CopyFrom(weights[old])
        tensor.name = new
        reference.graph.initializer.append(tensor)
    reference.graph.node.append(H.make_node(
        "Gemm", ["speaker_embedding", new_weight, new_bias], ["speaker_projection"],
        name="r2_reference_speaker_projection",
        **{attribute.name: H.get_attribute_value(attribute) for attribute in projection.attribute},
    ))
    reference.graph.output.append(H.make_tensor_value_info("speaker_projection", TP.FLOAT, [1, output_width]))
    graph.input.append(H.make_tensor_value_info("speaker_projection", TP.FLOAT, [1, output_width]))
    projection.op_type = "Identity"
    projection.input[:] = ["speaker_projection"]
    del projection.attribute[:]
    dce(model)
    return "reference_output"


def make_fine_stream(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(source)
    graph = model.graph
    producers = {value: node for node in graph.node for value in node.output}
    softmaxes = [node for node in graph.node if node.op_type == "Softmax"]
    if len(softmaxes) != 4:
        raise ValueError("Streaming fine-local requires the inspected four-layer architecture")
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    first_mask = producers[producers[softmaxes[0].input[0]].input[1]]
    if first_mask.op_type != "Where":
        raise ValueError("Fine-local scores must use a causal-window Where mask")
    neg, zero = first_mask.input[1:]
    graph.initializer.extend([
        const("r2_axis0", [0], np.int64), const("r2_axis1", [1], np.int64),
        const("r2_axis2", [2], np.int64), const("r2_axis3", [3], np.int64),
        const("r2_zero", 0, np.int64), const("r2_one", 1, np.int64),
        const("r2_history", 15, np.int64), const("r2_last15", [-15], np.int64),
        const("r2_max", [np.iinfo(np.int64).max], np.int64),
    ])
    pre = [
        H.make_node("Shape", ["fine_past.0.key"], ["r2_past1"], start=3, end=4),
        H.make_node("Squeeze", ["r2_past1", "r2_axis0"], ["r2_past"]),
        H.make_node("Shape", ["audio_tokens"], ["r2_frames1"], start=1, end=2),
        H.make_node("Squeeze", ["r2_frames1", "r2_axis0"], ["r2_frames"]),
        H.make_node("Add", ["r2_past", "r2_frames"], ["r2_total"]),
        H.make_node("Range", ["r2_zero", "r2_frames", "r2_one"], ["r2_queries0"]),
        H.make_node("Add", ["r2_queries0", "r2_past"], ["r2_queries"]),
        H.make_node("Range", ["r2_zero", "r2_total", "r2_one"], ["r2_keys"]),
        H.make_node("Unsqueeze", ["r2_queries", "r2_axis1"], ["r2_query_rows"]),
        H.make_node("Unsqueeze", ["r2_keys", "r2_axis0"], ["r2_key_columns"]),
        H.make_node("Sub", ["r2_query_rows", "r2_history"], ["r2_lower"]),
        H.make_node("Greater", ["r2_key_columns", "r2_query_rows"], ["r2_future"]),
        H.make_node("Less", ["r2_key_columns", "r2_lower"], ["r2_old"]),
        H.make_node("Or", ["r2_future", "r2_old"], ["r2_blocked"]),
        H.make_node("Where", ["r2_blocked", neg, zero], ["r2_causal_mask"]),
    ]
    after: dict[str, list[onnx.NodeProto]] = {}
    inputs = list(graph.input)
    outputs = [_bits_info(source)]
    for layer, softmax in enumerate(softmaxes):
        mask_add = producers[softmax.input[0]]
        if mask_add.op_type != "Add":
            raise ValueError(f"Unexpected fine-local mask at {softmax.name}")
        scores = producers[mask_add.input[0]]
        if scores.op_type != "MatMul":
            raise ValueError("Fine-local query scale must be folded before streaming")
        values = [node for node in graph.node if node.op_type == "MatMul" and softmax.output[0] in node.input]
        if len(values) != 1:
            raise ValueError("Fine-local attention must have one value product")
        value_product = values[0]
        key_source, value_source = scores.input[1], value_product.input[1]
        key_transpose = producers[key_source]
        reshape = producers[key_transpose.input[0]]
        shape = NH.to_array(initializers[reshape.input[1]])
        if list(shape) != [1, -1, 8, 64]:
            raise ValueError(f"Unsupported fine-local attention shape: {shape}")
        mask_add.input[1] = "r2_causal_mask"
        for kind, original, axis, dims, product in (
            ("key", key_source, 3, [1, 8, 64, "fine_past_frames"], scores),
            ("value", value_source, 2, [1, 8, "fine_past_frames", 64], value_product),
        ):
            past, present = f"fine_past.{layer}.{kind}", f"fine_present.{layer}.{kind}"
            combined = f"r2_fine_{layer}_{kind}_all"
            inputs.append(H.make_tensor_value_info(past, TP.FLOAT, dims))
            outputs.append(H.make_tensor_value_info(
                present, TP.FLOAT, ["fine_present_frames" if dim == "fine_past_frames" else dim for dim in dims],
            ))
            after.setdefault(original, []).extend([
                H.make_node("Concat", [past, original], [combined], axis=axis),
                H.make_node("Slice", [combined, "r2_last15", "r2_max", f"r2_axis{axis}"], [present]),
            ])
            product.input[1] = combined
    nodes = list(pre)
    for node in graph.node:
        nodes.append(node)
        for output in node.output:
            nodes.extend(after.get(output, []))
    del graph.node[:]
    graph.node.extend(nodes)
    return extract_model(model, inputs, outputs, "hojo_fine_stream")


def make_decoder_stream(source: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(source)
    graph = model.graph
    convolutions = [node for node in graph.node if node.op_type == "Conv" and "convnext." in node.name]
    if len(convolutions) != 8:
        raise ValueError("Streaming decoder requires the inspected eight ConvNeXt blocks")
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    channels = int(initializers[convolutions[0].input[1]].dims[0])
    bits_info = _bits_info(source)
    time_axis = len(bits_info.type.tensor_type.shape.dim) - 2
    front_output = convolutions[0].input[0]
    front = extract_model(
        source, [bits_info],
        [H.make_tensor_value_info(front_output, TP.FLOAT, [1, channels, "frames"])], "decoder_front",
    )
    front_names = {node.name for node in front.graph.node}
    for node in front.graph.node:
        node.output[:] = ["r2_dec_front_ready" if value == front_output else value for value in node.output]
    then_branch = H.make_graph(
        list(front.graph.node), "decoder_front_nonempty", [],
        [H.make_tensor_value_info("r2_dec_front_ready", TP.FLOAT, [1, channels, "frames"])],
    )
    else_branch = H.make_graph(
        [H.make_node("Identity", ["r2_dec_empty_front"], ["r2_dec_front_empty"])],
        "decoder_front_empty", [],
        [H.make_tensor_value_info("r2_dec_front_empty", TP.FLOAT, [1, channels, 0])],
    )
    graph.initializer.extend([
        const("r2_dec_axis2", [2], np.int64), const("r2_dec_axis0", [0], np.int64),
        const("r2_dec_one", 1, np.int64), const("r2_dec_zero", 0, np.int64),
        const("r2_dec_prefix_pads", [0, 0, 0, 0, 0], np.int64),
        const("r2_dec_last2", [-2], np.int64), const("r2_dec_max", [np.iinfo(np.int64).max], np.int64),
        const("r2_dec_center_start", [1], np.int64), const("r2_dec_center_end", [-1], np.int64),
        const("r2_dec_empty_front", np.empty((1, channels, 0), np.float32)),
    ])
    before: dict[str, list[onnx.NodeProto]] = {}
    inputs = [
        bits_info,
        H.make_tensor_value_info("is_first", TP.BOOL, []),
        H.make_tensor_value_info("is_final", TP.BOOL, []),
    ]
    outputs = [H.make_tensor_value_info(name, TP.FLOAT, [961, "spectral_frames"]) for name in ("vc_re", "vc_im")]
    prefix = [
        H.make_node("Shape", ["bits"], ["r2_dec_frames1"], start=time_axis, end=time_axis + 1),
        H.make_node("Squeeze", ["r2_dec_frames1", "r2_dec_axis0"], ["r2_dec_frames"]),
        H.make_node("Greater", ["r2_dec_frames", "r2_dec_zero"], ["r2_dec_has_frames"]),
        H.make_node("If", ["r2_dec_has_frames"], [front_output],
                    then_branch=then_branch, else_branch=else_branch),
        H.make_node("Where", ["is_first", "r2_dec_one", "r2_dec_zero"], ["r2_dec_start0"]),
        H.make_node("Unsqueeze", ["r2_dec_start0", "r2_dec_axis0"], ["r2_dec_start"]),
        H.make_node("Where", ["is_final", "r2_dec_one", "r2_dec_zero"], ["r2_dec_right0"]),
        H.make_node("Unsqueeze", ["r2_dec_right0", "r2_dec_axis0"], ["r2_dec_right"]),
        H.make_node("Concat", ["r2_dec_prefix_pads", "r2_dec_right"], ["r2_dec_pads"], axis=0),
    ]
    for layer, convolution in enumerate(convolutions):
        x = convolution.input[0]
        consumers = [node for node in graph.node if x in node.input and node.name != convolution.name]
        residuals = [node for node in consumers if node.op_type == "Add"]
        if len(consumers) != 1 or len(residuals) != 1:
            raise ValueError(f"Unexpected decoder residual consumers at {convolution.name}")
        pads = next(attribute for attribute in convolution.attribute if attribute.name == "pads")
        if list(pads.ints) != [1, 1]:
            raise ValueError("Streaming decoder requires symmetric three-tap convolutions")
        pads.ints[:] = [0, 0]
        tag = f"r2_conv_{layer}_"
        past, present = f"conv_past.{layer}", f"conv_present.{layer}"
        inputs.append(H.make_tensor_value_info(past, TP.FLOAT, [1, channels, 2]))
        outputs.append(H.make_tensor_value_info(present, TP.FLOAT, [1, channels, 2]))
        before[convolution.name] = [
            H.make_node("Concat", [past, x], [tag + "all"], axis=2),
            H.make_node("Slice", [tag + "all", "r2_dec_last2", "r2_dec_max", "r2_dec_axis2"], [present]),
            H.make_node("Slice", [tag + "all", "r2_dec_start", "r2_dec_max", "r2_dec_axis2"], [tag + "left"]),
            H.make_node("Pad", [tag + "left", "r2_dec_pads"], [tag + "ready"]),
            H.make_node("Slice", [tag + "ready", "r2_dec_center_start", "r2_dec_center_end", "r2_dec_axis2"],
                        [tag + "center"]),
        ]
        convolution.input[0] = tag + "ready"
        residuals[0].input[:] = [tag + "center" if name == x else name for name in residuals[0].input]
    nodes = list(prefix)
    for node in graph.node:
        if node.name in front_names:
            continue
        nodes.extend(before.get(node.name, []))
        nodes.append(node)
    del graph.node[:]
    graph.node.extend(nodes)
    return extract_model(model, inputs, outputs, "hojo_decoder_stream")


RNG_PREFIX = "r3_rng_"
_SPLITMIX = (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB)


def splitmix_uniform(seed: int | np.ndarray, step: int | np.ndarray) -> np.ndarray:
    """NumPy reference of the in-graph counter RNG: top 24 bits of SplitMix64(seed + (step + 1) * golden)."""
    golden, first, second = (np.uint64(value) for value in _SPLITMIX)
    with np.errstate(over="ignore"):
        x = np.asarray(seed, np.uint64) + (np.asarray(step).astype(np.uint64) + np.uint64(1)) * golden
        x = (x ^ (x >> np.uint64(30))) * first
        x = (x ^ (x >> np.uint64(27))) * second
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(40)).astype(np.float32) * np.float32(2.0 ** -24)


def rng_nodes(output: str = "uniform") -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """uint64 SplitMix64 over graph inputs rng_seed/rng_step (BitShift is defined for unsigned types only)."""
    p, n = RNG_PREFIX, H.make_node
    golden, first, second = _SPLITMIX
    initializers = [
        const(p + name, [value], np.uint64)
        for name, value in (("one", 1), ("golden", golden), ("m1", first), ("m2", second),
                            ("s30", 30), ("s27", 27), ("s31", 31), ("s40", 40))
    ] + [const(p + "u24", [2.0 ** -24], np.float32)]
    nodes = [
        n("Cast", ["rng_step"], [p + "step"], to=TP.UINT64),
        n("Add", [p + "step", p + "one"], [p + "step1"]),
        n("Mul", [p + "step1", p + "golden"], [p + "offset"]),
        n("Add", ["rng_seed", p + "offset"], [p + "x0"]),
        n("BitShift", [p + "x0", p + "s30"], [p + "h0"], direction="RIGHT"),
        n("BitwiseXor", [p + "x0", p + "h0"], [p + "y0"]),
        n("Mul", [p + "y0", p + "m1"], [p + "x1"]),
        n("BitShift", [p + "x1", p + "s27"], [p + "h1"], direction="RIGHT"),
        n("BitwiseXor", [p + "x1", p + "h1"], [p + "y1"]),
        n("Mul", [p + "y1", p + "m2"], [p + "x2"]),
        n("BitShift", [p + "x2", p + "s31"], [p + "h2"], direction="RIGHT"),
        n("BitwiseXor", [p + "x2", p + "h2"], [p + "x3"]),
        n("BitShift", [p + "x3", p + "s40"], [p + "top"], direction="RIGHT"),
        n("Cast", [p + "top"], [p + "top_f"], to=TP.FLOAT),
        n("Mul", [p + "top_f", p + "u24"], [output]),
    ]
    for index, node in enumerate(nodes):
        node.name = f"{p}{index}_{node.op_type}"
    return nodes, initializers


def _stochastic_branch(model: onnx.ModelProto) -> onnx.GraphProto:
    choices = [node for node in model.graph.node if node.op_type == "If" and node.name == "hj_choose_token"]
    if len(choices) != 1:
        raise ValueError("Counter RNG requires the Hojo sampler If node")
    return next(attribute.g for attribute in choices[0].attribute if attribute.name == "else_branch")


def counter_rng_sampler(source: onnx.ModelProto) -> onnx.ModelProto:
    """Replace the external `uniform` draw with in-graph SplitMix64 inside the stochastic sampler branch only."""
    model = onnx.ModelProto()
    model.CopyFrom(source)
    graph = model.graph
    uniform = [value for value in graph.input if value.name == "uniform"]
    readers = [node.name for node in graph.node if "uniform" in node.input]
    branch = _stochastic_branch(model)
    if len(uniform) != 1 or readers or sum("uniform" in node.input for node in branch.node) != 1:
        raise ValueError("Counter RNG expects `uniform` to be read only by the stochastic sampler branch")
    nodes, initializers = rng_nodes()
    kept = list(branch.node)
    del branch.node[:]
    branch.node.extend([*nodes, *kept])
    branch.initializer.extend(initializers)
    inputs = [value for value in graph.input if value.name != "uniform"]
    inputs.extend([H.make_tensor_value_info("rng_seed", TP.UINT64, [1]),
                   H.make_tensor_value_info("rng_step", TP.INT64, [1])])
    del graph.input[:]
    graph.input.extend(inputs)
    metadata = {item.key: item.value for item in model.metadata_props}
    H.set_model_props(model, {**metadata, "sampler_rng": "splitmix64_counter_v1"})
    onnx.checker.check_model(model)
    return model


def uniform_sampler_view(source: onnx.ModelProto) -> onnx.ModelProto:
    """Diagnostic inverse of counter_rng_sampler: drop the RNG nodes and expose `uniform` as an input again."""
    model = onnx.ModelProto()
    model.CopyFrom(source)
    branch = _stochastic_branch(model)
    kept = [node for node in branch.node if not node.name.startswith(RNG_PREFIX)]
    if len(kept) == len(branch.node):
        return model
    del branch.node[:]
    branch.node.extend(kept)
    initializers = [tensor for tensor in branch.initializer if not tensor.name.startswith(RNG_PREFIX)]
    del branch.initializer[:]
    branch.initializer.extend(initializers)
    inputs = [value for value in model.graph.input if value.name not in ("rng_seed", "rng_step")]
    inputs.append(H.make_tensor_value_info("uniform", TP.FLOAT, [1]))
    del model.graph.input[:]
    model.graph.input.extend(inputs)
    return model


ISTFT_HOP, ISTFT_CROP, ISTFT_STATE = 480, 720, 1440


def istft_envelopes() -> tuple[np.ndarray, np.ndarray]:
    """Inverse Hann^2 envelopes for the first frames and the final flush (same semantics as the R2 renderer)."""
    n_fft, hop = 1920, ISTFT_HOP
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
    square = (window * window).astype(np.float32)
    prefix = np.zeros((4, hop), np.float32)
    for count in range(4):
        envelope = np.zeros(hop, np.float32)
        for previous in range(count + 1):
            envelope += square[previous * hop:(previous + 1) * hop]
        prefix[count] = np.divide(1.0, envelope, out=np.zeros_like(envelope), where=envelope > 0)
    tail = np.zeros((3, 3 * hop), np.float32)
    for count in range(1, 4):
        for block in range(3):
            envelope = np.zeros(hop, np.float32)
            for previous in range(block + 1, min(count + block, 3) + 1):
                envelope += square[previous * hop:(previous + 1) * hop]
            tail[count - 1, block * hop:(block + 1) * hop] = np.divide(
                1.0, envelope, out=np.zeros_like(envelope), where=envelope > 0)
    return prefix, tail


def make_decoder_stream_pcm(source: onnx.ModelProto) -> onnx.ModelProto:
    """Streaming decoder whose spectra stay in the graph: stateful iSTFT, overlap-add and crop emit PCM directly."""
    return append_stream_istft(make_decoder_stream(source))


def append_stream_istft(decoder: onnx.ModelProto) -> onnx.ModelProto:
    """Replace a v1 streaming decoder's vc_re/vc_im outputs with the stateful in-graph iSTFT."""
    from Export_Hojo_TTS import istft_basis

    model = onnx.ModelProto()
    model.CopyFrom(decoder)
    graph = model.graph
    if [value.name for value in graph.output[:2]] != ["vc_re", "vc_im"]:
        raise ValueError("Streaming decoder must expose vc_re/vc_im before the PCM stage")
    cos_b, sin_b, _ = istft_basis()
    prefix, tail = istft_envelopes()
    hop, half = ISTFT_HOP, cos_b.shape[1] // 2
    p = "r3_istft_"
    graph.initializer.extend([
        # Same bytes as Hojo_Vocoder.onnx vc_cos_half / vc_sin_half so shared-initializer bundling deduplicates them.
        NH.from_array(np.ascontiguousarray(cos_b.T[:half + 1]), p + "cos_half"),
        NH.from_array(np.ascontiguousarray(sin_b.T[:half + 1]), p + "sin_half"),
        NH.from_array(prefix, p + "prefix_inverse"), NH.from_array(tail, p + "tail_inverse"),
        const(p + "zero_state", np.zeros((1, ISTFT_STATE), np.float32)),
        const(p + "mirror_start", [half - 1], np.int64), const(p + "mirror_end", [0], np.int64),
        const(p + "reverse", [-1], np.int64), const(p + "axis0", [0], np.int64), const(p + "axis1", [1], np.int64),
        const(p + "frame_shape", [-1, 4, hop], np.int64), const(p + "block_shape", [-1, hop], np.int64),
        const(p + "past_shape", [3, hop], np.int64), const(p + "state_shape", [1, ISTFT_STATE], np.int64),
        const(p + "flat", [-1], np.int64), const(p + "three", [3], np.int64), const(p + "zero2", [0, 0], np.int64),
        const(p + "zero1", [0], np.int64), const(p + "zero", 0, np.int64), const(p + "one", 1, np.int64),
        const(p + "three_s", 3, np.int64), const(p + "hop", hop, np.int64), const(p + "crop", [ISTFT_CROP], np.int64),
        const(p + "max", [np.iinfo(np.int64).max], np.int64), const(p + "zero_f", 0.0, np.float32),
        *[const(p + f"pads{j}", [j, 0, 0, 3 - j, 0, 0], np.int64) for j in range(4)],
    ])
    n = H.make_node
    graph.node.extend([
        # is_first ignores the carried state, so reset() only flips is_first.
        n("Where", ["is_first", p + "zero_state", "ola_past"], [p + "past"]),
        n("Squeeze", ["frames_past", p + "axis0"], [p + "past_frames"]),
        n("Where", ["is_first", p + "zero", p + "past_frames"], [p + "fp"]),
        # Mirrored half-basis inverse rFFT with the periodic Hann window folded in: [1920, S].
        n("MatMul", [p + "cos_half", "vc_re"], [p + "cos"]),
        n("MatMul", [p + "sin_half", "vc_im"], [p + "sin"]),
        n("Add", [p + "cos", p + "sin"], [p + "top"]),
        n("Sub", [p + "cos", p + "sin"], [p + "mirror"]),
        n("Slice", [p + "mirror", p + "mirror_start", p + "mirror_end", p + "axis0", p + "reverse"], [p + "bottom"]),
        n("Concat", [p + "top", p + "bottom"], [p + "frames_tn"], axis=0),
        n("Transpose", [p + "frames_tn"], [p + "frames_t"], perm=[1, 0]),
        # Overlap-add: four 480-sample phases of each frame land on blocks k..k+3, plus the carried 3 blocks.
        n("Reshape", [p + "frames_t", p + "frame_shape"], [p + "phases"]),
        n("Split", [p + "phases"], [p + f"phase{j}" for j in range(4)], axis=1, num_outputs=4),
        *[n("Pad", [p + f"phase{j}", p + f"pads{j}"], [p + f"shifted{j}"]) for j in range(4)],
        n("Reshape", [p + "past", p + "past_shape"], [p + "past3"]),
        n("Shape", ["vc_re"], [p + "S_v"], start=1, end=2),
        n("Concat", [p + "zero2", p + "S_v", p + "zero1"], [p + "past_pads"], axis=0),
        n("Pad", [p + "past3", p + "past_pads"], [p + "past_full"]),
        n("Unsqueeze", [p + "past_full", p + "axis1"], [p + "past_full3"]),
        n("Sum", [*[p + f"shifted{j}" for j in range(4)], p + "past_full3"], [p + "acc3"]),
        n("Reshape", [p + "acc3", p + "block_shape"], [p + "acc"]),
        # Completed blocks use prefix_inverse[min(frames_past + k, 3)].
        n("Squeeze", [p + "S_v", p + "axis0"], [p + "S"]),
        n("Range", [p + "zero", p + "S", p + "one"], [p + "k"]),
        n("Add", [p + "k", p + "fp"], [p + "frame_index"]),
        n("Min", [p + "frame_index", p + "three_s"], [p + "envelope_index"]),
        n("Gather", [p + "prefix_inverse", p + "envelope_index"], [p + "gains"], axis=0),
        n("Slice", [p + "acc", p + "zero1", p + "S_v", p + "axis0"], [p + "done"]),
        n("Mul", [p + "done", p + "gains"], [p + "done_scaled"]),
        n("Reshape", [p + "done_scaled", p + "flat"], [p + "main"]),
        n("Slice", [p + "acc", p + "S_v", p + "max", p + "axis0"], [p + "state_blocks"]),
        n("Reshape", [p + "state_blocks", p + "state_shape"], ["ola_present"]),
        n("Add", [p + "fp", p + "S"], [p + "total"]),
        n("Reshape", [p + "total", p + "flat"], ["frames_present"]),
        # Final flush: 720 tail samples with tail_inverse[min(total, 3) - 1].
        n("Min", [p + "total", p + "three_s"], [p + "total3"]),
        n("Sub", [p + "total3", p + "one"], [p + "tail_index"]),
        n("Gather", [p + "tail_inverse", p + "tail_index"], [p + "tail_gain"], axis=0),
        n("Reshape", [p + "state_blocks", p + "flat"], [p + "state_flat"]),
        n("Mul", [p + "state_flat", p + "tail_gain"], [p + "tail_scaled"]),
        n("Where", ["is_final", p + "crop", p + "zero1"], [p + "tail_length"]),
        n("Slice", [p + "tail_scaled", p + "zero1", p + "tail_length", p + "axis0"], [p + "tail"]),
        n("Concat", [p + "main", p + "tail"], [p + "stream"], axis=0),
        # Centered padding: global samples below 720 are never emitted.
        n("Mul", [p + "fp", p + "hop"], [p + "start"]),
        n("Sub", [p + "crop", p + "start"], [p + "skip0"]),
        n("Max", [p + "skip0", p + "zero1"], [p + "skip"]),
        n("Slice", [p + "stream", p + "skip", p + "max", p + "axis0"], ["pcm"]),
        # x * 0 sums to zero only when every value is finite (non-finite spectra fail the run).
        n("Mul", [p + "acc", p + "zero_f"], [p + "nan_probe"]),
        n("ReduceSum", [p + "nan_probe"], [p + "nan_sum"], keepdims=0),
        n("Equal", [p + "nan_sum", p + "zero_f"], ["pcm_finite"]),
    ])
    graph.input.extend([
        H.make_tensor_value_info("ola_past", TP.FLOAT, [1, ISTFT_STATE]),
        H.make_tensor_value_info("frames_past", TP.INT64, [1]),
    ])
    states = list(graph.output[2:])
    del graph.output[:]
    graph.output.extend([
        H.make_tensor_value_info("pcm", TP.FLOAT, ["pcm_samples"]),
        H.make_tensor_value_info("pcm_finite", TP.BOOL, []),
        H.make_tensor_value_info("ola_present", TP.FLOAT, [1, ISTFT_STATE]),
        H.make_tensor_value_info("frames_present", TP.INT64, [1]),
        *states,
    ])
    model.graph.name = "hojo_decoder_stream_pcm"
    dce(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def _replace_tensor_ref_recursive(graph: onnx.GraphProto, old_name: str, new_name: str, *, is_root: bool = True) -> None:
    """Replace `old_name` references across nodes and captured subgraphs while respecting local shadowing."""
    if not is_root:
        local_defs = {v.name for v in graph.input} | {t.name for t in graph.initializer}
        if old_name in local_defs or new_name in local_defs:
            return
    shadowed = False
    for node in graph.node:
        if not shadowed:
            for idx, inp in enumerate(node.input):
                if inp == old_name:
                    node.input[idx] = new_name
            for sg in iter_subgraphs(node):
                _replace_tensor_ref_recursive(sg, old_name, new_name, is_root=False)
        if not is_root and (old_name in node.output or new_name in node.output):
            shadowed = True
    if not shadowed:
        for out in graph.output:
            if out.name == old_name:
                out.name = new_name


def _purge_value_info_recursive(graph: onnx.GraphProto, names: set[str]) -> None:
    kept = [vi for vi in graph.value_info if vi.name not in names]
    del graph.value_info[:]
    graph.value_info.extend(kept)
    for node in graph.node:
        for sg in iter_subgraphs(node):
            local_defs = ({v.name for v in sg.input}
                          | {t.name for t in sg.initializer}
                          | {o for n in sg.node for o in n.output})
            _purge_value_info_recursive(sg, names - local_defs)


def _count_node_input_uses_recursive(graph: onnx.GraphProto, name: str, *, is_root: bool = True) -> int:
    if not is_root:
        if any(v.name == name for v in graph.input) or any(t.name == name for t in graph.initializer):
            return 0
    uses = 0
    shadowed = False
    for node in graph.node:
        if not shadowed:
            uses += sum(1 for inp in node.input if inp == name)
            for sg in iter_subgraphs(node):
                uses += _count_node_input_uses_recursive(sg, name, is_root=False)
        if name in node.output and not is_root:
            shadowed = True
    return uses


def unwrap_f16_neural_boundaries(
    lm: onnx.ModelProto,
    fine: onnx.ModelProto,
    decoder: onnx.ModelProto,
    *,
    require_lm_f16: bool = True,
) -> None:
    """Recursively unwrap redundant top-level FLOAT<->FLOAT16 neural boundary Casts and verify F16 hand-offs."""

    def unwrap_input(model: onnx.ModelProto, inp_name: str, label: str) -> None:
        vi = next((v for v in model.graph.input if v.name == inp_name), None)
        if vi is None:
            raise ValueError(f"{label}: missing expected input {inp_name}")
        elem = vi.type.tensor_type.elem_type
        if elem == TP.FLOAT16:
            return
        if elem != TP.FLOAT:
            raise ValueError(f"{label}: unexpected input {inp_name} dtype {elem}")
        cast_nodes = [
            (idx, n) for idx, n in enumerate(model.graph.node)
            if n.op_type == "Cast" and list(n.input) == [inp_name] and len(n.output) == 1
            and any(a.name == "to" and a.i == TP.FLOAT16 for a in n.attribute)
        ]
        graph_out_uses = sum(1 for out in model.graph.output if out.name == inp_name)
        if len(cast_nodes) != 1 or _count_node_input_uses_recursive(model.graph, inp_name) != 1 or graph_out_uses != 0:
            raise ValueError(f"{label}: input {inp_name} does not feed a unique F32->F16 boundary Cast")
        cast_idx, cast_node = cast_nodes[0]
        cast_out = cast_node.output[0]
        del model.graph.node[cast_idx]
        _replace_tensor_ref_recursive(model.graph, cast_out, inp_name)
        _purge_value_info_recursive(model.graph, {cast_out, inp_name})
        vi.type.tensor_type.elem_type = TP.FLOAT16

    def unwrap_output(model: onnx.ModelProto, out_name: str, label: str) -> None:
        vi = next((v for v in model.graph.output if v.name == out_name), None)
        if vi is None:
            raise ValueError(f"{label}: missing expected output {out_name}")
        elem = vi.type.tensor_type.elem_type
        if elem == TP.FLOAT16:
            return
        if elem != TP.FLOAT:
            raise ValueError(f"{label}: unexpected output {out_name} dtype {elem}")
        producers = [(idx, n) for idx, n in enumerate(model.graph.node) if out_name in n.output]
        cast_nodes = [
            (idx, n) for idx, n in producers
            if n.op_type == "Cast" and list(n.output) == [out_name] and len(n.input) == 1
            and any(a.name == "to" and a.i == TP.FLOAT for a in n.attribute)
        ]
        if len(producers) != 1 or len(cast_nodes) != 1 or _count_node_input_uses_recursive(model.graph, out_name) != 0:
            raise ValueError(f"{label}: output {out_name} is not produced by a unique F16->F32 boundary Cast")
        cast_idx, cast_node = cast_nodes[0]
        cast_in = cast_node.input[0]
        del model.graph.node[cast_idx]
        for node in model.graph.node:
            for idx, out in enumerate(node.output):
                if out == cast_in:
                    node.output[idx] = out_name
        _replace_tensor_ref_recursive(model.graph, cast_in, out_name)
        _purge_value_info_recursive(model.graph, {cast_in, out_name})
        vi.type.tensor_type.elem_type = TP.FLOAT16

    for v in fine.graph.input:
        if v.name == "hidden_states" or v.name.startswith("fine_past."):
            unwrap_input(fine, v.name, FINE_NAME)
    for v in fine.graph.output:
        if v.name == "bits" or v.name.startswith("fine_present."):
            unwrap_output(fine, v.name, FINE_NAME)

    for v in decoder.graph.input:
        if v.name == "bits" or v.name.startswith("conv_past."):
            unwrap_input(decoder, v.name, DECODER_NAME)
    for v in decoder.graph.output:
        if v.name.startswith("conv_present."):
            unwrap_output(decoder, v.name, DECODER_NAME)

    lm_hidden = next(v for v in lm.graph.output if v.name == "hidden")
    fine_hidden = next(v for v in fine.graph.input if v.name == "hidden_states")
    fine_bits = next(v for v in fine.graph.output if v.name == "bits")
    dec_bits = next(v for v in decoder.graph.input if v.name == "bits")
    allowed_lm_types = (TP.FLOAT16,) if require_lm_f16 else (TP.FLOAT16, TP.FLOAT)
    if (lm_hidden.type.tensor_type.elem_type not in allowed_lm_types
            or fine_hidden.type.tensor_type.elem_type != TP.FLOAT16
            or lm_hidden.type.tensor_type.shape.dim[-1].dim_value != fine_hidden.type.tensor_type.shape.dim[-1].dim_value):
        raise ValueError("LM hidden -> Fine hidden_states F16 boundary mismatch")
    if (fine_bits.type.tensor_type.elem_type != TP.FLOAT16
            or dec_bits.type.tensor_type.elem_type != TP.FLOAT16
            or fine_bits.type.tensor_type.shape.dim[-1].dim_value != dec_bits.type.tensor_type.shape.dim[-1].dim_value):
        raise ValueError("Fine bits -> Decoder bits F16 boundary mismatch")
    dec_in_types = {v.name: v.type.tensor_type.elem_type for v in decoder.graph.input}
    dec_out_types = {v.name: v.type.tensor_type.elem_type for v in decoder.graph.output}
    if (dec_in_types.get("ola_past") != TP.FLOAT16
            or dec_out_types.get("ola_present") != TP.FLOAT16
            or dec_out_types.get("pcm") not in (TP.FLOAT16, TP.FLOAT)):
        raise ValueError("Decoder must retain FLOAT16 OLA state and F16 or adapted F32 PCM")


def adapt_decoder_pcm_for_float_playback(decoder: onnx.ModelProto) -> None:
    """Append a single F16->F32 playback boundary Cast on `pcm` while keeping internal F16 quantization and OLA state."""
    output = next((value for value in decoder.graph.output if value.name == "pcm"), None)
    producers = [node for node in decoder.graph.node if "pcm" in node.output]
    internal_name = "pcm_f16_before_playback_cast"
    if output is not None and output.type.tensor_type.elem_type == TP.FLOAT:
        if (len(producers) != 1 or producers[0].op_type != "Cast"
                or list(producers[0].input) != [internal_name]
                or not any(attr.name == "to" and attr.i == TP.FLOAT for attr in producers[0].attribute)):
            raise ValueError("Adapted F32 PCM must be produced by the single playback boundary Cast")
        return
    if output is None or output.type.tensor_type.elem_type != TP.FLOAT16:
        raise ValueError("Decoder pcm must be an F16 graph output")
    names = {name for node in decoder.graph.node for name in (*node.input, *node.output)}
    names.update(value.name for value in (*decoder.graph.input, *decoder.graph.output, *decoder.graph.value_info))
    names.update(tensor.name for tensor in decoder.graph.initializer)
    if internal_name in names:
        raise ValueError(f"Decoder already uses {internal_name}")
    if len(producers) != 1:
        raise ValueError("Decoder pcm must have exactly one producer")
    _replace_tensor_ref_recursive(decoder.graph, "pcm", internal_name)
    for node in producers:
        for index, name in enumerate(node.output):
            if name == "pcm":
                node.output[index] = internal_name
    output.name = "pcm"
    output.type.tensor_type.elem_type = TP.FLOAT
    decoder.graph.node.append(
        H.make_node("Cast", [internal_name], ["pcm"], to=TP.FLOAT, name="pcm_float_playback_cast")
    )


def stabilize_lm_fp32_attention_and_activations(model: onnx.ModelProto, model_name: str = "Hojo_LM.onnx") -> int:
    """Wrap non-initializer attention MatMuls (QK^T, PV), Softmax, QuickGelu, and F16 Mul/Add (except *_rope_add) in FP32.

    Prevents ARMv8.2-A MlasHalfGemm 16-bit accumulation drift and runaway EOS failure while keeping all 41 weighted
    MatMuls, 31 norms, *_rope_add, KV caches, and hidden output in FLOAT16 with zero shared-weight size increase.
    """
    props = {entry.key: entry.value for entry in model.metadata_props}
    if props.get("lm_fp32_attention_activations") == "1" or any(n.name.startswith("z1_") for n in model.graph.node):
        return 0

    init_names = {t.name for t in model.graph.initializer}
    f16_tensors: set[str] = set()
    for vi in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if vi.type.tensor_type.elem_type == TP.FLOAT16:
            f16_tensors.add(vi.name)
    for t in model.graph.initializer:
        if t.data_type == TP.FLOAT16:
            f16_tensors.add(t.name)

    passthrough_ops = (
        "Gather", "Concat", "Transpose", "Slice", "Reshape", "Unsqueeze", "Squeeze",
        "Identity", "Split", "Expand", "Tile", "Pad", "Neg", KV_CUSTOM_OP,
    )
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if not node.output:
                continue
            is_f16 = False
            if node.op_type == "Cast" and any(a.name == "to" and a.i == TP.FLOAT16 for a in node.attribute):
                is_f16 = True
            elif node.op_type in NORM_OPS:
                is_f16 = True
            elif node.op_type in passthrough_ops and node.input and node.input[0] in f16_tensors:
                is_f16 = True
            elif node.op_type == "Where" and len(node.input) == 3 and node.input[1] in f16_tensors and node.input[2] in f16_tensors:
                is_f16 = True
            elif node.op_type in ("MatMul", "Mul", "Add", "Sub", "Div", "Softmax", "QuickGelu") and node.input and all(
                inp in f16_tensors for inp in node.input if inp
            ):
                is_f16 = True
            if is_f16:
                for out in node.output:
                    if out and out not in f16_tensors:
                        f16_tensors.add(out)
                        changed = True

    new_nodes: list[onnx.NodeProto] = []
    widen_cache: dict[str, str] = {}
    wrapped = 0
    for idx, node in enumerate(model.graph.node):
        wrap = False
        if node.op_type == "MatMul":
            has_weight = any(inp in init_names for inp in node.input)
            if not has_weight and all(inp in f16_tensors for inp in node.input):
                wrap = True
        elif node.op_type in ("Softmax", "QuickGelu"):
            if node.input and node.input[0] in f16_tensors:
                wrap = True
        elif node.op_type in ("Mul", "Add") and not (node.op_type == "Add" and node.name.endswith("_rope_add")):
            if node.input and all(inp in f16_tensors for inp in node.input if inp):
                wrap = True

        if not wrap:
            new_nodes.append(node)
            continue

        wrapped += 1
        for k, inp in enumerate(node.input):
            if inp in f16_tensors:
                if inp not in widen_cache:
                    cast_out = f"z1_fp32_in_{len(widen_cache)}"
                    new_nodes.append(H.make_node(
                        "Cast", [inp], [cast_out], to=TP.FLOAT, name=f"z1_widen_{len(widen_cache)}",
                    ))
                    widen_cache[inp] = cast_out
                node.input[k] = widen_cache[inp]
        orig_out = node.output[0]
        raw_fp32_out = f"z1_fp32_out_{idx}"
        node.output[0] = raw_fp32_out
        new_nodes.append(node)
        new_nodes.append(H.make_node(
            "Cast", [raw_fp32_out], [orig_out], to=TP.FLOAT16, name=f"z1_narrow_{idx}",
        ))

    if wrapped == 0:
        raise ValueError(f"{model_name}: stabilize_lm_fp32_attention_and_activations matched 0 nodes")
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.value_info[:]
    props["lm_fp32_attention_activations"] = "1"
    set_sorted_metadata(model, props)
    return wrapped


def streaming_metadata(
    fine: onnx.ModelProto,
    decoder: onnx.ModelProto,
    carrier: onnx.ModelProto | None,
    hop_length: int,
    n_fft: int,
) -> dict[str, str]:
    """Derive and verify streaming fine-local history length, decoder Conv delay, and iSTFT OLA geometry."""
    initializers = {tensor.name: tensor for tensor in fine.graph.initializer}
    history = set()
    for node in fine.graph.node:
        if node.op_type == "Slice" and any(name.startswith("fine_present.") for name in node.output):
            start = NH.to_array(initializers[node.input[1]])
            if start.size != 1 or int(start.item()) >= 0:
                raise ValueError("Fine state must retain a fixed trailing history")
            history.add(-int(start.item()))
    if len(history) != 1:
        raise ValueError("Fine states disagree on history length")

    conv_states = {
        value.name: value.type.tensor_type.shape.dim[-1].dim_value
        for value in decoder.graph.input if value.name.startswith("conv_past.")
    }
    if not conv_states:
        raise ValueError("Decoder has no streaming conv_past.* states")
    dec_init_shapes: dict[str, tuple[int, ...]] = {}
    if carrier is not None:
        dec_init_shapes.update({t.name: tuple(t.dims) for t in carrier.graph.initializer})
    dec_init_shapes.update({t.name: tuple(t.dims) for t in decoder.graph.initializer})
    convs = [node for node in decoder.graph.node if node.op_type == "Conv"]
    if len(convs) != len(conv_states):
        raise ValueError(f"Decoder Conv count ({len(convs)}) != conv_past state count ({len(conv_states)})")

    delay = 0
    for idx, conv in enumerate(convs):
        state_name = f"conv_past.{idx}"
        if state_name not in conv_states:
            raise ValueError(f"Missing expected ordered decoder state {state_name}")
        state_width = conv_states[state_name]
        attrs = {a.name: (list(a.ints) if a.ints else a.i) for a in conv.attribute}
        strides = attrs.get("strides", [1])
        pads = attrs.get("pads", [0, 0])
        dilations = attrs.get("dilations", [1])
        w_shape = dec_init_shapes.get(conv.input[1])
        kernel_shape = attrs.get("kernel_shape", [w_shape[-1]] if w_shape else [])
        if (strides != [1] or pads != [0, 0] or len(dilations) != 1 or len(kernel_shape) != 1
                or (w_shape is not None and w_shape[-1] != kernel_shape[0])):
            raise ValueError(f"Invalid streaming Conv attributes on {conv.name}: {attrs} weight={w_shape}")
        eff_hist = int(dilations[0]) * (int(kernel_shape[0]) - 1)
        if eff_hist <= 0 or eff_hist % 2 != 0 or eff_hist != state_width:
            raise ValueError(
                f"Conv {conv.name} history mismatch: dilation*(k-1)={eff_hist} != {state_name} width={state_width}"
            )
        delay += eff_hist // 2

    if n_fft <= hop_length or (n_fft - hop_length) % 2 != 0:
        raise ValueError(f"Invalid iSTFT geometry: n_fft={n_fft}, hop_length={hop_length}")
    expected_ola_width = n_fft - hop_length
    expected_crop = expected_ola_width // 2
    ola_in = next((v for v in decoder.graph.input if v.name == "ola_past"), None)
    ola_out = next((v for v in decoder.graph.output if v.name == "ola_present"), None)
    if ola_in is None or ola_out is None:
        raise ValueError("Decoder missing ola_past/ola_present tensors")
    ola_in_dims = [d.dim_value for d in ola_in.type.tensor_type.shape.dim]
    ola_out_dims = [d.dim_value for d in ola_out.type.tensor_type.shape.dim]
    if ola_in_dims != [1, expected_ola_width] or ola_out_dims != [1, expected_ola_width]:
        raise ValueError(f"OLA width mismatch: in={ola_in_dims} out={ola_out_dims} expected=[1, {expected_ola_width}]")
    crop_init = next((t for t in decoder.graph.initializer if t.name == "r3_istft_crop" and not t.external_data), None)
    if crop_init is None:
        raise ValueError("Decoder missing in-graph r3_istft_crop initializer")
    crop_arr = NH.to_array(crop_init).reshape(-1)
    if crop_arr.size == 0 or any(int(v) != expected_crop for v in crop_arr):
        raise ValueError(f"Decoder r3_istft_crop={crop_arr.tolist()} != (n_fft-hop_length)//2={expected_crop}")

    return {"fine_history_frames": str(history.pop()), "stream_decoder_delay_frames": str(delay)}


def convert_lm_to_inplaced_kv(source: onnx.ModelProto) -> onnx.ModelProto:
    """Convert Concat-KV Hojo_LM / Hojo_LM_Decode to static BNSH `com.custom.llm::KVCacheUpdate` in-place KV."""
    model = onnx.ModelProto()
    model.CopyFrom(source)
    graph = model.graph
    props = {item.key: item.value for item in model.metadata_props}
    if props.get("lm_kv_update") == "inplaced_kv":
        return model
    if any(node.name.startswith("z1_") for node in graph.node):
        raise ValueError("Run convert_lm_to_inplaced_kv before stabilize_lm_fp32_attention_and_activations")

    layers = int(props["num_layers"])
    kv_heads = int(props["kv_heads"])
    max_positions = int(props["max_positions"])
    head_dim = int(props["head_dim"])
    if kv_heads != 1:
        raise ValueError(f"inplaced_kv currently requires kv_heads == 1, got {kv_heads}")

    init_map = {tensor.name: tensor for tensor in graph.initializer}

    def ensure_init(name: str, value, dtype) -> str:
        if name not in init_map:
            tensor = const(name, value, dtype)
            graph.initializer.append(tensor)
            init_map[name] = tensor
        return name

    axis0_name = ensure_init("hj_axis0", [0], np.int64)
    axis2_name = "a8_axis2" if "a8_axis2" in init_map else ensure_init("hj_axis2", [2], np.int64)
    # Reuse existing [1, 1, -1, head_dim] / [1, 1, 1, head_dim] value reshape initializer if present.
    v_shape_name = None
    for candidate in ("a8_v_shape", "/lm/model/layers.0/self_attn/Transpose_2_output_0_shape"):
        if candidate in init_map:
            v_shape_name = candidate
            break
    if v_shape_name is None:
        v_shape_name = ensure_init("a8_v_shape", [1, kv_heads, -1, head_dim], np.int64)

    # 1. Add com.custom.llm opset v1 and insert `history_len: INT64[1]` right before past_key_values.0.key.
    if not any(opset.domain == KV_CUSTOM_DOMAIN for opset in model.opset_import):
        model.opset_import.append(H.make_opsetid(KV_CUSTOM_DOMAIN, 1))
    first_kv_idx = next(idx for idx, v in enumerate(graph.input) if v.name == "past_key_values.0.key")
    key0_dims = graph.input[first_kv_idx].type.tensor_type.shape.dim
    transposed_cache = len(key0_dims) == 4 and key0_dims[2].dim_value == head_dim
    if not any(v.name == "history_len" for v in graph.input):
        graph.input.insert(first_kv_idx, H.make_tensor_value_info("history_len", TP.INT64, [1]))

    # 2. Convert all KV cache I/O tensors to static FLOAT16 [1, kv_heads, max_positions, head_dim] BNSH layout.
    static_kv_dims = [1, kv_heads, max_positions, head_dim]
    for value in (*graph.input, *graph.output):
        if value.name.startswith(("past_key_values.", "present.")):
            value.type.tensor_type.elem_type = TP.FLOAT16
            del value.type.tensor_type.shape.dim[:]
            for dim_val in static_kv_dims:
                d = value.type.tensor_type.shape.dim.add()
                d.dim_value = dim_val

    # 3. Replace Shape(past_key_values.0.key) past-length extractors with `history_len` / Squeeze(`history_len`).
    producers = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    drop_nodes: set[int] = set()
    remap: dict[str, str] = {}
    prepend_nodes: list[onnx.NodeProto] = []
    need_scalar_past = False
    for idx, node in enumerate(graph.node):
        if node.op_type != "Shape" or not node.input or node.input[0] != "past_key_values.0.key":
            continue
        has_slice = any(a.name == "start" for a in node.attribute)
        if has_slice:
            remap[node.output[0]] = "history_len"
            drop_nodes.add(idx)
        else:
            shape_readers = consumers.get(node.output[0], [])
            if len(shape_readers) != 1 or shape_readers[0].op_type != "Gather":
                raise ValueError(f"Unsupported past_key_values.0.key Shape consumer: {[r.op_type for r in shape_readers]}")
            gather_node = shape_readers[0]
            gather_idx = next(i for i, n in enumerate(graph.node) if n is gather_node)
            remap[gather_node.output[0]] = "hj_past_len"
            drop_nodes.add(idx)
            drop_nodes.add(gather_idx)
            need_scalar_past = True

    if need_scalar_past and "hj_past_len" not in producers:
        prepend_nodes.append(H.make_node("Squeeze", ["history_len", axis0_name], ["hj_past_len"], name="Hojo_LM_Squeeze_3"))

    kept_nodes = list(prepend_nodes)
    for idx, node in enumerate(graph.node):
        if idx in drop_nodes:
            continue
        remap_node_inputs(node, remap)
        kept_nodes.append(node)
    del graph.node[:]
    graph.node.extend(kept_nodes)

    # Locate the total-length `[1]` tensor (`hj_end1 = history_len + seq_len`).
    if not any("hj_end1" in node.output for node in graph.node):
        raise ValueError("convert_lm_to_inplaced_kv requires `hj_end1` (history_len + seq_len)")

    is_decode = any(node.output and node.output[0] == "r2_mqa_0_kt" for node in graph.node)
    if is_decode:
        ensure_init("r2_mqa_v_shape", [1, -1, head_dim], np.int64)

    # 4. Replace each layer's K and V Concat nodes with KVCacheUpdate(layout="BNSH") + active-window Slice.
    producers = {output: node for node in graph.node for output in node.output}
    new_nodes: list[onnx.NodeProto] = []
    kv_rewrite: dict[str, str] = {}
    for node in graph.node:
        # Handle layer K/V Concat -> KVCacheUpdate + Slice.
        if node.op_type == "Concat" and node.output and node.output[0].startswith("present."):
            out_name = node.output[0]
            _, layer_str, kind = out_name.split(".")
            layer = int(layer_str)
            past_name = f"past_key_values.{layer}.{kind}"
            if len(node.input) != 2 or node.input[0] != past_name:
                raise ValueError(f"Expected 2-input Concat({past_name}, new) -> {out_name}")
            new_src = node.input[1]
            if kind == "key":
                # Convert pre-transposed [1, 1, D, S] key producer into BNSH [1, 1, S, D] via zero-copy Reshape.
                if transposed_cache:
                    k_prod = producers.get(new_src)
                    if k_prod is not None and k_prod.op_type == "Transpose":
                        k_orig = k_prod.input[0]
                        k_bnsh = f"hj_l{layer}_a8_k_bnsh"
                        new_nodes.append(H.make_node("Reshape", [k_orig, v_shape_name], [k_bnsh], name=k_bnsh))
                        new_src = k_bnsh
                active_name = f"hj_l{layer}_key_active_f16"
                new_nodes.append(H.make_node(
                    KV_CUSTOM_OP,
                    [past_name, new_src, "history_len", "hj_end1"],
                    [out_name],
                    name=f"hj_l{layer}_key_kv_update",
                    domain=KV_CUSTOM_DOMAIN,
                    layout="BNSH",
                ))
                new_nodes.append(H.make_node(
                    "Slice",
                    [out_name, axis0_name, "hj_end1", axis2_name],
                    [active_name],
                    name=f"hj_l{layer}_key_kv_slice",
                ))
                if transposed_cache and not is_decode:
                    k_t_name = f"/lm/model/layers.{layer}/self_attn/Transpose_3_output_0"
                    new_nodes.append(H.make_node(
                        "Transpose",
                        [active_name],
                        [k_t_name],
                        name=f"/lm/model/layers.{layer}/self_attn/Transpose_3",
                        perm=[0, 1, 3, 2],
                    ))
                    kv_rewrite[out_name] = k_t_name
                else:
                    kv_rewrite[out_name] = active_name
            else:
                active_name = f"hj_l{layer}_value_active_f16"
                new_nodes.append(H.make_node(
                    KV_CUSTOM_OP,
                    [past_name, new_src, "history_len", "hj_end1"],
                    [out_name],
                    name=f"hj_l{layer}_value_kv_update",
                    domain=KV_CUSTOM_DOMAIN,
                    layout="BNSH",
                ))
                new_nodes.append(H.make_node(
                    "Slice",
                    [out_name, axis0_name, "hj_end1", axis2_name],
                    [active_name],
                    name=f"hj_l{layer}_value_kv_slice",
                ))
                kv_rewrite[out_name] = active_name
            continue

        # In flattened MQA decode with pre-transposed cache, `Reshape(present.{l}.key, r2_mqa_k_shape) -> r2_mqa_{l}_kt`
        # becomes `Reshape(hj_l{l}_key_active_f16, r2_mqa_v_shape) -> r2_mqa_{l}_k` + `Transpose([0,2,1]) -> r2_mqa_{l}_kt`.
        if (transposed_cache and is_decode and node.op_type == "Reshape" and len(node.input) == 2
                and node.input[0] in kv_rewrite and node.input[0].endswith(".key")
                and node.output and node.output[0].endswith("_kt")):
            kt_out = node.output[0]
            k_3d = kt_out[:-1]  # r2_mqa_{layer}_k
            new_nodes.append(H.make_node("Reshape", [kv_rewrite[node.input[0]], "r2_mqa_v_shape"], [k_3d]))
            new_nodes.append(H.make_node("Transpose", [k_3d], [kt_out], perm=[0, 2, 1]))
            continue

        remap_node_inputs(node, kv_rewrite)
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    del graph.value_info[:]
    dce(model)
    props["lm_kv_update"] = "inplaced_kv"
    set_sorted_metadata(model, props)
    onnx.checker.check_model(model)
    return model


def export_tokenizer_tables(
    tokenizer_json: str | Path,
    target_dir: str | Path,
    *,
    prompts: Sequence[str] = GOLDEN_PROMPTS,
) -> dict[str, Path]:
    """Deterministically export hojo_bpe.txt, hojo_added.txt, hojo_chars.txt, and golden_prompt_ids.txt in pure Python."""
    from Inference_Hojo_TTS_ONNX import BpeTokenizer

    tokenizer_json = Path(tokenizer_json).resolve()
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    tok = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"]
    unk_id = vocab[tok["model"]["unk_token"]]

    bpe_lines = [f"V {len(vocab)} M {len(merges)} U {unk_id}"]
    for piece, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        bpe_lines.append(f"{idx} {piece.encode('utf-8').hex()}")
    for merge in merges:
        left, right = merge if isinstance(merge, list) else merge.split(" ", 1)
        bpe_lines.append(f"{left.encode('utf-8').hex()} {right.encode('utf-8').hex()}")
    bpe_text = "\n".join(bpe_lines) + "\n"

    added_lines = [
        f"{entry['id']} {1 if entry.get('normalized', False) else 0} {entry['content'].encode('utf-8').hex()}"
        for entry in tok["added_tokens"]
    ]
    added_text = "\n".join(added_lines) + "\n"

    re_w = re.compile(r"\w")
    re_s = re.compile(r"\s")

    def make_ranges(pred) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = prev = None
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            if pred(chr(cp)):
                if start is None:
                    start = prev = cp
                elif cp == prev + 1:
                    prev = cp
                else:
                    ranges.append((start, prev))
                    start = prev = cp
        if start is not None and prev is not None:
            ranges.append((start, prev))
        return ranges

    w_ranges = make_ranges(lambda c: bool(re_w.fullmatch(c)))
    s_ranges = make_ranges(lambda c: bool(re_s.fullmatch(c)))
    d_ranges = make_ranges(lambda c: unicodedata.category(c) == "Nd")
    lowers = [
        (cp, chr(cp).lower())
        for cp in range(0x110000)
        if not (0xD800 <= cp <= 0xDFFF) and chr(cp).lower() != chr(cp)
    ]
    char_lines: list[str] = []
    for tag, table in (("W", w_ranges), ("S", s_ranges), ("D", d_ranges)):
        char_lines.append(f"{tag} {len(table)}")
        char_lines.extend(f"{a:x} {b:x}" for a, b in table)
    char_lines.append(f"L {len(lowers)}")
    char_lines.extend(f"{cp:x} {low.encode('utf-8').hex()}" for cp, low in lowers)
    chars_text = "\n".join(char_lines) + "\n"

    bpe_tok = BpeTokenizer(tokenizer_json)
    golden_lines = [
        f"{prompt_text}\t{' '.join(str(tok_id) for tok_id in bpe_tok.encode(prompt_text))}"
        for prompt_text in prompts
    ]
    golden_text = "\n".join(golden_lines) + "\n"

    written: dict[str, Path] = {}
    for fname, content in (
        ("hojo_bpe.txt", bpe_text),
        ("hojo_added.txt", added_text),
        ("hojo_chars.txt", chars_text),
        ("golden_prompt_ids.txt", golden_text),
    ):
        dest = target_dir / fname
        dest.write_text(content, encoding="utf-8")
        written[fname] = dest
    return written
