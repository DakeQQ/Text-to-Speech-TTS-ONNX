"""Optimize or quantize Hojo-TTS ONNX graphs and verify structural/numerical gates.

Supports the surgical float16 Hojo pipeline (RMSNorm -> SimplifiedLayerNormalization, baked RoPE,
packed QKV/gate_up MatMuls, QuickGelu, SkipSimplifiedLayerNormalization, Audio8 pre-transposed K
cache attention, Gelu, ConvNeXt gamma fold, SkipLayerNormalization, half-basis mirror-symmetric
iSTFT, boundary-only Casts, and FP16-vs-FP32 parity gates) as well as Q8/Q4/DYNAMIC/F32 plans via
Optimize_ONNX_Common.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
from onnx import TensorProto as TP, helper as H, numpy_helper as NH


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (candidate / "Optimize_ONNX_Common.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break

import Export_Hojo_TTS as export_hojo  # noqa: E402
from Export_Hojo_TTS import dce  # noqa: E402
from Inference_Hojo_TTS_ONNX import (  # noqa: E402
    VOICE_ID, PROMPT_AUDIO_PATH, PROMPT_TEXT, MAX_NEW_TOKENS, RANDOM_SEED, BpeTokenizer, build_prompt, build_session_options,
    io_dtype, load_prompt_audio, normalize_text, parse_metadata, prompt_parts, resolve_voice_index, voice_lookup,
)
from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    process_model,
    read_onnx_metadata,
    replace_onnx_metadata,
    resolve_plan,
    uses_mixed_precision,
)
from Shared_Weights import (  # noqa: E402
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    SharedInitializerStore,
    bundle_shared_initializers,
    collect_live_external_names,
    collect_normalization_exclusions,
    inline_model_initializers,
    iter_subgraphs,
    set_sorted_metadata,
)
from Streaming_Hojo import StatefulVocoder  # noqa: E402


# ============================== USER CONFIG ==============================
SOURCE_FOLDER = SCRIPT_DIR / "HojoTTS_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "HojoTTS_Optimized"

# Per-model precision: "F16" | "F32" | "DYNAMIC" | "Q4" | "Q8"
LM_PRECISION = "F16"                        # Hojo_LM.onnx (autoregressive language model)
VOCODER_PRECISION = "F16"                   # Hojo_Vocoder.onnx (audio decoder / vocoder; F16 or F32 recommended)
REFERENCE_PREPROCESS_PRECISION = "F16"      # Hojo_ReferencePreprocess.onnx (80M voice-clone reference encoder)
FINE_STREAM_PRECISION: str | None = None    # Hojo_Fine_Stream.onnx weights; None follows the effective vocoder precision
MODEL_PRECISION = LM_PRECISION              # Backward-compatible alias for default LM precision
KEEP_IO_TYPE = False

MATMUL_ALGORITHM = "AFFINE_REFINE_V2"
BLOCK_SIZE = 64
ACCURACY_LEVEL = 4
OPTIMIZER_LEVEL = 2
BUNDLE_SHARED_INITIALIZERS = True      # Set True to deduplicate shared weights into Hojo_SharedInitializers.onnx.data.
STABILIZE_LM_FP32_ATTENTION = True     # Wrap F16 LM attention MatMuls, Softmax, QuickGelu, and non-rope_add Mul/Add in FP32.
DROP_WHOLE_VOCODER_IN_ADVANCED = True  # Omit Hojo_Vocoder.onnx in --advanced streaming packages to prune dead shared weights.
KV_UPDATE_MODE = "concat"              # "concat" (portable standard ORT) | "inplaced_kv" (com.custom.llm::KVCacheUpdate).
EXPORT_TOKENIZER_TABLES = False        # Export hojo_bpe.txt, hojo_added.txt, hojo_chars.txt, golden_prompt_ids.txt.

# Per-graph quantization and optimization plan (edit precisions above or customize each Plan directly).
MODEL_PLANS: dict[str, Plan] = {
    "Hojo_LM": Plan(
        method=LM_PRECISION,
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        opt_level=OPTIMIZER_LEVEL,
        optimize=True,
        transformer=True,
        external=False,
    ),
    "Hojo_Vocoder": Plan(
        method=VOCODER_PRECISION,
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        opt_level=OPTIMIZER_LEVEL,
        optimize=True,
        transformer=True,
        external=False,
    ),
    "Hojo_ReferencePreprocess": Plan(
        method=REFERENCE_PREPROCESS_PRECISION,
        algo=MATMUL_ALGORITHM,
        op_types=("MatMul",),
        axes=(0,),
        block_size=BLOCK_SIZE,
        accuracy_level=ACCURACY_LEVEL,
        opt_level=OPTIMIZER_LEVEL,
        optimize=True,
        transformer=True,
        external=False,
        first_slim_no_shape_infer=True,
        second_slim_no_shape_infer=True,
    ),
    "Hojo_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
        external=False,
    ),
}
# =========================================================================

LM, VOC, REF_PRE, META = (
    export_hojo.LM_MODEL_NAME,
    export_hojo.VOCODER_MODEL_NAME,
    export_hojo.REF_PREPROCESS_MODEL_NAME,
    export_hojo.METADATA_MODEL_NAME,
)
# speaker_embedding stays fp32 so Hojo_ReferencePreprocess output feeds the vocoder without a host cast.
FP32_IO = ["ctrl", "uniform", "wav", "speaker_embedding"]
VERIFY_TEXTS = ("你好！欢迎使用离线语音合成。", "今天天气不错，我们去公园散步吧。")
VERIFY_MAX_STEPS = MAX_NEW_TOKENS
MIN_GREEDY_AGREEMENT = 0.97
MAX_HIDDEN_REL_ERR = 0.01
MIN_VOCODER_SNR_DB = 45.0


class Graph:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.g = model.graph
        self.refresh()

    def refresh(self) -> None:
        self.prod = {o: n for n in self.g.node for o in n.output}
        self.cons: dict[str, list[onnx.NodeProto]] = {}
        for n in self.g.node:
            for i in export_hojo.node_inputs(n):
                self.cons.setdefault(i, []).append(n)
        self.init = {i.name: i for i in self.g.initializer}

    def value(self, name: str) -> np.ndarray | None:
        if name in self.init:
            return NH.to_array(self.init[name])
        n = self.prod.get(name)
        if n is not None and n.op_type == "Constant":
            return NH.to_array(n.attribute[0].t)
        return None

    def only_consumer(self, name: str, op: str) -> onnx.NodeProto | None:
        c = self.cons.get(name, [])
        return c[0] if len(c) == 1 and c[0].op_type == op else None

    def is_scalar(self, name: str, v: float) -> bool:
        a = self.value(name)
        return a is not None and a.size == 1 and float(a.ravel()[0]) == v

    def remove(self, nodes: list[onnx.NodeProto]) -> None:
        drop = set()
        for target in nodes:
            outputs = tuple(target.output)
            candidates = [(index, node) for index, node in enumerate(self.g.node) if tuple(node.output) == outputs]
            if len(candidates) > 1:
                candidates = [(index, node) for index, node in candidates
                              if (node.op_type, node.domain, tuple(node.input)) ==
                              (target.op_type, target.domain, tuple(target.input))]
            if not outputs or len(candidates) != 1:
                raise ValueError(f"Graph removal requires a unique existing node: {outputs}")
            drop.add(candidates[0][0])
        keep = [node for index, node in enumerate(self.g.node) if index not in drop]
        del self.g.node[:]
        self.g.node.extend(keep)
        self.refresh()

    def node(self, name: str) -> onnx.NodeProto | None:
        return next((n for n in self.g.node if n.name == name), None)


def hidden_width(gr: Graph, x: str) -> int | None:
    for v in list(gr.g.value_info) + list(gr.g.input):
        if v.name == x:
            d = v.type.tensor_type.shape.dim
            if d and d[-1].dim_value > 0:
                return int(d[-1].dim_value)
    return None


def fuse_rmsnorm(model: onnx.ModelProto) -> int:
    gr = Graph(model)
    fused = 0
    for pw in [n for n in gr.g.node if n.op_type == "Pow"]:
        if not gr.is_scalar(pw.input[1], 2.0):
            continue
        x = pw.input[0]
        rm = gr.only_consumer(pw.output[0], "ReduceMean")
        add = rm and gr.only_consumer(rm.output[0], "Add")
        sq = add and gr.only_consumer(add.output[0], "Sqrt")
        div = sq and gr.only_consumer(sq.output[0], "Div")
        if not div or not gr.is_scalar(div.input[0], 1.0):
            continue
        axes = gr.value(rm.input[1]) if len(rm.input) > 1 else None
        eps = gr.value(add.input[1] if add.input[0] == rm.output[0] else add.input[0])
        mul = gr.only_consumer(div.output[0], "Mul")
        if axes is None or list(axes.ravel()) != [-1] or eps is None or not mul or x not in mul.input:
            continue
        chain = [pw, rm, add, sq, div, mul]
        scale = gr.only_consumer(mul.output[0], "Mul")
        gamma = None
        if scale is not None:
            gamma = gr.value(scale.input[1] if scale.input[0] == mul.output[0] else scale.input[0])
        if gamma is not None and gamma.ndim == 1:
            chain.append(scale)
        else:
            gamma = None
        width = gamma.shape[0] if gamma is not None else hidden_width(gr, x)
        if width is None:
            continue
        name = f"{pw.name}_gamma"
        gr.g.initializer.append(NH.from_array((gamma if gamma is not None else np.ones(width)).astype(np.float32), name))
        node = H.make_node("SimplifiedLayerNormalization", [x, name], [chain[-1].output[0]], name=f"{pw.name}_rms",
                           axis=-1, epsilon=float(eps.ravel()[0]), stash_type=1)
        gr.g.node.insert(list(gr.g.node).index(chain[-1]), node)
        gr.remove(chain)
        fused += 1
    return fused


def standardize_rmsnorm(model: onnx.ModelProto) -> None:
    version = onnx.defs.get_schema("RMSNormalization").since_version
    for node in model.graph.node:
        if node.op_type == "SimplifiedLayerNormalization":
            node.op_type, node.domain = "RMSNormalization", ""
    for opset in model.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, version)


def bake_rotary(model: onnx.ModelProto, cfg: dict) -> int:
    gr = Graph(model)
    pairs = [(c, s) for c in gr.g.node if c.op_type == "Cos" for s in gr.g.node if s.op_type == "Sin" and s.input[0] == c.input[0]]
    _need(len(pairs) == 1, f"one rotary Cos/Sin pair, found {len(pairs)}")
    cos_n, sin_n = pairs[0]
    head_dim, theta, positions = int(cfg["head_dim"]), float(cfg["rope_theta"]), int(cfg["max_positions"])
    product = _walk(gr, cos_n.input[0], "MatMul", up=True)
    _need(product is not None, "rotary frequency-position MatMul")
    frequency = gr.value(product.input[0])
    _need(frequency is not None and frequency.size == head_dim // 2,
          "rotary inverse frequencies carried by the checkpoint")
    expected = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    _need(np.allclose(frequency.ravel(), expected, rtol=np.finfo(np.float32).eps ** 0.25),
          "checkpoint inverse frequencies match plain RoPE")
    temporary = onnx.ModelProto.FromString(model.SerializeToString())
    positions_input = product.input[1]
    del temporary.graph.node[:]
    temporary.graph.node.extend([
        H.make_node("MatMul", list(product.input), ["frequency_product"]),
        H.make_node("Transpose", ["frequency_product"], ["angles_half"], perm=[0, 2, 1]),
        H.make_node("Concat", ["angles_half", "angles_half"], ["angles"], axis=-1),
        H.make_node("Cos", ["angles"], ["cos_table"]),
        H.make_node("Sin", ["angles"], ["sin_table"]),
    ])
    del temporary.graph.input[:]
    temporary.graph.input.append(H.make_tensor_value_info(positions_input, TP.FLOAT, [1, 1, positions]))
    del temporary.graph.output[:]
    temporary.graph.output.extend([H.make_tensor_value_info(name, TP.FLOAT, [1, positions, head_dim])
                                   for name in ("cos_table", "sin_table")])
    del temporary.graph.initializer[:]
    temporary.graph.initializer.append(NH.from_array(frequency, product.input[0]))
    del temporary.graph.value_info[:]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(temporary.SerializeToString(), options, providers=["CPUExecutionProvider"])
    cos_t, sin_t = session.run(None, {positions_input: np.arange(positions, dtype=np.float32).reshape(1, 1, -1)})
    cos_t, sin_t = cos_t[0], sin_t[0]
    sin_t[:, :head_dim // 2] *= -1.0
    gr.g.initializer.extend([NH.from_array(cos_t[None], "rope_cos"), NH.from_array(sin_t[None], "rope_sin_signed"),
                             NH.from_array(np.array([1], np.int64), "rope_axis1")])
    idx = list(gr.g.node).index(cos_n)
    for table, old in (("rope_cos", cos_n), ("rope_sin_signed", sin_n)):
        gr.g.node.insert(idx, H.make_node("Slice", [table, "hj_past_len1", "hj_end1", "rope_axis1"],
                                          [old.output[0]], name=f"{table}_slice"))
    gr.remove([cos_n, sin_n])
    negs = 0
    for neg in [n for n in gr.g.node if n.op_type == "Neg"]:
        cat = gr.only_consumer(neg.output[0], "Concat")
        if cat is None or cat.input[0] != neg.output[0]:
            continue
        cat.input[0] = neg.input[0]
        gr.remove([neg])
        negs += 1
    return negs


def _const(gr: Graph, name: str, value, dtype) -> str:
    if name not in gr.init:
        t = NH.from_array(np.asarray(value, dtype=dtype), name)
        gr.g.initializer.append(t)
        gr.init[name] = t
    return name


def _need(cond, what: str) -> None:
    if not cond:
        raise RuntimeError(f"fusion precondition failed: {what}")


def pack_matmuls(gr: Graph, a: onnx.NodeProto, b: list[onnx.NodeProto], tag: str) -> list[str]:
    nodes = [a] + b
    ws = [gr.value(n.input[1]) for n in nodes]
    _need(all(w is not None and w.ndim == 2 for w in ws) and len({n.input[0] for n in nodes}) == 1, tag)
    w = np.concatenate(ws, axis=1)
    _const(gr, f"{tag}_w", w, w.dtype)
    sizes = _const(gr, f"{tag}_sizes", [x.shape[1] for x in ws], np.int64)
    outs = [n.output[0] for n in nodes]
    names = {n.name for n in nodes}
    gr.remove([n for n in gr.g.node if n.name in names])
    pos = next(i for i, n in enumerate(gr.g.node) if any(c in n.input for c in outs))
    gr.g.node.insert(pos, H.make_node("Split", [f"{tag}_out", sizes], outs, name=f"{tag}_split", axis=-1))
    gr.g.node.insert(pos, H.make_node("MatMul", [a.input[0], f"{tag}_w"], [f"{tag}_out"], name=f"{tag}_matmul"))
    gr.refresh()
    return outs


def _walk(gr: Graph, start: str, op: str, up: bool, weighted: bool = False) -> onnx.NodeProto | None:
    """Nearest producer (up) or consumer (down) of `start` with op type `op` (MatMul with a constant weight if weighted)."""
    frontier, seen = [start], {start}
    while frontier:
        nodes = [gr.prod[t] for t in frontier if t in gr.prod] if up else [c for t in frontier for c in gr.cons.get(t, [])]
        for n in nodes:
            if n.op_type == op and (not weighted or gr.value(n.input[1]) is not None):
                return n
        frontier = [t for n in nodes for t in (n.input if up else n.output) if t and t not in seen and t not in gr.init]
        seen.update(frontier)
    return None


def _layer_nodes(gr: Graph, layer: int) -> dict[str, onnx.NodeProto]:
    """Locate one decoder layer's attention/MLP nodes from its KV-cache outputs."""
    ck, cv = gr.prod.get(f"present.{layer}.key"), gr.prod.get(f"present.{layer}.value")
    _need(ck is not None and cv is not None and ck.op_type == cv.op_type == "Concat" and
          ck.input[0] == f"past_key_values.{layer}.key" and cv.input[0] == f"past_key_values.{layer}.value",
          f"layer {layer} KV Concat-grow")
    d = {"k_cat": ck, "v_cat": cv}
    d["kn"] = _walk(gr, ck.input[1], "SimplifiedLayerNormalization", up=True)
    d["mm"] = _walk(gr, ck.output[0], "MatMul", up=False)
    d["mm1"] = _walk(gr, cv.output[0], "MatMul", up=False)
    _need(all(d.values()), f"layer {layer} k_norm / attention MatMuls")
    d["qn"] = _walk(gr, d["mm"].input[0], "SimplifiedLayerNormalization", up=True)
    d["scale"] = gr.only_consumer(d["mm"].output[0], "Mul")
    d["add"] = d["scale"] and gr.only_consumer(d["scale"].output[0], "Add")
    d["o_proj"] = _walk(gr, d["mm1"].output[0], "MatMul", up=False, weighted=True)
    _need(all(d.values()), f"layer {layer} q_norm / score scale / mask Add / o_proj")
    return d


def fuse_lm(model: onnx.ModelProto, cfg: dict) -> Counter:
    gr = Graph(model)
    _need(not any(n.op_type == "QuickGelu" for n in gr.g.node), "model already fused")
    kv_heads, hd, layers = int(cfg["num_key_value_heads"]), int(cfg["head_dim"]), int(cfg["num_hidden_layers"])
    stats = Counter()
    for layer in range(layers):
        d = _layer_nodes(gr, layer)
        for x in ("q", "k"):
            r = gr.prod.get(d[f"{x}n"].input[0])
            proj = r is not None and r.op_type == "Reshape" and gr.prod.get(r.input[0])
            _need(proj and proj.op_type == "MatMul" and len(gr.cons[proj.output[0]]) == 1,
                  f"layer {layer} {x}_proj -> Reshape -> norm")
            d[f"{x}p"] = proj
        qp, kp = d["qp"], d["kp"]
        vp = _walk(gr, d["v_cat"].input[1], "MatMul", up=True, weighted=True)
        _need(vp is not None and vp.input[0] == qp.input[0] == kp.input[0], f"layer {layer} shared q/k/v input")
        for x in ("qn", "kn"):
            _need(list(gr.value(d[x].input[1]).shape) == [hd], f"layer {layer} q/k norm width")
        mm, mm1 = d["mm"].name, d["mm1"].name
        pack_matmuls(gr, qp, [kp, vp], f"hj_l{layer}_qkv")
        stats["packed MatMul"] += 1
        stats["KV repeat removed"] += _broadcast_kv(gr, layer, mm, mm1, kv_heads)
        _fuse_mlp_layer(gr, layer, d["o_proj"].name)
        stats["packed MatMul"] += 1
        stats["QuickGelu"] += 1
    stats["dead nodes removed"] += dce(model)
    stats["SkipSimplifiedLayerNormalization"] += _fuse_skip_norms(gr)
    _need(stats["SkipSimplifiedLayerNormalization"] == 2 * layers, f"LM fusion counts {dict(stats)}")
    _add_ms_domain(model)
    return stats


def fuse_attention_audio8(model: onnx.ModelProto, cfg: dict) -> Counter:
    gr = Graph(model)
    heads, kv_heads = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
    hd, layers = int(cfg["head_dim"]), int(cfg["num_hidden_layers"])
    qk = heads + kv_heads
    _need(kv_heads == 1, "pre-transposed K layout assumes one KV head")
    tables = {n.input[0]: n.output[0] for n in gr.g.node if n.op_type == "Slice" and n.input[0] in ("rope_cos", "rope_sin_signed")}
    _need(len(tables) == 2, "baked rotary tables (bake_rotary)")
    cos_t, sin_t = tables["rope_cos"], tables["rope_sin_signed"]
    masks = set()
    for layer in range(layers):
        d = _layer_nodes(gr, layer)
        sl = gr.prod.get(next(i for i in d["add"].input if i != d["scale"].output[0]))
        _need(sl is not None and sl.op_type == "Slice", f"layer {layer} scores + sliced mask")
        masks.add(sl.input[0])
    _need(len(masks) == 1 and next(iter(masks)) in gr.prod, f"one shared causal mask, found {masks}")
    mask = masks.pop()
    compute_dtype = gr.value(gr.prod[cos_t].input[0]).dtype
    axis2 = _const(gr, "a8_axis2", [2], np.int64)
    scale5 = _const(gr, "a8_qk_scale", np.array([hd ** -0.5] * heads + [1.0] * kv_heads).reshape(qk, 1), compute_dtype)
    shape5 = _const(gr, "a8_qk_shape", [1, -1, qk, hd], np.int64)
    half = _const(gr, "a8_half", [hd // 2, hd // 2], np.int64)
    qk_kv = _const(gr, "a8_qk_kv", [heads, kv_heads], np.int64)
    v_shape = _const(gr, "a8_v_shape", [1, kv_heads, -1, hd], np.int64)
    # Right after the baked RoPE slices: the causal mask is produced after layer 0's q/k projections.
    first = max(list(gr.g.node).index(gr.prod[t]) for t in (cos_t, sin_t)) + 1
    shared = []
    for tag, src in (("cos", cos_t), ("sin", sin_t)):
        shared += [H.make_node("Unsqueeze", [src, axis2], [f"a8_{tag}_u"], name=f"a8_{tag}_unsqueeze"),
                   H.make_node("Mul", [f"a8_{tag}_u", scale5], [f"a8_{tag}5"], name=f"a8_{tag}_scale")]
    for i, n in enumerate(shared):
        gr.g.node.insert(first + i, n)
    gr.refresh()
    stats = Counter()
    for layer in range(layers):
        p = f"hj_l{layer}_"
        d = _layer_nodes(gr, layer)
        qn, kn, mm, scale, add, cat = (d[x] for x in ("qn", "kn", "mm", "scale", "add", "k_cat"))
        split = gr.prod.get(gr.prod[qn.input[0]].input[0])
        _need(split is not None and split.op_type == "Split" and split is gr.prod.get(gr.prod[kn.input[0]].input[0]),
              f"layer {layer} packed q/k/v Split")
        _need(np.array_equal(gr.value(qn.input[1]), gr.value(kn.input[1])), f"layer {layer} q/k norm gamma equal")
        _need(abs(float(gr.value(scale.input[1]).ravel()[0]) - hd ** -0.5) < 1e-3, f"layer {layer} softmax scale")
        sizes = gr.value(split.input[1])
        _need(list(sizes) == [heads * hd, kv_heads * hd, kv_heads * hd], f"layer {layer} qkv split sizes")
        sizes2 = _const(gr, f"{p}qk_v_sizes", [qk * hd, kv_heads * hd], np.int64)
        v_out = split.output[2]
        split.input[1] = sizes2
        del split.output[:]
        split.output.extend([f"{p}a8_qk", v_out])
        x = f"{p}a8_"
        eps = next(a.f for a in qn.attribute if a.name == "epsilon")
        new = [
            H.make_node("Reshape", [f"{p}a8_qk", shape5], [x + "qk5"], name=x + "reshape"),
            H.make_node("SimplifiedLayerNormalization", [x + "qk5", qn.input[1]], [x + "qkn"], name=x + "qk_norm",
                        axis=-1, epsilon=eps, stash_type=1),
            H.make_node("Split", [x + "qkn", half], [x + "lo", x + "hi"], name=x + "rot_split", axis=-1),
            H.make_node("Concat", [x + "hi", x + "lo"], [x + "rot"], name=x + "rot_cat", axis=-1),
            H.make_node("Mul", [x + "qkn", "a8_cos5"], [x + "c"], name=x + "rope_cos"),
            H.make_node("Mul", [x + "rot", "a8_sin5"], [x + "s"], name=x + "rope_sin"),
            H.make_node("Add", [x + "c", x + "s"], [x + "qkr"], name=x + "rope_add"),
            H.make_node("Split", [x + "qkr", qk_kv], [x + "q", x + "k"], name=x + "qk_split", axis=2),
            H.make_node("Transpose", [x + "q"], [x + "qt"], name=x + "q_t", perm=[0, 2, 1, 3]),
            H.make_node("Transpose", [x + "k"], [x + "kt"], name=x + "k_t", perm=[0, 2, 3, 1]),
            H.make_node("Concat", [cat.input[0], x + "kt"], [f"present.{layer}.key"], name=x + "k_cat", axis=-1),
        ]
        pos = list(gr.g.node).index(split) + 1
        for i, n in enumerate(new):
            gr.g.node.insert(pos + i, n)
        mm.input[:] = [x + "qt", f"present.{layer}.key"]
        _need(scale.output[0] in add.input and gr.prod[next(i for i in add.input if i != scale.output[0])].op_type
              == "Slice", f"layer {layer} scores + sliced mask")
        add.input[:] = [mm.output[0], mask]
        gr.remove([cat])
        cat_v = gr.prod.get(f"present.{layer}.value")
        tr_v = gr.prod.get(cat_v.input[1]) if cat_v else None
        if tr_v is not None and tr_v.op_type == "Transpose":
            rs_v = gr.prod.get(tr_v.input[0])
            if rs_v is not None and rs_v.op_type == "Reshape" and rs_v.input[0] == v_out:
                rs_v.input[1] = v_shape
                cat_v.input[1] = rs_v.output[0]
                gr.remove([tr_v])
        stats["Audio8 attention"] += 1
    for n in gr.g.node:
        if n.op_type == "Shape" and n.input[0] == "past_key_values.0.key":
            if any(a.name == "start" for a in n.attribute):
                _need([(a.name, a.i) for a in n.attribute] in ([("end", 3), ("start", 2)], [("start", 2), ("end", 3)]),
                      "past-length Shape slice")
                del n.attribute[:]
                n.attribute.extend([H.make_attribute("start", 3), H.make_attribute("end", 4)])
            else:
                for c in gr.cons.get(n.output[0], []):
                    _need(c.op_type == "Gather" and gr.is_scalar(c.input[1], 2.0), "past-length Gather")
                    c.input[1] = _const(gr, "a8_c3", np.int64(3), np.int64)
            stats["past-length Shape moved"] += 1
    for v in list(gr.g.input) + list(gr.g.output):
        if v.name.endswith(".key") and (v.name.startswith("past_key_values.") or v.name.startswith("present.")):
            d = v.type.tensor_type.shape.dim
            if len(d) != 4:
                continue
            seq = d[2].dim_param or "kv_sequence_length"
            d[2].Clear()
            d[2].dim_value = hd
            d[3].Clear()
            d[3].dim_param = seq
    stats["dead nodes removed"] += dce(model)
    _need(stats["Audio8 attention"] == layers and stats["past-length Shape moved"] == 2, f"Audio8 counts {stats}")
    return stats


def _broadcast_kv(gr: Graph, layer: int, mm_k: str, mm_v: str, kv_heads: int) -> int:
    _need(kv_heads == 1, "KV broadcast needs one KV head")
    removed = 0
    for side, mm_name, idx in (("key", mm_k, 1), ("value", mm_v, 1)):
        mm = gr.node(mm_name)
        _need(mm is not None, f"layer {layer} {mm_name}")
        src = mm.input[idx]
        tr = gr.prod.get(src) if side == "key" else None
        if tr is not None:
            _need(tr.op_type == "Transpose" and list(tr.attribute[0].ints) == [0, 1, 3, 2], f"layer {layer} K^T")
            src = tr.input[0]
        rs = gr.prod.get(src)
        ex = rs and gr.prod.get(rs.input[0])
        un = ex and gr.prod.get(ex.input[0])
        present = f"present.{layer}.{side}"
        cache = un.input[0] if un is not None else ""
        cast = gr.prod.get(cache)
        _need(rs is not None and rs.op_type == "Reshape" and ex is not None and ex.op_type == "Expand"
              and un is not None and un.op_type == "Unsqueeze" and (
                  cache == present or cast is not None and cast.op_type == "Cast" and cast.input[0] == present
                  and any(a.name == "to" and a.i == TP.FLOAT for a in cast.attribute)
              ), f"layer {layer} repeat_kv chain for {side}")
        for c in gr.cons.get(rs.output[0], []):
            if c.op_type == "Transpose" or c is mm:
                continue
            g2 = gr.cons.get(c.output[0], [])
            _need(c.op_type == "Shape" and not c.attribute and len(g2) == 1 and g2[0].op_type == "Gather" and
                  gr.is_scalar(g2[0].input[1], 2.0), f"layer {layer} unexpected repeat_kv consumer {c.op_type}")
            c.input[0] = un.input[0]
        if tr is not None:
            tr.input[0] = un.input[0]
        else:
            mm.input[idx] = un.input[0]
        gr.refresh()
        removed += 1
    return removed


def _fuse_mlp_layer(gr: Graph, layer: int, o_proj: str) -> None:
    norm = _walk(gr, gr.node(o_proj).output[0], "SimplifiedLayerNormalization", up=False)
    mms = [c for c in gr.cons.get(norm.output[0], []) if c.op_type == "MatMul"] if norm else []
    gates = [m for m in mms if any(c.op_type == "Sigmoid" for c in gr.cons.get(m.output[0], []))]
    _need(len(mms) == 2 and len(gates) == 1, f"layer {layer} post-attention norm -> gate/up MatMuls")
    gate = gates[0]
    up = next(m for m in mms if m is not gate)
    sig = next(c for c in gr.cons[gate.output[0]] if c.op_type == "Sigmoid")
    silu = gr.only_consumer(sig.output[0], "Mul")
    _need(silu is not None and set(silu.input) == {gate.output[0], sig.output[0]}, f"layer {layer} SwiGLU")
    g_out, s_out, drop = gate.output[0], silu.output[0], {sig.name, silu.name}
    pack_matmuls(gr, gate, [up], f"hj_l{layer}_gate_up")
    gr.g.node.insert(list(gr.g.node).index(gr.node(silu.name)), H.make_node(
        "QuickGelu", [g_out], [s_out], name=f"hj_l{layer}_silu", domain="com.microsoft", alpha=1.0))
    gr.remove([n for n in gr.g.node if n.name in drop])


def _fuse_skip_norms(gr: Graph) -> int:
    gr.refresh()
    outs = {o.name for o in gr.g.output}
    fused = 0
    for norm_name in [n.name for n in gr.g.node if n.op_type == "SimplifiedLayerNormalization"]:
        norm = gr.node(norm_name)
        add = gr.prod.get(norm.input[0])
        if add is None or add.op_type != "Add" or any(i in gr.init for i in add.input) or add.output[0] in outs:
            continue
        eps = next(a.f for a in norm.attribute if a.name == "epsilon")
        new = H.make_node("SkipSimplifiedLayerNormalization", [add.input[0], add.input[1], norm.input[1]],
                          [norm.output[0], "", "", add.output[0]], name=f"{norm_name}_skip",
                          domain="com.microsoft", epsilon=eps)
        drop = {add.name, norm_name}
        # At the Add, not the norm: its residual output may be read by nodes placed in between.
        gr.g.node.insert(list(gr.g.node).index(add), new)
        gr.remove([n for n in gr.g.node if n.name in drop])
        fused += 1
    return fused


def fuse_gelu_fp16(model: onnx.ModelProto) -> int:
    gr = Graph(model)
    fused = 0
    for erf in [n for n in gr.g.node if n.op_type == "Erf"]:
        div = gr.prod.get(erf.input[0])
        add = gr.only_consumer(erf.output[0], "Add")
        mul = add and gr.only_consumer(add.output[0], "Mul")
        half = mul and gr.only_consumer(mul.output[0], "Mul")
        if not (div is not None and div.op_type == "Div" and half and len(gr.cons[div.output[0]]) == 1):
            continue
        x = div.input[0]
        sq2 = gr.value(div.input[1])
        c1 = gr.value(next(i for i in add.input if i != erf.output[0]))
        c5 = gr.value(next(i for i in half.input if i != mul.output[0]))
        if sq2 is None or c1 is None or c5 is None or x not in mul.input or sq2.size != 1 or \
                abs(float(sq2.ravel()[0]) - 2 ** 0.5) > 1e-3 or float(c1.ravel()[0]) != 1 or float(c5.ravel()[0]) != 0.5:
            continue
        gr.g.node.insert(list(gr.g.node).index(half), H.make_node(
            "Gelu", [x], [half.output[0]], name=f"{erf.name}_gelu", domain="com.microsoft"))
        gr.remove([div, erf, add, mul, half])
        fused += 1
    dce(model)
    _add_ms_domain(model)
    return fused


def _up(gr: Graph, name: str, ops: tuple[str, ...]) -> onnx.NodeProto | None:
    """Follow single-input layout ops (Transpose/Reshape) up to the first producer not in `ops`."""
    n = gr.prod.get(name)
    while n is not None and n.op_type in ops:
        n = gr.prod.get(n.input[0])
    return n


def fuse_vocoder_fp16(model: onnx.ModelProto, fp32_nodes: set[str] | None = None) -> Counter:
    gr = Graph(model)
    nm = gr.node
    fp32_nodes = fp32_nodes or set()
    stats = Counter()
    shared_mask = None
    for sm_name in [n.name for n in gr.g.node if n.op_type == "Softmax" and n.name not in fp32_nodes]:
        sm = nm(sm_name)
        add3 = gr.prod.get(sm.input[0])
        add2 = add3 is not None and add3.op_type == "Add" and next(
            (gr.prod[i] for i in add3.input if i in gr.prod and gr.prod[i].op_type == "Add"), None)
        div = add2 and next((gr.prod[i] for i in add2.input if i in gr.prod and gr.prod[i].op_type == "Div"), None)
        qk = div and gr.prod.get(div.input[0])
        _need(qk is not None and qk.op_type == "MatMul", f"{sm_name}: scores MatMul -> Div -> Add(band) -> Add(pad)")
        q_rs = gr.prod.get(gr.prod[qk.input[0]].input[0])
        sl0 = _up(gr, qk.input[0], ("Transpose", "Reshape"))
        bias = sl0 is not None and sl0.op_type == "Slice" and gr.prod.get(sl0.input[0])
        inp = bias and bias.op_type == "Add" and next(
            (gr.prod[i] for i in bias.input if i in gr.prod and gr.prod[i].op_type == "MatMul"), None)
        sl = [c for c in gr.cons.get(sl0.input[0], []) if c.op_type == "Slice"] if inp else []
        _need(len(sl) == 3 and q_rs is not None and q_rs.op_type == "Reshape", f"{sm_name}: in_proj -> 3 Slice q/k/v")
        sl.sort(key=lambda s: int(gr.value(s.input[1]).ravel()[0]))
        bname = next(i for i in bias.input if i != inp.output[0])
        w, bvec, sc = gr.value(inp.input[1]), gr.value(bname), gr.value(div.input[1])
        hd = int(gr.value(q_rs.input[1]).ravel()[-1])
        _need(w is not None and bvec is not None and len(gr.cons[inp.input[1]]) == 1 and len(gr.cons[bname]) == 1
              and sc is not None and sc.size == 1 and abs(float(sc.ravel()[0]) - hd ** 0.5) < 1e-3,
              f"{sm_name}: private in_proj weights and scale sqrt(head_dim={hd})")
        d = w.shape[1] // 3
        w, bvec = w.copy(), bvec.copy()
        w[:, :d] /= w.dtype.type(sc.ravel()[0])
        bvec[:d] /= bvec.dtype.type(sc.ravel()[0])
        gr.init[inp.input[1]].CopyFrom(NH.from_array(w, inp.input[1]))
        gr.init[bname].CopyFrom(NH.from_array(bvec, bname))
        sizes = _const(gr, "voc_qkv_sizes", [d, d, d], np.int64)
        tag = f"hj_fine{stats['fine attention']}_"
        gr.g.node.insert(list(gr.g.node).index(sl[0]), H.make_node(
            "Split", [bias.output[0], sizes], [s.output[0] for s in sl], name=tag + "qkv_split", axis=-1))
        div_name, add3_name, add2_name = div.name, add3.name, add2.name
        scores = div.input[0]
        band = next(i for i in add2.input if i != div.output[0])
        pad = next(i for i in add3.input if i != add2.output[0])
        _need(gr.prod[pad].op_type == "Mul" and "Expand" in {gr.prod[x].op_type for x in _ancestors(gr, pad)},
              f"{sm_name}: all-valid padding mask")
        gr.remove(sl)
        shared_mask = shared_mask or band
        nm(add2_name).input[:] = [scores, shared_mask]
        nm(sm_name).input[0] = nm(add2_name).output[0]
        gr.remove([nm(div_name), nm(add3_name)])
        stats["fine attention"] += 1
    for sig_name in [n.name for n in gr.g.node if n.op_type == "Sigmoid" and n.name not in fp32_nodes]:
        sig = nm(sig_name)
        mul = gr.only_consumer(sig.output[0], "Mul")
        _need(mul is not None and set(mul.input) == {sig.input[0], sig.output[0]}, f"{sig_name}: SiLU")
        gr.g.node.insert(list(gr.g.node).index(mul), H.make_node(
            "QuickGelu", [sig.input[0]], [mul.output[0]], name=f"{sig_name}_silu", domain="com.microsoft", alpha=1.0))
        gr.remove([sig, nm(mul.name)])
        stats["QuickGelu"] += 1
    stats["dead nodes removed"] += dce(model)
    gr.refresh()
    for mul_name in [n.name for n in gr.g.node if n.op_type == "Mul"]:
        mul = nm(mul_name)
        gname = next((i for i in mul.input if i in gr.init), None)
        add = gname and next((gr.prod[i] for i in mul.input if i != gname and i in gr.prod), None)
        mm = add is not None and add.op_type == "Add" and next(
            (gr.prod[i] for i in add.input if i in gr.prod and gr.prod[i].op_type == "MatMul"), None)
        if not mm or len(gr.cons[add.output[0]]) != 1:
            continue
        bname = next(i for i in add.input if i != mm.output[0])
        gamma, w, bvec = gr.value(gname), gr.value(mm.input[1]), gr.value(bname)
        if (gamma is None or w is None or bvec is None or gamma.shape != (w.shape[1],)
                or len(gr.cons[mm.input[1]]) != 1 or len(gr.cons[bname]) != 1):
            continue
        g32 = gamma.astype(np.float32)
        gr.init[mm.input[1]].CopyFrom(NH.from_array((w.astype(np.float32) * g32).astype(w.dtype), mm.input[1]))
        gr.init[bname].CopyFrom(NH.from_array((bvec.astype(np.float32) * g32).astype(bvec.dtype), bname))
        for n in gr.cons[mul.output[0]]:
            n.input[:] = [add.output[0] if i == mul.output[0] else i for i in n.input]
        gr.remove([mul])
        stats["ConvNeXt gamma folded"] += 1
    depthwise = sum(1 for n in gr.g.node if n.op_type == "Conv" and
                    next((a.i for a in n.attribute if a.name == "group"), 1) > 1)
    stats["SkipLayerNormalization"] += _fuse_skip_layernorm(gr, fp32_nodes)
    stats["iSTFT"] += _fuse_istft(gr)
    stats["dead nodes removed"] += dce(model)
    blocks = stats["fine attention"]
    _need(stats["QuickGelu"] == blocks and stats["ConvNeXt gamma folded"] == depthwise > 0 and
          stats["SkipLayerNormalization"] == 2 * blocks and stats["iSTFT"] == 1,
          f"vocoder counts {dict(stats)} (depthwise ConvNeXt convs {depthwise})")
    _add_ms_domain(model)
    return stats


def _ancestors(gr: Graph, name: str) -> list[str]:
    out, frontier, seen = [], [name], set()
    while frontier:
        value = frontier.pop()
        node = gr.prod.get(value)
        if node is not None and value not in seen:
            seen.add(value)
            out.append(value)
            frontier.extend(node.input)
    return out


def sensitive_vocoder_nodes(model: onnx.ModelProto) -> set[str]:
    gr = Graph(model)
    candidates = [(node, consumer) for node in gr.g.node if node.op_type == "Greater" and gr.is_scalar(node.input[1], 0)
                  for consumer in gr.cons.get(node.output[0], []) if consumer.op_type == "Where"
                  and gr.is_scalar(consumer.input[1], 1) and gr.is_scalar(consumer.input[2], -1)]
    _need(len(candidates) == 1, "one hard sign quantizer with values +1/-1")
    sign, bits = candidates[0]
    return {gr.prod[value].name for value in _ancestors(gr, sign.input[0])} | {sign.name, bits.name}


def _fuse_skip_layernorm(gr: Graph, fp32_nodes: set[str]) -> int:
    gr.refresh()
    outs = {o.name for o in gr.g.output}
    fused = 0
    for ln_name in [n.name for n in gr.g.node if n.op_type == "LayerNormalization" and n.name not in fp32_nodes]:
        ln = gr.node(ln_name)
        add = gr.prod.get(ln.input[0])
        if add is None or add.op_type != "Add" or add.output[0] in outs or len(ln.input) < 3:
            continue
        if any(i in gr.init or gr.prod.get(i) is None or gr.prod[i].op_type == "Unsqueeze" for i in add.input):
            continue
        eps = next((a.f for a in ln.attribute if a.name == "epsilon"), 1e-5)
        new = H.make_node("SkipLayerNormalization", [add.input[0], add.input[1], ln.input[1], ln.input[2]],
                          [ln.output[0], "", "", add.output[0]], name=f"{ln_name}_skip", domain="com.microsoft",
                          epsilon=eps)
        drop = {add.name, ln_name}
        # At the Add, not the norm: its residual output may be read by nodes placed in between.
        gr.g.node.insert(list(gr.g.node).index(add), new)
        gr.remove([n for n in gr.g.node if n.name in drop])
        fused += 1
    return fused


def _fuse_istft(gr: Graph) -> int:
    candidates = []
    for node in gr.g.node:
        if node.op_type != "Add" or len(node.input) != 2:
            continue
        pair = [gr.prod.get(value) for value in node.input]
        if not all(parent is not None and parent.op_type == "MatMul" for parent in pair):
            continue
        bases = [gr.value(parent.input[0]) for parent in pair]
        if (not all(basis is not None and basis.ndim == 2 for basis in bases)
                or bases[0].shape != bases[1].shape or bases[0].shape[1] != bases[0].shape[0] // 2 + 1):
            continue
        if _walk(gr, node.output[0], "Col2Im", up=False) is not None:
            candidates.append((node, pair, bases))
    _need(len(candidates) == 1, "one inverse-rfft basis MatMul pair feeding overlap-add")
    add, pair, bases = candidates[0]
    n_fft = bases[0].shape[0]
    mirror = np.arange(1, n_fft // 2)
    even = [np.allclose(basis[n_fft - mirror], basis[mirror], atol=1e-9) for basis in bases]
    _need(sum(even) == 1, "one even cosine basis and one odd sine basis")
    cosine = even.index(True)
    mre, mim = pair[cosine], pair[1 - cosine]
    cb, sb = bases[cosine], bases[1 - cosine]
    n_fft = cb.shape[0]
    half = n_fft // 2
    t = np.arange(1, half)
    _need(n_fft % 2 == 0 and np.allclose(cb[n_fft - t], cb[t], atol=1e-9) and
          np.allclose(sb[n_fft - t], -sb[t], atol=1e-9), "iSTFT basis mirror symmetry")
    _const(gr, "vc_axis0", [0], np.int64)
    _const(gr, "vc_cos_half", np.ascontiguousarray(cb[:half + 1]), cb.dtype)
    _const(gr, "vc_sin_half", np.ascontiguousarray(sb[:half + 1]), sb.dtype)
    _const(gr, "vc_rev_start", [half - 1], np.int64)
    _const(gr, "vc_rev_end", [0], np.int64)
    _const(gr, "vc_rev_step", [-1], np.int64)
    pos = list(gr.g.node).index(add)
    new = [
        H.make_node("MatMul", ["vc_cos_half", mre.input[1]], ["vc_p"], name="vc_istft_cos"),
        H.make_node("MatMul", ["vc_sin_half", mim.input[1]], ["vc_q"], name="vc_istft_sin"),
        H.make_node("Add", ["vc_p", "vc_q"], ["vc_top"], name="vc_istft_top"),
        H.make_node("Sub", ["vc_p", "vc_q"], ["vc_mir"], name="vc_istft_mirror"),
        H.make_node("Slice", ["vc_mir", "vc_rev_start", "vc_rev_end", "vc_axis0", "vc_rev_step"], ["vc_bot"],
                    name="vc_istft_reverse"),
        H.make_node("Concat", ["vc_top", "vc_bot"], [add.output[0]], name="vc_istft_frames", axis=0),
    ]
    for i, n in enumerate(new):
        gr.g.node.insert(pos + i, n)
    gr.remove([mre, mim, add])
    return 1


def _add_ms_domain(model: onnx.ModelProto) -> None:
    if not any(o.domain == "com.microsoft" for o in model.opset_import):
        model.opset_import.append(H.make_opsetid("com.microsoft", 1))


def downstream(model: onnx.ModelProto, roots: list[str]) -> list[str]:
    gr = Graph(model)
    todo, seen, names = list(roots), set(), []
    while todo:
        for node in gr.cons.get(todo.pop(), []):
            if id(node) not in seen:
                seen.add(id(node))
                names.append(node.name)
                todo.extend(node.output)
    return names


def to_fp16(model: onnx.ModelProto, fp32_nodes: set[str], fp32_io: list[str]) -> onnx.ModelProto:
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    info = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}
    is_float = lambda t: t in info and info[t].type.tensor_type.elem_type in (TP.FLOAT, TP.FLOAT16)
    inits = {i.name: i for i in g.initializer}
    fp32_local_names = set()
    subgraphs = [attribute.g for node in g.node for attribute in node.attribute
                 if attribute.type == onnx.AttributeProto.GRAPH]
    while subgraphs:
        subgraph = subgraphs.pop()
        fp32_local_names.update(tensor.name for tensor in subgraph.initializer if tensor.data_type == TP.FLOAT)
        fp32_local_names.update(node.output[0] for node in subgraph.node if node.op_type == "Constant"
                                and node.attribute[0].type == onnx.AttributeProto.TENSOR
                                and node.attribute[0].t.data_type == TP.FLOAT)
        subgraphs.extend(attribute.g for node in subgraph.node for attribute in node.attribute
                         if attribute.type == onnx.AttributeProto.GRAPH)
    used_as = {}
    for n in g.node:
        region = "f32" if n.name in fp32_nodes else "f16"
        for i in export_hojo.node_inputs(n):
            if i in inits and inits[i].data_type == TP.FLOAT:
                used_as.setdefault(i, set()).add(region)
    for name, regions in used_as.items():
        t = inits[name]
        a = NH.to_array(t)
        if "f16" in regions:
            h = np.clip(a, -65504.0, 65504.0).astype(np.float16)
            if "f32" in regions or name in fp32_local_names:
                # Slim may hoist branch constants: never let an FP16 outer name shadow their FP32 values.
                g.initializer.append(NH.from_array(h, name + "_f16"))
                for n in g.node:
                    if n.name not in fp32_nodes:
                        export_hojo.remap_node_inputs(n, {name: name + "_f16"})
            else:
                t.CopyFrom(NH.from_array(h, name))
    dtype: dict[str, int] = {}
    for v in g.input:
        if v.type.tensor_type.elem_type in (TP.FLOAT, TP.FLOAT16):
            dtype[v.name] = TP.FLOAT if v.name in fp32_io else TP.FLOAT16
    for i in g.initializer:
        if i.data_type in (TP.FLOAT, TP.FLOAT16):
            dtype[i.name] = i.data_type
    for n in g.node:
        for o in n.output:
            if is_float(o) or n.op_type == "SimplifiedLayerNormalization":
                dtype[o] = TP.FLOAT if n.name in fp32_nodes else TP.FLOAT16
    for n in g.node:
        if n.op_type == "Cast":
            to = next(a for a in n.attribute if a.name == "to")
            if to.i in (TP.FLOAT, TP.FLOAT16):
                to.i = TP.FLOAT if n.name in fp32_nodes else TP.FLOAT16
                dtype[n.output[0]] = to.i
        if n.op_type in ("Constant", "ConstantOfShape"):
            for a in n.attribute:
                if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TP.FLOAT and n.name not in fp32_nodes:
                    a.t.CopyFrom(NH.from_array(NH.to_array(a.t).astype(np.float16), a.t.name))
    outputs = {o.name for o in g.output}
    remap: dict[str, str] = {}
    casts: dict[tuple[str, int], str] = {}
    new_nodes: list[onnx.NodeProto] = []
    for n in g.node:
        export_hojo.remap_node_inputs(n, remap)
        want = TP.FLOAT if n.name in fp32_nodes else TP.FLOAT16
        if n.op_type == "Cast":
            to = next(a.i for a in n.attribute if a.name == "to")
            if dtype.get(n.input[0]) == to and n.output[0] not in outputs:
                remap[n.output[0]] = n.input[0]
                continue
        elif n.op_type != "Shape":
            for i in sorted(export_hojo.node_inputs(n)):
                have = dtype.get(i)
                if have is None or have == want or i in inits and inits[i].data_type == want:
                    continue
                key = (i, want)
                if key not in casts:
                    casts[key] = f"{i}_as_{'f32' if want == TP.FLOAT else 'f16'}"
                    new_nodes.append(H.make_node("Cast", [i], [casts[key]], to=want, name=casts[key] + "_cast"))
                export_hojo.remap_node_inputs(n, {i: casts[key]})
        new_nodes.append(n)
    for o in g.output:
        if o.type.tensor_type.elem_type not in (TP.FLOAT, TP.FLOAT16):
            continue
        want = TP.FLOAT if o.name in fp32_io else TP.FLOAT16
        if dtype.get(o.name) != want:
            src = o.name + "_pre"
            for n in new_nodes:
                n.output[:] = [src if x == o.name else x for x in n.output]
                n.input[:] = [src if x == o.name else x for x in n.input]
            new_nodes.append(H.make_node("Cast", [src], [o.name], to=want, name=o.name + "_out_cast"))
        o.type.tensor_type.elem_type = want
    for v in g.input:
        if v.type.tensor_type.elem_type in (TP.FLOAT, TP.FLOAT16):
            v.type.tensor_type.elem_type = dtype[v.name]
    del g.node[:]
    g.node.extend(new_nodes)
    del g.value_info[:]
    return model


def name_nodes(model: onnx.ModelProto, tag: str) -> None:
    for i, node in enumerate(model.graph.node):
        if not node.name:
            node.name = f"{tag}_{node.op_type}_{i}"


def stream_fp32_region(model: onnx.ModelProto) -> set[str]:
    """Decoder-stream nodes kept float32: everything except the backbone/head and ConvNeXt state ancestors."""
    gr = Graph(model)
    exp = next((node for node in gr.g.node if node.op_type == "Exp"), None)
    phase = gr.prod.get("phase")
    _need(exp is not None and phase is not None, "decoder stream Exp and phase producer")
    states = [value.name for value in gr.g.output if value.name.startswith("conv_present.")]
    frontier, half, seen = [exp.input[0], phase.input[0], *states], set(), set()
    while frontier:
        value = frontier.pop()
        node = gr.prod.get(value)
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        half.add(node.name)
        frontier.extend(export_hojo.node_inputs(node))
    return {node.name for node in gr.g.node if node.name not in half}


def stream_graph_to_fp16(model: onnx.ModelProto, fp32_nodes: set[str] | None = None) -> onnx.ModelProto:
    """Convert stream compute and floating I/O to float16 unless KEEP_IO_TYPE preserves the source I/O."""
    from onnxruntime.transformers.float16 import convert_float_to_float16

    converted = convert_float_to_float16(
        model,
        min_positive_val=OptimizerConfig.f16_min_positive_val,
        max_finite_val=OptimizerConfig.f16_max_finite_val,
        keep_io_types=KEEP_IO_TYPE,
        disable_shape_infer=False,
        node_block_list=sorted(fp32_nodes or ()),
        force_fp16_initializers=False,
    )
    graph = converted.graph
    # The converter appends its boundary Casts after their consumers; restore topological order, keeping node order stable.
    ready = {value.name for value in graph.input} | {tensor.name for tensor in graph.initializer} | {""}
    pending, ordered = list(graph.node), []
    while pending:
        later = []
        for node in pending:
            if export_hojo.node_inputs(node) <= ready:
                ordered.append(node)
                ready.update(node.output)
            else:
                later.append(node)
        if len(later) == len(pending):
            raise RuntimeError(f"Cannot order F16 stream graph; unresolved input of {later[0].name}")
        pending = later
    del graph.node[:]
    graph.node.extend(ordered)
    _drop_fp16_roundtrips(converted)
    return converted


def _drop_fp16_roundtrips(model: onnx.ModelProto) -> int:
    """Remove float32 -> float16 -> float32 Cast pairs the converter leaves between consecutive float32 nodes."""
    inferred = onnx.shape_inference.infer_shapes(model).graph
    types = {value.name: value.type.tensor_type.elem_type
             for value in list(inferred.input) + list(inferred.output) + list(inferred.value_info)}
    types.update({tensor.name: tensor.data_type for tensor in inferred.initializer})
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    outputs = {value.name for value in graph.output}
    cast_to = lambda node: next(attribute.i for attribute in node.attribute if attribute.name == "to")
    remap, drop, changed = {}, set(), set()
    for node in graph.node:
        if node.op_type != "Cast" or cast_to(node) != TP.FLOAT:
            continue
        half = producers.get(node.input[0])
        if not (half is not None and half.op_type == "Cast" and cast_to(half) == TP.FLOAT16
                and types.get(half.input[0]) == TP.FLOAT):
            continue
        if node.output[0] in outputs:
            node.op_type = "Identity"
            del node.attribute[:]
            node.input[0] = half.input[0]
            changed.add(node.name)
        else:
            remap[node.output[0]] = half.input[0]
            drop.add(node.name)
    if not drop and not changed:
        return 0
    kept = [node for node in graph.node if node.name not in drop]
    for node in kept:
        export_hojo.remap_node_inputs(node, remap)
    del graph.node[:]
    graph.node.extend(kept)
    del graph.value_info[:]
    dce(model)
    return len(drop)


COPY_OPS = ("Concat", "Pad", "Tile", "Expand", "Transpose", "ScatterND", "ScatterElements")
LAYOUT_OPS = ("Transpose", "Reshape", "Squeeze", "Unsqueeze", "Flatten")
SHAPE_OPS = ("Shape", "Range", "ConstantOfShape")
ELEMENTWISE_OPS = ("Add", "Sub", "Mul", "Div", "Neg", "Pow", "Where", "Sqrt", "Sigmoid", "Erf", "Equal",
                   "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Min", "Max", "Clip")


def metrics(model: onnx.ModelProto) -> dict:
    g = model.graph
    ops = Counter(n.op_type for n in g.node)
    cnt = lambda names: sum(ops[o] for o in names)
    return {"nodes": len(g.node), "init": len(g.initializer),
            "init_MB": round(sum(len(i.raw_data) or NH.to_array(i).nbytes for i in g.initializer) / 2 ** 20, 2),
            "Cast": ops["Cast"], "copy": cnt(COPY_OPS), "layout": cnt(LAYOUT_OPS), "shape": cnt(SHAPE_OPS),
            "elementwise": cnt(ELEMENTWISE_OPS), "MatMul": ops["MatMul"], "ops": dict(sorted(ops.items()))}


def report(tag: str, model: onnx.ModelProto) -> None:
    m = metrics(model)
    print(f"{tag}: " + " ".join(f"{k}={v}" for k, v in m.items() if k != "ops"))
    print(f"  ops {m['ops']}")


def io_report(model: onnx.ModelProto) -> None:
    g = model.graph
    print("  io", [(v.name, TP.DataType.Name(v.type.tensor_type.elem_type)) for v in list(g.input) + list(g.output)
                   if "." not in v.name or v.name.startswith(("past_key_values.0.", "present.0."))])


def _check_if_subgraphs(graph: onnx.GraphProto, name: str, bad: list[str]) -> None:
    for node in graph.node:
        if node.op_type == "If":
            branches = {attr.name: attr.g for attr in node.attribute if attr.HasField("g")}
            if set(branches) != {"then_branch", "else_branch"}:
                bad.append(f"{name}: If node {node.name} missing branch subgraphs")
            else:
                t_outs = [o.type.tensor_type.elem_type for o in branches["then_branch"].output]
                e_outs = [o.type.tensor_type.elem_type for o in branches["else_branch"].output]
                if t_outs != e_outs:
                    bad.append(f"{name}: If branch output dtype mismatch on {node.name}: {t_outs} != {e_outs}")
        for sg in iter_subgraphs(node):
            _check_if_subgraphs(sg, name, bad)


def _check_lm_fp32_stabilization(name: str, g: onnx.GraphProto, layers: int, props: dict[str, str], bad: list[str]) -> None:
    init_names = {t.name for t in g.initializer}
    fp32_tensors: set[str] = {
        v.name for v in (*g.input, *g.value_info) if v.type.tensor_type.elem_type == TP.FLOAT
    } | {t.name for t in g.initializer if t.data_type == TP.FLOAT}
    producers = {out: node for node in g.node for out in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in g.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
    progressed = True
    while progressed:
        progressed = False
        for node in g.node:
            is_fp32 = (
                (node.op_type == "Cast" and any(a.name == "to" and a.i == TP.FLOAT for a in node.attribute))
                or (node.op_type != "Cast" and node.input and all(inp in fp32_tensors for inp in node.input if inp))
            )
            if is_fp32:
                for out in node.output:
                    if out and out not in fp32_tensors:
                        fp32_tensors.add(out)
                        progressed = True
    attn_mm = weighted_mm = softmax_cnt = qgelu_cnt = 0
    is_mqa_decode = props.get("lm_attention_layout") == "flattened_mqa"
    for node in g.node:
        if node.op_type == "MatMul":
            if any(inp in init_names for inp in node.input):
                weighted_mm += 1
                if any(inp in fp32_tensors for inp in node.input):
                    bad.append(f"{name}: weighted MatMul {node.name} must remain FLOAT16")
            else:
                attn_mm += 1
                if not all(inp in fp32_tensors for inp in node.input):
                    bad.append(f"{name}: non-initializer attention MatMul {node.name} is not in FP32")
        elif node.op_type == "Softmax":
            softmax_cnt += 1
            if node.input[0] not in fp32_tensors:
                bad.append(f"{name}: Softmax {node.name} is not in FP32")
            if is_mqa_decode:
                up = producers.get(node.input[0])
                up_src = producers.get(up.input[0]) if up is not None and up.op_type == "Cast" else up
                down_nodes = consumers.get(node.output[0], [])
                down_cast = down_nodes[0] if len(down_nodes) == 1 and down_nodes[0].op_type == "Cast" else None
                down_after = consumers.get(down_cast.output[0], []) if down_cast is not None else down_nodes
                if (up_src is None or up_src.op_type != "Reshape" or "r2_mqa_score_shape" not in up_src.input
                        or not any(n.op_type == "Reshape" and "r2_mqa_prob_shape" in n.input for n in down_after)):
                    bad.append(f"{name}: MQA Reshape barriers around {node.name} were stripped")
        elif node.op_type == "QuickGelu":
            qgelu_cnt += 1
            if node.input[0] not in fp32_tensors:
                bad.append(f"{name}: QuickGelu {node.name} is not in FP32")
        elif node.op_type == "Add" and node.name.endswith("_rope_add"):
            if any(inp in fp32_tensors for inp in node.input):
                bad.append(f"{name}: {node.name} must remain FLOAT16 to preserve RoPE rounding")
    if (attn_mm, weighted_mm, softmax_cnt, qgelu_cnt) != (2 * layers, 4 * layers + 1, layers, layers):
        bad.append(
            f"{name}: unexpected FP32-stabilized LM topology attn_mm={attn_mm} weighted_mm={weighted_mm} "
            f"softmax={softmax_cnt} quickgelu={qgelu_cnt}"
        )


def check_graph(name: str, model: onnx.ModelProto, cfg: dict, expect: Counter | None = None,
                precision: str = "F16") -> list[str]:
    """Structural gates; `expect` holds vocoder targets counted on the fp32 source (Erf GELUs, ConvNeXt blocks)."""
    props = {entry.key: entry.value for entry in model.metadata_props}
    g = onnx.shape_inference.infer_shapes(model).graph
    ops = Counter(n.op_type for n in g.node)
    bad = []
    _check_if_subgraphs(g, name, bad)
    defined = {v.name for v in g.input} | {i.name for i in g.initializer} | {""}
    for n in g.node:
        late = [i for i in n.input if i not in defined]
        if late:
            bad.append(f"{name}: node {n.name} reads {late[0]} before it is produced (graph not topologically sorted)")
            break
        defined.update(n.output)
    infos = list(g.value_info) + list(g.input) + list(g.output)
    types = {value.name: value.type.tensor_type.elem_type for value in infos}
    types.update({tensor.name: tensor.data_type for tensor in g.initializer})
    for node in g.node:
        if node.op_type != "Cast":
            continue
        source = types.get(node.input[0])
        target = next(attribute.i for attribute in node.attribute if attribute.name == "to")
        if source == target:
            bad.append(f"{name}: redundant Cast at {node.name}")
        elif source in (TP.FLOAT, TP.FLOAT16) and target in (TP.FLOAT, TP.FLOAT16) and source != target:
            consumers = [child for child in g.node if node.output[0] in child.input]
            for child in consumers:
                if child.op_type == "Cast" and next(attribute.i for attribute in child.attribute if attribute.name == "to") == source:
                    # Intentional FP32->F16->FP32 mantissa quantization for LM stabilization or F16->F32 PCM playback.
                    if (node.name.startswith("z1_narrow_") and child.name.startswith("z1_widen_")) or (
                        child.name == "pcm_float_playback_cast" and source == TP.FLOAT and target == TP.FLOAT16
                    ):
                        continue
                    bad.append(f"{name}: round-trip floating Cast at {node.name}")
                    break
    if any(v.type.tensor_type.elem_type == TP.DOUBLE for v in infos) or \
            any(i.data_type == TP.DOUBLE for i in g.initializer):
        bad.append(f"{name}: float64 tensor present")
    if name in (LM, VOC):
        voice = next((v for v in g.input if v.name == "voice"), None)
        if (voice is not None) != bool(cfg["voices"]) or voice is not None and (
                voice.type.tensor_type.elem_type != TP.INT32 or voice.type.tensor_type.shape.dim
                or any(n.op_type != "Gather" for n in g.node if "voice" in n.input)):
            bad.append(f"{name}: preset graphs need a scalar int32 'voice' read only by Gather; clone graphs none")
    if name in (LM, "Hojo_LM_Decode.onnx"):
        layers = int(cfg["num_hidden_layers"])
        kv_ios = [v for v in list(g.input) + list(g.output) if v.name.startswith(("past_key_values.", "present."))]
        if not kv_ios or any(
            v.type.tensor_type.elem_type != TP.FLOAT16
            or not v.type.tensor_type.shape.dim
            or v.type.tensor_type.shape.dim[0].dim_value != 1
            for v in kv_ios
        ):
            bad.append(f"{name}: KV cache I/O is not float16 with static batch 1")
        elif precision.upper() != "F16" and name == LM:
            for v in g.output:
                if v.name.startswith("present."):
                    non_shape = [n for n in g.node if v.name in n.input and n.op_type != "Shape"]
                    if not non_shape or any(
                        n.op_type != "Cast" or next(a.i for a in n.attribute if a.name == "to") != TP.FLOAT
                        for n in non_shape
                    ):
                        bad.append(f"{name}: {v.name} must Cast back to float32 while computing")
                        break
        if props.get("lm_kv_update") == "inplaced_kv":
            kv_updates = [n for n in g.node if n.op_type == "KVCacheUpdate" and n.domain == "com.custom.llm"]
            if len(kv_updates) != 2 * layers:
                bad.append(f"{name}: expected {2 * layers} com.custom.llm::KVCacheUpdate ops, got {len(kv_updates)}")
            hist = next((v for v in g.input if v.name == "history_len"), None)
            if hist is None or hist.type.tensor_type.elem_type != TP.INT64 or [d.dim_value for d in hist.type.tensor_type.shape.dim] != [1]:
                bad.append(f"{name}: inplaced_kv requires static INT64[1] history_len input")
            expected_kv_shape = [1, int(cfg["num_key_value_heads"]), int(cfg["max_positions"]), int(cfg["head_dim"])]
            for v in kv_ios:
                if [d.dim_value for d in v.type.tensor_type.shape.dim] != expected_kv_shape:
                    bad.append(f"{name}: {v.name} shape must be static {expected_kv_shape}")
                    break
        if props.get("lm_fp32_attention_activations") == "1":
            _check_lm_fp32_stabilization(name, g, layers, props, bad)
    if name == "Hojo_Decoder_Stream.onnx" and props.get("streaming_vocoder_precision") == "F16":
        in_types = {v.name: v.type.tensor_type.elem_type for v in g.input}
        out_types = {v.name: v.type.tensor_type.elem_type for v in g.output}
        pcm_prods = [n for n in g.node if "pcm" in n.output]
        if (in_types.get("ola_past") != TP.FLOAT16 or out_types.get("ola_present") != TP.FLOAT16
                or out_types.get("pcm") != TP.FLOAT
                or len(pcm_prods) != 1 or pcm_prods[0].op_type != "Cast"
                or list(pcm_prods[0].input) != ["pcm_f16_before_playback_cast"]):
            bad.append(f"{name}: F16 streaming decoder must retain F16 OLA state and single F16->F32 pcm playback cast")
    if precision.upper() != "F16":
        return bad
    if name == LM:
        layers = int(cfg["num_hidden_layers"])
        norms = ops["RMSNormalization"] + ops["SimplifiedLayerNormalization"] + ops["SkipSimplifiedLayerNormalization"]
        if norms != 3 * layers + 1 or ops["SkipSimplifiedLayerNormalization"] != 2 * layers:
            bad.append(f"{name}: RMSNorm {norms} (skip {ops['SkipSimplifiedLayerNormalization']}) != "
                       f"{3 * layers + 1} ({2 * layers})")
        if ops["Transpose"] != 3 * layers:
            bad.append(f"{name}: {ops['Transpose']} Transpose, want {3 * layers} (q, new K row, attention out)")
        if ops["Expand"]:
            bad.append(f"{name}: {ops['Expand']} Expand left (repeat_kv KV copies)")
        splits = 4 * layers
        if ops["QuickGelu"] != layers or ops["Split"] != splits:
            bad.append(f"{name}: {ops['QuickGelu']} QuickGelu / {ops['Split']} Split, want {layers} / {splits}")
        if ops["Pow"] or ops["Neg"] or ops["Cos"] or ops["Sin"] or ops["Sigmoid"]:
            bad.append(f"{name}: unfused RMSNorm/rotary/SiLU ops remain (Pow/Neg/Cos/Sin/Sigmoid)")
    elif name == VOC:
        if expect is not None and (ops["Erf"] or ops["Gelu"] != expect["Erf"]):
            bad.append(f"{name}: GELU not fused ({ops['Gelu']} Gelu, {ops['Erf']} Erf, want {expect['Erf']})")
        if expect is not None and ops["SkipLayerNormalization"] != expect["ConvNeXt"]:
            bad.append(f"{name}: Vocos/fine fusions incomplete (Sigmoid {ops['Sigmoid']}, Slice {ops['Slice']}, "
                       f"Div {ops['Div']}, SkipLayerNormalization {ops['SkipLayerNormalization']}, want {dict(expect)})")
    return bad


def convert(src: Path, name: str, cfg: dict) -> tuple[bytes, list[str]]:
    model = inline_model_initializers(src / name)
    source_meta = {p.key: p.value for p in model.metadata_props}
    report(f"{name} fp32", model)
    is_lm = name == LM
    if is_lm:
        export_hojo.ensure_f16_kv_io(model)
    model = onnxslim.slim(model)
    if is_lm:
        model = onnx.shape_inference.infer_shapes(model)
        fused = fuse_rmsnorm(model)
        negs = bake_rotary(model, cfg)
        print(f"  fused RMSNorm: {fused}; Neg removed: {negs}")
        model = onnxslim.slim(model)
    name_nodes(model, Path(name).stem)
    report(f"{name} slim", model)
    fp32_io = list(FP32_IO)
    fine_fp32 = set()
    if is_lm:
        gr = Graph(model)
        head = next(node for node in gr.cons["hidden"] if node.op_type == "MatMul")
        block = downstream(model, list(head.output) + ["ctrl", "uniform"])
    else:
        fine_fp32 = sensitive_vocoder_nodes(model)
        block = set(downstream(model, ["mag", "phase"])) | fine_fp32
        if source_meta["tts_mode"] == "voice_clone":
            gr = Graph(model)
            block.update(gr.prod[value].name for root in ("mag", "phase") for value in _ancestors(gr, root))
        fp32_io.append("hidden_states")
    model = to_fp16(model, set(block), fp32_io)
    export_hojo.remove_redundant_casts(model)
    model = onnxslim.slim(model)
    report(f"{name} fp16", model)
    if is_lm:
        print(f"  LM fusions: {dict(fuse_lm(model, cfg))}")
        print(f"  Audio8 attention: {dict(fuse_attention_audio8(model, cfg))}")
        expect = None
    else:
        expect = Counter(Erf=fuse_gelu_fp16(model))
        stats = fuse_vocoder_fp16(model, fine_fp32 if source_meta["tts_mode"] == "voice_clone" else set())
        expect.update({"fine attention": stats["fine attention"], "ConvNeXt": stats["SkipLayerNormalization"]})
        print(f"  fp16 fusions: Gelu {expect['Erf']}")
        print(f"  Vocos/fine fusions: {dict(stats)}")
    if is_lm:
        standardize_rmsnorm(model)
    export_hojo.int32_gather_indices(model)
    del model.graph.value_info[:]
    if source_meta:
        H.set_model_props(model, source_meta)
    report(f"{name} final", model)
    io_report(model)
    data = model.SerializeToString()
    onnx.checker.check_model(model)
    ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    return data, check_graph(name, model, cfg, expect)


def reference_token(p: np.ndarray, top_p: float, u: float) -> int:
    p = p.astype(np.float32)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    excl = np.concatenate([[0.0], np.cumsum(ps)[:-1]]).astype(np.float32)
    keep = np.zeros(p.size, np.bool_)
    keep[order] = excl <= top_p
    pk = np.where(keep, p, np.float32(0))
    cdf = np.concatenate([[0.0], np.cumsum(pk)[:-1]]).astype(np.float32)
    cand = np.flatnonzero((cdf <= np.float32(u) * pk.sum(dtype=np.float32)) & (pk > 0))
    return int(cand[-1])


def sampler_model(model: onnx.ModelProto) -> onnx.ModelProto:
    model = onnx.ModelProto.FromString(model.SerializeToString())
    name_nodes(model, "sampler")
    gr = Graph(model)
    names = set(downstream(model, ["hidden", "ctrl", "uniform", "seen"]))
    hidden = next(value for value in gr.g.output if value.name == "hidden")
    token = next(value for value in gr.g.output if value.name == "token")
    softmax = next((node for node in gr.g.node if node.op_type == "Softmax" and node.name in names), None)
    if softmax is None:
        choice = next(node for node in gr.g.node if node.op_type == "If" and node.name in names)
        branches = {attribute.name: attribute.g for attribute in choice.attribute}
        sampled, greedy = branches["else_branch"], branches["then_branch"]
        softmax = next(node for node in sampled.node if node.op_type == "Softmax")
        # Diagnostic-only outputs expose the actual branch probabilities, including on greedy verification runs.
        until_softmax = list(sampled.node)[:list(sampled.node).index(softmax) + 1]
        greedy.node.extend(until_softmax)
        existing = {tensor.name for tensor in greedy.initializer}
        greedy.initializer.extend(tensor for tensor in sampled.initializer if tensor.name not in existing)
        for branch in (sampled, greedy):
            branch.output.extend([
                H.make_tensor_value_info(softmax.input[0], TP.FLOAT, [None]),
                H.make_tensor_value_info(softmax.output[0], TP.FLOAT, [None]),
            ])
        choice.output.extend(["sampler_scaled_logits", "sampler_probabilities"])
        scaled_name, probability_name = choice.output[-2:]
    else:
        scaled_name, probability_name = softmax.input[0], softmax.output[0]
    scaled_logits = H.make_tensor_value_info(scaled_name, TP.FLOAT, [None])
    probabilities = H.make_tensor_value_info(probability_name, TP.FLOAT, [None])
    producers = {output: node for node in gr.g.node for output in node.output}
    frontier, needed = ["token", scaled_name, probability_name], set()
    boundaries = {"hidden", "ctrl", "uniform", "seen"}
    while frontier:
        value = frontier.pop()
        if value in boundaries or value in needed:
            continue
        needed.add(value)
        producer = producers.get(value)
        if producer is not None:
            frontier.extend(export_hojo.node_inputs(producer))
    nodes = [node for node in gr.g.node if any(output in needed for output in node.output)]
    inputs = [value for value in gr.g.input if value.name in ("ctrl", "uniform", "seen")]
    del gr.g.node[:]
    gr.g.node.extend(nodes)
    del gr.g.input[:]
    gr.g.input.extend([hidden, *inputs])
    del gr.g.output[:]
    gr.g.output.extend([token, scaled_logits, probabilities])
    del gr.g.value_info[:]
    dce(model)
    return model


def _verify_inplaced_kv_offline(
    fp32: Path,
    target_dir: Path,
    loaded: dict[str, onnx.ModelProto],
    *,
    advanced: bool,
    min_snr_db: float,
    exact_fine_bits: bool,
) -> list[str]:
    """Run RNG + streaming vocoder parity checks when inplaced_kv custom ops library is unavailable."""
    so_base, _, disabled = build_session_options(False)
    so_base.log_severity_level = 3
    load_bytes = lambda data: ort.InferenceSession(data, so_base, providers=["CPUExecutionProvider"], disabled_optimizers=disabled)
    failed = _verify_counter_rng(loaded[LM], load_bytes)
    if "Hojo_LM_Decode.onnx" in loaded:
        failed += _verify_counter_rng(loaded["Hojo_LM_Decode.onnx"], load_bytes)
    if not advanced or failed:
        return failed
    from Advanced_Graphs import DECODER_NAME, FINE_NAME, make_fine_reference
    voc32_m = inline_model_initializers(fp32 / VOC)
    meta = parse_metadata({p.key: p.value for p in voc32_m.metadata_props})
    if meta["tts_mode"] != "preset_voices":
        return failed
    voc32 = load_bytes(voc32_m.SerializeToString())
    source_fine = load_bytes(make_fine_reference(voc32_m).SerializeToString())
    shared_path = target_dir / SHARED_MODEL_NAME
    store = SharedInitializerStore(shared_path, respect_prepack_exclusions=True) if shared_path.is_file() else None

    def load_target(name: str) -> ort.InferenceSession:
        so, _, dis = build_session_options(False)
        so.log_severity_level = 3
        path = target_dir / name
        if store is not None:
            store.attach_to(so, model_path=path)
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"], disabled_optimizers=dis)

    voice = np.array(0, np.int32)
    stateful = StatefulVocoder(
        load_target(FINE_NAME), load_target(DECODER_NAME), ort.RunOptions(), ort.OrtValue.ortvalue_from_numpy(voice),
    )
    rng = np.random.RandomState(1234)
    frames = 64
    hid = rng.randn(1, frames, meta["hidden_size"]).astype(np.float32) * np.float32(0.25)
    codes = rng.randint(meta["audio_token_first"], meta["audio_token_last"] + 1, size=(1, frames), dtype=np.int32)
    voc_feed = {"hidden_states": hid, "audio_tokens": codes, "voice": voice}
    w32 = voc32.run(None, voc_feed)[0].astype(np.float32)
    source_bits = source_fine.run(["bits"], voc_feed)[0]
    failed += _verify_stateful_wave(stateful, hid, codes, w32, source_bits, min_snr_db, exact_fine_bits)
    stateful.close()
    return failed


def verify(fp32: Path, target_dir: Path, tokenizer: Path, *,
           min_greedy: float = MIN_GREEDY_AGREEMENT, max_hidden_error: float = MAX_HIDDEN_REL_ERR,
           min_snr_db: float = MIN_VOCODER_SNR_DB, precision: str = "F16",
           advanced_folder: Path | None = None, exact_fine_bits: bool = True,
           custom_ops_library: Path | None = None) -> list[str]:
    so_base, _, disabled = build_session_options(False)
    so_base.log_severity_level = 3
    if custom_ops_library is not None:
        so_base.register_custom_ops_library(str(custom_ops_library))
    load_bytes = lambda data: ort.InferenceSession(data, so_base, providers=["CPUExecutionProvider"], disabled_optimizers=disabled)
    shared_path = target_dir / SHARED_MODEL_NAME
    shared_store = SharedInitializerStore(shared_path, respect_prepack_exclusions=True) if shared_path.is_file() else None

    def load_target(name: str, *, max_threads: int = 0) -> ort.InferenceSession:
        so, _, dis = build_session_options(False, max_threads=max_threads)
        so.log_severity_level = 3
        if custom_ops_library is not None:
            so.register_custom_ops_library(str(custom_ops_library))
        path = target_dir / name
        if shared_store is not None:
            shared_store.attach_to(so, model_path=path)
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"], disabled_optimizers=dis)

    lm32_m = inline_model_initializers(fp32 / LM)
    voc32_m = inline_model_initializers(fp32 / VOC)
    lm32, voc32 = load_bytes(lm32_m.SerializeToString()), load_bytes(voc32_m.SerializeToString())
    lm16 = load_target(LM)
    voc16 = load_target(VOC) if (target_dir / VOC).is_file() else None
    decode_name = "Hojo_LM_Decode.onnx"
    lm_decode = load_target(decode_name) if (target_dir / decode_name).is_file() else None
    meta = parse_metadata(lm32.get_modelmeta().custom_metadata_map)
    tts_mode = meta["tts_mode"]
    tok = BpeTokenizer(tokenizer)
    voice = None
    if meta["voices"]:
        try:
            voice = np.array(resolve_voice_index(VOICE_ID, *voice_lookup(meta["voices"])), np.int32)
        except ValueError as exc:
            print(f"  verify: {exc}; using voice 0")
            voice = np.array(0, np.int32)
    projected_speaker = None
    if tts_mode == "preset_voices":
        prefix, suffix = prompt_parts(meta, tok)
        spk_vec_fixed = None
    else:
        ref_sess = load_bytes(inline_model_initializers(fp32 / REF_PRE).SerializeToString())
        arg = ref_sess.get_inputs()[0]
        audio = load_prompt_audio(PROMPT_AUDIO_PATH, meta["in_sample_rate"], arg)
        ref_codes_arr, spk_vec_fixed = ref_sess.run(["prompt_audio_ids", "speaker_embedding"], {arg.name: audio})
        spk_vec_fixed = spk_vec_fixed.astype(np.float32)
        prefix, suffix = prompt_parts(meta, tok, normalize_text(PROMPT_TEXT), ref_codes_arr)
        if (target_dir / REF_PRE).is_file():
            ref_target = load_target(REF_PRE)
            target_arg = ref_target.get_inputs()[0]
            reference_outputs = ["prompt_audio_ids", "speaker_embedding"]
            projection = ref_target.get_modelmeta().custom_metadata_map.get("reference_speaker_projection_output")
            if projection:
                reference_outputs.append(projection)
            reference_values = ref_target.run(reference_outputs, {
                target_arg.name: np.asarray(audio, dtype=io_dtype(target_arg)),
            })
            target_codes, target_speaker = reference_values[:2]
            if projection:
                projected_speaker = ort.OrtValue.ortvalue_from_numpy(reference_values[2])
            speaker_error = float(np.max(np.abs(spk_vec_fixed - target_speaker)))
            codes_match = float(np.mean(ref_codes_arr == target_codes)) if ref_codes_arr.shape == target_codes.shape else 0.0
            print(f"  verify reference: prompt_audio_ids agreement {codes_match:.3f}, "
                  f"speaker embedding max abs diff {speaker_error:.9g}")
            if ref_codes_arr.shape != target_codes.shape or not np.isfinite(target_speaker).all():
                return ["reference preprocess produced invalid outputs"]
            if precision.upper() == "F32" and not np.array_equal(ref_codes_arr, target_codes):
                return ["reference preprocess changed prompt_audio_ids"]
    past = [i.name for i in lm32.get_inputs() if i.name.startswith("past_key_values.")]
    names = [o.name for o in lm32.get_outputs()]
    greedy = np.array([0.0, 0.95, 1.1], np.float32)
    sampling = np.array([0.8, 0.95, 1.1], np.float32)
    from Advanced_Graphs import uniform_sampler_view

    # Pipeline v2 draws `uniform` in-graph; the diagnostic view re-exposes it so draws can be checked against numpy.
    lm16_inlined = inline_model_initializers(target_dir / LM)
    samplers = [load_bytes(sampler_model(uniform_sampler_view(model)).SerializeToString()) for model in
                (lm32_m, lm16_inlined)]

    rng_failures = _verify_counter_rng(lm16_inlined, load_bytes)
    del lm16_inlined
    if (target_dir / decode_name).is_file():
        rng_failures += _verify_counter_rng(onnx.load(str(target_dir / decode_name), load_external_data=False), load_bytes)
    if rng_failures:
        return rng_failures

    def lm_feed(runtime: ort.InferenceSession, feed: dict, history_len: int = 0) -> dict:
        in_names = {value.name for value in runtime.get_inputs()}
        result = dict(feed)
        if "rng_seed" in in_names:
            result.pop("uniform", None)
            result.update(rng_seed=np.array([RANDOM_SEED], np.uint64), rng_step=np.zeros(1, np.int64))
        if "history_len" in in_names:
            result["history_len"] = np.array([history_len], np.int64)
        return result
    sampler_token_maps = []
    for sampler in samplers:
        sampler_meta = sampler.get_modelmeta().custom_metadata_map
        sampler_token_maps.append(
            np.unique(np.concatenate([
                np.arange(meta["audio_token_first"], meta["audio_token_last"] + 1, dtype=np.int32),
                np.array([meta["speech_end_id"]], np.int32),
            ])) if sampler_meta.get("audio_only_head") == "1" else None
        )
    stateful = source_fine = None
    if advanced_folder is not None:
        from Advanced_Graphs import DECODER_NAME, FINE_NAME, make_fine_reference
        source_fine = load_bytes(make_fine_reference(voc32_m).SerializeToString())
        stateful = StatefulVocoder(
            load_target(FINE_NAME), load_target(DECODER_NAME), ort.RunOptions(),
            ort.OrtValue.ortvalue_from_numpy(voice) if voice is not None else None,
        )
        if tts_mode == "voice_clone":
            if projected_speaker is None:
                raise ValueError("Advanced cloning verification needs a precomputed speaker projection")
            stateful.set_speaker(projected_speaker)
    del lm32_m, voc32_m
    rng = np.random.RandomState(42)
    draws = mismatches = 0
    softmax_error = 0.0
    failed = []
    empty = lambda sess: {arg.name: np.zeros([dim if isinstance(dim, int) else 0 for dim in arg.shape], io_dtype(arg))
                          for arg in sess.get_inputs() if arg.name in past}
    cast_feed = lambda sess, feed: {arg.name: np.asarray(feed[arg.name], io_dtype(arg)) for arg in sess.get_inputs()}
    for text in VERIFY_TEXTS:
        feed = prompt = build_prompt(tok, normalize_text(text), prefix, suffix)
        kv32, kv16 = empty(lm32), empty(lm16)
        seen = np.zeros(meta["vocab_size"], np.bool_)
        toks, h32, h16, agree = [], [], [], 0
        past_len = 0
        for step in range(VERIFY_MAX_STEPS):
            base = {"input_ids": feed, "ctrl": greedy, "uniform": np.zeros(1, np.float32), "seen": seen}
            if voice is not None:
                base["voice"] = voice
            o32 = dict(zip(names, lm32.run(None, {**base, **kv32})))
            target_lm = lm_decode if step and lm_decode is not None else lm16
            o16 = dict(zip([o.name for o in target_lm.get_outputs()],
                           target_lm.run(None, lm_feed(target_lm, {**base, **kv16}, history_len=past_len))))
            past_len += int(feed.shape[1])
            t32 = int(o32["token"][0, 0])
            agree += t32 == int(o16["token"][0, 0])
            u = float(rng.randint(0, 1 << 32, dtype=np.uint32) >> 8) / (1 << 24)
            for sampler, output, token_map in zip(samplers, (o32, o16), sampler_token_maps):
                drawn, scaled, probabilities = sampler.run(None, {"hidden": output["hidden"], "seen": seen,
                                                                 "ctrl": sampling, "uniform": np.array([u], np.float32)})
                exponential = np.exp(scaled - scaled.max())
                expected = exponential / exponential.sum(dtype=np.float32)
                softmax_error = max(softmax_error, float(np.max(np.abs(expected - probabilities))))
                expected_token = reference_token(probabilities, float(sampling[1]), u)
                if token_map is not None:
                    expected_token = int(token_map[expected_token])
                mismatches += int(drawn[0, 0]) != expected_token
                draws += 1
            toks.append(t32)
            h32.append(o32["hidden"][0, 0])
            h16.append(o16["hidden"][0, 0].astype(np.float32))
            kv32 = {k: o32[k.replace("past_key_values", "present")] for k in past}
            kv16 = {k: o16[k.replace("past_key_values", "present")] for k in past}
            seen = o32["seen_out"]
            feed = o32["token"]
            if t32 == meta["speech_end_id"]:
                break
        h32a, h16a = np.stack(h32), np.stack(h16)
        rel = float(np.linalg.norm(h32a - h16a) / np.linalg.norm(h32a))
        keep = [i for i, t in enumerate(toks) if meta["audio_token_first"] <= t <= meta["audio_token_last"]]
        codes = np.array([toks[i] for i in keep], np.int32)[None]
        hid = h32a[keep][None]
        voc_feed = {"hidden_states": hid, "audio_tokens": codes}
        voc_feed.update({"voice": voice} if voice is not None else {"speaker_embedding": spk_vec_fixed})
        w32 = voc32.run(None, voc_feed)[0].astype(np.float32)
        w16 = voc16.run(None, cast_feed(voc16, voc_feed))[0].astype(np.float32) if voc16 is not None else None
        if stateful is not None and source_fine is not None:
            source_bits = source_fine.run(["bits"], voc_feed)[0]
            failed += _verify_stateful_wave(stateful, hid, codes, w32, source_bits, min_snr_db, exact_fine_bits)
        if voice is not None and len(meta["voices"]) > 1:
            other = np.array((int(voice) + 1) % len(meta["voices"]), np.int32)
            first = {"input_ids": prompt, "ctrl": greedy, "uniform": np.zeros(1, np.float32),
                     "seen": np.zeros(meta["vocab_size"], np.bool_), "voice": other, **empty(lm16)}
            h_other = lm16.run(["hidden"], lm_feed(lm16, first, history_len=0))[0][0, 0].astype(np.float32)
            hidden_diff = float(np.max(np.abs(h_other - h16[0])))
            if voc16 is not None:
                w_other = voc16.run(None, cast_feed(voc16, {**voc_feed, "voice": other}))[0].astype(np.float32)
                wav_diff = float(np.max(np.abs(w_other - w16))) if w_other.shape == w16.shape else float("inf")
            elif stateful is not None:
                other_stateful = StatefulVocoder(
                    stateful.fine, stateful.decoder, ort.RunOptions(), ort.OrtValue.ortvalue_from_numpy(other),
                )
                w_cur = np.empty(codes.shape[1] * export_hojo.HOP, np.float32)
                w_oth = np.empty_like(w_cur)
                stateful.reset()
                stateful.process(hid, codes, w_cur, final=True)
                other_stateful.process(hid, codes, w_oth, final=True)
                wav_diff = float(np.max(np.abs(w_oth - w_cur)))
            else:
                wav_diff = 1.0
            print(f"  verify voice {int(voice)} -> {int(other)}: LM hidden max diff {hidden_diff:.4g}, wav max diff {wav_diff:.4g}")
            if not hidden_diff or not wav_diff:
                failed.append(f"voice {int(other)} reproduced voice {int(voice)} (LM hidden or waveform) for '{text}'")
        ratio = agree / len(toks)
        if w16 is not None:
            snr = float(10 * np.log10(np.sum(w32 ** 2) / max(float(np.sum((w32 - w16) ** 2)), 1e-20)))
            print(f"  verify '{text}': greedy {agree}/{len(toks)}, hidden rel err {rel:.4f}, "
                  f"vocoder raw SNR {snr:.1f} dB, {w16.size / meta['sample_rate']:.2f}s")
            if not np.isfinite(w16).all() or snr < min_snr_db:
                failed.append(f"vocoder raw SNR {snr:.1f} dB < {min_snr_db} (or non-finite) for '{text}'")
        else:
            print(f"  verify '{text}': greedy {agree}/{len(toks)}, hidden rel err {rel:.4f}, "
                  f"{codes.shape[1] * meta['hop_length'] / meta['sample_rate']:.2f}s (streaming-only)")
        if ratio < min_greedy:
            failed.append(f"greedy agreement {ratio:.3f} < {min_greedy} for '{text}'")
        if rel > max_hidden_error:
            failed.append(f"hidden rel err {rel:.4f} > {max_hidden_error} for '{text}'")
    # Verify unpunctuated single-character '好' reaches speech_end on Chinese reference voices (Guide H4).
    # Use deterministic 1-thread execution so x86_64 parallel MatMul reduction order cannot make EOS flaky.
    if tts_mode == "preset_voices":
        eos_pre = load_target(LM, max_threads=1)
        eos_dec = load_target(decode_name, max_threads=1) if (target_dir / decode_name).is_file() else None
        eos_id = meta["speech_end_id"]
        hao_prompt = build_prompt(tok, "好", prefix, suffix)
        max_steps = min(200, meta["max_positions"] - int(hao_prompt.shape[1]))
        hao_ctrl = np.array([0.8, 0.95, 1.05], np.float32)
        for v_name in ("hojo_zh_f_01", "hojo_zh_m_02"):
            if v_name not in meta["voices"]:
                continue
            v_arr = np.array(meta["voices"].index(v_name), np.int32)
            kv_h = empty(eos_pre)
            seen_h = np.zeros(meta["vocab_size"], np.bool_)
            cur_ids = hao_prompt
            p_len = 0
            ended = False
            for step in range(max_steps):
                sess = eos_pre if step == 0 or eos_dec is None else eos_dec
                step_feed = {"input_ids": cur_ids, "voice": v_arr, "ctrl": hao_ctrl,
                             "uniform": np.zeros(1, np.float32), "seen": seen_h, **kv_h}
                f_in = lm_feed(sess, step_feed, history_len=p_len)
                if "rng_step" in f_in:
                    f_in["rng_step"] = np.array([step], np.int64)
                res = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, f_in)))
                p_len += int(cur_ids.shape[1])
                tk = int(res["token"].item())
                if tk == eos_id:
                    if step < 10:
                        failed.append(f"Unpunctuated '好' voice={v_name} terminated prematurely at step={step}")
                    ended = True
                    break
                seen_h = res["seen_out"]
                cur_ids = res["token"]
                kv_h = {k: res[k.replace("past_key_values", "present")] for k in past}
            if not ended:
                failed.append(f"Unpunctuated '好' voice={v_name} failed to emit speech_end within {max_steps} steps")
            else:
                print(f"  verify EOS '好' ({v_name}): terminated cleanly at step {step}")
    print(f"  verify sampler: {draws - mismatches}/{draws} draws equal the full-sort numpy reference; "
          f"softmax max abs error {softmax_error:.9g}")
    if mismatches:
        failed.append(f"in-graph sampler differs from the full-sort reference in {mismatches}/{draws} draws")
    if stateful is not None:
        stateful.close()
    return failed


def _verify_counter_rng(model: onnx.ModelProto, load) -> list[str]:
    """Run the shipped in-branch RNG nodes standalone and require bit equality with the NumPy SplitMix64."""
    from Advanced_Graphs import RNG_PREFIX, splitmix_uniform

    if not any(value.name == "rng_seed" for value in model.graph.input):
        return []
    choice = next(node for node in model.graph.node if node.op_type == "If" and node.name == "hj_choose_token")
    branch = next(attribute.g for attribute in choice.attribute if attribute.name == "else_branch")
    nodes = [node for node in branch.node if node.name.startswith(RNG_PREFIX)]
    tensors = [tensor for tensor in branch.initializer if tensor.name.startswith(RNG_PREFIX)]
    probe = H.make_model(H.make_graph(
        nodes, "rng_probe",
        [H.make_tensor_value_info("rng_seed", TP.UINT64, ["n"]), H.make_tensor_value_info("rng_step", TP.INT64, ["n"])],
        [H.make_tensor_value_info("uniform", TP.FLOAT, ["n"])], tensors,
    ), opset_imports=[H.make_opsetid("", 20)])
    probe.ir_version = 8
    rng = np.random.default_rng(20261112)
    seeds = rng.integers(0, np.iinfo(np.int64).max, 65536, dtype=np.int64).astype(np.uint64) * np.uint64(3)
    steps = rng.integers(0, MAX_NEW_TOKENS, 65536, dtype=np.int64)
    actual = load(probe.SerializeToString()).run(None, {"rng_seed": seeds, "rng_step": steps})[0]
    expected = splitmix_uniform(seeds, steps)
    equal = np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    print(f"  verify counter RNG ({len(nodes)} in-branch nodes): {'bit-exact' if equal else 'MISMATCH'} "
          f"vs NumPy SplitMix64 on {seeds.size} (seed, step) pairs")
    return [] if equal and len(nodes) == 15 else ["in-graph counter RNG differs from the NumPy SplitMix64 reference"]


def _verify_stateful_wave(
    stateful: StatefulVocoder, hidden: np.ndarray, tokens: np.ndarray, reference: np.ndarray,
    source_bits: np.ndarray, minimum_snr: float, exact_source_bits: bool = True,
) -> list[str]:
    frames = tokens.shape[1]
    stateful.reset()
    full = np.empty(frames * export_hojo.HOP, np.float32)
    if stateful.process(hidden, tokens, full, final=True) != full.size:
        return ["Native vocoder produced the wrong full-utterance sample count"]
    expected_bits = stateful.last_bits.numpy()
    if expected_bits.shape != source_bits.shape:
        return ["Advanced fine-local sign-bit shape differs from the source"]
    source_bit_errors = int(np.count_nonzero(source_bits != expected_bits))
    stateful.reset()
    streamed = np.empty_like(full)
    bit_chunks, start, written = [], 0, 0
    while start < frames:
        stop = min(frames, 42 if start == 0 else start + 128)
        written += stateful.process(np.ascontiguousarray(hidden[:, start:stop]),
                                  np.ascontiguousarray(tokens[:, start:stop]), streamed[written:], final=stop == frames)
        bit_chunks.append(stateful.last_bits.numpy())
        start = stop
    if written != full.size or full.shape != reference.shape:
        return ["Stateful vocoder sample count differs from the source"]
    bit_axis = expected_bits.ndim - 2
    bit_errors = int(np.count_nonzero(np.concatenate(bit_chunks, axis=bit_axis) != expected_bits))
    denominator = max(float(np.sum((reference.astype(np.float64) - full) ** 2)), 1e-30)
    snr = float(10 * np.log10(np.sum(reference.astype(np.float64) ** 2) / denominator))
    stream_denominator = max(float(np.sum((full.astype(np.float64) - streamed) ** 2)), 1e-30)
    stream_snr = float(10 * np.log10(np.sum(full.astype(np.float64) ** 2) / stream_denominator))
    print(f"  verify stateful: source SNR {snr:.2f} dB, chunk SNR {stream_snr:.2f} dB, "
          f"source/chunk sign-bit mismatches {source_bit_errors}/{bit_errors} of {expected_bits.size}")
    failed = []
    # Reduced-precision fine weights may flip near-zero source signs; the SNR gate bounds their audible effect.
    if source_bit_errors and exact_source_bits:
        failed.append(f"Advanced fine-local changed {source_bit_errors}/{expected_bits.size} source sign bits")
    if bit_errors:
        failed.append(f"Stateful fine-local changed {bit_errors}/{expected_bits.size} sign bits")
    # Pure F16 fine weights may flip near-zero source signs (44.70 dB seen); allow 5 dB below the gate there, chunk-vs-full stays strict.
    source_minimum = minimum_snr if exact_source_bits else minimum_snr - 5.0
    if not np.isfinite(full).all() or snr < source_minimum or stream_snr < minimum_snr:
        failed.append(f"Stateful vocoder SNR {snr:.2f}/{stream_snr:.2f} dB < {source_minimum}/{minimum_snr}")
    return failed


def build_model_plans(
    meta: dict[str, str],
    precision: str | None = None,
    *,
    lm_precision: str | None = None,
    vocoder_precision: str | None = None,
    ref_precision: str | None = None,
) -> dict[str, Plan]:
    num_heads = int(meta["num_heads"])
    hidden_size = int(meta["hidden_size"])
    lm_plan = MODEL_PLANS["Hojo_LM"]
    voc_plan = MODEL_PLANS["Hojo_Vocoder"]
    meta_plan = MODEL_PLANS.get("Hojo_Metadata", Plan(method="F32", optimize=False, transformer=False, external=False))
    lm_method = (lm_precision or precision or lm_plan.method).upper()
    voc_method = (vocoder_precision or precision or voc_plan.method).upper()
    plans: dict[str, Plan] = {
        "Hojo_LM": replace(
            lm_plan,
            method=lm_method,
            num_heads=lm_plan.num_heads or num_heads,
            hidden_size=lm_plan.hidden_size or hidden_size,
        ),
        "Hojo_Vocoder": replace(voc_plan, method=voc_method),
        "Hojo_Metadata": replace(meta_plan, method=meta_plan.method.upper()),
    }
    if meta["tts_mode"] == "voice_clone":
        ref_plan = MODEL_PLANS["Hojo_ReferencePreprocess"]
        ref_method = (ref_precision or precision or ref_plan.method).upper()
        plans["Hojo_ReferencePreprocess"] = replace(ref_plan, method=ref_method)
    return plans


def _cfg_from_metadata(meta: dict[str, str]) -> dict[str, object]:
    keys = {"num_attention_heads": "num_heads", "num_key_value_heads": "kv_heads", "head_dim": "head_dim",
            "num_hidden_layers": "num_layers", "hidden_size": "hidden_size", "max_positions": "max_positions"}
    missing = [m for m in (*keys.values(), "rope_theta") if m not in meta]
    if missing:
        raise ValueError(f"exported package metadata lacks {missing}; re-export with Export_Hojo_TTS.py")
    cfg: dict[str, object] = {k: int(meta[m]) for k, m in keys.items()}
    cfg["rope_theta"] = float(meta["rope_theta"])
    cfg["voices"] = parse_metadata(meta)["voices"]
    return cfg


def optimize_package(
    src_folder: Path,
    out_folder: Path,
    *,
    precision: str | None = None,
    lm_precision: str | None = None,
    vocoder_precision: str | None = None,
    ref_precision: str | None = None,
    skip_verify: bool = False,
    bundle_shared: bool = BUNDLE_SHARED_INITIALIZERS,
    min_greedy: float | None = None,
    max_hidden_error: float | None = None,
    min_snr_db: float | None = None,
    advanced: bool = False,
    fold_coarse: bool = True,
    head_precision: str | None = None,
    audio_only_head: bool = False,
    fine_stream_precision: str | None = None,
    stabilize_lm_fp32: bool = STABILIZE_LM_FP32_ATTENTION,
    keep_whole_vocoder: bool = not DROP_WHOLE_VOCODER_IN_ADVANCED,
    kv_update: str = KV_UPDATE_MODE,
    custom_ops_library: Path | None = None,
    export_tokenizer_tables: bool = EXPORT_TOKENIZER_TABLES,
) -> int:
    src_folder = src_folder.expanduser().resolve()
    out_folder = out_folder.expanduser().resolve()
    if out_folder == src_folder:
        raise ValueError(f"Output folder must differ from source folder: {out_folder}")
    if kv_update not in {"concat", "inplaced_kv"}:
        raise ValueError(f"Unsupported kv_update mode: {kv_update!r} (expected 'concat' or 'inplaced_kv')")
    if custom_ops_library is None and os.environ.get("HOJO_KV_OPS_LIBRARY"):
        custom_ops_library = Path(os.environ["HOJO_KV_OPS_LIBRARY"]).expanduser().resolve()
    elif custom_ops_library is not None:
        custom_ops_library = custom_ops_library.expanduser().resolve()
    meta_source = src_folder / META
    meta = read_onnx_metadata(str(meta_source))
    parse_metadata(meta)
    cfg = _cfg_from_metadata(meta)
    plans = build_model_plans(
        meta,
        precision,
        lm_precision=lm_precision,
        vocoder_precision=vocoder_precision,
        ref_precision=ref_precision,
    )
    lm_method = plans["Hojo_LM"].method.upper()
    voc_method = plans["Hojo_Vocoder"].method.upper()
    ref_method = plans["Hojo_ReferencePreprocess"].method.upper() if "Hojo_ReferencePreprocess" in plans else "F32"
    fine_method = (fine_stream_precision or FINE_STREAM_PRECISION or voc_method).upper()
    if fine_method not in {"F16", "F32", "DYNAMIC", "Q4", "Q8"}:
        raise ValueError(f"Unsupported Hojo_Fine_Stream precision: {fine_method}")
    if min_greedy is None:
        min_greedy = MIN_GREEDY_AGREEMENT if lm_method in {"F16", "F32"} else (0.85 if lm_method == "Q8" else 0.70)
    if max_hidden_error is None:
        max_hidden_error = MAX_HIDDEN_REL_ERR if lm_method in {"F16", "F32"} else (0.05 if lm_method == "Q8" else 0.20)
    if min_snr_db is None:
        if voc_method == "F32" or (voc_method == "F16" and precision is not None and precision.upper() == "F16"):
            min_snr_db = MIN_VOCODER_SNR_DB
        elif voc_method == "F16":
            min_snr_db = 30.0
        elif voc_method == "Q8":
            min_snr_db = 10.0
        else:
            min_snr_db = 5.0
    if not 0 <= min_greedy <= 1 or max_hidden_error <= 0 or not np.isfinite(min_snr_db):
        raise ValueError("Invalid numerical verification thresholds")
    tools = {"onnx": onnx.__version__, "onnxslim": onnxslim.__version__}
    print(f"tools {tools}, onnxruntime {ort.__version__}, numpy {np.__version__}")
    published = out_folder
    out_folder = export_hojo.staging_dir(published)
    try:
        lm_post = (stabilize_lm_fp32 and lm_method == "F16") or kv_update == "inplaced_kv"
        postprocess = advanced or head_precision is not None or audio_only_head or lm_post
        base_bundle = bundle_shared and not postprocess
        skip_vocoder = advanced and not keep_whole_vocoder
        skip_reference = advanced
        if precision is not None and precision.upper() == "F16" and lm_method == "F16" and voc_method == "F16":
            rc = _optimize_f16(
                src_folder, out_folder, meta, cfg, base_bundle,
                skip_vocoder=skip_vocoder, skip_reference=skip_reference,
            )
        else:
            rc = _optimize_common(
                src_folder, out_folder, meta, cfg, plans, base_bundle,
                skip_vocoder=skip_vocoder, skip_reference=skip_reference,
            )
        if rc:
            return rc
        if head_precision is not None or audio_only_head:
            _rewrite_lm_head(src_folder, out_folder, precision=head_precision, audio_only=audio_only_head)
        if advanced:
            _build_advanced(
                src_folder, out_folder, fold_coarse=fold_coarse, decoder_precision=voc_method,
                fine_precision=fine_method, lm_method=lm_method, stabilize_lm_fp32=stabilize_lm_fp32,
                keep_whole_vocoder=keep_whole_vocoder, kv_update=kv_update,
            )
        elif lm_post:
            from Advanced_Graphs import convert_lm_to_inplaced_kv, stabilize_lm_fp32_attention_and_activations
            lm_m = inline_model_initializers(out_folder / LM)
            if kv_update == "inplaced_kv":
                lm_m = convert_lm_to_inplaced_kv(lm_m)
            if stabilize_lm_fp32 and lm_method == "F16":
                stabilize_lm_fp32_attention_and_activations(lm_m, LM)
            onnx.checker.check_model(lm_m, full_check=True)
            onnx.save(lm_m, out_folder / LM)
            carrier_meta = read_onnx_metadata(str(out_folder / META))
            carrier_meta["lm_kv_update"] = kv_update
            if stabilize_lm_fp32 and lm_method == "F16":
                carrier_meta["lm_fp32_attention_activations"] = "1"
            onnx.save(export_hojo.build_metadata_carrier(carrier_meta), str(out_folder / META))
        if bundle_shared and postprocess:
            updated = read_onnx_metadata(str(out_folder / META))
            _bundle(out_folder, [path for path in out_folder.glob("*.onnx") if path.name != META], updated)
            onnx.save(export_hojo.build_metadata_carrier(updated), str(out_folder / META))
        if export_tokenizer_tables:
            from Advanced_Graphs import export_tokenizer_tables as write_tok_tables
            write_tok_tables(src_folder / meta["tokenizer_file"], out_folder)
        failed = []
        loaded_for_bundle: dict[str, onnx.ModelProto] = {}
        shared_path = out_folder / SHARED_MODEL_NAME
        shared_store = SharedInitializerStore(shared_path, respect_prepack_exclusions=True) if shared_path.is_file() else None
        for path in sorted(out_folder.glob("*.onnx")):
            onnx.checker.check_model(str(path))
            model = onnx.load(str(path), load_external_data=False)
            loaded_for_bundle[path.name] = model
            if path.name == SHARED_MODEL_NAME:
                continue
            model_method = plans[path.stem].method if path.stem in plans else (
                lm_method if path.name == "Hojo_LM_Decode.onnx" else "F32"
            )
            failed += check_graph(path.name, model, cfg, precision=model_method)
            needs_custom = (kv_update == "inplaced_kv" and path.name in (LM, "Hojo_LM_Decode.onnx"))
            if not needs_custom or custom_ops_library is not None:
                sess_so = ort.SessionOptions()
                sess_so.log_severity_level = 3
                if custom_ops_library is not None:
                    sess_so.register_custom_ops_library(str(custom_ops_library))
                if shared_store is not None and path.name != META:
                    shared_store.attach_to(sess_so, model_path=path)
                ort.InferenceSession(str(path), sess_options=sess_so, providers=["CPUExecutionProvider"])
        if bundle_shared and SHARED_MODEL_NAME in loaded_for_bundle:
            carrier = loaded_for_bundle[SHARED_MODEL_NAME]
            canonical_names = {t.name for t in carrier.graph.initializer}
            live_ext = collect_live_external_names(loaded_for_bundle, SHARED_MODEL_NAME)
            if canonical_names != live_ext:
                failed.append(f"Dead shared tensors remain in carrier: {sorted(canonical_names - live_ext)}")
            exp_excl = collect_normalization_exclusions(loaded_for_bundle, SHARED_MODEL_NAME, canonical_names)
            meta_now = read_onnx_metadata(str(out_folder / META))
            act_excl = set(filter(None, meta_now.get("shared_initializer_prepack_excluded", "").split(",")))
            if act_excl != exp_excl:
                failed.append(f"Prepack exclusion mismatch: metadata={len(act_excl)} expected={len(exp_excl)}")
        if not skip_verify and not failed:
            if kv_update == "inplaced_kv" and custom_ops_library is None:
                print("[Verify] Skipping inplaced_kv LM ORT execution (set --custom-ops-library / HOJO_KV_OPS_LIBRARY to run); "
                      "running structural + ONNX checker + counter-RNG + streaming vocoder parity checks.")
                failed += _verify_inplaced_kv_offline(
                    src_folder, out_folder, loaded_for_bundle,
                    advanced=advanced, min_snr_db=min_snr_db, exact_fine_bits=fine_method == "F32",
                )
            else:
                failed += verify(src_folder, out_folder, src_folder / meta["tokenizer_file"],
                                 min_greedy=min_greedy, max_hidden_error=max_hidden_error,
                                 min_snr_db=min_snr_db, precision=ref_method,
                                 advanced_folder=out_folder if advanced else None,
                                 exact_fine_bits=fine_method == "F32",
                                 custom_ops_library=custom_ops_library)
        if failed:
            print("VERIFY FAILED:\n  " + "\n  ".join(failed), file=sys.stderr)
            return 1
        export_hojo.replace_dir_atomic(out_folder, published)
        print(f"saved {published}")
        return 0
    finally:
        shutil.rmtree(out_folder, ignore_errors=True)


def _rewrite_lm_head(src: Path, out: Path, *, precision: str | None, audio_only: bool) -> None:
    from Advanced_Graphs import extract_model, restrict_audio_head

    model = inline_model_initializers(out / LM)
    if precision is not None:
        if precision not in {"Q4", "Q8"}:
            raise ValueError("Additional head precision must be Q4 or Q8")
        hidden = next(value for value in model.graph.output if value.name == "hidden")
        if hidden.type.tensor_type.elem_type != TP.FLOAT:
            raise ValueError("Quantized head specialization requires an FP32-activation LM plan")
        source = inline_model_initializers(src / LM)
        vocabulary = int(read_onnx_metadata(str(out / META))["vocab_size"])
        head = extract_model(
            source, [hidden], [H.make_tensor_value_info("hj_logits3", TP.FLOAT, [1, 1, vocabulary])], "Hojo_Head",
        )
        plan = replace(MODEL_PLANS["Hojo_LM"], method=precision, optimize=False, transformer=False,
                       nodes_to_include=None, nodes_to_exclude=None, opt_level=0)
        with tempfile.TemporaryDirectory(prefix="hojo_head_") as temp:
            original, destination = Path(temp) / "source", Path(temp) / "output"
            original.mkdir()
            destination.mkdir()
            onnx.save(head, original / "Hojo_Head.onnx")
            config = OptimizerConfig(
                original_folder_path=str(original), optimized_folder_path=str(destination),
                model_plans={"Hojo_Head": plan}, block_size=BLOCK_SIZE, accuracy_level=ACCURACY_LEVEL,
                dynamic_default_tensor_type=TP.FLOAT,
            )
            process_model("Hojo_Head", resolve_plan(plan, config), config, mixed_precision=False)
            quantized = inline_model_initializers(destination / "Hojo_Head.onnx")
        export_hojo.prefix(quantized, "r2_head", {"hidden", "hj_logits3"})
        found, nodes = 0, []
        for node in model.graph.node:
            if "hj_logits3" in node.output:
                nodes.extend(quantized.graph.node)
                found += 1
            else:
                nodes.append(node)
        if found != 1:
            raise ValueError("Expected one replaceable LM output projection")
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        model.graph.initializer.extend(quantized.graph.initializer)
        del model.graph.value_info[:]
        dce(model)
    if audio_only:
        model = restrict_audio_head(model)
        print("[Experimental] Audio-only head changes sampling probabilities; full vocabulary is the default.")
    metadata = {item.key: item.value for item in model.metadata_props}
    if precision:
        metadata["lm_head_precision"] = precision
    H.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    onnx.save(model, out / LM)
    carrier = read_onnx_metadata(str(out / META))
    for key in ("audio_only_head", "sampling_vocab_size", "lm_head_precision"):
        if key in metadata:
            carrier[key] = metadata[key]
    onnx.save(export_hojo.build_metadata_carrier(carrier), out / META)


def _build_advanced(
    src: Path,
    out: Path,
    *,
    fold_coarse: bool,
    decoder_precision: str = "F32",
    fine_precision: str = "F32",
    lm_method: str = "F16",
    stabilize_lm_fp32: bool = STABILIZE_LM_FP32_ATTENTION,
    keep_whole_vocoder: bool = not DROP_WHOLE_VOCODER_IN_ADVANCED,
    kv_update: str = KV_UPDATE_MODE,
) -> None:
    from Advanced_Graphs import (
        DECODE_NAME, DECODER_NAME, FINE_NAME, adapt_decoder_pcm_for_float_playback,
        convert_lm_to_inplaced_kv, counter_rng_sampler, flatten_decode_mqa, fold_coarse_projection,
        fold_speaker_projection, make_decoder_stream_pcm, make_fine_stream, specialize_decode,
        stabilize_lm_fp32_attention_and_activations, store_weights_fp16, streaming_metadata,
        unwrap_f16_neural_boundaries,
    )
    from Shared_Weights import MIN_SHARED_INITIALIZER_ELEMENTS

    meta = read_onnx_metadata(str(out / META))
    if (int(meta["out_sample_rate"]) != export_hojo.MODEL_SAMPLE_RATE or meta["out_audio_dtype"] != "F32"
            or int(meta["n_fft"]) != 1920 or int(meta["hop_length"]) != 480):
        raise ValueError("Advanced CPU rendering currently requires native 24 kHz float32 output")
    model = onnx.shape_inference.infer_shapes(onnxslim.slim(inline_model_initializers(src / VOC)))
    name_nodes(model, "r2_vocoder")
    fuse_gelu_fp16(model)
    fuse_vocoder_fp16(model)
    if export_hojo.vocoder_context(model) != (70, 10):
        raise ValueError("Unsupported streaming vocoder receptive field")
    reference = inline_model_initializers(src / REF_PRE) if meta["tts_mode"] == "voice_clone" else None
    speaker_kind = fold_speaker_projection(model, reference)
    rows = fold_coarse_projection(model) if fold_coarse else 0
    fine = make_fine_stream(model)
    decoder = make_decoder_stream_pcm(model)
    f16_limits = {"min_elements": MIN_SHARED_INITIALIZER_ELEMENTS,
                  "min_positive": OptimizerConfig.f16_min_positive_val,
                  "max_finite": OptimizerConfig.f16_max_finite_val}
    widened = 0
    if decoder_precision == "F16":
        name_nodes(decoder, "r2_decoder_stream")
        decoder = stream_graph_to_fp16(decoder, stream_fp32_region(decoder))
    elif decoder_precision != "F32":
        widened = store_weights_fp16(decoder, **f16_limits)
    if fine_precision == "F16":
        name_nodes(fine, "r2_fine_stream")
        fine = stream_graph_to_fp16(fine)
    elif fine_precision != "F32":
        # Weight-only/dynamic quantization without graph fusions, which would not preserve the stream state I/O.
        plan = replace(MODEL_PLANS["Hojo_Vocoder"], method=fine_precision, optimize=False, transformer=False)
        with tempfile.TemporaryDirectory(prefix="hojo_fine_") as temp:
            original, destination = Path(temp) / "source", Path(temp) / "output"
            original.mkdir()
            destination.mkdir()
            onnx.save(fine, original / "Hojo_Fine_Stream.onnx")
            config = OptimizerConfig(
                original_folder_path=str(original), optimized_folder_path=str(destination),
                model_plans={"Hojo_Fine_Stream": plan}, block_size=BLOCK_SIZE, accuracy_level=ACCURACY_LEVEL,
                dynamic_default_tensor_type=TP.FLOAT,
            )
            process_model("Hojo_Fine_Stream", resolve_plan(plan, config), config, mixed_precision=False)
            fine = inline_model_initializers(destination / "Hojo_Fine_Stream.onnx")
    prefill = inline_model_initializers(out / LM)
    if decoder_precision == fine_precision == "F16":
        unwrap_f16_neural_boundaries(prefill, fine, decoder, require_lm_f16=(lm_method == "F16"))
    if decoder_precision == "F16":
        adapt_decoder_pcm_for_float_playback(decoder)
    decode = counter_rng_sampler(flatten_decode_mqa(specialize_decode(prefill)))
    prefill = counter_rng_sampler(prefill)
    if kv_update == "inplaced_kv":
        prefill = convert_lm_to_inplaced_kv(prefill)
        decode = convert_lm_to_inplaced_kv(decode)
    lm_wrapped = dec_wrapped = 0
    if stabilize_lm_fp32 and lm_method == "F16":
        lm_wrapped = stabilize_lm_fp32_attention_and_activations(prefill, LM)
        dec_wrapped = stabilize_lm_fp32_attention_and_activations(decode, DECODE_NAME)
    geom = streaming_metadata(fine, decoder, None, int(meta["hop_length"]), int(meta["n_fft"]))
    if reference is not None:
        meta["reference_speaker_projection_output"] = "speaker_projection"
    if not keep_whole_vocoder:
        (out / VOC).unlink(missing_ok=True)
        (out / (VOC + ".data")).unlink(missing_ok=True)
        meta.pop("model_file_name_vocoder", None)
    meta.pop("native_fft_library_file", None)
    meta.update({
        "advanced_pipeline_version": "2",
        "model_file_name_lm_decode": DECODE_NAME,
        "model_file_name_fine_stream": FINE_NAME,
        "model_file_name_decoder_stream": DECODER_NAME,
        "streaming_vocoder_precision": "F16" if decoder_precision == fine_precision == "F16" else "F32",
        "streaming_decoder_weight_precision": "F16" if widened or decoder_precision == "F16" else "F32",
        "streaming_fine_weight_precision": fine_precision,
        "stream_speaker_projection": speaker_kind,
        "fold_coarse_projection": str(int(fold_coarse)),
        "reference_prefix_cache": "1",
        "lm_kv_update": kv_update,
        "lm_decode_attention_layout": "flattened_mqa",
        "stream_first_chunk_frames": "128" if meta["tts_mode"] == "voice_clone" else "32",
        "stream_chunk_frames": "128",
        **geom,
    })
    if stabilize_lm_fp32 and lm_method == "F16":
        meta["lm_fp32_attention_activations"] = "1"
    if meta["tts_mode"] == "preset_voices" and decoder_precision == fine_precision == "F16":
        meta["in_audio_dtype"] = "F16"
        meta["out_audio_dtype"] = "F32"
    graphs: list[tuple[str, onnx.ModelProto]] = [
        (FINE_NAME, fine), (DECODER_NAME, decoder), (DECODE_NAME, decode), (LM, prefill),
    ]
    if reference is not None:
        graphs.append((REF_PRE, reference))
    del prefill, decode
    for name, result in graphs:
        if name != REF_PRE:
            export_hojo.int32_gather_indices(result)
        props = {item.key: item.value for item in result.metadata_props}
        props.update(meta)
        if not keep_whole_vocoder:
            props.pop("model_file_name_vocoder", None)
        if name == DECODE_NAME:
            props["lm_sequence_length"] = "1"
        set_sorted_metadata(result, props)
        onnx.checker.check_model(result, full_check=True)
        onnx.save(result, out / name)
    if keep_whole_vocoder and (out / VOC).is_file():
        voc_m = inline_model_initializers(out / VOC)
        props = {item.key: item.value for item in voc_m.metadata_props}
        props.update(meta)
        set_sorted_metadata(voc_m, props)
        onnx.save(voc_m, out / VOC)
    onnx.save(export_hojo.build_metadata_carrier(meta), out / META)
    print(f"[Advanced] Single-token decode, stateful vocoder (decoder={decoder_precision}, fine={fine_precision}), "
          f"in-graph iSTFT (pcm=F32), counter RNG; speaker={speaker_kind}, projected audio rows={rows}, "
          f"FP32-stabilized LM ops={lm_wrapped}/{dec_wrapped}, whole_vocoder={'kept' if keep_whole_vocoder else 'omitted'}, "
          f"LM KV={kv_update}.")


def _reference_preprocess(src: Path) -> bytes:
    """Slim and validate the FP32 reference graph."""
    model = onnxslim.slim(inline_model_initializers(src))
    onnx.checker.check_model(model)
    report(f"{REF_PRE} final", model)
    return model.SerializeToString()


def _bundle(out: Path, models: list[Path], meta_out: dict[str, str]) -> None:
    meta_out["shared_initializer_model_file"] = SHARED_MODEL_NAME
    meta_out["shared_initializer_data_file"] = SHARED_DATA_NAME
    stats = bundle_shared_initializers(out, model_paths=models, metadata=meta_out)
    print(f"[Shared weights] {stats['initializer_references']} references -> {stats['unique_initializers']} "
          f"unique tensors; deduplicated {stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB; "
          f"sub-tensor aliases={stats['subtensor_aliases']}; prepack_excluded={stats['prepack_excluded_count']}.")


def _optimize_f16(src_folder: Path, out: Path, meta: dict[str, str], cfg: dict,
                  bundle_shared: bool, *, skip_vocoder: bool = False, skip_reference: bool = False) -> int:
    built, failed = {}, []
    targets = (LM,) if skip_vocoder else (LM, VOC)
    for name in targets:
        built[name], bad = convert(src_folder, name, cfg)
        failed += bad
    if not skip_reference and (src_folder / REF_PRE).is_file():
        built[REF_PRE] = _reference_preprocess(src_folder / REF_PRE)
    if failed:
        print("VERIFY FAILED (nothing written):\n  " + "\n  ".join(failed), file=sys.stderr)
        return 1
    for name, data in built.items():
        (out / name).write_bytes(data)
    shutil.copy2(src_folder / export_hojo.TOKENIZER_FILE_NAME, out / export_hojo.TOKENIZER_FILE_NAME)
    meta_out = dict(meta)
    meta_out["activations_fp16"] = "1"
    meta_out.pop("shared_initializer_model_file", None)
    meta_out.pop("shared_initializer_data_file", None)
    if bundle_shared:
        _bundle(out, [out / name for name in built], meta_out)
    else:
        for name in built:
            replace_onnx_metadata(str(out / name), meta_out)
    onnx.save(export_hojo.build_metadata_carrier(meta_out), str(out / META))
    for name in built:
        print(f"{name}: {(out / name).stat().st_size} B (fp32 {(src_folder / name).stat().st_size} B)")
    return 0


def _optimize_common(src_folder: Path, out: Path, meta: dict[str, str], cfg: dict, plans: dict[str, Plan],
                     bundle_shared: bool, *, skip_vocoder: bool = False, skip_reference: bool = False) -> int:
    active_plans = {
        name: plan for name, plan in plans.items()
        if not (skip_vocoder and name == "Hojo_Vocoder") and not (skip_reference and name == "Hojo_ReferencePreprocess")
    }
    with tempfile.TemporaryDirectory(prefix="hojo_opt_src_") as tmp_src:
        stage_src = Path(tmp_src)
        for name, plan in active_plans.items():
            src_file = src_folder / f"{name}.onnx"
            if not src_file.is_file() or (name == "Hojo_LM" and plan.method.upper() == "F16"):
                continue
            m = inline_model_initializers(src_file)
            if name == "Hojo_LM":
                export_hojo.ensure_f16_kv_io(m)
                m = onnx.shape_inference.infer_shapes(onnxslim.slim(m))
                fuse_rmsnorm(m)
                bake_rotary(m, cfg)
                m = onnxslim.slim(m)
                name_nodes(m, "lm")
                print(f"  LM fusions: {dict(fuse_lm(m, cfg))}")
            elif name == "Hojo_Vocoder":
                m = onnxslim.slim(m)
                name_nodes(m, "vocoder")
                gelus = fuse_gelu_fp16(m)
                print(f"  vocoder fusions: Gelu {gelus}; {dict(fuse_vocoder_fp16(m))}")
            export_hojo.int32_gather_indices(m)
            onnx.save(m, str(stage_src / f"{name}.onnx"))
        config = OptimizerConfig(
            original_folder_path=str(stage_src),
            optimized_folder_path=str(out),
            model_plans=active_plans,
            block_size=BLOCK_SIZE,
            accuracy_level=ACCURACY_LEVEL,
            optimizer_level=OPTIMIZER_LEVEL,
            dynamic_default_tensor_type=onnx.TensorProto.FLOAT,
            f16_keep_io_types=KEEP_IO_TYPE,
        )
        resolved = {name: resolve_plan(plan, config) for name, plan in active_plans.items()}
        mixed = uses_mixed_precision(resolved.values())
        for name, rp in resolved.items():
            if name == "Hojo_LM" and rp.method == "F16":
                print(f"\nOptimizing {name} [{rp.method}]")
                data, bad = convert(src_folder, LM, cfg)
                if bad:
                    print("VERIFY FAILED:\n  " + "\n  ".join(bad), file=sys.stderr)
                    return 1
                (out / LM).write_bytes(data)
                meta_lm = dict(meta)
                meta_lm["activations_fp16"] = "1"
                meta_lm.pop("shared_initializer_model_file", None)
                meta_lm.pop("shared_initializer_data_file", None)
                replace_onnx_metadata(str(out / LM), meta_lm)
            elif (stage_src / f"{name}.onnx").is_file():
                print(f"\nOptimizing {name} [{rp.method}]")
                item_cfg = replace(config, shape_infer=False) if name == "Hojo_ReferencePreprocess" else config
                process_model(name, rp, item_cfg, mixed_precision=mixed)
                if name == "Hojo_LM":
                    model = inline_model_initializers(out / LM)
                    standardize_rmsnorm(model)
                    onnx.save(model, str(out / LM))
    shutil.copy2(src_folder / export_hojo.TOKENIZER_FILE_NAME, out / export_hojo.TOKENIZER_FILE_NAME)
    if bundle_shared:
        meta_out = dict(meta)
        targets = [out / f"{n}.onnx" for n in active_plans if n != "Hojo_Metadata" and (out / f"{n}.onnx").is_file()]
        _bundle(out, targets, meta_out)
        onnx.save(export_hojo.build_metadata_carrier(meta_out), str(out / META))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=SOURCE_FOLDER, help="exported HojoTTS_ONNX folder")
    parser.add_argument("--weights", type=Path, default=None,
                        help="optional official Hojo-TTS-Light release dir for one-shot export + optimize")
    parser.add_argument("--out", type=Path, default=OUTPUT_FOLDER, help="optimized output dir")
    parser.add_argument("--precision", default=None, choices=("F16", "F32", "DYNAMIC", "Q4", "Q8"),
                        help="override precision for all models (default: use per-model USER CONFIG)")
    parser.add_argument("--lm-precision", default=None, choices=("F16", "F32", "DYNAMIC", "Q4", "Q8"),
                        help="override Hojo_LM precision (default: LM_PRECISION)")
    parser.add_argument("--vocoder-precision", default=None, choices=("F16", "F32", "DYNAMIC", "Q4", "Q8"),
                        help="override Hojo_Vocoder precision (default: VOCODER_PRECISION)")
    parser.add_argument("--ref-precision", default=None, choices=("F16", "F32", "DYNAMIC", "Q4", "Q8"),
                        help="override Hojo_ReferencePreprocess precision (default: REFERENCE_PREPROCESS_PRECISION)")
    parser.add_argument("--fine-stream-precision", default=None, choices=("F16", "F32", "DYNAMIC", "Q4", "Q8"),
                        help="override Hojo_Fine_Stream weight precision (default: same as the vocoder precision)")
    parser.add_argument("--voices", nargs="+", default=None)
    parser.add_argument("--skip-verify", action="store_true", help="skip numerical parity gates (structural gates still run)")
    parser.add_argument("--min-greedy-agreement", type=float, default=None)
    parser.add_argument("--max-hidden-relative-error", type=float, default=None)
    parser.add_argument("--min-vocoder-snr-db", type=float, default=None)
    parser.add_argument("--bundle-shared", action="store_true", default=BUNDLE_SHARED_INITIALIZERS)
    parser.add_argument("--no-bundle-shared", dest="bundle_shared", action="store_false")
    parser.add_argument("--advanced", action=argparse.BooleanOptionalAction, default=True,
                        help="add single-token decode, counter RNG, FP32-stabilized LM and stateful vocoder with in-graph iSTFT (default: on; use --no-advanced to disable)")
    parser.add_argument("--stabilize-lm-fp32", action=argparse.BooleanOptionalAction, default=STABILIZE_LM_FP32_ATTENTION,
                        help="wrap F16 LM attention QK^T/PV, Softmax, QuickGelu, and non-rope_add Mul/Add in FP32 (default: on)")
    parser.add_argument("--keep-whole-vocoder", action="store_true", default=not DROP_WHOLE_VOCODER_IN_ADVANCED,
                        help="retain whole-utterance Hojo_Vocoder.onnx alongside streaming vocoder graphs in --advanced mode")
    parser.add_argument("--kv-update", choices=("concat", "inplaced_kv"), default=KV_UPDATE_MODE,
                        help="LM KV-cache update op: 'concat' (portable ORT default) or 'inplaced_kv' (com.custom.llm::KVCacheUpdate)")
    parser.add_argument("--custom-ops-library", type=Path, default=None,
                        help="optional KVCacheUpdate custom ops shared library for ORT verification of --kv-update inplaced_kv (or HOJO_KV_OPS_LIBRARY)")
    parser.add_argument("--export-tokenizer-tables", action="store_true", default=EXPORT_TOKENIZER_TABLES,
                        help="export deterministic hojo_bpe.txt, hojo_added.txt, hojo_chars.txt, and golden_prompt_ids.txt")
    parser.add_argument("--no-fold-coarse", dest="fold_coarse", action="store_false", default=True,
                        help="keep original audio embeddings instead of precomputing their fine-local projection")
    parser.add_argument("--head-precision", choices=("Q4", "Q8"), default=None,
                        help="experimental quantized output head; transformer precision and gates remain unchanged")
    parser.add_argument("--audio-only-head", action="store_true",
                        help="experimental vocabulary restriction to audio IDs/EOS; changes sampling behavior")
    args = parser.parse_args(argv)
    out = args.out.expanduser().resolve()
    options = {
        "precision": args.precision,
        "lm_precision": args.lm_precision,
        "vocoder_precision": args.vocoder_precision,
        "ref_precision": args.ref_precision,
        "skip_verify": args.skip_verify,
        "bundle_shared": args.bundle_shared,
        "min_greedy": args.min_greedy_agreement,
        "max_hidden_error": args.max_hidden_relative_error,
        "min_snr_db": args.min_vocoder_snr_db,
        "advanced": args.advanced,
        "fold_coarse": args.fold_coarse,
        "head_precision": args.head_precision,
        "audio_only_head": args.audio_only_head,
        "fine_stream_precision": args.fine_stream_precision,
        "stabilize_lm_fp32": args.stabilize_lm_fp32,
        "keep_whole_vocoder": args.keep_whole_vocoder,
        "kv_update": args.kv_update,
        "custom_ops_library": args.custom_ops_library,
        "export_tokenizer_tables": args.export_tokenizer_tables,
    }
    src = args.src.expanduser().resolve()
    if args.weights is None and (src / LM).is_file():
        return optimize_package(src, out, **options)
    weights = (args.weights or Path(export_hojo.download_path)).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="hojo_fp32_") as tmp:
        fp32 = Path(tmp) / "fp32"
        export_hojo.export_package(
            weights, fp32, voices_filter=tuple(args.voices) if args.voices else export_hojo.VOICES, bundle_shared=False,
        )
        return optimize_package(fp32, out, **options)


if __name__ == "__main__":
    raise SystemExit(main())
