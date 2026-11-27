"""Export Hojo-TTS-Light (40M preset voices and 80M voice cloning) into fused ONNX graphs.

Hojo_LM.onnx                  : (input_ids, [voice int32], ctrl, uniform, seen, past_kv) -> embedding + speaker injection + Qwen3 step + in-graph sampler.
Hojo_Vocoder.onnx             : (hidden, audio token ids, voice int32 / speaker_embedding) -> fine_local -> decoder -> iSTFT -> wav.
Hojo_ReferencePreprocess.onnx : (80M voice clone) prompt_audio -> (prompt_audio_ids, speaker_embedding).
Hojo_Metadata.onnx            : static package metadata carrier.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import io
import json
import os
import pickletools
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import onnx
import onnxslim
from onnx import AttributeProto, TensorProto as TP, helper as H, numpy_helper as NH

from Inference_Hojo_TTS_ONNX import resolve_voice_index, voice_lookup
from Shared_Weights import (
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    bundle_shared_initializers,
    set_sorted_metadata,
)


# ──────────────────────────────────────────────────────────────────────────────
# Export settings
# ──────────────────────────────────────────────────────────────────────────────
download_path = str(Path.home() / "Downloads" / "Hojo-TTS-Light-40M-V2")  # Hojo-TTS-Light-40M | Hojo-TTS-Light-40M-V2 | Hojo-TTS-Light
MAX_SEQ_LEN = 4096                   # Maximum LM context (prompt + generated tokens; must be > MAX_NEW_TOKENS in the inference script and <= config.max_position_embeddings).
DO_EXPORT = True
OPSET = 20                           # ONNX opset version (18+ required for Col2Im and ReduceMean axes input).
IN_SAMPLE_RATE = 24000               # Reference audio rate fed to ReferencePreprocess (80M voice clone).
OUT_SAMPLE_RATE = 24000              # Generated audio rate from Vocoder.
IN_AUDIO_DTYPE = "F32"               # "F16" | "F32" | "INT16"
OUT_AUDIO_DTYPE = "F32"              # "F16" | "F32" | "INT16"

# Usable Preset Voice IDs (accepts 0-based int32 index or exact string name):
# ---------------------------------------------------------------------------------
# • Hojo-TTS-Light-40M V1 (15 preset voices):
#     0: "hojo_en_f_01"   1: "hojo_en_f_02"   2: "hojo_en_f_03"   3: "hojo_en_f_04"
#     4: "hojo_en_f_05"   5: "hojo_en_f_06"   6: "hojo_en_f_07"
#     7: "hojo_en_m_01"   8: "hojo_en_m_02"   9: "hojo_en_m_03"  10: "hojo_en_m_04"
#    11: "hojo_en_m_05"
#    12: "hojo_en_u_01"
#    13: "hojo_zh_f_01"  14: "hojo_zh_f_02"
#
# • Hojo-TTS-Light-40M V2 (17 preset voices):
#     0: "hojo_en_f_01"   1: "hojo_en_f_02"   2: "hojo_en_f_03"   3: "hojo_en_f_04"
#     4: "hojo_en_f_05"   5: "hojo_en_f_06"   6: "hojo_en_f_07"   7: "hojo_en_f_08"
#     8: "hojo_en_m_01"   9: "hojo_en_m_02"  10: "hojo_en_m_03"  11: "hojo_en_m_04"
#    12: "hojo_en_m_05"
#    13: "hojo_zh_f_01"  14: "hojo_zh_f_02"  15: "hojo_zh_m_01"  16: "hojo_zh_m_02"
#
# • Hojo-TTS-Light 80M (Voice Cloning / Zero-Shot):
#     None (no preset voices). Uses reference prompt audio + text instead.
# ---------------------------------------------------------------------------------
VOICES: tuple[int | str, ...] | None = None  # None exports all voices in voice.npz; or unique 0-based int32 rows / names, e.g. (13, 14).
BUNDLE_SHARED_INITIALIZERS = False   # Set True to deduplicate shared weights into Hojo_SharedInitializers.onnx.data.


# ──────────────────────────────────────────────────────────────────────────────
# Fixed model constants & derived paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_FOLDER = SCRIPT_DIR / "HojoTTS_ONNX"
LM_MODEL_NAME = "Hojo_LM.onnx"
VOCODER_MODEL_NAME = "Hojo_Vocoder.onnx"
REF_PREPROCESS_MODEL_NAME = "Hojo_ReferencePreprocess.onnx"
METADATA_MODEL_NAME = "Hojo_Metadata.onnx"
TOKENIZER_FILE_NAME = "tokenizer.json"

HOP = 480
N_FFT = 1920
MODEL_SAMPLE_RATE = 24000
CODEC_SAMPLE_RATE = 16000
SPEAKER_EMB_SECONDS = 6.0
SPEAKER_MEL_N_FFT = 1024
SPEAKER_MEL_HOP_SIZE = 256
SPEAKER_MEL_FMIN = 0.0
SPEAKER_MEL_FMAX = 12000.0
_AUDIO_TP_DTYPES = {"F16": TP.FLOAT16, "F32": TP.FLOAT, "INT16": TP.INT16}
_AUDIO_TOKEN_RE = re.compile(r"\[(\d+)\]")


def const(name: str, value, dtype=None) -> onnx.TensorProto:
    return NH.from_array(np.asarray(value, dtype=dtype), name=name)


def _npy_header(raw: bytes) -> tuple[dict, int]:
    if not raw.startswith(b"\x93NUMPY"):
        raise ValueError("voice npz member is not an .npy array")
    major = raw[6]
    header_len = int.from_bytes(raw[8:10], "little") if major == 1 else int.from_bytes(raw[8:12], "little")
    start = 10 if major == 1 else 12
    header = ast.literal_eval(raw[start:start + header_len].decode("latin1").strip())
    return header, start + header_len


def _voice_ids_from_object_npy(raw: bytes) -> list[str]:
    header, payload_at = _npy_header(raw)
    shape = header.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 1:
        raise ValueError("voice_ids object array must be 1-d")
    count = int(shape[0])
    unicode_ops = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}
    lists: list[list[str]] = []
    current: list[str] | None = None
    for opcode, arg, _pos in pickletools.genops(raw[payload_at:]):
        name = opcode.name
        if name == "EMPTY_LIST":
            current = []
        elif current is not None and name in unicode_ops:
            if not isinstance(arg, str):
                raise ValueError("voice_ids pickle string is not text")
            current.append(arg)
        elif current is not None and name in {"MEMOIZE", "MARK"}:
            continue
        elif current is not None and name == "APPENDS":
            lists.append(current)
            current = None
        elif current is not None:
            raise ValueError(f"refusing voice_ids pickle opcode {name}")
    matches = [ids for ids in lists if len(ids) == count]
    if len(matches) != 1:
        raise ValueError("voice_ids object array is not a plain string list")
    return matches[0]


def load_npz_without_pickle(path: str | Path) -> dict[str, np.ndarray | list[str]]:
    arrays: dict[str, np.ndarray | list[str]] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.endswith(".npy"):
                continue
            key = info.filename[:-len(".npy")]
            raw = archive.read(info)
            header, _ = _npy_header(raw)
            descr = header.get("descr")
            if isinstance(descr, str) and (descr == "O" or descr.startswith("|O")):
                if key != "voice_ids":
                    raise ValueError(f"{key} is a pickled object array and was refused")
                arrays[key] = _voice_ids_from_object_npy(raw)
            else:
                arrays[key] = np.load(io.BytesIO(raw), allow_pickle=False)
    return arrays


def _promote_graph_bf16_to_fp32(graph: onnx.GraphProto) -> None:
    bf16, fp32 = TP.BFLOAT16, TP.FLOAT
    for init in graph.initializer:
        if init.data_type == bf16:
            arr = NH.to_array(init)
            init.CopyFrom(NH.from_array(np.asarray(arr, dtype=np.float32), name=init.name))
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == bf16:
            vi.type.tensor_type.elem_type = fp32
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == bf16:
                    attr.i = fp32
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                _promote_graph_bf16_to_fp32(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for sub_graph in attr.graphs:
                    _promote_graph_bf16_to_fp32(sub_graph)
            elif attr.type == AttributeProto.TENSOR and attr.t.data_type == bf16:
                arr = NH.to_array(attr.t)
                attr.t.CopyFrom(NH.from_array(np.asarray(arr, dtype=np.float32), name=attr.t.name))
            elif attr.type == AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    if tensor.data_type == bf16:
                        arr = NH.to_array(tensor)
                        tensor.CopyFrom(NH.from_array(np.asarray(arr, dtype=np.float32), name=tensor.name))
    del graph.value_info[:]


def load_fp32(path: Path) -> onnx.ModelProto:
    model = onnx.load(str(path), load_external_data=True)
    _promote_graph_bf16_to_fp32(model.graph)
    return model


def discover_source_bundle(src: Path) -> dict[str, object]:
    src = src.expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Hojo source directory not found: {src}")
    llm_files = sorted(src.glob("*-llm.onnx"))
    dec_files = sorted(src.glob("*-decoder.onnx"))
    npz_files = sorted(src.glob("*-voice.npz"))
    if len(llm_files) != 1 or len(dec_files) != 1 or len(npz_files) != 1:
        raise FileNotFoundError(
            f"Expected one *-llm.onnx, *-decoder.onnx, and *-voice.npz in {src}; "
            f"found llm={llm_files}, dec={dec_files}, npz={npz_files}"
        )
    stem = llm_files[0].name[:-len("-llm.onnx")]
    fine_path = src / f"{stem}-fine_local.onnx"
    enc_path = src / f"{stem}-encoder.onnx"
    spk_path = src / f"{stem}-speaker.onnx"
    if fine_path.is_file():
        tts_mode = "preset_voices"
    elif enc_path.is_file() and spk_path.is_file():
        tts_mode = "voice_clone"
    else:
        raise FileNotFoundError(
            f"Could not determine Hojo variant in {src}: need either {fine_path.name} "
            f"(40M preset voices) or {enc_path.name} + {spk_path.name} (80M voice clone)"
        )
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    return {
        "src": src,
        "stem": stem,
        "tts_mode": tts_mode,
        "config": cfg,
        "llm_path": llm_files[0],
        "dec_path": dec_files[0],
        "npz_path": npz_files[0],
        "fine_path": fine_path if fine_path.is_file() else None,
        "enc_path": enc_path if enc_path.is_file() else None,
        "spk_path": spk_path if spk_path.is_file() else None,
    }


def token_ids(src: Path, tts_mode: str = "preset_voices") -> dict[str, int]:
    tok = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    table = dict(tok["model"]["vocab"])
    table.update({a["content"]: a["id"] for a in tok["added_tokens"]})
    base_names = (
        "[target_text_start]",
        "[target_text_end]",
        "[target_speech_start]",
        "[target_speech_end]",
    )
    extra_names = (
        ("[spk_start]", "[spk_end]", "[spk_emb_0]")
        if tts_mode == "preset_voices"
        else ("[ref_text_start]", "[ref_text_end]", "[ref_speech_start]", "[ref_speech_end]")
    )
    ids = {n: int(table[n]) for n in (*base_names, *extra_names)}
    audio = sorted((int(m.group(1)), int(i)) for t, i in table.items() if (m := _AUDIO_TOKEN_RE.fullmatch(t)))
    if not audio:
        raise ValueError("tokenizer vocabulary has no indexed audio codes")
    codes, audio_ids = zip(*audio)
    if codes != tuple(range(len(codes))) or audio_ids != tuple(range(audio_ids[0], audio_ids[0] + len(codes))):
        raise ValueError("audio tokens [0]..[N] are not contiguous ids")
    ids["audio_token_first"], ids["audio_token_last"] = audio_ids[0], audio_ids[-1]
    if tts_mode == "preset_voices":
        slots = 0
        while f"[spk_emb_{slots}]" in table:
            if table[f"[spk_emb_{slots}]"] != ids["[spk_emb_0]"] + slots:
                raise ValueError("speaker placeholder tokens are not contiguous")
            slots += 1
        ids["spk_slots"] = slots
    else:
        ids["space_id"] = int(table[" "])
    return ids


def load_voices(
    src: Path,
    voices_filter: tuple[int | str, ...] | list[int | str] | None = VOICES,
) -> dict[str, object]:
    bundle = discover_source_bundle(src)
    data = load_npz_without_pickle(bundle["npz_path"])
    out: dict[str, object] = {
        "token_embedding": np.ascontiguousarray(data["token_embedding"], dtype=np.float32),
        "voice_ids": (),
    }
    if bundle["tts_mode"] == "preset_voices":
        all_voices = tuple(str(v) for v in data["voice_ids"])
        if any(not name or "," in name for name in all_voices):
            raise ValueError(f"voice_ids must be non-empty names without commas: {all_voices}")
        for key in ("speaker_embeds", "speaker_vecs"):
            if data[key].shape[0] != len(all_voices):
                raise ValueError(f"{key} has {data[key].shape[0]} rows for {len(all_voices)} voice_ids")
        lookup, listing = voice_lookup(all_voices)
        requested = range(len(all_voices)) if voices_filter is None or len(voices_filter) == 0 else voices_filter
        rows = [resolve_voice_index(item, lookup, listing) for item in requested]
        if len(set(rows)) != len(rows):
            raise ValueError(f"Duplicate preset voices requested: {[all_voices[i] for i in rows]}")
        out["voice_ids"] = tuple(all_voices[i] for i in rows)
        out["speaker_embeds"] = np.ascontiguousarray(data["speaker_embeds"][rows], dtype=np.float32)
        out["speaker_vecs"] = np.ascontiguousarray(data["speaker_vecs"][rows], dtype=np.float32)
    return out


def _slice_last_token(g: onnx.GraphProto, normed: str) -> None:
    """Cut the last position before the final layer's o_proj, so prefill runs o_proj, the last MLP and the final
    RMSNorm on one row. Everything after the last attention is position-wise:
        h = x + o_proj(attn);  y = h + mlp(norm(h));  normed = final_norm(y)
    """
    prod = {o: n for n in g.node for o in n.output}
    consts = {t.name for t in g.initializer} | {n.output[0] for n in g.node if n.op_type == "Constant"}
    data = lambda n: [i for i in n.input if i and i not in consts]

    def climb(start: str, stop) -> tuple[onnx.NodeProto | None, list[onnx.NodeProto]]:
        """Walk producers from `start` until `stop(node)`; return that node and the nodes passed on the way."""
        region, frontier = [], [start]
        while frontier:
            n = prod.get(frontier.pop())
            if n is None or any(n is r for r in region):
                continue
            if stop(n):
                return n, region
            region.append(n)
            frontier.extend(data(n))
        return None, region

    is_add = lambda n: n.op_type == "Add" and len(data(n)) == 2
    fin, norm_region = climb(normed, is_add)
    if fin is None:
        raise RuntimeError("final residual Add not found")
    # y = h + mlp(h): h is the input of `fin` that its other input's branch reads back.
    h_in, mlp_out = next(((a, b) for a, b in (fin.input, fin.input[::-1])
                          if climb(b, lambda n, a=a: a in n.input)[0] is not None), (None, None))
    h_add = prod.get(h_in) if h_in else None
    if h_add is None or not is_add(h_add):
        raise RuntimeError("last layer attention residual Add not found")
    _, mlp_region = climb(mlp_out, lambda n: h_in in n.output)
    o_proj = next((prod[i] for i in h_add.input if i in prod and prod[i].op_type == "MatMul" and prod[i].input[1] in consts), None)
    readers = lambda t: [n for n in g.node if t in n.input]
    inside = lambda nodes, region: all(any(n is r for r in region) for n in nodes)
    if (o_proj is None or not inside(readers(fin.output[0]), norm_region)
            or not inside(readers(h_in), mlp_region + [fin]) or readers(o_proj.output[0]) != [h_add]):
        raise RuntimeError("last layer tail is not position-wise (o_proj -> residual -> MLP -> final norm)")
    residual = next(i for i in h_add.input if i != o_proj.output[0])
    for node, k, tag in ((o_proj, 0, "attn"), (h_add, list(h_add.input).index(residual), "residual")):
        g.node.insert(list(g.node).index(node), H.make_node(
            "Slice", [node.input[k], "hj_neg1", "hj_i64_big", "hj_axis1"], [f"hj_last_{tag}"]))
        node.input[k] = f"hj_last_{tag}"
    for n in g.node:
        n.input[:] = ["hidden" if i == normed else i for i in n.input]
        n.output[:] = ["hidden" if o == normed else o for o in n.output]


def sampler_nodes(head_weight: str) -> list[onnx.NodeProto]:
    """Keep nucleus sampling off the greedy path without changing stochastic draws."""
    n = H.make_node
    stochastic = [
        n("Gather", ["ctrl", "hj_c1"], ["hj_top_p"]),
        n("Max", ["hj_temp", "hj_eps"], ["hj_temp_s"]),
        n("Div", ["hj_l2", "hj_temp_s"], ["hj_l3"]),
        n("Softmax", ["hj_l3"], ["hj_probs"], axis=0),
        n("Sub", ["hj_f1", "hj_top_p"], ["hj_tail"]),
        n("Mul", ["hj_tail", "hj_half_inv_v"], ["hj_thr"]),
        n("GreaterOrEqual", ["hj_probs", "hj_thr"], ["hj_cand_m"]),
        n("Compress", ["hj_probs", "hj_cand_m"], ["hj_cprobs"], axis=0),
        n("Compress", ["hj_vocab_idx", "hj_cand_m"], ["hj_cidx"], axis=0),
        n("Shape", ["hj_cprobs"], ["hj_ck"]),
        n("TopK", ["hj_cprobs", "hj_ck"], ["hj_sorted", "hj_corder"], axis=0, largest=1, sorted=1),
        n("Gather", ["hj_cidx", "hj_corder"], ["hj_order"], axis=0),
        n("CumSum", ["hj_sorted", "hj_c0"], ["hj_excl"], exclusive=1),
        n("LessOrEqual", ["hj_excl", "hj_top_p"], ["hj_keep_sorted"]),
        n("ScatterElements", ["hj_false_v", "hj_order", "hj_keep_sorted"], ["hj_keep"], axis=0),
        n("Where", ["hj_keep", "hj_probs", "hj_f0"], ["hj_pk"]),
        n("CumSum", ["hj_pk", "hj_c0"], ["hj_cdf_excl"], exclusive=1),
        n("ReduceSum", ["hj_pk", "hj_axis0"], ["hj_total"], keepdims=1),
        n("Mul", ["uniform", "hj_total"], ["hj_target"]),
        n("LessOrEqual", ["hj_cdf_excl", "hj_target"], ["hj_below"]),
        n("Greater", ["hj_pk", "hj_f0"], ["hj_live"]),
        n("And", ["hj_below", "hj_live"], ["hj_cand"]),
        n("Cast", ["hj_cand"], ["hj_cand_i32"], to=TP.INT32),
        n("ArgMax", ["hj_cand_i32"], ["hj_drawn"], axis=0, keepdims=1, select_last_index=1),
    ]
    greedy = H.make_graph(
        [n("ArgMax", ["hj_l2"], ["hj_greedy"], axis=0, keepdims=1)],
        "hojo_greedy", [], [H.make_tensor_value_info("hj_greedy", TP.INT64, [1])],
    )
    sampled = H.make_graph(
        stochastic, "hojo_stochastic", [], [H.make_tensor_value_info("hj_drawn", TP.INT64, [1])],
    )
    return [
        n("MatMul", ["hidden", head_weight], ["hj_logits3"]),
        n("Reshape", ["hj_logits3", "hj_neg1"], ["hj_l"]),
        n("Gather", ["ctrl", "hj_c0"], ["hj_temp"]),
        n("Gather", ["ctrl", "hj_c2"], ["hj_rep"]),
        n("Greater", ["hj_l", "hj_f0"], ["hj_pos_l"]),
        n("Div", ["hj_l", "hj_rep"], ["hj_ldiv"]),
        n("Mul", ["hj_l", "hj_rep"], ["hj_lmul"]),
        n("Where", ["hj_pos_l", "hj_ldiv", "hj_lmul"], ["hj_lpen"]),
        n("Where", ["seen", "hj_lpen", "hj_l"], ["hj_l2"]),
        n("LessOrEqual", ["hj_temp", "hj_f0"], ["hj_is_greedy"]),
        n("If", ["hj_is_greedy"], ["hj_tok64"], name="hj_choose_token",
          then_branch=greedy, else_branch=sampled),
        n("Reshape", ["hj_tok64", "hj_shape11"], ["hj_tok64_2d"]),
        n("Cast", ["hj_tok64_2d"], ["token"], to=TP.INT32),
        n("ScatterElements", ["seen", "hj_tok64", "hj_true1"], ["seen_out"], axis=0),
    ]


def _default_rope_theta(cfg: dict) -> float:
    """The optimizer bakes plain RoPE tables; reject scaled variants up front."""
    for key in ("rope_scaling", "rope_parameters"):
        rope = cfg.get(key) or {}
        if (any(rope[name] != "default" for name in ("rope_type", "type") if name in rope)
                or "factor" in rope and float(rope["factor"]) != 1.0):
            raise ValueError(f"config.{key}={rope!r}: only unscaled default RoPE is supported")
    return float(cfg["rope_theta"])


def metadata(
    bundle: dict[str, object],
    ids: dict[str, int],
    vocab: int,
    voices: tuple[str, ...],
    *,
    in_sample_rate: int = IN_SAMPLE_RATE,
    out_sample_rate: int = OUT_SAMPLE_RATE,
    in_audio_dtype: str = IN_AUDIO_DTYPE,
    out_audio_dtype: str = OUT_AUDIO_DTYPE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict[str, object]:
    cfg = bundle["config"]
    tts_mode = str(bundle["tts_mode"])
    meta: dict[str, object] = {
        "model": bundle["stem"],
        "tts_mode": tts_mode,
        "graph_layout": "hojo_lm_vocoder_v1",
        "sample_rate": out_sample_rate,
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "in_sample_rate": in_sample_rate,
        "out_sample_rate": out_sample_rate,
        "in_audio_dtype": in_audio_dtype.upper(),
        "out_audio_dtype": out_audio_dtype.upper(),
        "hop_length": HOP,
        "n_fft": N_FFT,
        "vocab_size": vocab,
        "voices": ",".join(voices),
        "num_layers": int(cfg["num_hidden_layers"]),
        "num_heads": int(cfg["num_attention_heads"]),
        "kv_heads": int(cfg["num_key_value_heads"]),
        "head_dim": int(cfg["head_dim"]),
        "hidden_size": int(cfg["hidden_size"]),
        "max_positions": min(int(cfg["max_position_embeddings"]), int(max_seq_len)),
        "rope_theta": _default_rope_theta(cfg),
        "text_start_id": ids["[target_text_start]"],
        "text_end_id": ids["[target_text_end]"],
        "speech_start_id": ids["[target_speech_start]"],
        "speech_end_id": ids["[target_speech_end]"],
        "audio_token_first": ids["audio_token_first"],
        "audio_token_last": ids["audio_token_last"],
        "model_file_name_lm": LM_MODEL_NAME,
        "model_file_name_vocoder": VOCODER_MODEL_NAME,
        "tokenizer_file": TOKENIZER_FILE_NAME,
    }
    if tts_mode == "preset_voices":
        meta.update({
            "spk_start_id": ids["[spk_start]"],
            "spk_end_id": ids["[spk_end]"],
            "spk_emb0_id": ids["[spk_emb_0]"],
            "spk_slots": ids["spk_slots"],
        })
    else:
        meta.update({
            "ref_text_start_id": ids["[ref_text_start]"],
            "ref_text_end_id": ids["[ref_text_end]"],
            "ref_speech_start_id": ids["[ref_speech_start]"],
            "ref_speech_end_id": ids["[ref_speech_end]"],
            "space_id": ids["space_id"],
            "model_file_name_reference_preprocess": REF_PREPROCESS_MODEL_NAME,
        })
    return meta


def upgrade_reduce_ops(model: onnx.ModelProto) -> None:
    """Opset 18+ moved ReduceMean axes from attribute to input."""
    g = model.graph
    for i, node in enumerate(g.node):
        if node.op_type != "ReduceMean":
            continue
        axes = next((a for a in node.attribute if a.name == "axes"), None)
        if axes is None:
            continue
        name = f"hj_axes_{i}"
        g.initializer.append(const(name, list(axes.ints), np.int64))
        node.attribute.remove(axes)
        node.input.append(name)


def ensure_f16_kv_io(model: onnx.ModelProto) -> None:
    """Force Hojo_LM KV cache I/O (past_key_values.* / present.*) to float16 and Cast back to float32 for compute."""
    g = model.graph
    kv_inputs = [v for v in g.input if v.name.startswith("past_key_values.")]
    kv_outputs = [v for v in g.output if v.name.startswith("present.")]
    if not kv_inputs or len(kv_inputs) != len(kv_outputs):
        raise ValueError("Hojo_LM graph is missing matching past_key_values.* inputs and present.* outputs")
    for v in kv_inputs + kv_outputs:
        v.type.tensor_type.elem_type = TP.FLOAT16
        d = v.type.tensor_type.shape.dim
        if d and not d[0].dim_value:
            d[0].Clear()
            d[0].dim_value = 1

    for node in list(g.node):
        if node.op_type == "Concat" and len(node.input) == 1 and node.input[0].startswith("past_key_values."):
            src_name, out_name = node.input[0], node.output[0]
            for consumer in g.node:
                consumer.input[:] = [src_name if x == out_name else x for x in consumer.input]
            g.node.remove(node)

    prod = {out: node for node in g.node for out in node.output}
    num_layers = len(kv_inputs) // 2
    for layer in range(num_layers):
        for kind in ("key", "value"):
            past_name = f"past_key_values.{layer}.{kind}"
            present_name = f"present.{layer}.{kind}"
            cat = prod.get(present_name)
            if cat is None or cat.op_type != "Concat" or len(cat.input) != 2 or cat.input[0] != past_name:
                raise RuntimeError(f"Expected 2-input Concat({past_name}, new) -> {present_name} in layer {layer}")
            new_src = cat.input[1]
            new_prod = prod.get(new_src)
            if new_prod is None or new_prod.op_type != "Cast" or not any(
                a.name == "to" and a.i == TP.FLOAT16 for a in new_prod.attribute
            ):
                new_f16 = f"hj_l{layer}_{kind}_new_f16"
                cat_idx = list(g.node).index(cat)
                g.node.insert(
                    cat_idx,
                    H.make_node("Cast", [new_src], [new_f16], to=TP.FLOAT16, name=f"hj_l{layer}_{kind}_to_f16"),
                )
                cat.input[1] = new_f16

            compute_consumers = [
                c for c in g.node if c is not cat and c.op_type != "Shape" and present_name in c.input
            ]
            if compute_consumers and not (
                len(compute_consumers) == 1
                and compute_consumers[0].op_type == "Cast"
                and any(a.name == "to" and a.i == TP.FLOAT for a in compute_consumers[0].attribute)
            ):
                present_f32 = f"hj_l{layer}_{kind}_f32"
                for c in compute_consumers:
                    c.input[:] = [present_f32 if x == present_name else x for x in c.input]
                cat_idx = list(g.node).index(cat)
                g.node.insert(
                    cat_idx + 1,
                    H.make_node("Cast", [present_name], [present_f32], to=TP.FLOAT, name=f"hj_l{layer}_{kind}_to_f32"),
                )


def build_lm(
    bundle: dict[str, object],
    voices: dict[str, object],
    ids: dict[str, int],
    *,
    meta: dict[str, object] | None = None,
) -> onnx.ModelProto:
    model = load_fp32(bundle["llm_path"])
    tts_mode = str(bundle["tts_mode"])
    g = model.graph
    head = next(n for n in g.node if "logits" in n.output)
    head_weight = head.input[1]
    g.node.remove(head)
    tok_emb = voices["token_embedding"]
    vocab, hidden_size = (int(d) for d in tok_emb.shape)
    kept_inputs = [i for i in g.input if i.name not in ("inputs_embeds", "position_ids")]
    del g.input[:]
    front_inputs = [H.make_tensor_value_info("input_ids", TP.INT32, [1, "sequence"])]
    if tts_mode == "preset_voices":
        front_inputs.append(H.make_tensor_value_info("voice", TP.INT32, []))
    front_inputs.extend([
        H.make_tensor_value_info("ctrl", TP.FLOAT, [3]),
        H.make_tensor_value_info("uniform", TP.FLOAT, [1]),
        H.make_tensor_value_info("seen", TP.BOOL, [vocab]),
    ])
    g.input.extend(front_inputs + kept_inputs)
    presents = [o for o in g.output if o.name.startswith("present.")]
    del g.output[:]
    g.output.extend([
        H.make_tensor_value_info("token", TP.INT32, [1, 1]),
        H.make_tensor_value_info("hidden", TP.FLOAT, [1, 1, hidden_size]),
        H.make_tensor_value_info("seen_out", TP.BOOL, [vocab]),
    ] + presents)
    inits = [const("hj_true1", [True], np.bool_)]
    if tts_mode == "preset_voices":
        slots = int(voices["speaker_embeds"].shape[1])
        if slots != ids["spk_slots"] or voices["speaker_embeds"].shape[2] != hidden_size:
            raise ValueError(f"speaker_embeds {voices['speaker_embeds'].shape} does not match "
                             f"{ids['spk_slots']} [spk_emb_*] tokens x hidden {hidden_size}")
        inits.extend([
            const("hj_voice_offsets", vocab - ids["[spk_emb_0]"] + np.arange(
                len(voices["voice_ids"]), dtype=np.int32) * slots),
            const("hj_spk0", ids["[spk_emb_0]"], np.int32),
            const("hj_i32_0", 0, np.int32),
            const("hj_i32_slots", slots, np.int32),
        ])
        tok_emb = np.concatenate((tok_emb, voices["speaker_embeds"].reshape(-1, hidden_size)), axis=0)
    inits.append(const("hj_token_embedding", tok_emb))
    inits.extend([
        const("hj_axis0", [0], np.int64),
        const("hj_i64_1", 1, np.int64),
        const("hj_vocab_idx", np.arange(vocab, dtype=np.int32)),
        const("hj_false_v", np.zeros(vocab), np.bool_),
        const("hj_half_inv_v", 0.5 / vocab, np.float32),
        const("hj_f1", 1.0, np.float32),
        const("hj_neg1", [-1], np.int64),
        const("hj_i64_big", [np.iinfo(np.int64).max], np.int64),
        const("hj_axis1", [1], np.int64),
        const("hj_shape11", [1, 1], np.int64),
        const("hj_f0", 0.0, np.float32),
        const("hj_eps", 1e-6, np.float32),
        const("hj_c0", 0, np.int32),
        const("hj_c1", 1, np.int32),
        const("hj_c2", 2, np.int32),
    ])
    g.initializer.extend(inits)
    if tts_mode == "preset_voices":
        embed_nodes = [
            H.make_node("Sub", ["input_ids", "hj_spk0"], ["hj_slot"]),
            H.make_node("GreaterOrEqual", ["hj_slot", "hj_i32_0"], ["hj_m1"]),
            H.make_node("Less", ["hj_slot", "hj_i32_slots"], ["hj_m2"]),
            H.make_node("And", ["hj_m1", "hj_m2"], ["hj_m"]),
            H.make_node("Gather", ["hj_voice_offsets", "voice"], ["hj_voice_offset"]),
            H.make_node("Add", ["input_ids", "hj_voice_offset"], ["hj_voice_ids"]),
            H.make_node("Where", ["hj_m", "hj_voice_ids", "input_ids"], ["hj_embedding_ids"]),
            H.make_node("Gather", ["hj_token_embedding", "hj_embedding_ids"], ["inputs_embeds"]),
        ]
    else:
        embed_nodes = [H.make_node("Gather", ["hj_token_embedding", "input_ids"], ["inputs_embeds"])]
    pos_nodes = [
        H.make_node("Shape", ["past_key_values.0.key"], ["hj_past_len1"], start=2, end=3),
        H.make_node("Shape", ["input_ids"], ["hj_seq1"], start=1, end=2),
        H.make_node("Add", ["hj_past_len1", "hj_seq1"], ["hj_end1"]),
        H.make_node("Squeeze", ["hj_past_len1"], ["hj_past_len"]),
        H.make_node("Squeeze", ["hj_end1"], ["hj_end"]),
        H.make_node("Range", ["hj_past_len", "hj_end", "hj_i64_1"], ["hj_pos"]),
        H.make_node("Unsqueeze", ["hj_pos", "hj_axis0"], ["position_ids"]),
    ]
    for node in reversed(embed_nodes + pos_nodes):
        g.node.insert(0, node)
    ensure_f16_kv_io(model)
    _slice_last_token(g, head.input[0])
    g.node.extend(sampler_nodes(head_weight))
    for o in model.opset_import:
        if o.domain in ("", "ai.onnx"):
            o.version = max(18, int(OPSET))
    meta_dict = meta or metadata(bundle, ids, vocab, voices["voice_ids"])
    H.set_model_props(model, {k: str(v) for k, v in meta_dict.items()})
    return model


def prefix(model: onnx.ModelProto, tag: str, keep: set[str]) -> None:
    """Rename every internal name of a subgraph so two graphs can be merged without collisions."""
    g = model.graph
    ren = lambda s: s if (not s or s in keep) else f"{tag}/{s}"
    for t in g.initializer:
        t.name = ren(t.name)
    for node in g.node:
        node.name = ren(node.name)
        node.input[:] = [ren(s) for s in node.input]
        node.output[:] = [ren(s) for s in node.output]
    for v in list(g.input) + list(g.output) + list(g.value_info):
        v.name = ren(v.name)


def istft_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real/imag inverse-rfft bases with the Hann window folded in, and window^2 for 'same' padding."""
    bins = N_FFT // 2 + 1
    window = np.hanning(N_FFT + 1)[:-1]
    k = np.arange(bins)[:, None]
    t = np.arange(N_FFT)[None, :]
    scale = np.full((bins, 1), 2.0 / N_FFT)
    scale[0] = scale[-1] = 1.0 / N_FFT
    ang = 2.0 * np.pi * k * t / N_FFT
    cos_b = (scale * np.cos(ang) * window).astype(np.float32)
    sin_b = (-scale * np.sin(ang) * window).astype(np.float32)
    sin_b[0] = 0.0
    sin_b[-1] = 0.0
    return cos_b, sin_b, (window ** 2).astype(np.float32)


def _scalar_value(g: onnx.GraphProto, name: str) -> float | None:
    for t in g.initializer:
        if t.name == name:
            a = NH.to_array(t)
            return float(a.ravel()[0]) if a.size == 1 else None
    for node in g.node:
        if node.op_type == "Constant" and node.output[0] == name and node.attribute[0].type == AttributeProto.TENSOR:
            a = NH.to_array(node.attribute[0].t)
            return float(a.ravel()[0]) if a.size == 1 else None
    return None


def _simplify_merged_bits_80m(dec: onnx.ModelProto) -> None:
    """Replace the merged 80M decoder's sign chain (x > 0 -> Where(+1, -1) -> Cast/Gather/Unsqueeze) by one Where."""
    g = dec.graph
    cons: dict[str, list[onnx.NodeProto]] = {}
    for node in g.node:
        for i in node.input:
            cons.setdefault(i, []).append(node)
    greaters = [nd for nd in g.node if nd.op_type == "Greater" and _scalar_value(g, nd.input[1]) == 0.0
                and [c.op_type for c in cons.get(nd.output[0], [])] == ["Where"]]
    if len(greaters) != 1:
        raise RuntimeError(f"expected one fine_local sign quantizer in the 80M decoder, found {len(greaters)}")
    gt = greaters[0]
    tail = cons[gt.output[0]][0]
    prod = {output: node for node in g.node for output in node.output}
    positive, negative = (prod.get(value) for value in tail.input[1:])
    if (positive is None or negative is None or positive.op_type != "ConstantOfShape" or negative.op_type != "Neg"
            or negative.input[0] != positive.output[0]
            or NH.to_array(next(attribute.t for attribute in positive.attribute if attribute.name == "value")).item() != 1.0):
        raise ValueError("80M sign quantizer must select +1 and -1")
    while [c.op_type for c in cons.get(tail.output[0], [])] in (["Cast"], ["Gather"], ["Unsqueeze"], ["Squeeze"]):
        nxt = cons[tail.output[0]][0]
        if nxt.op_type == "Cast" and next(attribute.i for attribute in nxt.attribute if attribute.name == "to") != TP.FLOAT:
            raise ValueError("80M sign chain must remain float32")
        if nxt.op_type == "Gather":
            axis = next((attribute.i for attribute in nxt.attribute if attribute.name == "axis"), 0)
            if axis != 0 or _scalar_value(g, nxt.input[1]) != 0.0:
                raise ValueError("80M sign chain must select batch row zero")
        if nxt.op_type in ("Unsqueeze", "Squeeze") and _scalar_value(g, nxt.input[1]) != 0.0:
            raise ValueError("80M sign chain must reshape only the batch axis")
        tail = nxt
    for c in cons.get(tail.output[0], []):
        c.input[:] = ["bits" if i == tail.output[0] else i for i in c.input]
    g.node.insert(list(g.node).index(gt) + 1, H.make_node("Where", [gt.output[0], "vc_f1", "vc_fm1"], ["bits"], name="vc_bits"))
    dce(dec)
    del g.value_info[:]


def node_inputs(node: onnx.NodeProto) -> set[str]:
    """Include values captured from outer scopes by ONNX control-flow subgraphs."""
    inputs = set(node.input)
    for attribute in node.attribute:
        graphs = [attribute.g] if attribute.type == AttributeProto.GRAPH else attribute.graphs
        for graph in graphs:
            defined = ({value.name for value in graph.input} | {tensor.name for tensor in graph.initializer}
                       | {output for child in graph.node for output in child.output})
            used = {value.name for value in graph.output} | {name for child in graph.node for name in node_inputs(child)}
            inputs.update(used - defined)
    return inputs


def remap_node_inputs(node: onnx.NodeProto, remap: dict[str, str]) -> None:
    """Rewrite captured values without renaming subgraph-local definitions."""
    node.input[:] = [remap.get(name, name) for name in node.input]
    for attribute in node.attribute:
        graphs = [attribute.g] if attribute.type == AttributeProto.GRAPH else attribute.graphs
        for graph in graphs:
            local = ({value.name for value in graph.input} | {tensor.name for tensor in graph.initializer}
                     | {output for child in graph.node for output in child.output})
            captured = {name: replacement for name, replacement in remap.items() if name not in local}
            for child in graph.node:
                remap_node_inputs(child, captured)
            for output in graph.output:
                output.name = captured.get(output.name, output.name)


def dce(model: onnx.ModelProto) -> int:
    """Drop nodes whose outputs are never read, then unreferenced initializers; returns removed node count."""
    g = model.graph
    outs = {o.name for o in g.output}
    removed = 0
    while True:
        used = {i for n in g.node for i in node_inputs(n)} | outs
        keep = [n for n in g.node if any(o in used for o in n.output if o)]
        if len(keep) == len(g.node):
            break
        removed += len(g.node) - len(keep)
        del g.node[:]
        g.node.extend(keep)
    used = {i for n in g.node for i in node_inputs(n)} | outs
    inits = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(inits)
    return removed


def int32_gather_indices(model: onnx.ModelProto) -> int:
    """Narrow constant Gather indices, not shape tensors or dynamic TopK/ArgMax outputs."""
    graph = model.graph
    values = {tensor.name: tensor for tensor in graph.initializer}
    values.update({node.output[0]: node.attribute[0].t for node in graph.node
                   if node.op_type == "Constant" and node.attribute[0].type == AttributeProto.TENSOR})
    converted: dict[str, str] = {}
    count = 0
    for node in graph.node:
        if node.op_type != "Gather":
            continue
        name = node.input[1]
        tensor = values.get(name)
        if tensor is None or tensor.data_type != TP.INT64:
            continue
        array = NH.to_array(tensor)
        bounds = np.iinfo(np.int32)
        if array.size and (array.min() < bounds.min or array.max() > bounds.max):
            continue
        if name not in converted:
            narrowed = name + "_gather_i32"
            if narrowed in values:
                raise ValueError(f"Gather initializer name collision: {narrowed}")
            narrowed_tensor = const(narrowed, array, np.int32)
            graph.initializer.append(narrowed_tensor)
            values[narrowed] = narrowed_tensor
            converted[name] = narrowed
        node.input[1] = converted[name]
        count += 1
    dce(model)
    return count


def vocoder_context(model: onnx.ModelProto) -> tuple[int, int] | None:
    """Certify the finite context of causal-window attention and stride-one 1-D convolutions."""
    graph = model.graph
    producers = {output: node for node in graph.node for output in node.output}
    softmaxes = [node for node in graph.node if node.op_type == "Softmax"]
    if not softmaxes:
        return None
    history = 0
    for softmax in softmaxes:
        frontier, masks, visited = [softmax.input[0]], [], set()
        while frontier:
            value = frontier.pop()
            node = producers.get(value)
            if node is None or value in visited:
                continue
            visited.add(value)
            if node.op_type == "Add":
                frontier.extend(node.input)
            elif node.op_type == "Cast":
                frontier.append(node.input[0])
            elif node.op_type == "Where":
                masks.append(node)
        matched = False
        for mask in masks:
            negative, zero = (_scalar_value(graph, name) for name in mask.input[1:])
            condition = producers.get(mask.input[0])
            if (negative is None or negative > -1e30 or zero != 0.0
                    or condition is None or condition.op_type != "Or"):
                continue
            comparisons = [producers.get(name) for name in condition.input]
            greater = next((node for node in comparisons if node is not None and node.op_type == "Greater"), None)
            less = next((node for node in comparisons if node is not None and node.op_type == "Less"), None)
            if greater is None or less is None or less.input[0] != greater.input[0]:
                continue
            columns, rows = (producers.get(name) for name in greater.input)
            if (columns is None or rows is None or columns.op_type != "Unsqueeze" or rows.op_type != "Unsqueeze"
                    or columns.input[0] != rows.input[0]
                    or _scalar_value(graph, columns.input[1]) != 0 or _scalar_value(graph, rows.input[1]) != 1):
                continue
            positions = producers.get(columns.input[0])
            if (positions is None or positions.op_type != "Range"
                    or _scalar_value(graph, positions.input[0]) != 0
                    or _scalar_value(graph, positions.input[2]) != 1):
                continue
            value, offset = less.input[1], 0.0
            while value != rows.output[0]:
                lower = producers.get(value)
                if lower is None or lower.op_type not in ("Add", "Sub"):
                    break
                constant = _scalar_value(graph, lower.input[1])
                if constant is not None:
                    offset += constant if lower.op_type == "Add" else -constant
                    value = lower.input[0]
                elif lower.op_type == "Add" and (constant := _scalar_value(graph, lower.input[0])) is not None:
                    offset += constant
                    value = lower.input[1]
                else:
                    break
            if value != rows.output[0] or offset > 0 or not offset.is_integer():
                continue
            history += -int(offset)
            matched = True
            break
        if not matched:
            return None
    left = right = convolutions = 0
    for node in graph.node:
        if node.op_type == "ConvTranspose":
            return None
        if node.op_type == "LayerNormalization" and next(
                (attribute.i for attribute in node.attribute if attribute.name == "axis"), -1) != -1:
            return None
        if node.op_type != "Conv":
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        kernel = list(attributes["kernel_shape"].ints) if "kernel_shape" in attributes else []
        pads = list(attributes["pads"].ints) if "pads" in attributes else []
        if (len(kernel) != 1 or kernel[0] <= 0 or len(pads) != 2 or any(pad < 0 for pad in pads)
                or sum(pads) != kernel[0] - 1
                or "strides" in attributes and list(attributes["strides"].ints) != [1]
                or "dilations" in attributes and list(attributes["dilations"].ints) != [1]
                or "auto_pad" in attributes and attributes["auto_pad"].s not in (b"", b"NOTSET")):
            return None
        left += pads[0]
        right += pads[1]
        convolutions += 1
    if not convolutions:
        return None
    overlap = ((N_FFT - HOP) // 2 + HOP - 1) // HOP
    return history + left + overlap, right + overlap


def _normalize_merged_decoder_80m(dec: onnx.ModelProto) -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """Rename the merged 80M fine_local+fine_decode graph to the 40M fine// and dec// scopes and quantize bits once."""
    keep = {"hidden_states", "coarse_embeddings", "speaker_embedding", "valid_mask", "mag", "phase"}

    def ren(s: str) -> str:
        if not s or s in keep:
            return s
        if s.startswith("/fine_local/"):
            return "fine//" + s[len("/fine_local/"):]
        if s.startswith("/fine_decode/"):
            return "dec//" + s[len("/fine_decode/"):]
        # Root-scope names ("/Constant") would otherwise collide with "/fine_local/Constant".
        return "fine//seam" + s if s.startswith("/") else "fine/" + s

    for t in dec.graph.initializer:
        t.name = ren(t.name)
    for node in dec.graph.node:
        node.name = ren(node.name)
        node.input[:] = [ren(s) for s in node.input]
        node.output[:] = [ren(s) for s in node.output]
    for v in list(dec.graph.input) + list(dec.graph.output) + list(dec.graph.value_info):
        v.name = ren(v.name)
    _simplify_merged_bits_80m(dec)
    return list(dec.graph.node), list(dec.graph.initializer)


def _input_dim(model: onnx.ModelProto, name: str, axis: int) -> int:
    v = next(v for v in model.graph.input if v.name == name)
    d = int(v.type.tensor_type.shape.dim[axis].dim_value)
    if d <= 0:
        raise ValueError(f"{name} axis {axis} has no static size")
    return d


def build_vocoder(
    bundle: dict[str, object],
    voices: dict[str, object],
    ids: dict[str, int],
    *,
    out_sample_rate: int = OUT_SAMPLE_RATE,
    out_audio_dtype: str = OUT_AUDIO_DTYPE,
    meta: dict[str, object] | None = None,
) -> onnx.ModelProto:
    tts_mode = str(bundle["tts_mode"])
    cos_b, sin_b, win_sq = istft_basis()
    n = H.make_node
    pad = (N_FFT - HOP) // 2
    if HOP <= 0 or N_FFT % HOP or (N_FFT - HOP) % 2:
        raise ValueError("iSTFT requires a positive dividing hop and symmetric integer padding")
    out_dtype = out_audio_dtype.upper()
    if out_dtype not in _AUDIO_TP_DTYPES:
        raise ValueError(f"Unsupported OUT_AUDIO_DTYPE={out_audio_dtype!r}; choose from {sorted(_AUDIO_TP_DTYPES)}")

    if tts_mode == "preset_voices":
        fine = load_fp32(bundle["fine_path"])
        dec = load_fp32(bundle["dec_path"])
        prefix(fine, "fine", {"hidden_states", "coarse_embeddings", "speaker_embedding", "valid_mask", "binary_logits"})
        prefix(dec, "dec", {"bits", "mag", "phase"})
        if voices["speaker_vecs"].shape[1] != _input_dim(fine, "speaker_embedding", 1):
            raise ValueError(f"speaker_vecs {voices['speaker_vecs'].shape} does not match fine_local speaker_embedding")
        pre = [
            n("Sub", ["audio_tokens", "vc_audio_first"], ["vc_audio_indices"]),
            n("Gather", ["vc_token_embedding", "vc_audio_indices"], ["coarse_embeddings"]),
            n("Gather", ["vc_speaker_vecs", "voice"], ["speaker_embedding"]),
            n("Shape", ["audio_tokens"], ["vc_codes_shape"]),
            n("Expand", ["vc_true", "vc_codes_shape"], ["valid_mask"]),
        ]
        mid = [
            n("Squeeze", ["binary_logits", "vc_axis0"], ["vc_logits"]),
            n("Greater", ["vc_logits", "vc_f0"], ["vc_pos"]),
            n("Where", ["vc_pos", "vc_f1", "vc_fm1"], ["bits"]),
        ]
        body_nodes = list(fine.graph.node) + mid + list(dec.graph.node)
        # [N, 1, D] rows let Gather(voice) emit the [1, D] speaker_embedding directly.
        body_inits = [const("vc_speaker_vecs", voices["speaker_vecs"][:, None])] + list(fine.graph.initializer) + list(dec.graph.initializer)
        third_input = H.make_tensor_value_info("voice", TP.INT32, [])
    else:
        dec = load_fp32(bundle["dec_path"])
        spk_dim = _input_dim(dec, "speaker_embedding", 1)
        body_nodes, body_inits = _normalize_merged_decoder_80m(dec)
        pre = [
            n("Sub", ["audio_tokens", "vc_audio_first"], ["vc_audio_indices"]),
            n("Gather", ["vc_token_embedding", "vc_audio_indices"], ["coarse_embeddings"]),
            n("Shape", ["audio_tokens"], ["vc_codes_shape"]),
            n("Expand", ["vc_true", "vc_codes_shape"], ["valid_mask"]),
        ]
        third_input = H.make_tensor_value_info("speaker_embedding", TP.FLOAT, [1, spk_dim])

    prod = {output: node for node in dec.graph.node for output in node.output}
    weights = {tensor.name: tensor for tensor in dec.graph.initializer}
    frontier, seen, spectral_width = [value.name for value in dec.graph.output], set(), None
    while frontier and spectral_width is None:
        value = frontier.pop()
        node = prod.get(value)
        if node is None or value in seen:
            continue
        seen.add(value)
        if node.op_type == "MatMul" and node.input[1] in weights:
            spectral_width = int(weights[node.input[1]].dims[-1])
        else:
            frontier.extend(node.input)
    if spectral_width != N_FFT + 2:
        raise ValueError(f"decoder magnitude/phase head width {spectral_width} does not match n_fft={N_FFT}")
    if voices["token_embedding"].shape[1] != int(bundle["config"]["hidden_size"]):
        raise ValueError("token embedding width does not match config.hidden_size")

    raw_wav = "wav" if (out_sample_rate == MODEL_SAMPLE_RATE and out_dtype == "F32") else "vc_wav_24k"
    post = [
        n("Cos", ["phase"], ["vc_cos"]),
        n("Sin", ["phase"], ["vc_sin"]),
        n("Mul", ["mag", "vc_cos"], ["vc_re"]),
        n("Mul", ["mag", "vc_sin"], ["vc_im"]),
        n("MatMul", ["vc_cos_bT", "vc_re"], ["vc_fr_re"]),
        n("MatMul", ["vc_sin_bT", "vc_im"], ["vc_fr_im"]),
        n("Add", ["vc_fr_re", "vc_fr_im"], ["vc_frames"]),
        n("Unsqueeze", ["vc_frames", "vc_axis0"], ["vc_cols"]),
        n("Col2Im", ["vc_cols", "vc_out_size", "vc_block"], ["vc_ola"], strides=[1, HOP]),
        n("Expand", ["vc_win_sq", "vc_env_shape"], ["vc_env_cols"]),
        n("Col2Im", ["vc_env_cols", "vc_out_size", "vc_block"], ["vc_env"], strides=[1, HOP]),
        n("Div", ["vc_ola", "vc_env"], ["vc_y"]),
        n("Reshape", ["vc_y", "vc_neg1"], ["vc_flat"]),
        n("Slice", ["vc_flat", "vc_pad", "vc_neg_pad", "vc_axis0"], [raw_wav]),
    ]
    inits = [
        const("vc_token_embedding", voices["token_embedding"][ids["audio_token_first"]:ids["audio_token_last"] + 1]),
        const("vc_audio_first", ids["audio_token_first"], np.int32),
        const("vc_axis0", [0], np.int64),
        const("vc_true", True, np.bool_),
        const("vc_f0", 0.0, np.float32),
        const("vc_f1", 1.0, np.float32),
        const("vc_fm1", -1.0, np.float32),
        const("vc_cos_bT", np.ascontiguousarray(cos_b.T)),
        const("vc_sin_bT", np.ascontiguousarray(sin_b.T)),
        const("vc_win_sq", win_sq.reshape(1, N_FFT, 1)),
        const("vc_block", [1, N_FFT], np.int64),
        const("vc_one", [1], np.int64),
        const("vc_one_one", [1, N_FFT], np.int64),
        const("vc_hop", [HOP], np.int64),
        const("vc_nfft", [N_FFT], np.int64),
        const("vc_neg1", [-1], np.int64),
        const("vc_pad", [pad], np.int64),
        const("vc_neg_pad", [-pad], np.int64),
    ]
    curr_wav = raw_wav
    if out_sample_rate != MODEL_SAMPLE_RATE:
        resampled_wav = "wav" if out_dtype == "F32" else "vc_wav_resampled"
        inits.extend([
            const("vc_axes_01", [0, 1], np.int64),
            const("vc_empty_roi", [], np.float32),
            const("vc_out_scales", [1.0, 1.0, float(out_sample_rate) / float(MODEL_SAMPLE_RATE)], np.float32),
        ])
        post.extend([
            n("Unsqueeze", [curr_wav, "vc_axes_01"], ["vc_wav_3d"]),
            n("Resize", ["vc_wav_3d", "vc_empty_roi", "vc_out_scales"], ["vc_wav_rs3d"],
              mode="linear", coordinate_transformation_mode="half_pixel"),
            n("Reshape", ["vc_wav_rs3d", "vc_neg1"], [resampled_wav]),
        ])
        curr_wav = resampled_wav
    if out_dtype == "F16":
        post.append(n("Cast", [curr_wav], ["wav"], to=TP.FLOAT16))
    elif out_dtype == "INT16":
        inits.extend([
            const("vc_i16_scale", 32767.0, np.float32),
            const("vc_i16_min", -32768.0, np.float32),
            const("vc_i16_max", 32767.0, np.float32),
        ])
        post.extend([
            n("Mul", [curr_wav, "vc_i16_scale"], ["vc_wav_scaled"]),
            n("Clip", ["vc_wav_scaled", "vc_i16_min", "vc_i16_max"], ["vc_wav_clamped"]),
            n("Cast", ["vc_wav_clamped"], ["wav"], to=TP.INT16),
        ])
    size_nodes = [
        n("Shape", ["phase"], ["vc_F"], start=1, end=2),
        n("Sub", ["vc_F", "vc_one"], ["vc_Fm1"]),
        n("Mul", ["vc_Fm1", "vc_hop"], ["vc_len0"]),
        n("Add", ["vc_len0", "vc_nfft"], ["vc_len"]),
        n("Concat", ["vc_one", "vc_len"], ["vc_out_size"], axis=0),
        n("Concat", ["vc_one_one", "vc_F"], ["vc_env_shape"], axis=0),
    ]
    graph = H.make_graph(
        pre + body_nodes + size_nodes + post,
        "hojo_vocoder",
        [
            H.make_tensor_value_info("hidden_states", TP.FLOAT, [1, "frames", int(voices["token_embedding"].shape[1])]),
            H.make_tensor_value_info("audio_tokens", TP.INT32, [1, "frames"]),
            third_input,
        ],
        [H.make_tensor_value_info("wav", _AUDIO_TP_DTYPES[out_dtype], ["samples"])],
        initializer=inits + body_inits,
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", max(18, int(OPSET)))], producer_name="Export_Hojo_TTS")
    model.ir_version = 8
    vocab = int(voices["token_embedding"].shape[0])
    meta_dict = meta or metadata(
        bundle, ids, vocab, voices["voice_ids"],
        out_sample_rate=out_sample_rate, out_audio_dtype=out_audio_dtype,
    )
    H.set_model_props(model, {k: str(v) for k, v in meta_dict.items()})
    return model


def slaney_mel_filterbank(
    sr: int = MODEL_SAMPLE_RATE,
    n_fft: int = SPEAKER_MEL_N_FFT,
    n_mels: int = 128,
    fmin: float = SPEAKER_MEL_FMIN,
    fmax: float = SPEAKER_MEL_FMAX,
) -> np.ndarray:
    """Slaney-normalized mel filterbank [n_mels, n_fft // 2 + 1] matching librosa.filters.mel."""
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    fftfreqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1, dtype=np.float64)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


def build_reference_preprocess(
    bundle: dict[str, object],
    ids: dict[str, int],
    *,
    in_sample_rate: int = IN_SAMPLE_RATE,
    in_audio_dtype: str = IN_AUDIO_DTYPE,
    meta: dict[str, object] | None = None,
) -> onnx.ModelProto:
    """80M voice-cloning reference preprocess: prompt_audio -> (prompt_audio_ids, speaker_embedding)."""
    in_dtype = in_audio_dtype.upper()
    if in_dtype not in _AUDIO_TP_DTYPES:
        raise ValueError(f"Unsupported IN_AUDIO_DTYPE={in_audio_dtype!r}; choose from {sorted(_AUDIO_TP_DTYPES)}")

    enc = onnxslim.slim(load_fp32(bundle["enc_path"]))
    spk = onnxslim.slim(load_fp32(bundle["spk_path"]))
    spk_out = spk.graph.output[0].name
    spk_dim = int(spk.graph.output[0].type.tensor_type.shape.dim[1].dim_value)
    if (spk_dim != _input_dim(load_fp32(bundle["dec_path"]), "speaker_embedding", 1)
            or SPEAKER_MEL_N_FFT <= SPEAKER_MEL_HOP_SIZE or SPEAKER_EMB_SECONDS <= 0
            or not 0 <= SPEAKER_MEL_FMIN < SPEAKER_MEL_FMAX <= MODEL_SAMPLE_RATE / 2):
        raise ValueError("speaker encoder/decoder or mel preprocessing dimensions do not match")
    for nd in spk.graph.node:
        nd.output[:] = ["speaker_embedding" if o == spk_out else o for o in nd.output]
    prefix(enc, "enc", {"wav", "codes"})
    prefix(spk, "spk", {"mel", "speaker_embedding"})

    n = H.make_node
    pre_nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        const("rp_empty_roi", [], np.float32),
        const("rp_code_shape", [1, -1], np.int64),
        const("rp_code_offset", ids["audio_token_first"], np.int32),
    ]
    audio_f32 = "prompt_audio"
    if in_dtype == "F16":
        pre_nodes.append(n("Cast", ["prompt_audio"], ["rp_audio_f32"], to=TP.FLOAT))
        audio_f32 = "rp_audio_f32"
    elif in_dtype == "INT16":
        inits.append(const("rp_i16_inv", 1.0 / 32768.0, np.float32))
        pre_nodes.extend([
            n("Cast", ["prompt_audio"], ["rp_audio_cast"], to=TP.FLOAT),
            n("Mul", ["rp_audio_cast", "rp_i16_inv"], ["rp_audio_f32"]),
        ])
        audio_f32 = "rp_audio_f32"

    if in_sample_rate == CODEC_SAMPLE_RATE:
        pre_nodes.append(n("Identity", [audio_f32], ["wav"]))
    else:
        inits.append(const("rp_enc_scales", [1.0, 1.0, float(CODEC_SAMPLE_RATE) / float(in_sample_rate)], np.float32))
        pre_nodes.append(
            n("Resize", [audio_f32, "rp_empty_roi", "rp_enc_scales"], ["wav"],
              mode="linear", coordinate_transformation_mode="half_pixel")
        )

    audio_24k = audio_f32
    if in_sample_rate != MODEL_SAMPLE_RATE:
        inits.append(const("rp_spk_scales", [1.0, 1.0, float(MODEL_SAMPLE_RATE) / float(in_sample_rate)], np.float32))
        pre_nodes.append(
            n("Resize", [audio_f32, "rp_empty_roi", "rp_spk_scales"], ["rp_audio_24k"],
              mode="linear", coordinate_transformation_mode="half_pixel")
        )
        audio_24k = "rp_audio_24k"

    max_samples = int(round(SPEAKER_EMB_SECONDS * MODEL_SAMPLE_RATE))
    bins = SPEAKER_MEL_N_FFT // 2 + 1
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(SPEAKER_MEL_N_FFT, dtype=np.float64) / SPEAKER_MEL_N_FFT)).astype(np.float32)
    k = np.arange(bins, dtype=np.float64)[:, None]
    t = np.arange(SPEAKER_MEL_N_FFT, dtype=np.float64)[None, :]
    ang = 2.0 * np.pi * k * t / float(SPEAKER_MEL_N_FFT)
    stft_kernel = np.concatenate([(np.cos(ang) * win).astype(np.float32), (-np.sin(ang) * win).astype(np.float32)], axis=0)[:, None, :]
    refl_pad = (SPEAKER_MEL_N_FFT - SPEAKER_MEL_HOP_SIZE) // 2
    mel_fb = slaney_mel_filterbank(n_mels=_input_dim(spk, spk.graph.input[0].name, 2))[None, :, :]

    inits.extend([
        const("rp_zero_1d", [0], np.int64),
        const("rp_max_samples", [max_samples], np.int64),
        const("rp_axis2", [2], np.int64),
        const("rp_pad_z5", [0, 0, 0, 0, 0], np.int64),
        const("rp_f0", 0.0, np.float32),
        const("rp_refl_pads", [0, 0, refl_pad, 0, 0, refl_pad], np.int64),
        const("rp_stft_kernel", stft_kernel),
        const("rp_split_bins", [bins, bins], np.int64),
        const("rp_spec_eps", 1e-9, np.float32),
        const("rp_mel_fb", mel_fb),
        const("rp_mel_floor", 1e-5, np.float32),
    ])
    mel_nodes = [
        n("Slice", [audio_24k, "rp_zero_1d", "rp_max_samples", "rp_axis2"], ["rp_cropped"]),
        n("Shape", ["rp_cropped"], ["rp_crop_len"], start=2, end=3),
        n("Sub", ["rp_max_samples", "rp_crop_len"], ["rp_pad_need"]),
        n("Max", ["rp_pad_need", "rp_zero_1d"], ["rp_pad_right"]),
        n("Concat", ["rp_pad_z5", "rp_pad_right"], ["rp_pad_spec"], axis=0),
        n("Pad", ["rp_cropped", "rp_pad_spec", "rp_f0"], ["rp_wav_6s"], mode="constant"),
        n("Pad", ["rp_wav_6s", "rp_refl_pads"], ["rp_wav_refl"], mode="reflect"),
        n("Conv", ["rp_wav_refl", "rp_stft_kernel"], ["rp_stft"], strides=[SPEAKER_MEL_HOP_SIZE]),
        n("Split", ["rp_stft", "rp_split_bins"], ["rp_re", "rp_im"], axis=1),
        n("Mul", ["rp_re", "rp_re"], ["rp_re2"]),
        n("Mul", ["rp_im", "rp_im"], ["rp_im2"]),
        n("Add", ["rp_re2", "rp_im2"], ["rp_pow"]),
        n("Add", ["rp_pow", "rp_spec_eps"], ["rp_pow_eps"]),
        n("Sqrt", ["rp_pow_eps"], ["rp_mag"]),
        n("MatMul", ["rp_mel_fb", "rp_mag"], ["rp_mel_lin"]),
        n("Max", ["rp_mel_lin", "rp_mel_floor"], ["rp_mel_clamped"]),
        n("Log", ["rp_mel_clamped"], ["rp_mel_log"]),
        n("Transpose", ["rp_mel_log"], ["mel"], perm=[0, 2, 1]),
    ]
    code_post = [
        n("Reshape", ["codes", "rp_code_shape"], ["rp_codes_2d"]),
        n("Cast", ["rp_codes_2d"], ["rp_codes_i32"], to=TP.INT32),
        n("Add", ["rp_codes_i32", "rp_code_offset"], ["prompt_audio_ids"]),
    ]
    graph = H.make_graph(
        pre_nodes + list(enc.graph.node) + code_post + mel_nodes + list(spk.graph.node),
        "hojo_reference_preprocess",
        [H.make_tensor_value_info("prompt_audio", _AUDIO_TP_DTYPES[in_dtype], [1, 1, "samples"])],
        [
            H.make_tensor_value_info("prompt_audio_ids", TP.INT32, [1, "audio_tokens"]),
            H.make_tensor_value_info("speaker_embedding", TP.FLOAT, [1, spk_dim]),
        ],
        initializer=inits + list(enc.graph.initializer) + list(spk.graph.initializer),
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", max(18, int(OPSET)))], producer_name="Export_Hojo_TTS")
    model.ir_version = 8
    upgrade_reduce_ops(model)
    if meta:
        H.set_model_props(model, {k: str(v) for k, v in meta.items()})
    return model


def build_metadata_carrier(meta: dict[str, object]) -> onnx.ModelProto:
    graph = H.make_graph(
        [H.make_node("Identity", ["metadata_marker"], ["metadata_marker_out"], name="metadata_identity")],
        "hojo_metadata",
        [H.make_tensor_value_info("metadata_marker", TP.INT64, [1])],
        [H.make_tensor_value_info("metadata_marker_out", TP.INT64, [1])],
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", max(18, int(OPSET)))], producer_name="Export_Hojo_TTS")
    model.ir_version = 8
    set_sorted_metadata(model, meta)
    return model


def remove_redundant_casts(model: onnx.ModelProto) -> None:
    inferred = onnx.shape_inference.infer_shapes(model).graph
    types = {value.name: value.type.tensor_type.elem_type
             for value in list(inferred.input) + list(inferred.output) + list(inferred.value_info)
             if value.type.tensor_type.elem_type}
    types.update({tensor.name: tensor.data_type for tensor in inferred.initializer})
    outputs = {value.name for value in model.graph.output}
    remap, nodes = {}, []
    for node in model.graph.node:
        remap_node_inputs(node, remap)
        if node.op_type == "Cast":
            target = next(attribute.i for attribute in node.attribute if attribute.name == "to")
            if types.get(node.input[0]) == target:
                if node.output[0] in outputs:
                    node.op_type = "Identity"
                    del node.attribute[:]
                else:
                    remap[node.output[0]] = node.input[0]
                    continue
            types[node.output[0]] = target
        elif node.input and node.input[0] in types and node.op_type in (
            "SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "RMSNormalization",
            "QuickGelu", "Gelu", "SkipLayerNormalization", "Transpose", "Reshape", "Split", "Concat", "Slice",
        ):
            for out in node.output:
                if out:
                    types.setdefault(out, types[node.input[0]])
        nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    dce(model)


def build_models(
    src: Path,
    *,
    voices_filter: tuple[int | str, ...] | list[int | str] | None = VOICES,
    in_sample_rate: int = IN_SAMPLE_RATE,
    out_sample_rate: int = OUT_SAMPLE_RATE,
    in_audio_dtype: str = IN_AUDIO_DTYPE,
    out_audio_dtype: str = OUT_AUDIO_DTYPE,
    max_seq_len: int = MAX_SEQ_LEN,
) -> tuple[onnx.ModelProto, onnx.ModelProto, onnx.ModelProto | None, dict[str, object]]:
    """Build checked fp32 (LM, Vocoder, optional ReferencePreprocess, metadata) in memory."""
    bundle = discover_source_bundle(src)
    tts_mode = str(bundle["tts_mode"])
    ids = token_ids(bundle["src"], tts_mode)
    voices = load_voices(bundle["src"], voices_filter)
    vocab = int(voices["token_embedding"].shape[0])
    meta = metadata(
        bundle,
        ids,
        vocab,
        voices["voice_ids"],
        in_sample_rate=in_sample_rate,
        out_sample_rate=out_sample_rate,
        in_audio_dtype=in_audio_dtype,
        out_audio_dtype=out_audio_dtype,
        max_seq_len=max_seq_len,
    )
    lm = build_lm(bundle, voices, ids, meta=meta)
    upgrade_reduce_ops(lm)
    int32_gather_indices(lm)
    remove_redundant_casts(lm)
    onnx.checker.check_model(lm, full_check=True)

    vocoder = build_vocoder(
        bundle,
        voices,
        ids,
        out_sample_rate=out_sample_rate,
        out_audio_dtype=out_audio_dtype,
        meta=meta,
    )
    context = vocoder_context(vocoder)
    if context is None:
        print("[Export] Audio-frame streaming disabled: unrecognized vocoder context.")
    else:
        meta["vocoder_left_context_frames"], meta["vocoder_right_context_frames"] = context
        for model in (lm, vocoder):
            H.set_model_props(model, {key: str(value) for key, value in meta.items()})
    upgrade_reduce_ops(vocoder)
    int32_gather_indices(vocoder)
    remove_redundant_casts(vocoder)
    onnx.checker.check_model(vocoder, full_check=True)

    ref_preprocess = None
    if tts_mode == "voice_clone":
        ref_preprocess = build_reference_preprocess(
            bundle,
            ids,
            in_sample_rate=in_sample_rate,
            in_audio_dtype=in_audio_dtype,
            meta=meta,
        )
        int32_gather_indices(ref_preprocess)
        remove_redundant_casts(ref_preprocess)
        onnx.checker.check_model(ref_preprocess, full_check=True)
    return lm, vocoder, ref_preprocess, meta


def export_package(
    src: Path,
    out: Path,
    *,
    voices_filter: tuple[int | str, ...] | list[int | str] | None = VOICES,
    in_sample_rate: int = IN_SAMPLE_RATE,
    out_sample_rate: int = OUT_SAMPLE_RATE,
    in_audio_dtype: str = IN_AUDIO_DTYPE,
    out_audio_dtype: str = OUT_AUDIO_DTYPE,
    max_seq_len: int = MAX_SEQ_LEN,
    bundle_shared: bool = BUNDLE_SHARED_INITIALIZERS,
) -> dict[str, object]:
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    stage = staging_dir(out)
    try:
        meta = _export_into(
            src, stage,
            voices_filter=voices_filter, in_sample_rate=in_sample_rate, out_sample_rate=out_sample_rate,
            in_audio_dtype=in_audio_dtype, out_audio_dtype=out_audio_dtype, max_seq_len=max_seq_len,
            bundle_shared=bundle_shared,
        )
        replace_dir_atomic(stage, out)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    print(f"[Export] Exported {meta['model']} ({meta['tts_mode']}) to {out}")
    if meta["voices"]:
        print(f"[Export] Preset voices (int32 index -> name): {voice_lookup(meta['voices'].split(','))[1]}")
    return meta


def staging_dir(out: Path) -> Path:
    """Empty sibling folder of `out` (same filesystem, so the final swap is a rename) with normal permissions."""
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        mask = os.umask(0)
        os.umask(mask)
        stage.chmod(0o777 & ~mask)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return stage


def replace_dir_atomic(stage: Path, out: Path) -> None:
    """Exchange directories atomically on Linux; elsewhere retain a recoverable backup during publication."""
    old = out.with_name(f".{out.name}.old")
    if old.exists():
        if out.exists():
            raise FileExistsError(f"Recovery package already exists: {old}")
        os.replace(old, out)
    if out.exists() and sys.platform == "linux":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            at_fdcwd, rename_exchange = -100, 2
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            renameat2.restype = ctypes.c_int
            if renameat2(at_fdcwd, os.fsencode(stage), at_fdcwd, os.fsencode(out), rename_exchange) == 0:
                shutil.rmtree(stage, ignore_errors=True)
                return
            error = ctypes.get_errno()
            if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
                raise OSError(error, os.strerror(error), str(out))
    if out.exists():
        os.replace(out, old)
    try:
        os.replace(stage, out)
    except BaseException:
        if old.exists() and not out.exists():
            os.replace(old, out)
        raise
    shutil.rmtree(old, ignore_errors=True)


def _export_into(
    src: Path,
    out: Path,
    *,
    voices_filter: tuple[int | str, ...] | list[int | str] | None,
    in_sample_rate: int,
    out_sample_rate: int,
    in_audio_dtype: str,
    out_audio_dtype: str,
    max_seq_len: int,
    bundle_shared: bool,
) -> dict[str, object]:
    lm, vocoder, ref_preprocess, meta = build_models(
        src,
        voices_filter=voices_filter,
        in_sample_rate=in_sample_rate,
        out_sample_rate=out_sample_rate,
        in_audio_dtype=in_audio_dtype,
        out_audio_dtype=out_audio_dtype,
        max_seq_len=max_seq_len,
    )
    if bundle_shared:
        meta["shared_initializer_model_file"] = SHARED_MODEL_NAME
        meta["shared_initializer_data_file"] = SHARED_DATA_NAME
        for m in (lm, vocoder, ref_preprocess):
            if m is not None:
                H.set_model_props(m, {k: str(v) for k, v in meta.items()})

    lm_path = out / LM_MODEL_NAME
    voc_path = out / VOCODER_MODEL_NAME
    meta_path = out / METADATA_MODEL_NAME
    onnx.save(lm, str(lm_path))
    onnx.save(vocoder, str(voc_path))
    model_paths = [lm_path, voc_path]
    if ref_preprocess is not None:
        ref_path = out / REF_PREPROCESS_MODEL_NAME
        onnx.save(ref_preprocess, str(ref_path))
        model_paths.append(ref_path)
    meta_model = build_metadata_carrier(meta)
    onnx.save(meta_model, str(meta_path))
    (out / TOKENIZER_FILE_NAME).write_bytes((src / TOKENIZER_FILE_NAME).read_bytes())

    if bundle_shared:
        stats = bundle_shared_initializers(out, model_paths=model_paths, metadata=meta)
        print(
            f"[Shared weights] {stats['initializer_references']} references -> "
            f"{stats['unique_initializers']} unique tensors; deduplicated "
            f"{stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB."
        )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=Path(download_path).expanduser())
    parser.add_argument("--out", type=Path, default=ONNX_FOLDER)
    parser.add_argument("--voices", nargs="+", default=None)
    parser.add_argument("--bundle-shared", action="store_true", default=BUNDLE_SHARED_INITIALIZERS)
    parser.add_argument("--no-bundle-shared", dest="bundle_shared", action="store_false")
    parser.add_argument("--no-demo", action="store_true")
    args = parser.parse_args(argv)
    if DO_EXPORT:
        export_package(
            args.src,
            args.out,
            voices_filter=tuple(args.voices) if args.voices else VOICES,
            bundle_shared=args.bundle_shared,
        )
    if argv is None and not args.no_demo and (SCRIPT_DIR / "Inference_Hojo_TTS_ONNX.py").is_file():
        print("\nStart running the Hojo-TTS demo via Inference_Hojo_TTS_ONNX.py ...")
        return subprocess.call([
            sys.executable,
            str(SCRIPT_DIR / "Inference_Hojo_TTS_ONNX.py"),
            "--onnx-folder",
            str(args.out.expanduser().resolve()),
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

