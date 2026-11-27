"""Regression checks for float16/float32 stateful vocoder graph I/O."""

from __future__ import annotations

import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP, helper as H, numpy_helper as NH

from Streaming_Hojo import HOP, StatefulVocoder, stream_pcm_end


def session(graph: onnx.GraphProto) -> ort.InferenceSession:
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 20)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def vocoder(fine_type: int, decoder_type: int, *, clone: bool = False, pcm_type: int | None = None) -> StatefulVocoder:
    pcm_elem = decoder_type if pcm_type is None else pcm_type
    inputs = [H.make_tensor_value_info("hidden_states", fine_type, [1, "frames", 512]),
              H.make_tensor_value_info("audio_tokens", TP.INT32, [1, "frames"])]
    outputs = [H.make_tensor_value_info("bits", fine_type, [1, "frames", 512])]
    if clone:
        inputs.append(H.make_tensor_value_info("speaker_projection", fine_type, [512]))
        nodes = [H.make_node("Add", ["hidden_states", "speaker_projection"], ["bits"])]
    else:
        inputs.append(H.make_tensor_value_info("voice", TP.INT32, []))
        nodes = [H.make_node("Identity", ["hidden_states"], ["bits"])]
    for layer in range(4):
        for kind in ("key", "value"):
            past, present = f"fine_past.{layer}.{kind}", f"fine_present.{layer}.{kind}"
            inputs.append(H.make_tensor_value_info(past, fine_type, [1, 1, "past_frames", 512]))
            outputs.append(H.make_tensor_value_info(present, fine_type, [1, 1, "past_frames", 512]))
            nodes.append(H.make_node("Identity", [past], [present]))
    fine = session(H.make_graph(nodes, "fine", inputs, outputs))
    inputs = [H.make_tensor_value_info("bits", decoder_type, [1, "frames", 512]),
              H.make_tensor_value_info("is_first", TP.BOOL, []),
              H.make_tensor_value_info("is_final", TP.BOOL, []),
              H.make_tensor_value_info("ola_past", decoder_type, [1, 3 * HOP]),
              H.make_tensor_value_info("frames_past", TP.INT64, [1])]
    outputs = [H.make_tensor_value_info("pcm", pcm_elem, ["samples"]),
               H.make_tensor_value_info("pcm_finite", TP.BOOL, []),
               H.make_tensor_value_info("ola_present", decoder_type, [1, 3 * HOP]),
               H.make_tensor_value_info("frames_present", TP.INT64, [1])]
    tensors = [NH.from_array(np.array([value], np.int64), name)
               for name, value in (("frame_axis", 1), ("hop", HOP), ("delay", 8), ("crop", 720), ("zero", 0))]
    nodes = [H.make_node("Shape", ["bits"], ["bits_shape"]),
             H.make_node("Gather", ["bits_shape", "frame_axis"], ["chunk_frames"], axis=0),
             H.make_node("Add", ["frames_past", "chunk_frames"], ["frames_present"]),
             H.make_node("Mul", ["frames_present", "hop"], ["final_end"])]
    for tag, source in (("current", "frames_present"), ("previous", "frames_past")):
        nodes.extend([H.make_node("Sub", [source, "delay"], [tag + "_delayed"]),
                      H.make_node("Mul", [tag + "_delayed", "hop"], [tag + "_samples"]),
                      H.make_node("Sub", [tag + "_samples", "crop"], [tag + "_cropped"]),
                      H.make_node("Max", [tag + "_cropped", "zero"], [tag + "_end"])])
    nodes.extend([H.make_node("Where", ["is_final", "final_end", "current_end"], ["pcm_end"]),
                  H.make_node("Sub", ["pcm_end", "previous_end"], ["pcm_size"]),
                  H.make_node("ConstantOfShape", ["pcm_size"], ["pcm"],
                              value=H.make_tensor("pcm_fill", pcm_elem, [1], [0.25])),
                  H.make_node("Constant", [], ["pcm_finite"], value=H.make_tensor("finite", TP.BOOL, [], [True])),
                  H.make_node("Identity", ["ola_past"], ["ola_present"])])
    for layer in range(8):
        past, present = f"conv_past.{layer}", f"conv_present.{layer}"
        inputs.append(H.make_tensor_value_info(past, decoder_type, [1, 512, 6]))
        outputs.append(H.make_tensor_value_info(present, decoder_type, [1, 512, 6]))
        nodes.append(H.make_node("Identity", [past], [present]))
    decoder = session(H.make_graph(nodes, "decoder", inputs, outputs, tensors))
    voice = None if clone else ort.OrtValue.ortvalue_from_numpy(np.array(0, np.int32))
    return StatefulVocoder(fine, decoder, ort.RunOptions(), voice)


def inputs(frames: int, dtype: type = np.float32) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.linspace(-0.5, 0.5, frames * 512, dtype=np.float32).reshape(1, frames, 512).astype(dtype)
    return hidden, np.zeros((1, frames), np.int32)


class StreamingDtypeTests(unittest.TestCase):
    def test_float32_and_float16_full_utterance(self) -> None:
        for elem_type, dtype in ((TP.FLOAT, np.float32), (TP.FLOAT16, np.float16)):
            for hidden_dtype in (np.float32, np.float16):
                with self.subTest(graph_type=elem_type, hidden_type=hidden_dtype):
                    runtime = vocoder(elem_type, elem_type)
                    hidden, tokens = inputs(12, hidden_dtype)
                    output = np.full(12 * HOP + 3, -2.0, np.float32)
                    self.assertEqual(runtime.process(hidden, tokens, output, final=True), 12 * HOP)
                    np.testing.assert_array_equal(output[:12 * HOP], 0.25)
                    np.testing.assert_array_equal(output[12 * HOP:], -2.0)
                    np.testing.assert_array_equal(runtime.last_bits.numpy(), hidden.astype(dtype))
                    with self.assertRaises(RuntimeError):
                        runtime.process(hidden, tokens, output, final=True)

    def test_float16_states_with_float32_playback_pcm(self) -> None:
        runtime = vocoder(TP.FLOAT16, TP.FLOAT16, pcm_type=TP.FLOAT)
        hidden, tokens = inputs(16, np.float16)
        output = np.full(16 * HOP + 4, -2.0, np.float32)
        self.assertEqual(runtime.process(hidden, tokens, output, final=True), 16 * HOP)
        self.assertEqual(runtime._pcm_buffer.size, 0)
        np.testing.assert_array_equal(output[:16 * HOP], 0.25)
        np.testing.assert_array_equal(output[16 * HOP:], -2.0)

    def test_mixed_precision_state_and_bits_handoffs(self) -> None:
        for fine_type, decoder_type in ((TP.FLOAT, TP.FLOAT16), (TP.FLOAT16, TP.FLOAT)):
            with self.subTest(fine_type=fine_type, decoder_type=decoder_type):
                runtime = vocoder(fine_type, decoder_type)
                hidden, tokens = inputs(42)
                expected = stream_pcm_end(42, final=False)
                output = np.empty(expected, np.float32)
                self.assertEqual(runtime.process(hidden, tokens, output, final=False), expected)
                np.testing.assert_array_equal(output, 0.25)
                hidden, tokens = inputs(11)
                remaining = 53 * HOP - expected
                self.assertEqual(runtime.process(hidden, tokens, np.empty(remaining, np.float32), final=True), remaining)
                self.assertEqual(runtime.produced_samples, 53 * HOP)

    def test_float16_chunks_empty_flush_and_reset(self) -> None:
        runtime = vocoder(TP.FLOAT16, TP.FLOAT16)
        for frames in (11, 17):
            hidden, tokens = inputs(frames)
            expected = stream_pcm_end(runtime._audio_frames + frames, final=False) - runtime.produced_samples
            output = np.full(expected + 2, -2.0, np.float32)
            self.assertEqual(runtime.process(hidden, tokens, output, final=False), expected)
            np.testing.assert_array_equal(output[:expected], 0.25)
            np.testing.assert_array_equal(output[expected:], -2.0)
        retained = output.copy()
        hidden, tokens = inputs(0)
        remaining = 28 * HOP - runtime.produced_samples
        self.assertEqual(runtime.process(hidden, tokens, np.empty(remaining, np.float32), final=True), remaining)
        np.testing.assert_array_equal(output, retained)
        self.assertEqual(runtime.produced_samples, 28 * HOP)
        runtime.reset()
        hidden, tokens = inputs(1)
        self.assertEqual(runtime.process(hidden, tokens, np.empty(HOP, np.float32), final=True), HOP)

    def test_float16_cloning_speaker_conversion(self) -> None:
        runtime = vocoder(TP.FLOAT16, TP.FLOAT16, clone=True)
        hidden, tokens = inputs(12)
        output = np.empty(12 * HOP, np.float32)
        with self.assertRaises(RuntimeError):
            runtime.process(hidden, tokens, output, final=True)
        speaker = np.full(512, 0.125, np.float32)
        runtime.set_speaker(ort.OrtValue.ortvalue_from_numpy(speaker))
        runtime.process(hidden, tokens, output, final=True)
        np.testing.assert_array_equal(runtime.last_bits.numpy(), hidden.astype(np.float16) + speaker.astype(np.float16))

    def test_shared_initializer_prepack_exclusions_and_filtering(self) -> None:
        import tempfile
        from pathlib import Path
        from Shared_Weights import SharedInitializerStore, bundle_shared_initializers

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            norm_w = NH.from_array(np.ones(8, np.float16), "norm_gamma")
            proj_w = NH.from_array(np.eye(8, dtype=np.float16), "proj_weight")
            m1 = H.make_model(
                H.make_graph(
                    [H.make_node("RMSNormalization", ["x", "norm_gamma"], ["n"], axis=-1, epsilon=1e-5),
                     H.make_node("MatMul", ["n", "proj_weight"], ["y"])],
                    "m1",
                    [H.make_tensor_value_info("x", TP.FLOAT16, [1, 8])],
                    [H.make_tensor_value_info("y", TP.FLOAT16, [1, 8])],
                    [norm_w, proj_w],
                ),
                opset_imports=[H.make_opsetid("", 23)],
            )
            m1.ir_version = 8
            m2 = H.make_model(
                H.make_graph(
                    [H.make_node("MatMul", ["x", "proj_weight"], ["y"])],
                    "m2",
                    [H.make_tensor_value_info("x", TP.FLOAT16, [1, 8])],
                    [H.make_tensor_value_info("y", TP.FLOAT16, [1, 8])],
                    [proj_w],
                ),
                opset_imports=[H.make_opsetid("", 23)],
            )
            m2.ir_version = 8
            onnx.save(m1, root / "m1.onnx")
            onnx.save(m2, root / "m2.onnx")
            stats = bundle_shared_initializers(root, model_paths=[root / "m1.onnx", root / "m2.onnx"])
            self.assertEqual(stats["unique_initializers"], 2)
            self.assertEqual(stats["prepack_excluded_count"], 1)
            store = SharedInitializerStore(root / "Hojo_SharedInitializers.onnx")
            self.assertEqual(len(store.excluded), 1)
            self.assertEqual(len(store.ort_values), 1)
            so = ort.SessionOptions()
            self.assertEqual(store.attach_to(so, root / "m2.onnx"), 1)

    def test_packaged_text_tokenizer_matches_json(self) -> None:
        import tempfile
        from pathlib import Path
        from Advanced_Graphs import GOLDEN_PROMPTS, export_tokenizer_tables
        from Inference_Hojo_TTS_ONNX import BpeTokenizer

        tok_json = Path(__file__).resolve().parent / "HojoTTS_ONNX" / "tokenizer.json"
        if not tok_json.is_file():
            self.skipTest("HojoTTS_ONNX/tokenizer.json not present")
        with tempfile.TemporaryDirectory() as tmp:
            written = export_tokenizer_tables(tok_json, tmp)
            t_json = BpeTokenizer(tok_json)
            t_txt = BpeTokenizer(written["hojo_bpe.txt"])
            for prompt in GOLDEN_PROMPTS:
                self.assertEqual(t_txt.encode(prompt), t_json.encode(prompt))

    def test_fp32_stabilization_preserves_rope_add_f16(self) -> None:
        from Advanced_Graphs import stabilize_lm_fp32_attention_and_activations

        w = NH.from_array(np.eye(4, dtype=np.float16), "w_proj")
        model = H.make_model(
            H.make_graph(
                [
                    H.make_node("MatMul", ["x", "w_proj"], ["q"], name="weighted_q"),
                    H.make_node("Mul", ["q", "cos"], ["c"], name="hj_l0_a8_rope_cos"),
                    H.make_node("Mul", ["q", "sin"], ["s"], name="hj_l0_a8_rope_sin"),
                    H.make_node("Add", ["c", "s"], ["qr"], name="hj_l0_a8_rope_add"),
                    H.make_node("MatMul", ["qr", "k"], ["scores"], name="/lm/model/layers.0/self_attn/MatMul"),
                    H.make_node("Softmax", ["scores"], ["probs"], name="/lm/model/layers.0/self_attn/Softmax", axis=-1),
                    H.make_node("QuickGelu", ["probs"], ["y"], name="hj_l0_silu", domain="com.microsoft", alpha=1.0),
                ],
                "mini_lm",
                [
                    H.make_tensor_value_info("x", TP.FLOAT16, [1, 4]),
                    H.make_tensor_value_info("cos", TP.FLOAT16, [1, 4]),
                    H.make_tensor_value_info("sin", TP.FLOAT16, [1, 4]),
                    H.make_tensor_value_info("k", TP.FLOAT16, [4, 4]),
                ],
                [H.make_tensor_value_info("y", TP.FLOAT16, [1, 4])],
                [w],
            ),
            opset_imports=[H.make_opsetid("", 20), H.make_opsetid("com.microsoft", 1)],
        )
        model.ir_version = 8
        wrapped = stabilize_lm_fp32_attention_and_activations(model, "mini_lm")
        self.assertEqual(wrapped, 5)
        rope_add = next(n for n in model.graph.node if n.name == "hj_l0_a8_rope_add")
        self.assertEqual(list(rope_add.input), ["c", "s"])
        weighted_q = next(n for n in model.graph.node if n.name == "weighted_q")
        self.assertEqual(list(weighted_q.input), ["x", "w_proj"])
        self.assertEqual(stabilize_lm_fp32_attention_and_activations(model, "mini_lm"), 0)

    def test_shared_weights_alignment_compaction_zero_padding(self) -> None:
        import tempfile
        from pathlib import Path
        from Shared_Weights import SharedInitializerStore, bundle_shared_initializers

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            odd_f16 = NH.from_array(np.ones((5, 1), np.float16), "odd_f16")   # 10 bytes (2 mod 4)
            f32_mat = NH.from_array(np.eye(4, dtype=np.float32), "f32_mat")   # 64 bytes (0 mod 4)
            m1 = H.make_model(
                H.make_graph(
                    [H.make_node("Add", ["x", "odd_f16"], ["y"])],
                    "m1",
                    [H.make_tensor_value_info("x", TP.FLOAT16, [5, 1])],
                    [H.make_tensor_value_info("y", TP.FLOAT16, [5, 1])],
                    [odd_f16],
                ),
                opset_imports=[H.make_opsetid("", 20)],
            )
            m1.ir_version = 8
            m2 = H.make_model(
                H.make_graph(
                    [H.make_node("MatMul", ["x", "f32_mat"], ["y"])],
                    "m2",
                    [H.make_tensor_value_info("x", TP.FLOAT, [4, 4])],
                    [H.make_tensor_value_info("y", TP.FLOAT, [4, 4])],
                    [f32_mat],
                ),
                opset_imports=[H.make_opsetid("", 20)],
            )
            m2.ir_version = 8
            onnx.save(m1, root / "m1.onnx")
            onnx.save(m2, root / "m2.onnx")
            stats = bundle_shared_initializers(root, model_paths=[root / "m1.onnx", root / "m2.onnx"])
            self.assertEqual(stats["unique_bytes"], 74)  # 64 + 10, 0 padding bytes
            store = SharedInitializerStore(root / "Hojo_SharedInitializers.onnx")
            self.assertEqual(len(store.ort_values), 2)

    def test_inplaced_kv_handles_transposed_and_bnsh_caches(self) -> None:
        from Advanced_Graphs import convert_lm_to_inplaced_kv

        for transposed in (True, False):
            with self.subTest(transposed=transposed):
                k_dims = [1, 1, 4, "past"] if transposed else [1, 1, "past", 4]
                k_axis = 3 if transposed else 2
                k_new = [1, 1, 4, "seq"] if transposed else [1, 1, "seq", 4]
                nodes = [
                    H.make_node("Shape", ["past_key_values.0.key"], ["hj_past_len1"], start=k_axis, end=k_axis + 1),
                    H.make_node("Shape", ["input_ids"], ["hj_seq1"], start=1, end=2),
                    H.make_node("Add", ["hj_past_len1", "hj_seq1"], ["hj_end1"]),
                    H.make_node("Concat", ["past_key_values.0.key", "k_in"], ["present.0.key"], axis=k_axis),
                    H.make_node("Concat", ["past_key_values.0.value", "v_in"], ["present.0.value"], axis=2),
                    H.make_node("Identity", ["present.0.key"], ["k_use"]),
                    H.make_node("Identity", ["present.0.value"], ["v_use"]),
                ]
                if transposed:
                    nodes.insert(3, H.make_node("Transpose", ["k_raw"], ["k_in"], perm=[0, 2, 3, 1]))
                    k_input = H.make_tensor_value_info("k_raw", TP.FLOAT16, [1, "seq", 1, 4])
                else:
                    k_input = H.make_tensor_value_info("k_in", TP.FLOAT16, k_new)
                model = H.make_model(
                    H.make_graph(
                        nodes, "mini_kv",
                        [
                            H.make_tensor_value_info("input_ids", TP.INT32, [1, "seq"]),
                            k_input,
                            H.make_tensor_value_info("v_in", TP.FLOAT16, [1, 1, "seq", 4]),
                            H.make_tensor_value_info("past_key_values.0.key", TP.FLOAT16, k_dims),
                            H.make_tensor_value_info("past_key_values.0.value", TP.FLOAT16, [1, 1, "past", 4]),
                        ],
                        [
                            H.make_tensor_value_info("present.0.key", TP.FLOAT16, k_dims),
                            H.make_tensor_value_info("present.0.value", TP.FLOAT16, [1, 1, "past", 4]),
                            H.make_tensor_value_info("k_use", TP.FLOAT16, k_dims),
                            H.make_tensor_value_info("v_use", TP.FLOAT16, [1, 1, "past", 4]),
                        ],
                    ),
                    opset_imports=[H.make_opsetid("", 20)],
                )
                model.ir_version = 8
                H.set_model_props(model, {"num_layers": "1", "kv_heads": "1", "max_positions": "64", "head_dim": "4"})
                converted = convert_lm_to_inplaced_kv(model)
                kv_ops = [n for n in converted.graph.node if n.op_type == "KVCacheUpdate"]
                self.assertEqual(len(kv_ops), 2)
                transposes = [n for n in converted.graph.node if n.op_type == "Transpose"]
                # Pre-transposed F16 cache needs Transpose_3 back for MatMul(Q, K^T); BNSH cache must NOT double-transpose.
                self.assertEqual(len(transposes), 1 if transposed else 0)

    def test_unwrap_f16_boundaries_mixed_lm_precision(self) -> None:
        from Advanced_Graphs import unwrap_f16_neural_boundaries

        lm_f32 = H.make_model(
            H.make_graph([], "lm", [], [H.make_tensor_value_info("hidden", TP.FLOAT, [1, 1, 512])]),
            opset_imports=[H.make_opsetid("", 20)],
        )
        fine = H.make_model(
            H.make_graph(
                [], "fine",
                [H.make_tensor_value_info("hidden_states", TP.FLOAT16, [1, "frames", 512])],
                [H.make_tensor_value_info("bits", TP.FLOAT16, ["frames", 128])],
            ),
            opset_imports=[H.make_opsetid("", 20)],
        )
        dec = H.make_model(
            H.make_graph(
                [], "dec",
                [
                    H.make_tensor_value_info("bits", TP.FLOAT16, ["frames", 128]),
                    H.make_tensor_value_info("ola_past", TP.FLOAT16, [1, 1440]),
                ],
                [
                    H.make_tensor_value_info("ola_present", TP.FLOAT16, [1, 1440]),
                    H.make_tensor_value_info("pcm", TP.FLOAT, ["samples"]),
                ],
            ),
            opset_imports=[H.make_opsetid("", 20)],
        )
        with self.assertRaises(ValueError):
            unwrap_f16_neural_boundaries(lm_f32, fine, dec, require_lm_f16=True)
        unwrap_f16_neural_boundaries(lm_f32, fine, dec, require_lm_f16=False)

    def test_tokenizer_added_tokens_and_pinyin_overrides(self) -> None:
        from pathlib import Path
        from Inference_Hojo_TTS_ONNX import BpeTokenizer

        tok_json = Path(__file__).resolve().parent / "HojoTTS_ONNX" / "tokenizer.json"
        if not tok_json.is_file():
            self.skipTest("HojoTTS_ONNX/tokenizer.json not present")
        tok = BpeTokenizer(tok_json)
        ids = tok.encode("[0]你好<zh_ni3>[spk_emb_0] [1234] [not_a_token] <not_pinyin> 2026")
        self.assertEqual(ids[0], tok.token_id("[0]"))
        self.assertIn(tok.token_id("<zh_ni3>"), ids)
        self.assertIn(tok.token_id("[spk_emb_0]"), ids)
        self.assertIn(tok.token_id("[1234]"), ids)


if __name__ == "__main__":
    unittest.main()