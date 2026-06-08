import gc
import site
import shutil
import soundfile as sf
import numpy as np
import onnxruntime
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

package_path = site.getsitepackages()[-1]
shutil.copyfile(r'./modeling_modified/common.py', package_path + r'/nemo/core/classes/common.py')
shutil.copyfile(r'./modeling_modified/audio_codec.py', package_path + r'/nemo/collections/tts/models/audio_codec.py')
from nemo.collections.tts.models import AudioCodecModel


path_kani    = r'/home/DakeQQ/Downloads/kani-tts-370m'                             # Set the folder path where the [kani-tts-370m, kani-tts-400m] project downloaded.
path_codec   = r'/home/DakeQQ/Downloads/nemo-nano-codec-22khz-0.6kbps-12.5fps/nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo' # The audio codec download path. URL: https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps
onnx_model_A = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/KaniTTS_Embed.onnx'           # Assign a path where the exported KaniTTS model stored.
onnx_model_B = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/KaniTTS_Main.onnx'            # Assign a path where the exported KaniTTS model stored.
onnx_model_C = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/Greedy_Search.onnx'           # Assign a path where the exported KaniTTS model stored.
onnx_model_D = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/First_Beam_Search.onnx'       # Assign a path where the exported KaniTTS model stored.
onnx_model_E = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/Second_Beam_Search.onnx'      # Assign a path where the exported KaniTTS model stored.
onnx_model_F = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/Apply_Penalty.onnx'           # Assign a path where the exported KaniTTS model stored.
onnx_model_G = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/KaniTTS_Codec.onnx'           # Assign a path where the exported KaniTTS model stored.
onnx_model_H = r'/home/DakeQQ/Downloads/KaniTTS_ONNX/Argmax.onnx'                  # Assign a path where the exported KaniTTS model stored.
generated_audio_path = r"./generated.wav"                                          # The generated audio path.

target_tts = [
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
]
# The test query after the export process.

speaker = 'jenny'

"""
kani-tts-370m multilingual:
    Speaker List:
        david — David, English (British)
        puck — Puck, English (Gemini)
        kore — Kore, English (Gemini)
        andrew — Andrew, English
        jenny — Jenny, English (Irish)
        simon — Simon, English
        katie — Katie, English
        seulgi — Seulgi, Korean
        bert — Bert, German
        thorsten — Thorsten, German (Hessisch)
        maria — Maria, Spanish
        mei — Mei, Chinese (Cantonese)
        ming — Ming, Chinese (Shanghai OpenAI)
        karim — Karim, Arabic
        nur — Nur, Arabic
"""

# ── Export ────────────────────────────────────────────────────────────────────
DO_EXPORT = True                # Export the ONNX models.
PREVENT_F16_OVERFLOW = False    # Set True when quantizing to Q4F16 / Q8F16 / F16.
USE_FLOAT16_KV    = True        # Store KV cache in float16 (less memory bandwidth).
USE_FLOAT16_CODEC = True        # Run NeMo Codec in float16.


# ── Decoding ──────────────────────────────────────────────────────────────────
STOP_TOKEN     = [64402]        # Stop token id for KaniTTS.
MAX_SEQ_LEN    = 1024           # Maximum decode length.
USE_BEAM_SEARCH = False         # True = beam search; False = greedy search.
BEAM_SIZE      = 5              # Number of beams.
TOP_K          = 5              # Top-k candidates per step.
REPEAT_PENALITY = 0.8           # Repetition penalty (0.0–1.0; 1.0 = none).
PENALITY_RANGE = 10             # Window of recent tokens to penalize.

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050          # Output audio sample rate (Hz).

# ── Hardware ──────────────────────────────────────────────────────────────────
MAX_THREADS    = 0              # CPU threads (0 = auto).
DEVICE_ID      = 0              # Device index.


class GREEDY_SEARCH(torch.nn.Module):
    """Select the token with the highest logit (greedy decoding)."""

    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id        = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.total_layers     = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits    = all_inputs[-3]
        save_id   = all_inputs[-2]
        beam_size = all_inputs[-1]

        row_logsumexp               = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob               = top_beam_logits - row_logsumexp

        for i in range(self.total_layers):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *([1] * (kv.dim() - 1)))

        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id          = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx   = top_beam_indices[[0]]

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.transpose(0, 1),
            top_beam_indices,
            max_logits_idx,
        )


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand existing beams."""

    def __init__(self, total_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.total_layers     = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits        = all_inputs[-5]
        save_id       = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size     = all_inputs[-2]
        top_k         = all_inputs[-1]

        row_logsumexp              = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        top_k_prob                 = top_k_logits - row_logsumexp
        current_prob               = (top_k_prob + previous_prob).view(-1)

        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index                 = flat_beam_indices // top_k
        top_beam_indices           = top_k_indices.view(-1)[flat_beam_indices]

        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx   = top_beam_indices[[0]]
        save_id          = torch.cat([gathered_save_id, top_beam_indices], dim=-1)

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.unsqueeze(-1),
            top_beam_indices,
            max_logits_idx,
        )


class APPLY_PENALTY(torch.nn.Module):
    """Apply a repetition penalty over the most recent `penalty_range` tokens."""

    def __init__(self):
        super(APPLY_PENALTY, self).__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized      = logits.gather(1, target_indices) * penalty_value
        logits         = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Return the argmax index over the vocabulary dimension."""

    def __init__(self):
        super(ARGMAX, self).__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class KANITTS_EMBED(torch.nn.Module):
    def __init__(self, kani_tts):
        super(KANITTS_EMBED, self).__init__()
        self.embed_tokens = kani_tts.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class KANITTS_MAIN(torch.nn.Module):
    """
    Optimized KaniTTS main transformer module.

    Optimizations applied:
      - flip()-based rotate_half (fastest ONNX pattern)
      - sum()-based RMS norm with rsqrt (eliminates division, uses single fused op)
      - float16 rotary buffers with [-sin, sin] pattern (reduced memory, compatible with flip)
      - Fused QKV projection with absorbed operator_norm weights
      - Fused QK norm weights with scale factors absorbed
      - GQA via broadcast (eliminates repeat_k/repeat_v memory copies)
      - Absorbed embedding_norm into lm_head weights
      - F16 KV cache support for reduced memory bandwidth
      - Absorbed ffn_norm weights for inline norm computation
      - Absorbed operator_norm into conv.in_proj for conv layers
    """

    def __init__(self, kani_tts, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers, num_conv_layers, num_attn_layers):
        super(KANITTS_MAIN, self).__init__()
        self.kani_tts = kani_tts

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.num_layers = num_layers
        self.num_conv_layers = num_conv_layers
        self.num_attn_layers = num_attn_layers
        self.qk_heads = num_heads + num_key_value_heads
        self.total_qkv_heads = self.qk_heads + num_key_value_heads
        self.qkv_split_sizes = [self.qk_heads, num_key_value_heads]
        self.qk_split_sizes = [num_heads, num_key_value_heads]
        self.kv_f16 = USE_FLOAT16_KV
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)

        # ── sum()-based RMS norm epsilon (eps_sum = hidden_size * eps) ────
        hidden_size = kani_tts.model.embed_tokens.embedding_dim
        variance_epsilon = float(1e-5)
        hidden_rms_norm_eps = hidden_size * variance_epsilon
        qk_rms_norm_eps = head_dim * variance_epsilon
        if PREVENT_F16_OVERFLOW:
            self.qk_rms_norm_eps *= self.overflow_scale.square()
        self.register_buffer("hidden_rms_norm_eps", torch.tensor([hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps", torch.tensor([qk_rms_norm_eps], dtype=torch.float32))

        # ── Norm scale factors (compensate sum vs mean) ──────────────────
        norm_factor = float(hidden_size ** 0.5)
        qk_norm_factor = float(head_dim ** 0.5)
        scale_factor = float(head_dim ** -0.25)
        combined_qk_scale = scale_factor * qk_norm_factor  # = head_dim^0.25

        # ── Precompute rotary embeddings as float16 with [-sin, sin] ─────
        position_ids = torch.arange(max_seq_len, dtype=torch.float32)
        inv_freq = kani_tts.model.pos_emb.inv_freq
        freqs = torch.outer(position_ids, inv_freq)  # (max_seq_len, head_dim//2)
        attention_scaling = kani_tts.model.pos_emb.attention_scaling

        # cos: [cos, cos], sin: [-sin, sin] for flip()-based rotate_half
        cos_emb = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * attention_scaling
        sin_emb = torch.cat([-freqs.sin(), freqs.sin()], dim=-1) * attention_scaling

        # Shape: (1, max_seq_len, 1, 1, head_dim) for broadcast with (B, S, 1, qk_heads, D)
        self.register_buffer("cos_rotary_pos_emb", cos_emb.unsqueeze(0).unsqueeze(2).unsqueeze(3).half())
        self.register_buffer("sin_rotary_pos_emb", sin_emb.unsqueeze(0).unsqueeze(2).unsqueeze(3).half())

        # ── Fuse weights ─────────────────────────────────────────────────
        self._fuse_weights(norm_factor, combined_qk_scale)

        # ── KV cache and conv state buffers ──────────────────────────────
        self.num_key_value_layers = num_attn_layers + num_attn_layers
        self.save_key = [None] * num_attn_layers
        self.save_value = [None] * num_attn_layers
        self.save_conv = [None] * num_conv_layers

        # ── Pre-computed per-layer constants ─────────────────────────────
        self.o_proj_in_features = num_heads * head_dim

    def _fuse_weights(self, norm_factor, combined_qk_scale):
        """
        Fuse and absorb normalization weights into projection matrices.

        For attention layers:
          - Fuse Q, K, V projections into single QKV linear
          - Absorb operator_norm.weight * sqrt(hidden_size) into QKV weights
          - Combine q_layernorm and k_layernorm weights with combined_qk_scale

        For conv layers:
          - Absorb operator_norm.weight * sqrt(hidden_size) into conv.in_proj

        For all layers:
          - Store ffn_norm.weight * sqrt(hidden_size) for inline computation

        Final layer:
          - Absorb embedding_norm.weight * sqrt(hidden_size) into lm_head
        """
        with torch.no_grad():
            for layer in self.kani_tts.model.layers:
                if layer.is_attention_layer:
                    self._fuse_attention_layer(layer, norm_factor, combined_qk_scale)
                else:
                    self._fuse_conv_layer(layer, norm_factor)
                self._fuse_ffn_norm(layer, norm_factor)

            # Absorb embedding_norm into lm_head
            final_norm_weight = self.kani_tts.model.embedding_norm.weight.unsqueeze(0) * norm_factor
            self.kani_tts.lm_head.weight.mul_(final_norm_weight)

    def _fuse_attention_layer(self, layer, norm_factor, combined_qk_scale):
        """Fuse QKV projections and absorb input/QK norms."""
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        # Create merged QKV linear
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = any(p.bias is not None for p in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        del attn.q_proj, attn.k_proj, attn.v_proj

        # Absorb operator_norm.weight into QKV weights
        input_norm_weight = layer.operator_norm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv

        # Fuse QK norm weights with combined scale
        attn.q_layernorm.weight.mul_(combined_qk_scale)
        attn.k_layernorm.weight.mul_(combined_qk_scale)
        q_norm_repeated = attn.q_layernorm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.k_layernorm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, self.qk_heads, self.head_dim),
            requires_grad=False
        )
        del attn.q_layernorm, attn.k_layernorm

    def _fuse_conv_layer(self, layer, norm_factor):
        """Absorb operator_norm.weight into conv.in_proj."""
        input_norm_weight = layer.operator_norm.weight.unsqueeze(0) * norm_factor
        layer.conv.in_proj.weight.mul_(input_norm_weight)

    def _fuse_ffn_norm(self, layer, norm_factor):
        """Absorb ffn_norm.weight into feed_forward's w1 and w3."""
        ffn_norm_weight = layer.ffn_norm.weight.unsqueeze(0) * norm_factor
        layer.feed_forward.w1.weight.mul_(ffn_norm_weight)
        layer.feed_forward.w3.weight.mul_(ffn_norm_weight)

    # ══════════════════════════════════════════════════════════════════════
    # Optimized Primitives
    # ══════════════════════════════════════════════════════════════════════

    def _rms_norm(self, x, eps):
        """sum()-based RMS norm: x * rsqrt(sum(x^2) + eps_sum).
        Avoids division; eps_sum = hidden_size * eps compensates for sum vs mean.
        """
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)

    def _rotate_half_qk(self, x, batch_size):
        """flip()-based rotate_half for combined QK tensor.
        x shape: (B, S, 1, qk_heads, D)
        Swaps the two halves of head_dim using view+flip+view.
        Combined with [-sin, sin] rotary buffer, produces standard RoPE.
        """
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    # ══════════════════════════════════════════════════════════════════════
    # Forward
    # ══════════════════════════════════════════════════════════════════════

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-3]
        history_len = all_inputs[-2]
        ids_len = all_inputs[-1]
        kv_seq_len = history_len + ids_len
        batch_size = hidden_states.shape[0]

        # Slice rotary embeddings (stored as float16, cast to float32)
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()

        kv_count = 0
        conv_count = 0
        for i, layer in enumerate(self.kani_tts.model.layers):

            if layer.is_attention_layer:
                # ── RMS norm (sum-based, weight already absorbed into QKV) ──
                hidden_states_norm = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

                # ── Fused QKV projection ────────────────────────────────────
                qkv = layer.self_attn.qkv(hidden_states_norm)
                qkv = qkv.reshape(batch_size, -1, 1, self.total_qkv_heads, self.head_dim)
                qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

                # ── QK RMS norm + fused weight ──────────────────────────────
                qk = self._rms_norm(qk, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight

                # ── Rotary embedding (flip-based) ───────────────────────────
                qk = qk * rotary_cos + self._rotate_half_qk(qk, batch_size) * rotary_sin

                # ── Split Q and K ───────────────────────────────────────────
                q, k = torch.split(qk, self.qk_split_sizes, dim=-2)

                # ── Q reshape for GQA: (B, S, 1, H, D) → (B, KVH, G, S, D) ─
                q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                q = q.permute(0, 2, 3, 1, 4)

                # ── K/V to cache layout ─────────────────────────────────────
                k = k.permute(0, 3, 2, 4, 1)   # (B, KVH, 1, D, S)
                v = v.transpose(1, 3)           # (B, KVH, 1, S, D)

                # ── Optional F16 KV cache ───────────────────────────────────
                if self.kv_f16:
                    k = k.half()
                    v = v.half()

                # ── Concatenate with KV cache ───────────────────────────────
                k = torch.cat((all_inputs[kv_count], k), dim=-1)
                v = torch.cat((all_inputs[kv_count + self.num_attn_layers], v), dim=-2)
                self.save_key[kv_count] = k
                self.save_value[kv_count] = v
                kv_count += 1

                # ── Attention (GQA via broadcast, no repeat needed) ─────────
                if self.kv_f16:
                    k = k.float()
                    v = v.float()

                attn = torch.softmax(torch.matmul(q, k), dim=-1, dtype=torch.float32)
                attn_out = torch.matmul(attn, v)

                # ── Reshape and output projection ───────────────────────────
                attn_out = attn_out.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, self.o_proj_in_features)
                attn_out = layer.self_attn.out_proj(attn_out)

            else:
                # ── Conv layer: RMS norm (weight absorbed into in_proj) ─────
                hidden_states_norm = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

                # ── Conv computation ────────────────────────────────────────
                BCx = layer.conv.in_proj(hidden_states_norm).transpose(-1, -2)
                B_val, C, x = BCx.chunk(3, dim=-2)
                Bx = B_val * x
                conv_state = torch.cat([all_inputs[conv_count + self.num_key_value_layers].float(), Bx], dim=-1)
                if conv_count == 0:
                    len_conv_state = conv_state.shape[-1]
                self.save_conv[conv_count] = conv_state[..., -2:].half()
                conv_count += 1
                conv_out = layer.conv.conv(conv_state)[..., :len_conv_state]
                conv_out = conv_out[..., -ids_len:]
                attn_out = layer.conv.out_proj((C * conv_out).transpose(-1, -2).contiguous())

            # ── Residual + FFN ──────────────────────────────────────────
            hidden_states = hidden_states + attn_out
            ffn_input = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)
            hidden_states = hidden_states + layer.feed_forward(ffn_input)

        # ── Final projection (embedding_norm absorbed into lm_head) ─────
        hidden_states = self._rms_norm(hidden_states[:, -1], self.hidden_rms_norm_eps)
        logits = self.kani_tts.lm_head(hidden_states)
        return *self.save_key, *self.save_value, *self.save_conv, logits, kv_seq_len


class NEMO_CODEC(torch.nn.Module):
    """
    Optimized NeMo Codec decoder module.

    Optimizations applied:
      - Vectorized FSQ dequantization (eliminates Python loop over 4 codebooks)
      - Fused scale: division replaced with pre-computed multiply + subtract
      - Codebook buffer in (1, 4, 1) layout for direct broadcast after transpose
      - Slice view [0:1] instead of fancy indexing [[0]] (avoids tensor copy)
      - Single unsqueeze + reshape replaces per-codebook slice + cat
    """

    def __init__(self, nemo_codec, tokeniser_length):
        super(NEMO_CODEC, self).__init__()
        self.tokeniser_length = tokeniser_length
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = 4032
        # Codebook in (1, 4, 1) layout — broadcasts directly with (1, 4, T) after transpose
        self.register_buffer("codebook",
            (torch.tensor([self.codebook_size * i for i in range(4)], dtype=torch.int32) + self.audio_tokens_start).view(1, 4, 1))
        self.scale = float(SAMPLE_RATE / 22050.0)

        # ── Vectorized FSQ dequantizer (all 4 codebooks processed in parallel) ──
        # Buffers shaped (1, 1, 4, 1) broadcast with (1, 4, 1, T) → (1, 4, 4, T)
        self.register_buffer("fsq_dim_base_index",
            torch.tensor([1, 9, 72, 576], dtype=torch.int32).view(1, 1, 4, 1))
        self.register_buffer("fsq_num_levels",
            torch.tensor([9, 8, 8, 7], dtype=torch.int32).view(1, 1, 4, 1))
        # Fused scale: (nonneg - offset) / scale → nonneg * inv_scale - bias
        fsq_scale = torch.tensor([4.0, 4.0, 4.0, 3.0], dtype=torch.float32)
        fsq_offset = torch.tensor([4.0, 4.0, 4.0, 3.0], dtype=torch.float32)
        self.register_buffer("fsq_inv_scale", (1.0 / fsq_scale).view(1, 1, 4, 1))
        self.register_buffer("fsq_bias", (fsq_offset / fsq_scale).view(1, 1, 4, 1))

        # ── Inline CausalHiFiGANDecoder ──
        decoder = nemo_codec.audio_decoder
        self.pre_conv = decoder.pre_conv
        self.activations = decoder.activations
        self.res_layers = decoder.res_layers
        self.up_sample_conv_layers = decoder.up_sample_conv_layers
        self.up_sample_rates = decoder.up_sample_rates
        self.post_activation = decoder.post_activation
        self.post_conv = decoder.post_conv

        # ── Fuse weights: remove weight norm & absorb output scale ──
        # Fuse weight_g * weight_v / ||weight_v|| → single weight tensor (eliminates runtime norm)
        for module in decoder.modules():
            if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
                torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')
        # Absorb int16 output scale (32767.0) into post_conv weights
        self.post_conv.conv.weight.mul_(32767.0)
        if self.post_conv.conv.bias is not None:
            self.post_conv.conv.bias.mul_(32767.0)

    def forward(self, decode_ids, num_decode):
        # Slice view (no copy) → reshape → transpose + subtract produces contiguous (1, 4, T)
        audio_codes = decode_ids[[0], 2:num_decode].reshape(1, -1, 4)
        audio_len = audio_codes.shape[1].unsqueeze(0)
        audio_codes = audio_codes.transpose(1, 2) - self.codebook  # (1, 4, T)

        with torch.autocast(device_type="cpu", dtype=torch.float16 if USE_FLOAT16_CODEC else torch.float32):
            # ── Vectorized FSQ Dequantize: (1, 4, T) → (1, 16, T) ──
            # Single unsqueeze expands codebook dim; broadcast handles all 4 codebooks simultaneously
            codes_nonneg = (audio_codes.unsqueeze(2) // self.fsq_dim_base_index) % self.fsq_num_levels  # (1, 4, 4, T)
            out = (codes_nonneg.float() * self.fsq_inv_scale - self.fsq_bias).reshape(1, 16, -1)  # (1, 16, T)

            # ── HiFi-GAN Decoder: (1, 16, T) → (1, 1, T_audio) ──
            out = self.pre_conv(inputs=out, input_len=audio_len)

            for act, res_layer, up_sample_conv, up_sample_rate in zip(
                self.activations, self.res_layers, self.up_sample_conv_layers, self.up_sample_rates
            ):
                audio_len = audio_len * up_sample_rate
                out = act(out)
                out = up_sample_conv(inputs=out, input_len=audio_len)
                out = res_layer(inputs=out, input_len=audio_len)

            out = self.post_activation(out)
            out = self.post_conv(inputs=out, input_len=audio_len)
            out = out.clamp(min=-32767.0, max=32767.0)

            if self.scale != 1.0:
                out = torch.nn.functional.interpolate(
                    out,
                    scale_factor=self.scale,
                    mode='linear',
                    align_corners=False
                )
                audio_len = out.shape[-1]
            audio_out = out.to(torch.int16)
        return audio_out, audio_len


if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():
        # Load the original model
        model = AutoModelForCausalLM.from_pretrained(path_kani, dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        head_dim = model.model.layers._modules['2'].self_attn.head_dim
        num_layers = model.config.num_hidden_layers
        num_conv_layers = model.config.layer_types.count("conv")
        num_attn_layers = num_layers - num_conv_layers
        num_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads
        hidden_size = model.model.embed_tokens.embedding_dim
        vocab_size = model.vocab_size

        # ── Export: Embed ────────────────────────────────────────────────────
        embed = KANITTS_EMBED(model)
        input_ids = torch.zeros((3, 10), dtype=torch.int32)
        torch.onnx.export(
            embed,
            (input_ids,),
            onnx_model_A,
            input_names=['input_ids'],
            output_names=['hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'ids_len'},
                'hidden_state': {0: 'batch', 1: 'ids_len'},
            },
            do_constant_folding=True,
            opset_version=17,
            dynamo=False
        )
        del embed
        del input_ids

        # ── Export: Main ─────────────────────────────────────────────────────
        kani_tts = KANITTS_MAIN(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers, num_conv_layers, num_attn_layers)
        batch_size = 3                                    # "3" is just a dummy value.
        ids_len = torch.tensor([10], dtype=torch.int64)   # "10" is just a dummy value.
        hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
        history_len = torch.tensor([0], dtype=torch.int64)
        kv_dtype = torch.float16 if USE_FLOAT16_KV else torch.float32
        past_keys = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, 0), dtype=kv_dtype)
        past_values = torch.zeros((batch_size, num_key_value_heads, 1, 0, head_dim), dtype=kv_dtype)
        conv_states = torch.zeros((batch_size, hidden_size, 0), dtype=kv_dtype)

        # Prepare input and output names
        all_inputs = []
        input_names = []
        output_names = []
        dynamic_axes = {'hidden_states': {0: 'batch', 1: 'ids_len'}}
        for i in range(num_attn_layers):
            name = f'in_key_{i}'
            input_names.append(name)
            all_inputs.append(past_keys)
            dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
            name = f'out_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 4: 'history_len_plus_ids_len'}
        for i in range(num_attn_layers):
            name = f'in_value_{i}'
            input_names.append(name)
            all_inputs.append(past_values)
            dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
            name = f'out_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
        for i in range(num_conv_layers):
            name = f'in_conv_{i}'
            input_names.append(name)
            all_inputs.append(conv_states)
            dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}
            name = f'out_conv_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}
        input_names.append('hidden_states')
        all_inputs.append(hidden_states)
        input_names.append('history_len')
        all_inputs.append(history_len)
        input_names.append('ids_len')
        all_inputs.append(ids_len)
        output_names.append('logits')
        output_names.append('kv_seq_len')
        dynamic_axes['logits'] = {0: 'batch'}

        # torch.onnx.export(
        #     kani_tts,
        #     tuple(all_inputs),
        #     onnx_model_B,
        #     input_names=input_names,
        #     output_names=output_names,
        #     dynamic_axes=dynamic_axes,
        #     do_constant_folding=True,
        #     opset_version=17,
        #     dynamo=False
        # )
        del hidden_states
        del ids_len
        del history_len
        del input_names
        del output_names
        del dynamic_axes
        del all_inputs
        del kani_tts
        del model

        # ── Export: Greedy Search ────────────────────────────────────────────
        greedy = GREEDY_SEARCH()
        beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)
        save_id = torch.zeros((beam_size, 10), dtype=torch.int32)

        torch.onnx.export(
            greedy,
            (logits, save_id),
            onnx_model_C,
            input_names=['logits', 'save_id_in'],
            output_names=['max_logits_idx', 'save_id_out'],
            dynamic_axes={
                'logits': {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'max_logits_idx': {0: 'batch'},
                'save_id_out': {0: 'batch', 1: 'history_len'}
            },
            do_constant_folding=True,
            opset_version=17,
            dynamo=False
        )
        del greedy

        # ── Export: Apply Penalty ────────────────────────────────────────────
        penalty_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALITY_RANGE], dtype=torch.int64)
        torch.onnx.export(
            APPLY_PENALTY(),
            (logits, save_id, penalty_value, penalty_range),
            onnx_model_F,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
            output_names=['logits_out'],
            dynamic_axes={
                'logits_in': {0: 'batch', 1: 'vocab_size'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'logits_out': {0: 'batch', 1: 'vocab_size'}
            },
            do_constant_folding=True,
            opset_version=17,
            dynamo=False
        )
        del penalty_value, penalty_range

        # ── Export: Argmax ───────────────────────────────────────────────────
        torch.onnx.export(
            ARGMAX(),
            (logits,),
            onnx_model_H,
            input_names=['logits'],
            output_names=['max_logits_idx'],
            dynamic_axes={
                'logits': {0: 'batch', 1: 'vocab_size'},
                'max_logits_idx': {0: 'batch'}
            },
            do_constant_folding=True,
            opset_version=17,
            dynamo=False
        )

        # ── Export: First Beam Search ────────────────────────────────────────
        num_layers_beam = num_attn_layers + num_attn_layers + num_conv_layers
        first_beam_search = FIRST_BEAM_SEARCH(num_layers_beam)
        topK = torch.tensor([TOP_K], dtype=torch.int64)
        previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
        past_keys_greedy = past_keys[[0]]
        past_values_greedy = past_values[[0]]
        conv_states_greedy = conv_states[[0]]

        all_inputs = []
        input_names = []
        output_names = []
        dynamic_axes = {}
        for i in range(num_attn_layers):
            name = f'in_key_{i}'
            input_names.append(name)
            all_inputs.append(past_keys_greedy)
            dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        for i in range(num_attn_layers):
            name = f'in_value_{i}'
            input_names.append(name)
            all_inputs.append(past_values_greedy)
            dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        for i in range(num_conv_layers):
            name = f'in_conv_{i}'
            input_names.append(name)
            all_inputs.append(conv_states_greedy)
            dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}

        # Output names for expanded KV/conv caches (no dynamic_axes for outputs — same shape as inputs after repeat)
        for i in range(num_attn_layers):
            name = f'out_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        for i in range(num_attn_layers):
            name = f'out_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        for i in range(num_conv_layers):
            name = f'out_conv_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}

        input_names.append('logits')
        all_inputs.append(logits[[0]])
        input_names.append('save_id_in')
        all_inputs.append(save_id)
        input_names.append('beam_size')
        all_inputs.append(beam_size)
        output_names.append('save_id_out')
        output_names.append('top_beam_prob')
        output_names.append('top_beam_indices')
        output_names.append('max_logits_idx')
        dynamic_axes['logits'] = {0: 'batch'}
        dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
        dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
        dynamic_axes['top_beam_prob'] = {0: 'batch'}
        dynamic_axes['top_beam_indices'] = {0: 'batch'}
        dynamic_axes['max_logits_idx'] = {0: 'batch'}

        torch.onnx.export(
            first_beam_search,
            tuple(all_inputs),
            onnx_model_D,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=17,
            dynamo=False
        )
        del first_beam_search

        # ── Export: Second Beam Search ───────────────────────────────────────
        # KV caches must have batch dim = BEAM_SIZE (they arrive already expanded from first beam step)
        past_keys_beam = torch.zeros((BEAM_SIZE, num_key_value_heads, 1, head_dim, 0), dtype=kv_dtype)
        past_values_beam = torch.zeros((BEAM_SIZE, num_key_value_heads, 1, 0, head_dim), dtype=kv_dtype)
        conv_states_beam = torch.zeros((BEAM_SIZE, hidden_size, 0), dtype=kv_dtype)

        all_inputs = []
        input_names = []
        output_names = []
        dynamic_axes = {}
        for i in range(num_attn_layers):
            name = f'in_key_{i}'
            input_names.append(name)
            all_inputs.append(past_keys_beam)
            dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
            name = f'out_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 4: 'kv_seq_len'}
        for i in range(num_attn_layers):
            name = f'in_value_{i}'
            input_names.append(name)
            all_inputs.append(past_values_beam)
            dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
            name = f'out_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 3: 'kv_seq_len'}
        for i in range(num_conv_layers):
            name = f'in_conv_{i}'
            input_names.append(name)
            all_inputs.append(conv_states_beam)
            dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}
            name = f'out_conv_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 2: 'conv_states_len'}
        input_names.append('logits')
        all_inputs.append(logits)
        input_names.append('save_id_in')
        all_inputs.append(save_id)
        input_names.append('previous_prob')
        all_inputs.append(previous_prob)
        input_names.append('beam_size')
        all_inputs.append(beam_size)
        input_names.append('topK')
        all_inputs.append(topK)
        output_names.append('save_id_out')
        output_names.append('top_beam_prob')
        output_names.append('top_beam_indices')
        output_names.append('max_logits_idx')
        dynamic_axes['logits'] = {0: 'batch'}
        dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
        dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
        dynamic_axes['previous_prob'] = {0: 'batch'}
        dynamic_axes['top_beam_prob'] = {0: 'batch'}
        dynamic_axes['top_beam_indices'] = {0: 'batch'}
        dynamic_axes['max_logits_idx'] = {0: 'batch'}

        second_beam_search = SECOND_BEAM_SEARCH(num_layers_beam)
        torch.onnx.export(
            second_beam_search,
            tuple(all_inputs),
            onnx_model_E,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=17,
            dynamo=False
        )

        # ── Export: NeMo Codec ───────────────────────────────────────────────
        decode_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int32)  # Dummy values
        num_decode = torch.tensor([decode_ids.shape[-1]], dtype=torch.int64)
        nemo_codec = AudioCodecModel.from_pretrained(path_codec, map_location=torch.device('cpu')).float().eval()
        tokeniser_length = AutoTokenizer.from_pretrained(path_kani).vocab_size
        nemo_codec = NEMO_CODEC(nemo_codec, tokeniser_length)
        torch.onnx.export(
            nemo_codec,
            (decode_ids, num_decode),
            onnx_model_G,
            input_names=['decode_ids', 'num_decode'],
            output_names=['audio_out', 'audio_out_len'],
            dynamic_axes={
                'decode_ids': {0: 'batch_size', 1: 'num_decode'},
                'audio_out': {2: 'audio_len'}
            },
            do_constant_folding=True,
            opset_version=17,
            dynamo=False
        )
        del decode_ids
        del nemo_codec
        del num_decode
        del tokeniser_length
        del second_beam_search
        del past_keys
        del past_values
        del conv_states
        del past_keys_greedy
        del past_values_greedy
        del conv_states_greedy
        del logits
        del previous_prob
        del save_id
        del topK
        del input_names
        del output_names
        del dynamic_axes
        del all_inputs
        gc.collect()

    print('\nExport done!\n\nStart running the KaniTTS by ONNXRuntime.\nNow loading . . . it could cost minutes.')


# ══════════════════════════════════════════════════════════════════════════════
# Run the exported model by ONNX Runtime
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4
session_opts.log_verbosity_level = 4
session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.enable_cpu_mem_arena = True
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

ORT_Accelerate_Providers = ['CPUExecutionProvider']
device_type = 'cpu'
provider_options = None

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = in_name_A[0].name
out_name_A = [out_name_A[0].name]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_B.get_providers()}")

in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
amount_of_outputs_B = len(out_name_B)
in_name_B = [in_name_B[i].name for i in range(len(in_name_B))]
out_name_B = [out_name_B[i].name for i in range(amount_of_outputs_B)]


generate_limit = MAX_SEQ_LEN - 5  # 5 = length of initial input_ids
num_keys_values = 0
for i in ort_session_B._outputs_meta:
    if len(i.shape) == 5:
        num_keys_values += 1
num_keys_values_convs = amount_of_outputs_B - 2
num_conv_layers = num_keys_values_convs - num_keys_values
num_layers = num_keys_values // 2
num_keys_values_convs_plus_1 = num_keys_values_convs + 1
num_keys_values_convs_plus_2 = num_keys_values_convs + 2
num_keys_values_convs_plus_3 = num_keys_values_convs + 3
num_keys_values_convs_plus_4 = num_keys_values_convs + 4
num_keys_values_convs_plus_5 = num_keys_values_convs + 5
num_keys_values_convs_plus_6 = num_keys_values_convs + 6
num_keys_values_convs_plus_7 = num_keys_values_convs + 7

kv_dtype = np.float16 if 'float16' in ort_session_B._inputs_meta[0].type else np.float32
hidden_dtype = np.float16 if 'float16' in ort_session_B._inputs_meta[num_keys_values_convs].type else np.float32

vocab_size = ort_session_B._outputs_meta[num_keys_values_convs].shape[1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
tokenizer = AutoTokenizer.from_pretrained(path_kani)
head_ids = np.array([[64403]], dtype=np.int32)       # For non-Python tokenizer = [64403, 1]
tail_ids = np.array([[2, 64404]], dtype=np.int32)


# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE


if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")


if USE_BEAM_SEARCH:
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]

    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]

    input_feed_D = {
        in_name_D[num_keys_values_convs_plus_2]: beam_size
    }

    input_feed_E = {
        in_name_E[num_keys_values_convs_plus_3]: beam_size,
        in_name_E[num_keys_values_convs_plus_4]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
    out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]
    input_feed_C = {}

# --- Argmax ---
ort_session_H = onnxruntime.InferenceSession(onnx_model_H, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_H = ort_session_H.get_inputs()
out_name_H = ort_session_H.get_outputs()
in_name_H = [in_name_H[i].name for i in range(len(in_name_H))]
out_name_H = [out_name_H[i].name for i in range(len(out_name_H))]
input_feed_H = {}


ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_G = ort_session_G.get_inputs()
out_name_G = ort_session_G.get_outputs()
in_name_G = [in_name_G[i].name for i in range(len(in_name_G))]
out_name_G = [out_name_G[i].name for i in range(len(out_name_G))]


USE_PENALTY = (REPEAT_PENALITY != 1.0)

if USE_PENALTY:
    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(len(out_name_F))]

    penalty_dtype = np.float16 if 'float16' in ort_session_F._inputs_meta[2].type else np.float32
    penalty_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([REPEAT_PENALITY], dtype=penalty_dtype), device_type, DEVICE_ID)
    penalty_range = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([PENALITY_RANGE], dtype=np.int64), device_type, DEVICE_ID)
    input_feed_F = {in_name_F[2]: penalty_value, in_name_F[3]: penalty_range}


init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[0].shape[1], 1, ort_session_B._inputs_meta[0].shape[3], 0), dtype=kv_dtype), device_type, DEVICE_ID)
init_past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_layers].shape[1], 1, 0, ort_session_B._inputs_meta[num_layers].shape[4]), dtype=kv_dtype), device_type, DEVICE_ID)
init_conv_states_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_keys_values].shape[1], 0), dtype=kv_dtype), device_type, DEVICE_ID)
init_save_id = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE if USE_BEAM_SEARCH else 1, 0), dtype=np.int32), device_type, DEVICE_ID)
blank_segment = np.zeros((1, 1, int(SAMPLE_RATE * 0.3)), dtype=np.int16)  # The blank for separate the generated audio. Default to 300ms. Edit it freely.

# Start to run
save_audio_out = []
start_time = time.time()
for sentence in target_tts:
    sentence = f"{speaker}: {sentence}"
    print(f"\n{sentence}")
    input_ids = tokenizer(sentence, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = np.concatenate([head_ids, input_ids, tail_ids], axis=1)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[1]], dtype=np.int64), device_type, DEVICE_ID)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
    history_len = init_history_len
    past_keys_B = init_past_keys_B
    past_values_B = init_past_values_B
    conv_states_B = init_conv_states_B
    save_id = init_save_id

    input_feed_A = {in_name_A: input_ids}
    hidden_states = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]

    input_feed_B = {
            in_name_B[num_keys_values_convs]: hidden_states,
            in_name_B[num_keys_values_convs_plus_1]: history_len,
            in_name_B[num_keys_values_convs_plus_2]: ids_len
        }

    for i in range(num_layers):
        input_feed_B[in_name_B[i]] = past_keys_B
    for i in range(num_layers, num_keys_values):
        input_feed_B[in_name_B[i]] = past_values_B
    for i in range(num_keys_values, num_keys_values_convs):
        input_feed_B[in_name_B[i]] = conv_states_B

    if USE_BEAM_SEARCH:
        input_feed_D[in_name_D[num_keys_values_convs]] = save_id
        input_feed_D[in_name_D[num_keys_values_convs_plus_1]] = save_id
    else:
        input_feed_C[in_name_C[1]] = save_id

    num_decode = 0
    start_decode = time.time()
    while num_decode < generate_limit:
        all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
        logits_out = all_outputs_B[num_keys_values_convs]

        if USE_BEAM_SEARCH:
            # Apply penalty before beam search
            if USE_PENALTY and num_decode >= PENALITY_RANGE:
                input_feed_F[in_name_F[0]] = logits_out
                input_feed_F[in_name_F[1]] = save_id
                logits_out = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)[0]

            if num_decode < 1:
                input_feed_D.update(zip(in_name_D[:num_keys_values_convs], all_outputs_B))
                input_feed_D[in_name_D[num_keys_values_convs]] = logits_out
                input_feed_D[in_name_D[num_keys_values_convs_plus_1]] = save_id
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                # Outputs: [*kv_caches, save_id_out, top_beam_prob, top_beam_indices, max_logits_idx]
                max_logits_idx = all_outputs_D[num_keys_values_convs_plus_3].numpy()
                save_id = all_outputs_D[num_keys_values_convs]
            else:
                input_feed_E.update(zip(in_name_E[:num_keys_values_convs], all_outputs_B))
                input_feed_E[in_name_E[num_keys_values_convs]] = logits_out
                input_feed_E[in_name_E[num_keys_values_convs_plus_1]] = save_id
                input_feed_E[in_name_E[num_keys_values_convs_plus_2]] = beam_prob
                all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                # Outputs: [*kv_caches, save_id_out, top_beam_prob, top_beam_indices, max_logits_idx]
                max_logits_idx = all_outputs_E[num_keys_values_convs_plus_3].numpy()
                save_id = all_outputs_E[num_keys_values_convs]

            if max_logits_idx in STOP_TOKEN:
                break

            if num_decode < 1:
                input_feed_B.update(zip(in_name_B[:num_keys_values_convs], all_outputs_D))
                input_feed_A[in_name_A] = all_outputs_D[num_keys_values_convs_plus_2]
                beam_prob = all_outputs_D[num_keys_values_convs_plus_1]
            else:
                input_feed_B.update(zip(in_name_B[:num_keys_values_convs], all_outputs_E))
                input_feed_A[in_name_A] = all_outputs_E[num_keys_values_convs_plus_2]
                beam_prob = all_outputs_E[num_keys_values_convs_plus_1]
            input_feed_B[in_name_B[num_keys_values_convs]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
        else:
            # Apply penalty before greedy/argmax
            if USE_PENALTY and num_decode >= PENALITY_RANGE:
                input_feed_F[in_name_F[0]] = logits_out
                input_feed_F[in_name_F[1]] = save_id
                logits_out = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)[0]

            input_feed_C[in_name_C[0]] = logits_out
            input_feed_C[in_name_C[1]] = save_id
            max_logits_idx_ort, save_id = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)

            max_logits_idx = max_logits_idx_ort.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN:
                break

            input_feed_B.update(zip(in_name_B[:num_keys_values_convs], all_outputs_B))
            input_feed_A[in_name_A] = max_logits_idx_ort
            input_feed_B[in_name_B[num_keys_values_convs]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
        input_feed_B[in_name_B[num_keys_values_convs_plus_1]] = all_outputs_B[num_keys_values_convs_plus_1]
        if num_decode < 1:
            input_feed_B[in_name_B[num_keys_values_convs_plus_2]] = init_ids_len_1
        num_decode += 1
    if num_decode > 0:
        print(f"\nDecode: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s")
        input_feed_G = {in_name_G[0]: save_id}
        input_feed_G[in_name_G[1]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([num_decode], dtype=np.int64), device_type, DEVICE_ID)
        audio_out = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)[0]
        save_audio_out.append(audio_out.numpy())
        save_audio_out.append(blank_segment)
    else:
        print("\n Generate Failed")
print(f"\nGenerate Complete.\n\nSaving to: {generated_audio_path}.\n\nTime Cost: {time.time() - start_time:.3f} Seconds")
audio_out = np.concatenate(save_audio_out, axis=-1).reshape(-1)
sf.write(generated_audio_path, audio_out, SAMPLE_RATE, format='WAVEX')
