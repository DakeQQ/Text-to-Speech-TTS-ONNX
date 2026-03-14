import gc
import time
import torch
import site
import shutil
import soundfile as sf
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from modeling_modified.text_normalize import TextNormalizer
from transformers import LlamaTokenizerFast


path_voxcpm                         = r'/home/DakeQQ/Downloads/VoxCPM1.5'                                # Set the folder path where the VoxCPM1.5 project downloaded.
onnx_model_Text_Embed               = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Text_Embed.onnx'       # Assign a path where the exported VoxCPM model stored.
onnx_model_VAE_Encoder              = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_VAE_Encoder.onnx'
onnx_model_Feat_Encoder             = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Feat_Encoder.onnx'
onnx_model_Feat_Cond                = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Feat_Cond.onnx'
onnx_model_Concat                   = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Concat.onnx'
onnx_model_Rotary_Mask_Text_Prefill = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Rotary_Mask_Text_Prefill.onnx'
onnx_model_Rotary_Mask_Text_Decode  = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Rotary_Mask_Text_Decode.onnx'
onnx_model_Main                     = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Main.onnx'
onnx_model_Feat_Decoder             = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Feat_Decoder.onnx'
onnx_model_VAE_Decoder              = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_VAE_Decoder.onnx'

prompt_audio_path = "./example/basic_ref_zh.wav"                                # optional: path to a prompt speech for voice cloning else None.
prompt_text = "对，这就是我，万人敬仰的太乙真人。"                                    # The reference text for the prompt speech.
target_tts = ["大家好，我现在正在大可奇奇体验AI科技。", "Hello everyone, I'm currently experiencing DakeQQ's AI technology."]  # The test query after the export process.
generated_audio_path = r"./generated.wav"                                       # The generated audio path.

# Model Config
DO_EXPORT = True                         # Whether to export the ONNX models

# === Decoding limits & tokens ===
STOP_TOKEN = [1]                         # The stop_id in VoxCPM is "1"
MAX_SEQ_LEN = 1024                       # The max decode length; cannot be changed after export. Free to edit it.
MIN_SEQ_LEN = 2                          # The min decode length. Free to edit it.
DECODE_LIMIT_FACTOR = 6                  # Decode length limit factor, integer >= 1. Free to edit it.

# === Audio configuration ===
IN_SAMPLE_RATE = 44100                      # Input prompt audio sample rate; cannot be changed after export
OUT_SAMPLE_RATE = 44100                     # Output audio sample rate; cannot be changed after export
MAX_PROMPT_AUDIO_LEN = 20 * IN_SAMPLE_RATE  # Max prompt audio length in samples. Free to edit it.

# === Guidance, diffusion & randomness ===
FIXED_TIMESTEPS = 10                     # Fixed timesteps; cannot be changed after export. Larger is finer but slower. Free to edit it.
CFG_VALUE = 2.5                          # Lower values result in more natural speech for long text, while higher values stay closer to the original sound features. Free to edit it.
RANDOM_SEED = 1                          # Global random seed. Free to edit it.

# === Feature flags ===
STREAMING = False                        # Enable streaming synthesis. Free to enable it. Unlike the official implementation, this version processes two latents at a time for faster performance, albeit with potential discontinuities during piece-by-piece decoding.
DYNAMIC_SHAPE_VAE_DECODE = True          # Use dynamic shape for VAE decoder. Free to enable it.
USE_TEXT_NORMALIZER = True               # Use text normalizer. Free to enable it.
USE_AUDIO_NORMALIZER = False             # Use an audio normalizer to stabilize loudness, though this may result in a loss of original audio characteristics. Free to enable it.
PREVENT_F16_OVERFLOW = False             # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
USE_F16_KV = False                       # Use float16 for key/value cache. Free to enable it. The quality of short sentences will decrease.

# === ONNX / runtime configuration ===
ORT_LOG = False                          # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16 = False                         # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers = []            # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
OPSET = 17                               # ONNX opset version. Free to edit it.
MAX_THREADS = 0                          # Parallel CPU threads, 0 for auto. Free to edit it.
DEVICE_ID = 0                            # Device id, default 0. Free to edit it.

py_site = site.getsitepackages()[-1]
shutil.copyfile('./modeling_modified/core.py', py_site + '/voxcpm/core.py')
shutil.copyfile('./modeling_modified/audio_vae.py', py_site + '/voxcpm/modules/audiovae/audio_vae.py')
from voxcpm import VoxCPM


# ══════════════════════════════════════════════════════════════════════════════
# Text Embedding Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_TEXT_EMBED(torch.nn.Module):
    def __init__(self, voxcpm):
        super(VOXCPM_TEXT_EMBED, self).__init__()
        self.voxcpm = voxcpm

    def forward(self, text_ids):
        text_embed = self.voxcpm.base_lm.embed_tokens(text_ids)
        return text_embed


# ══════════════════════════════════════════════════════════════════════════════
# VAE Encoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_VAE_ENCODER(torch.nn.Module):
    def __init__(self, voxcpm, in_sample_rate):
        super(VOXCPM_VAE_ENCODER, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self.inv_int16 = torch.tensor(1.0 / 32768.0, dtype=torch.float32).view(1, 1, -1)
        self.patch_len = self.voxcpm.patch_size * self.voxcpm.chunk_size
        self.pad_zeros = torch.zeros([1, 1, self.patch_len], dtype=torch.int8)
        self.in_sample_rate = in_sample_rate
        self.sr_scale = float(44100.0 / self.in_sample_rate)

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def forward(self, prompt_audio):
        prompt_audio = prompt_audio.float()
        if self.sr_scale > 1.0:
            prompt_audio = prompt_audio * self.inv_int16
            prompt_audio = torch.nn.functional.interpolate(
                prompt_audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False
            )
        elif self.sr_scale < 1.0:
            prompt_audio = torch.nn.functional.interpolate(
                prompt_audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False
            )
            prompt_audio = prompt_audio * self.inv_int16
        else:
            prompt_audio = prompt_audio * self.inv_int16
        padding_size = self.patch_len - prompt_audio.shape[-1] % self.patch_len
        prompt_audio = torch.cat([prompt_audio, self.pad_zeros[..., :padding_size].float()], dim=-1)
        audio_feat = self.voxcpm.audio_vae.encoder(prompt_audio)
        audio_feat = audio_feat.view(self.voxcpm.audio_vae.latent_dim, -1, self.voxcpm.patch_size).permute(1, 2, 0)
        return audio_feat


# ══════════════════════════════════════════════════════════════════════════════
# Feature Encoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_FEAT_ENCODER(torch.nn.Module):
    def __init__(self, voxcpm, max_prompt_audio_len, in_sample_rate):
        super(VOXCPM_FEAT_ENCODER, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self.head_dim = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_heads
        self.num_key_value_heads = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_key_value_heads
        self.num_key_value_groups = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_key_value_groups
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        max_prompt_feat_len = (max_prompt_audio_len // in_sample_rate * 44100) // (self.voxcpm.patch_size * self.voxcpm.chunk_size) + 1
        self.special_tokens = self.voxcpm.feat_encoder.special_token.expand(1, max_prompt_feat_len, 1, -1).squeeze(0).half()
        self.q_len = self.voxcpm.patch_size + 1  # Fixed to 5 for VoxCPM1.5
        position_ids = torch.arange(self.q_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = self.voxcpm.feat_encoder.encoder.rope_emb(position_ids)
        rope_emb_sin[:, :self.voxcpm.feat_encoder.encoder.rope_emb.dim // 2] *= -1.0
        self.rope_emb_cos = rope_emb_cos.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        self.rope_emb_sin = rope_emb_sin.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        self.split_size = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.head_dim // 2
        norm_factor = self.voxcpm.feat_encoder.encoder.config.hidden_size ** 0.5
        scale_factor = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.head_dim ** -0.25
        with torch.no_grad():
            for layer in self.voxcpm.feat_encoder.encoder.layers:
                # 1) Fuse q/k/v into qkv
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                in_features = int(q_proj.in_features)
                out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
                has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
                qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
                if has_bias:
                    z = lambda feat: torch.zeros(feat, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
                    qb = q_proj.bias if q_proj.bias is not None else z(q_proj.out_features)
                    kb = k_proj.bias if k_proj.bias is not None else z(k_proj.out_features)
                    vb = v_proj.bias if v_proj.bias is not None else z(v_proj.out_features)
                    qkv.bias.copy_(torch.cat([qb * scale_factor, kb * scale_factor, vb], dim=0))

                layer.self_attn.q_out_features = int(q_proj.out_features)
                layer.self_attn.k_out_features = int(k_proj.out_features)
                layer.self_attn.v_out_features = int(v_proj.out_features)
                layer.self_attn.qkv = qkv

                del layer.self_attn.q_proj
                del layer.self_attn.k_proj
                del layer.self_attn.v_proj

                # 2) Fuse input rmsnorm weight into qkv input columns
                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                del layer.input_layernorm

                # 3) Fuse post-attention rmsnorm weight into MLP gate/up input columns
                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj

                in_feat = gate.in_features
                out_feat = gate.out_features + up.out_features
                gate_up = torch.nn.Linear(in_feat, out_feat, bias=False)

                gate_weight = gate.weight * w
                up_weight = up.weight * w
                gate_up.weight.copy_(torch.cat([gate_weight, up_weight], dim=0))

                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj
                del layer.mlp.up_proj
                del layer.post_attention_layernorm

            # 4) Fuse final norm weight into enc_to_lm_proj
            w = self.voxcpm.feat_encoder.encoder.norm.weight.unsqueeze(0) * norm_factor
            self.voxcpm.enc_to_lm_proj.weight.mul_(w)
            del self.voxcpm.feat_encoder.encoder.norm

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        """Apply modified RMS normalization (with optional overflow scaling)."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True))

    def rotate_half(self, x):
        x = x.view(-1, self.q_len, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(-1, self.q_len, 1, self.qk_heads, self.head_dim)

    def forward(self, audio_feat):
        audio_feat_len = audio_feat.shape[0].unsqueeze(0)
        hidden_states = self.voxcpm.feat_encoder.in_proj(audio_feat)
        hidden_states = torch.cat([self.special_tokens[:audio_feat_len].float(), hidden_states], dim=-2)
        hidden_states = hidden_states.view(-1, self.q_len, self.voxcpm.feat_encoder.in_proj.out_features)
        for layer in self.voxcpm.feat_encoder.encoder.layers:
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, self.q_len, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * self.rope_emb_cos + self.rotate_half(qk) * self.rope_emb_sin
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.q_len, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)
            attn = torch.softmax(torch.matmul(q, k), dim=-1)
            attn = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(-1, self.q_len, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        feat_embed = hidden_states[:, 0]
        feat_embed = self._rms_norm(feat_embed)
        feat_embed = self.voxcpm.enc_to_lm_proj(feat_embed).unsqueeze(0)
        return feat_embed


# ══════════════════════════════════════════════════════════════════════════════
# Feature Conditioning Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_FEAT_COND(torch.nn.Module):
    def __init__(self, voxcpm):
        super(VOXCPM_FEAT_COND, self).__init__()
        self.voxcpm = voxcpm

    def forward(self, audio_feat):
        feat_cond = self.voxcpm.feat_decoder.estimator.cond_proj(audio_feat[[-1]])
        feat_cond = torch.cat([feat_cond, feat_cond], dim=0)
        return feat_cond


# ══════════════════════════════════════════════════════════════════════════════
# Concatenation Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_CONCAT(torch.nn.Module):
    def __init__(self):
        super(VOXCPM_CONCAT, self).__init__()
        pass

    def forward(self, embed_0, embed_1):
        concat_embed = torch.cat([embed_0, embed_1], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding & Attention Mask
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_ROTARY_MASK_PREFILL(torch.nn.Module):
    """Precompute rotary embeddings and causal mask for the prefill phase."""

    def __init__(self, voxcpm, max_seq_len):
        super().__init__()

        # Causal attention mask: upper triangle → -128
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

        # Precompute rotary embeddings
        cos, sin = self._build_rotary_table(voxcpm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos.half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half(), persistent=False)

    @staticmethod
    def _build_rotary_table(voxcpm, max_seq_len):
        position_ids = torch.arange(max_seq_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = voxcpm.base_lm.rope_emb(position_ids)
        rope_emb_sin[:, :voxcpm.base_lm.rope_emb.dim // 2] *= -1.0
        cos = rope_emb_cos.unsqueeze(1).unsqueeze(1)
        sin = rope_emb_sin.unsqueeze(1).unsqueeze(1)
        return cos, sin

    def forward(self, ids_len, history_len, mask):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[history_len:kv_seq_len].float()
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * mask).float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class VOXCPM_ROTARY_MASK_DECODE(torch.nn.Module):
    """Provide rotary embeddings for a single decode step."""

    def __init__(self, voxcpm, max_seq_len):
        super().__init__()
        cos, sin = VOXCPM_ROTARY_MASK_PREFILL._build_rotary_table(voxcpm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos.half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Main Transformer Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_MAIN(torch.nn.Module):
    """
    Main transformer module that processes hidden states through all decoder layers.

    Handles:
      - Fused QKV projection with pre-merged layer norms
      - Rotary positional embeddings (RoPE)
      - KV cache management with optional F16
      - Grouped-query attention (GQA)
      - Fused gate-up MLP projection
    """

    def __init__(self, voxcpm, max_seq_len):
        super(VOXCPM_MAIN, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim = self.voxcpm.base_lm.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_heads
        self.num_key_value_heads = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_key_value_heads
        self.num_key_value_groups = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_key_value_groups
        self.qk_heads = self.num_heads + self.num_key_value_heads

        # ── Overflow guard ───────────────────────────────────────────────
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)

        # ── Layer counts ─────────────────────────────────────────────────
        self.total_layers = self.voxcpm.base_lm.config.num_hidden_layers + self.voxcpm.residual_lm.config.num_hidden_layers

        # ── Per-layer output buffers ─────────────────────────────────────
        self.save_key = [None] * self.total_layers
        self.save_value = [None] * self.total_layers

        # ── Fuse & reshape weights for efficient inference ───────────────
        self.norm_factor = self.voxcpm.base_lm.config.hidden_size ** 0.5
        self.scale_factor_base = float(self.voxcpm.base_lm.layers._modules['0'].self_attn.head_dim ** -0.25)
        self._fuse_weights()

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self):
        """
        Merge separate Q/K/V projections into a single QKV linear,
        absorb RMSNorm weights into projection matrices, and fuse
        gate/up projections for the MLP.
        """
        with torch.no_grad():
            for layer in self.voxcpm.base_lm.layers:
                self._fuse_qkv_projection(layer)
                self._fuse_gate_up_projection(layer)

            for layer in self.voxcpm.residual_lm.layers:
                self._fuse_qkv_projection(layer)
                self._fuse_gate_up_projection(layer)

            # Absorb final RMSNorm into res_to_dit_proj
            final_norm_weight = self.voxcpm.residual_lm.norm.weight.unsqueeze(0) * self.norm_factor
            self.voxcpm.res_to_dit_proj.weight.mul_(final_norm_weight)
            del self.voxcpm.residual_lm.norm

    def _fuse_qkv_projection(self, layer):
        """Fuse Q, K, V projections and absorb input LayerNorm."""
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj

        # ── Create merged QKV linear ─────────────────────────────────
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight * self.scale_factor_base, k_proj.weight * self.scale_factor_base, v_proj.weight], dim=0))

        if has_bias:

            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=q_proj.weight.dtype, device=q_proj.weight.device)

            qkv.bias.copy_(torch.cat([_get_bias(q_proj) * self.scale_factor_base, _get_bias(k_proj) * self.scale_factor_base, _get_bias(v_proj)], dim=0))

        # Store split dimensions for later use
        layer.self_attn.q_out_features = int(q_proj.out_features)
        layer.self_attn.k_out_features = int(k_proj.out_features)
        layer.self_attn.v_out_features = int(v_proj.out_features)
        layer.self_attn.qkv = qkv

        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

        # ── Absorb input LayerNorm into QKV weights ─────────────────
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * self.norm_factor
        qkv.weight.mul_(input_norm_weight)
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * self.norm_factor
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        """Recursively replace exact GELU with tanh-approximated GELU for ONNX compatibility."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                VOXCPM_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        """Apply modified RMS normalization (with optional overflow scaling)."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True))

    def _rotate_half(self, x):
        """Rotate the last dimension by swapping and negating halves (for RoPE).
           Using flip() is more efficient than split() + concat() in ONNX Runtime.
        """
        x = x.view(-1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(-1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        feat_embed         = all_inputs[-6]
        concat_text_len    = all_inputs[-5]
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]

        for i, layer in enumerate(self.voxcpm.base_lm.layers):

            # ── Self-Attention ───────────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * rotary_pos_emb_cos + self._rotate_half(qk) * rotary_pos_emb_sin
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(1, 2, 0, 3)
            if USE_F16_KV:
                k = k.half()
                v = v.half()
            k = k.permute(2, 1, 3, 0)
            v = v.transpose(0, 2)

            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.total_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v

            if USE_F16_KV:
                k = k.float()
                v = v.float()

            attn = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
            attn = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network ─────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        hidden_states = self.voxcpm.base_lm.norm(hidden_states)
        fsq_layer_out = self.voxcpm.fsq_layer(hidden_states[:, concat_text_len:])
        hidden_states = hidden_states[:, :concat_text_len]
        lm_hidden = torch.cat([hidden_states, fsq_layer_out], dim=1)[:, [-1]]
        hidden_states = torch.cat([hidden_states, fsq_layer_out + feat_embed], dim=1)

        i = self.voxcpm.base_lm.config.num_hidden_layers
        for layer in self.voxcpm.residual_lm.layers:

            # ── Self-Attention ───────────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * rotary_pos_emb_cos + self._rotate_half(qk) * rotary_pos_emb_sin
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(1, 2, 0, 3)
            if USE_F16_KV:
                k = k.half()
                v = v.half()
            k = k.permute(2, 1, 3, 0)
            v = v.transpose(0, 2)

            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.total_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v

            if USE_F16_KV:
                k = k.float()
                v = v.float()

            attn = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
            attn = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network ─────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            i += 1

        # ── Final Projection ─────────────────────────────────────────
        residual_hidden = hidden_states[:, [-1]]
        residual_hidden = self._rms_norm(residual_hidden)
        dit_hidden_1 = self.voxcpm.lm_to_dit_proj(lm_hidden)
        dit_hidden_2 = self.voxcpm.res_to_dit_proj(residual_hidden)
        dit_hidden = dit_hidden_1 + dit_hidden_2
        random = torch.randn((1, self.voxcpm.patch_size, self.voxcpm.feat_decoder.in_channels), dtype=torch.float32)
        stop_flag = self.voxcpm.stop_head(self.voxcpm.stop_actn(self.voxcpm.stop_proj(lm_hidden))).argmax(dim=-1, keepdims=False).int()
        return *self.save_key, *self.save_value, random, dit_hidden, stop_flag


# ══════════════════════════════════════════════════════════════════════════════
# Feature Decoder Module (Diffusion)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_FEAT_DECODER(torch.nn.Module):
    def __init__(self, voxcpm, fixed_timesteps):
        super(VOXCPM_FEAT_DECODER, self).__init__()
        self.voxcpm = voxcpm
        self.head_dim = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.num_heads
        self.num_key_value_heads = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.num_key_value_heads
        self.num_key_value_groups = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.num_key_value_groups
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        sway_sampling_coef = 1.0
        t_span = torch.linspace(1, 0, fixed_timesteps + 1, dtype=torch.float32)
        t_span = (t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span))[1:]
        t = self.voxcpm.feat_decoder.estimator.time_embeddings(t_span[:-1])
        t = self.voxcpm.feat_decoder.estimator.time_mlp(t)
        self.dt = (t_span[:-1] - t_span[1:]).view(1, 1, -1)
        if self.voxcpm.feat_decoder.mean_mode:
            dt_in = self.voxcpm.feat_decoder.estimator.delta_time_mlp(self.voxcpm.feat_decoder.estimator.time_embeddings(self.dt)).unsqueeze(0)
        else:
            dt_in = self.voxcpm.feat_decoder.estimator.delta_time_mlp(self.voxcpm.feat_decoder.estimator.time_embeddings(torch.tensor([0], dtype=torch.float32)))
        self.t = (t + dt_in).unsqueeze(0)
        self.prefix_plus = self.voxcpm.patch_size + 1
        self.q_len = 9  # Fixed to 9 for VoxCPM1.5 CFM
        position_ids = torch.arange(self.q_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = self.voxcpm.feat_decoder.estimator.decoder.rope_emb(position_ids)
        rope_emb_sin[:, :self.voxcpm.feat_decoder.estimator.decoder.rope_emb.dim // 2] *= -1.0
        self.rope_emb_cos = rope_emb_cos.view(1, self.q_len, 1, 1, -1)
        self.rope_emb_sin = rope_emb_sin.view(1, self.q_len, 1, 1, -1)
        self.split_size = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.head_dim // 2
        scale_factor = self.voxcpm.feat_decoder.estimator.decoder.layers._modules['0'].self_attn.head_dim ** -0.25
        norm_factor = self.voxcpm.feat_decoder.estimator.config.hidden_size ** 0.5
        with torch.no_grad():
            for layer in self.voxcpm.feat_decoder.estimator.decoder.layers:
                # 1) Fuse q/k/v into qkv
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                in_features = int(q_proj.in_features)
                out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
                has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
                qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
                if has_bias:
                    z = lambda feat: torch.zeros(feat, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
                    qb = q_proj.bias if q_proj.bias is not None else z(q_proj.out_features)
                    kb = k_proj.bias if k_proj.bias is not None else z(k_proj.out_features)
                    vb = v_proj.bias if v_proj.bias is not None else z(v_proj.out_features)
                    qkv.bias.copy_(torch.cat([qb * scale_factor, kb * scale_factor, vb], dim=0))

                layer.self_attn.q_out_features = int(q_proj.out_features)
                layer.self_attn.k_out_features = int(k_proj.out_features)
                layer.self_attn.v_out_features = int(v_proj.out_features)
                layer.self_attn.qkv = qkv

                del layer.self_attn.q_proj
                del layer.self_attn.k_proj
                del layer.self_attn.v_proj

                # 2) Fuse input rmsnorm weight
                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                del layer.input_layernorm

                # 3) Fuse post-attention rmsnorm weight
                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj

                in_feat = gate.in_features
                out_feat = gate.out_features + up.out_features
                gate_up = torch.nn.Linear(in_feat, out_feat, bias=False)

                gate_weight = gate.weight * w
                up_weight = up.weight * w
                gate_up.weight.copy_(torch.cat([gate_weight, up_weight], dim=0))

                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj
                del layer.mlp.up_proj
                del layer.post_attention_layernorm

            # 4) Fuse final norm weight into out_proj
            w = self.voxcpm.feat_decoder.estimator.decoder.norm.weight.unsqueeze(0) * norm_factor
            self.voxcpm.feat_decoder.estimator.out_proj.weight.mul_(w)
            del self.voxcpm.feat_decoder.estimator.decoder.norm

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        """Apply modified RMS normalization (with optional overflow scaling)."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True))

    def rotate_half(self, x):
        x = x.view(-1, self.q_len, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(-1, self.q_len, 1, self.qk_heads, self.head_dim)

    def forward(self, step, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus):
        t = self.t[:, step]
        dt = self.dt[..., step]
        dit_hidden = dit_hidden + t
        dit_hidden = torch.cat([dit_hidden, t], dim=0)
        x = self.voxcpm.feat_decoder.estimator.in_proj(random)
        x = torch.cat([x, x], dim=0)
        hidden_states = torch.cat([dit_hidden, feat_cond, x], dim=1)
        for layer in self.voxcpm.feat_decoder.estimator.decoder.layers:
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, self.q_len, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * self.rope_emb_cos + self.rotate_half(qk) * self.rope_emb_sin
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.q_len, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)
            attn = torch.softmax(torch.matmul(q, k), dim=-1)
            attn = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(-1, self.q_len, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        hidden_states = hidden_states[:, self.prefix_plus:]
        hidden_states = self._rms_norm(hidden_states)
        hidden_states = self.voxcpm.feat_decoder.estimator.out_proj(hidden_states)
        dphi_dt, cfg_dphi_dt = hidden_states.split([1, 1], dim=0)
        positive_flat = dphi_dt.view(1, 1, -1)
        negative_flat = cfg_dphi_dt.view(1, 1, -1)
        dot_product = (positive_flat * negative_flat).sum(-1, keepdim=True)
        squared_norm = negative_flat.square().sum(-1, keepdim=True)
        st_star = dot_product / squared_norm
        dphi_dt = cfg_value_minus * cfg_dphi_dt * st_star + cfg_value * dphi_dt
        next_random = random - dt * dphi_dt
        next_step = step + 1
        return next_step, next_random


# ══════════════════════════════════════════════════════════════════════════════
# VAE Decoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_VAE_DECODE(torch.nn.Module):
    def __init__(self, voxcpm, output_sample_rate):
        super(VOXCPM_VAE_DECODE, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self.scale = float(output_sample_rate / 44100.0)
        self.single_decode_len = self.voxcpm.patch_size * self.voxcpm.chunk_size

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def forward(self, latent_pred):
        decode_audio = self.voxcpm.audio_vae.decode(latent_pred.transpose(-1, -2))
        if self.scale < 1.0:
            decode_audio = torch.nn.functional.interpolate(
                decode_audio,
                scale_factor=self.scale,
                mode='linear',
                align_corners=False
            )
            decode_audio = (decode_audio * 32767.0).clamp(min=-32768.0, max=32767.0)
        elif self.scale > 1.0:
            decode_audio = decode_audio * 32767.0
            decode_audio = torch.nn.functional.interpolate(
                decode_audio,
                scale_factor=self.scale,
                mode='linear',
                align_corners=False
            )
            decode_audio = decode_audio.clamp(min=-32768.0, max=32767.0)
        else:
            decode_audio = (decode_audio * 32767.0).clamp(min=-32768.0, max=32767.0)
        audio_out_len = decode_audio.shape[-1].unsqueeze(0)
        return decode_audio.to(torch.int16), audio_out_len


if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # Load Model & Extract Config
        # ══════════════════════════════════════════════════════════════════
        model = VoxCPM.from_pretrained(path_voxcpm, load_denoiser=False, optimize=False).tts_model
        model = model.float().to('cpu').eval()

        base_lm_num_layers    = model.base_lm.config.num_hidden_layers
        residual_lm_num_layers = model.residual_lm.config.num_hidden_layers
        total_layers           = base_lm_num_layers + residual_lm_num_layers
        head_dim               = model.base_lm.layers._modules['0'].self_attn.head_dim
        num_kv_heads           = model.base_lm.layers._modules['0'].self_attn.num_key_value_heads
        hidden_size            = model.base_lm.embed_tokens.embedding_dim
        feat_hidden_size       = model.feat_encoder.config.hidden_size
        patch_size             = model.patch_size
        feat_dim               = model.feat_dim
        feat_in_channels       = model.feat_decoder.in_channels
        cond_proj_out          = model.feat_decoder.estimator.cond_proj.out_features

        residual_head_dim      = model.residual_lm.layers._modules['0'].self_attn.head_dim
        residual_num_kv_heads  = model.residual_lm.layers._modules['0'].self_attn.num_key_value_heads

        # ══════════════════════════════════════════════════════════════════
        # Build Dummy Tensors for Tracing
        # ══════════════════════════════════════════════════════════════════
        ids_len     = torch.tensor([25], dtype=torch.int64)
        history_len = torch.tensor([0], dtype=torch.int64)
        kv_seq_len  = ids_len + history_len
        mask        = torch.tensor([1], dtype=torch.int8)

        kv_dtype = torch.float16 if USE_F16_KV else torch.float32

        # KV cache spec: list of (name, concat_dim)
        kv_specs = [('key', 3), ('value', 2)]

        base_kv_tensors = {
            'key':   torch.zeros((num_kv_heads, 1, head_dim, history_len), dtype=kv_dtype),
            'value': torch.zeros((num_kv_heads, 1, history_len, head_dim), dtype=kv_dtype),
        }
        residual_kv_tensors = {
            'key':   torch.zeros((residual_num_kv_heads, 1, residual_head_dim, history_len), dtype=kv_dtype),
            'value': torch.zeros((residual_num_kv_heads, 1, history_len, residual_head_dim), dtype=kv_dtype),
        }

        # ══════════════════════════════════════════════════════════════════
        # Helper: Build KV I/O names, tensors, and dynamic axes
        # ══════════════════════════════════════════════════════════════════
        def get_kv_io(base_kv, residual_kv, base_layers, residual_layers, seq_axis='history_len', out_seq_axis='kv_seq_len'):
            inputs, in_names, out_names, axes = [], [], [], {}
            total = base_layers + residual_layers
            for name, dim in kv_specs:
                for i in range(base_layers):
                    in_n  = f'in_{name}_{i}'
                    out_n = f'out_{name}_{i}'
                    inputs.append(base_kv[name])
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n]  = {dim: seq_axis}
                    axes[out_n] = {dim: out_seq_axis}
                for i in range(base_layers, total):
                    in_n  = f'in_{name}_{i}'
                    out_n = f'out_{name}_{i}'
                    inputs.append(residual_kv[name])
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n]  = {dim: seq_axis}
                    axes[out_n] = {dim: out_seq_axis}
            return inputs, in_names, out_names, axes

        # ══════════════════════════════════════════════════════════════════
        # Export: Text_Embed
        # ══════════════════════════════════════════════════════════════════
        text_ids = torch.zeros([1, 10], dtype=torch.int32)
        torch.onnx.export(
            VOXCPM_TEXT_EMBED(model),
            (text_ids,),
            onnx_model_Text_Embed,
            input_names=['text_ids'],
            output_names=['text_embed'],
            dynamic_axes={
                'text_ids':    {1: 'ids_len'},
                'text_embed':  {1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del text_ids

        # ══════════════════════════════════════════════════════════════════
        # Export: VAE_Encoder
        # ══════════════════════════════════════════════════════════════════
        prompt_audio = torch.zeros([1, 1, MAX_PROMPT_AUDIO_LEN], dtype=torch.int16)
        torch.onnx.export(
            VOXCPM_VAE_ENCODER(model, IN_SAMPLE_RATE),
            (prompt_audio,),
            onnx_model_VAE_Encoder,
            input_names=['prompt_audio'],
            output_names=['audio_feat'],
            dynamic_axes={
                'prompt_audio': {2: 'audio_len'},
                'audio_feat':   {0: 'audio_feat_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del prompt_audio

        # ══════════════════════════════════════════════════════════════════
        # Export: Feat_Encoder
        # ══════════════════════════════════════════════════════════════════
        audio_feat = torch.zeros([20, patch_size, feat_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM_FEAT_ENCODER(model, MAX_PROMPT_AUDIO_LEN, IN_SAMPLE_RATE),
            (audio_feat,),
            onnx_model_Feat_Encoder,
            input_names=['audio_feat'],
            output_names=['feat_embed'],
            dynamic_axes={
                'audio_feat': {0: 'audio_feat_len'},
                'feat_embed': {1: 'audio_feat_len'},
            },
            opset_version=OPSET,
            dynamo=False
        )
        del audio_feat

        # ══════════════════════════════════════════════════════════════════
        # Export: Feat_Cond
        # ══════════════════════════════════════════════════════════════════
        audio_feat = torch.zeros([20, patch_size, feat_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM_FEAT_COND(model),
            (audio_feat,),
            onnx_model_Feat_Cond,
            input_names=['audio_feat'],
            output_names=['feat_cond'],
            dynamic_axes={
                'audio_feat': {0: 'audio_feat_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del audio_feat

        # ══════════════════════════════════════════════════════════════════
        # Export: Concat
        # ══════════════════════════════════════════════════════════════════
        embed_0 = torch.zeros([1, 10, feat_hidden_size], dtype=torch.float32)
        embed_1 = torch.zeros([1, 10, hidden_size], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM_CONCAT(),
            (embed_0, embed_1),
            onnx_model_Concat,
            input_names=['embed_0', 'embed_1'],
            output_names=['concat_embed', 'concat_len'],
            dynamic_axes={
                'embed_0':      {1: 'embed_len_0', 2: 'embed_size'},
                'embed_1':      {1: 'embed_len_1', 2: 'embed_size'},
                'concat_embed': {1: 'concat_len', 2: 'embed_size'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del embed_0, embed_1

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary + Mask (Prefill)
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            VOXCPM_ROTARY_MASK_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len, mask),
            onnx_model_Rotary_Mask_Text_Prefill,
            input_names=['ids_len', 'history_len', 'mask'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos':     {0: 'ids_len'},
                'rotary_sin':     {0: 'ids_len'},
                'attention_mask': {2: 'ids_len', 3: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary + Mask (Decode)
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            VOXCPM_ROTARY_MASK_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Rotary_Mask_Text_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len_next'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Main (Transformer Layers)
        # ══════════════════════════════════════════════════════════════════
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(
            base_kv_tensors, residual_kv_tensors,
            base_lm_num_layers, residual_lm_num_layers
        )

        concat_text_len = torch.tensor([10], dtype=torch.int64)
        feat_embed      = torch.zeros([1, ids_len - concat_text_len, feat_hidden_size], dtype=torch.float32)
        hidden_states   = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
        rotary_cos      = torch.zeros((ids_len, 1, 1, head_dim), dtype=torch.float32)
        rotary_sin      = torch.zeros((ids_len, 1, 1, head_dim), dtype=torch.float32)
        attention_mask  = torch.zeros((1, 1, ids_len, kv_seq_len), dtype=torch.float32)

        model_Main = VOXCPM_MAIN(model, MAX_SEQ_LEN)

        all_inputs   = kv_ins + [feat_embed, concat_text_len, hidden_states, rotary_cos, rotary_sin, attention_mask]
        input_names  = kv_in_names + ['feat_embed', 'concat_text_len', 'hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
        output_names = kv_out_names + ['random', 'dit_hidden', 'stop_flag']
        dynamic_axes = {
            **kv_axes,
            'feat_embed':     {1: 'audio_feat_len'},
            'hidden_states':  {1: 'ids_len'},
            'rotary_cos':     {0: 'ids_len'},
            'rotary_sin':     {0: 'ids_len'},
            'attention_mask': {2: 'ids_len', 3: 'kv_seq_len'}
        }

        torch.onnx.export(
            model_Main,
            tuple(all_inputs),
            onnx_model_Main,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del model_Main, all_inputs, feat_embed, hidden_states, rotary_cos, rotary_sin, attention_mask
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: Feat_Decoder (Diffusion)
        # ══════════════════════════════════════════════════════════════════
        model_Feat_Decoder = VOXCPM_FEAT_DECODER(model, FIXED_TIMESTEPS)
        step            = torch.tensor([0], dtype=torch.int32)
        random          = torch.ones((1, patch_size, feat_in_channels), dtype=torch.float32)
        dit_hidden      = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
        feat_cond       = torch.zeros((2, patch_size, cond_proj_out), dtype=torch.float32)
        cfg_value       = torch.tensor([CFG_VALUE], dtype=torch.float32)
        cfg_value_minus = torch.tensor([1.0 - CFG_VALUE], dtype=torch.float32)

        torch.onnx.export(
            model_Feat_Decoder,
            (step, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus),
            onnx_model_Feat_Decoder,
            input_names=['step', 'random', 'dit_hidden', 'feat_cond', 'cfg_value', 'cfg_value_minus'],
            output_names=['next_step', 'next_random'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )
        del model_Feat_Decoder, step, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus

        # ══════════════════════════════════════════════════════════════════
        # Export: VAE_Decoder
        # ══════════════════════════════════════════════════════════════════
        model_VAE_Decoder = VOXCPM_VAE_DECODE(model, OUT_SAMPLE_RATE)
        latent_pred = torch.ones((1, patch_size + patch_size, feat_in_channels), dtype=torch.float32)

        torch.onnx.export(
            model_VAE_Decoder,
            (latent_pred,),
            onnx_model_VAE_Decoder,
            input_names=['latent_pred'],
            output_names=['audio_out', 'audio_out_len'],
            dynamic_axes={
                'latent_pred': {1: 'latent_pred_len'},
                'audio_out':   {2: 'audio_out_len'}
            } if DYNAMIC_SHAPE_VAE_DECODE else None,
            opset_version=OPSET,
            dynamo=False
        )
        del model_VAE_Decoder, latent_pred, model
        gc.collect()

    print(
        '\nExport done!\n\n'
        'Start running the VoxCPM by ONNXRuntime.\n'
        'Now loading . . . it could cost minutes.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def audio_normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


def mask_multichar_chinese_tokens(tokenizer):
    multichar_tokens = {
        token for token in tokenizer.vocab.keys()
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        def __init__(self, base_tokenizer) -> None:
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs):
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []

            for token in tokens:
                clean_token = token.replace("▁", "")

                if clean_token in self.multichar_tokens:
                    chars = list(clean_token)
                    processed.extend(chars)
                else:
                    processed.append(token)

            return processed

        def __call__(self, text: str, **kwargs):
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)


def create_ort_with_data(data, dtype, device, device_id):
    """Create an OrtValue from a Python list/scalar."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    """Create a zero-filled OrtValue with the given shape."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    """Create an ORT InferenceSession with standard options."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,                 # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 **3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',   # ["default", "high_performance", "minimum_power"] ; Default (Gpus first), HighPerformance (GPUs first), LowPower (NPUs first)
        'device_filter':              'gpu',                # [gpu, npu, any],
        'disable_metacommands':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'               # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
# --- Text Embed ---
ort_session_Text_Embed = create_session(onnx_model_Text_Embed, **packed_settings)
in_name_Text_Embed     = get_in_names(ort_session_Text_Embed)[0]
out_name_Text_Embed    = [get_out_names(ort_session_Text_Embed)[0]]

# --- VAE Encoder ---
ort_session_VAE_Encoder = create_session(onnx_model_VAE_Encoder, **packed_settings)
in_name_VAE_Encoder     = get_in_names(ort_session_VAE_Encoder)[0]
out_name_VAE_Encoder    = [get_out_names(ort_session_VAE_Encoder)[0]]

# --- Feat Encoder ---
ort_session_Feat_Encoder = create_session(onnx_model_Feat_Encoder, **packed_settings)
in_name_Feat_Encoder     = get_in_names(ort_session_Feat_Encoder)[0]
out_name_Feat_Encoder    = [get_out_names(ort_session_Feat_Encoder)[0]]

# --- Feat Cond ---
ort_session_Feat_Cond    = create_session(onnx_model_Feat_Cond, **packed_settings)
model_dtype_Feat_Cond    = np.float16 if 'float16' in ort_session_Feat_Cond._inputs_meta[0].type else np.float32
in_name_Feat_Cond        = get_in_names(ort_session_Feat_Cond)[0]
out_name_Feat_Cond       = [get_out_names(ort_session_Feat_Cond)[0]]

# --- Concat ---
ort_session_Concat = create_session(onnx_model_Concat, **packed_settings)
in_name_Concat     = get_in_names(ort_session_Concat)
out_name_Concat    = get_out_names(ort_session_Concat)

# --- Rotary + Mask (Text Prefill) ---
ort_session_Rotary_Mask_Text_Prefill = create_session(onnx_model_Rotary_Mask_Text_Prefill, **packed_settings)
in_name_Rotary_Mask_Text_Prefill     = get_in_names(ort_session_Rotary_Mask_Text_Prefill)
out_name_Rotary_Mask_Text_Prefill    = get_out_names(ort_session_Rotary_Mask_Text_Prefill)

# --- Rotary + Mask (Text Decode) ---
ort_session_Rotary_Mask_Text_Decode = create_session(onnx_model_Rotary_Mask_Text_Decode, **packed_settings)
in_name_Rotary_Mask_Text_Decode     = get_in_names(ort_session_Rotary_Mask_Text_Decode)
out_name_Rotary_Mask_Text_Decode    = get_out_names(ort_session_Rotary_Mask_Text_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
print(f"\nUsable Providers: {ort_session_Main.get_providers()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
model_dtype_Main       = np.float16 if 'float16' in ort_session_Main._inputs_meta[0].type else np.float32
in_name_Main           = get_in_names(ort_session_Main)
out_name_Main          = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)

# --- Feat Decoder ---
ort_session_Feat_Decoder = create_session(onnx_model_Feat_Decoder, **packed_settings)
model_dtype_Feat_Decoder = np.float16 if 'float16' in ort_session_Feat_Decoder._inputs_meta[2].type else np.float32
in_name_Feat_Decoder     = get_in_names(ort_session_Feat_Decoder)
out_name_Feat_Decoder    = get_out_names(ort_session_Feat_Decoder)

# --- VAE Decoder ---
ort_session_VAE_Decoder    = create_session(onnx_model_VAE_Decoder, **packed_settings)
model_dtype_VAE_Decoder    = np.float16 if 'float16' in ort_session_VAE_Decoder._inputs_meta[0].type else np.float32
DYNAMIC_SHAPE_VAE_DECODE   = isinstance(ort_session_VAE_Decoder._inputs_meta[0].shape[1], str)
in_name_VAE_Decoder        = get_in_names(ort_session_VAE_Decoder)[0]
out_name_VAE_Decoder       = get_out_names(ort_session_VAE_Decoder)
half_decode_len            = 7056  # Fixed for VoxCPM1.5


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
generate_limit  = MAX_SEQ_LEN - 1
num_keys_values = amount_of_outputs_Main - 3
num_layers      = num_keys_values // 2

# Derived index offsets
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5

_meta = ort_session_Main._inputs_meta


# ══════════════════════════════════════════════════════════════════════════════
# STATIC ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
# --- Scalars & Lengths ---
init_history_len       = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_concat_text_len   = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)

# --- Special Tokens ---
init_audio_start_ids   = create_ort_with_data([[101]], np.int32, device_type, DEVICE_ID)

# --- Masks ---
init_mask_prefill          = create_ort_with_data([1], np.int8, device_type, DEVICE_ID)
init_decode_attention_mask = create_ort_with_shape((1, 1, 1, 1), model_dtype_Main, device_type, DEVICE_ID)

# --- KV Cache & Embedding Shapes ---
shape_keys   = (_meta[0].shape[0],          1, _meta[0].shape[2],          0)
shape_vals   = (_meta[num_layers].shape[0],  1, 0, _meta[num_layers].shape[3])
shape_embed  = (1, 0, _meta[num_keys_values].shape[2])
shape_latent = (ort_session_VAE_Decoder._inputs_meta[0].shape[0], 0, ort_session_VAE_Decoder._inputs_meta[0].shape[2])

init_past_keys_Main   = create_ort_with_shape(shape_keys,    model_dtype_Main,        device_type, DEVICE_ID)
init_past_values_Main = create_ort_with_shape(shape_vals,    model_dtype_Main,        device_type, DEVICE_ID)
init_feat_embed       = create_ort_with_shape(shape_embed,   model_dtype_Main,        device_type, DEVICE_ID)
init_latent_pred      = create_ort_with_shape(shape_latent,  model_dtype_VAE_Decoder, device_type, DEVICE_ID)

# --- CFG Values ---
cfg_value       = create_ort_with_data([CFG_VALUE],       model_dtype_Feat_Decoder, device_type, DEVICE_ID)
cfg_value_minus = create_ort_with_data([1.0 - CFG_VALUE], model_dtype_Feat_Decoder, device_type, DEVICE_ID)

# --- Time Steps ---
timesteps      = FIXED_TIMESTEPS - 1
init_cfm_steps = create_ort_with_data([0], np.int32, device_type, DEVICE_ID)

# --- Audio Post-processing ---
blank_segment = np.zeros((1, 1, int(OUT_SAMPLE_RATE * 0.1)), dtype=np.int16)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION SETUP & IO FEEDS
# ══════════════════════════════════════════════════════════════════════════════
input_feed_Text_Embed               = {}
input_feed_VAE_Encoder              = {}
input_feed_Feat_Encoder             = {}
input_feed_Feat_Cond                = {}
input_feed_Concat                   = {}
input_feed_Rotary_Mask_Text_Prefill = {}
input_feed_Rotary_Mask_Text_Decode  = {}
input_feed_Main                     = {}
input_feed_Feat_Decoder             = {}
input_feed_VAE_Decoder              = {}

# Audio Start Embedding
input_feed_Text_Embed[in_name_Text_Embed] = init_audio_start_ids
audio_start_embed = ort_session_Text_Embed.run_with_ort_values(out_name_Text_Embed, input_feed_Text_Embed, run_options=run_options)[0]

# Feat Cond Initialization
input_feed_Feat_Cond[in_name_Feat_Cond] = create_ort_with_shape((1, ort_session_Feat_Cond._inputs_meta[0].shape[1], ort_session_Feat_Cond._inputs_meta[0].shape[2]), model_dtype_Feat_Cond, device_type, DEVICE_ID)
init_feat_cond_0 = ort_session_Feat_Cond.run_with_ort_values(out_name_Feat_Cond, input_feed_Feat_Cond, run_options=run_options)[0]

# Feat Decoder: Fixed Inputs
input_feed_Feat_Decoder[in_name_Feat_Decoder[4]] = cfg_value
input_feed_Feat_Decoder[in_name_Feat_Decoder[5]] = cfg_value_minus

# Rotary Mask Prefill: Fixed Inputs
input_feed_Rotary_Mask_Text_Prefill[in_name_Rotary_Mask_Text_Prefill[1]] = init_history_len
input_feed_Rotary_Mask_Text_Prefill[in_name_Rotary_Mask_Text_Prefill[2]] = init_mask_prefill


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & PROMPT HANDLING
# ══════════════════════════════════════════════════════════════════════════════
tokenizer       = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(path_voxcpm))
text_normalizer = TextNormalizer()

if prompt_audio_path:
    if prompt_text:
        use_prompt_audio = True
        audio = np.array(AudioSegment.from_file(prompt_audio_path).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
        if USE_AUDIO_NORMALIZER:
            audio = audio_normalizer(audio)
        audio = onnxruntime.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), device_type, DEVICE_ID)
    else:
        use_prompt_audio = False
        print("Warning: No prompt text provided, so the prompt audio will be ignored.\n")
else:
    use_prompt_audio = False
    print("Info: No prompt audio provided, using ransom seed to generate voice.\n")

count_time = time.time()
if use_prompt_audio:
    # VAE Encoder
    input_feed_VAE_Encoder[in_name_VAE_Encoder] = audio
    audio_feat = ort_session_VAE_Encoder.run_with_ort_values(out_name_VAE_Encoder, input_feed_VAE_Encoder, run_options=run_options)[0]

    # Feat Cond
    input_feed_Feat_Cond[in_name_Feat_Cond] = audio_feat
    init_feat_cond = ort_session_Feat_Cond.run_with_ort_values(out_name_Feat_Cond, input_feed_Feat_Cond, run_options=run_options)[0]

    # Text Processing
    if USE_TEXT_NORMALIZER:
        prompt_text = text_normalizer.normalize(prompt_text)
    prompt_ids      = np.array([tokenizer(prompt_text)], dtype=np.int32)
    prompt_text_len = prompt_ids.shape[-1]

    # Text Embed
    input_feed_Text_Embed[in_name_Text_Embed] = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_ids, device_type, DEVICE_ID)
    prompt_embed = ort_session_Text_Embed.run_with_ort_values(out_name_Text_Embed, input_feed_Text_Embed, run_options=run_options)[0]
else:
    init_feat_cond  = init_feat_cond_0
    prompt_text_len = 0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
save_audio_out = []

for sentence in target_tts:
    print(f"Convert to Speech: {sentence}")
    if USE_TEXT_NORMALIZER:
        sentence = text_normalizer.normalize(sentence)

    # --- Encode Target Text ---
    target_ids = np.array([tokenizer(sentence)], dtype=np.int32)
    input_feed_Text_Embed[in_name_Text_Embed] = onnxruntime.OrtValue.ortvalue_from_numpy(target_ids, device_type, DEVICE_ID)
    target_embed = ort_session_Text_Embed.run_with_ort_values(out_name_Text_Embed, input_feed_Text_Embed, run_options=run_options)[0]

    # --- Combine Embeddings ---
    if use_prompt_audio:
        input_feed_Concat[in_name_Concat[0]] = prompt_embed
        input_feed_Concat[in_name_Concat[1]] = target_embed
        target_embed, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)

    input_feed_Concat[in_name_Concat[0]] = target_embed
    input_feed_Concat[in_name_Concat[1]] = audio_start_embed
    concat_embed, concat_text_len = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)

    # --- Calculate Max Length & Initial Features ---
    if use_prompt_audio:
        input_feed_Feat_Encoder[in_name_Feat_Encoder] = audio_feat
        feat_embed = ort_session_Feat_Encoder.run_with_ort_values(out_name_Feat_Encoder, input_feed_Feat_Encoder, run_options=run_options)[0]

        input_feed_Concat[in_name_Concat[0]] = concat_embed
        input_feed_Concat[in_name_Concat[1]] = feat_embed
        concat_embed, ids_len = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
    else:
        feat_embed = init_feat_embed
        ids_len    = concat_text_len

    max_len = min((concat_text_len.numpy() - prompt_text_len) * DECODE_LIMIT_FACTOR + 10, generate_limit - ids_len.numpy())

    # --- Rotary Embeddings & Causal Mask (Prefill) ---
    input_feed_Rotary_Mask_Text_Prefill[in_name_Rotary_Mask_Text_Prefill[0]] = ids_len
    rotary_cos, rotary_sin, attention_mask, kv_seq_len = ort_session_Rotary_Mask_Text_Prefill.run_with_ort_values(
        out_name_Rotary_Mask_Text_Prefill, input_feed_Rotary_Mask_Text_Prefill, run_options=run_options)

    # --- Prepare Main Decoder Inputs ---
    input_feed_Main[in_name_Main[num_keys_values]]          = feat_embed
    input_feed_Main[in_name_Main[num_keys_values_plus_1]]   = concat_text_len
    input_feed_Main[in_name_Main[num_keys_values_plus_2]]   = concat_embed
    input_feed_Main[in_name_Main[num_keys_values_plus_3]]   = rotary_cos
    input_feed_Main[in_name_Main[num_keys_values_plus_4]]   = rotary_sin
    input_feed_Main[in_name_Main[num_keys_values_plus_5]]   = attention_mask

    # Reset KV Cache
    for i in range(num_layers):
        input_feed_Main[in_name_Main[i]] = init_past_keys_Main
    for i in range(num_layers, num_keys_values):
        input_feed_Main[in_name_Main[i]] = init_past_values_Main

    feat_cond = init_feat_cond

    if not STREAMING:
        save_latent = init_latent_pred if DYNAMIC_SHAPE_VAE_DECODE else []

    # ──────────────────────────────────────────────────────────────────────────
    # AUTO-REGRESSIVE DECODING
    # ──────────────────────────────────────────────────────────────────────────
    num_decode   = 0
    start_decode = time.time()

    while num_decode < max_len:
        # --- Transformer ---
        all_outputs_Main = ort_session_Main.run_with_ort_values(out_name_Main, input_feed_Main, run_options=run_options)

        # --- Flow Matching / Diffusion ---
        input_feed_Feat_Decoder[in_name_Feat_Decoder[0]] = init_cfm_steps
        input_feed_Feat_Decoder[in_name_Feat_Decoder[1]] = all_outputs_Main[num_keys_values]
        input_feed_Feat_Decoder[in_name_Feat_Decoder[2]] = all_outputs_Main[num_keys_values_plus_1]
        input_feed_Feat_Decoder[in_name_Feat_Decoder[3]] = feat_cond

        for i in range(timesteps):
            all_outputs_Feat_Decoder = ort_session_Feat_Decoder.run_with_ort_values(out_name_Feat_Decoder, input_feed_Feat_Decoder, run_options=run_options)
            input_feed_Feat_Decoder[in_name_Feat_Decoder[0]] = all_outputs_Feat_Decoder[0]
            input_feed_Feat_Decoder[in_name_Feat_Decoder[1]] = all_outputs_Feat_Decoder[1]

        latent_pred = all_outputs_Feat_Decoder[1]

        # --- Handle Output ---
        if STREAMING:
            if num_decode < 1:
                pre_latent_pred = latent_pred
            else:
                input_feed_Concat[in_name_Concat[0]] = pre_latent_pred
                input_feed_Concat[in_name_Concat[1]] = latent_pred
                save_latent, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
                input_feed_VAE_Decoder[in_name_VAE_Decoder] = save_latent
                audio_out, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
                pre_latent_pred = latent_pred
                audio_out = audio_out.numpy()
                if num_decode > 1:
                    audio_out = audio_out[..., half_decode_len:]
                save_audio_out.append(audio_out)
        else:
            if DYNAMIC_SHAPE_VAE_DECODE:
                input_feed_Concat[in_name_Concat[0]] = save_latent
                input_feed_Concat[in_name_Concat[1]] = latent_pred
                save_latent, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
            else:
                save_latent.append(latent_pred)

        # --- Check Stop Token ---
        if num_decode >= MIN_SEQ_LEN and all_outputs_Main[num_keys_values_plus_2].numpy() in STOP_TOKEN:
            break

        # --- Update Inputs for Next Iteration ---
        input_feed_Feat_Encoder[in_name_Feat_Encoder] = latent_pred
        feat_embed = ort_session_Feat_Encoder.run_with_ort_values(out_name_Feat_Encoder, input_feed_Feat_Encoder, run_options=run_options)[0]

        input_feed_Feat_Cond[in_name_Feat_Cond] = latent_pred
        feat_cond = ort_session_Feat_Cond.run_with_ort_values(out_name_Feat_Cond, input_feed_Feat_Cond, run_options=run_options)[0]

        input_feed_Main.update(zip(in_name_Main[:num_keys_values], all_outputs_Main))
        input_feed_Main[in_name_Main[num_keys_values]]        = feat_embed
        input_feed_Main[in_name_Main[num_keys_values_plus_2]] = feat_embed

        # Rotary embeddings for next decode step
        input_feed_Rotary_Mask_Text_Decode[in_name_Rotary_Mask_Text_Decode[0]] = kv_seq_len
        rotary_cos, rotary_sin, kv_seq_len = ort_session_Rotary_Mask_Text_Decode.run_with_ort_values(
            out_name_Rotary_Mask_Text_Decode, input_feed_Rotary_Mask_Text_Decode, run_options=run_options)
        input_feed_Main[in_name_Main[num_keys_values_plus_3]] = rotary_cos
        input_feed_Main[in_name_Main[num_keys_values_plus_4]] = rotary_sin

        if num_decode < 1:
            input_feed_Main[in_name_Main[num_keys_values_plus_1]] = init_concat_text_len
            input_feed_Main[in_name_Main[num_keys_values_plus_5]] = init_decode_attention_mask

        num_decode += 1
        print(f"    Decode: {num_decode}")

    print(f"\nDecode Speed: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s\n")

    # ──────────────────────────────────────────────────────────────────────────
    # FINALIZE SENTENCE AUDIO (NON-STREAMING)
    # ──────────────────────────────────────────────────────────────────────────
    if not STREAMING:
        if DYNAMIC_SHAPE_VAE_DECODE:
            input_feed_VAE_Decoder[in_name_VAE_Decoder] = save_latent
            audio_out, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
            save_audio_out.append(audio_out.numpy())
        else:
            input_feed_Concat[in_name_Concat[0]] = save_latent[0]
            input_feed_Concat[in_name_Concat[1]] = save_latent[1]
            concat_latent, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
            input_feed_VAE_Decoder[in_name_VAE_Decoder] = concat_latent
            audio_out, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
            save_audio_out.append(audio_out.numpy())
            for i in range(2, len(save_latent)):
                input_feed_Concat[in_name_Concat[0]] = save_latent[i - 1]
                input_feed_Concat[in_name_Concat[1]] = save_latent[i]
                concat_latent, _ = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)
                input_feed_VAE_Decoder[in_name_VAE_Decoder] = concat_latent
                audio_out, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
                audio_out = audio_out.numpy()[..., half_decode_len:]
                save_audio_out.append(audio_out)

    save_audio_out.append(blank_segment)


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING & STATS
# ══════════════════════════════════════════════════════════════════════════════
cost_time = time.time() - count_time
audio_out = np.concatenate(save_audio_out, axis=-1).reshape(-1)
if USE_AUDIO_NORMALIZER:
    audio_out = audio_normalizer(audio_out)
sf.write(generated_audio_path, audio_out, OUT_SAMPLE_RATE, format='WAVEX')

total_audio_duration = (audio_out.shape[-1] - blank_segment.shape[-1] * len(target_tts)) / OUT_SAMPLE_RATE
rtf = cost_time / total_audio_duration

print(f"\nGenerate Complete.")
print(f"Saving to: {generated_audio_path}")
print(f"Time Cost: {cost_time:.3f} Seconds")
print(f"RTF: {rtf:.3f}")
