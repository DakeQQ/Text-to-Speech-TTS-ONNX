import gc
import json
import math
import re
import time
from pathlib import Path
import numpy as np
import onnxruntime
import soundfile as sf
import torch
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import LlamaTokenizerFast
from voxcpm.model.voxcpm2 import VoxCPM2Model


# ══════════════════════════════════════════════════════════════════════════════
# USER-CONFIGURABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
path_voxcpm2                  = r'/home/DakeQQ/Downloads/VoxCPM2'

onnx_model_VAE_Encoder        = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_AudioVAE_Encode.onnx'
onnx_model_Feat_Encoder_Cond  = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Feat_Encoder_Cond.onnx'
onnx_model_Assemble           = {
                                    "voice_design":   r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Assemble_VoiceDesign.onnx',
                                    "continuation":   r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Assemble_Continuation.onnx',
                                    "reference_only": r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Assemble_ReferenceOnly.onnx',
                                    "combined":       r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Assemble_Combined.onnx',
}
onnx_model_Prefill            = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Prefill.onnx'
onnx_model_Rotary_Mask_Decode = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Rotary_Mask_Decode.onnx'
onnx_model_Main               = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Main.onnx'
onnx_model_Feat_Decoder       = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Feat_Decoder.onnx'
onnx_model_VAE_Decoder        = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_AudioVAE_Decode.onnx'
onnx_model_Concat             = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM2_Concat.onnx'

# === Prompt / target ===
reference_audio_path = "../example/basic_ref_zh.wav"                 # Speaker identity reference. Used by reference_only / combined.
prompt_audio_path    = "../example/basic_ref_zh.wav"                 # Continuation prompt audio. Used by continuation / combined.
prompt_text          = "对，这就是我，万人敬仰的太乙真人。"                # Transcript for prompt_audio_path. Used by continuation / combined.


DEMO_CONFIGS = [
    {
        "mode": "voice_design",             # voice_design — 你对模型说「用年轻女声说话」，它凭空造一个声音。没有任何参考音频输入。/ You tell the model to "speak in a young female voice," and it creates a voice out of thin air. There is no reference audio input.
        "reference_audio_path": None,
        "prompt_audio_path": None,
        "prompt_text": None,
        "target_texts": [
            "(用年轻女声说话)大家好，我现在正在大可奇奇体验AI科技。",
            "(speak in a young female voice)Hello everyone, I'm currently experiencing DakeQQ's AI technology.",
        ],
    },
    {
        "mode": "reference_only",           # reference_only — 你给一段参考音频，模型学它的「音色」（谁的声音），但语气语速由模型自己决定。/ You provide a reference audio clip, and the model learns its "timbre" (whose voice it is), but the tone and pace are determined by the model itself.
        "reference_audio_path": reference_audio_path,
        "prompt_audio_path": None,
        "prompt_text": None,
        "target_texts": [
            "大家好，我现在正在大可奇奇体验AI科技。",
            "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
        ],
    },
    {
        "mode": "continuation",             # continuation — 你给一段「上文音频 + 上文文字」，模型会像在接着说一样延续那段话的风格和韵律。音色也由 prompt 音频决定。/ You provide "the preceding audio + the preceding text," and the model continues the style and rhythm of that passage as if it were speaking again. The timbre is also determined by the prompt audio.
        "reference_audio_path": None,
        "prompt_audio_path": prompt_audio_path,
        "prompt_text": "对，这就是我，万人敬仰的太乙真人。",
        "target_texts": [
            "大家好，我现在正在大可奇奇体验AI科技。",
            "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
        ],
    },
    {
        "mode": "combined",                 # combined — reference 管「音色是谁」，prompt 管「怎么说」。两者可以是不同音频。 / `reference` determines "whose voice it is," and `prompt` determines "how it is spoken." The two can be different audio clips.
        "reference_audio_path": reference_audio_path,
        "prompt_audio_path": prompt_audio_path,
        "prompt_text": "对，这就是我，万人敬仰的太乙真人。",
        "target_texts": [
            "大家好，我现在正在大可奇奇体验AI科技。",
            "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
        ],
    },
]

# Model Config
DO_EXPORT = True                         # Whether to export the ONNX models

# === Decoding limits & tokens ===
STOP_TOKEN = [1]                         # The stop_id for VoxCPM2
MAX_SEQ_LEN = 2048                       # Max decode length; cannot be changed after export.
MIN_SEQ_LEN = 2                          # Min decode length before checking stop token.
DECODE_LIMIT_FACTOR = 6                  # Decode length limit factor, integer >= 1.

# === Audio configuration ===
IN_SAMPLE_RATE = 16000                   # Input audio sample rate (fixed at export time).
OUT_SAMPLE_RATE = 48000                  # Output audio sample rate (fixed at export time).

# === Guidance, diffusion & randomness ===
FIXED_TIMESTEPS = 10                     # Fixed timesteps for CFM diffusion; larger is finer but slower.
CFG_VALUE = 2.0                          # Classifier-free guidance scale.
RANDOM_SEED = 9527                       # Global random seed.

# === Feature flags ===
STREAMING = False                        # Enable streaming synthesis. Streaming-only Concat model exported.
USE_TEXT_NORMALIZER = True               # Use text normalizer.
USE_AUDIO_NORMALIZER = False             # Use audio normalizer to stabilize loudness.
PREVENT_F16_OVERFLOW = False             # Prevent float16 overflow. Set True for Q4F16/Q8F16/F16 quantization.
USE_F16_KV = True                        # Use float16 for key/value cache.

# === ONNX / runtime configuration ===
ORT_LOG = False                          # Enable ONNX Runtime logging for debugging.
ORT_FP16 = False                         # Set to True for FP16 ONNX Runtime settings.
ORT_Accelerate_Providers = []            # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
OPSET = 18                               # ONNX opset version.
MAX_THREADS = 0                          # Parallel CPU threads, 0 for auto.
DEVICE_ID = 0                            # Device id.


# ══════════════════════════════════════════════════════════════════════════════
# VAE Encoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_VAE_ENCODER(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_size = model.patch_size
        self.latent_dim = model.audio_vae.latent_dim
        self.patch_len = model.patch_size * math.prod(model.audio_vae.encoder_rates)

        encoder = model.audio_vae.encoder
        # Stage 0: initial conv (1 → encoder_dim)
        self.init_conv = encoder.block[0]
        # Stages 1-N: encoder blocks (each doubles channels, strides down)
        self.enc_blocks = torch.nn.ModuleList([encoder.block[i] for i in range(1, len(encoder.block))])
        # Final projection to latent space
        self.fc_mu = encoder.fc_mu

        # Fuse weights at init: remove weight norm from ALL convolutions and
        # precompute snake alpha reciprocals to eliminate runtime recomputation.
        with torch.no_grad():
            # Remove weight norm from init_conv and fuse 1/32768 normalization
            torch.nn.utils.remove_weight_norm(self.init_conv)
            self.init_conv.weight.mul_(1.0 / 32768.0)

            # Remove weight norm from all encoder block convolutions
            for block in self.enc_blocks:
                for unit_idx in range(3):  # 3 residual units per block
                    unit = block.block[unit_idx]
                    torch.nn.utils.remove_weight_norm(unit.block[1])  # dilated conv
                    torch.nn.utils.remove_weight_norm(unit.block[3])  # pointwise conv
                    # Precompute snake inv_alpha for each residual unit
                    unit.block[0].inv_alpha = (unit.block[0].alpha + 1e-9).reciprocal()
                    unit.block[2].inv_alpha = (unit.block[2].alpha + 1e-9).reciprocal()
                # Precompute snake inv_alpha before downsample
                block.block[3].inv_alpha = (block.block[3].alpha + 1e-9).reciprocal()
                # Remove weight norm from downsample conv
                torch.nn.utils.remove_weight_norm(block.block[4])

            # Remove weight norm from fc_mu projection
            torch.nn.utils.remove_weight_norm(self.fc_mu)

        # Pre-allocate int8 zero buffer for padding (sliced + cast at runtime)
        self.pad_buffer = torch.zeros((1, 1, self.patch_len), dtype=torch.int8)
        self.pad_buffer_right = torch.zeros((1, 1, self.patch_len), dtype=torch.float32)

    @staticmethod
    def _snake(x, alpha, inv_alpha):
        """Snake activation: x + (1/α) * sin²(αx), with precomputed inv_alpha."""
        return x + inv_alpha * torch.sin(alpha * x).square()

    def _residual_unit(self, x, unit):
        """CausalResidualUnit: Snake → DilatedConv → Snake → PointwiseConv, then residual add."""
        residual = x
        x = self._snake(x, unit.block[0].alpha, unit.block[0].inv_alpha)
        x = unit.block[1](x)
        x = self._snake(x, unit.block[2].alpha, unit.block[2].inv_alpha)
        x = unit.block[3](x)
        return residual + x

    def forward(self, audio):
        audio = audio.float()

        pad_len_left = self.patch_len - audio.shape[-1] % self.patch_len
        pad_buffer_left = self.pad_buffer[..., :pad_len_left].float()
        audio = torch.cat([pad_buffer_left, audio, self.pad_buffer_right], dim=-1)

        # Stage 0: Initial causal conv (1 → 128, k=7)
        x = self.init_conv(audio)

        # Stages 1-4: Encoder blocks (channels: 128→256→512→1024→2048, strides: 2,5,8,8)
        for block in self.enc_blocks:
            # Residual unit (dilation=1)
            x = self._residual_unit(x, block.block[0])
            # Residual unit (dilation=3)
            x = self._residual_unit(x, block.block[1])
            # Residual unit (dilation=9)
            x = self._residual_unit(x, block.block[2])
            # Snake activation before downsample
            x = self._snake(x, block.block[3].alpha, block.block[3].inv_alpha)
            # Strided downsample conv
            x = block.block[4](x)

        # fc_mu projection (2048 → latent_dim=64, k=3)
        latent = self.fc_mu(x)

        latent = latent.view(1, self.latent_dim, -1, self.patch_size)
        return latent.permute(0, 2, 3, 1).contiguous()


# ══════════════════════════════════════════════════════════════════════════════
# Fused Feature Encoder + Conditioning Module (2 calls → 1 call)
# Returns both feat_embed (for LM) and feat_cond (for diffusion) in one pass.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_FEAT_ENCODER_COND(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._replace_gelu_with_tanh_approximation(model.feat_encoder)

        encoder = model.feat_encoder.encoder
        layer0 = encoder.layers._modules['0']
        self.head_dim = layer0.self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = layer0.self_attn.num_heads
        self.num_key_value_heads = layer0.self_attn.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads = self.num_heads + self.num_key_value_heads

        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        self.rms_eps = torch.tensor([encoder.config.rms_norm_eps * encoder.config.hidden_size], dtype=torch.float32)
        if PREVENT_F16_OVERFLOW:
            self.rms_eps *= self.overflow_scale.square()

        self.q_len = model.patch_size + 1
        position_ids = torch.arange(self.q_len, dtype=torch.long)
        rope_emb_cos, rope_emb_sin = encoder.rope_emb(position_ids)
        rope_emb_sin[:, :encoder.rope_emb.dim // 2] *= -1.0
        self.rope_emb_cos = rope_emb_cos.view(1, self.q_len, 1, 1, -1)
        self.rope_emb_sin = rope_emb_sin.view(1, self.q_len, 1, 1, -1)

        self.special_tokens = model.feat_encoder.special_token.expand(1, MAX_SEQ_LEN, 1, -1).contiguous().half()

        norm_factor = encoder.config.hidden_size ** 0.5
        scale_factor = self.head_dim ** -0.25
        with torch.no_grad():
            for layer in encoder.layers:
                self._fuse_qkv(layer, scale_factor, norm_factor)
                self._fuse_gate_up(layer, norm_factor)
            # Absorb final norm into enc_to_lm_proj
            w = encoder.norm.weight.unsqueeze(0) * norm_factor
            model.enc_to_lm_proj.weight.mul_(w)
            del encoder.norm

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                VOXCPM2_FEAT_ENCODER_COND._replace_gelu_with_tanh_approximation(child)

    def _fuse_qkv(self, layer, scale_factor, norm_factor):
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj) * scale_factor, _get_bias(k_proj) * scale_factor, _get_bias(v_proj)], dim=0))
        layer.self_attn.q_out_features = int(q_proj.out_features)
        layer.self_attn.k_out_features = int(k_proj.out_features)
        layer.self_attn.qkv = qkv
        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj
        w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(w)
        del layer.input_layernorm

    def _fuse_gate_up(self, layer, norm_factor):
        w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * w, up.weight * w], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    def _rms_norm(self, x):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + self.rms_eps)

    def rotate_half(self, x):
        x = x.view(-1, self.q_len, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(-1, self.q_len, 1, self.qk_heads, self.head_dim)

    def forward(self, audio_feat):
        # audio_feat: (batch, seq_len, patch_size, feat_dim)
        seq_len = audio_feat.shape[1]

        # === Feature Encoder: produces feat_embed for the LM ===
        hidden_states = self.model.feat_encoder.in_proj(audio_feat)
        special_tokens = self.special_tokens[:, :seq_len, :, :].float()
        hidden_states = torch.cat([special_tokens, hidden_states], dim=2)
        hidden_states = hidden_states.reshape(seq_len, self.q_len, -1)

        for layer in self.model.feat_encoder.encoder.layers:
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
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        feat_embed = hidden_states[:, 0]
        feat_embed = self._rms_norm(feat_embed)
        feat_embed = self.model.enc_to_lm_proj(feat_embed)
        feat_embed = feat_embed.unsqueeze(0)

        # === Feature Conditioning: produces feat_cond for diffusion ===
        # Use last patch from input audio_feat for conditioning
        last_patch = audio_feat[:, [-1]]  # (batch, 1, patch_size, feat_dim)
        last_patch_squeezed = last_patch.squeeze(0)  # (patch_size, feat_dim)
        feat_cond = self.model.feat_decoder.estimator.cond_proj(last_patch_squeezed)  # (1, ps, cond_dim)
        feat_cond = torch.cat([feat_cond, feat_cond], dim=0)  # (2, ps, cond_dim)

        return feat_embed, feat_cond


# ══════════════════════════════════════════════════════════════════════════════
# Fused Prefill Module (Text_Embed + Segment Concat + Feat Extraction + Rotary_Mask)
# Uses segment indices to directly concat text_embed and feat_embed at their
# respective positions, eliminating the mask-multiply mixing approach.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_PREFILL(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        self.embed_tokens = model.base_lm.embed_tokens
        use_mup = bool(getattr(model.base_lm.config, 'use_mup', False))
        scale_emb = float(getattr(model.base_lm.config, 'scale_emb', 1)) if use_mup else 1.0
        if scale_emb != 1.0:
            with torch.no_grad():
                self.embed_tokens.weight.mul_(scale_emb)

        # Pre-allocate int8 causal attention mask buffer (sliced at runtime)
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

        # Precompute rotary embeddings
        position_ids = torch.arange(max_seq_len, dtype=torch.long)
        rope_emb_cos, rope_emb_sin = model.base_lm.rope_emb(position_ids)
        dim = rope_emb_cos.shape[-1]
        rope_emb_sin[:, :dim // 2] *= -1.0
        cos = rope_emb_cos.unsqueeze(1).unsqueeze(1)
        sin = rope_emb_sin.unsqueeze(1).unsqueeze(1)
        self.register_buffer("cos_rotary_pos_emb", cos.half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half(), persistent=False)

    def forward(self, text_ids, ids_len, feat_embed, audio_seg1_start, audio_seg1_end, concat_text_len, history_len):
        # 1. Embed text tokens
        text_embed = self.embed_tokens(text_ids)  # [1, seq_len, hidden]

        # 2. Concat embeddings by segment: text positions use text_embed, audio positions use feat_embed
        #    Layout: [text_before | audio_seg1 | text_after | trailing_audio]
        seg1_text = text_embed[:, :audio_seg1_start]
        seg2_audio = feat_embed[:, audio_seg1_start:audio_seg1_end]
        seg3_text = text_embed[:, audio_seg1_end:concat_text_len]
        seg4_audio = feat_embed[:, concat_text_len:]
        combined_embed = torch.cat([seg1_text, seg2_audio, seg3_text, seg4_audio], dim=1)

        # 3. Extract audio portions of feat_embed for Main model
        feat_embed_audio = torch.cat([seg2_audio, seg4_audio], dim=1)

        # 4. Compute rotary embeddings and causal mask
        kv_seq_len = ids_len + history_len                      # Add op prevents optimizer from removing this output
        rotary_cos = self.cos_rotary_pos_emb[:ids_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:ids_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()

        return combined_embed, feat_embed_audio, rotary_cos, rotary_sin, attention_mask, kv_seq_len


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding (Decode Only)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_ROTARY_MASK_DECODE(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.long)
        rope_emb_cos, rope_emb_sin = model.base_lm.rope_emb(position_ids)
        dim = rope_emb_cos.shape[-1]
        rope_emb_sin[:, :dim // 2] *= -1.0
        cos = rope_emb_cos.unsqueeze(1).unsqueeze(1)
        sin = rope_emb_sin.unsqueeze(1).unsqueeze(1)
        self.register_buffer("cos_rotary_pos_emb", cos.half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Fused Main Transformer (Base LM + Residual LM)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_MAIN(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        self.model = model
        self._replace_gelu_with_tanh_approximation(model)

        layer0 = model.base_lm.layers._modules['0']
        self.head_dim = layer0.self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = layer0.self_attn.num_heads
        self.num_key_value_heads = layer0.self_attn.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads = self.num_heads + self.num_key_value_heads

        self.base_layer_count = len(model.base_lm.layers)
        self.residual_layer_count = len(model.residual_lm.layers)
        self.total_layers = self.base_layer_count + self.residual_layer_count

        self.norm_factor = model.base_lm.config.hidden_size ** 0.5
        self.rms_eps = torch.tensor([model.base_lm.config.rms_norm_eps * model.base_lm.config.hidden_size], dtype=torch.float32)
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        if PREVENT_F16_OVERFLOW:
            self.rms_eps *= self.overflow_scale.square()
        scale_factor = self.head_dim ** -0.25

        self.use_mup = bool(getattr(model.base_lm.config, 'use_mup', False))
        if self.use_mup:
            scale_depth = float(getattr(model.base_lm.config, 'scale_depth', 1.0))
            base_num_hidden = int(model.base_lm.config.num_hidden_layers)
            residual_num_hidden = int(model.residual_lm.config.num_hidden_layers)
            self.base_mup_scale = scale_depth / math.sqrt(base_num_hidden)
            self.residual_mup_scale = scale_depth / math.sqrt(residual_num_hidden)

        self.save_key = [None] * self.total_layers
        self.save_value = [None] * self.total_layers

        # Pre-allocate zero buffer for feat_padded construction (avoids zeros_like)
        hidden_size = int(model.base_lm.config.hidden_size)
        self.zero_buffer = torch.zeros([1, max_seq_len, hidden_size], dtype=torch.int8)

        self._fuse_weights(scale_factor)

    def _fuse_weights(self, scale_factor):
        with torch.no_grad():
            for layer in self.model.base_lm.layers:
                self._fuse_qkv_projection(layer, scale_factor)
                self._fuse_gate_up_projection(layer)
                if self.use_mup:
                    layer.self_attn.o_proj.weight.mul_(self.base_mup_scale)
                    if layer.self_attn.o_proj.bias is not None:
                        layer.self_attn.o_proj.bias.mul_(self.base_mup_scale)
                    layer.mlp.down_proj.weight.mul_(self.base_mup_scale)
                    if layer.mlp.down_proj.bias is not None:
                        layer.mlp.down_proj.bias.mul_(self.base_mup_scale)
            for layer in self.model.residual_lm.layers:
                self._fuse_qkv_projection(layer, scale_factor)
                self._fuse_gate_up_projection(layer)
                if self.use_mup:
                    layer.self_attn.o_proj.weight.mul_(self.residual_mup_scale)
                    if layer.self_attn.o_proj.bias is not None:
                        layer.self_attn.o_proj.bias.mul_(self.residual_mup_scale)
                    layer.mlp.down_proj.weight.mul_(self.residual_mup_scale)
                    if layer.mlp.down_proj.bias is not None:
                        layer.mlp.down_proj.bias.mul_(self.residual_mup_scale)
            # Absorb residual_lm.norm into res_to_dit_proj
            final_norm_weight = self.model.residual_lm.norm.weight.unsqueeze(0) * self.norm_factor
            self.model.res_to_dit_proj.weight.mul_(final_norm_weight)
            del self.model.residual_lm.norm
            # Fuse lm_to_dit_proj and stop_proj (both take lm_hidden)
            lm_dit = self.model.lm_to_dit_proj
            stop = self.model.stop_proj
            self.lm_dit_out = int(lm_dit.out_features)
            self.stop_out = int(stop.out_features)
            has_bias = (lm_dit.bias is not None) or (stop.bias is not None)
            fused = torch.nn.Linear(int(lm_dit.in_features), self.lm_dit_out + self.stop_out, bias=has_bias)
            fused.weight.copy_(torch.cat([lm_dit.weight, stop.weight], dim=0))
            if has_bias:
                def _get_b(proj):
                    return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=lm_dit.weight.dtype, device=lm_dit.weight.device)
                fused.bias.copy_(torch.cat([_get_b(lm_dit), _get_b(stop)], dim=0))
            self.lm_dit_stop_proj = fused
            del self.model.lm_to_dit_proj, self.model.stop_proj

    def _fuse_qkv_projection(self, layer, scale_factor):
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj) * scale_factor, _get_bias(k_proj) * scale_factor, _get_bias(v_proj)], dim=0))
        layer.self_attn.q_out_features = int(q_proj.out_features)
        layer.self_attn.k_out_features = int(k_proj.out_features)
        layer.self_attn.v_out_features = int(v_proj.out_features)
        layer.self_attn.qkv = qkv
        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * self.norm_factor
        qkv.weight.mul_(input_norm_weight)
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer):
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * self.norm_factor
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                VOXCPM2_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + self.rms_eps)

    def _rotate_half(self, x):
        x = x.view(-1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(-1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        feat_embed = all_inputs[-8]
        audio_seg1_start = all_inputs[-7]
        audio_seg1_end = all_inputs[-6]
        concat_text_len = all_inputs[-5]
        hidden_states = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask = all_inputs[-1]

        # === BASE LM LAYERS (with rotary) ===
        for i, layer in enumerate(self.model.base_lm.layers):
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
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # === BASE NORM (kept at runtime — feeds both fsq_layer and fusion) ===
        hidden_states = self.model.base_lm.norm(hidden_states)

        # === FSQ SPLIT (universal 4-segment) ===
        text_before = hidden_states[:, :audio_seg1_start]
        audio_mid = self.model.fsq_layer(hidden_states[:, audio_seg1_start:audio_seg1_end])
        text_after = hidden_states[:, audio_seg1_end:concat_text_len]
        audio_trailing = self.model.fsq_layer(hidden_states[:, concat_text_len:])
        full_hidden = torch.cat([text_before, audio_mid, text_after, audio_trailing], dim=1)
        lm_hidden = full_hidden[:, [-1]]

        # === FEAT_PADDED (zeros at text positions, feat_embed at audio positions) ===
        seg1_pad = self.zero_buffer[:, :audio_seg1_start].float()
        seg2_feat = feat_embed[:, :audio_seg1_end - audio_seg1_start]
        seg3_pad = self.zero_buffer[:, :concat_text_len - audio_seg1_end].float()
        seg4_feat = feat_embed[:, audio_seg1_end - audio_seg1_start:]
        feat_padded = torch.cat([seg1_pad, seg2_feat, seg3_pad, seg4_feat], dim=1)
        hidden_states = self.model.fusion_concat_proj(torch.cat([full_hidden, feat_padded], dim=-1))

        # === RESIDUAL LM LAYERS (NO rotary) ===
        i = self.base_layer_count
        for layer in self.model.residual_lm.layers:
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            # NO rotary for residual layers
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
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            i += 1

        # === FINAL OUTPUTS ===
        residual_hidden = hidden_states[:, [-1]]
        residual_hidden = self._rms_norm(residual_hidden)
        # Fused lm_to_dit_proj + stop_proj (single matmul, then split)
        lm_dit_stop = self.lm_dit_stop_proj(lm_hidden)
        dit_hidden_1, stop_hidden = torch.split(lm_dit_stop, [self.lm_dit_out, self.stop_out], dim=-1)
        dit_hidden_2 = self.model.res_to_dit_proj(residual_hidden)
        # VoxCPM2: concatenate along seq dim for 2-token mu → (1, 2, dit_hidden_dim)
        dit_hidden = torch.cat([dit_hidden_1, dit_hidden_2], dim=1)

        random = torch.randn((1, self.model.patch_size, self.model.feat_decoder.in_channels), dtype=torch.float32)
        stop_flag = self.model.stop_head(self.model.stop_actn(stop_hidden)).argmax(dim=-1, keepdims=False).int()

        return *self.save_key, *self.save_value, random, dit_hidden, stop_flag


# ══════════════════════════════════════════════════════════════════════════════
# Feature Decoder Module (Full Diffusion Loop — all timesteps unrolled)
# Reduces timesteps session.run() calls to 1.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_FEAT_DECODER(torch.nn.Module):
    def __init__(self, model, fixed_timesteps):
        super().__init__()
        self.model = model
        self._replace_gelu_with_tanh_approximation(model.feat_decoder)

        decoder = model.feat_decoder.estimator.decoder
        layer0 = decoder.layers._modules['0']
        self.head_dim = layer0.self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = layer0.self_attn.num_heads
        self.num_key_value_heads = layer0.self_attn.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads = self.num_heads + self.num_key_value_heads

        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        self.rms_eps = torch.tensor([model.feat_decoder.estimator.config.rms_norm_eps * model.feat_decoder.estimator.config.hidden_size], dtype=torch.float32)
        if PREVENT_F16_OVERFLOW:
            self.rms_eps *= self.overflow_scale.square()

        self.patch_size = model.patch_size
        self.dit_hidden_dim = model.feat_decoder.estimator.config.hidden_size
        self.timesteps = fixed_timesteps

        # Pre-allocate mu_zeros buffer (1, 2, dit_hidden_dim)
        self.register_buffer("mu_zeros", torch.zeros(1, 2, self.dit_hidden_dim, dtype=torch.float32), persistent=False)

        # Pre-compute time embeddings for all steps
        sway_sampling_coef = 1.0
        t_span = torch.linspace(1, 0, fixed_timesteps + 1, dtype=torch.float32)
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        self.zero_init_steps = max(1, int(t_span.numel() * 0.04))

        active_t = t_span[self.zero_init_steps:-1]
        active_dt = t_span[self.zero_init_steps:-1] - t_span[self.zero_init_steps + 1:]
        self.timesteps = int(active_t.numel())
        self.dt = active_dt.view(1, 1, -1)

        mean_mode = getattr(model.feat_decoder, 'mean_mode', False)
        if self.timesteps > 0:
            t_embeds = model.feat_decoder.estimator.time_mlp(
                model.feat_decoder.estimator.time_embeddings(active_t)
            )
            if mean_mode:
                dt_embed = model.feat_decoder.estimator.delta_time_mlp(
                    model.feat_decoder.estimator.time_embeddings(active_dt)
                )
            else:
                dt_embed = model.feat_decoder.estimator.delta_time_mlp(
                    model.feat_decoder.estimator.time_embeddings(torch.zeros(1, dtype=torch.float32))
                )
            # Pre-compute all time embeddings (avoids time_mlp/time_embeddings at runtime)
            self.t_in_all = t_embeds + dt_embed  # (timesteps, hidden_dim)
        else:
            self.t_in_all = torch.empty(0, self.dit_hidden_dim, dtype=torch.float32)
        self.t_in_all = self.t_in_all.unsqueeze(0)
        self.t_in_all = torch.cat([self.t_in_all, self.t_in_all], dim=0)

        # VoxCPM2 DiT layout: [mu(2), t(1), cond(ps), x(ps)]
        self.q_len = 2 + 1 + self.patch_size + self.patch_size
        self.prefix_skip = 2 + 1 + self.patch_size

        # Pre-compute rotary
        position_ids = torch.arange(self.q_len, dtype=torch.long)
        rope_emb_cos, rope_emb_sin = decoder.rope_emb(position_ids)
        rope_emb_sin[:, :decoder.rope_emb.dim // 2] *= -1.0
        self.rope_emb_cos = rope_emb_cos.view(1, self.q_len, 1, 1, -1)
        self.rope_emb_sin = rope_emb_sin.view(1, self.q_len, 1, 1, -1)

        # Fuse all decoder layer weights
        norm_factor = model.feat_decoder.estimator.config.hidden_size ** 0.5
        scale_factor = self.head_dim ** -0.25
        with torch.no_grad():
            for layer in decoder.layers:
                self._fuse_qkv(layer, scale_factor, norm_factor)
                self._fuse_gate_up(layer, norm_factor)
            # Absorb final norm into out_proj
            w = decoder.norm.weight.unsqueeze(0) * norm_factor
            model.feat_decoder.estimator.out_proj.weight.mul_(w)
            del decoder.norm

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                VOXCPM2_FEAT_DECODER._replace_gelu_with_tanh_approximation(child)

    def _fuse_qkv(self, layer, scale_factor, norm_factor):
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj) * scale_factor, _get_bias(k_proj) * scale_factor, _get_bias(v_proj)], dim=0))
        layer.self_attn.q_out_features = int(q_proj.out_features)
        layer.self_attn.k_out_features = int(k_proj.out_features)
        layer.self_attn.qkv = qkv
        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj
        w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(w)
        del layer.input_layernorm

    def _fuse_gate_up(self, layer, norm_factor):
        w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate = layer.mlp.gate_proj
        up = layer.mlp.up_proj
        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * w, up.weight * w], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    def _rms_norm(self, x):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + self.rms_eps)

    def rotate_half(self, x):
        x = x.view(-1, self.q_len, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(-1, self.q_len, 1, self.qk_heads, self.head_dim)

    def _single_step(self, step, random, mu_in, feat_cond, dt_cfg, dt_cfg_minus):
        """Run a single diffusion step."""
        t_in = self.t_in_all[:, step]  # (1, 1, dit_dim)

        # Input projection for x
        x = self.model.feat_decoder.estimator.in_proj(random)  # (1, ps, dit_dim)
        x_in = torch.cat([x, x], dim=0)  # (2, ps, dit_dim)

        # Build sequence: [mu(2), t(1), cond(ps), x(ps)]
        hidden_states = torch.cat([mu_in, t_in, feat_cond, x_in], dim=1)  # (2, q_len, dit_dim)

        for layer in self.model.feat_decoder.estimator.decoder.layers:
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
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # Extract x positions, apply absorbed norm + out_proj
        hidden_states = hidden_states[:, self.prefix_skip:]
        hidden_states = self._rms_norm(hidden_states)
        dphi_dt = self.model.feat_decoder.estimator.out_proj(hidden_states)

        # CFG combination with precomputed dt*cfg products
        dphi_dt_positive, cfg_dphi_dt = dphi_dt.split([1, 1], dim=0)
        positive_flat = dphi_dt_positive.view(1, 1, -1)
        negative_flat = cfg_dphi_dt.view(1, 1, -1)
        dot_product = (positive_flat * negative_flat).sum(-1, keepdim=True)
        squared_norm = negative_flat.square().sum(-1, keepdim=True)
        st_star = dot_product / (squared_norm + 1e-8)

        # Euler step with fused dt*cfg scaling
        next_random = random - dt_cfg_minus * cfg_dphi_dt * st_star - dt_cfg * dphi_dt_positive
        return next_random

    def forward(self, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus):
        """Full diffusion loop unrolled, matching UnifiedCFM.solve_euler()."""
        # dit_hidden: (1, 2, dit_hidden_dim) — already correct shape from Main
        mu_in = torch.cat([dit_hidden, self.mu_zeros], dim=0)  # (2, 2, dit_dim)

        for step in range(self.timesteps):
            dt_step = self.dt[..., [step]]
            random = self._single_step([step], random, mu_in, feat_cond, dt_step * cfg_value, dt_step * cfg_value_minus)
        # Output (1, 1, patch_size, feat_in_channels) — matches Feat_Encoder_Cond input directly
        return random.unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# VAE Decoder Module
# Full decode pipeline inlined:
#   AudioVAE.decode(z, sr_cond)
#     → CausalDecoder.forward(z, sr_cond)
#       → bucketize sr_cond → init_conv_dw → init_conv_pw
#       → for each block: SampleRateConditionLayer(scale_bias) → CausalDecoderBlock
#         → CausalDecoderBlock: Snake → TransposeConv → ResUnit(d=1,3,9)
#           → CausalResidualUnit: Snake → DilConv → Snake → PwConv + residual
#       → final Snake → Conv → Tanh
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_VAE_DECODE(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._replace_gelu_with_tanh_approximation(model.audio_vae)
        self.patch_size = model.patch_size
        self.latent_dim = model.audio_vae.latent_dim

        decoder = model.audio_vae.decoder

        # Sample-rate conditioning boundaries
        self.sr_bin_boundaries = decoder.sr_bin_boundaries

        # Initial depthwise + pointwise convolutions (depthwise=True layout)
        self.init_conv_dw = decoder.model[0]
        self.init_conv_pw = decoder.model[1]

        # Decoder blocks and their paired sr-conditioning layers
        self.dec_blocks = torch.nn.ModuleList()
        self.sr_cond_layers = torch.nn.ModuleList()
        for i, layer in enumerate(decoder.model):
            if hasattr(layer, 'input_channels'):  # CausalDecoderBlock
                self.dec_blocks.append(layer)
                self.sr_cond_layers.append(decoder.sr_cond_model[i])

        # Final layers (after last decoder block)
        num_prefix = 2 + len(self.dec_blocks)  # 2 init convs + N blocks
        self.final_snake = decoder.model[num_prefix]      # Snake1d
        self.final_conv = decoder.model[num_prefix + 1]   # WNCausalConv1d → 1 channel
        # Tanh is inlined

        # Fuse weights at init: remove weight norm from ALL convolutions and
        # precompute snake alpha reciprocals to eliminate runtime recomputation.
        with torch.no_grad():
            # Remove weight norm from initial convolutions
            torch.nn.utils.remove_weight_norm(self.init_conv_dw)
            torch.nn.utils.remove_weight_norm(self.init_conv_pw)

            # Precompute inv_alpha for final snake
            self.final_snake.inv_alpha = (self.final_snake.alpha + 1e-9).reciprocal()

            # Remove weight norm from final conv
            torch.nn.utils.remove_weight_norm(self.final_conv)

            # Remove weight norm and precompute inv_alpha for all decoder blocks
            for block in self.dec_blocks:
                # block.block[0] = Snake1d before upsample
                block.block[0].inv_alpha = (block.block[0].alpha + 1e-9).reciprocal()
                # block.block[1] = WNCausalTransposeConv1d (strided upsample)
                torch.nn.utils.remove_weight_norm(block.block[1])
                # block.block[2..4] = 3 CausalResidualUnits (dilation=1,3,9)
                for unit_idx in range(2, 5):
                    unit = block.block[unit_idx]
                    # unit.block[0] = Snake1d, unit.block[1] = WNCausalConv1d (dilated)
                    # unit.block[2] = Snake1d, unit.block[3] = WNCausalConv1d (pointwise)
                    unit.block[0].inv_alpha = (unit.block[0].alpha + 1e-9).reciprocal()
                    torch.nn.utils.remove_weight_norm(unit.block[1])
                    unit.block[2].inv_alpha = (unit.block[2].alpha + 1e-9).reciprocal()
                    torch.nn.utils.remove_weight_norm(unit.block[3])

            # Remove weight norm from sr_cond out_layer convolutions and precompute inv_alpha
            for sr_cond_layer in self.sr_cond_layers:
                if hasattr(sr_cond_layer, 'out_layer') and not isinstance(sr_cond_layer.out_layer, torch.nn.Identity):
                    # out_layer = Sequential(Snake1d, WNCausalConv1d)
                    sr_cond_layer.out_layer[0].inv_alpha = (sr_cond_layer.out_layer[0].alpha + 1e-9).reciprocal()
                    torch.nn.utils.remove_weight_norm(sr_cond_layer.out_layer[1])

    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                VOXCPM2_VAE_DECODE._replace_gelu_with_tanh_approximation(child)

    @staticmethod
    def _snake(x, alpha, inv_alpha):
        """Snake activation: x + (1/α) * sin²(αx), with precomputed inv_alpha."""
        return x + inv_alpha * torch.sin(alpha * x).square()

    def _residual_unit(self, x, unit):
        """CausalResidualUnit: Snake → DilatedConv → Snake → PointwiseConv, then residual add."""
        residual = x
        x = self._snake(x, unit.block[0].alpha, unit.block[0].inv_alpha)
        x = unit.block[1](x)
        x = self._snake(x, unit.block[2].alpha, unit.block[2].inv_alpha)
        x = unit.block[3](x)
        return residual + x

    def _decoder_block(self, x, block):
        """CausalDecoderBlock: Snake → TransposeConv(upsample) → 3× residual units."""
        x = self._snake(x, block.block[0].alpha, block.block[0].inv_alpha)
        x = block.block[1](x)
        x = self._residual_unit(x, block.block[2])    # dilation=1
        x = self._residual_unit(x, block.block[3])    # dilation=3
        x = self._residual_unit(x, block.block[4])    # dilation=9
        return x

    def _apply_sr_cond(self, x, sr_cond_layer, sr_idx):
        """SampleRateConditionLayer (scale_bias): x * scale + bias, then out_layer."""
        scale = sr_cond_layer.scale_embed(sr_idx).unsqueeze(-1)
        bias = sr_cond_layer.bias_embed(sr_idx).unsqueeze(-1)
        x = x * scale + bias
        if hasattr(sr_cond_layer, 'out_layer') and not isinstance(sr_cond_layer.out_layer, torch.nn.Identity):
            x = self._snake(x, sr_cond_layer.out_layer[0].alpha, sr_cond_layer.out_layer[0].inv_alpha)
            x = sr_cond_layer.out_layer[1](x)
        return x

    def forward(self, latent_patches, sr_cond):
        # Reshape latent patches: (B, seq, patch, D) → (B, D, seq*patch)
        x = latent_patches.permute(0, 3, 1, 2).reshape(1, self.latent_dim, -1)

        # Bucketize sample rate → conditioning index
        sr_idx = torch.bucketize(sr_cond, self.sr_bin_boundaries)

        # Stage 0: Initial depthwise conv (latent_dim → latent_dim, k=7, grouped)
        x = self.init_conv_dw(x)
        # Stage 1: Pointwise conv (latent_dim → decoder_dim, k=1)
        x = self.init_conv_pw(x)

        # Stages 2-N: Decoder blocks with sample-rate conditioning
        # Each block: sr_cond(scale_bias) → Snake → TransposeConv(upsample) → 3× ResUnits
        for block, sr_cond_layer in zip(self.dec_blocks, self.sr_cond_layers):
            x = self._apply_sr_cond(x, sr_cond_layer, sr_idx)
            x = self._decoder_block(x, block)

        # Final: Snake → Conv(→1ch) → Tanh → int16 PCM
        x = self._snake(x, self.final_snake.alpha, self.final_snake.inv_alpha)
        x = self.final_conv(x)
        audio = torch.tanh(x)
        audio = (audio * 32767.0).to(torch.int16)

        return audio


# ══════════════════════════════════════════════════════════════════════════════
# Pre-Process Assembly Modules (one per mode — no control flow in forward())
# Moves all numpy token/mask/feat assembly into ONNX.
# Takes raw text_ids + ref/prompt audio feats → produces assembled tensors
# ready for Feat_Encoder_Cond and Prefill, eliminating runtime numpy ops.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_ASSEMBLE_VOICE_DESIGN(torch.nn.Module):
    def __init__(self, patch_size, latent_dim, max_seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        # len=1 static buffers (no cast needed)
        self.register_buffer("audio_seg1_start", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("audio_seg1_end", torch.zeros(1, dtype=torch.int64))
        # Pre-allocated int8 buffers (slice+cast at runtime)
        self.zero_buffer_4d = torch.zeros((1, max_seq_len, patch_size, latent_dim), dtype=torch.int8)

    def forward(self, text_ids):
        text_len = text_ids.shape[1]
        text_token = text_ids
        audio_feat = self.zero_buffer_4d[:, :text_len].float()
        concat_text_len = text_ids.shape[1].unsqueeze(0)
        ids_len = text_token.shape[1].unsqueeze(0)
        return text_token, audio_feat, self.audio_seg1_start, self.audio_seg1_end, concat_text_len, ids_len


class VOXCPM2_ASSEMBLE_CONTINUATION(torch.nn.Module):
    def __init__(self, patch_size, latent_dim, max_seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        # len=1 static buffers (no cast needed)
        self.register_buffer("audio_seg1_start", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("audio_seg1_end", torch.zeros(1, dtype=torch.int64))
        # Pre-allocated int8 buffers (slice+cast at runtime)
        self.zero_buffer_2d = torch.zeros((1, max_seq_len), dtype=torch.int8)
        self.zero_buffer_4d = torch.zeros((1, max_seq_len, patch_size, latent_dim), dtype=torch.int8)

    def forward(self, text_ids, prompt_audio_feat):
        text_len = text_ids.shape[1]
        prompt_len = prompt_audio_feat.shape[1]
        prompt_zeros = self.zero_buffer_2d[:, :prompt_len].int()
        text_token = torch.cat([text_ids, prompt_zeros], dim=1)
        text_pad = self.zero_buffer_4d[:, :text_len].float()
        audio_feat = torch.cat([text_pad, prompt_audio_feat], dim=1)
        concat_text_len = text_ids.shape[1].unsqueeze(0)
        ids_len = text_token.shape[1].unsqueeze(0)
        return text_token, audio_feat, self.audio_seg1_start, self.audio_seg1_end, concat_text_len, ids_len


class VOXCPM2_ASSEMBLE_REFERENCE_ONLY(torch.nn.Module):
    def __init__(self, patch_size, latent_dim, max_seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.register_buffer("zero_frame", torch.zeros((1, 1, patch_size, latent_dim), dtype=torch.float32))
        self.register_buffer("ref_start_token", torch.tensor([[103]], dtype=torch.int32))
        self.register_buffer("ref_end_token", torch.tensor([[104]], dtype=torch.int32))
        # len=1 static buffer (no cast needed)
        self.register_buffer("audio_seg1_start", torch.ones(1, dtype=torch.int64))
        # Pre-allocated int8 buffers (slice+cast at runtime)
        self.zero_buffer_2d = torch.zeros((1, max_seq_len), dtype=torch.int8)
        self.zero_buffer_4d = torch.zeros((1, max_seq_len, patch_size, latent_dim), dtype=torch.int8)

    def forward(self, text_ids, ref_audio_feat):
        text_len = text_ids.shape[1]
        ref_len = ref_audio_feat.shape[1]
        ref_zeros = self.zero_buffer_2d[:, :ref_len].int()
        text_token = torch.cat([self.ref_start_token, ref_zeros, self.ref_end_token, text_ids], dim=1)

        text_pad = self.zero_buffer_4d[:, :text_len].to(ref_audio_feat.dtype)
        audio_feat = torch.cat([self.zero_frame, ref_audio_feat, self.zero_frame, text_pad], dim=1)

        audio_seg1_end = (ref_len + 1).unsqueeze(0)
        concat_text_len = text_token.shape[1].unsqueeze(0)
        ids_len = text_token.shape[1].unsqueeze(0)
        return text_token, audio_feat, self.audio_seg1_start, audio_seg1_end, concat_text_len, ids_len


class VOXCPM2_ASSEMBLE_COMBINED(torch.nn.Module):
    def __init__(self, patch_size, latent_dim, max_seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.register_buffer("zero_frame", torch.zeros((1, 1, patch_size, latent_dim)))
        self.register_buffer("ref_start_token", torch.tensor([[103]], dtype=torch.int32))
        self.register_buffer("ref_end_token", torch.tensor([[104]], dtype=torch.int32))
        # len=1 static buffer (no cast needed)
        self.register_buffer("audio_seg1_start", torch.ones(1, dtype=torch.int64))
        # Pre-allocated int8 buffers (slice+cast at runtime)
        self.zero_buffer_2d = torch.zeros((1, max_seq_len), dtype=torch.int8)
        self.zero_buffer_4d = torch.zeros((1, max_seq_len, patch_size, latent_dim), dtype=torch.int8)

    def forward(self, text_ids, ref_audio_feat, prompt_audio_feat):
        text_len = text_ids.shape[1]
        ref_len = ref_audio_feat.shape[1]
        prompt_len = prompt_audio_feat.shape[1]
        ref_zeros = self.zero_buffer_2d[:, :ref_len].int()
        prompt_zeros = self.zero_buffer_2d[:, :prompt_len].int()
        text_token = torch.cat([self.ref_start_token, ref_zeros, self.ref_end_token, text_ids, prompt_zeros], dim=1)

        text_pad = self.zero_buffer_4d[:, :text_len].float()
        audio_feat = torch.cat([self.zero_frame, ref_audio_feat, self.zero_frame, text_pad, prompt_audio_feat], dim=1)

        audio_seg1_end = (ref_len + 1).unsqueeze(0)
        concat_text_len = (ref_len + 2 + text_len).unsqueeze(0)
        ids_len = text_token.shape[1].unsqueeze(0)
        return text_token, audio_feat, self.audio_seg1_start, audio_seg1_end, concat_text_len, ids_len


# ══════════════════════════════════════════════════════════════════════════════
# Concatenation Module (Streaming only)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_CONCAT(torch.nn.Module):
    def forward(self, embed_0, embed_1):
        concat_embed = torch.cat([embed_0, embed_1], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
if DO_EXPORT:
    print('Export start ...')
    Path(onnx_model_VAE_Encoder).parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        # Load Model
        model_dir = Path(path_voxcpm2).expanduser().resolve()
        model = VoxCPM2Model.from_local(str(model_dir), optimize=False, device='cpu')
        model = model.to(torch.float32).to('cpu').eval()

        # Read model config directly
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        lm_config = config["lm_config"]
        audio_vae_config = config["audio_vae_config"]

        hidden_size = lm_config["hidden_size"]
        head_dim = lm_config["kv_channels"]
        num_kv_heads = lm_config["num_key_value_heads"]
        base_layers = lm_config["num_hidden_layers"]
        residual_layers = config["residual_lm_num_layers"]
        total_layers = base_layers + residual_layers
        patch_size = config["patch_size"]
        feat_dim = config["feat_dim"]
        latent_dim = audio_vae_config["latent_dim"]
        encode_patch_len = patch_size * math.prod(audio_vae_config["encoder_rates"])
        feat_in_channels = model.feat_decoder.in_channels
        dit_hidden_dim = model.feat_decoder.estimator.config.hidden_size
        cond_proj_out = model.feat_decoder.estimator.cond_proj.out_features

        kv_dtype = torch.float16 if USE_F16_KV else torch.float32

        kv_specs = [('key', 3), ('value', 2)]

        base_kv_tensors = {
            'key': torch.zeros((num_kv_heads, 1, head_dim, 0), dtype=kv_dtype),
            'value': torch.zeros((num_kv_heads, 1, 0, head_dim), dtype=kv_dtype),
        }
        residual_kv_tensors = {
            'key': torch.zeros((num_kv_heads, 1, head_dim, 0), dtype=kv_dtype),
            'value': torch.zeros((num_kv_heads, 1, 0, head_dim), dtype=kv_dtype),
        }

        def get_kv_io(base_kv, residual_kv, n_base, n_residual, seq_axis='history_len', out_seq_axis='kv_seq_len'):
            inputs, in_names, out_names, axes = [], [], [], {}
            n_total = n_base + n_residual
            for name, dim in kv_specs:
                for idx in range(n_base):
                    in_n = f'in_{name}_{idx}'
                    out_n = f'out_{name}_{idx}'
                    inputs.append(base_kv[name])
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n] = {dim: seq_axis}
                    axes[out_n] = {dim: out_seq_axis}
                for idx in range(n_base, n_total):
                    in_n = f'in_{name}_{idx}'
                    out_n = f'out_{name}_{idx}'
                    inputs.append(residual_kv[name])
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n] = {dim: seq_axis}
                    axes[out_n] = {dim: out_seq_axis}
            return inputs, in_names, out_names, axes

        # ══════════════════════════════════════════════════════════════
        # Export: AudioVAE_Encode
        # ══════════════════════════════════════════════════════════════
        print('Exporting AudioVAE_Encode ...')
        prompt_audio = torch.zeros([1, 1, encode_patch_len * 2], dtype=torch.int16)
        torch.onnx.export(
            VOXCPM2_VAE_ENCODER(model),
            (prompt_audio,),
            onnx_model_VAE_Encoder,
            input_names=['audio'],
            output_names=['audio_feat'],
            dynamic_axes={'audio': {2: 'audio_samples'}, 'audio_feat': {1: 'audio_feat_len'}},
            opset_version=OPSET,
            dynamo=False
        )
        del prompt_audio

        # ══════════════════════════════════════════════════════════════
        # Export: Feat_Encoder_Cond (Fused)
        # ══════════════════════════════════════════════════════════════
        print('Exporting Feat_Encoder_Cond (fused) ...')
        audio_feat = torch.zeros([1, 10, patch_size, feat_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM2_FEAT_ENCODER_COND(model),
            (audio_feat,),
            onnx_model_Feat_Encoder_Cond,
            input_names=['audio_feat'],
            output_names=['feat_embed', 'feat_cond'],
            dynamic_axes={'audio_feat': {1: 'audio_feat_len'}, 'feat_embed': {1: 'audio_feat_len'}},
            opset_version=OPSET,
            dynamo=False
        )
        del audio_feat

        # ══════════════════════════════════════════════════════════════
        # Export: Assemble (per-mode — no control flow in each forward())
        # ══════════════════════════════════════════════════════════════
        _asm_text_ids = torch.zeros([1, 15], dtype=torch.int32)
        _asm_out_names = ['text_token', 'audio_feat', 'audio_seg1_start', 'audio_seg1_end', 'concat_text_len', 'ids_len']
        _asm_dyn_out = {'text_token': {1: 'total_len'}, 'audio_feat': {1: 'total_len'}}

        # voice_design
        print('Exporting Assemble (voice_design) ...')
        torch.onnx.export(
            VOXCPM2_ASSEMBLE_VOICE_DESIGN(patch_size, latent_dim, MAX_SEQ_LEN),
            (_asm_text_ids,),
            onnx_model_Assemble["voice_design"],
            input_names=['text_ids'],
            output_names=_asm_out_names,
            dynamic_axes={'text_ids': {1: 'text_len'}, **_asm_dyn_out},
            opset_version=OPSET,
            dynamo=False
        )

        # continuation
        print('Exporting Assemble (continuation) ...')
        _asm_prompt_feat = torch.zeros([1, 8, patch_size, latent_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM2_ASSEMBLE_CONTINUATION(patch_size, latent_dim, MAX_SEQ_LEN),
            (_asm_text_ids, _asm_prompt_feat),
            onnx_model_Assemble["continuation"],
            input_names=['text_ids', 'prompt_audio_feat'],
            output_names=_asm_out_names,
            dynamic_axes={'text_ids': {1: 'text_len'}, 'prompt_audio_feat': {1: 'prompt_len'}, **_asm_dyn_out},
            opset_version=OPSET,
            dynamo=False
        )

        # reference_only
        print('Exporting Assemble (reference_only) ...')
        _asm_ref_feat = torch.zeros([1, 5, patch_size, latent_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM2_ASSEMBLE_REFERENCE_ONLY(patch_size, latent_dim, MAX_SEQ_LEN),
            (_asm_text_ids, _asm_ref_feat),
            onnx_model_Assemble["reference_only"],
            input_names=['text_ids', 'ref_audio_feat'],
            output_names=_asm_out_names,
            dynamic_axes={'text_ids': {1: 'text_len'}, 'ref_audio_feat': {1: 'ref_len'}, **_asm_dyn_out},
            opset_version=OPSET,
            dynamo=False
        )

        # combined
        print('Exporting Assemble (combined) ...')
        torch.onnx.export(
            VOXCPM2_ASSEMBLE_COMBINED(patch_size, latent_dim, MAX_SEQ_LEN),
            (_asm_text_ids, _asm_ref_feat, _asm_prompt_feat),
            onnx_model_Assemble["combined"],
            input_names=['text_ids', 'ref_audio_feat', 'prompt_audio_feat'],
            output_names=_asm_out_names,
            dynamic_axes={'text_ids': {1: 'text_len'}, 'ref_audio_feat': {1: 'ref_len'}, 'prompt_audio_feat': {1: 'prompt_len'}, **_asm_dyn_out},
            opset_version=OPSET,
            dynamo=False
        )
        del _asm_text_ids, _asm_ref_feat, _asm_prompt_feat

        # ══════════════════════════════════════════════════════════════
        # Export: Prefill (Fused Text_Embed + Segment Concat + Feat Extraction + Rotary_Mask)
        # ══════════════════════════════════════════════════════════════
        print('Exporting Prefill (fused embed+concat+extract+rotary) ...')
        _prefill_seq_len = 25
        _prefill_audio_seg1_len = 5
        _prefill_text_after = 10
        text_ids = torch.zeros([1, _prefill_seq_len], dtype=torch.int32)
        ids_len = torch.tensor([_prefill_seq_len], dtype=torch.int64)
        feat_embed_dummy = torch.zeros([1, _prefill_seq_len, hidden_size], dtype=torch.float32)
        audio_seg1_start = torch.tensor([1], dtype=torch.int64)
        audio_seg1_end = torch.tensor([1 + _prefill_audio_seg1_len], dtype=torch.int64)
        concat_text_len_export = torch.tensor([1 + _prefill_audio_seg1_len + _prefill_text_after], dtype=torch.int64)
        history_len = torch.tensor([0], dtype=torch.int64)
        torch.onnx.export(
            VOXCPM2_PREFILL(model, MAX_SEQ_LEN),
            (text_ids, ids_len, feat_embed_dummy, audio_seg1_start, audio_seg1_end, concat_text_len_export, history_len),
            onnx_model_Prefill,
            input_names=['text_ids', 'ids_len', 'feat_embed', 'audio_seg1_start', 'audio_seg1_end', 'concat_text_len', 'history_len'],
            output_names=['combined_embed', 'feat_embed_audio', 'rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'text_ids': {1: 'seq_len'},
                'feat_embed': {1: 'seq_len'},
                'combined_embed': {1: 'seq_len'},
                'feat_embed_audio': {1: 'audio_feat_len'},
                'rotary_cos': {0: 'seq_len'},
                'rotary_sin': {0: 'seq_len'},
                'attention_mask': {2: 'seq_len', 3: 'seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del text_ids, feat_embed_dummy

        # ══════════════════════════════════════════════════════════════
        # Export: Rotary + Mask (Decode)
        # ══════════════════════════════════════════════════════════════
        print('Exporting Rotary_Mask_Decode ...')
        kv_seq_len = ids_len + history_len
        torch.onnx.export(
            VOXCPM2_ROTARY_MASK_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Rotary_Mask_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len_next'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════
        # Export: Main (Fused Base + Residual Transformer)
        # ══════════════════════════════════════════════════════════════
        print('Exporting Main (fused transformer) ...')
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(
            base_kv_tensors, residual_kv_tensors, base_layers, residual_layers
        )
        concat_text_len = torch.tensor([10], dtype=torch.int64)
        audio_seg1_start = torch.tensor([0], dtype=torch.int64)
        audio_seg1_end = torch.tensor([0], dtype=torch.int64)
        feat_embed = torch.zeros([1, int(ids_len) - int(concat_text_len), hidden_size], dtype=torch.float32)
        hidden_states = torch.ones((1, int(ids_len), hidden_size), dtype=torch.float32)
        rotary_cos = torch.zeros((int(ids_len), 1, 1, head_dim), dtype=torch.float32)
        rotary_sin = torch.zeros((int(ids_len), 1, 1, head_dim), dtype=torch.float32)
        attention_mask = torch.zeros((1, 1, int(ids_len), int(kv_seq_len)), dtype=torch.float32)

        model_Main = VOXCPM2_MAIN(model, MAX_SEQ_LEN)

        all_inputs = kv_ins + [feat_embed, audio_seg1_start, audio_seg1_end, concat_text_len, hidden_states, rotary_cos, rotary_sin, attention_mask]
        input_names = kv_in_names + ['feat_embed', 'audio_seg1_start', 'audio_seg1_end', 'concat_text_len', 'hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
        output_names = kv_out_names + ['random', 'dit_hidden', 'stop_flag']
        dynamic_axes = {
            **kv_axes,
            'feat_embed': {1: 'audio_feat_len'},
            'hidden_states': {1: 'ids_len'},
            'rotary_cos': {0: 'ids_len'},
            'rotary_sin': {0: 'ids_len'},
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

        # ══════════════════════════════════════════════════════════════
        # Export: Feat_Decoder (Full Diffusion Loop — no step input)
        # ══════════════════════════════════════════════════════════════
        print('Exporting Feat_Decoder (full loop) ...')
        model_Feat_Decoder = VOXCPM2_FEAT_DECODER(model, FIXED_TIMESTEPS)
        random = torch.ones((1, patch_size, feat_in_channels), dtype=torch.float32)
        dit_hidden = torch.zeros((1, 2, dit_hidden_dim), dtype=torch.float32)
        feat_cond = torch.zeros((2, patch_size, cond_proj_out), dtype=torch.float32)
        cfg_value_t = torch.tensor([2.0], dtype=torch.float32)
        cfg_value_minus_t = torch.tensor([-1.0], dtype=torch.float32)

        torch.onnx.export(
            model_Feat_Decoder,
            (random, dit_hidden, feat_cond, cfg_value_t, cfg_value_minus_t),
            onnx_model_Feat_Decoder,
            input_names=['random', 'dit_hidden', 'feat_cond', 'cfg_value', 'cfg_value_minus'],
            output_names=['latent_pred'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )
        del model_Feat_Decoder, random, dit_hidden, feat_cond, cfg_value_t, cfg_value_minus_t

        # ══════════════════════════════════════════════════════════════
        # Export: AudioVAE_Decode
        # ══════════════════════════════════════════════════════════════
        print('Exporting AudioVAE_Decode ...')
        model_VAE_Decoder = VOXCPM2_VAE_DECODE(model)
        latent_patches = torch.ones((1, 4, patch_size, latent_dim), dtype=torch.float32)
        sr_cond = torch.tensor([OUT_SAMPLE_RATE], dtype=torch.int32)

        torch.onnx.export(
            model_VAE_Decoder,
            (latent_patches, sr_cond),
            onnx_model_VAE_Decoder,
            input_names=['latent_patches', 'sr_cond'],
            output_names=['audio'],
            dynamic_axes={'latent_patches': {1: 'latent_seq_len'}, 'audio': {2: 'audio_len'}},
            opset_version=OPSET,
            dynamo=False
        )
        del model_VAE_Decoder, latent_patches, sr_cond

        # ══════════════════════════════════════════════════════════════
        # Export: Concat (Streaming only)
        # ══════════════════════════════════════════════════════════════
        print('Exporting Concat (streaming) ...')
        embed_0 = torch.zeros([1, 1, patch_size, latent_dim], dtype=torch.float32)
        embed_1 = torch.zeros([1, 1, patch_size, latent_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM2_CONCAT(),
            (embed_0, embed_1),
            onnx_model_Concat,
            input_names=['embed_0', 'embed_1'],
            output_names=['concat_embed', 'concat_len'],
            dynamic_axes={
                'embed_0': {1: 'embed_len_0'},
                'embed_1': {1: 'embed_len_1'},
                'concat_embed': {1: 'concat_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del embed_0, embed_1

        del model
        gc.collect()

    print(
        '\nExport done!\n\n'
        'Start running VoxCPM2 by ONNXRuntime.\n'
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


def create_ort_with_data(data, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
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


def mask_multichar_chinese_tokens(base_tokenizer):
    """Mask multi-char CJK tokens so the tokenizer splits them into single chars."""
    multichar_tokens = {
        token for token in base_tokenizer.get_vocab().keys()
        if len(token.replace("\u2581", "")) >= 2
        and all("\u4e00" <= c <= "\u9fff" for c in token.replace("\u2581", ""))
    }

    class CharTokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.multichar_tokens = multichar_tokens

        def __call__(self, text, **kwargs):
            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []
            for token in tokens:
                clean = token.replace("\u2581", "")
                if clean in self.multichar_tokens:
                    processed.extend(list(clean))
                else:
                    processed.append(token)
            return self.tokenizer.convert_tokens_to_ids(processed)

    return CharTokenizerWrapper(base_tokenizer)


def encode_audio_feat_ort(wav_path):
    """Load audio, pad like upstream, VAE encode → returns ORT value [1, seq, patch_size, latent_dim]."""
    audio = np.array(
        AudioSegment.from_file(wav_path).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(),
        dtype=np.int16
    )
    if USE_AUDIO_NORMALIZER:
        audio = audio_normalizer(audio)
    audio_ort = onnxruntime.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), device_type, DEVICE_ID)
    input_feed_VAE_Encoder[in_name_VAE_Encoder] = audio_ort
    return ort_session_VAE_Encoder.run_with_ort_values(out_name_VAE_Encoder, input_feed_VAE_Encoder, run_options=run_options)[0]


def validate_mode_inputs(mode, reference_path, prompt_path, prompt_text_value):
    """Validate user-configured inputs for the selected mode.

    reference_path is the timbre reference.
    prompt_path + prompt_text_value are the continuation pair.
    """
    if mode == "voice_design":
        return
    if mode == "reference_only":
        if not reference_path:
            raise ValueError("reference_only mode requires reference_audio_path.")
        return
    if mode == "continuation":
        if not prompt_path:
            raise ValueError("continuation mode requires prompt_audio_path.")
        if not prompt_text_value:
            raise ValueError("continuation mode requires prompt_text.")
        return
    if mode == "combined":
        if not reference_path:
            raise ValueError("combined mode requires reference_audio_path.")
        if not prompt_path:
            raise ValueError("combined mode requires prompt_audio_path.")
        if not prompt_text_value:
            raise ValueError("combined mode requires prompt_text.")
        return
    raise ValueError(f"Unsupported MODE: {mode}")


def prepare_mode_audio_features(mode, reference_path, prompt_path):
    """Encode only the audio inputs required by the selected mode.

    Returns:
        tuple(ref_audio_feat_ort, prompt_audio_feat_ort)
    """
    ref_audio_feat = empty_audio_feat_ort
    prompt_audio_feat = empty_audio_feat_ort

    if mode in ("reference_only", "combined"):
        ref_audio_feat = encode_audio_feat_ort(reference_path)
    if mode in ("continuation", "combined"):
        prompt_audio_feat = encode_audio_feat_ort(prompt_path)

    return ref_audio_feat, prompt_audio_feat

# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero': '1',
    'session.intra_op.allow_spinning': '1',
    'session.inter_op.allow_spinning': '1',
    'session.enable_quant_qdq_cleanup': '1',
    'session.qdq_matmulnbits_accuracy_level': '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level': '2',
    'optimization.enable_gelu_approximation': '1',
    'optimization.minimal_build_optimizations': '',
    'optimization.enable_cast_chain_elimination': '1',
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
        'device_type': 'CPU',
        'precision': 'ACCURACY',
        'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams': 1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer': False,
        'disable_dynamic_shapes': False
    }]
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id': DEVICE_ID,
        'gpu_mem_limit': 24 * (1024 ** 3),
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'sdpa_kernel': '2',
        'use_tf32': '1',
        'fuse_conv_bias': '0',
        'cudnn_conv_use_max_workspace': '1',
        'cudnn_conv1d_pad_to_nc1d': '0',
        'tunable_op_enable': '0',
        'tunable_op_tuning_enable': '0',
        'tunable_op_max_tuning_duration_ms': 10,
        'do_copy_in_default_stream': '0',
        'enable_cuda_graph': '0',
        'prefer_nhwc': '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream': '0'
    }]
    device_type = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id': DEVICE_ID,
        'performance_preference': 'high_performance',
        'device_filter': 'gpu',
        'disable_metacommands': 'false',
        'enable_graph_capture': 'false',
        'enable_graph_serialization': 'false'
    }]
    device_type = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts": session_opts,
    "_providers": ORT_Accelerate_Providers if ORT_Accelerate_Providers else None,
    "_provider_options": provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
print("Loading ONNX sessions...")

ort_session_VAE_Encoder = create_session(onnx_model_VAE_Encoder, **packed_settings)
in_name_VAE_Encoder = get_in_names(ort_session_VAE_Encoder)[0]
out_name_VAE_Encoder = [get_out_names(ort_session_VAE_Encoder)[0]]

ort_session_Feat_Encoder_Cond = create_session(onnx_model_Feat_Encoder_Cond, **packed_settings)
in_name_Feat_Encoder_Cond = get_in_names(ort_session_Feat_Encoder_Cond)[0]
out_name_Feat_Encoder_Cond = get_out_names(ort_session_Feat_Encoder_Cond)

# Pre-load all mode-specific Assemble sessions for the post-export demo
ort_sessions_Assemble = {}
for mode_key in onnx_model_Assemble:
    session = create_session(onnx_model_Assemble[mode_key], **packed_settings)
    ort_sessions_Assemble[mode_key] = {
        "session": session,
        "in_names": get_in_names(session),
        "out_names": get_out_names(session),
    }

ort_session_Prefill = create_session(onnx_model_Prefill, **packed_settings)
in_name_Prefill = get_in_names(ort_session_Prefill)
out_name_Prefill = get_out_names(ort_session_Prefill)

ort_session_Rotary_Mask_Decode = create_session(onnx_model_Rotary_Mask_Decode, **packed_settings)
in_name_Rotary_Mask_Decode = get_in_names(ort_session_Rotary_Mask_Decode)
out_name_Rotary_Mask_Decode = get_out_names(ort_session_Rotary_Mask_Decode)

ort_session_Main = create_session(onnx_model_Main, **packed_settings)
print(f"\nUsable Providers: {ort_session_Main.get_providers()}\n")

ort_session_Feat_Decoder = create_session(onnx_model_Feat_Decoder, **packed_settings)

ort_session_VAE_Decoder = create_session(onnx_model_VAE_Decoder, **packed_settings)

# --- Concat (streaming only) ---
if STREAMING:
    ort_session_Concat = create_session(onnx_model_Concat, **packed_settings)
    in_name_Concat = get_in_names(ort_session_Concat)
    out_name_Concat = get_out_names(ort_session_Concat)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main = get_in_names(ort_session_Main)
out_name_Main = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)

num_keys_values = amount_of_outputs_Main - 3  # last 3: random, dit_hidden, stop_flag
num_layers = num_keys_values // 2

num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
num_keys_values_plus_7 = num_keys_values + 7

kv_dtype_Main = np.float16 if 'float16' in ort_session_Main._inputs_meta[0].type else np.float32
hidden_dtype_Main = np.float16 if 'float16' in ort_session_Main._inputs_meta[num_keys_values_plus_4].type else np.float32


model_dtype_Feat_Decoder = np.float16 if 'float16' in ort_session_Feat_Decoder._inputs_meta[2].type else np.float32
model_dtype_VAE_Decoder = np.float16 if 'float16' in ort_session_VAE_Decoder._inputs_meta[0].type else np.float32
model_dtype_VAE_Decoder_out = np.int16

in_name_Feat_Decoder = get_in_names(ort_session_Feat_Decoder)
out_name_Feat_Decoder = get_out_names(ort_session_Feat_Decoder)

in_name_VAE_Decoder = get_in_names(ort_session_VAE_Decoder)
out_name_VAE_Decoder = get_out_names(ort_session_VAE_Decoder)
half_decode_len = 7680  # Fixed for VoxCPM2
_meta = ort_session_Main._inputs_meta


# ══════════════════════════════════════════════════════════════════════════════
# STATIC ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
init_concat_text_len = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_audio_seg1_start = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_audio_seg1_end = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_history_len = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
init_decode_attention_mask = create_ort_with_shape((1, 1, 1, 1), hidden_dtype_Main, device_type, DEVICE_ID)

# KV cache shapes
shape_keys = (_meta[0].shape[0], 1, _meta[0].shape[2], 0)
shape_vals = (_meta[num_layers].shape[0], 1, 0, _meta[num_layers].shape[3])
shape_embed = (1, 0, _meta[num_keys_values].shape[2])

init_past_keys_Main = create_ort_with_shape(shape_keys, kv_dtype_Main, device_type, DEVICE_ID)
init_past_values_Main = create_ort_with_shape(shape_vals, kv_dtype_Main, device_type, DEVICE_ID)
init_feat_embed = create_ort_with_shape(shape_embed, hidden_dtype_Main, device_type, DEVICE_ID)

# CFG Values
cfg_value_ort = create_ort_with_data([CFG_VALUE], hidden_dtype_Main, device_type, DEVICE_ID)
cfg_value_minus_ort = create_ort_with_data([1.0 - CFG_VALUE], hidden_dtype_Main, device_type, DEVICE_ID)

# sr_cond for VAE decode
sr_cond_ort = create_ort_with_data([OUT_SAMPLE_RATE], np.int32, device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT INPUT FEED DICTIONARIES
# ══════════════════════════════════════════════════════════════════════════════
input_feed_VAE_Encoder = {}
input_feed_Feat_Encoder_Cond = {}
input_feed_Assemble = {}
input_feed_Prefill = {}
input_feed_Rotary_Mask_Decode = {}
input_feed_Main = {}
input_feed_Feat_Decoder = {}
input_feed_VAE_Decoder = {}

# Fixed feeds
input_feed_Feat_Decoder[in_name_Feat_Decoder[3]] = cfg_value_ort
input_feed_Feat_Decoder[in_name_Feat_Decoder[4]] = cfg_value_minus_ort
input_feed_VAE_Decoder[in_name_VAE_Decoder[1]] = sr_cond_ort


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & SPECIAL TOKENS
# ══════════════════════════════════════════════════════════════════════════════
AUDIO_START_TOKEN = 101

tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(path_voxcpm2))

# Read patch_size / latent_dim from the VAE Encoder ONNX model output shape metadata
_vae_enc_out_shape = ort_session_VAE_Encoder._outputs_meta[0].shape
_patch_size = _vae_enc_out_shape[2]
_latent_dim = _vae_enc_out_shape[3]

# Encode prompt/reference audio once (cached as ORT values for all sentences)
# Empty ORT tensors for unused inputs (shape [1, 0, ps, ld])
empty_audio_feat_ort = create_ort_with_shape((1, 0, _patch_size, _latent_dim), hidden_dtype_Main, device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# POST-EXPORT DEMO: Run each mode once to verify all exported models work
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("Running post-export demos for all modes...")
print("═" * 70)

for demo_config in DEMO_CONFIGS:
    demo_mode = demo_config["mode"]
    demo_reference_audio_path = demo_config["reference_audio_path"]
    demo_prompt_audio_path = demo_config["prompt_audio_path"]
    demo_prompt_text = demo_config["prompt_text"]
    demo_targets = demo_config["target_texts"]

    print(f"\n{'─' * 50}")
    print(f"Demo Mode: {demo_mode}")
    print(f"Reference audio: {demo_reference_audio_path or 'None'}")
    print(f"Prompt audio: {demo_prompt_audio_path or 'None'}")
    print(f"Prompt text: {demo_prompt_text or 'None'}")
    print(f"{'─' * 50}")

    # Use pre-loaded mode-specific Assemble session
    demo_ort_session_Assemble = ort_sessions_Assemble[demo_mode]["session"]
    demo_in_name_Assemble = ort_sessions_Assemble[demo_mode]["in_names"]
    demo_out_name_Assemble = ort_sessions_Assemble[demo_mode]["out_names"]

    # Encode audio if needed
    validate_mode_inputs(demo_mode, demo_reference_audio_path, demo_prompt_audio_path, demo_prompt_text)
    demo_ref_audio_feat_ort, demo_prompt_audio_feat_ort = prepare_mode_audio_features(
        demo_mode,
        demo_reference_audio_path,
        demo_prompt_audio_path,
    )

    # Set fixed Assemble inputs
    demo_input_feed_Assemble = {}
    if demo_mode == "continuation":
        demo_input_feed_Assemble[demo_in_name_Assemble[1]] = demo_prompt_audio_feat_ort
    elif demo_mode == "reference_only":
        demo_input_feed_Assemble[demo_in_name_Assemble[1]] = demo_ref_audio_feat_ort
    elif demo_mode == "combined":
        demo_input_feed_Assemble[demo_in_name_Assemble[1]] = demo_ref_audio_feat_ort
        demo_input_feed_Assemble[demo_in_name_Assemble[2]] = demo_prompt_audio_feat_ort

    demo_audio_out = []
    demo_start = time.time()

    for sentence in demo_targets:
        print(f"\n  Convert to Speech: {sentence}")

        if USE_TEXT_NORMALIZER:
            from modeling_modified.text_normalize import TextNormalizer
            sentence = TextNormalizer().normalize(sentence)

        target_text = re.sub(r"\s+", " ", sentence.replace("\n", " ")).strip()
        target_ids = tokenizer(target_text)
        if demo_mode in ("continuation", "combined"):
            full_text = re.sub(r"\s+", " ", ((demo_prompt_text or "") + target_text).replace("\n", " ")).strip()
            text_ids = tokenizer(full_text) + [AUDIO_START_TOKEN]
        else:
            text_ids = target_ids + [AUDIO_START_TOKEN]
        text_ids_np = np.array([text_ids], dtype=np.int32)
        text_ids_ort = onnxruntime.OrtValue.ortvalue_from_numpy(text_ids_np, device_type, DEVICE_ID)

        # Assemble
        demo_input_feed_Assemble[demo_in_name_Assemble[0]] = text_ids_ort
        asm_out = demo_ort_session_Assemble.run_with_ort_values(demo_out_name_Assemble, demo_input_feed_Assemble, run_options=run_options)
        text_token_ort = asm_out[0]
        audio_feat_ort = asm_out[1]
        audio_seg1_start_ort = asm_out[2]
        audio_seg1_end_ort = asm_out[3]
        concat_text_len_ort = asm_out[4]
        ids_len_ort = asm_out[5]

        # Feat Encoder + Cond
        input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = audio_feat_ort
        feat_embed_ort, feat_cond_init = ort_session_Feat_Encoder_Cond.run_with_ort_values(out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)

        # Prefill
        input_feed_Prefill[in_name_Prefill[0]] = text_token_ort
        input_feed_Prefill[in_name_Prefill[1]] = ids_len_ort
        input_feed_Prefill[in_name_Prefill[2]] = feat_embed_ort
        input_feed_Prefill[in_name_Prefill[3]] = audio_seg1_start_ort
        input_feed_Prefill[in_name_Prefill[4]] = audio_seg1_end_ort
        input_feed_Prefill[in_name_Prefill[5]] = concat_text_len_ort
        input_feed_Prefill[in_name_Prefill[6]] = init_history_len

        prefill_out = ort_session_Prefill.run_with_ort_values(out_name_Prefill, input_feed_Prefill, run_options=run_options)
        combined_embed = prefill_out[0]
        feat_embed_audio = prefill_out[1]
        rotary_cos = prefill_out[2]
        rotary_sin = prefill_out[3]
        attention_mask = prefill_out[4]
        kv_seq_len = prefill_out[5]

        # Prepare Main inputs
        input_feed_Main[in_name_Main[num_keys_values]] = feat_embed_audio
        input_feed_Main[in_name_Main[num_keys_values_plus_1]] = audio_seg1_start_ort
        input_feed_Main[in_name_Main[num_keys_values_plus_2]] = audio_seg1_end_ort
        input_feed_Main[in_name_Main[num_keys_values_plus_3]] = concat_text_len_ort
        input_feed_Main[in_name_Main[num_keys_values_plus_4]] = combined_embed
        input_feed_Main[in_name_Main[num_keys_values_plus_5]] = rotary_cos
        input_feed_Main[in_name_Main[num_keys_values_plus_6]] = rotary_sin
        input_feed_Main[in_name_Main[num_keys_values_plus_7]] = attention_mask

        # Reset KV Cache
        for idx in range(num_layers):
            input_feed_Main[in_name_Main[idx]] = init_past_keys_Main
        for idx in range(num_layers, num_keys_values):
            input_feed_Main[in_name_Main[idx]] = init_past_values_Main

        feat_cond = feat_cond_init
        total_seq_len = int(ids_len_ort.numpy().item())
        max_len = min(int(len(target_ids) * DECODE_LIMIT_FACTOR + 10), MAX_SEQ_LEN - total_seq_len)

        save_latent_list = []
        num_decode = 0
        start_decode = time.time()

        if STREAMING:
            pre_latent_pred = None
            input_feed_Concat = {}

        while num_decode < max_len:
            all_outputs_Main = ort_session_Main.run_with_ort_values(out_name_Main, input_feed_Main, run_options=run_options)

            input_feed_Feat_Decoder[in_name_Feat_Decoder[0]] = all_outputs_Main[num_keys_values]
            input_feed_Feat_Decoder[in_name_Feat_Decoder[1]] = all_outputs_Main[num_keys_values_plus_1]
            input_feed_Feat_Decoder[in_name_Feat_Decoder[2]] = feat_cond
            latent_pred = ort_session_Feat_Decoder.run_with_ort_values(out_name_Feat_Decoder, input_feed_Feat_Decoder, run_options=run_options)[0]

            if STREAMING:
                if pre_latent_pred is None:
                    pre_latent_pred = latent_pred
                else:
                    input_feed_Concat[in_name_Concat[0]] = pre_latent_pred
                    input_feed_Concat[in_name_Concat[1]] = latent_pred
                    save_latent_ort = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)[0]
                    input_feed_VAE_Decoder[in_name_VAE_Decoder[0]] = save_latent_ort
                    audio_out_ort = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)[0]
                    pre_latent_pred = latent_pred
                    audio_out_np = audio_out_ort.numpy()
                    if num_decode > 1:
                        audio_out_np = audio_out_np[..., half_decode_len:]
                    demo_audio_out.append(audio_out_np)
            else:
                save_latent_list.append(latent_pred.numpy())

            if num_decode >= MIN_SEQ_LEN:
                if int(all_outputs_Main[num_keys_values_plus_2].numpy().item()) in STOP_TOKEN:
                    break

            input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = latent_pred
            feat_embed_decode, feat_cond = ort_session_Feat_Encoder_Cond.run_with_ort_values(out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)

            for idx in range(num_keys_values):
                input_feed_Main[in_name_Main[idx]] = all_outputs_Main[idx]
            input_feed_Main[in_name_Main[num_keys_values]] = feat_embed_decode
            input_feed_Main[in_name_Main[num_keys_values + 4]] = feat_embed_decode

            input_feed_Rotary_Mask_Decode[in_name_Rotary_Mask_Decode[0]] = kv_seq_len
            rotary_cos, rotary_sin, kv_seq_len = ort_session_Rotary_Mask_Decode.run_with_ort_values(out_name_Rotary_Mask_Decode, input_feed_Rotary_Mask_Decode, run_options=run_options)
            input_feed_Main[in_name_Main[num_keys_values_plus_5]] = rotary_cos
            input_feed_Main[in_name_Main[num_keys_values_plus_6]] = rotary_sin

            if num_decode < 1:
                input_feed_Main[in_name_Main[num_keys_values_plus_1]] = init_audio_seg1_start
                input_feed_Main[in_name_Main[num_keys_values_plus_2]] = init_audio_seg1_end
                input_feed_Main[in_name_Main[num_keys_values_plus_3]] = init_concat_text_len
                input_feed_Main[in_name_Main[num_keys_values_plus_7]] = init_decode_attention_mask

            num_decode += 1

        print(f"    Decoded {num_decode} tokens ({((num_decode + 1) / max(time.time() - start_decode, 1e-6)):.1f} tok/s)")

        if not STREAMING:
            if save_latent_list:
                stacked = np.concatenate(save_latent_list, axis=1)
                input_feed_VAE_Decoder[in_name_VAE_Decoder[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(stacked.astype(model_dtype_VAE_Decoder, copy=False), device_type, DEVICE_ID)
                audio_out = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)[0]
                demo_audio_out.append(audio_out.numpy())

    # Save demo output
    if demo_audio_out:
        demo_audio_all = np.concatenate([output.reshape(-1) for output in demo_audio_out], axis=-1)
        demo_out_path = f"./generated_demo_{demo_mode}.wav"
        Path(demo_out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(demo_out_path, demo_audio_all.reshape(-1), OUT_SAMPLE_RATE, subtype='PCM_16')
        demo_duration = demo_audio_all.shape[-1] / OUT_SAMPLE_RATE
        demo_cost = time.time() - demo_start
        print(f"  Saved: {demo_out_path} ({demo_duration:.2f}s audio, RTF={demo_cost / max(demo_duration, 1e-6):.3f})")

print("\n" + "═" * 70)
print("All mode demos complete.")
print("═" * 70)
