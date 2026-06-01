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


path_voxcpm                         = r'/home/DakeQQ/Downloads/VoxCPM1.5'                                     # Set the folder path where the VoxCPM1.5 project downloaded.
onnx_model_VAE_Encoder              = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_VAE_Encoder.onnx'           # Assign a path where the exported VoxCPM model stored.
onnx_model_Feat_Encoder_Cond        = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Feat_Encoder_Cond.onnx'
onnx_model_Prefill                  = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Prefill.onnx'
onnx_model_Rotary_Mask_Text_Decode  = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Rotary_Mask_Text_Decode.onnx'
onnx_model_Main                     = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Main.onnx'
onnx_model_Feat_Decoder             = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Feat_Decoder.onnx'
onnx_model_VAE_Decoder              = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_VAE_Decoder.onnx'
onnx_model_Concat                   = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Concat.onnx'                # Only used for streaming mode

prompt_audio_path = "./example/basic_ref_zh.wav"                                # optional: path to a prompt speech for voice cloning else None.
prompt_text = "对，这就是我，万人敬仰的太乙真人。"                                    # The reference text for the prompt speech.
target_tts = [                                                                  # The test query after the export process.
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
]
generated_audio_path = r"./generated.wav"                                       # The generated audio path.

# Model Config
DO_EXPORT = False                         # Whether to export the ONNX models

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
CFG_VALUE = 2.0                          # Lower values result in more natural speech for long text, while higher values stay closer to the original sound features. Free to edit it.
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
# VAE Encoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_VAE_ENCODER(torch.nn.Module):
    def __init__(self, voxcpm, in_sample_rate):
        super(VOXCPM_VAE_ENCODER, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self._remove_weight_norm(self.voxcpm.audio_vae.encoder)
        self.patch_len = self.voxcpm.patch_size * self.voxcpm.chunk_size
        self.pad_zeros = torch.zeros([1, 1, self.patch_len], dtype=torch.int8)
        self.in_sample_rate = in_sample_rate
        self.sr_scale = float(44100.0 / self.in_sample_rate)

        # Fuse inv_int16 (1/32768) scaling into the first conv layer's weights to eliminate runtime multiply
        with torch.no_grad():
            first_conv = self.voxcpm.audio_vae.encoder.block[0]
            first_conv.weight.mul_(1.0 / 32768.0)

    @staticmethod
    def _remove_weight_norm(module):
        for child in module.modules():
            try:
                torch.nn.utils.remove_weight_norm(child)
            except ValueError:
                pass

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def forward(self, prompt_audio):
        prompt_audio = prompt_audio.float()
        if self.sr_scale != 1.0:
            prompt_audio = torch.nn.functional.interpolate(prompt_audio, scale_factor=self.sr_scale, mode='linear', align_corners=False)
        padding_size = self.patch_len - prompt_audio.shape[-1] % self.patch_len
        prompt_audio = torch.cat([prompt_audio, self.pad_zeros[..., :padding_size].float()], dim=-1)
        audio_feat = self.voxcpm.audio_vae.encoder(prompt_audio)
        audio_feat = audio_feat.view(self.voxcpm.audio_vae.latent_dim, -1, self.voxcpm.patch_size).permute(1, 2, 0)
        return audio_feat


# ══════════════════════════════════════════════════════════════════════════════
# Fused Feature Encoder + Conditioning Module
# Replaces separate Feat_Encoder and Feat_Cond modules.
# Returns both feat_embed (for LM) and feat_cond (for diffusion) in one call.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_FEAT_ENCODER_COND(torch.nn.Module):
    def __init__(self, voxcpm, max_prompt_audio_len, in_sample_rate):
        super(VOXCPM_FEAT_ENCODER_COND, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)

        # === Feat Encoder geometry ===
        self.head_dim = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_heads
        self.num_key_value_heads = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_key_value_heads
        self.num_key_value_groups = self.voxcpm.feat_encoder.encoder.layers._modules['0'].self_attn.num_key_value_groups
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        self.rms_eps = torch.tensor([self.voxcpm.feat_encoder.encoder.config.rms_norm_eps * self.voxcpm.feat_encoder.encoder.config.hidden_size], dtype=torch.float32)

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
                del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                del layer.input_layernorm

                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj
                in_feat = gate.in_features
                out_feat = gate.out_features + up.out_features
                gate_up = torch.nn.Linear(in_feat, out_feat, bias=False)
                gate_up.weight.copy_(torch.cat([gate.weight * w, up.weight * w], dim=0))
                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

            w = self.voxcpm.feat_encoder.encoder.norm.weight.unsqueeze(0) * norm_factor
            self.voxcpm.enc_to_lm_proj.weight.mul_(w)
            del self.voxcpm.feat_encoder.encoder.norm

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + self.rms_eps)

    def rotate_half(self, x):
        x = x.view(-1, self.q_len, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(-1, self.q_len, 1, self.qk_heads, self.head_dim)

    def forward(self, audio_feat):
        # === Feature Encoder: produces feat_embed for the LM ===
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

        # === Feature Conditioning: produces feat_cond for diffusion ===
        feat_cond = self.voxcpm.feat_decoder.estimator.cond_proj(audio_feat[[-1]])
        feat_cond = torch.cat([feat_cond, feat_cond], dim=0)

        return feat_embed, feat_cond


# ══════════════════════════════════════════════════════════════════════════════
# Fused Prefill Module
# Replaces: Text_Embed + multiple Concat calls + Rotary_Mask_Prefill
# Produces the full prefill hidden_states, rotary embeddings, and causal mask
# in a single model call.
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_PREFILL(torch.nn.Module):
    def __init__(self, voxcpm, max_seq_len):
        super(VOXCPM_PREFILL, self).__init__()
        self.embed_tokens = voxcpm.base_lm.embed_tokens

        # Precompute audio_start_embed as a constant
        with torch.no_grad():
            self.audio_start_embed = self.embed_tokens(torch.tensor([[101]], dtype=torch.int32))  # [1, 1, hidden]

        # Causal attention mask
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

        # Precompute rotary embeddings
        position_ids = torch.arange(max_seq_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = voxcpm.base_lm.rope_emb(position_ids)
        rope_emb_sin[:, :voxcpm.base_lm.rope_emb.dim // 2] *= -1.0
        cos = rope_emb_cos.unsqueeze(1).unsqueeze(1)
        sin = rope_emb_sin.unsqueeze(1).unsqueeze(1)
        self.register_buffer("cos_rotary_pos_emb", cos.half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half(), persistent=False)

    def forward(self, prompt_text_ids, target_text_ids, feat_embed):
        # Embed text tokens
        prompt_embed = self.embed_tokens(prompt_text_ids)   # [1, prompt_len, hidden]
        target_embed = self.embed_tokens(target_text_ids)   # [1, target_len, hidden]

        # Build full sequence: [prompt_text | target_text | audio_start | feat_embed]
        text_embed = torch.cat([prompt_embed, target_embed, self.audio_start_embed], dim=1)
        concat_text_len = text_embed.shape[1].unsqueeze(0)

        hidden_states = torch.cat([text_embed, feat_embed], dim=1)
        ids_len = hidden_states.shape[1].unsqueeze(0)

        # Compute rotary embeddings and causal mask
        rotary_cos = self.cos_rotary_pos_emb[:ids_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:ids_len].float()
        attention_mask = (self.attention_mask[..., :ids_len, :ids_len]).float()

        return hidden_states, concat_text_len, rotary_cos, rotary_sin, attention_mask, ids_len


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding (Decode Only)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_ROTARY_MASK_DECODE(torch.nn.Module):
    def __init__(self, voxcpm, max_seq_len):
        super().__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = voxcpm.base_lm.rope_emb(position_ids)
        rope_emb_sin[:, :voxcpm.base_lm.rope_emb.dim // 2] *= -1.0
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
# Main Transformer Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_MAIN(torch.nn.Module):
    def __init__(self, voxcpm, max_seq_len):
        super(VOXCPM_MAIN, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)

        self.head_dim = self.voxcpm.base_lm.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_heads
        self.num_key_value_heads = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_key_value_heads
        self.num_key_value_groups = self.voxcpm.base_lm.layers._modules['0'].self_attn.num_key_value_groups
        self.qk_heads = self.num_heads + self.num_key_value_heads

        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        self.rms_eps = torch.tensor([self.voxcpm.base_lm.config.rms_norm_eps * self.voxcpm.base_lm.config.hidden_size], dtype=torch.float32)

        self.total_layers = self.voxcpm.base_lm.config.num_hidden_layers + self.voxcpm.residual_lm.config.num_hidden_layers
        self.save_key = [None] * self.total_layers
        self.save_value = [None] * self.total_layers

        self.norm_factor = self.voxcpm.base_lm.config.hidden_size ** 0.5
        self.scale_factor_base = float(self.voxcpm.base_lm.layers._modules['0'].self_attn.head_dim ** -0.25)
        self._fuse_weights()
        self._fuse_dit_stop_proj()

    def _fuse_dit_stop_proj(self):
        """Fuse lm_to_dit_proj and stop_proj into a single linear to reduce two matmuls to one."""
        with torch.no_grad():
            dit_proj = self.voxcpm.lm_to_dit_proj
            stop_proj = self.voxcpm.stop_proj
            in_features = dit_proj.in_features
            dit_out = dit_proj.out_features
            stop_out = stop_proj.out_features
            self.dit_out_features = dit_out
            self.stop_out_features = stop_out
            has_bias = (dit_proj.bias is not None) or (stop_proj.bias is not None)
            fused = torch.nn.Linear(in_features, dit_out + stop_out, bias=has_bias)
            fused.weight.copy_(torch.cat([dit_proj.weight, stop_proj.weight], dim=0))
            if has_bias:
                z = lambda feat: torch.zeros(feat, dtype=dit_proj.weight.dtype, device=dit_proj.weight.device)
                db = dit_proj.bias if dit_proj.bias is not None else z(dit_out)
                sb = stop_proj.bias if stop_proj.bias is not None else z(stop_out)
                fused.bias.copy_(torch.cat([db, sb], dim=0))
            self.fused_dit_stop_proj = fused
            del self.voxcpm.lm_to_dit_proj, self.voxcpm.stop_proj

    def _fuse_weights(self):
        with torch.no_grad():
            for layer in self.voxcpm.base_lm.layers:
                self._fuse_qkv_projection(layer)
                self._fuse_gate_up_projection(layer)
            for layer in self.voxcpm.residual_lm.layers:
                self._fuse_qkv_projection(layer)
                self._fuse_gate_up_projection(layer)
            final_norm_weight = self.voxcpm.residual_lm.norm.weight.unsqueeze(0) * self.norm_factor
            self.voxcpm.res_to_dit_proj.weight.mul_(final_norm_weight)
            del self.voxcpm.residual_lm.norm

    def _fuse_qkv_projection(self, layer):
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight * self.scale_factor_base, k_proj.weight * self.scale_factor_base, v_proj.weight], dim=0))
        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj) * self.scale_factor_base, _get_bias(k_proj) * self.scale_factor_base, _get_bias(v_proj)], dim=0))
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
                VOXCPM_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + self.rms_eps)

    def _rotate_half(self, x):
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

        hidden_states = self.voxcpm.base_lm.norm(hidden_states)
        fsq_layer_out = self.voxcpm.fsq_layer(hidden_states[:, concat_text_len:])
        hidden_states = hidden_states[:, :concat_text_len]
        lm_hidden = torch.cat([hidden_states, fsq_layer_out], dim=1)[:, [-1]]
        hidden_states = torch.cat([hidden_states, fsq_layer_out + feat_embed], dim=1)

        i = self.voxcpm.base_lm.config.num_hidden_layers
        for layer in self.voxcpm.residual_lm.layers:
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
            i += 1

        residual_hidden = hidden_states[:, [-1]]
        residual_hidden = self._rms_norm(residual_hidden)
        fused_out = self.fused_dit_stop_proj(lm_hidden)
        dit_hidden_1, stop_intermediate = torch.split(fused_out, [self.dit_out_features, self.stop_out_features], dim=-1)
        dit_hidden_2 = self.voxcpm.res_to_dit_proj(residual_hidden)
        dit_hidden = dit_hidden_1 + dit_hidden_2
        random = torch.randn((1, self.voxcpm.patch_size, self.voxcpm.feat_decoder.in_channels), dtype=torch.float32)
        stop_flag = self.voxcpm.stop_head(self.voxcpm.stop_actn(stop_intermediate)).argmax(dim=-1, keepdims=False).int()
        return *self.save_key, *self.save_value, random, dit_hidden, stop_flag


# ══════════════════════════════════════════════════════════════════════════════
# Fused Feature Decoder Module (Full Diffusion Loop)
# All timesteps are unrolled into a single forward pass.
# Reduces timesteps session.run() calls to 1.
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
        self.rms_eps = torch.tensor([self.voxcpm.feat_decoder.estimator.config.rms_norm_eps * self.voxcpm.feat_decoder.estimator.config.hidden_size], dtype=torch.float32)

        # Precompute all timestep data
        self.timesteps = fixed_timesteps
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
        self.t_all = (t + dt_in).unsqueeze(0)  # [1, timesteps-1, hidden]

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
                del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                del layer.input_layernorm

                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj
                in_feat = gate.in_features
                out_feat = gate.out_features + up.out_features
                gate_up = torch.nn.Linear(in_feat, out_feat, bias=False)
                gate_up.weight.copy_(torch.cat([gate.weight * w, up.weight * w], dim=0))
                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

            w = self.voxcpm.feat_decoder.estimator.decoder.norm.weight.unsqueeze(0) * norm_factor
            self.voxcpm.feat_decoder.estimator.out_proj.weight.mul_(w)
            del self.voxcpm.feat_decoder.estimator.decoder.norm

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + self.rms_eps)

    def rotate_half(self, x):
        x = x.view(-1, self.q_len, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(-1, self.q_len, 1, self.qk_heads, self.head_dim)

    def _single_step(self, step, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus):
        t = self.t_all[:, step]
        dt = self.dt[..., step]
        dit_hidden_t = dit_hidden + t
        dit_hidden_t = torch.cat([dit_hidden_t, t], dim=0)
        x = self.voxcpm.feat_decoder.estimator.in_proj(random)
        x = torch.cat([x, x], dim=0)
        hidden_states = torch.cat([dit_hidden_t, feat_cond, x], dim=1)
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
        return random - dt * dphi_dt

    def forward(self, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus):
        # Full diffusion loop unrolled - all timesteps in one call
        for step in range(self.timesteps - 1):
            random = self._single_step([step], random, dit_hidden, feat_cond, cfg_value, cfg_value_minus)
        return random



# ══════════════════════════════════════════════════════════════════════════════
# VAE Decoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_VAE_DECODE(torch.nn.Module):
    def __init__(self, voxcpm, output_sample_rate):
        super(VOXCPM_VAE_DECODE, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self._remove_weight_norm(self.voxcpm.audio_vae.decoder)
        self.scale = float(output_sample_rate / 44100.0)
        self.single_decode_len = self.voxcpm.patch_size * self.voxcpm.chunk_size

        # Fuse 32767 scale into the last conv layer's weights and replace Tanh with Hardtanh(-32767, 32767)
        with torch.no_grad():
            last_conv = self.voxcpm.audio_vae.decoder.model[-2]  # Conv1d before Tanh
            last_conv.weight.mul_(32767.0)
            if last_conv.bias is not None:
                last_conv.bias.mul_(32767.0)
            self.voxcpm.audio_vae.decoder.model[-1] = torch.nn.Hardtanh(min_val=-32767.0, max_val=32767.0)

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    @staticmethod
    def _remove_weight_norm(module):
        for child in module.modules():
            try:
                torch.nn.utils.remove_weight_norm(child)
            except ValueError:
                pass

    def forward(self, latent_pred):
        decode_audio = self.voxcpm.audio_vae.decode(latent_pred.transpose(-1, -2))
        if self.scale != 1.0:
            decode_audio = torch.nn.functional.interpolate(decode_audio, scale_factor=self.scale, mode='linear', align_corners=False)
        audio_out_len = decode_audio.shape[-1].unsqueeze(0)
        return decode_audio.to(torch.int16), audio_out_len


# ══════════════════════════════════════════════════════════════════════════════
# Concat Utility (for streaming VAE decode only)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM_CONCAT(torch.nn.Module):
    def forward(self, embed_0, embed_1):
        return torch.cat([embed_0, embed_1], dim=1)


if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # Load Model & Extract Config
        # ══════════════════════════════════════════════════════════════════
        model = VoxCPM.from_pretrained(path_voxcpm, load_denoiser=False, optimize=False).tts_model
        model = model.float().to('cpu').eval()

        base_lm_num_layers     = model.base_lm.config.num_hidden_layers
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

        kv_dtype = torch.float16 if USE_F16_KV else torch.float32

        # ══════════════════════════════════════════════════════════════════
        # Build Dummy Tensors for Tracing
        # ══════════════════════════════════════════════════════════════════
        ids_len     = torch.tensor([25], dtype=torch.int64)
        history_len = torch.tensor([0], dtype=torch.int64)
        kv_seq_len  = ids_len + history_len

        kv_specs = [('key', 3), ('value', 2)]
        base_kv_tensors = {
            'key':   torch.zeros((num_kv_heads, 1, head_dim, history_len), dtype=kv_dtype),
            'value': torch.zeros((num_kv_heads, 1, history_len, head_dim), dtype=kv_dtype),
        }
        residual_kv_tensors = {
            'key':   torch.zeros((residual_num_kv_heads, 1, residual_head_dim, history_len), dtype=kv_dtype),
            'value': torch.zeros((residual_num_kv_heads, 1, history_len, residual_head_dim), dtype=kv_dtype),
        }

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
        # Export: Fused Feat_Encoder_Cond
        # ══════════════════════════════════════════════════════════════════
        audio_feat = torch.zeros([20, patch_size, feat_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM_FEAT_ENCODER_COND(model, MAX_PROMPT_AUDIO_LEN, IN_SAMPLE_RATE),
            (audio_feat,),
            onnx_model_Feat_Encoder_Cond,
            input_names=['audio_feat'],
            output_names=['feat_embed', 'feat_cond'],
            dynamic_axes={
                'audio_feat': {0: 'audio_feat_len'},
                'feat_embed': {1: 'audio_feat_len'},
            },
            opset_version=OPSET,
            dynamo=False
        )
        del audio_feat

        # ══════════════════════════════════════════════════════════════════
        # Export: Fused Prefill
        # ══════════════════════════════════════════════════════════════════
        prompt_text_ids = torch.zeros([1, 5], dtype=torch.int32)
        target_text_ids = torch.zeros([1, 10], dtype=torch.int32)
        feat_embed_dummy = torch.zeros([1, 20, hidden_size], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM_PREFILL(model, MAX_SEQ_LEN),
            (prompt_text_ids, target_text_ids, feat_embed_dummy),
            onnx_model_Prefill,
            input_names=['prompt_text_ids', 'target_text_ids', 'feat_embed'],
            output_names=['hidden_states', 'concat_text_len', 'rotary_cos', 'rotary_sin', 'attention_mask', 'ids_len'],
            dynamic_axes={
                'prompt_text_ids': {1: 'prompt_len'},
                'target_text_ids': {1: 'target_len'},
                'feat_embed':      {1: 'feat_len'},
                'hidden_states':   {1: 'ids_len'},
                'rotary_cos':      {0: 'ids_len'},
                'rotary_sin':      {0: 'ids_len'},
                'attention_mask':  {2: 'ids_len', 3: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del prompt_text_ids, target_text_ids, feat_embed_dummy

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary_Mask_Decode
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
        # Export: Fused Feat_Decoder (Full Diffusion Loop)
        # ══════════════════════════════════════════════════════════════════
        model_Feat_Decoder = VOXCPM_FEAT_DECODER(model, FIXED_TIMESTEPS)
        random          = torch.ones((1, patch_size, feat_in_channels), dtype=torch.float32)
        dit_hidden      = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
        feat_cond       = torch.zeros((2, patch_size, cond_proj_out), dtype=torch.float32)
        cfg_value_t     = torch.tensor([CFG_VALUE], dtype=torch.float32)
        cfg_value_minus_t = torch.tensor([1.0 - CFG_VALUE], dtype=torch.float32)

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
        del model_VAE_Decoder, latent_pred

        # ══════════════════════════════════════════════════════════════════
        # Export: Concat (streaming utility only)
        # ══════════════════════════════════════════════════════════════════
        embed_0 = torch.zeros([1, patch_size, feat_in_channels], dtype=torch.float32)
        embed_1 = torch.zeros([1, patch_size, feat_in_channels], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM_CONCAT(),
            (embed_0, embed_1),
            onnx_model_Concat,
            input_names=['embed_0', 'embed_1'],
            output_names=['concat_out'],
            dynamic_axes={
                'embed_0':    {1: 'len_0'},
                'embed_1':    {1: 'len_1'},
                'concat_out': {1: 'concat_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del embed_0, embed_1

        del model
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
        'device_type':              'CPU',
        'precision':                'ACCURACY',
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 **3),
        'arena_extend_strategy':              'kNextPowerOfTwo',
        'cudnn_conv_algo_search':             'EXHAUSTIVE',
        'sdpa_kernel':                        '2',
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',
        'device_filter':              'gpu',
        'disable_metacommands':       'false',
        'enable_graph_capture':       'false',
        'enable_graph_serialization': 'false'
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
# --- VAE Encoder ---
ort_session_VAE_Encoder = create_session(onnx_model_VAE_Encoder, **packed_settings)
in_name_VAE_Encoder     = get_in_names(ort_session_VAE_Encoder)[0]
out_name_VAE_Encoder    = [get_out_names(ort_session_VAE_Encoder)[0]]

# --- Fused Feat Encoder + Cond ---
ort_session_Feat_Encoder_Cond = create_session(onnx_model_Feat_Encoder_Cond, **packed_settings)
in_name_Feat_Encoder_Cond     = get_in_names(ort_session_Feat_Encoder_Cond)[0]
out_name_Feat_Encoder_Cond    = get_out_names(ort_session_Feat_Encoder_Cond)

# --- Fused Prefill ---
ort_session_Prefill = create_session(onnx_model_Prefill, **packed_settings)
in_name_Prefill     = get_in_names(ort_session_Prefill)
out_name_Prefill    = get_out_names(ort_session_Prefill)

# --- Rotary Mask Decode ---
ort_session_Rotary_Mask_Text_Decode = create_session(onnx_model_Rotary_Mask_Text_Decode, **packed_settings)
in_name_Rotary_Mask_Text_Decode     = get_in_names(ort_session_Rotary_Mask_Text_Decode)
out_name_Rotary_Mask_Text_Decode    = get_out_names(ort_session_Rotary_Mask_Text_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
print(f"\nUsable Providers: {ort_session_Main.get_providers()}\n")

# --- Feat Decoder (full loop) ---
ort_session_Feat_Decoder = create_session(onnx_model_Feat_Decoder, **packed_settings)
model_dtype_Feat_Decoder = np.float16 if 'float16' in ort_session_Feat_Decoder._inputs_meta[1].type else np.float32
in_name_Feat_Decoder     = get_in_names(ort_session_Feat_Decoder)
out_name_Feat_Decoder    = get_out_names(ort_session_Feat_Decoder)

# --- VAE Decoder ---
ort_session_VAE_Decoder    = create_session(onnx_model_VAE_Decoder, **packed_settings)
model_dtype_VAE_Decoder    = np.float16 if 'float16' in ort_session_VAE_Decoder._inputs_meta[0].type else np.float32
DYNAMIC_SHAPE_VAE_DECODE   = isinstance(ort_session_VAE_Decoder._inputs_meta[0].shape[1], str)
in_name_VAE_Decoder        = get_in_names(ort_session_VAE_Decoder)[0]
out_name_VAE_Decoder       = get_out_names(ort_session_VAE_Decoder)
half_decode_len            = 7056  # Fixed for VoxCPM1.5

# --- Concat (streaming only) ---
if STREAMING:
    ort_session_Concat = create_session(onnx_model_Concat, **packed_settings)
    in_name_Concat     = get_in_names(ort_session_Concat)
    out_name_Concat    = get_out_names(ort_session_Concat)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
model_dtype_Main       = np.float16 if 'float16' in ort_session_Main._inputs_meta[0].type else np.float32
in_name_Main           = get_in_names(ort_session_Main)
out_name_Main          = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)

num_keys_values = amount_of_outputs_Main - 3
num_layers      = num_keys_values // 2

num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5

_meta = ort_session_Main._inputs_meta


# ══════════════════════════════════════════════════════════════════════════════
# STATIC ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
generate_limit = MAX_SEQ_LEN - 1

# --- Scalars & Lengths ---
init_concat_text_len   = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)

# --- Masks ---
init_decode_attention_mask = create_ort_with_shape((1, 1, 1, 1), model_dtype_Main, device_type, DEVICE_ID)

# --- KV Cache & Embedding Shapes ---
shape_keys   = (_meta[0].shape[0],          1, _meta[0].shape[2],          0)
shape_vals   = (_meta[num_layers].shape[0],  1, 0, _meta[num_layers].shape[3])
shape_embed  = (1, 0, _meta[num_keys_values].shape[2])

init_past_keys_Main   = create_ort_with_shape(shape_keys, model_dtype_Main, device_type, DEVICE_ID)
init_past_values_Main = create_ort_with_shape(shape_vals, model_dtype_Main, device_type, DEVICE_ID)
init_feat_embed       = create_ort_with_shape(shape_embed, model_dtype_Main, device_type, DEVICE_ID)

# --- Audio Post-processing ---
blank_segment = np.zeros((1, 1, int(OUT_SAMPLE_RATE * 0.1)), dtype=np.int16)

# --- Empty prompt IDs (for no-prompt case) ---
empty_prompt_ids = create_ort_with_data([[]], np.int32, device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION SETUP & IO FEEDS
# ══════════════════════════════════════════════════════════════════════════════
input_feed_VAE_Encoder              = {}
input_feed_Feat_Encoder_Cond        = {}
input_feed_Prefill                  = {}
input_feed_Rotary_Mask_Text_Decode  = {}
input_feed_Main                     = {}
input_feed_Feat_Decoder             = {}

# CFG Values
cfg_value_ort = create_ort_with_data([CFG_VALUE], np.float32, device_type, DEVICE_ID)
cfg_value_minus_ort = create_ort_with_data([1.0 - CFG_VALUE], np.float32, device_type, DEVICE_ID)
input_feed_VAE_Decoder              = {}

# Compute init_feat_cond (zero-input conditioning for no-prompt case)
_meta_fec = ort_session_Feat_Encoder_Cond._inputs_meta[0]
_zero_feat_shape = (1, _meta_fec.shape[1], _meta_fec.shape[2])
_zero_feat_dtype = np.float16 if 'float16' in _meta_fec.type else np.float32
input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = create_ort_with_shape(_zero_feat_shape, _zero_feat_dtype, device_type, DEVICE_ID)
_init_results = ort_session_Feat_Encoder_Cond.run_with_ort_values(out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)
init_feat_cond_0 = _init_results[1]


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & PROMPT HANDLING
# ══════════════════════════════════════════════════════════════════════════════
tokenizer       = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(path_voxcpm))
text_normalizer = TextNormalizer()

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCE AUDIO ENCODING (cached - computed once, reused for all sentences)
# ══════════════════════════════════════════════════════════════════════════════
if prompt_audio_path and prompt_text:
    use_prompt_audio = True

    # Load and encode audio
    audio = np.array(
        AudioSegment.from_file(prompt_audio_path).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(),
        dtype=np.int16
    )
    if USE_AUDIO_NORMALIZER:
        audio = audio_normalizer(audio)
    audio_ort = onnxruntime.OrtValue.ortvalue_from_numpy(audio.reshape(1, 1, -1), device_type, DEVICE_ID)

    # VAE Encode (once)
    input_feed_VAE_Encoder[in_name_VAE_Encoder] = audio_ort
    audio_feat = ort_session_VAE_Encoder.run_with_ort_values(out_name_VAE_Encoder, input_feed_VAE_Encoder, run_options=run_options)[0]

    # Feat Encoder + Cond (once) → cached feat_embed_full & feat_cond_init
    input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = audio_feat
    feat_embed_full, feat_cond_init = ort_session_Feat_Encoder_Cond.run_with_ort_values(
        out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)

    # Tokenize prompt text (once)
    if USE_TEXT_NORMALIZER:
        prompt_text = text_normalizer.normalize(prompt_text)
    prompt_ids_np      = np.array([tokenizer(prompt_text)], dtype=np.int32)
    prompt_text_len    = prompt_ids_np.shape[-1]
    prompt_ids_ort     = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_ids_np, device_type, DEVICE_ID)

    del audio, audio_ort
else:
    use_prompt_audio = False
    feat_embed_full  = init_feat_embed
    feat_cond_init   = init_feat_cond_0
    prompt_text_len  = 0
    prompt_ids_ort   = empty_prompt_ids

    if not prompt_audio_path:
        print("Info: No prompt audio provided, using random seed to generate voice.\n")
    else:
        print("Warning: No prompt text provided, so the prompt audio will be ignored.\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
save_audio_out = []
count_time = time.time()

for sentence in target_tts:
    print(f"Convert to Speech: {sentence}")
    if USE_TEXT_NORMALIZER:
        sentence = text_normalizer.normalize(sentence)

    # --- Tokenize target text ---
    target_ids_np = np.array([tokenizer(sentence)], dtype=np.int32)
    target_ids_ort = onnxruntime.OrtValue.ortvalue_from_numpy(target_ids_np, device_type, DEVICE_ID)

    # ──────────────────────────────────────────────────────────────────────────
    # PREFILL: Single fused call replaces Text_Embed + 3x Concat + Rotary_Mask
    # Input: prompt_text_ids, target_text_ids, feat_embed
    # Output: hidden_states, concat_text_len, rotary_cos, rotary_sin, attention_mask, ids_len
    # ──────────────────────────────────────────────────────────────────────────
    input_feed_Prefill[in_name_Prefill[0]] = prompt_ids_ort
    input_feed_Prefill[in_name_Prefill[1]] = target_ids_ort
    input_feed_Prefill[in_name_Prefill[2]] = feat_embed_full

    prefill_out      = ort_session_Prefill.run_with_ort_values(out_name_Prefill, input_feed_Prefill, run_options=run_options)
    hidden_states    = prefill_out[0]
    concat_text_len  = prefill_out[1]
    rotary_cos       = prefill_out[2]
    rotary_sin       = prefill_out[3]
    attention_mask   = prefill_out[4]
    ids_len_ort      = prefill_out[5]

    # Get scalar values for max_len calculation (one-time read, outside hot loop)
    concat_text_len_val = int(concat_text_len.numpy().item())
    ids_len_val         = int(ids_len_ort.numpy().item())
    max_len = min((concat_text_len_val - prompt_text_len) * DECODE_LIMIT_FACTOR + 10, generate_limit - ids_len_val)

    # --- Prepare Main Decoder Inputs (prefill step) ---
    input_feed_Main[in_name_Main[num_keys_values]]        = feat_embed_full
    input_feed_Main[in_name_Main[num_keys_values_plus_1]] = concat_text_len
    input_feed_Main[in_name_Main[num_keys_values_plus_2]] = hidden_states
    input_feed_Main[in_name_Main[num_keys_values_plus_3]] = rotary_cos
    input_feed_Main[in_name_Main[num_keys_values_plus_4]] = rotary_sin
    input_feed_Main[in_name_Main[num_keys_values_plus_5]] = attention_mask

    # Reset KV Cache
    for i in range(num_layers):
        input_feed_Main[in_name_Main[i]] = init_past_keys_Main
    for i in range(num_layers, num_keys_values):
        input_feed_Main[in_name_Main[i]] = init_past_values_Main

    feat_cond = feat_cond_init
    kv_seq_len = ids_len_ort

    # Latent accumulation (no Concat in loop for non-streaming)
    save_latent_list = []

    if STREAMING:
        pre_latent_pred = None
        input_feed_Concat = {}

    # ──────────────────────────────────────────────────────────────────────────
    # AUTO-REGRESSIVE DECODING
    # Hot loop: only 4 session.run() calls per step (was 14)
    #   1. Main (transformer)
    #   2. Feat_Decoder (full diffusion loop in one call)
    #   3. Feat_Encoder_Cond (fused feat encoding + conditioning)
    #   4. Rotary_Mask_Decode (next position)
    # ──────────────────────────────────────────────────────────────────────────
    num_decode   = 0
    start_decode = time.time()

    while num_decode < max_len:
        # --- 1. Main Transformer ---
        all_outputs_Main = ort_session_Main.run_with_ort_values(out_name_Main, input_feed_Main, run_options=run_options)

        # --- 2. Feat Decoder (ALL timesteps in one call) ---
        input_feed_Feat_Decoder[in_name_Feat_Decoder[0]] = all_outputs_Main[num_keys_values]         # random
        input_feed_Feat_Decoder[in_name_Feat_Decoder[1]] = all_outputs_Main[num_keys_values_plus_1]  # dit_hidden
        input_feed_Feat_Decoder[in_name_Feat_Decoder[2]] = feat_cond
        input_feed_Feat_Decoder[in_name_Feat_Decoder[3]] = cfg_value_ort
        input_feed_Feat_Decoder[in_name_Feat_Decoder[4]] = cfg_value_minus_ort

        latent_pred = ort_session_Feat_Decoder.run_with_ort_values(out_name_Feat_Decoder, input_feed_Feat_Decoder, run_options=run_options)[0]

        # --- Accumulate latent (no session.run() needed) ---
        if STREAMING:
            if pre_latent_pred is None:
                pre_latent_pred = latent_pred
            else:
                input_feed_Concat[in_name_Concat[0]] = pre_latent_pred
                input_feed_Concat[in_name_Concat[1]] = latent_pred
                save_latent_ort = ort_session_Concat.run_with_ort_values(out_name_Concat, input_feed_Concat, run_options=run_options)[0]
                input_feed_VAE_Decoder[in_name_VAE_Decoder] = save_latent_ort
                audio_out_ort, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
                pre_latent_pred = latent_pred
                audio_out_np = audio_out_ort.numpy()
                if num_decode > 1:
                    audio_out_np = audio_out_np[..., half_decode_len:]
                save_audio_out.append(audio_out_np)
        else:
            save_latent_list.append(latent_pred)

        # --- Check Stop Token ---
        if num_decode >= MIN_SEQ_LEN and all_outputs_Main[num_keys_values_plus_2].numpy() in STOP_TOKEN:
            break

        # --- 3. Fused Feat_Encoder_Cond (one call instead of two) ---
        input_feed_Feat_Encoder_Cond[in_name_Feat_Encoder_Cond] = latent_pred
        feat_embed_new, feat_cond = ort_session_Feat_Encoder_Cond.run_with_ort_values(out_name_Feat_Encoder_Cond, input_feed_Feat_Encoder_Cond, run_options=run_options)

        # --- Update Main inputs for next decode step ---
        input_feed_Main.update(zip(in_name_Main[:num_keys_values], all_outputs_Main))
        input_feed_Main[in_name_Main[num_keys_values]]        = feat_embed_new
        input_feed_Main[in_name_Main[num_keys_values_plus_2]] = feat_embed_new

        # --- 4. Rotary for next position ---
        input_feed_Rotary_Mask_Text_Decode[in_name_Rotary_Mask_Text_Decode[0]] = kv_seq_len
        rotary_cos, rotary_sin, kv_seq_len = ort_session_Rotary_Mask_Text_Decode.run_with_ort_values(out_name_Rotary_Mask_Text_Decode, input_feed_Rotary_Mask_Text_Decode, run_options=run_options)
        input_feed_Main[in_name_Main[num_keys_values_plus_3]] = rotary_cos
        input_feed_Main[in_name_Main[num_keys_values_plus_4]] = rotary_sin

        # First decode step: switch to decode-mode inputs
        if num_decode < 1:
            input_feed_Main[in_name_Main[num_keys_values_plus_1]] = init_concat_text_len
            input_feed_Main[in_name_Main[num_keys_values_plus_5]] = init_decode_attention_mask

        num_decode += 1
        print(f"    Decode: {num_decode}")

    print(f"\nDecode Speed: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s\n")

    # ──────────────────────────────────────────────────────────────────────────
    # FINALIZE SENTENCE AUDIO (NON-STREAMING)
    # One-time numpy conversion after decode loop completes
    # ──────────────────────────────────────────────────────────────────────────
    if not STREAMING:
        if DYNAMIC_SHAPE_VAE_DECODE:
            # Concatenate all latents at once (single numpy call, outside hot loop)
            all_latents = np.concatenate([lp.numpy() for lp in save_latent_list], axis=1)
            vae_input = onnxruntime.OrtValue.ortvalue_from_numpy(all_latents.astype(model_dtype_VAE_Decoder), device_type, DEVICE_ID)
            input_feed_VAE_Decoder[in_name_VAE_Decoder] = vae_input
            audio_out_ort, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
            save_audio_out.append(audio_out_ort.numpy())
        else:
            # Paired decode without Concat model (numpy on small tensors, outside loop)
            for i in range(len(save_latent_list) - 1):
                paired = np.concatenate([save_latent_list[i].numpy(), save_latent_list[i + 1].numpy()], axis=1)
                vae_input = onnxruntime.OrtValue.ortvalue_from_numpy(paired.astype(model_dtype_VAE_Decoder), device_type, DEVICE_ID)
                input_feed_VAE_Decoder[in_name_VAE_Decoder] = vae_input
                audio_out_ort, _ = ort_session_VAE_Decoder.run_with_ort_values(out_name_VAE_Decoder, input_feed_VAE_Decoder, run_options=run_options)
                audio_out_np = audio_out_ort.numpy()
                if i > 0:
                    audio_out_np = audio_out_np[..., half_decode_len:]
                save_audio_out.append(audio_out_np)

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
