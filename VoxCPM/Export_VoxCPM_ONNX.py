import gc
import time
import torch
import site
import shutil
import soundfile as sf
import numpy as np
import onnxruntime
from pydub import AudioSegment
from modeling_modified.text_normalize import TextNormalizer
from transformers import LlamaTokenizerFast


path_voxcpm = r'/home/DakeQQ/Downloads/VoxCPM1.5'                                 # Set the folder path where the VoxCPM1.5 project downloaded.
onnx_model_A = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Text_Embed.onnx'       # Assign a path where the exported VoxCPM model stored.
onnx_model_B = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_VAE_Encoder.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Feat_Encoder.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Feat_Cond.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Concat.onnx'
onnx_model_F = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Main.onnx'
onnx_model_G = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_Feat_Decoder.onnx'
onnx_model_H = r'/home/DakeQQ/Downloads/VoxCPM_ONNX/VoxCPM_VAE_Decoder.onnx'

prompt_audio_path = "./example/basic_ref_zh.wav"                                # optional: path to a prompt speech for voice cloning else None.
prompt_text = "对，这就是我，万人敬仰的太乙真人。"                                    # The reference text for the prompt speech.
target_tts = [ "大家好，我现在正在大可奇奇体验AI科技。", "Hello everyone, I'm currently experiencing DakeQQ's AI technology."]  # The test query after the export process.
generated_audio_path = r"./generated.wav"                                       # The generated audio path.

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
CFG_VALUE = 2.2                          # Lower values result in more natural speech for long text, while higher values stay closer to the original sound features. Free to edit it.
RANDOM_SEED = 1                          # Global random seed. Free to edit it.

# === Feature flags ===
STREAMING = False                        # Enable streaming synthesis. Free to enable it. Unlike the official implementation, this version processes two latents at a time for faster performance, albeit with potential discontinuities during piece-by-piece decoding.
DYNAMIC_SHAPE_VAE_DECODE = True          # Use dynamic shape for VAE decoder. Free to enable it.
USE_TEXT_NORMALIZER = True               # Use text normalizer. Free to enable it.
USE_AUDIO_NORMALIZER = False             # Use an audio normalizer to stabilize loudness, though this may result in a loss of original audio characteristics. Free to enable it.
PREVENT_F16_OVERFLOW = False             # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
USE_F16_KV = False                       # Use float16 for key/value cache. Free to enable it. The quality of short sentences will decrease.

# === ONNX / runtime configuration ===
OPSET = 17                               # ONNX opset version. Free to edit it.
MAX_THREADS = 0                          # Parallel CPU threads, 0 for auto. Free to edit it.
DEVICE_ID = 0                            # Device id, default 0. Free to edit it.


py_site = site.getsitepackages()[-1]
shutil.copyfile('./modeling_modified/core.py', py_site + '/voxcpm/core.py')
shutil.copyfile('./modeling_modified/audio_vae.py', py_site + '/voxcpm/modules/audiovae/audio_vae.py')
from voxcpm import VoxCPM


class VOXCPM_TEXT_EMBED(torch.nn.Module):
    def __init__(self, voxcpm):
        super(VOXCPM_TEXT_EMBED, self).__init__()
        self.voxcpm = voxcpm

    def forward(self, text_ids):
        text_embed = self.voxcpm.base_lm.embed_tokens(text_ids)
        return text_embed


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


class VOXCPM_FEAT_ENCODER(torch.nn.Module):
    def __init__(self, voxcpm, max_prompt_audio_len, in_sample_rate):
        super(VOXCPM_FEAT_ENCODER, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        max_prompt_feat_len = (max_prompt_audio_len // in_sample_rate * 44100) // (self.voxcpm.patch_size * self.voxcpm.chunk_size) + 1
        self.special_tokens = self.voxcpm.feat_encoder.special_token.expand(1, max_prompt_feat_len, 1, -1).squeeze(0).half()
        self.q_len = self.voxcpm.patch_size + 1  # Fixed to 5 for VoxCPM1.5
        position_ids = torch.arange(self.q_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = self.voxcpm.feat_encoder.encoder.rope_emb(position_ids)
        rope_emb_sin[:, :self.voxcpm.feat_encoder.encoder.rope_emb.dim // 2] *= -1.0
        self.rope_emb_cos_q = rope_emb_cos.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.rope_emb_sin_q = rope_emb_sin.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.rope_emb_cos_k = self.rope_emb_cos_q.transpose(-1, -2)
        self.rope_emb_sin_k = self.rope_emb_sin_q.transpose(-1, -2)
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

    def rotate_half(self, x, dim):
        x1, x2 = x.split(self.split_size, dim=dim)
        return torch.cat((x2, x1), dim=dim)

    def forward(self, audio_feat):
        audio_feat_len = audio_feat.shape[0].unsqueeze(0)
        hidden_states = self.voxcpm.feat_encoder.in_proj(audio_feat)
        hidden_states = torch.cat([self.special_tokens[:audio_feat_len].float(), hidden_states], dim=-2)
        hidden_states = hidden_states.view(-1, self.q_len, self.voxcpm.feat_encoder.in_proj.out_features)
        for layer in self.voxcpm.feat_encoder.encoder.layers:
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            qkv = layer.self_attn.qkv(hidden_states)
            q, k, v = torch.split(qkv, [layer.self_attn.q_out_features, layer.self_attn.k_out_features, layer.self_attn.v_out_features], dim=-1)
            q = q.view(-1, self.q_len, layer.self_attn.num_key_value_heads, layer.self_attn.num_key_value_groups, layer.self_attn.head_dim).permute(0, 2, 3, 1, 4)
            k = k.view(-1, self.q_len, 1, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).permute(0, 3, 2, 4, 1)
            v = v.view(-1, self.q_len, 1, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 3)
            q = q * self.rope_emb_cos_q + self.rotate_half(q, -1) * self.rope_emb_sin_q
            k = k * self.rope_emb_cos_k + self.rotate_half(k, -2) * self.rope_emb_sin_k
            attn = torch.nn.functional.softmax(torch.matmul(q, k), dim=-1, dtype=torch.float32)
            attn = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(-1, self.q_len, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        feat_embed = hidden_states[:, 0]
        if PREVENT_F16_OVERFLOW:
            feat_embed = feat_embed * self.overflow_scale
        feat_embed = feat_embed * torch.rsqrt(feat_embed.square().sum(-1, keepdim=True))
        feat_embed = self.voxcpm.enc_to_lm_proj(feat_embed).unsqueeze(0)
        return feat_embed


class VOXCPM_FEAT_COND(torch.nn.Module):
    def __init__(self, voxcpm):
        super(VOXCPM_FEAT_COND, self).__init__()
        self.voxcpm = voxcpm

    def forward(self, audio_feat):
        feat_cond = self.voxcpm.feat_decoder.estimator.cond_proj(audio_feat[[-1]])
        feat_cond = torch.cat([feat_cond, feat_cond], dim=0)
        return feat_cond


class VOXCPM_CONCAT(torch.nn.Module):
    def __init__(self):
        super(VOXCPM_CONCAT, self).__init__()
        pass

    def forward(self, embed_0, embed_1):
        concat_embed = torch.cat([embed_0, embed_1], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


class VOXCPM_MAIN(torch.nn.Module):
    def __init__(self, voxcpm, max_seq_len):
        super(VOXCPM_MAIN, self).__init__()
        self.voxcpm = voxcpm
        self._replace_gelu_with_tanh_approximation(self.voxcpm)
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        position_ids = torch.arange(max_seq_len, dtype=torch.int32)
        rope_emb_cos, rope_emb_sin = self.voxcpm.base_lm.rope_emb(position_ids)
        rope_emb_sin[:, :self.voxcpm.base_lm.rope_emb.dim // 2] *= -1.0
        self.cos_rotary_pos_emb = rope_emb_cos.unsqueeze(0).unsqueeze(0).half()
        self.sin_rotary_pos_emb = rope_emb_sin.unsqueeze(0).unsqueeze(0).half()
        self.total_layers = self.voxcpm.base_lm.config.num_hidden_layers + self.voxcpm.residual_lm.config.num_hidden_layers
        self.save_key = [None] * self.total_layers
        self.save_value = [None] * self.total_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.split_size_base = self.voxcpm.base_lm.layers._modules['0'].self_attn.head_dim // 2
        self.split_size_residual = self.voxcpm.residual_lm.layers._modules['0'].self_attn.head_dim // 2
        self.norm_factor = self.voxcpm.base_lm.config.hidden_size ** 0.5
        self.scale_factor_base = float(self.voxcpm.base_lm.layers._modules['0'].self_attn.head_dim ** -0.25)
        with torch.no_grad():
            self._fuse_layers(self.voxcpm.base_lm.layers)
            self._fuse_layers(self.voxcpm.residual_lm.layers)
            w = self.voxcpm.residual_lm.norm.weight.unsqueeze(0) * self.norm_factor
            self.voxcpm.res_to_dit_proj.weight.mul_(w)
            del self.voxcpm.residual_lm.norm

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _fuse_layers(self, layers):
        for layer in layers:
            q_proj = layer.self_attn.q_proj
            k_proj = layer.self_attn.k_proj
            v_proj = layer.self_attn.v_proj
            in_features = int(q_proj.in_features)
            out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
            has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
            qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
            qkv.weight.copy_(torch.cat([q_proj.weight * self.scale_factor_base, k_proj.weight * self.scale_factor_base, v_proj.weight], dim=0))
            if has_bias:
                z = lambda feat: torch.zeros(feat, dtype=q_proj.weight.dtype, device=q_proj.weight.device)
                qb = q_proj.bias if q_proj.bias is not None else z(q_proj.out_features)
                kb = k_proj.bias if k_proj.bias is not None else z(k_proj.out_features)
                vb = v_proj.bias if v_proj.bias is not None else z(v_proj.out_features)
                qkv.bias.copy_(torch.cat([qb * self.scale_factor_base, kb * self.scale_factor_base, vb], dim=0))

            layer.self_attn.q_out_features = int(q_proj.out_features)
            layer.self_attn.k_out_features = int(k_proj.out_features)
            layer.self_attn.v_out_features = int(v_proj.out_features)
            layer.self_attn.qkv = qkv

            del layer.self_attn.q_proj
            del layer.self_attn.k_proj
            del layer.self_attn.v_proj

            w = layer.input_layernorm.weight.unsqueeze(0) * self.norm_factor
            qkv.weight.mul_(w)
            del layer.input_layernorm

            w = layer.post_attention_layernorm.weight.unsqueeze(0) * self.norm_factor
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

    def rotate_half(self, x, dim):
        x1, x2 = x.split(self.split_size_base, dim=dim)
        return torch.cat((x2, x1), dim=dim)

    def forward(self, *all_inputs):
        history_len = all_inputs[-6]
        feat_embed = all_inputs[-5]
        concat_text_len = all_inputs[-4]
        hidden_states = all_inputs[-3]
        ids_len = all_inputs[-2]
        mask = all_inputs[-1]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2)
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * mask).float()
        for i, layer in enumerate(self.voxcpm.base_lm.layers):
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states_norm = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            qkv = layer.self_attn.qkv(hidden_states_norm)
            q, k, v = torch.split(qkv, [layer.self_attn.q_out_features, layer.self_attn.k_out_features, layer.self_attn.v_out_features], dim=-1)
            q = q.view(-1, layer.self_attn.num_key_value_heads, layer.self_attn.num_key_value_groups, layer.self_attn.head_dim).permute(1, 2, 0, 3)
            k = k.view(-1, 1, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).permute(2, 1, 3, 0)
            q = q * rotary_pos_emb_cos_q + self.rotate_half(q, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + self.rotate_half(k, -2) * rotary_pos_emb_sin_k
            if USE_F16_KV:
                k = k.half()
                v = v.half()
            v = v.view(-1, 1, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(0, 2)
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.total_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            if USE_F16_KV:
                k = k.float()
                v = v.float()
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        hidden_states = self.voxcpm.base_lm.norm(hidden_states)
        fsq_layer_out = self.voxcpm.fsq_layer(hidden_states[:, concat_text_len:])
        hidden_states = hidden_states[:, :concat_text_len]
        lm_hidden = torch.cat([hidden_states, fsq_layer_out], dim=1)[:, [-1]]
        hidden_states = torch.cat([hidden_states, fsq_layer_out + feat_embed], dim=1)
        i = self.voxcpm.base_lm.config.num_hidden_layers
        for layer in self.voxcpm.residual_lm.layers:
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states_norm = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            qkv = layer.self_attn.qkv(hidden_states_norm)
            q, k, v = torch.split(qkv, [layer.self_attn.q_out_features, layer.self_attn.k_out_features, layer.self_attn.v_out_features], dim=-1)
            q = q.view(-1, layer.self_attn.num_key_value_heads, layer.self_attn.num_key_value_groups, layer.self_attn.head_dim).permute(1, 2, 0, 3)
            k = k.view(-1, 1, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).permute(2, 1, 3, 0)
            q = q * rotary_pos_emb_cos_q + self.rotate_half(q, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + self.rotate_half(k, -2) * rotary_pos_emb_sin_k
            if USE_F16_KV:
                k = k.half()
                v = v.half()
            v = v.view(-1, 1, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(0, 2)
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.total_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            if USE_F16_KV:
                k = k.float()
                v = v.float()
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn = torch.matmul(attn, v).permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
            i += 1
        residual_hidden = hidden_states[:, [-1]]
        if PREVENT_F16_OVERFLOW:
            residual_hidden = residual_hidden * self.overflow_scale
        residual_hidden = residual_hidden * torch.rsqrt(residual_hidden.square().sum(-1, keepdim=True))
        dit_hidden_1 = self.voxcpm.lm_to_dit_proj(lm_hidden)
        dit_hidden_2 = self.voxcpm.res_to_dit_proj(residual_hidden)
        dit_hidden = dit_hidden_1 + dit_hidden_2
        random = torch.randn((1, self.voxcpm.patch_size, self.voxcpm.feat_decoder.in_channels), dtype=torch.float32)
        stop_flag = self.voxcpm.stop_head(self.voxcpm.stop_actn(self.voxcpm.stop_proj(lm_hidden))).argmax(dim=-1, keepdims=False).int()
        return *self.save_key, *self.save_value, kv_seq_len, random, dit_hidden, stop_flag


class VOXCPM_FEAT_DECODER(torch.nn.Module):
    def __init__(self, voxcpm, fixed_timesteps):
        super(VOXCPM_FEAT_DECODER, self).__init__()
        self.voxcpm = voxcpm
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
        self.rope_emb_cos_q = rope_emb_cos.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.rope_emb_sin_q = rope_emb_sin.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.rope_emb_cos_k = self.rope_emb_cos_q.transpose(-1, -2)
        self.rope_emb_sin_k = self.rope_emb_sin_q.transpose(-1, -2)
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

    def rotate_half(self, x, dim):
        x1, x2 = x.split(self.split_size, dim=dim)
        return torch.cat((x2, x1), dim=dim)

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
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states_norm = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            qkv = layer.self_attn.qkv(hidden_states_norm)
            q, k, v = torch.split(qkv, [layer.self_attn.q_out_features, layer.self_attn.k_out_features, layer.self_attn.v_out_features], dim=-1)
            q = q.view(-1, self.q_len, layer.self_attn.num_key_value_heads, layer.self_attn.num_key_value_groups, layer.self_attn.head_dim).permute(0, 2, 3, 1, 4)
            k = k.view(-1, self.q_len, 1, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).permute(0, 3, 2, 4, 1)
            v = v.view(-1, self.q_len, 1, layer.self_attn.num_key_value_heads, layer.self_attn.head_dim).transpose(1, 3)
            q = q * self.rope_emb_cos_q + self.rotate_half(q, -1) * self.rope_emb_sin_q
            k = k * self.rope_emb_cos_k + self.rotate_half(k, -2) * self.rope_emb_sin_k
            attn = torch.nn.functional.softmax(torch.matmul(q, k), dim=-1, dtype=torch.float32)
            attn = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(-1, self.q_len, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        hidden_states = hidden_states[:, self.prefix_plus:]
        if PREVENT_F16_OVERFLOW:
            hidden_states = hidden_states * self.overflow_scale
        hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
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


print('Export start ...')
with torch.inference_mode():
    model = VoxCPM.from_pretrained(path_voxcpm, load_denoiser=False, optimize=False).tts_model
    model = model.float().to('cpu').eval()

    # ============================================================================
    # Model A: Text Embedding
    # ============================================================================
    model_A = VOXCPM_TEXT_EMBED(model)
    text_ids = torch.zeros([1, 10], dtype=torch.int32)  # "10" is just a dummy value.
    torch.onnx.export(
        model_A,
        (text_ids,),
        onnx_model_A,
        input_names=['text_ids'],
        output_names=['text_embed'],
        dynamic_axes={
            'text_ids': {1: 'ids_len'},
            'text_embed': {1: 'ids_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del model_A
    del text_ids

    # ============================================================================
    # Model B: VAE Encoder
    # ============================================================================
    model_B = VOXCPM_VAE_ENCODER(model, IN_SAMPLE_RATE)
    prompt_audio = torch.zeros([1, 1, MAX_PROMPT_AUDIO_LEN], dtype=torch.int16)
    torch.onnx.export(
        model_B,
        (prompt_audio,),
        onnx_model_B,
        input_names=['prompt_audio'],
        output_names=['audio_feat'],
        dynamic_axes={
            'prompt_audio': {2: 'audio_len'},
            'audio_feat': {0: 'audio_feat_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del model_B
    del prompt_audio

    # ============================================================================
    # Model C: Feature Encoder
    # ============================================================================
    model_C = VOXCPM_FEAT_ENCODER(model, MAX_PROMPT_AUDIO_LEN, IN_SAMPLE_RATE)
    audio_feat = torch.zeros([20, model.patch_size, model.feat_dim], dtype=torch.float32)  # "20" is just a dummy value.
    torch.onnx.export(
        model_C,
        (audio_feat,),
        onnx_model_C,
        input_names=['audio_feat'],
        output_names=['feat_embed'],
        dynamic_axes={
            'audio_feat': {0: 'audio_feat_len'},
            'feat_embed': {1: 'audio_feat_len'},
        },
        opset_version=OPSET,
        dynamo=False
    )
    del model_C
    del audio_feat

    # ============================================================================
    # Model D: Feature Conditioning
    # ============================================================================
    model_D = VOXCPM_FEAT_COND(model)
    audio_feat = torch.zeros([20, model.patch_size, model.feat_dim], dtype=torch.float32)  # "20" is just a dummy value.
    torch.onnx.export(
        model_D,
        (audio_feat,),
        onnx_model_D,
        input_names=['audio_feat'],
        output_names=['feat_cond'],
        dynamic_axes={
            'audio_feat': {0: 'audio_feat_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del model_D
    del audio_feat

    # ============================================================================
    # Model E: Concatenation
    # ============================================================================
    model_E = VOXCPM_CONCAT()
    embed_0 = torch.zeros([1, 10, model.feat_encoder.config.hidden_size], dtype=torch.float32)     # "10" is just a dummy value.
    embed_1 = torch.zeros([1, 10, model.base_lm.embed_tokens.embedding_dim], dtype=torch.float32)
    torch.onnx.export(
        model_E,
        (embed_0, embed_1),
        onnx_model_E,
        input_names=['embed_0', 'embed_1'],
        output_names=['concat_embed', 'concat_len'],
        dynamic_axes={
            'embed_0': {1: 'embed_len_0', 2: 'embed_size'},
            'embed_1': {1: 'embed_len_1', 2: 'embed_size'},
            'concat_embed': {1: 'concat_len', 2: 'embed_size'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del model_E
    del embed_0
    del embed_1

    # ============================================================================
    # Model F: Main Model (with KV cache)
    # ============================================================================
    model_F = VOXCPM_MAIN(model, MAX_SEQ_LEN)

    # Extract model dimensions
    base_lm_head_dim = model.base_lm.layers._modules['0'].self_attn.head_dim
    base_lm_num_key_value_heads = model.base_lm.layers._modules['0'].self_attn.num_key_value_heads
    residual_lm_head_dim = model.residual_lm.layers._modules['0'].self_attn.head_dim
    residual_lm_num_key_value_heads = model.residual_lm.layers._modules['0'].self_attn.num_key_value_heads
    base_lm_num_attn_layers = model.base_lm.config.num_hidden_layers
    residual_lm_num_attn_layers = model.residual_lm.config.num_hidden_layers

    # Create dummy inputs
    ids_len = torch.tensor([25], dtype=torch.int64)              # "25" is just a dummy value.
    concat_text_len = torch.tensor([10], dtype=torch.int64)      # "10" is just a dummy value.
    feat_embed = torch.zeros([1, ids_len - concat_text_len, model.feat_encoder.config.hidden_size], dtype=torch.float32)
    hidden_states = torch.ones((1, ids_len, model.base_lm.embed_tokens.embedding_dim), dtype=torch.float32)
    history_len = torch.tensor([0], dtype=torch.int64)
    attention_mask = torch.tensor([1], dtype=torch.int8)

    if USE_F16_KV:
        kv_dtype = torch.float16
    else:
        kv_dtype = torch.float32
    base_lm_past_keys = torch.zeros((base_lm_num_key_value_heads, 1, base_lm_head_dim, 0), dtype=kv_dtype)
    base_lm_past_values = torch.zeros((base_lm_num_key_value_heads, 1, 0, base_lm_head_dim), dtype=kv_dtype)
    residual_lm_past_keys = torch.zeros((residual_lm_num_key_value_heads, 1, residual_lm_head_dim, 0), dtype=kv_dtype)
    residual_lm_past_values = torch.zeros((residual_lm_num_key_value_heads, 1, 0, residual_lm_head_dim), dtype=kv_dtype)

    # Prepare input and output names
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {}

    # Base LM keys
    for i in range(base_lm_num_attn_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(base_lm_past_keys)
        dynamic_axes[name] = {3: 'history_len'}

        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {3: 'kv_seq_len'}

    # Residual LM keys
    for i in range(base_lm_num_attn_layers, base_lm_num_attn_layers + residual_lm_num_attn_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(residual_lm_past_keys)
        dynamic_axes[name] = {3: 'history_len'}

        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {3: 'kv_seq_len'}

    # Base LM values
    for i in range(base_lm_num_attn_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(base_lm_past_values)
        dynamic_axes[name] = {2: 'history_len'}

        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'kv_seq_len'}

    # Residual LM values
    for i in range(base_lm_num_attn_layers, base_lm_num_attn_layers + residual_lm_num_attn_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(residual_lm_past_values)
        dynamic_axes[name] = {2: 'history_len'}

        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'kv_seq_len'}

    # Other inputs
    input_names.append('history_len')
    all_inputs.append(history_len)

    input_names.append('feat_embed')
    all_inputs.append(feat_embed)
    dynamic_axes["feat_embed"] = {1: 'audio_feat_len'}

    input_names.append('concat_text_len')
    all_inputs.append(concat_text_len)

    input_names.append('hidden_states')
    all_inputs.append(hidden_states)
    dynamic_axes["hidden_states"] = {1: 'ids_len'}

    input_names.append('ids_len')
    all_inputs.append(ids_len)

    input_names.append('attention_mask')
    all_inputs.append(attention_mask)

    # Other outputs
    output_names.append('kv_seq_len')
    output_names.append('random')
    output_names.append('dit_hidden')
    output_names.append('stop_flag')

    torch.onnx.export(
        model_F,
        tuple(all_inputs),
        onnx_model_F,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )

    del model_F
    del all_inputs
    del base_lm_past_keys
    del base_lm_past_values
    del residual_lm_past_keys
    del residual_lm_past_values
    del base_lm_num_attn_layers
    del residual_lm_num_attn_layers
    del feat_embed
    del hidden_states
    del ids_len
    del history_len
    del input_names
    del output_names
    del dynamic_axes
    del base_lm_head_dim
    del base_lm_num_key_value_heads
    del residual_lm_head_dim
    del residual_lm_num_key_value_heads

    # ============================================================================
    # Model G: Feature Decoder (Diffusion)
    # ============================================================================
    model_G = VOXCPM_FEAT_DECODER(model, FIXED_TIMESTEPS)
    step = torch.tensor([0], dtype=torch.int32)
    random = torch.ones((1, model.patch_size, model.feat_decoder.in_channels), dtype=torch.float32)
    dit_hidden = torch.zeros((1, 1, model.base_lm.embed_tokens.embedding_dim), dtype=torch.float32)
    feat_cond = torch.zeros((2, model.patch_size, model.feat_decoder.estimator.cond_proj.out_features), dtype=torch.float32)
    cfg_value = torch.tensor([CFG_VALUE], dtype=torch.float32)
    cfg_value_minus = torch.tensor([1.0 - CFG_VALUE], dtype=torch.float32)
    torch.onnx.export(
        model_G,
        (step, random, dit_hidden, feat_cond, cfg_value, cfg_value_minus),
        onnx_model_G,
        input_names=['step', 'random', 'dit_hidden', 'feat_cond', 'cfg_value', 'cfg_value_minus'],
        output_names=['next_step', 'next_random'],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False
    )
    del model_G
    del step
    del random
    del dit_hidden
    del feat_cond
    del cfg_value
    del cfg_value_minus

    # ============================================================================
    # Model H: VAE Decoder
    # ============================================================================
    model_H = VOXCPM_VAE_DECODE(model, OUT_SAMPLE_RATE)
    latent_pred = torch.ones((1, model.patch_size + model.patch_size, model.feat_decoder.in_channels), dtype=torch.float32)
    torch.onnx.export(
        model_H,
        (latent_pred,),
        onnx_model_H,
        input_names=['latent_pred'],
        output_names=['audio_out', 'audio_out_len'],
        dynamic_axes={
            'latent_pred': {1: 'latent_pred_len'},
            'audio_out': {2: 'audio_out_len'}
        } if DYNAMIC_SHAPE_VAE_DECODE else None,
        opset_version=OPSET,
        dynamo=False
    )
    del model_H
    del latent_pred
    del model
    gc.collect()
print('\nExport done!\n\nStart running the VoxCPM by ONNXRuntime.\nNow loading . . . it could cost minutes.')


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


# settings
onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
session_opts.log_severity_level = 4                 # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4                # Fatal level, it an adjustable value.
run_options.log_severity_level = 4                  # Fatal level, it an adjustable value.
run_options.log_verbosity_level = 4                 # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS     # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS     # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry('session.set_denormal_as_zero', '1')
session_opts.add_session_config_entry('session.intra_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.inter_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.enable_quant_qdq_cleanup', '1')
session_opts.add_session_config_entry('session.qdq_matmulnbits_accuracy_level', '4')
session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
session_opts.add_session_config_entry('session.graph_optimizations_loop_level', '2')
session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
session_opts.add_session_config_entry('optimization.enable_cast_chain_elimination', '1')
run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

ORT_Accelerate_Providers = ['CPUExecutionProvider']
device_type = 'cpu'
provider_options = None

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = in_name_A[0].name
out_name_A = [out_name_A[0].name]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B = in_name_B[0].name
out_name_B = [out_name_B[0].name]

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C = in_name_C[0].name
out_name_C = [out_name_C[0].name]

ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
model_dtype_D = ort_session_D._inputs_meta[0].type
if 'float16' in model_dtype_D:
    model_dtype_D = np.float16
else:
    model_dtype_D = np.float32
in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()
in_name_D = in_name_D[0].name
out_name_D = [out_name_D[0].name]

ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_E = ort_session_E.get_inputs()
out_name_E = ort_session_E.get_outputs()
in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]

ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_F.get_providers()}\n")
model_dtype_F = ort_session_F._inputs_meta[0].type
if 'float16' in model_dtype_F:
    model_dtype_F = np.float16
else:
    model_dtype_F = np.float32
in_name_F = ort_session_F.get_inputs()
out_name_F = ort_session_F.get_outputs()
amount_of_outputs_F = len(out_name_F)
in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
out_name_F = [out_name_F[i].name for i in range(amount_of_outputs_F)]

ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
model_dtype_G = ort_session_G._inputs_meta[2].type
if 'float16' in model_dtype_G:
    model_dtype_G = np.float16
else:
    model_dtype_G = np.float32
in_name_G = ort_session_G.get_inputs()
out_name_G = ort_session_G.get_outputs()
in_name_G = [in_name_G[i].name for i in range(len(in_name_G))]
out_name_G = [out_name_G[i].name for i in range(len(out_name_G))]

ort_session_H = onnxruntime.InferenceSession(onnx_model_H, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
model_dtype_H = ort_session_H._inputs_meta[0].type
if 'float16' in model_dtype_H:
    model_dtype_H = np.float16
else:
    model_dtype_H = np.float32

shape_value_in_H = ort_session_H._inputs_meta[0].shape[1]
if isinstance(shape_value_in_H, str):
    DYNAMIC_SHAPE_VAE_DECODE = True
else:
    DYNAMIC_SHAPE_VAE_DECODE = False

in_name_H = ort_session_H.get_inputs()
out_name_H = ort_session_H.get_outputs()
in_name_H = in_name_H[0].name
out_name_H = [out_name_H[i].name for i in range(len(out_name_H))]
half_decode_len = 7056  # Fixed for VoxCPM1.5

# ==============================================================================
# 1. Configuration & Constants Calculation
# ==============================================================================
generate_limit = MAX_SEQ_LEN - 1
num_keys_values = amount_of_outputs_F - 4
num_layers = num_keys_values // 2

# Pre-calculate indices for clarity
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5

# ==============================================================================
# 2. Initialize Static ORT Values (Scalars, Masks, & Buffers)
# ==============================================================================
# Simple Scalars and Lengths
init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_concat_text_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)

# Special Tokens
init_audio_start_ids = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[101]], dtype=np.int32), device_type, DEVICE_ID)

# Attention Masks
init_attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
init_attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)

# Large Zero Buffers (Cache & Embeddings)
shape_keys = (ort_session_F._inputs_meta[0].shape[0], 1, ort_session_F._inputs_meta[0].shape[2], 0)
shape_vals = (ort_session_F._inputs_meta[num_layers].shape[0], 1, 0, ort_session_F._inputs_meta[num_layers].shape[3])
shape_embed = (1, 0, ort_session_F._inputs_meta[num_keys_values_plus_1].shape[2])
shape_latent = (ort_session_H._inputs_meta[0].shape[0], 0, ort_session_H._inputs_meta[0].shape[2])

init_past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape_keys, dtype=model_dtype_F), device_type, DEVICE_ID)
init_past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape_vals, dtype=model_dtype_F), device_type, DEVICE_ID)
init_feat_embed = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape_embed, dtype=model_dtype_F), device_type, DEVICE_ID)
init_latent_pred = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape_latent, dtype=model_dtype_H), device_type, DEVICE_ID)

# Config Values (CFG)
cfg_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([CFG_VALUE], dtype=model_dtype_G), device_type, DEVICE_ID)
cfg_value_minus = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1.0 - CFG_VALUE], dtype=model_dtype_G), device_type, DEVICE_ID)

# Pre-calculate Time Steps
timesteps = FIXED_TIMESTEPS - 1
init_cfm_steps = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int32), device_type, DEVICE_ID)

# Audio Post-processing
blank_segment = np.zeros((1, 1, int(OUT_SAMPLE_RATE * 0.1)), dtype=np.int16)

# ==============================================================================
# 3. Session Setup & IO Bindings
# ==============================================================================
# Initialize Input Feeds
input_feed_A = {}
input_feed_B = {}
input_feed_C = {}
input_feed_D = {}
input_feed_E = {}
input_feed_F = {}
input_feed_G = {}
input_feed_H = {}
input_feed_I = {}

# Session A: Audio Start Embedding
input_feed_A[in_name_A] = init_audio_start_ids
audio_start_embed = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]

# Session D: IO Binding & Initialization
input_feed_D[in_name_D] = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros([1, ort_session_D._inputs_meta[0].shape[1], ort_session_D._inputs_meta[0].shape[2]], dtype=model_dtype_D), device_type, DEVICE_ID)
init_feat_cond_0 = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)[0]

# Session G: IO Binding & Fixed Inputs
input_feed_G[in_name_G[4]] = cfg_value
input_feed_G[in_name_G[5]] = cfg_value_minus

# ==============================================================================
# 4. Preprocessing & Prompt Handling
# ==============================================================================
tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(path_voxcpm))
text_normalizer = TextNormalizer()

# Handle Audio/Text Prompt
if prompt_audio_path:
    if prompt_text:
        use_prompt_audio = True
        # Process Audio
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
    # Run Audio Encoder (Session B)
    input_feed_B[in_name_B] = audio
    audio_feat = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0]
    
    # Run Feature Condition (Session D)
    input_feed_D[in_name_D] = audio_feat
    init_feat_cond = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)[0]
    
    # Process Text
    if USE_TEXT_NORMALIZER:
        prompt_text = text_normalizer.normalize(prompt_text)
    prompt_ids = np.array([tokenizer(prompt_text)], dtype=np.int32)
    prompt_text_len = prompt_ids.shape[-1]
    
    # Run Text Encoder (Session A)
    input_feed_A[in_name_A] = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_ids, device_type, DEVICE_ID)
    prompt_embed = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
else:
    # Use Defaults
    init_feat_cond = init_feat_cond_0
    prompt_text_len = 0

# ==============================================================================
# 5. Main Generation Loop
# ==============================================================================
save_audio_out = []

for sentence in target_tts:
    print(f"Convert to Speech: {sentence}")
    if USE_TEXT_NORMALIZER:
        sentence = text_normalizer.normalize(sentence)

    # 5.1 Encode Target Text
    target_ids = np.array([tokenizer(sentence)], dtype=np.int32)
    input_feed_A[in_name_A] = onnxruntime.OrtValue.ortvalue_from_numpy(target_ids, device_type, DEVICE_ID)
    target_embed = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
    
    # 5.2 Combine Embeddings (Session E)
    if use_prompt_audio:
        input_feed_E[in_name_E[0]] = prompt_embed
        input_feed_E[in_name_E[1]] = target_embed
        target_embed, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)

    input_feed_E[in_name_E[0]] = target_embed
    input_feed_E[in_name_E[1]] = audio_start_embed
    concat_embed, concat_text_len = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)

    # 5.3 Calculate Max Length & Initial Features
    if use_prompt_audio:
        input_feed_C[in_name_C] = audio_feat
        feat_embed = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)[0]
        
        input_feed_E[in_name_E[0]] = concat_embed
        input_feed_E[in_name_E[1]] = feat_embed
        concat_embed, ids_len = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
    else:
        feat_embed = init_feat_embed
        ids_len = concat_text_len

    max_len = min((concat_text_len.numpy() - prompt_text_len) * DECODE_LIMIT_FACTOR + 10, generate_limit - ids_len.numpy())

    # 5.4 Prepare Decoder Inputs (Session F)
    input_feed_F[in_name_F[num_keys_values]] = init_history_len
    input_feed_F[in_name_F[num_keys_values_plus_1]] = feat_embed
    input_feed_F[in_name_F[num_keys_values_plus_2]] = concat_text_len
    input_feed_F[in_name_F[num_keys_values_plus_3]] = concat_embed
    input_feed_F[in_name_F[num_keys_values_plus_4]] = ids_len
    input_feed_F[in_name_F[num_keys_values_plus_5]] = init_attention_mask_1

    # Reset KV Cache
    for i in range(num_layers):
        input_feed_F[in_name_F[i]] = init_past_keys_F
    for i in range(num_layers, num_keys_values):
        input_feed_F[in_name_F[i]] = init_past_values_F

    # Copy initial condition to avoid overwrite
    feat_cond = init_feat_cond

    # Prepare Latent Storage
    if not STREAMING:
        save_latent = init_latent_pred if DYNAMIC_SHAPE_VAE_DECODE else []

    # --------------------------------------------------------------------------
    # 5.5 Auto-regressive Decoding Loop
    # --------------------------------------------------------------------------
    num_decode = 0
    start_decode = time.time()
    
    while num_decode < max_len:
        # --- Run Transformer (Session F) ---
        all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)

        # --- Run Flow Matching / Diffusion (Session G) ---
        input_feed_G[in_name_G[0]] = init_cfm_steps
        input_feed_G[in_name_G[1]] = all_outputs_F[num_keys_values_plus_1]
        input_feed_G[in_name_G[2]] = all_outputs_F[num_keys_values_plus_2]
        input_feed_G[in_name_G[3]] = feat_cond

        for i in range(timesteps):
            all_outputs_G = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)
            input_feed_G[in_name_G[0]] = all_outputs_G[0]
            input_feed_G[in_name_G[1]] = all_outputs_G[1]

        latent_pred = all_outputs_G[1]

        # --- Handle Output (Stream or Save) ---
        if STREAMING:
            if num_decode < 1:
                pre_latent_pred = latent_pred
            else:
                input_feed_E[in_name_E[0]] = pre_latent_pred
                input_feed_E[in_name_E[1]] = latent_pred
                save_latent, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                input_feed_H[in_name_H] = save_latent
                audio_out, _ = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
                pre_latent_pred = latent_pred
                audio_out = audio_out.numpy()
                if num_decode > 1:
                    audio_out = audio_out[..., half_decode_len:]
                save_audio_out.append(audio_out)
        else:
            if DYNAMIC_SHAPE_VAE_DECODE:
                input_feed_E[in_name_E[0]] = save_latent
                input_feed_E[in_name_E[1]] = latent_pred
                save_latent, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
            else:
                save_latent.append(latent_pred)

        # --- Check Stop Token ---
        if num_decode >= MIN_SEQ_LEN and all_outputs_F[num_keys_values_plus_3].numpy() in STOP_TOKEN:
            break

        # --- Update Inputs for Next Iteration ---
        input_feed_C[in_name_C] = latent_pred
        feat_embed = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)[0]

        input_feed_D[in_name_D] = latent_pred
        feat_cond = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)[0]

        input_feed_F.update(zip(in_name_F[:num_keys_values_plus_1], all_outputs_F))
        input_feed_F[in_name_F[num_keys_values_plus_1]] = feat_embed
        input_feed_F[in_name_F[num_keys_values_plus_3]] = feat_embed

        if num_decode < 1:
            # First Step Initialization
            input_feed_F[in_name_F[num_keys_values_plus_2]] = init_concat_text_len
            input_feed_F[in_name_F[num_keys_values_plus_4]] = init_ids_len_1
            input_feed_F[in_name_F[num_keys_values_plus_5]] = init_attention_mask_0
        
        num_decode += 1
        print(f"    Decode: {num_decode}")

    print(f"\nDecode Speed: {((num_decode + 1) / (time.time() - start_decode)):.3f} token/s\n")

    # 5.6 Finalize Sentence Audio (Non-Streaming)
    if not STREAMING:
        if DYNAMIC_SHAPE_VAE_DECODE:
            input_feed_H[in_name_H] = save_latent
            audio_out, _ = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
            save_audio_out.append(audio_out.numpy())
        else:
            input_feed_E[in_name_E[0]] = save_latent[0]
            input_feed_E[in_name_E[1]] = save_latent[1]
            concat_latent, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
            input_feed_H[in_name_H] = concat_latent
            audio_out, _ = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
            save_audio_out.append(audio_out.numpy())
            for i in range(2, len(save_latent)):
                input_feed_E[in_name_E[0]] = save_latent[i - 1]
                input_feed_E[in_name_E[1]] = save_latent[i]
                concat_latent, _ = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                input_feed_H[in_name_H] = concat_latent
                audio_out, _ = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
                audio_out = audio_out.numpy()[..., half_decode_len:]
                save_audio_out.append(audio_out)

    save_audio_out.append(blank_segment)

# ==============================================================================
# 6. Post-Processing & Stats
# ==============================================================================
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
