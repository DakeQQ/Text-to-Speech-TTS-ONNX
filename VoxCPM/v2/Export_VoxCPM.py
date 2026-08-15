import gc
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from voxcpm.model.voxcpm2 import VoxCPM2Model


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT PATHS AND LOCAL IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
script_dir = Path(__file__).resolve().parent
raw_onnx_folder = script_dir / "VoxCPM_ONNX_Raw"
onnx_folder = script_dir / "VoxCPM_ONNX"
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
if str(script_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent.parent))
from Rewrite_VoxCPM_ONNX import rewrite_voxcpm2_onnx_folder
from Shared_Weights import (
    GraphComponent,
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    bundle_shared_initializers,
    compose_graphs,
)


# ══════════════════════════════════════════════════════════════════════════════
# USER-CONFIGURABLE EXPORT OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
path_voxcpm2 = Path.home() / "Downloads" / "VoxCPM2"  # Local VoxCPM2 model folder.

MAX_SEQ_LEN = 2048                       # Maximum exported context length.
IN_SAMPLE_RATE = 16000                   # Prompt-audio ONNX input rate.
OUT_SAMPLE_RATE = 48000                  # Generated-waveform ONNX output rate.
IN_AUDIO_DTYPE = "F32"                   # "F16" | "F32" | "INT16".
OUT_AUDIO_DTYPE = "F32"                  # "F16" | "F32" | "INT16".
FIXED_TIMESTEPS = 10                     # More diffusion steps improve quality but cost latency.

USE_F16_KV = True                        # Store the growing key/value cache in float16.
COMPUTE_IN_F32 = False                   # With F16 KV, upcast K/V for attention matmuls when True.

REORDER_DOWNPROJ_FOR_QUANT = True        # Exact MLP channel permutation for later quantization.
REORDER_OPROJ_FOR_QUANT = True           # Exact value/o-projection channel permutation.
REORDER_KEY = "absmean"                  # "absmean" | "L4" | "rms" | "std".


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL MODEL AND EXPORT CONTRACT
# ══════════════════════════════════════════════════════════════════════════════
STOP_TOKEN = {1}
MODEL_IN_SAMPLE_RATE = 16000
MODEL_OUT_SAMPLE_RATE = 48000
CFG_VALUE = 2.0
DYNAMIC_SHAPE_VAE_DECODE = True
OPSET = 20

_AUDIO_DTYPES = {
    "F16": torch.float16,
    "F32": torch.float32,
    "INT16": torch.int16,
}
# ══════════════════════════════════════════════════════════════════════════════
# EXPANDED-EXPORT GRAPH PATHS
# ══════════════════════════════════════════════════════════════════════════════
onnx_model_VAE_Encoder        = str(raw_onnx_folder / 'VoxCPM2_AudioVAE_Encode.onnx')
onnx_model_Feat_Encoder_Cond  = str(raw_onnx_folder / 'VoxCPM2_Feat_Encoder_Cond.onnx')
onnx_model_Assemble           = {
                                    "voice_design":   str(raw_onnx_folder / 'VoxCPM2_Assemble_VoiceDesign.onnx'),
                                    "continuation":   str(raw_onnx_folder / 'VoxCPM2_Assemble_Continuation.onnx'),
                                    "reference_only": str(raw_onnx_folder / 'VoxCPM2_Assemble_ReferenceOnly.onnx'),
                                    "combined":       str(raw_onnx_folder / 'VoxCPM2_Assemble_Combined.onnx'),
}
onnx_model_Prefill            = str(raw_onnx_folder / 'VoxCPM2_Prefill.onnx')
onnx_model_Rotary_Mask_Decode = str(raw_onnx_folder / 'VoxCPM2_Rotary_Mask_Decode.onnx')
onnx_model_Main               = str(raw_onnx_folder / 'VoxCPM2_Main.onnx')
onnx_model_Feat_Decoder       = str(raw_onnx_folder / 'VoxCPM2_Feat_Decoder.onnx')
onnx_model_VAE_Decoder        = str(raw_onnx_folder / 'VoxCPM2_AudioVAE_Decode.onnx')
onnx_model_Metadata           = str(raw_onnx_folder / 'VoxCPM2_Metadata.onnx')
onnx_model_Concat             = str(raw_onnx_folder / 'VoxCPM2_Concat.onnx')


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        variance = x.float().square().mean(dim=axis, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        return g.op(
            "SimplifiedLayerNormalization",
            x,
            scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )


def simplified_layer_norm(x, scale, epsilon, axis=-1):
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


def _channel_score(weight, key, dims):
    weight = weight.float()
    absolute = weight.abs()
    if key == "rms":
        return weight.square().mean(dim=dims).sqrt()
    if key == "L4":
        return absolute.pow(4).mean(dim=dims).pow(0.25)
    if key == "std":
        return weight.std(dim=dims)
    if key == "absmean":
        return absolute.mean(dim=dims)
    pass
def _reorder_transformer_channels(
    layers,
    num_heads,
    num_key_value_heads,
    head_dim,
    qk_heads,
):
    """Permute both sides of optional quantization-sensitive boundaries."""
    with torch.no_grad():
        for layer in layers:
            if REORDER_DOWNPROJ_FOR_QUANT:
                down_weight = layer.mlp.down_proj.weight
                permutation = torch.argsort(
                    _channel_score(down_weight, REORDER_KEY, (0,))
                )
                intermediate_size = layer.mlp.down_proj.in_features
                gate_up_weight = layer.mlp.gate_up_proj.weight
                layer.mlp.gate_up_proj.weight.copy_(
                    torch.cat(
                        [
                            gate_up_weight[:intermediate_size][permutation],
                            gate_up_weight[intermediate_size:][permutation],
                        ],
                        dim=0,
                    )
                )
                layer.mlp.down_proj.weight.copy_(down_weight[:, permutation])

            if REORDER_OPROJ_FOR_QUANT:
                heads_per_kv = num_heads // num_key_value_heads
                output_weight = layer.self_attn.o_proj.weight
                output_by_head = output_weight.view(
                    output_weight.shape[0], num_heads, head_dim
                )
                permutations = []
                for kv_head in range(num_key_value_heads):
                    grouped = output_by_head[
                        :,
                        kv_head * heads_per_kv:(kv_head + 1) * heads_per_kv,
                    ]
                    permutations.append(
                        torch.argsort(
                            _channel_score(grouped, REORDER_KEY, (0, 1))
                        )
                    )

                reordered_output = output_by_head.clone()
                for head in range(num_heads):
                    reordered_output[:, head] = output_by_head[
                        :, head, permutations[head // heads_per_kv]
                    ]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                qkv_weight = layer.self_attn.qkv.weight
                qkv_by_head = qkv_weight.view(
                    -1, head_dim, qkv_weight.shape[1]
                ).clone()
                for kv_head, permutation in enumerate(permutations):
                    qkv_by_head[qk_heads + kv_head] = qkv_by_head[
                        qk_heads + kv_head
                    ][permutation]
                qkv_weight.copy_(qkv_by_head.reshape_as(qkv_weight))
                if layer.self_attn.qkv.bias is not None:
                    qkv_bias = layer.self_attn.qkv.bias
                    qkv_bias_by_head = qkv_bias.view(-1, head_dim).clone()
                    for kv_head, permutation in enumerate(permutations):
                        qkv_bias_by_head[qk_heads + kv_head] = qkv_bias_by_head[
                            qk_heads + kv_head
                        ][permutation]
                    qkv_bias.copy_(qkv_bias_by_head.reshape_as(qkv_bias))


def prepare_causal_conv1d(conv):
    torch.nn.utils.remove_weight_norm(conv)
    left_padding = conv._CausalConv1d__padding * 2 - conv._CausalConv1d__output_padding
    if left_padding:
        conv.register_buffer(
            "onnx_left_padding",
            torch.zeros(1, conv.in_channels, left_padding, dtype=conv.weight.dtype, device=conv.weight.device),
            persistent=False,
        )
    else:
        conv.onnx_left_padding = None


def causal_conv1d(x, conv):
    if conv.onnx_left_padding is not None:
        x = torch.cat([conv.onnx_left_padding, x], dim=-1)
    return F.conv1d(x, conv.weight, conv.bias, conv.stride, 0, conv.dilation, conv.groups)


def prepare_snake(snake):
    snake.register_buffer("inv_alpha", (snake.alpha + 1e-9).reciprocal(), persistent=False)


# ══════════════════════════════════════════════════════════════════════════════
# VAE Encoder Module
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_VAE_ENCODER(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_size = model.patch_size
        self.latent_dim = model.audio_vae.latent_dim
        self.patch_len = model.patch_size * math.prod(model.audio_vae.encoder_rates)
        self.input_resample_scale = float(MODEL_IN_SAMPLE_RATE / IN_SAMPLE_RATE)

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
            # Remove weight norm from init_conv and fold integer PCM normalization.
            prepare_causal_conv1d(self.init_conv)
            if "int" in IN_AUDIO_DTYPE.lower():
                self.init_conv.weight.mul_(1.0 / 32768.0)

            # Remove weight norm from all encoder block convolutions
            for block in self.enc_blocks:
                for unit_idx in range(3):  # 3 residual units per block
                    unit = block.block[unit_idx]
                    prepare_causal_conv1d(unit.block[1])  # dilated conv
                    prepare_causal_conv1d(unit.block[3])  # pointwise conv
                    # Precompute snake inv_alpha for each residual unit
                    prepare_snake(unit.block[0])
                    prepare_snake(unit.block[2])
                # Precompute snake inv_alpha before downsample
                prepare_snake(block.block[3])
                # Remove weight norm from downsample conv
                prepare_causal_conv1d(block.block[4])

            # Remove weight norm from fc_mu projection
            prepare_causal_conv1d(self.fc_mu)

        # Pre-allocate zero buffers for padding.
        self.register_buffer("pad_buffer", torch.zeros((1, 1, self.patch_len), dtype=torch.float32), persistent=False)
        self.register_buffer("pad_buffer_right", torch.zeros((1, 1, self.patch_len), dtype=torch.float32), persistent=False)

    @staticmethod
    def _snake(x, alpha, inv_alpha):
        """Snake activation: x + (1/α) * sin²(αx), with precomputed inv_alpha."""
        return x + inv_alpha * torch.sin(alpha * x).square()

    def _residual_unit(self, x, unit):
        """CausalResidualUnit: Snake → DilatedConv → Snake → PointwiseConv, then residual add."""
        residual = x
        x = self._snake(x, unit.block[0].alpha, unit.block[0].inv_alpha)
        x = causal_conv1d(x, unit.block[1])
        x = self._snake(x, unit.block[2].alpha, unit.block[2].inv_alpha)
        x = causal_conv1d(x, unit.block[3])
        return residual + x

    def forward(self, audio):
        audio = audio.float()
        if self.input_resample_scale != 1.0:
            audio = F.interpolate(
                audio,
                scale_factor=self.input_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )

        pad_len_left = self.patch_len - audio.shape[-1] % self.patch_len
        pad_buffer_left = self.pad_buffer[..., :pad_len_left]
        audio = torch.cat([pad_buffer_left, audio, self.pad_buffer_right], dim=-1)

        # Stage 0: Initial causal conv (1 → 128, k=7)
        x = causal_conv1d(audio, self.init_conv)

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
            x = causal_conv1d(x, block.block[4])

        # fc_mu projection (2048 → latent_dim=64, k=3)
        latent = causal_conv1d(x, self.fc_mu)

        latent = latent.view(self.latent_dim, -1, self.patch_size)
        return latent.permute(1, 2, 0)


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
        self.register_buffer(
            "rotate_indices",
            torch.cat([
                torch.arange(self.head_dim_half, self.head_dim, dtype=torch.int32),
                torch.arange(self.head_dim_half, dtype=torch.int32),
            ]),
            persistent=False,
        )

        self.rms_norm_epsilon = float(encoder.config.rms_norm_eps)
        self.register_buffer(
            "rms_scale",
            torch.full(
                (encoder.config.hidden_size,),
                encoder.config.hidden_size ** -0.5,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.q_len = model.patch_size + 1
        position_ids = torch.arange(self.q_len, dtype=torch.long)
        rope_emb_cos, rope_emb_sin = encoder.rope_emb(position_ids)
        rope_emb_sin[:, :encoder.rope_emb.dim // 2] *= -1.0
        self.register_buffer("rope_emb_cos", rope_emb_cos.view(1, self.q_len, 1, 1, -1), persistent=False)
        self.register_buffer("rope_emb_sin", rope_emb_sin.view(1, self.q_len, 1, 1, -1), persistent=False)

        self.register_buffer(
            "special_token",
            model.feat_encoder.special_token.detach().squeeze(0).half().float().view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "special_token_indices",
            torch.zeros(MAX_SEQ_LEN, dtype=torch.int32),
            persistent=False,
        )

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
        _reorder_transformer_channels(
            encoder.layers,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.qk_heads,
        )

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
        return simplified_layer_norm(x, self.rms_scale, self.rms_norm_epsilon)

    def rotate_half(self, x):
        return torch.index_select(x, -1, self.rotate_indices)

    def forward(self, audio_feat):
        # audio_feat: (batch, seq_len, patch_size, feat_dim)
        seq_len = audio_feat.shape[0]

        # === Feature Encoder: produces feat_embed for the LM ===
        hidden_states = self.model.feat_encoder.in_proj(audio_feat)
        special_tokens = torch.index_select(self.special_token, 0, self.special_token_indices[:seq_len])
        hidden_states = torch.cat([special_tokens, hidden_states], dim=1)

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
        last_patch = audio_feat[[-1]]  # (1, patch_size, feat_dim)
        feat_cond = self.model.feat_decoder.estimator.cond_proj(last_patch)  # (1, ps, cond_dim)
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
        self.register_buffer(
            "attention_mask",
            (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128,
            persistent=False,
        )

        # Precompute rotary embeddings
        position_ids = torch.arange(max_seq_len, dtype=torch.long)
        rope_emb_cos, rope_emb_sin = model.base_lm.rope_emb(position_ids)
        dim = rope_emb_cos.shape[-1]
        rope_emb_sin[:, :dim // 2] *= -1.0
        cos = rope_emb_cos.unsqueeze(1).unsqueeze(1)
        sin = rope_emb_sin.unsqueeze(1).unsqueeze(1)
        self.register_buffer("cos_rotary_pos_emb", cos.half().float(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half().float(), persistent=False)

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
        rotary_cos = self.cos_rotary_pos_emb[:ids_len]
        rotary_sin = self.sin_rotary_pos_emb[:ids_len]
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
        self.register_buffer("cos_rotary_pos_emb", cos.half().float(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half().float(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[kv_seq_len]
        rotary_sin = self.sin_rotary_pos_emb[kv_seq_len]
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Fused Main Transformer (Base LM + Residual LM)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_MAIN(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        self.model = model
        self.compute_in_f32 = COMPUTE_IN_F32
        self._replace_gelu_with_tanh_approximation(model)

        layer0 = model.base_lm.layers._modules['0']
        self.head_dim = layer0.self_attn.head_dim
        self.head_dim_half = self.head_dim // 2
        self.num_heads = layer0.self_attn.num_heads
        self.num_key_value_heads = layer0.self_attn.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.register_buffer(
            "rotate_indices",
            torch.cat([
                torch.arange(self.head_dim_half, self.head_dim, dtype=torch.int32),
                torch.arange(self.head_dim_half, dtype=torch.int32),
            ]),
            persistent=False,
        )

        self.base_layer_count = len(model.base_lm.layers)
        self.residual_layer_count = len(model.residual_lm.layers)
        self.total_layers = self.base_layer_count + self.residual_layer_count

        self.norm_factor = model.base_lm.config.hidden_size ** 0.5
        self.rms_norm_epsilon = float(model.base_lm.config.rms_norm_eps)
        self.register_buffer(
            "rms_scale",
            torch.full(
                (model.base_lm.config.hidden_size,),
                model.base_lm.config.hidden_size ** -0.5,
                dtype=torch.float32,
            ),
            persistent=False,
        )
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
        self.register_buffer("zero_buffer", torch.zeros([1, max_seq_len, hidden_size], dtype=torch.int8), persistent=False)

        self._fuse_weights(scale_factor)
        _reorder_transformer_channels(
            [*self.model.base_lm.layers, *self.model.residual_lm.layers],
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.qk_heads,
        )

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
        return simplified_layer_norm(x, self.rms_scale, self.rms_norm_epsilon)

    def _rotate_half(self, x):
        return torch.index_select(x, -1, self.rotate_indices)

    def forward(self, *all_inputs):
        feat_embed = all_inputs[-8]
        audio_seg1_start = all_inputs[-7]
        audio_seg1_end = all_inputs[-6]
        concat_text_len = all_inputs[-5]
        hidden_states = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask = all_inputs[-1]

        # F16-compute path shares one F16 mask across all layers (cast once, not per-layer).
        attention_mask_f16 = attention_mask.half() if (USE_F16_KV and not self.compute_in_f32) else None

        # === BASE LM LAYERS (with rotary) ===
        for i, layer in enumerate(self.model.base_lm.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(-1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk = qk * rotary_pos_emb_cos + self._rotate_half(qk) * rotary_pos_emb_sin
            if USE_F16_KV and not self.compute_in_f32:
                qk = qk.half()                             # earliest clean point (post-RoPE): Q and K share one F16 cast
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(1, 2, 0, 3)
            if USE_F16_KV:
                if self.compute_in_f32:
                    k = k.half()                           # storage only; upcast K/V at the matmul below
                v = v.half()
            k = k.permute(2, 1, 3, 0)
            v = v.transpose(0, 2)
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.total_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            if USE_F16_KV:
                if self.compute_in_f32:
                    attn = torch.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1)
                    attn = torch.matmul(attn, v.float())
                else:
                    attn = torch.softmax(torch.matmul(q, k) + attention_mask_f16, dim=-1)
                    attn = torch.matmul(attn, v).float()
            else:
                attn = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
                attn = torch.matmul(attn, v)
            attn = attn.permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
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
            if USE_F16_KV and not self.compute_in_f32:
                qkv = qkv.half()
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            # NO rotary for residual layers
            q, k = torch.split(qk, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(-1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(1, 2, 0, 3)
            if USE_F16_KV and self.compute_in_f32:
                k = k.half()                               # storage only; upcast K/V at the matmul below
                v = v.half()
            k = k.permute(2, 1, 3, 0)
            v = v.transpose(0, 2)
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.total_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            if USE_F16_KV:
                if self.compute_in_f32:
                    attn = torch.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1)
                    attn = torch.matmul(attn, v.float())
                else:
                    attn = torch.softmax(torch.matmul(q, k) + attention_mask_f16, dim=-1)
                    attn = torch.matmul(attn, v).float()
            else:
                attn = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
                attn = torch.matmul(attn, v)
            attn = attn.permute(2, 0, 1, 3).reshape(1, -1, layer.self_attn.o_proj.in_features)
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

        self.rms_norm_epsilon = float(
            model.feat_decoder.estimator.config.rms_norm_eps
        )
        self.register_buffer(
            "rms_scale",
            torch.full(
                (model.feat_decoder.estimator.config.hidden_size,),
                model.feat_decoder.estimator.config.hidden_size ** -0.5,
                dtype=torch.float32,
            ),
            persistent=False,
        )

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
        self.register_buffer("dt", active_dt.view(1, 1, -1), persistent=False)

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
            t_in_all = t_embeds + dt_embed  # (timesteps, hidden_dim)
        else:
            t_in_all = torch.empty(0, self.dit_hidden_dim, dtype=torch.float32)
        t_in_all = t_in_all.unsqueeze(0)
        self.register_buffer("t_in_all", torch.cat([t_in_all, t_in_all], dim=0).detach(), persistent=False)

        # VoxCPM2 DiT layout: [mu(2), t(1), cond(ps), x(ps)]
        self.q_len = 2 + 1 + self.patch_size + self.patch_size
        self.prefix_skip = 2 + 1 + self.patch_size

        # Pre-compute rotary
        position_ids = torch.arange(self.q_len, dtype=torch.long)
        rope_emb_cos, rope_emb_sin = decoder.rope_emb(position_ids)
        rope_emb_sin[:, :decoder.rope_emb.dim // 2] *= -1.0
        self.register_buffer("rope_emb_cos", rope_emb_cos.view(1, self.q_len, 1, 1, -1), persistent=False)
        self.register_buffer("rope_emb_sin", rope_emb_sin.view(1, self.q_len, 1, 1, -1), persistent=False)
        self.register_buffer(
            "rotate_indices",
            torch.cat([
                torch.arange(self.head_dim_half, self.head_dim, dtype=torch.int32),
                torch.arange(self.head_dim_half, dtype=torch.int32),
            ]),
            persistent=False,
        )

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
        _reorder_transformer_channels(
            decoder.layers,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.qk_heads,
        )

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
        return simplified_layer_norm(x, self.rms_scale, self.rms_norm_epsilon)

    def rotate_half(self, x):
        return torch.index_select(x, -1, self.rotate_indices)

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
        dot_product = (dphi_dt_positive * cfg_dphi_dt).sum((1, 2), keepdim=True)
        squared_norm = cfg_dphi_dt.square().sum((1, 2), keepdim=True)
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
        return random


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
        self.output_resample_scale = float(OUT_SAMPLE_RATE / MODEL_OUT_SAMPLE_RATE)

        decoder = model.audio_vae.decoder
        sr_idx = int(torch.bucketize(
            torch.tensor([MODEL_OUT_SAMPLE_RATE], dtype=decoder.sr_bin_boundaries.dtype),
            decoder.sr_bin_boundaries,
        ).item())

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
            prepare_causal_conv1d(self.init_conv_dw)
            prepare_causal_conv1d(self.init_conv_pw)

            # Precompute inv_alpha for final snake
            prepare_snake(self.final_snake)

            # Remove weight norm from final conv
            prepare_causal_conv1d(self.final_conv)

            # Remove weight norm and precompute inv_alpha for all decoder blocks
            for block in self.dec_blocks:
                # block.block[0] = Snake1d before upsample
                prepare_snake(block.block[0])
                # block.block[1] = WNCausalTransposeConv1d (strided upsample)
                torch.nn.utils.remove_weight_norm(block.block[1])
                # block.block[2..4] = 3 CausalResidualUnits (dilation=1,3,9)
                for unit_idx in range(2, 5):
                    unit = block.block[unit_idx]
                    # unit.block[0] = Snake1d, unit.block[1] = WNCausalConv1d (dilated)
                    # unit.block[2] = Snake1d, unit.block[3] = WNCausalConv1d (pointwise)
                    prepare_snake(unit.block[0])
                    prepare_causal_conv1d(unit.block[1])
                    prepare_snake(unit.block[2])
                    prepare_causal_conv1d(unit.block[3])

            # Select fixed sample-rate conditioning and prepare optional output convolutions.
            for sr_cond_layer in self.sr_cond_layers:
                sr_cond_layer.register_buffer(
                    "fixed_scale",
                    sr_cond_layer.scale_embed.weight[sr_idx].view(1, -1, 1).clone(),
                    persistent=False,
                )
                sr_cond_layer.register_buffer(
                    "fixed_bias",
                    sr_cond_layer.bias_embed.weight[sr_idx].view(1, -1, 1).clone(),
                    persistent=False,
                )
                del sr_cond_layer.scale_embed, sr_cond_layer.bias_embed
                if hasattr(sr_cond_layer, 'out_layer') and not isinstance(sr_cond_layer.out_layer, torch.nn.Identity):
                    # out_layer = Sequential(Snake1d, WNCausalConv1d)
                    prepare_snake(sr_cond_layer.out_layer[0])
                    prepare_causal_conv1d(sr_cond_layer.out_layer[1])

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
        x = causal_conv1d(x, unit.block[1])
        x = self._snake(x, unit.block[2].alpha, unit.block[2].inv_alpha)
        x = causal_conv1d(x, unit.block[3])
        return residual + x

    def _decoder_block(self, x, block):
        """CausalDecoderBlock: Snake → TransposeConv(upsample) → 3× residual units."""
        x = self._snake(x, block.block[0].alpha, block.block[0].inv_alpha)
        x = block.block[1](x)
        x = self._residual_unit(x, block.block[2])    # dilation=1
        x = self._residual_unit(x, block.block[3])    # dilation=3
        x = self._residual_unit(x, block.block[4])    # dilation=9
        return x

    def _apply_sr_cond(self, x, sr_cond_layer):
        """SampleRateConditionLayer (scale_bias): x * scale + bias, then out_layer."""
        x = x * sr_cond_layer.fixed_scale + sr_cond_layer.fixed_bias
        if hasattr(sr_cond_layer, 'out_layer') and not isinstance(sr_cond_layer.out_layer, torch.nn.Identity):
            x = self._snake(x, sr_cond_layer.out_layer[0].alpha, sr_cond_layer.out_layer[0].inv_alpha)
            x = causal_conv1d(x, sr_cond_layer.out_layer[1])
        return x

    def forward(self, latent_patches):
        x = latent_patches.transpose(1, 2)

        # Stage 0: Initial depthwise conv (latent_dim → latent_dim, k=7, grouped)
        x = causal_conv1d(x, self.init_conv_dw)
        # Stage 1: Pointwise conv (latent_dim → decoder_dim, k=1)
        x = causal_conv1d(x, self.init_conv_pw)

        # Stages 2-N: Decoder blocks with sample-rate conditioning
        # Each block: sr_cond(scale_bias) → Snake → TransposeConv(upsample) → 3× ResUnits
        for block, sr_cond_layer in zip(self.dec_blocks, self.sr_cond_layers):
            x = self._apply_sr_cond(x, sr_cond_layer)
            x = self._decoder_block(x, block)

        # Final: Snake → Conv(→1ch) → Tanh → public-rate waveform.
        x = self._snake(x, self.final_snake.alpha, self.final_snake.inv_alpha)
        x = causal_conv1d(x, self.final_conv)
        generated_wav = torch.tanh(x)
        if self.output_resample_scale != 1.0:
            generated_wav = F.interpolate(
                generated_wav,
                scale_factor=self.output_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            generated_wav = (generated_wav * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16)
        elif "32" in OUT_AUDIO_DTYPE:
            generated_wav = generated_wav.float()
        else:
            generated_wav = generated_wav.half()

        return generated_wav, torch._shape_as_tensor(generated_wav)[-1].unsqueeze(0)


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
        self.register_buffer("zero_buffer_3d", torch.zeros((max_seq_len, patch_size, latent_dim), dtype=torch.int8), persistent=False)

    def forward(self, text_ids):
        text_len = text_ids.shape[1]
        text_token = text_ids
        audio_feat = self.zero_buffer_3d[:text_len].float()
        concat_text_len = text_len.unsqueeze(0)
        ids_len = concat_text_len
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
        self.register_buffer("zero_buffer_2d", torch.zeros((1, max_seq_len), dtype=torch.int8), persistent=False)
        self.register_buffer("zero_buffer_3d", torch.zeros((max_seq_len, patch_size, latent_dim), dtype=torch.int8), persistent=False)

    def forward(self, text_ids, prompt_audio_feat):
        text_len = text_ids.shape[1]
        prompt_len = prompt_audio_feat.shape[0]
        prompt_zeros = self.zero_buffer_2d[:, :prompt_len].int()
        text_token = torch.cat([text_ids, prompt_zeros], dim=1)
        text_pad = self.zero_buffer_3d[:text_len].float()
        audio_feat = torch.cat([text_pad, prompt_audio_feat], dim=0)
        concat_text_len = text_len.unsqueeze(0)
        ids_len = (text_len + prompt_len).unsqueeze(0)
        return text_token, audio_feat, self.audio_seg1_start, self.audio_seg1_end, concat_text_len, ids_len


class VOXCPM2_ASSEMBLE_REFERENCE_ONLY(torch.nn.Module):
    def __init__(self, patch_size, latent_dim, max_seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.register_buffer("zero_frame", torch.zeros((1, patch_size, latent_dim), dtype=torch.float32))
        self.register_buffer("ref_start_token", torch.tensor([[103]], dtype=torch.int32))
        self.register_buffer("ref_end_token", torch.tensor([[104]], dtype=torch.int32))
        # len=1 static buffer (no cast needed)
        self.register_buffer("audio_seg1_start", torch.ones(1, dtype=torch.int64))
        # Pre-allocated int8 buffers (slice+cast at runtime)
        self.register_buffer("zero_buffer_2d", torch.zeros((1, max_seq_len), dtype=torch.int8), persistent=False)
        self.register_buffer("zero_buffer_3d", torch.zeros((max_seq_len, patch_size, latent_dim), dtype=torch.int8), persistent=False)

    def forward(self, text_ids, ref_audio_feat):
        text_len = text_ids.shape[1]
        ref_len = ref_audio_feat.shape[0]
        ref_zeros = self.zero_buffer_2d[:, :ref_len].int()
        text_token = torch.cat([self.ref_start_token, ref_zeros, self.ref_end_token, text_ids], dim=1)

        text_pad = self.zero_buffer_3d[:text_len].float()
        audio_feat = torch.cat([self.zero_frame, ref_audio_feat, self.zero_frame, text_pad], dim=0)

        audio_seg1_end_scalar = ref_len + 1
        audio_seg1_end = audio_seg1_end_scalar.unsqueeze(0)
        concat_text_len = (audio_seg1_end_scalar + text_len + 1).unsqueeze(0)
        ids_len = concat_text_len
        return text_token, audio_feat, self.audio_seg1_start, audio_seg1_end, concat_text_len, ids_len


class VOXCPM2_ASSEMBLE_COMBINED(torch.nn.Module):
    def __init__(self, patch_size, latent_dim, max_seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.register_buffer("zero_frame", torch.zeros((1, patch_size, latent_dim)))
        self.register_buffer("ref_start_token", torch.tensor([[103]], dtype=torch.int32))
        self.register_buffer("ref_end_token", torch.tensor([[104]], dtype=torch.int32))
        # len=1 static buffer (no cast needed)
        self.register_buffer("audio_seg1_start", torch.ones(1, dtype=torch.int64))
        # Pre-allocated int8 buffers (slice+cast at runtime)
        self.register_buffer("zero_buffer_2d", torch.zeros((1, max_seq_len), dtype=torch.int8), persistent=False)
        self.register_buffer("zero_buffer_3d", torch.zeros((max_seq_len, patch_size, latent_dim), dtype=torch.int8), persistent=False)

    def forward(self, text_ids, ref_audio_feat, prompt_audio_feat):
        text_len = text_ids.shape[1]
        ref_len = ref_audio_feat.shape[0]
        prompt_len = prompt_audio_feat.shape[0]
        ref_zeros = self.zero_buffer_2d[:, :ref_len].int()
        prompt_zeros = self.zero_buffer_2d[:, :prompt_len].int()
        text_token = torch.cat([self.ref_start_token, ref_zeros, self.ref_end_token, text_ids, prompt_zeros], dim=1)

        text_pad = self.zero_buffer_3d[:text_len].float()
        audio_feat = torch.cat([self.zero_frame, ref_audio_feat, self.zero_frame, text_pad, prompt_audio_feat], dim=0)

        audio_seg1_end_scalar = ref_len + 1
        concat_text_len_scalar = audio_seg1_end_scalar + text_len + 1
        audio_seg1_end = audio_seg1_end_scalar.unsqueeze(0)
        concat_text_len = concat_text_len_scalar.unsqueeze(0)
        ids_len = (concat_text_len_scalar + prompt_len).unsqueeze(0)
        return text_token, audio_feat, self.audio_seg1_start, audio_seg1_end, concat_text_len, ids_len


# ══════════════════════════════════════════════════════════════════════════════
# Concatenation Module (Streaming only)
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_CONCAT(torch.nn.Module):
    def forward(self, embed_0, embed_1):
        concat_embed = torch.cat([embed_0, embed_1], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Graph components
# ══════════════════════════════════════════════════════════════════════════════
class VOXCPM2_PREFILL_INPUTS(torch.nn.Module):
    """Run v2 Prefill and own reusable empty KV/history constants."""

    def __init__(
        self,
        prefill,
        base_num_kv_heads,
        base_head_dim,
        residual_num_kv_heads,
        residual_head_dim,
        kv_dtype,
    ):
        super().__init__()
        self.prefill = prefill
        self.register_buffer(
            "empty_base_key",
            torch.zeros((base_num_kv_heads, 1, base_head_dim, 0), dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_base_value",
            torch.zeros((base_num_kv_heads, 1, 0, base_head_dim), dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_residual_key",
            torch.zeros(
                (residual_num_kv_heads, 1, residual_head_dim, 0),
                dtype=kv_dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "empty_residual_value",
            torch.zeros(
                (residual_num_kv_heads, 1, 0, residual_head_dim),
                dtype=kv_dtype,
            ),
            persistent=False,
        )
        self.register_buffer(
            "zero_history",
            torch.zeros((1,), dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        text_ids,
        ids_len,
        feat_embed,
        audio_seg1_start,
        audio_seg1_end,
        concat_text_len,
    ):
        outputs = self.prefill(
            text_ids,
            ids_len,
            feat_embed,
            audio_seg1_start,
            audio_seg1_end,
            concat_text_len,
            self.zero_history,
        )
        zero_dependency = ids_len * 0
        return (
            *outputs,
            self.empty_base_key + zero_dependency.to(self.empty_base_key.dtype),
            self.empty_base_value + zero_dependency.to(self.empty_base_value.dtype),
            self.empty_residual_key
            + zero_dependency.to(self.empty_residual_key.dtype),
            self.empty_residual_value
            + zero_dependency.to(self.empty_residual_value.dtype),
        )


class VOXCPM2_DECODE_INPUTS(torch.nn.Module):
    """Own decode-position RoPE, zero segment controls, mask, and length."""

    def __init__(self, model, max_seq_len):
        super().__init__()
        self.rotary = VOXCPM2_ROTARY_MASK_DECODE(model, max_seq_len)
        self.register_buffer(
            "zero_segment",
            torch.zeros((1,), dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "zero_attention_mask",
            torch.zeros((1, 1, 1, 1), dtype=torch.float32),
            persistent=False,
        )

    def forward(self, kv_seq_len):
        rotary_cos, rotary_sin, kv_seq_len_out = self.rotary(kv_seq_len)
        zero_dependency = kv_seq_len * 0
        zero_segment = self.zero_segment + zero_dependency
        return (
            rotary_cos,
            rotary_sin,
            zero_segment,
            zero_segment,
            zero_segment,
            self.zero_attention_mask
            + zero_dependency.to(self.zero_attention_mask.dtype),
            kv_seq_len_out,
        )


class VOXCPM2_MAIN_CORE(torch.nn.Module):
    """Expose v2 Main state/conditioning while removing its internal RNG output."""

    def __init__(self, main, kv_tensor_count):
        super().__init__()
        self.main = main
        self.kv_tensor_count = kv_tensor_count

    def forward(self, *inputs):
        outputs = self.main(*inputs)
        return (
            *outputs[:self.kv_tensor_count],
            outputs[-2],
            outputs[-1],
        )


class VOXCPM2_LATENT_ACCUMULATOR(torch.nn.Module):
    def forward(self, generated_latents, current_latent):
        return torch.cat((generated_latents, current_latent), dim=1)


class VOXCPM2_VAE_DECODE_STREAM(torch.nn.Module):
    def __init__(self, vae_decoder):
        super().__init__()
        self.vae_decoder = vae_decoder

    def forward(self, previous_latent, current_latent):
        return self.vae_decoder(
            torch.cat((previous_latent, current_latent), dim=1)
        )


# Metadata helpers — export the pipeline geometry once so inference never has to
# hand-duplicate the fixed-at-export constants (mirrors the ASR repo).
def build_model_metadata(*sections):
    """Flatten metadata sections into a ``{str: str}`` dict for ``metadata_props``."""
    metadata = {}
    for section in sections:
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                metadata[str(key)] = "1" if value else "0"
            elif isinstance(value, (list, tuple)):
                metadata[str(key)] = ",".join(str(item) for item in value)
            else:
                metadata[str(key)] = str(value)
    return metadata


def write_onnx_metadata(onnx_path, metadata):
    """Add/overwrite ``metadata_props`` in place, leaving any external-weight sidecar untouched."""
    import onnx

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in onnx_model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            onnx_model.metadata_props.add(key=key, value=value)
    onnx.save(onnx_model, onnx_path)


def replace_onnx_metadata(onnx_path, metadata):
    """Replace all metadata properties without loading external weights."""
    import onnx

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    del onnx_model.metadata_props[:]
    for key, value in metadata.items():
        onnx_model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(onnx_model, onnx_path)


class METADATA_CARRIER(torch.nn.Module):
    """Tiny identity graph that carries the static package contract."""

    def forward(self, marker):
        return marker


GRAPH_LAYOUT = "voxcpm2_prefill_decode_v1"
COMPONENT_FILES = {
    "feat_encoder": "VoxCPM2_Component_FeatEncoderCond.onnx",
    "prefill_inputs": "VoxCPM2_Component_PrefillInputs.onnx",
    "main_core": "VoxCPM2_Component_MainCore.onnx",
    "feat_decoder": "VoxCPM2_Component_FeatDecoder.onnx",
    "decode_inputs": "VoxCPM2_Component_DecodeInputs.onnx",
    "latent_accumulator": "VoxCPM2_Component_LatentAccumulator.onnx",
}
MODEL_FILES = {
    "vae_encoder": "VoxCPM2_AudioVAE_Encode.onnx",
    "decode_step": "VoxCPM2_DecodeStep.onnx",
    "vae_decoder": "VoxCPM2_AudioVAE_Decode.onnx",
    "vae_decoder_stream": "VoxCPM2_AudioVAE_Decode_Stream.onnx",
    "metadata": "VoxCPM2_Metadata.onnx",
}
PREFILL_FILES = {
    "voice_design": "VoxCPM2_MainPrefill_VoiceDesign.onnx",
    "continuation": "VoxCPM2_MainPrefill_Continuation.onnx",
    "reference_only": "VoxCPM2_MainPrefill_ReferenceOnly.onnx",
    "combined": "VoxCPM2_MainPrefill_Combined.onnx",
}
ASSEMBLE_COMPONENT_FILES = {
    "voice_design": "VoxCPM2_Assemble_VoiceDesign.onnx",
    "continuation": "VoxCPM2_Assemble_Continuation.onnx",
    "reference_only": "VoxCPM2_Assemble_ReferenceOnly.onnx",
    "combined": "VoxCPM2_Assemble_Combined.onnx",
}
MODE_PUBLIC_INPUTS = {
    "voice_design": ("text_ids",),
    "continuation": ("text_ids", "prompt_audio_feat"),
    "reference_only": ("text_ids", "ref_audio_feat"),
    "combined": ("text_ids", "ref_audio_feat", "prompt_audio_feat"),
}


def _export_component(
    module,
    inputs,
    path,
    input_names,
    output_names,
    dynamic_axes=None,
):
    module.eval()
    torch.onnx.export(
        module,
        inputs,
        str(path),
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
        external_data=True,
    )


def _kv_layout(
    base_key,
    base_value,
    residual_key,
    residual_value,
    base_layers,
    residual_layers,
):
    tensors = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    total_layers = base_layers + residual_layers
    for state_name, sequence_axis in (("key", 3), ("value", 2)):
        for layer_index in range(total_layers):
            is_base = layer_index < base_layers
            tensor = (
                base_key if state_name == "key" else base_value
            ) if is_base else (
                residual_key if state_name == "key" else residual_value
            )
            input_name = f"in_{state_name}_{layer_index}"
            output_name = f"out_{state_name}_{layer_index}"
            tensors.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            dynamic_axes[input_name] = {sequence_axis: "history_len"}
            dynamic_axes[output_name] = {sequence_axis: "kv_seq_len"}
    return tensors, input_names, output_names, dynamic_axes


def _compose_graphs(
    component_paths,
    assemble_paths,
    kv_in_names,
    kv_out_names,
    base_layers,
    output_folder,
):
    total_layers = len(kv_in_names) // 2
    empty_connections = {}
    for layer_index in range(total_layers):
        family = "base" if layer_index < base_layers else "residual"
        empty_connections[f"in_key_{layer_index}"] = f"empty_{family}_key"
        empty_connections[f"in_value_{layer_index}"] = f"empty_{family}_value"
    empty_connections.update(
        {
            "feat_embed": "feat_embed_audio",
            "audio_seg1_start": "audio_seg1_start",
            "audio_seg1_end": "audio_seg1_end",
            "concat_text_len": "concat_text_len",
            "hidden_states": "combined_embed",
            "rotary_cos": "rotary_cos",
            "rotary_sin": "rotary_sin",
            "attention_mask": "attention_mask",
        }
    )

    for mode, output_file in PREFILL_FILES.items():
        output_path = output_folder / output_file
        compose_graphs(
            [
                GraphComponent(assemble_paths[mode], "assemble/", {}),
                GraphComponent(
                    component_paths["feat_encoder"],
                    "feat_encoder/",
                    {"audio_feat": "audio_feat"},
                ),
                GraphComponent(
                    component_paths["prefill_inputs"],
                    "prefill/",
                    {
                        "text_ids": "text_token",
                        "ids_len": "ids_len",
                        "feat_embed": "feat_embed",
                        "audio_seg1_start": "audio_seg1_start",
                        "audio_seg1_end": "audio_seg1_end",
                        "concat_text_len": "concat_text_len",
                    },
                ),
                GraphComponent(
                    component_paths["main_core"],
                    "main/",
                    empty_connections,
                ),
                GraphComponent(
                    component_paths["feat_decoder"],
                    "feat_decoder/",
                    {
                        "dit_hidden": "dit_hidden",
                        "feat_cond": "feat_cond",
                    },
                    {"random": "noise"},
                ),
            ],
            output_path,
            (*kv_out_names, "latent_pred", "stop_flag", "kv_seq_len"),
            graph_name=f"voxcpm2_main_prefill_{mode}",
            input_names=(
                *MODE_PUBLIC_INPUTS[mode],
                "noise",
                "cfg_value",
                "cfg_value_minus",
            ),
        )

    decode_main_connections = {
        "feat_embed": "feat_embed",
        "audio_seg1_start": "zero_audio_seg1_start",
        "audio_seg1_end": "zero_audio_seg1_end",
        "concat_text_len": "zero_concat_text_len",
        "hidden_states": "feat_embed",
        "rotary_cos": "rotary_cos",
        "rotary_sin": "rotary_sin",
        "attention_mask": "zero_attention_mask",
    }
    compose_graphs(
        [
            GraphComponent(
                component_paths["feat_encoder"],
                "feat_encoder/",
                {},
                {"audio_feat": "previous_latent"},
            ),
            GraphComponent(
                component_paths["decode_inputs"],
                "decode_inputs/",
                {},
            ),
            GraphComponent(
                component_paths["main_core"],
                "main/",
                decode_main_connections,
            ),
            GraphComponent(
                component_paths["feat_decoder"],
                "feat_decoder/",
                {
                    "dit_hidden": "dit_hidden",
                    "feat_cond": "feat_cond",
                },
                {"random": "noise"},
            ),
            GraphComponent(
                component_paths["latent_accumulator"],
                "accumulator/",
                {"current_latent": "latent_pred"},
                {"generated_latents_in": "generated_latents_in"},
            ),
        ],
        output_folder / MODEL_FILES["decode_step"],
        (
            *kv_out_names,
            "latent_pred",
            "stop_flag",
            "kv_seq_len_out",
            "generated_latents",
        ),
        graph_name="voxcpm2_decode_step",
        input_names=(
            *kv_in_names,
            "previous_latent",
            "kv_seq_len",
            "noise",
            "cfg_value",
            "cfg_value_minus",
            "generated_latents_in",
        ),
    )

    for path in {*component_paths.values(), *assemble_paths.values()}:
        path.unlink()
        path.with_name(path.name + ".data").unlink(missing_ok=True)


def _finalize_shared_metadata(metadata, output_folder):
    for path in sorted(output_folder.glob("*.onnx")):
        replace_onnx_metadata(path, metadata)
    return dict(metadata)


def _install_package_folder(stage_folder, final_folder):
    backup_folder = None
    if final_folder.exists():
        backup_folder = final_folder.parent / (
            f".{final_folder.name}.expanded-backup-{uuid.uuid4().hex}"
        )
        os.replace(final_folder, backup_folder)
    try:
        os.replace(stage_folder, final_folder)
    except Exception:
        if backup_folder is not None and backup_folder.exists():
            os.replace(backup_folder, final_folder)
        pass
    if backup_folder is not None:
        shutil.rmtree(backup_folder)


def export_voxcpm2():
    print("VoxCPM2 export start ...")
    stage_folder = script_dir / ".VoxCPM_ONNX.stage"
    if raw_onnx_folder.exists():
        shutil.rmtree(raw_onnx_folder)
    raw_onnx_folder.mkdir(parents=True)
    if stage_folder.exists():
        shutil.rmtree(stage_folder)

    with torch.inference_mode():
        model_dir = Path(path_voxcpm2).expanduser().resolve()
        model = VoxCPM2Model.from_local(
            str(model_dir),
            optimize=False,
            device="cpu",
        )
        model = model.to(torch.float32).to("cpu").eval()
        with open(model_dir / "config.json", "r", encoding="utf-8") as handle:
            config = json.load(handle)
        audio_vae_config = config["audio_vae_config"]

        base_layers = len(model.base_lm.layers)
        residual_layers = len(model.residual_lm.layers)
        total_layers = base_layers + residual_layers
        kv_tensor_count = total_layers * 2
        base_layer = model.base_lm.layers[0]
        residual_layer = model.residual_lm.layers[0]
        base_head_dim = int(base_layer.self_attn.head_dim)
        base_num_kv_heads = int(base_layer.self_attn.num_key_value_heads)
        residual_head_dim = int(residual_layer.self_attn.head_dim)
        residual_num_kv_heads = int(
            residual_layer.self_attn.num_key_value_heads
        )
        hidden_size = int(model.base_lm.embed_tokens.embedding_dim)
        feat_hidden_size = int(model.enc_to_lm_proj.out_features)
        patch_size = int(model.patch_size)
        feat_dim = int(config["feat_dim"])
        latent_dim = int(model.audio_vae.latent_dim)
        feat_in_channels = int(model.feat_decoder.in_channels)
        dit_hidden_dim = int(model.feat_decoder.estimator.config.hidden_size)
        cond_proj_out = int(model.feat_decoder.estimator.cond_proj.out_features)
        encode_patch_len = patch_size * math.prod(audio_vae_config["encoder_rates"])
        model_samples_per_vae_frame = math.prod(audio_vae_config["decoder_rates"])
        streaming_crop_samples = (
            patch_size * model_samples_per_vae_frame
            * OUT_SAMPLE_RATE / MODEL_OUT_SAMPLE_RATE
        )
        streaming_crop_samples = int(streaming_crop_samples)
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32

        vae_encoder = VOXCPM2_VAE_ENCODER(model)
        feat_encoder = VOXCPM2_FEAT_ENCODER_COND(model)
        prefill_inputs = VOXCPM2_PREFILL_INPUTS(
            VOXCPM2_PREFILL(model, MAX_SEQ_LEN),
            base_num_kv_heads,
            base_head_dim,
            residual_num_kv_heads,
            residual_head_dim,
            kv_dtype,
        )
        decode_inputs = VOXCPM2_DECODE_INPUTS(model, MAX_SEQ_LEN)
        main_core = VOXCPM2_MAIN_CORE(
            VOXCPM2_MAIN(model, MAX_SEQ_LEN),
            kv_tensor_count,
        )
        feat_decoder = VOXCPM2_FEAT_DECODER(model, FIXED_TIMESTEPS)
        vae_decoder = VOXCPM2_VAE_DECODE(model)
        vae_decoder_stream = VOXCPM2_VAE_DECODE_STREAM(vae_decoder)
        latent_accumulator = VOXCPM2_LATENT_ACCUMULATOR()

        one_latent = torch.zeros((1, patch_size, latent_dim), dtype=torch.float32)
        two_latents = torch.zeros((1, patch_size * 2, latent_dim), dtype=torch.float32)
        _, one_length = vae_decoder(one_latent)
        _, two_length = vae_decoder(two_latents)
        del one_latent, two_latents

        metadata = build_model_metadata(
            {
                "graph_layout": GRAPH_LAYOUT,
                "model_file_name_vae_encoder": MODEL_FILES["vae_encoder"],
                "model_file_name_decode_step": MODEL_FILES["decode_step"],
                "model_file_name_vae_decoder": MODEL_FILES["vae_decoder"],
                "model_file_name_vae_decoder_stream": MODEL_FILES["vae_decoder_stream"],
                "model_file_name_metadata": MODEL_FILES["metadata"],
                **{
                    f"model_file_name_main_prefill_{mode}": file_name
                    for mode, file_name in PREFILL_FILES.items()
                },
                "shared_initializer_model_file": SHARED_MODEL_NAME,
                "shared_initializer_data_file": SHARED_DATA_NAME,
                "in_sample_rate": IN_SAMPLE_RATE,
                "out_sample_rate": OUT_SAMPLE_RATE,
                "max_seq_len": MAX_SEQ_LEN,
                "stop_token_ids": sorted(STOP_TOKEN),
                "audio_start_token_id": 101,
            },
        )

        component_paths = {
            name: raw_onnx_folder / file_name
            for name, file_name in COMPONENT_FILES.items()
        }
        assemble_paths = {
            mode: raw_onnx_folder / file_name
            for mode, file_name in ASSEMBLE_COMPONENT_FILES.items()
        }

        prompt_audio = torch.zeros(
            (1, 1, encode_patch_len * 2),
            dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()],
        )
        _export_component(
            vae_encoder,
            (prompt_audio,),
            raw_onnx_folder / MODEL_FILES["vae_encoder"],
            ("audio",),
            ("audio_feat",),
            {
                "audio": {2: "audio_samples"},
                "audio_feat": {0: "audio_feat_len"},
            },
        )
        del prompt_audio

        audio_feat = torch.zeros((10, patch_size, feat_dim), dtype=torch.float32)
        _export_component(
            feat_encoder,
            (audio_feat,),
            component_paths["feat_encoder"],
            ("audio_feat",),
            ("feat_embed", "feat_cond"),
            {
                "audio_feat": {0: "audio_feat_len"},
                "feat_embed": {1: "audio_feat_len"},
            },
        )
        del audio_feat

        text_ids = torch.zeros((1, 15), dtype=torch.int32)
        assemble_output_names = (
            "text_token",
            "audio_feat",
            "audio_seg1_start",
            "audio_seg1_end",
            "concat_text_len",
            "ids_len",
        )
        assemble_dynamic_outputs = {
            "text_token": {1: "total_len"},
            "audio_feat": {0: "total_len"},
        }
        prompt_feat = torch.zeros((8, patch_size, latent_dim), dtype=torch.float32)
        reference_feat = torch.zeros((5, patch_size, latent_dim), dtype=torch.float32)
        assemble_specs = {
            "voice_design": (
                VOXCPM2_ASSEMBLE_VOICE_DESIGN(patch_size, latent_dim, MAX_SEQ_LEN),
                (text_ids,),
                ("text_ids",),
                {"text_ids": {1: "text_len"}, **assemble_dynamic_outputs},
            ),
            "continuation": (
                VOXCPM2_ASSEMBLE_CONTINUATION(patch_size, latent_dim, MAX_SEQ_LEN),
                (text_ids, prompt_feat),
                ("text_ids", "prompt_audio_feat"),
                {
                    "text_ids": {1: "text_len"},
                    "prompt_audio_feat": {0: "prompt_len"},
                    **assemble_dynamic_outputs,
                },
            ),
            "reference_only": (
                VOXCPM2_ASSEMBLE_REFERENCE_ONLY(patch_size, latent_dim, MAX_SEQ_LEN),
                (text_ids, reference_feat),
                ("text_ids", "ref_audio_feat"),
                {
                    "text_ids": {1: "text_len"},
                    "ref_audio_feat": {0: "ref_len"},
                    **assemble_dynamic_outputs,
                },
            ),
            "combined": (
                VOXCPM2_ASSEMBLE_COMBINED(patch_size, latent_dim, MAX_SEQ_LEN),
                (text_ids, reference_feat, prompt_feat),
                ("text_ids", "ref_audio_feat", "prompt_audio_feat"),
                {
                    "text_ids": {1: "text_len"},
                    "ref_audio_feat": {0: "ref_len"},
                    "prompt_audio_feat": {0: "prompt_len"},
                    **assemble_dynamic_outputs,
                },
            ),
        }
        for mode, (module, inputs, input_names, dynamic_axes) in assemble_specs.items():
            _export_component(
                module,
                inputs,
                assemble_paths[mode],
                input_names,
                assemble_output_names,
                dynamic_axes,
            )
        del text_ids, prompt_feat, reference_feat, assemble_specs

        prefill_seq_len = 25
        prefill_audio_seg1_len = 5
        prefill_text_after = 10
        prefill_text_ids = torch.zeros((1, prefill_seq_len), dtype=torch.int32)
        ids_len = torch.tensor([prefill_seq_len], dtype=torch.int64)
        feat_embed_dummy = torch.zeros(
            (1, prefill_seq_len, feat_hidden_size),
            dtype=torch.float32,
        )
        audio_seg1_start = torch.tensor([1], dtype=torch.int64)
        audio_seg1_end = torch.tensor(
            [1 + prefill_audio_seg1_len],
            dtype=torch.int64,
        )
        concat_text_len = torch.tensor(
            [1 + prefill_audio_seg1_len + prefill_text_after],
            dtype=torch.int64,
        )
        _export_component(
            prefill_inputs,
            (
                prefill_text_ids,
                ids_len,
                feat_embed_dummy,
                audio_seg1_start,
                audio_seg1_end,
                concat_text_len,
            ),
            component_paths["prefill_inputs"],
            (
                "text_ids",
                "ids_len",
                "feat_embed",
                "audio_seg1_start",
                "audio_seg1_end",
                "concat_text_len",
            ),
            (
                "combined_embed",
                "feat_embed_audio",
                "rotary_cos",
                "rotary_sin",
                "attention_mask",
                "kv_seq_len",
                "empty_base_key",
                "empty_base_value",
                "empty_residual_key",
                "empty_residual_value",
            ),
            {
                "text_ids": {1: "seq_len"},
                "feat_embed": {1: "seq_len"},
                "combined_embed": {1: "seq_len"},
                "feat_embed_audio": {1: "audio_feat_len"},
                "rotary_cos": {0: "seq_len"},
                "rotary_sin": {0: "seq_len"},
                "attention_mask": {2: "seq_len", 3: "seq_len"},
            },
        )
        del prefill_text_ids, feat_embed_dummy

        history_length = 0
        base_key = torch.zeros(
            (base_num_kv_heads, 1, base_head_dim, history_length),
            dtype=kv_dtype,
        )
        base_value = torch.zeros(
            (base_num_kv_heads, 1, history_length, base_head_dim),
            dtype=kv_dtype,
        )
        residual_key = torch.zeros(
            (residual_num_kv_heads, 1, residual_head_dim, history_length),
            dtype=kv_dtype,
        )
        residual_value = torch.zeros(
            (residual_num_kv_heads, 1, history_length, residual_head_dim),
            dtype=kv_dtype,
        )
        kv_tensors, kv_in_names, kv_out_names, kv_dynamic_axes = _kv_layout(
            base_key,
            base_value,
            residual_key,
            residual_value,
            base_layers,
            residual_layers,
        )
        main_feat_embed = torch.zeros(
            (
                1,
                int(audio_seg1_end.item() - audio_seg1_start.item())
                + prefill_seq_len
                - int(concat_text_len.item()),
                feat_hidden_size,
            ),
            dtype=torch.float32,
        )
        hidden_states = torch.ones(
            (1, prefill_seq_len, hidden_size),
            dtype=torch.float32,
        )
        rotary_cos = torch.zeros(
            (prefill_seq_len, 1, 1, base_head_dim),
            dtype=torch.float32,
        )
        rotary_sin = torch.zeros_like(rotary_cos)
        attention_mask = torch.zeros(
            (1, 1, prefill_seq_len, prefill_seq_len),
            dtype=torch.float32,
        )
        main_inputs = kv_tensors + [
            main_feat_embed,
            audio_seg1_start,
            audio_seg1_end,
            concat_text_len,
            hidden_states,
            rotary_cos,
            rotary_sin,
            attention_mask,
        ]
        main_input_names = kv_in_names + [
            "feat_embed",
            "audio_seg1_start",
            "audio_seg1_end",
            "concat_text_len",
            "hidden_states",
            "rotary_cos",
            "rotary_sin",
            "attention_mask",
        ]
        main_dynamic_axes = {
            **kv_dynamic_axes,
            "feat_embed": {1: "audio_feat_len"},
            "hidden_states": {1: "ids_len"},
            "rotary_cos": {0: "ids_len"},
            "rotary_sin": {0: "ids_len"},
            "attention_mask": {2: "ids_len", 3: "kv_seq_len"},
        }
        _export_component(
            main_core,
            tuple(main_inputs),
            component_paths["main_core"],
            main_input_names,
            (*kv_out_names, "dit_hidden", "stop_flag"),
            main_dynamic_axes,
        )
        del main_inputs, hidden_states, rotary_cos, rotary_sin, attention_mask
        gc.collect()

        noise = torch.ones(
            (1, patch_size, feat_in_channels),
            dtype=torch.float32,
        )
        dit_hidden = torch.zeros(
            (1, 2, dit_hidden_dim),
            dtype=torch.float32,
        )
        feat_cond = torch.zeros(
            (2, patch_size, cond_proj_out),
            dtype=torch.float32,
        )
        cfg_value = torch.tensor([CFG_VALUE], dtype=torch.float32)
        cfg_value_minus = torch.tensor([1.0 - CFG_VALUE], dtype=torch.float32)
        _export_component(
            feat_decoder,
            (noise, dit_hidden, feat_cond, cfg_value, cfg_value_minus),
            component_paths["feat_decoder"],
            ("random", "dit_hidden", "feat_cond", "cfg_value", "cfg_value_minus"),
            ("latent_pred",),
        )

        kv_seq_len = torch.tensor([prefill_seq_len], dtype=torch.int64)
        _export_component(
            decode_inputs,
            (kv_seq_len,),
            component_paths["decode_inputs"],
            ("kv_seq_len",),
            (
                "rotary_cos",
                "rotary_sin",
                "zero_audio_seg1_start",
                "zero_audio_seg1_end",
                "zero_concat_text_len",
                "zero_attention_mask",
                "kv_seq_len_out",
            ),
        )

        generated_latents = torch.zeros(
            (1, patch_size * 2, latent_dim),
            dtype=torch.float32,
        )
        current_latent = torch.zeros(
            (1, patch_size, latent_dim),
            dtype=torch.float32,
        )
        _export_component(
            latent_accumulator,
            (generated_latents, current_latent),
            component_paths["latent_accumulator"],
            ("generated_latents_in", "current_latent"),
            ("generated_latents",),
            {
                "generated_latents_in": {1: "generated_latent_len"},
                "generated_latents": {1: "generated_latent_len_out"},
            },
        )

        _export_component(
            vae_decoder,
            (generated_latents,),
            raw_onnx_folder / MODEL_FILES["vae_decoder"],
            ("generated_latents",),
            ("audio_out", "audio_out_len"),
            {
                "generated_latents": {1: "generated_latent_len"},
                "audio_out": {2: "audio_out_len"},
            } if DYNAMIC_SHAPE_VAE_DECODE else None,
        )
        _export_component(
            vae_decoder_stream,
            (current_latent, current_latent),
            raw_onnx_folder / MODEL_FILES["vae_decoder_stream"],
            ("previous_latent", "current_latent"),
            ("audio_out", "audio_out_len"),
        )

        metadata_marker = torch.zeros((1,), dtype=torch.int64)
        _export_component(
            METADATA_CARRIER(),
            (metadata_marker,),
            raw_onnx_folder / MODEL_FILES["metadata"],
            ("metadata_marker",),
            ("metadata_marker_out",),
        )

        for target in sorted(raw_onnx_folder.glob("*.onnx")):
            write_onnx_metadata(target, metadata)
        print(
            f"[Metadata] Stamped {len(metadata)} keys into "
            f"{len(list(raw_onnx_folder.glob('*.onnx')))} component graph(s)."
        )

        del (
            model,
            vae_encoder,
            feat_encoder,
            prefill_inputs,
            decode_inputs,
            main_core,
            feat_decoder,
            vae_decoder,
            vae_decoder_stream,
            latent_accumulator,
        )
        gc.collect()

    rewrite_report = rewrite_voxcpm2_onnx_folder(
        raw_onnx_folder,
        stage_folder,
    )
    print("[Targeted rewrite]")
    for rewritten in rewrite_report["models"]:
        print(
            f"  {rewritten['model']}: {rewritten['raw_nodes']} -> "
            f"{rewritten['final_nodes']} nodes; Conv={rewritten['conv_rewrites']}, "
            f"ConvTranspose={rewritten['conv_transpose_rewrites']}"
        )

    bundle_targets = sorted(stage_folder.glob("*.onnx"))
    bundle_stats = bundle_shared_initializers(
        stage_folder,
        bundle_targets,
        metadata=metadata,
    )
    print(
        f"[Shared weights] {bundle_stats['initializer_references']} component "
        f"references -> {bundle_stats['unique_initializers']} exact tensors."
    )

    component_paths = {
        name: stage_folder / file_name
        for name, file_name in COMPONENT_FILES.items()
    }
    assemble_paths = {
        mode: stage_folder / file_name
        for mode, file_name in ASSEMBLE_COMPONENT_FILES.items()
    }
    _compose_graphs(
        component_paths,
        assemble_paths,
        kv_in_names,
        kv_out_names,
        base_layers,
        stage_folder,
    )

    expected_after_composition = {
        *MODEL_FILES.values(),
        *PREFILL_FILES.values(),
        SHARED_MODEL_NAME,
        SHARED_DATA_NAME,
    }
    for path in list(stage_folder.iterdir()):
        if path.is_file() and path.name not in expected_after_composition:
            path.unlink()

    metadata = _finalize_shared_metadata(metadata, stage_folder)
    _install_package_folder(stage_folder, onnx_folder)
    shutil.rmtree(raw_onnx_folder)

    print(f"[Cleanup] Removed temporary export folder: {raw_onnx_folder}")
    print("\nVoxCPM2 export done!")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    export_voxcpm2()


if __name__ == "__main__" and "--expanded" in sys.argv:
    print('Export start ...')
    if raw_onnx_folder.exists():
        shutil.rmtree(raw_onnx_folder)
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

        # Static package metadata is built while the model is loaded, then
        # stamped onto every exported graph at the end of the export loop.
        onnx_metadata = build_model_metadata(
            {
                "producer": Path(__file__).name,
                "in_sample_rate": IN_SAMPLE_RATE,
                "out_sample_rate": OUT_SAMPLE_RATE,
                "model_in_sample_rate": MODEL_IN_SAMPLE_RATE,
                "model_out_sample_rate": MODEL_OUT_SAMPLE_RATE,
                "input_audio_dtype": IN_AUDIO_DTYPE.upper(),
                "output_audio_dtype": OUT_AUDIO_DTYPE.upper(),
                "num_audio_inputs": 1,
                "max_seq_len": MAX_SEQ_LEN,
                "stop_token_ids": sorted(STOP_TOKEN),
                "fixed_timesteps": FIXED_TIMESTEPS,
                "use_f16_kv": USE_F16_KV,
                "compute_in_f32": COMPUTE_IN_F32,
                "kv_dtype": "float16" if USE_F16_KV else "float32",
                "activations_fp16": False,
                "simplified_layer_norm": True,
                "opset": OPSET,
            },
            {
                "base_lm_num_layers": base_layers,
                "residual_lm_num_layers": residual_layers,
                "total_layers": total_layers,
                "head_dim": head_dim,
                "num_key_value_heads": num_kv_heads,
                "hidden_size": hidden_size,
                "patch_size": patch_size,
                "feat_dim": feat_dim,
                "latent_dim": latent_dim,
                "dit_hidden_dim": dit_hidden_dim,
            },
        )

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
        prompt_audio = torch.zeros(
            [1, 1, encode_patch_len * 2],
            dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()],
        )
        torch.onnx.export(
            VOXCPM2_VAE_ENCODER(model),
            (prompt_audio,),
            onnx_model_VAE_Encoder,
            input_names=['audio'],
            output_names=['audio_feat'],
            dynamic_axes={'audio': {2: 'audio_samples'}, 'audio_feat': {0: 'audio_feat_len'}},
            opset_version=OPSET,
            dynamo=False
        )
        del prompt_audio

        # ══════════════════════════════════════════════════════════════
        # Export: Feat_Encoder_Cond (Fused)
        # ══════════════════════════════════════════════════════════════
        print('Exporting Feat_Encoder_Cond (fused) ...')
        audio_feat = torch.zeros([10, patch_size, feat_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM2_FEAT_ENCODER_COND(model),
            (audio_feat,),
            onnx_model_Feat_Encoder_Cond,
            input_names=['audio_feat'],
            output_names=['feat_embed', 'feat_cond'],
            dynamic_axes={'audio_feat': {0: 'audio_feat_len'}, 'feat_embed': {1: 'audio_feat_len'}},
            opset_version=OPSET,
            dynamo=False
        )
        del audio_feat

        # ══════════════════════════════════════════════════════════════
        # Export: Assemble (per-mode — no control flow in each forward())
        # ══════════════════════════════════════════════════════════════
        _asm_text_ids = torch.zeros([1, 15], dtype=torch.int32)
        _asm_out_names = ['text_token', 'audio_feat', 'audio_seg1_start', 'audio_seg1_end', 'concat_text_len', 'ids_len']
        _asm_dyn_out = {'text_token': {1: 'total_len'}, 'audio_feat': {0: 'total_len'}}

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
        _asm_prompt_feat = torch.zeros([8, patch_size, latent_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM2_ASSEMBLE_CONTINUATION(patch_size, latent_dim, MAX_SEQ_LEN),
            (_asm_text_ids, _asm_prompt_feat),
            onnx_model_Assemble["continuation"],
            input_names=['text_ids', 'prompt_audio_feat'],
            output_names=_asm_out_names,
            dynamic_axes={'text_ids': {1: 'text_len'}, 'prompt_audio_feat': {0: 'prompt_len'}, **_asm_dyn_out},
            opset_version=OPSET,
            dynamo=False
        )

        # reference_only
        print('Exporting Assemble (reference_only) ...')
        _asm_ref_feat = torch.zeros([5, patch_size, latent_dim], dtype=torch.float32)
        torch.onnx.export(
            VOXCPM2_ASSEMBLE_REFERENCE_ONLY(patch_size, latent_dim, MAX_SEQ_LEN),
            (_asm_text_ids, _asm_ref_feat),
            onnx_model_Assemble["reference_only"],
            input_names=['text_ids', 'ref_audio_feat'],
            output_names=_asm_out_names,
            dynamic_axes={'text_ids': {1: 'text_len'}, 'ref_audio_feat': {0: 'ref_len'}, **_asm_dyn_out},
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
            dynamic_axes={'text_ids': {1: 'text_len'}, 'ref_audio_feat': {0: 'ref_len'}, 'prompt_audio_feat': {0: 'prompt_len'}, **_asm_dyn_out},
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
        latent_patches = torch.ones((1, patch_size + patch_size, latent_dim), dtype=torch.float32)

        torch.onnx.export(
            model_VAE_Decoder,
            (latent_patches,),
            onnx_model_VAE_Decoder,
            input_names=['latent_patches'],
            output_names=['generated_wav', 'audio_len'],
            dynamic_axes={
                'latent_patches': {1: 'latent_seq_len'},
                'generated_wav': {2: 'generated_len'}
            } if DYNAMIC_SHAPE_VAE_DECODE else None,
            opset_version=OPSET,
            dynamo=False
        )
        del model_VAE_Decoder, latent_patches

        # ══════════════════════════════════════════════════════════════
        # Export: Concat (Streaming only)
        # ══════════════════════════════════════════════════════════════
        print('Exporting Concat (streaming) ...')
        embed_0 = torch.zeros([1, patch_size, latent_dim], dtype=torch.float32)
        embed_1 = torch.zeros([1, patch_size, latent_dim], dtype=torch.float32)
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

        # ── Metadata carrier + stamp the metadata onto every exported graph ──
        metadata_marker = torch.zeros((1,), dtype=torch.int64)
        torch.onnx.export(
            METADATA_CARRIER(),
            (metadata_marker,),
            onnx_model_Metadata,
            input_names=['metadata_marker'],
            output_names=['metadata_marker_out'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )
        del metadata_marker
        written = 0
        for target in sorted(str(p) for p in raw_onnx_folder.glob("*.onnx")):
            write_onnx_metadata(target, onnx_metadata)
            written += 1
        print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {written} ONNX graph(s).")

        rewrite_report = rewrite_voxcpm2_onnx_folder(
            raw_onnx_folder,
            onnx_folder,
            require_stream_decoder=False,
        )
        print("\n[Targeted ONNX rewrite]")
        print(json.dumps(rewrite_report, indent=2, sort_keys=True))

    shutil.rmtree(raw_onnx_folder)
    print(f"[Cleanup] Removed temporary export folder: {raw_onnx_folder}")
    print('\nExport done!')


if __name__ == "__main__":
    print("\nStart running the VoxCPM demo via Inference_VoxCPM_ONNX.py ...")
    raise SystemExit(subprocess.call([
        sys.executable,
        str(script_dir / "Inference_VoxCPM_ONNX.py"),
        "--onnx-folder",
        str(onnx_folder),
    ]))
