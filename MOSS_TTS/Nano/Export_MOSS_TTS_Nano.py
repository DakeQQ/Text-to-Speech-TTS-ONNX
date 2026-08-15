import gc
import math
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from Shared_Weights import (
    SHARED_DATA_NAME,
    SHARED_MODEL_NAME,
    build_decode_step_graphs,
    bundle_shared_initializers,
)


# Paths
download_path = Path.home() / 'Downloads' / 'MOSS-TTS-Nano-100M'  # Source model folder

MAX_SEQ_LEN = 2048    # Fixed prompt plus generated-frame cache limit


# Audio (the codec is a separate checkpoint)
IN_SAMPLE_RATE = 48000      # Public prompt-audio ONNX input rate.
OUT_SAMPLE_RATE = 48000     # Public generated-waveform ONNX output rate.
IN_AUDIO_DTYPE = "F32"      # "F16" | "F32" | "INT16".
OUT_AUDIO_DTYPE = "F32"     # "F16" | "F32" | "INT16".


# Export
DO_EXPORT = True
USE_F16_KV = True           # Saves memory but can change codes after repeated decode steps
COMPUTE_IN_F32 = False      # With f16 KV, use f32 attention for accuracy instead of f16 speed
OPSET = 20                  # ONNX opset

_AUDIO_DTYPES = {"F16": torch.float16, "F32": torch.float32, "INT16": torch.int16}
script_dir = Path(__file__).resolve().parent
onnx_folder = script_dir / "MOSS_TTS_Nano_ONNX"  # Export folder
onnx_folder.mkdir(parents=True, exist_ok=True)
raw_onnx_folder = onnx_folder / "raw"  # Temporary rewrite artifacts
raw_onnx_folder.mkdir(parents=True, exist_ok=True)


DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")
onnx_model_Metadata = str(onnx_folder / "MossTTSNano_Metadata.onnx")
onnx_model_Main_Prefill = {
    strategy: str(onnx_folder / f"MossTTSNano_MainPrefill_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Main_Decode = {
    strategy: str(onnx_folder / f"MossTTSNano_MainDecodeDecision_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Predictor_Frame = {
    strategy: str(onnx_folder / f"MossTTSNano_PredictorFrame_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Decode_Step = {
    strategy: str(onnx_folder / f"MossTTSNano_DecodeStep_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}

audio_tokenizer_path = Path.home() / 'Downloads' / 'MOSS-Audio-Tokenizer-Nano'  # Codec source folder
onnx_model_Audio_Encoder = str(onnx_folder / "MossAudioTokenizer_Encoder.onnx")
onnx_model_Audio_Decoder = str(onnx_folder / "MossAudioTokenizer_Decoder.onnx")


def load_moss_model(model_dir):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_cls = get_class_from_dynamic_module(
        "modeling_moss_tts_nano.MossTTSNanoForCausalLM", model_dir
    )
    # transformers 5.13 crashes on the remote class's list-valued ignore keys.
    model_cls._keys_to_ignore_on_load_unexpected = None
    model = model_cls.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("eager")
    return model


def load_audio_tokenizer(model_dir):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_cls = get_class_from_dynamic_module(
        "modeling_moss_audio_tokenizer.MossAudioTokenizerModel", model_dir
    )
    model = model_cls.from_pretrained(model_dir, dtype=torch.float32).eval()
    model.set_attention_implementation("sdpa")
    return model


def _codec_rope_tables(seq_len, head_dim, max_period, device):
    half     = head_dim // 2
    inv_freq = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(max_period) * 2 / head_dim))
    ts       = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, 1, seq_len, 1)
    freqs    = inv_freq * ts
    cos_half = torch.cos(freqs)
    sin_half = torch.sin(freqs)
    cos      = torch.stack((cos_half, cos_half), dim=-1).reshape(1, 1, seq_len, head_dim)
    sin      = torch.stack((-sin_half, sin_half), dim=-1).reshape(1, 1, seq_len, head_dim)
    return cos, sin


def _codec_apply_rope(x, cos, sin, permutation):
    return x * cos + x.index_select(-1, permutation) * sin


def _codec_attention_bias(attn, input_lengths, positions, max_seqlen, dtype):
    valid_k   = positions.view(1, 1, max_seqlen) < input_lengths.view(-1, 1, 1)
    if not attn.causal and attn.context is None:
        allowed = valid_k[:, None, :, :].expand(-1, 1, max_seqlen, -1)
    else:
        delta   = positions.view(1, max_seqlen, 1) - positions.view(1, 1, max_seqlen)
        allowed = torch.ones((1, max_seqlen, max_seqlen), device=positions.device, dtype=torch.bool)
        if attn.causal:
            allowed = allowed & (delta >= 0)
        if attn.context is not None:
            allowed = allowed & (delta < attn.context)
        allowed = (allowed & valid_k)[:, None, :, :]
    zero       = torch.zeros((), device=positions.device, dtype=dtype)
    # A finite minimum avoids NaNs for fully masked rows.
    mask_value = torch.full((), torch.finfo(dtype).min, device=positions.device, dtype=dtype)
    return torch.where(allowed, zero, mask_value)


def _codec_self_attention(attn, x, attn_bias, valid_q, cos, sin):
    batch_size, seq_len, _ = x.shape
    num_heads = attn.num_heads
    head_dim  = attn.embed_dim // num_heads
    qkv       = attn.in_proj(x).reshape(batch_size, seq_len, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    qk, v     = torch.split(qkv, [2, 1], dim=0)
    qk        = qk.reshape(2 * batch_size, num_heads, seq_len, head_dim)
    qk        = _codec_apply_rope(qk, cos, sin, attn._onnx_rope_permutation)
    qk        = qk.reshape(2, batch_size, num_heads, seq_len, head_dim)
    q, k      = qk.unbind(0)
    v         = v.squeeze(0)
    scores    = torch.matmul(q, k.transpose(-1, -2)) + attn_bias
    out       = torch.matmul(torch.softmax(scores, dim=-1), v)
    out       = torch.where(valid_q, out, torch.zeros((), device=out.device, dtype=out.dtype))
    out       = out.transpose(1, 2).reshape(batch_size, seq_len, attn.embed_dim)
    return attn.out_proj(out)


def _codec_transformer_layer(layer, x, attn_bias, valid_q, cos, sin):
    residual = x
    hidden   = _codec_self_attention(layer.self_attn, layer.norm1(x), attn_bias, valid_q, cos, sin)
    x        = residual + layer.layer_scale_1(hidden)
    residual = x
    x        = residual + layer.layer_scale_2(layer.ffn(layer.norm2(x)))
    return x


def _codec_fold_layer_scales(modules):
    with torch.no_grad():
        for module in modules:
            if module.module_type == "PatchedPretransform":
                continue
            for layer in module.transformer.layers:
                attention = layer.self_attn
                if not getattr(attention, "_onnx_qk_scale_folded", False):
                    head_dim = attention.embed_dim // attention.num_heads
                    qk_rows = 2 * attention.embed_dim
                    attention.in_proj.weight[:qk_rows].mul_(
                        head_dim ** -0.25
                    )
                    attention.register_buffer(
                        "_onnx_rope_permutation",
                        torch.arange(head_dim, dtype=torch.int64).view(-1, 2).flip(-1).reshape(-1),
                        persistent=False,
                    )
                    attention._onnx_qk_scale_folded = True
                pairs = (
                    ("layer_scale_1", attention.out_proj),
                    ("layer_scale_2", layer.ffn[-1]),
                )
                for scale_name, projection in pairs:
                    scale_module = getattr(layer, scale_name)
                    if isinstance(scale_module, torch.nn.Identity):
                        continue
                    scale = scale_module.scale.detach()
                    projection.weight.mul_(scale.unsqueeze(1))
                    if projection.bias is not None:
                        projection.bias.mul_(scale)
                    setattr(layer, scale_name, torch.nn.Identity())


def _codec_materialize_weight_norm(module):
    materialized = 0
    for child in module.modules():
        parametrizations = getattr(child, "parametrizations", None)
        if parametrizations is not None and "weight" in parametrizations:
            torch.nn.utils.parametrize.remove_parametrizations(child, "weight", leave_parametrized=True)
            materialized += 1
    return materialized


def _codec_fuse_encoder_patch_projections(encoder, quantizer):
    for index, patch_module in enumerate(encoder[:-1]):
        if patch_module.module_type != "PatchedPretransform":
            continue
        projected = encoder[index + 1]
        patch_size = int(patch_module.patch_size)
        linear     = projected.input_proj
        in_channels = linear.in_features // patch_size
        convolution = torch.nn.Conv1d(
            in_channels,
            linear.out_features,
            kernel_size=patch_size,
            stride=patch_size,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        convolution.weight = torch.nn.Parameter(
            linear.weight.detach().reshape(linear.out_features, in_channels, patch_size).contiguous(),
            requires_grad=False,
        )
        if linear.bias is not None:
            convolution.bias = torch.nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        projected.input_proj = convolution
        patch_module._onnx_fused_into_projection = True

    final_patch = encoder[-1]
    patch_size = int(final_patch.patch_size)
    projection = quantizer.input_proj
    in_channels = projection.in_channels // patch_size
    convolution = torch.nn.Conv1d(
        in_channels,
        projection.out_channels,
        kernel_size=patch_size,
        stride=patch_size,
        bias=projection.bias is not None,
        device=projection.weight.device,
        dtype=projection.weight.dtype,
    )
    convolution.weight = torch.nn.Parameter(
        projection.weight.detach().reshape(projection.out_channels, in_channels, patch_size).contiguous(),
        requires_grad=False,
    )
    if projection.bias is not None:
        convolution.bias = torch.nn.Parameter(projection.bias.detach().clone(), requires_grad=False)
    quantizer.input_proj = convolution
    final_patch._onnx_fused_into_projection = True


def _codec_projected_transformer(module, x, input_lengths):
    transformer = module.transformer
    if isinstance(module.input_proj, torch.nn.Conv1d):
        x = module.input_proj(x).transpose(1, 2)
    else:
        x = module.input_proj(x.transpose(1, 2))
    # Dynamic dimensions keep legacy-exporter reshapes symbolic.
    batch_size, seq_len, _ = x.shape
    attn0       = transformer.layers[0].self_attn
    positions   = torch.arange(seq_len, device=x.device, dtype=torch.long)
    attn_bias   = _codec_attention_bias(attn0, input_lengths, positions, seq_len, x.dtype)
    valid_q     = (positions.view(1, seq_len) < input_lengths.view(-1, 1)).view(batch_size, 1, seq_len, 1)
    cos, sin    = _codec_rope_tables(seq_len, attn0.embed_dim // attn0.num_heads, transformer.max_period, x.device)
    for layer in transformer.layers:
        x = _codec_transformer_layer(layer, x, attn_bias, valid_q, cos, sin)
    x = module.output_proj(x).transpose(1, 2)
    return x, input_lengths


def _codec_patched_encode(module, x, input_lengths):
    batch, dim, _ = x.shape
    patch = module.patch_size
    x = x.reshape(batch, dim, -1, patch).permute(0, 1, 3, 2).reshape(batch, dim * patch, -1)
    return x, input_lengths // patch


def _codec_patched_decode(module, x, input_lengths):
    batch, dim_patch, length = x.shape
    patch = module.patch_size
    dim   = dim_patch // patch
    x = x.reshape(batch, dim, patch, length).permute(0, 1, 3, 2).reshape(batch, dim, length * patch)
    return x, input_lengths * patch


def _codec_lfq_decode_latents(lfq, latents):
    encodings = F.normalize(latents.transpose(1, 2).reshape(-1, lfq.codebook_dim))
    scores  = 2 * (encodings @ lfq._cb_normalized_t) - lfq._cb_norm_sq
    indices = torch.argmax(scores, dim=1).reshape(1, -1)
    z_q     = F.embedding(indices, lfq.codebook.weight).transpose(1, 2)
    return z_q, indices


def _codec_lfq_quantize(lfq, z):
    z_e = lfq.in_proj(z)
    z_q, indices = _codec_lfq_decode_latents(lfq, z_e)
    z_q = lfq.out_proj(z_q)
    return z_q, indices


def _codec_residual_lfq_encode(quantizer, z, input_length, n_quantizers):
    z = quantizer.input_proj(z)
    max_time = z.shape[2]
    mask = (torch.arange(max_time, device=z.device) < input_length.unsqueeze(1)).unsqueeze(1)
    residual    = z * mask
    all_indices = []
    for i, lfq in enumerate(quantizer.quantizers):
        if i >= n_quantizers:
            break
        z_q_i, indices_i = _codec_lfq_quantize(lfq, residual)
        z_q_i = z_q_i * mask
        residual = residual - z_q_i
        all_indices.append(indices_i)
    all_indices = torch.stack(all_indices)
    return all_indices, input_length


def _codec_residual_lfq_decode(quantizer, codes, effective_codebook, codebook_offsets):
    emb = F.embedding(codes + codebook_offsets, effective_codebook).sum(dim=0)
    return quantizer.output_proj(emb.transpose(1, 2))


class AUDIO_ENCODER(torch.nn.Module):
    def __init__(self, model, in_sample_rate):
        super().__init__()
        self.encoder                   = model.encoder
        self.quantizer                 = model.quantizer
        self.number_channels           = int(model.number_channels)
        self.enable_channel_interleave = bool(model.enable_channel_interleave)
        self.downsample_rate           = int(model.downsample_rate)
        self.n_quantizers              = int(model.quantizer.num_quantizers)
        self.model_sample_rate         = int(model.sampling_rate)
        self.sr_scale                  = float(self.model_sample_rate / int(in_sample_rate))

        _codec_fold_layer_scales(self.encoder)
        _codec_materialize_weight_norm(self.quantizer)

        # Fold stereo channel interleave into the first patch projection.
        first_patch     = self.encoder[0]
        first_projected = self.encoder[1]
        patch_size = int(first_patch.patch_size)
        input_weight = first_projected.input_proj.weight.data
        if self.number_channels > 1 and self.enable_channel_interleave:
            input_weight = input_weight.view(
                input_weight.shape[0], patch_size // self.number_channels, self.number_channels
            ).permute(0, 2, 1).reshape(input_weight.shape[0], patch_size)
            first_patch.patch_size = patch_size // self.number_channels
            first_projected.input_dimension = input_weight.shape[1]
            self.enable_channel_interleave = False
        if "int" in IN_AUDIO_DTYPE.lower():
            input_weight = input_weight * (1.0 / 32768.0)
        input_weight = input_weight.contiguous()
        first_projected.input_proj.weight = torch.nn.Parameter(input_weight, requires_grad=False)
        first_projected.input_proj.in_features = input_weight.shape[1]
        _codec_fuse_encoder_patch_projections(self.encoder, self.quantizer)

        with torch.no_grad():
            for lfq in self.quantizer.quantizers:
                cb_normalized        = F.normalize(lfq.codebook.weight)
                lfq._cb_normalized_t = cb_normalized.t().contiguous()
                lfq._cb_norm_sq      = cb_normalized.pow(2).sum(1, keepdim=True).t().contiguous()

    def forward(self, prompt_audio):
        audio = prompt_audio.float()
        if self.sr_scale < 1.0:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )
        if self.sr_scale > 1.0:
            audio = torch.nn.functional.interpolate(
                audio,
                scale_factor=self.sr_scale,
                mode='linear',
                align_corners=False,
                recompute_scale_factor=False,
            )

        valid_len = torch._shape_as_tensor(audio)[2]

        pad_length = (self.downsample_rate - valid_len % self.downsample_rate) % self.downsample_rate
        audio      = torch.nn.functional.pad(audio, (0, pad_length))

        lengths = valid_len.view(1)
        if self.number_channels > 1 and self.enable_channel_interleave:
            audio   = audio.repeat(1, self.number_channels, 1)
            audio   = audio.transpose(1, 2).contiguous().view(1, 1, -1)
            lengths = lengths * self.number_channels

        hidden_states, hidden_lengths = audio, lengths
        for encoder_module in self.encoder:
            if encoder_module.module_type == "PatchedPretransform":
                if getattr(encoder_module, "_onnx_fused_into_projection", False):
                    hidden_lengths = hidden_lengths // encoder_module.patch_size
                else:
                    hidden_states, hidden_lengths = _codec_patched_encode(
                        encoder_module, hidden_states, hidden_lengths
                    )
            else:
                hidden_states, hidden_lengths = _codec_projected_transformer(encoder_module, hidden_states, hidden_lengths)

        audio_codes, audio_code_lengths = _codec_residual_lfq_encode(
            self.quantizer, hidden_states, hidden_lengths, self.n_quantizers
        )
        audio_codes = audio_codes.permute(1, 2, 0).contiguous().int()
        return audio_codes, audio_code_lengths.int()


class AUDIO_DECODER(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quantizer                 = model.quantizer
        self.decoder                   = model.decoder
        self.number_channels           = int(model.number_channels)
        self.enable_channel_interleave = bool(model.enable_channel_interleave)
        self.model_sample_rate         = int(model.sampling_rate)
        self.output_resample_scale     = float(OUT_SAMPLE_RATE / self.model_sample_rate)

        _codec_fold_layer_scales(self.decoder)
        _codec_materialize_weight_norm(self.quantizer)

        # Fold channel de-interleave into the final projection.
        if self.number_channels > 1 and self.enable_channel_interleave:
            final_projected = self.decoder[-2]
            final_patch     = self.decoder[-1]
            patch_size      = int(final_patch.patch_size)
            row_order = torch.arange(patch_size).view(-1, self.number_channels).transpose(0, 1).reshape(-1)
            with torch.no_grad():
                final_projected.output_proj.weight.data = final_projected.output_proj.weight.data.index_select(0, row_order)
                if final_projected.output_proj.bias is not None:
                    final_projected.output_proj.bias.data = final_projected.output_proj.bias.data.index_select(0, row_order)
            final_patch.patch_size = patch_size // self.number_channels
            self.enable_channel_interleave = False

        # Fold per-quantizer output projections into one packed codebook.
        self.num_quantizers = int(model.quantizer.num_quantizers)
        effective_codebooks = []
        with torch.no_grad():
            for lfq in self.quantizer.quantizers:
                effective = lfq.out_proj(lfq.codebook.weight.t().unsqueeze(0)).squeeze(0).t().contiguous()
                effective_codebooks.append(effective)
        codebook_size = effective_codebooks[0].shape[0]
        self.effective_codebook = torch.nn.Parameter(
            torch.cat(effective_codebooks, dim=0),
            requires_grad=False,
        )
        self.register_buffer(
            "codebook_offsets",
            (torch.arange(self.num_quantizers, dtype=torch.int64) * codebook_size).view(-1, 1, 1),
            persistent=True,
        )

    def forward(self, generated_codec):
        input_lengths = torch._shape_as_tensor(generated_codec)[1].view(1)
        codes = generated_codec.permute(2, 0, 1).long()
        hidden_states = _codec_residual_lfq_decode(
            self.quantizer, codes, self.effective_codebook, self.codebook_offsets
        )
        audio, lengths = hidden_states, input_lengths
        for decoder_module in self.decoder:
            if decoder_module.module_type == "PatchedPretransform":
                audio, lengths = _codec_patched_decode(decoder_module, audio, lengths)
            else:
                audio, lengths = _codec_projected_transformer(decoder_module, audio, lengths)
        audio = audio.clamp(min=-1.0, max=1.0)
        if self.output_resample_scale != 1.0:
            audio = F.interpolate(
                audio,
                scale_factor=self.output_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            return (audio * 32767.0).clamp(min=-32768.0, max=32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return audio.float()
        return audio.half()


class MOSS_EMBED(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        cfg           = model.config
        hidden        = int(cfg.gpt2_config.hidden_size)

        tables  = [model.transformer.wte.weight.data.float().clone()]
        offsets = [0]
        row_offset = tables[0].shape[0]
        pad_row = torch.zeros(1, hidden, dtype=torch.float32)
        for ch in range(int(cfg.n_vq)):
            audio_weight = model.audio_embeddings[ch].weight.data.float().clone()
            extended     = torch.cat([audio_weight, pad_row], dim=0)
            tables.append(extended)
            offsets.append(row_offset)
            row_offset += extended.shape[0]
        self.weight = torch.nn.Parameter(torch.cat(tables, dim=0), requires_grad=False)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int32), persistent=False)
        self.register_buffer("audio_offsets", torch.tensor(offsets[1:], dtype=torch.int32), persistent=False)
        assistant_slot_token_id = int(cfg.audio_assistant_slot_token_id)
        self.register_buffer(
            "assistant_slot_embedding",
            tables[0][assistant_slot_token_id:assistant_slot_token_id + 1].clone().half(),
            persistent=False,
        )

    def forward(self, input_ids):
        return F.embedding(input_ids + self.offsets, self.weight).sum(dim=2)

    def assistant_slot(self, reference):
        batch_size = torch._shape_as_tensor(reference)[0]
        return self.assistant_slot_embedding[:batch_size].float().unsqueeze(1)

    def embed_frame(self, frame_codec_ids):
        audio = F.embedding(frame_codec_ids + self.audio_offsets, self.weight).sum(dim=1).float().unsqueeze(1)
        return audio + self.assistant_slot(frame_codec_ids)

    def embed_feedback(self, codes, channel):
        return F.embedding(codes + self.audio_offsets[channel], self.weight).float()


class MOSS_EMBED_AUDIO(torch.nn.Module):
    def __init__(self, model, channel):
        super().__init__()
        self.register_buffer("weight", model.audio_embeddings[channel].weight.data.float().clone(), persistent=False)

    def forward(self, codes):
        return F.embedding(codes, self.weight)


# Recompute interleaved GPT-J RoPE; the checkpoint's non-persistent buffer may be uninitialized.
def _build_rope_tables(head_dim, base, max_seq_len):
    inv_freq  = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
    freqs     = positions * inv_freq
    cos_half  = torch.cos(freqs)
    cos_table = torch.stack((cos_half, cos_half), dim=-1).reshape(max_seq_len, head_dim)
    sin_half  = torch.sin(freqs)
    sin_table = torch.stack([-sin_half, sin_half], dim=-1).reshape(max_seq_len, head_dim)
    cos_table = cos_table.view(1, max_seq_len, 1, head_dim)
    sin_table = sin_table.view(1, max_seq_len, 1, head_dim)
    return cos_table, sin_table


class MOSS_MAIN_ROTARY_MASK_PREFILL(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        cfg      = model.config.gpt2_config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        base     = float(getattr(cfg, "rope_base", 10000.0))
        cos, sin = _build_rope_tables(head_dim, base, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos.half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half(), persistent=False)

        self.mask_dtype = torch.float16 if (USE_F16_KV and not COMPUTE_IN_F32) else torch.float32
        self.register_buffer("positions", torch.arange(max_seq_len, dtype=torch.int16), persistent=False)
        self.register_buffer("mask_zero", torch.tensor(0.0, dtype=self.mask_dtype), persistent=False)
        self.register_buffer(
            "mask_value", torch.tensor(torch.finfo(self.mask_dtype).min, dtype=self.mask_dtype), persistent=False
        )

    def forward(self, ids_len, history_len):
        kv_seq_len         = ids_len + history_len
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        query_positions    = self.positions[:ids_len].long() + history_len
        key_positions      = self.positions[:kv_seq_len].int()
        causal             = key_positions.view(1, 1, 1, -1) <= query_positions.view(1, 1, -1, 1)
        attention_mask     = torch.where(causal, self.mask_zero, self.mask_value)
        return rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, kv_seq_len


class MOSS_MAIN_ROTARY_DECODE(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        cfg      = model.config.gpt2_config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        base     = float(getattr(cfg, "rope_base", 10000.0))
        cos, sin = _build_rope_tables(head_dim, base, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos.half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half(), persistent=False)

    def forward(self, kv_seq_len):
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, kv_seq_len + 1


class MOSS_PREDICTOR_ROTARY(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        cfg      = model.local_transformer.config
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        base     = float(getattr(cfg, "rope_base", 10000.0))
        cos, sin = _build_rope_tables(head_dim, base, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", cos.half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin.half(), persistent=False)

    def forward(self, position):
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, position].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, position].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, position + 1


class _GPT2_STACK(torch.nn.Module):
    def __init__(self, transformer, cfg):
        super().__init__()
        self.h            = transformer.h
        self.hidden_size  = int(cfg.hidden_size)
        self.num_heads    = int(cfg.num_attention_heads)
        self.head_dim     = self.hidden_size // self.num_heads
        self.qk_heads     = self.num_heads + self.num_heads
        self.num_layers   = len(self.h)
        self.ln_eps       = float(cfg.layer_norm_epsilon)
        self.compute_in_f32 = COMPUTE_IN_F32
        self.register_buffer(
            "rope_permutation",
            torch.arange(self.head_dim, dtype=torch.int64).view(-1, 2).flip(-1).reshape(-1),
            persistent=False,
        )

        self.save_key   = [None] * self.num_layers
        self.save_value = [None] * self.num_layers

        self._fuse_weights()

    def _fuse_weights(self):
        scale_factor = self.head_dim ** -0.25
        two_h        = 2 * self.hidden_size
        with torch.no_grad():
            for layer in self.h:
                if getattr(layer, "_moss_onnx_fused", False):
                    continue
                # Fold attention scaling and LayerNorm affine terms into linear weights.
                c_attn = layer.attn.c_attn
                weight = c_attn.weight.data
                bias   = c_attn.bias.data
                weight[:two_h].mul_(scale_factor)
                bias[:two_h].mul_(scale_factor)
                ln1 = layer.ln_1
                bias.add_(weight @ ln1.bias.data)
                weight.mul_(ln1.weight.data.unsqueeze(0))

                fc_in = layer.mlp.fc_in
                ln2   = layer.ln_2
                fc_in.bias.data.add_(fc_in.weight.data @ ln2.bias.data)
                fc_in.weight.data.mul_(ln2.weight.data.unsqueeze(0))
                layer._moss_onnx_fused = True

    def _layer_norm(self, hidden_states):
        return F.layer_norm(hidden_states, (self.hidden_size,), None, None, self.ln_eps)

    def rotate_half(self, x):
        return x.index_select(-1, self.rope_permutation)

    def run(self, hidden_states, rotary_cos, rotary_sin, attention_mask, past_inputs, batch_size):
        for i, layer in enumerate(self.h):
            residual      = hidden_states
            hidden_states = self._layer_norm(hidden_states)
            qkv           = layer.attn.c_attn(hidden_states).view(batch_size, -1, self.qk_heads + self.num_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_heads], dim=2)
            qk            = qk * rotary_cos + self.rotate_half(qk) * rotary_sin

            if USE_F16_KV and not self.compute_in_f32:
                # Cast before layout ops so f16 attention stays fully f16.
                qk = qk.half()

            q, k = torch.split(qk, [self.num_heads, self.num_heads], dim=2)
            q    = q.transpose(1, 2)

            if USE_F16_KV:
                if self.compute_in_f32:
                    k = k.half()  # Storage only; attention matmul upcasts it
                v = v.half()

            k = torch.cat((past_inputs[i],                   k.permute(0, 2, 3, 1)), dim=-1)
            v = torch.cat((past_inputs[i + self.num_layers], v.transpose(1, 2)),     dim=-2)
            self.save_key[i]   = k
            self.save_value[i] = v

            if USE_F16_KV and self.compute_in_f32:
                scores = torch.matmul(q, k.float())
                value_compute = v.float()
            else:
                scores = torch.matmul(q, k)
                value_compute = v
            if attention_mask is not None:
                scores = scores + attention_mask
            attn = torch.softmax(scores, dim=-1)
            attn = torch.matmul(attn, value_compute).transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
            if USE_F16_KV and not self.compute_in_f32:
                attn = attn.float()
            hidden_states = residual + layer.attn.c_proj(attn)

            residual      = hidden_states
            hidden_states = self._layer_norm(hidden_states)
            hidden_states = residual + layer.mlp.fc_out(F.gelu(layer.mlp.fc_in(hidden_states), approximate="tanh"))
        return hidden_states


class MOSS_MAIN(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        cfg          = model.config.gpt2_config
        self.stack   = _GPT2_STACK(model.transformer, cfg)
        self.num_layers = self.stack.num_layers
        # ln_f stays affine because it feeds the local transformer's residual stream.
        self.ln_f_weight = torch.nn.Parameter(
            model.transformer.ln_f.weight.data.float().clone(),
            requires_grad=False,
        )
        self.ln_f_bias = torch.nn.Parameter(
            model.transformer.ln_f.bias.data.float().clone(),
            requires_grad=False,
        )
        self.ln_eps  = self.stack.ln_eps
        self.hidden_size = self.stack.hidden_size

    def forward(self, *all_inputs):
        hidden_states  = all_inputs[-4]
        rotary_cos     = all_inputs[-3]
        rotary_sin     = all_inputs[-2]
        attention_mask = all_inputs[-1]

        hidden_states = self.stack.run(hidden_states, rotary_cos, rotary_sin, attention_mask, all_inputs, 1)

        last_hidden_states = F.layer_norm(hidden_states[:, -1], (self.hidden_size,), self.ln_f_weight, self.ln_f_bias, self.ln_eps)
        return *self.stack.save_key, *self.stack.save_value, last_hidden_states.unsqueeze(1)


class MOSS_PREDICTOR(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        cfg          = model.local_transformer.config
        self.stack   = _GPT2_STACK(model.local_transformer, cfg)
        self.num_layers = self.stack.num_layers
        self.hidden_size = self.stack.hidden_size
        self.ln_eps  = self.stack.ln_eps

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-3]
        rotary_cos    = all_inputs[-2]
        rotary_sin    = all_inputs[-1]

        hidden_states = self.stack.run(hidden_states, rotary_cos, rotary_sin, None, all_inputs, 1)

        last_hidden_states = F.layer_norm(hidden_states[:, -1], (self.hidden_size,), None, None, self.ln_eps)
        return *self.stack.save_key, *self.stack.save_value, last_hidden_states


# Clone tied head weights before folding the local ln_f affine.
def _fold_layernorm_into_head(head_weight, ln_weight, ln_bias):
    fold_bias   = head_weight @ ln_bias
    fold_weight = head_weight * ln_weight.unsqueeze(0)
    return fold_weight, fold_bias


class MOSS_TEXT_HEAD(torch.nn.Module):
    """Decision logits: index 0 continues and index 1 stops."""

    def __init__(self, model):
        super().__init__()
        cfg       = model.config
        ln        = model.local_transformer.ln_f
        ln_weight = ln.weight.data.float().clone()
        ln_bias   = ln.bias.data.float().clone()
        rows      = torch.tensor([cfg.audio_assistant_slot_token_id, cfg.audio_end_token_id], dtype=torch.long)
        head_w    = model.text_lm_head.weight.data.index_select(0, rows).float().clone()
        fold_weight, fold_bias = _fold_layernorm_into_head(head_w, ln_weight, ln_bias)
        self.weight_transposed = torch.nn.Parameter(fold_weight.t().contiguous(), requires_grad=False)
        self.bias = torch.nn.Parameter(fold_bias, requires_grad=False)

    def forward(self, hidden_states):
        return torch.matmul(hidden_states, self.weight_transposed) + self.bias


class MOSS_AUDIO_HEAD(torch.nn.Module):
    def __init__(self, model, channel):
        super().__init__()
        ln        = model.local_transformer.ln_f
        ln_weight = ln.weight.data.float().clone()
        ln_bias   = ln.bias.data.float().clone()
        head_w    = model.audio_lm_heads[channel].weight.data.float().clone()
        fold_weight, fold_bias = _fold_layernorm_into_head(head_w, ln_weight, ln_bias)
        self.weight_transposed = torch.nn.Parameter(fold_weight.t().contiguous(), requires_grad=False)
        self.bias = torch.nn.Parameter(fold_bias, requires_grad=False)

    def forward(self, hidden_states):
        return torch.matmul(hidden_states, self.weight_transposed) + self.bias


class MOSS_PREDICTOR_FRAME_GREEDY(torch.nn.Module):
    def __init__(self, model, predictor, packed_embed, audio_heads):
        super().__init__()
        self.predictor = predictor
        self.packed_embed = packed_embed
        self.audio_heads = audio_heads
        self.num_layers = predictor.num_layers
        self.num_channels = len(audio_heads)

        cfg = model.local_transformer.config
        head_dim = int(cfg.hidden_size) // int(cfg.num_attention_heads)
        base = float(getattr(cfg, "rope_base", 10000.0))
        cos, sin = _build_rope_tables(head_dim, base, self.num_channels + 1)
        self.register_buffer("rotary_cos", cos.half(), persistent=True)
        self.register_buffer("rotary_sin", sin.half(), persistent=True)

    def _run(self, predictor_kv, capture_trace):
        kv = list(predictor_kv)
        hidden_states = self.packed_embed.assistant_slot(predictor_kv[0])
        tokens = []
        trace = []
        for channel in range(self.num_channels):
            outputs = self.predictor(
                *kv,
                hidden_states,
                self.rotary_cos[:, channel + 1:channel + 2].float(),
                self.rotary_sin[:, channel + 1:channel + 2].float(),
            )
            kv = list(outputs[:self.num_layers * 2])
            hidden = outputs[-1]
            token = torch.argmax(self.audio_heads[channel](hidden), dim=-1).int()
            tokens.append(token)
            if capture_trace:
                trace.append((tuple(kv), hidden, token))
            if channel + 1 < self.num_channels:
                hidden_states = self.packed_embed.embed_feedback(token.view(1, 1), channel)
        return torch.stack(tokens, dim=1), trace

    def forward(self, *predictor_kv):
        frame_codec_ids, _ = self._run(predictor_kv, False)
        return frame_codec_ids

    def eager_trace(self, *predictor_kv):
        return self._run(predictor_kv, True)


class MOSS_SIGN_AWARE_REPETITION_PENALTY(torch.autograd.Function):
    """Apply repetition penalty only at history IDs while preserving int32 ONNX indices."""

    @staticmethod
    def forward(ctx, logits, repetition_penalty, previous_ids):
        previous_ids_long = previous_ids.long()
        previous_logits = torch.gather(logits, 1, previous_ids_long)
        inv_penalty = torch.reciprocal(repetition_penalty)
        previous_scores = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits * inv_penalty,
        )
        return torch.scatter(logits, 1, previous_ids_long, previous_scores)

    @staticmethod
    def symbolic(g, logits, repetition_penalty, previous_ids):
        previous_logits = g.op("GatherElements", logits, previous_ids, axis_i=1)
        zero = g.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))
        inv_penalty = g.op("Reciprocal", repetition_penalty)
        previous_scores = g.op(
            "Where",
            g.op("Less", previous_logits, zero),
            g.op("Mul", previous_logits, repetition_penalty),
            g.op("Mul", previous_logits, inv_penalty),
        )
        return g.op("ScatterElements", logits, previous_ids, previous_scores, axis_i=1)


class MOSS_SAMPLER(torch.nn.Module):
    NEG_INF = float("-inf")

    def __init__(self, max_vocab_size):
        super().__init__()
        self.register_buffer("positions", torch.arange(max_vocab_size, dtype=torch.int32).unsqueeze(0), persistent=False)
        self.register_buffer("neg_inf", torch.tensor(self.NEG_INF, dtype=torch.float32), persistent=False)

    def sample(self, scores, temperature, top_k, top_p):
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        positions = self.positions[:, :sorted_scores.shape[-1]]
        keep_topk = positions < top_k
        sorted_scores = torch.where(keep_topk, sorted_scores, self.neg_inf)

        sorted_probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        keep = (cumulative_probabilities - sorted_probabilities) <= top_p

        kept_mass = torch.where(keep, cumulative_probabilities, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax((cumulative_probabilities >= threshold).int(), dim=-1, keepdim=True)
        return torch.gather(sorted_indices, 1, winner).squeeze(-1).int()

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        scores = MOSS_SIGN_AWARE_REPETITION_PENALTY.apply(logits, repetition_penalty, previous_ids)

        return self.sample(scores, temperature, top_k, top_p)


class MOSS_REPETITION_PENALTY(torch.nn.Module):
    def forward(self, logits, previous_ids, repetition_penalty):
        return MOSS_SIGN_AWARE_REPETITION_PENALTY.apply(logits, repetition_penalty, previous_ids)


class MOSS_MAIN_COMPACT_CORE(torch.nn.Module):
    def __init__(self, model, max_seq_len):
        super().__init__()
        self.transformer = MOSS_MAIN(model)
        self.num_layers = self.transformer.num_layers
        cfg = model.config.gpt2_config
        self.num_heads = int(cfg.num_attention_heads)
        self.head_dim = int(cfg.hidden_size) // self.num_heads
        base = float(getattr(cfg, "rope_base", 10000.0))
        cos, sin = _build_rope_tables(self.head_dim, base, max_seq_len)
        self.register_buffer("rotary_cos", cos.half(), persistent=True)
        self.register_buffer("rotary_sin", sin.half(), persistent=True)
        self.register_buffer("positions", torch.arange(max_seq_len, dtype=torch.int16), persistent=False)
        mask_dtype = torch.float16 if (USE_F16_KV and not COMPUTE_IN_F32) else torch.float32
        self.register_buffer("mask_zero", torch.tensor(0.0, dtype=mask_dtype), persistent=False)
        self.register_buffer(
            "mask_value",
            torch.tensor(torch.finfo(mask_dtype).min, dtype=mask_dtype),
            persistent=False,
        )
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        self.register_buffer(
            "empty_key",
            torch.zeros(1, self.num_heads, self.head_dim, 0, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_value",
            torch.zeros(1, self.num_heads, 0, self.head_dim, dtype=kv_dtype),
            persistent=False,
        )

    def prefill(self, hidden_states):
        ids_len = torch._shape_as_tensor(hidden_states)[1]
        positions = self.positions[:ids_len].int()
        causal = positions.view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
        attention_mask = torch.where(causal, self.mask_zero, self.mask_value)
        outputs = self.transformer(
            *([self.empty_key] * self.num_layers),
            *([self.empty_value] * self.num_layers),
            hidden_states,
            self.rotary_cos[:, :ids_len].float(),
            self.rotary_sin[:, :ids_len].float(),
            attention_mask,
        )
        return *outputs, ids_len.view(1)

    def decode(self, main_kv, hidden_states, kv_seq_len):
        outputs = self.transformer(
            *main_kv,
            hidden_states,
            self.rotary_cos[:, kv_seq_len].float(),
            self.rotary_sin[:, kv_seq_len].float(),
            None,
        )
        return *outputs, kv_seq_len + 1


class MOSS_PREDICTOR_COMPACT_CORE(torch.nn.Module):
    def __init__(self, model, n_vq):
        super().__init__()
        self.transformer = MOSS_PREDICTOR(model)
        self.num_layers = self.transformer.num_layers
        cfg = model.local_transformer.config
        self.num_heads = int(cfg.num_attention_heads)
        self.head_dim = int(cfg.hidden_size) // self.num_heads
        base = float(getattr(cfg, "rope_base", 10000.0))
        cos, sin = _build_rope_tables(self.head_dim, base, n_vq + 1)
        self.register_buffer("rotary_cos", cos.half(), persistent=True)
        self.register_buffer("rotary_sin", sin.half(), persistent=True)
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32
        self.register_buffer(
            "empty_key",
            torch.zeros(1, self.num_heads, self.head_dim, 0, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_value",
            torch.zeros(1, self.num_heads, 0, self.head_dim, dtype=kv_dtype),
            persistent=False,
        )

    def position_zero(self, hidden_states):
        return self.transformer(
            *([self.empty_key] * self.num_layers),
            *([self.empty_value] * self.num_layers),
            hidden_states,
            self.rotary_cos[:, :1].float(),
            self.rotary_sin[:, :1].float(),
        )

    def position(self, predictor_kv, hidden_states, position):
        position_plus = position + 1
        return self.transformer(
            *predictor_kv,
            hidden_states,
            self.rotary_cos[:, position:position_plus].float(),
            self.rotary_sin[:, position:position_plus].float(),
        )


class MOSS_MAIN_PREFILL_STRATEGY(torch.nn.Module):
    def __init__(self, packed_embed, main_core, predictor_core, text_head, strategy):
        super().__init__()
        self.packed_embed = packed_embed.half()
        self.main_core = main_core
        self.predictor_core = predictor_core
        self.text_head = text_head
        self.strategy = strategy
        self.text_sampler = MOSS_SAMPLER(2)

    def forward(self, *args):
        input_ids = args[0]
        main_outputs = self.main_core.prefill(self.packed_embed(input_ids).float())
        main_kv_count = self.main_core.num_layers * 2
        main_kv = main_outputs[:main_kv_count]
        global_hidden = main_outputs[-2]
        kv_seq_len = main_outputs[-1]
        predictor_outputs = self.predictor_core.position_zero(global_hidden)
        predictor_kv = predictor_outputs[:self.predictor_core.num_layers * 2]
        text_logits = self.text_head(predictor_outputs[-1])
        if self.strategy == "sampling":
            decision = self.text_sampler.sample(
                text_logits,
                args[1],
                args[2],
                args[3],
            )
        else:
            decision = torch.argmax(text_logits, dim=-1).int()
        return *main_kv, *predictor_kv, decision, kv_seq_len


class MOSS_PREDICTOR_FRAME_STRATEGY(torch.nn.Module):
    def __init__(self, packed_embed, predictor_core, audio_heads, strategy, audio_vocab_size):
        super().__init__()
        self.packed_embed = packed_embed
        self.predictor_core = predictor_core
        self.audio_heads = audio_heads
        self.strategy = strategy
        self.num_channels = len(audio_heads)
        self.penalty = MOSS_REPETITION_PENALTY()
        self.sampler = MOSS_SAMPLER(audio_vocab_size)

    def forward(self, *args):
        kv_count = self.predictor_core.num_layers * 2
        predictor_kv = list(args[:kv_count])
        generated_codec = args[kv_count]
        strategy_args = args[kv_count + 1:]
        hidden_states = self.packed_embed.assistant_slot(generated_codec)
        tokens = []
        for channel in range(self.num_channels):
            outputs = self.predictor_core.position(predictor_kv, hidden_states, channel + 1)
            predictor_kv = list(outputs[:kv_count])
            logits = self.audio_heads[channel](outputs[-1])
            history = generated_codec[:, :, channel]
            if self.strategy == "greedy":
                token = torch.argmax(logits, dim=-1).int()
            elif self.strategy == "penalty_greedy":
                token = torch.argmax(self.penalty(logits, history, strategy_args[0]), dim=-1).int()
            else:
                token = self.sampler(
                    logits,
                    strategy_args[0],
                    strategy_args[1],
                    strategy_args[2],
                    strategy_args[3],
                    history,
                )
            tokens.append(token)
            if channel + 1 < self.num_channels:
                hidden_states = self.packed_embed.embed_feedback(token.view(1, 1), channel)
        frame_codec_ids = torch.stack(tokens, dim=1)
        generated_codec = torch.cat([generated_codec, frame_codec_ids.unsqueeze(1)], dim=1)
        return frame_codec_ids, generated_codec


class MOSS_MAIN_DECODE_DECISION_STRATEGY(torch.nn.Module):
    def __init__(self, packed_embed, main_core, predictor_core, text_head, strategy):
        super().__init__()
        self.packed_embed = packed_embed
        self.main_core = main_core
        self.predictor_core = predictor_core
        self.text_head = text_head
        self.strategy = strategy
        self.text_sampler = MOSS_SAMPLER(2)

    def forward(self, *args):
        main_kv_count = self.main_core.num_layers * 2
        main_kv = args[:main_kv_count]
        frame_codec_ids = args[main_kv_count]
        kv_seq_len = args[main_kv_count + 1]
        strategy_args = args[main_kv_count + 2:]
        main_outputs = self.main_core.decode(
            main_kv,
            self.packed_embed.embed_frame(frame_codec_ids),
            kv_seq_len,
        )
        main_kv = main_outputs[:main_kv_count]
        global_hidden = main_outputs[-2]
        kv_seq_len = main_outputs[-1]
        predictor_outputs = self.predictor_core.position_zero(global_hidden)
        predictor_kv = predictor_outputs[:self.predictor_core.num_layers * 2]
        text_logits = self.text_head(predictor_outputs[-1])
        if self.strategy == "sampling":
            decision = self.text_sampler.sample(
                text_logits,
                strategy_args[0],
                strategy_args[1],
                strategy_args[2],
            )
        else:
            decision = torch.argmax(text_logits, dim=-1).int()
        return *main_kv, *predictor_kv, decision, kv_seq_len


def build_model_metadata(*sections):
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
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, onnx_path)


def replace_onnx_metadata(onnx_path, metadata):
    """Replace the metadata carrier contract without loading external weights."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save(model, onnx_path)


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


# KV keys grow on axis 3; values grow on axis 2.
KV_SPECS = [('key', 3), ('value', 2)]


def get_kv_io(kv_tensors, num_layers, out_seq_axis='kv_seq_len'):
    inputs, in_names, out_names, axes = [], [], [], {}
    for name, dim in KV_SPECS:
        tensor = kv_tensors[name]
        for i in range(num_layers):
            in_name, out_name = f'in_{name}_{i}', f'out_{name}_{i}'
            inputs.append(tensor)
            in_names.append(in_name)
            out_names.append(out_name)
            axes[in_name]  = {dim: 'history_len'}
            axes[out_name] = {dim: out_seq_axis}
    return inputs, in_names, out_names, axes


def run_compact_strategy_export():
    print('Compact strategy export start ...')
    with torch.inference_mode():
        for path in onnx_folder.glob("*.onnx*"):
            if path.is_file():
                path.unlink()
        for path in raw_onnx_folder.glob("*.onnx*"):
            if path.is_file():
                path.unlink()

        model = load_moss_model(download_path)

        cfg        = model.config
        gpt2_cfg   = cfg.gpt2_config
        hidden     = int(gpt2_cfg.hidden_size)
        num_heads  = int(gpt2_cfg.num_attention_heads)
        head_dim   = hidden // num_heads
        num_layers_main = int(gpt2_cfg.n_layer)
        num_layers_pred = len(model.local_transformer.h)
        n_vq        = int(cfg.n_vq)
        audio_vocab = int(cfg.audio_vocab_size)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)

        def encode_fixed(active_tokenizer, text):
            return list(active_tokenizer.encode(text, add_special_tokens=False))

        user_prompt_prefix_token_ids = (
            [int(cfg.im_start_token_id)]
            + encode_fixed(tokenizer, "user\n")
            + encode_fixed(tokenizer, "<user_inst>\n- Reference(s):\n")
        )
        user_prompt_after_reference_token_ids = encode_fixed(
            tokenizer,
            "\n- Instruction:\nNone\n- Tokens:\nNone\n- Quality:\nNone\n"
            "- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n"
        )
        assistant_prompt_prefix_token_ids = (
            encode_fixed(tokenizer, "\n</user_inst>")
            + [int(cfg.im_end_token_id)]
            + encode_fixed(tokenizer, "\n")
            + [int(cfg.im_start_token_id)]
            + encode_fixed(tokenizer, "assistant\n")
        )
        no_reference_token_ids = encode_fixed(tokenizer, "None")
        del tokenizer, encode_fixed

        trace_prompt_len = 8
        kv_dtype = torch.float16 if USE_F16_KV else torch.float32

        onnx_metadata = build_model_metadata(
            {
                "graph_layout": "strategy_prefill_decode_step",
                "in_sample_rate": IN_SAMPLE_RATE,
                "out_sample_rate": OUT_SAMPLE_RATE,
                "max_seq_len": MAX_SEQ_LEN,
                "use_f16_kv": USE_F16_KV,
                "compute_in_f32": COMPUTE_IN_F32,
                "shared_initializer_model_file": SHARED_MODEL_NAME,
                "shared_initializer_data_file": SHARED_DATA_NAME,
                "model_file_name_audio_encoder": Path(onnx_model_Audio_Encoder).name,
                "model_file_name_audio_decoder": Path(onnx_model_Audio_Decoder).name,
                **{
                    f"model_file_name_main_prefill_{strategy}": Path(
                        onnx_model_Main_Prefill[strategy]
                    ).name
                    for strategy in DECODE_STRATEGIES
                },
                **{
                    f"model_file_name_decode_step_{strategy}": Path(
                        onnx_model_Decode_Step[strategy]
                    ).name
                    for strategy in DECODE_STRATEGIES
                },
                "audio_vocab_size": audio_vocab,
            },
            {
                "audio_start_token_id": cfg.audio_start_token_id,
                "audio_end_token_id": cfg.audio_end_token_id,
                "audio_user_slot_token_id": cfg.audio_user_slot_token_id,
                "audio_assistant_slot_token_id": cfg.audio_assistant_slot_token_id,
                "audio_pad_token_id": cfg.audio_pad_token_id,
                "continue_decision_id": 0,
                "stop_decision_id": 1,
                "user_prompt_prefix_token_ids": user_prompt_prefix_token_ids,
                "user_prompt_after_reference_token_ids": (
                    user_prompt_after_reference_token_ids
                ),
                "assistant_prompt_prefix_token_ids": assistant_prompt_prefix_token_ids,
                "no_reference_token_ids": no_reference_token_ids,
            },
        )

        packed_embed = MOSS_EMBED(model)
        main_core = MOSS_MAIN_COMPACT_CORE(model, MAX_SEQ_LEN)
        predictor_core = MOSS_PREDICTOR_COMPACT_CORE(model, n_vq)
        text_head = MOSS_TEXT_HEAD(model)
        audio_heads = torch.nn.ModuleList([
            MOSS_AUDIO_HEAD(model, channel) for channel in range(n_vq)
        ])

        main_in_names = (
            [f'in_main_key_{index}' for index in range(num_layers_main)]
            + [f'in_main_value_{index}' for index in range(num_layers_main)]
        )
        main_out_names = (
            [f'out_main_key_{index}' for index in range(num_layers_main)]
            + [f'out_main_value_{index}' for index in range(num_layers_main)]
        )
        predictor_in_names = (
            [f'in_predictor_key_{index}' for index in range(num_layers_pred)]
            + [f'in_predictor_value_{index}' for index in range(num_layers_pred)]
        )
        predictor_out_names = (
            [f'out_predictor_key_{index}' for index in range(num_layers_pred)]
            + [f'out_predictor_value_{index}' for index in range(num_layers_pred)]
        )
        main_kv_axes = {}
        for index in range(num_layers_main):
            main_kv_axes[f'in_main_key_{index}'] = {3: 'history_len'}
            main_kv_axes[f'out_main_key_{index}'] = {3: 'kv_seq_len'}
            main_kv_axes[f'in_main_value_{index}'] = {2: 'history_len'}
            main_kv_axes[f'out_main_value_{index}'] = {2: 'kv_seq_len'}

        main_kv_inputs = (
            [torch.zeros(1, num_heads, head_dim, trace_prompt_len, dtype=kv_dtype)] * num_layers_main
            + [torch.zeros(1, num_heads, trace_prompt_len, head_dim, dtype=kv_dtype)] * num_layers_main
        )
        predictor_kv_inputs = (
            [torch.zeros(1, num_heads, head_dim, 1, dtype=kv_dtype)] * num_layers_pred
            + [torch.zeros(1, num_heads, 1, head_dim, dtype=kv_dtype)] * num_layers_pred
        )
        input_ids = torch.zeros(1, trace_prompt_len, n_vq + 1, dtype=torch.int32)
        frame_codec_ids = torch.zeros(1, n_vq, dtype=torch.int32)
        generated_codec = torch.zeros(1, 2, n_vq, dtype=torch.int32)
        kv_seq_len = torch.tensor([trace_prompt_len], dtype=torch.int64)
        text_temperature = torch.tensor([0.8], dtype=torch.float32)
        text_top_k = torch.tensor([2], dtype=torch.int32)
        text_top_p = torch.tensor([0.95], dtype=torch.float32)
        audio_temperature = torch.tensor([0.8], dtype=torch.float32)
        audio_top_k = torch.tensor([min(25, audio_vocab)], dtype=torch.int32)
        audio_top_p = torch.tensor([0.95], dtype=torch.float32)
        audio_repetition_penalty = torch.tensor([1.2], dtype=torch.float32)

        from Rewrite_MOSS_Sampler_TopK import rewrite_sampler_dynamic_topk

        def export_component(
            module,
            args,
            final_path,
            input_names,
            output_names,
            dynamic_axes,
            expected_sampler_matches=0,
        ):
            export_path = (
                str(raw_onnx_folder / Path(final_path).name)
                if expected_sampler_matches
                else final_path
            )
            torch.onnx.export(
                module,
                tuple(args),
                export_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                dynamo=False,
            )
            write_onnx_metadata(export_path, onnx_metadata)
            if expected_sampler_matches:
                rewrite = rewrite_sampler_dynamic_topk(
                    export_path,
                    final_path,
                    expected_match_count=expected_sampler_matches,
                )
                print(
                    f"[Sampler rewrite] {Path(final_path).name}: {rewrite['matched']} match(es), "
                    f"{rewrite['raw_nodes']} -> {rewrite['final_nodes']} nodes."
                )
        for strategy in DECODE_STRATEGIES:
            prefill = MOSS_MAIN_PREFILL_STRATEGY(
                packed_embed,
                main_core,
                predictor_core,
                text_head,
                strategy,
            )
            prefill_args = [input_ids]
            prefill_inputs = ['input_ids']
            prefill_axes = {
                'input_ids': {1: 'prompt_len'},
                **{name: axes for name, axes in main_kv_axes.items() if name.startswith('out_')},
            }
            if strategy == 'sampling':
                prefill_args.extend([text_temperature, text_top_k, text_top_p])
                prefill_inputs.extend(['text_temperature', 'text_top_k', 'text_top_p'])
            export_component(
                prefill,
                prefill_args,
                onnx_model_Main_Prefill[strategy],
                prefill_inputs,
                main_out_names + predictor_out_names + ['decision', 'kv_seq_len'],
                prefill_axes,
                expected_sampler_matches=1 if strategy == 'sampling' else 0,
            )
            del prefill

            predictor_frame = MOSS_PREDICTOR_FRAME_STRATEGY(
                packed_embed,
                predictor_core,
                audio_heads,
                strategy,
                audio_vocab,
            )
            predictor_args = predictor_kv_inputs + [generated_codec]
            predictor_inputs = predictor_in_names + ['generated_codec_in']
            predictor_axes = {
                'generated_codec_in': {1: 'generated_frames'},
                'generated_codec': {1: 'generated_frames_out'},
            }
            if strategy == 'penalty_greedy':
                predictor_args.append(audio_repetition_penalty)
                predictor_inputs.append('audio_repetition_penalty')
            elif strategy == 'sampling':
                predictor_args.extend([
                    audio_temperature,
                    audio_top_k,
                    audio_top_p,
                    audio_repetition_penalty,
                ])
                predictor_inputs.extend([
                    'audio_temperature',
                    'audio_top_k',
                    'audio_top_p',
                    'audio_repetition_penalty',
                ])
            export_component(
                predictor_frame,
                predictor_args,
                onnx_model_Predictor_Frame[strategy],
                predictor_inputs,
                ['frame_codec_ids', 'generated_codec'],
                predictor_axes,
                expected_sampler_matches=n_vq if strategy == 'sampling' else 0,
            )
            del predictor_frame

            main_decode = MOSS_MAIN_DECODE_DECISION_STRATEGY(
                packed_embed,
                main_core,
                predictor_core,
                text_head,
                strategy,
            )
            main_args = main_kv_inputs + [frame_codec_ids, kv_seq_len]
            main_inputs = main_in_names + ['frame_codec_ids', 'kv_seq_len']
            main_axes = dict(main_kv_axes)
            if strategy == 'sampling':
                main_args.extend([text_temperature, text_top_k, text_top_p])
                main_inputs.extend(['text_temperature', 'text_top_k', 'text_top_p'])
            export_component(
                main_decode,
                main_args,
                onnx_model_Main_Decode[strategy],
                main_inputs,
                main_out_names + predictor_out_names + ['decision', 'kv_seq_len_out'],
                main_axes,
                expected_sampler_matches=1 if strategy == 'sampling' else 0,
            )
            del main_decode

        decode_steps = build_decode_step_graphs(onnx_folder, DECODE_STRATEGIES)
        print(f"[DecodeStep] Built {len(decode_steps)} strategy graph(s); component graphs removed.")

        del packed_embed, main_core, predictor_core, text_head, audio_heads, model
        gc.collect()

        audio_model = load_audio_tokenizer(audio_tokenizer_path)
        audio_cfg   = audio_model.config

        n_quantizers    = int(audio_model.quantizer.num_quantizers)
        number_channels = int(audio_model.number_channels)
        sample_rate     = int(audio_model.sampling_rate)
        downsample_rate = int(audio_model.downsample_rate)
        scaled_samples_per_frame = downsample_rate * OUT_SAMPLE_RATE / sample_rate
        audio_metadata = build_model_metadata(
            {
                "samples_per_frame_per_channel": int(scaled_samples_per_frame),
            },
        )
        onnx_metadata.update(audio_metadata)

        sr_scale      = float(sample_rate / IN_SAMPLE_RATE)
        dummy_samples = max(1, int(round((downsample_rate * 10) / sr_scale)))
        waveform      = torch.zeros(
            (1, number_channels, dummy_samples),
            dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()],
        )
        torch.onnx.export(
            AUDIO_ENCODER(audio_model, IN_SAMPLE_RATE),
            (waveform,),
            onnx_model_Audio_Encoder,
            input_names=['prompt_audio'],
            output_names=['audio_codes', 'audio_code_lengths'],
            dynamic_axes={
                'prompt_audio':       {2: 'samples'},
                'audio_codes':        {1: 'frames'},
            },
            opset_version=OPSET,
            dynamo=False
        )
        del waveform

        dummy_frames = 16
        decoder_codes = torch.randint(
            0,
            int(audio_cfg.quantizer_kwargs["codebook_size"]),
            (1, dummy_frames, n_quantizers),
            dtype=torch.int32,
        )
        torch.onnx.export(
            AUDIO_DECODER(audio_model),
            (decoder_codes,),
            onnx_model_Audio_Decoder,
            input_names=['generated_codec'],
            output_names=['waveform'],
            dynamic_axes={
                'generated_codec': {1: 'frames'},
                'waveform':    {2: 'samples'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del decoder_codes, audio_model
        gc.collect()

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
        for target in sorted(onnx_folder.glob("*.onnx")):
            write_onnx_metadata(str(target), onnx_metadata)
            written += 1
        for target in sorted(raw_onnx_folder.glob("*.onnx")):
            write_onnx_metadata(str(target), onnx_metadata)
        print(f"[Metadata] Stamped {len(onnx_metadata)} keys into {written} final ONNX graph(s).")

        shared_stats = bundle_shared_initializers(onnx_folder, metadata=onnx_metadata)
        replace_onnx_metadata(onnx_model_Metadata, onnx_metadata)
        print(
            f"[Shared weights] {shared_stats['initializer_references']} references -> "
            f"{shared_stats['unique_initializers']} unique tensors; "
            f"deduplicated {shared_stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB."
        )
        shutil.rmtree(raw_onnx_folder)
        print(f"[Cleanup] Deleted temporary raw artifacts from {raw_onnx_folder}.")

    print('\nCompact strategy export done!')

run_compact_strategy_export()
print('\nStart running the MOSS-TTS Nano demo via Inference_MOSS_TTS_Nano_ONNX.py ...')
raise SystemExit(subprocess.call([
    sys.executable,
    str(script_dir / "Inference_MOSS_TTS_Nano_ONNX.py"),
    "--onnx-folder",
    str(onnx_folder),
]))
