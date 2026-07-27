import gc
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import lightning.pytorch.loggers as lightning_loggers

if not hasattr(lightning_loggers, "NeptuneLogger"):
    class _UnavailableNeptuneLogger:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("Install the optional 'neptune' package to enable NeptuneLogger.")

    lightning_loggers.NeptuneLogger = _UnavailableNeptuneLogger

from hydra.utils import instantiate
from nemo.collections.tts.losses.audio_codec_loss import (
    FeatureMatchingLoss,
    MultiResolutionMelLoss,
    MultiResolutionSTFTLoss,
    RelativeFeatureMatchingLoss,
    SISDRLoss,
    TimeDomainLoss,
)
from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.modules.audio_codec_modules import ResNetSpeakerEncoder
from nemo.collections.common.parts.utils import Snake
from nemo.collections.tts.modules.common import GaussianDropout
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.utils import logging, model_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

from Shared_Weights import SHARED_DATA_NAME, SHARED_MODEL_NAME, bundle_shared_initializers

script_dir = Path(__file__).resolve().parent
onnx_folder = script_dir / "KaniTTS_ONNX"
onnx_folder.mkdir(parents=True, exist_ok=True)

# ── User configuration ───────────────────────────────────────────────────────
downloads_folder = Path.home() / "Downloads"
path_kani    = str(downloads_folder / "kani-tts-370m")                          # Set the folder path where the [kani-tts-370m, kani-tts-400m] project downloaded.
path_codec   = str(downloads_folder / "nemo-nano-codec-22khz-0.6kbps-12.5fps" / "nemo-nano-codec-22khz-0.6kbps-12.5fps.nemo") # The audio codec download path. URL: https://huggingface.co/nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps
MAX_SEQ_LEN = 1024              # Maximum prompt + decode sequence length.
OUT_SAMPLE_RATE = 22050         # Public generated-waveform ONNX output rate.
OUT_AUDIO_DTYPE = "F32"         # "F16" | "F32" | "INT16".
MODEL_SAMPLE_RATE = 22050       # Native NeMo codec sample rate; do not edit.

PREVENT_F16_OVERFLOW = False    # Set True when quantizing to Q4F16 / Q8F16 / F16.
USE_FLOAT16_KV    = True        # Store KV cache in float16 (less memory bandwidth).
USE_FLOAT16_CODEC = True        # Run NeMo Codec in float16.
COMPUTE_IN_F32    = False       # Precision at the F16-cache arithmetic points (only affects USE_FLOAT16_KV=True).
                                #   True  = upcast the stored F16 K/V cache and compute attention in float32.
                                #   False = compute attention in float16, downcasting the small/temporary
                                #           operands (query, softmax weights) so the F16 cache is never upcast.
                                #   The K/V cache storage/output dtype stays float16 in both cases.

# Exact channel permutations that improve later weight-only quantization.
REORDER_DOWNPROJ_FOR_QUANT = True
REORDER_OPROJ_FOR_QUANT    = True
REORDER_KEY                = "absmean"  # "absmean" | "L4" | "rms" | "std"

# ── Fixed export contract ─────────────────────────────────────────────────────
DECODE_STRATEGIES = ("greedy", "penalty_greedy", "sampling")
STOP_TOKENS = ("<custom_token_2>",)
PROMPT_PREFIX_TOKENS = ("<custom_token_3>", "<|startoftext|>")
PROMPT_SUFFIX_TOKENS = ("<|endoftext|>", "<custom_token_4>")
OPSET = 20

_OUTPUT_AUDIO_DTYPES = {"F16", "F32", "INT16"}


onnx_model_Main_Prefill = {
    strategy: str(onnx_folder / f"KaniTTS_MainPrefill_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Decode_Step = {
    strategy: str(onnx_folder / f"KaniTTS_DecodeStep_{strategy}.onnx")
    for strategy in DECODE_STRATEGIES
}
onnx_model_Codec = str(onnx_folder / 'KaniTTS_Codec.onnx')
onnx_model_Metadata = str(onnx_folder / 'KaniTTS_Metadata.onnx')


def _restore_local_nemo_model(
    cls,
    model_name,
    refresh_cache=False,
    override_config_path=None,
    map_location=None,
    strict=True,
    return_config=False,
    trainer=None,
    save_restore_connector=None,
    return_model_file=False,
):
    if save_restore_connector is None:
        save_restore_connector = SaveRestoreConnector()

    if return_model_file:
        return model_name

    return cls.restore_from(
        restore_path=model_name,
        override_config_path=override_config_path,
        map_location=map_location,
        strict=strict,
        return_config=return_config,
        trainer=trainer,
        save_restore_connector=save_restore_connector,
    )


def _audio_codec_model_init(self, cfg, trainer=None):
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    cfg = model_utils.maybe_update_config_version(cfg)
    self.world_size = 1
    if trainer is not None:
        self.world_size = trainer.num_nodes * trainer.num_devices

    super(AudioCodecModel, self).__init__(cfg=cfg, trainer=trainer)

    self.sample_rate = cfg.sample_rate
    self.samples_per_frame = cfg.samples_per_frame

    self.disc_updates_per_period = cfg.get("disc_updates_per_period", 1)
    self.disc_update_period = cfg.get("disc_update_period", 1)
    if self.disc_updates_per_period > self.disc_update_period:
        raise ValueError(
            f'Number of discriminator updates ({self.disc_updates_per_period}) per period must be less or equal to the configured period ({self.disc_update_period})'
        )

    self.audio_encoder = instantiate(cfg.audio_encoder)

    encoder_noise_stdev = cfg.get("encoder_noise_stdev", 0.0)
    if encoder_noise_stdev:
        self.encoder_noise = GaussianDropout(stdev=encoder_noise_stdev)
    else:
        self.encoder_noise = None

    if "vector_quantizer" in cfg:
        self.vector_quantizer = instantiate(cfg.vector_quantizer)

        vq_output_types = list(self.vector_quantizer.output_types.keys())
        if len(vq_output_types) == 3 and vq_output_types[-1] == 'commit_loss':
            self.vector_quantizer_has_commit_loss = True
            logging.info('Vector quantizer supports commit loss.')
        else:
            self.vector_quantizer_has_commit_loss = False
            logging.info('Vector quantizer does not support commit loss.')
    else:
        logging.warning('Vector quantizer will not be used.')
        self.vector_quantizer = None

    self.audio_decoder = instantiate(cfg.audio_decoder)

    loss_resolutions = cfg.loss_resolutions
    mel_loss_dims = cfg.get("mel_loss_dims")
    mel_loss_log_guard = cfg.get("mel_loss_log_guard", 1.0)
    self.mel_loss_l1_scale = cfg.get("mel_loss_l1_scale", 1.0)
    self.mel_loss_l2_scale = cfg.get("mel_loss_l2_scale", 1.0)
    self.mel_loss_fn = MultiResolutionMelLoss(
        sample_rate=self.sample_rate,
        mel_dims=mel_loss_dims,
        resolutions=loss_resolutions,
        log_guard=mel_loss_log_guard,
    )

    stft_loss_log_guard = cfg.get("stft_loss_log_guard", 1.0)
    self.stft_loss_scale = cfg.get("stft_loss_scale", 0.0)
    self.stft_loss_fn = MultiResolutionSTFTLoss(
        resolutions=loss_resolutions,
        log_guard=stft_loss_log_guard,
    )

    self.time_domain_loss_scale = cfg.get("time_domain_loss_scale", 1.0)
    self.si_sdr_loss_scale = cfg.get("si_sdr_loss_scale", 0.0)
    self.time_domain_loss_fn = TimeDomainLoss()
    self.si_sdr_loss_fn = SISDRLoss()

    self.gen_loss_scale = cfg.get("gen_loss_scale", 1.0)
    self.feature_loss_scale = cfg.get("feature_loss_scale", 1.0)
    self.gen_loss_fn = instantiate(cfg.generator_loss)
    self.disc_loss_fn = instantiate(cfg.discriminator_loss)

    feature_loss_type = cfg.get("feature_loss_type", "relative")
    if feature_loss_type == "relative":
        self.feature_loss_fn = RelativeFeatureMatchingLoss()
    elif feature_loss_type == "absolute":
        self.feature_loss_fn = FeatureMatchingLoss()
    else:
        raise ValueError(f'Unknown feature loss type {feature_loss_type}.')

    if self.vector_quantizer:
        self.commit_loss_scale = cfg.get("commit_loss_scale", 1.0)
    else:
        self.commit_loss_scale = 0.0

    if self.commit_loss_scale > 0 and not self.vector_quantizer_has_commit_loss:
        raise ValueError('Commit loss is enabled but the quantizer does not support it.')

    self.use_scl_loss = cfg.get("use_scl_loss", False)
    self.scl_loss_scale = cfg.get("scl_loss_scale", False)
    if self.use_scl_loss:
        self.speaker_encoder = ResNetSpeakerEncoder()
        self.speaker_encoder.load_checkpoint(
            "https://huggingface.co/Edresson/Speaker_Encoder_H_ASP/resolve/main/pytorch_model.bin", strict=False
        )
        self.speaker_encoder.freeze()
        print("Speaker encoder loaded and frozen !!")

    self.use_asr_consitency_loss = False
    self.acl_loss_scale = False
    self.log_config = cfg.get("log_config", None)
    self.lr_schedule_interval = None
    self.automatic_optimization = False


def _audio_codec_load_state_dict(self, state_dict, strict=True):
    for key in list(state_dict.keys()):
        if self.use_scl_loss and "speaker_encoder." in key:
            del state_dict[key]
        if "discriminator" in key and ".slm_model.ssl_model." in key:
            del state_dict[key]

    super(AudioCodecModel, self).load_state_dict(state_dict, strict=False)


AudioCodecModel.from_pretrained = classmethod(_restore_local_nemo_model)
AudioCodecModel.__init__ = _audio_codec_model_init
AudioCodecModel.load_state_dict = _audio_codec_load_state_dict


class APPLY_PENALTY(torch.nn.Module):
    """Apply a repetition penalty over the most recent `penalty_range` tokens."""

    def __init__(self):
        super(APPLY_PENALTY, self).__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized      = logits.gather(1, target_indices) * penalty_value
        logits         = logits.scatter(1, target_indices, penalized)
        return logits


class SIGN_AWARE_REPETITION_PENALTY(torch.autograd.Function):
    """Apply sampling repetition penalty while preserving int32 ONNX indices."""

    @staticmethod
    def forward(ctx, logits, repetition_penalty, previous_ids):
        previous_ids_long = previous_ids.long()
        previous_logits = torch.gather(logits, 1, previous_ids_long)
        previous_scores = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits / repetition_penalty,
        )
        return torch.scatter(logits, 1, previous_ids_long, previous_scores)

    @staticmethod
    def symbolic(g, logits, repetition_penalty, previous_ids):
        previous_logits = g.op("GatherElements", logits, previous_ids, axis_i=1)
        zero = g.op("Constant", value_t=torch.tensor(0.0, dtype=torch.float32))
        previous_scores = g.op(
            "Where",
            g.op("Less", previous_logits, zero),
            g.op("Mul", previous_logits, repetition_penalty),
            g.op("Div", previous_logits, repetition_penalty),
        )
        return g.op("ScatterElements", logits, previous_ids, previous_scores, axis_i=1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    """Top-K/Top-P categorical sampling with sign-aware repetition penalty."""

    def __init__(self, vocab_size):
        super().__init__()
        self.register_buffer("one", torch.tensor([1], dtype=torch.int64), persistent=False)
        self.register_buffer("vocab_size", torch.tensor([vocab_size], dtype=torch.int64), persistent=False)

    def sample(self, scores, temperature, top_k, top_p, greedy_scores):
        top_k = torch.minimum(torch.maximum(top_k, self.one), self.vocab_size)
        sorted_scores, sorted_indices = torch.topk(scores, k=top_k, dim=-1, largest=True, sorted=True)
        sorted_probabilities = torch.softmax(sorted_scores / temperature, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        keep = (cumulative_probabilities - sorted_probabilities) <= top_p

        kept_mass = torch.where(keep, cumulative_probabilities, 0.0).amax(dim=-1, keepdim=True)
        threshold = torch.rand_like(kept_mass) * kept_mass
        winner = torch.argmax((cumulative_probabilities >= threshold).int(), dim=-1, keepdim=True)
        sampled_id = torch.gather(sorted_indices, 1, winner).int()
        greedy_id = torch.argmax(greedy_scores, dim=-1, keepdim=True).int()
        return torch.where(top_k == self.one, greedy_id, sampled_id)

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        scores = SIGN_AWARE_REPETITION_PENALTY.apply(logits, repetition_penalty, previous_ids)
        sampled_id = self.sample(scores, temperature, top_k, top_p, logits)
        save_ids = torch.cat([previous_ids, sampled_id], dim=-1)
        return sampled_id, save_ids


class KANITTS_TOKEN_STRATEGY(torch.nn.Module):
    """Select and append one token using exactly one configured strategy."""

    def __init__(self, strategy, vocab_size):
        super().__init__()
        if strategy not in DECODE_STRATEGIES:
            raise ValueError(f"Unsupported decode strategy: {strategy!r}")
        self.strategy = strategy
        self.penalty = APPLY_PENALTY()
        self.sampling = TOPK_TOPP_SAMPLING(vocab_size)

    def forward(self, logits, save_ids, *controls):
        if self.strategy == "penalty_greedy":
            penalty_value, penalty_range = controls
            penalized_logits = self.penalty(logits, save_ids, penalty_value, penalty_range)
            use_penalty = torch._shape_as_tensor(save_ids)[1:2] >= penalty_range
            logits = torch.where(use_penalty, penalized_logits, logits)
        elif self.strategy == "sampling":
            return self.sampling(logits, *controls, save_ids)

        next_token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return next_token, torch.cat([save_ids, next_token], dim=-1)


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
    - Float32 rotary buffers with [-sin, sin] pattern (no runtime casts, compatible with flip)
    - Precomputed additive causal mask shared by all attention layers
      - Fused QKV projection with absorbed operator_norm weights
    - Fused FFN gate/up projection with absorbed ffn_norm weights
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
        ffn_intermediate_size = kani_tts.model.layers[0].feed_forward.w1.out_features
        self.ffn_split_sizes = [ffn_intermediate_size, ffn_intermediate_size]
        self.kv_f16 = USE_FLOAT16_KV
        self.compute_in_f32 = COMPUTE_IN_F32
        self.register_buffer("overflow_scale", torch.tensor([0.01], dtype=torch.float32))

        # ── sum()-based RMS norm epsilon (eps_sum = hidden_size * eps) ────
        hidden_size = kani_tts.model.embed_tokens.embedding_dim
        self.hidden_size = hidden_size
        self.conv_split_sizes = [hidden_size] * 3
        conv_paddings = {
            layer.conv.conv.padding[0]
            for layer in kani_tts.model.layers
            if not layer.is_attention_layer
        }
        if len(conv_paddings) != 1:
            raise ValueError(f"Expected one shared conv padding, found {sorted(conv_paddings)}.")
        self.conv_padding = conv_paddings.pop()
        variance_epsilon = float(1e-5)
        hidden_rms_norm_eps = hidden_size * variance_epsilon
        qk_rms_norm_eps = head_dim * variance_epsilon
        if PREVENT_F16_OVERFLOW:
            hidden_rms_norm_eps *= 0.01 ** 2
            qk_rms_norm_eps *= 0.01 ** 2
        self.register_buffer("hidden_rms_norm_eps", torch.tensor([hidden_rms_norm_eps], dtype=torch.float32))
        self.register_buffer("qk_rms_norm_eps", torch.tensor([qk_rms_norm_eps], dtype=torch.float32))

        # ── Norm scale factors (compensate sum vs mean) ──────────────────
        norm_factor = float(hidden_size ** 0.5)
        qk_norm_factor = float(head_dim ** 0.5)
        scale_factor = float(head_dim ** -0.25)
        combined_qk_scale = scale_factor * qk_norm_factor  # = head_dim^0.25

        # ── Precompute float32 rotary embeddings with [-sin, sin] ────────
        position_ids = torch.arange(max_seq_len, dtype=torch.float32)
        rotary_embedding = getattr(kani_tts.model, "rotary_emb", None)
        if rotary_embedding is None:
            rotary_embedding = getattr(kani_tts.model, "pos_emb", None)
        if rotary_embedding is None:
            raise AttributeError("LFM2 model exposes neither rotary_emb nor pos_emb.")
        inv_freq = rotary_embedding.inv_freq
        freqs = torch.outer(position_ids, inv_freq)  # (max_seq_len, head_dim//2)
        attention_scaling = rotary_embedding.attention_scaling

        # cos: [cos, cos], sin: [-sin, sin] for flip()-based rotate_half
        cos_emb = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * attention_scaling
        sin_emb = torch.cat([-freqs.sin(), freqs.sin()], dim=-1) * attention_scaling

        # Shape: (1, max_seq_len, 1, 1, head_dim) for broadcast with (B, S, 1, qk_heads, D)
        self.register_buffer("cos_rotary_pos_emb", cos_emb.unsqueeze(0).unsqueeze(2).unsqueeze(3).half())
        self.register_buffer("sin_rotary_pos_emb", sin_emb.unsqueeze(0).unsqueeze(2).unsqueeze(3).half())

        causal_mask = torch.full((max_seq_len, max_seq_len), -128, dtype=torch.int8)
        causal_mask = torch.triu(causal_mask, diagonal=1).view(1, 1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        # ── Fuse weights ─────────────────────────────────────────────────
        self._fuse_weights(norm_factor, combined_qk_scale)
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)
        self.layers = self.kani_tts.model.layers
        del self.kani_tts

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

            # Detach the tied LM head before folding the final norm so the embedding
            # weights already exported by KANITTS_EMBED cannot be mutated.
            final_norm_weight = self.kani_tts.model.embedding_norm.weight.unsqueeze(0) * norm_factor
            original_lm_head = self.kani_tts.lm_head
            fused_lm_head = torch.nn.Linear(
                original_lm_head.in_features,
                original_lm_head.out_features,
                bias=original_lm_head.bias is not None,
                device=original_lm_head.weight.device,
                dtype=original_lm_head.weight.dtype,
            )
            fused_lm_head.weight.copy_(original_lm_head.weight)
            fused_lm_head.weight.mul_(final_norm_weight)
            if original_lm_head.bias is not None:
                fused_lm_head.bias.copy_(original_lm_head.bias)
            self.lm_head = fused_lm_head
            del self.kani_tts.lm_head
            del self.kani_tts.model.embed_tokens
            del self.kani_tts.model.embedding_norm

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
        del layer.operator_norm

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
        del layer.operator_norm

    def _fuse_ffn_norm(self, layer, norm_factor):
        """Fuse FFN gate/up projections and absorb ffn_norm.weight."""
        feed_forward = layer.feed_forward
        w1, w3 = feed_forward.w1, feed_forward.w3
        ffn_norm_weight = layer.ffn_norm.weight.unsqueeze(0) * norm_factor
        has_bias = w1.bias is not None or w3.bias is not None
        w13 = torch.nn.Linear(
            w1.in_features,
            w1.out_features + w3.out_features,
            bias=has_bias,
            device=w1.weight.device,
            dtype=w1.weight.dtype,
        )
        with torch.no_grad():
            w13.weight.copy_(torch.cat([w1.weight, w3.weight], dim=0) * ffn_norm_weight)
            if has_bias:
                zero_bias = torch.zeros(w1.out_features, device=w1.weight.device, dtype=w1.weight.dtype)
                w13.bias.copy_(torch.cat([
                    w1.bias if w1.bias is not None else zero_bias,
                    w3.bias if w3.bias is not None else zero_bias,
                ]))
        feed_forward.w13 = w13
        del feed_forward.w1, feed_forward.w3
        del layer.ffn_norm

    @staticmethod
    def _channel_score(weight, key, dims):
        absolute = weight.abs()
        if key == "rms":
            return weight.square().mean(dim=dims).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(dim=dims).pow(0.25)
        if key == "std":
            flattened = weight.movedim(dims, tuple(range(len(dims)))).reshape(-1, weight.shape[-1])
            return flattened.std(0)
        if key == "absmean":
            return absolute.mean(dim=dims)
        raise ValueError(f"Unsupported REORDER_KEY: {key!r}")

    def _reorder_downproj_for_quant(self, key):
        """Apply one exact FFN permutation to w13 rows/bias and w2 columns."""
        with torch.no_grad():
            for layer in self.kani_tts.model.layers:
                feed_forward = layer.feed_forward
                down_weight = feed_forward.w2.weight
                permutation = torch.argsort(self._channel_score(down_weight, key, (0,)))
                intermediate_size = feed_forward.w2.in_features
                gate_up_weight = feed_forward.w13.weight
                feed_forward.w13.weight.copy_(torch.cat([
                    gate_up_weight[:intermediate_size][permutation],
                    gate_up_weight[intermediate_size:][permutation],
                ], dim=0))
                if feed_forward.w13.bias is not None:
                    gate_up_bias = feed_forward.w13.bias
                    feed_forward.w13.bias.copy_(torch.cat([
                        gate_up_bias[:intermediate_size][permutation],
                        gate_up_bias[intermediate_size:][permutation],
                    ], dim=0))
                feed_forward.w2.weight.copy_(down_weight[:, permutation])

    def _reorder_oproj_for_quant(self, key):
        """Permute V channels and matching out-projection columns within each KV group."""
        heads_per_kv = self.num_heads // self.num_key_value_heads
        with torch.no_grad():
            for layer in self.kani_tts.model.layers:
                if not layer.is_attention_layer:
                    continue
                attention = layer.self_attn
                output_weight = attention.out_proj.weight
                output_by_head = output_weight.view(output_weight.shape[0], self.num_heads, self.head_dim)
                permutations = []
                for kv_head in range(self.num_key_value_heads):
                    grouped = output_by_head[:, kv_head * heads_per_kv:(kv_head + 1) * heads_per_kv]
                    permutations.append(torch.argsort(self._channel_score(grouped, key, (0, 1))))

                reordered_output = output_by_head.clone()
                for head in range(self.num_heads):
                    reordered_output[:, head] = output_by_head[:, head, permutations[head // heads_per_kv]]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                qkv_weight = attention.qkv.weight
                qkv_by_head = qkv_weight.view(self.total_qkv_heads, self.head_dim, -1).clone()
                for kv_head, permutation in enumerate(permutations):
                    qkv_by_head[self.qk_heads + kv_head] = qkv_by_head[self.qk_heads + kv_head][permutation]
                qkv_weight.copy_(qkv_by_head.reshape_as(qkv_weight))
                if attention.qkv.bias is not None:
                    qkv_bias = attention.qkv.bias.view(self.total_qkv_heads, self.head_dim).clone()
                    for kv_head, permutation in enumerate(permutations):
                        qkv_bias[self.qk_heads + kv_head] = qkv_bias[self.qk_heads + kv_head][permutation]
                    attention.qkv.bias.copy_(qkv_bias.reshape_as(attention.qkv.bias))

    # ══════════════════════════════════════════════════════════════════════
    # Optimized Primitives
    # ══════════════════════════════════════════════════════════════════════

    def _rms_norm(self, x, eps):
        """sum()-based RMS norm: x * rsqrt(sum(x^2) + eps_sum).
        Avoids division; eps_sum = hidden_size * eps compensates for sum vs mean.
        """
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)

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

        # Slice rotary embeddings already stored in their F32 compute dtype.
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()

        kv_count = 0
        conv_count = 0
        for layer in self.layers:

            if layer.is_attention_layer:
                # ── RMS norm (sum-based, weight already absorbed into QKV) ──
                hidden_states_norm = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

                # ── Fused QKV projection ────────────────────────────────────
                qkv = layer.self_attn.qkv(hidden_states_norm)
                qkv = qkv.reshape(1, -1, 1, self.total_qkv_heads, self.head_dim)
                qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)

                # ── QK RMS norm + fused weight ──────────────────────────────
                qk = self._rms_norm(qk, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight

                # ── Rotary embedding (flip-based) ───────────────────────────
                qk = qk * rotary_cos + self._rotate_half_qk(qk, 1) * rotary_sin

                # Earliest clean F16 cast (F16-compute mode only): rotary is the last
                # op forcing F32 (F32 cos/sin tables), so this is the earliest clean
                # cast point. Casting the q/k common ancestor once lets the following
                # split/reshape/permute run in F16, so the query reaches the matmul
                # already downcast and the large, growing K cache is never upcast.
                if self.kv_f16 and not self.compute_in_f32:
                    qk = qk.half()

                # ── Split Q and K ───────────────────────────────────────────
                q, k = torch.split(qk, self.qk_split_sizes, dim=-2)

                # ── Q reshape for GQA: (B, S, 1, H, D) → (B, KVH, G, S, D) ─
                q = q.reshape(1, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
                q = q.permute(0, 2, 3, 1, 4)

                # ── K/V to cache layout ─────────────────────────────────────
                k = k.permute(0, 3, 2, 4, 1)   # (B, KVH, 1, D, S)
                v = v.transpose(1, 3)           # (B, KVH, 1, S, D)

                # ── Optional F16 KV cache ───────────────────────────────────
                if self.kv_f16:
                    v = v.half()
                    if self.compute_in_f32:
                        k = k.half()   # F32 compute path still stores the key cache in F16.

                # ── Concatenate with KV cache ───────────────────────────────
                k = torch.cat((all_inputs[kv_count], k), dim=-1)
                v = torch.cat((all_inputs[kv_count + self.num_attn_layers], v), dim=-2)
                self.save_key[kv_count] = k
                self.save_value[kv_count] = v
                kv_count += 1

                # ── Attention (GQA via broadcast, no repeat needed) ─────────
                if self.kv_f16:
                    if self.compute_in_f32:
                        # Upcast the stored F16 cache at the use point (query is F32).
                        attn = torch.softmax(torch.matmul(q, k.float()), dim=-1)
                        attn_out = torch.matmul(attn, v.float())
                    else:
                        # F16 compute: query is already F16; keep the stored F16 cache
                        # and its F16 softmax weights instead of upcasting the cache.
                        attn = torch.softmax(torch.matmul(q, k), dim=-1)
                        attn_out = torch.matmul(attn, v).float()
                else:
                    attn = torch.softmax(torch.matmul(q, k), dim=-1)
                    attn_out = torch.matmul(attn, v)

                # ── Reshape and output projection ───────────────────────────
                attn_out = attn_out.permute(0, 3, 1, 2, 4).reshape(1, -1, self.o_proj_in_features)
                attn_out = layer.self_attn.out_proj(attn_out)

            else:
                # ── Conv layer: RMS norm (weight absorbed into in_proj) ─────
                hidden_states_norm = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)

                # ── Conv computation ────────────────────────────────────────
                BCx = layer.conv.in_proj(hidden_states_norm)
                B_val, C, x = torch.split(BCx, self.conv_split_sizes, dim=-1)
                Bx = B_val * x
                previous_conv_state = all_inputs[conv_count + self.num_key_value_layers]
                if self.kv_f16:
                    previous_conv_state = previous_conv_state.float()
                # Assemble state channels-last so ORT cannot propagate the projection transpose into Concat.
                conv_state = torch.cat([previous_conv_state.transpose(-1, -2), Bx], dim=-2)
                saved_conv_state = conv_state[:, -self.conv_padding:].transpose(-1, -2)
                if self.kv_f16:
                    saved_conv_state = saved_conv_state.half()
                self.save_conv[conv_count] = saved_conv_state
                conv_count += 1
                conv_out = layer.conv.conv(conv_state.transpose(-1, -2))
                conv_out = conv_out[..., -ids_len - self.conv_padding:-self.conv_padding]
                attn_out = layer.conv.out_proj(C * conv_out.transpose(-1, -2))

            # ── Residual + FFN ──────────────────────────────────────────
            hidden_states = hidden_states + attn_out
            ffn_input = self._rms_norm(hidden_states, self.hidden_rms_norm_eps)
            gate, up = torch.split(layer.feed_forward.w13(ffn_input), self.ffn_split_sizes, dim=-1)
            hidden_states = hidden_states + layer.feed_forward.w2(torch.nn.functional.silu(gate) * up)

        # ── Final projection (embedding_norm absorbed into lm_head) ─────
        hidden_states = self._rms_norm(hidden_states[:, -1], self.hidden_rms_norm_eps)
        logits = self.lm_head(hidden_states)
        return *self.save_key, *self.save_value, *self.save_conv, logits, kv_seq_len


class KANITTS_MAIN_PREFILL_STRATEGY(torch.nn.Module):
    """Embed a prompt, initialize recurrent state, and select the first token."""

    def __init__(self, embed, main_core, strategy, vocab_size):
        super().__init__()
        self.embed = embed
        self.main_core = main_core
        self.strategy_name = strategy
        self.strategy = KANITTS_TOKEN_STRATEGY(strategy, vocab_size)
        self.num_attn_layers = main_core.num_attn_layers
        self.num_conv_layers = main_core.num_conv_layers
        kv_dtype = torch.float16 if USE_FLOAT16_KV else torch.float32
        self.register_buffer(
            "empty_key",
            torch.zeros(1, main_core.num_key_value_heads, 1, main_core.head_dim, 0, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_value",
            torch.zeros(1, main_core.num_key_value_heads, 1, 0, main_core.head_dim, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer(
            "empty_conv",
            torch.zeros(1, main_core.hidden_size, main_core.conv_padding, dtype=kv_dtype),
            persistent=False,
        )
        self.register_buffer("zero_history_len", torch.zeros(1, dtype=torch.int64), persistent=False)

    def forward(self, input_ids, *controls):
        hidden_states = self.embed(input_ids)
        ids_len = torch._shape_as_tensor(hidden_states)[1:2]
        outputs = self.main_core(
            *([self.empty_key] * self.num_attn_layers),
            *([self.empty_value] * self.num_attn_layers),
            *([self.empty_conv] * self.num_conv_layers),
            hidden_states,
            self.zero_history_len,
            ids_len,
        )
        state_count = self.num_attn_layers * 2 + self.num_conv_layers
        logits = outputs[state_count]
        if self.strategy_name == "sampling":
            next_token = self.strategy.sampling.sample(logits, *controls, logits)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True).int()
        return *outputs[:state_count], next_token, outputs[state_count + 1]


class KANITTS_DECODE_STEP_STRATEGY(torch.nn.Module):
    """Advance all recurrent state and select one token in a single ORT graph call."""

    def __init__(self, embed, main_core, strategy, vocab_size):
        super().__init__()
        self.embed = embed
        self.main_core = main_core
        self.strategy = KANITTS_TOKEN_STRATEGY(strategy, vocab_size)
        self.state_count = main_core.num_attn_layers * 2 + main_core.num_conv_layers

    def forward(self, *args):
        states = args[:self.state_count]
        current_token = args[self.state_count]
        save_ids = args[self.state_count + 1]
        history_len = args[self.state_count + 2]
        controls = args[self.state_count + 3:]
        hidden_states = self.embed(current_token)
        ids_len = torch._shape_as_tensor(hidden_states)[1:2]
        outputs = self.main_core(*states, hidden_states, history_len, ids_len)
        next_token, save_ids_out = self.strategy(outputs[self.state_count], save_ids, *controls)
        return *outputs[:self.state_count], next_token, save_ids_out, outputs[self.state_count + 1]


class FIXED_RANK_SNAKE(torch.nn.Module):
    """Rank-3 Snake activation with immutable scales prepared before export."""

    def __init__(self, snake):
        super(FIXED_RANK_SNAKE, self).__init__()
        alpha = snake.alpha.detach().clone()
        self.register_buffer("alpha", alpha)
        self.register_buffer("inverse_alpha", (alpha + 1e-9).reciprocal())

    def forward(self, x):
        x = x.float()
        periodic = torch.sin(self.alpha * x)
        return x + self.inverse_alpha * (periodic * periodic)


class NEMO_CODEC(torch.nn.Module):
    """
    Optimized NeMo Codec decoder module.

    Optimizations applied:
            - Precomputed FSQ lookup table (replaces integer Div/Mod and pointwise dequantization)
            - Joint int32 Gather for all four codebooks
            - Identity sequence masks removed from the single full-length decode path
            - Static causal padding/cropping derived from immutable convolution geometry
            - Final audio length computed once from the token count and total upsample rate
    """

    CODEBOOK_SIZE = 4032

    def __init__(self, nemo_codec, tokeniser_length):
        super(NEMO_CODEC, self).__init__()
        self.tokeniser_length = tokeniser_length
        self.audio_tokens_start = self.tokeniser_length + 10
        self.codebook_size = self.CODEBOOK_SIZE
        self.register_buffer(
            "codebook_offsets",
            (torch.arange(4, dtype=torch.int32) * self.codebook_size + self.audio_tokens_start).view(1, 1, 4),
        )
        self.scale = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)

        # ── Immutable FSQ dequantization table: (code, FSQ dimension) ────────
        fsq_indices = torch.arange(self.codebook_size, dtype=torch.int32).unsqueeze(1)
        fsq_dim_base = torch.tensor([1, 9, 72, 576], dtype=torch.int32).unsqueeze(0)
        fsq_num_levels = torch.tensor([9, 8, 8, 7], dtype=torch.int32).unsqueeze(0)
        fsq_inv_scale = torch.tensor([0.25, 0.25, 0.25, 1.0 / 3.0], dtype=torch.float32).unsqueeze(0)
        fsq_codebook = ((fsq_indices // fsq_dim_base) % fsq_num_levels).float() * fsq_inv_scale - 1.0
        self.register_buffer("fsq_codebook", fsq_codebook)

        # ── Inline CausalHiFiGANDecoder ──
        decoder = nemo_codec.audio_decoder
        self.pre_conv = decoder.pre_conv
        self.activations = decoder.activations
        self.res_layers = decoder.res_layers
        self.up_sample_conv_layers = decoder.up_sample_conv_layers
        self.up_sample_rates = decoder.up_sample_rates
        self.total_up_sample_rate = math.prod(self.up_sample_rates)
        if self.total_up_sample_rate % 4:
            raise ValueError("Codec upsample rate must be divisible by the four interleaved codebooks.")
        self.samples_per_decode_id = self.total_up_sample_rate // 4
        self.post_activation = decoder.post_activation
        self.post_conv = decoder.post_conv

        def replace_snake_modules(parent):
            replaced = 0
            for child_name, child in list(parent.named_children()):
                if isinstance(child, Snake):
                    setattr(parent, child_name, FIXED_RANK_SNAKE(child))
                    replaced += 1
                else:
                    replaced += replace_snake_modules(child)
            return replaced

        replaced_snake = replace_snake_modules(decoder)
        if not replaced_snake:
            raise ValueError("Expected at least one Snake activation in the NeMo codec decoder.")

        causal_convs = [self.pre_conv, self.post_conv]
        for res_layer in self.res_layers:
            for res_block in res_layer.res_blocks:
                for block in res_block.res_blocks:
                    causal_convs.extend((block.input_conv, block.skip_conv))
        for conv in causal_convs:
            if conv.conv.stride != (1,) or conv.extra_pad_mode != "constant":
                raise ValueError("Mask-free codec export requires stride-1 causal convolutions with constant padding.")
            conv.conv.padding = ((conv.conv.kernel_size[0] - 1) * conv.conv.dilation[0],)

        # ── Fuse weight normalization into each convolution weight ──
        # Fuse weight_g * weight_v / ||weight_v|| → single weight tensor (eliminates runtime norm)
        for module in decoder.modules():
            if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
                torch.nn.utils.parametrize.remove_parametrizations(module, 'weight')

    @staticmethod
    def _causal_conv(module, inputs):
        padding = module.conv.padding[0]
        hidden_states = module.conv(inputs)
        if padding:
            hidden_states = hidden_states[..., :-padding]
        return module.activation(hidden_states)

    @staticmethod
    def _causal_conv_transpose(module, inputs):
        hidden_states = module.conv(inputs)
        end = -module.padding_right if module.padding_right else None
        hidden_states = hidden_states[..., module.padding_left:end]
        return module.activation(hidden_states)

    def _residual_block(self, block, inputs):
        hidden_states = block.input_activation(inputs)
        hidden_states = self._causal_conv(block.input_conv, hidden_states)
        hidden_states = block.skip_activation(hidden_states)
        hidden_states = self._causal_conv(block.skip_conv, hidden_states)
        return inputs + block.dropout(hidden_states)

    def _res_layer(self, layer, inputs):
        residuals = []
        for res_block in layer.res_blocks:
            hidden_states = inputs
            for block in res_block.res_blocks:
                hidden_states = self._residual_block(block, hidden_states)
            residuals.append(hidden_states)
        return sum(residuals) / len(residuals)

    def forward(self, decode_ids, num_decode):
        # Keep the first hypothesis, then expose interleaved codebooks as (1, T, 4).
        audio_codes = decode_ids[[0], 2:num_decode].reshape(1, -1, 4) - self.codebook_offsets
        audio_len = (num_decode - 2) * self.samples_per_decode_id

        with torch.autocast(device_type="cpu", dtype=torch.float16 if USE_FLOAT16_CODEC else torch.float32):
            # ── FSQ lookup: (1, T, 4) → (1, T, 4, 4) → (1, 16, T) ──
            out = torch.nn.functional.embedding(audio_codes, self.fsq_codebook)
            out = out.permute(0, 2, 3, 1).reshape(1, 16, -1)

            # ── HiFi-GAN Decoder: (1, 16, T) → (1, 1, T_audio) ──
            out = self._causal_conv(self.pre_conv, out)

            for act, res_layer, up_sample_conv in zip(
                self.activations, self.res_layers, self.up_sample_conv_layers
            ):
                out = act(out)
                out = self._causal_conv_transpose(up_sample_conv, out)
                out = self._res_layer(res_layer, out)

            out = self.post_activation(out)
            out = self._causal_conv(self.post_conv, out)
            out = out.clamp(min=-1.0, max=1.0)

            if self.scale != 1.0:
                out = torch.nn.functional.interpolate(
                    out,
                    scale_factor=self.scale,
                    mode='linear',
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                audio_len = torch._shape_as_tensor(out)[-1:]
            if "int" in OUT_AUDIO_DTYPE.lower():
                audio_out = (out * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16)
            elif "32" in OUT_AUDIO_DTYPE:
                audio_out = out.float()
            else:
                audio_out = out.half()
        return audio_out, audio_len


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers — export the pipeline geometry once so inference never has to
# hand-duplicate the fixed-at-export constants (mirrors the ASR repo).
# ─────────────────────────────────────────────────────────────────────────────
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


def resolve_token_ids(tokenizer, tokens):
    token_ids = tokenizer.convert_tokens_to_ids(list(tokens))
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != len(tokens)
        or any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids)
        or (
            tokenizer.unk_token_id is not None
            and any(token_id == tokenizer.unk_token_id for token_id in token_ids)
        )
    ):
        raise ValueError(f"KaniTTS tokenizer cannot resolve fixed tokens: {tokens!r}")
    return token_ids


def write_onnx_metadata(onnx_path, metadata):
    """Add/overwrite ``metadata_props`` in place, leaving any external-weight sidecar untouched."""
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
    """Tiny identity graph that carries the static package contract."""

    def forward(self, marker):
        return marker


def _validate_export_settings():
    if MAX_SEQ_LEN < 1:
        raise ValueError("MAX_SEQ_LEN must be at least one.")
    if OUT_SAMPLE_RATE < 1:
        raise ValueError("OUT_SAMPLE_RATE must be at least one.")
    if OUT_AUDIO_DTYPE.upper() not in _OUTPUT_AUDIO_DTYPES:
        raise ValueError(
            f"Unsupported OUT_AUDIO_DTYPE={OUT_AUDIO_DTYPE!r}; "
            f"expected one of {tuple(sorted(_OUTPUT_AUDIO_DTYPES))}."
        )
    if REORDER_KEY not in {"absmean", "L4", "rms", "std"}:
        raise ValueError(f"Unsupported REORDER_KEY: {REORDER_KEY!r}.")


def _export_onnx(module, args, path, input_names, output_names, dynamic_axes=None):
    torch.onnx.export(
        module,
        tuple(args),
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False,
    )


def run_compact_strategy_export():
    if onnx_folder.exists():
        shutil.rmtree(onnx_folder)
    onnx_folder.mkdir(parents=True)
    print("Compact KaniTTS export start ...")
    _validate_export_settings()

    with torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(
            path_kani,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
        attention_layers = [layer for layer in model.model.layers if layer.is_attention_layer]
        if not attention_layers:
            raise ValueError("KaniTTS LFM2 model has no attention layers.")
        head_dim = attention_layers[0].self_attn.head_dim
        num_layers = model.config.num_hidden_layers
        num_conv_layers = model.config.layer_types.count("conv")
        num_attn_layers = num_layers - num_conv_layers
        num_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads
        hidden_size = model.model.embed_tokens.embedding_dim
        vocab_size = model.vocab_size
        kv_dtype = torch.float16 if USE_FLOAT16_KV else torch.float32

        embed = KANITTS_EMBED(model)
        main_core = KANITTS_MAIN(
            model,
            MAX_SEQ_LEN,
            num_heads,
            num_key_value_heads,
            head_dim,
            num_layers,
            num_conv_layers,
            num_attn_layers,
        )
        conv_state_len = main_core.conv_padding

        state_inputs = (
            [torch.zeros(1, num_key_value_heads, 1, head_dim, 10, dtype=kv_dtype)] * num_attn_layers
            + [torch.zeros(1, num_key_value_heads, 1, 10, head_dim, dtype=kv_dtype)] * num_attn_layers
            + [torch.zeros(1, hidden_size, conv_state_len, dtype=kv_dtype)] * num_conv_layers
        )
        state_input_names = (
            [f"in_key_{index}" for index in range(num_attn_layers)]
            + [f"in_value_{index}" for index in range(num_attn_layers)]
            + [f"in_conv_{index}" for index in range(num_conv_layers)]
        )
        state_output_names = (
            [f"out_key_{index}" for index in range(num_attn_layers)]
            + [f"out_value_{index}" for index in range(num_attn_layers)]
            + [f"out_conv_{index}" for index in range(num_conv_layers)]
        )
        state_axes = {}
        for index in range(num_attn_layers):
            state_axes[f"in_key_{index}"] = {4: "history_len"}
            state_axes[f"out_key_{index}"] = {4: "kv_seq_len"}
            state_axes[f"in_value_{index}"] = {3: "history_len"}
            state_axes[f"out_value_{index}"] = {3: "kv_seq_len"}

        input_ids = torch.zeros(1, 10, dtype=torch.int32)
        current_token = torch.zeros(1, 1, dtype=torch.int32)
        save_ids = torch.zeros(1, 10, dtype=torch.int32)
        history_len = torch.tensor([10], dtype=torch.int64)
        dummy_control_tensors = {
            "penalty_value": torch.tensor([1.0], dtype=torch.float32),
            "penalty_range": torch.tensor([1], dtype=torch.int64),
            "temperature": torch.tensor([1.0], dtype=torch.float32),
            "top_k": torch.tensor([min(50, vocab_size)], dtype=torch.int64),
            "top_p": torch.tensor([1.0], dtype=torch.float32),
            "repetition_penalty": torch.tensor([1.0], dtype=torch.float32),
        }

        for strategy in DECODE_STRATEGIES:
            if strategy == "greedy":
                decode_control_names = []
            elif strategy == "penalty_greedy":
                decode_control_names = ["penalty_value", "penalty_range"]
            else:
                decode_control_names = ["temperature", "top_k", "top_p", "repetition_penalty"]
            prefill_control_names = ["temperature", "top_k", "top_p"] if strategy == "sampling" else []
            prefill_controls = [dummy_control_tensors[name] for name in prefill_control_names]
            decode_controls = [dummy_control_tensors[name] for name in decode_control_names]

            prefill = KANITTS_MAIN_PREFILL_STRATEGY(embed, main_core, strategy, vocab_size)
            prefill_axes = {
                **{name: axes for name, axes in state_axes.items() if name.startswith("out_")},
                "input_ids": {1: "ids_len"},
            }
            _export_onnx(
                prefill,
                [input_ids, *prefill_controls],
                onnx_model_Main_Prefill[strategy],
                ["input_ids", *prefill_control_names],
                [*state_output_names, "next_token", "kv_seq_len"],
                prefill_axes,
            )
            del prefill

            decode_step = KANITTS_DECODE_STEP_STRATEGY(embed, main_core, strategy, vocab_size)
            decode_axes = {
                **state_axes,
                "save_ids_in": {1: "save_ids_len"},
                "save_ids_out": {1: "save_ids_len_out"},
            }
            _export_onnx(
                decode_step,
                [*state_inputs, current_token, save_ids, history_len, *decode_controls],
                onnx_model_Decode_Step[strategy],
                [*state_input_names, "current_token", "save_ids_in", "history_len", *decode_control_names],
                [*state_output_names, "next_token", "save_ids_out", "kv_seq_len"],
                decode_axes,
            )
            del decode_step

        del model, embed, main_core, state_inputs, dummy_control_tensors
        gc.collect()

        tokenizer = AutoTokenizer.from_pretrained(path_kani)
        tokeniser_length = tokenizer.vocab_size
        stop_token_ids = resolve_token_ids(tokenizer, STOP_TOKENS)
        prompt_prefix_token_ids = resolve_token_ids(tokenizer, PROMPT_PREFIX_TOKENS)
        prompt_suffix_token_ids = resolve_token_ids(tokenizer, PROMPT_SUFFIX_TOKENS)
        del tokenizer
        nemo_model = AudioCodecModel.from_pretrained(
            path_codec,
            map_location=torch.device("cpu"),
        ).float().eval()
        codec = NEMO_CODEC(nemo_model, tokeniser_length)
        model_codec_samples_per_decode_id = codec.samples_per_decode_id
        codec_samples_per_decode_id = (
            model_codec_samples_per_decode_id
            * OUT_SAMPLE_RATE
            / MODEL_SAMPLE_RATE
        )
        if not codec_samples_per_decode_id.is_integer():
            raise ValueError(
                "OUT_SAMPLE_RATE must map each codec token to a whole number of samples; "
                f"got {codec_samples_per_decode_id} samples per token."
            )
        codec_samples_per_decode_id = int(codec_samples_per_decode_id)
        dummy_audio_tokens = (
            torch.arange(4, dtype=torch.int32) * NEMO_CODEC.CODEBOOK_SIZE
            + tokeniser_length
            + 10
        ).view(1, 4)
        decode_ids = torch.cat([torch.zeros(1, 2, dtype=torch.int32), dummy_audio_tokens], dim=1)
        num_decode = torch.tensor([decode_ids.shape[-1]], dtype=torch.int64)
        _export_onnx(
            codec,
            [decode_ids, num_decode],
            onnx_model_Codec,
            ["decode_ids", "num_decode"],
            ["audio_out", "audio_out_len"],
            {
                "decode_ids": {1: "num_decode"},
                "audio_out": {2: "audio_len"},
            },
        )
        del nemo_model, codec, decode_ids, num_decode, dummy_audio_tokens
        gc.collect()

        _export_onnx(
            METADATA_CARRIER(),
            [torch.zeros(1, dtype=torch.int64)],
            onnx_model_Metadata,
            ["metadata_marker"],
            ["metadata_marker_out"],
        )

        onnx_metadata = build_model_metadata(
            {
                "graph_layout": "strategy_prefill_decode_step",
                "max_seq_len": MAX_SEQ_LEN,
                "stop_token_ids": stop_token_ids,
                "prompt_prefix_token_ids": prompt_prefix_token_ids,
                "prompt_suffix_token_ids": prompt_suffix_token_ids,
                "shared_initializer_model_file": SHARED_MODEL_NAME,
                "shared_initializer_data_file": SHARED_DATA_NAME,
                "model_file_name_codec": Path(onnx_model_Codec).name,
                "vocab_size": vocab_size,
                "use_float16_kv": USE_FLOAT16_KV,
                "compute_in_f32": COMPUTE_IN_F32,
                "out_sample_rate": OUT_SAMPLE_RATE,
                "codec_prefix_token_count": 2,
                "codec_token_alignment": 4,
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
            },
        )
        final_graphs = (
            [Path(onnx_model_Main_Prefill[strategy]) for strategy in DECODE_STRATEGIES]
            + [Path(onnx_model_Decode_Step[strategy]) for strategy in DECODE_STRATEGIES]
            + [Path(onnx_model_Codec), Path(onnx_model_Metadata)]
        )
        for target in final_graphs:
            write_onnx_metadata(target, onnx_metadata)
        print(f"[Metadata] Stamped {len(onnx_metadata)} keys into {len(final_graphs)} graphs.")

        shared_stats = bundle_shared_initializers(
            onnx_folder,
            model_paths=final_graphs,
            metadata=onnx_metadata,
        )
        replace_onnx_metadata(onnx_model_Metadata, onnx_metadata)
        print(
            f"[Shared weights] {shared_stats['initializer_references']} references -> "
            f"{shared_stats['unique_initializers']} unique tensors; deduplicated "
            f"{shared_stats['deduplicated_bytes'] / (1024 * 1024):.2f} MiB."
        )

    print("Compact KaniTTS export done!")


if __name__ == "__main__":
    run_compact_strategy_export()
    print("\nStart running inference via Inference_Kani_TTS_ONNX.py ...")
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "Inference_Kani_TTS_ONNX.py"),
            "--onnx-folder",
            str(onnx_folder),
        ],
        check=True,
    )
