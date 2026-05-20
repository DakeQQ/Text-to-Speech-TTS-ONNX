import gc
import site
import shutil
import time
import concurrent.futures
import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torchaudio
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer
from STFT_Process import STFT_Process



# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
download_path                            = r'/home/DakeQQ/Downloads/Qwen3-TTS-12Hz-0.6B-Base'                  # Source model folder [0.6B-Base / 1.7B-Base / 0.6B-CustomVoice / 1.7B-CustomVoice / 1.7B-VoiceDesign]
onnx_model_Embed_A                       = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Embed_A.onnx'
onnx_model_Embed_B                       = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Embed_B.onnx'
onnx_model_Embed_C                       = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Embed_C.onnx'
onnx_model_Embed_D                       = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Embed_D.onnx'
onnx_model_Preprocess                    = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Preprocess.onnx'
onnx_model_Encoder                       = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Encoder.onnx'
onnx_model_Predictor                     = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Predictor.onnx'
onnx_model_Pred_LmHead                   = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Predictor_LmHead.onnx'
onnx_model_Main                          = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Main.onnx'
onnx_model_Decoder                       = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Decoder.onnx'
onnx_model_Decoder_Stream                = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Decoder_Stream.onnx'
onnx_model_Main_Rotary_Mask_Text_Prefill = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Main_Rotary_Mask_Text_Prefill.onnx'
onnx_model_Main_Rotary_Mask_Text_Decode  = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Main_Rotary_Mask_Text_Decode.onnx'
onnx_model_Pred_Rotary_Mask_Text_Prefill = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Predictor_Rotary_Mask_Text_Prefill.onnx'
onnx_model_Pred_Rotary_Mask_Text_Decode  = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/QwenTTS_Predictor_Rotary_Mask_Text_Decode.onnx'
onnx_model_Gather_0                      = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/Gather_0.onnx'
onnx_model_Concat_Embed                  = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/Concat_Embed.onnx'
onnx_model_Concat_Ids                    = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/Concat_Ids.onnx'
onnx_model_Slide_Window                  = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/Slide_Window.onnx'
onnx_model_Greedy                        = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/Greedy_Search.onnx'
onnx_model_First_Beam                    = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/First_Beam_Search.onnx'
onnx_model_Second_Beam                   = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/Second_Beam_Search.onnx'
onnx_model_Penalty                       = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/Apply_Penalty.onnx'
onnx_model_Argmax                        = r'/home/DakeQQ/Downloads/QwenTTS_ONNX/Argmax.onnx'


# ─────────────────────────────────────────────────────────────────────────────
# Targets TTS
# ─────────────────────────────────────────────────────────────────────────────
generated_audio_path = r"./generated.wav"           # Output audio file
target_tts           = [                            # Texts to synthesize
    "大家好，我现在正在大可奇奇体验AI科技。",
    "Hello everyone, I'm currently experiencing DakeQQ's AI technology."
]


# ─────────────────────────────────────────────────────────────────────────────
# Language & Generation Settings
# ─────────────────────────────────────────────────────────────────────────────
TTS_LANGUAGE = "Chinese"                            # Options: [English, German, Spanish, Chinese, Japanese, French, Korean, Russian, Italian, Portuguese]

# For Voice Clone
prompt_audio_path = "./example/basic_ref_zh.wav"    # Reference audio for voice cloning
prompt_text       = "对，这就是我，万人敬仰的太乙真人。"  # Transcription of the reference audio

# For Custom Voice
SPEAKER_NAME  = "Vivian"                            # only used when MODE == "custom_voice". Supported: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
INSTRUCT_TEXT = "Speak very happily"                # Optional style instruction. Empty string = no instruct.
                                                    # English examples: "Speak cheerfully.", "Whisper gently.", "Speak slowly and calmly.",
                                                    #                   "Read with excitement.", "Speak in a sad tone.", "Narrate like a storyteller.",
                                                    #                   "Speak with anger.", "Talk softly.", "Speak fast with urgency."
                                                    # 中文示例:          "开心地说。", "轻声细语。", "慢慢地、平静地说。",
                                                    #                   "兴奋地朗读。", "用悲伤的语气说。", "像讲故事一样叙述。",
                                                    #                   "愤怒地说。", "温柔地说。", "急促而紧迫地说。"
# For Voice Design
VOICE_DESCRIPTION = "A young female with a warm, gentle tone and slight breathiness"  # Natural language voice description.

MAX_SEQ_LEN   = 1024                                # Maximum decode length (fixed at export time)
MIN_SEQ_LEN   = 2                                   # Minimum decode length (editable at runtime)
STOP_TOKEN    = [2150]                              # EOS token id for QwenTTS — Do not change


# ─────────────────────────────────────────────────────────────────────────────
# Audio settings
# ─────────────────────────────────────────────────────────────────────────────
IN_SAMPLE_RATE       = 24000                        # Prompt audio sample rate  (fixed at export time)
OUT_SAMPLE_RATE      = 24000                        # Output audio sample rate  (fixed at export time)
MAX_PROMPT_AUDIO_LEN = 20 * IN_SAMPLE_RATE          # Maximum prompt audio length in samples (fixed at export time, '20' means 20 seconds, Voice Clone only)

WINDOW_TYPE          = 'hann'                       # Window function      — edit carefully
N_MELS               = 128                          # Number of Mel bands  — Do not edit
NFFT_STFT            = 1024                         # FFT size             — Do not edit
WINDOW_LENGTH        = 1024                         # Window length        — Do not edit
HOP_LENGTH           = 256                          # Hop length (samples) — Do not edit
SAMPLES_PER_CODEC_FRAME = 1920                      # # Fixed value for the Qwen3-TTS — Do not edit

# ─────────────────────────────────────────────────────────────────────────────
# Decoding settings
# ─────────────────────────────────────────────────────────────────────────────
USE_BEAM_SEARCH = False                             # False → greedy decoding
MAX_BEAM_SIZE   = 10                                # Maximum beam width (fixed at export time)
BEAM_SIZE       = 3                                 # Active beam width
TOP_K           = 3                                 # Top-K sampling parameter
PENALTY_RANGE   = 5                                 # Recent-token window for repetition penalty
REPEAT_PENALTY  = 0.8                               # Repetition penalty coefficient (1.0 = disabled)


# ─────────────────────────────────────────────────────────────────────────────
# Runtime / optimisation flags
# ─────────────────────────────────────────────────────────────────────────────
DO_EXPORT                = True                     # Set True to run the export pipeline
STREAMING                = False                    # True → streaming decode with static N-frame Decoder (sliding window)
STREAM_WINDOW_FRAMES     = 7                        # Streaming sliding window frame count, Lower is faster but affects quality. (at least ≥ 3, recommended ≥ 7, fixed at export time)
USE_F16_KV               = True                     # Use float16 KV cache (saves memory, may reduce quality)
USE_F16_ENCODER          = False                    # Pre-process the encoder in FP16 format for better GPU utilization.
USE_AUDIO_NORMALIZER     = False                    # Normalize output loudness (may alter voice characteristics)
ORT_LOG                  = False                    # Enable ONNX Runtime logging (disable for best performance)
ORT_FP16                 = False                    # FP16 ORT settings (ARM64-v8.2a or newer required for CPU)
ORT_Accelerate_Providers = []                       # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
OPSET                    = 18                       # ONNX opset version
MAX_THREADS              = 0                        # CPU thread count (0 = auto)
DEVICE_ID                = 0                        # Device index


# ─────────────────────────────────────────────────────────────────────────────
# Patch site-packages with modified model files before importing qwen_tts
# ─────────────────────────────────────────────────────────────────────────────
site_package_path = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/modeling_mimi.py",                   site_package_path + "/transformers/models/mimi/modeling_mimi.py")
shutil.copyfile("./modeling_modified/modeling_qwen3_tts.py",              site_package_path + "/qwen_tts/core/models/modeling_qwen3_tts.py")
shutil.copyfile("./modeling_modified/modeling_qwen3_tts_tokenizer_v2.py", site_package_path + "/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py")

# transformers==4.57.3
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2 as mod
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2ConvNeXtBlock,
    Qwen3TTSTokenizerV2ConvNeXtBlockUnfused,
    SnakeBeta
)


# ─────────────────────────────────────────────────────────────────────────────
# Mode Selection (mutually exclusive)
# ─────────────────────────────────────────────────────────────────────────────
# Options: "voice_clone", "custom_voice", or "voice_design" (one per export run)
if "custom" in download_path.lower():
    MODE = "custom_voice"
elif "design" in download_path.lower():
    MODE = "voice_design"
else:
    MODE = "voice_clone"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Tensor Utility Modules
# ─────────────────────────────────────────────────────────────────────────────
class GATHER_0(torch.nn.Module):
    """Get the first codec from a batch."""

    def __init__(self):
        super().__init__()

    def forward(self, input_ids):
        return input_ids[[0]]


class CONCAT_IDS(torch.nn.Module):
    """Concatenate two codec tensors along the sequence dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, codec_0, codec_1):
        return torch.cat([codec_0, codec_1], dim=1)
    

class CONCAT_EMBED(torch.nn.Module):
    """Concatenate two codec embeddings along the sequence dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, codec_embed_0, codec_embed_1):
        return torch.cat([codec_embed_0, codec_embed_1], dim=1)


class SLIDE_WINDOW(torch.nn.Module):
    def __init__(self, tts):
        super().__init__()
        self.num_code_groups = tts.model.talker.code_predictor.model.config.num_code_groups

    def forward(self, codec_0, codec_1):
        return torch.cat([codec_0[:, self.num_code_groups:], codec_1], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Modules
# ─────────────────────────────────────────────────────────────────────────────
class TTS_EMBED_A(torch.nn.Module):
    """Text token → projected text embedding (used for both reference and target text)."""

    def __init__(self, tts):
        super().__init__()
        self.tts = tts.model
        self._replace_gelu_with_tanh_approximation(self.tts)
        self.talker_text_embed = self.tts.talker.model.text_embedding
        self.text_projection   = self.tts.talker.text_projection

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def forward(self, text_ids):
        return self.text_projection(self.talker_text_embed(text_ids))


class TTS_EMBED_B(torch.nn.Module):
    """Input token ids → talker codec embedding."""

    def __init__(self, tts):
        super().__init__()
        self.talker_input_embed = tts.model.talker.model.codec_embedding

    def forward(self, codec_ids):
        return self.talker_input_embed(codec_ids)


class TTS_EMBED_C(torch.nn.Module):
    """Codec ids → code-predictor codec embedding for all RVQ layers (fused)."""

    def __init__(self, tts):
        super().__init__()
        self.talker_code_predictor_embed = tts.model.talker.code_predictor.model.codec_embedding
        self.num_code_groups = tts.model.talker.code_predictor.model.config.num_code_groups

    def forward(self, codec_ids, codec_embed, trailing_text_hidden, gather_id):
        codec_ids = codec_ids.reshape(self.num_code_groups, -1)
        codec_embed += trailing_text_hidden[:, gather_id]
        for layer in range(len(self.talker_code_predictor_embed)):
            codec_embed += self.talker_code_predictor_embed._modules[f'{layer}'](codec_ids[[layer + 1]])
        return codec_embed


class TTS_EMBED_D(torch.nn.Module):
    """Codec ids → code-predictor codec embedding for one RVQ layer (split)."""

    def __init__(self, tts, layer):
        super().__init__()
        self.talker_code_predictor_embed = tts.model.talker.code_predictor.model.codec_embedding._modules[f'{layer}']

    def forward(self, codec_ids):
        return self.talker_code_predictor_embed(codec_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary Position Embedding & Attention Mask Modules
# ─────────────────────────────────────────────────────────────────────────────
class TTS_MAIN_ROTARY_MASK_PREFILL(torch.nn.Module):
    """
    Compute rotary position embeddings and a causal attention mask for the
    main talker prefill (multi-token) phase.
    """

    def __init__(self, tts, max_seq_len):
        super().__init__()
        self.tts           = tts.model.talker.model
        self.mrope_section = tts.model.talker.config.rope_scaling['mrope_section']
        head_dim_half      = tts.model.talker.config.head_dim // 2
        modality_num       = len(self.mrope_section)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).view(1, 1, 1, -1).expand(3, 1, 1, -1)
        inv_freq     = self.tts.rotary_emb.inv_freq.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        idx_theta    = (inv_freq @ position_ids).transpose(2, 3)
        cos          = torch.cat([torch.cos(idx_theta)] * 2, dim=-1)
        sin          = torch.cat([torch.sin(idx_theta)] * 2, dim=-1)

        cos = torch.cat([self.apply_interleaved_rope(cos[..., :head_dim_half], modality_num)] * 2, dim=-1)
        sin = torch.cat([self.apply_interleaved_rope(sin[..., :head_dim_half], modality_num)] * 2, dim=-1)
        sin[..., :head_dim_half] = sin[..., :head_dim_half] * -1.0

        cos = cos.unsqueeze(2).unsqueeze(2).half()
        sin = sin.unsqueeze(2).unsqueeze(2).half()

        # Causal mask: -128 for masked positions (used with int8 KV cache arithmetic)
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

    def apply_interleaved_rope(self, x, modality_num):
        x_t = x[0].clone()
        index_ranges = []
        for i, n in enumerate(self.mrope_section[1:], 1):
            index_ranges.append((i, n * modality_num))
        for beg_idx, end_idx in index_ranges:
            x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
        return x_t

    def forward(self, ids_len, history_len):
        kv_seq_len         = ids_len + history_len
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask     = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, kv_seq_len


class TTS_MAIN_ROTARY_MASK_DECODE(torch.nn.Module):
    """
    Compute rotary position embeddings for the main talker decode (single-token)
    phase and advance the KV sequence length counter.
    """

    def __init__(self, tts, max_seq_len):
        super().__init__()
        self.tts           = tts.model.talker.model
        self.mrope_section = tts.model.talker.config.rope_scaling['mrope_section']
        head_dim_half      = tts.model.talker.config.head_dim // 2
        modality_num       = len(self.mrope_section)

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).view(1, 1, 1, -1).expand(3, 1, 1, -1)
        inv_freq     = self.tts.rotary_emb.inv_freq.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        idx_theta    = (inv_freq @ position_ids).transpose(2, 3)
        cos          = torch.cat([torch.cos(idx_theta)] * 2, dim=-1)
        sin          = torch.cat([torch.sin(idx_theta)] * 2, dim=-1)

        cos = torch.cat([self.apply_interleaved_rope(cos[..., :head_dim_half], modality_num)] * 2, dim=-1)
        sin = torch.cat([self.apply_interleaved_rope(sin[..., :head_dim_half], modality_num)] * 2, dim=-1)
        sin[..., :head_dim_half] = sin[..., :head_dim_half] * -1.0

        cos = cos.unsqueeze(2).unsqueeze(2).half()
        sin = sin.unsqueeze(2).unsqueeze(2).half()

        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

    def apply_interleaved_rope(self, x, modality_num):
        x_t = x[0].clone()
        index_ranges = []
        for i, n in enumerate(self.mrope_section[1:], 1):
            index_ranges.append((i, n * modality_num))
        for beg_idx, end_idx in index_ranges:
            x_t[..., beg_idx:end_idx:modality_num] = x[beg_idx, ..., beg_idx:end_idx:modality_num]
        return x_t

    def forward(self, kv_seq_len):
        kv_seq_len_next    = kv_seq_len + 1
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, kv_seq_len_next


class TTS_PREDICTOR_ROTARY_MASK_PREFILL(torch.nn.Module):
    """
    Compute rotary position embeddings and a causal attention mask for the
    code-predictor prefill (multi-token) phase.
    """

    def __init__(self, tts, max_seq_len):
        super().__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq     = tts.model.talker.code_predictor.model.rotary_emb.inv_freq
        idx_theta    = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos          = torch.cat([torch.cos(idx_theta)] * 2, dim=-1).half()
        sin          = torch.cat([-torch.sin(idx_theta), torch.sin(idx_theta)], dim=-1).half()

        # Causal mask: -128 for masked positions (used with int8 KV cache arithmetic)
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

    def forward(self, ids_len, history_len):
        kv_seq_len         = ids_len + history_len
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask     = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, kv_seq_len


class TTS_PREDICTOR_ROTARY_MASK_DECODE(torch.nn.Module):
    """
    Compute rotary position embeddings for the code-predictor decode (single-token)
    phase and advance the KV sequence length counter.
    """

    def __init__(self, tts, max_seq_len):
        super().__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq     = tts.model.talker.code_predictor.model.rotary_emb.inv_freq
        idx_theta    = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos          = torch.cat([torch.cos(idx_theta)] * 2, dim=-1).half()
        sin          = torch.cat([-torch.sin(idx_theta), torch.sin(idx_theta)], dim=-1).half()

        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next    = kv_seq_len + 1
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, kv_seq_len_next


# ─────────────────────────────────────────────────────────────────────────────
# Audio Pipeline Modules
# ─────────────────────────────────────────────────────────────────────────────
class TTS_ENCODER(torch.nn.Module):
    """
    Encode a raw audio waveform into:
      - ref_code       : RVQ codec tokens (used as in-context reference)
      - ref_code_len   : number of codec frames
      - speaker_embed  : speaker identity embedding derived from Mel spectrogram
    """

    def __init__(self, tts, in_sample_rate, max_seq_len, stft_model, nfft_stft, n_mels):
        super().__init__()
        self.tts             = tts
        self._replace_gelu_with_tanh_approximation(self.tts.model)
        self.encoder         = self.tts.model.speech_tokenizer.model.encoder.eval()
        self.speaker_encoder = self.tts.model.speaker_encoder.eval()

        for param in self.tts.model.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False

        self._fuse_encoder_weights()

        # Pre-computed values
        self.stft_model = stft_model
        self.in_sample_rate = in_sample_rate
        self.sr_scale   = float(24000.0 / self.in_sample_rate)
        self.eps        = torch.tensor([1e-5], dtype=torch.float32)
        self.inv_int16  = torch.tensor(1.0 / 32768.0, dtype=torch.float32).view(1, 1, -1)
        self.fbank      = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 0, in_sample_rate // 2, n_mels, in_sample_rate, "slaney", 'slaney')).transpose(0, 1).unsqueeze(0)

        self.num_heads     = self.encoder.encoder_transformer.layers._modules['0'].self_attn.num_heads
        self.qk_heads      = self.num_heads + self.num_heads
        self.head_dim      = self.encoder.encoder_transformer.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2

        position_ids      = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq          = self.encoder.encoder_transformer.layers._modules['0'].self_attn.rotary_emb.inv_freq
        idx_theta         = (position_ids * inv_freq).unsqueeze(1).unsqueeze(0)
        cos, sin          = torch.cos(idx_theta), torch.sin(idx_theta)
        self.rope_emb_cos = torch.cat([cos,  cos], dim=-1)
        self.rope_emb_sin = torch.cat([-sin, sin], dim=-1)

    # ── Weight Fusion ─────────────────────────────────────────────────────────

    def _fuse_encoder_weights(self):
        """Fuse QKV projections, layer norms, and layer scales for the encoder transformer."""
        scale_factor = self.encoder.encoder_transformer.layers._modules['0'].self_attn.head_dim ** -0.25
        with torch.no_grad():
            for layer in self.encoder.encoder_transformer.layers:
                self._fuse_qkv_projection(layer, scale_factor)
                self._fuse_input_layernorm_into_qkv(layer)
                self._fuse_post_layernorm_into_mlp(layer)
                self._fuse_layer_scales(layer)

    def _fuse_qkv_projection(self, layer, scale_factor):
        """Fuse Q, K, V projections into a single QKV linear."""
        q_proj, k_proj, v_proj = (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        )
        in_features  = q_proj.in_features
        out_features = q_proj.out_features + k_proj.out_features + v_proj.out_features
        qkv          = torch.nn.Linear(in_features, out_features, bias=(q_proj.bias is not None))
        qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
        if q_proj.bias is not None:
            qkv.bias.copy_(torch.cat([q_proj.bias * scale_factor, k_proj.bias * scale_factor, v_proj.bias], dim=0))
        layer.self_attn.qkv   = qkv
        layer.self_attn.q_dim = q_proj.out_features
        layer.self_attn.k_dim = k_proj.out_features
        layer.self_attn.v_dim = v_proj.out_features
        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

    def _fuse_input_layernorm_into_qkv(self, layer):
        """Absorb input_layernorm affine parameters into the QKV projection."""
        norm   = layer.input_layernorm
        linear = layer.self_attn.qkv
        if linear.bias is not None:
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        else:
            linear.bias = torch.nn.Parameter(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = norm.bias = None

    def _fuse_post_layernorm_into_mlp(self, layer):
        """Absorb post_attention_layernorm affine parameters into MLP fc1."""
        norm   = layer.post_attention_layernorm
        linear = layer.mlp.fc1
        if linear.bias is not None:
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        else:
            linear.bias = torch.nn.Parameter(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = norm.bias = None

    def _fuse_layer_scales(self, layer):
        """Fuse self-attention and MLP layer scales into output projections."""
        scale  = layer.self_attn_layer_scale.scale
        linear = layer.self_attn.o_proj
        if linear.bias is not None:
            linear.bias.data.mul_(scale.data)
        linear.weight.data.mul_(scale.data.unsqueeze(1))

        scale  = layer.mlp_layer_scale.scale
        linear = layer.mlp.fc2
        if linear.bias is not None:
            linear.bias.data.mul_(scale.data)
        linear.weight.data.mul_(scale.data.unsqueeze(1))

    # ── Utility ───────────────────────────────────────────────────────────────

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def rotate_half(self, x):
        """Rotate using flip() — more efficient than split()+cat() in ONNX Runtime."""
        x = x.view(1, -1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(1, -1, self.qk_heads, self.head_dim)

    def forward(self, prompt_audio):
        # Resample and normalize to [-1, 1]
        prompt_audio = prompt_audio.float()
        if self.sr_scale < 1.0:
            prompt_audio = torch.nn.functional.interpolate(prompt_audio, scale_factor=self.sr_scale, mode='linear', align_corners=False)
        prompt_audio = prompt_audio * self.inv_int16
        if self.sr_scale > 1.0:
            prompt_audio = torch.nn.functional.interpolate(prompt_audio, scale_factor=self.sr_scale, mode='linear', align_corners=False)

        # Encode audio through the convolutional encoder
        hidden_states = self.encoder.encoder(prompt_audio).transpose(1, 2)
        ids_len       = hidden_states.shape[1]
        rope_emb_cos  = self.rope_emb_cos[:, :ids_len].float()
        rope_emb_sin  = self.rope_emb_sin[:, :ids_len].float()

        # Transformer layers with fused RoPE attention
        for layer in self.encoder.encoder_transformer.layers:
            residual      = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            qkv           = layer.self_attn.qkv(hidden_states)
            qkv           = qkv.reshape(1, -1, self.qk_heads + self.num_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_heads], dim=-2)
            qk            = qk * rope_emb_cos + self.rotate_half(qk) * rope_emb_sin
            q, k          = torch.split(qk, [self.num_heads, self.num_heads], dim=-2)
            q             = q.transpose(1, 2)
            k             = k.permute(0, 2, 3, 1)
            v             = v.transpose(1, 2)
            attn          = torch.softmax(torch.matmul(q, k), dim=-1)
            attn          = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual      = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp.fc2(layer.mlp.activation_fn(layer.mlp.fc1(hidden_states)))

        # Downsample and quantise to RVQ codes
        embeddings = self.encoder.downsample(hidden_states.transpose(1, 2))
        ref_code   = self.encoder.quantizer.encode(embeddings, self.tts.model.speech_tokenizer.config.encoder_valid_num_quantizers)
        ref_code   = ref_code.squeeze(1)

        # Compute speaker embedding from log-Mel spectrogram
        real_part, imag_part = self.stft_model(prompt_audio)
        mel_features         = (torch.matmul(self.fbank, torch.sqrt(real_part * real_part + imag_part * imag_part)) + self.eps).log()
        speaker_embed        = self.speaker_encoder(mel_features)
        ref_code_len         = ref_code.shape[1].unsqueeze(0)

        return ref_code, ref_code_len, speaker_embed.float()


class TTS_PREPROCESS(torch.nn.Module):
    """
    Build the full talker input embedding from:
      - reference-audio codec embedding
      - speaker embedding
      - language embedding
      - reference text embedding
      - target text embedding

    Returns talker_input_embed, trailing_text_hidden, and the sequence length.

    Modes:
      - voice_clone:  speaker_embed is from speaker encoder (x-vector), codec_embed from ref audio.
      - custom_voice: speaker_embed is from speaker id token, codec_embed is empty.
      - voice_design: no speaker_embed at all, codec_embed is empty. Voice identity comes from instruct.
    """

    def __init__(self, tts, mode="voice_clone"):
        super().__init__()
        self.tts = tts
        self.mode = mode
        self.talker_text_embed  = self.tts.model.talker.model.text_embedding
        self.talker_input_embed = self.tts.model.talker.model.codec_embedding

        config = self.tts.model.config

        # Pre-compute special-token embeddings
        sp_tokens = torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]], dtype=torch.int32)
        self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = \
            self.tts.model.talker.text_projection(self.talker_text_embed(sp_tokens)).chunk(3, dim=1)

        # Pre-compute fixed codec prefix / suffix embeddings
        if mode == "voice_design":
            # voice_design: no speaker token → codec prefix is [think, think_bos, language, think_eos, pad, bos]
            # The pad portion aligns with: pad*4 + bos = 5 positions
            self._talker_input_embed = torch.cat([self.tts_pad_embed.expand(-1, 4, -1), self.tts_bos_embed], dim=1)
        else:
            # voice_clone / custom_voice: codec prefix includes speaker → pad*5 + bos = 6 positions
            self._talker_input_embed = torch.cat([self.tts_pad_embed.expand(-1, 5, -1), self.tts_bos_embed], dim=1)
        self.codec_bos_embed     = self.talker_input_embed(torch.tensor([[config.talker_config.codec_bos_id]],                                            dtype=torch.int32))
        self.codec_think_embed   = self.talker_input_embed(torch.tensor([[config.talker_config.codec_think_id, config.talker_config.codec_think_bos_id]], dtype=torch.int32))
        self.codec_eos_embed     = self.talker_input_embed(torch.tensor([[config.talker_config.codec_think_eos_id]],                                      dtype=torch.int32))
        self.codec_pad_embed     = self.talker_input_embed(torch.tensor([[config.talker_config.codec_pad_id]],                                            dtype=torch.int32))

        # Role header embedding
        system_head     = "<|im_start|>assistant\n"
        system_head_ids = self.tts.processor(text=system_head, return_tensors="pt", padding=True)["input_ids"].int()
        self._talker_input_embed_role = self.tts.model.talker.text_projection(self.talker_text_embed(system_head_ids))

    def forward(self, *args):
        if self.mode == "voice_design":
            # voice_design: forward(language_embed, target_text_embed)
            language_embed, target_text_embed = args
            return self._forward_voice_design(language_embed, target_text_embed)
        else:
            # voice_clone / custom_voice: forward(codec_embed, speaker_embed, language_embed, ref_prompt_text_embed, target_text_embed)
            codec_embed, speaker_embed, language_embed, ref_prompt_text_embed, target_text_embed = args
            return self._forward_default(codec_embed, speaker_embed, language_embed, ref_prompt_text_embed, target_text_embed)

    def _forward_default(self, codec_embed, speaker_embed, language_embed, ref_prompt_text_embed, target_text_embed):
        # Prepend BOS to the codec sequence
        codec_embed = torch.cat([self.codec_bos_embed, codec_embed], dim=1)
        codec_len   = codec_embed.shape[1].unsqueeze(0)

        # Build text sequence and pad to match codec length
        text_embed = torch.cat([ref_prompt_text_embed, target_text_embed, self.tts_eos_embed], dim=1)
        text_len   = text_embed.shape[1].unsqueeze(0)
        text_embed = torch.cat([text_embed, self.tts_pad_embed.repeat(1, (codec_len - text_len).clamp(min=0), 1)], dim=1)

        # Build codec conditioning prefix and combine with role header
        codec_input_embed   = torch.cat([self.codec_think_embed, language_embed, self.codec_eos_embed, speaker_embed, self.codec_pad_embed], dim=1)
        _talker_input_embed = self._talker_input_embed + codec_input_embed
        talker_input_embed  = torch.cat([self._talker_input_embed_role, _talker_input_embed], dim=1)

        # Interleave text and codec embeddings
        icl_input_embed      = text_embed[:, :codec_len] + codec_embed
        trailing_text_hidden = torch.cat([text_embed[:, codec_len:], self.tts_pad_embed], dim=1)
        trailing_len_minus   = trailing_text_hidden.shape[1].unsqueeze(0).int() - 1
        hidden_states        = torch.cat([talker_input_embed, icl_input_embed], dim=1)
        ids_len              = hidden_states.shape[1].unsqueeze(0)
        return hidden_states, ids_len, trailing_text_hidden, trailing_len_minus

    def _forward_voice_design(self, language_embed, target_text_embed):
        # voice_design: no ref audio, no speaker → streaming-style text + codec interleave
        # Build text sequence: target_text + eos
        text_embed = torch.cat([target_text_embed, self.tts_eos_embed], dim=1)

        # Build codec conditioning prefix (no speaker): [think, think_bos, language, think_eos, pad]
        codec_input_embed   = torch.cat([self.codec_think_embed, language_embed, self.codec_eos_embed, self.codec_pad_embed], dim=1)
        _talker_input_embed = self._talker_input_embed + codec_input_embed
        talker_input_embed  = torch.cat([self._talker_input_embed_role, _talker_input_embed], dim=1)

        # For voice_design with no ref_code, the first text token is combined with codec_bos
        first_text_token    = text_embed[:, :1] + self.codec_bos_embed
        talker_input_embed  = torch.cat([talker_input_embed, first_text_token], dim=1)

        # Remaining text tokens become trailing_text_hidden for streaming
        trailing_text_hidden = torch.cat([text_embed[:, 1:], self.tts_pad_embed], dim=1)
        trailing_len_minus   = trailing_text_hidden.shape[1].unsqueeze(0).int() - 1
        hidden_states        = talker_input_embed
        ids_len              = hidden_states.shape[1].unsqueeze(0)
        return hidden_states, ids_len, trailing_text_hidden, trailing_len_minus


class TTS_DECODER(torch.nn.Module):
    """
    Decode RVQ codec tokens back to a raw audio waveform.
    - voice_clone: Combines reference codec tokens with the generated sequence.
    - custom_voice: Decodes only the generated sequence (no ref_code).
    """

    def __init__(self, tts, output_sample_rate, max_seq_len, mode="voice_clone"):
        super().__init__()
        self.tts = tts
        self.mode = mode
        self._replace_gelu_with_tanh_approximation(self.tts.model)
        self.decoder         = self.tts.model.speech_tokenizer.model.decoder.eval()
        self.hidden_size     = self.decoder.config.hidden_size
        self.num_code_groups = self.tts.model.talker.code_predictor.model.config.num_code_groups
        self.scale           = output_sample_rate / 24000.0
        self.upsample_rate   = self.tts.model.speech_tokenizer.model.decode_upsample_rate

        for param in self.tts.model.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.decoder.pre_transformer.parameters():
            param.requires_grad = False

        self._fuse_decoder_weights()

        self.num_heads     = self.decoder.pre_transformer.layers._modules['0'].self_attn.config.num_attention_heads
        self.qk_heads      = self.num_heads + self.num_heads
        self.head_dim      = self.decoder.pre_transformer.layers._modules['0'].self_attn.head_dim
        self.head_dim_half = self.head_dim // 2

        position_ids      = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq          = self.decoder.pre_transformer.rotary_emb.inv_freq
        idx_theta         = (position_ids * inv_freq).unsqueeze(1).unsqueeze(0)
        cos, sin          = torch.cos(idx_theta), torch.sin(idx_theta)
        self.rope_emb_cos = torch.cat([cos,  cos], dim=-1).half()
        self.rope_emb_sin = torch.cat([-sin, sin], dim=-1).half()

    # ── Weight Fusion ─────────────────────────────────────────────────────────

    def _fuse_decoder_weights(self):
        """Fuse QKV projections, layer norms, layer scales, and final norm for the decoder pre-transformer."""
        scale_factor = self.decoder.pre_transformer.layers._modules['0'].self_attn.head_dim ** -0.25
        norm_factor  = self.hidden_size ** 0.5

        with torch.no_grad():
            for layer in self.decoder.pre_transformer.layers:
                self._fuse_qkv_projection(layer, scale_factor)
                self._fuse_input_layernorm_into_qkv(layer, norm_factor)
                self._fuse_gate_up_projection(layer, norm_factor)
                self._fuse_layer_scales(layer)

            # Fuse final pre-transformer norm into output projection
            final_norm_weight = self.decoder.pre_transformer.norm.weight.unsqueeze(0) * norm_factor
            self.decoder.pre_transformer.output_proj.weight.mul_(final_norm_weight)
            del self.decoder.pre_transformer.norm

    def _fuse_qkv_projection(self, layer, scale_factor):
        """Fuse Q, K, V projections into a single QKV linear with scale-factor baked in."""
        q_proj, k_proj, v_proj = (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
        )
        in_features  = q_proj.in_features
        out_features = q_proj.out_features + k_proj.out_features + v_proj.out_features
        qkv          = torch.nn.Linear(in_features, out_features, bias=(q_proj.bias is not None))
        qkv.weight.copy_(torch.cat([q_proj.weight * scale_factor, k_proj.weight * scale_factor, v_proj.weight], dim=0))
        if q_proj.bias is not None:
            qkv.bias.copy_(torch.cat([q_proj.bias * scale_factor, k_proj.bias * scale_factor, v_proj.bias], dim=0))
        layer.self_attn.qkv   = qkv
        layer.self_attn.q_dim = q_proj.out_features
        layer.self_attn.k_dim = k_proj.out_features
        layer.self_attn.v_dim = v_proj.out_features
        del layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj

    def _fuse_input_layernorm_into_qkv(self, layer, norm_factor):
        """Absorb input_layernorm (RMSNorm) into the QKV projection."""
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        layer.self_attn.qkv.weight.mul_(input_norm_weight)
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up         = layer.mlp.gate_proj, layer.mlp.up_proj
        gate_up          = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    def _fuse_layer_scales(self, layer):
        """Fuse self-attention and MLP layer scales into output projections."""
        scale  = layer.self_attn_layer_scale.scale
        linear = layer.self_attn.o_proj
        if linear.bias is not None:
            linear.bias.data.mul_(scale.data)
        linear.weight.data.mul_(scale.data.unsqueeze(1))

        scale  = layer.mlp_layer_scale.scale
        linear = layer.mlp.down_proj
        if linear.bias is not None:
            linear.bias.data.mul_(scale.data)
        linear.weight.data.mul_(scale.data.unsqueeze(1))

    # ── Utility ───────────────────────────────────────────────────────────────

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        """Apply modified RMS normalization (with optional overflow scaling)."""
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True))

    def rotate_half(self, x):
        """Rotate using flip() — more efficient than split()+cat() in ONNX Runtime."""
        x = x.view(1, -1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(1, -1, self.qk_heads, self.head_dim)

    def forward(self, generated_codec):
        concat_codec = generated_codec.reshape(1, -1, self.num_code_groups).transpose(1, 2)

        hidden_states = self.decoder.quantizer.decode(concat_codec)
        hidden_states = self.decoder.pre_conv(hidden_states).transpose(1, 2)
        hidden_states = self.decoder.pre_transformer.input_proj(hidden_states)

        ids_len      = hidden_states.shape[1].unsqueeze(0)
        rope_emb_cos = self.rope_emb_cos[:, :ids_len].float()
        rope_emb_sin = self.rope_emb_sin[:, :ids_len].float()

        # Pre-transformer (full-attention over the entire codec sequence)
        for layer in self.decoder.pre_transformer.layers:
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv           = layer.self_attn.qkv(hidden_states)
            qkv           = qkv.reshape(1, -1, self.qk_heads + self.num_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_heads], dim=-2)
            qk            = qk * rope_emb_cos + self.rotate_half(qk) * rope_emb_sin
            q, k          = torch.split(qk, [self.num_heads, self.num_heads], dim=-2)
            q             = q.transpose(1, 2)
            k             = k.permute(0, 2, 3, 1)
            v             = v.transpose(1, 2)
            attn          = torch.softmax(torch.matmul(q, k), dim=-1)
            attn          = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, [layer.mlp.down_proj.in_features] * 2, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # Final norm, output projection, and convolutional decode
        hidden_states = self._rms_norm(hidden_states)
        hidden_states = self.decoder.pre_transformer.output_proj(hidden_states)
        generated_wav = hidden_states.transpose(1, 2)

        for blocks in self.decoder.upsample:
            for block in blocks:
                generated_wav = block(generated_wav)
        for block in self.decoder.decoder:
            generated_wav = block(generated_wav)

        generated_wav = generated_wav[..., : ids_len * self.upsample_rate]

        if self.scale < 1.0:
            generated_wav = torch.nn.functional.interpolate(generated_wav, scale_factor=self.scale, mode='linear', align_corners=False)
        generated_wav = generated_wav * 32767.0
        if self.scale > 1.0:
            generated_wav = torch.nn.functional.interpolate(generated_wav, scale_factor=self.scale, mode='linear', align_corners=False)

        generated_wav = generated_wav.clamp(min=-32768.0, max=32767.0).to(torch.int16)
        generated_len = generated_wav.shape[-1].unsqueeze(0)
        return generated_wav, generated_len


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Model Modules
# ─────────────────────────────────────────────────────────────────────────────
class TTS_MAIN(torch.nn.Module):
    """
    Main talker auto-regressive transformer.
    Returns updated KV caches, the final hidden state, and the greedy codec token.
    """

    def __init__(self, tts):
        super().__init__()
        self.tts = tts.model.talker
        self._replace_gelu_with_tanh_approximation(self.tts.model)

        self.head_dim             = self.tts.config.head_dim
        self.head_dim_half        = self.head_dim // 2
        self.hidden_size          = self.tts.config.hidden_size
        self.num_heads            = self.tts.config.num_attention_heads
        self.num_key_value_heads  = self.tts.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads             = self.num_heads + self.num_key_value_heads
        self.num_layers           = self.tts.config.num_hidden_layers
        self.num_layers_2         = self.num_layers * 2
        self.num_layers_3         = self.num_layers * 3
        self.num_layers_4         = self.num_layers * 4
        self.num_layers_5         = self.num_layers * 5

        suppress_ids = [
            token_id
            for token_id in range(self.tts.config.vocab_size - 1024, self.tts.config.vocab_size)
            if token_id != self.tts.config.codec_eos_token_id
        ]
        suppress_logits_bias = torch.zeros((1, self.tts.config.vocab_size), dtype=torch.float32)
        suppress_logits_bias[..., suppress_ids] = -1e7
        self.register_buffer("suppress_logits_bias", suppress_logits_bias, persistent=False)

        self.save_key   = [None] * self.num_layers
        self.save_value = [None] * self.num_layers

        self._fuse_weights()

    # ── Weight Fusion ─────────────────────────────────────────────────────────

    def _fuse_weights(self):
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = self.hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5
        with torch.no_grad():
            for layer in self.tts.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn           = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        in_features    = int(q_proj.in_features)
        out_features   = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias       = any(p.bias is not None for p in (q_proj, k_proj, v_proj))
        qkv            = torch.nn.Linear(in_features, out_features, bias=has_bias)

        attn.q_out_features  = int(q_proj.out_features)
        attn.k_out_features  = int(k_proj.out_features)
        attn.v_out_features  = int(v_proj.out_features)
        attn.qkv_in_features = in_features
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        del attn.q_proj, attn.k_proj, attn.v_proj

        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)
        q_norm_repeated     = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated     = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight   = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up           = layer.mlp.gate_proj, layer.mlp.up_proj
        gate_up            = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ── Utility ───────────────────────────────────────────────────────────────

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True))

    def rotate_half(self, x, batch_size):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]

        for i, layer in enumerate(self.tts.model.layers):
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv           = layer.self_attn.qkv(hidden_states)
            qkv           = qkv.reshape(1, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk            = self._rms_norm(qk) * layer.self_attn.qk_norm_weight
            qk_rot        = qk * rotary_pos_emb_cos + self.rotate_half(qk, 1) * rotary_pos_emb_sin
            q, k          = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q             = q.reshape(1, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q             = q.permute(0, 2, 3, 1, 4)

            if USE_F16_KV:
                k, v = k.half(), v.half()

            k = torch.cat((all_inputs[i],                   k.permute(0, 3, 2, 4, 1)), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v.transpose(1, 3)),        dim=-2)
            self.save_key[i]   = k
            self.save_value[i] = v

            if USE_F16_KV:
                k, v = k.float(), v.float()

            attn          = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
            attn          = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(1, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, [layer.mlp.down_proj.in_features] * 2, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        last_hidden_states = self.tts.model.norm(hidden_states[:, -1])
        logits             = self.tts.codec_head(last_hidden_states) + self.suppress_logits_bias

        return *self.save_key, *self.save_value, last_hidden_states.unsqueeze(1), logits


class TTS_PREDICTOR(torch.nn.Module):
    """
    RVQ code-predictor transformer.
    Accepts KV caches + hidden states and returns updated KV caches
    together with the final hidden state (fed to the LM heads).
    """

    def __init__(self, tts):
        super().__init__()
        self.tts = tts.model.talker.code_predictor
        self._replace_gelu_with_tanh_approximation(self.tts.model)

        self.head_dim             = self.tts.config.head_dim
        self.head_dim_half        = self.head_dim // 2
        self.hidden_size          = self.tts.config.hidden_size
        self.num_heads            = self.tts.config.num_attention_heads
        self.num_key_value_heads  = self.tts.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads             = self.num_heads + self.num_key_value_heads
        self.num_layers           = self.tts.config.num_hidden_layers
        self.num_layers_2         = self.num_layers * 2
        self.num_layers_3         = self.num_layers * 3
        self.num_layers_4         = self.num_layers * 4
        self.num_layers_5         = self.num_layers * 5

        self.save_key   = [None] * self.num_layers
        self.save_value = [None] * self.num_layers

        self._fuse_weights()

    # ── Weight Fusion ─────────────────────────────────────────────────────────

    def _fuse_weights(self):
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = self.hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5
        with torch.no_grad():
            for layer in self.tts.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn           = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        in_features    = int(q_proj.in_features)
        out_features   = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias       = any(p.bias is not None for p in (q_proj, k_proj, v_proj))
        qkv            = torch.nn.Linear(in_features, out_features, bias=has_bias)

        attn.q_out_features  = int(q_proj.out_features)
        attn.k_out_features  = int(k_proj.out_features)
        attn.v_out_features  = int(v_proj.out_features)
        attn.qkv_in_features = in_features
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        del attn.q_proj, attn.k_proj, attn.v_proj

        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)
        q_norm_repeated     = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated     = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight   = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up           = layer.mlp.gate_proj, layer.mlp.up_proj
        gate_up            = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ── Utility ───────────────────────────────────────────────────────────────

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True))

    def rotate_half(self, x, batch_size):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        batch_size         = hidden_states.shape[0].unsqueeze(0)
        hidden_states = self.tts.small_to_mtp_projection(hidden_states)

        for i, layer in enumerate(self.tts.model.layers):
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            qkv           = layer.self_attn.qkv(hidden_states)
            qkv           = qkv.reshape(batch_size, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v         = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            qk            = self._rms_norm(qk) * layer.self_attn.qk_norm_weight
            qk_rot        = qk * rotary_pos_emb_cos + self.rotate_half(qk, batch_size) * rotary_pos_emb_sin
            q, k          = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q             = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q             = q.permute(0, 2, 3, 1, 4)

            if USE_F16_KV:
                k, v = k.half(), v.half()

            k = torch.cat((all_inputs[i],                  k.permute(0, 3, 2, 4, 1)), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v.transpose(1, 3)),        dim=-2)
            self.save_key[i]   = k
            self.save_value[i] = v

            if USE_F16_KV:
                k, v = k.float(), v.float()

            attn          = torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1)
            attn          = torch.matmul(attn, v).permute(0, 3, 1, 2, 4).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)
            gate_up       = layer.mlp.gate_up_proj(hidden_states)
            gate, up      = torch.split(gate_up, [layer.mlp.down_proj.in_features] * 2, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        hidden_states = self._rms_norm(hidden_states[:, -1])
        return *self.save_key, *self.save_value, hidden_states


class TTS_PREDICTOR_LM_HEAD(torch.nn.Module):
    """
    LM head for one RVQ code group.
    Fuses the final layer-norm into the linear projection weight.
    """

    def __init__(self, tts, indices):
        super().__init__()
        self.tts    = tts.model.talker.code_predictor
        hidden_size = self.tts.config.hidden_size
        norm_factor = hidden_size ** 0.5

        with torch.no_grad():
            w                = self.tts.model.norm.weight.unsqueeze(0) * norm_factor
            original_lm_head = self.tts.lm_head._modules[f"{indices}"]
            self.lm_head     = torch.nn.Linear(original_lm_head.in_features, original_lm_head.out_features, bias=False)
            self.lm_head.weight.copy_(original_lm_head.weight * w)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


# ─────────────────────────────────────────────────────────────────────────────
# Decoding Strategy Modules
# ─────────────────────────────────────────────────────────────────────────────
class GREEDY_SEARCH(torch.nn.Module):
    """Select the token with the highest logit (greedy decoding)."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id        = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_layers):
        super().__init__()
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
        super().__init__()
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
        super().__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized      = logits.gather(1, target_indices) * penalty_value
        logits         = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Return the argmax index over the vocabulary dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():

        # ── Phase 1 : Model initialisation ──────────────────────────────────
        mod.Qwen3TTSTokenizerV2ConvNeXtBlock = Qwen3TTSTokenizerV2ConvNeXtBlockUnfused
        model = Qwen3TTSModel.from_pretrained(download_path, device_map="cpu", dtype=torch.float32, attn_implementation="eager")
        mod.Qwen3TTSTokenizerV2ConvNeXtBlock = Qwen3TTSTokenizerV2ConvNeXtBlock

        for i, upsample_block in enumerate(model.model.speech_tokenizer.model.decoder.upsample):
            for j, module in enumerate(upsample_block):
                if isinstance(module, Qwen3TTSTokenizerV2ConvNeXtBlockUnfused):
                    model.model.speech_tokenizer.model.decoder.upsample[i][j] = Qwen3TTSTokenizerV2ConvNeXtBlock.from_unfused(module)

        for layer in model.model.speech_tokenizer.model.decoder.quantizer.rvq_first.vq.layers:
            layer._codebook.precompute_embedding()
        for layer in model.model.speech_tokenizer.model.decoder.quantizer.rvq_rest.vq.layers:
            layer._codebook.precompute_embedding()
        for module in model.model.speech_tokenizer.model.decoder.decoder.modules():
            if isinstance(module, SnakeBeta):
                module.precompute()

        model.model = model.model.eval()

        stft_model = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE, pad_mode='constant', center_pad=True).eval()

        # ── Phase 2 : Shared constants & KV-cache helpers ────────────────────
        head_dim              = model.model.talker.config.head_dim
        hidden_size           = model.model.talker.config.hidden_size
        hidden_size_pred      = model.model.talker.code_predictor.config.hidden_size
        num_heads             = model.model.talker.config.num_attention_heads
        num_kv_heads          = model.model.talker.config.num_key_value_heads
        vocab_size            = model.model.talker.code_predictor.config.vocab_size
        num_code_groups       = model.model.talker.code_predictor.config.num_code_groups
        NUM_CODE_GROUPS_MINUS = num_code_groups - 1

        batch_size  = 1
        ids_len     = torch.tensor([10],        dtype=torch.int64)
        history_len = torch.tensor([0],         dtype=torch.int64)
        kv_seq_len  = ids_len + history_len
        beam_size   = torch.tensor([batch_size], dtype=torch.int64)

        kv_specs   = [('key', 4), ('value', 3)]
        kv_dtype   = torch.float16 if USE_F16_KV else torch.float32
        kv_tensors = {
            'key':   torch.zeros([batch_size, num_kv_heads, 1, head_dim,    history_len], dtype=kv_dtype),
            'value': torch.zeros([batch_size, num_kv_heads, 1, history_len, head_dim],    dtype=kv_dtype)
        }

        def get_kv_io(tensors_dict, _num_layers, batch_axis='batch', seq_axis='history_len', out_seq_axis='kv_seq_len'):
            """Build input tensors, name lists, and dynamic-axes dicts for KV caches."""
            inputs, in_names, out_names, axes = [], [], [], {}
            for name, dim in kv_specs:
                t = tensors_dict[name]
                for i in range(_num_layers):
                    in_n, out_n = f'in_{name}_{i}', f'out_{name}_{i}'
                    inputs.append(t)
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n]  = {0: batch_axis, dim: seq_axis}
                    axes[out_n] = {0: batch_axis, dim: out_seq_axis}
            return inputs, in_names, out_names, axes

        # ── Phase 3 : Embedding & gather exports ─────────────────────────────
        text_ids = torch.zeros([1, ids_len], dtype=torch.int32)
        torch.onnx.export(
            TTS_EMBED_A(model),
            (text_ids,),
            onnx_model_Embed_A,
            input_names=['text_ids'],
            output_names=['text_embed'],
            dynamic_axes={
                'text_ids':   {1: 'ids_len'},
                'text_embed': {1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del text_ids

        codec_ids = torch.zeros([1, ids_len], dtype=torch.int32)
        torch.onnx.export(
            TTS_EMBED_B(model),
            (codec_ids,),
            onnx_model_Embed_B,
            input_names=['codec_ids'],
            output_names=['codec_embed'],
            dynamic_axes={
                'codec_ids':   {1: 'ids_len'},
                'codec_embed': {1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        codec_ids            = torch.zeros([num_code_groups, ids_len], dtype=torch.int32)
        codec_embed_0        = torch.zeros([1, ids_len, hidden_size],  dtype=torch.float32)
        trailing_text_hidden = torch.zeros([1, ids_len, hidden_size],  dtype=torch.float32)
        gather_id            = torch.tensor([0], dtype=torch.int32)
        torch.onnx.export(
            TTS_EMBED_C(model),
            (codec_ids, codec_embed_0, trailing_text_hidden, gather_id),
            onnx_model_Embed_C,
            input_names=['codec_ids', 'codec_embed_0', 'trailing_text_hidden', 'gather_id'],
            output_names=['codec_embed_sum'],
            dynamic_axes={
                'codec_ids':            {0: 'num_code_groups', 1: 'ids_len'},
                'codec_embed_0':        {1: 'ids_len'},
                'trailing_text_hidden': {1: 'ids_len'},
                'codec_embed_sum':      {1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del codec_embed_0, trailing_text_hidden, gather_id

        path_name = onnx_model_Embed_D.split('.')[0]
        input_ids = torch.zeros([batch_size, 1], dtype=torch.int32)
        for i in range(NUM_CODE_GROUPS_MINUS):
            torch.onnx.export(
                TTS_EMBED_D(model, i),
                (input_ids,),
                f'{path_name}_{i}.onnx',
                input_names=['input_ids'],
                output_names=['codec_embed'],
                dynamic_axes={
                    'input_ids':   {0: 'batch'},
                    'codec_embed': {0: 'batch'}
                },
                opset_version=OPSET,
                dynamo=False
            )
        del input_ids

        torch.onnx.export(
            GATHER_0(),
            (codec_ids,),
            onnx_model_Gather_0,
            input_names=['codec_ids'],
            output_names=['target_id'],
            dynamic_axes={'codec_ids': {0: 'num_code_groups', 1: 'ids_len'}},
            opset_version=OPSET,
            dynamo=False
        )
        del codec_ids

        # ── Phase 4 : Concatenation / element-wise exports ───────────────────
        codec_embed_0 = torch.zeros([1, 1, hidden_size], dtype=torch.float32)
        codec_embed_1 = torch.zeros([1, 1, hidden_size], dtype=torch.float32)
        torch.onnx.export(
            CONCAT_EMBED(),
            (codec_embed_0, codec_embed_1),
            onnx_model_Concat_Embed,
            input_names=['codec_embed_0', 'codec_embed_1'],
            output_names=['codec_embed_concat'],
            dynamic_axes={
                'codec_embed_0':      {1: 'ids_len_0'},
                'codec_embed_1':      {1: 'ids_len_1'},
                'codec_embed_concat': {1: 'ids_len_plus'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del codec_embed_1

        codec_0 = torch.ones([num_code_groups, 1],  dtype=torch.int32)
        codec_1 = torch.ones([num_code_groups, 10], dtype=torch.int32)
        torch.onnx.export(
            CONCAT_IDS(),
            (codec_0, codec_1),
            onnx_model_Concat_Ids,
            input_names=['codec_0', 'codec_1'],
            output_names=['codec_concat'],
            dynamic_axes={
                'codec_0':      {0: 'group_count', 1: 'ids_len_0'},
                'codec_1':      {0: 'group_count', 1: 'ids_len_1'},
                'codec_concat': {0: 'group_count', 1: 'ids_len_plus'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del codec_0, codec_1

        slide_window_input_0 = torch.zeros([1, num_code_groups * STREAM_WINDOW_FRAMES], dtype=torch.int32)
        slide_window_input_1 = torch.zeros([1, num_code_groups], dtype=torch.int32)
        torch.onnx.export(
            SLIDE_WINDOW(model),
            (slide_window_input_0, slide_window_input_1),
            onnx_model_Slide_Window,
            input_names=['codec_0', 'codec_1'],
            output_names=['codec_slide'],
            dynamic_axes={
                'codec_0':     {1: 'window_len'},
                'codec_1':     {1: 'frame_len'},
                'codec_slide': {1: 'window_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del slide_window_input_0, slide_window_input_1

        # ── Phase 5 : Pre-processing & speech encoder exports ────────────────
        language_embed        = torch.zeros([1, 1,  hidden_size], dtype=torch.float32)
        target_text_embed     = torch.zeros([1, 10, hidden_size], dtype=torch.float32)

        if MODE == "voice_design":
            # voice_design: no speaker, no ref audio — only language_embed + target_text_embed
            torch.onnx.export(
                TTS_PREPROCESS(model, mode="voice_design"),
                (language_embed, target_text_embed),
                onnx_model_Preprocess,
                input_names=['language_embed', 'target_text_embed'],
                output_names=['hidden_states', 'ids_len', 'trailing_text_hidden', 'trailing_len_minus'],
                dynamic_axes={
                    'target_text_embed':    {1: 'ids_len'},
                    'hidden_states':        {1: 'ids_len'},
                    'trailing_text_hidden': {1: 'trailing_len'}
                },
                opset_version=OPSET,
                dynamo=False
            )
            del codec_embed_0, language_embed, target_text_embed
            del stft_model
        else:
            speaker_embed         = torch.zeros([1, 1,  hidden_size], dtype=torch.float32)
            ref_prompt_text_embed = torch.zeros([1, 10, hidden_size], dtype=torch.float32)
            torch.onnx.export(
                TTS_PREPROCESS(model, mode=MODE),
                (codec_embed_0, speaker_embed, language_embed, ref_prompt_text_embed, target_text_embed),
                onnx_model_Preprocess,
                input_names=['codec_embed', 'speaker_embed', 'language_embed', 'ref_prompt_text_embed', 'target_text_embed'],
                output_names=['hidden_states', 'ids_len', 'trailing_text_hidden', 'trailing_len_minus'],
                dynamic_axes={
                    'codec_embed':           {1: 'ids_len'},
                    'ref_prompt_text_embed': {1: 'ids_len'},
                    'target_text_embed':     {1: 'ids_len'},
                    'hidden_states':         {1: 'ids_len'},
                    'trailing_text_hidden':  {1: 'trailing_len'}
                },
                opset_version=OPSET,
                dynamo=False
            )
            del codec_embed_0, speaker_embed, language_embed, ref_prompt_text_embed, target_text_embed

            # Encoder: only exported for voice_clone mode
            if MODE == "voice_clone":
                prompt_audio = torch.zeros([1, 1, MAX_PROMPT_AUDIO_LEN], dtype=torch.int16)
                with torch.amp.autocast('cpu', dtype=torch.float16, enabled=USE_F16_ENCODER):
                    torch.onnx.export(
                        TTS_ENCODER(model, IN_SAMPLE_RATE, MAX_SEQ_LEN, stft_model, NFFT_STFT, N_MELS),
                        (prompt_audio,),
                        onnx_model_Encoder,
                        input_names=['prompt_audio'],
                        output_names=['ref_code', 'ref_code_len', 'speaker_embed'],
                        dynamic_axes={
                            'prompt_audio': {2: 'audio_len'},
                            'ref_code':     {1: 'ref_code_len'}
                        },
                        opset_version=OPSET,
                        dynamo=False
                    )
                    del prompt_audio, stft_model
            else:
                # custom_voice: encoder not needed
                del stft_model

        # ── Phase 6 : Rotary embedding & attention-mask exports ──────────────
        torch.onnx.export(
            TTS_MAIN_ROTARY_MASK_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Main_Rotary_Mask_Text_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos':     {1: 'ids_len'},
                'rotary_sin':     {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        torch.onnx.export(
            TTS_MAIN_ROTARY_MASK_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Main_Rotary_Mask_Text_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )
        torch.onnx.export(
            TTS_PREDICTOR_ROTARY_MASK_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Pred_Rotary_Mask_Text_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos':     {1: 'ids_len'},
                'rotary_sin':     {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        torch.onnx.export(
            TTS_PREDICTOR_ROTARY_MASK_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Main_Rotary_Mask_Text_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )
        del history_len

        # ── Phase 7 : Code-predictor & LM-head exports ───────────────────────
        num_layers_predictor = model.model.talker.code_predictor.config.num_hidden_layers
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, num_layers_predictor, 'batch', 'history_len', 'kv_seq_len')
        hidden_states  = torch.ones([batch_size, ids_len, hidden_size], dtype=torch.float32)
        rotary_cos     = torch.zeros([1, ids_len, 1, 1, head_dim],      dtype=torch.float32)
        rotary_sin     = rotary_cos
        attention_mask = torch.zeros([1, 1, 1, ids_len, kv_seq_len],    dtype=torch.float32)

        all_inputs_base  = kv_ins + [hidden_states, rotary_cos, rotary_sin, attention_mask]
        input_names_base = kv_in_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
        torch.onnx.export(
            TTS_PREDICTOR(model),
            tuple(all_inputs_base),
            onnx_model_Predictor,
            input_names=input_names_base,
            output_names=kv_out_names + ['last_hidden_state'],
            dynamic_axes={
                **kv_axes,
                'hidden_states':     {0: 'batch', 1: 'ids_len'},
                'rotary_cos':        {1: 'ids_len'},
                'rotary_sin':        {1: 'ids_len'},
                'attention_mask':    {3: 'ids_len', 4: 'kv_seq_len'},
                'last_hidden_state': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del kv_seq_len

        path_name         = onnx_model_Pred_LmHead.split('.')[0]
        hidden_states = torch.ones([batch_size, ids_len, hidden_size_pred], dtype=torch.float32)
        last_hidden_state = hidden_states[:, -1]
        for i in range(NUM_CODE_GROUPS_MINUS):
            torch.onnx.export(
                TTS_PREDICTOR_LM_HEAD(model, i),
                (last_hidden_state,),
                f'{path_name}_{i}.onnx',
                input_names=['last_hidden_state'],
                output_names=['logits'],
                dynamic_axes={
                    'last_hidden_state': {0: 'batch'},
                    'logits':            {0: 'batch'}
                },
                opset_version=OPSET,
                dynamo=False
            )
        del last_hidden_state

        # ── Phase 8 : Main talker transformer export ─────────────────────────
        num_layers = model.model.talker.config.num_hidden_layers
        hidden_states = torch.ones([batch_size, ids_len, hidden_size], dtype=torch.float32)
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, num_layers, 'batch', 'history_len', 'kv_seq_len')
        all_inputs_base  = kv_ins + [hidden_states, rotary_cos, rotary_sin, attention_mask]
        input_names_base = kv_in_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
        torch.onnx.export(
            TTS_MAIN(model),
            tuple(all_inputs_base),
            onnx_model_Main,
            input_names=input_names_base,
            output_names=kv_out_names + ['last_hidden_state', 'max_ids'],
            dynamic_axes={
                **kv_axes,
                'hidden_states':     {1: 'ids_len'},
                'rotary_cos':        {1: 'ids_len'},
                'rotary_sin':        {1: 'ids_len'},
                'attention_mask':    {3: 'ids_len', 4: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del ids_len, hidden_states, rotary_cos, rotary_sin, attention_mask
        del all_inputs_base, input_names_base
        del kv_ins, kv_in_names, kv_out_names, kv_axes

        # ── Phase 9 : Audio decoder export ───────────────────────────────────
        ref_code        = torch.zeros([num_code_groups, 10], dtype=torch.int32)
        ref_code_len    = torch.tensor([10],                 dtype=torch.int64)
        generated_codec = torch.zeros([1, num_code_groups * 10], dtype=torch.int32)
        decoder = TTS_DECODER(model, OUT_SAMPLE_RATE, MAX_SEQ_LEN, mode="voice_clone" if MODE == "voice_clone" else "custom_voice")
        del model
        gc.collect()

        torch.onnx.export(
            decoder,
            (generated_codec,),
            onnx_model_Decoder,
            input_names=['generated_codec'],
            output_names=['generated_wav', 'generated_len'],
            dynamic_axes={
                'generated_codec': {1: 'generated_codec_len'},
                'generated_wav':   {2: 'generated_wav_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        if STREAMING:
            # Export streaming decoder with fixed static input shape (1, num_code_groups * STREAM_WINDOW_FRAMES).
            # No dynamic axes → all tensor shapes are compile-time constants in the ONNX graph.
            generated_codec_stream = torch.zeros([1, num_code_groups * STREAM_WINDOW_FRAMES], dtype=torch.int32)
            torch.onnx.export(
                decoder,
                (generated_codec_stream,),
                onnx_model_Decoder_Stream,
                input_names=['generated_codec'],
                output_names=['generated_wav', 'generated_len'],
                dynamic_axes=None,
                opset_version=OPSET,
                dynamo=False
            )
            del generated_codec_stream

        del decoder, ref_code, ref_code_len, generated_codec

        # ── Phase 10 : Decoding strategy & beam-search exports ───────────────
        logits     = torch.ones((beam_size, vocab_size), dtype=torch.float32)
        save_id_in = torch.zeros((beam_size, 10),        dtype=torch.int32)

        torch.onnx.export(
            GREEDY_SEARCH(),
            (logits, save_id_in),
            onnx_model_Greedy,
            input_names=['logits', 'save_id_in'],
            output_names=['max_logits_idx', 'save_id_out'],
            dynamic_axes={
                'logits':         {0: 'batch'},
                'save_id_in':     {0: 'batch', 1: 'history_len'},
                'max_logits_idx': {0: 'batch'},
                'save_id_out':    {0: 'batch', 1: 'history_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALTY_RANGE],  dtype=torch.int64)
        torch.onnx.export(
            APPLY_PENALTY(),
            (logits, save_id_in, penalty_value, penalty_range),
            onnx_model_Penalty,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
            output_names=['logits_out'],
            dynamic_axes={
                'logits_in':  {0: 'batch', 1: 'vocab_size'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'logits_out': {0: 'batch', 1: 'vocab_size'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del penalty_value, penalty_range

        torch.onnx.export(
            ARGMAX(),
            (logits,),
            onnx_model_Argmax,
            input_names=['logits'],
            output_names=['max_logits_idx'],
            dynamic_axes={
                'logits':         {0: 'batch', 1: 'vocab_size'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        num_layers_beam = num_layers_predictor * len(kv_specs)

        kv_tensors_greedy = {k: v[[0]] for k, v in kv_tensors.items()}
        kv_ins_g, kv_in_names_g, kv_out_names_g, kv_axes_g = get_kv_io(kv_tensors_greedy, num_layers_predictor, 'batch', 'history_len', 'history_len')
        kv_axes_g = {k: v for k, v in kv_axes_g.items() if k not in kv_out_names_g}
        torch.onnx.export(
            FIRST_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins_g + [logits[[0]], save_id_in, beam_size]),
            onnx_model_First_Beam,
            input_names=kv_in_names_g + ['logits', 'save_id_in', 'beam_size'],
            output_names=['out_' + n[3:] for n in kv_in_names_g] + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={
                **kv_axes_g,
                'logits':           {0: 'batch'},
                'save_id_in':       {0: 'batch', 1: 'history_len'},
                'top_beam_prob':    {0: 'batch'},
                'top_beam_indices': {0: 'batch'},
                'max_logits_idx':   {0: 'batch'},
                'batch_indices':    {0: 'batch'},
                'save_id_out':      {0: 'batch', 1: 'history_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del kv_tensors_greedy, kv_ins_g, kv_in_names_g, kv_out_names_g, kv_axes_g

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(
            kv_tensors, num_layers_predictor, 'batch', 'history_len', 'kv_seq_len'
        )
        previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
        topK_t        = torch.tensor([TOP_K],        dtype=torch.int64)
        torch.onnx.export(
            SECOND_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + [logits, save_id_in, previous_prob, beam_size, topK_t]),
            onnx_model_Second_Beam,
            input_names=kv_in_names + ['logits', 'save_id_in', 'previous_prob', 'beam_size', 'topK'],
            output_names=kv_out_names + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={
                **kv_axes,
                'logits':           {0: 'batch'},
                'save_id_in':       {0: 'batch', 1: 'history_len'},
                'previous_prob':    {0: 'batch'},
                'save_id_out':      {0: 'batch', 1: 'history_len'},
                'top_beam_prob':    {0: 'batch'},
                'top_beam_indices': {0: 'batch'},
                'max_logits_idx':   {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del kv_tensors, kv_ins, kv_in_names, kv_out_names, kv_axes
        del logits, save_id_in, previous_prob, beam_size, topK_t
        gc.collect()

    print('\nExport done!\n\nStart running the TTS by ONNXRuntime.\nNow loading . . . it could cost minutes.')



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


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE ID MAPPING
# ══════════════════════════════════════════════════════════════════════════════
LANGUAGE_ID_MAP = {
    'english':    2050,
    'german':     2053,
    'spanish':    2054,
    'chinese':    2055,
    'japanese':   2058,
    'french':     2061,
    'korean':     2064,
    'russian':    2069,
    'italian':    2070,
    'portuguese': 2071,
}

# Speaker ID and dialect mappings (custom_voice mode)
# These values come from the model's config.json talker_config.
SPEAKER_ID_MAP = {
    'serena':   3066,
    'vivian':   3065,
    'uncle_fu': 3010,
    'ryan':     3061,
    'aiden':    2861,
    'ono_anna': 2873,
    'sohee':    2864,
    'eric':     2875,
    'dylan':    2878,
}
SPEAKER_DIALECT_MAP = {
    'serena':   False,
    'vivian':   False,
    'uncle_fu': False,
    'ryan':     False,
    'aiden':    False,
    'ono_anna': False,
    'sohee':    False,
    'eric':     'sichuan_dialect',
    'dylan':    'beijing_dialect',
}
DIALECT_LANGUAGE_ID_MAP = {
    'beijing_dialect':  2074,
    'sichuan_dialect':  2062,
}

# Resolve language_id with dialect override for custom_voice
# (Mirrors inference_custom_voice.py dialect handling)
language_id = LANGUAGE_ID_MAP[TTS_LANGUAGE.lower()]
if MODE == "custom_voice":
    speaker_key = SPEAKER_NAME.lower()
    if speaker_key not in SPEAKER_ID_MAP:
        raise ValueError(f"Unknown speaker '{SPEAKER_NAME}'. Supported: {list(SPEAKER_ID_MAP.keys())}")
    speaker_id_value = SPEAKER_ID_MAP[speaker_key]
    # Dialect override: if language is Chinese and speaker has a dialect, use dialect language_id
    dialect = SPEAKER_DIALECT_MAP.get(speaker_key, False)
    if dialect and TTS_LANGUAGE.lower() in ('chinese', 'auto'):
        language_id = DIALECT_LANGUAGE_ID_MAP[dialect]
elif MODE == "voice_design":
    # voice_design: no speaker needed, voice identity comes from VOICE_DESCRIPTION
    if not VOICE_DESCRIPTION or not VOICE_DESCRIPTION.strip():
        raise ValueError("VOICE_DESCRIPTION is required for voice_design mode.")


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
        'gpu_mem_limit':                      24 * (1024 ** 3),   # 24 GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',                # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',                # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',  # ["default", "high_performance", "minimum_power"]
        'device_filter':              'gpu',               # [gpu, npu, any]
        'disable_metacommands':       'false',             # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',             # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'              # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device        = 'cpu' if 'dml' in device_type else device_type

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}


# ══════════════════════════════════════════════════════════════════════════════
# DECODING STRATEGY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = (REPEAT_PENALTY != 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & STOP TOKENS & PROMPT
# ══════════════════════════════════════════════════════════════════════════════
tokenizer      = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
STOP_TOKEN_SET = set(STOP_TOKEN)

if MODE == "voice_clone":
    prompt_tokens = tokenizer(prompt_text, return_tensors='np')['input_ids'].astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
NUM_CODE_GROUPS_MINUS = 15  # Fixed value for the QwenTTS-0.6B

# --- Embed A ---
ort_session_Embed_A = create_session(onnx_model_Embed_A, **packed_settings)
in_name_Embed_A     = get_in_names(ort_session_Embed_A)
out_name_Embed_A    = get_out_names(ort_session_Embed_A)

# --- Embed B ---
ort_session_Embed_B = create_session(onnx_model_Embed_B, **packed_settings)
in_name_Embed_B     = get_in_names(ort_session_Embed_B)
out_name_Embed_B    = get_out_names(ort_session_Embed_B)

# --- Embed C ---
ort_session_Embed_C = create_session(onnx_model_Embed_C, **packed_settings)
in_name_Embed_C     = get_in_names(ort_session_Embed_C)
out_name_Embed_C    = get_out_names(ort_session_Embed_C)
in_meta_Embed_C     = ort_session_Embed_C._inputs_meta

# --- Embed D (multi-group) ---
path_name           = onnx_model_Embed_D.split('.')[0]
ort_session_Embed_D = [create_session(f'{path_name}_{i}.onnx', **packed_settings) for i in range(NUM_CODE_GROUPS_MINUS)]
in_name_Embed_D     = get_in_names(ort_session_Embed_D[0])
out_name_Embed_D    = get_out_names(ort_session_Embed_D[0])

# --- Gather / Concat ---
ort_session_Gather_0     = create_session(onnx_model_Gather_0, **packed_settings)
in_name_Gather_0         = get_in_names(ort_session_Gather_0)
out_name_Gather_0        = get_out_names(ort_session_Gather_0)

ort_session_Concat_Embed = create_session(onnx_model_Concat_Embed, **packed_settings)
in_name_Concat_Embed     = get_in_names(ort_session_Concat_Embed)
out_name_Concat_Embed    = get_out_names(ort_session_Concat_Embed)

ort_session_Concat_Ids   = create_session(onnx_model_Concat_Ids, **packed_settings)
in_name_Concat_Ids       = get_in_names(ort_session_Concat_Ids)
out_name_Concat_Ids      = get_out_names(ort_session_Concat_Ids)

# --- Preprocess ---
ort_session_Preprocess = create_session(onnx_model_Preprocess, **packed_settings)
in_name_Preprocess     = get_in_names(ort_session_Preprocess)
out_name_Preprocess    = get_out_names(ort_session_Preprocess)

# --- Encoder (voice_clone only) ---
if MODE == "voice_clone":
    ort_session_Encoder = create_session(onnx_model_Encoder, **packed_settings)
    in_name_Encoder     = get_in_names(ort_session_Encoder)
    out_name_Encoder    = get_out_names(ort_session_Encoder)

# --- LM Head (multi-group) ---
path_name          = onnx_model_Pred_LmHead.split('.')[0]
ort_session_LmHead = [create_session(f'{path_name}_{i}.onnx', **packed_settings) for i in range(NUM_CODE_GROUPS_MINUS)]
in_name_LmHead     = get_in_names(ort_session_LmHead[0])
out_name_LmHead    = get_out_names(ort_session_LmHead[0])
vocab_size_LmHead  = ort_session_LmHead[0]._outputs_meta[0].shape[1]

# --- Decoder ---
ort_session_Decoder = create_session(onnx_model_Decoder, **packed_settings)
in_name_Decoder     = get_in_names(ort_session_Decoder)
out_name_Decoder    = get_out_names(ort_session_Decoder)

# --- Decoder Stream (streaming mode only) ---
if STREAMING:
    ort_session_Decoder_Stream = create_session(onnx_model_Decoder_Stream, **packed_settings)
    in_name_Decoder_Stream     = get_in_names(ort_session_Decoder_Stream)
    out_name_Decoder_Stream    = get_out_names(ort_session_Decoder_Stream)
    STREAM_WINDOW_FRAMES = ort_session_Decoder_Stream.get_outputs()[0].shape[-1] // SAMPLES_PER_CODEC_FRAME

    ort_session_Slide_Window = create_session(onnx_model_Slide_Window, **packed_settings)
    in_name_Slide_Window     = get_in_names(ort_session_Slide_Window)
    out_name_Slide_Window    = get_out_names(ort_session_Slide_Window)

# --- Main Rotary ---
ort_session_Main_Rotary_Text_Prefill = create_session(onnx_model_Main_Rotary_Mask_Text_Prefill, **packed_settings)
in_name_Main_Rotary_Text_Prefill     = get_in_names(ort_session_Main_Rotary_Text_Prefill)
out_name_Main_Rotary_Text_Prefill    = get_out_names(ort_session_Main_Rotary_Text_Prefill)

ort_session_Main_Rotary_Text_Decode  = create_session(onnx_model_Main_Rotary_Mask_Text_Decode, **packed_settings)
in_name_Main_Rotary_Text_Decode      = get_in_names(ort_session_Main_Rotary_Text_Decode)
out_name_Main_Rotary_Text_Decode     = get_out_names(ort_session_Main_Rotary_Text_Decode)

# --- Predictor Rotary ---
ort_session_Predictor_Rotary_Text_Prefill = create_session(onnx_model_Pred_Rotary_Mask_Text_Prefill, **packed_settings)
in_name_Predictor_Rotary_Text_Prefill     = get_in_names(ort_session_Predictor_Rotary_Text_Prefill)
out_name_Predictor_Rotary_Text_Prefill    = get_out_names(ort_session_Predictor_Rotary_Text_Prefill)

ort_session_Predictor_Rotary_Text_Decode  = create_session(onnx_model_Main_Rotary_Mask_Text_Decode, **packed_settings)
in_name_Predictor_Rotary_Text_Decode      = get_in_names(ort_session_Predictor_Rotary_Text_Decode)
out_name_Predictor_Rotary_Text_Decode     = get_out_names(ort_session_Predictor_Rotary_Text_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")

in_name_Main  = get_in_names(ort_session_Main)
out_name_Main = get_out_names(ort_session_Main)
in_meta_Main  = ort_session_Main._inputs_meta

# Derived index offsets
num_keys_values_Main        = len(out_name_Main) - 2
num_layers_Main             = num_keys_values_Main // 2
num_keys_values_Main_plus_1 = num_keys_values_Main + 1
num_keys_values_Main_plus_2 = num_keys_values_Main + 2
num_keys_values_Main_plus_3 = num_keys_values_Main + 3
num_keys_values_Main_plus_4 = num_keys_values_Main + 4

# Partitioned name lists
in_name_Main_kv      = in_name_Main[:num_keys_values_Main]
in_name_Main_others  = in_name_Main[num_keys_values_Main:]
out_name_Main_kv     = out_name_Main[:num_keys_values_Main]
out_name_Main_others = out_name_Main[num_keys_values_Main:]

# Dtype introspection
kv_dtype_Main     = np.float16 if 'float16' in in_meta_Main[0].type else np.float32
hidden_dtype_Main = np.float16 if 'float16' in in_meta_Main[num_keys_values_Main].type else np.float32

# Initial KV cache
init_past_keys_Main   = create_ort_with_shape((1, in_meta_Main[0].shape[1],               1, in_meta_Main[0].shape[3],               0), kv_dtype_Main, kv_device, DEVICE_ID)
init_past_values_Main = create_ort_with_shape((1, in_meta_Main[num_layers_Main].shape[1], 1, 0, in_meta_Main[num_layers_Main].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)

# --- Predictor ---
ort_session_Predictor = create_session(onnx_model_Predictor, **packed_settings)

in_name_Predictor         = get_in_names(ort_session_Predictor)
out_name_Predictor        = get_out_names(ort_session_Predictor)
in_meta_Predictor         = ort_session_Predictor._inputs_meta
num_keys_values_Predictor = len(out_name_Predictor) - 1
num_layers_Predictor      = num_keys_values_Predictor // 2

# Derived index offsets
num_keys_values_Predictor_plus_1 = num_keys_values_Predictor + 1
num_keys_values_Predictor_plus_2 = num_keys_values_Predictor + 2
num_keys_values_Predictor_plus_3 = num_keys_values_Predictor + 3
num_keys_values_Predictor_plus_4 = num_keys_values_Predictor + 4

# Partitioned name lists
in_name_Predictor_kv      = in_name_Predictor[:num_keys_values_Predictor]
in_name_Predictor_others  = in_name_Predictor[num_keys_values_Predictor:]
out_name_Predictor_kv     = out_name_Predictor[:num_keys_values_Predictor]
out_name_Predictor_hidden = out_name_Predictor[num_keys_values_Predictor]

# Dtype introspection
kv_dtype_Predictor     = np.float16 if 'float16' in in_meta_Predictor[0].type else np.float32
hidden_dtype_Predictor = np.float16 if 'float16' in in_meta_Predictor[num_keys_values_Predictor].type else np.float32

if hidden_dtype_Predictor != hidden_dtype_Main:
    raise ValueError(f"Hidden state dtype mismatch between Main and Predictor: {hidden_dtype_Main} vs {hidden_dtype_Predictor}")

# Initial KV cache
init_past_keys_Predictor   = create_ort_with_shape((1, in_meta_Predictor[0].shape[1],                    1, in_meta_Predictor[0].shape[3],                    0), kv_dtype_Predictor, kv_device, DEVICE_ID)
init_past_values_Predictor = create_ort_with_shape((1, in_meta_Predictor[num_layers_Predictor].shape[1], 1, 0, in_meta_Predictor[num_layers_Predictor].shape[4]), kv_dtype_Predictor, kv_device, DEVICE_ID)

# --- Greedy ---
ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
in_name_Greedy     = get_in_names(ort_session_Greedy)
out_name_Greedy    = get_out_names(ort_session_Greedy)

# --- Argmax ---
ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
in_name_Argmax     = get_in_names(ort_session_Argmax)
out_name_Argmax    = get_out_names(ort_session_Argmax)

# --- Beam Search (optional) ---
if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")

    ort_session_First_Beam     = create_session(onnx_model_First_Beam, **packed_settings)
    in_name_First_Beam         = get_in_names(ort_session_First_Beam)
    out_name_First_Beam        = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_kv      = in_name_First_Beam[:num_keys_values_Predictor]
    in_name_First_Beam_logits  = in_name_First_Beam[num_keys_values_Predictor]
    out_name_First_Beam_kv     = out_name_First_Beam[:num_keys_values_Predictor]
    out_name_First_Beam_others = out_name_First_Beam[num_keys_values_Predictor_plus_1:]

    ort_session_Second_Beam     = create_session(onnx_model_Second_Beam, **packed_settings)
    in_name_Second_Beam         = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam        = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_kv      = in_name_Second_Beam[:num_keys_values_Predictor]
    in_name_Second_Beam_logits  = in_name_Second_Beam[num_keys_values_Predictor]
    out_name_Second_Beam_kv     = out_name_Second_Beam[:num_keys_values_Predictor]
    out_name_Second_Beam_others = out_name_Second_Beam[num_keys_values_Predictor_plus_1:]

    beam_ids_buf   = create_ort_with_shape([BEAM_SIZE, 1], np.int32,               device_type, DEVICE_ID)
    beam_score_buf = create_ort_with_shape([BEAM_SIZE, 1], hidden_dtype_Predictor, device_type, DEVICE_ID)

# --- Penalty (optional) ---
if USE_PENALTY:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
    in_name_Penalty     = get_in_names(ort_session_Penalty)
    out_name_Penalty    = get_out_names(ort_session_Penalty)

    penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
if MODE == "voice_clone":
    input_ids_prompt = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_tokens, device_type, DEVICE_ID)
init_history_len          = create_ort_with_data([0],                                           np.int64,          device_type, DEVICE_ID)
init_predictor_ids_len    = create_ort_with_data([2],                                           np.int64,          device_type, DEVICE_ID)
init_generated_codec      = create_ort_with_shape([1, 0],                                       np.int32,          device_type, DEVICE_ID)
top_k                     = create_ort_with_data([TOP_K],                                       np.int64,          device_type, DEVICE_ID)
beam_size                 = create_ort_with_data([BEAM_SIZE],                                   np.int64,          device_type, DEVICE_ID)
gather_id_0               = create_ort_with_data([0],                                           np.int32,          device_type, DEVICE_ID)
gather_id_cache           = [create_ort_with_data([i],                                          np.int32,          device_type, DEVICE_ID) for i in range(MAX_SEQ_LEN)]
init_trailing_text_hidden = create_ort_with_shape([1, 1, in_meta_Embed_C[2].shape[2]],          hidden_dtype_Main, device_type, DEVICE_ID)
init_predictor_save_id    = create_ort_with_shape([BEAM_SIZE if USE_BEAM_SEARCH else 1, 0],     np.int32,          device_type, DEVICE_ID)
init_main_greedy_ids      = create_ort_with_shape([1, 0],                                       np.int32,          device_type, DEVICE_ID)
init_decode_attn_mask     = create_ort_with_shape([1, 1, 1, 1, 1],                              hidden_dtype_Main, device_type, DEVICE_ID)
init_frame_codec_ids      = create_ort_with_shape([1, 0],                                       np.int32,          device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# INPUT FEED DICTIONARIES
# ══════════════════════════════════════════════════════════════════════════════
input_feed_Encoder                       = {}
input_feed_Embed_A                       = {}
input_feed_Embed_B                       = {}
input_feed_Embed_C                       = {}
input_feed_Embed_D                       = {}
input_feed_Gather_0                      = {}
input_feed_Concat_Embed                  = {}
input_feed_Concat_Ids                    = {}
input_feed_Concat_Aux                    = {}
input_feed_Preprocess                    = {}
input_feed_Main_Rotary_Text_Prefill      = {}
input_feed_Main_Rotary_Text_Decode       = {}
input_feed_Predictor_Rotary_Text_Prefill = {}
input_feed_Predictor_Rotary_Text_Decode  = {}
input_feed_Main                          = {}
input_feed_Predictor                     = {}
input_feed_LmHead                        = {}
input_feed_Argmax                        = {}
input_feed_Greedy                        = {}
input_feed_First_Beam                    = {}
input_feed_Second_Beam                   = {}
input_feed_Penalty                       = {}
input_feed_Decoder                       = {}
_stream_concat_feed                      = {}  # dedicated feed dict for streaming window concat


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT AUDIO ENCODING & REFERENCE EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════
save_generated_wav = []
empty_segment = np.zeros(int(OUT_SAMPLE_RATE * 0.2), dtype=np.int16)  # 200ms
encoder_time = 0.0

# Common: compute language_embed for all modes
input_feed_Embed_B[in_name_Embed_B[0]] = create_ort_with_data([[language_id]], np.int32, device_type, DEVICE_ID)
language_embed = ort_session_Embed_B.run_with_ort_values(out_name_Embed_B, input_feed_Embed_B, run_options=run_options)[0]

if MODE == "voice_clone":
    # Encode reference audio to obtain ref_code, speaker_embed
    prompt_audio = np.array(AudioSegment.from_file(prompt_audio_path).set_channels(1).set_frame_rate(IN_SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if USE_AUDIO_NORMALIZER:
        prompt_audio = audio_normalizer(prompt_audio)
    prompt_audio = onnxruntime.OrtValue.ortvalue_from_numpy(prompt_audio.reshape(1, 1, -1), device_type, DEVICE_ID)

    input_feed_Encoder[in_name_Encoder[0]] = prompt_audio
    encoder_start = time.perf_counter()
    ref_code, ref_code_len, speaker_embed = ort_session_Encoder.run_with_ort_values(out_name_Encoder, input_feed_Encoder, run_options=run_options)
    encoder_time = time.perf_counter() - encoder_start
    print(f'\nEncoder time: {encoder_time:.2f} s')

    input_feed_Embed_A[in_name_Embed_A[0]] = input_ids_prompt
    ref_prompt_text_embed = ort_session_Embed_A.run_with_ort_values(out_name_Embed_A, input_feed_Embed_A, run_options=run_options)[0]

    input_feed_Gather_0[in_name_Gather_0[0]] = ref_code
    ref_ids_0 = ort_session_Gather_0.run_with_ort_values(out_name_Gather_0, input_feed_Gather_0, run_options=run_options)[0]

    input_feed_Embed_B[in_name_Embed_B[0]] = ref_ids_0
    codec_embed_0 = ort_session_Embed_B.run_with_ort_values(out_name_Embed_B, input_feed_Embed_B, run_options=run_options)[0]

    input_feed_Embed_C[in_name_Embed_C[0]] = ref_code
    input_feed_Embed_C[in_name_Embed_C[1]] = codec_embed_0
    input_feed_Embed_C[in_name_Embed_C[2]] = init_trailing_text_hidden
    input_feed_Embed_C[in_name_Embed_C[3]] = gather_id_0
    codec_embed = ort_session_Embed_C.run_with_ort_values(out_name_Embed_C, input_feed_Embed_C, run_options=run_options)[0]

elif MODE == "custom_voice":
    # Obtain speaker_embed by passing speaker_id through Embed_B
    input_feed_Embed_B[in_name_Embed_B[0]] = create_ort_with_data([[speaker_id_value]], np.int32, device_type, DEVICE_ID)
    speaker_embed = ort_session_Embed_B.run_with_ort_values(out_name_Embed_B, input_feed_Embed_B, run_options=run_options)[0]

    codec_embed           = create_ort_with_shape([1, 0, in_meta_Embed_C[2].shape[2]], np.float32, device_type, DEVICE_ID)
    ref_prompt_text_embed = create_ort_with_shape([1, 0, in_meta_Embed_C[2].shape[2]], np.float32, device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-POPULATE FIXED INPUT FEED ENTRIES
# ══════════════════════════════════════════════════════════════════════════════

# Preprocess fixed inputs
if MODE == "voice_design":
    input_feed_Preprocess[in_name_Preprocess[0]] = language_embed
else:
    input_feed_Preprocess[in_name_Preprocess[0]] = codec_embed
    input_feed_Preprocess[in_name_Preprocess[1]] = speaker_embed
    input_feed_Preprocess[in_name_Preprocess[2]] = language_embed
    input_feed_Preprocess[in_name_Preprocess[3]] = ref_prompt_text_embed

# Predictor Rotary Text Prefill fixed inputs
input_feed_Predictor_Rotary_Text_Prefill[in_name_Predictor_Rotary_Text_Prefill[0]] = init_predictor_ids_len
input_feed_Predictor_Rotary_Text_Prefill[in_name_Predictor_Rotary_Text_Prefill[1]] = init_history_len

# Predictor KV cache initial state
for i in range(num_layers_Predictor):
    input_feed_Predictor[in_name_Predictor[i]] = init_past_keys_Predictor
for i in range(num_layers_Predictor, num_keys_values_Predictor):
    input_feed_Predictor[in_name_Predictor[i]] = init_past_values_Predictor

# Penalty fixed inputs
if USE_PENALTY:
    input_feed_Penalty[in_name_Penalty[2]] = penalty_value
    input_feed_Penalty[in_name_Penalty[3]] = penalty_range

if USE_BEAM_SEARCH:
    input_feed_First_Beam[in_name_First_Beam[num_keys_values_Predictor_plus_2]]   = beam_size
    input_feed_Second_Beam[in_name_Second_Beam[num_keys_values_Predictor_plus_3]] = beam_size
    input_feed_Second_Beam[in_name_Second_Beam[num_keys_values_Predictor_plus_4]] = top_k


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING DECODE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _stream_decode(window_ort, is_first_decode):
    _t0 = time.perf_counter()
    wav_ort = ort_session_Decoder_Stream.run_with_ort_values(out_name_Decoder_Stream, {in_name_Decoder_Stream[0]: window_ort}, run_options=run_options)[0]
    return wav_ort, is_first_decode, time.perf_counter() - _t0


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTOR STEPS
# ══════════════════════════════════════════════════════════════════════════════
def predictor_steps(codec_token_main, last_hidden_state_Main, gather_id):
    if not USE_BEAM_SEARCH:
        input_feed_Concat_Aux[in_name_Concat_Ids[0]] = init_frame_codec_ids
        input_feed_Concat_Aux[in_name_Concat_Ids[1]] = codec_token_main
        frame_codec_buf = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]

    predictor_save_id = init_predictor_save_id

    input_feed_Embed_B[in_name_Embed_B[0]] = codec_token_main
    codec_embed_main = ort_session_Embed_B.run_with_ort_values(out_name_Embed_B, input_feed_Embed_B, run_options=run_options)[0]

    input_feed_Concat_Embed[in_name_Concat_Embed[0]] = last_hidden_state_Main
    input_feed_Concat_Embed[in_name_Concat_Embed[1]] = codec_embed_main
    hidden_states_predictor = ort_session_Concat_Embed.run_with_ort_values(out_name_Concat_Embed, input_feed_Concat_Embed, run_options=run_options)[0]

    rotary_cos_Predictor, rotary_sin_Predictor, attention_mask_Predictor, kv_seq_len_Predictor = ort_session_Predictor_Rotary_Text_Prefill.run_with_ort_values(out_name_Predictor_Rotary_Text_Prefill, input_feed_Predictor_Rotary_Text_Prefill, run_options=run_options)

    input_feed_Predictor[in_name_Predictor_others[3]] = attention_mask_Predictor

    # Reset Predictor KV cache to empty for this new predictor sequence
    for i in range(num_layers_Predictor):
        input_feed_Predictor[in_name_Predictor[i]] = init_past_keys_Predictor
    for i in range(num_layers_Predictor, num_keys_values_Predictor):
        input_feed_Predictor[in_name_Predictor[i]] = init_past_values_Predictor

    is_prefill_step_Predictor = True

    for num_decode_Predictor in range(NUM_CODE_GROUPS_MINUS):
        input_feed_Predictor[in_name_Predictor_others[0]] = hidden_states_predictor
        input_feed_Predictor[in_name_Predictor_others[1]] = rotary_cos_Predictor
        input_feed_Predictor[in_name_Predictor_others[2]] = rotary_sin_Predictor

        all_outputs_Predictor       = ort_session_Predictor.run_with_ort_values(out_name_Predictor, input_feed_Predictor, run_options=run_options)
        last_hidden_state_Predictor = all_outputs_Predictor[num_keys_values_Predictor]

        input_feed_LmHead[in_name_LmHead[0]] = last_hidden_state_Predictor
        logits_Predictor = ort_session_LmHead[num_decode_Predictor].run_with_ort_values(out_name_LmHead, input_feed_LmHead, run_options=run_options)[0]

        if USE_PENALTY and not is_prefill_step_Predictor:
            input_feed_Penalty[in_name_Penalty[0]] = logits_Predictor
            input_feed_Penalty[in_name_Penalty[1]] = predictor_save_id
            logits_Predictor = ort_session_Penalty.run_with_ort_values(out_name_Penalty, input_feed_Penalty, run_options=run_options)[0]

        if USE_BEAM_SEARCH:
            if is_prefill_step_Predictor:
                input_feed_First_Beam.update(zip(in_name_First_Beam_kv, all_outputs_Predictor))
                input_feed_First_Beam[in_name_First_Beam_logits]                            = logits_Predictor
                input_feed_First_Beam[in_name_First_Beam[num_keys_values_Predictor_plus_1]] = predictor_save_id
                all_outputs_Predictor = ort_session_First_Beam.run_with_ort_values(out_name_First_Beam, input_feed_First_Beam, run_options=run_options)
            else:
                input_feed_Second_Beam.update(zip(in_name_Second_Beam_kv, all_outputs_Predictor))
                input_feed_Second_Beam[in_name_Second_Beam_logits]                            = logits_Predictor
                input_feed_Second_Beam[in_name_Second_Beam[num_keys_values_Predictor_plus_1]] = predictor_save_id
                input_feed_Second_Beam[in_name_Second_Beam[num_keys_values_Predictor_plus_2]] = predictor_beam_score
                all_outputs_Predictor = ort_session_Second_Beam.run_with_ort_values(out_name_Second_Beam, input_feed_Second_Beam, run_options=run_options)

            predictor_save_id      = all_outputs_Predictor[num_keys_values_Predictor]
            predictor_beam_score   = all_outputs_Predictor[num_keys_values_Predictor_plus_1]
            next_codec_ids         = all_outputs_Predictor[num_keys_values_Predictor_plus_2]
            codec_token_predictor  = all_outputs_Predictor[num_keys_values_Predictor_plus_3]
        else:
            if USE_PENALTY:
                input_feed_Greedy[in_name_Greedy[0]] = logits_Predictor
                input_feed_Greedy[in_name_Greedy[1]] = predictor_save_id
                codec_token_predictor, predictor_save_id = ort_session_Greedy.run_with_ort_values(out_name_Greedy, input_feed_Greedy, run_options=run_options)
            else:
                input_feed_Argmax[in_name_Argmax[0]] = logits_Predictor
                codec_token_predictor = ort_session_Argmax.run_with_ort_values(out_name_Argmax, input_feed_Argmax, run_options=run_options)[0]

            next_codec_ids = codec_token_predictor

        input_feed_Predictor.update(zip(in_name_Predictor_kv, all_outputs_Predictor))

        input_feed_Embed_D[in_name_Embed_D[0]] = next_codec_ids
        hidden_states_predictor = ort_session_Embed_D[num_decode_Predictor].run_with_ort_values(out_name_Embed_D, input_feed_Embed_D, run_options=run_options)[0]

        input_feed_Predictor_Rotary_Text_Decode[in_name_Predictor_Rotary_Text_Decode[0]] = kv_seq_len_Predictor
        rotary_cos_Predictor, rotary_sin_Predictor, kv_seq_len_Predictor = ort_session_Predictor_Rotary_Text_Decode.run_with_ort_values(out_name_Predictor_Rotary_Text_Decode, input_feed_Predictor_Rotary_Text_Decode, run_options=run_options)

        if is_prefill_step_Predictor:
            input_feed_Predictor[in_name_Predictor_others[3]] = init_decode_attn_mask
            is_prefill_step_Predictor = False

        if not USE_BEAM_SEARCH:
            input_feed_Concat_Aux[in_name_Concat_Ids[0]] = frame_codec_buf
            input_feed_Concat_Aux[in_name_Concat_Ids[1]] = codec_token_predictor
            frame_codec_buf = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]

    if USE_BEAM_SEARCH:
        input_feed_Gather_0[in_name_Gather_0[0]] = predictor_save_id
        best_predictor_save_id = ort_session_Gather_0.run_with_ort_values(out_name_Gather_0, input_feed_Gather_0, run_options=run_options)[0]

        input_feed_Concat_Aux[in_name_Concat_Ids[0]] = codec_token_main
        input_feed_Concat_Aux[in_name_Concat_Ids[1]] = best_predictor_save_id
        frame_codec_ids = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]
    else:
        if USE_PENALTY:
            input_feed_Concat_Aux[in_name_Concat_Ids[0]] = codec_token_main
            input_feed_Concat_Aux[in_name_Concat_Ids[1]] = predictor_save_id
            frame_codec_ids = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]

        else:
            frame_codec_ids = frame_codec_buf

    input_feed_Embed_C[in_name_Embed_C[0]] = frame_codec_ids
    input_feed_Embed_C[in_name_Embed_C[1]] = codec_embed_main
    input_feed_Embed_C[in_name_Embed_C[3]] = gather_id
    hidden_states_main = ort_session_Embed_C.run_with_ort_values(out_name_Embed_C, input_feed_Embed_C, run_options=run_options)[0]

    input_feed_Concat_Ids[in_name_Concat_Ids[1]] = frame_codec_ids
    generated_codec = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Ids, run_options=run_options)[0]

    return hidden_states_main, generated_codec, frame_codec_ids


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
total_audio_samples = 0
total_generation_time = 0.0
total_decoder_time = 0.0

# custom_voice / voice_design: pre-compute instruct embed once (shared across all targets)
instruct_embed = None
instruct_text = ""
if MODE == "custom_voice" and INSTRUCT_TEXT:
    instruct_text = INSTRUCT_TEXT
elif MODE == "voice_design":
    instruct_text = VOICE_DESCRIPTION

if instruct_text:
    instruct_prompt = "<|im_start|>system\n" + instruct_text + "<|im_end|>\n"
    instruct_tokens = tokenizer(instruct_prompt, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids_instruct = onnxruntime.OrtValue.ortvalue_from_numpy(instruct_tokens, device_type, DEVICE_ID)
    input_feed_Embed_A[in_name_Embed_A[0]] = input_ids_instruct
    instruct_embed = ort_session_Embed_A.run_with_ort_values(out_name_Embed_A, input_feed_Embed_A, run_options=run_options)[0]

for target_idx, target in enumerate(target_tts):
    target_tokens    = tokenizer(target, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids_target = onnxruntime.OrtValue.ortvalue_from_numpy(target_tokens, device_type, DEVICE_ID)
    main_greedy_ids  = init_main_greedy_ids
    is_prefill_step_Main = True
    main_time_total = 0.0
    predictor_time_total = 0.0

    if STREAMING:
        _stream_frame_window = []
        _stream_futures      = [] 
        _stream_decode_count = 0  
        _stream_executor     = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    input_feed_Embed_A[in_name_Embed_A[0]] = input_ids_target
    target_text_embed = ort_session_Embed_A.run_with_ort_values(out_name_Embed_A, input_feed_Embed_A, run_options=run_options)[0]

    if MODE == "voice_design":
        # voice_design preprocess: inputs are (language_embed, target_text_embed)
        input_feed_Preprocess[in_name_Preprocess[1]] = target_text_embed
    else:
        input_feed_Preprocess[in_name_Preprocess[4]] = target_text_embed
    hidden_states, ids_len, trailing_text_hidden, trailing_len_minus = ort_session_Preprocess.run_with_ort_values(out_name_Preprocess, input_feed_Preprocess, run_options=run_options)

    # custom_voice / voice_design: prepend instruct embed before hidden_states
    if instruct_embed is not None:
        input_feed_Concat_Embed[in_name_Concat_Embed[0]] = instruct_embed
        input_feed_Concat_Embed[in_name_Concat_Embed[1]] = hidden_states
        hidden_states = ort_session_Concat_Embed.run_with_ort_values(out_name_Concat_Embed, input_feed_Concat_Embed, run_options=run_options)[0]
        # Update ids_len to include instruct tokens
        ids_len = create_ort_with_data([hidden_states.shape()[1]], np.int64, device_type, DEVICE_ID)

    trailing_len_minus_numpy               = trailing_len_minus.numpy()
    input_feed_Embed_C[in_name_Embed_C[2]] = trailing_text_hidden

    input_feed_Main_Rotary_Text_Prefill[in_name_Main_Rotary_Text_Prefill[0]] = ids_len
    input_feed_Main_Rotary_Text_Prefill[in_name_Main_Rotary_Text_Prefill[1]] = init_history_len
    rotary_cos_Main, rotary_sin_Main, attention_mask_Main, kv_seq_len_Main   = ort_session_Main_Rotary_Text_Prefill.run_with_ort_values(out_name_Main_Rotary_Text_Prefill, input_feed_Main_Rotary_Text_Prefill, run_options=run_options)

    input_feed_Main[in_name_Main_others[3]] = attention_mask_Main

    # Reset Main KV cache to empty for this new target
    for i in range(num_layers_Main):
        input_feed_Main[in_name_Main[i]] = init_past_keys_Main
    for i in range(num_layers_Main, num_keys_values_Main):
        input_feed_Main[in_name_Main[i]] = init_past_values_Main

    num_decode_Main = 0
    generate_limit  = MAX_SEQ_LEN - ids_len.numpy()

    while num_decode_Main < generate_limit:
        input_feed_Main[in_name_Main_others[0]] = hidden_states
        input_feed_Main[in_name_Main_others[1]] = rotary_cos_Main
        input_feed_Main[in_name_Main_others[2]] = rotary_sin_Main

        main_step_start        = time.perf_counter()
        all_outputs_Main       = ort_session_Main.run_with_ort_values(out_name_Main, input_feed_Main, run_options=run_options)
        main_time_total       += time.perf_counter() - main_step_start
        last_hidden_state_Main = all_outputs_Main[num_keys_values_Main]
        logits_Main            = all_outputs_Main[num_keys_values_Main_plus_1]

        if USE_PENALTY and num_decode_Main >= PENALTY_RANGE:
            input_feed_Penalty[in_name_Penalty[0]] = logits_Main
            input_feed_Penalty[in_name_Penalty[1]] = main_greedy_ids
            logits_Main = ort_session_Penalty.run_with_ort_values(out_name_Penalty, input_feed_Penalty, run_options=run_options)[0]

        input_feed_Argmax[in_name_Argmax[0]] = logits_Main
        codec_token_main = ort_session_Argmax.run_with_ort_values(out_name_Argmax, input_feed_Argmax, run_options=run_options)[0]

        max_logits_idx = codec_token_main.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET and num_decode_Main >= MIN_SEQ_LEN:
            break

        input_feed_Concat_Aux[in_name_Concat_Ids[0]] = main_greedy_ids
        input_feed_Concat_Aux[in_name_Concat_Ids[1]] = codec_token_main
        main_greedy_ids = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, input_feed_Concat_Aux, run_options=run_options)[0]

        if is_prefill_step_Main:
            input_feed_Main[in_name_Main_others[3]]      = init_decode_attn_mask
            input_feed_Concat_Ids[in_name_Concat_Ids[0]] = init_generated_codec
            is_prefill_step_Main = False

        gather_id = gather_id_cache[num_decode_Main] if num_decode_Main <= trailing_len_minus_numpy else trailing_len_minus
        predictor_step_start = time.perf_counter()
        hidden_states, generated_codec, _frame_codec = predictor_steps(codec_token_main, last_hidden_state_Main, gather_id)
        predictor_time_total += time.perf_counter() - predictor_step_start
        input_feed_Concat_Ids[in_name_Concat_Ids[0]] = generated_codec

        if STREAMING:
            _stream_frame_window.append(_frame_codec)
            if len(_stream_frame_window) > STREAM_WINDOW_FRAMES:
                _stream_frame_window.pop(0)
            # Launch async decoder as soon as STREAM_WINDOW_FRAMES frames are available.
            if len(_stream_frame_window) == STREAM_WINDOW_FRAMES:
                _is_first = (_stream_decode_count == 0)
                if _is_first:
                    # First window: build by concatenating all frames
                    _window_ort = _stream_frame_window[0]
                    for _fi in range(1, STREAM_WINDOW_FRAMES):
                        _stream_concat_feed[in_name_Concat_Ids[0]] = _window_ort
                        _stream_concat_feed[in_name_Concat_Ids[1]] = _stream_frame_window[_fi]
                        _window_ort = ort_session_Concat_Ids.run_with_ort_values(out_name_Concat_Ids, _stream_concat_feed, run_options=run_options)[0]
                else:
                    # Subsequent windows: slide by dropping first frame and appending new frame
                    _stream_concat_feed[in_name_Slide_Window[0]] = _window_ort
                    _stream_concat_feed[in_name_Slide_Window[1]] = _frame_codec
                    _window_ort = ort_session_Slide_Window.run_with_ort_values(out_name_Slide_Window, _stream_concat_feed, run_options=run_options)[0]
                _stream_decode_count += 1
                _stream_futures.append(_stream_executor.submit(_stream_decode, _window_ort, _is_first))

        input_feed_Main_Rotary_Text_Decode[in_name_Main_Rotary_Text_Decode[0]] = kv_seq_len_Main
        rotary_cos_Main, rotary_sin_Main, kv_seq_len_Main = ort_session_Main_Rotary_Text_Decode.run_with_ort_values(out_name_Main_Rotary_Text_Decode, input_feed_Main_Rotary_Text_Decode, run_options=run_options)
        input_feed_Main.update(zip(in_name_Main_kv, all_outputs_Main))

        num_decode_Main += 1

    # Decoder
    if STREAMING:
        # Wait for all background decoder tasks to complete, then assemble the audio.
        _stream_executor.shutdown(wait=True)
        _stream_results = [f.result() for f in _stream_futures]
        if _stream_results:
            # Convert OrtValues to numpy only here (final assembly before soundfile write).
            _wav_chunks = []
            _stream_decoder_time = 0.0
            for _wav_ort, _is_first, _elapsed in _stream_results:
                _wav_arr = _wav_ort.numpy().reshape(-1)
                _wav_chunks.append(_wav_arr if _is_first else _wav_arr[-SAMPLES_PER_CODEC_FRAME:])
                _stream_decoder_time += _elapsed
            generated_wav = np.concatenate(_wav_chunks)
            if USE_AUDIO_NORMALIZER:
                generated_wav = audio_normalizer(generated_wav)
            decoder_time = _stream_decoder_time
        else:
            input_feed_Decoder[in_name_Decoder[0]] = generated_codec
            _fb_start = time.perf_counter()
            _fb = ort_session_Decoder.run_with_ort_values(out_name_Decoder, input_feed_Decoder, run_options=run_options)[0]
            decoder_time = time.perf_counter() - _fb_start
            generated_wav = _fb.numpy().reshape(-1)
            if USE_AUDIO_NORMALIZER:
                generated_wav = audio_normalizer(generated_wav)
    else:
        # Standard non-streaming: decode the full accumulated codec in one shot.
        input_feed_Decoder[in_name_Decoder[0]] = generated_codec
        decoder_start = time.perf_counter()
        generated_wav = ort_session_Decoder.run_with_ort_values(out_name_Decoder, input_feed_Decoder, run_options=run_options)[0]
        decoder_time = time.perf_counter() - decoder_start
        generated_wav = generated_wav.numpy().reshape(-1)
        if USE_AUDIO_NORMALIZER:
            generated_wav = audio_normalizer(generated_wav)

    # Timing statistics for this target
    main_tokens_per_sec = num_decode_Main / main_time_total if main_time_total > 0 else 0
    predictor_tokens_per_sec = (num_decode_Main * NUM_CODE_GROUPS_MINUS) / predictor_time_total if predictor_time_total > 0 else 0
    audio_duration = len(generated_wav) / OUT_SAMPLE_RATE
    target_gen_time = main_time_total + predictor_time_total + decoder_time
    total_audio_samples += len(generated_wav)
    total_generation_time += target_gen_time
    total_decoder_time += decoder_time

    print(f'\n┌──────────────────┬──────────┬──────────┬──────────────┐')
    print(f'│  Target {target_idx:<9}│          │          │              │')
    print(f'├──────────────────┼──────────┼──────────┼──────────────┤')
    print(f'│ Stage            │   Tokens │  Time(s) │    Tokens/s  │')
    print(f'├──────────────────┼──────────┼──────────┼──────────────┤')
    print(f'│ Main             │ {num_decode_Main:>8d} │ {main_time_total:>8.2f} │ {main_tokens_per_sec:>12.2f} │')
    print(f'│ Predictor        │ {num_decode_Main * NUM_CODE_GROUPS_MINUS:>8d} │ {predictor_time_total:>8.2f} │ {predictor_tokens_per_sec:>12.2f} │')
    print(f'│ Decoder          │        — │ {decoder_time:>8.2f} │            — │')
    print(f'├──────────────────┼──────────┴──────────┴──────────────┤')
    print(f'│ Audio duration   │ {audio_duration:>8.2f} s{" ":>25}│')
    print(f'│ Target RTF       │ {target_gen_time / audio_duration:>8.3f}{" ":>27}│' if audio_duration > 0 else f'│ Target RTF       │{"N/A":>9}{" ":>27}│')
    print(f'└──────────────────┴────────────────────────────────────┘')

    save_generated_wav.append(generated_wav)
    save_generated_wav.append(empty_segment)  # Append silence between target sentence
if save_generated_wav:
    total_audio_duration = total_audio_samples / OUT_SAMPLE_RATE
    overall_rtf = (encoder_time + total_generation_time) / total_audio_duration if total_audio_duration > 0 else float('inf')
    print(f'\n┌─────────────────────────┬─────────────────┐')
    print(f'│    Overall Statistics   │                 │')
    print(f'├─────────────────────────┼─────────────────┤')
    print(f'│ Encoder time            │ {encoder_time:>8.2f} s      │')
    print(f'│ Total decoder time      │ {total_decoder_time:>8.2f} s      │')
    print(f'│ Total generation time   │ {total_generation_time:>8.2f} s      │')
    print(f'│ Total audio duration    │ {total_audio_duration:>8.2f} s      │')
    print(f'│ Overall RTF             │ {overall_rtf:>8.3f}        │')
    print(f'└─────────────────────────┴─────────────────┘')

    save_generated_wav = np.concatenate(save_generated_wav)
    sf.write(generated_audio_path, save_generated_wav, OUT_SAMPLE_RATE, format='WAVEX')
    
