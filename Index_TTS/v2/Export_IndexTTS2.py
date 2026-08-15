"""Export official IndexTTS2 v2 or v2.5 modules to a compact ONNX package."""

# pyright: reportMissingImports=false

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


# User configuration
#
# This single exporter supports both official releases. The retained v2.5
# package is selected by default; set this to ``"2"`` for IndexTTS2 v2. The
# release-specific adapter preserves original graph inputs, codec path, and CFM
# schedule.
MODEL_VERSION = "2.5"  # "2" | "2.5"

home_path = Path.home()
project_path = home_path / "Downloads" / "index-tts-main"
_MODEL_PATHS = {
    "2": home_path / "Downloads" / "IndexTTS-2",
    "2.5": home_path / "Downloads" / "IndexTTS-2.5",
}
_TEXT_TOKENIZER_FILES = {
    "2": "bpe.model",
    "2.5": "multilingual_zh_ja_yue_char_del.tiktoken",
}
_OUTPUT_FOLDER_NAMES = {
    "2": "IndexTTS2_ONNX",
    "2.5": "IndexTTS2_5_ONNX",
}
if MODEL_VERSION not in _MODEL_PATHS:
    raise ValueError(f"Unsupported MODEL_VERSION: {MODEL_VERSION!r}")
models_path = _MODEL_PATHS[MODEL_VERSION]
TEXT_TOKENIZER_FILE = _TEXT_TOKENIZER_FILES[MODEL_VERSION]

MAX_SIGNAL_LENGTH = 2048
USE_F16_KV = True
COMPUTE_IN_F32 = False
OPSET = 20
CFM_STEPS = 25
IN_SAMPLE_RATE = 22050
OUT_SAMPLE_RATE = 22050
IN_AUDIO_DTYPE = "F32"
OUT_AUDIO_DTYPE = "F32"
EMOTION_TEXT_MAX_SEQ_LENGTH = 1024
EMOTION_TEXT_REORDER_DOWNPROJ = True
EMOTION_TEXT_REORDER_KEY = "absmean"
EMOTION_TEXT_KV_DTYPE = "F16"


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
for import_path in (project_path, REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from Index_TTS import _indextts2_export_common as shared  # noqa: E402
from Index_TTS.v2.STFT_Process import STFT_Process  # noqa: E402


class V2Reference(nn.Module):
    """Build v2's reference condition through the residual codec quantizer."""

    def __init__(
        self,
        semantic_codec: nn.Module,
        length_regulator: nn.Module,
        campplus: nn.Module,
        cfm_projection: nn.Module,
    ) -> None:
        super().__init__()
        self.semantic_codec = semantic_codec.eval()
        self.length_regulator = length_regulator.eval()
        self.campplus = campplus.eval()
        self.cfm_projection = cfm_projection.eval()

    def forward(
        self,
        semantic_features: torch.Tensor,
        reference_mel: torch.Tensor,
        style_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, continuous_reference = self.semantic_codec.quantize(semantic_features)
        target_lengths = torch._shape_as_tensor(reference_mel)[2:3]
        prompt_condition = self.length_regulator(
            continuous_reference,
            ylens=target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]
        style = self.campplus(style_features)
        reference_hidden, null_hidden = self.cfm_projection(
            reference_mel,
            style,
            prompt_condition,
        )
        return style, reference_hidden, null_hidden


class V2Conditioning(nn.Module):
    """Run v2's speaker-conditioning and shared emotion matrix path."""

    def __init__(
        self,
        gpt: nn.Module,
        speaker_matrix: torch.Tensor,
        emotion_matrix: torch.Tensor,
    ) -> None:
        super().__init__()
        self.gpt = gpt.eval()
        self.emotion_matrix = shared.IndexTTS2EmotionMatrix(
            speaker_matrix,
            emotion_matrix,
        )

    def forward(
        self,
        speaker_features: torch.Tensor,
        speaker_lengths: torch.Tensor,
        emotion_features: torch.Tensor,
        emotion_lengths: torch.Tensor,
        emotion_alpha: torch.Tensor,
        style: torch.Tensor,
        emotion_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        speaker_latent = self.gpt.get_conditioning(
            speaker_features.transpose(1, 2),
            speaker_lengths,
        )
        base_emotion = self.gpt.merge_emovec(
            speaker_features,
            emotion_features,
            speaker_lengths,
            emotion_lengths,
            alpha=emotion_alpha,
        )
        emotion_vector = self.emotion_matrix(
            style,
            base_emotion,
            emotion_weights,
        )
        return speaker_latent, emotion_vector


class V2TargetPreprocess(nn.Module):
    """Construct the v2 prompt with the two learned speed embeddings."""

    def __init__(self, gpt: nn.Module) -> None:
        super().__init__()
        self.gpt = gpt.eval()
        self.register_buffer(
            "text_position_table",
            gpt.text_pos_embedding.emb.weight.detach().half(),
        )
        self.register_buffer(
            "mel_position_table",
            gpt.mel_pos_embedding.emb.weight.detach().half(),
        )
        self.register_buffer(
            "start_text_id",
            torch.tensor([[gpt.start_text_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "stop_text_id",
            torch.tensor([[gpt.stop_text_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "start_mel_id",
            torch.tensor([[gpt.start_mel_token]], dtype=torch.int32),
            persistent=False,
        )
        self.register_buffer(
            "normal_speed_id",
            torch.zeros(1, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "half_speed_id",
            torch.ones(1, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        speaker_latent: torch.Tensor,
        emotion_vector: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_with_bounds = torch.cat(
            (self.start_text_id, text_ids, self.stop_text_id),
            dim=1,
        )
        text_length = torch._shape_as_tensor(text_with_bounds)[1:2]
        text_hidden = self.gpt.text_embedding(text_with_bounds)
        text_hidden = text_hidden + self.text_position_table[:text_length].float()
        conditioned_speaker = speaker_latent + emotion_vector.unsqueeze(1)
        half_speed = self.gpt.speed_emb(self.half_speed_id).unsqueeze(1)
        normal_speed = self.gpt.speed_emb(self.normal_speed_id).unsqueeze(1)
        prompt_hidden = torch.cat(
            (conditioned_speaker, half_speed, normal_speed, text_hidden),
            dim=1,
        )
        start_mel_hidden = self.gpt.mel_embedding(self.start_mel_id)
        batch_size = torch._shape_as_tensor(text_ids)[0]
        start_mel_hidden = start_mel_hidden + self.mel_position_table[:batch_size].float()
        return torch.cat((prompt_hidden, start_mel_hidden), dim=1)


class V2AcousticConditioning(nn.Module):
    """Match v2's quantizer-plus-GPT acoustic conditioning path."""

    def __init__(
        self,
        semantic_quantizer: nn.Module,
        gpt_projection: nn.Module,
        length_regulator: nn.Module,
        cfm_projection: nn.Module,
    ) -> None:
        super().__init__()
        self.semantic_quantizer = semantic_quantizer.eval()
        self.gpt_projection = gpt_projection.eval()
        self.length_regulator = length_regulator.eval()
        self.cfm_projection = cfm_projection.eval()

    def forward(
        self,
        mel_codes: torch.Tensor,
        gpt_latent: torch.Tensor,
        cfg_rate: torch.Tensor,
        style: torch.Tensor,
        reference_hidden: torch.Tensor,
        null_hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        semantic_hidden = self.semantic_quantizer.vq2emb(
            mel_codes.unsqueeze(1)
        ).transpose(1, 2)
        semantic_hidden = semantic_hidden + self.gpt_projection(gpt_latent)
        code_length = torch._shape_as_tensor(mel_codes)[1:2]
        target_length = (code_length.float() * 1.72).int()
        condition = self.length_regulator(
            semantic_hidden,
            ylens=target_length,
            n_quantizers=3,
            f0=None,
        )[0]
        target_hidden = self.cfm_projection.project_without_prompt(style, condition)
        conditional_hidden = torch.cat((reference_hidden, target_hidden), dim=1)
        static_hidden = torch.cat(
            (conditional_hidden, null_hidden.expand_as(conditional_hidden)),
            dim=0,
        )
        target_mask = torch.cat(
            (
                torch.zeros_like(reference_hidden[:, :, :1]),
                torch.ones_like(target_hidden[:, :, :1]),
            ),
            dim=1,
        )
        cfg_scales = torch.cat((1.0 + cfg_rate, -cfg_rate)).view(2, 1, 1)
        cfg_scale_sum = cfg_scales.sum().reshape(1)
        return static_hidden, target_length, cfg_scales, cfg_scale_sum, target_mask


class V2ExportAdapter(shared.ExportAdapter):
    def resolve_auxiliary_paths(self, config: Any) -> dict[str, str]:
        from indextts.utils.model_download import ensure_models_available

        return ensure_models_available(
            str(models_path),
            bigvgan_repo=str(config.vocoder.name),
        )

    def load_gpt(self, config: Any) -> nn.Module:
        shared._install_transformers_compatibility_modules()
        from indextts.gpt.model_v2 import UnifiedVoice

        gpt = UnifiedVoice(
            **OmegaConf.to_container(config.gpt, resolve=True),
            use_accel=False,
        )
        checkpoint = torch.load(
            models_path / str(config.gpt_checkpoint),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        gpt.load_state_dict(state, strict=True)
        del checkpoint, state
        gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=True, half=False)
        return shared.freeze(gpt)

    def load_acoustic_modules(
        self,
        config: Any,
        auxiliary_paths: dict[str, str],
    ) -> tuple[nn.Module, nn.Module, nn.Module]:
        import safetensors.torch

        from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
        from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
        from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec

        semantic_codec = RepCodec(cfg=config.semantic_codec)
        safetensors.torch.load_model(semantic_codec, auxiliary_paths["semantic_codec"])
        semantic_codec = shared.freeze(semantic_codec)
        shared.remove_all_weight_norm(semantic_codec)
        s2mel = MyModel(config.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            str(models_path / str(config.s2mel_checkpoint)),
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        s2mel = shared.freeze(s2mel)
        shared.remove_all_weight_norm(s2mel)
        s2mel.models["cfm"].estimator.setup_caches(
            max_batch_size=1,
            max_seq_length=8192,
        )
        campplus = CAMPPlus(
            feat_dim=80,
            embedding_size=int(config.s2mel.style_encoder.dim),
        )
        campplus_state = torch.load(
            auxiliary_paths["campplus"],
            map_location="cpu",
            weights_only=True,
        )
        campplus.load_state_dict(campplus_state, strict=True)
        return semantic_codec, s2mel, shared.freeze(campplus)

    def make_reference(
        self,
        semantic_codec: nn.Module,
        s2mel: nn.Module,
        campplus: nn.Module,
        cfm_projection: nn.Module,
    ) -> nn.Module:
        return V2Reference(
            semantic_codec,
            s2mel.models["length_regulator"],
            campplus,
            cfm_projection,
        )

    def make_conditioning(
        self,
        gpt: nn.Module,
        speaker_matrix: torch.Tensor,
        emotion_matrix: torch.Tensor,
    ) -> nn.Module:
        return V2Conditioning(gpt, speaker_matrix, emotion_matrix)

    def speaker_latent_example(self, hidden_size: int) -> torch.Tensor:
        return torch.zeros(1, 32, hidden_size)

    def target_export(
        self,
        gpt: nn.Module,
        speaker_latent: torch.Tensor,
        emotion_vector: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
        return (
            V2TargetPreprocess(gpt),
            (speaker_latent, emotion_vector, text_ids),
            ["speaker_latent", "emotion_vector", "text_ids"],
        )

    def make_latent(
        self,
        gpt: nn.Module,
        main_core: nn.Module,
        config: Any,
        semantic_hidden_size: int,
    ) -> nn.Module:
        del config, semantic_hidden_size
        return shared.IndexTTS2Latent(gpt, main_core)

    def acoustic_export(
        self,
        semantic_codec: nn.Module,
        s2mel: nn.Module,
        cfm_projection: nn.Module,
        config: Any,
        style_embed_size: int,
        semantic_hidden_size: int,
    ) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
        del semantic_hidden_size
        acoustic = V2AcousticConditioning(
            semantic_codec.quantizer,
            s2mel.models["gpt_layer"],
            s2mel.models["length_regulator"],
            cfm_projection,
        )
        return (
            acoustic,
            (
                torch.zeros(1, 20, dtype=torch.int32),
                torch.zeros(1, 20, int(config.gpt.model_dim)),
                torch.tensor([0.7], dtype=torch.float32),
                torch.zeros(1, style_embed_size),
                torch.zeros(1, 100, int(config.s2mel.DiT.hidden_dim)),
                cfm_projection.null_hidden,
            ),
            [
                "mel_codes",
                "gpt_latent",
                "cfg_rate",
                "style",
                "reference_hidden",
                "null_hidden",
            ],
        )

    def cfm_time_schedule(
        self,
        steps: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_times = torch.arange(steps, dtype=torch.float32, device=device) / steps
        time_span = torch.arange(steps + 1, dtype=torch.float32, device=device) / steps
        return step_times, time_span[1:] - time_span[:-1]

    def metadata_fields(self, config: Any) -> dict[str, Any]:
        del config
        return {
            "model_version": "2",
            "speaker_conditioning_mode": "speed",
            "target_language_embedding": False,
            "use_gpt_latent": True,
        }


V2_PROFILE = shared.ExportProfile(
    model_version="2",
    script_dir=SCRIPT_DIR,
    project_path=project_path,
    models_path=_MODEL_PATHS["2"],
    output_folder_name=_OUTPUT_FOLDER_NAMES["2"],
    text_tokenizer_file=_TEXT_TOKENIZER_FILES["2"],
    stft_process=STFT_Process,
    max_signal_length=MAX_SIGNAL_LENGTH,
    use_f16_kv=USE_F16_KV,
    compute_in_f32=COMPUTE_IN_F32,
    opset=OPSET,
    cfm_steps=CFM_STEPS,
    in_sample_rate=IN_SAMPLE_RATE,
    out_sample_rate=OUT_SAMPLE_RATE,
    in_audio_dtype=IN_AUDIO_DTYPE,
    out_audio_dtype=OUT_AUDIO_DTYPE,
    emotion_text_max_seq_length=EMOTION_TEXT_MAX_SEQ_LENGTH,
    emotion_text_reorder_downproj=EMOTION_TEXT_REORDER_DOWNPROJ,
    emotion_text_reorder_key=EMOTION_TEXT_REORDER_KEY,
    emotion_text_kv_dtype=EMOTION_TEXT_KV_DTYPE,
)
V25_PROFILE = shared.ExportProfile(
    model_version="2.5",
    script_dir=SCRIPT_DIR,
    project_path=project_path,
    models_path=_MODEL_PATHS["2.5"],
    output_folder_name=_OUTPUT_FOLDER_NAMES["2.5"],
    text_tokenizer_file=_TEXT_TOKENIZER_FILES["2.5"],
    stft_process=STFT_Process,
    max_signal_length=MAX_SIGNAL_LENGTH,
    use_f16_kv=USE_F16_KV,
    compute_in_f32=COMPUTE_IN_F32,
    opset=OPSET,
    cfm_steps=CFM_STEPS,
    in_sample_rate=IN_SAMPLE_RATE,
    out_sample_rate=OUT_SAMPLE_RATE,
    in_audio_dtype=IN_AUDIO_DTYPE,
    out_audio_dtype=OUT_AUDIO_DTYPE,
    emotion_text_max_seq_length=EMOTION_TEXT_MAX_SEQ_LENGTH,
    emotion_text_reorder_downproj=EMOTION_TEXT_REORDER_DOWNPROJ,
    emotion_text_reorder_key=EMOTION_TEXT_REORDER_KEY,
    emotion_text_kv_dtype=EMOTION_TEXT_KV_DTYPE,
)
PROFILE = V2_PROFILE if MODEL_VERSION == "2" else V25_PROFILE


def main() -> None:
    adapter = V2ExportAdapter() if MODEL_VERSION == "2" else None
    shared.run_export(PROFILE, adapter)


if __name__ == "__main__":
    main()
    print("\nStart running the IndexTTS2 demo via Inference_IndexTTS2_ONNX.py ...")
    raise SystemExit(subprocess.call([
        sys.executable,
        str(SCRIPT_DIR / "Inference_IndexTTS2_ONNX.py"),
        "--onnx-folder",
        str(PROFILE.onnx_folder),
    ]))