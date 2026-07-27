"""Shared voice-clone reference audio catalog for the TTS inference scripts.

Every voice-cloning demo in this repo points at the same Chinese reference clip stored under
``example/`` together with its transcript, so the per-model ``example/`` folders were consolidated
here. Inference scripts add the repo root to ``sys.path`` and import the helper they need::

    from Example_Audio import model_reference
    prompt_audio_path, prompt_text = model_reference("qwen_tts")

or, when only the audio path is needed::

    from Example_Audio import reference_audio_path
    reference_audio = reference_audio_path("indextts")
"""

from pathlib import Path


EXAMPLE_AUDIO_ROOT = Path(__file__).resolve().parent / "Example_Audio"

# Transcript that matches basic_ref_zh.wav.
_BASIC_REF_ZH_TEXT = "对，这就是我，万人敬仰的太乙真人。"

# model name -> (reference audio filename under EXAMPLE_AUDIO_ROOT, transcript of that audio)
_MODEL_REFERENCES = {
    "indextts": ("basic_ref_zh.wav", _BASIC_REF_ZH_TEXT),
    "qwen_tts": ("basic_ref_zh.wav", _BASIC_REF_ZH_TEXT),
    "moss_tts": ("basic_ref_zh.wav", _BASIC_REF_ZH_TEXT),
    "voxcpm":   ("basic_ref_zh.wav", _BASIC_REF_ZH_TEXT),
}


def example_audio_path(filename):
    """Absolute path (str) to a reference clip stored under ``example/``."""
    return str(EXAMPLE_AUDIO_ROOT / filename)


def _reference(model_name):
    try:
        return _MODEL_REFERENCES[model_name]
    except KeyError as exc:
        names = ", ".join(sorted(_MODEL_REFERENCES))
        raise ValueError(f"Unknown demo reference model {model_name!r}. Available models: {names}") from exc


def model_reference(model_name):
    """Return ``(reference_audio_path, prompt_text)`` for a model's voice-clone demo."""
    filename, prompt_text = _reference(model_name)
    return example_audio_path(filename), prompt_text


def reference_audio_path(model_name):
    """Return the absolute path (str) to a model's demo reference audio."""
    return example_audio_path(_reference(model_name)[0])


def reference_prompt_text(model_name):
    """Return the transcript of a model's demo reference audio."""
    return _reference(model_name)[1]
