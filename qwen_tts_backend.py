"""Qwen3-TTS inference backend for the audiobook workflow.

The module deliberately imports the heavyweight Qwen and PyTorch packages only
inside the functions that need them.  Text preparation and chunk planning can
therefore be used on machines where the TTS runtime is not installed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from audiobook_config import LANGUAGE, NARRATION_INSTRUCTION, VOICE_NAME


@runtime_checkable
class NarrationChunkLike(Protocol):
    """Structural type accepted by :func:`generate_chunk`."""

    text: str


def verify_tts_dependencies() -> None:
    """Verify that Qwen3-TTS is importable and a CUDA device is available.

    Raises:
        RuntimeError: If the TTS packages are missing or CUDA is unavailable.
    """

    try:
        import torch
        from qwen_tts import Qwen3TTSModel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-TTS is not installed. Run: .venv/bin/python -m pip "
            "install -r requirements.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS generation requires a CUDA GPU.")


def load_qwen_model(model_name_or_path: str) -> Any:
    """Load a Qwen3-TTS CustomVoice model on the first CUDA device."""

    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"Loading {model_name_or_path} on {torch.cuda.get_device_name(0)}...")
    return Qwen3TTSModel.from_pretrained(
        model_name_or_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )


def verify_supported_voice(model: Any, voice_name: str = VOICE_NAME) -> None:
    """Raise when *model* does not expose the configured narrator voice."""

    supported_speakers = {
        str(speaker).casefold() for speaker in model.get_supported_speakers()
    }
    if voice_name.casefold() not in supported_speakers:
        raise RuntimeError(f"{voice_name} is not supported by the selected model.")


def generate_chunk(
    model: Any,
    chunk: NarrationChunkLike,
    *,
    voice_name: str = VOICE_NAME,
    language: str = LANGUAGE,
    instruction: str = NARRATION_INSTRUCTION,
) -> tuple[np.ndarray, int]:
    """Generate one narration chunk with the configured CustomVoice settings.

    Only ``chunk.text`` is spoken.  Any neighboring context held by the chunk
    remains metadata, matching the workflow's non-spoken-context policy.
    """

    wavs, sample_rate = model.generate_custom_voice(
        text=chunk.text,
        language=language,
        speaker=voice_name,
        instruct=instruction,
    )
    audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    return audio, int(sample_rate)


__all__ = [
    "NarrationChunkLike",
    "generate_chunk",
    "load_qwen_model",
    "verify_supported_voice",
    "verify_tts_dependencies",
]
