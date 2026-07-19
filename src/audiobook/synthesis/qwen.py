"""Qwen3-TTS inference backend for the audiobook workflow.

The module deliberately imports the heavyweight Qwen and PyTorch packages only
inside the functions that need them.  Text preparation and chunk planning can
therefore be used on machines where the TTS runtime is not installed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..config import LANGUAGE, NARRATION_INSTRUCTION, VOICE_NAME


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


def _as_mono_float32(wav: Any) -> np.ndarray:
    """Coerce a decoded Qwen waveform into a flat float32 mono array."""

    return np.asarray(wav, dtype=np.float32).reshape(-1)


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
    return _as_mono_float32(wavs[0]), int(sample_rate)


def design_reference_clip(
    design_model: Any,
    *,
    ref_text: str,
    instruct: str,
    language: str = LANGUAGE,
) -> tuple[np.ndarray, int]:
    """Render a reference clip with the VoiceDesign model from a persona string.

    The returned audio is meant to be handed to :func:`build_voice_clone_prompt`
    so the designed persona can be reused as a stable cloned narrator.
    """

    wavs, sample_rate = design_model.generate_voice_design(
        text=ref_text,
        language=language,
        instruct=instruct,
    )
    return _as_mono_float32(wavs[0]), int(sample_rate)


def build_voice_clone_prompt(
    clone_model: Any,
    *,
    ref_audio: np.ndarray,
    sample_rate: int,
    ref_text: str | None = None,
) -> Any:
    """Precompute a reusable clone prompt from a reference clip.

    Building the prompt once and passing it to every :func:`generate_clone_chunk`
    call keeps the narrator identical across the whole book and avoids
    re-extracting reference features for each chunk.

    With *ref_text* the model runs in-context and carries both timbre and the
    reference's prosody.  Without it only the speaker embedding is available, so
    the clone keeps the voice's identity but reads with its own delivery.
    """

    return clone_model.create_voice_clone_prompt(
        ref_audio=(ref_audio, int(sample_rate)),
        ref_text=ref_text,
        x_vector_only_mode=not ref_text,
    )


def generate_clone_chunk(
    clone_model: Any,
    chunk: NarrationChunkLike,
    *,
    voice_clone_prompt: Any,
    language: str = LANGUAGE,
) -> tuple[np.ndarray, int]:
    """Generate one narration chunk by cloning a prepared reference voice.

    Delivery and prosody are carried by ``voice_clone_prompt`` (built from the
    reference clip), so no per-chunk style instruction is supplied.
    """

    wavs, sample_rate = clone_model.generate_voice_clone(
        text=chunk.text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
    )
    return _as_mono_float32(wavs[0]), int(sample_rate)


__all__ = [
    "NarrationChunkLike",
    "build_voice_clone_prompt",
    "design_reference_clip",
    "generate_chunk",
    "generate_clone_chunk",
    "load_qwen_model",
    "verify_supported_voice",
    "verify_tts_dependencies",
]
