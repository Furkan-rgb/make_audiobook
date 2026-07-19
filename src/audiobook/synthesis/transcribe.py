"""Transcribe a reference recording so it can clone prosody, not just timbre.

Cloning in ICL mode pairs the reference audio with the words that produced it,
so a recording is only half a reference until its transcript is known.  This
module recovers that half with Whisper when no transcript was supplied.

The transcript is a best effort, not a source of truth: it is written beside the
audio for review because a misheard word teaches the clone a wrong alignment.
"""

from __future__ import annotations

import numpy as np

from ..config import (
    ASR_MIN_WORDS_PER_SECOND,
    ASR_MODEL,
    ASR_SAMPLE_RATE,
)


def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    *,
    model: str = ASR_MODEL,
    min_words_per_second: float = ASR_MIN_WORDS_PER_SECOND,
) -> str | None:
    """Transcribe already-conditioned reference *audio*.

    Takes the same mono array the clone model receives so both hear the same
    thing.  Returns ``None`` when speech recognition is unavailable or the
    result is too thin to be a real transcript, leaving the caller to fall back
    to timbre-only cloning.
    """

    try:
        import librosa
        import torch
        from transformers import pipeline
    except ImportError:
        return None

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    seconds = len(audio) / sample_rate
    if sample_rate != ASR_SAMPLE_RATE:
        audio = librosa.resample(
            y=audio, orig_sr=int(sample_rate), target_sr=ASR_SAMPLE_RATE
        )

    print(f"Transcribing {seconds:.1f}s of reference audio with {model}...")
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model,
        device=0 if torch.cuda.is_available() else -1,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    result = recognizer(
        {"raw": audio, "sampling_rate": ASR_SAMPLE_RATE},
        generate_kwargs={"language": "en", "task": "transcribe"},
    )
    text = str(result.get("text", "")).strip()

    # Whisper answers silence with confident nonsense, so judge the transcript
    # against how much audio it claims to cover.
    if len(text.split()) < seconds * min_words_per_second:
        print("  transcript too thin to trust; falling back to timbre-only.")
        return None
    return text


__all__ = ["transcribe"]
