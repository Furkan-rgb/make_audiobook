"""Resolve narrator reference voices from voice folders or plain audio files.

A reference voice is whatever the clone model conditions on.  It can come from
two places, and both resolve to the same :class:`ReferenceVoice`:

* a designed voice folder written by ``design_voice.py`` (``voices/<name>/``),
  which carries a transcript and the persona it was rendered from;
* any audio file on disk, e.g. a recording of your own voice.

Decoding goes through ffmpeg rather than ``soundfile`` because recorders often
emit stream-style files whose headers declare no total sample count, which
libsndfile refuses to seek.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import (
    REFERENCE_MAX_INTERNAL_SILENCE_MS,
    REFERENCE_PEAK_DBFS,
    REFERENCE_SAMPLE_RATE,
    REFERENCE_TRANSCRIBE,
    REFERENCE_TRIM_PAD_MS,
    REFERENCE_TRIM_TOP_DB,
    VOICE_REFERENCE_AUDIO_FILENAME,
    VOICE_REFERENCE_METADATA_FILENAME,
    VOICES_DIR,
)

AUDIO_SUFFIXES = frozenset(
    {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif"}
)


@dataclass(frozen=True)
class ReferenceVoice:
    """A decoded reference clip plus whatever is known about how it was made."""

    slug: str
    audio: np.ndarray
    sample_rate: int
    ref_text: str | None
    instruct: str | None
    source: Path

    @property
    def x_vector_only(self) -> bool:
        """True when no transcript is available, so only timbre can be cloned."""

        return not self.ref_text


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    *,
    top_db: float = REFERENCE_TRIM_TOP_DB,
    pad_ms: int = REFERENCE_TRIM_PAD_MS,
    max_internal_silence_ms: int = REFERENCE_MAX_INTERNAL_SILENCE_MS,
) -> np.ndarray:
    """Drop silence at the ends of *audio* and cap the gaps inside it.

    The transcript accounts for words, never for pauses, so silence the clone
    model sees is silence it may reproduce in every chunk.  Detection is
    relative to the clip's own level: a recording's floor is its noise, not
    zero.  A short pad is kept around each segment so breaths and plosive
    onsets survive.
    """

    import librosa

    intervals = librosa.effects.split(audio, top_db=top_db)
    if not len(intervals):
        return audio

    pad = int(sample_rate * pad_ms / 1000)
    max_gap = int(sample_rate * max_internal_silence_ms / 1000)
    pieces: list[np.ndarray] = []
    previous_end: int | None = None

    for raw_start, raw_end in intervals:
        start = max(0, int(raw_start) - pad)
        end = min(len(audio), int(raw_end) + pad)
        if previous_end is None:
            pieces.append(audio[start:end])
        elif start <= previous_end:
            # Pads overlap, so the segments are contiguous already.
            pieces.append(audio[previous_end:end])
        else:
            pieces.append(audio[previous_end : previous_end + min(start - previous_end, max_gap)])
            pieces.append(audio[start:end])
        previous_end = end

    return np.concatenate(pieces)


def load_reference_audio(
    path: Path,
    *,
    sample_rate: int = REFERENCE_SAMPLE_RATE,
    peak_dbfs: float = REFERENCE_PEAK_DBFS,
) -> tuple[np.ndarray, int]:
    """Decode *path* to mono float32 at *sample_rate*, trimmed and levelled.

    Beyond channel mixing, resampling and silence trimming, only a single gain
    factor is applied.  No EQ or dynamics, so a clip that is already clean
    passes through unchanged in everything but length and level.  Trimming runs
    before the gain so measured level reflects speech alone.
    """

    import soundfile as sf

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to decode reference audio.")

    with tempfile.TemporaryDirectory() as temp_dir:
        decoded = Path(temp_dir) / "reference.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(path),
                "-ac", "1",
                "-ar", str(sample_rate),
                "-c:a", "pcm_s16le",
                str(decoded),
            ],
            check=True,
        )
        audio, decoded_rate = sf.read(decoded, dtype="float32")

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = trim_silence(audio, int(decoded_rate))
    peak = float(np.abs(audio).max())
    if peak > 0.0:
        audio = audio * (10.0 ** (peak_dbfs / 20.0) / peak)
    return audio, int(decoded_rate)


def _sidecar_ref_text(path: Path) -> str | None:
    """Read a transcript sitting beside *path* as ``<stem>.txt`` or ``<stem>.json``."""

    text_path = path.with_suffix(".txt")
    if text_path.exists():
        return text_path.read_text(encoding="utf-8").strip() or None

    json_path = path.with_suffix(".json")
    if json_path.exists():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        return metadata.get("ref_text") or None

    return None


def reference_audio_path(voice_dir: Path) -> Path | None:
    """The reference clip inside a voice folder, whatever it was encoded as.

    Designed voices are always written as ``reference.wav``, but an imported
    recording keeps the encoding it arrived in — re-encoding a FLAC to WAV to
    satisfy a filename would be a lossy step taken for no reason.
    """

    canonical = voice_dir / VOICE_REFERENCE_AUDIO_FILENAME
    if canonical.exists():
        return canonical
    stem = canonical.stem
    for child in sorted(voice_dir.glob(f"{stem}.*")):
        if child.suffix.lower() in AUDIO_SUFFIXES:
            return child
    return None


def _load_voice_folder(voice_dir: Path, *, sample_rate: int) -> ReferenceVoice:
    """Load a voice folder: a designed one, or an imported recording."""

    metadata = json.loads(
        (voice_dir / VOICE_REFERENCE_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    audio_path = reference_audio_path(voice_dir)
    if audio_path is None:
        raise FileNotFoundError(
            f"{voice_dir} has no reference clip beside its {VOICE_REFERENCE_METADATA_FILENAME}."
        )
    audio, rate = load_reference_audio(audio_path, sample_rate=sample_rate)
    return ReferenceVoice(
        slug=metadata.get("slug", voice_dir.name),
        audio=audio,
        sample_rate=rate,
        ref_text=metadata.get("ref_text") or None,
        instruct=metadata.get("instruct"),
        source=audio_path,
    )


def _load_audio_file(
    path: Path, *, sample_rate: int, transcribe_missing: bool
) -> ReferenceVoice:
    """Load a bare recording and recover its transcript if one is not on disk.

    A recovered transcript is cached beside the audio so it is reviewed once,
    corrected if wrong, and reused by every later run.
    """

    audio, rate = load_reference_audio(path, sample_rate=sample_rate)
    ref_text = _sidecar_ref_text(path)

    if ref_text is None and transcribe_missing:
        from .transcribe import transcribe

        ref_text = transcribe(audio, rate)
        if ref_text:
            sidecar = path.with_suffix(".txt")
            sidecar.write_text(ref_text + "\n", encoding="utf-8")
            print(f"  wrote {sidecar} — check it word for word and fix any errors.")

    return ReferenceVoice(
        slug=path.stem,
        audio=audio,
        sample_rate=rate,
        ref_text=ref_text,
        instruct=None,
        source=path,
    )


def resolve_voice(
    spec: str | Path,
    *,
    voices_dir: Path = VOICES_DIR,
    ref_text: str | None = None,
    sample_rate: int = REFERENCE_SAMPLE_RATE,
    transcribe_missing: bool = REFERENCE_TRANSCRIBE,
) -> ReferenceVoice:
    """Resolve *spec* to a reference voice.

    *spec* may be a designed voice name (``warm_male``), a path to an audio file
    (``voices/Self.flac``), or a bare filename inside *voices_dir*.  The
    transcript is taken from the first source that has one: an explicit
    *ref_text*, a sidecar file, then speech recognition.  If none produces one
    the voice falls back to timbre-only cloning.
    """

    candidate = Path(spec)
    voice_dir = voices_dir / str(spec)
    # An explicit transcript makes recognition pointless work.
    transcribe_missing = transcribe_missing and not ref_text

    if (voice_dir / VOICE_REFERENCE_METADATA_FILENAME).exists():
        voice = _load_voice_folder(voice_dir, sample_rate=sample_rate)
    elif candidate.is_file():
        voice = _load_audio_file(
            candidate, sample_rate=sample_rate, transcribe_missing=transcribe_missing
        )
    elif (voices_dir / candidate.name).is_file():
        voice = _load_audio_file(
            voices_dir / candidate.name,
            sample_rate=sample_rate,
            transcribe_missing=transcribe_missing,
        )
    else:
        raise FileNotFoundError(
            f"No voice named {str(spec)!r}: it is neither a designed voice in "
            f"{voices_dir} nor a readable audio file. Design one with "
            f"'python design_voice.py {spec}', or point at a recording."
        )

    if ref_text:
        return ReferenceVoice(
            slug=voice.slug,
            audio=voice.audio,
            sample_rate=voice.sample_rate,
            ref_text=ref_text,
            instruct=voice.instruct,
            source=voice.source,
        )
    return voice


def describe(voice: ReferenceVoice) -> str:
    """One-line summary of what the clone model will actually condition on."""

    seconds = len(voice.audio) / voice.sample_rate
    mode = (
        "timbre only (no transcript)"
        if voice.x_vector_only
        else "timbre + prosody (transcript available)"
    )
    return f"{voice.slug}: {seconds:.1f}s from {voice.source} — {mode}"


__all__ = [
    "AUDIO_SUFFIXES",
    "ReferenceVoice",
    "describe",
    "load_reference_audio",
    "reference_audio_path",
    "resolve_voice",
    "trim_silence",
]
