"""Audio joining and chaptered M4B assembly for the audiobook workflow."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import soundfile as sf

from audiobook_config import (
    CHAPTER_SILENCE_MS,
    CHUNK_CROSSFADE_MS,
    PARAGRAPH_SILENCE_MS,
    SECTION_SILENCE_MS,
    VOICE_NAME,
)


class BoundaryChunkLike(Protocol):
    """Structural type needed when joining independently generated chunks."""

    boundary_after: str


def verify_audio_dependencies() -> None:
    """Verify that the FFmpeg executable needed for M4B output is available."""

    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required to create the chaptered M4B file.")


def crossfade(left: np.ndarray, right: np.ndarray, samples: int) -> np.ndarray:
    """Overlap two one-dimensional signals with a linear crossfade."""

    samples = min(samples, len(left), len(right))
    if samples <= 0:
        return np.concatenate((left, right))
    fade_in = np.linspace(0.0, 1.0, samples, endpoint=True, dtype=np.float32)
    overlap = left[-samples:] * (1.0 - fade_in) + right[:samples] * fade_in
    return np.concatenate((left[:-samples], overlap, right[samples:]))


def fade_in(audio: np.ndarray, samples: int) -> np.ndarray:
    """Return a copy with a linear fade applied to its beginning."""

    samples = min(samples, len(audio))
    if samples <= 0:
        return audio
    faded = audio.copy()
    ramp = np.linspace(0.0, 1.0, samples, endpoint=True, dtype=np.float32)
    faded[:samples] *= ramp
    return faded


def fade_out(audio: np.ndarray, samples: int) -> np.ndarray:
    """Return a copy with a linear fade applied to its end."""

    samples = min(samples, len(audio))
    if samples <= 0:
        return audio
    faded = audio.copy()
    ramp = np.linspace(1.0, 0.0, samples, endpoint=True, dtype=np.float32)
    faded[-samples:] *= ramp
    return faded


# Compatibility aliases keep the established helper names available while the
# public spellings make the pure operations easier to discover and unit-test.
_crossfade = crossfade
_fade_in = fade_in
_fade_out = fade_out


def assemble_chunk_audio(
    chunks: Sequence[BoundaryChunkLike],
    audio_segments: Sequence[np.ndarray],
    sample_rate: int,
) -> np.ndarray:
    """Join generated requests without inserting sentence-level pauses.

    A continuation within a split paragraph is directly crossfaded.  Paragraph
    and section/scene boundaries instead receive their configured short gaps.
    """

    if len(chunks) != len(audio_segments):
        raise ValueError("Each narration chunk must have one audio segment")
    if not audio_segments:
        return np.array([], dtype=np.float32)

    crossfade_samples = round(sample_rate * CHUNK_CROSSFADE_MS / 1000)
    result = np.asarray(audio_segments[0], dtype=np.float32).reshape(-1)
    for previous, segment in zip(chunks, audio_segments[1:]):
        following = np.asarray(segment, dtype=np.float32).reshape(-1)
        if previous.boundary_after == "continuation":
            result = crossfade(result, following, crossfade_samples)
            continue

        silence_ms = (
            SECTION_SILENCE_MS
            if previous.boundary_after in {"section", "scene"}
            else PARAGRAPH_SILENCE_MS
        )
        result = fade_out(result, crossfade_samples)
        following = fade_in(following, crossfade_samples)
        silence = np.zeros(round(sample_rate * silence_ms / 1000), dtype=np.float32)
        result = np.concatenate((result, silence, following))
    return result


def add_chapter_silence(
    audio: np.ndarray,
    sample_rate: int,
    silence_ms: int = CHAPTER_SILENCE_MS,
) -> np.ndarray:
    """Append the configured inter-chapter silence to a mono signal."""

    flattened = np.asarray(audio, dtype=np.float32).reshape(-1)
    silence = np.zeros(round(sample_rate * silence_ms / 1000), dtype=np.float32)
    return np.concatenate((flattened, silence))


def write_chapter_wav(
    temp_dir: Path,
    chapter_index: int,
    chapter_audio: np.ndarray,
    sample_rate: int,
    *,
    add_trailing_silence: bool = True,
) -> tuple[str, int]:
    """Write a numbered chapter WAV and return its name and duration in ms."""

    temp_dir.mkdir(parents=True, exist_ok=True)
    audio = (
        add_chapter_silence(chapter_audio, sample_rate)
        if add_trailing_silence
        else np.asarray(chapter_audio, dtype=np.float32).reshape(-1)
    )
    wav_name = f"part_{chapter_index:03d}.wav"
    sf.write(temp_dir / wav_name, audio, sample_rate)
    duration_ms = round(len(audio) / sample_rate * 1000)
    return wav_name, duration_ms


def _escape_ffmetadata(value: str) -> str:
    """Escape a value for FFmetadata key/value syntax."""

    return re.sub(r"([\\=;#])", r"\\\1", value).replace("\n", " ")


def create_ffmpeg_metadata(
    chapters_metadata: Sequence[tuple[str, int, int]],
    metadata_file: Path,
    *,
    title: str = "Audiobook Generated with Qwen3-TTS",
    artist: str = f"Qwen3-TTS / {VOICE_NAME}",
) -> None:
    """Write global tags and millisecond chapter markers as FFmetadata."""

    with metadata_file.open("w", encoding="utf-8") as handle:
        handle.write(";FFMETADATA1\n")
        handle.write(f"title={_escape_ffmetadata(title)}\n")
        handle.write(f"artist={_escape_ffmetadata(artist)}\n\n")
        for chapter_title, start_ms, end_ms in chapters_metadata:
            handle.write("[CHAPTER]\n")
            handle.write("TIMEBASE=1/1000\n")
            handle.write(f"START={start_ms}\n")
            handle.write(f"END={end_ms}\n")
            handle.write(f"title={_escape_ffmetadata(chapter_title)}\n\n")


def merge_chapters(
    temp_dir: Path,
    wav_files: Sequence[str],
    chapters_metadata: Sequence[tuple[str, int, int]],
    output_path: Path,
) -> None:
    """Concatenate chapter WAVs and encode a chaptered 64-kbps AAC M4B."""

    file_list = temp_dir / "files.txt"
    metadata_path = temp_dir / "metadata.txt"
    with file_list.open("w", encoding="utf-8") as handle:
        for wav_file in wav_files:
            handle.write(f"file '{wav_file}'\n")
    create_ffmpeg_metadata(chapters_metadata, metadata_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        file_list.name,
        "-i",
        metadata_path.name,
        "-map_metadata",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(output_path.resolve()),
    ]
    subprocess.run(command, check=True, cwd=temp_dir)


__all__ = [
    "BoundaryChunkLike",
    "add_chapter_silence",
    "assemble_chunk_audio",
    "create_ffmpeg_metadata",
    "crossfade",
    "fade_in",
    "fade_out",
    "merge_chapters",
    "verify_audio_dependencies",
    "write_chapter_wav",
]
