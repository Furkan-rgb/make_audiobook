"""Audio joining, loudness matching, and chaptered M4B assembly.

Loudness is handled in two deliberately separate stages.  Chunks generated
independently by the TTS model drift a little in level, so within each chapter
they are gently matched against the chapter median using an *active-speech*
measurement (``active_speech_rms``/``match_chunk_loudness``).  The finished
concatenation is then normalized exactly once to a predictable playback level
with FFmpeg's measured two-pass EBU R128 ``loudnorm`` (``merge_chapters``).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol, Sequence

import numpy as np
import soundfile as sf

from ..config import (
    CHAPTER_SILENCE_MS,
    CHUNK_CROSSFADE_MS,
    OUTPUT_TARGET_LRA,
    OUTPUT_TARGET_LUFS,
    OUTPUT_TRUE_PEAK_DBTP,
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


# Frames whose RMS sits below this level (~ -60 dBFS) are treated as silence:
# digital black, breaths, and room tone all fall under it while even quiet
# speech stays well above.
_SILENCE_FLOOR_RMS = 1e-3


def active_speech_rms(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 50.0,
    relative_gate_db: float = -25.0,
) -> float:
    """Measure the loudness of the *spoken* portion of a chunk.

    Whole-waveform RMS is dragged down by pauses, breaths, and trailing
    silence, so chunks with more pauses would be boosted louder than chunks of
    continuous speech.  Instead the signal is framed (~50 ms), near-silent
    frames are discarded, and a relative gate derived from a robust high
    percentile keeps only frames carrying actual speech.  Returns ``0.0`` for
    empty, effectively silent, or non-finite (malformed) audio so callers can
    skip such chunks rather than amplify noise.
    """

    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        return 0.0

    frame_length = max(1, round(sample_rate * frame_ms / 1000.0))
    usable_length = len(flat) - len(flat) % frame_length
    if usable_length == 0:
        frames = flat[np.newaxis, :]  # Shorter than one frame: measure whole.
    else:
        frames = flat[:usable_length].reshape(-1, frame_length)
    frame_rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    usable = frame_rms[frame_rms > _SILENCE_FLOOR_RMS]
    if usable.size == 0:
        return 0.0
    # Gate relative to a robust high percentile rather than the maximum, so a
    # single hot transient cannot push quieter-but-real speech under the gate.
    reference = np.percentile(usable, 90.0)
    speech = usable[usable >= reference * 10.0 ** (relative_gate_db / 20.0)]
    return float(np.sqrt(np.mean(np.square(speech))))


@dataclass(frozen=True)
class ChunkLoudnessDiagnostics:
    """Record of the gain applied to one chunk during loudness matching."""

    active_rms_before: float
    active_rms_after: float
    normalization_gain_db: float

    def to_manifest(self) -> dict[str, float]:
        """JSON-safe values rounded to meaningful precision for the manifest."""

        return {
            "active_rms_before": round(self.active_rms_before, 6),
            "active_rms_after": round(self.active_rms_after, 6),
            "normalization_gain_db": round(self.normalization_gain_db, 2),
        }


class LoudnessMatchResult(NamedTuple):
    """Matched chapter segments plus one diagnostics record per chunk."""

    segments: list[np.ndarray]
    diagnostics: list[ChunkLoudnessDiagnostics]


def match_chunk_loudness(
    audio_segments: Sequence[np.ndarray],
    sample_rate: int,
    *,
    max_adjustment_db: float = 3.0,
    sample_peak_dbfs: float = -1.5,
) -> LoudnessMatchResult:
    """Gently align a chapter's chunks to their median active-speech loudness.

    Each chunk moves at most ``max_adjustment_db`` toward the chapter median,
    so generation drift between chunks is evened out while natural dynamics —
    a hushed passage, an emphatic line — survive.  Empty or effectively silent
    chunks are returned untouched.  Gain is plain scaling with a sample-peak
    cap at ``sample_peak_dbfs``; no compression or limiting is applied.
    """

    segments = [
        np.asarray(segment, dtype=np.float32).reshape(-1)
        for segment in audio_segments
    ]
    measurements = [active_speech_rms(segment, sample_rate) for segment in segments]
    voiced = [measured for measured in measurements if measured > 0.0]
    target = float(np.median(voiced)) if voiced else 0.0
    peak_ceiling = 10.0 ** (sample_peak_dbfs / 20.0)

    matched: list[np.ndarray] = []
    diagnostics: list[ChunkLoudnessDiagnostics] = []
    for segment, measured in zip(segments, measurements):
        if measured <= 0.0 or target <= 0.0:
            matched.append(segment)
            diagnostics.append(ChunkLoudnessDiagnostics(measured, measured, 0.0))
            continue

        gain_db = float(
            np.clip(
                20.0 * np.log10(target / measured),
                -max_adjustment_db,
                max_adjustment_db,
            )
        )
        gain = 10.0 ** (gain_db / 20.0)
        # Peak safety only ever lowers the gain: a boost that would push
        # samples past the ceiling is reduced until the peak fits.
        peak = float(np.max(np.abs(segment)))
        if peak > 0.0 and peak * gain > peak_ceiling:
            gain = peak_ceiling / peak
            gain_db = 20.0 * float(np.log10(gain))

        matched.append(np.multiply(segment, gain, dtype=np.float32))
        # Plain scaling changes RMS by exactly the gain factor.
        diagnostics.append(
            ChunkLoudnessDiagnostics(measured, measured * gain, gain_db)
        )
    return LoudnessMatchResult(matched, diagnostics)


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


# The measurements the loudnorm analysis pass must yield for the second pass.
_LOUDNORM_MEASUREMENT_KEYS = (
    "input_i",
    "input_tp",
    "input_lra",
    "input_thresh",
    "target_offset",
)


def _run_ffmpeg(arguments: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one FFmpeg invocation, raising a clear error on failure."""

    command = ["ffmpeg", "-y", "-hide_banner", "-nostats", *arguments]
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed ({' '.join(command)}):\n{completed.stderr.strip()}"
        )
    return completed


def _concat_source_args(file_list_name: str) -> list[str]:
    """Input arguments reading the concatenated chapter WAV list."""

    return ["-f", "concat", "-safe", "0", "-i", file_list_name]


def _loudnorm_filter(extra_settings: Sequence[str] = ()) -> str:
    """Build a loudnorm filter string for the configured output targets."""

    settings = [
        f"I={OUTPUT_TARGET_LUFS}",
        f"TP={OUTPUT_TRUE_PEAK_DBTP}",
        f"LRA={OUTPUT_TARGET_LRA}",
        *extra_settings,
    ]
    return "loudnorm=" + ":".join(settings)


def _parse_loudnorm_measurements(stderr: str) -> dict[str, float]:
    """Extract the analysis pass's JSON measurement block from FFmpeg stderr.

    loudnorm prints its statistics as a flat JSON object at the end of the
    log, after unrelated demuxer and filter chatter, so the block is located
    from the end of the stream rather than assumed to be the whole output.
    """

    start = stderr.rfind("{")
    end = stderr.find("}", start)
    if start == -1 or end == -1:
        raise RuntimeError(
            "FFmpeg loudnorm analysis produced no measurement JSON."
        )
    try:
        payload = json.loads(stderr[start : end + 1])
        return {key: float(payload[key]) for key in _LOUDNORM_MEASUREMENT_KEYS}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"FFmpeg loudnorm measurements could not be parsed: {exc}"
        ) from exc


def measure_output_loudness(temp_dir: Path, file_list_name: str) -> dict[str, float]:
    """First loudnorm pass: analyze the concatenated book without encoding."""

    completed = _run_ffmpeg(
        [
            *_concat_source_args(file_list_name),
            "-af",
            _loudnorm_filter(("print_format=json",)),
            "-f",
            "null",
            "-",
        ],
        cwd=temp_dir,
    )
    return _parse_loudnorm_measurements(completed.stderr)


def merge_chapters(
    temp_dir: Path,
    wav_files: Sequence[str],
    chapters_metadata: Sequence[tuple[str, int, int]],
    output_path: Path,
) -> None:
    """Concatenate chapter WAVs into a normalized, chaptered 64-kbps AAC M4B.

    The output is normalized with FFmpeg's two-pass EBU R128 ``loudnorm``:
    an analysis pass measures the whole book, then the measured values drive a
    linear second pass so the entire audiobook receives one uniform gain
    instead of the drifting dynamic adjustment single-pass loudnorm applies.
    """

    file_list = temp_dir / "files.txt"
    metadata_path = temp_dir / "metadata.txt"
    with file_list.open("w", encoding="utf-8") as handle:
        for wav_file in wav_files:
            handle.write(f"file '{wav_file}'\n")
    create_ffmpeg_metadata(chapters_metadata, metadata_path)

    measured = measure_output_loudness(temp_dir, file_list.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    second_pass_filter = _loudnorm_filter(
        (
            f"measured_I={measured['input_i']:.2f}",
            f"measured_TP={measured['input_tp']:.2f}",
            f"measured_LRA={measured['input_lra']:.2f}",
            f"measured_thresh={measured['input_thresh']:.2f}",
            f"offset={measured['target_offset']:.2f}",
            "linear=true",
            "print_format=summary",
        )
    )
    _run_ffmpeg(
        [
            "-v",
            "warning",
            *_concat_source_args(file_list.name),
            "-i",
            metadata_path.name,
            "-map_metadata",
            "1",
            "-af",
            second_pass_filter,
            # 48 kHz keeps true-peak processing and AAC encoding on a standard
            # rate regardless of the TTS model's native output rate.
            "-ar",
            "48000",
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            str(output_path.resolve()),
        ],
        cwd=temp_dir,
    )


__all__ = [
    "BoundaryChunkLike",
    "ChunkLoudnessDiagnostics",
    "LoudnessMatchResult",
    "active_speech_rms",
    "add_chapter_silence",
    "assemble_chunk_audio",
    "create_ffmpeg_metadata",
    "crossfade",
    "fade_in",
    "fade_out",
    "match_chunk_loudness",
    "measure_output_loudness",
    "merge_chapters",
    "verify_audio_dependencies",
    "write_chapter_wav",
]
