"""Orchestrate preparation and narration without coupling their providers."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from tqdm import tqdm

from .assembly.audio import (
    assemble_chunk_audio,
    merge_chapters,
    verify_audio_dependencies,
    write_chapter_wav,
)
from .config import (
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_PREPARED_MARKDOWN_FILENAME,
    DEFAULT_PREPARED_SCRIPT_FILENAME,
    DEFAULT_PREVIEW_OUTPUT_FILENAME,
    ACTIVE_VOICE,
    LANGUAGE,
    LOCAL_VOICE_CLONE_MODEL_PATH,
    NARRATION_INSTRUCTION,
    TARGET_CHUNK_DURATION_SECONDS,
    TTS_BACKEND,
    VOICE_CLONE_MODEL,
    VOICE_NAME,
    VOICES_DIR,
)
from .extraction import parse_book_to_chapters, source_media_type
from .synthesis.qwen import (
    build_voice_clone_prompt,
    generate_chunk,
    generate_clone_chunk,
    load_qwen_model,
    verify_supported_voice,
    verify_tts_dependencies,
)
from .synthesis.voices import describe, resolve_voice
from .chunking.semantic import NarrationChunk, build_chunk_plan, display_chunk_plan


@dataclass(frozen=True)
class PreparationWorkflowOptions:
    """Provider-neutral configuration for producing a prepared script."""

    # Any supported book format; the extraction backend is chosen from it.
    source_path: Path
    output_dir: Path
    provider_name: str
    model: str
    base_url: str
    timeout_seconds: float
    script_path: Path | None = None
    preview_chapters: int | None = None
    preview_units: int | None = None
    force: bool = False


@dataclass(frozen=True)
class NarrationWorkflowOptions:
    """Configuration for turning prepared chapters into a chaptered M4B."""

    output_dir: Path
    tts_model: str
    script_path: Path | None = None
    preview_chapters: int | None = None
    preview_chunks: int | None = None
    dry_run: bool = False
    keep_temp: bool = False
    preparation_was_preview: bool = False
    # Reference voice for the clone backend. ``None`` falls back to the
    # committed ACTIVE_VOICE, so the CLI keeps its existing behaviour while a
    # caller running several narrations can choose per run.
    voice: str | None = None


def resolve_script_path(output_dir: Path, script_path: Path | None) -> Path:
    """Resolve the canonical JSON artifact path for a workflow invocation."""

    return script_path or output_dir / DEFAULT_PREPARED_SCRIPT_FILENAME


def prepared_markdown_path(script_path: Path) -> Path:
    """Return the human-readable companion path next to the JSON artifact."""

    if script_path.name == DEFAULT_PREPARED_SCRIPT_FILENAME:
        return script_path.with_name(DEFAULT_PREPARED_MARKDOWN_FILENAME)
    return script_path.with_suffix(".md")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_prepared_markdown(book: Any, path: Path) -> None:
    """Write a clean, human-readable view of a prepared-book artifact."""

    lines = [f"# {book.title}", ""]
    lines.extend(
        [
            f"> Prepared by {book.provider_metadata.name} / "
            f"{book.provider_metadata.model}",
            "",
        ]
    )
    for chapter in sorted(book.chapters, key=lambda item: item.index):
        prepared_text = chapter.prepared_text.strip()
        first_line = prepared_text.splitlines()[0].lstrip("# ").strip()
        if first_line.casefold() != chapter.title.strip().casefold():
            lines.extend([f"## {chapter.title}", ""])
        lines.extend([prepared_text, ""])
    _atomic_write_text(path, "\n".join(lines).rstrip() + "\n")


def _book_title_from_source(path: Path) -> str:
    stem = re.sub(r"(?i)(pdf|epub)$", "", path.stem).strip(" _-")
    words = re.sub(r"[_-]+", " ", stem).strip()
    return words.title() or "Audiobook"


def _create_preparation_provider(options: PreparationWorkflowOptions):
    from .preparation import create_provider

    return create_provider(
        options.provider_name,
        model=options.model,
        base_url=options.base_url,
        timeout=options.timeout_seconds,
    )


def prepare_narration_script(options: PreparationWorkflowOptions):
    """Extract a book, adapt its prose, and checkpoint a prepared-book artifact."""

    from .preparation import (
        NarrationPreparationPipeline,
        SourceMetadata,
        load_prepared_book,
        save_prepared_book,
        sha256_file,
    )

    if options.preview_chapters is not None and options.preview_chapters <= 0:
        raise ValueError("--preview-chapters must be positive")
    if options.preview_units is not None and options.preview_units <= 0:
        raise ValueError("--preview-units must be positive")
    if options.timeout_seconds <= 0:
        raise ValueError("--provider-timeout must be positive")

    chapters = parse_book_to_chapters(options.source_path)
    if options.preview_chapters is not None:
        chapters = chapters[: options.preview_chapters]

    script_path = resolve_script_path(options.output_dir, options.script_path)
    resume_from = None
    if script_path.exists() and not options.force:
        resume_from = load_prepared_book(script_path)

    source = SourceMetadata(
        path=str(options.source_path),
        sha256=sha256_file(options.source_path),
        size_bytes=options.source_path.stat().st_size,
        media_type=source_media_type(options.source_path),
    )
    provider = _create_preparation_provider(options)
    pipeline = NarrationPreparationPipeline(provider)

    def checkpoint(book: Any, *, validate: bool = False) -> None:
        # Per-unit checkpoints skip the whole-book re-validation; the final save
        # (validate=True) is the integrity gate for the persisted artifact.
        save_prepared_book(book, script_path, validate=validate)
        write_prepared_markdown(book, prepared_markdown_path(script_path))

    # The unit count is only known once segmentation has run, which happens
    # inside prepare_book, so the bar is opened by the first progress report.
    bar: tqdm | None = None

    def progress(done: int, total: int) -> None:
        nonlocal bar
        if bar is None:
            print(f"Adapting {total} units with {provider.metadata.model}...")
            bar = tqdm(total=total, desc="Preparing", unit="unit")
        bar.update(done - bar.n)

    try:
        book = pipeline.prepare_book(
            chapters,
            book_title=_book_title_from_source(options.source_path),
            source_metadata=source,
            resume_from=resume_from,
            checkpoint=checkpoint,
            max_prose_units=options.preview_units,
            progress=progress,
        )
        checkpoint(book, validate=True)
    finally:
        if bar is not None:
            bar.close()
        provider.close()

    print(f"Prepared narration script: {script_path}")
    print(f"Reviewable text: {prepared_markdown_path(script_path)}")
    return book


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preparation_manifest(book: Any | None, script_path: Path | None) -> dict[str, Any]:
    if book is None:
        return {}
    metadata: dict[str, Any] = {
        "schema_version": book.schema_version,
        "title": book.title,
        "complete": book.complete,
        "source_sha256": book.source_sha256,
        "prepared_sha256": book.prepared_sha256,
        "provider": book.provider_metadata.to_dict(),
    }
    if script_path is not None and script_path.exists():
        metadata.update(
            {
                "path": str(script_path),
                "artifact_sha256": _sha256_file(script_path),
            }
        )
    return metadata


def narration_chapters(book: Any) -> list[tuple[str, str]]:
    """Adapt a validated prepared book to the existing chunk-planner contract."""

    return [
        (chapter.title, chapter.prepared_text)
        for chapter in sorted(book.chapters, key=lambda item: item.index)
        if chapter.prepared_text.strip()
    ]


@dataclass(frozen=True)
class _Narrator:
    """A loaded TTS voice and the metadata recorded for a narration run."""

    label: str
    model_path: str
    instruction: str | None
    generate: Callable[[Any], tuple[np.ndarray, int]]


def _load_clone_narrator(voice_spec: str | None = None) -> _Narrator:
    """Load the Base clone model and lock in the reference voice for this run."""

    try:
        voice = resolve_voice(voice_spec or ACTIVE_VOICE, voices_dir=VOICES_DIR)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{exc} Set ACTIVE_VOICE before narrating with "
            "TTS_BACKEND='voice_clone'."
        ) from exc
    print(describe(voice))

    model_path = str(
        LOCAL_VOICE_CLONE_MODEL_PATH
        if LOCAL_VOICE_CLONE_MODEL_PATH.exists()
        else VOICE_CLONE_MODEL
    )
    model = load_qwen_model(model_path)
    # Encode the reference once; every chunk reuses it for a stable narrator.
    prompt = build_voice_clone_prompt(
        model,
        ref_audio=voice.audio,
        sample_rate=voice.sample_rate,
        ref_text=voice.ref_text,
    )
    return _Narrator(
        label=voice.slug,
        model_path=model_path,
        instruction=voice.instruct,
        generate=lambda chunk: generate_clone_chunk(
            model, chunk, voice_clone_prompt=prompt, language=LANGUAGE
        ),
    )


def _load_narrator(tts_model: str, voice_spec: str | None = None) -> _Narrator:
    """Build the narrator selected by ``TTS_BACKEND``.

    ``tts_model`` is the CustomVoice checkpoint requested on the command line; it
    is used only by the built-in-speaker backend. The clone backend loads the
    Base checkpoint and the requested reference clip instead.
    """

    verify_tts_dependencies()
    if TTS_BACKEND == "voice_clone":
        return _load_clone_narrator(voice_spec)
    if TTS_BACKEND != "custom_voice":
        raise ValueError(f"Unknown TTS_BACKEND: {TTS_BACKEND!r}")
    model = load_qwen_model(tts_model)
    verify_supported_voice(model)
    return _Narrator(
        label=VOICE_NAME,
        model_path=tts_model,
        instruction=NARRATION_INSTRUCTION,
        generate=lambda chunk: generate_chunk(model, chunk),
    )


def narrate_chapters(
    chapters: Sequence[tuple[str, str]],
    options: NarrationWorkflowOptions,
    *,
    prepared_book: Any | None = None,
) -> Path | None:
    """Generate a chaptered audiobook from already prepared chapter text."""

    if options.preview_chapters is not None:
        if options.preview_chapters <= 0:
            raise ValueError("--preview-chapters must be positive")
        chapters = chapters[: options.preview_chapters]
    if options.preview_chunks is not None and options.preview_chunks <= 0:
        raise ValueError("--preview-chunks must be positive")

    plan = build_chunk_plan(chapters, options.preview_chunks)
    if not plan:
        raise RuntimeError("No narratable text was found in the prepared script.")
    display_chunk_plan(plan)
    if options.dry_run:
        return None

    verify_audio_dependencies()
    narrator = _load_narrator(options.tts_model, options.voice)

    temp_dir = options.output_dir / "temp_parts"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    chapter_timings: list[tuple[str, int, int]] = []
    wav_files: list[str] = []
    manifest: list[dict[str, Any]] = []
    current_time_ms = 0
    global_chunk_index = 0
    sample_rate: int | None = None

    total_chunks = sum(len(chunks) for _, chunks in plan)
    print(f"Generating {total_chunks} chunks with {narrator.label}...")
    for chapter_index, (title, chunks) in enumerate(plan):
        audio_segments: list[np.ndarray] = []
        for chapter_chunk_index, chunk in enumerate(
            tqdm(chunks, desc=title, leave=False, unit="chunk")
        ):
            audio, generated_rate = narrator.generate(chunk)
            if sample_rate is None:
                sample_rate = generated_rate
            elif generated_rate != sample_rate:
                raise RuntimeError("Qwen returned inconsistent sample rates.")

            duration_seconds = len(audio) / generated_rate
            audio_segments.append(audio)
            item = asdict(chunk)
            item.update(
                {
                    "chapter": title,
                    "chapter_index": chapter_index,
                    "chunk_index": global_chunk_index,
                    "chapter_chunk_index": chapter_chunk_index,
                    "char_count": chunk.char_count,
                    "duration_seconds": round(duration_seconds, 3),
                    "duration_target_met": (
                        TARGET_CHUNK_DURATION_SECONDS[0]
                        <= duration_seconds
                        <= TARGET_CHUNK_DURATION_SECONDS[1]
                    ),
                }
            )
            manifest.append(item)
            global_chunk_index += 1

        if sample_rate is None:
            continue
        chapter_audio = assemble_chunk_audio(chunks, audio_segments, sample_rate)
        wav_name, duration_ms = write_chapter_wav(
            temp_dir, chapter_index, chapter_audio, sample_rate
        )
        chapter_timings.append((title, current_time_ms, current_time_ms + duration_ms))
        current_time_ms += duration_ms
        wav_files.append(wav_name)

    manifest_path = options.output_dir / "chunk_manifest.json"
    options.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = options.script_path
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "tts_backend": TTS_BACKEND,
                "tts_model": narrator.model_path,
                "voice": narrator.label,
                "instruction": narrator.instruction,
                "prepared_script": _preparation_manifest(prepared_book, script_path),
                "chunks": manifest,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    is_preview = (
        options.preview_chapters is not None
        or options.preview_chunks is not None
        or options.preparation_was_preview
    )
    output_name = (
        DEFAULT_PREVIEW_OUTPUT_FILENAME if is_preview else DEFAULT_OUTPUT_FILENAME
    )
    output_path = options.output_dir / output_name
    print(f"Merging {len(wav_files)} chapters into {output_path}...")
    merge_chapters(temp_dir, wav_files, chapter_timings, output_path)
    if not options.keep_temp:
        shutil.rmtree(temp_dir)
    print(f"Audiobook ready: {output_path}")
    print(f"Chunk diagnostics: {manifest_path}")
    return output_path


def narrate_prepared_script(options: NarrationWorkflowOptions) -> Path | None:
    """Load and validate a prepared JSON artifact before narration."""

    from .preparation import load_prepared_book

    script_path = resolve_script_path(options.output_dir, options.script_path)
    book = load_prepared_book(script_path)
    return narrate_chapters(
        narration_chapters(book),
        NarrationWorkflowOptions(
            output_dir=options.output_dir,
            tts_model=options.tts_model,
            script_path=script_path,
            preview_chapters=options.preview_chapters,
            preview_chunks=options.preview_chunks,
            dry_run=options.dry_run,
            keep_temp=options.keep_temp,
            preparation_was_preview=(
                options.preparation_was_preview or not book.complete
            ),
            voice=options.voice,
        ),
        prepared_book=book,
    )


__all__ = [
    "NarrationWorkflowOptions",
    "PreparationWorkflowOptions",
    "narrate_chapters",
    "narrate_prepared_script",
    "narration_chapters",
    "prepare_narration_script",
    "prepared_markdown_path",
    "resolve_script_path",
    "write_prepared_markdown",
]
