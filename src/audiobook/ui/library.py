"""Enumerate and mutate the voice library.

Listing is the backend's job — ``SynthesisProvider.voices`` answers "what can
narrate", wherever each voice lives — so this module only delegates it.  What
it adds is mutation of the file-backed voices in the voices directory
(transcripts, renames, deletes, imports), which stays on the same on-disk
layout the CLI uses: the directory is the database.  Voices without files
(a backend's own speakers) cannot be mutated, and the helpers refuse them.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SYNTHESIS_PROVIDER,
    VOICE_REFERENCE_METADATA_FILENAME,
    VOICES_DIR,
)
from ..synthesis.providers import VoiceInfo, create_synthesis_provider
from ..synthesis.voices import AUDIO_SUFFIXES

# The older name for a catalog row, kept for existing callers.
VoiceEntry = VoiceInfo


def list_voices() -> list[VoiceEntry]:
    """Every selectable voice, as the configured backend exposes them."""

    return list(create_synthesis_provider(DEFAULT_SYNTHESIS_PROVIDER).voices())


def find_voice(spec: str) -> VoiceEntry | None:
    """Look a voice up by the spec stored in the dropdown."""

    return next((v for v in list_voices() if v.spec == spec), None)


def save_transcript(entry: VoiceEntry, text: str) -> str:
    """Write a corrected transcript back to wherever that voice keeps it.

    An empty transcript is a meaningful choice, not a mistake: it drops the
    voice to timbre-only cloning, which is the right move when the recording
    and its transcript disagree and there is no time to fix the words.
    """

    if entry.builtin:
        raise ValueError("This voice lives in the synthesis backend and has no transcript to edit.")

    text = text.strip()
    if entry.folder:
        metadata = json.loads(entry.transcript_path.read_text(encoding="utf-8"))
        metadata["ref_text"] = text
        entry.transcript_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif text:
        entry.transcript_path.write_text(text + "\n", encoding="utf-8")
    elif entry.transcript_path.exists():
        entry.transcript_path.unlink()

    mode = "timbre + prosody" if text else "timbre only"
    return f"Saved to {entry.transcript_path} — this voice now clones {mode}."


def rename_voice(entry: VoiceEntry, new_name: str) -> str:
    """Rename a voice on disk and return the spec it now answers to.

    Renaming must move everything that makes the voice findable — the voice
    folder, or a loose recording plus its transcript sidecar — or the next
    ``list_voices`` would show a half-renamed orphan.
    """

    if entry.builtin:
        raise ValueError("This voice lives in the synthesis backend and cannot be renamed.")

    new_name = new_name.strip().replace("/", "_")
    if not new_name:
        raise ValueError("Give the voice a new name.")

    if entry.folder:
        source_dir = entry.audio_path.parent
        target_dir = source_dir.with_name(new_name)
        if target_dir.exists():
            raise ValueError(f"{target_dir} already exists.")
        source_dir.rename(target_dir)
        metadata_path = target_dir / VOICE_REFERENCE_METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["slug"] = new_name
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return new_name

    target = entry.audio_path.with_name(new_name + entry.audio_path.suffix)
    if target.exists():
        raise ValueError(f"{target} already exists.")
    entry.audio_path.rename(target)
    if entry.transcript_path.exists():
        entry.transcript_path.rename(target.with_suffix(".txt"))
    return str(target)


def delete_voice(entry: VoiceEntry) -> str:
    """Remove a voice from disk entirely."""

    if entry.builtin:
        raise ValueError("This voice lives in the synthesis backend and cannot be deleted.")
    if entry.folder:
        shutil.rmtree(entry.audio_path.parent)
        return f"Deleted {entry.audio_path.parent}."
    entry.audio_path.unlink()
    if entry.transcript_path.exists():
        entry.transcript_path.unlink()
    return f"Deleted {entry.audio_path}."


def _write_reference_metadata(
    voice_dir: Path, *, slug: str, ref_text: str, source_name: str
) -> None:
    """Write the ``reference.json`` that makes a folder a voice."""

    (voice_dir / VOICE_REFERENCE_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "slug": slug,
                "instruct": None,  # a recording has no persona it was rendered from
                "ref_text": ref_text,
                "source": source_name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def import_recording(
    upload_path: str | Path,
    name: str = "",
    ref_text: str = "",
    voices_dir: Path = VOICES_DIR,
) -> str:
    """Save an uploaded recording as a voice folder and return its spec.

    Gradio hands over a temp file that is cleaned up later, so the audio has to
    be copied somewhere permanent before it can become a voice.  It lands in
    ``voices/<name>/`` alongside its metadata, the same shape a designed voice
    has: one voice is one directory, whatever it was made from.  The upload's
    encoding is preserved — re-encoding to match a filename would lose quality
    to no end.
    """

    source = Path(upload_path)
    slug = (name.strip() or source.stem).replace("/", "_")
    voice_dir = voices_dir / slug
    voice_dir.mkdir(parents=True, exist_ok=True)
    destination = voice_dir / f"reference{source.suffix.lower()}"
    if destination.resolve() != source.resolve():
        shutil.copyfile(source, destination)
    _write_reference_metadata(
        voice_dir, slug=slug, ref_text=ref_text.strip(), source_name=source.name
    )
    return slug


def migrate_loose_recordings(voices_dir: Path = VOICES_DIR) -> list[str]:
    """Fold recordings saved directly in ``voices/`` into their own folders.

    Older versions wrote an imported recording as ``voices/<name>.wav`` plus a
    ``<name>.txt`` sidecar.  Both layouts still load, but two of them is one
    too many for rename, delete and the picker to keep explaining, so this
    moves the old shape to the new one.  Idempotent, and it refuses to touch a
    recording whose name is already taken by a folder.
    """

    if not voices_dir.exists():
        return []

    moved: list[str] = []
    for child in sorted(voices_dir.iterdir()):
        if not child.is_file() or child.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        voice_dir = voices_dir / child.stem
        if voice_dir.exists():
            continue
        sidecar = child.with_suffix(".txt")
        ref_text = sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""
        voice_dir.mkdir(parents=True)
        child.rename(voice_dir / f"reference{child.suffix.lower()}")
        _write_reference_metadata(
            voice_dir, slug=child.stem, ref_text=ref_text, source_name=child.name
        )
        if sidecar.exists():
            sidecar.unlink()
        moved.append(child.stem)
    return moved


def list_prepared_scripts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Prepared-book artifacts available to narrate."""

    if not output_dir.exists():
        return []
    return sorted(
        path
        for path in output_dir.rglob("*.json")
        if path.name not in {"chunk_manifest.json"} and _is_prepared_book(path)
    )


def _is_prepared_book(path: Path) -> bool:
    """Structural check so unrelated JSON never reaches the narrator.

    The keys are looked for anywhere in the file, not in a fixed-size head:
    ``chapters`` follows the source and provider metadata, which carry the
    prompt and can push it well past any prefix worth guessing.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return '"chapters"' in text and '"schema_version"' in text


def list_audiobooks(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Finished and preview audiobooks, newest first."""

    if not output_dir.exists():
        return []
    return sorted(output_dir.rglob("*.m4b"), key=lambda p: p.stat().st_mtime, reverse=True)


__all__ = [
    "VoiceEntry",
    "delete_voice",
    "find_voice",
    "import_recording",
    "list_audiobooks",
    "list_prepared_scripts",
    "list_voices",
    "migrate_loose_recordings",
    "rename_voice",
    "save_transcript",
]
