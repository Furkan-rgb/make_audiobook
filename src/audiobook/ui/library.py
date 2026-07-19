"""Enumerate and mutate the voices directory.

``resolve_voice`` answers "load this one"; the frontend also needs "what is
there" and "save this change", which is all this module adds.  It stays on the
same on-disk layout the CLI uses, so a voice created here works from the command
line and vice versa — the directory is the database.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    DEFAULT_OUTPUT_DIR,
    VOICE_REFERENCE_AUDIO_FILENAME,
    VOICE_REFERENCE_METADATA_FILENAME,
    VOICES_DIR,
)
from ..synthesis.voices import AUDIO_SUFFIXES, reference_audio_path


@dataclass(frozen=True)
class VoiceEntry:
    """A selectable voice, described without decoding its audio.

    Listing must stay cheap: the picker refreshes on every tab switch, and
    decoding every reference to fill a dropdown would make that crawl.
    """

    spec: str
    label: str
    audio_path: Path
    transcript_path: Path
    ref_text: str | None
    instruct: str | None
    designed: bool
    folder: bool = True

    @property
    def has_transcript(self) -> bool:
        return bool(self.ref_text)


def _folder_voice(voice_dir: Path) -> VoiceEntry | None:
    """Read a voice folder — the layout every voice is written in.

    Designed voices and imported recordings share it: a reference clip and a
    ``reference.json`` beside it.  What distinguishes them is the persona the
    designed one was rendered from, which a recording has no equivalent of.
    """

    metadata_path = voice_dir / VOICE_REFERENCE_METADATA_FILENAME
    audio_path = reference_audio_path(voice_dir)
    if not metadata_path.exists() or audio_path is None:
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    instruct = metadata.get("instruct") or None
    return VoiceEntry(
        spec=voice_dir.name,
        label=f"{voice_dir.name}  ({'designed' if instruct else 'recording'})",
        audio_path=audio_path,
        transcript_path=metadata_path,
        ref_text=metadata.get("ref_text") or None,
        instruct=instruct,
        designed=bool(instruct),
        folder=True,
    )


def _loose_voice(path: Path) -> VoiceEntry:
    """A recording sitting directly in ``voices/``, as older versions wrote them.

    Still listed and still usable, so nothing anyone saved stops working; new
    imports go into folders, and :func:`migrate_loose_recordings` folds these in.
    """

    transcript_path = path.with_suffix(".txt")
    ref_text = (
        transcript_path.read_text(encoding="utf-8").strip()
        if transcript_path.exists()
        else None
    )
    return VoiceEntry(
        spec=str(path),
        label=f"{path.stem}  (recording)",
        audio_path=path,
        transcript_path=transcript_path,
        ref_text=ref_text or None,
        instruct=None,
        designed=False,
        folder=False,
    )


def list_voices(voices_dir: Path = VOICES_DIR) -> list[VoiceEntry]:
    """Every usable voice: voice folders first, then any loose recordings."""

    if not voices_dir.exists():
        return []

    folders = [
        entry
        for child in sorted(voices_dir.iterdir())
        if child.is_dir() and (entry := _folder_voice(child)) is not None
    ]
    loose = [
        _loose_voice(child)
        for child in sorted(voices_dir.iterdir())
        if child.is_file() and child.suffix.lower() in AUDIO_SUFFIXES
    ]
    return folders + loose


def find_voice(spec: str, voices_dir: Path = VOICES_DIR) -> VoiceEntry | None:
    """Look a voice up by the spec stored in the dropdown."""

    return next((v for v in list_voices(voices_dir) if v.spec == spec), None)


def save_transcript(entry: VoiceEntry, text: str) -> str:
    """Write a corrected transcript back to wherever that voice keeps it.

    An empty transcript is a meaningful choice, not a mistake: it drops the
    voice to timbre-only cloning, which is the right move when the recording
    and its transcript disagree and there is no time to fix the words.
    """

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
        ref_text = (
            sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else ""
        )
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
    """Cheap structural check so unrelated JSON never reaches the narrator."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    return '"chapters"' in head and '"schema_version"' in head


def list_audiobooks(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Finished and preview audiobooks, newest first."""

    if not output_dir.exists():
        return []
    return sorted(
        output_dir.rglob("*.m4b"), key=lambda p: p.stat().st_mtime, reverse=True
    )


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
