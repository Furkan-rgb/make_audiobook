"""Schema-v1 hashing, validation, and atomic JSON artifact I/O."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .types import PreparedBook, SCHEMA_VERSION, SourceMetadata


class ArtifactValidationError(ValueError):
    """A prepared-book artifact is malformed or fails integrity checks."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def source_metadata_for_path(
    path: str | Path, *, media_type: str | None = None
) -> SourceMetadata:
    source_path = Path(path)
    stat = source_path.stat()
    return SourceMetadata(
        path=str(source_path),
        sha256=sha256_file(source_path),
        size_bytes=stat.st_size,
        media_type=media_type,
    )


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def refresh_hashes(book: PreparedBook) -> PreparedBook:
    """Refresh every derived hash in place and return ``book``."""

    if not book.created_at:
        book.created_at = datetime.now(timezone.utc).isoformat()
    for chapter in book.chapters:
        for unit in chapter.units:
            unit.source_sha256 = sha256_text(unit.source_text)
            unit.prepared_sha256 = sha256_text(unit.prepared_text)
        chapter.source_sha256 = sha256_text(chapter.source_text)
        chapter.normalized_sha256 = sha256_text(chapter.normalized_text)
        chapter.prepared_sha256 = sha256_text(chapter.prepared_text)
    book.source_sha256 = _canonical_hash(
        [
            {"index": chapter.index, "title": chapter.title, "text": chapter.source_text}
            for chapter in sorted(book.chapters, key=lambda item: item.index)
        ]
    )
    book.prepared_sha256 = _canonical_hash(
        [
            {
                "index": chapter.index,
                "title": chapter.title,
                "text": chapter.prepared_text,
            }
            for chapter in sorted(book.chapters, key=lambda item: item.index)
        ]
    )
    return book


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_artifact(book: PreparedBook) -> None:
    """Validate schema, structure, JSON compatibility, and all text hashes."""

    issues: list[str] = []
    if book.schema_version != SCHEMA_VERSION:
        issues.append(
            f"unsupported schema_version {book.schema_version}; expected {SCHEMA_VERSION}"
        )
    if not book.title.strip():
        issues.append("book title is blank")
    if not book.provider_metadata.name.strip() or not book.provider_metadata.model.strip():
        issues.append("provider metadata must include name and model")
    if not book.created_at:
        issues.append("created_at is blank")
    if book.source_metadata.sha256 is not None and not _is_sha256(
        book.source_metadata.sha256
    ):
        issues.append("source_metadata.sha256 is not a lowercase SHA256 digest")

    chapter_indexes: set[int] = set()
    unit_ids: set[str] = set()
    for chapter in book.chapters:
        label = f"chapter {chapter.index}"
        if chapter.index in chapter_indexes:
            issues.append(f"duplicate {label}")
        chapter_indexes.add(chapter.index)
        if not chapter.title.strip():
            issues.append(f"{label} title is blank")
        if chapter.source_sha256 != sha256_text(chapter.source_text):
            issues.append(f"{label} source hash mismatch")
        if chapter.normalized_sha256 != sha256_text(chapter.normalized_text):
            issues.append(f"{label} normalized hash mismatch")
        if chapter.prepared_sha256 != sha256_text(chapter.prepared_text):
            issues.append(f"{label} prepared hash mismatch")

        positions: set[int] = set()
        for unit in chapter.units:
            unit_label = f"{label} unit {unit.position}"
            if unit.position in positions:
                issues.append(f"duplicate position for {unit_label}")
            positions.add(unit.position)
            if not unit.unit_id or unit.unit_id in unit_ids:
                issues.append(f"duplicate or blank unit_id for {unit_label}")
            unit_ids.add(unit.unit_id)
            if unit.kind not in {"prose", "heading", "scene_marker"}:
                issues.append(f"invalid kind for {unit_label}: {unit.kind!r}")
            if not unit.source_text.strip() or not unit.prepared_text.strip():
                issues.append(f"blank source or prepared text for {unit_label}")
            if unit.source_sha256 != sha256_text(unit.source_text):
                issues.append(f"source hash mismatch for {unit_label}")
            if unit.prepared_sha256 != sha256_text(unit.prepared_text):
                issues.append(f"prepared hash mismatch for {unit_label}")
            if unit.kind != "prose":
                if unit.prepared_text != unit.source_text:
                    issues.append(f"structural text changed for {unit_label}")
                if unit.provider_metadata is not None:
                    issues.append(f"structural unit has provider metadata for {unit_label}")

    expected_source = _canonical_hash(
        [
            {"index": chapter.index, "title": chapter.title, "text": chapter.source_text}
            for chapter in sorted(book.chapters, key=lambda item: item.index)
        ]
    )
    expected_prepared = _canonical_hash(
        [
            {
                "index": chapter.index,
                "title": chapter.title,
                "text": chapter.prepared_text,
            }
            for chapter in sorted(book.chapters, key=lambda item: item.index)
        ]
    )
    if book.source_sha256 != expected_source:
        issues.append("book source hash mismatch")
    if book.prepared_sha256 != expected_prepared:
        issues.append("book prepared hash mismatch")
    try:
        json.dumps(book.to_dict(), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        issues.append(f"artifact contains non-JSON metadata: {exc}")
    if issues:
        raise ArtifactValidationError("Invalid prepared-book artifact: " + "; ".join(issues))


def save_prepared_book(book: PreparedBook, path: str | Path) -> Path:
    """Hash, validate, and atomically replace a schema-v1 JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    refresh_hashes(book)
    validate_artifact(book)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                book.to_dict(), handle, ensure_ascii=False, indent=2, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_prepared_book(path: str | Path) -> PreparedBook:
    """Load and integrity-check a schema-v1 prepared-book artifact."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"Malformed JSON artifact {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError("Prepared-book artifact root must be an object")
    try:
        book = PreparedBook.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"Malformed schema-v1 artifact: {exc}") from exc
    validate_artifact(book)
    return book


atomic_save_prepared_book = save_prepared_book


def render_prepared_markdown(book: PreparedBook) -> str:
    """Render the prepared script without injecting any new spoken headings."""

    return "\n\n".join(
        chapter.prepared_text
        for chapter in sorted(book.chapters, key=lambda item: item.index)
        if chapter.prepared_text
    ).rstrip() + "\n"


def save_prepared_markdown(book: PreparedBook, path: str | Path) -> Path:
    """Atomically write the human-reviewable prepared narration script."""

    validate_artifact(book)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(render_prepared_markdown(book))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination
