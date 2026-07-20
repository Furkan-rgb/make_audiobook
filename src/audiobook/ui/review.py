"""Read-only inspection of a prepared-book artifact.

The Review step exists so that GPU-hours are never spent narrating text nobody
looked at.  Reading the prose is only half of that.  The rest is knowing which
model wrote it, whether the run actually finished, whether the source book has changed
since, and — above all — which units the model *altered*, because reviewing a
book is reviewing its changes, not re-reading the source.

All of that is recorded in the JSON artifact and dropped by the rendered
markdown, which is a flat concatenation of prepared prose.  This module reads
the JSON and answers those questions without writing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..preparation import PreparedBook, load_prepared_book, sha256_file

# Narration pace assumed when turning a word count into a duration.  It matches
# the pace NARRATION_INSTRUCTION asks the model for, so the estimate describes
# this pipeline rather than audiobooks in general.
WORDS_PER_MINUTE = 140.0

_CACHE: dict[Path, tuple[float, int, PreparedBook]] = {}
_HASHES: dict[Path, tuple[float, int, str]] = {}


def _cached_digest(path: Path) -> str:
    """Hash a source file once per version of it, not once per glance.

    Selecting a prepared script re-checks that its source is unchanged, and a
    book file is large enough that re-hashing on every click is felt.  Keyed on mtime
    and size, so an actually-changed file is still hashed again.
    """

    stat = path.stat()
    signature = (stat.st_mtime, stat.st_size)
    cached = _HASHES.get(path)
    if cached is not None and cached[:2] == signature:
        return cached[2]
    digest = sha256_file(path)
    _HASHES[path] = (*signature, digest)
    return digest


def load_artifact(path: str | Path) -> PreparedBook:
    """Load a prepared book, reusing the last load of an unchanged file.

    Every panel in the Review section asks a different question of the same
    artifact, and a book is a multi-megabyte JSON parse.  The cache is keyed on
    mtime and size so a re-prepared artifact is never served stale.
    """

    resolved = Path(path)
    stat = resolved.stat()
    signature = (stat.st_mtime, stat.st_size)
    cached = _CACHE.get(resolved)
    if cached is not None and cached[:2] == signature:
        return cached[2]
    book = load_prepared_book(resolved)
    _CACHE[resolved] = (*signature, book)
    return book


@dataclass(frozen=True)
class FlaggedUnit:
    """One unit worth a human's attention, with the context to judge it."""

    unit_id: str
    position: int
    chapter_index: int
    chapter_title: str
    source_text: str
    warnings: tuple[str, ...]
    # sentence, category, original, replacement, reason
    edits: tuple[tuple[int, str, str, str, str], ...]

    @property
    def label(self) -> str:
        """Dropdown text: severity, location, and enough prose to recognise it."""

        excerpt = " ".join(self.source_text.split())[:60]
        marker = "⚠" if self.warnings else "·"
        counts = []
        if self.warnings:
            counts.append(f"{len(self.warnings)} warning{'s' if len(self.warnings) > 1 else ''}")
        if self.edits:
            counts.append(f"{len(self.edits)} edit{'s' if len(self.edits) > 1 else ''}")
        return f"{marker} ch{self.chapter_index}  {', '.join(counts)}  — {excerpt}…"


def flagged_units(book: PreparedBook) -> list[FlaggedUnit]:
    """Units the model changed or complained about, most serious first.

    Units it passed through untouched are not review material: they are the
    author's own sentences, already trusted.  Sorting warnings above edits puts
    the model's own doubts at the top of the list.
    """

    flagged: list[FlaggedUnit] = []
    for chapter in sorted(book.chapters, key=lambda item: item.index):
        for unit in chapter.units:
            if not unit.warnings and not unit.edits:
                continue
            flagged.append(
                FlaggedUnit(
                    unit_id=unit.unit_id,
                    position=unit.position,
                    chapter_index=chapter.index,
                    chapter_title=chapter.title,
                    source_text=unit.source_text,
                    warnings=tuple(unit.warnings),
                    edits=tuple(
                        (
                            edit.sentence,
                            edit.category,
                            edit.original,
                            edit.replacement,
                            edit.reason,
                        )
                        for edit in unit.edits
                    ),
                )
            )
    flagged.sort(
        key=lambda item: (
            not item.warnings,
            -len(item.warnings),
            -len(item.edits),
            item.chapter_index,
            item.position,
        )
    )
    return flagged


def render_unit(unit: FlaggedUnit | None) -> str:
    """Show what changed in one unit, and nothing else.

    The full source and prepared text are deliberately not repeated here: on a
    real book they bury the edits under two copies of the prose, and the Full
    text tab already holds the result.  Each edit quotes the words it touched,
    which is the part a reviewer has to judge.
    """

    if unit is None:
        return "_Select a unit to see what changed._"
    lines = [f"**{unit.chapter_title or f'Chapter {unit.chapter_index}'}** · `{unit.unit_id}`", ""]
    if unit.warnings:
        lines.append("**The model flagged:**")
        lines.extend(f"- {warning}" for warning in unit.warnings)
        lines.append("")
    if unit.edits:
        # Every edit here was applied: the prepared text is the source with
        # exactly these splices in it.  Anything the model proposed that could
        # not be placed was refused and appears above as a warning instead.
        lines.append("| Sentence | Category | Original | Replacement | Reason |")
        lines.append("| --- | --- | --- | --- | --- |")
        for sentence, category, original, replacement, reason in unit.edits:
            cells = [
                (value.replace("|", "\\|").replace("\n", " ") or "—")
                for value in (category, original, replacement, reason)
            ]
            anchor = str(sentence) if sentence else "—"
            lines.append(
                f"| {anchor} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |"
            )
    elif not unit.warnings:
        lines.append("_No edits recorded._")
    return "\n".join(lines)


def estimate_duration(book: PreparedBook) -> tuple[int, float]:
    """Word count and narrated minutes, so 'GPU-hours' becomes a number."""

    words = sum(len(unit.prepared_text.split()) for chapter in book.chapters for unit in chapter.units)
    return words, words / WORDS_PER_MINUTE


def _format_minutes(minutes: float) -> str:
    hours, remainder = divmod(int(round(minutes)), 60)
    return f"{hours} h {remainder:02d} min" if hours else f"{remainder} min"


def _source_line(book: PreparedBook) -> str:
    """Whether the artifact still describes the book sitting on disk."""

    source = book.source_metadata
    if not source.path:
        return "- Source: _not recorded_"
    path = Path(source.path)
    if not path.exists():
        return f"- Source: `{path}` — **missing now**; cannot confirm it is the book this came from."
    if not source.sha256:
        return f"- Source: `{path}` (no hash recorded)"
    if _cached_digest(path) != source.sha256:
        return (
            f"- Source: `{path}` — **has changed since preparation**. "
            "Re-prepare, or this narrates an older edition."
        )
    return f"- Source: `{path}` ✓ unchanged"


def _model_lines(book: PreparedBook, selected_model: str | None) -> list[str]:
    """Which model wrote this, including units that a different one wrote.

    Cached units survive a model change, so a book can be a mixture: the header
    says ``qwen3.6:35b`` while most of its prose came from whatever was
    configured a month ago.  That is invisible in the prose and decisive for
    whether the review means anything.
    """

    provider = book.provider_metadata
    # The policy is a paragraph of prose and belongs in the artifact, not in a
    # header line; the prompt version is what distinguishes two runs.
    lines = [
        f"- Prepared by **{provider.name} · {provider.model}** "
        f"(prompt {book.prompt_version})"
    ]
    seen: dict[str, int] = {}
    for chapter in book.chapters:
        for unit in chapter.units:
            if unit.provider_metadata is None:
                continue
            key = f"{unit.provider_metadata.name} · {unit.provider_metadata.model}"
            seen[key] = seen.get(key, 0) + 1
    others = {key: count for key, count in seen.items() if key != f"{provider.name} · {provider.model}"}
    if others:
        mixture = ", ".join(f"{count}× {key}" for key, count in sorted(others.items()))
        lines.append(
            f"- ⚠ **Mixed models**: some units were prepared by {mixture} and reused from cache."
        )
    if selected_model and selected_model != provider.model:
        lines.append(
            f"- Prepare is currently set to **{selected_model}**; re-running with it "
            "invalidates every cached unit and re-prepares the whole book."
        )
    return lines


def summarize(path: str | Path, selected_model: str | None = None) -> str:
    """The header that says whether this artifact is worth reviewing at all."""

    resolved = Path(path)
    try:
        book = load_artifact(resolved)
    except FileNotFoundError:
        return f"`{resolved}` no longer exists."
    except Exception as exc:  # artifact validation, malformed JSON
        return f"**Cannot read `{resolved}`:** {exc}"

    units = [unit for chapter in book.chapters for unit in chapter.units]
    cache_hits = sum(1 for unit in units if unit.cache_hit)
    flagged = flagged_units(book)
    words, minutes = estimate_duration(book)

    lines = [f"### {book.title}"]
    if not book.complete:
        lines.append(
            "> ⚠ **Partial artifact.** This was a preview run — narrating it "
            "produces part of a book, not an abridged one."
        )
    chapters = len(book.chapters)
    lines.append(
        f"- {chapters} chapter{'s' if chapters != 1 else ''} · {len(units)} units "
        f"· {words:,} words · ≈ {_format_minutes(minutes)} narrated"
    )
    lines.extend(_model_lines(book, selected_model))
    if units:
        lines.append(
            f"- {cache_hits}/{len(units)} units came from cache · "
            f"{len(flagged)} unit{'s' if len(flagged) != 1 else ''} to review"
        )
    lines.append(_source_line(book))
    if book.created_at:
        lines.append(f"- Prepared {book.created_at}")
    return "\n".join(lines)


__all__ = [
    "FlaggedUnit",
    "WORDS_PER_MINUTE",
    "estimate_duration",
    "flagged_units",
    "load_artifact",
    "render_unit",
    "summarize",
]
