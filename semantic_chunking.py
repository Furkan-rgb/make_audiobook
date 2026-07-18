"""Build coherent narration chunks from prepared chapter text.

The hierarchy is section to paragraph to sentence: paragraphs remain intact
when practical, adjacent short paragraphs may share a request, and sentences
are considered only when a paragraph exceeds the configured soft maximum.
Neighboring context is retained as metadata and is never added to spoken text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from audiobook_config import (
    CONTEXT_CHARS,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    TARGET_CHUNK_CHARS,
)


RE_SCENE_BREAK = re.compile(r"^(?:(?:\*\s*){3,}|-{3,}|_{3,}|~{3,})$")
RE_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[A-Z0-9\"'“‘(\[])"
)
RE_DIALOGUE = re.compile(r"^(?:[\"'“‘]|[-—]\s)")


@dataclass
class TextUnit:
    """An indivisible paragraph or sentence unit used while packing chunks."""

    text: str
    boundary_after: str
    paragraph_index: int
    is_dialogue: bool


@dataclass
class TextSection:
    """A group of paragraphs delimited by a subheading or scene break."""

    paragraphs: list[str]
    boundary_after: str = "section"


@dataclass
class NarrationChunk:
    """One TTS request plus unspoken neighboring-context metadata."""

    text: str
    boundary_after: str
    previous_context: str = ""
    following_context: str = ""

    @property
    def char_count(self) -> int:
        """Return the number of characters that will be narrated."""
        return len(self.text)


def _normalize_paragraph(block: str) -> str:
    """Normalize intraparagraph whitespace and remove Markdown heading marks."""
    block = block.strip()
    if block.startswith("#"):
        block = block.lstrip("# ")
    return " ".join(block.split())


def split_into_sections(content: str) -> list[TextSection]:
    """Split content at explicit scene breaks and Markdown subheadings."""
    sections: list[TextSection] = []
    current: list[str] = []
    pending_headings: list[str] = []

    def flush(boundary_after: str = "section") -> None:
        if current:
            sections.append(TextSection(current.copy(), boundary_after))
            current.clear()
        elif boundary_after == "scene" and sections:
            sections[-1].boundary_after = "scene"

    for raw_block in re.split(r"\n\s*\n", content):
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        if RE_SCENE_BREAK.fullmatch(raw_block):
            flush("scene")
            pending_headings.clear()
            continue

        paragraph = _normalize_paragraph(raw_block)
        if not paragraph:
            continue
        is_heading = raw_block.startswith("#") and paragraph[0].isupper()
        if is_heading:
            flush()
            pending_headings.append(paragraph)
            continue

        if pending_headings:
            paragraph = "\n\n".join([*pending_headings, paragraph])
            pending_headings.clear()
        current.append(paragraph)

    if pending_headings:
        if current:
            current.extend(pending_headings)
        elif sections:
            sections[-1].paragraphs.extend(pending_headings)
        else:
            current.extend(pending_headings)
    flush()

    return sections


def sentence_split(text: str) -> list[str]:
    """Split sentences only on demand, with a no-download regex fallback."""
    try:
        import nltk

        return [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    except (ImportError, LookupError):
        return [part.strip() for part in RE_SENTENCE_BOUNDARY.split(text) if part.strip()]


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Group complete sentences; leave an indivisible long sentence intact."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = sentence_split(paragraph)
    if len(sentences) <= 1:
        return [paragraph]

    parts: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        proposed = current_length + len(sentence) + (1 if current else 0)
        if current and proposed > max_chars:
            parts.append(" ".join(current))
            current = [sentence]
            current_length = len(sentence)
        else:
            current.append(sentence)
            current_length = proposed
    if current:
        parts.append(" ".join(current))
    return parts


def _make_text_units(content: str, max_chars: int) -> list[TextUnit]:
    """Convert section paragraphs into atomic units for chunk packing."""
    units: list[TextUnit] = []
    paragraph_index = 0
    for section in split_into_sections(content):
        section_start = len(units)
        for paragraph in section.paragraphs:
            parts = split_long_paragraph(paragraph, max_chars)
            for index, part in enumerate(parts):
                units.append(
                    TextUnit(
                        text=part,
                        boundary_after=(
                            "paragraph" if index == len(parts) - 1 else "continuation"
                        ),
                        paragraph_index=paragraph_index,
                        is_dialogue=bool(RE_DIALOGUE.match(part)),
                    )
                )
            paragraph_index += 1
        if len(units) > section_start:
            units[-1].boundary_after = section.boundary_after
    return units


def _join_units(units: Sequence[TextUnit]) -> str:
    """Join units using spaces within paragraphs and blank lines between them."""
    pieces: list[str] = []
    for index, unit in enumerate(units):
        pieces.append(unit.text)
        if index < len(units) - 1:
            pieces.append(" " if unit.boundary_after == "continuation" else "\n\n")
    return "".join(pieces)


def _context_tail(text: str, limit: int) -> str:
    """Return at most ``limit`` trailing characters, aligned to a word."""
    if len(text) <= limit:
        return text
    clipped = text[-limit:]
    first_space = clipped.find(" ")
    return clipped[first_space + 1 :] if first_space >= 0 else clipped


def _context_head(text: str, limit: int) -> str:
    """Return at most ``limit`` leading characters, aligned to a word."""
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    last_space = clipped.rfind(" ")
    return clipped[:last_space] if last_space >= 0 else clipped


def make_narration_chunks(
    content: str,
    min_chars: int = MIN_CHUNK_CHARS,
    target_chars: int = TARGET_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    context_chars: int = CONTEXT_CHARS,
) -> list[NarrationChunk]:
    """Create coherent chunks from sections, paragraphs, then sentences."""
    if min_chars <= 0 or target_chars < min_chars or target_chars > max_chars:
        raise ValueError(
            "Chunk sizes must satisfy 0 < min_chars <= target_chars <= max_chars"
        )

    units = _make_text_units(content, max_chars)
    chunks: list[NarrationChunk] = []
    current: list[TextUnit] = []

    def flush() -> None:
        if current:
            chunks.append(
                NarrationChunk(
                    text=_join_units(current),
                    boundary_after=current[-1].boundary_after,
                )
            )
            current.clear()

    for unit in units:
        if current:
            proposed_length = len(_join_units([*current, unit]))
            dialogue_exchange = current[-1].is_dialogue and unit.is_dialogue
            reached_target = len(_join_units(current)) >= target_chars
            if proposed_length > max_chars or (reached_target and not dialogue_exchange):
                flush()

        current.append(unit)
        if unit.boundary_after in {"section", "scene"}:
            flush()
    flush()

    # Avoid treating a short heading or final paragraph like an independent
    # recording. Merge it with a neighbor when the soft maximum and hard scene
    # boundaries allow it.
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if chunk.char_count >= min_chars or len(chunks) == 1:
            index += 1
            continue

        if index + 1 < len(chunks) and chunk.boundary_after != "scene":
            following = chunks[index + 1]
            separator = " " if chunk.boundary_after == "continuation" else "\n\n"
            combined_text = chunk.text + separator + following.text
            if len(combined_text) <= max_chars:
                chunks[index : index + 2] = [
                    NarrationChunk(combined_text, following.boundary_after)
                ]
                continue

        if index and chunks[index - 1].boundary_after != "scene":
            previous = chunks[index - 1]
            separator = " " if previous.boundary_after == "continuation" else "\n\n"
            combined_text = previous.text + separator + chunk.text
            if len(combined_text) <= max_chars:
                chunks[index - 1 : index + 1] = [
                    NarrationChunk(combined_text, chunk.boundary_after)
                ]
                index = max(0, index - 1)
                continue

        index += 1

    for index, chunk in enumerate(chunks):
        if index:
            chunk.previous_context = _context_tail(chunks[index - 1].text, context_chars)
        if index + 1 < len(chunks):
            chunk.following_context = _context_head(chunks[index + 1].text, context_chars)
    return chunks


def build_chunk_plan(
    chapters: Sequence[tuple[str, str]],
    preview_chunks: int | None = None,
) -> list[tuple[str, list[NarrationChunk]]]:
    """Build a per-chapter chunk plan, optionally capped across the whole book."""
    plan: list[tuple[str, list[NarrationChunk]]] = []
    remaining = preview_chunks
    for title, content in chapters:
        chunks = make_narration_chunks(content)
        if remaining is not None:
            chunks = chunks[:remaining]
            remaining -= len(chunks)
        if chunks:
            plan.append((title, chunks))
        if remaining is not None and remaining <= 0:
            break
    return plan


def display_chunk_plan(plan: Sequence[tuple[str, Sequence[NarrationChunk]]]) -> None:
    """Print concise chunk counts, size ranges, and indivisible overages."""
    total_chunks = sum(len(chunks) for _, chunks in plan)
    print(f"Planned {total_chunks} chunks across {len(plan)} chapters.")
    for title, chunks in plan:
        sizes = [chunk.char_count for chunk in chunks]
        oversized = sum(size > MAX_CHUNK_CHARS for size in sizes)
        print(
            f"  {title}: {len(chunks)} chunks, "
            f"{min(sizes)}-{max(sizes)} chars"
            + (f", {oversized} indivisible long sentences" if oversized else "")
        )


__all__ = [
    "NarrationChunk",
    "RE_DIALOGUE",
    "RE_SCENE_BREAK",
    "RE_SENTENCE_BOUNDARY",
    "TextSection",
    "TextUnit",
    "_context_head",
    "_context_tail",
    "_join_units",
    "_make_text_units",
    "_normalize_paragraph",
    "build_chunk_plan",
    "display_chunk_plan",
    "make_narration_chunks",
    "sentence_split",
    "split_into_sections",
    "split_long_paragraph",
]
