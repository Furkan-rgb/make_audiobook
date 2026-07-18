"""Structure-aware segmentation for narration preparation providers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .normalization import is_markdown_heading, is_scene_marker
from .types import UnitKind


DEFAULT_TARGET_UNIT_CHARS = 3_500
DEFAULT_MAX_UNIT_CHARS = 6_000
DEFAULT_CONTEXT_CHARS = 600

_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[A-Z0-9\"'“‘(\[])"
)


@dataclass(frozen=True)
class SourceUnit:
    """A deterministic unit; only ``prose`` units are sent to a provider."""

    position: int
    kind: UnitKind
    text: str
    previous_context: str = ""
    following_context: str = ""


def _split_sentences(paragraph: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY_RE.split(paragraph)]
    return [part for part in parts if part]


def _split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = _split_sentences(paragraph)
    if len(sentences) < 2:
        # Never sever an indivisible sentence merely to satisfy a soft limit.
        return [paragraph]

    groups: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence])
        if current and len(candidate) > max_chars:
            groups.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        groups.append(" ".join(current))
    return groups


def _clip_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[-limit:]
    space = clipped.find(" ")
    return clipped[space + 1 :] if space >= 0 else clipped


def _clip_head(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    space = clipped.rfind(" ")
    return clipped[:space] if space >= 0 else clipped


def _group_prose(paragraphs: Sequence[str], target_chars: int, max_chars: int) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    for paragraph in paragraphs:
        for part in _split_oversized_paragraph(paragraph, max_chars):
            candidate = "\n\n".join([*current, part])
            if current and (
                len(candidate) > max_chars
                or len("\n\n".join(current)) >= target_chars
            ):
                groups.append("\n\n".join(current))
                current = [part]
            else:
                current.append(part)
    if current:
        groups.append("\n\n".join(current))
    return groups


def segment_text(
    normalized_text: str,
    *,
    target_chars: int = DEFAULT_TARGET_UNIT_CHARS,
    max_chars: int = DEFAULT_MAX_UNIT_CHARS,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[SourceUnit]:
    """Create provider-sized prose units and passthrough structural units."""

    if target_chars <= 0 or max_chars < target_chars:
        raise ValueError("Unit sizes must satisfy 0 < target_chars <= max_chars")
    if context_chars < 0:
        raise ValueError("context_chars cannot be negative")

    raw_units: list[tuple[UnitKind, str]] = []
    pending_prose: list[str] = []

    def flush_prose() -> None:
        if pending_prose:
            raw_units.extend(
                ("prose", text)
                for text in _group_prose(
                    pending_prose, target_chars=target_chars, max_chars=max_chars
                )
            )
            pending_prose.clear()

    for block in re.split(r"\n\s*\n", normalized_text):
        block = block.strip()
        if not block:
            continue
        if is_markdown_heading(block):
            flush_prose()
            raw_units.append(("heading", block))
        elif is_scene_marker(block):
            flush_prose()
            raw_units.append(("scene_marker", block))
        else:
            pending_prose.append(block)
    flush_prose()

    prose_positions = [
        position for position, (kind, _text) in enumerate(raw_units) if kind == "prose"
    ]
    previous_prose: dict[int, str] = {}
    following_prose: dict[int, str] = {}
    for index, position in enumerate(prose_positions):
        if index:
            previous_prose[position] = raw_units[prose_positions[index - 1]][1]
        if index + 1 < len(prose_positions):
            following_prose[position] = raw_units[prose_positions[index + 1]][1]

    return [
        SourceUnit(
            position=position,
            kind=kind,
            text=text,
            previous_context=_clip_tail(previous_prose.get(position, ""), context_chars),
            following_context=_clip_head(
                following_prose.get(position, ""), context_chars
            ),
        )
        for position, (kind, text) in enumerate(raw_units)
    ]
