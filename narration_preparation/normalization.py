"""Conservative, deterministic cleanup before model-assisted adaptation."""

from __future__ import annotations

import re
import unicodedata


MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)\S?.*$")
SCENE_MARKER_RE = re.compile(
    r"^\s*(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,}|(?:~\s*){3,})\s*$"
)
_LINE_END_HYPHEN_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[a-z])")
_SOFT_HYPHEN_RE = re.compile("\u00ad")
_SPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_RE = re.compile(r"\n[ \t]*\n+")


def is_markdown_heading(text: str) -> bool:
    """Return whether a block is a Markdown ATX heading."""

    return bool(MARKDOWN_HEADING_RE.fullmatch(text.strip()))


def is_scene_marker(text: str) -> bool:
    """Return whether a block is an explicit Markdown scene divider."""

    return bool(SCENE_MARKER_RE.fullmatch(text.strip()))


def _logical_blocks(text: str) -> list[str]:
    """Split prose, headings, and scene markers without losing structure."""

    blocks: list[str] = []
    prose_lines: list[str] = []

    def flush_prose() -> None:
        if prose_lines:
            blocks.append("\n".join(prose_lines))
            prose_lines.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush_prose()
            continue
        if is_markdown_heading(stripped) or is_scene_marker(stripped):
            flush_prose()
            blocks.append(stripped)
            continue
        prose_lines.append(stripped)
    flush_prose()
    return blocks


def normalize_paragraph(text: str) -> str:
    """Normalize whitespace and wrapped lines inside one prose paragraph."""

    text = unicodedata.normalize("NFC", text)
    text = _SOFT_HYPHEN_RE.sub("", text)
    text = _LINE_END_HYPHEN_RE.sub("", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2007", " ").replace("\u202f", " ")
    text = re.sub(r"(?<=\w)-[ \t]+(?=\w)", "-", text)
    text = text.replace("\n", " ")
    return _SPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    """Normalize extracted text while retaining paragraph boundaries.

    Headings and explicit scene markers remain separate blocks so that the
    segmentation stage can keep them out of provider requests.
    """

    if not isinstance(text, str):
        raise TypeError("Text to normalize must be a string")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SOFT_HYPHEN_RE.sub("", text)
    text = _BLANK_RE.sub("\n\n", text).strip()

    normalized: list[str] = []
    for block in _logical_blocks(text):
        if is_markdown_heading(block):
            heading = _SPACE_RE.sub(" ", block).strip()
            normalized.append(heading)
        elif is_scene_marker(block):
            normalized.append(block.strip())
        else:
            paragraph = normalize_paragraph(block)
            if paragraph:
                normalized.append(paragraph)
    return "\n\n".join(normalized)
