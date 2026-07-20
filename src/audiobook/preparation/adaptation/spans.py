"""Cutting a passage into addressable sentences, and rendering it for a model.

Everything here is deterministic and reversible: a span is a pair of offsets
into the source, never a copy of it. The gaps between spans — single spaces,
paragraph breaks — are left in the source untouched, which is what lets an edit
be spliced in without any part of the passage being rebuilt from parts.
"""

from __future__ import annotations

import re


# Alternative one is a sentence terminator with its trailing closing quotes and
# any footnote marker riding on it, followed by whitespace and something that
# can open a sentence. Alternative two is a paragraph break, which ends a
# sentence whether or not the author punctuated it. Both capture the separator
# so a span can end where the whitespace begins.
_BOUNDARY_RE = re.compile(
    r"""
    (?: [.!?] ["'”’)\]]* (?: \[[\d\s,;-]+\] )? ["'”’)\]]* )
    (?P<gap>\s+)
    (?= ["'“‘(\[]* [A-Z0-9] )
    |
    (?P<para>\n[ \t]*\n\s*)
    """,
    re.VERBOSE,
)

# A period after an abbreviation or an initial is not a sentence end. Getting
# this wrong costs nothing structurally — it only fragments the view a model
# reads — but a passage numbered "3: Dr." invites nonsense anchors.
_ABBREVIATION_RE = re.compile(
    r"(?:\b[A-Z]|\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Rev|Hon|Gen|Col|Capt|Lt|Sgt"
    r"|vs|etc|al|cf|ca|approx|Fig|No|Vol|Ch|pp|ed|eds|trans|Inc|Ltd|Co|Corp"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec))\.$",
    re.UNICODE,
)

# The label a numbered line carries, and the pattern that takes it back off.
# "n:" rather than "[n]" deliberately: a bracketed label is indistinguishable
# from the numeric reference markers a model is told to delete, and it will
# dutifully propose deleting the labels — hundreds of them, one per sentence.
_LABEL_RE = re.compile(r"^\s*\d+:\s*")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Offsets of each sentence, excluding the separators between them.

    The spans plus the gaps between them reconstitute ``text`` exactly.
    """

    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY_RE.finditer(text):
        paragraph_break = match.group("para") is not None
        end = match.start("para") if paragraph_break else match.start("gap")
        if not paragraph_break and _ABBREVIATION_RE.search(text[cursor:end]):
            continue
        span = _trim(text, cursor, end)
        if span is not None:
            spans.append(span)
        cursor = match.end()
    tail = _trim(text, cursor, len(text))
    if tail is not None:
        spans.append(tail)
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def numbered_view(text: str) -> str:
    """The passage as ``n: sentence`` lines, blank line between paragraphs.

    This is what a model is shown, and what the ``sentence`` field of an edit
    refers to. It is not merely an addressing scheme: measured against the same
    passages sent as plain prose, the numbering is what makes a small model
    attend sentence by sentence and propose any edits at all.
    """

    spans = sentence_spans(text)
    lines: list[str] = []
    previous_end: int | None = None
    for number, (start, end) in enumerate(spans, start=1):
        if previous_end is not None and "\n" in text[previous_end:start]:
            lines.append("")
        lines.append(f"{number}: {text[start:end]}")
        previous_end = end
    return "\n".join(lines)


def strip_label(quoted: str) -> str | None:
    """``quoted`` without a leading view label; None if it was only a label.

    Models quote the line as they saw it, label and all. The labels are this
    module's own injection into the prompt, so taking one back off restores
    what the model meant rather than guessing at it.
    """

    stripped = _LABEL_RE.sub("", quoted, count=1)
    if stripped == quoted:
        return quoted
    return stripped if stripped.strip() else None


__all__ = ["numbered_view", "sentence_spans", "strip_label"]
