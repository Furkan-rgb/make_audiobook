"""Deciding which characters of the source an edit refers to.

An edit arrives as a quotation plus a sentence number. The quotation is what a
model gets right and the number is what it gets wrong — reliably off by one
down a column of dialogue — so the quotation is primary here and the number
only breaks ties. What is never guessed at is a quotation that occurs more than
once: there the number is the only thing that could tell the occurrences apart,
and a wrong number would rewrite the wrong one.
"""

from __future__ import annotations

import re

from ..types import PreparationEdit


# Length-preserving folds only. Offsets found in the folded text are used
# directly against the real text, so a substitution that changed the character
# count would corrupt every span after it.
_FOLD = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "–": "-",
        "—": "-",
        "‐": "-",
        "‑": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2007": " ",
        "\n": " ",
        "\t": " ",
    }
)
_WHITESPACE_RUN_RE = re.compile(r"\s+")
_AMBIGUOUS = "ambiguous"


def fold(text: str) -> str:
    """Normalize characters a model is likely to retype differently, 1:1."""

    return text.translate(_FOLD)


def collapse(text: str) -> str:
    """Fold, then squeeze whitespace runs — for needles only, never haystacks."""

    return _WHITESPACE_RUN_RE.sub(" ", fold(text)).strip()


def _locate(haystack: str, quoted: str) -> tuple[int, int] | str | None:
    """Where ``quoted`` sits in ``haystack``, ``_AMBIGUOUS``, or None.

    Three passes, each tolerating a little more of what a model does when it
    retypes a quotation: verbatim, then curly quotes and dashes folded to their
    plain forms, then whitespace runs collapsed.
    """

    folded = fold(haystack)
    seen: set[tuple[str, str]] = set()
    for hay, needle in (
        (haystack, quoted),
        (folded, fold(quoted)),
        (folded, collapse(quoted)),
    ):
        if not needle or (hay, needle) in seen:
            continue
        seen.add((hay, needle))
        occurrences = hay.count(needle)
        if occurrences > 1:
            return _AMBIGUOUS
        if occurrences == 1:
            # The needle's own length is the span it covers: every fold is
            # 1:1, and the collapsing pass only ever matches a haystack with
            # no whitespace runs left to account for.
            return hay.index(needle), len(needle)
    return None


def resolve_edit(
    text: str, spans: list[tuple[int, int]], edit: PreparationEdit
) -> tuple[int, int] | str:
    """The span an edit replaces, or a reason it cannot be placed."""

    if not edit.original.strip():
        return "no original text to anchor the edit to"

    index = edit.sentence - 1
    if 0 <= index < len(spans):
        start, end = spans[index]
        found = _locate(text[start:end], edit.original)
        if isinstance(found, tuple):
            offset, length = found
            return _guard(text, start + offset, start + offset + length)
        if found == _AMBIGUOUS:
            return "the original text appears more than once in that sentence"

    found = _locate(text, edit.original)
    if isinstance(found, tuple):
        offset, length = found
        return _guard(text, offset, offset + length)
    if found == _AMBIGUOUS:
        return (
            f"it is not in sentence {edit.sentence}, and the same wording "
            "appears in more than one other sentence"
        )
    if not 0 <= index < len(spans):
        return (
            f"sentence {edit.sentence} does not exist (the passage has "
            f"{len(spans)})"
        )
    return "the original text is nowhere in the passage"


def _guard(text: str, start: int, end: int) -> tuple[int, int] | str:
    """A located span, unless splicing it would rebuild passage structure.

    Normalized prose holds no newline inside a paragraph, so a newline inside a
    located span means the match crossed a paragraph break — possible because
    the folds map newlines to spaces for matching. Structure is the one thing
    no edit may touch, however faithful its wording.
    """

    if "\n" in text[start:end]:
        return "it crosses a paragraph break, which adaptation never touches"
    return (start, end)


def empties_paragraph(text: str, start: int, end: int, replacement: str) -> bool:
    """Whether applying this edit would leave its paragraph with no words.

    A paragraph reduced to nothing does not disappear — its separators stay,
    stacking blank lines the author never wrote.
    """

    left = text.rfind("\n\n", 0, start)
    paragraph_start = 0 if left < 0 else left + 2
    right = text.find("\n\n", end)
    paragraph_end = len(text) if right < 0 else right
    remaining = text[paragraph_start:start] + replacement + text[end:paragraph_end]
    return not remaining.strip()


__all__ = ["collapse", "empties_paragraph", "fold", "resolve_edit"]
