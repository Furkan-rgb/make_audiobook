"""Sentence-anchored edits, and their deterministic application to source text.

A provider is asked for the *changes* it would make, never for a rewritten
passage.  A small local model retyping three thousand characters of prose will
eventually drop a clause, and nothing downstream can tell that apart from a
deliberate adaptation.  Asking only for edits removes the failure mode instead
of measuring it: prose nobody edited is byte-identical to the source because it
is literally the same string.

Edits are addressed by quoting the text they change, with a sentence number as
a disambiguating hint.  The quote is primary because it is what live runs show
a 12B model getting right; the number is what it gets wrong — reliably off by
one down a column of dialogue — so it decides only between repeated occurrences
and is otherwise advisory.  Anchoring is resolved against character spans, and
a span that would cross a paragraph break is refused outright, so paragraph
boundaries are never rebuilt from parts and therefore cannot be lost.
"""

from __future__ import annotations

from dataclasses import replace
import re

from .types import PreparationEdit
from .validation import ValidationPolicy, mask_citations, validate_edit


# Alternative one is a sentence terminator with its trailing closing quotes,
# followed by whitespace and something that can open a sentence.  Alternative
# two is a paragraph break, which ends a sentence whether or not the author
# punctuated it.  Both capture the separator itself so a span can end where the
# whitespace begins.
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
# this wrong costs nothing structurally — it only fragments the numbered view
# the model reads — but a passage numbered "[3] Dr." invites nonsense anchors.
_ABBREVIATION_RE = re.compile(
    r"(?:\b[A-Z]|\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|Rev|Hon|Gen|Col|Capt|Lt|Sgt"
    r"|vs|etc|al|cf|ca|approx|Fig|No|Vol|Ch|pp|ed|eds|trans|Inc|Ltd|Co|Corp"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec))\.$",
    re.UNICODE,
)

# Length-preserving folds only.  Offsets found in the folded text are used
# directly against the original, so a substitution that changed the character
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
_TIGHT_PUNCTUATION = ".,;:!?)]}"


def _fold(text: str) -> str:
    """Normalize characters a model is likely to retype differently, 1:1."""

    return text.translate(_FOLD)


def _collapse(text: str) -> str:
    return _WHITESPACE_RUN_RE.sub(" ", _fold(text)).strip()


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of each sentence, excluding the separators between them.

    The spans plus the gaps between them reconstitute ``text`` exactly; nothing
    here rejoins anything, which is what makes edit application lossless.
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

    This is what the model sees and what its ``sentence`` anchors refer to.
    The label is grep-style on purpose: an earlier ``[n]`` form was
    indistinguishable from the numeric reference markers the model is told to
    delete, and it dutifully proposed deleting the labels, by the hundred.
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


# The numbered view's own label, when a model quotes it as if it were text.
# Only the current "n:" form: a bracketed "[3]" really is book notation — a
# footnote marker whose deletion is a legitimate, common edit.
_LABEL_RE = re.compile(r"^\s*\d+:\s*")


def _strip_label(original: str) -> str | None:
    """``original`` without a leading view label; None if it was only a label.

    The labels are this module's own injection into the prompt, so removing
    them from a quote is restoring the model's words, not guessing at them.
    """

    stripped = _LABEL_RE.sub("", original, count=1)
    if stripped == original:
        return original
    return stripped if stripped.strip() else None


_AMBIGUOUS = "ambiguous"


def _locate(haystack: str, original: str) -> tuple[int, int] | str | None:
    """Where ``original`` sits in ``haystack``, ``_AMBIGUOUS``, or None.

    Three passes, each tolerating a little more of what a model does when it
    retypes a quotation: verbatim, then curly quotes and dashes folded to their
    plain forms, then whitespace runs collapsed. Every fold is 1:1 in length, so
    an offset found in a folded string is an offset in the real one.
    """

    folded = _fold(haystack)
    seen: set[tuple[str, str]] = set()
    for hay, needle in (
        (haystack, original),
        (folded, _fold(original)),
        (folded, _collapse(original)),
    ):
        if not needle or (hay, needle) in seen:
            continue
        seen.add((hay, needle))
        occurrences = hay.count(needle)
        if occurrences > 1:
            return _AMBIGUOUS
        if occurrences == 1:
            # The needle's own length is the span it covers: every fold is
            # 1:1, and the collapsing pass only ever matches a haystack that
            # has no whitespace runs left to account for.
            return hay.index(needle), len(needle)
    return None


def resolve_edit(
    text: str, spans: list[tuple[int, int]], edit: PreparationEdit
) -> tuple[int, int] | str:
    """The absolute span an edit replaces, or why it cannot be placed.

    The sentence number is a hint that disambiguates, not a gate. Small models
    slip a line when counting — an off-by-one down a column of dialogue is the
    common case — and refusing a quotation that occurs exactly once in the
    passage would discard a perfectly unambiguous edit over a clerical error.
    What is never guessed at is a quotation that appears more than once: there,
    the anchor is the only thing that could tell them apart, and a wrong anchor
    means the wrong occurrence gets rewritten.
    """

    if not edit.original.strip():
        return "no original text to anchor the edit to"

    index = edit.sentence - 1
    if 0 <= index < len(spans):
        start, end = spans[index]
        found = _locate(text[start:end], edit.original)
        if isinstance(found, tuple):
            offset, length = found
            return _guard_span(text, start + offset, start + offset + length)
        if found == _AMBIGUOUS:
            return "the original text appears more than once in that sentence"

    found = _locate(text, edit.original)
    if isinstance(found, tuple):
        offset, length = found
        return _guard_span(text, offset, offset + length)
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


def _guard_span(text: str, start: int, end: int) -> tuple[int, int] | str:
    """A located span, unless splicing it would rebuild passage structure.

    Normalized prose contains no newlines inside a paragraph, so any newline
    inside a located span means the match crossed a paragraph break — possible
    because the folds map newlines to spaces for matching. Structure is the one
    thing no edit may touch, however faithful its wording.
    """

    if "\n" in text[start:end]:
        return "it crosses a paragraph break, which adaptation never touches"
    return (start, end)


def _excerpt(text: str, limit: int = 60) -> str:
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else flattened[: limit - 1] + "…"


def _rejection(edit: PreparationEdit, reason: str) -> str:
    return (
        f"Dropped a {edit.category or 'unspecified'} edit to sentence "
        f"{edit.sentence} — {reason}. The original wording was kept: "
        f"“{_excerpt(edit.original)}”"
    )


def _splice(text: str, start: int, end: int, replacement: str) -> str:
    """Replace a span, tidying the whitespace a deletion leaves behind.

    Only the immediate neighbours are touched, and only for a deletion, so a
    paragraph break can never be closed up by removing a citation next to it.
    """

    before, after = text[:start], text[end:]
    if replacement.strip():
        return before + replacement + after
    if before.endswith(" ") and (
        after.startswith(" ") or after[:1] in _TIGHT_PUNCTUATION
    ):
        before = before[:-1]
    return before + after


def apply_edits(
    source: str,
    edits: list[PreparationEdit],
    *,
    policy: ValidationPolicy | None = None,
) -> tuple[str, list[PreparationEdit], list[str]]:
    """Apply what is safely applicable and report what was refused.

    Returns the prepared text, the edits actually applied, and one warning per
    refused edit.  A refusal is never fatal: the source sentence survives, and
    the reviewer is told a change was proposed and dropped, which is strictly
    more information than a rewritten passage would have given them.
    """

    policy = policy or ValidationPolicy()
    spans = sentence_spans(source)
    accepted: list[tuple[int, int, PreparationEdit]] = []
    warnings: list[str] = []
    budget = max(
        int(len(source) * policy.maximum_edited_fraction),
        policy.edited_slack_chars,
    )
    spent = 0
    label_only = 0

    for edit in edits:
        # A quote that begins with — or is nothing but — a view label is
        # interface residue, not book content. Strip it or set it aside; the
        # label-only ones are aggregated below rather than reported one by
        # one, because they say nothing about the book a reviewer must judge.
        stripped = _strip_label(edit.original)
        if stripped is None:
            label_only += 1
            continue
        if stripped != edit.original:
            edit = replace(edit, original=stripped)
        outcome = resolve_edit(source, spans, edit)
        if isinstance(outcome, str):
            warnings.append(_rejection(edit, outcome))
            continue
        start, end = outcome
        issue = validate_edit(source[start:end], edit.replacement, policy=policy)
        if issue is not None:
            warnings.append(_rejection(edit, issue))
            continue
        if any(start < other_end and other_start < end for other_start, other_end, _ in accepted):
            warnings.append(_rejection(edit, "it overlaps an earlier edit"))
            continue
        # Deleting citations is the job, not the risk: a preface can be a third
        # parentheses by weight. The budget therefore counts only prose an edit
        # would rewrite, using the same citation mask the whole-passage
        # retention check uses.
        cost = 0 if not mask_citations(source[start:end]).strip() else end - start
        if spent + cost > budget:
            warnings.append(
                _rejection(
                    edit,
                    f"the edits already rewrite {policy.maximum_edited_fraction:.0%} "
                    "of this passage, which is as much as adaptation may touch",
                )
            )
            continue
        spent += cost
        # Record the sentence the span actually landed in, not the one the
        # model claimed: an off-by-one anchor recovered by the passage-wide
        # search must not survive into the review table as a wrong number.
        landed = next(
            (
                number
                for number, (span_start, span_end) in enumerate(spans, start=1)
                if span_start <= start < span_end
            ),
            edit.sentence,
        )
        if landed != edit.sentence:
            edit = replace(edit, sentence=landed)
        accepted.append((start, end, edit))

    if label_only:
        warnings.append(
            f"Ignored {label_only} proposed edit"
            f"{'s' if label_only > 1 else ''} that targeted the numbered "
            "view's sentence labels rather than the passage text."
        )

    prepared = source
    for start, end, edit in sorted(accepted, key=lambda item: item[0], reverse=True):
        prepared = _splice(prepared, start, end, edit.replacement)
    return prepared, [edit for _start, _end, edit in accepted], warnings


__all__ = [
    "apply_edits",
    "numbered_view",
    "resolve_edit",
    "sentence_spans",
]
