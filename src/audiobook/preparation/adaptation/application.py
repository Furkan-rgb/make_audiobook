"""Turning a list of proposed edits into prepared text.

A provider is asked for the *changes* it would make, never for a rewritten
passage. A small local model retyping three thousand characters of prose will
eventually drop a clause, and nothing downstream can tell that apart from a
deliberate adaptation. Asking only for edits removes the failure mode instead
of measuring it: prose nobody edited is byte-identical to the source because it
is literally the same string.

Refusing an edit is never fatal. The source survives, the reviewer is told what
was proposed and why it was dropped, and the run continues — a book is a
hundred passages, and one over-eager title page must not end it.
"""

from __future__ import annotations

from dataclasses import replace
import re

from ..types import PreparationEdit
from ..validation import (
    ValidationPolicy,
    lexical_retention,
    validate_edit,
    words_dropped,
)
from .anchoring import empties_paragraph, resolve_edit
from .spans import sentence_spans, strip_label


_WHITESPACE_RUN = re.compile(r"\s+")
_TIGHT_PUNCTUATION = ".,;:!?)]}"


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
    """Replace a span, tidying the whitespace the change leaves at its seams.

    Only the immediate neighbours are touched, so a paragraph break can never
    be closed up by removing a citation next to it.
    """

    before, after = text[:start], text[end:]
    if replacement.strip():
        # Normalized prose is single-spaced and newline-free within a
        # paragraph; a replacement carrying its own padding must not import
        # spacing the source does not use.
        cleaned = _WHITESPACE_RUN.sub(" ", replacement)
        if cleaned.startswith(" ") and (not before or before[-1].isspace()):
            cleaned = cleaned.lstrip()
        if cleaned.endswith(" ") and (not after or after[0].isspace()):
            cleaned = cleaned.rstrip()
        return before + cleaned + after
    if before.endswith(" ") and (
        not after or after.startswith(" ") or after[0] in _TIGHT_PUNCTUATION
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
    refusal.
    """

    policy = policy or ValidationPolicy()
    spans = sentence_spans(source)
    accepted: list[tuple[int, int, PreparationEdit]] = []
    warnings: list[str] = []
    label_only = 0

    for edit in edits:
        # A quote that begins with — or is nothing but — a view label is
        # interface residue, not book content. Strip it or set it aside; the
        # label-only ones are counted rather than reported one by one, because
        # they say nothing about the book a reviewer has to judge.
        stripped = strip_label(edit.original)
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
        if empties_paragraph(source, start, end, edit.replacement):
            warnings.append(_rejection(edit, "it would leave its paragraph empty"))
            continue
        if any(
            start < other_end and other_start < end
            for other_start, other_end, _ in accepted
        ):
            warnings.append(_rejection(edit, "it overlaps an earlier edit"))
            continue

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
            f"Ignored {label_only} proposed edit{'s' if label_only > 1 else ''} "
            "that targeted the numbered view's sentence labels rather than the "
            "passage text."
        )

    def splice_all(items: list[tuple[int, int, PreparationEdit]]) -> str:
        text = source
        for start, end, edit in sorted(items, key=lambda item: item[0], reverse=True):
            text = _splice(text, start, end, edit.replacement)
        return text

    # Each edit is individually legal, yet together they can still strip a
    # passage past the point where it is the author's text. Back off the
    # heaviest cutter until what remains clears the floor: dropping edits
    # always converges, because the untouched source retains everything.
    prepared = splice_all(accepted)
    while (
        accepted
        and lexical_retention(source, prepared) < policy.minimum_lexical_retention
    ):
        victim = max(
            accepted,
            key=lambda item: words_dropped(source[item[0] : item[1]], item[2].replacement),
        )
        accepted.remove(victim)
        warnings.append(
            _rejection(
                victim[2],
                "together the edits cut more of the passage than adaptation "
                f"allows, so the largest were undone to stay above "
                f"{policy.minimum_lexical_retention:.0%} of the author's words",
            )
        )
        prepared = splice_all(accepted)

    return prepared, [edit for _start, _end, edit in accepted], warnings


__all__ = ["apply_edits"]
