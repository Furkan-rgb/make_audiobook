"""Scoring a prepared passage against the gold answer for it.

Because a provider only ever proposes edits, prepared text is the source with a
few spans spliced and nothing else touched. That is what makes exact scoring
possible: diffing the output against the source recovers the changes the model
actually caused, and each one either lands on a span the gold answer asked for
or it does not. So this is a detection problem with real true positives, false
positives, and false negatives — not a similarity score that reports 99% for
everything.

The asymmetry that matters is between the two ways of being wrong. Leaving a
citation in makes a paragraph slightly tedious to listen to. Changing a word
the author wrote makes the book say something else, and no listener can tell.
The first only costs recall; the second fails the case outright.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import statistics
from typing import Any, Sequence

from ..preparation import PreparationEdit
from ..preparation.adaptation.anchoring import collapse, resolve_edit
from ..preparation.adaptation.spans import sentence_spans, strip_label
from ..preparation.validation import ValidationPolicy, validate_edit, words_changed
from .corpus import BenchmarkCase, ExpectedEdit


# Weights for the single headline number. Recall leads because coverage is what
# a model is for; precision and exactness refine the ranking among models that
# already clear the fidelity gate, which is the thing no weighting can trade
# away — a case with a substantive false positive scores zero regardless.
RECALL_WEIGHT = 0.5
PRECISION_WEIGHT = 0.3
EXACTNESS_WEIGHT = 0.2


@dataclass(frozen=True)
class ChangeRegion:
    """A span the model actually changed, in source and in output offsets."""

    source_start: int
    source_end: int
    source_text: str
    output_text: str

    def overlaps(self, start: int, end: int) -> bool:
        if self.source_start == self.source_end:
            # A pure insertion has no width; it belongs to the span it sits in.
            return start <= self.source_start <= end
        return self.source_start < end and start < self.source_end


@dataclass
class ExpectedOutcome:
    """What became of one required change."""

    anchor: str
    category: str
    status: str  # "exact", "approximate", or "missed"
    observed: str = ""
    why: str = ""

    @property
    def found(self) -> bool:
        return self.status != "missed"


@dataclass
class UnexpectedChange:
    """A change to the passage that the gold answer did not ask for."""

    source_text: str
    output_text: str
    words_changed: int
    severity: str  # "substantive" or "cosmetic"
    trap_label: str | None = None


@dataclass
class ProtocolStats:
    """How well a model obeyed the edits-only contract, apart from taste.

    A small local model fails here in ways that look like quality failures but
    are not: it retypes a quotation inexactly and the edit is discarded, or it
    quotes the numbered view's own label. Separating these out is what tells a
    prompt problem apart from a judgement problem.
    """

    proposed: int = 0
    applied: int = 0
    unanchored: int = 0
    ambiguous: int = 0
    oversized: int = 0
    label_only: int = 0
    refused: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class CaseScore:
    """One model's answer to one case, judged."""

    case_id: str
    tier: str
    categories: list[str]
    outcomes: list[ExpectedOutcome]
    unexpected: list[UnexpectedChange]
    protocol: ProtocolStats
    recall: float
    precision: float
    exactness: float
    fidelity_pass: bool
    score: float
    passed: bool
    output_matches_gold: bool
    prepared_text: str = field(default="", repr=False)
    gold_text: str = field(default="", repr=False)
    error: str | None = None

    @property
    def substantive_false_positives(self) -> int:
        return sum(1 for item in self.unexpected if item.severity == "substantive")

    def to_dict(self) -> dict[str, Any]:
        # The two full texts are omitted: they are reproducible from the corpus
        # and the run, and including them would triple the artifact for no gain.
        payload = asdict(self)
        payload.pop("prepared_text", None)
        payload.pop("gold_text", None)
        payload["substantive_false_positives"] = self.substantive_false_positives
        return payload


def change_regions(source: str, prepared: str) -> list[ChangeRegion]:
    """Every span in which ``prepared`` differs from ``source``, atomically.

    One :class:`ChangeRegion` per non-equal opcode the differ emits, with
    nearby changes deliberately left unmerged. Correctness scoring judges each
    change against the gold spans on its own, and merging a substantive change
    into an adjacent permitted one — a citation deletion and the reworded word
    beside it — would let the second hide behind the first. Coalescing the
    regions for a readable diff, if ever wanted, is a separate presentation step.
    """

    matcher = SequenceMatcher(None, source, prepared, autojunk=False)
    return [
        ChangeRegion(
            source_start=i1,
            source_end=i2,
            source_text=source[i1:i2],
            output_text=prepared[j1:j2],
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]


def _trimmed_source_span(region: ChangeRegion) -> tuple[int, int]:
    """``region``'s source span with surrounding whitespace removed.

    The differ often sweeps an incidental space into a deletion, reporting
    " (Smith 1999)" where the anchor is "(Smith 1999)". That space sits just
    outside the expected span but carries none of the change's meaning, so it is
    trimmed before any containment test rather than being read as the change
    reaching past the span.
    """

    text = region.source_text
    lead = len(text) - len(text.lstrip())
    trail = len(text) - len(text.rstrip())
    start = region.source_start + lead
    end = region.source_end - trail
    # An all-whitespace change collapses to a point: it carries no words and is
    # cosmetic wherever it falls, so it need not be contained to be permitted.
    return (start, start) if start > end else (start, end)


def contained_in_expected(region: ChangeRegion, expected: ExpectedEdit) -> bool:
    """Whether the whole of ``region`` falls inside one expected-edit span.

    Permission is containment, not overlap. A replacement or deletion is
    expected only when its complete (whitespace-trimmed) source span lies within
    the gold span; a pure insertion, only when its point does. A change that
    reaches past the span into protected prose crosses the boundary and is not
    excused for the part of it that happens to overlap.
    """

    start, end = _trimmed_source_span(region)
    if start == end:
        # A pure insertion has no width; it belongs to the span it sits in.
        return expected.start <= start <= expected.end
    return expected.start <= start and end <= expected.end


def _project(source: str, prepared: str, start: int, end: int) -> str:
    """The text ``prepared`` holds where ``source[start:end]`` used to be.

    Offsets are mapped through the diff rather than searched for, so a span
    that was deleted outright projects to the empty string instead of to
    whatever happens to sit nearby.
    """

    opcodes = SequenceMatcher(None, source, prepared, autojunk=False).get_opcodes()
    output_start = len(prepared)
    output_end = len(prepared)
    for tag, i1, i2, j1, j2 in opcodes:
        if i1 <= start < i2 or (i1 == i2 == start):
            output_start = j1 + (start - i1) if tag == "equal" else j1
            break
    for tag, i1, i2, j1, j2 in opcodes:
        if i1 < end <= i2:
            output_end = j1 + (end - i1) if tag == "equal" else j2
            break
    return prepared[output_start:max(output_start, output_end)]


def _matches_accepted(observed: str, accept: Sequence[str]) -> bool:
    """Whether an observed replacement is one of the accepted wordings.

    Compared through the same fold the anchoring uses, so a model that types a
    straight apostrophe where the book has a curly one is not marked wrong for
    it.
    """

    folded = collapse(observed)
    return any(folded == collapse(variant) for variant in accept)


def protocol_stats(
    source: str,
    proposed: Sequence[PreparationEdit],
    applied: Sequence[PreparationEdit],
    warnings: Sequence[str],
    *,
    policy: ValidationPolicy | None = None,
) -> ProtocolStats:
    """Classify why proposed edits did not survive, without string-matching.

    Every judgement is re-derived from the same functions the applier uses, so
    this stays correct if the applier's messages are ever reworded.
    """

    policy = policy or ValidationPolicy()
    spans = sentence_spans(source)
    stats = ProtocolStats(
        proposed=len(proposed),
        applied=len(applied),
        refused=len(warnings),
        warnings=list(warnings),
    )
    for edit in proposed:
        stripped = strip_label(edit.original)
        if stripped is None:
            stats.label_only += 1
            continue
        candidate = (
            edit if stripped == edit.original else PreparationEdit(
                category=edit.category,
                original=stripped,
                replacement=edit.replacement,
                reason=edit.reason,
                sentence=edit.sentence,
            )
        )
        outcome = resolve_edit(source, spans, candidate)
        if isinstance(outcome, str):
            if "more than one" in outcome:
                stats.ambiguous += 1
            else:
                stats.unanchored += 1
            continue
        start, end = outcome
        if validate_edit(source[start:end], candidate.replacement, policy=policy):
            stats.oversized += 1
    return stats


def _trap_label(case: BenchmarkCase, region: ChangeRegion) -> str | None:
    for trap in case.traps:
        if region.overlaps(trap.start, trap.end):
            return trap.label
    return None


def _classify(case: BenchmarkCase, region: ChangeRegion) -> UnexpectedChange:
    changed = words_changed(region.source_text, region.output_text)
    cosmetic = changed == 0 or collapse(region.source_text) == collapse(
        region.output_text
    )
    return UnexpectedChange(
        source_text=region.source_text,
        output_text=region.output_text,
        words_changed=changed,
        severity="cosmetic" if cosmetic else "substantive",
        trap_label=_trap_label(case, region),
    )


def _outcome(
    case: BenchmarkCase,
    expected: ExpectedEdit,
    prepared: str,
    regions: Sequence[ChangeRegion],
) -> ExpectedOutcome:
    touched = any(region.overlaps(expected.start, expected.end) for region in regions)
    if not touched:
        return ExpectedOutcome(
            anchor=expected.anchor,
            category=expected.category,
            status="missed",
            observed=expected.anchor,
            why=expected.why,
        )
    observed = _project(case.source, prepared, expected.start, expected.end)
    return ExpectedOutcome(
        anchor=expected.anchor,
        category=expected.category,
        status="exact" if _matches_accepted(observed, expected.accept) else "approximate",
        observed=observed,
        why=expected.why,
    )


def score_case(
    case: BenchmarkCase,
    prepared: str,
    *,
    proposed: Sequence[PreparationEdit] = (),
    applied: Sequence[PreparationEdit] = (),
    warnings: Sequence[str] = (),
    error: str | None = None,
) -> CaseScore:
    """Judge one prepared passage against the case's gold answer."""

    stats = protocol_stats(case.source, proposed, applied, warnings)
    if error is not None:
        return CaseScore(
            case_id=case.id,
            tier=case.tier,
            categories=list(case.categories),
            outcomes=[],
            unexpected=[],
            protocol=stats,
            recall=0.0,
            precision=0.0,
            exactness=0.0,
            fidelity_pass=False,
            score=0.0,
            passed=False,
            output_matches_gold=False,
            gold_text=case.prepared,
            error=error,
        )

    regions = change_regions(case.source, prepared)
    outcomes = [_outcome(case, item, prepared, regions) for item in case.expect]
    unexpected = [
        _classify(case, region)
        for region in regions
        if not any(
            contained_in_expected(region, item) for item in case.expect
        )
    ]

    found = sum(1 for item in outcomes if item.found)
    exact = sum(1 for item in outcomes if item.status == "exact")
    # A case with nothing to do is answered perfectly by doing nothing, so its
    # recall and exactness are one by definition. Only precision can be lost.
    recall = found / len(outcomes) if outcomes else 1.0
    exactness = exact / len(outcomes) if outcomes else 1.0
    precision = (
        found / (found + len(unexpected)) if (found + len(unexpected)) else 1.0
    )
    fidelity_pass = not any(item.severity == "substantive" for item in unexpected)
    score = (
        RECALL_WEIGHT * recall
        + PRECISION_WEIGHT * precision
        + EXACTNESS_WEIGHT * exactness
    ) if fidelity_pass else 0.0

    return CaseScore(
        case_id=case.id,
        tier=case.tier,
        categories=list(case.categories),
        outcomes=outcomes,
        unexpected=unexpected,
        protocol=stats,
        recall=recall,
        precision=precision,
        exactness=exactness,
        fidelity_pass=fidelity_pass,
        score=score,
        passed=fidelity_pass and recall == 1.0 and exactness == 1.0 and not unexpected,
        output_matches_gold=prepared.strip() == case.prepared,
        prepared_text=prepared,
        gold_text=case.prepared,
    )


@dataclass
class Breakdown:
    """Aggregate scores for one slice of the corpus."""

    label: str
    cases: int
    passed: int
    score: float
    recall: float
    precision: float
    exactness: float
    fidelity_failures: int


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(label: str, scores: Sequence[CaseScore]) -> Breakdown:
    return Breakdown(
        label=label,
        cases=len(scores),
        passed=sum(1 for item in scores if item.passed),
        score=_mean([item.score for item in scores]),
        recall=_mean([item.recall for item in scores]),
        precision=_mean([item.precision for item in scores]),
        exactness=_mean([item.exactness for item in scores]),
        fidelity_failures=sum(1 for item in scores if not item.fidelity_pass),
    )


def breakdowns(scores: Sequence[CaseScore], keys: str) -> list[Breakdown]:
    """Per-tier or per-category aggregates, in a stable order."""

    grouped: dict[str, list[CaseScore]] = {}
    for item in scores:
        labels = [item.tier] if keys == "tier" else item.categories
        for label in labels:
            grouped.setdefault(label, []).append(item)
    return [summarize(label, grouped[label]) for label in sorted(grouped)]


def edit_signature(edits: Sequence[PreparationEdit]) -> frozenset[tuple[str, str]]:
    """A repetition-comparable identity for what a model proposed.

    Compared as edits rather than as text because text consistency is trivially
    perfect for a model that proposes nothing, which is exactly the failure a
    determinism figure should not reward.
    """

    return frozenset(
        (collapse(edit.original), collapse(edit.replacement)) for edit in edits
    )


def determinism(signatures: Sequence[frozenset[tuple[str, str]]]) -> float | None:
    """Mean pairwise agreement between repetitions of the same case."""

    if len(signatures) < 2:
        return None
    agreements: list[float] = []
    for index, first in enumerate(signatures):
        for second in signatures[index + 1 :]:
            union = first | second
            agreements.append(
                len(first & second) / len(union) if union else 1.0
            )
    return _mean(agreements)


def trap_failures(scores: Sequence[CaseScore]) -> list[tuple[str, int]]:
    """Which named traps a model fell into, most frequent first.

    Counted once per case run, not once per changed region: an edit that
    strips both quotation marks off a line disturbs the trap span in two
    places, and reporting that as two failures would overstate a single
    lapse of judgement.
    """

    counter: Counter[str] = Counter()
    for item in scores:
        for label in {
            change.trap_label for change in item.unexpected if change.trap_label
        }:
            counter[label] += 1
    return counter.most_common()


__all__ = [
    "EXACTNESS_WEIGHT",
    "PRECISION_WEIGHT",
    "RECALL_WEIGHT",
    "Breakdown",
    "CaseScore",
    "ChangeRegion",
    "ExpectedOutcome",
    "ProtocolStats",
    "UnexpectedChange",
    "breakdowns",
    "change_regions",
    "contained_in_expected",
    "determinism",
    "edit_signature",
    "protocol_stats",
    "score_case",
    "summarize",
    "trap_failures",
]
