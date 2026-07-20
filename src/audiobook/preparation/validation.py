"""Safety validation for model-prepared narration prose."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata

from .types import PreparationRequest, PreparationResult


_PAREN_RE = re.compile(r"\([^()]*\)")
_NUMERIC_REFERENCE_RE = re.compile(r"\[(?:\s*\d+[a-z]?(?:\s*[-,;]\s*\d+[a-z]?)*\s*)\]")
_YEAR_RE = re.compile(r"\b(?:1[5-9]|20)\d{2}[a-z]?\b", re.IGNORECASE)
_AUTHOR_MARKER_RE = re.compile(r"\bet\s+al\.?\b|\b[A-Z][A-Za-z'’.-]+(?:\s+and\s+[A-Z])?", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SUMMARY_LEAD_RE = re.compile(
    r"^\s*(?:in summary|to summarize|summary:|this (?:passage|section|text) "
    r"(?:describes|discusses|explains|is about|summarizes))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationPolicy:
    """Thresholds for conservative, meaning-preserving adaptation."""

    minimum_lexical_retention: float = 0.72
    maximum_expansion_ratio: float = 1.60
    expansion_slack_chars: int = 240
    # Per-edit limits. An adaptation is a citation removed or a symbol spoken
    # aloud; an edit that swallows a whole sentence is a paraphrase wearing an
    # edit's clothes, and these are what tell the two apart.
    maximum_edit_span_chars: int = 400
    maximum_replacement_ratio: float = 3.0
    replacement_slack_chars: int = 40
    # Past this length an edit must keep the words it touches. A short span may
    # legitimately say things differently — "§5" becomes "section five" — but a
    # long one that does is a paraphrase, which is the failure this whole
    # design exists to prevent. Citation-shaped text is masked out first, so
    # deleting a whole bibliographic list still costs nothing.
    verbatim_edit_chars: int = 60
    minimum_edit_retention: float = 0.9
    # Deletion is the one edit the retention check cannot see — there is no
    # replacement to compare — and silently dropping a claim wholesale is the
    # worst failure this stage can produce. Citation-shaped text is masked out
    # first, so removing a long bibliographic list still costs nothing.
    maximum_prose_deletion_chars: int = 120
    maximum_edited_fraction: float = 0.25
    # A fraction alone would forbid a single legitimate repair in a two-line
    # passage. Below this many characters the per-edit limits are the binding
    # constraint, as they are for the expansion check above.
    edited_slack_chars: int = 200

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_lexical_retention <= 1.0:
            raise ValueError("minimum_lexical_retention must be between 0 and 1")
        if self.maximum_expansion_ratio < 1.0:
            raise ValueError("maximum_expansion_ratio must be at least 1")
        if self.expansion_slack_chars < 0:
            raise ValueError("expansion_slack_chars cannot be negative")
        if self.maximum_edit_span_chars <= 0:
            raise ValueError("maximum_edit_span_chars must be positive")
        if self.maximum_replacement_ratio < 1.0:
            raise ValueError("maximum_replacement_ratio must be at least 1")
        if self.replacement_slack_chars < 0:
            raise ValueError("replacement_slack_chars cannot be negative")
        if not 0.0 < self.maximum_edited_fraction <= 1.0:
            raise ValueError("maximum_edited_fraction must be within (0, 1]")
        if self.edited_slack_chars < 0:
            raise ValueError("edited_slack_chars cannot be negative")
        if self.verbatim_edit_chars < 0:
            raise ValueError("verbatim_edit_chars cannot be negative")
        if not 0.0 <= self.minimum_edit_retention <= 1.0:
            raise ValueError("minimum_edit_retention must be between 0 and 1")
        if self.maximum_prose_deletion_chars <= 0:
            raise ValueError("maximum_prose_deletion_chars must be positive")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_lexical_retention": self.minimum_lexical_retention,
            "maximum_expansion_ratio": self.maximum_expansion_ratio,
            "expansion_slack_chars": self.expansion_slack_chars,
            "maximum_edit_span_chars": self.maximum_edit_span_chars,
            "maximum_replacement_ratio": self.maximum_replacement_ratio,
            "replacement_slack_chars": self.replacement_slack_chars,
            "maximum_edited_fraction": self.maximum_edited_fraction,
            "edited_slack_chars": self.edited_slack_chars,
            "verbatim_edit_chars": self.verbatim_edit_chars,
            "minimum_edit_retention": self.minimum_edit_retention,
            "maximum_prose_deletion_chars": self.maximum_prose_deletion_chars,
        }


@dataclass(frozen=True)
class ValidationReport:
    lexical_retention: float
    expansion_ratio: float
    source_token_count: int
    prepared_token_count: int


class PreparationValidationError(ValueError):
    """Raised when a provider response is unsafe to narrate."""

    def __init__(self, issues: list[str], report: ValidationReport | None = None):
        self.issues = issues
        self.report = report
        super().__init__("Narration preparation rejected: " + "; ".join(issues))


def mask_citations(text: str) -> str:
    """Remove citation-shaped spans for fair lexical-retention measurement.

    This intentionally affects validation only. It is not used to rewrite the
    source and therefore cannot accidentally delete a meaningful parenthesis.
    """

    text = _NUMERIC_REFERENCE_RE.sub(" ", text)

    def replace_parenthesis(match: re.Match[str]) -> str:
        content = match.group(0)
        has_year = bool(_YEAR_RE.search(content))
        citation_list = ";" in content or content.count(",") >= 1
        has_author = bool(_AUTHOR_MARKER_RE.search(content))
        return " " if has_year and (has_author or citation_list) else content

    return _PAREN_RE.sub(replace_parenthesis, text)


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", mask_citations(text)).casefold()
    return _TOKEN_RE.findall(normalized)


def _lexical_retention(source: list[str], prepared: list[str]) -> float:
    if not source:
        return 1.0
    source_counts = Counter(source)
    prepared_counts = Counter(prepared)
    retained = sum(
        min(count, prepared_counts[token]) for token, count in source_counts.items()
    )
    return retained / sum(source_counts.values())


def validate_edit(
    original: str,
    replacement: str,
    *,
    policy: ValidationPolicy | None = None,
) -> str | None:
    """Why one resolved edit is too large to be an adaptation, or None.

    The whole-passage checks below can only notice damage after it has been
    done. This runs before anything is applied, on a span whose boundaries are
    already known, which is the only point at which a bad edit is free to
    refuse.
    """

    policy = policy or ValidationPolicy()
    if len(original) > policy.maximum_edit_span_chars:
        return (
            f"it rewrites {len(original)} characters at once, more than the "
            f"{policy.maximum_edit_span_chars} an adaptation may touch in one edit"
        )
    limit = max(
        len(original) * policy.maximum_replacement_ratio,
        len(original) + policy.replacement_slack_chars,
    )
    if len(replacement) > limit:
        return (
            f"its replacement is {len(replacement)} characters for "
            f"{len(original)} of source, which adds rather than adapts"
        )
    if _SUMMARY_LEAD_RE.search(replacement):
        return "its replacement opens with summary-style framing"
    prose = mask_citations(original).strip()
    if not replacement.strip():
        if len(prose) > policy.maximum_prose_deletion_chars:
            return (
                f"it deletes {len(prose)} characters of prose outright, more "
                f"than the {policy.maximum_prose_deletion_chars} an adaptation "
                "may drop in one edit"
            )
    elif len(prose) > policy.verbatim_edit_chars:
        retention = _lexical_retention(_tokens(prose), _tokens(replacement))
        if retention < policy.minimum_edit_retention:
            return (
                f"it rewrites a long span but keeps only {retention:.0%} of its "
                "words, which is paraphrase rather than adaptation"
            )
    return None


def validate_preparation(
    request: PreparationRequest,
    result: PreparationResult,
    *,
    policy: ValidationPolicy | None = None,
) -> ValidationReport:
    """Reject blank, inflated, or summary-like provider output."""

    policy = policy or ValidationPolicy()
    prepared = result.prepared_text.strip()
    issues: list[str] = []
    if not prepared:
        raise PreparationValidationError(["prepared text is blank"])

    source_masked = mask_citations(request.source_text).strip()
    prepared_masked = mask_citations(prepared).strip()
    source_tokens = _tokens(request.source_text)
    prepared_tokens = _tokens(prepared)
    retention = _lexical_retention(source_tokens, prepared_tokens)
    expansion = len(prepared_masked) / max(1, len(source_masked))
    report = ValidationReport(
        lexical_retention=retention,
        expansion_ratio=expansion,
        source_token_count=len(source_tokens),
        prepared_token_count=len(prepared_tokens),
    )

    expansion_limit = max(
        len(source_masked) * policy.maximum_expansion_ratio,
        len(source_masked) + policy.expansion_slack_chars,
    )
    if len(prepared_masked) > expansion_limit:
        issues.append(
            f"extreme expansion ({expansion:.2f}x; allowed "
            f"{policy.maximum_expansion_ratio:.2f}x plus short-text slack)"
        )
    if retention < policy.minimum_lexical_retention:
        issues.append(
            f"lexical retention {retention:.1%} is below "
            f"{policy.minimum_lexical_retention:.1%}; output may be a summary"
        )
    if _SUMMARY_LEAD_RE.search(prepared) and not _SUMMARY_LEAD_RE.search(
        request.source_text
    ):
        issues.append("output adds summary-style framing")
    if issues:
        raise PreparationValidationError(issues, report)
    return report
