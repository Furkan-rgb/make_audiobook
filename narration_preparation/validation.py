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

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_lexical_retention <= 1.0:
            raise ValueError("minimum_lexical_retention must be between 0 and 1")
        if self.maximum_expansion_ratio < 1.0:
            raise ValueError("maximum_expansion_ratio must be at least 1")
        if self.expansion_slack_chars < 0:
            raise ValueError("expansion_slack_chars cannot be negative")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "minimum_lexical_retention": self.minimum_lexical_retention,
            "maximum_expansion_ratio": self.maximum_expansion_ratio,
            "expansion_slack_chars": self.expansion_slack_chars,
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
