"""What a benchmark run produced, as data and as something a person reads.

The old report answered "how much did each model change the text?", which
every model answers identically. This one answers "what did each model get
wrong?", so it leads with a leaderboard and ends with an appendix that shows,
case by case, the exact diff between the gold answer and what a model did.
The appendix is the part that changes anyone's mind about a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import unified_diff
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

from ..preparation import DEFAULT_PROMPT_VERSION, SAMPLING_OPTIONS, PreparationEdit
from .scoring import Breakdown, CaseScore


BENCHMARK_SCHEMA_VERSION = 2


@dataclass
class CaseRun:
    """One model's attempt at one case, once."""

    case_id: str
    model: str
    repetition: int
    seconds: float
    score: CaseScore
    # The seed this run generated under. Recorded per run because it advances
    # with the repetition, so a reader can reproduce any single attempt.
    seed: int | None = None
    # Kept verbatim because the scored result cannot explain a model that
    # proposed the right change and merely mis-anchored it.
    proposed: list[PreparationEdit] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "repetition": self.repetition,
            "seed": self.seed,
            "seconds": round(self.seconds, 3),
            "proposed_edits": [edit.to_dict() for edit in self.proposed],
            **self.score.to_dict(),
        }


@dataclass
class ProtocolTotals:
    """Contract compliance summed over every case a model attempted."""

    proposed: int = 0
    applied: int = 0
    unanchored: int = 0
    ambiguous: int = 0
    oversized: int = 0
    label_only: int = 0
    refused: int = 0

    @property
    def anchor_failure_rate(self) -> float:
        return (self.unanchored + self.ambiguous) / self.proposed if self.proposed else 0.0


@dataclass
class ModelReport:
    """Everything the benchmark learned about one model."""

    model: str
    overall: Breakdown
    by_tier: list[Breakdown]
    by_category: list[Breakdown]
    protocol: ProtocolTotals
    trap_failures: list[tuple[str, int]]
    determinism: float | None
    mean_seconds: float
    total_seconds: float
    errored_cases: int
    # The provider options this variant actually ran under — the explicit
    # budgets and think mode, plus whichever sampling options were sent (none,
    # under native sampling). Read from the provider's own metadata.
    provider_options: dict[str, Any] = field(default_factory=dict)
    runs: list[CaseRun] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "overall": asdict(self.overall),
            "by_tier": [asdict(item) for item in self.by_tier],
            "by_category": [asdict(item) for item in self.by_category],
            "protocol": {
                **asdict(self.protocol),
                "anchor_failure_rate": self.protocol.anchor_failure_rate,
            },
            "trap_failures": [
                {"label": label, "count": count} for label, count in self.trap_failures
            ],
            "determinism": self.determinism,
            "mean_seconds": round(self.mean_seconds, 3),
            "total_seconds": round(self.total_seconds, 3),
            "errored_cases": self.errored_cases,
            "provider_options": self.provider_options,
        }


@dataclass
class BenchmarkReport:
    created_at: str
    provider_name: str
    models: list[str]
    repetitions: int
    corpus_size: int
    corpus_tiers: dict[str, int]
    models_reports: list[ModelReport]
    runs: list[CaseRun]
    json_path: Path
    markdown_path: Path
    # Sampling policy of the run. The benchmark has one: model-native defaults,
    # so every sampling option is omitted and the seed sequence (one per
    # repetition) is the only generation control the benchmark still pins.
    native_sampling: bool = True
    seeds: list[int] = field(default_factory=list)
    prompt_version: str = DEFAULT_PROMPT_VERSION
    schema_version: int = BENCHMARK_SCHEMA_VERSION
    # Filled in by the runner after the markdown and JSON are on disk; the plots
    # are a convenience view of them, so a run without plots is still complete.
    plot_paths: list[Path] = field(default_factory=list)

    @property
    def supplied_sampling(self) -> list[str]:
        """Sampling options any model actually sent, in canonical order.

        Empty under native sampling. Derived from the providers' own metadata
        rather than assumed, so the report never claims an option was omitted
        that a run in fact supplied.
        """

        sent = {
            key
            for item in self.models_reports
            for key in item.provider_options
            if key in SAMPLING_OPTIONS
        }
        return [name for name in SAMPLING_OPTIONS if name in sent]

    @property
    def omitted_sampling(self) -> list[str]:
        """Sampling options left for the model package to supply."""

        sent = set(self.supplied_sampling)
        return [name for name in SAMPLING_OPTIONS if name not in sent]

    @property
    def ranked(self) -> list[ModelReport]:
        """Best first: fidelity is the primary key, the score only breaks ties.

        Ranking on the composite alone would let a model buy back a fidelity
        failure with volume, and a model that changes the author's words is not
        redeemable by covering more citations.
        """

        return sorted(
            self.models_reports,
            key=lambda item: (
                item.overall.fidelity_failures,
                -item.overall.score,
                -item.overall.passed,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "configuration": {
                "provider": self.provider_name,
                "models": self.models,
                "repetitions": self.repetitions,
                "corpus_size": self.corpus_size,
                "corpus_tiers": self.corpus_tiers,
                "cache_reuse": False,
                "prompt_version": self.prompt_version,
                "sampling": {
                    # The benchmark's one behaviour: each model generates under
                    # its package's own sampling defaults. `supplied` lists what
                    # the benchmark still sends explicitly; `omitted` lists the
                    # sampling options dropped so the model package provides them.
                    "native_sampling": self.native_sampling,
                    "seeds": self.seeds,
                    "supplied": self.supplied_sampling,
                    "omitted": self.omitted_sampling,
                },
            },
            "models": [item.to_dict() for item in self.models_reports],
            "runs": [run.to_dict() for run in self.runs],
        }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _escape(model: str) -> str:
    return model.replace("|", "\\|")


def _leaderboard(report: BenchmarkReport) -> list[str]:
    lines = [
        "| Model | Score | Cases passed | Fidelity failures | Recall | Precision "
        "| Exactness | Determinism | Mean s/case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.ranked:
        overall = item.overall
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_escape(item.model)}`",
                    f"{overall.score:.3f}",
                    f"{overall.passed}/{overall.cases}",
                    str(overall.fidelity_failures),
                    _percent(overall.recall),
                    _percent(overall.precision),
                    _percent(overall.exactness),
                    _percent(item.determinism),
                    f"{item.mean_seconds:.2f}",
                ]
            )
            + " |"
        )
    return lines


def _matrix(report: BenchmarkReport, attribute: str, heading: str) -> list[str]:
    """One row per model, one column per tier or category."""

    labels: list[str] = []
    for item in report.models_reports:
        for breakdown in getattr(item, attribute):
            if breakdown.label not in labels:
                labels.append(breakdown.label)
    if not labels:
        return []
    labels.sort()

    lines = [
        "",
        f"## {heading}",
        "",
        "Mean case score, and passed/total in parentheses.",
        "",
        "| Model | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for item in report.ranked:
        cells = []
        by_label = {
            breakdown.label: breakdown for breakdown in getattr(item, attribute)
        }
        for label in labels:
            breakdown = by_label.get(label)
            cells.append(
                "—"
                if breakdown is None
                else f"{breakdown.score:.2f} ({breakdown.passed}/{breakdown.cases})"
            )
        lines.append(f"| `{_escape(item.model)}` | " + " | ".join(cells) + " |")
    return lines


def _protocol_table(report: BenchmarkReport) -> list[str]:
    lines = [
        "",
        "## Contract compliance",
        "",
        "Edits a model proposed that the applier could not use. These are prompt "
        "and formatting failures rather than judgement failures, and they are "
        "worth separating: a model that anchors badly may still have good taste.",
        "",
        "| Model | Proposed | Applied | Unanchorable | Ambiguous | Oversized "
        "| Label-only | Anchor failure rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.ranked:
        protocol = item.protocol
        lines.append(
            f"| `{_escape(item.model)}` | {protocol.proposed} | {protocol.applied} "
            f"| {protocol.unanchored} | {protocol.ambiguous} | {protocol.oversized} "
            f"| {protocol.label_only} | {_percent(protocol.anchor_failure_rate)} |"
        )
    return lines


def _trap_table(report: BenchmarkReport) -> list[str]:
    if not any(item.trap_failures for item in report.models_reports):
        return []
    lines = [
        "",
        "## Traps sprung",
        "",
        "Each entry is a model changing something the corpus marked as text that "
        "must survive. Every one of these is a passage where the audiobook would "
        "no longer say what the author wrote.",
        "",
        "| Model | Trap | Times |",
        "|---|---|---:|",
    ]
    for item in report.ranked:
        for label, count in item.trap_failures:
            lines.append(f"| `{_escape(item.model)}` | `{label}` | {count} |")
    return lines


def _failure_appendix(report: BenchmarkReport, limit: int) -> list[str]:
    lines = [
        "",
        "## Failure appendix",
        "",
        "Every case a model did not pass, worst first. `-` is the gold answer, "
        "`+` is what the model produced.",
    ]
    for item in report.ranked:
        failures = [
            run
            for run in item.runs
            if not run.score.passed and run.repetition == 1
        ]
        failures.sort(key=lambda run: (run.score.fidelity_pass, run.score.score))
        lines.extend(["", f"### `{item.model}`", ""])
        if not failures:
            lines.append("Passed every case.")
            continue
        shown = failures[:limit]
        for run in shown:
            score = run.score
            lines.extend(
                [
                    f"#### {score.case_id} ({score.tier})",
                    "",
                    f"- Score {score.score:.2f}"
                    f"{'' if score.fidelity_pass else ' — **fidelity failure**'}",
                ]
            )
            if score.error:
                lines.extend([f"- Error: `{score.error}`", ""])
                continue
            for outcome in score.outcomes:
                if outcome.status == "missed":
                    lines.append(f"- Missed: `{outcome.anchor}` — {outcome.why}")
                elif outcome.status == "approximate":
                    lines.append(
                        f"- Wrong wording for `{outcome.anchor}`: produced "
                        f"`{outcome.observed}`"
                    )
            for change in score.unexpected:
                label = f" (trap `{change.trap_label}`)" if change.trap_label else ""
                lines.append(
                    f"- Unrequested {change.severity} change{label}: "
                    f"`{change.source_text}` → `{change.output_text}`"
                )
            diff = list(
                unified_diff(
                    score.gold_text.splitlines(),
                    score.prepared_text.splitlines(),
                    fromfile="gold",
                    tofile=item.model,
                    lineterm="",
                    n=1,
                )
            )
            if diff:
                lines.extend(["", "```diff", *diff, "```"])
            lines.append("")
        if len(failures) > len(shown):
            lines.append(
                f"_{len(failures) - len(shown)} further failing cases omitted; "
                "the full list is in `benchmark.json`._"
            )
    return lines


def render_markdown(report: BenchmarkReport, *, appendix_limit: int = 8) -> str:
    tiers = ", ".join(
        f"{count} {label}" for label, count in sorted(report.corpus_tiers.items())
    )
    seeds = ", ".join(str(seed) for seed in report.seeds) or "—"
    omitted = ", ".join(report.omitted_sampling) or "none"
    supplied = ", ".join(report.supplied_sampling) or "none"
    sampling_line = (
        "- Sampling: model-native defaults — the benchmark omits "
        f"{omitted} so each model package's own values are used"
        if report.native_sampling
        else f"- Sampling: benchmark-supplied ({supplied})"
    )
    lines = [
        "# Narration preparation benchmark",
        "",
        f"- Provider: `{report.provider_name}`",
        f"- Corpus: {report.corpus_size} cases ({tiers})",
        f"- Repetitions per model: {report.repetitions}",
        "- Preparation-cache reuse: disabled",
        f"- Prompt version: `{report.prompt_version}`",
        sampling_line,
        f"- Seeds (one per repetition): {seeds}",
        f"- Explicitly controlled options: seed, num_ctx, num_predict, think",
        f"- Run at: {report.created_at}",
        "",
        "## Leaderboard",
        "",
        "Ranked by fidelity failures first, then by score. A case scores zero if "
        "the model changed anything the gold answer did not ask it to change. "
        "Counts are over every case run, so a two-repetition benchmark of a "
        f"{report.corpus_size}-case corpus totals "
        f"{report.corpus_size * report.repetitions}.",
        "",
        *_leaderboard(report),
        *_matrix(report, "by_tier", "By tier"),
        *_matrix(report, "by_category", "By category"),
        *_protocol_table(report),
        *_trap_table(report),
        *_failure_appendix(report, appendix_limit),
        "",
        "## How to read this",
        "",
        "**Score** is `0.5 x recall + 0.3 x precision + 0.2 x exactness`, and it "
        "is zero for any case with a substantive unrequested change. **Recall** is "
        "the share of required edits the model made; **precision** counts the "
        "unrequested changes against it; **exactness** asks whether the wording it "
        "produced was one the corpus accepts. **Determinism** is agreement between "
        "repetitions, compared as edit sets rather than as text, so that a model "
        "which proposes nothing does not score perfectly for it.",
        "",
        "A fidelity failure is not a bad score, it is a wrong book. Read the "
        "failure appendix before reading anything else.",
        "",
    ]
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def write_report(report: BenchmarkReport, *, appendix_limit: int = 8) -> None:
    atomic_write(
        report.json_path,
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )
    atomic_write(
        report.markdown_path,
        render_markdown(report, appendix_limit=appendix_limit),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkReport",
    "CaseRun",
    "ModelReport",
    "ProtocolTotals",
    "atomic_write",
    "render_markdown",
    "utc_now",
    "write_report",
]
