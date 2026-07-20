"""Reproducible, provider-neutral narration-preparation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import statistics
import tempfile
from time import perf_counter
from typing import Any, Callable, Sequence

from ..extraction import parse_book_to_chapters, source_media_type
from ..preparation import (
    NarrationPreparationPipeline,
    NarrationPreparationProvider,
    PreparationRequest,
    PreparationResult,
    PreparedBook,
    SourceMetadata,
    create_provider,
    mask_citations,
    save_prepared_book,
    save_prepared_markdown,
    source_metadata_for_path,
    validate_preparation,
)


BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_BENCHMARK_MODELS = (
    "gemma4:12b",
    "gemma4:26b",
    "gemma4:31b",
)


@dataclass(frozen=True)
class BenchmarkOptions:
    """Configuration shared by every model in one preparation benchmark."""

    source_path: Path
    output_dir: Path
    provider_name: str
    models: tuple[str, ...]
    base_url: str
    timeout_seconds: float
    preview_chapters: int = 1
    preview_units: int = 1
    repetitions: int = 1

    def __post_init__(self) -> None:
        cleaned_models = tuple(model.strip() for model in self.models if model.strip())
        if not cleaned_models:
            raise ValueError("At least one benchmark model is required")
        if len(set(cleaned_models)) != len(cleaned_models):
            raise ValueError("Benchmark model names must be unique")
        if cleaned_models != self.models:
            object.__setattr__(self, "models", cleaned_models)
        if not self.provider_name.strip():
            raise ValueError("Benchmark provider cannot be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("Benchmark provider timeout must be positive")
        if self.preview_chapters <= 0:
            raise ValueError("Benchmark chapter count must be positive")
        if self.preview_units <= 0:
            raise ValueError("Benchmark unit count must be positive")
        if self.repetitions <= 0:
            raise ValueError("Benchmark repetitions must be positive")


@dataclass
class QualityMetrics:
    """Automatic preservation indicators; human review remains authoritative."""

    source_chars: int = 0
    prepared_chars: int = 0
    prose_units: int = 0
    lexical_retention: float | None = None
    minimum_unit_retention: float | None = None
    expansion_ratio: float | None = None
    source_similarity: float | None = None
    citation_target_similarity: float | None = None
    citation_shaped_chars_before: int = 0
    citation_shaped_chars_after: int = 0
    citation_reduction: float | None = None
    paragraph_boundaries_preserved: bool = True
    edit_count: int = 0
    warning_count: int = 0


@dataclass
class ModelRunResult:
    """One isolated provider/model/repetition result."""

    model: str
    repetition: int
    success: bool
    wall_seconds: float
    provider_seconds: float
    provider_calls: int
    artifact_path: str | None = None
    markdown_path: str | None = None
    prepared_sha256: str | None = None
    quality: QualityMetrics | None = None
    error: str | None = None
    prepared_text: str = field(default="", repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("prepared_text", None)
        return payload


@dataclass
class ModelSummary:
    model: str
    successful_runs: int
    total_runs: int
    mean_wall_seconds: float | None
    minimum_wall_seconds: float | None
    mean_provider_seconds: float | None
    mean_lexical_retention: float | None
    minimum_unit_retention: float | None
    mean_source_similarity: float | None
    mean_citation_target_similarity: float | None
    mean_citation_reduction: float | None
    paragraph_boundaries_preserved: bool | None
    mean_edit_count: float | None
    mean_warning_count: float | None
    consistency: float | None


@dataclass
class PairwiseComparison:
    first_model: str
    second_model: str
    output_similarity: float
    character_delta: int


@dataclass
class BenchmarkReport:
    """Machine-readable benchmark result with human-review report paths."""

    created_at: str
    source: SourceMetadata
    options: BenchmarkOptions
    selected_chapters: list[str]
    runs: list[ModelRunResult]
    summaries: list[ModelSummary]
    comparisons: list[PairwiseComparison]
    json_path: Path
    markdown_path: Path
    schema_version: int = BENCHMARK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "source": self.source.to_dict(),
            "configuration": {
                "source_path": str(self.options.source_path),
                "output_dir": str(self.options.output_dir),
                "provider": self.options.provider_name,
                "models": list(self.options.models),
                "base_url": self.options.base_url,
                "timeout_seconds": self.options.timeout_seconds,
                "preview_chapters": self.options.preview_chapters,
                "preview_units": self.options.preview_units,
                "repetitions": self.options.repetitions,
                "cache_reuse": False,
            },
            "selected_chapters": self.selected_chapters,
            "model_summaries": [asdict(summary) for summary in self.summaries],
            "pairwise_comparisons": [
                asdict(comparison) for comparison in self.comparisons
            ],
            "runs": [run.to_dict() for run in self.runs],
        }


ProviderFactory = Callable[..., NarrationPreparationProvider]


class _TimedProvider:
    """Protocol-preserving wrapper that measures only provider preparation calls."""

    def __init__(self, provider: NarrationPreparationProvider) -> None:
        self.provider = provider
        self.call_seconds: list[float] = []

    @property
    def metadata(self):
        return self.provider.metadata

    def check_available(self) -> None:
        self.provider.check_available()

    def prepare(self, request: PreparationRequest) -> PreparationResult:
        started = perf_counter()
        try:
            return self.provider.prepare(request)
        finally:
            self.call_seconds.append(perf_counter() - started)

    def close(self) -> None:
        self.provider.close()


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "model"


def _book_title(path: Path) -> str:
    title = re.sub(r"[_-]+", " ", path.stem).strip()
    return title.title() or "Audiobook Benchmark"


def _citation_shaped_chars(text: str) -> int:
    return max(0, len(text) - len(mask_citations(text)))


def _comparison_text(text: str) -> str:
    return " ".join(text.split())


def _quality_metrics(book: PreparedBook) -> QualityMetrics:
    prose_units = [
        (chapter.title, unit)
        for chapter in book.chapters
        for unit in chapter.units
        if unit.kind == "prose"
    ]
    reports = []
    for chapter_title, unit in prose_units:
        reports.append(
            validate_preparation(unit.source_text, unit.prepared_text)
        )

    source_text = "\n\n".join(unit.source_text for _title, unit in prose_units)
    prepared_text = "\n\n".join(
        unit.prepared_text for _title, unit in prose_units
    )
    source_tokens = sum(report.source_token_count for report in reports)
    retained_weighted = sum(
        report.lexical_retention * report.source_token_count for report in reports
    )
    citation_before = _citation_shaped_chars(source_text)
    citation_after = _citation_shaped_chars(prepared_text)
    source_masked_length = len(mask_citations(source_text).strip())
    prepared_masked_length = len(mask_citations(prepared_text).strip())
    return QualityMetrics(
        source_chars=len(source_text),
        prepared_chars=len(prepared_text),
        prose_units=len(prose_units),
        lexical_retention=(
            retained_weighted / source_tokens if source_tokens else 1.0
        ),
        minimum_unit_retention=(
            min(report.lexical_retention for report in reports) if reports else 1.0
        ),
        expansion_ratio=prepared_masked_length / max(1, source_masked_length),
        source_similarity=SequenceMatcher(
            None, source_text, prepared_text, autojunk=False
        ).ratio(),
        citation_target_similarity=SequenceMatcher(
            None,
            _comparison_text(mask_citations(source_text)),
            _comparison_text(prepared_text),
            autojunk=False,
        ).ratio(),
        citation_shaped_chars_before=citation_before,
        citation_shaped_chars_after=citation_after,
        citation_reduction=(
            1.0 - (citation_after / citation_before) if citation_before else None
        ),
        paragraph_boundaries_preserved=all(
            unit.source_text.count("\n\n") == unit.prepared_text.count("\n\n")
            for _title, unit in prose_units
        ),
        edit_count=sum(len(unit.edits) for _title, unit in prose_units),
        warning_count=sum(len(unit.warnings) for _title, unit in prose_units),
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _summarize_model(model: str, runs: Sequence[ModelRunResult]) -> ModelSummary:
    successful = [run for run in runs if run.success and run.quality is not None]
    quality = [run.quality for run in successful if run.quality is not None]
    citation_values = [
        item.citation_reduction
        for item in quality
        if item.citation_reduction is not None
    ]
    baseline = successful[0].prepared_text if successful else ""
    consistency_values = [
        SequenceMatcher(None, baseline, run.prepared_text, autojunk=False).ratio()
        for run in successful[1:]
    ]
    return ModelSummary(
        model=model,
        successful_runs=len(successful),
        total_runs=len(runs),
        mean_wall_seconds=_mean([run.wall_seconds for run in successful]),
        minimum_wall_seconds=(
            min(run.wall_seconds for run in successful) if successful else None
        ),
        mean_provider_seconds=_mean(
            [run.provider_seconds for run in successful]
        ),
        mean_lexical_retention=_mean(
            [item.lexical_retention for item in quality if item.lexical_retention is not None]
        ),
        minimum_unit_retention=(
            min(
                item.minimum_unit_retention
                for item in quality
                if item.minimum_unit_retention is not None
            )
            if quality
            else None
        ),
        mean_source_similarity=_mean(
            [item.source_similarity for item in quality if item.source_similarity is not None]
        ),
        mean_citation_target_similarity=_mean(
            [
                item.citation_target_similarity
                for item in quality
                if item.citation_target_similarity is not None
            ]
        ),
        mean_citation_reduction=_mean(citation_values),
        paragraph_boundaries_preserved=(
            all(item.paragraph_boundaries_preserved for item in quality)
            if quality
            else None
        ),
        mean_edit_count=_mean([float(item.edit_count) for item in quality]),
        mean_warning_count=_mean([float(item.warning_count) for item in quality]),
        consistency=(
            _mean(consistency_values)
            if consistency_values
            else (1.0 if len(successful) == 1 else None)
        ),
    )


def _pairwise_comparisons(
    models: Sequence[str], runs: Sequence[ModelRunResult]
) -> list[PairwiseComparison]:
    first_success = {
        model: next(
            (run for run in runs if run.model == model and run.success),
            None,
        )
        for model in models
    }
    comparisons: list[PairwiseComparison] = []
    for first_index, first_model in enumerate(models):
        first = first_success[first_model]
        if first is None:
            continue
        for second_model in models[first_index + 1 :]:
            second = first_success[second_model]
            if second is None:
                continue
            comparisons.append(
                PairwiseComparison(
                    first_model=first_model,
                    second_model=second_model,
                    output_similarity=SequenceMatcher(
                        None,
                        first.prepared_text,
                        second.prepared_text,
                        autojunk=False,
                    ).ratio(),
                    character_delta=len(second.prepared_text) - len(first.prepared_text),
                )
            )
    return comparisons


def _format_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _format_percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _markdown_link(path: str | None, root: Path) -> str:
    if path is None:
        return "—"
    destination = Path(path)
    try:
        display = destination.relative_to(root)
    except ValueError:
        display = destination
    return f"[{display}]({display.as_posix()})"


def _render_markdown(report: BenchmarkReport) -> str:
    lines = [
        "# Narration preparation benchmark",
        "",
        f"- Provider: `{report.options.provider_name}`",
        f"- Source: `{report.options.source_path}`",
        f"- Chapters sampled: {report.options.preview_chapters}",
        f"- Prose units sampled: {report.options.preview_units}",
        f"- Repetitions per model: {report.options.repetitions}",
        "- Cache reuse: disabled",
        "",
        "## Model summary",
        "",
        "| Model | Successful | Mean wall (s) | Provider (s) | Lexical retention | Citation-target similarity | Citation-shape reduction | Paragraphs | Consistency |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for summary in report.summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary.model.replace('|', '\\|')}`",
                    f"{summary.successful_runs}/{summary.total_runs}",
                    _format_seconds(summary.mean_wall_seconds),
                    _format_seconds(summary.mean_provider_seconds),
                    _format_percent(summary.mean_lexical_retention),
                    _format_percent(summary.mean_citation_target_similarity),
                    _format_percent(summary.mean_citation_reduction),
                    (
                        "yes"
                        if summary.paragraph_boundaries_preserved is True
                        else "no"
                        if summary.paragraph_boundaries_preserved is False
                        else "—"
                    ),
                    _format_percent(summary.consistency),
                ]
            )
            + " |"
        )

    if report.comparisons:
        lines.extend(
            [
                "",
                "## Pairwise prepared-text comparison",
                "",
                "| First | Second | Output similarity | Character delta |",
                "|---|---|---:|---:|",
            ]
        )
        for comparison in report.comparisons:
            lines.append(
                f"| `{comparison.first_model}` | `{comparison.second_model}` | "
                f"{comparison.output_similarity:.1%} | "
                f"{comparison.character_delta:+d} |"
            )

    lines.extend(["", "## Runs", ""])
    for run in report.runs:
        status = "passed" if run.success else "failed"
        lines.extend(
            [
                f"### {run.model} — run {run.repetition} ({status})",
                "",
                f"- Wall time: {run.wall_seconds:.2f} seconds",
                f"- Provider calls: {run.provider_calls} in "
                f"{run.provider_seconds:.2f} seconds",
            ]
        )
        if run.success and run.quality is not None:
            lines.extend(
                [
                    f"- Lexical retention: {_format_percent(run.quality.lexical_retention)}",
                    f"- Minimum unit retention: {_format_percent(run.quality.minimum_unit_retention)}",
                    f"- Source similarity: {_format_percent(run.quality.source_similarity)}",
                    f"- Citation-stripped target similarity: "
                    f"{_format_percent(run.quality.citation_target_similarity)}",
                    f"- Citation-shaped characters: "
                    f"{run.quality.citation_shaped_chars_before} → "
                    f"{run.quality.citation_shaped_chars_after}",
                    f"- Reported edits/warnings: {run.quality.edit_count}/"
                    f"{run.quality.warning_count}",
                    f"- Prepared JSON: {_markdown_link(run.artifact_path, report.options.output_dir)}",
                    f"- Reading copy: {_markdown_link(run.markdown_path, report.options.output_dir)}",
                ]
            )
        else:
            lines.append(f"- Error: `{run.error or 'unknown error'}`")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "Automatic metrics detect obvious deletion, expansion, citation retention, "
            "format drift, and inconsistent output. They do not prove semantic fidelity. "
            "Review the reading copies, especially names, dates, quotations, and "
            "parenthetical qualifications, before changing the production default.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
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


def benchmark_preparation(
    options: BenchmarkOptions,
    *,
    provider_factory: ProviderFactory = create_provider,
    chapters: Sequence[tuple[str, str]] | None = None,
    source_metadata: SourceMetadata | None = None,
) -> BenchmarkReport:
    """Run identical source units through every configured model and report results."""

    selected = (
        list(chapters)
        if chapters is not None
        else parse_book_to_chapters(options.source_path)
    )
    selected = selected[: options.preview_chapters]
    if not selected:
        raise RuntimeError("No chapters were available for the benchmark")
    source = source_metadata or source_metadata_for_path(
        options.source_path, media_type=source_media_type(options.source_path)
    )
    options.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[ModelRunResult] = []

    # Round-robin repetitions avoid giving one model every adjacent warm-cache
    # run before the next model is loaded.
    for repetition in range(1, options.repetitions + 1):
        for model_index, model in enumerate(options.models, start=1):
            model_dir = options.output_dir / (
                f"{model_index:02d}_{_safe_component(model)}"
            )
            run_dir = model_dir / f"run_{repetition:02d}"
            artifact_path = run_dir / "prepared_book.json"
            markdown_path = run_dir / "prepared_book.md"
            provider: NarrationPreparationProvider | None = None
            timed: _TimedProvider | None = None
            book: PreparedBook | None = None
            error: str | None = None
            started = perf_counter()
            try:
                provider = provider_factory(
                    options.provider_name,
                    model=model,
                    base_url=options.base_url,
                    timeout=options.timeout_seconds,
                )
                timed = _TimedProvider(provider)
                pipeline = NarrationPreparationPipeline(timed)

                def checkpoint(current: PreparedBook) -> None:
                    save_prepared_book(current, artifact_path)
                    save_prepared_markdown(current, markdown_path)

                book = pipeline.prepare_book(
                    selected,
                    book_title=_book_title(options.source_path),
                    source_metadata=source,
                    checkpoint=checkpoint,
                    max_prose_units=options.preview_units,
                )
                checkpoint(book)
            except Exception as exc:  # Continue so one model cannot hide the others.
                error = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    if timed is not None:
                        timed.close()
                    elif provider is not None:
                        provider.close()
                except Exception as exc:
                    if error is None:
                        error = f"{type(exc).__name__} during provider close: {exc}"
            wall_seconds = perf_counter() - started
            success = book is not None and error is None
            runs.append(
                ModelRunResult(
                    model=model,
                    repetition=repetition,
                    success=success,
                    wall_seconds=wall_seconds,
                    provider_seconds=sum(timed.call_seconds) if timed else 0.0,
                    provider_calls=len(timed.call_seconds) if timed else 0,
                    artifact_path=str(artifact_path) if success else None,
                    markdown_path=str(markdown_path) if success else None,
                    prepared_sha256=book.prepared_sha256 if success and book else None,
                    quality=_quality_metrics(book) if success and book else None,
                    error=error,
                    prepared_text=book.prepared_text if success and book else "",
                )
            )

    summaries = [
        _summarize_model(model, [run for run in runs if run.model == model])
        for model in options.models
    ]
    comparisons = _pairwise_comparisons(options.models, runs)
    report = BenchmarkReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        source=source,
        options=options,
        selected_chapters=[title for title, _text in selected],
        runs=runs,
        summaries=summaries,
        comparisons=comparisons,
        json_path=options.output_dir / "benchmark.json",
        markdown_path=options.output_dir / "comparison.md",
    )
    _atomic_write(
        report.json_path,
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )
    _atomic_write(report.markdown_path, _render_markdown(report))
    return report


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_BENCHMARK_MODELS",
    "BenchmarkOptions",
    "BenchmarkReport",
    "ModelRunResult",
    "QualityMetrics",
    "benchmark_preparation",
]
