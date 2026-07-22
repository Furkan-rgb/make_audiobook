"""Running every model over the same gold corpus, under production rules.

The point of a benchmark is that the only thing which differs between two
columns of the table is the model. So each case is a fresh provider request
with no cache, no resume, and no neighbouring case to condition on, and what
comes back is applied by the same applier production uses, with the same
validation policy. A model is scored on the prose a listener would actually
have heard, not on the JSON it emitted.

Each variant is taken to exhaustion — every repetition of every case — before
the next is loaded, and variants are ordered so a model and its thinking
counterpart run back to back. On a local server, loading a model is tens of
seconds of moving gigabytes into VRAM; interleaving models across repetitions,
as an earlier version did, reloaded each one once per repetition and spent more
time swapping weights than preparing text. The cost of finishing sooner is that
timing is no longer insulated from a machine that slows over a long run — a
model measured late looks slower than one measured early — but scores are
unaffected, because every case is independent and each is generated under a
fixed per-repetition seed, and timing is already reported as specific to the
machine.

Sampling is left to each model package: the benchmark sends no temperature or
other sampling option, so a model runs under the policy it ships with rather
than one the harness imposed. What stays pinned is everything that would make a
comparison unfair otherwise — the prompt, schema, thinking mode, context and
output budgets, and a deterministic seed sequence shared across models, so that
repetition N of every model runs under the same seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable, Sequence, TextIO

from ..preparation import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_POLICY,
    DEFAULT_PROMPT_VERSION,
    SAMPLING_OPTIONS,
    NarrationPreparationProvider,
    PreparationEdit,
    PreparationRequest,
    ValidationPolicy,
    apply_edits,
    create_provider,
    system_prompt_for,
    validate_preparation,
)
from .corpus import BenchmarkCase, load_corpus
from .plots import write_plots
from .report import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkReport,
    CaseRun,
    ModelReport,
    ProtocolTotals,
    utc_now,
    write_report,
)
from .scoring import (
    breakdowns,
    determinism,
    edit_signature,
    score_case,
    summarize,
    trap_failures,
)


DEFAULT_BENCHMARK_MODELS = (
    "gemma4:12b",
    "gemma4:26b",
    "gemma4:31b",
)

# The seed repetition 1 runs under; later repetitions step up from here. Fixed
# so a run is reproducible and every model sees the same seed for the same
# repetition, while successive repetitions still vary the generation rather than
# reusing one seed — the point of running a model more than once under native
# sampling is to see whether it stays safe across several generations.
BASE_SEED = 42

ProviderFactory = Callable[..., NarrationPreparationProvider]


def seed_for_repetition(repetition: int, base: int = BASE_SEED) -> int:
    """The seed a 1-based ``repetition`` generates under: ``base + index``.

    Repetition 1 is always ``base`` (never ``base + 1``), so the first run of
    every model is reproducible from the base seed alone.
    """

    if repetition < 1:
        raise ValueError("Repetition numbers are 1-based")
    return base + repetition - 1


@dataclass(frozen=True)
class BenchmarkOptions:
    """Configuration shared by every model in one preparation benchmark."""

    output_dir: Path
    provider_name: str
    models: tuple[str, ...]
    base_url: str
    timeout_seconds: float
    repetitions: int = 1
    corpus_dir: Path | None = None
    tiers: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    quick: bool = False
    # Each model is scored once per think mode, so ``(False, True)`` compares a
    # model with and without reasoning as two separately ranked entries. The
    # default keeps thinking off, which is how the pipeline runs in production.
    think_modes: tuple[bool, ...] = (False,)
    # Models the provider says cannot think, so the thinking run is skipped for
    # them rather than filed as forty-eight identical provider errors.
    no_think_models: tuple[str, ...] = ()
    appendix_limit: int = 8
    # Which system prompt every model runs under. Changing it is how a run scores
    # a new prompt version: run once per version and compare the two leaderboards.
    prompt_version: str = DEFAULT_PROMPT_VERSION
    validation_policy: ValidationPolicy = field(default_factory=ValidationPolicy)

    def __post_init__(self) -> None:
        cleaned = tuple(model.strip() for model in self.models if model.strip())
        if not cleaned:
            raise ValueError("At least one benchmark model is required")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Benchmark model names must be unique")
        if cleaned != self.models:
            object.__setattr__(self, "models", cleaned)
        modes = tuple(dict.fromkeys(bool(mode) for mode in self.think_modes))
        if not modes:
            raise ValueError("At least one think mode is required")
        if modes != self.think_modes:
            object.__setattr__(self, "think_modes", modes)
        if not self.provider_name.strip():
            raise ValueError("Benchmark provider cannot be blank")
        if self.timeout_seconds <= 0:
            raise ValueError("Benchmark provider timeout must be positive")
        if self.repetitions <= 0:
            raise ValueError("Benchmark repetitions must be positive")
        if self.appendix_limit < 0:
            raise ValueError("Benchmark appendix limit cannot be negative")
        # Fail here, not after gigabytes of weights have loaded, if the prompt
        # version is misspelled or not one the package carries.
        system_prompt_for(self.prompt_version)

    @property
    def variants(self) -> tuple["BenchmarkVariant", ...]:
        """The (model, think) pairs a run scores, each with a display label.

        Thinking is folded into the label rather than a separate column so the
        rest of the report — leaderboard, breakdowns, appendix — treats a model
        and its thinking counterpart as two independent competitors, which is
        exactly what a with/without comparison wants.
        """

        blocked = set(self.no_think_models)
        result: list[BenchmarkVariant] = []
        for model in self.models:
            modes = self.think_modes
            if model in blocked:
                modes = tuple(mode for mode in modes if not mode)
            for think in modes:
                result.append(
                    BenchmarkVariant(
                        label=f"{model} +think" if think else model,
                        model=model,
                        think=think,
                    )
                )
        return tuple(result)


@dataclass(frozen=True)
class BenchmarkVariant:
    """One competitor in a run: a model, a think mode, and the label they share."""

    label: str
    model: str
    think: bool


class _Attempt:
    """One provider call and everything the pipeline made of it."""

    def __init__(self) -> None:
        self.proposed: list[PreparationEdit] = []
        self.applied: list[PreparationEdit] = []
        self.warnings: list[str] = []
        self.prepared: str = ""
        self.error: str | None = None


def _attempt(
    provider: NarrationPreparationProvider,
    case: BenchmarkCase,
    policy: ValidationPolicy,
) -> _Attempt:
    """Prepare one case exactly as production would, capturing every stage.

    A failure anywhere — a provider that times out, a response that will not
    parse, an applied result that validation rejects — is recorded against the
    case and the run continues. One difficult passage must not decide a model
    comparison by ending it early.
    """

    attempt = _Attempt()
    request = PreparationRequest(
        unit_id=case.id,
        chapter_title=case.chapter_title,
        source_text=case.source,
        previous_context=case.previous_context,
        following_context=case.following_context,
        prompt_version=provider.metadata.prompt_version,
        policy=DEFAULT_POLICY,
    )
    try:
        result = provider.prepare(request)
        attempt.proposed = list(result.edits)
        prepared, applied, refusals = apply_edits(
            case.source, list(result.edits), policy=policy
        )
        attempt.applied = applied
        attempt.warnings = [*result.warnings, *refusals]
        validate_preparation(case.source, prepared, policy=policy)
        attempt.prepared = prepared.strip()
    except Exception as exc:
        attempt.error = f"{type(exc).__name__}: {exc}"
    return attempt


def _model_report(
    model: str,
    runs: Sequence[CaseRun],
    provider_options: dict[str, Any] | None = None,
) -> ModelReport:
    scores = [run.score for run in runs]
    protocol = ProtocolTotals()
    for score in scores:
        protocol.proposed += score.protocol.proposed
        protocol.applied += score.protocol.applied
        protocol.unanchored += score.protocol.unanchored
        protocol.ambiguous += score.protocol.ambiguous
        protocol.oversized += score.protocol.oversized
        protocol.label_only += score.protocol.label_only
        protocol.refused += score.protocol.refused

    by_case: dict[str, list[CaseRun]] = {}
    for run in runs:
        by_case.setdefault(run.case_id, []).append(run)
    agreements = [
        value
        for value in (
            determinism([edit_signature(run.proposed) for run in group])
            for group in by_case.values()
        )
        if value is not None
    ]

    seconds = [run.seconds for run in runs]
    return ModelReport(
        model=model,
        overall=summarize(model, scores),
        by_tier=breakdowns(scores, "tier"),
        by_category=breakdowns(scores, "category"),
        protocol=protocol,
        trap_failures=trap_failures(scores),
        determinism=(sum(agreements) / len(agreements) if agreements else None),
        mean_seconds=(sum(seconds) / len(seconds) if seconds else 0.0),
        total_seconds=sum(seconds),
        errored_cases=sum(1 for score in scores if score.error is not None),
        provider_options=dict(provider_options or {}),
        runs=list(runs),
    )


def _provider_options(provider: NarrationPreparationProvider | None) -> dict[str, Any]:
    """The options a provider declares it will run under, or empty if it failed.

    Read from the provider's own metadata rather than assumed, so the report
    records what was actually sent — including that the sampling options were
    omitted for the model package to supply.
    """

    if provider is None:
        return {}
    try:
        return dict(provider.metadata.parameters)
    except Exception:
        return {}


def benchmark_preparation(
    options: BenchmarkOptions,
    *,
    provider_factory: ProviderFactory = create_provider,
    cases: Sequence[BenchmarkCase] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> BenchmarkReport:
    """Score every configured model against the gold corpus and write a report."""

    selected = list(cases) if cases is not None else load_corpus(
        options.corpus_dir,
        tiers=options.tiers or None,
        categories=options.categories or None,
        ids=options.case_ids or None,
        limit_per_tier=3 if options.quick else None,
    )
    if not selected:
        raise RuntimeError("No benchmark cases were available")
    options.output_dir.mkdir(parents=True, exist_ok=True)

    variants = options.variants
    runs: list[CaseRun] = []
    seeds = [seed_for_repetition(rep) for rep in range(1, options.repetitions + 1)]
    variant_options: dict[str, dict[str, Any]] = {}
    total = len(selected) * len(variants) * options.repetitions
    if progress is not None:
        progress(0, total)

    for variant in variants:
        provider: NarrationPreparationProvider | None = None
        setup_error: str | None = None
        try:
            # No sampling option is passed here: the provider omits every one by
            # default, so each model runs under its package's native sampling.
            # This is the same policy production uses — the benchmark does not
            # impose a sampling profile of its own.
            provider = provider_factory(
                options.provider_name,
                model=variant.model,
                base_url=options.base_url,
                timeout=options.timeout_seconds,
                think=variant.think,
                prompt_version=options.prompt_version,
            )
            provider.check_available()
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

        variant_options[variant.label] = _provider_options(provider)

        try:
            for repetition in range(1, options.repetitions + 1):
                seed = seed_for_repetition(repetition)
                if provider is not None:
                    # One provider stays loaded across every repetition of this
                    # variant; only the seed advances, so repetition N runs under
                    # the same seed for every model without reloading weights.
                    provider.seed = seed
                for case in selected:
                    started = perf_counter()
                    if provider is None or setup_error is not None:
                        attempt = _Attempt()
                        attempt.error = setup_error or "provider was not created"
                    else:
                        attempt = _attempt(
                            provider, case, options.validation_policy
                        )
                    elapsed = perf_counter() - started
                    score = score_case(
                        case,
                        attempt.prepared or case.source,
                        proposed=attempt.proposed,
                        applied=attempt.applied,
                        warnings=attempt.warnings,
                        error=attempt.error,
                    )
                    runs.append(
                        CaseRun(
                            case_id=case.id,
                            model=variant.label,
                            repetition=repetition,
                            seed=seed,
                            seconds=elapsed,
                            score=score,
                            proposed=list(attempt.proposed),
                        )
                    )
                    if progress is not None:
                        progress(len(runs), total)
        finally:
            if provider is not None:
                try:
                    provider.close()
                except Exception:
                    # A provider that will not shut down cleanly has already
                    # given us its answers; losing them here helps nobody.
                    pass

    tier_counts: dict[str, int] = {}
    for case in selected:
        tier_counts[case.tier] = tier_counts.get(case.tier, 0) + 1

    # The benchmark's one behaviour is native sampling, so unless a variant's
    # provider reports having sent sampling options, the run inherited them all.
    native_sampling = not any(
        key in SAMPLING_OPTIONS
        for parameters in variant_options.values()
        for key in parameters
    )

    report = BenchmarkReport(
        created_at=utc_now(),
        provider_name=options.provider_name,
        models=[variant.label for variant in variants],
        repetitions=options.repetitions,
        corpus_size=len(selected),
        corpus_tiers=tier_counts,
        models_reports=[
            _model_report(
                variant.label,
                [run for run in runs if run.model == variant.label],
                variant_options.get(variant.label),
            )
            for variant in variants
        ],
        runs=runs,
        native_sampling=native_sampling,
        seeds=seeds,
        prompt_version=options.prompt_version,
        json_path=options.output_dir / "benchmark.json",
        markdown_path=options.output_dir / "comparison.md",
    )
    write_report(report, appendix_limit=options.appendix_limit)
    try:
        report.plot_paths = write_plots(report, options.output_dir / "plots")
    except Exception:
        # The run's numbers are already on disk; a plotting failure must not be
        # allowed to discard a benchmark that may have taken hours to produce.
        report.plot_paths = []
    return report


def benchmark_progress(done: int, total: int) -> None:
    """One rewritten line of progress: a full benchmark is long and mostly silent."""

    print(f"\rBenchmarking case {done}/{total}", end="", file=sys.stderr, flush=True)


def default_output_dir(base: Path = Path("output")) -> Path:
    """A timestamped run directory under ``<base>/benchmarks/``."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / "benchmarks" / timestamp


def print_summary(report: BenchmarkReport, *, file: TextIO | None = None) -> None:
    """Print the artifact paths and the ranked leaderboard for a finished run.

    Shared by the CLI and the driver script so both report a run identically;
    the errored-case warning always goes to stderr regardless of ``file``.
    """

    out = file if file is not None else sys.stdout
    print(f"\nBenchmark JSON: {report.json_path}", file=out)
    print(f"Comparison report: {report.markdown_path}", file=out)
    if report.plot_paths:
        print(f"Plots: {report.plot_paths[0].parent}", file=out)
    print(file=out)
    width = max((len(item.model) for item in report.models_reports), default=0)
    for item in report.ranked:
        overall = item.overall
        print(
            f"  {item.model:<{width}}  score {overall.score:.3f}   "
            f"passed {overall.passed}/{overall.cases}   "
            f"fidelity failures {overall.fidelity_failures}",
            file=out,
        )
    errored = sum(item.errored_cases for item in report.models_reports)
    if errored:
        print(
            f"\nWarning: {errored} of {len(report.runs)} case runs errored; "
            "see the report for details.",
            file=sys.stderr,
        )


def run(
    *,
    models: Sequence[str] = DEFAULT_BENCHMARK_MODELS,
    think_modes: Sequence[bool] = (False,),
    repetitions: int = 1,
    provider: str = "ollama",
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout: float = 300.0,
    tiers: Sequence[str] = (),
    categories: Sequence[str] = (),
    case_ids: Sequence[str] = (),
    quick: bool = False,
    no_think_models: Sequence[str] = (),
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    output_dir: Path | str | None = None,
    provider_factory: ProviderFactory = create_provider,
    cases: Sequence[BenchmarkCase] | None = None,
    progress: Callable[[int, int], None] | None = benchmark_progress,
    show_summary: bool = True,
) -> BenchmarkReport:
    """Build options from plain values, score every model, and write the report.

    The file-first way to run a benchmark: a driver script sets the arguments
    as ordinary Python and calls this, instead of assembling a
    :class:`BenchmarkOptions` and threading a progress callback by hand. It is a
    thin convenience over :func:`benchmark_preparation`, which remains the seam
    the tests drive.
    """

    options = BenchmarkOptions(
        output_dir=Path(output_dir) if output_dir is not None else default_output_dir(),
        provider_name=provider,
        models=tuple(models),
        base_url=base_url,
        timeout_seconds=timeout,
        repetitions=repetitions,
        tiers=tuple(tiers),
        categories=tuple(categories),
        case_ids=tuple(case_ids),
        quick=quick,
        think_modes=tuple(think_modes),
        no_think_models=tuple(no_think_models),
        prompt_version=prompt_version,
    )
    report = benchmark_preparation(
        options, provider_factory=provider_factory, cases=cases, progress=progress
    )
    if show_summary:
        print_summary(report)
    return report


__all__ = [
    "BASE_SEED",
    "BENCHMARK_SCHEMA_VERSION",
    "DEFAULT_BENCHMARK_MODELS",
    "BenchmarkOptions",
    "BenchmarkReport",
    "BenchmarkVariant",
    "CaseRun",
    "ModelReport",
    "benchmark_preparation",
    "benchmark_progress",
    "default_output_dir",
    "print_summary",
    "run",
    "seed_for_repetition",
]
