"""Gold-corpus benchmarks for the narration-preparation stage.

A model is scored against passages whose correct preparation is already known,
so the result says which edits it got right rather than merely how much text it
changed.
"""

from .corpus import (
    CATEGORIES,
    DEFAULT_CORPUS_DIR,
    TIERS,
    BenchmarkCase,
    CorpusError,
    lint_case,
    load_case,
    load_corpus,
)
from .run import (
    BASE_SEED,
    DEFAULT_BENCHMARK_MODELS,
    BenchmarkOptions,
    BenchmarkVariant,
    benchmark_preparation,
    benchmark_progress,
    default_output_dir,
    print_summary,
    run,
    seed_for_repetition,
)
from .plots import write_plots
from .report import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkReport,
    CaseRun,
    ModelReport,
    render_markdown,
)
from .scoring import CaseScore, change_regions, score_case

__all__ = [
    "BASE_SEED",
    "BENCHMARK_SCHEMA_VERSION",
    "CATEGORIES",
    "DEFAULT_BENCHMARK_MODELS",
    "DEFAULT_CORPUS_DIR",
    "TIERS",
    "BenchmarkCase",
    "BenchmarkOptions",
    "BenchmarkReport",
    "BenchmarkVariant",
    "CaseRun",
    "CaseScore",
    "CorpusError",
    "ModelReport",
    "benchmark_preparation",
    "benchmark_progress",
    "change_regions",
    "default_output_dir",
    "lint_case",
    "load_case",
    "load_corpus",
    "print_summary",
    "render_markdown",
    "run",
    "score_case",
    "seed_for_repetition",
    "write_plots",
]
