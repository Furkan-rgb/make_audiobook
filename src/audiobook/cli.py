"""CLI for the modular book → prepared script → Qwen3-TTS workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .benchmarking import (
    CATEGORIES,
    DEFAULT_BENCHMARK_MODELS,
    TIERS,
    BenchmarkOptions,
    benchmark_preparation,
    benchmark_progress,
    default_output_dir,
    print_summary,
)
from .preparation import DEFAULT_PROMPT_VERSION, SYSTEM_PROMPTS

from .assembly.audio import (
    _crossfade,
    _fade_in,
    _fade_out,
    assemble_chunk_audio,
    create_ffmpeg_metadata,
    merge_chapters,
)
from .config import (
    CHAPTER_SILENCE_MS,
    CHUNK_CROSSFADE_MS,
    CONTEXT_CHARS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_BOOK_PATH,
    DEFAULT_PREPARATION_MODEL,
    DEFAULT_PREPARATION_PROVIDER,
    DEFAULT_PREPARED_MARKDOWN_FILENAME,
    DEFAULT_PREPARED_SCRIPT_FILENAME,
    DEFAULT_PREVIEW_OUTPUT_FILENAME,
    DEFAULT_PROVIDER_BASE_URL,
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    LANGUAGE,
    LOCAL_TTS_MODEL_PATH,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    NARRATION_INSTRUCTION,
    PARAGRAPH_SILENCE_MS,
    SECTION_SILENCE_MS,
    TARGET_CHUNK_CHARS,
    TARGET_CHUNK_DURATION_SECONDS,
    TTS_MODEL,
    VOICE_NAME,
)
from .workflow import (
    NarrationWorkflowOptions,
    PreparationWorkflowOptions,
    narrate_chapters,
    narrate_prepared_script,
    narration_chapters,
    prepare_narration_script,
    resolve_script_path,
)
from .extraction import (
    SUPPORTED_SOURCE_SUFFIXES,
    parse_book_to_chapters,
    parse_epub_to_chapters,
)
from .extraction.pdf import (
    RE_BOLD,
    RE_CITATIONS_BRACKET,
    RE_CITATIONS_PAREN,
    RE_CODE,
    RE_FIGS,
    RE_HYPHENS,
    RE_IMGS,
    RE_LINKS,
    RE_NEWLINES,
    RE_NUMBERED_CHAPTER,
    RE_PAGENUMS,
    RE_PART_BOOKMARK,
    RE_STANDALONE_PAGE_NUMBER,
    RE_WHITESPACE,
    _join_markdown_pages,
    clean_text_segment,
    parse_pdf_to_chapters,
)
from .chunking.semantic import (
    NarrationChunk,
    RE_CLAUSE_BOUNDARY,
    RE_DIALOGUE,
    RE_SCENE_BREAK,
    RE_SENTENCE_BOUNDARY,
    TextSection,
    TextUnit,
    _context_head,
    _context_tail,
    _greedy_pack,
    _join_units,
    _make_text_units,
    _normalize_paragraph,
    _split_long_sentence,
    build_chunk_plan,
    display_chunk_plan,
    make_narration_chunks,
    sentence_split,
    split_into_sections,
    split_long_paragraph,
)


# Compatibility names retained for scripts that imported the former monolith.
BOOK_PATH = DEFAULT_BOOK_PATH
PDF_PATH = BOOK_PATH
OUTPUT_FOLDER = DEFAULT_OUTPUT_DIR
OUTPUT_FILENAME = DEFAULT_OUTPUT_FILENAME
PREVIEW_OUTPUT_FILENAME = DEFAULT_PREVIEW_OUTPUT_FILENAME
PREPARED_SCRIPT_FILENAME = DEFAULT_PREPARED_SCRIPT_FILENAME
PREPARED_MARKDOWN_FILENAME = DEFAULT_PREPARED_MARKDOWN_FILENAME
PREPARATION_PROVIDER = DEFAULT_PREPARATION_PROVIDER
PREPARATION_MODEL = DEFAULT_PREPARATION_MODEL
PREPARATION_PROVIDER_BASE_URL = DEFAULT_PROVIDER_BASE_URL
PREPARATION_PROVIDER_TIMEOUT_SECONDS = DEFAULT_PROVIDER_TIMEOUT_SECONDS
LOCAL_MODEL_PATH = LOCAL_TTS_MODEL_PATH


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_FOLDER)


def _add_script_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--script",
        type=Path,
        help="Prepared-book JSON path (default: <output-dir>/prepared_book.json).",
    )


def _add_preparation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--book",
        "--pdf",
        dest="book",
        type=Path,
        default=BOOK_PATH,
        help=(
            "Book to extract; the backend follows the extension "
            f"({', '.join(SUPPORTED_SOURCE_SUFFIXES)})."
        ),
    )
    parser.add_argument("--provider", default=PREPARATION_PROVIDER)
    parser.add_argument("--preparation-model", default=PREPARATION_MODEL)
    parser.add_argument(
        "--provider-base-url",
        default=PREPARATION_PROVIDER_BASE_URL,
        help="Base URL used by the selected preparation provider.",
    )
    parser.add_argument(
        "--provider-timeout",
        type=float,
        default=PREPARATION_PROVIDER_TIMEOUT_SECONDS,
        metavar="SECONDS",
    )
    parser.add_argument(
        "--force-preparation",
        action="store_true",
        help="Ignore compatible cached units and adapt them again.",
    )
    parser.add_argument(
        "--preview-units",
        type=int,
        metavar="N",
        help="Prepare only the first N editable units.",
    )


def _add_narration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tts-model",
        "--model",
        dest="tts_model",
        default=str(LOCAL_MODEL_PATH if LOCAL_MODEL_PATH.exists() else TTS_MODEL),
        help="Downloaded Qwen3-TTS directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--preview-chunks",
        type=int,
        metavar="N",
        help="Narrate only the first N semantic chunks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display the prepared-text chunk plan without loading Qwen3-TTS.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated chapter WAV files after a successful merge.",
    )


def _add_preview_chapters_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preview-chapters",
        type=int,
        metavar="N",
        help="Process only the first N detected chapters.",
    )


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default=PREPARATION_PROVIDER)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_MODELS),
        metavar="MODEL",
        help="Provider model identifiers to compare using identical source units.",
    )
    parser.add_argument(
        "--provider-base-url",
        default=PREPARATION_PROVIDER_BASE_URL,
        help="Base URL used by the selected preparation provider.",
    )
    parser.add_argument(
        "--provider-timeout",
        type=float,
        default=PREPARATION_PROVIDER_TIMEOUT_SECONDS,
        metavar="SECONDS",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        metavar="N",
        help="Run every model N times to measure timing and edit determinism.",
    )
    parser.add_argument(
        "--think",
        choices=["off", "on", "both"],
        default="off",
        help=(
            "Reasoning mode: off (default), on, or both to score each model "
            "with and without thinking as two ranked entries. Models that do "
            "not support thinking are reported and skipped for the thinking run."
        ),
    )
    parser.add_argument(
        "--prompt-version",
        choices=sorted(SYSTEM_PROMPTS),
        default=DEFAULT_PROMPT_VERSION,
        help=(
            "System prompt version every model runs under (default: "
            f"{DEFAULT_PROMPT_VERSION}). Run once per version to compare prompts."
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Gold corpus directory (default: the corpus shipped with the package).",
    )
    parser.add_argument(
        "--tier",
        dest="tiers",
        nargs="+",
        default=[],
        choices=list(TIERS),
        metavar="TIER",
        help=f"Score only these corpus tiers ({', '.join(TIERS)}).",
    )
    parser.add_argument(
        "--category",
        dest="categories",
        nargs="+",
        default=[],
        choices=list(CATEGORIES),
        metavar="CATEGORY",
        help="Score only cases exercising these edit categories.",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        nargs="+",
        default=[],
        metavar="ID",
        help="Score only these case ids, for reproducing one failure.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke subset: the first three cases of every tier, for checking a "
            "provider is wired up without paying for a full run."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Benchmark artifact directory (default: output/benchmarks/<timestamp>).",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit subcommands while retaining the old one-command syntax."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Extract and adapt a book into a reviewable narration script.",
    )
    _add_output_argument(prepare_parser)
    _add_script_argument(prepare_parser)
    _add_preparation_arguments(prepare_parser)
    _add_preview_chapters_argument(prepare_parser)

    narrate_parser = subparsers.add_parser(
        "narrate",
        help="Generate an audiobook from an existing prepared script.",
    )
    _add_output_argument(narrate_parser)
    _add_script_argument(narrate_parser)
    _add_narration_arguments(narrate_parser)
    _add_preview_chapters_argument(narrate_parser)

    all_parser = subparsers.add_parser(
        "all",
        help="Prepare a book and narrate the resulting script.",
    )
    _add_output_argument(all_parser)
    _add_script_argument(all_parser)
    _add_preparation_arguments(all_parser)
    _add_narration_arguments(all_parser)
    _add_preview_chapters_argument(all_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Compare narration-preparation models on identical prose units.",
    )
    _add_benchmark_arguments(benchmark_parser)

    subparsers.add_parser(
        "check",
        help="Probe every configured dependency and report, without running.",
    )

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        raw_args = ["all"]
    elif raw_args[0] not in {
        "prepare",
        "narrate",
        "all",
        "benchmark",
        "check",
        "-h",
        "--help",
    }:
        raw_args.insert(0, "all")
    return parser.parse_args(raw_args)


def _preparation_options(args: argparse.Namespace) -> PreparationWorkflowOptions:
    preview_units = args.preview_units
    # A one-chunk all-in-one preview should not prepare an entire book first.
    if (
        args.command == "all"
        and args.preview_chunks is not None
        and args.preview_chapters is None
        and preview_units is None
    ):
        preview_units = 1
    return PreparationWorkflowOptions(
        source_path=args.book,
        output_dir=args.output_dir,
        script_path=args.script,
        provider_name=args.provider,
        model=args.preparation_model,
        base_url=args.provider_base_url,
        timeout_seconds=args.provider_timeout,
        preview_chapters=args.preview_chapters,
        preview_units=preview_units,
        force=args.force_preparation,
    )


def _narration_options(
    args: argparse.Namespace,
    *,
    preparation_was_preview: bool = False,
) -> NarrationWorkflowOptions:
    return NarrationWorkflowOptions(
        output_dir=args.output_dir,
        script_path=resolve_script_path(args.output_dir, args.script),
        tts_model=args.tts_model,
        preview_chapters=args.preview_chapters,
        preview_chunks=args.preview_chunks,
        dry_run=args.dry_run,
        keep_temp=args.keep_temp,
        preparation_was_preview=preparation_was_preview,
    )


_THINK_MODES = {"off": (False,), "on": (True,), "both": (False, True)}


def _models_without_thinking(args: argparse.Namespace) -> tuple[str, ...]:
    """Which requested models the provider says cannot think.

    Best-effort and Ollama-specific: a probe that cannot reach the server or
    does not recognise the provider simply reports nothing, so a down server
    delays the useful error to the run rather than aborting option-building.
    """

    if args.provider.strip().casefold() != "ollama":
        return ()
    try:
        from .preparation.providers import fetch_model_capabilities
    except ImportError:
        return ()
    blocked: list[str] = []
    for model in args.models:
        capabilities = fetch_model_capabilities(
            args.provider_base_url, model, timeout=min(args.provider_timeout, 15.0)
        )
        # An empty set means the probe could not answer; only exclude a model
        # when the server answered and thinking was absent from the list.
        if capabilities and "thinking" not in capabilities:
            blocked.append(model)
    return tuple(blocked)


def _benchmark_options(args: argparse.Namespace) -> BenchmarkOptions:
    think_modes = _THINK_MODES[args.think]
    no_think_models: tuple[str, ...] = ()
    if True in think_modes:
        no_think_models = _models_without_thinking(args)
        if no_think_models:
            print(
                "Skipping the thinking run for models without a thinking "
                f"capability: {', '.join(no_think_models)}.",
                file=sys.stderr,
            )
    return BenchmarkOptions(
        output_dir=args.output_dir or default_output_dir(OUTPUT_FOLDER),
        provider_name=args.provider,
        models=tuple(args.models),
        base_url=args.provider_base_url,
        timeout_seconds=args.provider_timeout,
        repetitions=args.repetitions,
        corpus_dir=args.corpus_dir,
        tiers=tuple(args.tiers),
        categories=tuple(args.categories),
        case_ids=tuple(args.case_ids),
        quick=args.quick,
        think_modes=think_modes,
        no_think_models=no_think_models,
        prompt_version=args.prompt_version,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.command == "check":
            from .preflight import format_report, passed, run_preflight

            results = run_preflight()
            print(format_report(results))
            if not passed(results):
                raise SystemExit(1)
            return
        if args.command == "benchmark":
            report = benchmark_preparation(_benchmark_options(args), progress=benchmark_progress)
            print_summary(report)
            errored = sum(item.errored_cases for item in report.models_reports)
            if errored == len(report.runs):
                raise RuntimeError("Every benchmark case failed to run")
            return
        if args.command == "prepare":
            prepare_narration_script(_preparation_options(args))
            return
        if args.command == "narrate":
            narrate_prepared_script(_narration_options(args))
            return

        preparation_options = _preparation_options(args)
        book = prepare_narration_script(preparation_options)
        narration_options = _narration_options(
            args,
            preparation_was_preview=(
                preparation_options.preview_chapters is not None
                or preparation_options.preview_units is not None
            ),
        )
        narrate_chapters(
            narration_chapters(book),
            narration_options,
            prepared_book=book,
        )
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
