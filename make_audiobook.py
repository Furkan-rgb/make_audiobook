"""CLI for the modular PDF → prepared script → Qwen3-TTS workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from audio_assembly import (
    _crossfade,
    _fade_in,
    _fade_out,
    assemble_chunk_audio,
    create_ffmpeg_metadata,
    merge_chapters,
)
from audiobook_config import (
    CHAPTER_SILENCE_MS,
    CHUNK_CROSSFADE_MS,
    CONTEXT_CHARS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_PDF_PATH,
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
from audiobook_workflow import (
    NarrationWorkflowOptions,
    PreparationWorkflowOptions,
    narrate_chapters,
    narrate_prepared_script,
    narration_chapters,
    prepare_narration_script,
    resolve_script_path,
)
from pdf_extraction import (
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
from qwen_tts_backend import generate_chunk, load_qwen_model
from semantic_chunking import (
    NarrationChunk,
    RE_DIALOGUE,
    RE_SCENE_BREAK,
    RE_SENTENCE_BOUNDARY,
    TextSection,
    TextUnit,
    _context_head,
    _context_tail,
    _join_units,
    _make_text_units,
    _normalize_paragraph,
    build_chunk_plan,
    display_chunk_plan,
    make_narration_chunks,
    sentence_split,
    split_into_sections,
    split_long_paragraph,
)


# Compatibility names retained for scripts that imported the former monolith.
PDF_PATH = DEFAULT_PDF_PATH
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
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit subcommands while retaining the old one-command syntax."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Extract and adapt a PDF into a reviewable narration script.",
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
        help="Prepare a PDF and narrate the resulting script.",
    )
    _add_output_argument(all_parser)
    _add_script_argument(all_parser)
    _add_preparation_arguments(all_parser)
    _add_narration_arguments(all_parser)
    _add_preview_chapters_argument(all_parser)

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        raw_args = ["all"]
    elif raw_args[0] not in {"prepare", "narrate", "all", "-h", "--help"}:
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
        pdf_path=args.pdf,
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
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
