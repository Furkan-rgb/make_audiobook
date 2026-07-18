"""Build a chaptered audiobook with Qwen3-TTS and semantic narration chunks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
from tqdm import tqdm


# --- BOOK CONFIGURATION ---
PDF_PATH = Path("book.pdf")
OUTPUT_FOLDER = Path("audiobook_output")
OUTPUT_FILENAME = "audiobook.m4b"
PREVIEW_OUTPUT_FILENAME = "audiobook_preview.m4b"

# --- QWEN CONFIGURATION ---
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LOCAL_MODEL_PATH = Path("models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
VOICE_NAME = "Aiden"
LANGUAGE = "English"
NARRATION_INSTRUCTION = (
    "Professional audiobook narration. Calm, natural and measured. "
    "Keep a steady reading pace of approximately 125 to 140 words per minute. "
    "Do not slow down for names, dates or parenthetical citations. "
    "Maintain flowing continuity between sentences. Use restrained emotion "
    "and subtle dialogue differentiation. Avoid exaggerated pauses."
)

# These are soft text-size guides. Generated durations are recorded in the
# chunk manifest so they can be tuned toward the 30-90 second audio target.
MIN_CHUNK_CHARS = 300
TARGET_CHUNK_CHARS = 500
MAX_CHUNK_CHARS = 700
CONTEXT_CHARS = 240
TARGET_CHUNK_DURATION_SECONDS = (30.0, 90.0)

# A split inside one long paragraph is crossfaded directly. Natural paragraph
# and section boundaries receive a modest gap instead of sentence-level pauses.
CHUNK_CROSSFADE_MS = 30
PARAGRAPH_SILENCE_MS = 150
SECTION_SILENCE_MS = 250
CHAPTER_SILENCE_MS = 500

# --- TEXT REGEXES ---
RE_BOLD = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
RE_CODE = re.compile(r"`([^`]+)`")
RE_LINKS = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_IMGS = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
RE_HYPHENS = re.compile(r"\s*-\s*\n\s*")
RE_PAGENUMS = re.compile(r"(\d+)\s*\n\s*(?=\S)")
RE_NEWLINES = re.compile(r"\n{3,}")
RE_WHITESPACE = re.compile(r"[ \t]+")
RE_FIGS = re.compile(r"\b(fig\.|figure|table)\s*\d+[.:]*", re.IGNORECASE)
RE_CITATIONS_PAREN = re.compile(r"\(\s*\d+\s*\)")
RE_CITATIONS_BRACKET = re.compile(r"\[\s*\d+\s*\]")
RE_SCENE_BREAK = re.compile(r"^(?:(?:\*\s*){3,}|-{3,}|_{3,}|~{3,})$")
RE_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?])(?:[\"'”’)]*)\s+(?=[A-Z0-9\"'“‘(\[])"
)
RE_DIALOGUE = re.compile(r"^(?:[\"'“‘]|[-—]\s)")
RE_NUMBERED_CHAPTER = re.compile(r"^\s*\d+\s+\S")
RE_PART_BOOKMARK = re.compile(r"^\s*[IVXLCDM]+\s+\S", re.IGNORECASE)
RE_STANDALONE_PAGE_NUMBER = re.compile(
    r"(?m)^[ \t]*#{0,6}[ \t]*(?:\d[ \t]*)+[ \t]*$"
)

@dataclass
class TextUnit:
    text: str
    boundary_after: str
    paragraph_index: int
    is_dialogue: bool


@dataclass
class TextSection:
    paragraphs: list[str]
    boundary_after: str = "section"


@dataclass
class NarrationChunk:
    text: str
    boundary_after: str
    previous_context: str = ""
    following_context: str = ""

    @property
    def char_count(self) -> int:
        return len(self.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_FOLDER)
    parser.add_argument(
        "--model",
        default=str(LOCAL_MODEL_PATH if LOCAL_MODEL_PATH.exists() else TTS_MODEL),
        help="Downloaded model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--preview-chapters",
        type=int,
        metavar="N",
        help="Generate only the first N detected chapters.",
    )
    parser.add_argument(
        "--preview-chunks",
        type=int,
        metavar="N",
        help="Generate only the first N semantic chunks across the book.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and display the chunk plan without loading Qwen.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated chapter WAV files after a successful merge.",
    )
    return parser.parse_args()


def verify_dependencies(require_tts: bool = True) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required to create the chaptered M4B file.")

    if require_tts:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-TTS is not installed. Run: .venv/bin/python -m pip "
                "install -r requirements.txt"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-TTS generation requires a CUDA GPU.")


def clean_text_segment(text: str) -> str:
    """Remove PDF/Markdown noise while preserving paragraph boundaries."""
    text = RE_IMGS.sub("", text)
    text = RE_LINKS.sub(r"\1", text)
    text = RE_BOLD.sub(r"\1", text)
    text = RE_CODE.sub(r"\1", text)
    text = RE_HYPHENS.sub("", text)
    text = RE_PAGENUMS.sub(r"\1 ", text)
    text = RE_FIGS.sub("", text)
    text = RE_CITATIONS_PAREN.sub("", text)
    text = RE_CITATIONS_BRACKET.sub("", text)
    text = RE_NEWLINES.sub("\n\n", text)
    text = RE_WHITESPACE.sub(" ", text)
    return text.strip()


def _join_markdown_pages(page_texts: Sequence[str]) -> str:
    """Join page-top sentence continuations without inventing a paragraph."""
    result = ""
    for page_text in page_texts:
        page_text = page_text.strip()
        if not page_text:
            continue
        if not result:
            result = page_text
            continue

        visible_start = page_text.lstrip("#*_ `\t")
        continues_sentence = bool(visible_start) and (
            visible_start[0].islower() or visible_start[0] in ",;:)”’"
        )
        if continues_sentence:
            page_text = re.sub(r"^#+[ \t]*", "", page_text, count=1)
            if result.endswith("-"):
                result = result[:-1] + page_text
            else:
                result += " " + page_text
        else:
            result += "\n\n" + page_text
    return result


def parse_pdf_to_chapters(pdf_path: Path) -> list[tuple[str, str]]:
    """Parse a PDF using embedded bookmarks, avoiding numbered body lists."""
    import pymupdf
    import pymupdf4llm

    print(f"Parsing PDF structure: {pdf_path}...")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document = pymupdf.open(pdf_path)
    bookmarks: list[tuple[str, int]] = []
    structural_bookmarks: list[tuple[str, int]] = []
    for _level, raw_title, page_number in document.get_toc():
        title = " ".join(raw_title.split())
        if RE_NUMBERED_CHAPTER.match(title):
            bookmarks.append((title, page_number))
            structural_bookmarks.append((title, page_number))
        elif RE_PART_BOOKMARK.match(title):
            structural_bookmarks.append((title, page_number))

    if not bookmarks:
        print("No numbered chapter bookmarks detected; treating the PDF as one chapter.")
        markdown = pymupdf4llm.to_markdown(document, page_chunks=False)
        return [("Audiobook", clean_text_segment(markdown))]

    # Embedded bookmarks often omit preface/introduction entries. Include exact
    # standard-section headings found before the first numbered chapter.
    first_chapter_page = bookmarks[0][1]
    prelude_titles = {"preface", "introduction", "foreword", "prologue"}
    prelude_pages: dict[str, tuple[str, int]] = {}
    for page_index in range(first_chapter_page - 1):
        lines = [line.strip() for line in document[page_index].get_text().splitlines()]
        for line in lines[:8]:
            if line.lower() in prelude_titles:
                prelude_pages[line.lower()] = (line.title(), page_index + 1)
                break
    bookmarks.extend(prelude_pages.values())
    structural_bookmarks.extend(prelude_pages.values())

    def searchable(text: str) -> str:
        return "".join(character.lower() for character in text if character.isalnum())

    def align_to_heading(title: str, page_hint: int) -> int:
        needle = searchable(title)
        candidate_pages = [page_hint]
        for offset in range(1, 4):
            candidate_pages.extend((page_hint - offset, page_hint + offset))
        for page_number in candidate_pages:
            if not 1 <= page_number <= document.page_count:
                continue
            if needle in searchable(document[page_number - 1].get_text()):
                return page_number
        return page_hint

    bookmarks = sorted(
        {(title, align_to_heading(title, page)) for title, page in bookmarks},
        key=lambda item: item[1],
    )
    structural_pages = sorted(
        {
            align_to_heading(title, page)
            for title, page in structural_bookmarks
        }
    )
    first_narrated_page = bookmarks[0][1]
    page_chunks = pymupdf4llm.to_markdown(
        document,
        pages=list(range(first_narrated_page - 1, document.page_count)),
        page_chunks=True,
    )
    text_by_page = {
        item["metadata"]["page_number"]: RE_STANDALONE_PAGE_NUMBER.sub("", item["text"])
        for item in page_chunks
    }

    chapters: list[tuple[str, str]] = []
    for title, start_page in bookmarks:
        next_boundaries = [page for page in structural_pages if page > start_page]
        end_page = next_boundaries[0] - 1 if next_boundaries else document.page_count
        markdown = _join_markdown_pages(
            [
                text_by_page.get(page_number, "")
                for page_number in range(start_page, end_page + 1)
            ]
        )
        content = clean_text_segment(markdown)
        spoken_title = re.sub(r"^\d+\s+", "", title).strip()
        if content.lower().startswith(spoken_title.lower()) and not content.startswith("#"):
            content = "# " + content
        if content:
            chapters.append((title, content))

    print(f"Detected {len(chapters)} chapters from PDF bookmarks.")
    return chapters


def _normalize_paragraph(block: str) -> str:
    block = block.strip()
    if block.startswith("#"):
        block = block.lstrip("# ")
    return " ".join(block.split())


def split_into_sections(content: str) -> list[TextSection]:
    """Split content at explicit scene breaks and Markdown subheadings."""
    sections: list[TextSection] = []
    current: list[str] = []
    pending_headings: list[str] = []

    def flush(boundary_after: str = "section") -> None:
        if current:
            sections.append(TextSection(current.copy(), boundary_after))
            current.clear()
        elif boundary_after == "scene" and sections:
            sections[-1].boundary_after = "scene"

    for raw_block in re.split(r"\n\s*\n", content):
        raw_block = raw_block.strip()
        if not raw_block:
            continue
        if RE_SCENE_BREAK.fullmatch(raw_block):
            flush("scene")
            pending_headings.clear()
            continue

        paragraph = _normalize_paragraph(raw_block)
        if not paragraph:
            continue
        is_heading = raw_block.startswith("#") and paragraph[0].isupper()
        if is_heading:
            flush()
            pending_headings.append(paragraph)
            continue

        if pending_headings:
            paragraph = "\n\n".join([*pending_headings, paragraph])
            pending_headings.clear()
        current.append(paragraph)

    if pending_headings:
        if current:
            current.extend(pending_headings)
        elif sections:
            sections[-1].paragraphs.extend(pending_headings)
        else:
            current.extend(pending_headings)
    flush()

    return sections


def sentence_split(text: str) -> list[str]:
    """Split only when a paragraph is too long, with a no-download fallback."""
    try:
        import nltk

        return [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    except (ImportError, LookupError):
        return [part.strip() for part in RE_SENTENCE_BOUNDARY.split(text) if part.strip()]


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Group whole sentences; an indivisible long sentence remains intact."""
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = sentence_split(paragraph)
    if len(sentences) <= 1:
        return [paragraph]

    parts: list[str] = []
    current: list[str] = []
    current_length = 0
    for sentence in sentences:
        proposed = current_length + len(sentence) + (1 if current else 0)
        if current and proposed > max_chars:
            parts.append(" ".join(current))
            current = [sentence]
            current_length = len(sentence)
        else:
            current.append(sentence)
            current_length = proposed
    if current:
        parts.append(" ".join(current))
    return parts


def _make_text_units(content: str, max_chars: int) -> list[TextUnit]:
    units: list[TextUnit] = []
    paragraph_index = 0
    for section in split_into_sections(content):
        section_start = len(units)
        for paragraph in section.paragraphs:
            parts = split_long_paragraph(paragraph, max_chars)
            for index, part in enumerate(parts):
                units.append(
                    TextUnit(
                        text=part,
                        boundary_after=(
                            "paragraph" if index == len(parts) - 1 else "continuation"
                        ),
                        paragraph_index=paragraph_index,
                        is_dialogue=bool(RE_DIALOGUE.match(part)),
                    )
                )
            paragraph_index += 1
        if len(units) > section_start:
            units[-1].boundary_after = section.boundary_after
    return units


def _join_units(units: Sequence[TextUnit]) -> str:
    pieces: list[str] = []
    for index, unit in enumerate(units):
        pieces.append(unit.text)
        if index < len(units) - 1:
            pieces.append(" " if unit.boundary_after == "continuation" else "\n\n")
    return "".join(pieces)


def _context_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[-limit:]
    first_space = clipped.find(" ")
    return clipped[first_space + 1 :] if first_space >= 0 else clipped


def _context_head(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    last_space = clipped.rfind(" ")
    return clipped[:last_space] if last_space >= 0 else clipped


def make_narration_chunks(
    content: str,
    min_chars: int = MIN_CHUNK_CHARS,
    target_chars: int = TARGET_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    context_chars: int = CONTEXT_CHARS,
) -> list[NarrationChunk]:
    """Create coherent chunks from sections, paragraphs, then sentences."""
    if min_chars <= 0 or target_chars < min_chars or target_chars > max_chars:
        raise ValueError(
            "Chunk sizes must satisfy 0 < min_chars <= target_chars <= max_chars"
        )

    units = _make_text_units(content, max_chars)
    chunks: list[NarrationChunk] = []
    current: list[TextUnit] = []

    def flush() -> None:
        if current:
            chunks.append(
                NarrationChunk(
                    text=_join_units(current),
                    boundary_after=current[-1].boundary_after,
                )
            )
            current.clear()

    for unit in units:
        if current:
            proposed_length = len(_join_units([*current, unit]))
            dialogue_exchange = current[-1].is_dialogue and unit.is_dialogue
            reached_target = len(_join_units(current)) >= target_chars
            if proposed_length > max_chars or (reached_target and not dialogue_exchange):
                flush()

        current.append(unit)
        if unit.boundary_after in {"section", "scene"}:
            flush()
    flush()

    # Avoid treating a short heading or final paragraph like an independent
    # recording. Merge it with a neighbor when the soft maximum and hard scene
    # boundaries allow it.
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if chunk.char_count >= min_chars or len(chunks) == 1:
            index += 1
            continue

        if index + 1 < len(chunks) and chunk.boundary_after != "scene":
            following = chunks[index + 1]
            separator = " " if chunk.boundary_after == "continuation" else "\n\n"
            combined_text = chunk.text + separator + following.text
            if len(combined_text) <= max_chars:
                chunks[index : index + 2] = [
                    NarrationChunk(combined_text, following.boundary_after)
                ]
                continue

        if index and chunks[index - 1].boundary_after != "scene":
            previous = chunks[index - 1]
            separator = " " if previous.boundary_after == "continuation" else "\n\n"
            combined_text = previous.text + separator + chunk.text
            if len(combined_text) <= max_chars:
                chunks[index - 1 : index + 1] = [
                    NarrationChunk(combined_text, chunk.boundary_after)
                ]
                index = max(0, index - 1)
                continue

        index += 1

    for index, chunk in enumerate(chunks):
        if index:
            chunk.previous_context = _context_tail(chunks[index - 1].text, context_chars)
        if index + 1 < len(chunks):
            chunk.following_context = _context_head(chunks[index + 1].text, context_chars)
    return chunks


def _crossfade(left: np.ndarray, right: np.ndarray, samples: int) -> np.ndarray:
    samples = min(samples, len(left), len(right))
    if samples <= 0:
        return np.concatenate((left, right))
    fade_in = np.linspace(0.0, 1.0, samples, endpoint=True, dtype=np.float32)
    overlap = left[-samples:] * (1.0 - fade_in) + right[:samples] * fade_in
    return np.concatenate((left[:-samples], overlap, right[samples:]))


def _fade_in(audio: np.ndarray, samples: int) -> np.ndarray:
    samples = min(samples, len(audio))
    if samples <= 0:
        return audio
    faded = audio.copy()
    ramp = np.linspace(0.0, 1.0, samples, endpoint=True, dtype=np.float32)
    faded[:samples] *= ramp
    return faded


def _fade_out(audio: np.ndarray, samples: int) -> np.ndarray:
    samples = min(samples, len(audio))
    if samples <= 0:
        return audio
    faded = audio.copy()
    ramp = np.linspace(1.0, 0.0, samples, endpoint=True, dtype=np.float32)
    faded[-samples:] *= ramp
    return faded


def assemble_chunk_audio(
    chunks: Sequence[NarrationChunk],
    audio_segments: Sequence[np.ndarray],
    sample_rate: int,
) -> np.ndarray:
    """Join generated requests without inserting sentence-level pauses."""
    if len(chunks) != len(audio_segments):
        raise ValueError("Each narration chunk must have one audio segment")
    if not audio_segments:
        return np.array([], dtype=np.float32)

    crossfade_samples = round(sample_rate * CHUNK_CROSSFADE_MS / 1000)
    result = np.asarray(audio_segments[0], dtype=np.float32).reshape(-1)
    for previous, segment in zip(chunks, audio_segments[1:]):
        following = np.asarray(segment, dtype=np.float32).reshape(-1)
        if previous.boundary_after == "continuation":
            result = _crossfade(result, following, crossfade_samples)
            continue

        silence_ms = (
            SECTION_SILENCE_MS
            if previous.boundary_after in {"section", "scene"}
            else PARAGRAPH_SILENCE_MS
        )
        result = _fade_out(result, crossfade_samples)
        following = _fade_in(following, crossfade_samples)
        silence = np.zeros(round(sample_rate * silence_ms / 1000), dtype=np.float32)
        result = np.concatenate((result, silence, following))
    return result


def load_qwen_model(model_name_or_path: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"Loading {model_name_or_path} on {torch.cuda.get_device_name(0)}...")
    return Qwen3TTSModel.from_pretrained(
        model_name_or_path,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )


def generate_chunk(model, chunk: NarrationChunk) -> tuple[np.ndarray, int]:
    """Speak only chunk.text; neighboring context intentionally stays metadata."""
    wavs, sample_rate = model.generate_custom_voice(
        text=chunk.text,
        language=LANGUAGE,
        speaker=VOICE_NAME,
        instruct=NARRATION_INSTRUCTION,
    )
    return np.asarray(wavs[0], dtype=np.float32).reshape(-1), sample_rate


def _escape_ffmetadata(value: str) -> str:
    return re.sub(r"([\\=;#])", r"\\\1", value).replace("\n", " ")


def create_ffmpeg_metadata(
    chapters_metadata: Sequence[tuple[str, int, int]],
    metadata_file: Path,
) -> None:
    with metadata_file.open("w", encoding="utf-8") as handle:
        handle.write(";FFMETADATA1\n")
        handle.write("title=Audiobook Generated with Qwen3-TTS\n")
        handle.write("artist=Qwen3-TTS / Aiden\n\n")
        for title, start_ms, end_ms in chapters_metadata:
            handle.write("[CHAPTER]\n")
            handle.write("TIMEBASE=1/1000\n")
            handle.write(f"START={start_ms}\n")
            handle.write(f"END={end_ms}\n")
            handle.write(f"title={_escape_ffmetadata(title)}\n\n")


def build_chunk_plan(
    chapters: Sequence[tuple[str, str]],
    preview_chunks: int | None = None,
) -> list[tuple[str, list[NarrationChunk]]]:
    plan: list[tuple[str, list[NarrationChunk]]] = []
    remaining = preview_chunks
    for title, content in chapters:
        chunks = make_narration_chunks(content)
        if remaining is not None:
            chunks = chunks[:remaining]
            remaining -= len(chunks)
        if chunks:
            plan.append((title, chunks))
        if remaining is not None and remaining <= 0:
            break
    return plan


def display_chunk_plan(plan: Sequence[tuple[str, Sequence[NarrationChunk]]]) -> None:
    total_chunks = sum(len(chunks) for _, chunks in plan)
    print(f"Planned {total_chunks} chunks across {len(plan)} chapters.")
    for title, chunks in plan:
        sizes = [chunk.char_count for chunk in chunks]
        oversized = sum(size > MAX_CHUNK_CHARS for size in sizes)
        print(
            f"  {title}: {len(chunks)} chunks, "
            f"{min(sizes)}-{max(sizes)} chars"
            + (f", {oversized} indivisible long sentences" if oversized else "")
        )


def merge_chapters(
    temp_dir: Path,
    wav_files: Sequence[str],
    chapters_metadata: Sequence[tuple[str, int, int]],
    output_path: Path,
) -> None:
    file_list = temp_dir / "files.txt"
    metadata_path = temp_dir / "metadata.txt"
    with file_list.open("w", encoding="utf-8") as handle:
        for wav_file in wav_files:
            handle.write(f"file '{wav_file}'\n")
    create_ffmpeg_metadata(chapters_metadata, metadata_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        file_list.name,
        "-i",
        metadata_path.name,
        "-map_metadata",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(output_path.resolve()),
    ]
    subprocess.run(command, check=True, cwd=temp_dir)


def run_audiobook_generation(args: argparse.Namespace) -> Path | None:
    verify_dependencies(require_tts=not args.dry_run)
    chapters = parse_pdf_to_chapters(args.pdf)
    if args.preview_chapters is not None:
        if args.preview_chapters <= 0:
            raise ValueError("--preview-chapters must be positive")
        chapters = chapters[: args.preview_chapters]
    if args.preview_chunks is not None and args.preview_chunks <= 0:
        raise ValueError("--preview-chunks must be positive")

    plan = build_chunk_plan(chapters, args.preview_chunks)
    if not plan:
        raise RuntimeError("No narratable text was found in the selected chapters.")
    display_chunk_plan(plan)
    if args.dry_run:
        return None

    model = load_qwen_model(args.model)
    supported_speakers = {speaker.lower() for speaker in model.get_supported_speakers()}
    if VOICE_NAME.lower() not in supported_speakers:
        raise RuntimeError(f"{VOICE_NAME} is not supported by the selected model.")

    temp_dir = args.output_dir / "temp_parts"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    chapter_timings: list[tuple[str, int, int]] = []
    wav_files: list[str] = []
    manifest: list[dict] = []
    current_time_ms = 0
    global_chunk_index = 0
    sample_rate: int | None = None

    print(f"Generating {sum(len(chunks) for _, chunks in plan)} chunks with {VOICE_NAME}...")
    for chapter_index, (title, chunks) in enumerate(plan):
        audio_segments: list[np.ndarray] = []
        for chapter_chunk_index, chunk in enumerate(
            tqdm(chunks, desc=title, leave=False, unit="chunk")
        ):
            audio, generated_rate = generate_chunk(model, chunk)
            if sample_rate is None:
                sample_rate = generated_rate
            elif generated_rate != sample_rate:
                raise RuntimeError("Qwen returned inconsistent sample rates.")

            duration_seconds = len(audio) / generated_rate
            audio_segments.append(audio)
            item = asdict(chunk)
            item.update(
                {
                    "chapter": title,
                    "chapter_index": chapter_index,
                    "chunk_index": global_chunk_index,
                    "chapter_chunk_index": chapter_chunk_index,
                    "char_count": chunk.char_count,
                    "duration_seconds": round(duration_seconds, 3),
                    "duration_target_met": (
                        TARGET_CHUNK_DURATION_SECONDS[0]
                        <= duration_seconds
                        <= TARGET_CHUNK_DURATION_SECONDS[1]
                    ),
                }
            )
            manifest.append(item)
            global_chunk_index += 1

        if sample_rate is None:
            continue
        chapter_audio = assemble_chunk_audio(chunks, audio_segments, sample_rate)
        chapter_audio = np.concatenate(
            (
                chapter_audio,
                np.zeros(round(sample_rate * CHAPTER_SILENCE_MS / 1000), dtype=np.float32),
            )
        )
        wav_name = f"part_{chapter_index:03d}.wav"
        sf.write(temp_dir / wav_name, chapter_audio, sample_rate)

        duration_ms = round(len(chapter_audio) / sample_rate * 1000)
        chapter_timings.append((title, current_time_ms, current_time_ms + duration_ms))
        current_time_ms += duration_ms
        wav_files.append(wav_name)

    manifest_path = args.output_dir / "chunk_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "voice": VOICE_NAME,
                "instruction": NARRATION_INSTRUCTION,
                "chunks": manifest,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    is_preview = args.preview_chapters is not None or args.preview_chunks is not None
    output_name = PREVIEW_OUTPUT_FILENAME if is_preview else OUTPUT_FILENAME
    output_path = args.output_dir / output_name
    print(f"Merging {len(wav_files)} chapters into {output_path}...")
    merge_chapters(temp_dir, wav_files, chapter_timings, output_path)
    if not args.keep_temp:
        shutil.rmtree(temp_dir)
    print(f"Audiobook ready: {output_path}")
    print(f"Chunk diagnostics: {manifest_path}")
    return output_path


def main() -> None:
    try:
        run_audiobook_generation(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
