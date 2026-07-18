"""Extract chapter-oriented narration text from PDF and Markdown input.

This module owns the deterministic cleanup that happens before any optional
model-assisted narration adaptation.  It deliberately preserves paragraph and
chapter structure so later workflow stages can make semantic decisions without
depending on PDF-specific details.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence


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
RE_NUMBERED_CHAPTER = re.compile(r"^\s*\d+\s+\S")
RE_PART_BOOKMARK = re.compile(r"^\s*[IVXLCDM]+\s+\S", re.IGNORECASE)
RE_STANDALONE_PAGE_NUMBER = re.compile(
    r"(?m)^[ \t]*#{0,6}[ \t]*(?:\d[ \t]*)+[ \t]*$"
)


def clean_text_segment(text: str) -> str:
    """Remove PDF and Markdown noise while preserving paragraph boundaries."""
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
    """Parse a PDF into chapters using bookmarks instead of numbered body text.

    Embedded bookmark page hints are aligned with nearby visible headings,
    structural part bookmarks delimit content without being narrated, and
    common unbookmarked front-matter sections are included when present.
    """
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
        {align_to_heading(title, page) for title, page in structural_bookmarks}
    )
    first_narrated_page = bookmarks[0][1]
    page_chunks = pymupdf4llm.to_markdown(
        document,
        pages=list(range(first_narrated_page - 1, document.page_count)),
        page_chunks=True,
    )
    text_by_page = {
        item["metadata"]["page_number"]: RE_STANDALONE_PAGE_NUMBER.sub(
            "", item["text"]
        )
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


__all__ = [
    "RE_BOLD",
    "RE_CITATIONS_BRACKET",
    "RE_CITATIONS_PAREN",
    "RE_CODE",
    "RE_FIGS",
    "RE_HYPHENS",
    "RE_IMGS",
    "RE_LINKS",
    "RE_NEWLINES",
    "RE_NUMBERED_CHAPTER",
    "RE_PAGENUMS",
    "RE_PART_BOOKMARK",
    "RE_STANDALONE_PAGE_NUMBER",
    "RE_WHITESPACE",
    "_join_markdown_pages",
    "clean_text_segment",
    "parse_pdf_to_chapters",
]
