# make_audiobook

Turn a PDF book into a chapterized `.m4b` audiobook using the [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) neural text-to-speech model.

The script extracts the text from a PDF, detects chapters from its table of contents, narrates each chapter with Kokoro, and stitches everything into a single audiobook file with embedded chapter markers — so players can jump between chapters.

## Features

- **PDF → speech** — extracts text with [`pymupdf4llm`](https://pypi.org/project/pymupdf4llm/) and cleans up markdown artifacts, page numbers, citations, figure/table references, and hyphenated line breaks before narration.
- **Automatic chapter detection** — parses the table of contents (e.g. `Chapter 1: ...`, `Part I - ...`, `1.2 ...`) plus standard sections (Introduction, Preface, Bibliography, etc.). Falls back to a single chapter when no usable TOC is found.
- **Chapter markers in the output** — generates an FFmetadata file so the final `.m4b` has navigable chapters.
- **Natural pacing** — inserts short pauses between sentences and paragraphs.
- **GPU acceleration** — uses CUDA automatically when an ONNX Runtime GPU provider is available, otherwise runs on CPU.
- **Preview mode** — render just the first couple of chapters to test voice and settings before committing to a full run.

## Requirements

- **Python 3.9+**
- **[FFmpeg](https://ffmpeg.org/)** — must be installed and on your `PATH` (used to merge audio and embed chapters). The script exits with an error if it isn't found.
- Python packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The Kokoro ONNX model (`kokoro-v1.0.onnx`) and the voices file (`voices-v1.0.bin`) are downloaded automatically on first run if they aren't already present in the project directory.

> NLTK sentence-tokenizer data (`punkt` / `punkt_tab`) is also downloaded automatically the first time you run the script.

## Usage

1. Place your PDF in the project directory.
2. Open `kokoro_make_audiobook.py` and edit the configuration block near the top:

   ```python
   PDF_PATH = "book.pdf"            # your input PDF
   OUTPUT_FOLDER = "audiobook_output"
   OUTPUT_FILENAME = "audiobook.m4b"

   VOICE_NAME = "am_michael"        # e.g. am_michael, af_bella
   SPEED = 1.1                      # narration speed multiplier

   PREVIEW_MODE = False             # True = render only the first few chapters
   PREVIEW_CHAPTERS_LIMIT = 2
   ```

3. Run it:

   ```bash
   python kokoro_make_audiobook.py
   ```

The finished audiobook is written to `audiobook_output/<OUTPUT_FILENAME>`.

## Configuration reference

| Setting | Description | Default |
| --- | --- | --- |
| `PDF_PATH` | Path to the input PDF | `book.pdf` |
| `OUTPUT_FOLDER` | Directory for the output and temporary files | `audiobook_output` |
| `OUTPUT_FILENAME` | Name of the final audiobook (`.m4b` recommended for chapters) | `audiobook.m4b` |
| `VOICE_NAME` | Kokoro voice to narrate with | `am_michael` |
| `SPEED` | Narration speed multiplier | `1.1` |
| `PREVIEW_MODE` | Render only the first `PREVIEW_CHAPTERS_LIMIT` chapters | `False` |
| `PREVIEW_CHAPTERS_LIMIT` | Number of chapters rendered in preview mode | `2` |
| `SAMPLE_RATE` | Output sample rate (Hz) | `24000` |
| `PAUSE_BETWEEN_SENTENCES` | Silence between sentences (seconds) | `0.3` |
| `PAUSE_BETWEEN_PARAGRAPHS` | Silence between paragraphs (seconds) | `0.8` |
| `MAX_BATCH_CHARS` | Max characters per TTS batch | `500` |

To change the output to MP3 instead of `.m4b`/AAC, swap the FFmpeg encoder in the `cmd` list from `aac` to `libmp3lame` (note that chapter markers are best supported by `.m4b`).

## How it works

1. **Verify dependencies** — checks for FFmpeg and downloads NLTK tokenizer data if needed.
2. **Parse the PDF** (`parse_pdf_to_chapters`) — converts the PDF to markdown, detects chapter titles from the TOC, splits the body into chapters, and cleans each chapter's text.
3. **Batch & narrate** — splits chapters into sentence batches (capped at `MAX_BATCH_CHARS`) and synthesizes each batch with Kokoro, writing one WAV per chapter and tracking chapter durations.
4. **Build metadata** (`create_ffmpeg_metadata`) — writes an FFmetadata file with `START`/`END` timestamps per chapter.
5. **Merge** — FFmpeg concatenates the per-chapter WAVs, encodes to AAC at 64 kbps (good for speech), and embeds the chapter metadata into the final file. Temporary files are cleaned up on success.

## Project structure

```
make_audiobook/
├── kokoro_make_audiobook.py   # the full pipeline
├── requirements.txt           # Python dependencies
├── voices-v1.0.bin            # Kokoro voices (auto-downloaded if missing)
└── book.pdf                   # your input PDF (git-ignored; provide your own)
```

## Notes

- Input PDFs (`*.pdf`), the model file (`*.onnx`), and the `audiobook_output/` directory are git-ignored, so your source material and generated files stay out of version control.
- Generation is CPU/GPU intensive and can take a while for full-length books — use `PREVIEW_MODE` to dial in voice and speed first.

## Acknowledgements

- [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) for the speech model.
- [pymupdf4llm](https://github.com/pymupdf/RAG) for PDF-to-markdown extraction.
