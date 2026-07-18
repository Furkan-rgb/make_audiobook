# Qwen3-TTS audiobook generator

Convert a PDF into a chaptered `.m4b` audiobook with
[`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
and the calm Aiden narration voice.

## Features

- Uses PDF bookmarks for reliable audiobook chapter markers.
- Builds coherent narration chunks from scenes, paragraphs, and dialogue.
- Keeps paragraphs intact where possible and splits only oversized paragraphs at
  sentence boundaries.
- Sends each semantic chunk as one Qwen request, with no fixed sentence pauses.
- Uses a 30 ms crossfade for continuations and small gaps at paragraph, section,
  and chapter boundaries.
- Records neighboring non-spoken context and generated duration diagnostics in
  `audiobook_output/chunk_manifest.json`.
- Supports dry runs, short chunk previews, and chapter previews before a full
  render.

## Requirements

- Python 3.12
- FFmpeg available on `PATH`
- A CUDA-capable GPU
- Enough disk space for the Qwen3-TTS 1.7B model

## Setup

Create the virtual environment, install the Python dependencies, and download
the model from Hugging Face:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

The virtual environment, downloaded models, input PDFs, and generated audio are
ignored by Git.

## Usage

Supply an input PDF with `--pdf`. First inspect its chapter and chunk plan
without loading the TTS model:

```bash
.venv/bin/python make_audiobook.py --pdf book.pdf --dry-run
```

Generate a one-chunk end-to-end voice and pacing preview:

```bash
.venv/bin/python make_audiobook.py --pdf book.pdf --preview-chunks 1
```

Generate the first chapter or the complete audiobook:

```bash
.venv/bin/python make_audiobook.py --pdf book.pdf --preview-chapters 1
.venv/bin/python make_audiobook.py --pdf book.pdf
```

By default, preview output is written to
`audiobook_output/audiobook_preview.m4b`; a full run writes
`audiobook_output/audiobook.m4b`.

To generate the short Aiden sample used while selecting the voice:

```bash
.venv/bin/python qwen_tts_sample.py
```

Run `make_audiobook.py --help` for model, output directory, preview, and
temporary-file options.

## Narration and chunking

The fixed instruction asks Aiden for professional audiobook narration that is
calm, natural, measured, and continuous, with restrained emotion and subtle
dialogue differentiation.

The chunker follows this hierarchy:

1. PDF bookmark chapters
2. Scene or section boundaries
3. Complete paragraphs and related dialogue exchanges
4. Sentences, only when a paragraph must be split

The current empirical target is 500 characters with a 700-character soft
maximum, tuned toward roughly 30–90 seconds of audio per Qwen request.
Indivisible long sentences may exceed the soft maximum. Neighboring text is
retained as metadata rather than injected into the spoken text.

## Project structure

```text
make_audiobook/
├── make_audiobook.py       # PDF parsing, semantic chunking, TTS, and M4B output
├── qwen_tts_sample.py      # short Aiden voice sample generator
├── test_make_audiobook.py  # chunking and audio-assembly tests
├── requirements.txt
└── book.pdf                # your local input (ignored by Git)
```
