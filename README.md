# Qwen3-TTS audiobook generator

This project converts the configured PDF into a chaptered M4B with
`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` and the Aiden voice.

## Setup

Python 3.12, FFmpeg, and a CUDA GPU are required. The current `.venv` and model
download are already prepared. For a fresh setup:

```bash
PYENV_VERSION=3.12.10 python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

The model directory, virtual environment, and generated audio are ignored by
Git.

## Usage

Inspect the PDF chapter and chunk plan without loading the TTS model:

```bash
.venv/bin/python make_audiobook.py --dry-run
```

Generate one end-to-end preview chunk:

```bash
.venv/bin/python make_audiobook.py --preview-chunks 1
```

Generate a chapter preview or the complete audiobook:

```bash
.venv/bin/python make_audiobook.py --preview-chapters 1
.venv/bin/python make_audiobook.py
```

Preview output is written to `audiobook_output/reparative_therapy_preview.m4b`.
The full run writes `audiobook_output/reparative_therapy_complete.m4b`.

To regenerate the short Aiden voice sample:

```bash
.venv/bin/python qwen_tts_sample.py
```

## Chunking

The chunker follows this hierarchy:

1. PDF bookmarks become audiobook chapters.
2. Explicit scene breaks and subheadings become section boundaries.
3. Complete paragraphs are combined into coherent chunks.
4. Oversized paragraphs are split only at sentence boundaries.
5. Adjacent dialogue paragraphs stay together where the soft maximum permits.

The current empirical target is 500 characters with a 700-character soft
maximum. It was tuned from the included Aiden preview to target approximately
30–90 seconds per request. Indivisible long sentences may exceed the soft
maximum.

Neighboring text is retained as non-spoken metadata in
`audiobook_output/chunk_manifest.json`. Only the chunk text and fixed narration
instruction are sent to Qwen. Continuations use a 30 ms crossfade; separate
paragraph and section requests receive 150 ms and 250 ms gaps respectively.
