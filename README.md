# Modular audiobook workflow

Convert a PDF into a prepared narration script, review it, and then generate a
chaptered `.m4b` audiobook with
[`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
and the calm Aiden voice.

The workflow deliberately separates editorial preparation from speech
generation:

```text
PDF extraction
    → deterministic text normalization
    → provider-neutral narration preparation
    → validated, reviewable prepared script
    → semantic narration chunks
    → Qwen3-TTS
    → chaptered M4B
```

## Narration preparation

Printed books often contain material that is useful on the page but unpleasant
to hear: long author-year citations, reference markers, footnotes, visual list
punctuation, broken line wrapping, and extraction artifacts. The preparation
stage adapts those presentation details for listening while preserving the
author's substantive prose.

The default provider is local Ollama with `gemma4:31b`. Its policy explicitly
forbids summarizing, censoring, softening, editorializing, modernizing, or adding
transitions. Structured responses contain:

- the prepared passage;
- an audit record of material edits;
- warnings for ambiguous cases.

Every result is checked for blank output, suspicious expansion, and low lexical
retention after citation-shaped spans are excluded from the comparison.
Headings and scene markers never enter the model request. Preparation is
checkpointed after every prose unit and compatible units are reused when a run
is resumed.

The canonical artifact is `output/prepared_book.json`. It contains source,
normalized, and prepared text; chapter and unit structure; SHA-256 integrity
hashes; provider/model/prompt metadata; edits; and warnings. A clean reading
copy is written beside it as `output/prepared_book.md`.

## Requirements

- Python 3.12
- FFmpeg on `PATH` for the final M4B stage
- A CUDA-capable GPU for Qwen3-TTS
- [Ollama](https://ollama.com/) for the default local preparation provider
- Disk space for the Ollama and Qwen3-TTS models

## Setup

Create the environment and download the TTS model from Hugging Face:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Start Ollama and install the default preparation model:

```bash
ollama serve
ollama pull gemma4:31b
```

The provider is configurable, so another installed Ollama model can be selected
with `--preparation-model`. Future hosted adapters implement the same
`NarrationPreparationProvider` interface; extraction, validation, artifacts,
chunking, and TTS do not depend on Ollama response types.

## Recommended workflow

First prepare a small sample from the opening chapter:

```bash
.venv/bin/python make_audiobook.py prepare \
  --pdf book.pdf \
  --preview-chapters 1 \
  --preview-units 1
```

Review:

```text
output/prepared_book.md
output/prepared_book.json
```

Inspect the semantic TTS plan without loading Qwen:

```bash
.venv/bin/python make_audiobook.py narrate \
  --script output/prepared_book.json \
  --dry-run
```

Generate a one-chunk audio preview:

```bash
.venv/bin/python make_audiobook.py narrate \
  --script output/prepared_book.json \
  --preview-chunks 1
```

When the sample is satisfactory, prepare the complete book and then narrate it:

```bash
.venv/bin/python make_audiobook.py prepare --pdf book.pdf
.venv/bin/python make_audiobook.py narrate
```

Preparation resumes from compatible units in the existing JSON artifact. Use
`--force-preparation` only when every selected unit should be generated again.

## One-command workflow

`all` runs preparation and narration sequentially. Ollama is unloaded before
Qwen3-TTS is loaded so the two models do not compete for GPU memory.

```bash
.venv/bin/python make_audiobook.py all --pdf book.pdf
```

For a fast end-to-end preview:

```bash
.venv/bin/python make_audiobook.py all \
  --pdf book.pdf \
  --preview-units 1 \
  --preview-chunks 1
```

The original option-only form remains compatible and is treated as `all`:

```bash
.venv/bin/python make_audiobook.py --pdf book.pdf
```

## Semantic TTS chunking

Prepared text follows this hierarchy:

1. PDF bookmark chapters
2. Scene or section boundaries
3. Complete paragraphs and related dialogue exchanges
4. Sentences, only when an oversized paragraph must be split

The current empirical target is 500 characters with a 700-character soft
maximum, tuned toward roughly 30–90 seconds of audio per Qwen request.
Neighboring text is retained as non-spoken metadata. Continuations use a 30 ms
crossfade; separately generated paragraphs and sections receive only small
boundary-sensitive gaps.

## Modules

```text
make_audiobook.py                 # CLI only
audiobook_config.py               # shared defaults
audiobook_workflow.py             # prepare/narrate orchestration
pdf_extraction.py                 # PDF bookmarks and extraction cleanup
narration_preparation/
├── normalization.py              # deterministic, idempotent cleanup
├── segmentation.py               # provider-sized prose units and context
├── prompting.py                  # provider-neutral policy and JSON schema
├── validation.py                 # preservation safeguards
├── artifacts.py                  # hashes, validation, atomic persistence
├── pipeline.py                   # cache, resume, and checkpoints
└── providers/
    ├── base.py                   # provider protocol and shared errors
    └── ollama.py                 # local Ollama adapter
semantic_chunking.py              # coherent TTS request construction
qwen_tts_backend.py               # Qwen3-TTS/Aiden inference
audio_assembly.py                 # crossfades, chapter WAVs, and M4B output
sample/qwen_tts_sample.py         # short voice sample generator
tests/                            # unit and workflow tests
```

## Verification

Run the complete offline test suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The virtual environment, downloaded models, input PDFs, generated audio,
prepared scripts, and voice-sample WAV files are ignored by Git.
