# Modular audiobook workflow

Convert a PDF into a prepared narration script, review it, and then generate a
chaptered `.m4b` audiobook with Qwen3-TTS. The narrator can be a built-in
speaker or a bespoke voice you design yourself (see
[Custom narrator voice](#custom-narrator-voice)); runs default to the
designed `warm_male` voice.

The workflow deliberately separates editorial preparation from speech
generation:

```text
PDF extraction
    → deterministic text normalization
    → provider-neutral listening adaptation
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

The model-assisted operation is called **listening adaptation**; together with
deterministic normalization and validation it forms the broader narration
preparation stage. Gemma receives grouped prose units rather than isolated
paragraphs. Paragraph boundaries remain intact, headings and scene markers
bypass the model, and neighboring prose is supplied only as non-output context.

The default provider is local Ollama with `gemma4:12b`. Its policy explicitly
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
.venv/bin/python -m pip install -e .
.venv/bin/hf download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Start Ollama and install the default preparation model:

```bash
ollama serve
ollama pull gemma4:12b
```

The provider is configurable, so another installed Ollama model can be selected
with `--preparation-model`. Future hosted adapters implement the same
`NarrationPreparationProvider` interface; extraction, validation, artifacts,
chunking, and TTS do not depend on Ollama response types.

## Preparation-model benchmark

The manually reviewed results that informed the current 12B default are
recorded in
[`docs/preparation-model-benchmarks.md`](docs/preparation-model-benchmarks.md).

Compare the default Gemma variants on exactly the same normalized prose unit:

```bash
.venv/bin/audiobook benchmark \
  --pdf book.pdf \
  --models gemma4:12b gemma4:26b gemma4:31b \
  --preview-chapters 1 \
  --preview-units 1
```

Every model and repetition runs without preparation-cache reuse and is unloaded
afterward. The benchmark writes a timestamped directory under
`output/benchmarks/` containing:

- `comparison.md`, the human-review report;
- `benchmark.json`, machine-readable metrics and configuration;
- isolated `prepared_book.md` and `prepared_book.json` artifacts for every run.

Reported indicators include wall and provider time, lexical retention, minimum
unit retention, source similarity, similarity to a citation-stripped listening
target, citation-shaped character reduction,
paragraph-boundary preservation, edits, warnings, cross-model similarity, and
repeat consistency. These indicators catch obvious failures but do not replace
human review of names, dates, quotations, qualifications, and intended citation
removals.

Use `--repetitions 3` when measuring consistency or timing. Any future model
identifier can be supplied through `--models`; future provider adapters use the
same command with `--provider` and the provider-specific model identifiers.

## Recommended workflow

First prepare a small sample from the opening chapter:

```bash
.venv/bin/audiobook prepare \
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
.venv/bin/audiobook narrate \
  --script output/prepared_book.json \
  --dry-run
```

Generate a one-chunk audio preview:

```bash
.venv/bin/audiobook narrate \
  --script output/prepared_book.json \
  --preview-chunks 1
```

When the sample is satisfactory, prepare the complete book and then narrate it:

```bash
.venv/bin/audiobook prepare --pdf book.pdf
.venv/bin/audiobook narrate
```

Preparation resumes from compatible units in the existing JSON artifact. Use
`--force-preparation` only when every selected unit should be generated again.

## One-command workflow

`all` runs preparation and narration sequentially. Ollama is unloaded before
Qwen3-TTS is loaded so the two models do not compete for GPU memory.

```bash
.venv/bin/audiobook all --pdf book.pdf
```

For a fast end-to-end preview:

```bash
.venv/bin/audiobook all \
  --pdf book.pdf \
  --preview-units 1 \
  --preview-chunks 1
```

The original launcher and option-only form remain compatible and are treated
as `all`:

```bash
.venv/bin/python make_audiobook.py --pdf book.pdf
```

## Custom narrator voice

The narrator is chosen by `TTS_BACKEND` in `src/audiobook/config.py`:

- `custom_voice` — a built-in Qwen3-TTS speaker named by `VOICE_NAME`
  (Aiden, Ryan, Serena, …) on the CustomVoice checkpoint.
- `voice_clone` — a bespoke narrator built with the **design-then-clone**
  pipeline. The VoiceDesign model renders one reference clip from a
  natural-language description, and every book chunk is cloned from that clip so
  the voice stays perfectly consistent across the whole book. `ACTIVE_VOICE`
  selects which designed voice to use.

Designed voices live in `voices/<name>/` (a `reference.wav` plus its
`reference.json` recipe). This repo ships `warm_male`. To create your own:

```bash
# 1. design a voice from a description -> voices/gentle_reader/
.venv/bin/python design_voice.py gentle_reader \
  --instruct "A soft-spoken female narrator in her thirties, warm and unhurried, neutral American accent."

# 2. hear it on sample passages -> voices/gentle_reader/previews/
.venv/bin/python clone_voice.py gentle_reader

# 3. once happy, set ACTIVE_VOICE = "gentle_reader" in src/audiobook/config.py
```

Both steps need the VoiceDesign and Base checkpoints
(`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` and `-Base`), which download
automatically on first use or can be fetched with `hf download`.

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
src/audiobook/
├── cli.py                        # command-line interface
├── config.py                     # shared defaults
├── workflow.py                   # prepare/narrate orchestration
├── benchmarking/
│   └── preparation.py            # model timing, quality metrics, reports
├── extraction/
│   └── pdf.py                    # PDF bookmarks and extraction cleanup
├── preparation/
│   ├── normalization.py          # deterministic, idempotent cleanup
│   ├── segmentation.py           # provider-sized prose units and context
│   ├── prompting.py              # provider-neutral policy and JSON schema
│   ├── validation.py             # preservation safeguards
│   ├── artifacts.py              # hashes and atomic persistence
│   ├── pipeline.py               # cache, resume, and checkpoints
│   └── providers/
│       ├── base.py               # provider protocol and shared errors
│       ├── registry.py           # provider-neutral construction
│       └── ollama.py             # local Ollama adapter
├── chunking/
│   └── semantic.py               # coherent TTS request construction
├── synthesis/
│   └── qwen.py                   # Qwen3-TTS custom-voice and clone inference
└── assembly/
    └── audio.py                  # crossfades, WAVs, and M4B output

design_voice.py                   # design a narrator voice from a description
clone_voice.py                    # preview a designed voice on sample text
voices/<name>/                    # reference.wav + reference.json per voice
make_audiobook.py                 # legacy compatibility launcher
sample/qwen_tts_sample.py         # short built-in-speaker sample generator
tests/                            # focused module and workflow tests
```

Each domain package has a narrow API and can be tested without running later
stages. The preparation preview exercises extraction, normalization, listening
adaptation, validation, and artifact persistence. `narrate --dry-run` exercises
artifact loading and semantic chunking without loading Qwen, while
`--preview-chunks 1` isolates a single synthesis request and its audio output.

## Verification

Run the complete offline test suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The installed CLI can also be invoked as `.venv/bin/python -m audiobook`.

The virtual environment, downloaded models, input PDFs, generated audio,
prepared scripts, and voice-sample WAV files are ignored by Git.
