# Narration-preparation model benchmarks

Preparation models are scored against a committed **gold corpus**: passages
whose correct preparation is known in advance, so a run reports which edits a
model got right rather than how much text it changed. This document describes
how that works and why. Raw run artifacts are written under
`output/benchmarks/` and are intentionally not tracked by Git.

## Why a gold corpus

A narration-preparation model is shown a passage and asked for the small edits
that make it listenable — a bibliographic citation removed, `§5` spoken as
"section 5", an extraction artifact repaired — and nothing else. The
[contract](../src/audiobook/preparation/prompting.py) is edits-only: the model
never rewrites the passage, so the prepared text is the source with a handful of
spans spliced and everything else byte-identical.

The first benchmark measured each model against the *source*: lexical retention,
similarity, citation-shape reduction, expansion ratio. Every one of those
answers "how much did the model change?", and on real book prose the answer is
"very little" for every model. In the 2026-07-18 Gemma 4 run below, three models
scored 99.7–99.8% lexical retention and 100% repeat consistency — while manual
review found one of them had changed "relational" to "non-relational" and
dropped a word from a phrase. The automatic metrics were blind to it because a
one-word substitution is, by any similarity measure, almost no change at all.

The fix is ground truth. Each case in the corpus carries the passage, the edits
it requires, and the exact text those edits produce. Scoring diffs a model's
output against the source to recover the spans it actually changed, then checks
each span against the gold answer: a required edit made is a true positive, a
required edit missed is a false negative, and — the case that mattered — a change
the gold answer did not ask for is a false positive, even when it is a single
word.

## The corpus

Forty-eight cases live under
[`src/audiobook/benchmarking/cases/`](../src/audiobook/benchmarking/cases/), one
JSON file each. They are original pastiche rather than extracts from real books,
so the corpus is copyright-clean, committable, and able to plant a specific trap
exactly where it is wanted. Each case is a single normalized prose unit of
roughly 400–900 characters — short on purpose, so a full multi-model run is one
provider call per case and stays tractable.

A case names an `anchor` for each required edit (the smallest verbatim span to
change) and a list of `accept` wordings (`"§5"` may become `"section 5"` or
`"section five"`; the first is canonical). It may also name `traps`: spans that
must survive untouched, each with a label that turns a failure from "changed
characters 214–218" into "sprang `historical-year-must-stay`".

The four tiers answer different questions:

| Tier | n | What it tests |
|---|---:|---|
| `core` | 18 | The five real edit categories: author-year citations, numeric and footnote reference markers, visual notation (`§`, `Fig.`, `cf.`), list punctuation, and extraction artifacts (broken words, `DSM-11`→`DSM-II`, ligatures) |
| `noop` | 12 | Clean prose whose correct answer is an empty edit list — dialogue, memoir, prose name-lists, legitimate hyphenated compounds. The single most under-tested behaviour, and the one the prompt itself calls "a correct and common answer" |
| `trap` | 12 | A legitimate edit paired with adjacent bait: a removable citation beside a bare historical year, a hedge that must survive verbatim, dialogue quote marks, a meaning-bearing parenthetical |
| `robustness` | 6 | Prompt injection inside book text, summarization bait, and blunt, dated, or since-disproven material that a faithful narrator must not soften, modernize, or fact-check |

### Trust in the corpus

A benchmark is only worth running if every case in it is provably fair, so no
case is trusted on the author's word. [`lint_case`](../src/audiobook/benchmarking/corpus.py)
runs the gold edits through the *production applier* and requires that they
reproduce the case's `prepared` text exactly. That single check proves the gold
answer is reachable through the real edits-only contract — every anchor
resolves, no edit exceeds the per-edit size limit or crosses a paragraph break,
none overlaps another, and together they clear the retention floor. It also
requires that the source already be in normalized form (so a case never tests
text no model would see) and that it segment to exactly one prose unit. The
whole corpus is linted on load, and a test fails if any case drifts.

## Scoring

Each case is prepared exactly as production would: a fresh provider request with
no cache and no resume, applied by the same applier and checked by the same
validation policy. A model is scored on the prose a listener would actually have
heard, not on the JSON it emitted. Per case:

- **Fidelity** — no changed span outside an expected one. A substantive
  unrequested change (one that alters words, as opposed to cosmetic whitespace
  or punctuation) is a hard failure: the case scores zero regardless of anything
  else it got right. This is the metric the old benchmark lacked.
- **Recall** — the share of required edits the model made.
- **Precision** — true positives over true positives plus unrequested changes.
- **Exactness** — of the edits made, how many produced an accepted wording.
- **Composite score** — `0.5 × recall + 0.3 × precision + 0.2 × exactness`,
  forced to zero on a fidelity failure. Recall leads because coverage is the
  point; fidelity gates because a changed word is not redeemable by volume.

Alongside these the report tracks **contract compliance** (edits that could not
be anchored, were ambiguous, were oversized rewrites, or targeted the numbered
view's own labels — prompt problems rather than judgement problems),
**determinism** (agreement between repetitions, compared as edit sets so a model
that proposes nothing does not score a perfect 100%), and **cost** (wall seconds
per case).

Models are ranked by fidelity failures first, then by composite score. The
report leads with a leaderboard, breaks the score down by tier and by category,
lists which named traps each model sprang, and ends with a failure appendix that
shows every non-passing case as a diff between the gold text and the model's
output. The appendix is the part that changes anyone's mind about a model.

## Running it

```bash
.venv/bin/audiobook benchmark \
  --models gemma4:12b gemma4:26b gemma4:31b \
  --repetitions 2
```

Useful flags: `--tier`, `--category`, and `--case` narrow a run to a slice of the
corpus; `--quick` runs a balanced three-cases-per-tier smoke subset; `--corpus-dir`
points at an alternative corpus. The command writes `comparison.md`,
`benchmark.json`, and a `plots/` directory of PNG charts to a timestamped
directory under `output/benchmarks/`.

The CLI is a thin wrapper: the runner is `run()` in
[`benchmarking/run.py`](../src/audiobook/benchmarking/run.py), which the
[`run_benchmark.py`](../run_benchmark.py) driver script at the repo root calls
with plain Python values instead of flags — edit its settings and run
`python run_benchmark.py`. Both paths build the same `BenchmarkOptions` and go
through the same `benchmark_preparation()`, the seam the tests drive with a
scripted in-memory provider.

### Sampling and seeds

Sampling is left to each model package. The benchmark sends no `temperature`,
`top_k`, `top_p`, `min_p`, `typical_p`, `presence_penalty`, `frequency_penalty`,
or `repeat_penalty`: each is omitted from the request entirely — not sent as
`null` — so Ollama falls back to the defaults the model ships with, and a model
is measured under its own policy rather than one the harness imposed. This is the
only sampling behaviour; there is no temperature-zero mode and no `--sampling`
flag. Normal production preparation uses the same native sampling — neither path
sends a sampling option — unless a caller deliberately overrides one.

Everything else that would otherwise make a comparison unfair stays pinned and
identical across models: the prompt and its version, the JSON schema, the
thinking mode, the context and output budgets (`num_ctx`/`num_predict`, floored
higher for thinking runs), and the seed. Seeds run as a deterministic sequence —
repetition 1 uses `42`, repetition 2 uses `43`, and so on — the same for every
model, so a repetition is reproducible while successive repetitions still test
whether a model stays safe across several native-sampling generations.
`benchmark.json` records that native sampling was used, which options were
omitted, the explicit provider options, and the seed for every run.

### Thinking

`--think both` scores each model twice, once direct and once with reasoning
enabled, and files the two as separate competitors (`gemma4:12b` and `gemma4:12b
+think`) so a single leaderboard answers whether thinking earns its cost;
`--think on` runs only the thinking pass. A reasoning model emits its chain
before the JSON answer, and the two share the context window and the output
budget, so a thinking run is given a larger floor for both automatically —
otherwise a long chain truncates the answer and loses the unit. Ollama reports a
model's capabilities, so a model that cannot think is skipped for the thinking
pass with a note rather than charged forty-eight identical errors. Thinking is
several times slower per case; the `speed.png` chart is where that cost is read
against whatever accuracy it bought.

### Plots

Each run draws three PNG charts under `plots/` with matplotlib, so they embed
directly in Markdown and on GitHub and need no renderer to view:

- `scores.png` ranks the composite score and colours any competitor with a
  fidelity failure, so a wrong-book result cannot hide behind a tall bar.
- `by-tier.png` breaks each competitor's score across the four tiers.
- `speed.png` plots mean seconds per case, fastest first.

---

# Historical record

The run below predates gold scoring. It is kept because its manual review is the
evidence that motivated the rewrite: the automatic metrics agreed with every
model equally, and a human caught the failure that decided the result. Its
numbers are not comparable to a gold-corpus run.

## Gemma 4 comparison — 2026-07-18

The benchmark compared the locally installed Ollama models using identical
source text and preparation settings.

### Test scope

| Setting | Value |
|---|---|
| Source | `reparative-therapy-nicolosipdf.pdf` |
| Chapters selected | 5 |
| Chapters represented before the unit cap | Preface; Introduction; Chapter 1; Chapter 2 |
| Prose units processed per run | 8 |
| Source characters per run | 18,856 |
| Repetitions per model | 2 |
| Preparation-cache reuse | Disabled |
| Temperature / seed | `0.0` / `42` |
| Output contract | Structured JSON |

The eight-unit global cap represented one Preface unit, two Introduction
units, two Chapter 1 units, and three Chapter 2 units. Every model received the
same units. Models were run in round-robin order and unloaded between runs.

### Results

| Model | Successful runs | Mean wall time | Mean provider time | Lexical retention | Citation-target similarity | Citation-shaped characters | Paragraphs preserved | Repeat consistency |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| `gemma4:12b` | 2/2 | 78.42 s | 77.99 s | 99.7% | 99.8% | 478 → 0 | Yes | 100.0% |
| `gemma4:26b` | 2/2 | 45.87 s | 45.44 s | 99.7% | 99.0% | 478 → 0 | Yes | 100.0% |
| `gemma4:31b` | 2/2 | 148.40 s | 147.98 s | 99.8% | 99.9% | 478 → 0 | Yes | 100.0% |

Timing is specific to the local Ollama installation and hardware (an RTX 4090).
In this run,
the 26B model was unexpectedly faster than 12B, so parameter count alone should
not be treated as a speed prediction.

### Manual quality review

| Model | Finding | Decision |
|---|---|---|
| `gemma4:12b` | Removed long bibliographic lists, preserved meaning and paragraph structure, and repaired obvious extraction artifacts. No substantive errors were found in the reviewed sample. | Recommended default |
| `gemma4:26b` | Removed citation years but retained long author-name lists in the Preface. It also changed “relational” to “non-relational” and dropped “time” from “enough time and education.” | Rejected despite fastest timing |
| `gemma4:31b` | Closely matched 12B and correctly repaired `DSM-11` to `DSM-II`, but missed one `fearful-ness` extraction artifact and took about 1.9 times as long as 12B. | Good but unnecessary for this task |

The automatic citation-shape metric reported complete removal for all three
models because author-only lists no longer matched the author-year pattern.
Manual review was therefore decisive in rejecting 26B. This is exactly the gap
the gold corpus closes: the "non-relational" substitution is now a fidelity
failure a machine detects, and the retained author lists are missed edits that
cost recall.

### Conclusion

`gemma4:12b` is the production default. Its prepared text was almost identical
to the 31B output, deterministic across both repetitions, and materially safer
than the 26B output. It also has the smallest model footprint, although it was
not the fastest model in this particular benchmark.
