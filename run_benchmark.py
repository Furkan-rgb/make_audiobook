"""Score narration-preparation models against the gold corpus.

The file-first alternative to `audiobook benchmark`: edit the settings below and
run it. Both entry points call the same runner, so results are identical.

    .venv/bin/python run_benchmark.py

Results land in a timestamped directory under output/benchmarks/ — comparison.md,
benchmark.json, and a plots/ folder of SVG charts.
"""

from audiobook.benchmarking import run
from audiobook.preparation import DEFAULT_PROMPT_VERSION

# --- settings -------------------------------------------------------------
# Any model identifier the provider can serve. `ollama list` shows what is
# installed locally.
MODELS = (
    "gemma4:12b",
    "gemma4:26b",
    "gemma4:31b",
    "qwen3.6:27b",
    "qwen3.6:35b",
)

# Reasoning modes to score, each a separately ranked entry:
#   (False,)       direct only
#   (True,)        thinking only
#   (False, True)  both, to compare a model with and without thinking
THINK_MODES = (False, True)

# Times to run every model. Two or more enables the determinism metric
# (edit-set agreement across repetitions).
REPETITIONS = 2

PROVIDER = "ollama"
BASE_URL = "http://127.0.0.1:11434"
TIMEOUT_SECONDS = 300.0

# Optional narrowing — leave empty for the whole 48-case corpus.
TIERS = ()          # e.g. ("trap", "robustness")
CATEGORIES = ()     # e.g. ("bibliographic_citation",)
CASE_IDS = ()       # e.g. ("core-001-citation-author-year-list",)
QUICK = False       # True = three cases per tier, a fast smoke test

# Models the provider cannot run with thinking, to skip their thinking pass
# instead of recording it as errors. Ollama reports this, so this is usually
# left empty.
NO_THINK_MODELS = ()

# System prompt every model runs under. Defaults to the current version; pin an
# older one (e.g. "narration-preparation-v4") and run again to compare prompts,
# since the same corpus scores whichever prompt is sent.
PROMPT_VERSION = DEFAULT_PROMPT_VERSION

# None = output/benchmarks/<timestamp>. Set a path to pin the location.
OUTPUT_DIR = None
# --------------------------------------------------------------------------


if __name__ == "__main__":
    run(
        models=MODELS,
        think_modes=THINK_MODES,
        repetitions=REPETITIONS,
        provider=PROVIDER,
        base_url=BASE_URL,
        timeout=TIMEOUT_SECONDS,
        tiers=TIERS,
        categories=CATEGORIES,
        case_ids=CASE_IDS,
        quick=QUICK,
        no_think_models=NO_THINK_MODELS,
        prompt_version=PROMPT_VERSION,
        output_dir=OUTPUT_DIR,
    )
