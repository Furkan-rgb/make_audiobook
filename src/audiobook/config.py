"""Shared defaults for the modular audiobook workflow."""

from pathlib import Path

# Book and artifact locations.  The source may be any format the extraction
# backends support (PDF or EPUB); the default is only what the CLI and UI
# preselect when a file sits beside the project.
DEFAULT_BOOK_PATH = Path("book.pdf")
DEFAULT_PDF_PATH = DEFAULT_BOOK_PATH  # Retained for scripts using the old name.
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_FILENAME = "audiobook.m4b"
DEFAULT_PREVIEW_OUTPUT_FILENAME = "audiobook_preview.m4b"
DEFAULT_PREPARED_SCRIPT_FILENAME = "prepared_book.json"
DEFAULT_PREPARED_MARKDOWN_FILENAME = "prepared_book.md"

# Narration preparation.  Each provider adapter reads its own entry here to
# build the menu the frontend offers, so adding a model is a config edit rather
# than a code change.  Keep the lists to models you can actually reach: a local
# model that is not pulled yet is fetched automatically at preflight (unless
# auto_pull is off), but one that is not on your plan (or misspelled) only
# fails once extraction has already run.  API keys are never stored here — an
# adapter names the environment variable it reads, and the value stays in your
# shell.
PREPARATION_PROVIDERS = {
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "models": ("gemma4:31b",),
        "auto_pull": True,
        "think": False,
    },
}
DEFAULT_PREPARATION_PROVIDER = "ollama"
DEFAULT_PREPARATION_MODEL = PREPARATION_PROVIDERS["ollama"]["models"][0]
DEFAULT_PROVIDER_BASE_URL = PREPARATION_PROVIDERS["ollama"]["base_url"]
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 600.0

# Qwen3-TTS
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LOCAL_TTS_MODEL_PATH = Path("models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
# Fallback built-in speaker, reported by the backend descriptor only when the
# checkpoint's own roster cannot be read yet (i.e. before its first download).
VOICE_NAME = "Aiden"
LANGUAGE = "English"

# There is no backend switch: ACTIVE_VOICE below names the narrator, and the
# kind of voice it names decides the synthesis path.  A built-in speaker
# renders natively on the CustomVoice checkpoint; a designed voice or a
# recording narrates through the design-then-clone pipeline, where every book
# chunk is cloned from one reference clip so the voice stays consistent.

# Design-then-clone checkpoints (only needed for file-backed voices).
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LOCAL_VOICE_DESIGN_MODEL_PATH = Path("models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
VOICE_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LOCAL_VOICE_CLONE_MODEL_PATH = Path("models/Qwen3-TTS-12Hz-1.7B-Base")

# TTS synthesis providers.  Mirrors PREPARATION_PROVIDERS: each backend adapter
# reads its own entry to learn which checkpoints to load, so swapping Qwen for
# another TTS model is a new adapter plus an entry here rather than edits across
# the workflow.  Each role maps to (local checkpoint dir, Hugging Face id); the
# local copy is used when present, otherwise the id is downloaded on first use.
# The values reference the constants above so there is a single source of truth.
SYNTHESIS_PROVIDERS = {
    "qwen": {
        "design": (LOCAL_VOICE_DESIGN_MODEL_PATH, VOICE_DESIGN_MODEL),
        "clone": (LOCAL_VOICE_CLONE_MODEL_PATH, VOICE_CLONE_MODEL),
        "custom_voice": (LOCAL_TTS_MODEL_PATH, TTS_MODEL),
        "voice_name": VOICE_NAME,
    },
}
# TODO: Select a synthesis provider per capability instead of one default for
# everything.  Each adapter already declares what it serves (supports_design /
# supports_clone / supports_narrate on its descriptor), and every call site
# resolves the provider by name through the registry — so the remaining work
# is replacing this single default with a role map, e.g.
#     SYNTHESIS_ROLES = {"design": "qwen", "clone": "qwen", "narrate": "other"}
# validated against each provider's declared capabilities at preflight.  That
# would let one model design voices while another clones or narrates.
DEFAULT_SYNTHESIS_PROVIDER = "qwen"

# Natural-language persona the VoiceDesign model renders into the reference clip.
# Voice cloning carries prosody from the reference audio, so this description
# should capture both *who* the narrator is and *how* they read.
VOICE_DESIGN_INSTRUCT = (
    "A male audiobook narrator in his late forties. Low-to-mid pitch with warm "
    "chest resonance, clean and full-bodied with no breathiness, rasp, or vocal "
    "fry. Neutral American accent, crisp consonants, fully articulated word "
    "endings. Reads at a steady unhurried pace with even energy across the "
    "phrase, narrow pitch variation, and a gentle downward inflection at each "
    "sentence end. Plain narration, not dramatised, no character voices. "
    "Recorded close to a large-diaphragm microphone in a dry, acoustically "
    "treated studio: no reverb, no room tone, no background noise or music."
)
# Representative narration read aloud to create the reference clip. Cloning
# carries prosody from this audio, so match the book's register and give the
# clone varied cadence: a declarative, a longer subordinate clause, a comma
# list. Aim for ~15-20 seconds of plain narration with no dialogue.
VOICE_REFERENCE_TEXT = "At first, the village seemed quiet, almost ordinary. Then a warm breeze moved through the open window, carrying the scent of rain and wood smoke from the hills. Clara paused, listened, and smiled. Whatever waited beyond the road, she would meet it with patience, curiosity, and a steady voice."
# Narrator voices. ACTIVE_VOICE selects the narrator and accepts any of:
#   "warm_male"        — a designed voice in voices/<name>/ (design_voice.py)
#   "voices/Self.flac" — any audio file, e.g. a recording of your own voice
#   "Aiden"            — a built-in speaker of the synthesis backend, rendered
#                        natively rather than cloned; anything on disk with
#                        the same name shadows it
# A recording clones timbre on its own; to carry prosody as well, supply its
# transcript in a sidecar <stem>.txt or via clone_voice.py --ref-text.
VOICES_DIR = Path("voices")
ACTIVE_VOICE = "warm_male"
VOICE_REFERENCE_AUDIO_FILENAME = "reference.wav"
VOICE_REFERENCE_METADATA_FILENAME = "reference.json"
# Every reference is decoded to mono at this rate and scaled to this peak before
# cloning, so recordings and designed clips reach the model on equal terms.
REFERENCE_SAMPLE_RATE = 24000
REFERENCE_PEAK_DBFS = -3.0
# Silence in a reference is unaccounted for by its transcript, so the clone can
# learn it as part of the speaker's delivery. Gaps are trimmed at the ends and
# capped in the middle; the threshold is relative to the clip's own level, since
# a recording's "silence" is its noise floor rather than digital black.
# 30 dB, not librosa's 60: a room's tone can sit only ~34 dB below speech, and a
# threshold above that reads the room as quiet talking. Internal gaps are capped
# rather than removed, so erring aggressive here costs pause length, not words.
REFERENCE_TRIM_TOP_DB = 30.0
REFERENCE_TRIM_PAD_MS = 50
REFERENCE_MAX_INTERNAL_SILENCE_MS = 300
# Transcribing a recording lets it clone prosody, not just timbre. The result is
# written beside the audio as <stem>.txt so it can be corrected and reused.
REFERENCE_TRANSCRIBE = True
# Full large-v3 rather than -turbo: turbo's pruned decoder often drops
# punctuation, and an unpunctuated transcript teaches the clone flat prosody.
ASR_MODEL = "openai/whisper-large-v3"
ASR_SAMPLE_RATE = 16000
# Whisper invents fluent text on near-silent input. A transcript thinner than
# this many words per second of audio is treated as a hallucination.
ASR_MIN_WORDS_PER_SECOND = 0.5
NARRATION_INSTRUCTION = (
    "Warm, engaging professional audiobook narration. Read naturally and clearly "
    "at a relaxed pace, with expressive phrasing and emotion appropriate to the text. "
    "Maintain smooth continuity, use natural pauses and vary tone gently to keep the "
    "listening experience pleasant and immersive. Give dialogue subtle character "
    "distinction without exaggerated acting. Avoid monotony, rushed delivery and "
    "overly long pauses."
)


# Semantic narration chunks
MIN_CHUNK_CHARS = 300
TARGET_CHUNK_CHARS = 500
MAX_CHUNK_CHARS = 700
CONTEXT_CHARS = 240
TARGET_CHUNK_DURATION_SECONDS = (30.0, 90.0)

# Boundaries between separately generated TTS chunks
CHUNK_CROSSFADE_MS = 30
PARAGRAPH_SILENCE_MS = 150
SECTION_SILENCE_MS = 250
CHAPTER_SILENCE_MS = 500

# Final-output loudness. Chunk matching only evens out level drift between
# independently generated chunks within a chapter; the finished audiobook is
# then normalized once, with FFmpeg's measured two-pass EBU R128 loudnorm, so
# every book plays back at the same predictable level. These targets apply only
# to the final M4B, never to individual chunks.
OUTPUT_TARGET_LUFS = -23.0
OUTPUT_TRUE_PEAK_DBTP = -2.0
OUTPUT_TARGET_LRA = 7.0
