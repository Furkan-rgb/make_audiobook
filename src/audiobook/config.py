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
        "models": ("gemma4:26b",),
        "auto_pull": True,
        "think": True,
    },
}
DEFAULT_PREPARATION_PROVIDER = "ollama"
DEFAULT_PREPARATION_MODEL = PREPARATION_PROVIDERS["ollama"]["models"][0]
DEFAULT_PROVIDER_BASE_URL = PREPARATION_PROVIDERS["ollama"]["base_url"]
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 600.0

# Qwen3-TTS
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LOCAL_TTS_MODEL_PATH = Path("models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
VOICE_NAME = "Aiden"
LANGUAGE = "English"

# Narrator backend:
#   "custom_voice" — a built-in speaker (VOICE_NAME) on the CustomVoice model.
#   "voice_clone"  — a bespoke narrator produced by the design-then-clone
#                    pipeline: the VoiceDesign model synthesizes one reference
#                    clip from VOICE_DESIGN_INSTRUCT, and every book chunk is
#                    cloned from that clip so the voice stays consistent.
TTS_BACKEND = "voice_clone"

# Design-then-clone checkpoints (only needed when TTS_BACKEND == "voice_clone").
VOICE_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LOCAL_VOICE_DESIGN_MODEL_PATH = Path("models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
VOICE_CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LOCAL_VOICE_CLONE_MODEL_PATH = Path("models/Qwen3-TTS-12Hz-1.7B-Base")

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
VOICE_REFERENCE_TEXT = (
    "The distinction matters more than it first appears. What separates the two "
    "cases is not scale, or cost, or even timing, but who is left to absorb the "
    "consequences. Once that question is asked plainly, the earlier arguments "
    "begin to look like descriptions of the same problem."
)
# Narrator voices. ACTIVE_VOICE selects the reference the clone model conditions
# on, and accepts either form:
#   "warm_male_v2"     — a designed voice in voices/<name>/ (design_voice.py)
#   "voices/Self.flac" — any audio file, e.g. a recording of your own voice
# A recording clones timbre on its own; to carry prosody as well, supply its
# transcript in a sidecar <stem>.txt or via clone_voice.py --ref-text.
VOICES_DIR = Path("voices")
ACTIVE_VOICE = "warm_male_v2"
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
ASR_MODEL = "openai/whisper-large-v3-turbo"
ASR_SAMPLE_RATE = 16000
# Whisper invents fluent text on near-silent input. A transcript thinner than
# this many words per second of audio is treated as a hallucination.
ASR_MIN_WORDS_PER_SECOND = 0.5
NARRATION_INSTRUCTION = (
    "Professional audiobook narration. Calm, natural and measured. "
    "Keep a steady reading pace of approximately 125 to 140 words per minute. "
    "Do not slow down for names, dates or parenthetical citations. "
    "Maintain flowing continuity between sentences. Use restrained emotion "
    "and subtle dialogue differentiation. Avoid exaggerated pauses."
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
