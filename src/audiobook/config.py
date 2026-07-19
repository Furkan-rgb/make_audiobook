"""Shared defaults for the modular audiobook workflow."""

from pathlib import Path


# Book and artifact locations
DEFAULT_PDF_PATH = Path("book.pdf")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_OUTPUT_FILENAME = "audiobook.m4b"
DEFAULT_PREVIEW_OUTPUT_FILENAME = "audiobook_preview.m4b"
DEFAULT_PREPARED_SCRIPT_FILENAME = "prepared_book.json"
DEFAULT_PREPARED_MARKDOWN_FILENAME = "prepared_book.md"

# Narration-preparation provider
DEFAULT_PREPARATION_PROVIDER = "ollama"
DEFAULT_PREPARATION_MODEL = "gemma4:12b"
DEFAULT_PROVIDER_BASE_URL = "http://127.0.0.1:11434"
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
    "A warm, middle-aged male audiobook narrator with a resonant, unhurried "
    "voice and quiet gravitas. Neutral American accent. Calm, measured, and "
    "authoritative, like a seasoned professional reader of serious non-fiction."
)
# Representative narration read aloud to create the reference clip. Keep it a
# couple of plain declarative sentences (~8-12 seconds) with no dialogue.
VOICE_REFERENCE_TEXT = (
    "By morning, the decision no longer seemed complicated. The house was "
    "silent, and pale light crossed the hallway floor as he considered what "
    "the day would ask of him."
)
# Designed narrator voices. Each voice lives in voices/<name>/ as a reference
# clip plus its metadata; ACTIVE_VOICE selects the one used by book runs.
VOICES_DIR = Path("voices")
ACTIVE_VOICE = "warm_male"
VOICE_REFERENCE_AUDIO_FILENAME = "reference.wav"
VOICE_REFERENCE_METADATA_FILENAME = "reference.json"
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
