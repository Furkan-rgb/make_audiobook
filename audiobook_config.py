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
DEFAULT_PREPARATION_MODEL = "gemma4:31b"
DEFAULT_PROVIDER_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_PROVIDER_TIMEOUT_SECONDS = 600.0

# Qwen3-TTS
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
LOCAL_TTS_MODEL_PATH = Path("models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
VOICE_NAME = "Aiden"
LANGUAGE = "English"
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
