import math
import os
import re
import shutil
import subprocess
import sys
import time

import nltk
import numpy as np
import pymupdf4llm
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
PDF_PATH = "reparative-therapy-nicolosipdf.pdf"
OUTPUT_FOLDER = "audiobook_output"
OUTPUT_FILENAME = "reparative_therapy_complete.m4b"  # .m4b is best for chapters

# Voice: am_michael, af_bella, etc.
VOICE_NAME = "am_michael"
SPEED = 1.1

# PREVIEW SETTINGS
PREVIEW_MODE = False
PREVIEW_CHAPTERS_LIMIT = 2

# AUDIO SETTINGS
SAMPLE_RATE = 24000
PAUSE_BETWEEN_SENTENCES = 0.3
PAUSE_BETWEEN_PARAGRAPHS = 0.8
MAX_BATCH_CHARS = 500

# --- REGEX SETUP ---
RE_BOLD = re.compile(r"\*{1,2}([^*]+)\*{1,2}")
RE_CODE = re.compile(r"`([^`]+)`")
RE_LINKS = re.compile(r"\[([^\]]+)\]\([^)]+\)")
RE_IMGS = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
RE_HYPHENS = re.compile(r"\s*-\s*\n\s*")
RE_PAGENUMS = re.compile(r"(\d+)\s*\n\s*(?=\S)")
RE_NEWLINES = re.compile(r"\n{3,}")
RE_WHITESPACE = re.compile(r"[ \t]+")
RE_FIGS = re.compile(r"\b(fig\.|figure|table)\s*\d+[.:]*", re.IGNORECASE)
RE_CITATIONS_PAREN = re.compile(r"\(\s*\d+\s*\)")
RE_CITATIONS_BRACKET = re.compile(r"\[\s*\d+\s*\]")
RE_CHAPTER_HEADER = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)


def verify_dependencies():
    """Ensure NLTK and FFmpeg are available."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("📥 Downloading NLTK data...")
        try:
            nltk.download("punkt_tab")
            nltk.download("punkt")
        except Exception:
            pass

    if not shutil.which("ffmpeg"):
        print("❌ CRITICAL: FFmpeg is missing.")
        print("   You strictly need FFmpeg to embed chapters.")
        sys.exit(1)


def clean_text_segment(text):
    text = RE_BOLD.sub(r"\1", text)
    text = RE_CODE.sub(r"\1", text)
    text = RE_LINKS.sub(r"\1", text)
    text = RE_IMGS.sub("", text)
    text = RE_HYPHENS.sub("", text)
    text = RE_PAGENUMS.sub(r"\1 ", text)
    text = RE_NEWLINES.sub("\n\n", text)
    text = RE_WHITESPACE.sub(" ", text)
    text = RE_FIGS.sub("", text)
    text = RE_CITATIONS_PAREN.sub("", text)
    text = RE_CITATIONS_BRACKET.sub("", text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(""", "'").replace(""", "'")
    return text.strip()


def parse_pdf_to_chapters(pdf_path):
    print(f"📖 Parsing PDF structure: {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False, margins=(50, 50, 50, 50))

    chapters = []
    lines = md_text.split("\n")
    current_title = "Intro / Front Matter"
    current_buffer = []

    for line in lines:
        match = RE_CHAPTER_HEADER.match(line)
        if match:
            if current_buffer:
                clean_content = clean_text_segment("\n".join(current_buffer))
                if len(clean_content) > 50:
                    chapters.append((current_title, clean_content))

            current_title = match.group(2).strip()
            current_buffer = []
        else:
            current_buffer.append(line)

    if current_buffer:
        clean_content = clean_text_segment("\n".join(current_buffer))
        if len(clean_content) > 50:
            chapters.append((current_title, clean_content))

    print(f"📑 Found {len(chapters)} sections.")
    return chapters


def batch_sentences(sentences, max_chars=MAX_BATCH_CHARS):
    batches = []
    current_batch = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_len = len(sentence)
        if sentence_len > max_chars:
            if current_batch:
                batches.append(" ".join(current_batch))
                current_batch = []
                current_length = 0
            batches.append(sentence)
        elif current_length + sentence_len + 1 > max_chars:
            if current_batch:
                batches.append(" ".join(current_batch))
            current_batch = [sentence]
            current_length = sentence_len
        else:
            current_batch.append(sentence)
            current_length += sentence_len + 1
    if current_batch:
        batches.append(" ".join(current_batch))
    return batches


def download_kokoro_models():
    import urllib.request

    model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    if not os.path.exists("kokoro-v1.0.onnx"):
        urllib.request.urlretrieve(model_url, "kokoro-v1.0.onnx")
    if not os.path.exists("voices-v1.0.bin"):
        urllib.request.urlretrieve(voices_url, "voices-v1.0.bin")
    return "kokoro-v1.0.onnx", "voices-v1.0.bin"


def create_ffmpeg_metadata(chapters_metadata, metadata_file):
    """Creates the FFmetadata file format."""
    with open(metadata_file, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        f.write("title=Audiobook Generated by Kokoro\n")
        f.write("artist=Kokoro AI\n\n")

        for title, start_ms, end_ms in chapters_metadata:
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_ms}\n")
            f.write(f"END={end_ms}\n")
            f.write(f"title={title}\n\n")


def run_audiobook_gen():
    verify_dependencies()

    # 1. Setup Model
    try:
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        if "CUDAExecutionProvider" in ort.get_available_providers():
            print("🚀 GPU DETECTED")
    except ImportError:
        sys.exit(1)

    model_path, voices_path = download_kokoro_models()
    kokoro = Kokoro(model_path, voices_path)

    # 2. Setup Directories
    chapters = parse_pdf_to_chapters(PDF_PATH)
    if PREVIEW_MODE:
        chapters = chapters[:PREVIEW_CHAPTERS_LIMIT]

    temp_dir = os.path.join(OUTPUT_FOLDER, "temp_parts")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    file_list_path = os.path.join(temp_dir, "files.txt")
    metadata_path = os.path.join(temp_dir, "metadata.txt")

    chapter_timings = []  # (title, start_ms, end_ms)
    wav_files = []

    current_time_ms = 0
    silence_sentence = np.zeros(int(SAMPLE_RATE * PAUSE_BETWEEN_SENTENCES), dtype=np.float32)
    silence_paragraph = np.zeros(int(SAMPLE_RATE * PAUSE_BETWEEN_PARAGRAPHS), dtype=np.float32)

    # 3. Generate Audio per Chapter
    print(f"\n🎙️ Generating {len(chapters)} chapters...")

    for i, (title, content) in enumerate(chapters):
        safe_name = f"part_{i:03d}.wav"
        wav_path = os.path.join(temp_dir, safe_name)

        print(f"  🔹 {title}")

        # Prepare Text
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 0]
        batches = []
        for p in paragraphs:
            for b in batch_sentences(nltk.sent_tokenize(p)):
                batches.append(b)

        if not batches:
            continue

        # Generate Audio to WAV
        with sf.SoundFile(wav_path, mode="w", samplerate=SAMPLE_RATE, channels=1) as f:
            for j, batch in enumerate(tqdm(batches, leave=False, unit="batch")):
                audio, _ = kokoro.create(text=batch, voice=VOICE_NAME, speed=SPEED, lang="en-us")
                if audio is not None:
                    f.write(audio)
                    f.write(silence_paragraph if j < len(batches) - 1 else silence_sentence)

        # Calculate Duration for Metadata
        info = sf.info(wav_path)
        duration_ms = int(info.duration * 1000)

        start_ms = current_time_ms
        end_ms = current_time_ms + duration_ms
        chapter_timings.append((title, start_ms, end_ms))

        current_time_ms = end_ms
        wav_files.append(safe_name)

    # 4. Create FFmpeg Input Files
    # a. File List for Concat
    with open(file_list_path, "w") as f:
        for wav in wav_files:
            # Escape path for ffmpeg
            f.write(f"file '{wav}'\n")

    # b. Metadata File
    create_ffmpeg_metadata(chapter_timings, metadata_path)

    # 5. Merge and Convert using FFmpeg
    final_output = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    print(f"\n🔄 Merging into single file with chapters: {final_output}")

    # FFmpeg command explanation:
    # -f concat: Use concatenation demuxer
    # -safe 0: Allow relative paths
    # -i file_list_path: The list of wavs
    # -i metadata_path: The chapter info
    # -map_metadata 1: Map metadata from input #1 (the text file)
    # -c:a aac -b:a 64k: Encode to AAC (m4b standard) at 64k bitrate (good for speech)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        file_list_path,
        "-i",
        metadata_path,
        "-map_metadata",
        "1",
        "-c:a",
        "aac",  # Use 'libmp3lame' here if you strictly want MP3
        "-b:a",
        "64k",  # Bitrate
        final_output,
    ]

    try:
        subprocess.run(cmd, check=True, cwd=temp_dir)  # Run inside temp dir to handle paths easily
        print("✅ Success! Cleaning up temp files...")
        shutil.rmtree(temp_dir)
        print(f"🎉 Audiobook ready: {final_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e}")


if __name__ == "__main__":
    run_audiobook_gen()
