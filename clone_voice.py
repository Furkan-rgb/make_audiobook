"""Preview a designed voice by cloning sample narration with it.

Loads ``voices/<name>/reference.wav``, builds a reusable clone prompt with the
Qwen3-TTS Base model, and reads one or more passages so you can hear the voice
exactly as a book run would render it. Clips are written to
``voices/<name>/previews/``.

    python clone_voice.py warm_male
    python clone_voice.py warm_male --text "Any sentence you want to hear."
"""

import argparse
import json
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from audiobook.config import (
    LANGUAGE,
    LOCAL_VOICE_CLONE_MODEL_PATH,
    VOICE_CLONE_MODEL,
    VOICE_REFERENCE_AUDIO_FILENAME,
    VOICE_REFERENCE_METADATA_FILENAME,
    VOICES_DIR,
)
from audiobook.synthesis.qwen import build_voice_clone_prompt

DEFAULT_PASSAGES = [
    "By morning, the decision no longer seemed complicated. The house was "
    "silent, and pale light crossed the hallway floor. He picked up the "
    "suitcase without looking back.",
    "The argument that follows rests on a single, easily overlooked premise: "
    "that the mind and the body are not, in the end, separate accountants "
    "keeping separate books.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name", help="Voice name under voices/ (e.g. warm_male)."
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Passage to read (repeatable). Defaults to two sample passages.",
    )
    parser.add_argument("--voices-dir", type=Path, default=VOICES_DIR)
    parser.add_argument(
        "--model",
        default=str(
            LOCAL_VOICE_CLONE_MODEL_PATH
            if LOCAL_VOICE_CLONE_MODEL_PATH.exists()
            else VOICE_CLONE_MODEL
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Voice cloning requires a CUDA GPU.")

    voice_dir = args.voices_dir / args.name
    audio_path = voice_dir / VOICE_REFERENCE_AUDIO_FILENAME
    metadata_path = voice_dir / VOICE_REFERENCE_METADATA_FILENAME
    if not audio_path.exists() or not metadata_path.exists():
        raise SystemExit(
            f"No voice named '{args.name}' in {voice_dir}. Create it first with "
            f"'python design_voice.py {args.name}'."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ref_text = metadata["ref_text"]
    ref_audio, sample_rate = sf.read(audio_path)

    print(f"Loading {args.model} on {torch.cuda.get_device_name(0)}...")
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=torch.bfloat16
    )
    prompt = build_voice_clone_prompt(
        model,
        ref_audio=ref_audio,
        sample_rate=int(sample_rate),
        ref_text=ref_text,
    )

    preview_dir = voice_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    passages = args.texts or DEFAULT_PASSAGES
    for index, passage in enumerate(passages, start=1):
        print(f"Cloning passage {index}/{len(passages)}...")
        wavs, clone_rate = model.generate_voice_clone(
            text=passage,
            language=LANGUAGE,
            voice_clone_prompt=prompt,
        )
        out_path = preview_dir / f"preview_{index}.wav"
        sf.write(out_path, wavs[0], clone_rate)
        print(f"  wrote {out_path}")

    print(f"\nPreviews for '{args.name}' are in {preview_dir}")


if __name__ == "__main__":
    main()
