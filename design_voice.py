"""Design a narrator voice from a natural-language description.

Renders a short reference clip with the Qwen3-TTS VoiceDesign model and stores it
under ``voices/<name>/`` (a ``reference.wav`` plus ``reference.json``). Preview
the result with ``clone_voice.py`` and select it for book runs by setting
``ACTIVE_VOICE`` in ``src/audiobook/config.py``.

    python design_voice.py warm_male
    python design_voice.py gentle_reader --instruct "A soft-spoken female narrator..."
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
    LOCAL_VOICE_DESIGN_MODEL_PATH,
    VOICE_DESIGN_INSTRUCT,
    VOICE_DESIGN_MODEL,
    VOICE_REFERENCE_AUDIO_FILENAME,
    VOICE_REFERENCE_METADATA_FILENAME,
    VOICE_REFERENCE_TEXT,
    VOICES_DIR,
)
from audiobook.synthesis.qwen import design_reference_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "name", help="Voice name; saved to voices/<name>/ (e.g. warm_male)."
    )
    parser.add_argument(
        "--instruct",
        default=VOICE_DESIGN_INSTRUCT,
        help="Natural-language persona. Because cloning carries prosody from the "
        "clip, describe both who the narrator is and how they read.",
    )
    parser.add_argument(
        "--ref-text",
        default=VOICE_REFERENCE_TEXT,
        help="Sentence(s) read aloud to create the reference clip.",
    )
    parser.add_argument(
        "--model",
        default=str(
            LOCAL_VOICE_DESIGN_MODEL_PATH
            if LOCAL_VOICE_DESIGN_MODEL_PATH.exists()
            else VOICE_DESIGN_MODEL
        ),
    )
    parser.add_argument("--voices-dir", type=Path, default=VOICES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Voice design requires a CUDA GPU.")

    print(f"Loading {args.model} on {torch.cuda.get_device_name(0)}...")
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=torch.bfloat16
    )

    print(f"Designing voice '{args.name}'...")
    audio, sample_rate = design_reference_clip(
        model,
        ref_text=args.ref_text,
        instruct=args.instruct,
        language=LANGUAGE,
    )

    voice_dir = args.voices_dir / args.name
    voice_dir.mkdir(parents=True, exist_ok=True)
    sf.write(voice_dir / VOICE_REFERENCE_AUDIO_FILENAME, audio, sample_rate)
    (voice_dir / VOICE_REFERENCE_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "slug": args.name,
                "instruct": args.instruct,
                "ref_text": args.ref_text,
                "sample_rate": sample_rate,
                "design_model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved voice to {voice_dir}")
    print(f"Preview it:  python clone_voice.py {args.name}")


if __name__ == "__main__":
    main()
