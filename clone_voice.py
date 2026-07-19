"""Preview a narrator voice by cloning sample narration with it.

The voice can be a designed voice folder or any audio file — point at a
recording and it is decoded, levelled and cloned as-is. Supplying the
recording's transcript upgrades the clone from timbre-only to timbre plus
prosody; without one it still sounds like the speaker, but reads in the model's
own cadence. Clips are written next to the reference.

    python clone_voice.py voices/Self.flac
    python clone_voice.py voices/Self.flac --ref-text "What I actually said."
    python clone_voice.py warm_male_v2 --text "Any sentence you want to hear."
"""

import argparse
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from audiobook.config import (
    ACTIVE_VOICE,
    LANGUAGE,
    LOCAL_VOICE_CLONE_MODEL_PATH,
    VOICE_CLONE_MODEL,
    VOICES_DIR,
)
from audiobook.synthesis.qwen import build_voice_clone_prompt
from audiobook.synthesis.voices import describe, resolve_voice

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
        "voice",
        nargs="?",
        default=ACTIVE_VOICE,
        help="Designed voice name (warm_male_v2) or audio file "
        "(voices/Self.flac). Defaults to ACTIVE_VOICE.",
    )
    parser.add_argument(
        "--ref-text",
        help="Transcript of the reference audio, word for word. Carries the "
        "reference's prosody into the clone. Read from a sidecar <stem>.txt "
        "when present; omit entirely to clone timbre only.",
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


def preview_dir_for(voice) -> Path:
    """Put previews under the voice folder, or beside a standalone recording."""

    if voice.source.name == "reference.wav":
        return voice.source.parent / "previews"
    return voice.source.parent / f"{voice.source.stem}_previews"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Voice cloning requires a CUDA GPU.")

    try:
        voice = resolve_voice(
            args.voice, voices_dir=args.voices_dir, ref_text=args.ref_text
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(describe(voice))

    print(f"Loading {args.model} on {torch.cuda.get_device_name(0)}...")
    model = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=torch.bfloat16
    )
    prompt = build_voice_clone_prompt(
        model,
        ref_audio=voice.audio,
        sample_rate=voice.sample_rate,
        ref_text=voice.ref_text,
    )

    preview_dir = preview_dir_for(voice)
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

    print(f"\nPreviews for '{voice.slug}' are in {preview_dir}")


if __name__ == "__main__":
    main()
