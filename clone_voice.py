"""Preview a narrator voice by cloning sample narration with it.

The voice can be a designed voice folder or any audio file — point at a
recording and it is decoded, levelled and cloned as-is. Supplying the
recording's transcript upgrades the clone from timbre-only to timbre plus
prosody; without one it still sounds like the speaker, but reads in the model's
own cadence. Clips are written next to the reference.

    python clone_voice.py voices/Self.flac
    python clone_voice.py voices/Self.flac --ref-text "What I actually said."
    python clone_voice.py warm_male --text "Any sentence you want to hear."
"""

import argparse
from pathlib import Path

import soundfile as sf

from audiobook.config import (
    ACTIVE_VOICE,
    DEFAULT_SYNTHESIS_PROVIDER,
    LANGUAGE,
    VOICES_DIR,
)
from audiobook.synthesis.providers import (
    SynthesisUnavailableError,
    create_synthesis_provider,
    synthesis_descriptor,
)
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
        help="Designed voice name (warm_male) or audio file "
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
        default=None,
        help="Clone checkpoint to load; defaults to the configured backend's.",
    )
    return parser.parse_args()


def preview_dir_for(voice) -> Path:
    """Put previews under the voice folder, or beside a standalone recording."""

    if voice.source.name == "reference.wav":
        return voice.source.parent / "previews"
    return voice.source.parent / f"{voice.source.stem}_previews"


def main() -> None:
    args = parse_args()

    descriptor = synthesis_descriptor(DEFAULT_SYNTHESIS_PROVIDER)
    if not descriptor.supports_clone:
        raise SystemExit(f"The {descriptor.label} backend cannot clone voices.")
    provider = create_synthesis_provider(DEFAULT_SYNTHESIS_PROVIDER, clone_model=args.model)
    try:
        provider.check_available()
    except SynthesisUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        voice = resolve_voice(args.voice, voices_dir=args.voices_dir, ref_text=args.ref_text)
    except FileNotFoundError as exc:
        provider.close()
        raise SystemExit(str(exc)) from exc
    print(describe(voice))

    preview_dir = preview_dir_for(voice)
    preview_dir.mkdir(parents=True, exist_ok=True)
    passages = args.texts or DEFAULT_PASSAGES
    try:
        prompt = provider.clone(
            ref_audio=voice.audio,
            sample_rate=voice.sample_rate,
            ref_text=voice.ref_text,
        )
        for index, passage in enumerate(passages, start=1):
            print(f"Cloning passage {index}/{len(passages)}...")
            clip = provider.generate(text=passage, language=LANGUAGE, voice=prompt)
            out_path = preview_dir / f"preview_{index}.wav"
            sf.write(out_path, clip.audio, clip.sample_rate)
            print(f"  wrote {out_path}")
    finally:
        provider.close()

    print(f"\nPreviews for '{voice.slug}' are in {preview_dir}")


if __name__ == "__main__":
    main()
