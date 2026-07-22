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
from pathlib import Path

import soundfile as sf

from audiobook.config import (
    DEFAULT_SYNTHESIS_PROVIDER,
    LANGUAGE,
    VOICE_DESIGN_INSTRUCT,
    VOICE_REFERENCE_AUDIO_FILENAME,
    VOICE_REFERENCE_METADATA_FILENAME,
    VOICE_REFERENCE_TEXT,
    VOICES_DIR,
)
from audiobook.synthesis.providers import (
    SynthesisUnavailableError,
    create_synthesis_provider,
    synthesis_descriptor,
)


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
        default=None,
        help="Design checkpoint to load; defaults to the configured backend's.",
    )
    parser.add_argument("--voices-dir", type=Path, default=VOICES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    descriptor = synthesis_descriptor(DEFAULT_SYNTHESIS_PROVIDER)
    if not descriptor.supports_design:
        raise SystemExit(f"The {descriptor.label} backend cannot design voices.")
    provider = create_synthesis_provider(
        DEFAULT_SYNTHESIS_PROVIDER, design_model=args.model
    )
    try:
        provider.check_available()
    except SynthesisUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        print(f"Designing voice '{args.name}'...")
        clip = provider.design(
            persona=args.instruct,
            ref_text=args.ref_text,
            language=LANGUAGE,
        )
        design_model = args.model or provider.resident_checkpoint()
    finally:
        provider.close()

    voice_dir = args.voices_dir / args.name
    voice_dir.mkdir(parents=True, exist_ok=True)
    sf.write(voice_dir / VOICE_REFERENCE_AUDIO_FILENAME, clip.audio, clip.sample_rate)
    (voice_dir / VOICE_REFERENCE_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "slug": args.name,
                "instruct": args.instruct,
                "ref_text": args.ref_text,
                "sample_rate": clip.sample_rate,
                "design_model": design_model,
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
