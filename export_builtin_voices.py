"""Export the built-in Qwen3-TTS speakers as reference voices.

Built-in speakers (Aiden, Ryan, Serena, ...) live inside the CustomVoice
checkpoint as embeddings, so the voice library cannot see them.  This renders
each one reading ``VOICE_REFERENCE_TEXT`` and stores the clip under
``voices/<name>/`` — the same layout designed voices use — after which they
appear in the picker and narrate through the clone pipeline like any other
voice.

A clone of a rendered clip is a second-generation copy of the speaker: timbre
lands very close, but the delivery is fixed to this one clip and the
``instruct`` style hint no longer applies.  That is the same trade every
designed voice already makes, and in exchange the transcript matches the audio
exactly, which is the ideal case for prosody cloning.

    python export_builtin_voices.py                 # every speaker in the checkpoint
    python export_builtin_voices.py aiden serena    # just these
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

from audiobook.config import (
    DEFAULT_SYNTHESIS_PROVIDER,
    LANGUAGE,
    NARRATION_INSTRUCTION,
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
        "speakers",
        nargs="*",
        help="Built-in speaker names to export; defaults to the checkpoint's full roster.",
    )
    parser.add_argument(
        "--ref-text",
        default=VOICE_REFERENCE_TEXT,
        help="Sentence(s) each speaker reads to create their reference clip.",
    )
    parser.add_argument(
        "--instruction",
        default=NARRATION_INSTRUCTION,
        help="Style hint for the rendering; the clip carries this delivery into every later clone.",
    )
    parser.add_argument("--voices-dir", type=Path, default=VOICES_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render speakers whose voice folder already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    descriptor = synthesis_descriptor(DEFAULT_SYNTHESIS_PROVIDER)
    if not descriptor.supports_builtin_voice:
        raise SystemExit(f"The {descriptor.label} backend has no built-in speakers.")
    speakers = args.speakers or list(descriptor.builtin_voices)

    provider = create_synthesis_provider(DEFAULT_SYNTHESIS_PROVIDER)
    try:
        provider.check_available()
    except SynthesisUnavailableError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        for speaker in speakers:
            voice_dir = args.voices_dir / speaker
            if voice_dir.exists() and not args.force:
                print(f"Skipping '{speaker}' — {voice_dir} exists (use --force to re-render).")
                continue
            print(f"Rendering built-in speaker '{speaker}'...")
            clip = provider.generate(
                text=args.ref_text,
                language=LANGUAGE,
                voice=speaker,
                instruction=args.instruction,
            )
            voice_dir.mkdir(parents=True, exist_ok=True)
            sf.write(voice_dir / VOICE_REFERENCE_AUDIO_FILENAME, clip.audio, clip.sample_rate)
            (voice_dir / VOICE_REFERENCE_METADATA_FILENAME).write_text(
                json.dumps(
                    {
                        "slug": speaker,
                        "builtin_speaker": speaker,
                        "instruct": None,  # not a designed persona; see generation_instruction
                        "ref_text": args.ref_text,
                        "sample_rate": clip.sample_rate,
                        "custom_voice_model": provider.resident_checkpoint(),
                        "generation_instruction": args.instruction,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved voice to {voice_dir}")
    finally:
        provider.close()


if __name__ == "__main__":
    main()
