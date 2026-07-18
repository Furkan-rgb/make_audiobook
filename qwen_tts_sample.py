"""Generate a small Qwen3-TTS audiobook voice comparison."""

import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from make_audiobook import (
    LOCAL_MODEL_PATH,
    NARRATION_INSTRUCTION,
    TTS_MODEL,
    VOICE_NAME,
)

PREVIOUS_CONTEXT = "Daniel had spent the night considering whether to leave."
TEXT_TO_NARRATE = (
    "By morning, the decision no longer seemed complicated. The house was "
    "silent, and pale light crossed the hallway floor. He picked up the suitcase "
    "without looking back."
)
FOLLOWING_CONTEXT = "Outside, Sarah was already waiting beside the car."


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=[VOICE_NAME],
        help="CustomVoice speakers to render (for example: Aiden Ryan).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("audiobook_output/voice_samples"),
    )
    parser.add_argument(
        "--model",
        default=str(LOCAL_MODEL_PATH if LOCAL_MODEL_PATH.exists() else TTS_MODEL),
        help="A Hugging Face model id or downloaded model directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-TTS sample generation requires a CUDA GPU.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model} on {torch.cuda.get_device_name(0)}...")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    supported_speakers = {speaker.lower() for speaker in model.get_supported_speakers()}
    for speaker in args.speakers:
        if speaker.lower() not in supported_speakers:
            raise ValueError(f"Unsupported speaker: {speaker}")

        print(f"Generating {speaker} sample...")
        wavs, sample_rate = model.generate_custom_voice(
            text=TEXT_TO_NARRATE,
            language="English",
            speaker=speaker,
            instruct=NARRATION_INSTRUCTION,
        )
        output_path = args.output_dir / f"qwen3_tts_{speaker.lower()}.wav"
        sf.write(output_path, wavs[0], sample_rate)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
