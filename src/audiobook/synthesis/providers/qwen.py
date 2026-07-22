"""Qwen3-TTS backend adapter.

Qwen serves the three narration verbs from three separate checkpoints
(VoiceDesign, Base, CustomVoice) that do not fit in VRAM together, so this
adapter keeps exactly one resident and evicts on switch.  That single-slot
loader also replaces the four hand-rolled ``Qwen3TTSModel.from_pretrained``
call sites the workflow, UI and scripts used to each keep their own copy of.

Heavyweight imports (``torch``, ``qwen_tts``) stay inside the methods that need
them, so preparation and chunk planning still run on a machine without the TTS
runtime installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import (
    AudioClip,
    SynthesisDescriptor,
    SynthesisResponseError,
    SynthesisUnavailableError,
    VoiceInfo,
)


def _configured() -> dict[str, Any]:
    """Read this backend's entry from config at call time, not import time."""

    from ...config import SYNTHESIS_PROVIDERS

    return SYNTHESIS_PROVIDERS["qwen"]


def _resolve_checkpoint(spec: tuple[Path, str]) -> str:
    """Prefer a local checkpoint directory, falling back to a Hugging Face id."""

    local_path, remote_id = spec
    return str(local_path if Path(local_path).exists() else remote_id)


def _builtin_speaker_roster() -> tuple[str, ...]:
    """Speaker names baked into the CustomVoice checkpoint, without loading it.

    The roster lives in the checkpoint's ``config.json`` under
    ``talker_config.spk_id``, so a local copy answers from disk.  When only the
    Hugging Face id is configured the roster is unknown until download, and the
    single configured default is reported instead.
    """

    import json

    cfg = _configured()
    config_path = Path(cfg["custom_voice"][0]) / "config.json"
    try:
        talker = json.loads(config_path.read_text(encoding="utf-8"))["talker_config"]
        roster = tuple(sorted(talker["spk_id"]))
    except (OSError, KeyError, ValueError):
        return (cfg["voice_name"],)
    return roster or (cfg["voice_name"],)


def _as_mono_float32(wav: Any) -> np.ndarray:
    """Coerce a decoded Qwen waveform into a flat float32 mono array."""

    return np.asarray(wav, dtype=np.float32).reshape(-1)


class _QwenVoice:
    """Wraps a Qwen clone prompt so it satisfies the opaque ``Voice`` protocol."""

    __slots__ = ("prompt",)

    def __init__(self, prompt: Any) -> None:
        self.prompt = prompt


class _QwenBuiltinVoice:
    """A speaker baked into the CustomVoice checkpoint, addressed by name.

    Which of the two handle types ``load_voice`` returns is this adapter's
    business alone: callers pass either straight back to ``generate`` and the
    right checkpoint and call path are chosen here.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


class QwenSynthesisProvider:
    """Qwen3-TTS: three checkpoints, one resident at a time.

    Construct with no arguments to take the checkpoints from config, or override
    any role — the scripts pass a ``--model`` this way, and the workflow passes
    the CustomVoice checkpoint chosen on the command line.
    """

    def __init__(
        self,
        *,
        design_model: str | Path | None = None,
        clone_model: str | Path | None = None,
        custom_voice_model: str | Path | None = None,
        voice_name: str | None = None,
        device: str = "cuda:0",
    ) -> None:
        cfg = _configured()
        self._paths = {
            "design": (
                str(design_model)
                if design_model is not None
                else _resolve_checkpoint(cfg["design"])
            ),
            "clone": (
                str(clone_model) if clone_model is not None else _resolve_checkpoint(cfg["clone"])
            ),
            "custom_voice": (
                str(custom_voice_model)
                if custom_voice_model is not None
                else _resolve_checkpoint(cfg["custom_voice"])
            ),
        }
        self._voice_name = voice_name or cfg["voice_name"]
        self._device = device
        self._loaded: tuple[str, Any] | None = None
        self._verified_speakers: set[str] = set()

    # -- capability & availability --------------------------------------

    @classmethod
    def describe(cls) -> SynthesisDescriptor:
        return SynthesisDescriptor(
            name="qwen",
            label="Qwen3-TTS",
            local=True,
            requires_cuda=True,
            supports_design=True,
            supports_clone=True,
            supports_narrate=True,
        )

    def check_available(self) -> None:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel  # noqa: F401
        except ImportError as exc:
            raise SynthesisUnavailableError(
                "Qwen3-TTS is not installed. Run: .venv/bin/python -m pip "
                "install -r requirements.txt"
            ) from exc
        if not torch.cuda.is_available():
            raise SynthesisUnavailableError("Qwen3-TTS generation requires a CUDA GPU.")

    # -- the voice catalog ----------------------------------------------

    def voices(self, voices_dir: Path | None = None) -> tuple[VoiceInfo, ...]:
        """File-backed voices on disk, then the checkpoint's built-in speakers.

        Both kinds are Qwen's to know about: reference clips condition the
        clone pipeline, built-in names address CustomVoice embeddings.  A
        voice on disk shadows a built-in of the same name (case-insensitive)
        so a voice someone made always wins the spec.  Cheap by design — the
        roster comes from the checkpoint's config file, never a loaded model.
        """

        from ..voices import list_reference_voices

        if voices_dir is None:
            from ...config import VOICES_DIR

            voices_dir = VOICES_DIR

        file_backed = list_reference_voices(voices_dir)
        taken = {info.spec.casefold() for info in file_backed}
        builtins = tuple(
            VoiceInfo(spec=name, label=f"{name}  (built-in)", kind="built-in")
            for name in _builtin_speaker_roster()
            if name.casefold() not in taken
        )
        return file_backed + builtins

    def load_voice(
        self, spec: str, *, ref_text: str | None = None
    ) -> "_QwenVoice | _QwenBuiltinVoice":
        """Resolve *spec* to a generate handle, whichever kind it names.

        Disk is consulted first, mirroring :meth:`voices` shadowing: a voice
        folder or recording wins over a built-in of the same name.  A
        file-backed voice is decoded and its clone prompt precomputed once,
        so reusing the handle keeps the narrator identical across a book.
        """

        from ...config import VOICE_REFERENCE_METADATA_FILENAME, VOICES_DIR
        from ..voices import describe, resolve_voice

        candidate = Path(spec)
        on_disk = (
            (VOICES_DIR / spec / VOICE_REFERENCE_METADATA_FILENAME).exists()
            or candidate.is_file()
            or (VOICES_DIR / candidate.name).is_file()
        )
        if not on_disk:
            for name in _builtin_speaker_roster():
                if name.casefold() == spec.casefold():
                    return _QwenBuiltinVoice(name)

        voice = resolve_voice(spec, voices_dir=VOICES_DIR, ref_text=ref_text)
        print(describe(voice))
        return self.clone(
            ref_audio=voice.audio,
            sample_rate=voice.sample_rate,
            ref_text=voice.ref_text,
        )

    # -- model residency ------------------------------------------------

    def _model_for(self, role: str) -> Any:
        """Return the checkpoint for *role*, evicting whichever one is resident."""

        import torch
        from qwen_tts import Qwen3TTSModel

        path = self._paths[role]
        if self._loaded is not None and self._loaded[0] == path:
            return self._loaded[1]

        if self._loaded is not None:
            print(f"Unloading {self._loaded[0]}...")
            self._loaded = None
            torch.cuda.empty_cache()

        print(f"Loading {path} on {torch.cuda.get_device_name(0)}...")
        model = Qwen3TTSModel.from_pretrained(path, device_map=self._device, dtype=torch.bfloat16)
        self._loaded = (path, model)
        return model

    # -- the core three -------------------------------------------------

    def design(self, *, persona: str, ref_text: str, language: str) -> AudioClip:
        model = self._model_for("design")
        wavs, sample_rate = model.generate_voice_design(
            text=ref_text,
            language=language,
            instruct=persona,
        )
        return AudioClip(_as_mono_float32(wavs[0]), int(sample_rate))

    def clone(
        self,
        *,
        ref_audio: np.ndarray,
        sample_rate: int,
        ref_text: str | None,
    ) -> _QwenVoice:
        model = self._model_for("clone")
        prompt = model.create_voice_clone_prompt(
            ref_audio=(ref_audio, int(sample_rate)),
            ref_text=ref_text,
            x_vector_only_mode=not ref_text,
        )
        return _QwenVoice(prompt)

    def generate(
        self,
        *,
        text: str,
        language: str,
        voice: _QwenVoice | _QwenBuiltinVoice | str,
        instruction: str | None = None,
    ) -> AudioClip:
        if isinstance(voice, _QwenVoice):
            # Delivery is carried by the reference clip; the instruction is
            # deliberately ignored so callers can pass one unconditionally.
            model = self._model_for("clone")
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice.prompt,
            )
        else:
            speaker = voice.name if isinstance(voice, _QwenBuiltinVoice) else voice
            model = self._model_for("custom_voice")
            self._verify_supported_voice(model, speaker)
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruction,
            )
        return AudioClip(_as_mono_float32(wavs[0]), int(sample_rate))

    def close(self) -> None:
        if self._loaded is None:
            return
        import torch

        self._loaded = None
        torch.cuda.empty_cache()

    def resident_checkpoint(self) -> str | None:
        """Path of the checkpoint currently in VRAM, for run diagnostics."""

        return self._loaded[0] if self._loaded is not None else None

    # -- helpers --------------------------------------------------------

    def _verify_supported_voice(self, model: Any, voice_name: str) -> None:
        """Raise the first time a built-in speaker the model lacks is requested."""

        if voice_name in self._verified_speakers:
            return
        supported = {str(speaker).casefold() for speaker in model.get_supported_speakers()}
        if voice_name.casefold() not in supported:
            raise SynthesisResponseError(f"{voice_name} is not supported by the selected model.")
        self._verified_speakers.add(voice_name)


__all__ = ["QwenSynthesisProvider"]
