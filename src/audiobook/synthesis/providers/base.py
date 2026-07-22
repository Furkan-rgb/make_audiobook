"""Provider protocol and types shared by local and hosted TTS backends.

This mirrors ``preparation/providers/base.py``: a structural ``Protocol`` plus a
declarative descriptor a frontend can read before any model is loaded.  The
three verbs a narrator needs — design a voice, clone a voice, speak text — are
the whole interface, so swapping Qwen for another backend is a new adapter here
rather than edits spread across the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np


class AudioClip(NamedTuple):
    """A rendered mono waveform and the rate it was produced at.

    A ``NamedTuple`` so existing call sites that unpack ``audio, rate = ...``
    keep working, while new code can read ``clip.audio`` / ``clip.sample_rate``.
    """

    audio: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class SynthesisDescriptor:
    """What a TTS backend can do and what it needs, known without loading it.

    Preflight and the UI decide which actions to offer before any checkpoint is
    resident, so capability is declared here: a backend that cannot design a
    voice says so, rather than failing once the action is invoked.
    """

    name: str
    label: str
    local: bool = False
    requires_cuda: bool = False
    supports_design: bool = False          # persona text -> reference clip
    supports_clone: bool = False           # reference clip -> reusable voice
    supports_builtin_voice: bool = False   # named speaker baked into the model
    builtin_voices: tuple[str, ...] = ()
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Synthesis descriptor needs a name")
        if not (
            self.supports_design
            or self.supports_clone
            or self.supports_builtin_voice
        ):
            raise ValueError(
                f"Synthesis backend {self.name!r} declares no capabilities"
            )


class SynthesisError(RuntimeError):
    """Base error for TTS backends."""


class SynthesisUnavailableError(SynthesisError):
    """The backend's package or hardware is missing."""


class SynthesisResponseError(SynthesisError):
    """The backend returned an invalid or unusable result."""


@runtime_checkable
class Voice(Protocol):
    """Opaque, backend-defined handle returned by :meth:`SynthesisProvider.clone`.

    Callers pass it straight back to :meth:`SynthesisProvider.generate` and
    never inspect it: for Qwen it wraps a precomputed clone prompt, for another
    backend it might be a speaker embedding or a remote voice id.  It is only
    valid while the provider that produced it is open.
    """


@runtime_checkable
class SynthesisProvider(Protocol):
    """Minimal interface implemented by Qwen and any future TTS backend."""

    @classmethod
    def describe(cls) -> SynthesisDescriptor:
        """Capabilities and requirements, answerable without loading a model."""
        ...

    def check_available(self) -> None:
        """Raise :class:`SynthesisUnavailableError` if deps/hardware are missing."""
        ...

    def design(self, *, persona: str, ref_text: str, language: str) -> AudioClip:
        """Render a reference clip from a natural-language persona."""
        ...

    def clone(
        self,
        *,
        ref_audio: np.ndarray,
        sample_rate: int,
        ref_text: str | None,
    ) -> Voice:
        """Precompute a reusable narrator identity from a reference clip.

        The returned handle is bound to this provider instance and stays valid
        until :meth:`close` is called; reuse it across every ``generate`` call
        so the narrator is identical for the whole book.
        """
        ...

    def generate(
        self,
        *,
        text: str,
        language: str,
        voice: Voice | str,
        instruction: str | None = None,
    ) -> AudioClip:
        """Speak *text*.

        ``voice`` is either a :meth:`clone` handle or the name of a built-in
        speaker.  ``instruction`` is a style hint honoured by built-in speakers;
        cloned voices carry their delivery from the reference and ignore it.
        """
        ...

    def close(self) -> None:
        """Release any resident model and free its memory."""
        ...


__all__ = [
    "AudioClip",
    "SynthesisDescriptor",
    "SynthesisError",
    "SynthesisProvider",
    "SynthesisResponseError",
    "SynthesisUnavailableError",
    "Voice",
]
