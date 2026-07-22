"""Provider protocol and types shared by local and hosted TTS backends.

This mirrors ``preparation/providers/base.py``: a structural ``Protocol`` plus a
declarative descriptor a frontend can read before any model is loaded.  The
three verbs a narrator needs — design a voice, clone a voice, speak text — are
the whole interface, so swapping Qwen for another backend is a new adapter here
rather than edits spread across the workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
    voice says so, rather than failing once the action is invoked.  Which
    voices exist is not a capability — every backend answers that through
    :meth:`SynthesisProvider.voices`.
    """

    name: str
    label: str
    local: bool = False
    requires_cuda: bool = False
    supports_design: bool = False  # persona text -> reference clip
    supports_clone: bool = False  # reference clip -> reusable voice
    supports_narrate: bool = False  # speak text with a voice from voices()
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Synthesis descriptor needs a name")
        if not (self.supports_design or self.supports_clone or self.supports_narrate):
            raise ValueError(f"Synthesis backend {self.name!r} declares no capabilities")


@dataclass(frozen=True)
class VoiceInfo:
    """One selectable narrator voice, as a backend exposes it.

    Where a voice comes from — a speaker baked into a checkpoint, a designed
    reference clip, an imported recording, a remote voice id — is the
    backend's business; callers pick from this list and hand the ``spec``
    back to :meth:`SynthesisProvider.load_voice`.  ``kind`` and the file
    fields are presentation metadata: the frontend uses them to label entries
    and to offer file edits where files exist, never to choose a synthesis
    path.
    """

    spec: str
    label: str
    kind: str  # e.g. "built-in", "designed", "recording" — display only
    audio_path: Path | None = None  # None when the voice has no files
    transcript_path: Path | None = None
    ref_text: str | None = None
    instruct: str | None = None
    folder: bool = False  # file layout: a voice folder vs a loose recording

    @property
    def file_backed(self) -> bool:
        return self.audio_path is not None

    @property
    def builtin(self) -> bool:
        return not self.file_backed

    @property
    def designed(self) -> bool:
        return self.kind == "designed"

    @property
    def has_transcript(self) -> bool:
        return bool(self.ref_text)


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
    """Minimal interface implemented by Qwen and any future TTS backend.

    A backend only honours the verbs its descriptor declares — design, clone,
    narrate — and may raise :class:`SynthesisError` from the others.  Callers
    consult the descriptor before offering an action, so an unsupported verb
    is a disabled affordance rather than a runtime surprise.  ``voices`` and
    ``load_voice`` belong to the narrate capability: a backend that cannot
    narrate exposes an empty catalog.
    """

    @classmethod
    def describe(cls) -> SynthesisDescriptor:
        """Capabilities and requirements, answerable without loading a model."""
        ...

    def check_available(self) -> None:
        """Raise :class:`SynthesisUnavailableError` if deps/hardware are missing."""
        ...

    def voices(self) -> tuple[VoiceInfo, ...]:
        """Every narrator this backend can speak with, wherever each lives.

        Must stay cheap — no model may be loaded to answer it — because the
        frontend refreshes the list on every tab switch and preflight reads
        it before anything is resident.
        """
        ...

    def load_voice(self, spec: str, *, ref_text: str | None = None) -> Voice:
        """Prepare the narrator *spec* names and return its generate handle.

        *spec* is a value from :meth:`voices` (or anything the backend
        documents, e.g. a path to a recording).  How the voice is realised —
        native speaker embedding, precomputed clone prompt, remote id — is
        the backend's choice; callers pass the handle to :meth:`generate`
        without inspecting it.  *ref_text* optionally overrides the
        transcript a file-backed voice would otherwise carry.

        Raises :class:`FileNotFoundError` when *spec* names nothing this
        backend knows.
        """
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
        voice: Voice,
        instruction: str | None = None,
    ) -> AudioClip:
        """Speak *text* with a :meth:`load_voice` or :meth:`clone` handle.

        ``instruction`` is a style hint; a voice whose delivery is fixed by a
        reference clip is free to ignore it, so callers may always pass one.
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
    "VoiceInfo",
]
