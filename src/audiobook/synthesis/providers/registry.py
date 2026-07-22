"""Small registry that keeps narration orchestration backend-agnostic.

A direct parallel of ``preparation/providers/registry.py``.  The function names
are prefixed with ``synthesis_`` so a module can import both registries without
a clash (``workflow.py`` uses the preparation one in the same file).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import SynthesisDescriptor, SynthesisProvider


SynthesisFactory = Callable[..., SynthesisProvider]
_PROVIDERS: dict[str, SynthesisFactory] = {}


def register_synthesis_provider(
    name: str,
    factory: SynthesisFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a synthesis backend factory under a case-insensitive name.

    Registration is where the protocol is enforced: the descriptor is built
    once, here, so an adapter that cannot describe its capabilities fails at
    import rather than when a frontend tries to offer it.
    """

    key = name.strip().casefold()
    if not key:
        raise ValueError("Synthesis provider name cannot be blank")
    if key in _PROVIDERS and not replace:
        raise ValueError(f"Synthesis provider already registered: {name}")

    describe = getattr(factory, "describe", None)
    if not callable(describe):
        raise TypeError(
            f"Synthesis provider {name!r} must implement describe() -> "
            "SynthesisDescriptor"
        )
    descriptor = describe()
    if not isinstance(descriptor, SynthesisDescriptor):
        raise TypeError(
            f"Synthesis provider {name!r}.describe() must return a "
            "SynthesisDescriptor"
        )

    _PROVIDERS[key] = factory


def available_synthesis_providers() -> tuple[str, ...]:
    """Return registered provider names in stable display order."""

    return tuple(sorted(_PROVIDERS))


def synthesis_descriptor(name: str) -> SynthesisDescriptor:
    """The capabilities a registered backend declares.

    Rebuilt on each call rather than cached at registration: descriptors read
    config, so a checkpoint repointed in config shows up on the next refresh
    instead of after a restart.
    """

    key = name.strip().casefold()
    try:
        factory = _PROVIDERS[key]
    except KeyError as exc:
        installed = ", ".join(available_synthesis_providers()) or "none"
        raise ValueError(
            f"Unknown synthesis provider: {name}. Installed providers: {installed}"
        ) from exc
    return factory.describe()


def synthesis_descriptors() -> tuple[SynthesisDescriptor, ...]:
    """Descriptors for every registered backend, in display order."""

    return tuple(
        synthesis_descriptor(name) for name in available_synthesis_providers()
    )


def create_synthesis_provider(name: str, **configuration: Any) -> SynthesisProvider:
    """Construct a registered backend without leaking its concrete type."""

    key = name.strip().casefold()
    try:
        factory = _PROVIDERS[key]
    except KeyError as exc:
        installed = ", ".join(available_synthesis_providers()) or "none"
        raise ValueError(
            f"Unknown synthesis provider: {name}. Installed providers: {installed}"
        ) from exc
    return factory(**configuration)


__all__ = [
    "SynthesisFactory",
    "available_synthesis_providers",
    "create_synthesis_provider",
    "register_synthesis_provider",
    "synthesis_descriptor",
    "synthesis_descriptors",
]
