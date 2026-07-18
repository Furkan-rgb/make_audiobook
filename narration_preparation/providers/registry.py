"""Small registry that keeps workflow orchestration provider-agnostic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import NarrationPreparationProvider


ProviderFactory = Callable[..., NarrationPreparationProvider]
_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(
    name: str,
    factory: ProviderFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a provider factory under a case-insensitive CLI name."""

    key = name.strip().casefold()
    if not key:
        raise ValueError("Provider name cannot be blank")
    if key in _PROVIDERS and not replace:
        raise ValueError(f"Preparation provider already registered: {name}")
    _PROVIDERS[key] = factory


def available_providers() -> tuple[str, ...]:
    """Return registered provider names in stable display order."""

    return tuple(sorted(_PROVIDERS))


def create_provider(name: str, **configuration: Any) -> NarrationPreparationProvider:
    """Construct a registered provider without leaking its response types."""

    key = name.strip().casefold()
    try:
        factory = _PROVIDERS[key]
    except KeyError as exc:
        installed = ", ".join(available_providers()) or "none"
        raise ValueError(
            f"Unknown preparation provider: {name}. Installed providers: {installed}"
        ) from exc
    return factory(**configuration)


__all__ = [
    "ProviderFactory",
    "available_providers",
    "create_provider",
    "register_provider",
]
