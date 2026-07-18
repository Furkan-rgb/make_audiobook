"""Provider protocol and errors shared by local and hosted adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import PreparationRequest, PreparationResult, ProviderMetadata


class ProviderError(RuntimeError):
    """Base error for narration-preparation providers."""


class ProviderUnavailableError(ProviderError):
    """The provider service or requested model is unavailable."""


class ProviderResponseError(ProviderError):
    """The provider returned an invalid or unsuccessful response."""


@runtime_checkable
class NarrationPreparationProvider(Protocol):
    """Minimal interface implemented by Ollama and future hosted providers."""

    @property
    def metadata(self) -> ProviderMetadata:
        ...

    def check_available(self) -> None:
        ...

    def prepare(self, request: PreparationRequest) -> PreparationResult:
        ...

    def close(self) -> None:
        ...
