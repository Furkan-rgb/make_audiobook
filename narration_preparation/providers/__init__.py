"""Narration-preparation provider adapters."""

from .base import (
    NarrationPreparationProvider,
    ProviderError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from .ollama import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    OllamaProvider,
)
from .registry import (
    ProviderFactory,
    available_providers,
    create_provider,
    register_provider,
)

register_provider("ollama", OllamaProvider)

__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "NarrationPreparationProvider",
    "OllamaProvider",
    "ProviderFactory",
    "ProviderError",
    "ProviderResponseError",
    "ProviderUnavailableError",
    "available_providers",
    "create_provider",
    "register_provider",
]
