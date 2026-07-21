"""Narration-preparation provider adapters."""

from .base import (
    NarrationPreparationProvider,
    ProviderDescriptor,
    ProviderError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from .ollama import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    SAMPLING_OPTIONS,
    OllamaProvider,
    fetch_model_capabilities,
)
from .registry import (
    ProviderFactory,
    available_providers,
    create_provider,
    provider_descriptor,
    provider_descriptors,
    register_provider,
)

register_provider("ollama", OllamaProvider)

__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "SAMPLING_OPTIONS",
    "NarrationPreparationProvider",
    "OllamaProvider",
    "ProviderDescriptor",
    "ProviderFactory",
    "ProviderError",
    "ProviderResponseError",
    "ProviderUnavailableError",
    "available_providers",
    "create_provider",
    "fetch_model_capabilities",
    "provider_descriptor",
    "provider_descriptors",
    "register_provider",
]
