"""Text-to-speech provider adapters.

Importing this package registers the built-in backends, so a caller only needs
``create_synthesis_provider(name)`` to get one.  Adding a backend is a new
adapter module plus a single ``register_synthesis_provider`` line here.
"""

from .base import (
    AudioClip,
    SynthesisDescriptor,
    SynthesisError,
    SynthesisProvider,
    SynthesisResponseError,
    SynthesisUnavailableError,
    Voice,
)
from .qwen import QwenSynthesisProvider
from .registry import (
    SynthesisFactory,
    available_synthesis_providers,
    create_synthesis_provider,
    register_synthesis_provider,
    synthesis_descriptor,
    synthesis_descriptors,
)

register_synthesis_provider("qwen", QwenSynthesisProvider)

__all__ = [
    "AudioClip",
    "QwenSynthesisProvider",
    "SynthesisDescriptor",
    "SynthesisError",
    "SynthesisFactory",
    "SynthesisProvider",
    "SynthesisResponseError",
    "SynthesisUnavailableError",
    "Voice",
    "available_synthesis_providers",
    "create_synthesis_provider",
    "register_synthesis_provider",
    "synthesis_descriptor",
    "synthesis_descriptors",
]
