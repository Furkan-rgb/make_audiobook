"""Prepare extracted prose for faithful, listenable audiobook narration.

The stage in one line: normalize the extracted text, cut it into units, ask a
model for the *edits* each prose unit needs, apply them deterministically, and
save a reviewable artifact.

Where things live:

- :mod:`normalization` and :mod:`segmentation` shape raw text into units, and
  decide which of them are prose worth sending to a model at all.
- :mod:`adaptation` owns the edits-only contract end to end: addressing a
  passage, anchoring a quoted edit in it, and splicing the survivors in.
- :mod:`validation` holds the thresholds, measured in words with citations
  masked out, plus the passage-level checks.
- :mod:`prompting` is the wire format; :mod:`providers` are transports.
- :mod:`pipeline` orchestrates, caches, and checkpoints; :mod:`artifacts`
  hashes and persists.

The layering rule worth keeping: providers parse, the pipeline decides. No
adapter applies edits, so two providers given identical model output cannot
produce different prose.
"""

from .adaptation import apply_edits, numbered_view, resolve_edit, sentence_spans
from .artifacts import (
    ArtifactValidationError,
    atomic_save_prepared_book,
    load_prepared_book,
    refresh_hashes,
    render_prepared_markdown,
    save_prepared_book,
    save_prepared_markdown,
    sha256_file,
    sha256_text,
    source_metadata_for_path,
    validate_artifact,
)
from .normalization import (
    is_display_line,
    is_markdown_heading,
    is_scene_marker,
    normalize_paragraph,
    normalize_text,
)
from .pipeline import (
    CheckpointCallback,
    NarrationPreparationPipeline,
    PreparationPipeline,
    prepare_book,
)
from .prompting import (
    RESPONSE_JSON_SCHEMA,
    SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    build_messages,
    parse_structured_response,
    system_prompt_for,
)
from .providers import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    SAMPLING_OPTIONS,
    NarrationPreparationProvider,
    OllamaProvider,
    ProviderDescriptor,
    ProviderError,
    ProviderFactory,
    ProviderResponseError,
    ProviderUnavailableError,
    available_providers,
    create_provider,
    fetch_model_capabilities,
    provider_descriptor,
    provider_descriptors,
    register_provider,
)
from .segmentation import (
    DEFAULT_CONTEXT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    DEFAULT_TARGET_UNIT_CHARS,
    SourceUnit,
    segment_text,
)
from .types import (
    DEFAULT_POLICY,
    DEFAULT_PROMPT_VERSION,
    SCHEMA_VERSION,
    PreparationEdit,
    PreparationRequest,
    PreparationResult,
    PreparedBook,
    PreparedChapter,
    PreparedUnit,
    ProviderMetadata,
    SourceMetadata,
    UnitKind,
)
from .validation import (
    PreparationValidationError,
    ValidationPolicy,
    ValidationReport,
    lexical_retention,
    mask_citations,
    validate_edit,
    validate_preparation,
    words_changed,
)

__all__ = [
    # Data contracts
    "DEFAULT_POLICY",
    "DEFAULT_PROMPT_VERSION",
    "SCHEMA_VERSION",
    "PreparationEdit",
    "PreparationRequest",
    "PreparationResult",
    "PreparedBook",
    "PreparedChapter",
    "PreparedUnit",
    "ProviderMetadata",
    "SourceMetadata",
    "SourceUnit",
    "UnitKind",
    # Shaping text into units
    "DEFAULT_CONTEXT_CHARS",
    "DEFAULT_MAX_UNIT_CHARS",
    "DEFAULT_TARGET_UNIT_CHARS",
    "is_display_line",
    "is_markdown_heading",
    "is_scene_marker",
    "normalize_paragraph",
    "normalize_text",
    "segment_text",
    # The edits-only adaptation contract
    "apply_edits",
    "numbered_view",
    "resolve_edit",
    "sentence_spans",
    # Limits and checks
    "PreparationValidationError",
    "ValidationPolicy",
    "ValidationReport",
    "lexical_retention",
    "mask_citations",
    "validate_edit",
    "validate_preparation",
    "words_changed",
    # Asking a model
    "RESPONSE_JSON_SCHEMA",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPTS",
    "build_messages",
    "parse_structured_response",
    "system_prompt_for",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "SAMPLING_OPTIONS",
    "NarrationPreparationProvider",
    "OllamaProvider",
    "ProviderDescriptor",
    "ProviderError",
    "ProviderFactory",
    "ProviderResponseError",
    "ProviderUnavailableError",
    "available_providers",
    "create_provider",
    "fetch_model_capabilities",
    "provider_descriptor",
    "provider_descriptors",
    "register_provider",
    # Running and persisting a book
    "CheckpointCallback",
    "NarrationPreparationPipeline",
    "PreparationPipeline",
    "prepare_book",
    "ArtifactValidationError",
    "atomic_save_prepared_book",
    "load_prepared_book",
    "refresh_hashes",
    "render_prepared_markdown",
    "save_prepared_book",
    "save_prepared_markdown",
    "sha256_file",
    "sha256_text",
    "source_metadata_for_path",
    "validate_artifact",
]
