"""Data contracts for provider-neutral narration preparation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SCHEMA_VERSION = 1
DEFAULT_PROMPT_VERSION = "narration-preparation-v1"
DEFAULT_POLICY = (
    "Adapt presentation for listening without summarizing, censoring, "
    "softening, modernizing, or otherwise changing the author's meaning."
)

UnitKind = Literal["prose", "heading", "scene_marker"]


@dataclass(frozen=True)
class ProviderMetadata:
    """Reproducible identity and settings for a preparation provider."""

    name: str
    model: str
    prompt_version: str = DEFAULT_PROMPT_VERSION
    base_url: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "parameters": self.parameters,
        }
        if self.base_url is not None:
            payload["base_url"] = self.base_url
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProviderMetadata":
        return cls(
            name=str(payload["name"]),
            model=str(payload["model"]),
            prompt_version=str(
                payload.get("prompt_version", DEFAULT_PROMPT_VERSION)
            ),
            base_url=(
                str(payload["base_url"])
                if payload.get("base_url") is not None
                else None
            ),
            parameters=dict(payload.get("parameters", {})),
        )


@dataclass(frozen=True)
class SourceMetadata:
    """Metadata for the input from which the narration script was extracted."""

    path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"extra": self.extra}
        for key in ("path", "sha256", "size_bytes", "media_type"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceMetadata":
        return cls(
            path=str(payload["path"]) if payload.get("path") is not None else None,
            sha256=(
                str(payload["sha256"])
                if payload.get("sha256") is not None
                else None
            ),
            size_bytes=(
                int(payload["size_bytes"])
                if payload.get("size_bytes") is not None
                else None
            ),
            media_type=(
                str(payload["media_type"])
                if payload.get("media_type") is not None
                else None
            ),
            extra=dict(payload.get("extra", {})),
        )


@dataclass(frozen=True)
class PreparationRequest:
    """One prose unit submitted to a narration-preparation provider.

    Neighbor context is reference-only and must never be copied into
    ``prepared_text``.
    """

    chapter_title: str
    source_text: str
    previous_context: str = ""
    following_context: str = ""
    unit_id: str = ""
    prompt_version: str = DEFAULT_PROMPT_VERSION
    policy: str = DEFAULT_POLICY


@dataclass(frozen=True)
class PreparationEdit:
    """An auditable material edit reported by the provider."""

    category: str
    original: str = ""
    replacement: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "original": self.original,
            "replacement": self.replacement,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreparationEdit":
        return cls(
            category=str(payload.get("category", "unspecified")),
            original=str(payload.get("original", "")),
            replacement=str(payload.get("replacement", "")),
            reason=str(payload.get("reason", "")),
        )


@dataclass
class PreparationResult:
    """Provider response before it is incorporated into a book artifact."""

    prepared_text: str
    edits: list[PreparationEdit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provider_metadata: ProviderMetadata | None = None


@dataclass
class PreparedUnit:
    """A prepared prose unit or an untouched structural marker."""

    unit_id: str
    position: int
    kind: UnitKind
    source_text: str
    prepared_text: str
    source_sha256: str = ""
    prepared_sha256: str = ""
    cache_key: str = ""
    edits: list[PreparationEdit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provider_metadata: ProviderMetadata | None = None
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "position": self.position,
            "kind": self.kind,
            "source_text": self.source_text,
            "prepared_text": self.prepared_text,
            "source_sha256": self.source_sha256,
            "prepared_sha256": self.prepared_sha256,
            "cache_key": self.cache_key,
            "edits": [edit.to_dict() for edit in self.edits],
            "warnings": self.warnings,
            "provider_metadata": (
                self.provider_metadata.to_dict()
                if self.provider_metadata is not None
                else None
            ),
            "cache_hit": self.cache_hit,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreparedUnit":
        metadata = payload.get("provider_metadata")
        return cls(
            unit_id=str(payload["unit_id"]),
            position=int(payload["position"]),
            kind=str(payload["kind"]),  # type: ignore[arg-type]
            source_text=str(payload["source_text"]),
            prepared_text=str(payload["prepared_text"]),
            source_sha256=str(payload.get("source_sha256", "")),
            prepared_sha256=str(payload.get("prepared_sha256", "")),
            cache_key=str(payload.get("cache_key", "")),
            edits=[
                PreparationEdit.from_dict(item)
                for item in payload.get("edits", [])
            ],
            warnings=[str(item) for item in payload.get("warnings", [])],
            provider_metadata=(
                ProviderMetadata.from_dict(metadata) if metadata is not None else None
            ),
            cache_hit=bool(payload.get("cache_hit", False)),
        )


@dataclass
class PreparedChapter:
    """One source chapter and its independently prepared units."""

    index: int
    title: str
    source_text: str
    normalized_text: str
    units: list[PreparedUnit] = field(default_factory=list)
    source_sha256: str = ""
    normalized_sha256: str = ""
    prepared_sha256: str = ""

    @property
    def prepared_text(self) -> str:
        return "\n\n".join(
            unit.prepared_text.strip()
            for unit in sorted(self.units, key=lambda item: item.position)
            if unit.prepared_text.strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "source_text": self.source_text,
            "normalized_text": self.normalized_text,
            "source_sha256": self.source_sha256,
            "normalized_sha256": self.normalized_sha256,
            "prepared_sha256": self.prepared_sha256,
            "prepared_text": self.prepared_text,
            "units": [unit.to_dict() for unit in self.units],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreparedChapter":
        chapter = cls(
            index=int(payload["index"]),
            title=str(payload["title"]),
            source_text=str(payload["source_text"]),
            normalized_text=str(payload["normalized_text"]),
            units=[
                PreparedUnit.from_dict(item) for item in payload.get("units", [])
            ],
            source_sha256=str(payload.get("source_sha256", "")),
            normalized_sha256=str(payload.get("normalized_sha256", "")),
            prepared_sha256=str(payload.get("prepared_sha256", "")),
        )
        if (
            payload.get("prepared_text") is not None
            and str(payload["prepared_text"]) != chapter.prepared_text
        ):
            raise ValueError("chapter prepared_text does not match its units")
        return chapter


@dataclass
class PreparedBook:
    """Versioned, reviewable input artifact for the narration stage."""

    title: str
    source_metadata: SourceMetadata
    provider_metadata: ProviderMetadata
    chapters: list[PreparedChapter] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    prompt_version: str = DEFAULT_PROMPT_VERSION
    policy: str = DEFAULT_POLICY
    created_at: str = ""
    source_sha256: str = ""
    prepared_sha256: str = ""
    complete: bool = True

    @property
    def prepared_text(self) -> str:
        return "\n\n".join(
            chapter.prepared_text
            for chapter in sorted(self.chapters, key=lambda item: item.index)
            if chapter.prepared_text
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "source_metadata": self.source_metadata.to_dict(),
            "provider_metadata": self.provider_metadata.to_dict(),
            "prompt_version": self.prompt_version,
            "policy": self.policy,
            "created_at": self.created_at,
            "source_sha256": self.source_sha256,
            "prepared_sha256": self.prepared_sha256,
            "prepared_text": self.prepared_text,
            "complete": self.complete,
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreparedBook":
        book = cls(
            schema_version=int(payload["schema_version"]),
            title=str(payload["title"]),
            source_metadata=SourceMetadata.from_dict(payload["source_metadata"]),
            provider_metadata=ProviderMetadata.from_dict(
                payload["provider_metadata"]
            ),
            prompt_version=str(
                payload.get("prompt_version", DEFAULT_PROMPT_VERSION)
            ),
            policy=str(payload.get("policy", DEFAULT_POLICY)),
            created_at=str(payload.get("created_at", "")),
            source_sha256=str(payload.get("source_sha256", "")),
            prepared_sha256=str(payload.get("prepared_sha256", "")),
            complete=bool(payload.get("complete", True)),
            chapters=[
                PreparedChapter.from_dict(item)
                for item in payload.get("chapters", [])
            ],
        )
        if (
            payload.get("prepared_text") is not None
            and str(payload["prepared_text"]) != book.prepared_text
        ):
            raise ValueError("book prepared_text does not match its chapters")
        return book
