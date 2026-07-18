"""Resumable orchestration for provider-neutral narration preparation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Callable, Sequence

from .artifacts import refresh_hashes, validate_artifact
from .normalization import normalize_text
from .providers.base import NarrationPreparationProvider
from .segmentation import (
    DEFAULT_CONTEXT_CHARS,
    DEFAULT_MAX_UNIT_CHARS,
    DEFAULT_TARGET_UNIT_CHARS,
    SourceUnit,
    segment_text,
)
from .types import (
    DEFAULT_POLICY,
    PreparedBook,
    PreparedChapter,
    PreparedUnit,
    PreparationRequest,
    PreparationResult,
    SourceMetadata,
)
from .validation import ValidationPolicy, validate_preparation


CheckpointCallback = Callable[[PreparedBook], None]


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class NarrationPreparationPipeline:
    """Normalize, segment, adapt, validate, cache, and checkpoint a book."""

    def __init__(
        self,
        provider: NarrationPreparationProvider,
        *,
        target_unit_chars: int = DEFAULT_TARGET_UNIT_CHARS,
        max_unit_chars: int = DEFAULT_MAX_UNIT_CHARS,
        context_chars: int = DEFAULT_CONTEXT_CHARS,
        policy: str = DEFAULT_POLICY,
        validation_policy: ValidationPolicy | None = None,
    ) -> None:
        if target_unit_chars <= 0 or max_unit_chars < target_unit_chars:
            raise ValueError(
                "Unit sizes must satisfy 0 < target_unit_chars <= max_unit_chars"
            )
        if context_chars < 0:
            raise ValueError("context_chars cannot be negative")
        self.provider = provider
        self.target_unit_chars = target_unit_chars
        self.max_unit_chars = max_unit_chars
        self.context_chars = context_chars
        self.policy = policy
        self.validation_policy = validation_policy or ValidationPolicy()

    def _cache_key(self, request: PreparationRequest) -> str:
        return _stable_hash(
            {
                "source_text": request.source_text,
                "chapter_title": request.chapter_title,
                "previous_context": request.previous_context,
                "following_context": request.following_context,
                "prompt_version": request.prompt_version,
                "policy": request.policy,
                "validation_policy": self.validation_policy.to_dict(),
                "provider": self.provider.metadata.to_dict(),
            }
        )

    @staticmethod
    def _cache_from(book: PreparedBook | None) -> dict[str, PreparedUnit]:
        cache: dict[str, PreparedUnit] = {}
        if book is None:
            return cache
        # A corrupt resume artifact must not silently seed future narration.
        validate_artifact(book)
        for chapter in book.chapters:
            for unit in chapter.units:
                if unit.kind == "prose" and unit.cache_key:
                    cache[unit.cache_key] = unit
        return cache

    @staticmethod
    def _checkpoint(book: PreparedBook, callback: CheckpointCallback | None) -> None:
        if callback is None:
            return
        refresh_hashes(book)
        validate_artifact(book)
        callback(book)

    def _segments(self, normalized_text: str) -> list[SourceUnit]:
        return segment_text(
            normalized_text,
            target_chars=self.target_unit_chars,
            max_chars=self.max_unit_chars,
            context_chars=self.context_chars,
        )

    def prepare_book(
        self,
        chapters: Sequence[tuple[str, str]],
        *,
        book_title: str = "Audiobook",
        source_metadata: SourceMetadata | None = None,
        resume_from: PreparedBook | None = None,
        checkpoint: CheckpointCallback | None = None,
        check_provider: bool = True,
        max_prose_units: int | None = None,
    ) -> PreparedBook:
        """Prepare chapters, resuming matching units from a prior artifact.

        ``max_prose_units`` is a book-wide call cap intended for previews. If
        reached, structural units immediately preceding the next prose unit are
        retained and the returned artifact has ``complete=False``.
        """

        if not book_title.strip():
            raise ValueError("book_title cannot be blank")
        if max_prose_units is not None and max_prose_units < 0:
            raise ValueError("max_prose_units cannot be negative")

        planned: list[tuple[int, str, str, str, list[SourceUnit]]] = []
        for chapter_index, chapter in enumerate(chapters):
            if not isinstance(chapter, (tuple, list)) or len(chapter) != 2:
                raise TypeError("Each chapter must be a (title, text) pair")
            chapter_title, source_text = chapter
            if not isinstance(chapter_title, str) or not isinstance(source_text, str):
                raise TypeError("Chapter title and text must be strings")
            normalized = normalize_text(source_text)
            planned.append(
                (
                    chapter_index,
                    chapter_title,
                    source_text,
                    normalized,
                    self._segments(normalized),
                )
            )
        total_prose_units = sum(
            segment.kind == "prose"
            for _index, _title, _source, _normalized, segments in planned
            for segment in segments
        )
        call_limit = (
            total_prose_units if max_prose_units is None else max_prose_units
        )
        will_be_partial = call_limit < total_prose_units

        book = PreparedBook(
            title=book_title,
            source_metadata=source_metadata or SourceMetadata(),
            provider_metadata=self.provider.metadata,
            prompt_version=self.provider.metadata.prompt_version,
            policy=self.policy,
            complete=False,
        )
        cache = self._cache_from(resume_from)
        prose_units_processed = 0
        availability_checked = False
        stop = False

        for chapter_index, chapter_title, source_text, normalized, segments in planned:
            prepared_chapter = PreparedChapter(
                index=chapter_index,
                title=chapter_title,
                source_text=source_text,
                normalized_text=normalized,
            )
            book.chapters.append(prepared_chapter)

            for segment in segments:
                if segment.kind == "prose" and prose_units_processed >= call_limit:
                    stop = True
                    break
                source_digest = hashlib.sha256(segment.text.encode("utf-8")).hexdigest()
                unit_id = (
                    f"chapter-{chapter_index:04d}-unit-{segment.position:04d}-"
                    f"{source_digest[:12]}"
                )

                if segment.kind != "prose":
                    prepared_chapter.units.append(
                        PreparedUnit(
                            unit_id=unit_id,
                            position=segment.position,
                            kind=segment.kind,
                            source_text=segment.text,
                            prepared_text=segment.text,
                        )
                    )
                    self._checkpoint(book, checkpoint)
                    continue

                request = PreparationRequest(
                    unit_id=unit_id,
                    chapter_title=chapter_title,
                    source_text=segment.text,
                    previous_context=segment.previous_context,
                    following_context=segment.following_context,
                    prompt_version=self.provider.metadata.prompt_version,
                    policy=self.policy,
                )
                cache_key = self._cache_key(request)
                cached = cache.get(cache_key)
                if cached is not None:
                    cached_result = PreparationResult(
                        prepared_text=cached.prepared_text,
                        edits=deepcopy(cached.edits),
                        warnings=list(cached.warnings),
                        provider_metadata=cached.provider_metadata,
                    )
                    validate_preparation(
                        request, cached_result, policy=self.validation_policy
                    )
                    unit = deepcopy(cached)
                    unit.unit_id = unit_id
                    unit.position = segment.position
                    unit.source_text = segment.text
                    unit.cache_key = cache_key
                    unit.cache_hit = True
                else:
                    if check_provider and not availability_checked:
                        self.provider.check_available()
                        availability_checked = True
                    result = self.provider.prepare(request)
                    validate_preparation(
                        request, result, policy=self.validation_policy
                    )
                    unit = PreparedUnit(
                        unit_id=unit_id,
                        position=segment.position,
                        kind="prose",
                        source_text=segment.text,
                        prepared_text=result.prepared_text.strip(),
                        cache_key=cache_key,
                        edits=result.edits,
                        warnings=result.warnings,
                        provider_metadata=(
                            result.provider_metadata or self.provider.metadata
                        ),
                    )
                prepared_chapter.units.append(unit)
                prose_units_processed += 1
                self._checkpoint(book, checkpoint)

            if stop:
                # Retain a chapter only if it contains structural context or
                # completed prose. An empty next chapter conveys no progress.
                if not prepared_chapter.units:
                    book.chapters.pop()
                break

        book.complete = not will_be_partial
        refresh_hashes(book)
        validate_artifact(book)
        self._checkpoint(book, checkpoint)
        return book


PreparationPipeline = NarrationPreparationPipeline


def prepare_book(
    chapters: Sequence[tuple[str, str]],
    provider: NarrationPreparationProvider,
    **kwargs: object,
) -> PreparedBook:
    """Convenience wrapper around :class:`NarrationPreparationPipeline`."""

    pipeline_keys = {
        "target_unit_chars",
        "max_unit_chars",
        "context_chars",
        "policy",
        "validation_policy",
    }
    pipeline_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in pipeline_keys
    }
    pipeline = NarrationPreparationPipeline(provider, **pipeline_kwargs)  # type: ignore[arg-type]
    return pipeline.prepare_book(chapters, **kwargs)  # type: ignore[arg-type]
