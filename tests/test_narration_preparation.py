import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

import make_audiobook
from audiobook_config import DEFAULT_PREPARATION_MODEL, TTS_MODEL
from audiobook_workflow import narration_chapters
from narration_preparation import (
    ArtifactValidationError,
    DEFAULT_PROMPT_VERSION,
    NarrationPreparationPipeline,
    OllamaProvider,
    PreparationEdit,
    PreparationRequest,
    PreparationResult,
    PreparationValidationError,
    PreparedBook,
    PreparedChapter,
    PreparedUnit,
    ProviderMetadata,
    ProviderResponseError,
    ProviderUnavailableError,
    RESPONSE_JSON_SCHEMA,
    SourceMetadata,
    load_prepared_book,
    normalize_text,
    refresh_hashes,
    save_prepared_book,
    segment_text,
    validate_preparation,
)
from narration_preparation.providers.ollama import DEFAULT_OLLAMA_MODEL
from semantic_chunking import build_chunk_plan


PREFACE_SOURCE = """The psychotherapy of male homosexuals has been explored for many years. What is new
in this book is the interweaving of several strands of clinical research: the development
of male gender-identity (Abelin 1975, Greenacre 1957, Greenson 1968, Greenspan 1982,
Kohl- berg 1966, LaTorre 1979, Mahler 1955, Moberly 1983, Money and Ehrhardt 1972,
Ross 1979, Stoller 1968), histories of family dynamics (Bell and Weinberg 1978, Bieber et
al. 1962, Green 1987, Higham 1976, Money and Russo 1979, Tyson 1985), and the
techniques of psychodynamic psychotherapy of male homosexuality (Gershman 1953,
Hadden 1966, Hamilton 1939, Hatterer 1970, Horner 1989, Masters and Johnson 1979,
Nun- berg 1938, Ovesey 1969, Socarides 1978, van den Aardweg 1986, Winnicott 1965).

I would like to thank a number of people, without whose help this book could not
have been written. I want to express my appreciation to my office staff—Jennie Gohn,
Margaret Guiteras, Edith Joanis, Joan Multerer, and Cindy Anctil, and very special
appreciation to my research assistant, Jeanne Armstrong, whose many hours in the
library made this book possible."""

PREFACE_PREPARED = """The psychotherapy of male homosexuals has been explored for many years. What is new in this book is the interweaving of several strands of clinical research: the development of male gender-identity, histories of family dynamics, and the techniques of psychodynamic psychotherapy of male homosexuality.

I would like to thank a number of people, without whose help this book could not have been written. I want to express my appreciation to my office staff—Jennie Gohn, Margaret Guiteras, Edith Joanis, Joan Multerer, and Cindy Anctil, and very special appreciation to my research assistant, Jeanne Armstrong, whose many hours in the library made this book possible."""


class FakeProvider:
    def __init__(self, *, model="gemma4:31b", transform=None):
        self._metadata = ProviderMetadata(
            name="fake",
            model=model,
            prompt_version=DEFAULT_PROMPT_VERSION,
            parameters={"temperature": 0.0},
        )
        self.transform = transform
        self.calls = []
        self.availability_checks = 0
        self.closed = False

    @property
    def metadata(self):
        return self._metadata

    def check_available(self):
        self.availability_checks += 1

    def prepare(self, request):
        self.calls.append(request)
        if self.transform is not None:
            result = self.transform(request)
        else:
            result = PreparationResult(prepared_text=request.source_text)
        if result.provider_metadata is None:
            result.provider_metadata = self.metadata
        return result

    def close(self):
        self.closed = True


class FakeHTTPResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, _limit=-1):
        return self.body


def make_single_unit_book(source_text, prepared_text):
    metadata = ProviderMetadata(name="fake", model="gemma4:31b")
    unit = PreparedUnit(
        unit_id="chapter-0000-unit-0000-fixture",
        position=0,
        kind="prose",
        source_text=source_text,
        prepared_text=prepared_text,
        provider_metadata=metadata,
    )
    chapter = PreparedChapter(
        index=0,
        title="Preface",
        source_text=source_text,
        normalized_text=source_text,
        units=[unit],
    )
    book = PreparedBook(
        title="Unicode Book",
        source_metadata=SourceMetadata(extra={"language": "Français"}),
        provider_metadata=metadata,
        chapters=[chapter],
    )
    return refresh_hashes(book)


class NormalizationAndSegmentationTests(unittest.TestCase):
    def test_normalization_is_idempotent_and_retains_logical_paragraphs(self):
        source = (
            "  Cafe\u0301 was care-\r\nfully set with a soft\u00adhyphen.  \r\n"
            "The same paragraph continues.\r\n\r\n"
            "Second\u00a0paragraph stays separate.\r\n\r\n"
            "# A Heading   \r\n\r\n***\r\n"
        )

        normalized = normalize_text(source)

        self.assertEqual(normalize_text(normalized), normalized)
        self.assertEqual(
            normalized.split("\n\n"),
            [
                "Café was carefully set with a softhyphen. The same paragraph continues.",
                "Second paragraph stays separate.",
                "# A Heading",
                "***",
            ],
        )

    def test_segmentation_preserves_structure_and_splits_only_prose(self):
        first = "The first paragraph carries enough detail to remain one coherent unit."
        second = "The second paragraph also carries enough detail to stay independent."
        third = "The final paragraph belongs to the next explicitly marked scene."
        normalized = normalize_text(
            f"# Preface\n\n{first}\n\n{second}\n\n***\n\n# Later\n\n{third}"
        )

        units = segment_text(
            normalized,
            target_chars=55,
            max_chars=90,
            context_chars=200,
        )

        self.assertEqual(
            [unit.kind for unit in units],
            ["heading", "prose", "prose", "scene_marker", "heading", "prose"],
        )
        self.assertEqual(units[0].text, "# Preface")
        self.assertEqual(units[3].text, "***")
        self.assertEqual(units[4].text, "# Later")
        self.assertEqual(
            [unit.text for unit in units if unit.kind == "prose"],
            [first, second, third],
        )
        prose = [unit for unit in units if unit.kind == "prose"]
        self.assertEqual(prose[0].following_context, second)
        self.assertEqual(prose[1].previous_context, first)
        self.assertEqual(prose[1].following_context, third)
        self.assertNotIn("***", prose[1].following_context)

    def test_oversized_prose_units_split_at_complete_sentence_boundaries(self):
        sentences = [
            "This sentence is one complete and deliberate thought.",
            "This sentence is another complete and deliberate thought.",
            "This sentence closes the complete and deliberate passage.",
        ]

        units = segment_text(
            " ".join(sentences),
            target_chars=40,
            max_chars=70,
            context_chars=0,
        )

        prose = [unit.text for unit in units if unit.kind == "prose"]
        self.assertGreater(len(prose), 1)
        self.assertEqual(" ".join(prose), " ".join(sentences))
        self.assertTrue(all(unit.endswith(".") for unit in prose))


class PreparationValidationTests(unittest.TestCase):
    def test_preface_citation_lists_are_removed_by_a_fake_provider(self):
        def adapt(_request):
            return PreparationResult(
                prepared_text=PREFACE_PREPARED,
                edits=[
                    PreparationEdit(
                        category="bibliographic_citations",
                        original="Three parenthetical author-year citation lists",
                        replacement="",
                        reason="Visual-only sourcing is difficult to listen to.",
                    )
                ],
            )

        provider = FakeProvider(transform=adapt)
        pipeline = NarrationPreparationPipeline(
            provider,
            target_unit_chars=10_000,
            max_unit_chars=12_000,
            context_chars=200,
        )

        book = pipeline.prepare_book(
            [("Preface", PREFACE_SOURCE)],
            book_title="Example",
        )

        self.assertEqual(len(provider.calls), 1)
        self.assertIn("Abelin 1975", provider.calls[0].source_text)
        self.assertEqual(book.chapters[0].prepared_text, PREFACE_PREPARED)
        self.assertNotIn("Abelin 1975", book.chapters[0].prepared_text)
        self.assertIn("Jeanne Armstrong", book.chapters[0].prepared_text)

    def test_validator_accepts_removal_of_author_year_citations(self):
        request = PreparationRequest(
            chapter_title="Preface",
            source_text=(
                "The finding (Smith 1999, Jones 2001, Patel 2004) remained "
                "important to the argument and its stated qualification."
            ),
        )
        result = PreparationResult(
            prepared_text=(
                "The finding remained important to the argument and its stated "
                "qualification."
            )
        )

        report = validate_preparation(request, result)

        self.assertEqual(report.lexical_retention, 1.0)

    def test_validator_rejects_summary_framing_even_when_source_is_retained(self):
        source = "The house was silent, and pale light crossed the hallway floor."
        request = PreparationRequest(chapter_title="Chapter One", source_text=source)

        with self.assertRaises(PreparationValidationError) as raised:
            validate_preparation(
                request,
                PreparationResult(prepared_text="In summary, " + source),
            )

        self.assertTrue(
            any("summary-style" in issue for issue in raised.exception.issues)
        )

    def test_validator_rejects_blank_provider_output(self):
        request = PreparationRequest(
            chapter_title="Chapter One",
            source_text="A complete source sentence remains available.",
        )

        with self.assertRaisesRegex(PreparationValidationError, "blank"):
            validate_preparation(request, PreparationResult(prepared_text=" \n "))


class ArtifactTests(unittest.TestCase):
    def test_unicode_artifact_round_trip_and_hash_validation(self):
        source = "Café — ‘naïve’ — λόγος — 中文."
        prepared = "Café — “naïve” — λόγος — 中文."
        book = make_single_unit_book(source, prepared)

        with TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "prepared_book.json"
            save_prepared_book(book, artifact_path)

            raw = artifact_path.read_text(encoding="utf-8")
            loaded = load_prepared_book(artifact_path)

        self.assertIn("中文", raw)
        self.assertNotIn("\\u4e2d", raw)
        self.assertEqual(loaded.chapters[0].prepared_text, prepared)
        self.assertEqual(loaded.prepared_sha256, book.prepared_sha256)

    def test_modified_artifact_text_is_rejected_when_hashes_are_stale(self):
        book = make_single_unit_book("Original source.", "Prepared narration.")

        with TemporaryDirectory() as temporary_directory:
            artifact_path = Path(temporary_directory) / "prepared_book.json"
            save_prepared_book(book, artifact_path)
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            payload["chapters"][0]["units"][0]["prepared_text"] = "Corrupted text."
            payload["chapters"][0]["prepared_text"] = "Corrupted text."
            payload["prepared_text"] = "Corrupted text."
            artifact_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ArtifactValidationError, "hash mismatch"):
                load_prepared_book(artifact_path)


class PipelineCacheTests(unittest.TestCase):
    FIRST = "Alpha detail " * 6 + "ends here."
    SECOND = "Beta detail " * 6 + "ends here."

    @staticmethod
    def pipeline(provider):
        return NarrationPreparationPipeline(
            provider,
            target_unit_chars=50,
            max_unit_chars=100,
            context_chars=40,
        )

    def test_matching_units_are_reused_without_contacting_provider(self):
        chapters = [("One", f"{self.FIRST}\n\n{self.SECOND}")]
        initial_provider = FakeProvider()
        initial = self.pipeline(initial_provider).prepare_book(
            chapters,
            book_title="Cache Test",
        )
        resumed_provider = FakeProvider()

        resumed = self.pipeline(resumed_provider).prepare_book(
            chapters,
            book_title="Cache Test",
            resume_from=initial,
        )

        self.assertEqual(len(initial_provider.calls), 2)
        self.assertEqual(resumed_provider.calls, [])
        self.assertEqual(resumed_provider.availability_checks, 0)
        self.assertTrue(
            all(unit.cache_hit for unit in resumed.chapters[0].units if unit.kind == "prose")
        )

    def test_provider_or_source_changes_invalidate_cached_units(self):
        chapters = [("One", f"{self.FIRST}\n\n{self.SECOND}")]
        original = self.pipeline(FakeProvider()).prepare_book(
            chapters,
            book_title="Cache Test",
        )

        changed_model = FakeProvider(model="gemma4:custom")
        self.pipeline(changed_model).prepare_book(
            chapters,
            book_title="Cache Test",
            resume_from=original,
        )
        changed_source = FakeProvider()
        self.pipeline(changed_source).prepare_book(
            [("One", f"{self.FIRST}\n\n{self.SECOND} changed")],
            book_title="Cache Test",
            resume_from=original,
        )

        self.assertEqual(len(changed_model.calls), 2)
        self.assertGreaterEqual(len(changed_source.calls), 1)

    def test_max_prose_units_returns_a_valid_partial_artifact(self):
        second = "Second detail " * 6 + "ends here."
        third = "Third detail " * 6 + "ends here."
        source = (
            f"# Opening\n\n{self.FIRST}\n\n***\n\n"
            f"# Later\n\n{second}\n\n{third}"
        )
        provider = FakeProvider()

        book = self.pipeline(provider).prepare_book(
            [("One", source)],
            book_title="Partial Test",
            max_prose_units=1,
        )

        self.assertFalse(book.complete)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(
            [unit.kind for unit in book.chapters[0].units],
            ["heading", "prose", "scene_marker", "heading"],
        )


class ModularWorkflowTests(unittest.TestCase):
    def test_only_prepared_text_is_passed_to_semantic_chunking(self):
        source = "SOURCE-ONLY-CITATION (Smith 1999) must never reach narration."
        prepared = "The prepared narration is the only spoken passage."
        book = make_single_unit_book(source, prepared)

        chapters = narration_chapters(book)
        plan = build_chunk_plan(chapters)
        spoken = "\n".join(
            chunk.text for _title, chunks in plan for chunk in chunks
        )

        self.assertEqual(chapters, [("Preface", prepared)])
        self.assertEqual(spoken, prepared)
        self.assertNotIn("SOURCE-ONLY-CITATION", spoken)
        self.assertNotIn("Smith 1999", spoken)


class OllamaProviderTests(unittest.TestCase):
    def test_payload_disables_streaming_and_thinking_and_parses_schema(self):
        inner_payload = {
            "prepared_text": "A clean prepared passage.",
            "edits": [
                {
                    "category": "citation",
                    "original": "(Smith 1999)",
                    "replacement": "",
                    "reason": "Visual-only citation",
                }
            ],
            "warnings": ["Fixture warning"],
        }
        response_body = json.dumps(
            {"message": {"content": json.dumps(inner_payload)}}
        ).encode("utf-8")
        captured = []

        def fake_urlopen(request, timeout):
            captured.append((request, timeout))
            return FakeHTTPResponse(response_body)

        provider = OllamaProvider(
            model="gemma4:31b",
            timeout=17.0,
            unload_on_close=False,
        )
        request = PreparationRequest(
            chapter_title="Preface",
            source_text="A clean passage (Smith 1999).",
        )

        with patch(
            "narration_preparation.providers.ollama.urlopen",
            side_effect=fake_urlopen,
        ):
            result = provider.prepare(request)

        self.assertEqual(len(captured), 1)
        sent_request, timeout = captured[0]
        payload = json.loads(sent_request.data.decode("utf-8"))
        self.assertEqual(sent_request.get_method(), "POST")
        self.assertTrue(sent_request.full_url.endswith("/api/chat"))
        self.assertEqual(timeout, 17.0)
        self.assertIs(payload["stream"], False)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["format"], RESPONSE_JSON_SCHEMA)
        self.assertEqual(payload["model"], "gemma4:31b")
        self.assertIn("A clean passage", payload["messages"][1]["content"])
        self.assertEqual(result.prepared_text, inner_payload["prepared_text"])
        self.assertEqual(result.edits[0].category, "citation")
        self.assertEqual(result.warnings, ["Fixture warning"])

    def test_malformed_transport_and_structured_json_are_rejected(self):
        provider = OllamaProvider(unload_on_close=False)
        request = PreparationRequest(
            chapter_title="One",
            source_text="A complete source passage.",
        )

        with patch(
            "narration_preparation.providers.ollama.urlopen",
            return_value=FakeHTTPResponse(b"{not-json"),
        ):
            with self.assertRaisesRegex(ProviderResponseError, "malformed JSON"):
                provider.prepare(request)

        outer = json.dumps(
            {"message": {"content": "not structured json"}}
        ).encode("utf-8")
        with patch(
            "narration_preparation.providers.ollama.urlopen",
            return_value=FakeHTTPResponse(outer),
        ):
            with self.assertRaisesRegex(ProviderResponseError, "JSON output contract"):
                provider.prepare(request)

    def test_ollama_error_and_connection_failure_have_typed_errors(self):
        provider = OllamaProvider(unload_on_close=False)
        request = PreparationRequest(
            chapter_title="One",
            source_text="A complete source passage.",
        )
        error_body = json.dumps({"error": "model runner failed"}).encode("utf-8")

        with patch(
            "narration_preparation.providers.ollama.urlopen",
            return_value=FakeHTTPResponse(error_body),
        ):
            with self.assertRaisesRegex(ProviderResponseError, "model runner failed"):
                provider.prepare(request)

        with patch(
            "narration_preparation.providers.ollama.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(ProviderUnavailableError, "ollama serve"):
                provider.prepare(request)


class CommandLineTests(unittest.TestCase):
    def test_subcommands_expose_only_their_relevant_model_options(self):
        prepare_args = make_audiobook.parse_args(["prepare"])
        narrate_args = make_audiobook.parse_args(["narrate"])
        all_args = make_audiobook.parse_args(["all"])

        self.assertEqual(prepare_args.command, "prepare")
        self.assertTrue(hasattr(prepare_args, "preparation_model"))
        self.assertFalse(hasattr(prepare_args, "tts_model"))
        self.assertEqual(narrate_args.command, "narrate")
        self.assertTrue(hasattr(narrate_args, "tts_model"))
        self.assertFalse(hasattr(narrate_args, "preparation_model"))
        self.assertEqual(all_args.command, "all")
        self.assertTrue(hasattr(all_args, "preparation_model"))
        self.assertTrue(hasattr(all_args, "tts_model"))

    def test_default_is_installed_gemma_and_is_distinct_from_qwen_tts(self):
        args = make_audiobook.parse_args([])

        self.assertEqual(args.command, "all")
        self.assertEqual(DEFAULT_PREPARATION_MODEL, "gemma4:31b")
        self.assertEqual(DEFAULT_OLLAMA_MODEL, "gemma4:31b")
        self.assertEqual(args.preparation_model, "gemma4:31b")
        self.assertIn("Qwen3-TTS", args.tts_model)
        self.assertNotEqual(args.preparation_model, args.tts_model)
        self.assertIn("Qwen3-TTS", TTS_MODEL)

    def test_preparation_and_tts_models_can_be_overridden_independently(self):
        args = make_audiobook.parse_args(
            [
                "all",
                "--preparation-model",
                "gemma4:custom",
                "--tts-model",
                "/models/qwen-custom",
            ]
        )

        self.assertEqual(args.preparation_model, "gemma4:custom")
        self.assertEqual(args.tts_model, "/models/qwen-custom")

    def test_legacy_options_route_to_the_all_workflow(self):
        args = make_audiobook.parse_args(["--dry-run", "--preview-chunks", "1"])

        self.assertEqual(args.command, "all")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.preview_chunks, 1)


if __name__ == "__main__":
    unittest.main()
