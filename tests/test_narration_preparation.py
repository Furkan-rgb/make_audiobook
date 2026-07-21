import re
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import URLError

from audiobook import cli as cli_module
from audiobook.config import DEFAULT_PREPARATION_MODEL, TTS_MODEL
from audiobook.workflow import narration_chapters
from audiobook.preparation import (
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
    ValidationPolicy,
    apply_edits,
    is_display_line,
    lexical_retention,
    load_prepared_book,
    normalize_text,
    numbered_view,
    parse_structured_response,
    refresh_hashes,
    save_prepared_book,
    segment_text,
    sentence_spans,
    validate_preparation,
)
from audiobook.preparation.providers.ollama import DEFAULT_OLLAMA_MODEL, SAMPLING_OPTIONS
from audiobook.chunking.semantic import build_chunk_plan

from support import (
    PREFACE_PREPARED,
    PREFACE_SOURCE,
    FakeHTTPResponse,
    FakeProvider,
    citation_edits,
    make_single_unit_book,
)


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


class ProviderBoundaryTests(unittest.TestCase):
    """A provider proposes; the pipeline disposes."""

    def test_the_pipeline_applies_edits_a_provider_only_reports(self):
        source = "The finding (Smith 1999) held under replication."

        def propose(_request):
            return PreparationResult(
                edits=[
                    PreparationEdit(
                        category="bibliographic_citation",
                        original="(Smith 1999)",
                        replacement="",
                        sentence=1,
                    )
                ]
            )

        provider = FakeProvider(transform=propose)
        book = NarrationPreparationPipeline(provider).prepare_book(
            [("One", source)], book_title="Example"
        )

        unit = book.chapters[0].units[0]
        self.assertEqual(unit.prepared_text, "The finding held under replication.")
        self.assertEqual(len(unit.edits), 1)

    def test_a_provider_cannot_smuggle_prose_past_the_edit_contract(self):
        # The guarantee the boundary buys: an adapter has no channel for text
        # at all, so an edit that will not anchor changes nothing, and the
        # refusal is on the record instead of silently altering the book.
        source = "The finding held under replication."

        def propose(_request):
            return PreparationResult(
                edits=[
                    PreparationEdit(
                        category="rewrite",
                        original="wording that is not in the passage",
                        replacement="Something else entirely.",
                        sentence=1,
                    )
                ]
            )

        book = NarrationPreparationPipeline(FakeProvider(transform=propose)).prepare_book(
            [("One", source)], book_title="Example"
        )

        unit = book.chapters[0].units[0]
        self.assertEqual(unit.prepared_text, source)
        self.assertEqual(unit.edits, [])
        self.assertIn("nowhere in the passage", unit.warnings[0])


class DisplayLineTests(unittest.TestCase):
    """Front matter is narrated verbatim, so it never costs a provider call."""

    FRONT_MATTER = (
        "THORNS OF REGRET\n\nBy\n\nMrs. Alex. McVeigh Miller\n\n"
        "STREET & SMITH CORPORATION PUBLISHERS 79-89 Seventh Avenue, New York\n\n"
        "Copyright, 1899 By Norman L. Munro"
    )

    def test_laid_out_lines_are_classified_apart_from_prose(self):
        prose = (
            "The day came when, through her inability to cope with the keen, "
            "avaricious man of business, he became owner of everything."
        )

        units = segment_text(normalize_text(f"{self.FRONT_MATTER}\n\n{prose}"))

        self.assertEqual(
            [unit.kind for unit in units],
            ["display_line"] * 5 + ["prose"],
        )
        self.assertEqual(units[-1].text, prose)

    def test_a_dialogue_lead_in_is_prose_however_short(self):
        # The distinction that matters on a novel: a colon ends a line that
        # introduces speech, and there are hundreds of them.
        for line in ("He laughed mockingly:", "She said hoarsely:"):
            with self.subTest(line):
                self.assertFalse(is_display_line(line))
        for line in ("AUTHOR OF", "Yours truly,", "OR,"):
            with self.subTest(line):
                self.assertTrue(is_display_line(line))

    def test_display_lines_reach_the_artifact_without_a_provider_call(self):
        provider = FakeProvider()
        pipeline = NarrationPreparationPipeline(provider)

        book = pipeline.prepare_book(
            [("Front Matter", self.FRONT_MATTER)], book_title="Example"
        )

        self.assertEqual(provider.calls, [])
        units = book.chapters[0].units
        self.assertTrue(all(unit.kind == "display_line" for unit in units))
        self.assertTrue(all(unit.prepared_text == unit.source_text for unit in units))
        self.assertEqual(book.chapters[0].prepared_text, self.FRONT_MATTER)


class PreparationValidationTests(unittest.TestCase):
    def test_preface_citation_lists_are_removed_by_a_fake_provider(self):
        def adapt(request):
            return parse_structured_response(
                {"edits": citation_edits(request.source_text), "warnings": []}
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
        prepared = (
            "The finding remained important to the argument and its stated "
            "qualification."
        )

        report = validate_preparation(request.source_text, prepared)

        self.assertEqual(report.lexical_retention, 1.0)

    def test_validator_rejects_summary_framing_even_when_source_is_retained(self):
        source = "The house was silent, and pale light crossed the hallway floor."
        request = PreparationRequest(chapter_title="Chapter One", source_text=source)

        with self.assertRaises(PreparationValidationError) as raised:
            validate_preparation(request.source_text, "In summary, " + source)

        self.assertTrue(
            any("summary-style" in issue for issue in raised.exception.issues)
        )

    def test_validator_rejects_blank_provider_output(self):
        request = PreparationRequest(
            chapter_title="Chapter One",
            source_text="A complete source sentence remains available.",
        )

        with self.assertRaisesRegex(PreparationValidationError, "blank"):
            validate_preparation(request.source_text, " \n ")


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

    def test_progress_reports_the_total_before_the_first_provider_call(self):
        chapters = [("One", f"{self.FIRST}\n\n{self.SECOND}")]
        reports: list[tuple[int, int]] = []

        self.pipeline(FakeProvider()).prepare_book(
            chapters,
            book_title="Progress Test",
            progress=lambda done, total: reports.append((done, total)),
        )

        self.assertEqual(reports, [(0, 2), (1, 2), (2, 2)])

    def test_progress_total_follows_the_preview_cap(self):
        chapters = [("One", f"{self.FIRST}\n\n{self.SECOND}")]
        reports: list[tuple[int, int]] = []

        self.pipeline(FakeProvider()).prepare_book(
            chapters,
            book_title="Progress Test",
            max_prose_units=1,
            progress=lambda done, total: reports.append((done, total)),
        )

        self.assertEqual(reports, [(0, 1), (1, 1)])


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
    def setUp(self):
        # These tests exercise the adapter's own mechanics, so pin them to its
        # code defaults instead of whatever PREPARATION_PROVIDERS the project
        # ships. Without this, the config-driven `think` fallback (and any future
        # config default) leaks in — e.g. raising num_predict out from under the
        # truncation test — so the suite would fail on a config edit, not a code
        # bug. The dedicated fallback tests below re-patch this per case.
        patcher = patch(
            "audiobook.preparation.providers.ollama._configured",
            return_value={},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_payload_disables_streaming_and_thinking_and_applies_returned_edits(self):
        inner_payload = {
            "edits": [
                {
                    "sentence": 1,
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
            "audiobook.preparation.providers.ollama.urlopen",
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
        self.assertIn("1: A clean passage", payload["messages"][1]["content"])
        self.assertNotIn("prepared_text", payload["format"]["properties"])
        # A provider reports what the model proposed and nothing more: no
        # prepared prose, because deciding what the passage becomes is the
        # pipeline's job and must not vary by adapter.
        self.assertFalse(hasattr(result, "prepared_text"))
        self.assertEqual(result.edits[0].category, "citation")
        self.assertEqual(result.edits[0].sentence, 1)
        self.assertEqual(result.edits[0].original, "(Smith 1999)")
        self.assertEqual(result.warnings, ["Fixture warning"])
        self.assertEqual(result.provider_metadata.model, "gemma4:31b")

    def test_think_stays_off_and_leaves_budgets_untouched_without_config(self):
        # With no config entry the adapter's own default governs: thinking off,
        # and the direct-run context/output budgets left as constructed.
        provider = OllamaProvider(unload_on_close=False)
        self.assertIs(provider.think, False)
        self.assertEqual((provider.num_ctx, provider.num_predict), (8192, 4096))
        self.assertIs(provider.metadata.parameters["think"], False)

    def test_unset_think_defers_to_config_and_raises_budgets(self):
        # An unset think reads the project config, mirroring auto_pull, so a run
        # can turn reasoning on from config.py alone. A thinking model emits its
        # trace ahead of the JSON, so both budgets are floored up to fit it.
        with patch(
            "audiobook.preparation.providers.ollama._configured",
            return_value={"think": True},
        ):
            provider = OllamaProvider(unload_on_close=False)
        self.assertIs(provider.think, True)
        self.assertGreaterEqual(provider.num_ctx, 16384)
        self.assertGreaterEqual(provider.num_predict, 8192)
        self.assertIs(provider.metadata.parameters["think"], True)

    def test_explicit_think_overrides_config_in_both_directions(self):
        # The benchmark path passes think explicitly per variant, so an explicit
        # value must win over config whichever way the two disagree.
        with patch(
            "audiobook.preparation.providers.ollama._configured",
            return_value={"think": True},
        ):
            forced_off = OllamaProvider(think=False, unload_on_close=False)
        self.assertIs(forced_off.think, False)
        self.assertEqual(
            (forced_off.num_ctx, forced_off.num_predict), (8192, 4096)
        )

        with patch(
            "audiobook.preparation.providers.ollama._configured",
            return_value={"think": False},
        ):
            forced_on = OllamaProvider(think=True, unload_on_close=False)
        self.assertIs(forced_on.think, True)

    def test_enabled_think_is_sent_in_the_chat_payload(self):
        # Guards the whole path config -> provider.think -> request body. The
        # sibling "disables" test only sees think off, so a regression that
        # hard-coded think off (or dropped it from the payload) would let a
        # thinking run silently degrade to a direct answer with nothing red.
        captured = []

        def fake_urlopen(request, timeout):
            captured.append(request)
            return FakeHTTPResponse(
                json.dumps(
                    {"message": {"content": json.dumps({"edits": [], "warnings": []})}}
                ).encode("utf-8")
            )

        with patch(
            "audiobook.preparation.providers.ollama._configured",
            return_value={"think": True},
        ):
            provider = OllamaProvider(unload_on_close=False)
        request = PreparationRequest(
            chapter_title="One", source_text="A complete source passage."
        )
        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.prepare(request)

        payload = json.loads(captured[0].data.decode("utf-8"))
        self.assertIs(payload["think"], True)

    def _sent_payload(self, provider):
        """Prepare one passage through ``provider`` and return the request body."""

        captured = []

        def fake_urlopen(request, timeout):
            captured.append(request)
            return FakeHTTPResponse(
                json.dumps(
                    {"message": {"content": json.dumps({"edits": [], "warnings": []})}}
                ).encode("utf-8")
            )

        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.prepare(
                PreparationRequest(
                    chapter_title="One", source_text="A complete source passage."
                )
            )
        return json.loads(captured[0].data.decode("utf-8"))

    def test_native_sampling_omits_every_sampling_option_but_keeps_the_budgets(self):
        # The benchmark builds the provider with temperature=None, meaning
        # "inherit the model package's default": no sampling option is sent, so
        # the model's own policy governs, while the controlled budgets, seed, and
        # think mode are all still explicit.
        provider = OllamaProvider(temperature=None, unload_on_close=False)
        payload = self._sent_payload(provider)
        options = payload["options"]

        for name in SAMPLING_OPTIONS:
            # Absent, not present-with-null: the key must not be in the request.
            self.assertNotIn(name, options)
        self.assertEqual(options["seed"], 42)
        self.assertEqual(options["num_ctx"], 8192)
        self.assertEqual(options["num_predict"], 4096)
        self.assertIs(payload["think"], False)
        self.assertIs(provider.native_sampling, True)
        self.assertIs(provider.metadata.parameters["native_sampling"], True)
        self.assertNotIn("temperature", provider.metadata.parameters)

    def test_supplied_sampling_options_are_sent_and_the_rest_stay_absent(self):
        # A value given for an option is sent and overrides the model default;
        # the options left as None remain omitted, and their absence is what lets
        # the package supply them.
        provider = OllamaProvider(
            temperature=None, top_p=0.9, repeat_penalty=1.2, unload_on_close=False
        )
        options = self._sent_payload(provider)["options"]

        self.assertEqual(options["top_p"], 0.9)
        self.assertEqual(options["repeat_penalty"], 1.2)
        self.assertNotIn("temperature", options)
        self.assertNotIn("top_k", options)
        self.assertIs(provider.native_sampling, False)

    def test_production_default_uses_native_sampling(self):
        # Native sampling is the default for normal production preparation too,
        # not only the benchmark: a provider built without naming a sampling
        # policy — the way production constructs it — omits every sampling option
        # so the model package's own defaults govern generation.
        provider = OllamaProvider(unload_on_close=False)
        options = self._sent_payload(provider)["options"]

        for name in SAMPLING_OPTIONS:
            self.assertNotIn(name, options)
        # The controlled budgets and seed are still explicit.
        self.assertEqual(options["seed"], 42)
        self.assertIn("num_ctx", options)
        self.assertIn("num_predict", options)
        self.assertIs(provider.native_sampling, True)
        self.assertIs(provider.metadata.parameters["native_sampling"], True)

    def test_native_sampling_keeps_the_mode_context_and_output_floors(self):
        # The budgets are capacity ceilings, not quality tuning, so they hold at
        # their per-mode floors regardless of the sampling policy.
        direct = OllamaProvider(temperature=None, think=False, unload_on_close=False)
        self.assertGreaterEqual(direct.num_ctx, 8192)
        self.assertGreaterEqual(direct.num_predict, 4096)

        thinking = OllamaProvider(temperature=None, think=True, unload_on_close=False)
        self.assertGreaterEqual(thinking.num_ctx, 16384)
        self.assertGreaterEqual(thinking.num_predict, 8192)

    def test_malformed_transport_and_structured_json_are_rejected(self):
        provider = OllamaProvider(unload_on_close=False)
        request = PreparationRequest(
            chapter_title="One",
            source_text="A complete source passage.",
        )

        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            return_value=FakeHTTPResponse(b"{not-json"),
        ):
            with self.assertRaisesRegex(ProviderResponseError, "malformed JSON"):
                provider.prepare(request)

        outer = json.dumps(
            {"message": {"content": "not structured json"}}
        ).encode("utf-8")
        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            return_value=FakeHTTPResponse(outer),
        ):
            with self.assertRaisesRegex(ProviderResponseError, "JSON output contract"):
                provider.prepare(request)

    def test_a_truncated_response_names_the_output_limit_that_caused_it(self):
        provider = OllamaProvider(num_predict=1024, unload_on_close=False)
        truncated = json.dumps(
            {
                "message": {"content": '{"edits": [{"sentence": 1, "original": "abc'},
                "done_reason": "length",
            }
        ).encode("utf-8")

        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            return_value=FakeHTTPResponse(truncated),
        ):
            with self.assertRaisesRegex(ProviderResponseError, "1024-token output limit"):
                provider.prepare(
                    PreparationRequest(
                        chapter_title="One", source_text="A complete source passage."
                    )
                )

    def test_ollama_error_and_connection_failure_have_typed_errors(self):
        provider = OllamaProvider(unload_on_close=False)
        request = PreparationRequest(
            chapter_title="One",
            source_text="A complete source passage.",
        )
        error_body = json.dumps({"error": "model runner failed"}).encode("utf-8")

        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            return_value=FakeHTTPResponse(error_body),
        ):
            with self.assertRaisesRegex(ProviderResponseError, "model runner failed"):
                provider.prepare(request)

        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(ProviderUnavailableError, "ollama serve"):
                provider.prepare(request)

    def test_missing_model_is_pulled_and_then_reverified(self):
        progress = []
        provider = OllamaProvider(
            model="gemma4:12b",
            unload_on_close=False,
            on_pull_progress=progress.append,
        )
        empty_tags = json.dumps({"models": []}).encode("utf-8")
        installed_tags = json.dumps(
            {"models": [{"name": "gemma4:12b"}]}
        ).encode("utf-8")
        pull_stream = b"\n".join(
            [
                json.dumps({"status": "pulling manifest"}).encode("utf-8"),
                json.dumps(
                    {
                        "status": "pulling 5f4c",
                        "completed": 6 * 1024**3,
                        "total": 12 * 1024**3,
                    }
                ).encode("utf-8"),
                json.dumps({"status": "success"}).encode("utf-8"),
            ]
        )
        endpoints = []

        def fake_urlopen(request, timeout):
            endpoints.append(request.full_url.rsplit("/", 1)[-1])
            if request.full_url.endswith("/api/pull"):
                return FakeHTTPResponse(pull_stream)
            # The first probe finds nothing; the one after the pull finds it.
            return FakeHTTPResponse(
                installed_tags if "pull" in endpoints else empty_tags
            )

        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            side_effect=fake_urlopen,
        ):
            provider.check_available()

        self.assertEqual(endpoints, ["tags", "pull", "tags"])
        self.assertIn("50% of 12.0 GB", "\n".join(progress))
        self.assertTrue(any("is not installed" in line for line in progress))

    def test_failed_pull_and_disabled_auto_pull_report_the_manual_command(self):
        empty_tags = json.dumps({"models": []}).encode("utf-8")

        manual = OllamaProvider(
            model="gemma4:12b", unload_on_close=False, auto_pull=False
        )
        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            return_value=FakeHTTPResponse(empty_tags),
        ):
            with self.assertRaisesRegex(
                ProviderUnavailableError, "ollama pull gemma4:12b"
            ):
                manual.check_available()

        automatic = OllamaProvider(
            model="typo:12b", unload_on_close=False, on_pull_progress=lambda _line: None
        )
        pull_error = json.dumps({"error": "file does not exist"}).encode("utf-8")

        def fake_urlopen(request, timeout):
            if request.full_url.endswith("/api/pull"):
                return FakeHTTPResponse(pull_error)
            return FakeHTTPResponse(empty_tags)

        with patch(
            "audiobook.preparation.providers.ollama.urlopen",
            side_effect=fake_urlopen,
        ):
            with self.assertRaisesRegex(ProviderResponseError, "could not pull"):
                automatic.check_available()


class CommandLineTests(unittest.TestCase):
    def test_subcommands_expose_only_their_relevant_model_options(self):
        prepare_args = cli_module.parse_args(["prepare"])
        narrate_args = cli_module.parse_args(["narrate"])
        all_args = cli_module.parse_args(["all"])

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
        args = cli_module.parse_args([])

        self.assertEqual(args.command, "all")
        # The configured default and the adapter's standalone fallback (used when
        # the preparation package runs without a project config) are kept in step.
        self.assertEqual(DEFAULT_PREPARATION_MODEL, "gemma4:26b")
        self.assertEqual(DEFAULT_OLLAMA_MODEL, "gemma4:26b")
        self.assertEqual(args.preparation_model, "gemma4:26b")
        self.assertIn("Qwen3-TTS", args.tts_model)
        self.assertNotEqual(args.preparation_model, args.tts_model)
        self.assertIn("Qwen3-TTS", TTS_MODEL)

    def test_preparation_and_tts_models_can_be_overridden_independently(self):
        args = cli_module.parse_args(
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
        args = cli_module.parse_args(["--dry-run", "--preview-chunks", "1"])

        self.assertEqual(args.command, "all")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.preview_chunks, 1)


if __name__ == "__main__":
    unittest.main()
