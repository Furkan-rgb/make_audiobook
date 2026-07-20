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
from audiobook.preparation.providers.ollama import DEFAULT_OLLAMA_MODEL
from audiobook.chunking.semantic import build_chunk_plan


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

    def __iter__(self):
        # urlopen responses iterate line by line; /api/pull streams NDJSON.
        return iter(self.body.splitlines(keepends=True))


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


def _citation_edits(source: str) -> list[dict[str, object]]:
    """What a well-behaved model returns for the preface: three deletions.

    Built by locating the parentheticals rather than by hand so the fixture
    states the model's *intent* — remove the author-year lists — while the
    exact characters, and therefore the spacing of the result, are the
    pipeline's problem to get right.
    """

    spans = sentence_spans(source)
    edits: list[dict[str, object]] = []
    for match in re.finditer(r"\([^()]*\b\d{4}\b[^()]*\)", source):
        sentence = next(
            number
            for number, (start, end) in enumerate(spans, start=1)
            if start <= match.start() and match.end() <= end
        )
        edits.append(
            {
                "sentence": sentence,
                "category": "bibliographic_citation",
                "original": match.group(0),
                "replacement": "",
                "reason": "Visual-only sourcing is difficult to listen to.",
            }
        )
    return edits


class EditApplicationTests(unittest.TestCase):
    PASSAGE = (
        "Kahneman and Tversky (1979) showed that losses loom larger.[3] "
        "The effect survived replication.\n\n"
        "Later work qualified the size of the asymmetry (Gal 2006)."
    )

    def test_a_passage_with_no_edits_is_returned_byte_identical(self):
        prepared, applied, warnings = apply_edits(self.PASSAGE, [])

        self.assertEqual(prepared, self.PASSAGE)
        self.assertEqual((applied, warnings), ([], []))

    def test_numbering_matches_the_spans_edits_are_anchored_to(self):
        spans = sentence_spans(self.PASSAGE)
        view = numbered_view(self.PASSAGE)

        self.assertEqual(len(spans), 3)
        for number, (start, end) in enumerate(spans, start=1):
            self.assertIn(f"{number}: {self.PASSAGE[start:end]}", view)
        # The paragraph break survives as a blank line, so the model can see
        # that sentence 3 begins a new paragraph.
        self.assertIn("\n\n3: ", view)
        # The label must not look like the "[n]" reference markers the model
        # is told to delete, or it proposes deleting the labels themselves.
        # The passage's own "[3]" footnote marker must survive untouched.
        self.assertTrue(
            all(re.fullmatch(r"\d+: .*", line) for line in view.split("\n") if line)
        )
        self.assertIn("larger.[3]", view)

    def test_deletions_close_up_the_spacing_they_leave_behind(self):
        prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="citation", original="(1979)", replacement="", sentence=1
                ),
                PreparationEdit(
                    category="marker", original="[3]", replacement="", sentence=1
                ),
                PreparationEdit(
                    category="citation", original=" (Gal 2006)", replacement="", sentence=3
                ),
            ],
        )

        self.assertEqual(warnings, [])
        self.assertEqual(len(applied), 3)
        self.assertEqual(
            prepared,
            "Kahneman and Tversky showed that losses loom larger. "
            "The effect survived replication.\n\n"
            "Later work qualified the size of the asymmetry.",
        )

    def test_a_misquoted_original_is_dropped_and_the_source_survives(self):
        prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="citation",
                    original="(Kahneman 1979)",
                    replacement="",
                    sentence=1,
                )
            ],
        )

        self.assertEqual(prepared, self.PASSAGE)
        self.assertEqual(applied, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("nowhere in the passage", warnings[0])

    def test_typographic_differences_in_the_quote_still_anchor(self):
        source = 'She said “it was over” — plainly (Reed 1998).'

        prepared, applied, _warnings = apply_edits(
            source,
            [
                PreparationEdit(
                    category="citation",
                    original=' (Reed 1998)',
                    replacement="",
                    sentence=1,
                ),
                PreparationEdit(
                    category="notation",
                    original='"it was over" -',
                    replacement="it was over,",
                    sentence=1,
                ),
            ],
        )

        self.assertEqual(len(applied), 2)
        self.assertEqual(prepared, "She said it was over, plainly.")

    def test_an_ambiguous_original_is_refused_rather_than_guessed_at(self):
        source = "He read (1994) and she read (1994) that winter."

        prepared, applied, warnings = apply_edits(
            source,
            [
                PreparationEdit(
                    category="citation", original="(1994)", replacement="", sentence=1
                )
            ],
        )

        self.assertEqual(prepared, source)
        self.assertEqual(applied, [])
        self.assertIn("more than once", warnings[0])

    def test_an_off_by_one_anchor_recovers_when_the_wording_is_unique(self):
        # Down a column of dialogue a 12B model reliably slips a line; the
        # quoted text is unambiguous, so refusing it over the number would
        # discard a good edit for a clerical error.
        prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="citation",
                    original="(Gal 2006)",
                    replacement="",
                    sentence=2,  # actually sentence 3
                )
            ],
        )

        self.assertEqual(warnings, [])
        self.assertEqual(len(applied), 1)
        # The record shows where the edit landed, not the model's miscount.
        self.assertEqual(applied[0].sentence, 3)
        self.assertNotIn("Gal 2006", prepared)

    def test_a_misanchored_edit_whose_wording_repeats_elsewhere_is_refused(self):
        source = (
            "The word garden appears here. It names a garden of another kind. "
            "The third sentence mentions no such place at all."
        )

        prepared, applied, warnings = apply_edits(
            source,
            [
                PreparationEdit(
                    category="artifact",
                    original="garden",
                    replacement="orchard",
                    sentence=3,
                )
            ],
        )

        self.assertEqual(prepared, source)
        self.assertEqual(applied, [])
        self.assertIn("more than one other sentence", warnings[0])

    def test_wording_the_model_invented_is_reported_as_nowhere(self):
        prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="visual_notation",
                    original="“The effect survived replication.”",
                    replacement="The effect survived replication.",
                    sentence=2,
                )
            ],
        )

        # The source has no quotation marks there: the model fabricated them,
        # and a fabricated original must not fuzzy-match its way into a splice.
        self.assertEqual(prepared, self.PASSAGE)
        self.assertEqual(applied, [])
        self.assertIn("nowhere in the passage", warnings[0])

    def test_a_quote_that_includes_its_view_label_still_anchors(self):
        # gemma quotes the line as it saw it, label and all. The label is our
        # own injection into the prompt, so removing it restores the model's
        # words rather than guessing at them.
        prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="citation",
                    original="3: Later work qualified the size of the asymmetry (Gal 2006).",
                    replacement="Later work qualified the size of the asymmetry.",
                    sentence=3,
                )
            ],
        )

        self.assertEqual(warnings, [])
        self.assertEqual(len(applied), 1)
        self.assertTrue(prepared.endswith("asymmetry."))
        self.assertIn("\n\n", prepared)

    def test_edits_aimed_at_the_labels_themselves_are_ignored_in_bulk(self):
        # The failure this format change fixes: a model told to delete "[n]"
        # reference markers saw "[n]" labels and proposed deleting every one.
        prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="reference_marker",
                    original=f"{number}:",
                    replacement="",
                    sentence=number,
                )
                for number in (1, 2, 3)
            ],
        )

        self.assertEqual(prepared, self.PASSAGE)
        self.assertEqual(applied, [])
        # One aggregate line, not three near-identical ones: they say nothing
        # about the book a reviewer has to judge.
        self.assertEqual(len(warnings), 1)
        self.assertIn("Ignored 3 proposed edits", warnings[0])

    def test_a_match_across_a_paragraph_break_is_refused(self):
        # The folds map newlines to spaces for matching, so a quote spanning
        # two paragraphs can locate — but splicing it would delete the break,
        # the one piece of structure no edit may touch.
        prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="visual_notation",
                    original="replication.\n\nLater work",
                    replacement="replication. Later work",
                    sentence=2,
                )
            ],
        )

        self.assertEqual(prepared, self.PASSAGE)
        self.assertEqual(applied, [])
        self.assertIn("crosses a paragraph break", warnings[0])

    def test_deleting_a_long_run_of_prose_outright_is_refused(self):
        # Deletion has no replacement for the retention check to inspect, so
        # it gets its own ceiling; contrast with the preface test, where far
        # longer deletions pass because they are citation-shaped.
        sentence = (
            "The committee recorded a series of objections that the author "
            "considered essential to the argument and repeated at length."
        )

        prepared, applied, warnings = apply_edits(
            sentence + " A second sentence stands here.",
            [
                PreparationEdit(
                    category="extraction_artifact",
                    original=sentence,
                    replacement="",
                    sentence=1,
                )
            ],
        )

        self.assertIn(sentence, prepared)
        self.assertEqual(applied, [])
        self.assertIn("deletes 122 characters of prose outright", warnings[0])

    def test_a_long_span_rewrite_that_discards_the_authors_words_is_refused(self):
        sentence = (
            "The experiment continued for eleven further weeks despite the "
            "objections recorded by the committee in its interim report."
        )

        _prepared, applied, warnings = apply_edits(
            sentence,
            [
                PreparationEdit(
                    category="visual_notation",
                    original=sentence,
                    replacement="The study went on for months over objections.",
                    sentence=1,
                )
            ],
        )

        self.assertEqual(applied, [])
        self.assertIn("paraphrase rather than adaptation", warnings[0])

    def test_overlapping_edits_keep_the_first_and_report_the_second(self):
        prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="citation", original="(1979)", replacement="", sentence=1
                ),
                PreparationEdit(
                    category="citation",
                    original="Tversky (1979) showed",
                    replacement="Tversky showed",
                    sentence=1,
                ),
            ],
        )

        self.assertEqual(len(applied), 1)
        self.assertIn("overlaps an earlier edit", warnings[0])
        self.assertIn("Kahneman and Tversky showed", prepared)

    def test_a_nonsense_anchor_still_recovers_unique_wording(self):
        # An anchor that does not even exist is the same clerical error as an
        # off-by-one, so unique wording is still worth recovering; sentence=0
        # additionally covers edits from pre-anchor artifacts.
        for sentence in (9, 0, -1):
            with self.subTest(sentence=sentence):
                prepared, applied, warnings = apply_edits(
                    self.PASSAGE,
                    [
                        PreparationEdit(
                            category="c",
                            original="losses",
                            replacement="setbacks",
                            sentence=sentence,
                        )
                    ],
                )

                self.assertEqual(warnings, [])
                self.assertEqual(applied[0].sentence, 1)
                self.assertIn("setbacks loom larger", prepared)

    def test_edits_that_cannot_anchor_at_all_are_refused(self):
        for edit, expected in (
            (
                PreparationEdit(category="c", original="", replacement="x", sentence=1),
                "no original text",
            ),
            (
                PreparationEdit(
                    category="c", original="absent words", replacement="x", sentence=9
                ),
                "sentence 9 does not exist",
            ),
        ):
            with self.subTest(original=edit.original):
                prepared, applied, warnings = apply_edits(self.PASSAGE, [edit])

                self.assertEqual(prepared, self.PASSAGE)
                self.assertEqual(applied, [])
                self.assertEqual(len(warnings), 1)
                self.assertIn(expected, warnings[0])

    def test_a_rewrite_wearing_an_edits_clothes_is_refused(self):
        source = "A" * 500 + ". A short second sentence."

        _prepared, applied, warnings = apply_edits(
            source,
            [
                PreparationEdit(
                    category="rewrite",
                    original="A" * 500,
                    replacement="Something else entirely.",
                    sentence=1,
                )
            ],
        )

        self.assertEqual(applied, [])
        self.assertIn("more than the 400", warnings[0])

    def test_an_inflating_replacement_is_refused(self):
        _prepared, applied, warnings = apply_edits(
            self.PASSAGE,
            [
                PreparationEdit(
                    category="expansion",
                    original="losses",
                    replacement="losses, which is to say the perceived forfeiture of "
                    "something already held, a notion the authors return to",
                    sentence=1,
                )
            ],
        )

        self.assertEqual(applied, [])
        self.assertIn("adds rather than adapts", warnings[0])

    def test_rewriting_prose_wholesale_runs_out_of_budget(self):
        sentences = [f"Sentence {number:02d} of the passage stands here." for number in range(20)]
        source = " ".join(sentences)
        # No slack, so the budget is exactly a quarter of the passage and the
        # arithmetic in the assertion is the policy, not a coincidence.
        policy = ValidationPolicy(maximum_edited_fraction=0.25, edited_slack_chars=0)
        affordable = int(len(source) * 0.25) // len(sentences[0])
        rewrites = [
            PreparationEdit(
                category="rewrite",
                original=sentence,
                replacement=f"Sentence {number:02d} stands here.",
                sentence=number + 1,
            )
            for number, sentence in enumerate(sentences)
        ]

        _prepared, applied, warnings = apply_edits(source, rewrites, policy=policy)

        self.assertEqual(len(applied), affordable)
        self.assertEqual(len(warnings), len(sentences) - affordable)
        self.assertTrue(all("rewrite 25%" in warning for warning in warnings))

    def test_a_citation_dense_passage_is_not_capped_by_the_budget(self):
        source = normalize_text(PREFACE_SOURCE)

        prepared, applied, warnings = apply_edits(
            source,
            [PreparationEdit.from_dict(item) for item in _citation_edits(source)],
        )

        self.assertEqual(warnings, [])
        self.assertEqual(len(applied), 3)
        self.assertEqual(prepared, PREFACE_PREPARED)


class PreparationValidationTests(unittest.TestCase):
    def test_preface_citation_lists_are_removed_by_a_fake_provider(self):
        def adapt(request):
            return parse_structured_response(
                request,
                {
                    "edits": _citation_edits(request.source_text),
                    "warnings": [],
                },
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
        # The passage is the source with the edit spliced out, not anything the
        # model retyped: it never sent prose back at all.
        self.assertEqual(result.prepared_text, "A clean passage.")
        self.assertEqual(result.edits[0].category, "citation")
        self.assertEqual(result.edits[0].sentence, 1)
        self.assertEqual(result.warnings, ["Fixture warning"])

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
        self.assertEqual(DEFAULT_PREPARATION_MODEL, "gemma4:12b")
        self.assertEqual(DEFAULT_OLLAMA_MODEL, "gemma4:12b")
        self.assertEqual(args.preparation_model, "gemma4:12b")
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
