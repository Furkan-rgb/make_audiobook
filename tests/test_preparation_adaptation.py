"""Tests for the edits-only adaptation contract.

These cover the subsystem that decides what a model's proposed edits do to a
passage: how the passage is addressed, how a quoted edit is anchored in it,
which edits are refused, and how the survivors are spliced in.
"""

import re
import unittest

from audiobook.preparation import (
    PreparationEdit,
    ValidationPolicy,
    apply_edits,
    lexical_retention,
    normalize_text,
    numbered_view,
    sentence_spans,
)

from support import PREFACE_PREPARED, PREFACE_SOURCE, citation_edits


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

    def test_an_edit_that_would_empty_its_paragraph_is_refused(self):
        # Front matter is one short paragraph per line, so trimming "By" and
        # "AUTHOR OF" hollows the structure out a line at a time: the words go
        # but the separators stay, stacking blank lines.
        source = "By\n\nMrs. Alex. McVeigh Miller\n\nTHORNS OF REGRET"

        prepared, applied, warnings = apply_edits(
            source,
            [
                PreparationEdit(
                    category="extraction_artifact",
                    original="By",
                    replacement="",
                    sentence=1,
                )
            ],
        )

        self.assertEqual(prepared, source)
        self.assertEqual(applied, [])
        self.assertIn("leave its paragraph empty", warnings[0])

    def test_individually_legal_edits_that_together_strip_a_passage_back_off(self):
        # Every edit here is small and legal on its own; together they take the
        # passage under the retention floor. That must cost the edits, not the
        # unit — a run of a hundred units cannot die on one title page.
        source = (
            "Alpha bravo charlie delta.\n\nEcho foxtrot golf hotel.\n\n"
            "India juliett kilo lima.\n\nMike november oscar papa."
        )
        edits = [
            PreparationEdit(
                category="extraction_artifact",
                original=original,
                replacement="",
                sentence=number,
            )
            for number, original in enumerate(
                ("Alpha bravo", "Echo foxtrot", "India juliett", "Mike november"),
                start=1,
            )
        ]

        prepared, applied, warnings = apply_edits(source, edits)

        self.assertLess(len(applied), len(edits))
        self.assertGreaterEqual(
            lexical_retention(source, prepared), ValidationPolicy().minimum_lexical_retention
        )
        self.assertTrue(any("together the edits cut more" in w for w in warnings))
        # Backing off restores whole words, never partial ones.
        self.assertEqual(source.count("\n\n"), prepared.count("\n\n"))

    def test_a_replacement_does_not_import_its_own_spacing(self):
        prepared, applied, _warnings = apply_edits(
            "Mrs. Alex. McVeigh Miller wrote it.",
            [
                PreparationEdit(
                    category="extraction_artifact",
                    original="Mrs. Alex.",
                    replacement="Mrs. Alex ",
                    sentence=1,
                )
            ],
        )

        self.assertEqual(len(applied), 1)
        self.assertNotIn("  ", prepared)
        self.assertEqual(prepared, "Mrs. Alex McVeigh Miller wrote it.")

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

    def test_edits_too_large_to_be_adaptations_are_refused(self):
        # One rule covers every shape of over-reach, which is why these are one
        # test: an edit may change only a handful of words, counted with
        # citations masked out. Deleting a clause, paraphrasing it, and padding
        # it with commentary are the same mistake seen from different angles.
        sentence = (
            "The committee recorded a series of objections that the author "
            "considered essential to the argument and repeated at length."
        )
        passage = sentence + " A second sentence stands here."
        for label, original, replacement in (
            ("deletes a clause outright", sentence, ""),
            ("paraphrases it", sentence, "The committee objected at length."),
            (
                "pads it with commentary",
                "objections",
                "objections, which the reader may recall from the earlier "
                "discussion of the committee's founding charter",
            ),
        ):
            with self.subTest(label):
                prepared, applied, warnings = apply_edits(
                    passage,
                    [
                        PreparationEdit(
                            category="rewrite",
                            original=original,
                            replacement=replacement,
                            sentence=1,
                        )
                    ],
                )

                self.assertEqual(prepared, passage)
                self.assertEqual(applied, [])
                self.assertIn("an adaptation may change in one edit", warnings[0])

    def test_a_citation_list_is_free_to_delete_however_long_it_is(self):
        # The counterpart to the rule above, and the reason it counts words
        # rather than characters: this deletion is longer than any of them and
        # costs nothing, because masked citations leave no words behind.
        citations = (
            "(Abelin 1975, Greenacre 1957, Greenson 1968, Greenspan 1982, "
            "Kohlberg 1966, LaTorre 1979, Mahler 1955, Moberly 1983)"
        )
        passage = f"The development of gender-identity {citations} took decades."

        prepared, applied, warnings = apply_edits(
            passage,
            [
                PreparationEdit(
                    category="bibliographic_citation",
                    original=citations,
                    replacement="",
                    sentence=1,
                )
            ],
        )

        self.assertEqual(warnings, [])
        self.assertEqual(len(applied), 1)
        self.assertEqual(
            prepared, "The development of gender-identity took decades."
        )

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

    def test_a_citation_dense_passage_survives_a_full_pass(self):
        source = normalize_text(PREFACE_SOURCE)

        prepared, applied, warnings = apply_edits(
            source,
            [PreparationEdit.from_dict(item) for item in citation_edits(source)],
        )

        self.assertEqual(warnings, [])
        self.assertEqual(len(applied), 3)
        self.assertEqual(prepared, PREFACE_PREPARED)
