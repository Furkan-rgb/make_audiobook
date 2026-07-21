"""Tests for the gold-corpus preparation benchmark.

The corpus is only worth running if every case in it is provably fair, and the
score is only worth reading if it separates the failures that matter from the
ones that do not. Both claims are tested here: the shipped corpus is linted in
full, and the scorer is driven by providers whose right answer is known in
advance — including the one that quietly changes a word, which is the failure
the previous benchmark could not see.
"""

import io
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from audiobook import cli
from audiobook.benchmarking import (
    BASE_SEED,
    CATEGORIES,
    TIERS,
    BenchmarkOptions,
    CorpusError,
    benchmark_preparation,
    change_regions,
    default_output_dir,
    load_corpus,
    print_summary,
    run,
    score_case,
    seed_for_repetition,
)
from audiobook.benchmarking.corpus import case_from_dict, lint_case
from audiobook.preparation import (
    DEFAULT_PROMPT_VERSION,
    PreparationEdit,
    PreparationResult,
    ProviderMetadata,
    apply_edits,
)


CITATION_SOURCE = (
    "The finding (Smith 1999) remained central to the argument, and the "
    "committee reaffirmed it in 1974. A second sentence preserves every "
    "substantive qualification."
)
CITATION_PREPARED = (
    "The finding remained central to the argument, and the committee "
    "reaffirmed it in 1974. A second sentence preserves every substantive "
    "qualification."
)
NOOP_SOURCE = (
    "“You never asked,” she said, and set the cup down without looking at "
    "him. The rain had not let up all afternoon, and the window was grey "
    "with it."
)


def citation_payload(**overrides):
    payload = {
        "id": "fixture-citation",
        "tier": "trap",
        "categories": ["bibliographic_citation"],
        "source": CITATION_SOURCE,
        "expect": [
            {
                "anchor": "(Smith 1999)",
                "accept": [""],
                "category": "bibliographic_citation",
                "why": "Visual-only sourcing.",
            }
        ],
        "traps": [
            {"span": "in 1974", "label": "historical-year-must-stay"},
            {
                "span": "every substantive qualification",
                "label": "qualification-must-stay",
            },
        ],
        "prepared": CITATION_PREPARED,
    }
    payload.update(overrides)
    return payload


def noop_payload(**overrides):
    payload = {
        "id": "fixture-noop",
        "tier": "noop",
        "categories": ["no_edit"],
        "source": NOOP_SOURCE,
        "expect": [],
        "traps": [{"span": "“You never asked,”", "label": "dialogue-quotes-must-stay"}],
        "prepared": NOOP_SOURCE,
    }
    payload.update(overrides)
    return payload


CITATION_CASE = case_from_dict(citation_payload())
NOOP_CASE = case_from_dict(noop_payload())

GOLD_EDIT = CITATION_CASE.expect[0].as_edit()
# Deletes a historical year the corpus marks as text that must survive.
TRAP_EDIT = PreparationEdit(
    category="bibliographic_citation",
    original="in 1974",
    replacement="",
    reason="Looks like a citation.",
    sentence=1,
)
# Swaps a comma for a semicolon: unrequested, but it changes no words.
COSMETIC_EDIT = PreparationEdit(
    category="visual_notation",
    original="argument,",
    replacement="argument;",
    reason="Punctuation preference.",
    sentence=1,
)
# Quoted wrongly, so it can never be placed in the passage.
UNANCHORABLE_EDIT = PreparationEdit(
    category="bibliographic_citation",
    original="(Smith, 1999)",
    replacement="",
    reason="Retyped rather than copied.",
    sentence=1,
)
# A rewrite wearing an edit's clothes; the applier refuses it on size.
OVERSIZED_EDIT = PreparationEdit(
    category="visual_notation",
    original="remained central to the argument, and the committee reaffirmed it",
    replacement="was important",
    reason="Shorter.",
    sentence=1,
)


def citation_deletion_case(source, *, anchor="(Smith 1999)", prepared=None, cid="fixture-adjacent"):
    """A case whose one expected edit is deleting ``anchor`` from ``source``.

    Built straight from a payload rather than through the applier because these
    scoring fixtures feed ``score_case`` a hand-written model output directly:
    the point is to control the exact diff against the source, which a real
    applier run would not let a test do.
    """

    return case_from_dict(
        {
            "id": cid,
            "tier": "trap",
            "categories": ["bibliographic_citation"],
            "source": source,
            "expect": [
                {
                    "anchor": anchor,
                    "accept": [""],
                    "category": "bibliographic_citation",
                    "why": "Visual-only sourcing.",
                }
            ],
            "prepared": source if prepared is None else prepared,
        }
    )


# The citation deletion and the reworded word sit one comma apart, so the differ
# reports them as two atomic changes: the deletion the gold answer asked for,
# and a substantive "remained" -> "buckled" it did not.
ADJACENT_SOURCE = "The result (Smith 1999), long central, remained sound."
ADJACENT_CASE = citation_deletion_case(
    ADJACENT_SOURCE, prepared="The result, long central, remained sound."
)
ADJACENT_CORRECT = "The result, long central, remained sound."
ADJACENT_WITH_EXTRA_EDIT = "The result, long central, buckled sound."

# One deletion swallows the citation *and* the protected prose beside it, so the
# single change crosses the expected span's boundary instead of sitting inside
# it.
BOUNDARY_SOURCE = "The result (Smith 1999) mattered greatly here."
BOUNDARY_CASE = citation_deletion_case(
    BOUNDARY_SOURCE, prepared="The result mattered greatly here.", cid="fixture-boundary"
)
BOUNDARY_OVERREACH = "The result here."


def prepare(case, edits):
    """Run edits through the production applier, as the benchmark does."""

    prepared, applied, warnings = apply_edits(case.source, list(edits))
    return prepared.strip(), applied, warnings


def judge(case, edits):
    prepared, applied, warnings = prepare(case, edits)
    return score_case(
        case, prepared, proposed=list(edits), applied=applied, warnings=warnings
    )


class ScriptedProvider:
    """A provider whose answer to each passage is decided by the test.

    Models the benchmark's real provider closely enough to exercise the run
    metadata: its declared parameters carry no sampling option and flag
    ``native_sampling``, matching a provider the runner built with
    ``temperature=None``, and it records the seed it was told to generate under
    so the seed sequence can be checked at the provider, not just in the report.
    """

    def __init__(self, model, script, registry):
        self._metadata = ProviderMetadata(
            name="fake",
            model=model,
            prompt_version=DEFAULT_PROMPT_VERSION,
            parameters={
                "seed": 42,
                "num_ctx": 8192,
                "num_predict": 4096,
                "think": False,
                "structured_output": True,
                "native_sampling": True,
            },
        )
        self.model = model
        self.script = script
        self.calls = 0
        self.closed = False
        self.availability_checks = 0
        self.seeds_seen = []
        registry.append(self)

    @property
    def metadata(self):
        return self._metadata

    def check_available(self):
        self.availability_checks += 1

    def prepare(self, request):
        self.calls += 1
        # The runner sets `seed` before each repetition; capture what generation
        # would have run under at the moment it is asked.
        self.seeds_seen.append(getattr(self, "seed", None))
        return PreparationResult(
            edits=list(self.script(request.source_text, self.calls)),
            provider_metadata=self.metadata,
        )

    def close(self):
        self.closed = True


def build_scripts():
    """Fresh, independently stateful scripts for one benchmark run.

    Built per run rather than shared at module scope because the flaky script
    must carry state across repetitions, and leaking that state between tests
    would make them order-dependent.
    """

    def oracle(source, _call):
        return [GOLD_EDIT] if source == CITATION_SOURCE else []

    def lazy(_source, _call):
        return []

    def vandal(source, _call):
        return [GOLD_EDIT, TRAP_EDIT] if source == CITATION_SOURCE else []

    # The runner creates a fresh provider per repetition, so a per-provider
    # counter cannot tell repetitions apart; this one persists across them and
    # answers the citation case differently each time it is asked.
    citation_seen = {"count": 0}

    def flaky(source, _call):
        if source != CITATION_SOURCE:
            return []
        citation_seen["count"] += 1
        return [GOLD_EDIT] if citation_seen["count"] % 2 == 1 else []

    return {
        "model-oracle": oracle,
        "model-lazy": lazy,
        "model-vandal": vandal,
        "model-flaky": flaky,
    }


class CorpusTests(unittest.TestCase):
    def test_shipped_corpus_is_valid_ground_truth(self):
        cases = load_corpus()
        tiers = Counter(case.tier for case in cases)
        categories = Counter(
            category for case in cases for category in case.categories
        )

        # load_corpus lints every case, so reaching here already proves each
        # gold answer is reproduced by the production applier.
        self.assertEqual(len(cases), 48)
        self.assertEqual(set(tiers), set(TIERS))
        self.assertEqual(set(categories), set(CATEGORIES))
        self.assertEqual(len({case.id for case in cases}), len(cases))
        self.assertEqual(len({case.source for case in cases}), len(cases))

    def test_every_trap_case_carries_real_work_as_well_as_bait(self):
        # Without this, a model that never edits anything would score full
        # marks on the tier built to catch over-editing.
        for case in load_corpus(tiers=["trap"]):
            with self.subTest(case=case.id):
                self.assertTrue(case.expect)
                self.assertTrue(case.traps)

    def test_quick_subset_stays_balanced_across_tiers(self):
        cases = load_corpus(limit_per_tier=3)
        self.assertEqual(
            Counter(case.tier for case in cases),
            Counter({tier: 3 for tier in TIERS}),
        )

    def test_lint_rejects_a_gold_answer_the_applier_does_not_reproduce(self):
        case = case_from_dict(
            citation_payload(prepared="The finding remained central.")
        )
        issues = lint_case(case)
        self.assertTrue(
            any("does not reproduce the prepared text" in issue for issue in issues),
            issues,
        )

    def test_lint_rejects_an_ambiguous_anchor(self):
        with self.assertRaises(CorpusError) as caught:
            case_from_dict(
                citation_payload(
                    source="A finding (Smith 1999) and another (Smith 1999) here.",
                    expect=[
                        {
                            "anchor": "(Smith 1999)",
                            "accept": [""],
                            "category": "bibliographic_citation",
                        }
                    ],
                )
            )
        self.assertIn("ambiguous", str(caught.exception))

    def test_lint_rejects_source_that_normalization_would_change(self):
        case = case_from_dict(
            citation_payload(source=CITATION_SOURCE.replace("The finding", "The  finding"))
        )
        self.assertTrue(
            any("normalized form" in issue for issue in lint_case(case)),
            lint_case(case),
        )

    def test_lint_rejects_a_trap_that_overlaps_an_expected_edit(self):
        case = case_from_dict(
            citation_payload(
                traps=[{"span": "(Smith 1999)", "label": "contradictory"}]
            )
        )
        self.assertTrue(
            any("overlaps expected edit" in issue for issue in lint_case(case)),
            lint_case(case),
        )

    def test_lint_rejects_a_trap_tier_case_with_nothing_to_do(self):
        case = case_from_dict(
            citation_payload(expect=[], prepared=CITATION_SOURCE, categories=["no_edit"])
        )
        self.assertTrue(
            any("at least one expected edit" in issue for issue in lint_case(case)),
            lint_case(case),
        )


class ChangeRegionTests(unittest.TestCase):
    def test_a_single_deletion_is_reported_as_one_region(self):
        prepared, _applied, _warnings = prepare(CITATION_CASE, [GOLD_EDIT])
        regions = change_regions(CITATION_CASE.source, prepared)

        self.assertEqual(len(regions), 1)
        self.assertIn("Smith 1999", regions[0].source_text)
        self.assertEqual(regions[0].output_text.strip(), "")

    def test_identical_text_has_no_regions(self):
        self.assertEqual(change_regions(NOOP_SOURCE, NOOP_SOURCE), [])


class ScoringTests(unittest.TestCase):
    def test_the_gold_answer_scores_perfectly(self):
        score = judge(CITATION_CASE, [GOLD_EDIT])

        self.assertEqual(score.score, 1.0)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.precision, 1.0)
        self.assertEqual(score.exactness, 1.0)
        self.assertTrue(score.fidelity_pass)
        self.assertTrue(score.passed)
        self.assertTrue(score.output_matches_gold)

    def test_doing_nothing_misses_the_edit_without_harming_the_passage(self):
        score = judge(CITATION_CASE, [])

        self.assertEqual(score.recall, 0.0)
        self.assertTrue(score.fidelity_pass)
        self.assertEqual([item.status for item in score.outcomes], ["missed"])
        self.assertFalse(score.passed)

    def test_doing_nothing_is_a_perfect_answer_to_a_noop_case(self):
        score = judge(NOOP_CASE, [])

        self.assertEqual(score.score, 1.0)
        self.assertTrue(score.passed)

    def test_a_substantive_unrequested_change_fails_the_case_outright(self):
        # The failure the old benchmark scored at 99.7% retention.
        score = judge(CITATION_CASE, [GOLD_EDIT, TRAP_EDIT])

        self.assertEqual(score.recall, 1.0)
        self.assertFalse(score.fidelity_pass)
        self.assertEqual(score.score, 0.0)
        self.assertFalse(score.passed)
        self.assertEqual(score.substantive_false_positives, 1)
        self.assertEqual(
            [item.trap_label for item in score.unexpected],
            ["historical-year-must-stay"],
        )

    def test_a_substantive_change_beside_a_permitted_edit_still_fails(self):
        # The blind spot the merge-then-overlap scorer had: the model makes the
        # expected citation deletion but also rewords the adjacent prose, and
        # the two changes are close enough that merging them into one region —
        # which then overlapped the gold span — excused the reword. Scored on
        # atomic changes with containment, the reword is caught on its own.
        score = score_case(ADJACENT_CASE, ADJACENT_WITH_EXTRA_EDIT)

        self.assertEqual(score.recall, 1.0)
        self.assertFalse(score.fidelity_pass)
        self.assertEqual(score.score, 0.0)
        self.assertFalse(score.passed)
        self.assertEqual(score.substantive_false_positives, 1)
        # The unexpected change names the reworded word, not the citation whose
        # deletion the gold answer permitted.
        substantive = [item for item in score.unexpected if item.severity == "substantive"]
        self.assertEqual(len(substantive), 1)
        self.assertIn("remain", substantive[0].source_text)
        self.assertIn("buckl", substantive[0].output_text)
        self.assertNotIn("Smith", substantive[0].source_text)

    def test_a_normal_citation_deletion_beside_that_word_still_scores_full(self):
        # The control for the case above: with nothing changed but the citation,
        # the incidental space the deletion sweeps up must not be read as a
        # change reaching past the anchor, so the case still scores perfectly.
        score = score_case(ADJACENT_CASE, ADJACENT_CORRECT)

        self.assertEqual(score.score, 1.0)
        self.assertTrue(score.fidelity_pass)
        self.assertTrue(score.passed)
        self.assertEqual(score.unexpected, [])

    def test_a_change_crossing_the_expected_boundary_is_unexpected(self):
        # One deletion takes the citation and the protected words next to it. It
        # overlaps the gold span but is not contained by it, so the overlap buys
        # it nothing: the change is unexpected and the collateral damage to the
        # prose is a fidelity failure.
        score = score_case(BOUNDARY_CASE, BOUNDARY_OVERREACH)

        self.assertFalse(score.fidelity_pass)
        self.assertEqual(score.score, 0.0)
        self.assertFalse(score.passed)
        self.assertEqual(score.substantive_false_positives, 1)
        crossing = score.unexpected[0]
        self.assertEqual(crossing.severity, "substantive")
        # The change spans past the citation into the prose it was not asked to
        # touch.
        self.assertIn("(Smith 1999)", crossing.source_text)
        self.assertIn("mattered", crossing.source_text)

    def test_a_word_preserving_change_costs_precision_but_not_fidelity(self):
        score = judge(CITATION_CASE, [GOLD_EDIT, COSMETIC_EDIT])

        self.assertTrue(score.fidelity_pass)
        self.assertEqual(score.recall, 1.0)
        self.assertEqual(score.precision, 0.5)
        self.assertEqual([item.severity for item in score.unexpected], ["cosmetic"])
        self.assertGreater(score.score, 0.0)
        self.assertFalse(score.passed)

    def test_touching_a_noop_case_is_punished(self):
        edit = PreparationEdit(
            category="visual_notation",
            original="“You never asked,”",
            replacement="You never asked,",
            reason="Quotes are visual.",
            sentence=1,
        )
        score = judge(NOOP_CASE, [edit])

        self.assertFalse(score.passed)
        self.assertEqual(score.precision, 0.0)
        # Removing both quote marks disturbs the trapped span in two places;
        # every disturbance is attributed to the same trap.
        self.assertTrue(score.unexpected)
        self.assertEqual(
            {item.trap_label for item in score.unexpected},
            {"dialogue-quotes-must-stay"},
        )

    def test_a_misquoted_anchor_is_a_contract_failure_not_a_taste_failure(self):
        score = judge(CITATION_CASE, [UNANCHORABLE_EDIT])

        self.assertEqual(score.protocol.proposed, 1)
        self.assertEqual(score.protocol.applied, 0)
        self.assertEqual(score.protocol.unanchored, 1)
        self.assertEqual(score.recall, 0.0)
        self.assertTrue(score.fidelity_pass)

    def test_a_rewrite_is_counted_as_an_oversized_edit(self):
        score = judge(CITATION_CASE, [OVERSIZED_EDIT])

        self.assertEqual(score.protocol.oversized, 1)
        self.assertEqual(score.protocol.applied, 0)
        self.assertTrue(score.fidelity_pass)

    def test_an_accepted_alternative_wording_scores_as_exact(self):
        payload = citation_payload(
            id="fixture-variants",
            tier="core",
            expect=[
                {
                    "anchor": "(Smith 1999)",
                    "accept": ["", "as Smith showed"],
                    "category": "bibliographic_citation",
                }
            ],
        )
        case = case_from_dict(payload)
        self.assertEqual(lint_case(case), [])

        alternative = PreparationEdit(
            category="bibliographic_citation",
            original="(Smith 1999)",
            replacement="as Smith showed",
            sentence=1,
        )
        score = judge(case, [alternative])

        self.assertEqual(score.exactness, 1.0)
        self.assertTrue(score.passed)
        self.assertFalse(score.output_matches_gold)

    def test_a_provider_error_scores_zero_without_crashing(self):
        score = score_case(
            CITATION_CASE, CITATION_CASE.source, error="TimeoutError: too slow"
        )

        self.assertEqual(score.score, 0.0)
        self.assertFalse(score.fidelity_pass)
        self.assertEqual(score.error, "TimeoutError: too slow")


class BenchmarkRunnerTests(unittest.TestCase):
    def options(self, output_dir, models, repetitions=1):
        return BenchmarkOptions(
            output_dir=output_dir,
            provider_name="fake",
            models=models,
            base_url="http://127.0.0.1:11434",
            timeout_seconds=30,
            repetitions=repetitions,
        )

    def run_benchmark(self, models, repetitions=1, cases=None):
        providers = []
        scripts = build_scripts()

        def factory(_name, *, model, **_configuration):
            return ScriptedProvider(model, scripts[model], providers)

        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        report = benchmark_preparation(
            self.options(Path(temporary.name), models, repetitions),
            provider_factory=factory,
            cases=cases if cases is not None else [CITATION_CASE, NOOP_CASE],
        )
        return report, providers

    def test_a_model_that_changes_the_authors_words_ranks_last(self):
        report, providers = self.run_benchmark(
            ("model-oracle", "model-lazy", "model-vandal")
        )
        ranked = [item.model for item in report.ranked]
        by_model = {item.model: item for item in report.models_reports}

        self.assertEqual(ranked[0], "model-oracle")
        self.assertEqual(ranked[-1], "model-vandal")
        self.assertEqual(by_model["model-oracle"].overall.score, 1.0)
        self.assertEqual(by_model["model-vandal"].overall.fidelity_failures, 1)
        self.assertEqual(by_model["model-lazy"].overall.fidelity_failures, 0)
        # The lazy model outranks the vandal despite doing no useful work.
        self.assertLess(ranked.index("model-lazy"), ranked.index("model-vandal"))
        self.assertTrue(all(provider.closed for provider in providers))

    def test_tier_and_category_breakdowns_separate_the_failure(self):
        report, _providers = self.run_benchmark(("model-vandal",))
        item = report.models_reports[0]
        by_tier = {breakdown.label: breakdown for breakdown in item.by_tier}

        self.assertEqual(by_tier["trap"].score, 0.0)
        self.assertEqual(by_tier["noop"].score, 1.0)
        self.assertEqual([label for label, _count in item.trap_failures],
                         ["historical-year-must-stay"])

    def test_determinism_notices_a_model_that_changes_its_mind(self):
        report, _providers = self.run_benchmark(
            ("model-flaky", "model-oracle"), repetitions=2
        )
        by_model = {item.model: item for item in report.models_reports}

        self.assertEqual(by_model["model-oracle"].determinism, 1.0)
        self.assertLess(by_model["model-flaky"].determinism, 1.0)

    def test_repetitions_run_under_a_sequential_seed_shared_across_models(self):
        report, providers = self.run_benchmark(
            ("model-oracle", "model-lazy"), repetitions=3
        )

        # Two cases per repetition, so each repetition's seed appears twice, and
        # the first repetition runs under the base seed rather than base + 1.
        for provider in providers:
            self.assertEqual(provider.seeds_seen, [42, 42, 43, 43, 44, 44])
        # Every model saw the identical sequence: the same seed for the same
        # repetition, deterministic across models.
        self.assertEqual(providers[0].seeds_seen, providers[1].seeds_seen)
        self.assertEqual(seed_for_repetition(1), BASE_SEED)
        self.assertEqual(
            [seed_for_repetition(rep) for rep in (1, 2, 3)], [42, 43, 44]
        )
        # And each run records the seed it generated under.
        for repetition, expected in ((1, 42), (2, 43), (3, 44)):
            self.assertEqual(
                {run.seed for run in report.runs if run.repetition == repetition},
                {expected},
            )

    def test_the_report_records_native_sampling_and_the_seed_sequence(self):
        report, _providers = self.run_benchmark(
            ("model-oracle", "model-vandal"), repetitions=2
        )
        payload = json.loads(report.json_path.read_text(encoding="utf-8"))
        sampling = payload["configuration"]["sampling"]

        # The metadata states plainly that model-native sampling was used, and
        # names both the options it still supplied and the ones it omitted.
        self.assertTrue(sampling["native_sampling"])
        self.assertEqual(sampling["seeds"], [42, 43])
        self.assertEqual(sampling["supplied"], [])
        self.assertIn("temperature", sampling["omitted"])
        self.assertEqual(len(sampling["omitted"]), 8)
        self.assertEqual(payload["configuration"]["prompt_version"], DEFAULT_PROMPT_VERSION)
        # The seed is recorded on every run for reproducing a single attempt.
        self.assertEqual(
            {run["seed"] for run in payload["runs"] if run["repetition"] == 1}, {42}
        )
        # And the explicit provider budgets are recorded per model.
        options = payload["models"][0]["provider_options"]
        self.assertIn("num_ctx", options)
        self.assertIn("num_predict", options)
        self.assertNotIn("temperature", options)

        markdown = report.markdown_path.read_text(encoding="utf-8")
        self.assertIn("model-native defaults", markdown)
        self.assertIn("Seeds (one per repetition): 42, 43", markdown)

    def test_a_provider_that_cannot_start_does_not_hide_the_others(self):
        providers = []
        scripts = build_scripts()

        def factory(_name, *, model, **_configuration):
            if model == "model-broken":
                raise RuntimeError("fixture provider failure")
            return ScriptedProvider(model, scripts[model], providers)

        with TemporaryDirectory() as temporary:
            report = benchmark_preparation(
                self.options(Path(temporary), ("model-broken", "model-oracle")),
                provider_factory=factory,
                cases=[CITATION_CASE, NOOP_CASE],
            )
        by_model = {item.model: item for item in report.models_reports}

        self.assertEqual(by_model["model-broken"].errored_cases, 2)
        self.assertEqual(by_model["model-oracle"].errored_cases, 0)
        self.assertEqual(by_model["model-oracle"].overall.score, 1.0)
        self.assertIn(
            "fixture provider failure",
            by_model["model-broken"].runs[0].score.error,
        )

    def test_it_writes_a_json_artifact_and_a_readable_report(self):
        report, _providers = self.run_benchmark(
            ("model-oracle", "model-vandal")
        )
        payload = json.loads(report.json_path.read_text(encoding="utf-8"))
        markdown = report.markdown_path.read_text(encoding="utf-8")

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["configuration"]["corpus_size"], 2)
        self.assertFalse(payload["configuration"]["cache_reuse"])
        # The full texts are reproducible from the corpus; storing them would
        # triple the artifact for nothing.
        self.assertNotIn("prepared_text", payload["runs"][0])
        self.assertNotIn("gold_text", payload["runs"][0])
        self.assertIn("proposed_edits", payload["runs"][0])
        self.assertIn("## Leaderboard", markdown)
        self.assertIn("## Failure appendix", markdown)
        self.assertIn("historical-year-must-stay", markdown)
        self.assertIn("Traps sprung", markdown)

    def test_every_model_is_asked_exactly_the_same_questions(self):
        _report, providers = self.run_benchmark(
            ("model-oracle", "model-lazy"), repetitions=2
        )
        # One provider per variant, loaded once and kept for both repetitions,
        # so each model is asked all four of its case runs (two cases, twice)
        # without being reloaded between them.
        self.assertEqual([provider.calls for provider in providers], [4, 4])
        self.assertEqual(
            [provider.availability_checks for provider in providers], [1, 1]
        )

    def test_each_variant_is_loaded_once_for_all_repetitions(self):
        # A model is taken to exhaustion before the next loads, so a two-model,
        # two-repetition run creates two providers, not four: no model is
        # unloaded and reloaded between its repetitions.
        _report, providers = self.run_benchmark(
            ("model-oracle", "model-lazy"), repetitions=2
        )
        self.assertEqual(len(providers), 2)
        self.assertEqual([provider.model for provider in providers],
                         ["model-oracle", "model-lazy"])

    def test_thinking_scores_each_mode_as_its_own_entry(self):
        seen: list[tuple[str, bool]] = []
        scripts = build_scripts()
        providers: list[ScriptedProvider] = []

        def factory(_name, *, model, think=False, **_configuration):
            seen.append((model, think))
            return ScriptedProvider(model, scripts[model], providers)

        with TemporaryDirectory() as temporary:
            options = BenchmarkOptions(
                output_dir=Path(temporary),
                provider_name="fake",
                models=("model-oracle",),
                base_url="http://127.0.0.1:11434",
                timeout_seconds=30,
                think_modes=(False, True),
            )
            report = benchmark_preparation(
                options,
                provider_factory=factory,
                cases=[CITATION_CASE, NOOP_CASE],
            )

        self.assertEqual(
            {item.model for item in report.models_reports},
            {"model-oracle", "model-oracle +think"},
        )
        self.assertIn(("model-oracle", True), seen)
        self.assertIn(("model-oracle", False), seen)

    def test_it_draws_png_plots_alongside_the_report(self):
        report, _providers = self.run_benchmark(("model-oracle", "model-vandal"))
        plots_dir = report.json_path.parent / "plots"

        self.assertEqual(
            {path.name for path in report.plot_paths},
            {"scores.png", "by-tier.png", "speed.png"},
        )
        data = (plots_dir / "scores.png").read_bytes()
        # PNG magic number, and large enough to be a real render rather than a
        # blank canvas.
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertGreater(len(data), 2000)


class BenchmarkRunConvenienceTests(unittest.TestCase):
    def factory(self, providers):
        scripts = build_scripts()

        def make(_name, *, model, **_configuration):
            return ScriptedProvider(model, scripts[model], providers)

        return make

    def test_run_drives_the_same_runner_from_plain_values(self):
        providers = []
        with TemporaryDirectory() as temporary:
            report = run(
                models=("model-oracle", "model-vandal"),
                provider="fake",
                output_dir=Path(temporary),
                provider_factory=self.factory(providers),
                cases=[CITATION_CASE, NOOP_CASE],
                progress=None,
                show_summary=False,
            )
            self.assertTrue(report.json_path.exists())
            self.assertTrue(report.markdown_path.exists())
        self.assertEqual([item.model for item in report.ranked][0], "model-oracle")

    def test_default_output_dir_is_timestamped_under_benchmarks(self):
        path = default_output_dir(Path("output"))
        self.assertEqual(path.parent, Path("output") / "benchmarks")
        self.assertTrue(path.name.endswith("Z"))

    def test_print_summary_lists_artifacts_and_ranking(self):
        providers = []
        with TemporaryDirectory() as temporary:
            report = run(
                models=("model-oracle",),
                provider="fake",
                output_dir=Path(temporary),
                provider_factory=self.factory(providers),
                cases=[CITATION_CASE, NOOP_CASE],
                progress=None,
                show_summary=False,
            )
            buffer = io.StringIO()
            print_summary(report, file=buffer)
        text = buffer.getvalue()
        self.assertIn("Comparison report:", text)
        self.assertIn("model-oracle", text)
        self.assertIn("fidelity failures", text)


class BenchmarkVariantTests(unittest.TestCase):
    def options(self, **overrides):
        base = dict(
            output_dir=Path("unused"),
            provider_name="fake",
            models=("model-a", "model-b"),
            base_url="http://127.0.0.1:11434",
            timeout_seconds=30,
        )
        base.update(overrides)
        return BenchmarkOptions(**base)

    def test_both_modes_make_two_ranked_entries_per_model(self):
        variants = self.options(think_modes=(False, True)).variants
        self.assertEqual(
            [variant.label for variant in variants],
            ["model-a", "model-a +think", "model-b", "model-b +think"],
        )
        self.assertEqual(
            [variant.think for variant in variants], [False, True, False, True]
        )

    def test_a_model_without_thinking_keeps_only_its_direct_run(self):
        variants = self.options(
            think_modes=(False, True), no_think_models=("model-b",)
        ).variants
        self.assertEqual(
            [variant.label for variant in variants],
            ["model-a", "model-a +think", "model-b"],
        )

    def test_repeated_think_modes_collapse_to_one_each(self):
        self.assertEqual(
            self.options(think_modes=(False, False, True)).think_modes,
            (False, True),
        )


class BenchmarkCommandLineTests(unittest.TestCase):
    def test_it_accepts_default_and_filtered_runs(self):
        defaults = cli.parse_args(["benchmark"])
        filtered = cli.parse_args(
            [
                "benchmark",
                "--models",
                "local:model-a",
                "hosted:model-b",
                "--tier",
                "trap",
                "--repetitions",
                "3",
                "--quick",
            ]
        )

        self.assertEqual(defaults.models, ["gemma4:12b", "gemma4:26b", "gemma4:31b"])
        self.assertEqual(defaults.tiers, [])
        self.assertFalse(defaults.quick)
        self.assertEqual(filtered.models, ["local:model-a", "hosted:model-b"])
        self.assertEqual(filtered.tiers, ["trap"])
        self.assertEqual(filtered.repetitions, 3)
        self.assertTrue(filtered.quick)

    def test_it_builds_options_the_runner_accepts(self):
        options = cli._benchmark_options(
            cli.parse_args(["benchmark", "--category", "no_edit"])
        )

        self.assertEqual(options.categories, ("no_edit",))
        self.assertEqual(options.provider_name, "ollama")
        self.assertTrue(options.models)
        # The default run leaves thinking off, matching production.
        self.assertEqual(options.think_modes, (False,))

    def test_cli_and_driver_route_through_the_one_runner(self):
        # The file-first run() (which run_benchmark.py calls) is a thin wrapper
        # over the runner, so a driver run reaches the same code as the CLI.
        report = MagicMock()
        report.models_reports = []
        with patch(
            "audiobook.benchmarking.run.benchmark_preparation", return_value=report
        ) as runner:
            run(
                models=("model-a",),
                provider="fake",
                output_dir=Path("unused"),
                provider_factory=lambda *args, **kwargs: None,
                cases=[],
                progress=None,
                show_summary=False,
            )
        runner.assert_called_once()

        # The CLI benchmark command calls that same runner.
        cli_report = MagicMock()
        cli_report.models_reports = [MagicMock(errored_cases=0)]
        cli_report.runs = [MagicMock()]
        with patch(
            "audiobook.cli.benchmark_preparation", return_value=cli_report
        ) as cli_runner, patch("audiobook.cli.print_summary"):
            cli.main(["benchmark", "--provider", "fake", "--models", "model-a"])
        cli_runner.assert_called_once()

    def test_think_both_expands_to_two_modes_without_probing_a_server(self):
        # A non-Ollama provider skips the capability probe, so this builds
        # options without any network call.
        options = cli._benchmark_options(
            cli.parse_args(
                ["benchmark", "--provider", "fake", "--think", "both",
                 "--models", "model-a", "model-b"]
            )
        )

        self.assertEqual(options.think_modes, (False, True))
        self.assertEqual(len(options.variants), 4)


if __name__ == "__main__":
    unittest.main()
