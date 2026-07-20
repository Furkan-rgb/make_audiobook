import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audiobook import cli
from audiobook.benchmarking import BenchmarkOptions, benchmark_preparation
from audiobook.preparation import (
    DEFAULT_PROMPT_VERSION,
    PreparationEdit,
    PreparationResult,
    ProviderMetadata,
    SourceMetadata,
)


SOURCE = (
    "The finding (Smith 1999, Jones 2001) remained central to the argument.\n\n"
    "A second paragraph preserves every substantive qualification."
)
PREPARED = (
    "The finding remained central to the argument.\n\n"
    "A second paragraph preserves every substantive qualification."
)


class BenchmarkFakeProvider:
    def __init__(self, model, registry, *, fail=False):
        self._metadata = ProviderMetadata(
            name="fake",
            model=model,
            prompt_version=DEFAULT_PROMPT_VERSION,
            parameters={"temperature": 0.0},
        )
        self.model = model
        self.registry = registry
        self.fail = fail
        self.closed = False
        self.calls = 0
        registry.append(self)

    @property
    def metadata(self):
        return self._metadata

    def check_available(self):
        return None

    def prepare(self, request):
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture model failure")
        return PreparationResult(
            edits=(
                [
                    PreparationEdit(
                        category="bibliographic_citation",
                        original="(Smith 1999, Jones 2001)",
                        replacement="",
                        reason="Visual-only citation",
                    )
                ]
                if self.model == "model-small"
                else []
            ),
            provider_metadata=self.metadata,
        )

    def close(self):
        self.closed = True


class PreparationBenchmarkTests(unittest.TestCase):
    def options(self, output_dir, models=("model-small", "model-large"), repetitions=2):
        return BenchmarkOptions(
            source_path=Path("fixture.pdf"),
            output_dir=output_dir,
            provider_name="fake",
            models=models,
            base_url="http://127.0.0.1:11434",
            timeout_seconds=30,
            preview_chapters=1,
            preview_units=1,
            repetitions=repetitions,
        )

    def test_benchmark_writes_isolated_artifacts_metrics_and_comparisons(self):
        providers = []

        def factory(_name, *, model, **_configuration):
            return BenchmarkFakeProvider(model, providers)

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "benchmark"
            report = benchmark_preparation(
                self.options(output_dir),
                provider_factory=factory,
                chapters=[("Preface", SOURCE)],
                source_metadata=SourceMetadata(extra={"fixture": True}),
            )
            payload = json.loads(report.json_path.read_text(encoding="utf-8"))
            markdown = report.markdown_path.read_text(encoding="utf-8")

            self.assertEqual(len(report.runs), 4)
            self.assertTrue(all(run.success for run in report.runs))
            self.assertTrue(all(provider.closed for provider in providers))
            self.assertTrue(all(provider.calls == 1 for provider in providers))
            self.assertEqual(len(report.comparisons), 1)
            self.assertLess(report.comparisons[0].output_similarity, 1.0)
            self.assertNotIn("prepared_text", payload["runs"][0])
            self.assertEqual(payload["configuration"]["cache_reuse"], False)
            self.assertIn("model-small", markdown)
            self.assertIn("Pairwise prepared-text comparison", markdown)
            self.assertTrue(
                all(Path(run.artifact_path).exists() for run in report.runs)
            )
            small = next(
                summary for summary in report.summaries if summary.model == "model-small"
            )
            large = next(
                summary for summary in report.summaries if summary.model == "model-large"
            )
            self.assertEqual(small.mean_citation_reduction, 1.0)
            self.assertGreater(
                small.mean_citation_target_similarity,
                large.mean_citation_target_similarity,
            )
            self.assertEqual(small.consistency, 1.0)

    def test_one_model_failure_does_not_hide_successful_models(self):
        providers = []

        def factory(_name, *, model, **_configuration):
            return BenchmarkFakeProvider(
                model,
                providers,
                fail=model == "model-broken",
            )

        with TemporaryDirectory() as temporary_directory:
            report = benchmark_preparation(
                self.options(
                    Path(temporary_directory),
                    models=("model-broken", "model-small"),
                    repetitions=1,
                ),
                provider_factory=factory,
                chapters=[("Preface", SOURCE)],
                source_metadata=SourceMetadata(extra={"fixture": True}),
            )

        self.assertFalse(report.runs[0].success)
        self.assertIn("fixture model failure", report.runs[0].error)
        self.assertTrue(report.runs[1].success)
        self.assertTrue(all(provider.closed for provider in providers))

    def test_cli_accepts_default_and_future_model_lists(self):
        defaults = cli.parse_args(["benchmark"])
        custom = cli.parse_args(
            [
                "benchmark",
                "--models",
                "local:model-a",
                "hosted:model-b",
                "--repetitions",
                "3",
            ]
        )

        self.assertEqual(
            defaults.models,
            ["gemma4:12b", "gemma4:26b", "gemma4:31b"],
        )
        self.assertEqual(custom.models, ["local:model-a", "hosted:model-b"])
        self.assertEqual(custom.repetitions, 3)


if __name__ == "__main__":
    unittest.main()
