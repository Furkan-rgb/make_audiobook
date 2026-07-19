import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from audiobook.ui.library import list_prepared_scripts


def _artifact(padding: int) -> str:
    """A prepared book whose metadata pushes "chapters" past `padding` bytes."""

    return json.dumps(
        {
            "schema_version": 1,
            "title": "Book",
            "provider_metadata": {"prompt": "x" * padding},
            "chapters": [{"index": 0, "title": "One", "units": []}],
        },
        indent=2,
    )


class ListPreparedScriptsTests(unittest.TestCase):
    def test_finds_artifact_with_large_leading_metadata(self):
        # The prompt in provider metadata is easily bigger than any fixed-size
        # head read, and "chapters" is serialized after it.
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            script = output_dir / "prepared_book.json"
            script.write_text(_artifact(20_000), encoding="utf-8")

            self.assertEqual(list_prepared_scripts(output_dir), [script])

    def test_ignores_unrelated_json(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "chunk_manifest.json").write_text(
                json.dumps({"chapters": [], "schema_version": 1}), encoding="utf-8"
            )
            (output_dir / "notes.json").write_text('{"hello": 1}', encoding="utf-8")

            self.assertEqual(list_prepared_scripts(output_dir), [])


if __name__ == "__main__":
    unittest.main()
