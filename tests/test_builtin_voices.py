import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from audiobook.synthesis.providers import SynthesisDescriptor, VoiceInfo
from audiobook.synthesis.providers.qwen import (
    QwenSynthesisProvider,
    _QwenBuiltinVoice,
)
from audiobook.ui.library import delete_voice, rename_voice, save_transcript


class CapabilityDeclarationTests(unittest.TestCase):
    """A backend states which of the three verbs it serves; nothing is implied."""

    def test_qwen_declares_all_three(self):
        descriptor = QwenSynthesisProvider.describe()
        self.assertTrue(descriptor.supports_design)
        self.assertTrue(descriptor.supports_clone)
        self.assertTrue(descriptor.supports_narrate)

    def test_a_partial_backend_is_valid(self):
        # e.g. a future design-only provider: allowed, and the frontend
        # simply offers nothing but the design tab.
        descriptor = SynthesisDescriptor(name="d", label="D", supports_design=True)
        self.assertFalse(descriptor.supports_narrate)

    def test_a_backend_declaring_nothing_fails_at_registration_time(self):
        with self.assertRaises(ValueError):
            SynthesisDescriptor(name="void", label="Void")


def _write_folder_voice(voices_dir: Path, name: str, instruct: str | None = None) -> None:
    voice_dir = voices_dir / name
    voice_dir.mkdir(parents=True)
    (voice_dir / "reference.wav").write_bytes(b"RIFF")
    (voice_dir / "reference.json").write_text(
        json.dumps({"slug": name, "instruct": instruct, "ref_text": "hello"}),
        encoding="utf-8",
    )


class QwenVoiceCatalogTests(unittest.TestCase):
    """The adapter's voices(): one catalog, whatever each voice's origin."""

    def test_file_backed_voices_come_first_then_backend_speakers(self):
        with (
            TemporaryDirectory() as directory,
            patch(
                "audiobook.synthesis.providers.qwen._builtin_speaker_roster",
                return_value=("Aiden", "Serena"),
            ),
        ):
            voices_dir = Path(directory)
            _write_folder_voice(voices_dir, "warm_male", instruct="a narrator")

            voices = QwenSynthesisProvider().voices(voices_dir=voices_dir)

            self.assertEqual([v.spec for v in voices], ["warm_male", "Aiden", "Serena"])
            self.assertEqual(voices[0].kind, "designed")
            self.assertEqual(voices[1].kind, "built-in")
            self.assertTrue(voices[1].builtin)
            self.assertIsNone(voices[1].audio_path)

    def test_a_voice_on_disk_shadows_a_backend_speaker_of_the_same_name(self):
        with (
            TemporaryDirectory() as directory,
            patch(
                "audiobook.synthesis.providers.qwen._builtin_speaker_roster",
                return_value=("Aiden",),
            ),
        ):
            voices_dir = Path(directory)
            _write_folder_voice(voices_dir, "aiden")

            voices = QwenSynthesisProvider().voices(voices_dir=voices_dir)

            self.assertEqual(len(voices), 1)
            self.assertTrue(voices[0].file_backed)  # the folder wins the spec

    def test_catalog_lists_backend_speakers_even_without_a_voices_directory(self):
        with (
            TemporaryDirectory() as directory,
            patch(
                "audiobook.synthesis.providers.qwen._builtin_speaker_roster",
                return_value=("Aiden",),
            ),
        ):
            missing = Path(directory) / "does-not-exist"

            voices = QwenSynthesisProvider().voices(voices_dir=missing)

            self.assertEqual([v.spec for v in voices], ["Aiden"])


class QwenLoadVoiceTests(unittest.TestCase):
    """load_voice() resolves a spec without the caller knowing its kind."""

    def test_backend_speaker_resolves_without_touching_disk_or_gpu(self):
        with (
            TemporaryDirectory() as directory,
            patch(
                "audiobook.synthesis.providers.qwen._builtin_speaker_roster",
                return_value=("Aiden",),
            ),
            patch("audiobook.config.VOICES_DIR", Path(directory) / "none"),
        ):
            handle = QwenSynthesisProvider().load_voice("aiden")  # case-insensitive

            self.assertIsInstance(handle, _QwenBuiltinVoice)
            self.assertEqual(handle.name, "Aiden")

    def test_unknown_spec_raises_file_not_found(self):
        with (
            TemporaryDirectory() as directory,
            patch(
                "audiobook.synthesis.providers.qwen._builtin_speaker_roster",
                return_value=(),
            ),
            patch("audiobook.config.VOICES_DIR", Path(directory) / "none"),
        ):
            with self.assertRaises(FileNotFoundError):
                QwenSynthesisProvider().load_voice("no-such-voice")


class LibraryMutationGuardTests(unittest.TestCase):
    def test_voices_without_files_refuse_mutation(self):
        entry = VoiceInfo(spec="Aiden", label="Aiden  (built-in)", kind="built-in")

        with self.assertRaises(ValueError):
            save_transcript(entry, "text")
        with self.assertRaises(ValueError):
            rename_voice(entry, "other")
        with self.assertRaises(ValueError):
            delete_voice(entry)


class CatalogLookupTests(unittest.TestCase):
    def test_workflow_matches_backend_speakers_case_insensitively(self):
        from audiobook.workflow import _voice_info

        class StubProvider:
            def voices(self):
                return (
                    VoiceInfo(spec="warm_male", label="w", kind="designed", audio_path=Path("x")),
                    VoiceInfo(spec="Aiden", label="a", kind="built-in"),
                )

        provider = StubProvider()
        self.assertEqual(_voice_info(provider, "warm_male").kind, "designed")
        self.assertEqual(_voice_info(provider, "aiden").spec, "Aiden")
        # File-backed specs stay exact: no case-folding surprises on disk.
        self.assertIsNone(_voice_info(provider, "WARM_MALE"))
        self.assertIsNone(_voice_info(provider, "missing"))

    def test_preflight_reports_nothing_for_an_unknown_active_voice(self):
        import audiobook.preflight as preflight

        original = preflight.ACTIVE_VOICE
        preflight.ACTIVE_VOICE = "definitely-not-a-voice-xyz"
        try:
            self.assertIsNone(preflight._active_voice_info())
        finally:
            preflight.ACTIVE_VOICE = original


if __name__ == "__main__":
    unittest.main()
