"""Loudness handling: chapter chunk matching and final two-pass loudnorm."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from audiobook.assembly.audio import (
    active_speech_rms,
    match_chunk_loudness,
    merge_chapters,
)

SAMPLE_RATE = 24000


def speech(amplitude, seconds=1.0, frequency=220.0):
    """A steady sine standing in for speech at a known RMS (amplitude / √2)."""

    t = np.arange(round(SAMPLE_RATE * seconds), dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(round(SAMPLE_RATE * seconds), dtype=np.float32)


def db_between(louder_rms, quieter_rms):
    return 20.0 * np.log10(louder_rms / quieter_rms)


class ActiveSpeechRmsTests(unittest.TestCase):
    def test_empty_input_measures_zero(self):
        self.assertEqual(active_speech_rms(np.array([], dtype=np.float32), SAMPLE_RATE), 0.0)

    def test_silent_input_measures_zero(self):
        self.assertEqual(active_speech_rms(silence(1.0), SAMPLE_RATE), 0.0)

    def test_near_silent_noise_measures_zero(self):
        rng = np.random.default_rng(seed=7)
        noise = (1e-4 * rng.standard_normal(SAMPLE_RATE)).astype(np.float32)
        self.assertEqual(active_speech_rms(noise, SAMPLE_RATE), 0.0)

    def test_leading_and_trailing_silence_do_not_dilute_measurement(self):
        voiced = speech(0.2, seconds=1.0)
        padded = np.concatenate([silence(2.0), voiced, silence(2.0)])

        measured = active_speech_rms(padded, SAMPLE_RATE)

        # Whole-waveform RMS of the padded signal would be √(1/5) ≈ 0.45 of
        # the speech level; the active measurement must stay at the speech RMS.
        expected = 0.2 / np.sqrt(2.0)
        self.assertAlmostEqual(measured, expected, delta=expected * 0.05)

    def test_measurement_matches_unpadded_speech(self):
        voiced = speech(0.15, seconds=2.0)
        padded = np.concatenate([silence(3.0), voiced, silence(3.0)])

        unpadded_rms = active_speech_rms(voiced, SAMPLE_RATE)
        padded_rms = active_speech_rms(padded, SAMPLE_RATE)

        self.assertGreater(padded_rms, unpadded_rms * 0.9)
        self.assertLess(padded_rms, unpadded_rms * 1.1)

    def test_output_is_finite(self):
        self.assertTrue(np.isfinite(active_speech_rms(speech(0.3), SAMPLE_RATE)))
        malformed = np.array([np.nan, np.inf, 0.5], dtype=np.float32)
        self.assertEqual(active_speech_rms(malformed, SAMPLE_RATE), 0.0)


class MatchChunkLoudnessTests(unittest.TestCase):
    def test_no_segments_yield_no_results(self):
        segments, diagnostics = match_chunk_loudness([], SAMPLE_RATE)
        self.assertEqual(segments, [])
        self.assertEqual(diagnostics, [])

    def test_empty_segment_remains_empty(self):
        segments, diagnostics = match_chunk_loudness(
            [speech(0.2), np.array([], dtype=np.float32)], SAMPLE_RATE
        )
        self.assertEqual(len(segments[1]), 0)
        self.assertEqual(diagnostics[1].normalization_gain_db, 0.0)

    def test_silent_segment_is_not_amplified(self):
        quiet_silence = silence(1.0)
        segments, diagnostics = match_chunk_loudness(
            [speech(0.2), quiet_silence, speech(0.25)], SAMPLE_RATE
        )

        self.assertTrue(np.array_equal(segments[1], quiet_silence))
        self.assertEqual(diagnostics[1].normalization_gain_db, 0.0)
        self.assertEqual(diagnostics[1].to_manifest()["normalization_gain_db"], 0.0)

    def test_quiet_and_loud_chunks_move_closer_together(self):
        segments, _ = match_chunk_loudness(
            [speech(0.1), speech(0.14), speech(0.2)], SAMPLE_RATE
        )

        before = db_between(
            active_speech_rms(speech(0.2), SAMPLE_RATE),
            active_speech_rms(speech(0.1), SAMPLE_RATE),
        )
        after = db_between(
            active_speech_rms(segments[2], SAMPLE_RATE),
            active_speech_rms(segments[0], SAMPLE_RATE),
        )
        self.assertLess(after, before)
        self.assertLess(after, 1.0)

    def test_adjustment_never_exceeds_three_db(self):
        very_quiet = speech(0.02)
        segments, diagnostics = match_chunk_loudness(
            [very_quiet, speech(0.2), speech(0.2)], SAMPLE_RATE
        )

        for record in diagnostics:
            self.assertLessEqual(abs(record.normalization_gain_db), 3.0)
        boosted = active_speech_rms(segments[0], SAMPLE_RATE)
        original = active_speech_rms(very_quiet, SAMPLE_RATE)
        self.assertAlmostEqual(db_between(boosted, original), 3.0, places=3)

    def test_sample_peaks_stay_below_the_ceiling(self):
        spiky = speech(0.1)
        spiky[100] = 0.8  # A transient near full scale in a quiet chunk.
        segments, diagnostics = match_chunk_loudness(
            [spiky, speech(0.2), speech(0.2)], SAMPLE_RATE
        )

        ceiling = 10.0 ** (-1.5 / 20.0)
        self.assertLessEqual(float(np.max(np.abs(segments[0]))), ceiling + 1e-6)
        # The boost was reduced for peak safety rather than clipped afterwards.
        self.assertLess(diagnostics[0].normalization_gain_db, 3.0)
        self.assertGreater(diagnostics[0].normalization_gain_db, 0.0)

    def test_output_is_flat_mono_float32(self):
        column = np.asarray(speech(0.2), dtype=np.float64).reshape(-1, 1)
        segments, _ = match_chunk_loudness([column, speech(0.15)], SAMPLE_RATE)

        for segment in segments:
            self.assertEqual(segment.ndim, 1)
            self.assertEqual(segment.dtype, np.float32)

    def test_natural_relative_differences_survive(self):
        segments, _ = match_chunk_loudness(
            [speech(0.05), speech(0.3)], SAMPLE_RATE
        )

        after = db_between(
            active_speech_rms(segments[1], SAMPLE_RATE),
            active_speech_rms(segments[0], SAMPLE_RATE),
        )
        # 15.6 dB apart before; ±3 dB of matching must leave a clear gap.
        self.assertGreater(after, 6.0)

    def test_diagnostics_match_the_applied_gain(self):
        segments, diagnostics = match_chunk_loudness(
            [speech(0.1), speech(0.14), speech(0.2)], SAMPLE_RATE
        )

        for segment, record in zip(segments, diagnostics):
            expected_after = record.active_rms_before * 10.0 ** (
                record.normalization_gain_db / 20.0
            )
            self.assertAlmostEqual(record.active_rms_after, expected_after, places=5)
            remeasured = active_speech_rms(segment, SAMPLE_RATE)
            self.assertAlmostEqual(remeasured, record.active_rms_after, places=4)
            manifest = record.to_manifest()
            self.assertEqual(
                set(manifest),
                {"active_rms_before", "active_rms_after", "normalization_gain_db"},
            )
            for value in manifest.values():
                self.assertIsInstance(value, float)


ANALYSIS_STDERR = """\
Input #0, concat, from 'files.txt':
  Duration: N/A, start: 0.000000, bitrate: 768 kb/s
[Parsed_loudnorm_0 @ 0x5555]
{
    "input_i" : "-27.61",
    "input_tp" : "-4.47",
    "input_lra" : "18.06",
    "input_thresh" : "-39.20",
    "output_i" : "-22.03",
    "output_tp" : "-2.00",
    "output_lra" : "6.30",
    "output_thresh" : "-33.51",
    "normalization_type" : "linear",
    "target_offset" : "0.58"
}
"""

CHAPTERS = [("Chapter One", 0, 1000), ("Chapter Two", 1000, 2500)]


def completed(command, returncode=0, stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout="", stderr=stderr)


class MergeChaptersLoudnormTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp.name)
        self.output_path = self.temp_dir / "out" / "audiobook.m4b"
        self.addCleanup(self._temp.cleanup)

    def merge(self, run_mock):
        with mock.patch(
            "audiobook.assembly.audio.subprocess.run", side_effect=run_mock
        ) as patched:
            merge_chapters(
                self.temp_dir, ["part_000.wav", "part_001.wav"], CHAPTERS, self.output_path
            )
        return patched

    def test_two_passes_run_with_measured_values(self):
        responses = iter(
            [
                completed([], stderr=ANALYSIS_STDERR),
                completed([]),
            ]
        )
        patched = self.merge(lambda command, **kwargs: next(responses))

        self.assertEqual(patched.call_count, 2)
        first = patched.call_args_list[0].args[0]
        second = patched.call_args_list[1].args[0]

        # First pass: analysis only over the concatenated chapter list.
        self.assertIn("loudnorm=I=-23.0:TP=-2.0:LRA=7.0:print_format=json", first)
        self.assertEqual(first[-3:], ["-f", "null", "-"])
        self.assertIn("files.txt", first)

        # Second pass: measured values drive a linear normalization.
        second_filter = second[second.index("-af") + 1]
        for expected in (
            "I=-23.0",
            "TP=-2.0",
            "LRA=7.0",
            "measured_I=-27.61",
            "measured_TP=-4.47",
            "measured_LRA=18.06",
            "measured_thresh=-39.20",
            "offset=0.58",
            "linear=true",
        ):
            self.assertIn(expected, second_filter)
        self.assertIn("-ar", second)
        self.assertEqual(second[second.index("-ar") + 1], "48000")
        self.assertEqual(second[second.index("-c:a") + 1], "aac")
        self.assertEqual(second[-1], str(self.output_path.resolve()))

    def test_chapter_metadata_remains_mapped(self):
        responses = iter([completed([], stderr=ANALYSIS_STDERR), completed([])])
        patched = self.merge(lambda command, **kwargs: next(responses))

        second = patched.call_args_list[1].args[0]
        self.assertEqual(second[second.index("-map_metadata") + 1], "1")
        self.assertIn("metadata.txt", second)
        metadata = (self.temp_dir / "metadata.txt").read_text(encoding="utf-8")
        self.assertIn("[CHAPTER]", metadata)
        self.assertIn("title=Chapter One", metadata)
        self.assertIn("END=2500", metadata)

    def test_malformed_analysis_output_raises_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "loudnorm"):
            self.merge(
                lambda command, **kwargs: completed(command, stderr="no json here")
            )

    def test_unparseable_measurements_raise_a_clear_error(self):
        stderr = '{ "output_i" : "-22.03" }'
        with self.assertRaisesRegex(RuntimeError, "loudnorm"):
            self.merge(lambda command, **kwargs: completed(command, stderr=stderr))

    def test_failed_analysis_command_raises_a_clear_error(self):
        with self.assertRaisesRegex(RuntimeError, "FFmpeg failed"):
            self.merge(
                lambda command, **kwargs: completed(
                    command, returncode=1, stderr="boom"
                )
            )

    def test_failed_encode_command_raises_a_clear_error(self):
        responses = iter(
            [
                completed([], stderr=ANALYSIS_STDERR),
                completed([], returncode=1, stderr="encoder exploded"),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "encoder exploded"):
            self.merge(lambda command, **kwargs: next(responses))


if __name__ == "__main__":
    unittest.main()
