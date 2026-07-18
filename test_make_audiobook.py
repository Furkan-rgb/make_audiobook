import unittest

import numpy as np

import make_audiobook as audiobook


class SemanticChunkingTests(unittest.TestCase):
    def test_page_top_sentence_continuation_is_not_a_new_paragraph(self):
        joined = audiobook._join_markdown_pages(
            ["The relationship may be due to many", "# influences, including these."],
        )
        self.assertEqual(
            joined,
            "The relationship may be due to many influences, including these.",
        )

    def test_spaced_page_number_removal_preserves_following_heading(self):
        markdown = "# 1 3\n\n# Chapter Title\n\n# SECTION\n\nThe section begins."
        cleaned = audiobook.RE_STANDALONE_PAGE_NUMBER.sub("", markdown)
        sections = audiobook.split_into_sections(cleaned)

        self.assertEqual(len(sections), 1)
        self.assertEqual(
            sections[0].paragraphs,
            ["Chapter Title\n\nSECTION\n\nThe section begins."],
        )

    def test_short_paragraphs_are_combined_and_preserved(self):
        first = "First paragraph " + "quiet words " * 29
        second = "Second paragraph " + "soft words " * 20
        third = "Third paragraph " + "later words " * 64
        chunks = audiobook.make_narration_chunks(
            f"{first}\n\n{second}\n\n{third}",
            target_chars=700,
            max_chars=1000,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, f"{first.strip()}\n\n{second.strip()}")
        self.assertEqual(chunks[1].text, third.strip())

    def test_long_paragraph_splits_only_between_sentences(self):
        sentences = [f"Sentence {index} carries the same deliberate rhythm." for index in range(20)]
        paragraph = " ".join(sentences)
        parts = audiobook.split_long_paragraph(paragraph, max_chars=210)

        self.assertGreater(len(parts), 1)
        self.assertEqual(" ".join(parts), paragraph)
        for part in parts:
            self.assertTrue(part.endswith("."))

    def test_single_oversized_sentence_is_not_cut(self):
        sentence = "A" * 1500 + "."
        self.assertEqual(audiobook.split_long_paragraph(sentence, 1300), [sentence])

    def test_dialogue_exchange_stays_together_when_it_fits(self):
        narration = "Narration " + "continues calmly " * 55
        first_reply = '“Are you ready?” Daniel asked.'
        second_reply = '“I think so,” Sarah replied.'
        chunks = audiobook.make_narration_chunks(
            f"{narration}\n\n{first_reply}\n\n{second_reply}",
            target_chars=700,
            max_chars=1200,
        )

        dialogue_chunks = [
            chunk for chunk in chunks if first_reply in chunk.text or second_reply in chunk.text
        ]
        self.assertEqual(len(dialogue_chunks), 1)
        self.assertIn(first_reply, dialogue_chunks[0].text)
        self.assertIn(second_reply, dialogue_chunks[0].text)

    def test_scene_break_forces_a_chunk_boundary(self):
        first = "First scene " + "quietly unfolds " * 20
        second = "Second scene " + "slowly begins " * 20
        chunks = audiobook.make_narration_chunks(
            f"{first}\n\n***\n\n{second}",
            target_chars=850,
            max_chars=1300,
        )

        self.assertEqual([chunk.text for chunk in chunks], [first.strip(), second.strip()])
        self.assertEqual(chunks[0].boundary_after, "scene")

    def test_neighbor_context_is_metadata_not_spoken_text(self):
        first = "First section " + "one " * 220
        second = "Second section " + "two " * 220
        chunks = audiobook.make_narration_chunks(
            f"{first}\n\n***\n\n{second}",
            context_chars=80,
        )

        self.assertEqual(len(chunks), 2)
        self.assertNotIn(chunks[0].following_context, chunks[0].text)
        self.assertEqual(
            chunks[0].following_context,
            audiobook._context_head(second.strip(), 80),
        )
        self.assertEqual(
            chunks[1].previous_context,
            audiobook._context_tail(first.strip(), 80),
        )


class AudioAssemblyTests(unittest.TestCase):
    def test_continuation_uses_only_a_short_crossfade(self):
        sample_rate = 1000
        chunks = [
            audiobook.NarrationChunk("first", "continuation"),
            audiobook.NarrationChunk("second", "paragraph"),
        ]
        segments = [np.ones(1000, dtype=np.float32), np.ones(1000, dtype=np.float32)]
        joined = audiobook.assemble_chunk_audio(chunks, segments, sample_rate)

        self.assertEqual(len(joined), 2000 - audiobook.CHUNK_CROSSFADE_MS)

    def test_paragraph_and_section_gaps_are_boundary_sensitive(self):
        sample_rate = 1000
        segment = np.ones(1000, dtype=np.float32)
        paragraph_chunks = [
            audiobook.NarrationChunk("first", "paragraph"),
            audiobook.NarrationChunk("second", "section"),
        ]
        section_chunks = [
            audiobook.NarrationChunk("first", "section"),
            audiobook.NarrationChunk("second", "section"),
        ]

        paragraph_audio = audiobook.assemble_chunk_audio(
            paragraph_chunks, [segment, segment], sample_rate
        )
        section_audio = audiobook.assemble_chunk_audio(
            section_chunks, [segment, segment], sample_rate
        )

        self.assertEqual(len(paragraph_audio), 2000 + audiobook.PARAGRAPH_SILENCE_MS)
        self.assertEqual(len(section_audio), 2000 + audiobook.SECTION_SILENCE_MS)


if __name__ == "__main__":
    unittest.main()
