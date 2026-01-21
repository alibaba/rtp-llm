from typing import List
from unittest import TestCase, main
from unittest.mock import MagicMock, Mock

from rtp_llm.openai.api_datatype import FinisheReason
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, StreamStatus
from rtp_llm.utils.word_util import get_stop_word_slices


class RemoveStopWordIdsTest(TestCase):
    """Test _remove_stop_word_ids method which truncates token sequences at stop words."""

    def setUp(self):
        # Create a minimal mock renderer with necessary attributes
        self.renderer = Mock(spec=CustomChatRenderer)
        self.renderer.eos_token_id = 2
        self.renderer.stop_words_id_list = [[151643], [151644], [151645]]
        self.renderer.get_all_extra_stop_word_ids_list = Mock(return_value=[])

        # Bind the actual method to our mock
        self.renderer._remove_stop_word_ids = (
            CustomChatRenderer._remove_stop_word_ids.__get__(self.renderer)
        )

    def test_truncate_at_eos(self):
        # EOS token in middle of sequence - should truncate
        output_ids = [100, 101, 2, 103, 104]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_truncate_at_eos_multiple(self):
        # Multiple EOS tokens - should truncate at FIRST
        output_ids = [100, 2, 102, 2, 104]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_eos_at_beginning(self):
        # EOS at beginning - should return empty
        output_ids = [2, 100, 101]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [])

    def test_eos_at_end(self):
        # EOS at end - should truncate
        output_ids = [100, 101, 2]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_no_eos(self):
        # No EOS token - should return unchanged
        output_ids = [100, 101, 102]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101, 102])

    def test_truncate_at_stop_word_sequence(self):
        # Stop word sequence in middle
        output_ids = [100, 101, 151643, 103, 104]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_truncate_at_multi_token_stop_word(self):
        # Multi-token stop word sequence
        self.renderer.stop_words_id_list = [[200, 201, 202]]
        output_ids = [100, 200, 201, 202, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_multiple_stop_words_truncate_at_first(self):
        # Multiple different stop words - should truncate at earliest
        self.renderer.stop_words_id_list = [[151643], [151644]]
        output_ids = [100, 101, 151644, 102, 151643, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_stop_word_at_beginning(self):
        # Stop word at beginning - should return empty
        output_ids = [151643, 100, 101]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [])

    def test_stop_word_at_end(self):
        # Stop word at end - should truncate
        output_ids = [100, 101, 151643]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101])

    def test_no_stop_words(self):
        # No stop words in sequence
        output_ids = [100, 101, 102, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 101, 102, 103])

    def test_eos_before_stop_word(self):
        # Both EOS and stop word, EOS comes first - should truncate at EOS
        output_ids = [100, 2, 151643, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_stop_word_before_eos(self):
        # Both stop word and EOS, stop word comes first - should truncate at stop word
        output_ids = [100, 151643, 2, 103]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])  # Truncates at stop word position 1

    def test_partial_stop_word_match(self):
        # Partial match of multi-token stop word - should NOT truncate
        self.renderer.stop_words_id_list = [[200, 201, 202]]
        output_ids = [100, 200, 201, 999]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100, 200, 201, 999])

    def test_overlapping_stop_words(self):
        # Overlapping stop word sequences
        self.renderer.stop_words_id_list = [[200, 201], [201, 202]]
        output_ids = [100, 200, 201, 202, 103]
        # Should match [200, 201] first at position 1
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])

    def test_empty_sequence(self):
        # Empty output_ids
        output_ids = []
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [])

    def test_mtp_scenario(self):
        # Simulate MTP (Multiple Token Prediction) where 3 tokens generated at once
        # Stop word appears in the middle of the 3-token chunk
        self.renderer.stop_words_id_list = [[151643]]
        # Generated tokens: [100, 151643, 102] - stop word in middle
        output_ids = [98, 99, 100, 151643, 102]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        # Should truncate at position 3 (where stop word starts)
        self.assertEqual(result, [98, 99, 100])

    def test_extra_stop_words(self):
        # Test with extra stop words from get_all_extra_stop_word_ids_list
        self.renderer.get_all_extra_stop_word_ids_list = Mock(return_value=[[300, 301]])
        output_ids = [100, 300, 301, 102]
        result = self.renderer._remove_stop_word_ids(output_ids, [])
        self.assertEqual(result, [100])


class ProcessStopWordsTest(TestCase):
    """Test _process_stop_words method which handles string-level stop word processing."""

    def setUp(self):
        # Create a minimal mock renderer
        self.renderer = Mock(spec=CustomChatRenderer)
        self.renderer._process_stop_words = (
            CustomChatRenderer._process_stop_words.__get__(self.renderer)
        )
        self.status = StreamStatus(Mock())
        self.status.finish_reason = None

    def test_truncate_at_complete_stop_word(self):
        # Complete stop word found - should truncate and set finish_reason
        delta_string = "Hello<|observation|>world"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "Hello")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)
        self.assertFalse(should_buffer)

    def test_partial_stop_word_streaming(self):
        # Partial stop word at end in streaming mode - should buffer
        delta_string = "Hello<|obs"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = get_stop_word_slices(["<|observation|>"])

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        # stop_words_str lookup doesn't find complete match, so no truncation
        # But stop_word_slice_list detects partial match at end, so should buffer
        self.assertEqual(truncated, "Hello<|obs")  # No truncation from stop_words_str
        self.assertIsNone(self.status.finish_reason)  # No complete stop word found
        self.assertTrue(should_buffer)  # Should buffer because partial match detected

    def test_no_stop_word(self):
        # No stop word - should pass through unchanged
        delta_string = "Hello world"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = ["<|observation|>"]

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "Hello world")
        self.assertIsNone(self.status.finish_reason)
        self.assertFalse(should_buffer)

    def test_empty_string(self):
        # Empty string - should return empty, no buffering
        delta_string = ""
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "")
        self.assertFalse(should_buffer)

    def test_multiple_stop_words_truncate_at_first(self):
        # Multiple stop words - should truncate at earliest
        delta_string = "Start<|user|>middle<|observation|>end"
        stop_words_str = ["<|observation|>", "<|user|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "Start")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)

    def test_complete_before_partial(self):
        # Complete stop word found - should NOT buffer even if partial match exists
        delta_string = "Hello<|observation|>"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = ["<|observation|>"]

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "Hello")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)
        self.assertFalse(should_buffer)  # Complete match takes precedence

    def test_non_streaming_mode(self):
        # Non-streaming mode - same truncation behavior
        delta_string = "Hello<|observation|>world"
        stop_words_str = ["<|observation|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, False, self.status
        )

        self.assertEqual(truncated, "Hello")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)
        self.assertFalse(should_buffer)

    def test_unicode_stop_words(self):
        # Unicode stop words
        delta_string = "文本<|结束|>后续"
        stop_words_str = ["<|结束|>"]
        stop_word_slice_list = []

        truncated, should_buffer = self.renderer._process_stop_words(
            delta_string, stop_words_str, stop_word_slice_list, True, self.status
        )

        self.assertEqual(truncated, "文本")
        self.assertEqual(self.status.finish_reason, FinisheReason.stop)


if __name__ == "__main__":
    main()
