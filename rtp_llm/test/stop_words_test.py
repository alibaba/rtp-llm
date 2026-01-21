from unittest import TestCase, main

from rtp_llm.utils.word_util import (
    get_stop_word_slices,
    is_truncated,
    truncate_response_with_stop_words,
)


class StopWordTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_truncate_streaming_mode(self):
        # 1. 基本功能 - 单个停止词
        response = "Hello world, this is a test"
        stop_words = ["world"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words), "Hello "
        )

        # 2. 多个停止词 - 取最早出现的
        response = "Start middle end"
        stop_words = ["end", "middle"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words), "Start "
        )

        # 3. 停止词在开头 - 立即截断
        response = "Stop word at start"
        stop_words = ["Stop"]
        self.assertEqual(truncate_response_with_stop_words(response, stop_words), "")

        # 4. 没有匹配的停止词 - 返回完整响应
        response = "No stop words here"
        stop_words = ["missing"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words), response
        )

        # 5. 空停止词被忽略
        response = "Ignore empty stop words"
        stop_words = ["", "empty"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words), "Ignore "
        )

        # 6. 空响应
        response = ""
        stop_words = ["test"]
        self.assertEqual(truncate_response_with_stop_words(response, stop_words), "")

        # 7. 重叠停止词
        response = "abcde"
        stop_words = ["bc", "cd"]
        self.assertEqual(truncate_response_with_stop_words(response, stop_words), "a")

    def test_truncate_non_streaming_mode(self):
        # 1. 基本功能 - 单个停止词
        response = "Hello world, this is a test"
        stop_words = ["world"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, False), "Hello "
        )

        # 2. 多个停止词 - 取最早出现的
        response = "Start middle end"
        stop_words = ["end", "middle"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, False), "Start "
        )

        # 3. 停止词在开头
        response = "Stop word at start"
        stop_words = ["Stop"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, False), ""
        )

        # 4. 没有匹配的停止词
        response = "No stop words here"
        stop_words = ["missing"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, False), response
        )

        # 5. 空停止词被忽略
        response = "Ignore empty stop words"
        stop_words = ["", "empty"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, False), "Ignore "
        )

        # 6. 停止词在末尾
        response = "End with stop"
        stop_words = ["stop"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, False), "End with "
        )

    def test_truncate_slice_mode(self):
        # slice=True mode: matches partial or complete stop words at the END
        # 1. Complete stop word at end - should truncate
        response = "Hello<|observation|>"
        stop_words = ["<|observation|>"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, True, True), "Hello"
        )

        # 2. Stop word in middle - should NOT truncate (slice mode only checks end)
        response = "Hello<|observation|>world"
        stop_words = ["<|observation|>"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, True, True),
            response,
        )

        # 3. Partial stop word at end - SHOULD match and truncate
        response = "Hello<|obs"
        stop_words = get_stop_word_slices(["<|observation|>"])
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, True, True),
            "Hello",
        )

        # 4. Multiple stop words - match first one
        response = "Hello<|user|>"
        stop_words = ["<|observation|>", "<|user|>"]
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, True, True), "Hello"
        )

        # 5. Single character partial match
        response = "Hello<"
        stop_words = get_stop_word_slices(["<|observation|>"])
        self.assertEqual(
            truncate_response_with_stop_words(response, stop_words, True, True), "Hello"
        )

    def test_truncate_edge_cases(self):
        # Special characters in stop words
        response = "Code: ```python\nprint('hello')\n```"
        stop_words = ["```"]
        result = truncate_response_with_stop_words(response, stop_words, True)
        self.assertEqual(result, "Code: ")

        # Unicode stop words
        response = "文本<|结束|>后续"
        stop_words = ["<|结束|>"]
        result = truncate_response_with_stop_words(response, stop_words, True)
        self.assertEqual(result, "文本")

        # Very long stop word
        long_stop = "a" * 100
        response = "Start" + long_stop + "End"
        stop_words = [long_stop]
        result = truncate_response_with_stop_words(response, stop_words, True)
        self.assertEqual(result, "Start")


class IsTruncatedTest(TestCase):
    def test_partial_match_at_end(self):
        # is_truncated with pre-computed slices should detect partial stop words at end

        # 1. Ends with partial stop word - should return True
        text = "Hello<|obs"
        stop_words = ["<|observation|>"]
        slices = get_stop_word_slices(stop_words)
        self.assertTrue(is_truncated(text, slices, True, True))

        # 2. Ends with complete stop word - should return True (complete is also partial)
        text = "Hello<|observation|>"
        stop_words = ["<|observation|>"]
        slices = get_stop_word_slices(stop_words)
        self.assertTrue(is_truncated(text, slices, True, True))

        # 3. Doesn't end with stop word - should return False
        text = "Hello world"
        stop_words = ["<|observation|>"]
        slices = get_stop_word_slices(stop_words)
        self.assertFalse(is_truncated(text, slices, True, True))

        # 4. Partial match in middle, not at end - should return False
        text = "Hello<|obsworld"
        stop_words = ["<|observation|>"]
        slices = get_stop_word_slices(stop_words)
        self.assertFalse(is_truncated(text, slices, True, True))

    def test_multiple_stop_words(self):
        # 1. Matches one of multiple stop words (partial match)
        text = "Hello<|us"
        stop_words = ["<|observation|>", "<|user|>"]
        self.assertTrue(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

        # 2. Partial match with first char
        text = "Hello<"
        stop_words = ["<|observation|>", "<|user|>"]
        self.assertTrue(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

        # 3. No match with any stop word
        text = "Hello world"
        stop_words = ["<|observation|>", "<|user|>"]
        self.assertFalse(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

    def test_edge_cases(self):
        # 1. Empty string
        text = ""
        stop_words = ["<|observation|>"]
        self.assertFalse(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

        # 2. Empty stop words list
        text = "Hello"
        stop_words = []
        self.assertFalse(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

        # 3. Stop word longer than text
        text = "Hi"
        stop_words = ["<|observation|>"]
        self.assertFalse(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

        # 4. Single character partial match
        text = "Hello<"
        stop_words = ["<|observation|>"]
        self.assertTrue(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

    def test_unicode_partial_match(self):
        # Complete unicode match at end
        text = "文本<|结束|>"
        stop_words = ["<|结束|>"]
        self.assertTrue(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

        # Partial unicode match
        text = "文本<|结"
        stop_words = ["<|结束|>"]
        self.assertTrue(
            is_truncated(text, get_stop_word_slices(stop_words), True, True)
        )

    def test_streaming_vs_non_streaming(self):
        # Test the difference between streaming and non-streaming modes with slice=True
        stop_words = ["<|observation|>"]
        slices = get_stop_word_slices(stop_words)

        # Case 1: Text ends with partial stop word
        text_ends_with = "Hello<|obs"
        # Streaming mode: endswith check - should match
        self.assertTrue(is_truncated(text_ends_with, slices, True, True))
        # Non-streaming mode: find anywhere - should also match
        self.assertTrue(is_truncated(text_ends_with, slices, False, True))

        # Case 2: Partial stop word in middle, not at end
        text_middle = "Hello<|obsworld"
        # Streaming mode: endswith check - should NOT match (doesn't end with slice)
        self.assertFalse(is_truncated(text_middle, slices, True, True))
        # Non-streaming mode: find anywhere - SHOULD match ("<|obs" is in the string)
        self.assertTrue(is_truncated(text_middle, slices, False, True))

        # Case 3: No partial stop word anywhere
        text_none = "Hello world"
        # Both modes should return False
        self.assertFalse(is_truncated(text_none, slices, True, True))
        self.assertFalse(is_truncated(text_none, slices, False, True))


if __name__ == "__main__":
    main()
