from unittest import TestCase, main

from rtp_llm.utils.word_util import truncate_response_with_stop_words


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


if __name__ == "__main__":
    main()
