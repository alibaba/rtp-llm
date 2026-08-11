import logging
import unittest

from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser


class _RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class GlmReasoningHotPathTest(unittest.TestCase):
    def test_streaming_parser_does_not_log_generated_text_at_info(self):
        parser = ReasoningParser(model_type="glm45", force_reasoning=True)
        handler = _RecordHandler()
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        try:
            reasoning, content = parser.parse_stream_chunk("private reasoning")
            self.assertEqual("private reasoning", reasoning)
            self.assertEqual("", content)

            reasoning, content = parser.parse_stream_chunk("</think>answer")
            self.assertEqual("", reasoning)
            self.assertEqual("answer", content)
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(original_level)

        reasoning_logs = [
            record
            for record in handler.records
            if "REASONING_DEBUG" in record.getMessage()
        ]
        self.assertEqual([], reasoning_logs)
        logged_text = "\n".join(record.getMessage() for record in handler.records)
        self.assertNotIn("private reasoning", logged_text)
        self.assertNotIn("</think>answer", logged_text)


if __name__ == "__main__":
    unittest.main()
