import unittest

from rtp_llm.config.generate_config import RequestFormat
from rtp_llm.frontend.glm5_prompt import maybe_wrap_glm5_raw_prompt


class Glm5RawPromptTest(unittest.TestCase):
    def test_wraps_plain_glm5_raw_prompt(self):
        self.assertEqual(
            maybe_wrap_glm5_raw_prompt("glm_5", "你是谁", RequestFormat.RAW),
            "<|user|>你是谁\n<|assistant|><think></think>",
        )

    def test_keeps_formatted_glm5_prompt(self):
        prompt = "<|user|>你是谁\n<|assistant|><think></think>"
        self.assertEqual(
            maybe_wrap_glm5_raw_prompt("glm_5", prompt, RequestFormat.RAW), prompt
        )

    def test_keeps_non_raw_or_other_model_prompt(self):
        self.assertEqual(
            maybe_wrap_glm5_raw_prompt("glm_5", "你是谁", RequestFormat.CHAT_API),
            "你是谁",
        )
        self.assertEqual(
            maybe_wrap_glm5_raw_prompt("deepseek3", "你是谁", RequestFormat.RAW),
            "你是谁",
        )


if __name__ == "__main__":
    unittest.main()
