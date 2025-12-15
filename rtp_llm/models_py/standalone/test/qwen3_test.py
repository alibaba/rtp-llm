import logging
from unittest import TestCase, main

import torch

from rtp_llm.models_py.standalone.auto_model import AutoModel

logging.basicConfig(
    level="INFO",
    format="[process-%(process)d][%(name)s][%(asctime)s.%(msecs)03d][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Qwen3AutoPyModelTest(TestCase):

    def setUp(self):
        # test_msg1 consist of 20 tokens
        self.test_msg1 = [{"role": "user", "content": "你好，请用较长篇幅介绍自己"}]
        self.max_new_tokens1 = 45
        self.expected_output_text1 = "你好！我是你的AI助手，我是一个基于深度学习的多模态语言模型，专为用户提供自然、流畅的对话体验。我能够理解多种语言，包括中文、英文、西班牙语、法语"

        self.test_msg2 = [{"role": "user", "content": "3.9和3.11哪个大"}]
        self.max_new_tokens2 = 50
        self.expected_output_text2 = "3.9 和 3.11 中，**3.9 大于 3.11**。"

        self.max_total_tokens = 100  # max_total_tokens is about kv_cache capacity
        self.tokens_per_block = 2
        self.model = AutoModel.from_pretrained(
            model_path_or_name="Qwen/Qwen3-0.6B",
            max_total_tokens=self.max_total_tokens,
            tokens_per_block=self.tokens_per_block,
        )
        logging.info(f"model created")

    def test_qwen3_auto_model(self):
        # test compute_dtype
        self.assertEqual(self.model.compute_dtype, torch.bfloat16)

        # test simple message
        output_text1 = self._run_message(
            self.test_msg1, max_new_tokens=self.max_new_tokens1
        )
        logging.info(f"output_text1: {output_text1}")
        self.assertEqual(output_text1, self.expected_output_text1)

        output_text2 = self._run_message(
            self.test_msg2, max_new_tokens=self.max_new_tokens2
        )
        logging.info(f"output_text2: {output_text2}")
        self.assertEqual(output_text2, self.expected_output_text2)

        # test max_mew_tokens exceed max_total_tokens
        with self.assertRaises(AssertionError) as context:
            self._run_message(self.test_msg1, max_new_tokens=self.max_new_tokens1 + 100)
        self.assertEqual("sequence_length is too long", str(context.exception))

    def _run_message(
        self,
        message: list[dict[str, str]],
        max_new_tokens: int = 1000,
    ) -> str:
        input_text = self.model.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        output_text = self._run_raw_text(input_text, max_new_tokens)
        return output_text

    def _run_raw_text(self, text: str, max_new_tokens: int) -> str:
        input_ids = self.model.tokenizer.encode(text)
        output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        output_text = self.model.tokenizer.decode(output_ids)
        return output_text


if __name__ == "__main__":
    main()
