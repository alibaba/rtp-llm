import os
from typing import Any
from unittest import TestCase, main

from transformers import AutoTokenizer

from rtp_llm.config.engine_config import SpecialTokens
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer,
)
from rtp_llm.pipeline.pipeline import Pipeline


class MockGenerateEnvConfig:
    """Mock GenerateEnvConfig for testing."""
    def __init__(self, think_mode=0, think_end_token_id=-1, think_end_tag="</think>\n\n"):
        self.think_mode = think_mode
        self.think_end_token_id = think_end_token_id
        self.think_end_tag = think_end_tag


class GenerateConfigTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(os.getcwd(), "rtp_llm/test")

    def _create_generate_config(self):
        return {
            "stop_words_str": ["hello", "what's your name"],
            "stop_words_list": [[8848]],
            "top_k": 1,
            "top_p": 0.95,
            "temperature": 0.8,
            "max_new_tokens": 100,
        }

    def _create_generate_config_for_select_tokens_id(self):
        return {"select_tokens_id": [0, 3]}

    def _create_kwargs(self):
        return {
            "stop_words_str": ["hi"],
            "stop_words_list": [[1551]],
            "top_k": 2,
            "top_p": 0.5,
            "max_new_tokens": 20,
        }

    def test_simple(self):
        special_tokens = SpecialTokens()
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=None,
        )
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)

        generate_config = Pipeline.create_generate_config(
            generate_config={},
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=None,
            **self._create_generate_config(),
        )
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)

    def test_kwargs_overwrite(self):
        special_tokens = SpecialTokens()
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=None,
            **self._create_kwargs(),
        )
        self.assertEqual(generate_config.stop_words_list, [[1551]])
        self.assertEqual(generate_config.stop_words_str, ["hi"])
        self.assertEqual(generate_config.top_k, 2)
        self.assertEqual(generate_config.top_p, 0.5)
        self.assertEqual(generate_config.max_new_tokens, 20)

    def test_stop_words_merge(self):
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=None,
        )
        self.assertEqual(generate_config.stop_words_list, [[8848], [1233, 19912]])
        self.assertEqual(
            generate_config.stop_words_str, ["hello", "what's your name", "gg"]
        )

    def test_stop_words_merge_with_toeknizer(self):
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
            generate_env_config=None,
        )
        self.assertEqual(
            generate_config.stop_words_list,
            [[8848], [1233, 19912], [14990], [12555, 594, 697, 829], [14398]],
        )
        self.assertEqual(
            generate_config.stop_words_str, ["hello", "what's your name", "gg"]
        )

    def test_select_tokens_id(self):
        special_tokens = SpecialTokens()
        generate_config = Pipeline.create_generate_config(
            generate_config=self._create_generate_config_for_select_tokens_id(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=None,
        )
        self.assertEqual(generate_config.select_tokens_id, [0, 3])
        self.assertEqual(generate_config.select_tokens_str, [])

        with self.assertRaisesRegex(Exception, "should be less than vocab_size"):
            generate_config = Pipeline.create_generate_config(
                generate_config=self._create_generate_config_for_select_tokens_id(),
                vocab_size=2,
                special_tokens=special_tokens,
                tokenizer=None,
                generate_env_config=None,
            )

    def test_same(self):
        special_tokens = SpecialTokens()
        special_tokens.stop_words_id_list = [[1233, 19912]]
        special_tokens.stop_words_str_list = ["gg"]

        a = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=None,
        )
        b = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=None,
        )
        a.gen_hash_value()
        b.gen_hash_value()
        self.assertTrue(a.is_same(b))

    def test_add_thinking_params(self):
        generate_env_config = MockGenerateEnvConfig(think_mode=1, think_end_token_id=102)
        special_tokens = SpecialTokens()
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 109})
        generate_config = Pipeline.create_generate_config(
            generate_config=generate_config_dict,
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
            generate_env_config=generate_env_config,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 109)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [102])

    def test_add_thinking_params_with_think_token(self):
        generate_env_config = MockGenerateEnvConfig(
            think_mode=1, 
            think_end_token_id=-1, 
            think_end_tag="</think>"
        )
        special_tokens = SpecialTokens()
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 20})
        generate_config = Pipeline.create_generate_config(
            generate_config=generate_config_dict,
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
            generate_env_config=generate_env_config,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 20)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [151649])

    def test_add_thinking_params_with_think_token_2(self):
        generate_env_config = MockGenerateEnvConfig(
            think_mode=1, 
            think_end_token_id=-1, 
            think_end_tag="</think>\n\n"
        )
        special_tokens = SpecialTokens()
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 20})
        generate_config = Pipeline.create_generate_config(
            generate_config=generate_config_dict,
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=tokenizer,
            generate_env_config=generate_env_config,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 20)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [151649, 271])


if __name__ == "__main__":
    main()
