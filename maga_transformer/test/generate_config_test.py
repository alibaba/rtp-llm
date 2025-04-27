import os
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.models.starcoder import StarcoderTokenizer
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from transformers import AutoTokenizer
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from unittest import TestCase, main
from typing import Any

class GenerateConfigTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(
            os.getcwd(), 'maga_transformer/test'
        )

    def _create_generate_config(self):
        return {
            "stop_words_str": ["hello", "what's your name"],
            "stop_words_list": [[8848]],
            "top_k": 1,
            "top_p": 0.95,
            "temperature": 0.8,
            "max_new_tokens": 100
        }

    def _create_generate_config_for_select_tokens_id(self):
        return {
            "select_tokens_id": [0, 3]
        }

    def _create_kwargs(self):
        return {
            "stop_words_str": ["hi"],
            "stop_words_list": [[1551]],
            "top_k": 2,
            "top_p": 0.5,
            "max_new_tokens": 20
        }

    def test_simple(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        generate_config = Pipeline.create_generate_config(tokenizer=None, vocab_size=100,
                                                          special_tokens=parameter.special_tokens, generate_config=self._create_generate_config())
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)

        generate_config = Pipeline.create_generate_config(tokenizer=None, vocab_size=100,
                                                          special_tokens=parameter.special_tokens, generate_config={}, **self._create_generate_config())
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)


    def test_kwargs_overwrite(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        generate_config = Pipeline.create_generate_config(tokenizer=None, vocab_size=100,
                                                          special_tokens=parameter.special_tokens,
                                                          generate_config=self._create_generate_config(),**self._create_kwargs())
        self.assertEqual(generate_config.stop_words_list, [[1551]])
        self.assertEqual(generate_config.stop_words_str, ["hi"])
        self.assertEqual(generate_config.top_k, 2)
        self.assertEqual(generate_config.top_p, 0.5)
        self.assertEqual(generate_config.max_new_tokens, 20)

    def test_stop_words_merge(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        parameter.special_tokens.stop_words_id_list = [[1233, 19912]]
        parameter.special_tokens.stop_words_str_list = ["gg"]
        generate_config = Pipeline.create_generate_config(tokenizer=None, vocab_size=100,
                                                          special_tokens=parameter.special_tokens, generate_config=self._create_generate_config())
        self.assertEqual(generate_config.stop_words_list, [[8848], [1233, 19912]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name", "gg"])
    
    def test_stop_words_merge_with_toeknizer(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        parameter.special_tokens.stop_words_id_list = [[1233, 19912]]
        parameter.special_tokens.stop_words_str_list = ["gg"]
        tokenizer = QWenTokenizer(f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken")
        generate_config = Pipeline.create_generate_config(tokenizer=tokenizer, vocab_size=100,
                                                          special_tokens=parameter.special_tokens, generate_config=self._create_generate_config())
        self.assertEqual(generate_config.stop_words_list, [[8848], [1233, 19912], [14990], [12555, 594, 697, 829], [14398]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name", "gg"])
        
    def test_select_tokens_id(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        generate_config = Pipeline.create_generate_config(tokenizer=None, vocab_size=100,
                                                          special_tokens=parameter.special_tokens,
                                                          generate_config=self._create_generate_config_for_select_tokens_id())
        self.assertEqual(generate_config.select_tokens_id, [0, 3])
        self.assertEqual(generate_config.select_tokens_str, [])

        with self.assertRaisesRegex(Exception, "should be less than vocab_size"):
            generate_config = Pipeline.create_generate_config(tokenizer=None, vocab_size=2,
                                                          special_tokens=parameter.special_tokens,
                                                          generate_config=self._create_generate_config_for_select_tokens_id())

    def test_same(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        parameter.special_tokens.stop_words_id_list = [[1233, 19912]]
        parameter.special_tokens.stop_words_str_list = ["gg"]

        a = Pipeline.create_generate_config(tokenizer=None, vocab_size=100,
                                            special_tokens=parameter.special_tokens, generate_config=self._create_generate_config())
        b = Pipeline.create_generate_config(tokenizer=None, vocab_size=100,
                                            special_tokens=parameter.special_tokens, generate_config=self._create_generate_config())
        a.gen_hash_value()
        b.gen_hash_value()
        self.assertTrue(a.is_same(b))
    
    def test_add_thinking_params(self):
        os.environ["THINK_MODE"] = "1"
        os.environ["THINK_END_TOKEN_ID"] = "102"
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        tokenizer = QWenTokenizer(f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken")
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({
            "max_thinking_tokens": 109
        })
        generate_config = Pipeline.create_generate_config(tokenizer=tokenizer, vocab_size=100,
                                                          special_tokens=parameter.special_tokens, generate_config=generate_config_dict)
        self.assertEqual(generate_config.max_thinking_tokens, 109)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [102])

    def test_add_thinking_params_with_think_token(self):
        os.environ["THINK_MODE"] = "1"
        os.environ["THINK_END_TOKEN_ID"] = "-1"
        os.environ["THINK_END_TAG"] = "</think>"
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({
            "max_thinking_tokens": 20
        })
        generate_config = Pipeline.create_generate_config(tokenizer=tokenizer, vocab_size=100,
                                                          special_tokens=parameter.special_tokens, generate_config=generate_config_dict)
        self.assertEqual(generate_config.max_thinking_tokens, 20)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [151649])

    def test_add_thinking_params_with_think_token_2(self):
        os.environ["THINK_MODE"] = "1"
        os.environ["THINK_END_TOKEN_ID"] = "-1"
        os.environ["THINK_END_TAG"] = "</think>\n\n"
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({
            "max_thinking_tokens": 20
        })
        generate_config = Pipeline.create_generate_config(tokenizer=tokenizer, vocab_size=100,
                                                          special_tokens=parameter.special_tokens, generate_config=generate_config_dict)
        self.assertEqual(generate_config.max_thinking_tokens, 20)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [151649, 271])

if __name__ == '__main__':
    main()
