import os
from typing import Any, List, Optional
from unittest import TestCase, main

from transformers import AutoTokenizer

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.frontend.frontend_worker import FrontendWorker
from rtp_llm.frontend.openai_endpoint import OpenaiEndpoint
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest, GenerateConfig


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
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        generate_config = FrontendWorker.build_generation_config(
            tokenizer=None,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=self._create_generate_config(),
        )
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)

        generate_config = FrontendWorker.build_generation_config(
            tokenizer=None,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config={},
            **self._create_generate_config(),
        )
        self.assertEqual(generate_config.stop_words_list, [[8848]])
        self.assertEqual(generate_config.stop_words_str, ["hello", "what's your name"])
        self.assertEqual(generate_config.top_k, 1)
        self.assertEqual(generate_config.top_p, 0.95)
        self.assertEqual(generate_config.max_new_tokens, 100)

    def test_kwargs_overwrite(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        generate_config = FrontendWorker.build_generation_config(
            tokenizer=None,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=self._create_generate_config(),
            **self._create_kwargs(),
        )
        self.assertEqual(generate_config.stop_words_list, [[1551]])
        self.assertEqual(generate_config.stop_words_str, ["hi"])
        self.assertEqual(generate_config.top_k, 2)
        self.assertEqual(generate_config.top_p, 0.5)
        self.assertEqual(generate_config.max_new_tokens, 20)

    def test_stop_words_merge(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        parameter.special_tokens.stop_words_id_list = [[1233, 19912]]
        parameter.special_tokens.stop_words_str_list = ["gg"]
        generate_config = FrontendWorker.build_generation_config(
            tokenizer=None,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=self._create_generate_config(),
        )
        self.assertEqual(generate_config.stop_words_list, [[8848], [1233, 19912]])
        self.assertEqual(
            generate_config.stop_words_str, ["hello", "what's your name", "gg"]
        )

    def test_stop_words_merge_with_toeknizer(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        parameter.special_tokens.stop_words_id_list = [[1233, 19912]]
        parameter.special_tokens.stop_words_str_list = ["gg"]
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        generate_config = FrontendWorker.build_generation_config(
            tokenizer=tokenizer,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=self._create_generate_config(),
        )
        self.assertEqual(
            generate_config.stop_words_list,
            [[8848], [1233, 19912], [14990], [12555, 594, 697, 829], [14398]],
        )
        self.assertEqual(
            generate_config.stop_words_str, ["hello", "what's your name", "gg"]
        )

    def test_select_tokens_id(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        generate_config = FrontendWorker.build_generation_config(
            tokenizer=None,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=self._create_generate_config_for_select_tokens_id(),
        )
        self.assertEqual(generate_config.select_tokens_id, [0, 3])
        self.assertEqual(generate_config.select_tokens_str, [])

        with self.assertRaisesRegex(Exception, "should be less than vocab_size"):
            generate_config = FrontendWorker.build_generation_config(
                tokenizer=None,
                vocab_size=2,
                special_tokens=parameter.special_tokens,
                generate_config=self._create_generate_config_for_select_tokens_id(),
            )

    def test_same(self):
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        parameter.special_tokens.stop_words_id_list = [[1233, 19912]]
        parameter.special_tokens.stop_words_str_list = ["gg"]

        a = FrontendWorker.build_generation_config(
            tokenizer=None,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=self._create_generate_config(),
        )
        b = FrontendWorker.build_generation_config(
            tokenizer=None,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=self._create_generate_config(),
        )
        a.gen_hash_value()
        b.gen_hash_value()
        self.assertTrue(a.is_same(b))

    def test_add_thinking_params(self):
        StaticConfig.generate_env_config.think_mode = 1
        StaticConfig.generate_env_config.think_end_token_id = 102
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 109})
        generate_config = FrontendWorker.build_generation_config(
            tokenizer=tokenizer,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=generate_config_dict,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 109)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [102])

    def test_add_thinking_params_with_think_token(self):
        StaticConfig.generate_env_config.think_mode = 1
        StaticConfig.generate_env_config.think_end_token_id = -1
        StaticConfig.generate_env_config.think_end_tag = "</think>"
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 20})
        generate_config = FrontendWorker.build_generation_config(
            tokenizer=tokenizer,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=generate_config_dict,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 20)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [151649])

    def test_add_thinking_params_with_think_token_2(self):
        StaticConfig.generate_env_config.think_mode = 1
        StaticConfig.generate_env_config.think_end_token_id = -1
        StaticConfig.generate_env_config.think_end_tag = "</think>\n\n"
        parameter = GptInitModelParameters(0, 0, 0, 0, 0)
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        generate_config_dict = self._create_generate_config()
        generate_config_dict.update({"max_thinking_tokens": 20})
        generate_config = FrontendWorker.build_generation_config(
            tokenizer=tokenizer,
            vocab_size=100,
            special_tokens=parameter.special_tokens,
            generate_config=generate_config_dict,
        )
        self.assertEqual(generate_config.max_thinking_tokens, 20)
        self.assertEqual(generate_config.in_think_mode, True)
        self.assertEqual(generate_config.end_think_token_ids, [151649, 271])


class OpenaiGenerateConfigTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(
            os.getcwd(), "rtp_llm/test/model_test/fake_test/testdata"
        )
        self.tokenizer = QWenTokenizer(
            os.path.join(self.test_data_path, "qwen_7b/tokenizer/qwen.tiktoken"),
            *args,
            **kwargs,
        )

    def _get_model_config(self, model_config=None):
        model_config = model_config if model_config is not None else {}
        model_config = {
            "head_num": 1024,
            "size_per_head": 1024,
            "layer_num": 1024,
            "max_seq_len": 1024,
            "vocab_size": 1024,
            **model_config,
        }
        return GptInitModelParameters(**model_config)

    def _generate_config_with_stop_word(
        self,
        model_stop_word_str: Optional[List[str]] = None,
        model_stop_word_list: Optional[List[str]] = None,
        env_stop_word_str: Optional[str] = None,
        env_stop_word_list: Optional[str] = None,
        req_stop: Optional[List[str]] = None,
        req_config_stop_word_str: Optional[List[str]] = None,
        req_config_stop_word_list: Optional[List[List[int]]] = None,
    ):
        model_config = self._get_model_config()
        if model_stop_word_str is not None:
            model_config.special_tokens.stop_words_str_list = model_stop_word_str
        if model_stop_word_list is not None:
            model_config.special_tokens.stop_words_id_list = model_stop_word_list
        if env_stop_word_str is not None:
            model_config.py_env_configs.generate_env_config.stop_words_str = (
                env_stop_word_str
            )
        if env_stop_word_list is not None:
            model_config.py_env_configs.generate_env_config.stop_words_list = (
                env_stop_word_list
            )

        openai_endpoint = OpenaiEndpoint(model_config, self.tokenizer, None)

        request = ChatCompletionRequest(messages=[])
        if req_stop is not None:
            request.stop = req_stop
        if req_config_stop_word_str is not None:
            if request.extra_configs is None:
                request.extra_configs = GenerateConfig()
            request.extra_configs.stop_words_str = req_config_stop_word_str
        if req_config_stop_word_list is not None:
            if request.extra_configs is None:
                request.extra_configs = GenerateConfig()
            request.extra_configs.stop_words_list = req_config_stop_word_list

        return openai_endpoint._extract_generation_config(request)

    def assert_config_stop_word(
        self,
        expect_stop_word_str: Optional[List[str]] = None,
        expect_stop_word_list: Optional[List[List[str]]] = None,
        **kwargs,
    ):
        config = self._generate_config_with_stop_word(**kwargs)
        if expect_stop_word_str is not None:
            self.assertEqual(
                sorted(config.stop_words_str), sorted(expect_stop_word_str)
            )
        if expect_stop_word_list is not None:
            self.assertEqual(
                sorted(config.stop_words_list), sorted(expect_stop_word_list)
            )

    def test_stop_word_config(self):
        self.assert_config_stop_word(
            expect_stop_word_str=["<|im_end|>", "<|endoftext|>"],
            expect_stop_word_list=[[151643], [151645]],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "model stop word",
                "another model stop word",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2528, 2936, 3409],
                [41963, 1614, 2936, 3409],
            ],
            model_stop_word_str=["model stop word", "another model stop word"],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "model stop list",
                "another model stop list",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2528, 2936, 1140],
                [41963, 1614, 2936, 1140],
            ],
            model_stop_word_list=[[2528, 2936, 1140], [41963, 1614, 2936, 1140]],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "env stop word",
                "another env stop word",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [3160, 2936, 3409],
                [41963, 6105, 2936, 3409],
            ],
            env_stop_word_str='["env stop word", "another env stop word"]',
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "env stop list",
                "another env stop list",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [3160, 2936, 1140],
                [41963, 6105, 2936, 1140],
            ],
            env_stop_word_list="[[3160, 2936, 1140], [41963, 6105, 2936, 1140]]",
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "req stop word",
                "another req stop word",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2958, 2936, 3409],
                [41963, 4232, 2936, 3409],
            ],
            req_stop=["req stop word", "another req stop word"],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",
                "req config stop word",
                "another config req stop word",
            ],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2958, 2193, 2936, 3409],
                [41963, 2193, 4232, 2936, 3409],
            ],
            req_config_stop_word_str=[
                "req config stop word",
                "another config req stop word",
            ],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=["<|im_end|>", "<|endoftext|>"],
            expect_stop_word_list=[
                [151643],
                [151645],
                [2958, 2193, 2936, 1140],
                [41963, 2193, 4232, 2936, 1140],
            ],
            req_config_stop_word_list=[
                [2958, 2193, 2936, 1140],
                [41963, 2193, 4232, 2936, 1140],
            ],
        )

        self.assert_config_stop_word(
            expect_stop_word_str=[
                "<|im_end|>",
                "<|endoftext|>",  # default stop word
                "model stop word",
                "another model stop word",  # model_stop_word_str
                "model stop list",
                "another model stop list",  # model_stop_word_list
                "env stop word",
                "another env stop word",  # env_stop_word_str
                "env stop list",
                "another env stop list",  # env_stop_word_list
                "req stop word",
                "another req stop word",  # req_stop
                "req config stop word",
                "another config req stop word",  # req_config_stop_word_str
                "dup stop word",
                "dup stop list",  # duplicate stop word
            ],
            expect_stop_word_list=[
                [151643],
                [151645],  # default stop word list
                [2528, 2936, 3409],
                [41963, 1614, 2936, 3409],  # model_stop_word_str
                [2528, 2936, 1140],
                [41963, 1614, 2936, 1140],  # model_stop_word_list
                [3160, 2936, 3409],
                [41963, 6105, 2936, 3409],  # env_stop_word_str
                [3160, 2936, 1140],
                [41963, 6105, 2936, 1140],  # env_stop_word_list
                [2958, 2936, 3409],
                [41963, 4232, 2936, 3409],  # req_stop
                [2958, 2193, 2936, 3409],
                [41963, 2193, 4232, 2936, 3409],  # req_config_stop_word_str
                [2958, 2193, 2936, 1140],
                [41963, 2193, 4232, 2936, 1140],  # req_config_stop_word_list
                [21912, 2936, 1140],
                [21912, 2936, 3409],  # duplicate stop word
            ],
            model_stop_word_str=[
                "model stop word",
                "another model stop word",
                "dup stop word",
            ],
            model_stop_word_list=[
                [2528, 2936, 1140],
                [41963, 1614, 2936, 1140],
                [21912, 2936, 1140],
            ],
            env_stop_word_str='["env stop word", "another env stop word", "dup stop word"]',
            env_stop_word_list="[[3160, 2936, 1140], [41963, 6105, 2936, 1140], [21912, 2936, 1140]]",
            req_stop=["req stop word", "another req stop word", "dup stop word"],
            req_config_stop_word_str=[
                "req config stop word",
                "another config req stop word",
                "dup stop word",
            ],
            req_config_stop_word_list=[
                [2958, 2193, 2936, 1140],
                [41963, 2193, 4232, 2936, 1140],
                [21912, 2936, 1140],
            ],
        )


if __name__ == "__main__":
    main()
