import json
import os
from typing import Any, List, Optional
from unittest import TestCase, main
from unittest.mock import patch

from transformers import AutoTokenizer

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.config.response_format_builder import (
    ReasoningFormat,
    ResponseFormatBuilder,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest, GenerateConfig
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.ops import SpecialTokens
from rtp_llm.pipeline.pipeline import Pipeline


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
            generate_env_config=GenerateEnvConfig(),
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
            generate_env_config=GenerateEnvConfig(),
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
            generate_env_config=GenerateEnvConfig(),
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
            generate_env_config=GenerateEnvConfig(),
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
            generate_env_config=GenerateEnvConfig(),
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
            generate_env_config=GenerateEnvConfig(),
        )
        self.assertEqual(generate_config.select_tokens_id, [0, 3])
        self.assertEqual(generate_config.select_tokens_str, [])

        with self.assertRaisesRegex(Exception, "should be less than vocab_size"):
            generate_config = Pipeline.create_generate_config(
                generate_config=self._create_generate_config_for_select_tokens_id(),
                vocab_size=2,
                special_tokens=special_tokens,
                tokenizer=None,
                generate_env_config=GenerateEnvConfig(),
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
            generate_env_config=GenerateEnvConfig(),
        )
        b = Pipeline.create_generate_config(
            generate_config=self._create_generate_config(),
            vocab_size=100,
            special_tokens=special_tokens,
            tokenizer=None,
            generate_env_config=GenerateEnvConfig(),
        )
        a.gen_hash_value()
        b.gen_hash_value()
        self.assertTrue(a.is_same(b))

    def test_add_thinking_params(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
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
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>"
        special_tokens = SpecialTokens()
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = BaseTokenizer(tokenizer_path)
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
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>\n\n"
        special_tokens = SpecialTokens()
        tokenizer_path = f"{self.test_data_path}/model_test/fake_test/testdata/deepseek_r1_qwen_14b_tokenizer"
        tokenizer = BaseTokenizer(tokenizer_path)
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
        special_tokens = SpecialTokens()
        if model_stop_word_str is not None:
            special_tokens.stop_words_str_list = model_stop_word_str
        if model_stop_word_list is not None:
            special_tokens.stop_words_id_list = model_stop_word_list

        generate_env_config = GenerateEnvConfig()
        if env_stop_word_str is not None:
            generate_env_config.stop_words_str = env_stop_word_str
        if env_stop_word_list is not None:
            generate_env_config.stop_words_list = env_stop_word_list

        # Create ModelConfig object
        model_config = ModelConfig()
        model_config.generate_env_config = generate_env_config
        model_config.render_config = RenderConfig()
        model_config.special_tokens = special_tokens
        model_config.max_seq_len = 1024
        model_config.template_type = None
        model_config.model_name = ""
        model_config.ckpt_path = ""

        openai_endpoint = OpenaiEndpoint(
            model_config=model_config,
            misc_config=PyMiscellaneousConfig(),
            vit_config=VitConfig(),
            tokenizer=self.tokenizer,
            backend_rpc_server_visitor=None,
        )

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


class GrammarMultiSequenceConfigTest(TestCase):
    """Frontend normalization preserves fields; backend owns capability validation."""

    def _apply(self, **fields):
        cfg = GenerateConfig(**fields)
        cfg.validate()
        ResponseFormatBuilder(cfg).apply()
        return cfg

    def _assert_rejected(self, exception_type: ExceptionType, **fields):
        cfg = GenerateConfig(**fields)
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.validate()
            ResponseFormatBuilder(cfg).apply()
        self.assertEqual(ctx.exception.exception_type, exception_type)

    def test_grammar_field_plus_multi_sequence_is_preserved(self):
        cases = [
            {"json_schema": '{"type": "object"}', "num_beams": 4},
            {
                "response_format": {"type": "regex", "pattern": r"\d+"},
                "variable_num_beams": [1, 3],
            },
            {
                "response_format": '{"type":"json_object"}',
                "num_return_sequences": 2,
            },
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                cfg = self._apply(**fields)
                self.assertTrue(
                    any(
                        getattr(cfg, field) is not None
                        for field in ("json_schema", "regex", "ebnf", "structural_tag")
                    )
                )

    def test_grammar_or_beam_alone_allowed(self):
        for fields in [
            {"json_schema": '{"type": "object"}'},
            {"num_beams": 4},
            {"num_beams": 4, "response_format": {"type": "text"}},
        ]:
            with self.subTest(fields=fields):
                self._apply(**fields)

    def test_empty_direct_grammar_field_rejected(self):
        self._assert_rejected(ExceptionType.ERROR_INPUT_FORMAT_ERROR, regex="")


class ResponseFormatProjectionTest(TestCase):
    """rf projected to typed fields and cleared by ResponseFormatBuilder."""

    def _terminate_without_stop_token(self, cfg: GenerateConfig) -> bool:
        return ResponseFormatBuilder.grammar_terminate_without_stop_token(cfg)

    def _validate(
        self,
        cfg: GenerateConfig,
        reasoning_format: Optional[ReasoningFormat] = None,
    ):
        cfg.validate()
        ResponseFormatBuilder(cfg, reasoning_format=reasoning_format).apply()

    def _enable_thinking(
        self,
        cfg: GenerateConfig,
        think_end_tag: str = "</think>\n\n",
        think_end_token_id: int = -1,
    ):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = think_end_token_id
        generate_env_config.think_end_tag = think_end_tag
        cfg.add_thinking_params(
            None,
            generate_env_config,
            normalize_response_format=False,
        )
        return ReasoningFormat.from_generate_env_config(generate_env_config)

    def test_response_format_projected_to_typed_grammar_field(self):
        cases = [
            (
                GenerateConfig(
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"schema": {"type": "string"}},
                    }
                ),
                "json_schema",
                '{"type":"string"}',
                True,
            ),
            (
                GenerateConfig(response_format={"type": "json_object"}),
                "json_schema",
                '{"type":"object"}',
                True,
            ),
            (
                GenerateConfig(response_format={"type": "regex", "pattern": r"\d+"}),
                "regex",
                r"\d+",
                False,
            ),
            (
                GenerateConfig(response_format='{"type":"json_object"}'),
                "json_schema",
                '{"type":"object"}',
                True,
            ),
        ]
        for cfg, field, expected, terminate_without_stop_token in cases:
            with self.subTest(field=field, expected=expected):
                self._validate(cfg)
                self.assertIsNone(cfg.response_format)
                self.assertEqual(getattr(cfg, field), expected)
                self.assertEqual(
                    self._terminate_without_stop_token(cfg),
                    terminate_without_stop_token,
                )

    def test_response_format_clears_or_overrides_stale_typed_field(self):
        cfg = GenerateConfig(
            response_format={"type": "text"},
            json_schema='{"type": "object"}',
        )
        self._validate(cfg)
        self.assertIsNone(cfg.response_format)
        self.assertIsNone(cfg.json_schema)
        self.assertFalse(self._terminate_without_stop_token(cfg))

        cfg = GenerateConfig(
            response_format={"type": "regex", "pattern": r"[a-z]+"},
            json_schema='{"type": "object"}',
        )
        self._validate(cfg)
        self.assertIsNone(cfg.response_format)
        self.assertIsNone(cfg.json_schema)
        self.assertEqual(cfg.regex, r"[a-z]+")
        self.assertFalse(self._terminate_without_stop_token(cfg))

    def test_response_format_recursion_error_is_reported_as_input_error(self):
        cfg = GenerateConfig(
            response_format=(
                '{"type":"structural_tag","structural_tag":'
                '{"type":"structural_tag","format":{}}}'
            )
        )

        with patch(
            "rtp_llm.config.response_format_builder.parse_response_format",
            side_effect=RecursionError("maximum JSON nesting depth exceeded"),
        ):
            with self.assertRaises(FtRuntimeException) as ctx:
                self._validate(cfg)
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )

    def test_direct_grammar_dict_normalized(self):
        cfg = GenerateConfig(json_schema={"type": "object"})
        self._validate(cfg)
        self.assertEqual(cfg.json_schema, '{"type":"object"}')

    def test_reasoning_json_schema_wrapped_as_structural_tag(self):
        cfg = GenerateConfig(
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
            max_thinking_tokens=64,
        )
        reasoning_format = self._enable_thinking(cfg)
        self._validate(cfg, reasoning_format=reasoning_format)

        self.assertIsNone(cfg.response_format)
        self.assertIsNone(cfg.json_schema)
        self.assertIsNone(cfg.regex)
        self.assertIsNone(cfg.ebnf)
        self.assertTrue(self._terminate_without_stop_token(cfg))

        structural_tag = json.loads(cfg.structural_tag)
        self.assertEqual(structural_tag["type"], "structural_tag")
        elements = structural_tag["format"]["elements"]
        self.assertEqual(elements[0]["type"], "tag")
        self.assertEqual(elements[0]["begin"], "")
        self.assertEqual(elements[0]["end"], "</think>\n\n")
        self.assertEqual(elements[0]["content"], {"type": "any_text", "max_tokens": 64})
        self.assertEqual(elements[1]["type"], "json_schema")
        self.assertEqual(elements[1]["json_schema"], {"type": "object"})
        self.assertEqual(elements[1]["style"], "json")

    def test_reasoning_uses_token_end_when_think_end_token_id_is_configured(self):
        cfg = GenerateConfig(
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
            max_thinking_tokens=16,
        )
        reasoning_format = self._enable_thinking(
            cfg, think_end_tag="this-string-should-not-be-used", think_end_token_id=123
        )
        self._validate(cfg, reasoning_format=reasoning_format)

        structural_tag = json.loads(cfg.structural_tag)
        elements = structural_tag["format"]["elements"]
        self.assertEqual(elements[0]["end"], {"type": "token", "token": 123})
        self.assertTrue(self._terminate_without_stop_token(cfg))

    def test_reasoning_without_grammar_wraps_any_text_structural_tag(self):
        cfg = GenerateConfig(response_format={"type": "text"})
        reasoning_format = self._enable_thinking(cfg)
        self._validate(cfg, reasoning_format=reasoning_format)

        self.assertIsNone(cfg.json_schema)
        structural_tag = json.loads(cfg.structural_tag)
        elements = structural_tag["format"]["elements"]
        self.assertEqual(elements[0]["type"], "tag")
        self.assertEqual(elements[1], {"type": "any_text"})
        self.assertFalse(self._terminate_without_stop_token(cfg))

    def test_reasoning_final_structural_tag_with_existing_budget_rejected(self):
        cfg = GenerateConfig(
            structural_tag={
                "type": "structural_tag",
                "format": {"type": "any_text", "max_tokens": 3},
            }
        )
        reasoning_format = self._enable_thinking(cfg)

        with self.assertRaises(FtRuntimeException) as ctx:
            self._validate(cfg, reasoning_format=reasoning_format)
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSUPPORTED_OPERATION
        )

    def test_deeply_nested_structural_tag_is_reported_as_input_error(self):
        depth = 2000
        nested_format = '{"child":' * depth + "{}" + "}" * depth
        cfg = GenerateConfig(
            structural_tag=('{"type":"structural_tag","format":' + nested_format + "}")
        )
        reasoning_format = self._enable_thinking(cfg)

        with self.assertRaises(FtRuntimeException) as ctx:
            self._validate(cfg, reasoning_format=reasoning_format)
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )


class RawUpdateAndGrammarConflictTest(TestCase):
    """Raw request updates still coerce response_format and reject grammar conflicts."""

    def _terminate_without_stop_token(self, cfg: GenerateConfig) -> bool:
        return ResponseFormatBuilder.grammar_terminate_without_stop_token(cfg)

    def test_update_and_pop_coerces_string_envelope(self):
        cfg = GenerateConfig()
        remain = cfg.update_and_pop(
            {"response_format": '{"type":"regex","pattern":"\\\\d+"}', "stranger": 1}
        )
        self.assertEqual(remain, {"stranger": 1})
        cfg.validate()
        ResponseFormatBuilder(cfg).apply()
        self.assertEqual(cfg.regex, r"\d+")

    def test_internal_terminate_flag_is_not_dumped_as_user_config(self):
        cfg = GenerateConfig(
            grammar_terminate_without_stop_token=True,
        )
        self.assertFalse(self._terminate_without_stop_token(cfg))
        self.assertNotIn(
            "grammar_terminate_without_stop_token", GenerateConfig.model_fields
        )
        self.assertNotIn("grammar_terminate_without_stop_token", cfg.model_dump())

    def test_update_rejects_malformed_envelope(self):
        cfg = GenerateConfig()
        cfg.update({"response_format": {"type": "json_schema"}})
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.validate()
            ResponseFormatBuilder(cfg).apply()
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )

    def test_update_and_pop_rejects_invalid_json_envelope(self):
        cfg = GenerateConfig()
        remain = cfg.update_and_pop(
            {"response_format": '{"type":"json_object"', "stranger": 1}
        )
        self.assertEqual(remain, {"stranger": 1})
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.validate()
            ResponseFormatBuilder(cfg).apply()
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )

    def test_multiple_typed_grammar_fields_rejected(self):
        cfg = GenerateConfig(json_schema='{"type": "object"}', regex=r"\d+")
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.validate()
            ResponseFormatBuilder(cfg).apply()
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSUPPORTED_OPERATION
        )


if __name__ == "__main__":
    main()
