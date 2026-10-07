import copy
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from unittest import TestCase, main

from transformers import AutoTokenizer

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import ThinkingMode, thinking_mode_from_value
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.config.response_format import ResponseFormat
from rtp_llm.config.response_format_compiler import (
    ReasoningFormat,
    restore_final_constraint,
    validate_engine_ready,
)
from rtp_llm.config.thinking_mode import normalize_think_mode
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest, GenerateConfig
from rtp_llm.openai.api_datatype import ResponseFormat as OpenAIResponseFormat
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer
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

    def test_think_mode_accepts_strings_and_legacy_aliases(self):
        cases = {
            "disabled": ("disabled", ThinkingMode.DISABLED),
            "adaptive": ("adaptive", ThinkingMode.ADAPTIVE),
            "enabled": ("enabled", ThinkingMode.ENABLED),
            "0": ("disabled", ThinkingMode.DISABLED),
            0: ("disabled", ThinkingMode.DISABLED),
            "1": ("enabled", ThinkingMode.ENABLED),
            1: ("enabled", ThinkingMode.ENABLED),
        }
        for value, (normalized, mode) in cases.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_think_mode(value), normalized)
                self.assertEqual(thinking_mode_from_value(value), mode)

        with self.assertRaises(ValueError):
            normalize_think_mode("auto")

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
        generate_env_config.think_end_tag = "</think>\\n\\n"
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

    def test_add_thinking_params_does_not_check_tokenizer_length(self):
        class Tokenizer:
            def __len__(self):
                raise AssertionError("tokenizer length should not be checked")

            def encode(self, text, add_special_tokens=False):
                return [123]

        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>"

        generate_config = GenerateConfig()
        generate_config.add_thinking_params(Tokenizer(), generate_env_config)

        self.assertEqual(generate_config.end_think_token_ids, [123])

    def test_tool_choice_dict_is_validated_at_request_parse(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object"},
                },
            }
        ]

        request = ChatCompletionRequest(
            messages=[],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        self.assertEqual(request.tool_choice["function"]["name"], "get_weather")

        cases = [
            (
                {"type": "bad"},
                tools,
                "tool_choice.type must be 'function'",
            ),
            (
                {"type": "function"},
                tools,
                "tool_choice.function must be an object",
            ),
            (
                {"type": "function", "function": {}},
                tools,
                "tool_choice.function.name must be a non-empty string",
            ),
            (
                {"type": "function", "function": {"name": "missing"}},
                tools,
                "tool_choice function .* is not in tools",
            ),
            (
                {"type": "function", "function": {"name": "get_weather"}},
                None,
                "tool_choice function requires non-empty tools",
            ),
            (
                "required",
                None,
                "tool_choice='required' requires non-empty tools",
            ),
        ]
        for tool_choice, case_tools, message in cases:
            with self.subTest(tool_choice=tool_choice):
                with self.assertRaisesRegex(ValueError, message):
                    ChatCompletionRequest(
                        messages=[],
                        tools=case_tools,
                        tool_choice=tool_choice,
                    )


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

    def _extract_openai_generation_config(
        self,
        request: ChatCompletionRequest,
        generate_env_config: Optional[GenerateEnvConfig] = None,
    ):
        model_config = ModelConfig()
        model_config.generate_env_config = generate_env_config or GenerateEnvConfig()
        model_config.render_config = RenderConfig()
        model_config.special_tokens = SpecialTokens()
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
        return openai_endpoint._extract_generation_config(request)

    def _assert_reasoning_envelope_wraps_json_object(self, config: GenerateConfig):
        """in_think_mode moves the final constraint inside the reasoning tag."""
        self.assertIsNone(config.json_schema)
        structural_tag = config.structural_tag
        self.assertEqual(structural_tag["type"], "structural_tag")
        elements = structural_tag["format"]["elements"]
        self.assertEqual(elements[0]["type"], "tag")
        self.assertEqual(elements[0]["end"], "</think>\n\n")
        self.assertEqual(elements[1]["type"], "json_schema")
        self.assertEqual(
            elements[1]["json_schema"],
            {"anyOf": [{"type": "object"}, {"type": "array"}]},
        )

    def _generate_config_with_stop_word(
        self,
        model_stop_word_str: Optional[List[str]] = None,
        model_stop_word_list: Optional[List[str]] = None,
        env_stop_word_str: Optional[str] = None,
        env_stop_word_list: Optional[str] = None,
        req_stop: Optional[Union[str, List[str]]] = None,
        req_config_stop_word_str: Optional[List[str]] = None,
        req_config_stop_word_list: Optional[List[List[int]]] = None,
        response_format: Optional[Union[str, Dict[str, Any]]] = None,
        json_format: Optional[bool] = None,
        enable_thinking: Optional[bool] = False,
        thinking_budget: Optional[int] = None,
        input_ids: Optional[List[int]] = None,
        thinking_mode: Optional[ThinkingMode] = None,
        env_think_mode: Optional[Union[str, int]] = None,
    ):
        special_tokens = SpecialTokens()
        if model_stop_word_str is not None:
            special_tokens.stop_words_str_list = model_stop_word_str
        if model_stop_word_list is not None:
            special_tokens.stop_words_id_list = model_stop_word_list

        generate_env_config = GenerateEnvConfig()
        if env_think_mode is not None:
            generate_env_config.think_mode = env_think_mode
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

        request = ChatCompletionRequest(
            messages=[],
            response_format=response_format,
            json_format=json_format,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        if thinking_mode is not None:
            if request.extra_configs is None:
                request.extra_configs = GenerateConfig()
            request.extra_configs.thinking_mode = thinking_mode

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

        return openai_endpoint._extract_generation_config(request, input_ids=input_ids)

    def test_response_format_is_finalized_before_generation(self):
        config = self._generate_config_with_stop_word(
            response_format='{"type":"json_object"}'
        )
        self.assertIsNone(config.response_format)
        json_object_schema = {"anyOf": [{"type": "object"}, {"type": "array"}]}
        self.assertEqual(config.json_schema, json_object_schema)

        legacy_config = self._generate_config_with_stop_word(json_format=True)
        self.assertIsNone(legacy_config.response_format)
        self.assertEqual(legacy_config.json_schema, json_object_schema)

    def test_extra_configs_may_not_carry_structured_output(self):
        request = ChatCompletionRequest(
            messages=[],
            extra_configs=GenerateConfig(
                response_format={
                    "type": "json_schema",
                    "json_schema": {"schema": {"type": "object"}},
                }
            ),
        )
        with self.assertRaises(FtRuntimeException) as ctx:
            self._extract_openai_generation_config(request)
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )
        self.assertIn("top-level response_format", ctx.exception.message)

        request = ChatCompletionRequest(
            messages=[],
            extra_configs=GenerateConfig(json_schema='{"type": "object"}'),
        )
        with self.assertRaises(FtRuntimeException) as ctx:
            self._extract_openai_generation_config(request)
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )

    def test_extra_configs_max_thinking_tokens_zero_disables_thinking(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        request = ChatCompletionRequest(
            messages=[],
            extra_configs=GenerateConfig(max_thinking_tokens=0),
            enable_thinking=True,
        )

        self.assertTrue(request.disable_thinking())
        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.max_thinking_tokens, 0)
        self.assertEqual(config.end_think_token_ids, [102])

    def test_disable_thinking_zeroes_backend_thinking_budget(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        request = ChatCompletionRequest(
            messages=[],
            extra_configs=GenerateConfig(max_thinking_tokens=16),
            enable_thinking=False,
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.max_thinking_tokens, 0)
        self.assertEqual(config.end_think_token_ids, [102])

    def test_openai_max_completion_tokens_thinking_budget_keeps_backend_limit(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        request = ChatCompletionRequest(
            messages=[],
            max_tokens=200,
            max_completion_tokens=100,
            thinking_budget=10,
            enable_thinking=True,
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(config.max_thinking_tokens, 10)
        self.assertTrue(config.in_think_mode)

    def test_openai_max_completion_tokens_respects_max_tokens_total_cap(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        request = ChatCompletionRequest(
            messages=[],
            max_tokens=105,
            max_completion_tokens=100,
            thinking_budget=10,
            enable_thinking=True,
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(config.max_thinking_tokens, 10)

    def test_openai_max_completion_tokens_does_not_add_default_thinking_budget(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        request = ChatCompletionRequest(
            messages=[],
            max_completion_tokens=100,
            enable_thinking=True,
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertEqual(config.max_new_tokens, 100)
        self.assertEqual(config.max_thinking_tokens, 32000)
        self.assertTrue(config.in_think_mode)

    def test_openai_max_completion_tokens_non_positive_is_unset(self):
        request = ChatCompletionRequest(
            messages=[],
            max_tokens=64,
            max_completion_tokens=0,
        )
        config = self._extract_openai_generation_config(request)
        self.assertEqual(config.max_new_tokens, 64)

        request = ChatCompletionRequest(messages=[], max_completion_tokens=-1)
        config = self._extract_openai_generation_config(request)
        self.assertEqual(config.max_new_tokens, 32000)

    def test_request_level_thinking_adds_think_end_tokens_when_env_mode_off(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 0
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>\n\n"
        request = ChatCompletionRequest(
            messages=[], thinking_budget=10, enable_thinking=True
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertTrue(config.in_think_mode)
        self.assertEqual(config.max_thinking_tokens, 10)
        self.assertEqual(
            config.end_think_token_ids,
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False),
        )

    def test_top_level_enable_thinking_enables_backend_for_json_object(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 0
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>\n\n"
        request = ChatCompletionRequest(
            messages=[],
            response_format={"type": "json_object"},
            enable_thinking=True,
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertTrue(config.in_think_mode)
        self._assert_reasoning_envelope_wraps_json_object(config)
        self.assertEqual(
            config.end_think_token_ids,
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False),
        )

    def test_chat_template_enable_thinking_enables_backend_for_json_object(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 0
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>\n\n"
        request = ChatCompletionRequest(
            messages=[],
            response_format={"type": "json_object"},
            chat_template_kwargs={"enable_thinking": True},
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertTrue(config.in_think_mode)
        self._assert_reasoning_envelope_wraps_json_object(config)
        self.assertEqual(
            config.end_think_token_ids,
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False),
        )

    def test_extra_config_chat_template_enable_thinking_enables_backend(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 0
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>\n\n"
        request = ChatCompletionRequest(
            messages=[],
            response_format={"type": "json_object"},
            extra_configs=GenerateConfig(
                chat_template_kwargs={"enable_thinking": True}
            ),
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertTrue(config.in_think_mode)
        self._assert_reasoning_envelope_wraps_json_object(config)
        self.assertEqual(
            config.end_think_token_ids,
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False),
        )

    def test_renderer_chat_constraints_are_applied_to_generate_config(self):
        class Renderer:
            def apply_chat_completion_constraints(self, request, config):
                config.structural_tag = '{"type":"test"}'

        config = GenerateConfig()

        OpenaiEndpoint._apply_renderer_chat_constraints(
            Renderer(),
            ChatCompletionRequest(messages=[]),
            config,
        )

        self.assertEqual(config.structural_tag, '{"type":"test"}')

    def test_default_renderer_chat_constraints_allow_non_forcing_tool_choice(self):
        renderer = CustomChatRenderer.__new__(CustomChatRenderer)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object"},
                },
            }
        ]

        for tool_choice in (None, "auto", "none"):
            with self.subTest(tool_choice=tool_choice):
                OpenaiEndpoint._apply_renderer_chat_constraints(
                    renderer,
                    ChatCompletionRequest(
                        messages=[],
                        tools=tools,
                        tool_choice=tool_choice,
                    ),
                    GenerateConfig(),
                )

        with self.assertRaisesRegex(Exception, "is not supported"):
            OpenaiEndpoint._apply_renderer_chat_constraints(
                renderer,
                ChatCompletionRequest(
                    messages=[],
                    tools=tools,
                    tool_choice="required",
                ),
                GenerateConfig(),
            )

    def test_text_response_format_is_finalized_before_generation(self):
        config = self._generate_config_with_stop_word(response_format={"type": "text"})

        self.assertIsNone(config.response_format)
        self.assertIsNone(config.json_schema)
        self.assertIsNone(config.regex)
        self.assertIsNone(config.ebnf)
        self.assertIsNone(config.structural_tag)

    def test_unresolved_openai_thinking_uses_disabled_fallback_even_with_budget(self):
        request = ChatCompletionRequest(messages=[], thinking_budget=32000)
        self.assertEqual(request.resolve_thinking_mode(), ThinkingMode.DISABLED)
        self.assertIsNone(request.get_enable_thinking())

    def test_unspecified_openai_thinking_uses_disabled_final_constraint(self):
        config = self._generate_config_with_stop_word(
            response_format={"type": "json_object"},
            enable_thinking=None,
        )

        self.assertEqual(config.thinking_mode, ThinkingMode.DISABLED)
        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.max_thinking_tokens, 0)
        self.assertIsNone(config.structural_tag)
        self.assertEqual(config.begin_think_token_ids, [])
        self.assertEqual(config.end_think_token_ids, [])
        self.assertEqual(
            config.json_schema,
            {"anyOf": [{"type": "object"}, {"type": "array"}]},
        )

    def test_unspecified_openai_thinking_inherits_env_mode(self):
        cases = {
            "disabled": (ThinkingMode.DISABLED, None),
            "adaptive": (ThinkingMode.ADAPTIVE, "or"),
            "enabled": (ThinkingMode.ENABLED, "sequence"),
            "0": (ThinkingMode.DISABLED, None),
            "1": (ThinkingMode.ENABLED, "sequence"),
        }
        for env_mode, (expected, expected_format_type) in cases.items():
            with self.subTest(env_mode=env_mode):
                config = self._generate_config_with_stop_word(
                    enable_thinking=None,
                    env_think_mode=env_mode,
                )
                self.assertEqual(config.thinking_mode, expected)
                self.assertEqual(config.in_think_mode, expected == ThinkingMode.ENABLED)
                if expected_format_type is None:
                    self.assertIsNone(config.structural_tag)
                    self.assertEqual(config.max_thinking_tokens, 0)
                else:
                    self.assertEqual(
                        config.structural_tag["format"]["type"],
                        expected_format_type,
                    )
                    self.assertEqual(config.max_thinking_tokens, 32000)

    def test_openai_positive_budget_inherits_env_mode(self):
        cases = {
            "disabled": ThinkingMode.DISABLED,
            "adaptive": ThinkingMode.ADAPTIVE,
            "enabled": ThinkingMode.ENABLED,
        }
        for env_mode, expected in cases.items():
            with self.subTest(env_mode=env_mode):
                config = self._generate_config_with_stop_word(
                    enable_thinking=None,
                    thinking_budget=32000,
                    env_think_mode=env_mode,
                )

                self.assertEqual(config.thinking_mode, expected)
                self.assertEqual(
                    config.max_thinking_tokens,
                    0 if expected == ThinkingMode.DISABLED else 32000,
                )

    def test_explicit_openai_thinking_overrides_env_mode(self):
        disabled = self._generate_config_with_stop_word(
            enable_thinking=False,
            env_think_mode="enabled",
        )
        enabled = self._generate_config_with_stop_word(
            enable_thinking=True,
            env_think_mode="disabled",
        )

        self.assertEqual(disabled.thinking_mode, ThinkingMode.DISABLED)
        self.assertEqual(enabled.thinking_mode, ThinkingMode.ENABLED)

    def test_adaptive_prompt_with_think_start_continues_as_enabled(self):
        begin_ids = self.tokenizer.encode("<think>\n", add_special_tokens=False)
        config = self._generate_config_with_stop_word(
            enable_thinking=None,
            input_ids=[1, 2, *begin_ids],
            thinking_mode=ThinkingMode.ADAPTIVE,
        )

        self.assertEqual(config.thinking_mode, ThinkingMode.ENABLED)
        self.assertTrue(config.in_think_mode)
        reasoning_tag = config.structural_tag["format"]["elements"][0]
        self.assertEqual(reasoning_tag["begin"], "")

    def test_explicit_openai_thinking_boolean_selects_fixed_mode(self):
        self.assertEqual(
            ChatCompletionRequest(
                messages=[], enable_thinking=True
            ).resolve_thinking_mode(),
            ThinkingMode.ENABLED,
        )
        self.assertEqual(
            ChatCompletionRequest(
                messages=[], enable_thinking=False
            ).resolve_thinking_mode(),
            ThinkingMode.DISABLED,
        )
        request = ChatCompletionRequest(
            messages=[],
            enable_thinking=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertFalse(request.get_enable_thinking())
        self.assertEqual(request.resolve_thinking_mode(), ThinkingMode.DISABLED)
        adaptive_request = ChatCompletionRequest(
            messages=[],
            extra_configs=GenerateConfig(
                thinking_mode=ThinkingMode.ADAPTIVE,
                chat_template_kwargs={"enable_thinking": True},
            ),
        )
        self.assertEqual(
            adaptive_request.resolve_thinking_mode(), ThinkingMode.ADAPTIVE
        )
        self.assertTrue(adaptive_request.get_enable_thinking())

    def test_enabled_openai_thinking_keeps_legacy_empty_grammar_begin(self):
        config = self._generate_config_with_stop_word(enable_thinking=True)

        self.assertEqual(config.thinking_mode, ThinkingMode.ENABLED)
        self.assertTrue(config.in_think_mode)
        reasoning_tag = config.structural_tag["format"]["elements"][0]
        self.assertEqual(reasoning_tag["begin"], "")
        self.assertEqual(config.begin_think_token_ids, [])

    def test_invalid_chat_template_thinking_mode_is_rejected(self):
        for value in ("auto", "Adaptive", True, 1, None):
            with self.subTest(value=value):
                request = ChatCompletionRequest(
                    messages=[],
                    chat_template_kwargs={"thinking_mode": value},
                )

                with self.assertRaisesRegex(
                    ValueError,
                    "chat_template_kwargs.thinking_mode must be one of",
                ):
                    request.resolve_thinking_mode()

    def test_adaptive_thinking_initializes_boundaries_without_forcing_think(self):
        env = GenerateEnvConfig()
        env.think_start_tag = "<think>"
        env.think_end_token_id = 102
        config = GenerateConfig(
            thinking_mode=ThinkingMode.ADAPTIVE,
            max_thinking_tokens=32000,
        )

        config.add_thinking_params(self.tokenizer, env)

        self.assertFalse(config.in_think_mode)
        self.assertEqual(
            config.begin_think_token_ids,
            self.tokenizer.encode("<think>", add_special_tokens=False),
        )
        self.assertEqual(config.end_think_token_ids, [102])
        branches = config.structural_tag["format"]["elements"]
        self.assertEqual(config.structural_tag["format"]["type"], "or")
        self.assertEqual(branches[0]["elements"][0]["begin"], "<think>")
        self.assertEqual(branches[0]["elements"][0]["content"]["max_tokens"], 32000)
        self.assertEqual(branches[1]["type"], "any_text")
        self.assertEqual(branches[1]["excludes"], ["<think>", "</think>\n\n"])

    def test_disabled_thinking_keeps_configured_end_token_metadata(self):
        env = GenerateEnvConfig()
        env.think_mode = "disabled"
        env.think_end_token_id = 102
        config = GenerateConfig()

        config.add_thinking_params(self.tokenizer, env)

        self.assertEqual(config.thinking_mode, ThinkingMode.DISABLED)
        self.assertFalse(config.in_think_mode)
        self.assertEqual(config.end_think_token_ids, [102])
        self.assertIsNone(config.structural_tag)

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

        # Request-level stop words are natural-language phrases: each yields
        # two token sequences (bare + leading-space variant) because byte-level
        # BPE merges a leading space into the first token. Model/env stop words
        # are template special tokens resolved via the renderer and keep a
        # single sequence; the asymmetry below is intentional.
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
                [2958, 2936, 3409],  # "req stop word"
                [4232, 2936, 3409],  # leading-space variant
                [41963, 4232, 2936, 3409],  # "another req stop word"
                [2441, 4232, 2936, 3409],  # leading-space variant
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
                [2958, 2193, 2936, 3409],  # "req config stop word"
                [4232, 2193, 2936, 3409],  # leading-space variant
                [41963, 2193, 4232, 2936, 3409],  # "another config req stop word"
                [2441, 2193, 4232, 2936, 3409],  # leading-space variant
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
                [4232, 2936, 3409],
                [41963, 4232, 2936, 3409],
                [2441, 4232, 2936, 3409],  # req_stop (+ leading-space variants)
                [2958, 2193, 2936, 3409],
                [4232, 2193, 2936, 3409],
                [41963, 2193, 4232, 2936, 3409],
                [2441, 2193, 4232, 2936, 3409],  # req_config_stop_word_str (+ variants)
                [2958, 2193, 2936, 1140],
                [
                    41963,
                    2193,
                    4232,
                    2936,
                    1140,
                ],  # req_config_stop_word_list (ids as-is)
                [21912, 2936, 1140],
                [21912, 2936, 3409],
                [22737, 2936, 3409],  # duplicate stop word (+ leading-space variant)
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

    def test_request_stop_word_edge_cases(self):
        default_stop_ids = [[151643], [151645]]

        # Empty stop word must not produce any entry.
        config = self._generate_config_with_stop_word(req_stop=[""])
        self.assertEqual(sorted(config.stop_words_list), sorted(default_stop_ids))

        # Whitespace-only stop word produces exactly one entry: the guard must
        # not add a second (double-space) variant.
        space_ids = self.tokenizer.encode(" ", add_special_tokens=False)
        config = self._generate_config_with_stop_word(req_stop=[" "])
        self.assertEqual(
            sorted(config.stop_words_list),
            sorted(default_stop_ids + [space_ids]),
        )

        # Already-space-prefixed stop word produces exactly one entry too.
        word = " leading space word"
        word_ids = self.tokenizer.encode(word, add_special_tokens=False)
        config = self._generate_config_with_stop_word(req_stop=[word])
        self.assertEqual(
            sorted(config.stop_words_list),
            sorted(default_stop_ids + [word_ids]),
        )

    def test_request_stop_str_form(self):
        # The OpenAI contract allows request.stop to be a bare string.
        config = self._generate_config_with_stop_word(req_stop="req stop word")
        self.assertIn("req stop word", config.stop_words_str)
        self.assertEqual(
            sorted(config.stop_words_list),
            sorted([[151643], [151645], [2958, 2936, 3409], [4232, 2936, 3409]]),
        )

    def test_request_stop_not_mutated(self):
        # _extract_generation_config must not mutate the caller's request.stop
        # in place when folding extra_configs.stop_words_str into it.
        req_stop = ["req stop word"]
        self._generate_config_with_stop_word(
            req_stop=req_stop,
            req_config_stop_word_str=["req config stop word"],
        )
        self.assertEqual(req_stop, ["req stop word"])

    def test_tokenize_request_stop_words_fallback(self):
        # Tokenizers whose encode() does not accept add_special_tokens fall
        # back to a bare encode() call and log a warning.
        class _LegacyTokenizer:
            def encode(self, text):
                return [ord(c) for c in text]

        endpoint = SimpleNamespace(tokenizer=_LegacyTokenizer())
        with self.assertLogs(level="WARNING"):
            ids = OpenaiEndpoint._tokenize_request_stop_words(endpoint, ["ab"])
        self.assertEqual(ids, [[97, 98], [32, 97, 98]])


class GrammarMultiSequenceConfigTest(TestCase):
    """Grammar and thinking requests reject multi-sequence/beam configuration."""

    def _apply(self, **fields):
        cfg = GenerateConfig(**fields)
        cfg.finalize_response_format()
        return cfg

    def _assert_rejected(self, exception_type: ExceptionType, **fields):
        cfg = GenerateConfig(**fields)
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.finalize_response_format()
        self.assertEqual(ctx.exception.exception_type, exception_type)

    def test_grammar_field_plus_multi_sequence_is_rejected(self):
        cases = [
            {"json_schema": '{"type": "object"}', "num_beams": 4},
            {
                "response_format": {"type": "regex", "pattern": r"\d+"},
                "variable_num_beams": [1, 3],
            },
            {
                "response_format": {"type": "json_object"},
                "num_return_sequences": 2,
            },
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                self._assert_rejected(
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    **fields,
                )

    def test_openai_wire_response_format_plus_multi_sequence_is_rejected(self):
        request = ChatCompletionRequest(
            messages=[], response_format={"type": "json_object"}
        )
        self.assertIsInstance(request.response_format, OpenAIResponseFormat)

        for fields in ({"num_beams": 4}, {"num_return_sequences": 2}):
            with self.subTest(fields=fields):
                config = GenerateConfig(**fields)
                # Mirror OpenaiEndpoint._extract_generation_config(), which
                # assigns the already parsed wire model directly.
                config.response_format = request.response_format
                with self.assertRaises(FtRuntimeException) as ctx:
                    config.finalize_response_format()
                self.assertEqual(
                    ctx.exception.exception_type,
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                )
                self.assertIn(
                    "grammar-constrained decoding does not support",
                    ctx.exception.message,
                )

    def test_thinking_plus_multi_sequence_is_rejected_before_engine(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        cases = [
            {"num_beams": 4},
            {"variable_num_beams": [1, 3]},
            {"num_return_sequences": 2},
        ]

        for fields in cases:
            with self.subTest(fields=fields):
                config = GenerateConfig(**fields)
                with self.assertRaises(FtRuntimeException) as ctx:
                    config.add_thinking_params(
                        tokenizer=None,
                        generate_env_config=generate_env_config,
                    )
                self.assertEqual(
                    ctx.exception.exception_type,
                    ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                )
                self.assertIn(
                    "thinking mode does not support beam search or "
                    "num_return_sequences > 1",
                    ctx.exception.message,
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
    """response_format is compiled to one engine grammar constraint."""

    def test_json_schema_wire_alias_round_trip(self):
        payload = {
            "type": "json_schema",
            "json_schema": {"name": "item", "schema": {"type": "object"}},
        }

        response_format = ResponseFormat.model_validate(payload)

        self.assertIsNotNone(response_format)
        self.assertEqual(response_format.model_dump(exclude_none=True), payload)
        self.assertEqual(response_format.json_schema.schema_, {"type": "object"})

    def _validate(
        self,
        cfg: GenerateConfig,
        reasoning_format: Optional[ReasoningFormat] = None,
    ):
        cfg.finalize_response_format(reasoning_format=reasoning_format)

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
        cfg.in_think_mode = True
        cfg.end_think_token_ids = (
            [think_end_token_id] if think_end_token_id != -1 else []
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
                {"type": "string"},
            ),
            (
                GenerateConfig(response_format={"type": "json_object"}),
                "json_schema",
                {"anyOf": [{"type": "object"}, {"type": "array"}]},
            ),
            (
                GenerateConfig(response_format={"type": "regex", "pattern": r"\d+"}),
                "regex",
                r"\d+",
            ),
        ]
        for cfg, field, expected in cases:
            with self.subTest(field=field, expected=expected):
                self._validate(cfg)
                self.assertIsNone(cfg.response_format)
                self.assertEqual(cfg.model_dump()[field], expected)

    def test_openai_response_format_model_is_canonicalized(self):
        cfg = GenerateConfig()
        cfg.response_format = OpenAIResponseFormat(
            type="json_schema",
            json_schema={
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        )

        self._validate(cfg)

        self.assertIsNone(cfg.response_format)
        self.assertEqual(
            cfg.json_schema,
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

    def test_response_format_conflicts_with_typed_grammar_field(self):
        cfg = GenerateConfig(
            response_format={"type": "regex", "pattern": r"[a-z]+"},
            json_schema='{"type": "object"}',
        )
        with self.assertRaises(FtRuntimeException) as ctx:
            self._validate(cfg)
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )

    def test_direct_json_schema_is_kept_structured(self):
        for value in ({"type": "object"}, '{"type": "object"}'):
            with self.subTest(value=value):
                cfg = GenerateConfig(json_schema=value)
                self._validate(cfg)
                self.assertEqual(cfg.json_schema, {"type": "object"})

    def test_reasoning_json_schema_wrapped_as_structural_tag(self):
        cfg = GenerateConfig(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "examples": [{"type": "any_text", "max_tokens": 1}],
                    }
                },
            },
            max_thinking_tokens=64,
        )
        reasoning_format = self._enable_thinking(cfg)
        self._validate(cfg, reasoning_format=reasoning_format)

        self.assertIsNone(cfg.response_format)
        self.assertIsNone(cfg.json_schema)
        self.assertIsNone(cfg.regex)
        self.assertIsNone(cfg.ebnf)

        structural_tag = cfg.structural_tag
        self.assertEqual(structural_tag["type"], "structural_tag")
        elements = structural_tag["format"]["elements"]
        self.assertEqual(elements[0]["type"], "tag")
        self.assertEqual(elements[0]["begin"], "")
        self.assertEqual(elements[0]["end"], "</think>\n\n")
        self.assertEqual(elements[0]["content"], {"type": "any_text", "max_tokens": 64})
        self.assertEqual(elements[1]["type"], "json_schema")
        self.assertEqual(
            elements[1]["json_schema"],
            {
                "type": "object",
                "examples": [{"type": "any_text", "max_tokens": 1}],
            },
        )
        self.assertEqual(elements[1]["style"], "json")

    def test_reasoning_preparation_is_idempotent(self):
        cfg = GenerateConfig(
            response_format={"type": "json_object"},
            max_thinking_tokens=64,
        )
        reasoning_format = self._enable_thinking(cfg)

        first_constraint = cfg.finalize_response_format(
            reasoning_format=reasoning_format
        )
        first_envelope = copy.deepcopy(cfg.structural_tag)
        second_constraint = cfg.finalize_response_format(
            reasoning_format=reasoning_format
        )

        self.assertEqual(second_constraint, first_constraint)
        self.assertEqual(cfg.structural_tag, first_envelope)

    def test_explicit_final_constraint_can_be_restored_for_phase2(self):
        cfg = GenerateConfig(
            response_format={"type": "regex", "pattern": r"[a-z]+"},
            max_thinking_tokens=64,
        )
        reasoning_format = self._enable_thinking(cfg)
        final_constraint = cfg.finalize_response_format(
            reasoning_format=reasoning_format
        )

        cfg.in_think_mode = False
        restore_final_constraint(cfg, final_constraint)

        self.assertIsNone(cfg.structural_tag)
        self.assertEqual(cfg.regex, r"[a-z]+")
        validate_engine_ready(cfg)

    def test_reasoning_can_generate_non_empty_begin_tag(self):
        cfg = GenerateConfig(
            response_format={"type": "json_object"},
            in_think_mode=True,
            max_thinking_tokens=16,
        )
        reasoning_format = ReasoningFormat(
            tag_begin="<think>\n",
            tag_end="</think>\n\n",
        )
        self._validate(cfg, reasoning_format=reasoning_format)

        structural_tag = cfg.structural_tag
        reasoning_tag = structural_tag["format"]["elements"][0]
        self.assertEqual(reasoning_tag["begin"], "<think>\n")
        self.assertEqual(reasoning_tag["end"], "</think>\n\n")

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

        structural_tag = cfg.structural_tag
        elements = structural_tag["format"]["elements"]
        self.assertEqual(elements[0]["end"], {"type": "token", "token": 123})

    def test_reasoning_without_grammar_wraps_any_text_structural_tag(self):
        cfg = GenerateConfig(response_format={"type": "text"})
        reasoning_format = self._enable_thinking(cfg)
        self._validate(cfg, reasoning_format=reasoning_format)

        self.assertIsNone(cfg.json_schema)
        structural_tag = cfg.structural_tag
        elements = structural_tag["format"]["elements"]
        self.assertEqual(elements[0]["type"], "tag")
        self.assertEqual(elements[1], {"type": "any_text"})

    def test_adaptive_reasoning_uses_think_or_final_branches(self):
        cfg = GenerateConfig(
            response_format={"type": "json_object"},
            thinking_mode=ThinkingMode.ADAPTIVE,
            max_thinking_tokens=17,
            begin_think_token_ids=[10],
            end_think_token_ids=[11],
        )
        reasoning_format = ReasoningFormat(
            tag_begin="<think>",
            tag_end="</think>",
        )

        self._validate(cfg, reasoning_format=reasoning_format)

        adaptive = cfg.structural_tag["format"]
        self.assertEqual(adaptive["type"], "or")
        think_branch, no_think_branch = adaptive["elements"]
        reasoning_tag = think_branch["elements"][0]
        self.assertEqual(reasoning_tag["begin"], "<think>")
        self.assertEqual(reasoning_tag["end"], "</think>")
        self.assertEqual(reasoning_tag["content"]["max_tokens"], 17)
        self.assertEqual(think_branch["elements"][1]["type"], "json_schema")
        self.assertEqual(no_think_branch["type"], "json_schema")

    def test_adaptive_text_branch_excludes_think_boundaries(self):
        cfg = GenerateConfig(
            thinking_mode=ThinkingMode.ADAPTIVE,
            max_thinking_tokens=9,
            begin_think_token_ids=[10],
            end_think_token_ids=[11],
        )
        self._validate(
            cfg,
            reasoning_format=ReasoningFormat(
                tag_begin="<think>",
                tag_end="</think>",
            ),
        )

        no_think_branch = cfg.structural_tag["format"]["elements"][1]
        self.assertEqual(no_think_branch["type"], "any_text")
        self.assertEqual(no_think_branch["excludes"], ["<think>", "</think>"])

    def test_adaptive_reasoning_requires_begin_tag(self):
        cfg = GenerateConfig(
            thinking_mode=ThinkingMode.ADAPTIVE,
            max_thinking_tokens=9,
            begin_think_token_ids=[10],
            end_think_token_ids=[11],
        )
        with self.assertRaises(FtRuntimeException) as ctx:
            self._validate(
                cfg,
                reasoning_format=ReasoningFormat(
                    tag_begin="",
                    tag_end="</think>",
                ),
            )
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )

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

    def test_reasoning_does_not_accept_caller_built_reasoning_envelope(self):
        cfg = GenerateConfig(
            structural_tag={
                "type": "structural_tag",
                "format": {
                    "type": "sequence",
                    "elements": [
                        {
                            "type": "tag",
                            "begin": "",
                            "content": {"type": "any_text", "max_tokens": 3},
                            "end": "</think>\n\n",
                        },
                        {
                            "type": "json_schema",
                            "json_schema": {"type": "object"},
                        },
                    ],
                },
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

        with self.assertRaises(FtRuntimeException) as ctx:
            GenerateConfig(
                structural_tag=(
                    '{"type":"structural_tag","format":' + nested_format + "}"
                )
            )
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )


class RawUpdateAndGrammarConflictTest(TestCase):
    """Raw config updates cannot bypass response_format normalization."""

    def test_update_and_pop_accepts_string_envelope(self):
        cfg = GenerateConfig()
        remain = cfg.update_and_pop(
            {"response_format": '{"type":"regex","pattern":"\\\\d+"}', "stranger": 1}
        )
        self.assertEqual(remain, {"stranger": 1})
        self.assertIsInstance(cfg.response_format, ResponseFormat)
        cfg.finalize_response_format()
        self.assertEqual(cfg.regex, r"\d+")

    def test_service_terminate_flag_is_not_dumped_as_user_config(self):
        cfg = GenerateConfig(
            grammar_terminate_without_stop_token=True,
        )
        self.assertNotIn(
            "grammar_terminate_without_stop_token", GenerateConfig.model_fields
        )
        self.assertNotIn("grammar_terminate_without_stop_token", cfg.model_dump())

    def test_update_ignores_service_only_terminate_flag(self):
        cfg = GenerateConfig()
        cfg.update(
            {
                "grammar_terminate_without_stop_token": True,
                "max_new_tokens": 42,
            }
        )
        self.assertEqual(cfg.max_new_tokens, 42)
        self.assertIn("max_new_tokens", cfg.model_fields_set)

    def test_update_and_pop_preserves_service_only_terminate_flag(self):
        cfg = GenerateConfig()
        remain = cfg.update_and_pop(
            {
                "grammar_terminate_without_stop_token": True,
                "max_new_tokens": 42,
            }
        )
        self.assertEqual(remain, {"grammar_terminate_without_stop_token": True})
        self.assertEqual(cfg.max_new_tokens, 42)
        self.assertIn("max_new_tokens", cfg.model_fields_set)

    def test_update_rejects_malformed_envelope(self):
        cfg = GenerateConfig()
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.update({"response_format": {"type": "json_schema"}})
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )
        self.assertIn("response_format", str(ctx.exception))
        self.assertIn("requires json_schema.schema", str(ctx.exception))
        self.assertIsNone(cfg.response_format)

    def test_update_and_pop_rejects_invalid_json_envelope(self):
        cfg = GenerateConfig()
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.update_and_pop(
                {"response_format": '{"type":"json_object"', "stranger": 1}
            )
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )
        self.assertIn("response_format", str(ctx.exception))
        self.assertIn("Expecting", str(ctx.exception))
        self.assertIsNone(cfg.response_format)

    def test_multiple_typed_grammar_fields_rejected(self):
        cfg = GenerateConfig(json_schema='{"type": "object"}', regex=r"\d+")
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.finalize_response_format()
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSUPPORTED_OPERATION
        )

    def test_update_paths_keep_json_grammar_fields_structured(self):
        cfg = GenerateConfig()

        cfg.update({"json_schema": '{"type":"object"}'})
        remain = cfg.update_and_pop(
            {
                "structural_tag": '{"format":{"type":"regex","pattern":"a"}}',
                "stranger": 1,
            }
        )

        self.assertEqual(cfg.json_schema, {"type": "object"})
        self.assertEqual(
            cfg.structural_tag,
            {"format": {"type": "regex", "pattern": "a"}},
        )
        self.assertEqual(remain, {"stranger": 1})


if __name__ == "__main__":
    main()
