import json
import os
from typing import Any, List, Optional
from unittest import TestCase, main

from transformers import AutoTokenizer

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest, GenerateConfig
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

    def test_response_format_to_grammar_config(self):
        request = ChatCompletionRequest(
            messages=[],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            },
        )
        config = GenerateConfig()
        OpenaiEndpoint._apply_response_format(request.response_format, config)
        self.assertEqual(json.loads(config.json_schema), {"type": "object"})

        request = ChatCompletionRequest(messages=[], json_format=True)
        self.assertTrue(request.json_format)

        config = GenerateConfig()
        OpenaiEndpoint._apply_json_format(config)
        self.assertEqual(json.loads(config.json_schema), {"type": "object"})

        config = GenerateConfig(json_schema={"type": "object"})
        config.validate()
        self.assertEqual(config.json_schema, '{"type":"object"}')

        config = GenerateConfig(json_format=True)
        config.validate()
        self.assertEqual(config.json_schema, '{"type":"object"}')
        self.assertFalse(config.force_disable_sp_run)

        with self.assertRaisesRegex(Exception, "beam search"):
            GenerateConfig(num_beams=2, regex="[0-9]+").validate()

    def test_openai_response_format_structural_tag_is_not_public_api(self):
        with self.assertRaises(Exception):
            ChatCompletionRequest(
                messages=[],
                response_format={
                    "type": "structural_tag",
                    "format": {"type": "tag"},
                },
            )

    def test_response_format_text_clears_stale_extra_config_response_format(self):
        config = GenerateConfig(
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "object"}},
            }
        )
        self.assertTrue(config._has_grammar_constraint())

        request = ChatCompletionRequest(
            messages=[],
            response_format={"type": "text"},
        )
        OpenaiEndpoint._apply_response_format(request.response_format, config)

        self.assertIsNone(config.response_format)
        self.assertFalse(config._has_grammar_constraint())

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

    def test_thinking_budget_default_and_invalid_values(self):
        self.assertEqual(GenerateConfig().max_thinking_tokens, 131072)
        self.assertEqual(GenerateConfig().max_completion_tokens, 0)

        for max_thinking_tokens in [0, -1]:
            generate_config = GenerateConfig(
                in_think_mode=True,
                max_thinking_tokens=max_thinking_tokens,
                end_think_token_ids=[102],
            )
            with self.assertRaises(Exception):
                generate_config.validate()

    def test_thinking_requires_completion_budget_greater_than_thinking_budget(self):
        GenerateConfig(
            in_think_mode=True,
            max_new_tokens=8,
            max_thinking_tokens=5,
            max_completion_tokens=6,
            end_think_token_ids=[102],
        ).validate()

        for max_completion_tokens in [0, 5]:
            generate_config = GenerateConfig(
                in_think_mode=True,
                max_new_tokens=8,
                max_thinking_tokens=5,
                max_completion_tokens=max_completion_tokens,
                end_think_token_ids=[102],
            )
            with self.assertRaises(Exception):
                generate_config.validate()

        GenerateConfig(
            max_new_tokens=8,
            max_thinking_tokens=5,
            max_completion_tokens=5,
            in_think_mode=False,
        ).validate()
        GenerateConfig(max_new_tokens=8, max_completion_tokens=0).validate()

    def test_add_thinking_params_with_think_token(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>"
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
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>\n\n"
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

    def test_extra_configs_max_thinking_tokens_zero_is_invalid(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        request = ChatCompletionRequest(
            messages=[],
            extra_configs=GenerateConfig(max_thinking_tokens=0),
            enable_thinking=True,
        )

        self.assertFalse(request.disable_thinking())
        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertEqual(config.max_thinking_tokens, 0)
        with self.assertRaises(Exception):
            config.validate()

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

    def test_disable_thinking_keeps_default_backend_thinking_budget(self):
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
        self.assertEqual(config.max_thinking_tokens, 16)
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

        self.assertEqual(config.max_new_tokens, 200)
        self.assertEqual(config.max_completion_tokens, 100)
        self.assertEqual(config.max_thinking_tokens, 10)
        self.assertTrue(config.in_think_mode)

    def test_openai_max_tokens_and_max_completion_tokens_pass_through_independently(
        self,
    ):
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

        self.assertEqual(config.max_new_tokens, 105)
        self.assertEqual(config.max_completion_tokens, 100)
        self.assertEqual(config.max_thinking_tokens, 10)

    def test_openai_max_completion_tokens_does_not_add_default_thinking_budget(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        request = ChatCompletionRequest(
            messages=[],
            max_completion_tokens=200000,
            enable_thinking=True,
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertEqual(config.max_new_tokens, 131072)
        self.assertEqual(config.max_completion_tokens, 200000)
        self.assertEqual(config.max_thinking_tokens, 131072)
        self.assertTrue(config.in_think_mode)

    def test_openai_negative_thinking_budget_is_invalid(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_end_token_id = 102
        request = ChatCompletionRequest(
            messages=[],
            thinking_budget=-1,
            enable_thinking=True,
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertEqual(config.max_thinking_tokens, -1)
        with self.assertRaises(Exception):
            config.validate()

    def test_request_level_thinking_adds_think_end_tokens_when_env_mode_off(self):
        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 0
        generate_env_config.think_end_token_id = -1
        generate_env_config.think_end_tag = "</think>\n\n"
        request = ChatCompletionRequest(
            messages=[],
            thinking_budget=10,
            max_completion_tokens=20,
            enable_thinking=True,
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
            max_completion_tokens=200000,
            enable_thinking=True,
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertTrue(config.in_think_mode)
        self.assertEqual(json.loads(config.json_schema), {"type": "object"})
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
            max_completion_tokens=200000,
            chat_template_kwargs={"enable_thinking": True},
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertTrue(config.in_think_mode)
        self.assertEqual(json.loads(config.json_schema), {"type": "object"})
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
            max_completion_tokens=200000,
            extra_configs=GenerateConfig(
                chat_template_kwargs={"enable_thinking": True}
            ),
        )

        config = self._extract_openai_generation_config(request, generate_env_config)

        self.assertTrue(config.in_think_mode)
        self.assertEqual(json.loads(config.json_schema), {"type": "object"})
        self.assertEqual(
            config.end_think_token_ids,
            self.tokenizer.encode("</think>\n\n", add_special_tokens=False),
        )

    def test_openai_max_completion_tokens_zero_allowed_negative_invalid(self):
        request = ChatCompletionRequest(
            messages=[],
            max_tokens=64,
            max_completion_tokens=0,
        )
        config = self._extract_openai_generation_config(request)
        self.assertEqual(config.max_new_tokens, 64)
        self.assertEqual(config.max_completion_tokens, 0)
        config.validate()

        request = ChatCompletionRequest(messages=[], max_completion_tokens=-1)
        config = self._extract_openai_generation_config(request)
        self.assertEqual(config.max_completion_tokens, -1)
        with self.assertRaises(Exception):
            config.validate()

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
