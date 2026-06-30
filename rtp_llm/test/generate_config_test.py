import os
from typing import Any, List, Optional
from unittest import TestCase, main

from pydantic import ValidationError
from transformers import AutoTokenizer

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.config.response_format import ResponseFormat
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


class GrammarBeamSearchRejectionTest(TestCase):
    """validate() must reject beam search + grammar-constrained decoding."""

    def _assert_rejected(self, **fields):
        cfg = GenerateConfig(**fields)
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.validate()
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSUPPORTED_OPERATION
        )

    def _assert_accepted(self, **fields):
        GenerateConfig(**fields).validate()

    def test_grammar_field_plus_beam_rejected(self):
        # Direct grammar field (incl. empty/falsy values) × beam-search knob.
        grammar_fields = [
            ("json_schema", '{"type": "object"}'),
            ("json_schema", {}),
            ("json_schema", ""),
            ("regex", r"\d+"),
            ("regex", ""),
            ("ebnf", "root ::= 'a'"),
            ("ebnf", ""),
            ("structural_tag", '{"begin": "<t>", "end": "</t>"}'),
            ("structural_tag", ""),
        ]
        beam_knobs = [
            {"num_beams": 4},
            {"variable_num_beams": [1, 3]},
            {"num_return_sequences": 2},
        ]
        for grammar_key, grammar_val in grammar_fields:
            for beam in beam_knobs:
                with self.subTest(grammar=grammar_key, value=grammar_val, beam=beam):
                    self._assert_rejected(**{grammar_key: grammar_val}, **beam)

    def test_response_format_plus_beam_rejected(self):
        # Every grammar-resolving rf type rejects; string + dict envelopes both flow through.
        formats = [
            {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
            {"type": "json_object"},
            {"type": "regex", "pattern": r"\d+"},
            '{"type": "json_object"}',
        ]
        for rf in formats:
            with self.subTest(response_format=rf):
                self._assert_rejected(num_beams=2, response_format=rf)

    def test_response_format_no_grammar_allowed(self):
        # `text` and empty envelopes do not set a grammar, so beam stays allowed.
        for rf in [{"type": "text"}, {}, "", None]:
            with self.subTest(response_format=rf):
                self._assert_accepted(num_beams=2, response_format=rf)

    def test_response_format_unknown_type_rejected(self):
        # `type` is a closed Literal; pydantic rejects at construction time.
        with self.assertRaises(ValidationError):
            GenerateConfig(response_format={"type": "something_else"})

    def test_response_format_missing_payload_rejected(self):
        # Missing/empty/wrong-typed inner payload: pydantic rejects before validate runs.
        cases = [
            {"type": "json_schema"},  # no inner
            {"type": "json_schema", "json_schema": {}},  # missing schema
            {
                "type": "json_schema",
                "json_schema": {"name": "foo"},
            },  # the silent-relax case
            {"type": "json_schema", "json_schema": 42},  # wrong inner type
            {"type": "json_schema", "json_schema": [1, 2]},  # wrong inner type
            {"type": "regex"},
            {"type": "regex", "pattern": ""},
            {"type": "regex", "pattern": {"foo": "bar"}},  # wrong pattern type
            {"type": "ebnf"},
            {"type": "ebnf", "grammar": ""},
            {"type": "structural_tag"},
            {"type": "structural_tag", "structural_tag": ""},
        ]
        for rf in cases:
            with self.subTest(response_format=rf):
                with self.assertRaises(ValidationError):
                    GenerateConfig(response_format=rf)

    def test_response_format_json_object_no_payload_required(self):
        # `json_object` is the any-JSON shortcut and intentionally has no inner payload.
        self._assert_accepted(num_beams=1, response_format={"type": "json_object"})

    def test_grammar_or_beam_alone_allowed(self):
        # Sanity: each side in isolation must validate.
        self._assert_accepted(json_schema='{"type": "object"}')
        self._assert_accepted(num_beams=4)
        self._assert_accepted(
            num_beams=4, json_schema=None, regex=None, ebnf=None, structural_tag=None
        )


class ResponseFormatProjectionTest(TestCase):
    """rf projected to typed fields and cleared by validate(); rf wins over stale extra_configs."""

    def test_json_schema_envelope_projected(self):
        cfg = GenerateConfig(
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": {"type": "string"}},
            }
        )
        cfg.validate()
        self.assertIsNone(cfg.response_format)
        self.assertEqual(cfg.json_schema, '{"type":"string"}')
        self.assertIsNone(cfg.regex)
        self.assertIsNone(cfg.ebnf)
        self.assertIsNone(cfg.structural_tag)

    def test_json_object_envelope_projected_to_any_json(self):
        cfg = GenerateConfig(response_format={"type": "json_object"})
        cfg.validate()
        self.assertIsNone(cfg.response_format)
        self.assertEqual(cfg.json_schema, '{"type": "object"}')

    def test_regex_envelope_projected(self):
        cfg = GenerateConfig(response_format={"type": "regex", "pattern": r"\d+"})
        cfg.validate()
        self.assertIsNone(cfg.response_format)
        self.assertEqual(cfg.regex, r"\d+")

    def test_text_envelope_clears_grammar(self):
        cfg = GenerateConfig(
            response_format={"type": "text"},
            json_schema='{"type": "object"}',
        )
        cfg.validate()
        # rf=text wins: stale extra_configs grammar is cleared.
        self.assertIsNone(cfg.response_format)
        self.assertIsNone(cfg.json_schema)

    def test_envelope_overrides_stale_typed_field(self):
        # extra_configs.json_schema is overridden by top-level response_format.
        cfg = GenerateConfig(
            response_format={"type": "regex", "pattern": r"[a-z]+"},
            json_schema='{"type": "object"}',
        )
        cfg.validate()
        self.assertIsNone(cfg.response_format)
        self.assertIsNone(cfg.json_schema)
        self.assertEqual(cfg.regex, r"[a-z]+")

    def test_string_envelope_accepted(self):
        cfg = GenerateConfig(response_format='{"type":"json_object"}')
        cfg.validate()
        self.assertIsNone(cfg.response_format)
        self.assertEqual(cfg.json_schema, '{"type": "object"}')

    def test_blank_string_envelope_treated_as_none(self):
        cfg = GenerateConfig(response_format="   ")
        cfg.validate()
        self.assertIsNone(cfg.response_format)
        self.assertIsNone(cfg.json_schema)


class GrammarFieldNormalizationTest(TestCase):
    """Typed grammar fields are normalized before backend/RPC consumption."""

    def test_direct_json_schema_dict_normalized(self):
        cfg = GenerateConfig(json_schema={"type": "object", "title": "测试"})
        cfg.validate()
        self.assertEqual(cfg.json_schema, '{"type":"object","title":"测试"}')

    def test_direct_structural_tag_dict_normalized(self):
        cfg = GenerateConfig(
            structural_tag={
                "type": "structural_tag",
                "structures": [{"begin": "<answer>", "end": "</answer>"}],
            }
        )
        cfg.validate()
        self.assertEqual(
            cfg.structural_tag,
            '{"type":"structural_tag","structures":[{"begin":"<answer>","end":"</answer>"}]}',
        )


class RawUpdateResponseFormatCoercionTest(TestCase):
    """update / update_and_pop must run the rf coercer; raw HTTP path goes through update_and_pop."""

    def test_update_coerces_dict_envelope(self):
        cfg = GenerateConfig()
        cfg.update({"response_format": {"type": "json_object"}})
        self.assertIsInstance(cfg.response_format, ResponseFormat)
        cfg.validate()
        self.assertIsNone(cfg.response_format)
        self.assertEqual(cfg.json_schema, '{"type": "object"}')

    def test_update_and_pop_coerces_string_envelope(self):
        cfg = GenerateConfig()
        remain = cfg.update_and_pop(
            {"response_format": '{"type":"regex","pattern":"\\\\d+"}', "stranger": 1}
        )
        self.assertEqual(remain, {"stranger": 1})
        self.assertIsInstance(cfg.response_format, ResponseFormat)
        cfg.validate()
        self.assertEqual(cfg.regex, r"\d+")

    def test_update_rejects_malformed_envelope(self):
        cfg = GenerateConfig()
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.update({"response_format": {"type": "json_schema"}})
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )

    def test_update_and_pop_rejects_invalid_json_envelope(self):
        cfg = GenerateConfig()
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.update_and_pop(
                {"response_format": '{"type":"json_object"', "stranger": 1}
            )
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )


class GrammarConstraintMutualExclusionTest(TestCase):
    """Only one typed grammar field may be set per request once envelope is projected."""

    def _assert_rejected(self, **fields):
        cfg = GenerateConfig(**fields)
        with self.assertRaises(FtRuntimeException) as ctx:
            cfg.validate()
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.UNSUPPORTED_OPERATION
        )

    def test_json_schema_plus_regex_rejected(self):
        self._assert_rejected(json_schema='{"type": "object"}', regex=r"\d+")

    def test_envelope_overrides_typed_field_no_conflict(self):
        # rf wins over stale typed fields, so this is NOT a multi-grammar conflict.
        cfg = GenerateConfig(
            json_schema='{"type": "object"}',
            response_format={"type": "json_object"},
        )
        cfg.validate()
        self.assertIsNone(cfg.response_format)
        self.assertEqual(cfg.json_schema, '{"type": "object"}')


if __name__ == "__main__":
    main()
