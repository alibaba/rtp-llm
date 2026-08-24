#!/usr/bin/env python3

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.async_decoder_engine.xgrammar_bootstrap import bootstrap_grammar_config


class _Tokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]


def _model(tokenizer, vocab_size=4, model_type=""):
    special_tokens = SimpleNamespace(
        eos_token_id=3,
        stop_words_id_list=[[2], [1, 2]],
    )
    return SimpleNamespace(
        tokenizer=SimpleNamespace(tokenizer=tokenizer),
        model_config=SimpleNamespace(
            model_type=model_type,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        ),
    )


def _engine(backend="xgrammar"):
    return SimpleNamespace(
        grammar_config=SimpleNamespace(
            grammar_backend=backend,
            tokenizer_info_json="stale",
        )
    )


class XGrammarBootstrapTest(unittest.TestCase):
    def test_huggingface_fast_tokenizer_path_is_preserved(self):
        tokenizer = _Tokenizer({"A": 0})
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "hf-json")
        engine = _engine()

        with mock.patch(
            "rtp_llm.ops.build_xgrammar_tokenizer_info_json",
            return_value="hf-info",
        ) as build:
            bootstrap_grammar_config(engine, _model(tokenizer))

        self.assertEqual(engine.grammar_config.tokenizer_info_json, "hf-info")
        build.assert_called_once_with({"A": 0}, "hf-json", 4, [2, 3])

    def test_tiktoken_uses_byte_level_vocab_metadata(self):
        tokenizer = _Tokenizer({"A": 0, "\u0120": 1})
        tokenizer.byte_decoder = {"A": 65, "\u0120": 32}
        tokenizer.model = SimpleNamespace(decode_single_token_bytes=lambda _: b"A")
        engine = _engine()

        with mock.patch(
            "rtp_llm.ops.build_xgrammar_tokenizer_info_json_from_vocab",
            return_value="tiktoken-info",
        ) as build:
            bootstrap_grammar_config(engine, _model(tokenizer))

        self.assertEqual(engine.grammar_config.tokenizer_info_json, "tiktoken-info")
        build.assert_called_once_with({"A": 0, "\u0120": 1}, 2, 4, [2, 3], False)

    def test_unsupported_tokenizer_disables_grammar(self):
        engine = _engine()
        with self.assertLogs(level="WARNING"):
            bootstrap_grammar_config(engine, _model(_Tokenizer({"A": 0})))
        self.assertEqual(engine.grammar_config.tokenizer_info_json, "")

    def test_disabled_backend_clears_stale_metadata(self):
        engine = _engine("none")
        bootstrap_grammar_config(engine, _model(_Tokenizer({"A": 0})))
        self.assertEqual(engine.grammar_config.tokenizer_info_json, "")

    def test_renderer_pretokenized_chat_constraints_reach_engine_config(self):
        tokenizer = _Tokenizer({"A": 0})
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "hf-json")
        engine = _engine()

        class _Renderer:
            @classmethod
            def pretokenized_chat_constraints(cls):
                return {
                    "reasoning": {
                        "prompt_tail": "TR",
                        "structural_tag": {"type": "reasoning"},
                    },
                    "response": {
                        "prompt_tail": "RS",
                        "structural_tag": {"type": "response"},
                    },
                }

        with (
            mock.patch(
                "rtp_llm.ops.build_xgrammar_tokenizer_info_json",
                return_value="hf-info",
            ),
            mock.patch(
                "rtp_llm.openai.renderer_factory_register.get_renderer_class",
                return_value=_Renderer,
            ),
        ):
            bootstrap_grammar_config(engine, _model(tokenizer, model_type="test_model"))

        grammar = engine.grammar_config
        self.assertEqual(grammar.reasoning_prompt_tail_token_ids, [84, 82])
        self.assertEqual(grammar.response_prompt_tail_token_ids, [82, 83])
        self.assertEqual(
            json.loads(grammar.reasoning_structural_tag), {"type": "reasoning"}
        )
        self.assertEqual(
            json.loads(grammar.response_structural_tag), {"type": "response"}
        )
        self.assertEqual(grammar.reasoning_completion_boundary_token_ids, [])
        self.assertEqual(grammar.response_completion_boundary_token_ids, [])

    def test_renderer_completion_boundaries_reach_engine_config(self):
        tokenizer = _Tokenizer({"A": 0})
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "hf-json")
        engine = _engine()

        class _Renderer:
            @classmethod
            def pretokenized_chat_constraints(cls):
                return {
                    "reasoning": {
                        "prompt_tail": "TR",
                        "completion_boundary": "MC",
                    },
                    "response": {
                        "prompt_tail": "RS",
                        "completion_boundary": "MC",
                    },
                }

        with (
            mock.patch(
                "rtp_llm.ops.build_xgrammar_tokenizer_info_json",
                return_value="hf-info",
            ),
            mock.patch(
                "rtp_llm.openai.renderer_factory_register.get_renderer_class",
                return_value=_Renderer,
            ),
        ):
            bootstrap_grammar_config(engine, _model(tokenizer, model_type="test_model"))

        grammar = engine.grammar_config
        self.assertEqual(grammar.reasoning_structural_tag, "")
        self.assertEqual(grammar.response_structural_tag, "")
        self.assertEqual(grammar.reasoning_completion_boundary_token_ids, [77, 67])
        self.assertEqual(grammar.response_completion_boundary_token_ids, [77, 67])

    def test_bad_pretokenized_defaults_do_not_disable_request_grammar(self):
        tokenizer = _Tokenizer({"A": 0})
        tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: "hf-json")
        engine = _engine()

        class _BadRenderer:
            @classmethod
            def pretokenized_chat_constraints(cls):
                return {
                    "reasoning": {"prompt_tail": "", "structural_tag": {}},
                    "response": {"prompt_tail": "", "structural_tag": {}},
                }

        with (
            mock.patch(
                "rtp_llm.ops.build_xgrammar_tokenizer_info_json",
                return_value="hf-info",
            ),
            mock.patch(
                "rtp_llm.openai.renderer_factory_register.get_renderer_class",
                return_value=_BadRenderer,
            ),
            self.assertLogs(level="WARNING"),
        ):
            bootstrap_grammar_config(engine, _model(tokenizer, model_type="test_model"))

        self.assertEqual(engine.grammar_config.tokenizer_info_json, "hf-info")
        self.assertEqual(engine.grammar_config.reasoning_structural_tag, "")
        self.assertEqual(
            engine.grammar_config.reasoning_completion_boundary_token_ids, []
        )

    def test_explicit_vocab_builder_decodes_byte_level_tokens(self):
        from rtp_llm.ops import (
            build_xgrammar_tokenizer_info_json_from_vocab,
            ensure_engine_ops_loaded,
        )

        ensure_engine_ops_loaded()
        info = json.loads(
            build_xgrammar_tokenizer_info_json_from_vocab(
                {"A": 0, "\u0120": 1, "<eos>": 2},
                2,
                4,
                [2],
                False,
            )
        )

        self.assertEqual(info["vocab_type"], 2)
        self.assertEqual(info["vocab_size"], 4)
        self.assertFalse(info["add_prefix_space"])
        self.assertEqual(info["stop_token_ids"], [2])
        self.assertEqual(info["decoded_vocab"], ["A", " ", "<eos>", ""])
        self.assertEqual(info["special_token_ids"], [3])


if __name__ == "__main__":
    unittest.main()
