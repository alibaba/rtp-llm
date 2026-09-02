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


def _model(tokenizer, vocab_size=4):
    special_tokens = SimpleNamespace(
        eos_token_id=3,
        stop_words_id_list=[[2], [1, 2]],
    )
    return SimpleNamespace(
        tokenizer=SimpleNamespace(tokenizer=tokenizer),
        model_config=SimpleNamespace(
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
