import unittest
import sys
import types

from rtp_llm.structured_output.xgrammar_frontend import (
    XGrammarFrontendCompiler,
    canonicalize_schema,
    fingerprint_tokenizer,
)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    name_or_path = "fake-tokenizer"

    def __len__(self):
        return 4

    def get_vocab(self):
        return {"<pad>": 0, "{": 1, "}": 2, "a": 3}


class FakeTokenizerInfo:
    @staticmethod
    def from_huggingface(tokenizer, vocab_size=None, stop_token_ids=None):
        return FakeTokenizerInfo()

    def serialize_json(self):
        return '{"fake":"tokenizer"}'


class FakeCompiledGrammar:
    def serialize_json(self):
        return '{"fake":"compiled"}'


class FakeGrammarCompiler:
    def __init__(self, tokenizer_info, max_threads=1, cache_enabled=True):
        self.tokenizer_info = tokenizer_info

    def compile_builtin_json_grammar(self):
        return FakeCompiledGrammar()

    def compile_json_schema(self, schema, strict_mode=True):
        return FakeCompiledGrammar()


class XGrammarFrontendTest(unittest.TestCase):
    def setUp(self):
        fake_xgrammar = types.ModuleType("xgrammar")
        fake_xgrammar.__version__ = "0.2.1"
        fake_xgrammar.TokenizerInfo = FakeTokenizerInfo
        fake_xgrammar.GrammarCompiler = FakeGrammarCompiler
        sys.modules["xgrammar"] = fake_xgrammar
        self.tokenizer = FakeTokenizer()

    def test_canonicalize_schema_is_stable(self):
        left = {"required": ["answer"], "type": "object", "properties": {"answer": {"type": "string"}}}
        right = {"properties": {"answer": {"type": "string"}}, "type": "object", "required": ["answer"]}
        self.assertEqual(canonicalize_schema(left), canonicalize_schema(right))

    def test_tokenizer_fingerprint_is_stable(self):
        self.assertEqual(fingerprint_tokenizer(self.tokenizer), fingerprint_tokenizer(FakeTokenizer()))

    def test_compile_cache_hit_and_eviction(self):
        compiler = XGrammarFrontendCompiler(capacity=1, thread_num=1)
        compiler.compile_response_format({"type": "json_object"}, self.tokenizer)
        hit = compiler.compile_response_format({"type": "json_object"}, self.tokenizer)
        self.assertTrue(hit.cache_hit)
        self.assertEqual(compiler.stats()["xgrammar_compile_cache_hit_total"], 1)

        compiler.compile_response_format(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            },
            self.tokenizer,
        )
        self.assertEqual(compiler.stats()["xgrammar_compile_cache_entries"], 1)
        self.assertEqual(compiler.stats()["xgrammar_compile_cache_eviction_total"], 1)

    def test_rejects_invalid_response_format(self):
        compiler = XGrammarFrontendCompiler(capacity=1, thread_num=1)
        with self.assertRaises(ValueError):
            compiler.compile_response_format({"type": "text"}, self.tokenizer)


if __name__ == "__main__":
    unittest.main()
