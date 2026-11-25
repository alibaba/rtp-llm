import string
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

from rtp_llm.async_decoder_engine.xgrammar_bootstrap import bootstrap_grammar_config
from rtp_llm.config.grammar_tokenizer_info import (
    _make_xgrammar_bindings_findable,
    build_model_grammar_tokenizer_info_json,
    grammar_stop_token_ids,
)
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.config.stop_words import resolve_stop_word_ids
from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class GrammarStopWordsTest(unittest.TestCase):
    def setUp(self):
        vocab = {c: i for i, c in enumerate(string.printable)}
        backend = Tokenizer(BPE(vocab, []))
        backend.decoder = decoders.ByteLevel()
        backend.add_special_tokens(["<eos>", "<turn>", "<checkpoint>", "<startup>"])
        self.hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")
        self.turn, self.checkpoint, self.startup = self.hf.convert_tokens_to_ids(
            ["<turn>", "<checkpoint>", "<startup>"]
        )
        self.eos = self.hf.eos_token_id
        self.tok = BaseTokenizer.__new__(BaseTokenizer)
        self.tok.tokenizer = self.hf
        self.tok.config_json = {"eos_token_id": [self.eos, self.checkpoint]}
        self.config = SimpleNamespace(
            generate_env_config=GenerateEnvConfig(),
            render_config=RenderConfig(),
            special_tokens=SimpleNamespace(
                stop_words_id_list=[], stop_words_str_list=[], eos_token_id=self.eos
            ),
            ckpt_path="",
            model_type="test",
            max_seq_len=128,
            template_type="",
            vocab_size=len(self.hf),
        )
        factory = patch(
            "rtp_llm.openai.renderer_factory.ChatRendererFactory.get_renderer"
        )
        self.addCleanup(factory.stop)
        factory.start().return_value.get_all_extra_stop_word_ids_list.return_value = [
            [self.turn]
        ]
        self.config.generate_env_config.stop_words_str = '["<startup>", "}"]'
        self.config.generate_env_config.stop_words_list = f"[[{self.turn}, {self.eos}]]"

    def resolve(self):
        return resolve_stop_word_ids(self.tok, self.config, None, None)

    def test_startup_and_force(self):
        sequences = self.resolve()
        self.assertIn(self.tok.encode("}"), sequences)
        self.assertIn([self.turn, self.eos], sequences)
        self.assertEqual(
            grammar_stop_token_ids(self.tok, self.config, sequences, []),
            sorted([self.eos, self.turn, self.checkpoint, self.startup]),
        )
        self.config.generate_env_config.force_stop_words = True
        self.assertEqual(
            grammar_stop_token_ids(self.tok, self.config, self.resolve(), []),
            sorted([self.eos, self.startup]),
        )
        self.config.generate_env_config.stop_words_str = ""
        self.config.generate_env_config.stop_words_list = ""
        self.assertEqual(self.resolve(), [])
        self.assertEqual(
            grammar_stop_token_ids(self.tok, self.config, [], []), [self.eos]
        )

    def test_compiled_metadata_on_admission_and_backend(self):
        _make_xgrammar_bindings_findable()
        import xgrammar as xgr

        sequences = self.resolve()
        for override in ([], [self.eos, self.turn]):
            with self.subTest(override=override):
                grammar = SimpleNamespace(
                    grammar_backend="xgrammar", override_stop_tokens=override
                )
                engine = SimpleNamespace(grammar_config=grammar, misc_config=None)
                model = SimpleNamespace(
                    tokenizer=self.tok, model_config=self.config, vit_config=None
                )
                bootstrap_grammar_config(engine, model)
                admission = build_model_grammar_tokenizer_info_json(
                    self.tok, self.config, sequences, override
                )
                expected = sorted(
                    override or [self.eos, self.turn, self.checkpoint, self.startup]
                )
                for metadata in (admission, grammar.tokenizer_info_json):
                    info = xgr.TokenizerInfo.deserialize_json(metadata)
                    self.assertEqual(sorted(info.stop_token_ids), expected)
                    compiled = xgr.GrammarCompiler(info).compile_builtin_json_grammar()
                    matcher = xgr.GrammarMatcher(compiled)
                    for token in self.tok.encode('{"a":"'):
                        self.assertTrue(matcher.accept_token(token))
                    mask = xgr.allocate_token_bitmask(1, len(self.hf))
                    matcher.fill_next_token_bitmask(mask)
                    self.assertEqual(
                        (int(mask[0, self.turn // 32]) >> (self.turn % 32)) & 1, 0
                    )
                    self.assertFalse(matcher.accept_token(self.turn))
                    for token in self.tok.encode('ok"}'):
                        self.assertTrue(matcher.accept_token(token))
                    self.assertTrue(matcher.accept_token(self.turn))
                    self.assertTrue(matcher.is_terminated())


if __name__ == "__main__":
    unittest.main()
