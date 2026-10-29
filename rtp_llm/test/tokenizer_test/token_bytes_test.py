import unittest

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rtp_llm.frontend.tokenizer_factory.tokenizers.base_tokenizer import BaseTokenizer


class TokenBytesTest(unittest.TestCase):
    def wrap(self, tokenizer):
        wrapper = BaseTokenizer.__new__(BaseTokenizer)
        wrapper.tokenizer = tokenizer
        return wrapper

    def test_bytelevel_recovers_incomplete_utf8_and_space(self):
        native = Tokenizer(models.BPE({"Ã": 0, "©": 1, "Ġ": 2, "Ā": 3, "Ċ": 4}, []))
        native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        native.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=native)
        wrapper = self.wrap(tokenizer)
        self.assertEqual(tokenizer.decode([0]), "\ufffd")
        self.assertEqual(tokenizer.decode([0, 1]), "é")
        self.assertEqual(
            [wrapper.token_id_to_bytes(i) for i in range(5)],
            [b"\xc3", b"\xa9", b" ", b"\x00", b"\n"],
        )

    def test_byte_fallback_recovers_fragment(self):
        native = Tokenizer(models.BPE({"<0xC3>": 0, "<0xA9>": 1}, []))
        native.decoder = decoders.ByteFallback()
        wrapper = self.wrap(PreTrainedTokenizerFast(tokenizer_object=native))
        self.assertEqual(wrapper.token_id_to_bytes(0), b"\xc3")
        self.assertEqual(wrapper.token_id_to_bytes(1), b"\xa9")

    def test_unknown_decoder_does_not_invent_bytes(self):
        native = Tokenizer(models.WordLevel({"\ufffd": 0}))
        wrapper = self.wrap(PreTrainedTokenizerFast(tokenizer_object=native))
        self.assertIsNone(wrapper.token_id_to_bytes(0))


if __name__ == "__main__":
    unittest.main()
