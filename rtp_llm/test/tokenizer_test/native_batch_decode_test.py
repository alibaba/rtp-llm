import unittest
from unittest.mock import patch

from rtp_llm.frontend.tokenizer_factory.tokenizers import base_tokenizer
from tokenizers import Tokenizer, decoders, models


@unittest.skipIf(base_tokenizer.TokenizersBackend is None, "requires Transformers v5")
class NativeBatchDecodeTest(unittest.TestCase):
    def setUp(self):
        native = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "hello": 1, ".": 2, "世界": 3, "[EOS]": 4},
                unk_token="[UNK]",
            )
        )
        native.decoder = decoders.WordPiece(cleanup=False)
        self.tokenizer = base_tokenizer.TokenizersBackend(
            tokenizer_object=native,
            eos_token="[EOS]",
            clean_up_tokenization_spaces=False,
        )
        self.wrapper = base_tokenizer.BaseTokenizer.__new__(
            base_tokenizer.BaseTokenizer
        )
        self.wrapper.tokenizer = self.tokenizer

    def compare(self, rows, **kwargs):
        expected = [self.tokenizer._decode(row, **kwargs) for row in rows]
        with patch.object(base_tokenizer, "ENABLE_NATIVE_BATCH_DECODE", True):
            self.assertEqual(self.wrapper.batch_decode(rows, **kwargs), expected)
        with patch.object(base_tokenizer, "ENABLE_NATIVE_BATCH_DECODE", False):
            self.assertEqual(self.wrapper.batch_decode(rows, **kwargs), expected)

    def test_native_batch_semantics(self):
        for skip in [False, True]:
            self.compare([[1, 2, 4], [], [3, 0]], skip_special_tokens=skip)
        self.compare([])
        self.compare([[4]])  # default must retain special tokens

    def test_cleanup_and_scalar_compatibility(self):
        self.compare([[1, 2]], clean_up_tokenization_spaces=True)
        self.tokenizer.clean_up_tokenization_spaces = True
        self.compare([[1, 2]])
        self.compare([[1, 2]], clean_up_tokenization_spaces=False)
        self.compare([1, 3])

    def test_custom_decode_is_not_bypassed(self):
        with patch.object(
            self.tokenizer, "_decode", side_effect=lambda row, **kw: "custom"
        ) as decode:
            with patch.object(base_tokenizer, "ENABLE_NATIVE_BATCH_DECODE", True):
                self.assertEqual(
                    self.wrapper.batch_decode([[1], [3]]), ["custom", "custom"]
                )
            self.assertEqual(decode.call_count, 2)


if __name__ == "__main__":
    unittest.main()
