import unittest
from unittest.mock import Mock, patch

from tokenizers import Tokenizer, decoders, models

from rtp_llm.frontend.tokenizer_factory.tokenizers import base_tokenizer

NativeBackend = (
    base_tokenizer.TokenizersBackend or base_tokenizer.PreTrainedTokenizerFast
)


@unittest.skipIf(NativeBackend is None, "requires a Transformers fast tokenizer")
class NativeBatchDecodeTest(unittest.TestCase):
    def setUp(self):
        native = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "hello": 1, ".": 2, "世界": 3, "[EOS]": 4},
                unk_token="[UNK]",
            )
        )
        native.decoder = decoders.WordPiece(cleanup=False)
        self.tokenizer = NativeBackend(
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
        self.compare(((), (1, 2), (3, 4)))
        self.compare([[1, 3], [1, 2, 4], []] * 334)

    def test_native_decoder_is_used(self):
        decoder = Mock(wraps=self.tokenizer._tokenizer)
        self.tokenizer._tokenizer = decoder
        with patch.object(base_tokenizer, "ENABLE_NATIVE_BATCH_DECODE", True):
            self.wrapper.batch_decode([[1, 3], [4], []])
        decoder.decode_batch.assert_called_once_with(
            [[1, 3], [4], []], skip_special_tokens=False
        )
        decoder.decode.assert_not_called()

    def test_source_tokenizer_flag_and_empty_batch(self):
        if (
            base_tokenizer.PreTrainedTokenizerFast is None
            or getattr(self.tokenizer._decode, "__func__", None)
            is not base_tokenizer.PreTrainedTokenizerFast._decode
        ):
            self.skipTest("requires the Transformers 4.x decode contract")
        with patch.object(base_tokenizer, "ENABLE_NATIVE_BATCH_DECODE", True):
            self.wrapper.batch_decode([[1]], use_source_tokenizer=True)
            self.assertTrue(self.tokenizer._decode_use_source_tokenizer)
            self.wrapper.batch_decode([])
            self.assertTrue(self.tokenizer._decode_use_source_tokenizer)
            self.wrapper.batch_decode([[3]])
            self.assertFalse(self.tokenizer._decode_use_source_tokenizer)

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
