import unittest
from unittest.mock import Mock, patch

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

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
        self.assertEqual(self.wrapper.batch_decode(rows, **kwargs), expected)

    def test_native_batch_semantics(self):
        for skip in [False, True]:
            self.compare([[1, 2, 4], [], [3, 0]], skip_special_tokens=skip)
        self.compare([])
        self.compare([[4]])  # default must retain special tokens
        self.compare(((), (1, 2), (3, 4)))
        self.compare([[1, 3], [1, 2, 4], []] * 334)

    def test_bytelevel_bpe_with_runtime_added_tokens(self):
        native = Tokenizer(models.BPE(unk_token="[UNK]"))
        native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        native.decoder = decoders.ByteLevel()
        native.train_from_iterator(
            ["hello 世界 café 🙂 .", "hello hello world", "space and punctuation!"],
            trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "[EOS]"]),
        )
        self.tokenizer = NativeBackend(
            tokenizer_object=native,
            eos_token="[EOS]",
            clean_up_tokenization_spaces=False,
        )
        self.tokenizer.add_tokens(
            [AddedToken("<added>", lstrip=True, normalized=False)]
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<special>"]})
        self.wrapper.tokenizer = self.tokenizer
        rows = [
            self.tokenizer.encode(text, add_special_tokens=False)
            for text in (
                "hello 世界 café 🙂",
                "hello <added><added>世界",
                "<special>hello<special>[EOS]",
                "",
                "hello .",
            )
        ]
        for skip in (False, True):
            for cleanup in (False, True):
                self.compare(
                    rows, skip_special_tokens=skip, clean_up_tokenization_spaces=cleanup
                )

    def test_native_decoder_is_used(self):
        decoder = Mock(wraps=self.tokenizer._tokenizer)
        self.tokenizer._tokenizer = decoder
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
            self.assertEqual(
                self.wrapper.batch_decode([[1], [3]]), ["custom", "custom"]
            )
            self.assertEqual(decode.call_count, 2)

    def test_missing_native_backend_keeps_legacy_decode(self):
        class LegacyTokenizer:
            clean_up_tokenization_spaces = False

            def _decode(self, row, **kwargs):
                return f"legacy:{row}"

        self.wrapper.tokenizer = LegacyTokenizer()
        with patch.object(
            base_tokenizer, "NATIVE_DECODE_IMPLEMENTATIONS", (LegacyTokenizer._decode,)
        ):
            self.assertEqual(
                self.wrapper.batch_decode([[1], []]), ["legacy:[1]", "legacy:[]"]
            )


if __name__ == "__main__":
    unittest.main()
