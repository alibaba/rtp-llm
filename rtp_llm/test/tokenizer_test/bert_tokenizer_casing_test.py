import unittest

from tokenizers import Tokenizer, models, normalizers

from rtp_llm.frontend.tokenizer_factory.tokenizers.bert_tokenizer import BertTokenizer


class BertTokenizerCasingTest(unittest.TestCase):
    def tokenizer(self, lowercase):
        tokenizer = Tokenizer(models.WordPiece({"[UNK]": 0}, unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.BertNormalizer(
            lowercase=lowercase, strip_accents=False
        )
        return tokenizer

    def test_explicit_casing_overrides_serialized_normalizer(self):
        for lowercase, expected in ((True, "école"), (False, "ÉCOLE")):
            with self.subTest(lowercase=lowercase):
                tokenizer = self.tokenizer(not lowercase)
                kwargs = BertTokenizer._transformers_v5_kwargs(
                    {"do_lower_case": lowercase}, tokenizer
                )
                self.assertIs(kwargs["tokenizer_object"], tokenizer)
                self.assertEqual(tokenizer.normalizer.normalize_str("ÉCOLE"), expected)

    def test_unspecified_casing_preserves_serialized_normalizer(self):
        tokenizer = self.tokenizer(False)
        BertTokenizer._transformers_v5_kwargs({}, tokenizer)
        self.assertEqual(tokenizer.normalizer.normalize_str("ÉCOLE"), "ÉCOLE")

    def test_missing_serialized_tokenizer_is_supported(self):
        kwargs = BertTokenizer._transformers_v5_kwargs({"do_lower_case": True})
        self.assertNotIn("tokenizer_object", kwargs)


if __name__ == "__main__":
    unittest.main()
