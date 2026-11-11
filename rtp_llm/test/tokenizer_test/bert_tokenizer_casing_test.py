import unittest

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from transformers import RobertaTokenizerFast

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

    def test_non_bert_normalizer_is_not_rewritten(self):
        tokenizer = self.tokenizer(False)
        tokenizer.normalizer = normalizers.Lowercase()
        original = tokenizer.normalizer.__getstate__()
        BertTokenizer._transformers_v5_kwargs({"do_lower_case": False}, tokenizer)
        self.assertEqual(tokenizer.normalizer.__getstate__(), original)

    def test_roberta_byte_level_bpe_keeps_case_sensitive_tokens(self):
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.train_from_iterator(
            ["HELLO hello ÉCOLE école"],
            trainers.BpeTrainer(
                vocab_size=100,
                special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
            ),
        )
        before = backend.to_str()
        kwargs = BertTokenizer._transformers_v5_kwargs({"do_lower_case": True}, backend)
        self.assertEqual(backend.to_str(), before)
        tokenizer = RobertaTokenizerFast(**kwargs)
        self.assertEqual(
            tokenizer.encode("HELLO", add_special_tokens=False),
            backend.encode("HELLO").ids,
        )
        self.assertNotEqual(
            tokenizer.encode("HELLO", add_special_tokens=False),
            tokenizer.encode("hello", add_special_tokens=False),
        )


if __name__ == "__main__":
    unittest.main()
