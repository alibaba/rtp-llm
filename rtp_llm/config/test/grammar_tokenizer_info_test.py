import unittest
from typing import Dict, List

from rtp_llm.config.grammar_tokenizer_info import _is_byte_level_tokenizer


class FakeTokenizer:
    def __init__(self, token_by_id: Dict[int, str], encoded_ids: List[int]):
        self._token_by_id = token_by_id
        self._encoded_ids = encoded_ids
        self.add_special_tokens = None

    def encode(self, text: str, *, add_special_tokens: bool) -> List[int]:
        self.add_special_tokens = add_special_tokens
        return self._encoded_ids

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return [self._token_by_id[token_id] for token_id in token_ids]


class LegacyTokenizer:
    def encode(self, text: str) -> List[int]:
        return [1, 2]

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        return ["<bos>", "\u0120"]


class GrammarTokenizerInfoTest(unittest.TestCase):
    def test_byte_level_detection_disables_special_tokens(self):
        tokenizer = FakeTokenizer({1: "\u0120"}, [1])

        self.assertTrue(_is_byte_level_tokenizer(tokenizer))
        self.assertFalse(tokenizer.add_special_tokens)

    def test_byte_level_detection_handles_empty_encoding(self):
        tokenizer = FakeTokenizer({}, [])

        self.assertFalse(_is_byte_level_tokenizer(tokenizer))

    def test_byte_level_detection_ignores_prefix_token(self):
        tokenizer = FakeTokenizer({1: "<bos>", 2: "\u0120"}, [1, 2])

        self.assertTrue(_is_byte_level_tokenizer(tokenizer))

    def test_byte_level_detection_rejects_raw_space_after_prefix(self):
        tokenizer = FakeTokenizer({1: "<bos>", 2: " "}, [1, 2])

        self.assertFalse(_is_byte_level_tokenizer(tokenizer))

    def test_byte_level_detection_supports_legacy_encode_signature(self):
        self.assertTrue(_is_byte_level_tokenizer(LegacyTokenizer()))


if __name__ == "__main__":
    unittest.main()
