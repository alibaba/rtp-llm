import json
import sys
import types
import unittest
from typing import Dict, List, Optional
from unittest import mock

from rtp_llm.config import grammar_tokenizer_info
from rtp_llm.config.grammar_tokenizer_info import (
    _build_encoded_vocab,
    _is_byte_level_tokenizer,
)


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


class FakeBackendTokenizer:
    def to_str(self) -> str:
        return '{"backend":"fake"}'


class FakeBuildTokenizer:
    def __init__(self, vocab: Dict[str, int]):
        self._vocab = vocab
        self.backend_tokenizer = FakeBackendTokenizer()

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab


class GrammarTokenizerInfoTest(unittest.TestCase):
    def build_tokenizer_info(
        self,
        tokenizer: FakeBuildTokenizer,
        *,
        model_vocab_size: int = 0,
        stop_token_ids: Optional[List[int]] = None,
        is_fast: bool = False,
        is_tiktoken: bool = False,
        is_sentencepiece: bool = False,
        is_byte_level: bool = False,
    ):
        if stop_token_ids is None:
            stop_token_ids = [2]
        serializer = mock.Mock(return_value="serialized")
        fake_ops = types.ModuleType("rtp_llm.ops")
        fake_ops.serialize_grammar_tokenizer_info = serializer
        with (
            mock.patch.dict(sys.modules, {"rtp_llm.ops": fake_ops}),
            mock.patch.object(
                grammar_tokenizer_info,
                "_is_fast_tokenizer",
                return_value=is_fast,
            ),
            mock.patch.object(
                grammar_tokenizer_info,
                "_is_tiktoken_tokenizer",
                return_value=is_tiktoken,
            ),
            mock.patch.object(
                grammar_tokenizer_info,
                "_is_sentencepiece_tokenizer",
                return_value=is_sentencepiece,
            ),
            mock.patch.object(
                grammar_tokenizer_info,
                "_is_byte_level_tokenizer",
                return_value=is_byte_level,
            ),
        ):
            result = grammar_tokenizer_info.build_grammar_tokenizer_info_json(
                tokenizer,
                model_vocab_size=model_vocab_size,
                stop_token_ids=stop_token_ids,
            )
        return result, serializer

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

    def test_build_encoded_vocab_rejects_empty_vocab(self):
        with self.assertRaisesRegex(ValueError, "tokenizer vocab is empty"):
            _build_encoded_vocab({}, 0)

    def test_build_encoded_vocab_rejects_negative_model_vocab_size(self):
        with self.assertRaisesRegex(ValueError, "negative model_vocab_size -1"):
            _build_encoded_vocab({"a": 0}, -1)

    def test_build_encoded_vocab_rejects_negative_token_id(self):
        with self.assertRaisesRegex(ValueError, "negative token id -1"):
            _build_encoded_vocab({"a": -1}, 1)

    def test_build_encoded_vocab_preserves_sparse_token_ids(self):
        encoded_vocab, vocab_size = _build_encoded_vocab(
            {"zero": 0, "three": 3},
            2,
        )

        self.assertEqual(encoded_vocab, ["zero", "", "", "three"])
        self.assertEqual(vocab_size, 4)

    def test_build_encoded_vocab_uses_larger_model_vocab_size(self):
        encoded_vocab, vocab_size = _build_encoded_vocab(
            {"zero": 0, "three": 3},
            6,
        )

        self.assertEqual(encoded_vocab, ["zero", "", "", "three", "", ""])
        self.assertEqual(vocab_size, 6)

    def test_build_tokenizer_info_serializes_fast_tokenizer(self):
        result, serializer = self.build_tokenizer_info(
            FakeBuildTokenizer({"a": 0, "c": 2}),
            model_vocab_size=4,
            stop_token_ids=[2, 3],
            is_fast=True,
        )

        self.assertEqual(result, "serialized")
        encoded_vocab, metadata_json = serializer.call_args.args
        self.assertEqual(encoded_vocab, ["a", "", "c", ""])
        self.assertEqual(
            json.loads(metadata_json),
            {
                "vocab_size": 4,
                "stop_token_ids": [2, 3],
                "hf_tokenizer_json": '{"backend":"fake"}',
            },
        )

    def test_build_tokenizer_info_serializes_byte_level_tiktoken(self):
        result, serializer = self.build_tokenizer_info(
            FakeBuildTokenizer({"a": 0}),
            stop_token_ids=[0],
            is_tiktoken=True,
            is_byte_level=True,
        )

        self.assertEqual(result, "serialized")
        encoded_vocab, metadata_json = serializer.call_args.args
        self.assertEqual(encoded_vocab, ["a"])
        self.assertEqual(
            json.loads(metadata_json),
            {
                "vocab_size": 1,
                "stop_token_ids": [0],
                "vocab_type": "BYTE_LEVEL",
                "add_prefix_space": False,
            },
        )

    def test_build_tokenizer_info_serializes_raw_tiktoken(self):
        _, serializer = self.build_tokenizer_info(
            FakeBuildTokenizer({"a": 0}),
            is_tiktoken=True,
            is_byte_level=False,
        )

        _, metadata_json = serializer.call_args.args
        self.assertEqual(json.loads(metadata_json)["vocab_type"], "RAW")

    def test_build_tokenizer_info_serializes_sentencepiece(self):
        result, serializer = self.build_tokenizer_info(
            FakeBuildTokenizer({"a": 0, "<0x0A>": 1}),
            stop_token_ids=[1],
            is_sentencepiece=True,
        )

        self.assertEqual(result, "serialized")
        encoded_vocab, metadata_json = serializer.call_args.args
        self.assertEqual(encoded_vocab, ["a", "<0x0A>"])
        self.assertEqual(
            json.loads(metadata_json),
            {
                "vocab_size": 2,
                "stop_token_ids": [1],
                "vocab_type": "BYTE_FALLBACK",
                "add_prefix_space": True,
            },
        )

    def test_build_tokenizer_info_rejects_unsupported_tokenizer(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tokenizer type"):
            self.build_tokenizer_info(FakeBuildTokenizer({"a": 0}))

    def test_build_tokenizer_info_rejects_empty_stop_token_ids(self):
        with self.assertRaisesRegex(ValueError, "stop_token_ids cannot be empty"):
            self.build_tokenizer_info(
                FakeBuildTokenizer({"a": 0}),
                stop_token_ids=[],
                is_fast=True,
            )


if __name__ == "__main__":
    unittest.main()
