import os
import unittest
from types import SimpleNamespace
from unittest import mock

from rtp_llm.models.bert import _configure_bert_uqi


class BertUqiConfigTest(unittest.TestCase):
    @staticmethod
    def make_config(vocab_size=1000):
        return SimpleNamespace(
            vocab_size=vocab_size,
            bert_uqi_config=SimpleNamespace(
                enabled=False,
                segment_token_id=-1,
                separator_token_id=-1,
            ),
        )

    def test_defaults_are_disabled(self):
        config = self.make_config(vocab_size=2)
        with mock.patch.dict(os.environ, {}, clear=True):
            _configure_bert_uqi(config)
        self.assertFalse(config.bert_uqi_config.enabled)
        self.assertEqual(config.bert_uqi_config.segment_token_id, 2)
        self.assertEqual(config.bert_uqi_config.separator_token_id, 102)

    def test_equivalent_alias_spellings_are_accepted(self):
        config = self.make_config()
        env = {
            "ENABLE_BERT_UQI_ATTENTION": "true",
            "USE_VISION_BERT_UQI_BLOCK_MASK": "1",
            "BERT_UQI_SEGMENT_TOKEN_ID": "7",
            "VISION_BERT_CLS_UQI_TOKEN_ID": "007",
            "BERT_UQI_SEPARATOR_TOKEN_ID": "9",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            _configure_bert_uqi(config)
        self.assertTrue(config.bert_uqi_config.enabled)
        self.assertEqual(config.bert_uqi_config.segment_token_id, 7)
        self.assertEqual(config.bert_uqi_config.separator_token_id, 9)

    def test_conflicting_aliases_are_rejected(self):
        config = self.make_config()
        env = {
            "ENABLE_BERT_UQI_ATTENTION": "1",
            "USE_VISION_BERT_UQI_BLOCK_MASK": "0",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "conflicts with legacy alias"):
                _configure_bert_uqi(config)

    def test_delimiters_must_be_distinct_and_in_vocabulary(self):
        config = self.make_config(vocab_size=10)
        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_BERT_UQI_ATTENTION": "1",
                "BERT_UQI_SEGMENT_TOKEN_ID": "9",
                "BERT_UQI_SEPARATOR_TOKEN_ID": "9",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "distinct"):
                _configure_bert_uqi(config)

        with mock.patch.dict(
            os.environ,
            {
                "ENABLE_BERT_UQI_ATTENTION": "1",
                "BERT_UQI_SEPARATOR_TOKEN_ID": "10",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "within the model vocabulary"):
                _configure_bert_uqi(config)


if __name__ == "__main__":
    unittest.main()
