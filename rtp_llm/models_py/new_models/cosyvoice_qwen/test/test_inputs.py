"""CosyVoice external feature path using the same injector as the serving model."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.models_py.new_models.cosyvoice_qwen.inputs import embed_external_features


class CosyVoiceInputsTest(unittest.TestCase):
    def setUp(self):
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.arange(60, dtype=torch.float).reshape(15, 4)
        )
        self.injector = MultimodalEmbeddingInjector()

    def test_features_and_text_in_one_sequence(self):
        inputs = SimpleNamespace(
            input_ids=torch.tensor([14, -200, -200, 6]),
            embedding_inputs=SimpleNamespace(
                text_tokens_mask=torch.tensor([1, 0, 0, 1])
            ),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[torch.full((2, 4), -1.0)],
                mm_features_locs=torch.tensor([1]),
            ),
        )
        actual = embed_external_features(self.embedding, self.injector, inputs)
        expected = torch.cat(
            [
                self.embedding.weight[14:15],
                torch.full((2, 4), -1.0),
                self.embedding.weight[6:7],
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_text_only_and_empty_mask(self):
        ids = torch.tensor([13, 0])
        for mask in (None, torch.empty(0, dtype=torch.bool)):
            inputs = SimpleNamespace(
                input_ids=ids, embedding_inputs=SimpleNamespace(text_tokens_mask=mask)
            )
            torch.testing.assert_close(
                embed_external_features(self.embedding, self.injector, inputs),
                self.embedding(ids),
                rtol=0,
                atol=0,
            )

    def test_bad_mask_or_location_not_silently_accepted(self):
        inputs = SimpleNamespace(
            input_ids=torch.tensor([4, 8]),
            embedding_inputs=SimpleNamespace(text_tokens_mask=torch.ones(1)),
        )
        with self.assertRaisesRegex(ValueError, "must match input_ids"):
            embed_external_features(self.embedding, self.injector, inputs)
        inputs.embedding_inputs = None
        inputs.multimodal_inputs = SimpleNamespace(
            multimodal_features=[torch.zeros(2, 4)], mm_features_locs=torch.tensor([1])
        )
        with self.assertRaises(IndexError):
            embed_external_features(self.embedding, self.injector, inputs)
