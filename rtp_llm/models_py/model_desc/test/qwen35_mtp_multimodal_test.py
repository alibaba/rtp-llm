import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.models_py.model_desc.qwen3_next_mtp_multimodal import (
    mtp_word_embedding,
)
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)


class Qwen35MtpMultimodalTest(unittest.TestCase):
    def test_text_only_keeps_embedding_call_unchanged(self):
        input_ids = torch.tensor([1, 2], dtype=torch.int32)
        inputs = SimpleNamespace(
            input_ids=input_ids,
            multimodal_inputs=SimpleNamespace(multimodal_features=[]),
        )
        embed_tokens = Mock(return_value=torch.ones((2, 2)))
        injector = Mock()

        result = mtp_word_embedding(embed_tokens, injector, inputs)

        self.assertIs(result, embed_tokens.return_value)
        embed_tokens.assert_called_once_with(input_ids)
        injector.assert_not_called()

    def test_shifted_visual_ids_receive_features_in_each_request(self):
        # C++ shifted tokens, mask and locations together; Python must not shift again.
        input_ids = torch.tensor([1000, 1001, 2, 3, 5, 1002, 6, 7])
        mask = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1], dtype=torch.int32)
        features = [
            torch.tensor([[20.0, 21.0], [22.0, 23.0]]),
            torch.tensor([[30.0, 31.0]]),
        ]
        locs = torch.tensor([0, 5], dtype=torch.int32)
        inputs = SimpleNamespace(
            input_ids=input_ids,
            embedding_inputs=SimpleNamespace(text_tokens_mask=mask),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=features, mm_features_locs=locs
            ),
            attention_inputs={
                "full": SimpleNamespace(input_lengths=torch.tensor([4, 4])),
                "linear": SimpleNamespace(input_lengths=torch.tensor([4, 4])),
            },
        )
        weight = torch.arange(20, dtype=torch.float32).reshape(10, 2)

        def embed_tokens(ids, *, text_tokens_mask):
            safe_ids = torch.where(text_tokens_mask.bool(), ids, 0)
            # An unshifted mask would leave out-of-vocabulary ids at this lookup.
            return weight[safe_ids].clone()

        result = mtp_word_embedding(
            embed_tokens, MultimodalEmbeddingInjector(), inputs
        )

        expected = torch.tensor(
            [
                [20, 21],
                [22, 23],
                [4, 5],
                [6, 7],
                [10, 11],
                [30, 31],
                [12, 13],
                [14, 15],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result, expected)
        self.assertEqual(mask.tolist(), [0, 0, 1, 1, 1, 0, 1, 1])
        self.assertEqual(locs.tolist(), [0, 5])
        self.assertEqual(input_ids.tolist(), [1000, 1001, 2, 3, 5, 1002, 6, 7])

    def test_multimodal_input_requires_mask_before_embedding(self):
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1000, 2]),
            embedding_inputs=SimpleNamespace(text_tokens_mask=None),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[torch.ones((1, 2))]
            ),
        )
        embed_tokens = Mock()
        with self.assertRaisesRegex(ValueError, "requires text_tokens_mask"):
            mtp_word_embedding(embed_tokens, Mock(), inputs)
        embed_tokens.assert_not_called()

    def test_cp_adjacent_feature_runs_are_injected_without_merging(self):
        # One global image was shifted, then CP selected [0:4, 12:16].
        # Its two local feature runs touch but remain separate tensors.
        features = [
            torch.tensor([[20.0, 21.0], [22.0, 23.0], [24.0, 25.0]]),
            torch.tensor([[42.0, 43.0]]),
        ]
        mask = torch.tensor([1, 0, 0, 0, 0, 1, 1, 1], dtype=torch.int32)
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 1000, 1001, 1002, 1011, 2, 3, 4]),
            embedding_inputs=SimpleNamespace(text_tokens_mask=mask),
            multimodal_inputs=SimpleNamespace(
                multimodal_features=features,
                mm_features_locs=torch.tensor([1, 4], dtype=torch.int32),
            ),
        )
        weight = torch.arange(20, dtype=torch.float32).reshape(10, 2)

        def embed_tokens(ids, *, text_tokens_mask):
            return weight[torch.where(text_tokens_mask.bool(), ids, 0)].clone()

        result = mtp_word_embedding(embed_tokens, MultimodalEmbeddingInjector(), inputs)
        expected = torch.tensor(
            [[2, 3], [20, 21], [22, 23], [24, 25], [42, 43], [4, 5], [6, 7], [8, 9]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result, expected)

    def test_rejects_mask_length_mismatch_before_embedding(self):
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            embedding_inputs=SimpleNamespace(text_tokens_mask=torch.ones(3)),
            multimodal_inputs=SimpleNamespace(multimodal_features=[torch.ones(1, 2)]),
        )
        embed_tokens = Mock()
        with self.assertRaisesRegex(ValueError, "mask length"):
            mtp_word_embedding(embed_tokens, Mock(), inputs)
        embed_tokens.assert_not_called()


if __name__ == "__main__":
    unittest.main()
