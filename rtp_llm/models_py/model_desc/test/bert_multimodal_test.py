"""Tests for BertModel multimodal-feature splicing.

BertModel.forward must inject vision features into hidden_states at the
positions advertised by bert_embedding_inputs.mm_features_locs, leaving all
other token positions untouched. The forward delegates to the shared
MultimodalEmbeddingInjector, so this test verifies the wiring end-to-end
without needing real model weights.
"""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.models_py.modules import MultimodalEmbeddingInjector


class _FmhaStub:
    def __init__(self):
        self.fmha_params = None


class BertMultimodalForwardTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        self.device = torch.device("cuda:0")
        torch.manual_seed(0)

    def _build_model(self, hidden_size: int, dtype: torch.dtype) -> BertModel:
        model = BertModel.__new__(BertModel)
        # Replace heavy submodules with identity-style stubs so forward exercises
        # only the multimodal splice path that we want to test.
        model.embed_tokens = lambda input_ids, *args, **kwargs: torch.zeros(
            input_ids.shape[0], hidden_size, device=self.device, dtype=dtype
        )
        model.pre_decoder_layernorm = lambda x: x
        model.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        model.layers = []
        model.layer_num = 0
        model.kv_cache = None
        model.prepare_fmha_impl = lambda inputs: _FmhaStub()
        return model

    @staticmethod
    def _make_inputs(seq_len: int, features, locs, device, dtype):
        bert_inputs = SimpleNamespace(
            combo_position_ids=torch.empty(0),
            position_encoding=torch.empty(0),
            combo_tokens_type_ids=torch.empty(0),
            token_type_embedding=torch.empty(0),
            input_embedding_scalar=1.0,
            multimodal_features=features,
            mm_features_locs=(
                torch.tensor(locs, device=device, dtype=torch.int32)
                if locs
                else torch.empty(0, device=device, dtype=torch.int32)
            ),
        )
        return SimpleNamespace(
            input_ids=torch.zeros(seq_len, device=device, dtype=torch.int64),
            bert_embedding_inputs=bert_inputs,
        )

    def test_features_overwrite_target_positions_only(self) -> None:
        hidden_size = 16
        seq_len = 12
        dtype = torch.float16
        model = self._build_model(hidden_size, dtype)

        feat0 = torch.randn(3, hidden_size, device=self.device, dtype=dtype)
        feat1 = torch.randn(2, hidden_size, device=self.device, dtype=dtype)
        locs = [1, 7]

        inputs = self._make_inputs(seq_len, [feat0, feat1], locs, self.device, dtype)
        out = model.forward(inputs).hidden_states

        expected = torch.zeros(seq_len, hidden_size, device=self.device, dtype=dtype)
        expected[1:4] = feat0
        expected[7:9] = feat1
        torch.testing.assert_close(out, expected)

    def test_empty_features_leaves_hidden_states_untouched(self) -> None:
        hidden_size = 8
        seq_len = 5
        dtype = torch.float16
        model = self._build_model(hidden_size, dtype)

        inputs = self._make_inputs(seq_len, [], [], self.device, dtype)
        out = model.forward(inputs).hidden_states

        torch.testing.assert_close(
            out,
            torch.zeros(seq_len, hidden_size, device=self.device, dtype=dtype),
        )

    def test_negative_loc_truncates_prefix(self) -> None:
        # loc < 0 means part of the feature is already in the reused KV prefix;
        # the injector must drop the head rows and place the tail at position 0.
        hidden_size = 4
        seq_len = 6
        dtype = torch.float16
        model = self._build_model(hidden_size, dtype)

        feat = torch.randn(4, hidden_size, device=self.device, dtype=dtype)
        inputs = self._make_inputs(seq_len, [feat], [-2], self.device, dtype)
        out = model.forward(inputs).hidden_states

        expected = torch.zeros(seq_len, hidden_size, device=self.device, dtype=dtype)
        expected[0:2] = feat[2:]
        torch.testing.assert_close(out, expected)

    def test_out_of_range_loc_raises(self) -> None:
        hidden_size = 4
        seq_len = 4
        dtype = torch.float16
        model = self._build_model(hidden_size, dtype)

        feat = torch.randn(3, hidden_size, device=self.device, dtype=dtype)
        inputs = self._make_inputs(seq_len, [feat], [2], self.device, dtype)
        with self.assertRaises(IndexError):
            model.forward(inputs)

    def test_locs_length_mismatch_raises(self) -> None:
        hidden_size = 4
        seq_len = 6
        dtype = torch.float16
        model = self._build_model(hidden_size, dtype)

        feat = torch.randn(2, hidden_size, device=self.device, dtype=dtype)
        inputs = self._make_inputs(seq_len, [feat], [0, 3], self.device, dtype)
        with self.assertRaises(ValueError):
            model.forward(inputs)


if __name__ == "__main__":
    unittest.main()
