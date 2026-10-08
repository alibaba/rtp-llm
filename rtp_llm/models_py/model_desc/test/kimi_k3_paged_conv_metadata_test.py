import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.ops import HybridAttentionType


class FakeKDAlayer(nn.Module):
    layer_type = HybridAttentionType.LINEAR

    def __init__(self):
        super().__init__()
        self.attention = SimpleNamespace(
            prefill=SimpleNamespace(use_paged_conv=True)
        )
        self.metadata = None

    def forward(self, hidden, anchors, fmha, cache, attention_inputs, metadata, valid_mask):
        self.metadata = metadata
        return hidden


class KimiK3PagedConvMetadataTest(unittest.TestCase):
    def run_model(self, prefix, cache):
        model = object.__new__(KimiK3Model)
        nn.Module.__init__(model)
        layer = FakeKDAlayer()
        model.layers = nn.ModuleList([layer])
        model.kv_cache = SimpleNamespace(get_layer_cache=lambda _: cache) if cache else None
        model.tp_size = 1
        model.tp_rank = 0
        model.num_blocks = 0
        model.use_paged_conv_prefill = True
        primary = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32),
            prefix_lengths=torch.tensor([prefix], dtype=torch.int32),
            valid_token_mask=None,
        )
        inputs = SimpleNamespace(input_ids=torch.zeros(64, dtype=torch.int64))
        classic = object()
        paged = object()
        with (
            patch("rtp_llm.models_py.model_desc.kimi_k3.get_primary_attention_inputs", return_value=primary),
            patch("rtp_llm.models_py.model_desc.kimi_k3.select_attention_inputs_for_layer", return_value=primary),
            patch("rtp_llm.models_py.model_desc.kimi_k3.prepare_causal_conv1d_metadata", return_value=classic) as classic_prepare,
            patch("rtp_llm.models_py.model_desc.kimi_k3.prepare_paged_short_conv_metadata", return_value=paged) as paged_prepare,
        ):
            model._forward_layers(torch.zeros((64, 8)), inputs, fmha_impl=object())
        return layer.metadata, classic, paged, classic_prepare, paged_prepare

    def test_aligned_paged_cache_does_not_prepare_unused_classic_metadata(self):
        metadata, _, paged, classic_prepare, paged_prepare = self.run_model(
            prefix=0, cache=object()
        )
        classic_prepare.assert_not_called()
        paged_prepare.assert_called_once()
        self.assertIsNone(metadata.prefill_conv1d_meta)
        self.assertIs(metadata.prefill_paged_conv_meta, paged)

    def test_unaligned_prefix_retains_classic_fallback_metadata(self):
        metadata, classic, paged, classic_prepare, paged_prepare = self.run_model(
            prefix=1, cache=object()
        )
        classic_prepare.assert_called_once()
        paged_prepare.assert_called_once()
        self.assertIs(metadata.prefill_conv1d_meta, classic)
        self.assertIs(metadata.prefill_paged_conv_meta, paged)

    def test_missing_cache_retains_classic_metadata(self):
        metadata, classic, _, classic_prepare, _ = self.run_model(prefix=0, cache=None)
        classic_prepare.assert_called_once()
        self.assertIs(metadata.prefill_conv1d_meta, classic)


if __name__ == "__main__":
    unittest.main()
