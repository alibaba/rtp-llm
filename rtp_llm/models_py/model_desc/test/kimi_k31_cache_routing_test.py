import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.ops import HybridAttentionType


class RecordingLayer(nn.Module):
    def __init__(self, sparse):
        super().__init__()
        self.layer_type = HybridAttentionType.NONE
        self.attention = SimpleNamespace(indexer=object() if sparse else None)
        self.received = None

    def forward(self, hidden, anchors, fmha, cache, inputs, metadata, valid_mask):
        self.received = (fmha, cache, inputs)
        return hidden


class TaggedCache:
    def __init__(self):
        self.full = SimpleNamespace(tag="full")
        self.indexer = SimpleNamespace(tag="indexer_kv")

    def get_layer_cache_groups(self, index):
        # Deliberately reverse sparse topology order: routing must use tags.
        return [self.full] if index == 0 else [self.indexer, self.full]

    def get_layer_cache(self, index):
        if index != 0:
            raise RuntimeError("a sparse layer cannot use the sole-cache API")
        return self.full


class KimiK31CacheRoutingTest(unittest.TestCase):
    def test_forward_keeps_dense_and_sparse_group_ownership(self):
        model = object.__new__(KimiK3Model)
        nn.Module.__init__(model)
        dense, sparse = RecordingLayer(False), RecordingLayer(True)
        model.layers = nn.ModuleList([dense, sparse])
        model.kv_cache = cache = TaggedCache()
        model.tp_size, model.tp_rank = 1, 0
        model.num_blocks = 0
        model.use_paged_conv_prefill = False
        full_inputs = SimpleNamespace(
            is_prefill=False, is_target_verify=False, valid_token_mask=None
        )
        index_inputs = object()
        inputs = SimpleNamespace(
            input_ids=torch.zeros(2, dtype=torch.int64),
            attention_inputs={"full": full_inputs, "indexer_kv": index_inputs},
        )
        full_impl, index_impl = object(), object()
        model._forward_layers(
            torch.zeros((2, 8)), inputs,
            {"full": full_impl, "indexer_kv": index_impl},
        )
        self.assertEqual(dense.received, (full_impl, cache.full, full_inputs))
        self.assertEqual(sparse.received, (
            {"full": full_impl, "indexer_kv": index_impl},
            {"full": cache.full, "indexer_kv": cache.indexer}, full_inputs,
        ))


if __name__ == "__main__":
    unittest.main()
