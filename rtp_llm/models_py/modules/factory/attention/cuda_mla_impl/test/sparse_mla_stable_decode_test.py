import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
    SparseMlaFp8Op,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.test.sparse_mla_decode_op_test import (
    FP8KVCacheLayout,
    quantize_k_cache,
)


class SparseMlaStableDecodeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.q = torch.randn(13, 64, 576, device="cuda", dtype=torch.bfloat16)
        keys = torch.randn(34, 64, 1, 576, device="cuda", dtype=torch.bfloat16)
        self.cache = quantize_k_cache(keys, FP8KVCacheLayout.V32_FP8Sparse)
        self.indices = (
            torch.arange(2048, device="cuda", dtype=torch.int32)
            .expand(13, -1)
            .contiguous()
        )
        self.table = torch.arange(1, 34, device="cuda", dtype=torch.int32).unsqueeze(0)
        self.params = SimpleNamespace(
            batch_indice_h=torch.zeros(13, dtype=torch.int32),
            batch_indice_d=torch.zeros(13, device="cuda", dtype=torch.int32),
        )
        with patch.dict(os.environ, {"DETERMINISTIC_GEMM": "1"}):
            self.operator = SparseMlaFp8Op(
                64, 512, 64, 192, 64, 1.0, 2048, use_cuda_graph=True
            )
        self.operator.plan(self.params, self.table)

    def reference(self):
        return torch.cat(
            [
                flash_mla_with_kvcache(
                    q=row.unsqueeze(0).unsqueeze(0),
                    k_cache=self.cache.view(torch.uint8),
                    block_table=self.table,
                    cache_seqlens=None,
                    head_dim_v=512,
                    tile_scheduler_metadata=get_mla_metadata()[0],
                    is_fp8_kvcache=True,
                    indices=(indices + 64).reshape(1, 1, -1),
                    softmax_scale=self.operator.scale,
                )[0].squeeze(0)
                for row, indices in zip(self.q, self.indices)
            ],
            dim=0,
        )

    def forward(self):
        return self.operator.forward(self.q, self.cache, self.indices, layer_id=0)

    def test_prime_query_count_and_cuda_graph_replay_match_single_queries(self):
        expected = self.reference()
        for _ in range(3):
            torch.testing.assert_close(self.forward(), expected, rtol=0, atol=0)
        self.assertEqual(set(self.operator._chunk_sched_meta), {1, 4})
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.forward()
        pointers = {
            size: meta.tile_scheduler_metadata.data_ptr()
            for size, meta in self.operator._chunk_sched_meta.items()
        }
        for _ in range(5):
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.q.mul_(0.5)
        expected = self.reference()
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        self.assertEqual(
            pointers,
            {
                size: meta.tile_scheduler_metadata.data_ptr()
                for size, meta in self.operator._chunk_sched_meta.items()
            },
        )

    def test_graph_shape_guard_remains_enforced(self):
        changed = SimpleNamespace(batch_indice_h=torch.zeros(14, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "changed scheduler shape"):
            self.operator.plan(changed, self.table)


if __name__ == "__main__":
    unittest.main()
