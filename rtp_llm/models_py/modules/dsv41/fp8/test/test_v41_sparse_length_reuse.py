"""Sparse plan consumers must not recompute already-owned causal lengths."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_sparse_decode as sparse
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8


class LengthOps(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.add.Tensor, torch.ops.aten.floor_divide.default):
            self.ops.append(func)
        return func(*args, **(kwargs or {}))


class SparseLengthReuseCPU(unittest.TestCase):
    def setUp(self):
        self.positions = torch.tensor([31, 32, 33, 63, 64, 65], dtype=torch.int64)
        self.x = torch.empty(2, 3, 8, dtype=torch.bfloat16)
        self.qr = torch.empty(2, 3, 16, dtype=torch.bfloat16)
        self.q = torch.empty(6, 32 * 128, dtype=torch.bfloat16)
        self.weights = torch.empty(6, 32, dtype=torch.bfloat16)
        self.output = torch.empty(6, 512, dtype=torch.int32)
        keys = indexer.DecodeIndexerKeys(
            torch.empty(1, 128, 68, dtype=torch.uint8),
            torch.ones(2, 256, dtype=torch.int32), 32768, 128,
        )
        self.shared = {
            "global": {20: keys},
            "candidates": torch.zeros(6, 2048, dtype=torch.int32),
        }
        self.layer = SimpleNamespace(
            is_index_source=True, layer_id=28, kv_source_layer_id=20,
            index_n_heads=32, index_head_dim=128, index_topk=512,
            compress_ratio=1, decode_sparse_mqa=True, rope_head_dim=64,
            freqs_cis=torch.empty(66, 32, dtype=torch.complex64),
            v41_config=dict(candidate_source_layer_id=20, candidate_topk_blocks=2048,
                            candidate_block_size=8),
            _shared_attention=self.shared,
        )

    def call(self):
        return AttentionV41FP8._select_indices_decode(
            self.layer, self.x, self.qr, self.positions,
            index_q=self.q, head_weights=self.weights,
        )

    def test_cached_plan_does_not_launch_length_add_or_divide(self):
        plan = object()
        self.shared["decode_sparse_plan"] = plan
        with patch.object(sparse, "is_supported", return_value=True), \
             patch.object(sparse, "make_plan") as make, \
             patch.object(sparse, "select_decode", return_value=self.output) as select, \
             LengthOps() as count:
            self.assertIs(self.call(), self.output)
        make.assert_not_called()
        self.assertIs(select.call_args.args[5], plan)
        self.assertEqual(count.ops, [])

    def test_first_plan_adds_once_without_division_and_then_reuses(self):
        plan = object()
        with patch.object(sparse, "is_supported", return_value=True), \
             patch.object(sparse, "make_plan", return_value=plan) as make, \
             patch.object(sparse, "select_decode", return_value=self.output), \
             LengthOps() as count:
            self.call()
            self.call()
        self.assertEqual(count.ops, [torch.ops.aten.add.Tensor])
        make.assert_called_once()
        torch.testing.assert_close(make.call_args.args[1], self.positions + 1)
        self.assertIs(self.shared["decode_sparse_plan"], plan)

    def test_general_scorer_retains_compressed_causal_lengths(self):
        class ReachedScorer(Exception):
            pass

        for ratio, enabled in ((2, True), (1, False)):
            with self.subTest(ratio=ratio, sparse=enabled):
                self.layer.compress_ratio = ratio
                self.layer.decode_sparse_mqa = enabled
                with patch.object(indexer, "score_decode_indexer",
                                  side_effect=ReachedScorer) as score:
                    with self.assertRaises(ReachedScorer):
                        self.call()
                torch.testing.assert_close(
                    score.call_args.args[5],
                    ((self.positions + 1) // ratio).view(2, 3),
                )


if __name__ == "__main__":
    unittest.main()
