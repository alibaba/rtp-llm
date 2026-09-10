"""T24 platform-neutral EP strategy and decode-graph contracts.

These tests deliberately do not initialize distributed, DeepEP, Mega, PPU, or
CUDA.  They prove only selection policy and ownership invariants; a passing
test is not DP8 acceptance.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import torch
from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
    DSv4DecodeFmhaImpl,
    DSv4DecodeFmhaImplConfig,
)
from rtp_llm.models_py.modules.dsv4.fp8 import indexer


class T24StrategyAndGraphContractTest(unittest.TestCase):
    def test_decode_topk_capture_miss_is_fail_closed(self):
        # A graph capture must never allocate its workspace lazily.
        saved_cache = dict(indexer._decode_topk_workspace_cache)
        try:
            indexer._decode_topk_workspace_cache.clear()
            with mock.patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "warmed before graph capture"
                ):
                    indexer._get_decode_topk_workspace(torch.device("cuda"))
        finally:
            indexer._decode_topk_workspace_cache.clear()
            indexer._decode_topk_workspace_cache.update(saved_cache)

    def test_each_graph_instance_owns_distinct_metadata(self):
        cfg = DSv4DecodeFmhaImplConfig(
            max_batch_size=2,
            q_len=1,
            window_size=8,
            head_dim=32,
            max_seq_len=64,
            compress_ratios=[4, 128],
            index_topk=4,
        )
        instances = [
            DSv4DecodeFmhaImpl(cfg, device=torch.device("cpu")) for _ in range(4)
        ]
        pointers = [impl.metadata.start_pos.data_ptr() for impl in instances]
        self.assertEqual(len(set(pointers)), len(pointers))
        before = [impl.metadata.start_pos.clone() for impl in instances]
        instances[2].prepare_cuda_graph(
            type(
                "Attn",
                (),
                {"sequence_lengths": torch.tensor([11, 13], dtype=torch.int32)},
            )()
        )
        self.assertTrue(
            torch.equal(
                instances[2].metadata.start_pos,
                torch.tensor([11, 13], dtype=torch.int32),
            )
        )
        with mock.patch.object(
            instances[2], "prepare", wraps=instances[2].prepare
        ) as prepare:
            instances[2].prepare_cuda_graph(
                type(
                    "Attn",
                    (),
                    {"sequence_lengths": torch.tensor([11, 13], dtype=torch.int32)},
                )()
            )
            self.assertTrue(prepare.call_args.kwargs["forbid_realloc"])
        for i, impl in enumerate(instances):
            if i != 2:
                self.assertTrue(torch.equal(impl.metadata.start_pos, before[i]))


if __name__ == "__main__":
    unittest.main()
