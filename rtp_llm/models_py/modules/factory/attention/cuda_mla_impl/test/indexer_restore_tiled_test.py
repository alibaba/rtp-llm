"""Byte-exact tiled restore and fallback tests, independent of communications."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.fused_indexer_cp_assemble import (
    _restore_packed_indexer_k,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TiledIndexerRestoreTest(unittest.TestCase):
    def test_shapes_indices_tails_and_fallback(self):
        torch.manual_seed(183)
        for n in (0, 1, 31, 4095, 4096, 8193, 110195, 321430):
            for kd, sd in ((128, 1), (128, 4), (64, 1), (127, 3)):
                for dtype in (torch.int32, torch.int64):
                    with self.subTest(n=n, kd=kd, sd=sd, dtype=dtype):
                        data = torch.randint(
                            0, 256, (n + 37, kd + sd), device="cuda", dtype=torch.uint8
                        )
                        idx = torch.randint(0, n + 37, (n,), device="cuda", dtype=dtype)
                        ref = data[idx.long()]
                        for flag in ("0", "1"):
                            k = torch.full(
                                (n + 2, kd), 171, device="cuda", dtype=torch.uint8
                            )
                            s = torch.full(
                                (n + 2, sd), 171, device="cuda", dtype=torch.uint8
                            )
                            with patch.dict(
                                "os.environ", {"RTP_LLM_CP_TILED_INDEXER_RESTORE": flag}
                            ):
                                _restore_packed_indexer_k(data, k[1:-1], s[1:-1], idx)
                            self.assertTrue(torch.equal(k[1:-1], ref[:, :kd]))
                            self.assertTrue(torch.equal(s[1:-1], ref[:, kd:]))
                            self.assertTrue(bool((k[[0, -1]] == 171).all()))
                            self.assertTrue(bool((s[[0, -1]] == 171).all()))

    def test_graph_replay_refreshes_input_and_indices(self):
        n = 8193
        data = torch.zeros(n + 37, 132, device="cuda", dtype=torch.uint8)
        idx = torch.arange(n, device="cuda", dtype=torch.int64)
        k = torch.empty(n, 128, device="cuda", dtype=torch.uint8)
        s = torch.empty(n, 4, device="cuda", dtype=torch.uint8)
        with patch.dict("os.environ", {"RTP_LLM_CP_TILED_INDEXER_RESTORE": "1"}):
            _restore_packed_indexer_k(data, k, s, idx)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _restore_packed_indexer_k(data, k, s, idx)
        for _ in range(5):
            data.random_(0, 256)
            idx.random_(0, n + 37)
            graph.replay()
            ref = data[idx]
            self.assertTrue(torch.equal(k, ref[:, :128]))
            self.assertTrue(torch.equal(s, ref[:, 128:]))
