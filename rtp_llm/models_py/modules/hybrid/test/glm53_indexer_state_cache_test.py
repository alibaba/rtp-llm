"""Keep GLM53 compressor state live when CUDA graphs omit publication tables."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import PoolBackedModule
from rtp_llm.models_py.modules.dsv4.fp8.glm53_compressor import CompressorFP8
from rtp_llm.models_py.modules.hybrid.glm53_indexer import Indexer


class _Cache:
    def __init__(self, state_tokens_per_block):
        self.state_tokens_per_block = state_tokens_per_block
        self.pools = {
            "indexer_kv": torch.zeros((64, 32 * 132), dtype=torch.uint8, device="cuda"),
            "indexer_state": torch.zeros(
                (16, 8 * 256), dtype=torch.float32, device="cuda"
            ),
        }

    def get_layer_cache(self, layer, tag):
        assert layer == 0
        return SimpleNamespace(kv_cache_base=self.pools[tag])

    def get_seq_size_per_block(self, tag):
        return self.state_tokens_per_block if tag == "indexer_state" else 128

    def get_kernel_seq_size_per_block(self, tag):
        return self.get_seq_size_per_block(tag)


class Glm53IndexerStateCacheTest(unittest.TestCase):
    def test_graph_replay_tracks_state_execution_pages(self):
        self.assertTrue(torch.cuda.is_available(), "This target requires its CUDA pool")
        # The 1024-token state rows cover the P/D CP=8 layout; KV pages still
        # cover 128 raw tokens. Width five exercises MTP across a state-row edge.
        for state_tpb in (128, 1024):
            for query_len in (1, 5):
                with self.subTest(state_tpb=state_tpb, query_len=query_len):
                    self._check_replay(state_tpb, query_len)

    def _check_replay(self, state_tpb, query_len):
        compressor = CompressorFP8.__new__(CompressorFP8)
        PoolBackedModule.__init__(compressor)
        compressor.compress_ratio = 4
        compressor.set_cp_ctx(None)
        indexer = Indexer.__new__(Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.layer_idx = 0
        indexer.index_head_dim = 128
        indexer.compressed_indexer = compressor

        state_pages = [[2, 3, 4], [5, 6, 7]]
        state_table = torch.tensor(state_pages, dtype=torch.int32, device="cuda")
        kv_columns = 3 * state_tpb // 128
        kv_table = torch.arange(
            1, 2 * kv_columns + 1, dtype=torch.int32, device="cuda"
        ).view(2, kv_columns)
        inputs = {
            "indexer_kv": SimpleNamespace(kv_cache_kernel_block_id_device=kv_table),
            # CudaGraphRunner supplies execution tables; CacheStore publication
            # tables are intentionally undefined in its captured model inputs.
            "indexer_state": SimpleNamespace(
                kv_cache_kernel_block_id_device=state_table,
                kv_cache_block_id_device=None,
            ),
        }
        indexer._bind_compressed_pools(_Cache(state_tpb), inputs)

        starts = [state_tpb - 3, 2 * state_tpb]
        positions_list = [
            start + offset for start in starts for offset in range(query_len)
        ]
        positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
        request_ids = torch.arange(
            2, dtype=torch.int64, device="cuda"
        ).repeat_interleave(query_len)
        sequence_starts = torch.tensor(starts, dtype=torch.int32, device="cuda")
        cu_sequence = torch.arange(
            0, 2 * query_len + 1, query_len, dtype=torch.int32, device="cuda"
        )

        def prepare():
            return compressor.prepare_metadata(
                positions,
                request_ids,
                has_prefix=True,
                is_batched=query_len > 1,
                seq_start_per_req=sequence_starts,
                cu_seq_per_req=cu_sequence,
            )

        def expected():
            return torch.tensor(
                [
                    state_pages[row // query_len][position // state_tpb] * 8
                    + position % 8
                    for row, position in enumerate(positions_list)
                ],
                dtype=torch.int64,
                device="cuda",
            )

        warmup = prepare()
        self.assertIsNotNone(
            warmup.state_slots,
            "Missing publication tables must not select the unbound warmup path",
        )
        torch.testing.assert_close(warmup.state_slots, expected())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = prepare()
        graph.replay()
        torch.testing.assert_close(captured.state_slots, expected())

        # A reused graph must follow a new request's allocation at the same row.
        state_pages[0][0] = 9
        state_table[0, 0] = 9
        graph.replay()
        torch.testing.assert_close(captured.state_slots, expected())


if __name__ == "__main__":
    unittest.main()
