"""Integer preparation must match eager semantics, including padding and replay."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.fp8.decode._fused_prepare_meta_triton import (
    fill_state_pool_slots,
    fused_update_decode_meta_pure,
    try_snapshot_block_tables,
)
from rtp_llm.models_py.speculative._dspark_prepare_triton import (
    try_map_context_rows,
    try_prepare_proposal,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class BoundaryPrepareTest(unittest.TestCase):
    def test_commit_rows_with_empty_requests_and_gaps(self):
        for batch in (1, 3, 7, 32, 64, 65, 128, 257):
            for width in (1, 5, 6):
                lengths = torch.arange(batch, device="cuda", dtype=torch.int64) % (
                    width + 1
                )
                starts = (
                    torch.arange(batch, device="cuda", dtype=torch.int64) * width + 2
                )
                if batch > 1:
                    starts[1] = starts[0]  # searchsorted(right=True) tie handling
                ends = starts + lengths + 32000
                rows = batch * width + 3
                actual = try_map_context_rows(starts, lengths, ends, rows)
                index = torch.arange(rows, device="cuda")
                req = torch.searchsorted(starts, index, right=True) - 1
                safe = req.clamp(0, batch - 1)
                valid = (
                    (req >= 0)
                    & (index >= starts[safe])
                    & (index < starts[safe] + lengths[safe])
                )
                positions = ends[safe] - lengths[safe] + index - starts[safe]
                expected = (
                    torch.where(valid, req, -1).int(),
                    torch.where(valid, positions, -1).int(),
                )
                for a, e in zip(actual, expected):
                    self.assertTrue(torch.equal(a, e), (batch, width))

    def test_proposal_casts_padding_and_graph_replay(self):
        for batch in (0, 1, 7, 32, 64, 65, 128):
            for width in (1, 3, 5, 6):
                for dtype in (torch.int32, torch.int64):
                    ids = torch.arange(batch * width, device="cuda", dtype=dtype)
                    prefixes = torch.arange(batch + 1, device="cuda", dtype=dtype) * 100
                    if dtype == torch.int64:
                        prefixes += 2**32  # cast before int64 position addition
                    graph = torch.cuda.CUDAGraph()
                    try_prepare_proposal(ids, prefixes, width=width, noise_token_id=1)
                    with torch.cuda.graph(graph):
                        actual = try_prepare_proposal(
                            ids, prefixes, width=width, noise_token_id=1
                        )
                    for shift in (0, 7):
                        prefixes.add_(shift)
                        graph.replay()
                        p = prefixes[:batch].int()
                        expected_ids = torch.full(
                            (batch, width), 1, device="cuda", dtype=torch.int32
                        )
                        expected_ids[:, 0] = ids.reshape(batch, width)[:, 0].int()
                        expected = (
                            expected_ids,
                            p.long()[:, None] + torch.arange(width, device="cuda"),
                            p,
                            p > 0,
                        )
                        for a, e in zip(actual, expected):
                            self.assertTrue(
                                torch.equal(a, e), (batch, width, dtype, shift)
                            )

    def test_state_slots_strides_sentinels_and_wrap(self):
        for batch in (1, 7, 64, 65):
            for width in (1, 5, 6):
                for entries, tokens in ((2, 128), (16, 16), (128, 16384)):
                    table = (
                        torch.arange(
                            batch * 14, device="cuda", dtype=torch.int32
                        ).reshape(batch, 14)[:, ::2]
                        - 2
                    )
                    requests = torch.arange(batch, device="cuda").repeat_interleave(
                        width
                    )
                    positions = torch.arange(batch * width, device="cuda") * tokens - 1
                    output = torch.full(
                        (positions.numel() + 3,), -99, device="cuda", dtype=torch.int64
                    )
                    fill_state_pool_slots(
                        table, positions, requests, output, entries, tokens
                    )
                    block = table.long()[
                        requests, (positions // tokens) % table.shape[1]
                    ]
                    expected = torch.where(
                        block > 0, block * entries + positions % entries, -1
                    )
                    self.assertTrue(torch.equal(output[: positions.numel()], expected))
                    self.assertTrue((output[positions.numel() :] == -99).all().item())

    def test_page_snapshots_strides_padding_and_replay(self):
        for batch in (1, 7, 64, 65):
            sources, destinations = {}, {}
            for table in range(7):
                # Exercise both narrower/wider sources and noncontiguous views.
                rows = max(0, batch + table - 3)
                cols = table + 1
                sources[table] = torch.arange(
                    rows * cols * 2, device="cuda", dtype=torch.int32
                ).reshape(rows, cols * 2)[:, ::2]
                destinations[table] = torch.full(
                    (batch + 3, 10 * 2), -99, device="cuda", dtype=torch.int32
                )[:, ::2]
            graph = torch.cuda.CUDAGraph()
            self.assertTrue(try_snapshot_block_tables(destinations, sources, batch))
            with torch.cuda.graph(graph):
                self.assertTrue(try_snapshot_block_tables(destinations, sources, batch))
            pointers = [x.data_ptr() for x in destinations.values()]
            for shift in (1, 11):
                for value in sources.values():
                    value.add_(shift)
                expected = {key: value.clone() for key, value in destinations.items()}
                for key, src in sources.items():
                    dst = expected[key]
                    rows, cols = min(src.shape[0], dst.shape[0]), min(
                        src.shape[1], dst.shape[1]
                    )
                    dst[batch:].zero_()
                    dst[:rows, :cols].copy_(src[:rows, :cols])
                    dst[:rows, cols:].zero_()
                graph.replay()
                for key in sources:
                    self.assertTrue(
                        torch.equal(destinations[key], expected[key]), (batch, key)
                    )
                self.assertEqual(
                    pointers, [x.data_ptr() for x in destinations.values()]
                )

    def test_aux_metadata_partial_batch_and_replay(self):
        for batch in (1, 7, 64, 65):
            for width in (1, 5, 6):
                for window, topk in ((16, 0), (128, 64)):
                    capacity = batch + 3

                    def buf(shape, dtype=torch.int32):
                        return torch.full(shape, -99, device="cuda", dtype=dtype)

                    meta = SimpleNamespace(
                        q_len_per_req=width,
                        window_size=window,
                        start_pos=buf((capacity,)),
                        position_ids=buf((capacity * width,)),
                        position_ids_long=buf((capacity * width,), torch.int64),
                        decode_seq_start_per_req=buf((capacity,), torch.int64),
                        topk_buffer_compressed=buf((capacity, width, topk)),
                        slot_mapping_compressed={},
                        compressed_lens={},
                        compressed_lens_per_token={},
                        topk_total_by_ratio={},
                        compressed_buffer_t_dim_per_ratio={},
                        slot_mapping_swa=buf((capacity * width,)),
                        topk_window_idxs=buf((capacity, width, window)),
                        swa_abs_idx=buf((capacity, width, window)),
                    )
                    starts = torch.arange(batch, device="cuda", dtype=torch.int32) * 127
                    fused_update_decode_meta_pure(meta, starts, 65536)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        fused_update_decode_meta_pure(meta, starts, 65536)
                    for shift in (0, 131):
                        starts.add_(shift)
                        graph.replay()
                        expected = (
                            starts[:, None]
                            + torch.arange(width, device="cuda", dtype=torch.int32)
                        ).flatten()
                        self.assertTrue(
                            torch.equal(meta.position_ids[: batch * width], expected)
                        )
                        self.assertTrue(
                            torch.equal(
                                meta.position_ids_long[: batch * width], expected.long()
                            )
                        )
                        self.assertTrue(
                            torch.equal(
                                meta.decode_seq_start_per_req[:batch], starts.long()
                            )
                        )
                        self.assertTrue(
                            (meta.topk_buffer_compressed[:batch] == -1).all().item()
                        )
                        for tensor in (meta.position_ids, meta.position_ids_long):
                            self.assertTrue(
                                (tensor[batch * width :] == -99).all().item()
                            )
                        self.assertTrue(
                            (meta.topk_buffer_compressed[batch:] == -99).all().item()
                        )


if __name__ == "__main__":
    unittest.main()
