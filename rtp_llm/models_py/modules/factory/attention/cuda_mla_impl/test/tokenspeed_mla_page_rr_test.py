"""TokenSpeed 0.2.11 Page-RR masks and packed-query integration on CUDA."""

import math
import unittest
from itertools import accumulate
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.page_rr_mla_metadata import (
    PageRRMlaMetadata,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.tokenspeed_mla_page_rr import (
    tokenspeed_mla_page_rr_decode,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/TokenSpeed MLA")
class TokenspeedMlaPageRRTest(unittest.TestCase):
    def test_target_verify_owner_boundaries(self):
        for rank in (0, 1, 15):
            with self.subTest(rank=rank):
                self.check_case(torch.float8_e4m3fn, 4, rank)

    def test_q1_and_bf16_local_length_fallback(self):
        for dtype, queries in (
            (torch.float8_e4m3fn, 1),
            (torch.bfloat16, 1),
            (torch.bfloat16, 4),
        ):
            with self.subTest(dtype=dtype, queries=queries):
                self.check_case(dtype, queries, 1)

    def test_q1_update_and_virtual_rows(self):
        for dtype in (torch.float8_e4m3fn, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.check_case(dtype, 1, 15, mtp_update=True)

    def test_fixed_update_and_graph_capacity(self):
        for rank in (0, 1, 15):
            with self.subTest(rank=rank):
                self.check_case(torch.float8_e4m3fn, 4, rank, mtp_update=True)
        self.check_case(torch.bfloat16, 4, 1, mtp_update=True)

    def test_nonreplicated_head_major_query_fallback(self):
        # This is the layout returned by the unchanged head-axis AllGather;
        # distributed communication itself is covered by its own tests.
        self.check_case(torch.float8_e4m3fn, 4, 1, head_major=True)

    def check_case(
        self,
        dtype,
        queries,
        rank,
        head_major=False,
        mtp_update=False,
    ):
        page, world, heads, dim, value = 128, 16, 96, 576, 512
        scale = 192**-0.5
        ends = (1, 127, 128, 129, 1919, 1920, 2047, 2048, 2049)
        lengths = [queries] * len(ends)
        if mtp_update:
            ends, lengths = (128, 1, 129, 1920, 1), [queries, 0, queries, queries, 0]
        batch = len(ends)
        physical = batch * queries
        rows = [
            (request, offset) if offset < length else (None, 0)
            for request, length in enumerate(lengths)
            for offset in range(queries)
        ]

        def local_length(end):
            # Enumerate global token ownership, independently of kernel math.
            return sum((position // page) % world == rank for position in range(end))

        capacity = max(local_length(end + max(lengths) + 1) for end in ends)
        width = max(1, math.ceil(capacity / page))
        torch.manual_seed(20261028)
        query = torch.randn((physical, heads, dim), device="cuda").to(dtype)
        if head_major:
            query = query.transpose(0, 1).contiguous().transpose(0, 1)
        cache = torch.randn((batch * width + 1, page, dim), device="cuda").to(dtype)
        storage = torch.zeros((batch, width + 5), dtype=torch.int32, device="cuda")
        table = storage[:, :width]
        table.copy_(
            torch.arange(1, batch * width + 1, dtype=torch.int32, device="cuda").view(
                batch, width
            )
        )
        if mtp_update:
            # C++ fills virtual request tables with the reserved null block.
            for request, length in enumerate(lengths):
                if not length:
                    table[request].zero_()
        prefixes = torch.tensor(
            [end - 1 for end in ends], dtype=torch.int32, device="cuda"
        )
        inputs = SimpleNamespace(
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            physical_token_count=physical,
            is_target_verify=queries > 1 and not mtp_update,
            is_mtp_draft_update=mtp_update,
            is_cuda_graph=False,
            prefix_lengths_device=prefixes,
            sequence_lengths_plus_1_device=prefixes + 1,
            kv_cache_kernel_block_id_device=table,
            cu_seqlens_device=torch.tensor(
                list(accumulate([0] + lengths)), dtype=torch.int32, device="cuda"
            ),
            valid_token_mask=torch.tensor(
                [request is not None for request, _ in rows], device="cuda"
            ),
        )
        metadata = PageRRMlaMetadata(page, page, world, rank)
        compact = queries > 1 and dtype == torch.float8_e4m3fn and not head_major
        metadata.prepare_decode(inputs, compact_queries=compact)
        workspace = torch.empty(64 * 1024**2, dtype=torch.int8, device="cuda")
        shape = (batch, queries, heads, dim)

        def forward():
            metadata.prepare_decode(
                inputs,
                forbid_realloc=True,
                compact_queries=compact,
            )
            return tokenspeed_mla_page_rr_decode(
                query.view(shape),
                cache,
                workspace,
                value,
                dim - value,
                metadata.query_block_tables,
                metadata.local_causal_lens,
                width * page,
                scale,
                local_seq_lens=metadata.local_seq_lens,
                **(
                    {
                        "global_causal_lens": metadata.global_causal_lens,
                        "cp_world": world,
                        "cp_rank": rank,
                        "cp_interleave_size": page,
                    }
                    if compact
                    else {}
                ),
            )

        def check(outputs, increment):
            expected_lengths, expected_output, expected_lse = [], [], []
            for row, (request, offset) in enumerate(rows):
                length = (
                    0
                    if request is None
                    else local_length(ends[request] + increment + offset)
                )
                expected_lengths.append(length)
                if not length:
                    expected_output.append(torch.zeros((heads, value), device="cuda"))
                    expected_lse.append(torch.full((heads,), -math.inf, device="cuda"))
                    continue
                pages = table[request, : math.ceil(length / page)].long()
                kv = cache[pages].reshape(-1, dim)[:length].float()
                scores = query[row].float() @ kv.T * scale
                expected_output.append(scores.softmax(-1) @ kv[:, :value])
                expected_lse.append(torch.logsumexp(scores, -1) / math.log(2))
            self.assertEqual(
                metadata.local_causal_lens.view(-1).cpu().tolist(), expected_lengths
            )
            if compact:
                self.assertEqual(
                    metadata.global_causal_lens.cpu().tolist(),
                    [
                        end + increment + queries - 1 if length else 0
                        for end, length in zip(ends, lengths)
                    ],
                )
                torch.testing.assert_close(metadata.query_block_tables, table)
            else:
                self.assertIsNone(metadata.global_causal_lens)
            output, lse = (tensor.reshape(physical, heads, -1) for tensor in outputs)
            lse = lse.squeeze(-1)
            expected_output, expected_lse = torch.stack(expected_output), torch.stack(
                expected_lse
            )
            torch.testing.assert_close(
                output.float(), expected_output, atol=0.13, rtol=1e-5
            )
            torch.testing.assert_close(lse, expected_lse, atol=0.05, rtol=1e-4)
            empty = torch.tensor(expected_lengths, device="cuda") == 0
            self.assertTrue(bool((output[empty] == 0).all()))
            self.assertTrue(bool(torch.isneginf(lse[empty]).all()))
            if mtp_update:
                inactive_rows = [
                    row for row, (request, _) in enumerate(rows) if request is None
                ]
                self.assertEqual(
                    metadata.positions_d[inactive_rows].cpu().tolist(),
                    [0] * len(inactive_rows),
                )
                self.assertEqual(
                    metadata.slot_mapping[inactive_rows].cpu().tolist(),
                    [-1] * len(inactive_rows),
                )
                # Acceptance selects one candidate per live request; it does
                # not shorten the fixed Q rows passed to draft attention.
                accepted = [
                    (request + increment) % queries + 1 if length else 0
                    for request, length in enumerate(lengths)
                ]
                selected = [
                    request * queries + count - 1
                    for request, count in enumerate(accepted)
                    if count
                ]
                torch.testing.assert_close(
                    output[selected].float(),
                    expected_output[selected],
                    atol=0.13,
                    rtol=1e-5,
                )

        for _ in range(3):
            eager = forward()
        check(eager, 0)
        inputs.is_cuda_graph = True
        if mtp_update:
            live_cu = inputs.cu_seqlens_device.clone()
            live_mask = inputs.valid_token_mask.clone()
            inputs.cu_seqlens_device.zero_()
            # Query lengths alone must suppress empty captured requests.
            inputs.valid_token_mask.fill_(True)
            metadata.query_block_tables.fill_(-7)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        graph.replay()
        torch.cuda.synchronize()
        if mtp_update:
            self.assertTrue(bool((metadata.local_causal_lens == 0).all()))
            self.assertTrue(bool((metadata.positions_d == 0).all()))
            self.assertTrue(bool((metadata.slot_mapping == -1).all()))
            if compact:
                self.assertTrue(bool((metadata.local_seq_lens == 0).all()))
                self.assertTrue(bool((metadata.global_causal_lens == 0).all()))
                torch.testing.assert_close(metadata.query_block_tables, table)
            self.assertTrue(bool((captured[0] == 0).all()))
            self.assertTrue(bool(torch.isneginf(captured[1]).all()))
            inputs.cu_seqlens_device.copy_(live_cu)
            inputs.valid_token_mask.copy_(live_mask)
            graph.replay()
            torch.cuda.synchronize()
        for actual, expected in zip(captured, eager):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if mtp_update:
            # SP padding can keep Q dummy input rows in cu_seqlens. Unlike
            # graph-capacity tails above, only the live device mask rejects
            # this virtual request once its query length is nonzero.
            inputs.input_lengths[-1] = queries
            inputs.cu_seqlens_device[-1] += queries
            graph.replay()
            torch.cuda.synchronize()
            check(captured, 0)
            inputs.input_lengths[-1] = 0
            inputs.cu_seqlens_device.copy_(live_cu)
        # Cross ownership boundaries using device data while keeping graph shape
        # and all metadata buffers fixed, including the global final CP bound.
        inputs.prefix_lengths_device.add_(1)
        inputs.sequence_lengths_plus_1_device.add_(1)
        graph.replay()
        torch.cuda.synchronize()
        check(captured, 1)


if __name__ == "__main__":
    unittest.main()
