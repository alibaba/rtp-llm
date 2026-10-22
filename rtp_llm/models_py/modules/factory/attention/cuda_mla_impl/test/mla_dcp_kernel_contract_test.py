"""Wire packing and base-2 merge, including Triton's token=1 specialization.

GPU kernels are real. The wire transfer is constructed locally, so this test
makes no claim about distributed all-to-all or PD request routing.
"""
import unittest

import torch

from rtp_llm.models_py.triton_kernels.mla.page_rr import _prepare_decode_metadata
from rtp_llm.models_py.triton_kernels.mla.dcp import (
    _combine_a2a,
    _pack_a2a,
)


class MlaDcpKernelContractTest(unittest.TestCase):
    def test_full_kv_verify_and_draft_update_page_boundaries_eager_and_graph(self):
        for kernel_page in (64, 128):
            for mtp_update in (False, True):
                with self.subTest(kernel_page=kernel_page, mtp_update=mtp_update):
                    self._check_full_kv_metadata(kernel_page, mtp_update)

    def _check_full_kv_metadata(self, kernel_page, mtp_update):
        device, rows, queries, width = "cuda", 8, 4, 192
        bases = torch.tensor([4094, 8190], dtype=torch.int32, device=device)
        capture = torch.zeros_like(bases)
        offsets = torch.tensor([0, 4, 5], dtype=torch.int32, device=device)
        mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool, device=device)
        table = torch.arange(1, 1 + 2 * width, dtype=torch.int32, device=device).view(2, width)
        query_tables = torch.empty(rows, width, dtype=torch.int32, device=device)
        positions = torch.empty(rows, dtype=torch.int32, device=device)
        slots = torch.empty(rows, dtype=torch.int64, device=device)
        lengths = torch.empty(rows, dtype=torch.int32, device=device)

        def forward():
            if not mtp_update:
                # Ordinary/Verify tables are expanded by prepare_decode before
                # the kernel; only ragged MTP updates copy tables in the kernel.
                query_tables.view(2, queries, width).copy_(table[:, None, :])
            _prepare_decode_metadata[(rows if mtp_update else 1,)](
                bases, capture, offsets, mask, table, query_tables, positions, slots, lengths,
                rows, width, width, queries=queries, page_size=4096,
                kernel_page_size=kernel_page, cp_size=1, cp_rank=0,
                multi_query_update=True, graph_decode=False, mtp_update=mtp_update,
                request_count=2, has_valid_mask=mtp_update, block=128,
            )

        def check():
            base_values = bases.tolist()
            expected_tables = []
            for row in range(rows):
                active = not mtp_update or row < 5
                batch, offset = divmod(row, queries)
                position = base_values[batch] + offset if active else 0
                self.assertEqual(positions[row].item(), position)
                self.assertEqual(lengths[row].item(), position + 1 if active else 0)
                expected_slot = (
                    table[batch, position // kernel_page].item() * kernel_page + position % kernel_page
                    if active else -1
                )
                self.assertEqual(slots[row].item(), expected_slot)
                expected_tables.append(table[batch] if active else torch.zeros_like(table[batch]))
            torch.testing.assert_close(query_tables, torch.stack(expected_tables), rtol=0, atol=0)

        # Materialize before capture; then change the live bases for replay.
        forward()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        bases.add_(2)
        graph.replay()
        torch.cuda.synchronize()
        check()

    def test_tp4_tp8_one_and_many_tokens_eager_and_graph(self):
        for cp in (4, 8):
            for tokens in (1, 4, 7):
                with self.subTest(cp=cp, tokens=tokens):
                    self.check_wire(cp, tokens)

    def check_wire(self, cp, tokens):
        local_heads, dim = 8, 512
        heads = local_heads * cp
        values = torch.arange(1, cp + 1, device="cuda", dtype=torch.float32)
        lengths = torch.arange(1, cp + 1, device="cuda", dtype=torch.int32)[:, None].expand(cp, tokens).clone()
        lengths[-1] = 0
        if tokens > 1:
            lengths[:, -1] = 0
        partial = [torch.full((tokens, heads, dim), float(i + 1), device="cuda", dtype=torch.bfloat16) for i in range(cp)]
        lse = [row.float().log2()[:, None].expand(tokens, heads).contiguous() for row in lengths]
        # Poison empty rows. Packing must publish exactly O=0/LSE=-inf.
        for i in range(cp):
            partial[i][lengths[i] == 0] = float("nan")
            lse[i][lengths[i] == 0] = float("nan")
        packed = [torch.empty((cp, tokens, local_heads, dim + 2), device="cuda", dtype=torch.bfloat16) for _ in range(cp)]
        received = torch.empty_like(packed[0])
        output = torch.empty((local_heads, tokens, dim), device="cuda", dtype=torch.bfloat16)
        destination = cp - 1

        def forward():
            for i in range(cp):
                _pack_a2a[(tokens, heads)](
                    partial[i], lse[i], lengths[i], packed[i], packed[i].view(torch.float32), tokens,
                    heads=heads, local_heads=local_heads, dim=dim, block=512,
                )
                received[i].copy_(packed[i][destination])
            _combine_a2a[(tokens, local_heads)](
                received, received.view(torch.float32), output, tokens,
                local_heads=local_heads, dim=dim, cp_size=cp, block=512,
            )

        def check():
            for i in range(cp):
                expected_lse = torch.where(lengths[i] > 0, lengths[i].float().log2(), -float("inf"))
                torch.testing.assert_close(
                    received[i].view(torch.float32)[..., -1],
                    expected_lse[:, None].expand(tokens, local_heads), rtol=0, atol=0,
                )
                self.assertTrue((received[i, lengths[i] == 0, :, :dim] == 0).all().item())
            denominator = lengths.float().sum(0)
            expected = torch.where(denominator > 0, (lengths.float() * values[:, None]).sum(0) / denominator, 0)
            expected = expected[None, :, None].expand(local_heads, tokens, dim).to(torch.bfloat16)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

        for _ in range(3):
            forward()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        # Replaying must use the new values, with the same graph metadata.
        for i in range(cp):
            partial[i][lengths[i] > 0] = float(2 * (i + 1))
        values.mul_(2)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        check()


if __name__ == "__main__":
    unittest.main()
