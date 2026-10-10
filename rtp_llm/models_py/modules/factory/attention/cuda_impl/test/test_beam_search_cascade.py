import math
import unittest

import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

from rtp_llm.models_py.modules.factory.attention.cuda_impl.beam_search_cascade import (
    BeamSearchCascadeAttention,
)


class BeamSearchCascadeTest(unittest.TestCase):
    def test_small_graph_buckets_with_reused_workspace(self):
        torch.manual_seed(20261010)
        for batch in (128, 256, 512, 1024):
            with self.subTest(batch=batch):
                # Exercise dirty pool storage and different graph launch sizes.
                work = torch.full(
                    (128 * 1024**2,), 255, dtype=torch.uint8, device="cuda"
                )
                op = BeamSearchCascadeAttention(
                    batch, 16, 8, 128, 16, 8, 32, torch.bfloat16, True, work
                )
                table = torch.zeros(batch, 32, dtype=torch.int32, pin_memory=True)
                table[: batch // 2, :31] = torch.arange(31, dtype=torch.int32)
                table[batch // 2 :, :31] = torch.arange(31, dtype=torch.int32) + 32
                table[:, 31] = torch.arange(batch, dtype=torch.int32) + 64
                lengths = torch.full((batch,), 501, dtype=torch.int32, pin_memory=True)
                q = torch.randn(batch, 16, 128, dtype=torch.bfloat16, device="cuda")
                kv = torch.randn(
                    batch + 64, 2, 8, 16, 128, dtype=q.dtype, device="cuda"
                )
                op.plan(table, lengths)
                for _ in range(3):
                    op.run(q, kv)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    result = op.run(q, kv)
                for _ in range(10):
                    graph.replay()
                torch.cuda.synchronize()
                reference_work = torch.zeros_like(work)
                reference = BatchDecodeWithPagedKVCacheWrapper(reference_work, "HND")
                reference.plan(
                    torch.arange(batch + 1, dtype=torch.int32) * 32,
                    table.flatten().cuda(),
                    lengths % 16 + 1,
                    16,
                    8,
                    128,
                    16,
                    q_data_type=q.dtype,
                    kv_data_type=q.dtype,
                )
                torch.testing.assert_close(
                    result, reference.run(q, kv), atol=0.004, rtol=0.02
                )

    def test_dynamic_groups_reuse_one_graph(self):
        torch.manual_seed(20261010)
        batch, page, dim, max_pages = 2048, 16, 128, 33
        for heads, kv_heads, dtype in (
            (8, 8, torch.bfloat16),
            (16, 8, torch.bfloat16),
            (16, 8, torch.float16),
        ):
            with self.subTest(heads=heads, dtype=dtype):
                work = torch.empty(128 * 1024**2, dtype=torch.uint8, device="cuda")
                op = BeamSearchCascadeAttention(
                    batch,
                    heads,
                    kv_heads,
                    dim,
                    page,
                    8,
                    max_pages,
                    dtype,
                    True,
                    work,
                )
                q = torch.randn(batch, heads, dim, device="cuda", dtype=dtype)
                # Full prefix pages are shared; every beam owns two private pages.
                kv = torch.randn(
                    8 * max_pages + 2 * batch + 1,
                    2,
                    kv_heads,
                    page,
                    dim,
                    device="cuda",
                    dtype=dtype,
                )
                normal = BatchDecodeWithPagedKVCacheWrapper(
                    torch.zeros_like(work), "HND", use_tensor_cores=False
                )
                graph = torch.cuda.CUDAGraph()
                cases = [
                    ([512] * 4, 500, "shared"),
                    ([512, 512, 512, 256, 256], 511, "shared"),
                    ([256] * 8, 512, "shared"),
                    ([512, 256], 257, "shared"),  # padded query rows
                    ([512] * 4, 31, "shared"),  # too-short prefix -> full tail
                    ([512] * 4, 500, "distinct"),
                    ([512] * 4, 500, "partial"),  # diverging later pages
                    ([512, 512, 512, 256, 256], 500, "mixed_lengths"),
                    ([512] * 4, 500, "shared"),  # original partition again
                ]
                for ci, (widths, history, layout) in enumerate(cases):
                    table = torch.zeros(
                        (batch, max_pages), dtype=torch.int32, pin_memory=True
                    )
                    lengths = torch.zeros(batch, dtype=torch.int32, pin_memory=True)
                    first = 0
                    shared = history // page
                    for group, width in enumerate(widths):
                        for row in range(first, first + width):
                            ids = (
                                torch.arange(shared, dtype=torch.int32)
                                + group * max_pages
                                + 1
                            )
                            table[row, :shared] = ids
                            table[row, shared] = 8 * max_pages + 2 * row + 1
                            if layout == "distinct":
                                table[row, 0] = 8 * max_pages + 2 * row + 2
                            elif layout == "partial":
                                table[row, 10] = 8 * max_pages + 2 * row + 2
                        lengths[first : first + width] = (
                            history - group * 21
                            if layout == "mixed_lengths"
                            else history
                        )
                        first += width
                    op.plan(table, lengths)
                    torch.cuda.synchronize()
                    if ci == 0:
                        for _ in range(3):
                            op.run(q, kv)
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph):
                            output = op.run(q, kv)
                    graph.replay()
                    torch.cuda.synchronize()
                    counts = (lengths + page) // page
                    ptr = torch.cat(
                        (torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int())
                    )
                    indices = table[
                        torch.arange(max_pages)[None, :] < counts[:, None]
                    ].cuda()
                    normal.plan(
                        ptr,
                        indices,
                        lengths % page + 1,
                        heads,
                        kv_heads,
                        dim,
                        page,
                        q_data_type=dtype,
                        kv_data_type=dtype,
                    )
                    reference = normal.run(q, kv)
                    torch.testing.assert_close(output, reference, atol=0.004, rtol=0.02)
                    # Independent FP32 softmax reference for first/last valid query.
                    for row in (0, first - 1):
                        data = kv[table[row, : counts[row]].long()]
                        data = data.permute(1, 2, 0, 3, 4).reshape(2, kv_heads, -1, dim)
                        data = data[:, :, : int(lengths[row]) + 1].float()
                        k = data[0].repeat_interleave(heads // kv_heads, 0)
                        v = data[1].repeat_interleave(heads // kv_heads, 0)
                        weights = torch.einsum(
                            "hd,hld->hl", q[row].float(), k
                        ) / math.sqrt(dim)
                        expected = torch.einsum("hl,hld->hd", weights.softmax(-1), v)
                        torch.testing.assert_close(
                            output[row].float(), expected, atol=0.004, rtol=0.02
                        )
                    if layout == "distinct" or history < 128:
                        self.assertEqual(op.shared_tokens, 0)
                    else:
                        self.assertGreater(op.shared_tokens, 0)

    def test_rejects_page_capacity_overflow(self):
        work = torch.empty(128 * 1024**2, dtype=torch.uint8, device="cuda")
        op = BeamSearchCascadeAttention(
            32, 8, 8, 128, 16, 2, 8, torch.bfloat16, True, work
        )
        with self.assertRaisesRegex(ValueError, "captured capacity"):
            op.plan(
                torch.zeros(32, 8, dtype=torch.int32),
                torch.full((32,), 128, dtype=torch.int32),
            )


if __name__ == "__main__":
    unittest.main()
