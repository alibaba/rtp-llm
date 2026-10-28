import unittest

import torch

from rtp_llm.models_py.triton_kernels.attention.beam_search_cascade import (
    RocmBeamSearchCascadeAttention,
)


def make_case(
    rows=128,
    heads=16,
    kv_heads=8,
    dim=128,
    history=500,
    dtype=torch.bfloat16,
    linear_v=False,
    distinct=False,
    seed=20261021,
    page=16,
):
    torch.manual_seed(seed)
    max_pages = (history + page) // page
    prefix_pages = history // page
    groups = (rows + 127) // 128
    blocks = groups * max_pages + rows + 1
    k = torch.randn(blocks, kv_heads, page, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    x = 8
    kp = (
        k.reshape(blocks, kv_heads, page, dim // x, x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view_as(k)
    )
    if linear_v:
        vp = v.transpose(-1, -2).contiguous().view_as(v)
    else:
        vp = (
            v.reshape(blocks, kv_heads, page // x, x, dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
            .view_as(v)
        )
    kv = torch.stack((kp, vp), 1)
    table = torch.zeros((rows, max_pages), dtype=torch.int32)
    for row in range(rows):
        table[row, :prefix_pages] = (
            torch.arange(prefix_pages) + (row // 128) * max_pages
        )
        table[row, prefix_pages] = groups * max_pages + row + 1
    if distinct and prefix_pages:
        table[:, 0] = torch.arange(rows) + groups * max_pages + 1
    lengths = torch.full((rows,), history, dtype=torch.int32)
    q = torch.randn(rows, heads, dim, dtype=dtype, device="cuda")
    return q, kv, table, lengths, k, v


def reference(q, k, v, table, lengths, row):
    page = k.shape[2]
    count = (int(lengths[row]) + page) // page
    ids = table[row, :count].to(device=q.device, dtype=torch.long)
    kk = k[ids].transpose(0, 1).flatten(1, 2)[:, : int(lengths[row]) + 1].float()
    vv = v[ids].transpose(0, 1).flatten(1, 2)[:, : int(lengths[row]) + 1].float()
    kk = kk.repeat_interleave(q.shape[1] // kk.shape[0], 0)
    vv = vv.repeat_interleave(q.shape[1] // vv.shape[0], 0)
    probs = (
        torch.einsum("hd,hld->hl", q[row].float(), kk)
        .mul(q.shape[-1] ** -0.5)
        .softmax(-1)
    )
    return torch.einsum("hl,hld->hd", probs, vv)


class RocmBeamCascadeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.hip is None or not torch.cuda.is_available():
            raise RuntimeError("ROCm GPU required for cascade correctness tests")

    def check_case(self, case, linear_v=False, op=None):
        q, kv, table, lengths, k, v = case
        if op is None:
            op = RocmBeamSearchCascadeAttention(
                q.shape[0],
                q.shape[1],
                k.shape[1],
                q.shape[-1],
                k.shape[2],
                table.shape[1],
                q.dtype,
                q.device,
                linear_v=linear_v,
            )
        op.plan(table, lengths)
        out = op.run(q, kv)
        for row in sorted(set((0, q.shape[0] // 2, q.shape[0] - 1))):
            expected = reference(q, k, v, table, lengths, row)
            torch.testing.assert_close(
                out[row].float(), expected, atol=0.005, rtol=0.03
            )
        self.assertTrue(torch.isfinite(out).all().item())
        return op

    def test_wide_beams_mha_gqa_and_both_cache_layouts(self):
        for rows in (32, 128, 256, 1024):
            for heads, kv_heads in ((8, 8), (16, 8)):
                for dtype in (torch.bfloat16, torch.float16):
                    for linear_v in (False, True):
                        with self.subTest(
                            rows=rows, heads=heads, dtype=dtype, linear_v=linear_v
                        ):
                            self.check_case(
                                make_case(
                                    rows=rows,
                                    heads=heads,
                                    kv_heads=kv_heads,
                                    dtype=dtype,
                                    linear_v=linear_v,
                                ),
                                linear_v,
                            )

    def test_page_boundaries_short_and_unshared_histories(self):
        for history in (0, 7, 127, 128, 255, 511, 512):
            for distinct in (False, True):
                with self.subTest(history=history, distinct=distinct):
                    op = self.check_case(
                        make_case(rows=65, history=history, distinct=distinct)
                    )
                    if distinct or history < 128:
                        self.assertEqual(op.shared_tokens, 0)

    def test_checks_every_page_and_mixed_lengths(self):
        case = make_case(rows=256)
        q, kv, table, lengths, k, v = case
        table[3, 12] = table[3, -1]
        lengths[17:33] = 257
        op = self.check_case(case)
        self.assertEqual(int(op.cuts_host[0]), 12 * 16)
        self.assertEqual(int(op.cuts_host[1]), 31 * 16)

    def test_other_supported_head_shapes_and_ratios(self):
        for heads, kv_heads, dim in ((16, 4, 64), (16, 2, 128), (16, 8, 256)):
            with self.subTest(heads=heads, kv_heads=kv_heads, dim=dim):
                self.check_case(
                    make_case(rows=128, heads=heads, kv_heads=kv_heads, dim=dim)
                )

    def test_nonuniform_softmax_with_large_query_key_and_value_range(self):
        q, kv, table, lengths, k, v = make_case(rows=128)
        q.mul_(2)
        k.mul_(2)
        v.mul_(5)
        kv[:, 0].mul_(2)
        kv[:, 1].mul_(5)
        op = RocmBeamSearchCascadeAttention(128, 16, 8, 128, 16, 32, q.dtype, q.device)
        op.plan(table, lengths)
        output = op.run(q, kv)
        for row in (0, 64, 127):
            expected = reference(q, k, v, table, lengths, row)
            error = output[row].float() - expected
            self.assertLess((error.norm() / expected.norm()).item(), 0.01)
            self.assertLess(
                error.abs().max().item(), expected.abs().max().item() * 0.02 + 0.01
            )

    def test_unused_page_padding_does_not_need_valid_ids(self):
        case = make_case(rows=128)
        case[3][0] = 257
        case[2][0, 17:] = -1
        self.check_case(case)

    def test_other_page_sizes_use_triton_prefix(self):
        for page in (8, 32, 64):
            with self.subTest(page=page):
                op = self.check_case(make_case(page=page))
                self.assertFalse(op.native_prefix)
                self.assertGreater(op.shared_tokens, 0)

    def test_large_pool_fallback_keeps_native_planned_cuts(self):
        case = make_case(rows=256)
        q, kv, table, lengths, _, _ = case
        table[3, 12] = table[3, -1]
        op = RocmBeamSearchCascadeAttention(256, 16, 8, 128, 16, 32, q.dtype, q.device)
        self.assertTrue(op.native_prefix)
        # Exercise the same fallback without allocating a multi-GiB fixture.
        op.NATIVE_PREFIX_BYTE_LIMIT = kv.numel() * kv.element_size()
        self.check_case(case, op=op)
        self.assertEqual(int(op.cuts_host[0]), 12 * 16)
        self.assertEqual(int(op.cuts_host[1]), 31 * 16)

    def test_large_physical_page_ids_use_64_bit_offsets(self):
        q, kv, table, lengths, k, v = make_case(rows=32)
        offset = (1 << 31) // kv.stride(0) + 1
        # Reserve just over 4 GiB, but initialize only the referenced pages.
        # The first live page exceeds a signed int32 element offset.
        large_kv = torch.empty(
            (offset + kv.shape[0], *kv.shape[1:]),
            device=kv.device,
            dtype=kv.dtype,
        )
        large_kv[offset:].copy_(kv)
        op = RocmBeamSearchCascadeAttention(32, 16, 8, 128, 16, 32, q.dtype, q.device)
        op.plan(table + offset, lengths)
        out = op.run(q, large_kv)
        for row in (0, 16, 31):
            expected = reference(q, k, v, table, lengths, row)
            torch.testing.assert_close(
                out[row].float(), expected, atol=0.005, rtol=0.03
            )

    def test_graph_replay_changes_metadata_query_and_cache(self):
        case = make_case(rows=256)
        q, kv, table, lengths, k, v = case
        op = self.check_case(case)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                op.run(q, kv)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op.run(q, kv)
        address = output.data_ptr()
        for seed, distinct, history in (
            (1, False, 500),
            (2, True, 500),
            (3, False, 257),
            (4, False, 500),
        ):
            new = make_case(rows=256, seed=seed, distinct=distinct)
            q.copy_(new[0])
            kv.copy_(new[1])
            table.copy_(new[2])
            lengths.fill_(history)
            op.plan(table, lengths)
            graph.replay()
            for row in (0, 127, 255):
                expected = reference(q, new[4], new[5], table, lengths, row)
                torch.testing.assert_close(
                    output[row].float(), expected, atol=0.005, rtol=0.03
                )
            self.assertEqual(output.data_ptr(), address)

    def test_unaligned_request_boundaries_and_graph_replay(self):
        # Two production-sized requests must not lose their shared prefix at
        # row 500 merely because it is not a multiple of the query tile size.
        for force_fallback in (False, True):
            with self.subTest(force_fallback=force_fallback):
                q, kv, table, lengths, k, v = make_case(rows=1000)
                first_prefix = table[0, :-1].clone()
                second_prefix = table[500, :-1].clone()
                table[:500, :-1] = first_prefix
                table[500:, :-1] = second_prefix
                op = RocmBeamSearchCascadeAttention(
                    1000,
                    16,
                    8,
                    128,
                    16,
                    32,
                    q.dtype,
                    q.device,
                    max_requests=2,
                )
                if force_fallback:
                    op.NATIVE_PREFIX_BYTE_LIMIT = kv.numel() * kv.element_size()
                op.plan(table, lengths)
                self.assertIn(500, op.qptr_host.tolist())
                self.assertEqual(op.shared_tokens, 1000 * 496)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        op.run(q, kv)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = op.run(q, kv)
                addresses = (
                    output.data_ptr(),
                    op.prefix_qptr.data_ptr(),
                    op.row_cuts.data_ptr(),
                )
                for boundary in (500, 503, 128, 500):
                    table[:boundary, :-1] = first_prefix
                    table[boundary:, :-1] = second_prefix
                    op.plan(table, lengths)
                    self.assertEqual(op.shared_tokens, 1000 * 496)
                    graph.replay()
                    for row in (0, boundary - 1, boundary, 999):
                        expected = reference(q, k, v, table, lengths, row)
                        torch.testing.assert_close(
                            output[row].float(), expected, atol=0.005, rtol=0.03
                        )
                    self.assertEqual(
                        addresses,
                        (
                            output.data_ptr(),
                            op.prefix_qptr.data_ptr(),
                            op.row_cuts.data_ptr(),
                        ),
                    )

    def test_invalid_metadata_is_rejected_before_launch(self):
        q, kv, table, lengths, _, _ = make_case()
        op = RocmBeamSearchCascadeAttention(128, 16, 8, 128, 16, 32, q.dtype, q.device)
        with self.assertRaisesRegex(ValueError, "capacity"):
            op.plan(table, lengths + 512)
        with self.assertRaisesRegex(ValueError, "narrow"):
            op.plan(table[:, :1], lengths)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            op.plan(table - 1, lengths)
        with self.assertRaisesRegex(ValueError, "CPU int32"):
            op.plan(table.cuda(), lengths)


if __name__ == "__main__":
    unittest.main()
