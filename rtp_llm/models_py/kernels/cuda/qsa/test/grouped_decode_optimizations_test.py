"""Numerical and live-metadata checks for SM120 short-query specializations."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import sparse_paged_gqa_attn


class _RejectVectorAttention:
    def __getitem__(self, grid):
        raise AssertionError("business QSA must not launch per-head vector attention")


class GroupedDecodeOptimizationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 required")

    def fixture(self, batch=2, queries=4, kv_heads=1, packed=True):
        torch.manual_seed(917)
        heads, d, pages = 12 * kv_heads, 256, 37
        q = torch.randn(batch, heads, queries, d, device="cuda", dtype=torch.bfloat16)
        payload = 2 * kv_heads * 128 * d
        storage = torch.randn(pages, payload + 64, device="cuda").to(
            torch.float8_e4m3fn
        )
        cache = storage[:, :payload].view(pages, 2, kv_heads, 128, d)
        scale = torch.ones(pages, 2, kv_heads, 128, device="cuda")
        table = torch.arange(1, 33, device="cuda", dtype=torch.int32).repeat(batch, 1)
        table[0, 1] = 0
        table[-1, 2] = pages
        lengths = torch.full((batch, queries), 4000, device="cuda", dtype=torch.int32)
        selected = torch.randint(
            0,
            4000,
            (batch, queries, 2052 if packed else 2051),
            device="cuda",
            dtype=torch.int32,
        )
        selected[..., :64] = -1  # An empty tile preceding live tiles.
        selected[..., 66] = 5000
        if packed:
            selected[..., -1] = 2051
            selected[0, 0, -1] = 0
            selected[-1, -1, -1] = 129
        gate = torch.randn(
            batch, queries, heads, d, device="cuda", dtype=torch.bfloat16
        )
        return q, cache, table, lengths, selected, scale, gate

    @staticmethod
    def reference(args, packed=True, unit=True, with_gate=True):
        q, cache, table, lengths, selected, scale, gate = args
        result = torch.zeros_like(q)
        kv_heads = cache.shape[2]
        for b in range(q.shape[0]):
            for row in range(q.shape[2]):
                count = max(0, min(2051, int(selected[b, row, -1]))) if packed else 2051
                ids = selected[b, row, :count].long()
                ids = ids[
                    (ids >= 0) & (ids < lengths[b, row]) & (ids < table.shape[1] * 128)
                ]
                pages = table[b, ids // 128].long()
                good = (pages > 0) & (pages < cache.shape[0])
                pages, ids = pages[good], ids[good]
                if not ids.numel():
                    continue
                for h in range(kv_heads):
                    k = cache[pages, 0, h, ids % 128].float()
                    v = cache[pages, 1, h, ids % 128].float()
                    if not unit:
                        k *= scale[pages, 0, h, ids % 128, None]
                        v *= scale[pages, 1, h, ids % 128, None]
                    k, v = k.bfloat16().double(), v.bfloat16().double()
                    hs = slice(h * 12, (h + 1) * 12)
                    p = (q[b, hs, row].double() @ k.T / 16).softmax(-1)
                    result[b, hs, row] = (p @ v).bfloat16()
        if with_gate:
            result = (
                result.float()
                * gate.transpose(1, 2).float().sigmoid().bfloat16().float()
            ).bfloat16()
        return result

    @staticmethod
    def run_attention(args, packed=True, unit=True, with_gate=True, **kwargs):
        q, cache, table, lengths, selected, scale, gate = args
        return sparse_paged_gqa_attn(
            q,
            cache,
            table,
            lengths,
            selected,
            page_size=128,
            kv_head_num=cache.shape[2],
            kv_scale=scale,
            selected_has_count=packed,
            kv_scale_is_unity=unit,
            validate_selected=False,
            output_gate=gate if with_gate else None,
            **kwargs
        )

    def test_unit_scale_does_not_read_poisoned_scale_storage(self):
        for kv_heads, batch, queries in ((1, 2, 4), (1, 4, 1), (2, 4, 4)):
            with self.subTest(kv_heads=kv_heads, batch=batch, queries=queries):
                args = self.fixture(batch, queries, kv_heads)
                args[5].fill_(float("nan"))
                actual = self.run_attention(args)
                self.assertTrue(bool(actual.isfinite().all()))
                torch.testing.assert_close(
                    actual, self.reference(args), atol=0.02, rtol=0.02
                )

    def test_real_scales_retain_bf16_dequantization(self):
        args = self.fixture()
        args[5].uniform_(0.1, 2.0)
        actual = self.run_attention(args, unit=False)
        torch.testing.assert_close(
            actual, self.reference(args, unit=False), atol=0.02, rtol=0.02
        )

    def test_unpacked_tail_and_clamped_packed_counts(self):
        args = self.fixture(batch=8, packed=False)
        args[5].fill_(float("nan"))
        torch.testing.assert_close(
            self.run_attention(args, packed=False),
            self.reference(args, packed=False),
            atol=0.02,
            rtol=0.02,
        )
        args = self.fixture()
        args[4][0, 0, -1] = -3
        args[4][0, 1, -1] = 4000
        args[5].fill_(float("nan"))
        torch.testing.assert_close(
            self.run_attention(args), self.reference(args), atol=0.02, rtol=0.02
        )

    def test_unit_graph_refreshes_counts_pages_lengths_and_gate(self):
        args = self.fixture(packed=True)
        args[5].fill_(float("nan"))
        for _ in range(2):
            self.run_attention(args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.run_attention(args, graph_capture=True)
        args[2].copy_(args[2].flip(1))
        args[3][1, 1] = 65
        args[4][0, 0, -1] = 2051
        args[4][1, 3, -1] = 0
        args[6].mul_(0.25)
        graph.replay()
        torch.testing.assert_close(actual, self.reference(args), atol=0.02, rtol=0.02)
        args[3].zero_()
        graph.replay()
        torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
        graph.reset()

    def test_direct_conversion_rejects_old_selected_ptxas(self):
        args = self.fixture()
        with patch(
            "triton.backends.nvidia.compiler.get_ptxas",
            return_value=SimpleNamespace(
                version="12.9", path="/usr/local/cuda-13.2/bin/ptxas"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "13.2"):
                self.run_attention(args)

    def test_extended_verify_uses_tensor_cores_with_real_scales(self):
        # A lost business-shape dispatch must fail even if the vector kernel
        # still produces a numerically valid answer. Real kernels run otherwise.
        for batch, queries, kv_heads, packed in (
            (1, 1, 2, False),
            (1, 5, 1, False),
            (2, 7, 1, True),
            (4, 8, 2, False),
            (8, 3, 1, True),
            (8, 10, 1, True),
            (16, 15, 2, True),
        ):
            with self.subTest(batch=batch, queries=queries, kv_heads=kv_heads):
                args = self.fixture(batch, queries, kv_heads, packed=packed)
                args[5].uniform_(0.1, 2.0)
                with patch(
                    "rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha."
                    "_sparse_paged_online_split_kernel",
                    _RejectVectorAttention(),
                ):
                    actual = self.run_attention(args, packed=packed, unit=False)
                torch.testing.assert_close(
                    actual,
                    self.reference(args, packed=packed, unit=False),
                    atol=0.02,
                    rtol=0.02,
                )

    def test_extended_verify_graph_refreshes_all_operands(self):
        for queries, kv_heads, unit in ((7, 1, False), (10, 2, True)):
            with self.subTest(queries=queries, kv_heads=kv_heads, unit=unit):
                args = self.fixture(2, queries, kv_heads, packed=True)
                args[5].fill_(float("nan")) if unit else args[5].uniform_(0.2, 1.5)
                with patch(
                    "rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha."
                    "_sparse_paged_online_split_kernel",
                    _RejectVectorAttention(),
                ):
                    for _ in range(2):
                        self.run_attention(args, unit=unit)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = self.run_attention(args, unit=unit, graph_capture=True)
                args[0].mul_(0.5)
                args[2].copy_(args[2].flip(1))
                args[3][1, 1] = 129
                args[4][..., :2051].copy_(args[4][..., :2051].roll(67, -1))
                args[4][0, 0, -1] = 2051
                args[4][1, -1, -1] = 0
                if not unit:
                    args[5].uniform_(0.1, 2.0)
                args[6].mul_(0.25)
                graph.replay()
                torch.testing.assert_close(
                    actual, self.reference(args, unit=unit), atol=0.02, rtol=0.02
                )
                args[3].zero_()
                graph.replay()
                torch.testing.assert_close(
                    actual, torch.zeros_like(actual), atol=0, rtol=0
                )
                graph.reset()

    def test_extended_verify_explicit_fp16_pv(self):
        args = self.fixture(2, 8, 1, packed=False)
        args[5].uniform_(0.1, 2.0)
        with patch(
            "rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha."
            "_sparse_paged_online_split_kernel",
            _RejectVectorAttention(),
        ):
            actual = self.run_attention(args, packed=False, unit=False, fp16_pv=True)
        torch.testing.assert_close(
            actual,
            self.reference(args, packed=False, unit=False),
            atol=0.02,
            rtol=0.02,
        )


if __name__ == "__main__":
    unittest.main()
