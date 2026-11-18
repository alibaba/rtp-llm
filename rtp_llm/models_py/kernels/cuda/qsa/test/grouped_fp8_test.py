import unittest
from unittest.mock import patch

import torch
import triton

import rtp_llm.models_py.kernels.cuda.qsa.grouped_sparse_paged_fmha as grouped
from rtp_llm.models_py.kernels.cuda.qsa.chunk_paged_gqa import (
    _supports_tensor_core_arch,
)
from rtp_llm.models_py.kernels.cuda.qsa.grouped_sparse_paged_fmha import (
    grouped_sparse_paged_split,
)
from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import sparse_paged_gqa_attn
from rtp_llm.models_py.kernels.cuda.qsa.test.reference.sparse_paged import _reference


class GroupedFp8Test(unittest.TestCase):
    def fixture(self, heads, kv_heads, queries):
        torch.manual_seed(763)
        q = torch.randn(8, heads, queries, 256, device="cuda", dtype=torch.bfloat16)
        scale = torch.rand(35, 2, kv_heads, 128, device="cuda") * 0.8 + 0.2
        source = torch.randn(35, 2, kv_heads, 128, 256, device="cuda") * 0.3
        cache = (source / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        table = torch.stack(
            [
                torch.arange(1, 33, device="cuda", dtype=torch.int32).roll(i)
                for i in range(8)
            ]
        )
        lengths = torch.full((8, queries), 3000, device="cuda", dtype=torch.int32)
        lengths[2] = 97
        lengths[5] = 0
        selected = torch.arange(2051, device="cuda", dtype=torch.int32).repeat(
            8, queries, 1
        )
        selected = torch.where(selected < lengths.unsqueeze(-1), selected, -1)
        return q, cache, scale, table, lengths, selected

    def run_grouped(
        self, q, cache, scale, table, lengths, selected, kv_heads, **kwargs
    ):
        return sparse_paged_gqa_attn(
            q,
            cache,
            table,
            lengths,
            selected,
            page_size=128,
            kv_head_num=kv_heads,
            kv_scale=scale,
            **kwargs,
        )

    def test_measured_tp2_split_dispatch(self):
        for batch, queries, splits in (
            (1, 1, 32),
            (1, 4, 32),
            (4, 1, 32),
            (4, 4, 32),
            (2, 1, 32),
            (2, 4, 32),
            (8, 1, 32),
            (8, 4, 16),
        ):
            q, cache, scale, table, lengths, selected = self.fixture(12, 1, queries)
            q, table, lengths, selected = (
                t[:batch].contiguous() for t in (q, table, lengths, selected)
            )
            with patch.object(
                grouped,
                "grouped_sparse_paged_split",
                wraps=grouped.grouped_sparse_paged_split,
            ) as called:
                actual = self.run_grouped(q, cache, scale, table, lengths, selected, 1)
                self.assertIsNotNone(called.call_args)
                expected_splits = (
                    (32 if batch * queries <= 12 else 16)
                    if torch.cuda.get_device_capability(q.device)[0] == 10
                    else splits
                )
                self.assertEqual(called.call_args.kwargs["splits"], expected_splits)
            expected = _reference(
                q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
            )
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    @staticmethod
    def fp64_reference(q, cache, scale, table, lengths, selected):
        # Independent dense attention after the reader's BF16 dequant boundary.
        dequantized = (cache.float() * scale.unsqueeze(-1)).to(torch.bfloat16)
        result = torch.zeros_like(q)
        for batch in range(q.shape[0]):
            for query in range(q.shape[2]):
                indices = selected[batch, query]
                indices = indices[(indices >= 0) & (indices < lengths[batch, query])]
                if not indices.numel():
                    continue
                pages = table[batch, indices.long() // 128].long()
                valid = (pages > 0) & (pages < cache.shape[0])
                pages, indices = pages[valid], indices[valid]
                if not indices.numel():
                    continue
                keys = dequantized[pages, 0, 0, indices.long() % 128].double()
                values = dequantized[pages, 1, 0, indices.long() % 128].double()
                scores = q[batch, :, query].double() @ keys.T / 16.0
                result[batch, :, query] = (scores.softmax(-1) @ values).to(result.dtype)
        return result

    def test_tuned_tp2_short_long_and_cancellation_against_fp64(self):
        for batch, queries in ((1, 4), (4, 1), (4, 4)):
            for history in (129, 1023, 4096):
                q, cache, scale, table, lengths, selected = self.fixture(12, 1, queries)
                q, table, lengths, selected = (
                    t[:batch].contiguous() for t in (q, table, lengths, selected)
                )
                lengths.fill_(history)
                # Include duplicate indices and holes in the selected set.
                selected.masked_fill_(selected >= history, -1)
                selected[..., 1] = selected[..., 0]
                selected[..., 7] = -1
                # Alternate signs and use a wide, non-unit scale distribution.
                cache[:, 1].copy_(
                    (
                        cache[:, 1].float()
                        * torch.where(
                            torch.arange(128, device="cuda")[:, None] % 2 == 0,
                            1.0,
                            -1.0,
                        )
                    ).to(cache.dtype)
                )
                scale.mul_(torch.linspace(0.01, 4.0, 128, device="cuda"))
                actual = self.run_grouped(q, cache, scale, table, lengths, selected, 1)
                expected = self.fp64_reference(
                    q, cache, scale, table, lengths, selected
                )
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
                nrmse = (actual.float() - expected.float()).square().mean().sqrt() / (
                    expected.float().square().mean().sqrt() + 1e-8
                )
                # Single BF16 PV rounds probabilities once. Independent
                # original/direct/single-PV calibration places this fixture
                # near 0.22% relative L2; pointwise 0.02/0.02 stays unchanged.
                self.assertLess(float(nrmse), 4e-3)

    def test_tuned_tp2_graph_refreshes_pages_scales_and_selected(self):
        for batch, queries in ((1, 4), (4, 1), (4, 4)):
            q, cache, scale, table, lengths, selected = self.fixture(12, 1, queries)
            q, table, lengths, selected = (
                t[:batch].contiguous() for t in (q, table, lengths, selected)
            )
            for _ in range(2):
                self.run_grouped(q, cache, scale, table, lengths, selected, 1)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = self.run_grouped(
                    q, cache, scale, table, lengths, selected, 1, graph_capture=True
                )
            for factor in (0.5, 2.0):
                q.mul_(0.75)
                scale.mul_(factor)
                table.copy_(table.flip(1))
                table[0, 0] = 0
                selected.copy_(selected.flip(-1))
                graph.replay()
                expected = self.fp64_reference(
                    q, cache, scale, table, lengths, selected
                )
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            lengths.zero_()
            graph.replay()
            torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
            graph.reset()

    def test_tp2_tp4_tp8_and_two_kv_heads_with_scaled_padded_cache(self):
        for heads, kv_heads in [(3, 1), (6, 1), (12, 1), (24, 2)]:
            for queries in (1, 4):
                with self.subTest(heads=heads, kv_heads=kv_heads, queries=queries):
                    q, cache, scale, table, lengths, selected = self.fixture(
                        heads, kv_heads, queries
                    )
                    packed = torch.empty(
                        35, cache[0].numel() + 37, device="cuda", dtype=cache.dtype
                    )
                    packed[:, : cache[0].numel()].copy_(cache.flatten(1))
                    packed_scale = torch.full(
                        (35, scale[0].numel() + 13), float("nan"), device="cuda"
                    )
                    packed_scale[:, : scale[0].numel()].copy_(scale.flatten(1))
                    expected = _reference(
                        q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                    )
                    with patch.object(
                        grouped,
                        "grouped_sparse_paged_split",
                        wraps=grouped.grouped_sparse_paged_split,
                    ) as called:
                        got = self.run_grouped(
                            q, packed, packed_scale, table, lengths, selected, kv_heads
                        )
                        self.assertEqual(called.call_count, 1)
                    torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
                    self.assertTrue(
                        torch.isnan(packed_scale[:, scale[0].numel() :]).all()
                    )

    def test_graph_updates_fp8_values_scales_pages_and_zero_rows(self):
        q, cache, scale, table, lengths, selected = self.fixture(12, 1, 4)
        for _ in range(2):
            self.run_grouped(q, cache, scale, table, lengths, selected, 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = self.run_grouped(
                q, cache, scale, table, lengths, selected, 1, graph_capture=True
            )
        for factor in (0.5, 2.0):
            q.mul_(0.75)
            scale.mul_(factor)
            cache.copy_((torch.randn_like(cache.float()) * 0.3).to(cache.dtype))
            table.copy_(table.flip(1))
            graph.replay()
            torch.cuda.synchronize()
            expected = _reference(
                q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
            )
            torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
        lengths.zero_()
        graph.replay()
        torch.testing.assert_close(got, torch.zeros_like(got))

    def test_small_batches_and_intermediate_query_lengths(self):
        for batch in range(1, 9):
            queries = 1 + batch % 4
            with self.subTest(batch=batch, queries=queries):
                q, cache, scale, table, lengths, selected = self.fixture(12, 1, queries)
                q, table, lengths, selected = (
                    x[:batch].contiguous() for x in (q, table, lengths, selected)
                )
                with patch.object(
                    grouped,
                    "grouped_sparse_paged_split",
                    wraps=grouped.grouped_sparse_paged_split,
                ) as called:
                    got = self.run_grouped(q, cache, scale, table, lengths, selected, 1)
                    self.assertEqual(called.call_count, 1)
                torch.testing.assert_close(
                    got,
                    _reference(
                        q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                    ),
                    atol=2e-2,
                    rtol=2e-2,
                )

    def test_two_kv_graph_live_selected_tail_and_invalid_pages(self):
        q, cache, scale, table, lengths, selected = self.fixture(24, 2, 3)
        q, table, lengths, selected = (
            x[:7].contiguous() for x in (q, table, lengths, selected)
        )
        selected.fill_(-1)
        selected[..., -3:] = torch.tensor(
            [90, 91, 92], device="cuda", dtype=torch.int32
        )
        selected.masked_fill_(selected >= lengths.unsqueeze(-1), -1)
        for _ in range(2):
            self.run_grouped(q, cache, scale, table, lengths, selected, 2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            got = self.run_grouped(
                q, cache, scale, table, lengths, selected, 2, graph_capture=True
            )
        for tail in ([0, 1, 2], [90, 91, 92]):
            selected[..., -3:] = torch.tensor(tail, device="cuda", dtype=torch.int32)
            scale.mul_(0.75)
            table.copy_(table.flip(1))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                got,
                _reference(
                    q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                ),
                atol=2e-2,
                rtol=2e-2,
            )
        selected.fill_(-1)
        graph.replay()
        torch.testing.assert_close(got, torch.zeros_like(got))
        selected[..., -3:] = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
        table.fill_(cache.shape[0])
        graph.replay()
        torch.testing.assert_close(got, torch.zeros_like(got))

    def test_small_batches_and_gqa_ratios_use_tensor_cores(self):
        for batch, heads, queries in [(1, 3, 1), (3, 12, 1), (1, 6, 4)]:
            with self.subTest(batch=batch, heads=heads, queries=queries):
                q, cache, scale, table, lengths, selected = self.fixture(
                    heads, 1, queries
                )
                q, table, lengths, selected = (
                    x[:batch].contiguous() for x in (q, table, lengths, selected)
                )
                with patch.object(
                    grouped,
                    "grouped_sparse_paged_split",
                    wraps=grouped.grouped_sparse_paged_split,
                ) as called:
                    got = self.run_grouped(q, cache, scale, table, lengths, selected, 1)
                    self.assertEqual(called.call_count, 1)
                torch.testing.assert_close(
                    got,
                    _reference(
                        q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
                    ),
                    atol=2e-2,
                    rtol=2e-2,
                )

    def test_fp8_grouped_kernel_is_selected_automatically(self):
        q, cache, scale, table, lengths, selected = self.fixture(12, 1, 1)
        with patch.object(
            grouped,
            "grouped_sparse_paged_split",
            wraps=grouped.grouped_sparse_paged_split,
        ) as called:
            got = self.run_grouped(q, cache, scale, table, lengths, selected, 1)
            self.assertEqual(called.call_count, 1)
        torch.testing.assert_close(
            got,
            _reference(
                q, cache.float() * scale.unsqueeze(-1), table, lengths, selected
            ),
            atol=2e-2,
            rtol=2e-2,
        )

    def gate_fixture(self, batch, queries):
        q, cache, scale, table, lengths, selected = self.fixture(12, 1, queries)
        q, table, lengths, selected = [
            t[:batch].contiguous() for t in (q, table, lengths, selected)
        ]
        gate = torch.randn(batch, queries, 12, 256, device="cuda", dtype=torch.bfloat16)
        return q, cache, scale, table, lengths, selected, gate

    def run_gated_core(self, tensors, gated, graph=False):
        q, cache, scale, table, lengths, selected, gate = tensors
        return sparse_paged_gqa_attn(
            q,
            cache,
            table,
            lengths,
            selected,
            page_size=128,
            kv_head_num=1,
            kv_scale=scale,
            graph_capture=graph,
            output_gate=gate if gated else None,
        )

    def test_exact_gate_rounding_and_owner_layout(self):
        for batch in (1, 4):
            for queries in (1, 4):
                tensors = self.gate_fixture(batch, queries)
                gate = tensors[-1]
                gate.flatten()[:3] = torch.tensor(
                    [float("-inf"), float("inf"), float("nan")],
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                expected = self.run_gated_core(tensors, False).transpose(
                    1, 2
                ) * torch.sigmoid(gate)
                actual = self.run_gated_core(tensors, True).transpose(1, 2)
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0, equal_nan=True
                )
                self.assertTrue(actual.is_contiguous())

    def test_graph_refreshes_gates_and_fp32_page_scales(self):
        tensors = self.gate_fixture(4, 4)
        for _ in range(3):
            self.run_gated_core(tensors, True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.run_gated_core(tensors, True, graph=True)
        for seed in (7370, 7371, 7372):
            torch.manual_seed(seed)
            tensors[-1].normal_()
            tensors[2].uniform_(0.2, 1)
            expected = (
                self.run_gated_core(tensors, False).transpose(1, 2)
                * torch.sigmoid(tensors[-1])
            ).transpose(1, 2)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class GroupedDecodeRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not _supports_tensor_core_arch(
            torch.cuda.get_device_capability()
        ):
            raise unittest.SkipTest("SM10x or SM120 required")

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
            **kwargs,
        )

    def test_large_value_fp16_pv_falls_back_to_single_bf16_pv(self):
        q, cache, table, lengths, selected, scale, _ = self.fixture()
        # Scale V only: keep nontrivial probabilities while exceeding FP16's
        # range, so a residual PV term cannot be hidden by one-hot softmax.
        scale[:, 1].fill_(2.0**17)
        default = torch.empty((2, 12, 4, 8, 258), device=q.device)
        fallback = torch.empty_like(default)
        for fp16_pv, partial in ((False, default), (True, fallback)):
            grouped_sparse_paged_split(
                q,
                cache,
                table,
                lengths,
                selected,
                partial,
                page_size=128,
                splits=8,
                scale=scale,
                selected_has_count=True,
                fp16_pv=fp16_pv,
            )
        torch.testing.assert_close(fallback, default, atol=0, rtol=0)

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

    def test_unit_scale_falls_back_without_direct_fp8_support(self):
        from rtp_llm.models_py.kernels.cuda.qsa import chunk_paged_gqa as chunk

        target = triton.runtime.driver.active.get_current_target()
        args = self.fixture()
        expected = self.reference(args)
        # Unit scale is a producer guarantee, including when conversion falls
        # back: do not read irrelevant (potentially absent) scale storage.
        args[5].fill_(float("nan"))
        for capability in ((8, 9), (9, 0), (12, 0)):
            with self.subTest(capability=capability), patch(
                "torch.cuda.get_device_capability", return_value=capability
            ), patch.object(
                chunk,
                "_require_fp8_toolchain",
                side_effect=RuntimeError("CUDA 13.2 required"),
            ):
                actual = self.run_attention(args)
            self.assertEqual(target, triton.runtime.driver.active.get_current_target())
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)

    def test_extended_verify_uses_tensor_cores_with_real_scales(self):
        # A lost grouped dispatch must fail even if another path still produces
        # a numerically valid answer. The spy executes the real GPU kernels.
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
                    "rtp_llm.models_py.kernels.cuda.qsa.grouped_sparse_paged_fmha."
                    "grouped_sparse_paged_split",
                    wraps=grouped_sparse_paged_split,
                ) as launch:
                    actual = self.run_attention(args, packed=packed, unit=False)
                    launch.assert_called_once()
                torch.testing.assert_close(
                    actual,
                    self.reference(args, packed=packed, unit=False),
                    atol=0.02,
                    rtol=0.02,
                )

    def test_extended_verify_graph_refreshes_all_operands(self):
        for queries, kv_heads, unit in ((4, 1, True), (7, 1, False), (10, 2, True)):
            with self.subTest(queries=queries, kv_heads=kv_heads, unit=unit):
                args = self.fixture(2, queries, kv_heads, packed=True)
                args[5].fill_(float("nan")) if unit else args[5].uniform_(0.2, 1.5)
                with patch(
                    "rtp_llm.models_py.kernels.cuda.qsa.grouped_sparse_paged_fmha."
                    "grouped_sparse_paged_split",
                    wraps=grouped_sparse_paged_split,
                ) as launch:
                    for _ in range(2):
                        self.run_attention(args, unit=unit)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = self.run_attention(args, unit=unit, graph_capture=True)
                    self.assertEqual(launch.call_count, 3)  # Two warmups and capture.
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


if __name__ == "__main__":
    unittest.main()
