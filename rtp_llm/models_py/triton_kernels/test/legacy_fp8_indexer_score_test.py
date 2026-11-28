"""Compare both legacy cache layouts against FP32 scores on SM100/SM103."""

import sys
import unittest
from unittest import mock

import rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score as score_mod
import torch
import triton
from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
    legacy_fp8_mqa_logits,
    legacy_fp8_paged_mqa_logits,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class LegacyFP8IndexerScoreTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4021)
        torch.backends.cuda.matmul.allow_tf32 = False

    def reference(self, q, k, scale, weight):
        return (
            torch.einsum("mhd,nd->mhn", q.float(), k.float()).relu()
            * weight.unsqueeze(-1)
        ).sum(1) * scale

    def test_ragged_fp32_weights_scales_and_mask(self):
        for heads, queries, keys in [(1, 1, 3), (7, 5, 67), (64, 17, 131)]:
            with self.subTest(heads=heads, queries=queries, keys=keys):
                q = torch.randn(queries, heads, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
                scales = torch.rand(keys, device="cuda") * 0.01371 + 0.000123
                weights = torch.randn(queries, heads, 2, device="cuda")[..., 0]
                starts = torch.arange(queries, device="cuda", dtype=torch.int32) % keys
                ends = torch.minimum(
                    starts + keys // 2 + 1, torch.full_like(starts, keys)
                )
                actual = legacy_fp8_mqa_logits(q, (k, scales), weights, starts, ends)
                expected = self.reference(q, k, scales, weights)
                columns = torch.arange(keys, device="cuda")
                expected.masked_fill_(
                    (columns < starts[:, None]) | (columns >= ends[:, None]),
                    0.0,
                )
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_prefill_packed_causal_clean_and_edge_windows(self):
        for heads, queries, keys, clean in [
            (64, 1, 65, False),
            (64, 17, 131, True),
            (64, 146, 400, False),
            (64, 298, 520, True),
            (64, 301, 513, False),
            (64, 301, 513, True),
            (32, 37, 257, False),
            (32, 260, 640, True),
            (16, 25, 193, True),
            (16, 260, 640, False),
        ]:
            with self.subTest(heads=heads, queries=queries, keys=keys, clean=clean):
                q = torch.randn(queries, heads, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
                scales = torch.rand(keys, device="cuda") * 0.01 + 0.0005
                weights = torch.randn(
                    queries, heads, device="cuda", dtype=torch.float32
                )
                starts = torch.zeros(queries, device="cuda", dtype=torch.int32)
                ends = torch.linspace(
                    0, keys, queries, device="cuda", dtype=torch.int32
                )
                # Inject empty and shifted windows to exercise early-exit + clamping.
                if queries >= 4:
                    starts[0], ends[0] = 5, 5
                    starts[1], ends[1] = 30, 10
                    starts[2], ends[2] = keys - 17, keys
                actual = legacy_fp8_mqa_logits(
                    q, (k, scales), weights, starts, ends, clean_logits=clean
                )
                expected = self.reference(q, k, scales, weights)
                columns = torch.arange(keys, device="cuda")
                pad = float("-inf") if clean else 0.0
                expected.masked_fill_(
                    (columns < starts[:, None]) | (columns >= ends[:, None]),
                    pad,
                )
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_paged_next_n_permutation_and_context_lengths(self):
        batch, next_n, heads, page, pages = 3, 2, 64, 16, 12
        q = torch.randn(batch, next_n, heads, 128, device="cuda").to(
            torch.float8_e4m3fn
        )
        keys = torch.randn(pages, page, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(pages, page, device="cuda") * 0.00457 + 0.000019
        # The fused pool is block-planar, matching the legacy DeepGEMM
        # contract the kernel preserves: each page stores page_size*128 K
        # bytes followed by page_size*4 scale bytes. The [pages, page, 1,
        # 132] shape is the nominal container DeepGEMM indexes internally.
        packed = torch.empty(pages, page * 132, dtype=torch.uint8, device="cuda")
        packed[:, : page * 128] = keys.view(torch.uint8).reshape(pages, page * 128)
        packed[:, page * 128 :] = scales.view(torch.uint8).reshape(pages, page * 4)
        packed = packed.view(pages, page, 1, 132)
        table = torch.randperm(pages, device="cuda").to(torch.int32).reshape(batch, 4)
        lens = torch.tensor(
            [[0, 3], [31, 42], [63, 64]], device="cuda", dtype=torch.int32
        )
        weights = torch.randn(batch * next_n, heads, device="cuda")
        actual = legacy_fp8_paged_mqa_logits(q, packed, weights, lens, table, 64)
        for b in range(batch):
            physical = table[b].long()
            k = keys[physical].reshape(-1, 128)
            scale = scales[physical].reshape(-1)
            expected = self.reference(
                q[b], k, scale, weights[b * next_n : (b + 1) * next_n]
            )
            expected.masked_fill_(
                torch.arange(64, device="cuda") >= lens[b, :, None], 0.0
            )
            torch.testing.assert_close(
                actual[b * next_n : (b + 1) * next_n], expected, rtol=2e-5, atol=2e-5
            )

    def test_paged_single_page_fast_path_and_invalid_pages(self):
        for batch, next_n, heads, page in [
            (2, 4, 64, 64),
            (2, 3, 64, 64),
            (3, 1, 64, 64),
            (2, 4, 32, 32),
            (2, 2, 7, 64),
        ]:
            with self.subTest(batch=batch, next_n=next_n, heads=heads, page=page):
                table_width = 4
                pages = batch * table_width + 2
                max_ctx = table_width * page + 16
                q = torch.randn(batch, next_n, heads, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                keys = torch.randn(pages, page, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                scales = torch.rand(pages, page, device="cuda") * 0.005 + 0.0001
                packed = torch.empty(
                    pages, page * 132, dtype=torch.uint8, device="cuda"
                )
                packed[:, : page * 128] = keys.view(torch.uint8).reshape(
                    pages, page * 128
                )
                packed[:, page * 128 :] = scales.view(torch.uint8).reshape(
                    pages, page * 4
                )
                packed = packed.view(pages, page, 1, 132)
                table = (
                    torch.randperm(pages, device="cuda", dtype=torch.int32)[
                        : batch * table_width
                    ]
                    .reshape(batch, table_width)
                    .contiguous()
                )
                # Inject negative and out-of-range page entries on row 0 / last page.
                table[0, 1] = -1
                table[-1, -1] = pages + 5
                lens = torch.randint(
                    0, max_ctx + 1, (batch, next_n), device="cuda", dtype=torch.int32
                )
                lens[0, 0] = 0
                weights = torch.randn(batch * next_n, heads, device="cuda")
                actual = legacy_fp8_paged_mqa_logits(
                    q, packed, weights, lens, table, max_ctx
                )
                cols = torch.arange(max_ctx, device="cuda")
                logical_pages = cols // page
                for b in range(batch):
                    for s in range(next_n):
                        row = b * next_n + s
                        l_row = min(int(lens[b, s].item()), max_ctx)
                        expected_row = torch.zeros(
                            max_ctx, dtype=torch.float32, device="cuda"
                        )
                        valid_cols = cols[:l_row]
                        lp = logical_pages[:l_row]
                        in_tbl = lp < table_width
                        if in_tbl.any():
                            pp = table[b, lp[in_tbl]].long()
                            pp_ok = (pp >= 0) & (pp < pages)
                            sel_cols = valid_cols[in_tbl][pp_ok]
                            sel_pp = pp[pp_ok]
                            sel_off = sel_cols % page
                            k_sel = keys[sel_pp, sel_off]
                            s_sel = scales[sel_pp, sel_off]
                            ref_vals = self.reference(
                                q[b : b + 1, s],
                                k_sel,
                                s_sel,
                                weights[row : row + 1],
                            )[0]
                            expected_row[sel_cols] = ref_vals
                        torch.testing.assert_close(
                            actual[row], expected_row, rtol=2e-5, atol=2e-5
                        )

    def test_bf16_dot_fallback_guard_and_zero_pool(self):
        # Verify PPU/non-CUDA fallback path (USE_FP8_DOT=False) and empty pool/table.
        q = torch.randn(4, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn(70, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(70, device="cuda") * 0.01 + 0.001
        weights = torch.randn(4, 64, device="cuda", dtype=torch.float32)
        starts = torch.zeros(4, device="cuda", dtype=torch.int32)
        ends = torch.tensor([1, 17, 64, 70], device="cuda", dtype=torch.int32)

        q_bq2 = torch.randn(260, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        k_bq2 = torch.randn(520, 128, device="cuda").to(torch.float8_e4m3fn)
        s_bq2 = torch.rand(520, device="cuda") * 0.01 + 0.001
        w_bq2 = torch.randn(260, 64, device="cuda", dtype=torch.float32)
        ks_bq2 = torch.zeros(260, device="cuda", dtype=torch.int32)
        ke_bq2 = torch.arange(1, 261, device="cuda", dtype=torch.int32)

        q_p = torch.randn(2, 2, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        keys = torch.randn(4, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        p_scales = torch.rand(4, 32, device="cuda") * 0.01 + 0.001
        packed = torch.empty(4, 32 * 132, dtype=torch.uint8, device="cuda")
        packed[:, : 32 * 128] = keys.view(torch.uint8).reshape(4, 32 * 128)
        packed[:, 32 * 128 :] = p_scales.view(torch.uint8).reshape(4, 32 * 4)
        packed = packed.view(4, 32, 1, 132)
        table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
        lens = torch.tensor([[12, 45], [0, 64]], device="cuda", dtype=torch.int32)
        w_p = torch.randn(4, 64, device="cuda", dtype=torch.float32)

        with mock.patch.object(score_mod, "_use_native_fp8_dot", return_value=False):
            actual = legacy_fp8_mqa_logits(
                q, (k, scales), weights, starts, ends, clean_logits=True
            )
            expected = self.reference(q, k, scales, weights)
            cols = torch.arange(70, device="cuda")
            expected.masked_fill_(
                (cols < starts[:, None]) | (cols >= ends[:, None]),
                float("-inf"),
            )
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

            actual_bq2 = legacy_fp8_mqa_logits(
                q_bq2, (k_bq2, s_bq2), w_bq2, ks_bq2, ke_bq2, clean_logits=False
            )
            exp_bq2 = self.reference(q_bq2, k_bq2, s_bq2, w_bq2)
            cols2 = torch.arange(520, device="cuda")
            exp_bq2.masked_fill_(
                (cols2 < ks_bq2[:, None]) | (cols2 >= ke_bq2[:, None]),
                0.0,
            )
            torch.testing.assert_close(actual_bq2, exp_bq2, rtol=2e-5, atol=2e-5)

            actual_p = legacy_fp8_paged_mqa_logits(q_p, packed, w_p, lens, table, 64)
            for b in range(2):
                phys = table[b].long()
                exp_p = self.reference(
                    q_p[b],
                    keys[phys].reshape(-1, 128),
                    p_scales[phys].reshape(-1),
                    w_p[b * 2 : (b + 1) * 2],
                )
                exp_p.masked_fill_(
                    torch.arange(64, device="cuda") >= lens[b, :, None], 0.0
                )
                torch.testing.assert_close(
                    actual_p[b * 2 : (b + 1) * 2], exp_p, rtol=2e-5, atol=2e-5
                )

        # Zero-width table must return exact zeros without reading pool.
        empty_table = torch.empty((2, 0), device="cuda", dtype=torch.int32)
        zero_out = legacy_fp8_paged_mqa_logits(q_p, packed, w_p, lens, empty_table, 64)
        torch.testing.assert_close(zero_out, torch.zeros_like(zero_out))

    def test_scalar_one_specialization_edge_cases(self):
        # Guard against Triton specializing runtime scalar ints == 1 into Python ints.
        for heads, queries, keys, clean in [
            (1, 1, 1, False),
            (64, 1, 1, True),
            (64, 3, 1, False),
            (64, 3, 1, True),
        ]:
            with self.subTest(
                kernel="prefill_bq1",
                heads=heads,
                queries=queries,
                keys=keys,
                clean=clean,
            ):
                q = torch.randn(queries, heads, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                k = torch.randn(keys, 128, device="cuda").to(torch.float8_e4m3fn)
                scales = torch.rand(keys, device="cuda") * 0.01 + 0.001
                weights = torch.randn(
                    queries, heads, device="cuda", dtype=torch.float32
                )
                starts = torch.zeros(queries, device="cuda", dtype=torch.int32)
                ends = torch.ones(queries, device="cuda", dtype=torch.int32)
                if queries >= 3:
                    ends[1] = 0
                actual = legacy_fp8_mqa_logits(
                    q, (k, scales), weights, starts, ends, clean_logits=clean
                )
                expected = self.reference(q, k, scales, weights)
                cols = torch.arange(keys, device="cuda")
                pad = float("-inf") if clean else 0.0
                expected.masked_fill_(
                    (cols < starts[:, None]) | (cols >= ends[:, None]),
                    pad,
                )
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

        with mock.patch.object(score_mod, "_get_num_sms", return_value=1):
            for clean in (False, True):
                with self.subTest(
                    kernel="prefill_bq2", heads=64, queries=16, keys=1, clean=clean
                ):
                    q = torch.randn(16, 64, 128, device="cuda").to(torch.float8_e4m3fn)
                    k = torch.randn(1, 128, device="cuda").to(torch.float8_e4m3fn)
                    scales = torch.rand(1, device="cuda") * 0.01 + 0.001
                    weights = torch.randn(16, 64, device="cuda", dtype=torch.float32)
                    starts = torch.zeros(16, device="cuda", dtype=torch.int32)
                    ends = torch.arange(16, device="cuda", dtype=torch.int32) % 2
                    ends[0] = 1
                    actual = legacy_fp8_mqa_logits(
                        q, (k, scales), weights, starts, ends, clean_logits=clean
                    )
                    expected = self.reference(q, k, scales, weights)
                    cols = torch.arange(1, device="cuda")
                    pad = float("-inf") if clean else 0.0
                    expected.masked_fill_(
                        (cols < starts[:, None]) | (cols >= ends[:, None]),
                        pad,
                    )
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

        for batch, next_n, heads, page in [
            (1, 1, 1, 64),
            (1, 1, 64, 16),
        ]:
            with self.subTest(
                kernel="paged_bq1", batch=batch, next_n=next_n, heads=heads, page=page
            ):
                pages, table_width, max_ctx = 1, 1, 1
                q = torch.randn(batch, next_n, heads, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                keys = torch.randn(pages, page, 128, device="cuda").to(
                    torch.float8_e4m3fn
                )
                scales = torch.rand(pages, page, device="cuda") * 0.01 + 0.001
                packed = torch.empty(
                    pages, page * 132, dtype=torch.uint8, device="cuda"
                )
                packed[:, : page * 128] = keys.view(torch.uint8).reshape(
                    pages, page * 128
                )
                packed[:, page * 128 :] = scales.view(torch.uint8).reshape(
                    pages, page * 4
                )
                packed = packed.view(pages, page, 1, 132)
                table = torch.zeros(
                    (batch, table_width), device="cuda", dtype=torch.int32
                )
                weights = torch.randn(
                    batch * next_n, heads, device="cuda", dtype=torch.float32
                )
                for length_val in (1, 0):
                    lens = torch.full(
                        (batch, next_n), length_val, device="cuda", dtype=torch.int32
                    )
                    actual = legacy_fp8_paged_mqa_logits(
                        q, packed, weights, lens, table, max_ctx
                    )
                    if length_val == 0:
                        expected = torch.zeros_like(actual)
                    else:
                        expected = self.reference(
                            q[0], keys[0, :1], scales[0, :1], weights
                        )
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

        with self.subTest(kernel="paged_bq2", batch=16, next_n=2, heads=64, page=64):
            batch, next_n, heads, page, pages, table_width, max_ctx = (
                16,
                2,
                64,
                64,
                1,
                1,
                1,
            )
            q = torch.randn(batch, next_n, heads, 128, device="cuda").to(
                torch.float8_e4m3fn
            )
            keys = torch.randn(pages, page, 128, device="cuda").to(torch.float8_e4m3fn)
            scales = torch.rand(pages, page, device="cuda") * 0.01 + 0.001
            packed = torch.empty(pages, page * 132, dtype=torch.uint8, device="cuda")
            packed[:, : page * 128] = keys.view(torch.uint8).reshape(pages, page * 128)
            packed[:, page * 128 :] = scales.view(torch.uint8).reshape(pages, page * 4)
            packed = packed.view(pages, page, 1, 132)
            table = torch.zeros((batch, table_width), device="cuda", dtype=torch.int32)
            lens = torch.randint(
                0, 2, (batch, next_n), device="cuda", dtype=torch.int32
            )
            lens[0, 0] = 1
            lens[1, 0] = 0
            weights = torch.randn(
                batch * next_n, heads, device="cuda", dtype=torch.float32
            )
            actual = legacy_fp8_paged_mqa_logits(
                q, packed, weights, lens, table, max_ctx
            )
            expected = self.reference(
                q.view(batch * next_n, heads, 128), keys[0, :1], scales[0, :1], weights
            )
            expected.masked_fill_(lens.reshape(-1, 1) == 0, 0.0)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def _run_benchmarks():
    props = torch.cuda.get_device_properties(0)
    print(
        f"\n=== [legacy_fp8_indexer_score bench on {props.name} "
        f"SM={props.major}.{props.minor} SMs={props.multi_processor_count} "
        f"torch={torch.__version__} triton={triton.__version__}] ===",
        flush=True,
    )
    torch.manual_seed(0)

    print("\n[Prefill: legacy_fp8_mqa_logits]", flush=True)
    print(
        f"{'M':>6} {'N':>6} {'H':>4} {'clean':>6} | {'median_ms':>10} {'TFLOP/s':>8}",
        flush=True,
    )
    prefill_shapes = [
        (7584, 7585, 64, False),
        (7584, 7585, 64, True),
        (298, 3625, 64, False),
        (266, 1284, 64, False),
        (290, 290, 64, False),
        (146, 400, 64, False),
    ]
    for m, n, h, clean in prefill_shapes:
        q = torch.randn(m, h, 128, device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn(n, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(n, device="cuda") * 0.01 + 0.001
        weights = torch.randn(m, h, device="cuda", dtype=torch.float32)
        prefix = max(0, n - m)
        starts = torch.zeros(m, device="cuda", dtype=torch.int32)
        ends = torch.arange(
            prefix + 1, prefix + m + 1, device="cuda", dtype=torch.int32
        ).clamp_max(n)
        ms = triton.testing.do_bench(
            lambda: legacy_fp8_mqa_logits(
                q, (k, scales), weights, starts, ends, clean_logits=clean
            ),
            warmup=25,
            rep=200,
        )
        # Approximate active MACs: 2 * sum(ends - starts) * H * 128
        active_tokens = float((ends - starts).clamp_min(0).sum().item())
        tflops = (2.0 * active_tokens * h * 128.0) / (ms * 1e-3) / 1e12
        print(
            f"{m:6d} {n:6d} {h:4d} {str(clean):>6} | {ms:10.4f} {tflops:8.2f}",
            flush=True,
        )
        del q, k, scales, weights, starts, ends
        torch.cuda.empty_cache()

    print("\n[Paged Decode: legacy_fp8_paged_mqa_logits]", flush=True)
    print(
        f"{'B':>4} {'next_n':>6} {'H':>4} {'T':>6} {'page':>5} {'mode':>8} | {'median_ms':>10}",
        flush=True,
    )
    paged_shapes = [
        (24, 4, 64, 4096, 64, "full"),
        (24, 4, 64, 4096, 64, "varlen"),
        (24, 1, 64, 4096, 64, "full"),
        (1, 1, 64, 4096, 64, "full"),
    ]
    for b, next_n, h, t, page, mode in paged_shapes:
        pages_per_req = (t + page - 1) // page
        total_pages = b * pages_per_req + 4
        q = torch.randn(b, next_n, h, 128, device="cuda").to(torch.float8_e4m3fn)
        kv_cache = torch.randint(
            0, 127, (total_pages, page, 1, 132), dtype=torch.uint8, device="cuda"
        )
        # Put well-formed fp32 scales in last 4*page bytes per page
        scales = (
            torch.rand(total_pages, page, device="cuda", dtype=torch.float32) * 0.01
            + 0.001
        )
        kv_cache.view(total_pages, page * 132)[:, page * 128 :] = scales.view(
            torch.uint8
        ).reshape(total_pages, page * 4)
        table = (
            torch.randperm(total_pages, device="cuda", dtype=torch.int32)[
                : b * pages_per_req
            ]
            .reshape(b, pages_per_req)
            .contiguous()
        )
        if mode == "full":
            lens = torch.full((b, next_n), t, dtype=torch.int32, device="cuda")
        else:
            base = torch.linspace(256, t, b, device="cuda", dtype=torch.int32)[:, None]
            lens = (
                base + torch.arange(next_n, device="cuda", dtype=torch.int32)[None, :]
            ).clamp_max(t)
        weights = torch.randn(b * next_n, h, device="cuda", dtype=torch.float32)
        ms = triton.testing.do_bench(
            lambda: legacy_fp8_paged_mqa_logits(q, kv_cache, weights, lens, table, t),
            warmup=25,
            rep=200,
        )
        print(
            f"{b:4d} {next_n:6d} {h:4d} {t:6d} {page:5d} {mode:>8} | {ms:10.4f}",
            flush=True,
        )
        del q, kv_cache, scales, table, lens, weights
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_bench = "--bench" in sys.argv
    if run_bench:
        sys.argv = [a for a in sys.argv if a != "--bench"]
    if run_bench:
        _run_benchmarks()
    else:
        unittest.main()
