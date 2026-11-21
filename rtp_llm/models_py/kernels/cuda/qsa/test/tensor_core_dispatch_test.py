"""All accepted attention geometries must execute BF16 matrix products."""

import unittest
from unittest.mock import patch

import torch

import rtp_llm.models_py.kernels.cuda.qsa.grouped_sparse_paged_fmha as grouped
import rtp_llm.models_py.kernels.cuda.qsa.sparse_fmha as dense
from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import sparse_paged_gqa_attn


class GroupedLaunchPolicyTest(unittest.TestCase):
    def launch(self, batch, heads, kv_heads, queries, dim, table_width):
        calls = []

        class Recorder:
            def __getitem__(self, grid):
                return lambda *args, **kwargs: calls.append((grid, kwargs))

        q = torch.empty(batch, heads, queries, dim, device="meta", dtype=torch.bfloat16)
        cache = torch.empty(2, 2, kv_heads, 4, dim, device="meta", dtype=q.dtype)
        table = torch.empty(batch, table_width, device="meta", dtype=torch.int32)
        lengths = torch.empty(batch, queries, device="meta", dtype=torch.int32)
        selected = torch.empty(batch, queries, 1, device="meta", dtype=torch.int32)
        with patch.object(grouped, "_grouped_sparse_paged_split_kernel", Recorder()):
            grouped.grouped_sparse_paged_split(
                q, cache, table, lengths, selected, q, page_size=4, splits=1
            )
        return calls[0]

    def test_long_grid_also_respects_x_limit(self):
        grid, _ = self.launch(128, 128, 128, 131072, 1, 1)
        self.assertLessEqual(grid[0], 2**31 - 1)
        self.assertLessEqual(grid[1], 65535)
        self.assertGreaterEqual(grid[0] * grid[1], 128 * 128 * 131072)

    def test_wide_block_table_requires_wide_addressing(self):
        _, options = self.launch(3, 1, 1, 1, 8, 2**30 + 1)
        self.assertTrue(options["WIDE_ADDRESS"])


class TensorCoreDispatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires CUDA")

    def assert_attention(self, actual, expected):
        self.assertTrue(bool(actual.isfinite().all()))
        norm = expected.norm(dim=-1)
        error = (actual.double() - expected).norm(dim=-1) / norm.clamp_min(1e-30)
        self.assertLess(float(error.max()), 0.008)
        torch.testing.assert_close(
            actual[norm == 0].double(), expected[norm == 0], atol=0, rtol=0
        )

    def run_matrix_kernel(self, kernel, invoke, *, wide_address=None):
        compiled = []
        run = kernel.run

        def record(*args, **kwargs):
            if wide_address is not None:
                kwargs["WIDE_ADDRESS"] = wide_address
            result = run(*args, **kwargs)
            compiled.append(result)
            return result

        with patch.object(kernel, "run", side_effect=record):
            output = invoke()
        self.assertEqual(len(compiled), 1, "attention bypassed the Tensor Core kernel")
        products = [
            line
            for line in compiled[0].asm["ttir"].splitlines()
            if " = tt.dot " in line
        ]
        self.assertEqual(len(products), 2, products)
        self.assertTrue(all(line.count("xbf16>") == 2 for line in products))
        return output

    def test_long_query_grid_preserves_batch_head_and_split_mapping(self):
        # One selected token has an exact V oracle. Exercise both sides of
        # CUDA's grid.y limit, multiple head tiles, batches and split reduction.
        for batch, heads, kv_heads, queries, width in (
            (1, 1, 1, 65535, 1),
            (2, 1, 1, 65536, 1),
            (1, 2, 2, 32767, 1),
            (2, 2, 2, 32768, 65),
            (1, 17, 1, 32768, 1),
        ):
            with self.subTest(shape=(batch, heads, kv_heads, queries, width)):
                q = torch.zeros(
                    batch, heads, queries, 8, device="cuda", dtype=torch.bfloat16
                )
                cache = (
                    torch.arange((batch + 1) * 2 * kv_heads * 4 * 8, device="cuda")
                    .remainder(127)
                    .to(q.dtype)
                    .reshape(batch + 1, 2, kv_heads, 4, 8)
                )
                table = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)[
                    :, None
                ]
                lengths = torch.full(
                    (batch, queries), 4, device="cuda", dtype=torch.int32
                )
                selected = torch.full(
                    (batch, queries, width), -1, device="cuda", dtype=torch.int32
                )
                selected[..., -1] = torch.arange(queries, device="cuda") % 4
                actual = sparse_paged_gqa_attn(
                    q,
                    cache,
                    table,
                    lengths,
                    selected,
                    page_size=4,
                    kv_head_num=kv_heads,
                    validate_selected=False,
                )
                expected = cache[table[:, 0].long(), 1][
                    :, :, torch.arange(queries, device="cuda") % 4
                ].repeat_interleave(heads // kv_heads, dim=1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_padded_two_dimensional_attention_and_merge_grids(self):
        # Shrink the x limit to exercise padding and 64-bit row reconstruction
        # on real hardware without allocating multi-gigabyte operands.
        with patch.object(grouped, "_MAX_GRID_X", 257, create=True):
            self.test_long_query_grid_preserves_batch_head_and_split_mapping()

    def test_paged_small_batches_and_general_geometry_use_tensor_cores(self):
        # These used to select the vector kernel by batch, GQA, width, page or D.
        cases = (
            (2, 12, 1, 1, 256, 128, 2051),
            (4, 12, 1, 1, 256, 128, 2051),
            (1, 12, 1, 4, 256, 128, 2051),
            (1, 3, 1, 1, 256, 128, 2051),
            (3, 12, 1, 1, 256, 128, 2051),
            (1, 6, 1, 7, 256, 128, 2051),
            (1, 2, 2, 2, 8, 4, 3),
            (2, 6, 2, 3, 64, 32, 65),
            (1, 40, 2, 3, 32, 32, 257),
            (1, 1, 1, 2, 512, 4, 3),
            (1, 1, 1, 2, 1024, 4, 3),
        )
        for batch, heads, kv_heads, queries, dim, page, width in cases:
            for fp8 in (False, True):
                with self.subTest(
                    shape=(batch, heads, queries, dim, page, width), fp8=fp8
                ):
                    torch.manual_seed(610)
                    q = torch.randn(
                        batch, heads, queries, dim, device="cuda", dtype=torch.bfloat16
                    )
                    cache = torch.randn(
                        4, 2, kv_heads, page, dim, device="cuda", dtype=q.dtype
                    )
                    scales = None
                    reference_cache = cache.double()
                    if fp8:
                        cache = cache.to(torch.float8_e4m3fn)
                        scales = torch.rand(cache.shape[:-1], device="cuda") * 0.5 + 0.5
                        reference_cache = (
                            (cache.float() * scales[..., None]).bfloat16().double()
                        )
                    table = torch.tensor(
                        [3, 1, 2], device="cuda", dtype=torch.int32
                    ).repeat(batch, 1)
                    lengths = torch.full(
                        (batch, queries), 3 * page, device="cuda", dtype=torch.int32
                    )
                    selected = torch.randint(
                        0,
                        3 * page,
                        (batch, queries, width),
                        device="cuda",
                        dtype=torch.int32,
                    )
                    selected[..., width // 2 :] = -1
                    expected = torch.zeros_like(q, dtype=torch.float64)
                    for b in range(batch):
                        for s in range(queries):
                            ids = selected[b, s].long()
                            ids = ids[ids >= 0]
                            k = reference_cache[
                                table[b, ids // page].long(), 0, :, ids % page
                            ].transpose(0, 1)
                            v = reference_cache[
                                table[b, ids // page].long(), 1, :, ids % page
                            ].transpose(0, 1)
                            k, v = (
                                x.repeat_interleave(heads // kv_heads, dim=0)
                                for x in (k, v)
                            )
                            p = (
                                q[b, :, s, None].double() @ k.transpose(1, 2) / dim**0.5
                            ).softmax(-1)
                            expected[b, :, s] = (p @ v).squeeze(1)

                    def invoke():
                        return sparse_paged_gqa_attn(
                            q,
                            cache,
                            table,
                            lengths,
                            selected,
                            page_size=page,
                            kv_scale=scales,
                        )

                    actual = self.run_matrix_kernel(
                        grouped._grouped_sparse_paged_split_kernel, invoke
                    )
                    self.assert_attention(actual, expected)
                    if dim in (8, 64):
                        # Exercise both address specializations for direct output
                        # and split reduction without allocating multi-GiB inputs.
                        wide = self.run_matrix_kernel(
                            grouped._grouped_sparse_paged_split_kernel,
                            invoke,
                            wide_address=True,
                        )
                        torch.testing.assert_close(wide, actual, atol=0, rtol=0)

    def test_dense_short_rows_and_large_head_groups_use_tensor_cores(self):
        for heads, kv_heads, dim, width in (
            (1, 1, 8, 3),
            (6, 2, 64, 17),
            (40, 2, 32, 65),
            (12, 1, 256, 257),
            (1, 1, 512, 17),
            (1, 1, 1024, 17),
        ):
            with self.subTest(heads=heads, dim=dim, width=width):
                torch.manual_seed(611)
                q = torch.randn(1, heads, 3, dim, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(1, kv_heads, 33, dim, device="cuda", dtype=q.dtype)
                v = torch.randn_like(k)
                selected = torch.randint(
                    0, 33, (1, 3, width), device="cuda", dtype=torch.int32
                )
                selected[:, 0] = -1
                expected = torch.zeros_like(q, dtype=torch.float64)
                for s in (1, 2):
                    ids = selected[0, s].long()
                    kk, vv = (
                        x[0, :, ids]
                        .double()
                        .repeat_interleave(heads // kv_heads, dim=0)
                        for x in (k, v)
                    )
                    p = (
                        q[0, :, s, None].double() @ kk.transpose(1, 2) / dim**0.5
                    ).softmax(-1)
                    expected[0, :, s] = (p @ vv).squeeze(1)
                actual = self.run_matrix_kernel(
                    dense._sparse_prefill_grouped_kernel,
                    lambda: dense.sparse_prefill_attn(q, k, v, selected),
                )
                self.assert_attention(actual, expected)


if __name__ == "__main__":
    unittest.main()
