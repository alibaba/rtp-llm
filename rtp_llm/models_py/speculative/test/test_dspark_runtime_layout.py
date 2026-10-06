"""Layout metadata must not create one Triton binary per context width."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.dspark_swa import _paged_swa, paged_gqa_swa
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_grouped_index_score import (
    _ragged_grouped_index_score_kernel,
    q8kv4_ragged_grouped_index_score,
)


def variants(kernel):
    return len(kernel.device_caches[torch.cuda.current_device()][0])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class RuntimeLayoutTest(unittest.TestCase):
    def test_swa_page_table_width_is_runtime_data(self):
        torch.manual_seed(1004)
        q = torch.randn(3, 7, 4, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(3, 7, 1, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        cache = torch.randn(9, 2, 1, 128, 128, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(9, device="cuda", dtype=torch.int32).view(3, 3)
        lens = torch.tensor([0, 129, 257], device="cuda", dtype=torch.int32)
        live = torch.tensor([7, 3, 0], device="cuda", dtype=torch.int32)
        for causal in (False, True):
            expected = None
            compiled = None
            for width in (3, 5, 16, 17, 257):
                table = torch.full((3, width), -1, device="cuda", dtype=torch.int32)
                table[:, :3].copy_(ids)
                out = torch.empty_like(q)

                def run():
                    return paged_gqa_swa(
                        q,
                        k,
                        v,
                        cache,
                        table,
                        lens,
                        live,
                        window_left=127,
                        causal=causal,
                        out=out,
                    )

                run()
                if expected is None:
                    expected = out.clone()
                    compiled = variants(_paged_swa)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                self.assertEqual(variants(_paged_swa), compiled)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                out.fill_(100)
                graph.replay()
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                self.assertEqual(int(out[2].count_nonzero()), 0)
                # Reuse the same graph addresses after lengths/slots change.
                table[:, :3].copy_(ids.roll(1, dims=1))
                lens.copy_(
                    torch.tensor([129, 1, 257], device="cuda", dtype=torch.int32)
                )
                live.copy_(torch.tensor([3, 1, 0], device="cuda", dtype=torch.int32))
                run()
                changed = out.clone()
                out.fill_(100)
                graph.replay()
                torch.testing.assert_close(out, changed, atol=0, rtol=0)
                self.assertEqual(int(out[2].count_nonzero()), 0)
                table[:, :3].copy_(ids)
                lens.copy_(
                    torch.tensor([0, 129, 257], device="cuda", dtype=torch.int32)
                )
                live.copy_(torch.tensor([7, 3, 0], device="cuda", dtype=torch.int32))

    def test_ragged_score_rows_columns_and_stride_are_runtime_data(self):
        torch.manual_seed(1004)
        # Uniform FP4 pages: dequantization gives one, so all valid pages
        # score identically independent of their physical address.
        packed = torch.full((4, 8192), 0x22, device="cuda", dtype=torch.uint8)
        scales = torch.ones((4, 1024), device="cuda", dtype=torch.float8_e4m3fn)
        q = torch.ones((16, 4, 128), device="cuda", dtype=torch.float8_e4m3fn)
        compiled = None
        for tokens in (9, 16):
            offsets = [0, 1, 4, tokens]
            cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
            lens = torch.full((tokens,), 256, device="cuda", dtype=torch.int32)
            for columns, stride in ((3, 3), (5, 7), (16, 16), (17, 19), (127, 129)):
                backing = torch.full(
                    (tokens, stride), 2000000000, device="cuda", dtype=torch.int32
                )
                table = backing[:, :columns]
                table[:, :2].copy_(
                    torch.tensor([0, 1], device="cuda", dtype=torch.int32)
                )
                out = torch.empty(
                    (4, tokens, columns), device="cuda", dtype=torch.float32
                )

                def run():
                    return q8kv4_ragged_grouped_index_score(
                        q[:tokens],
                        packed,
                        scales,
                        table,
                        lens,
                        out,
                        cu_seqlens=cu,
                        max_query_width=16,
                        init_blocks=0,
                        local_blocks=0,
                        sm_scale=1.0,
                    )

                run()
                if compiled is None:
                    compiled = variants(_ragged_grouped_index_score_kernel)
                self.assertEqual(variants(_ragged_grouped_index_score_kernel), compiled)
                torch.testing.assert_close(
                    out[:, :, :2], torch.full_like(out[:, :, :2], 128), atol=0, rtol=0
                )
                self.assertTrue(bool(torch.isneginf(out[:, :, 2:]).all()))
                expected = out.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                out.fill_(100)
                graph.replay()
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                # Heterogeneous physical pages reveal wrong-page selection.
                packed[0].fill_(0x11)
                packed[1].fill_(0x22)
                packed[2].fill_(0x44)
                packed[3].fill_(0x66)
                table[:, :2].copy_(
                    torch.tensor([3, 2], device="cuda", dtype=torch.int32)
                )
                lens.fill_(129)
                lens[0] = 128
                cu.copy_(
                    torch.tensor([0, 2, 5, tokens], device="cuda", dtype=torch.int32)
                )
                run()
                changed = out.clone()
                torch.testing.assert_close(
                    changed[:, :, 0],
                    torch.full_like(changed[:, :, 0], 512),
                    atol=0,
                    rtol=0,
                )
                self.assertTrue(bool(torch.isneginf(changed[:, 0, 1:]).all()))
                torch.testing.assert_close(
                    changed[:, 1:, 1],
                    torch.full_like(changed[:, 1:, 1], 256),
                    atol=0,
                    rtol=0,
                )
                out.fill_(100)
                graph.replay()
                torch.testing.assert_close(out, changed, atol=0, rtol=0)
                self.assertEqual(variants(_ragged_grouped_index_score_kernel), compiled)
                packed.fill_(0x22)
                table[:, :2].copy_(
                    torch.tensor([0, 1], device="cuda", dtype=torch.int32)
                )
                lens.fill_(256)
                cu.copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
