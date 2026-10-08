"""Match fused output bytes/scales to BF16 combine + production MXFP8 quant."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_attention import (
    LOG2E_F32,
    _q8kv4_combine,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_combine_mxfp8 import (
    _combine_mxfp8,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CombineMxfp8Test(unittest.TestCase):
    def test_quantization_and_graph_replay(self):
        if torch.cuda.get_device_capability() != (10, 3):
            self.skipTest("MXFP8 fused consumer is gated to SM103")
        for rows in (1, 3, 5, 20, 40, 100):
            with self.subTest(rows=rows):
                self.check_rows(rows)

    def check_rows(self, rows):
        torch.manual_seed(1847 + rows)
        partial = torch.randn(rows, 4, 16, 16, 128, device="cuda", dtype=torch.bfloat16)
        lse = torch.randn(rows, 4, 16, 16, device="cuda")
        counts = torch.randint(0, 17, (rows, 4), device="cuda", dtype=torch.int32)
        valid = torch.ones(rows, device="cuda", dtype=torch.bool)
        bf16 = torch.empty(rows, 64, 128, device="cuda", dtype=torch.bfloat16)
        fp8 = torch.empty_like(bf16, dtype=torch.float8_e4m3fn)
        aligned_m = (rows + 3) // 4 * 4
        scales = torch.empty_strided(
            (rows, 64), (1, aligned_m), device="cuda", dtype=torch.int32
        )
        mask_args = dict(
            valid_token_mask=valid,
            HAS_VALID_TOKEN_MASK=True,
            valid_token_mask_stride=valid.stride(0),
        )

        def launch():
            _combine_mxfp8[(rows, 4)](
                partial,
                lse,
                counts,
                fp8.view(torch.uint8),
                scales,
                ALIGNED_M=aligned_m,
                LOG2E=LOG2E_F32,
                num_warps=4,
                **mask_args
            )

        def check():
            _q8kv4_combine[(rows, 4)](
                partial,
                lse,
                counts,
                bf16,
                bf16.stride(0),
                bf16.stride(1),
                LOG2E=LOG2E_F32,
                num_warps=4,
                **mask_args
            )
            import flashinfer

            expected, scale_u8 = flashinfer.mxfp8_quantize(
                bf16.reshape(rows, -1),
                is_sf_swizzled_layout=False,
                alignment=32,
                backend="cute-dsl",
            )
            # Independent byte packing reference for the int32 TMA ABI;
            # avoid coupling the oracle to the production Triton packer.
            scale_groups = scale_u8.reshape(rows, 64, 4).to(torch.int64)
            shifts = torch.arange(4, device="cuda", dtype=torch.int64) * 8
            expected_scales = (scale_groups << shifts).sum(-1).to(torch.int32)
            self.assertTrue(
                torch.equal(
                    fp8.view(torch.uint8).reshape(rows, -1), expected.view(torch.uint8)
                )
            )
            self.assertTrue(torch.equal(scales, expected_scales))

        launch()
        check()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        pointers = (fp8.data_ptr(), scales.data_ptr())
        for live, amplitude in (
            (0, 1.0),
            (rows // 2, 1e-4),
            (rows, 64.0),
            (max(0, rows - 1), 1.0),
        ):
            valid.copy_(torch.arange(rows, device="cuda") < live)
            partial.normal_().mul_(amplitude)
            partial[~valid] = float("nan")
            counts.random_(0, 17)
            counts[0, 0] = 0
            lse.normal_().mul_(64)
            graph.replay()
            check()
            self.assertEqual(pointers, (fp8.data_ptr(), scales.data_ptr()))


if __name__ == "__main__":
    unittest.main()
