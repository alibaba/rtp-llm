"""Native Engram transition parity, changing-input graph replay and ownership."""

import os
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class EngramTransitionCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (
            torch.cuda.get_device_capability() != (10, 3)
            or torch.cuda.get_device_properties(0).multi_processor_count != 148
        ):
            raise unittest.SkipTest("SM103 / 148 SMs required")
        from rtp_llm.models_py.modules.dsv41.hc import engram_cuda

        if not engram_cuda.available():
            raise RuntimeError(
                "Build RTP's CUDA13 Engram bindings before running this test"
            )
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        torch.manual_seed(41)
        cls.projection = V41MXFP8Linear(
            (torch.randn(25600, 6144, device="cuda") * 0.1).to(torch.float8_e4m3fn),
            torch.ones((800, 192), device="cuda").to(torch.float8_e8m0fnu),
        )

    def assert_exact(self, actual, expected):
        for i, (got, ref) in enumerate(zip(actual, expected)):
            with self.subTest(output=i):
                self.assertEqual(got.dtype, ref.dtype)
                self.assertEqual(got.shape, ref.shape)
                self.assertTrue(
                    torch.equal(
                        got.contiguous().view(torch.uint8),
                        ref.contiguous().view(torch.uint8),
                    )
                )

    def make_pair(self, m):
        from rtp_llm.models_py.modules.dsv41.test.test_v41_mega_mhc import make_case
        from rtp_llm.models_py.modules.dsv41.engram import gated_engram_residual
        from rtp_llm.models_py.modules.dsv41.hc.engram_cuda import State
        from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import prepare_tf32_weight
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            sgl_per_token_group_quant_fp8_v3,
        )

        case = make_case((m,))
        previous, next_hc = case["previous"], case["next_hc"]
        x, residual = case["attn_out"], case["residual"]
        post, comb = case["post"].clone(), case["comb"].clone()
        shifted = previous.pre_mix_out.reshape(m, 4, 1).clone()
        previous.pre_mix_out = shifted.reshape(m, 4)
        rows = torch.randn(m, 6144, device="cuda", dtype=torch.bfloat16) * 0.25
        a, sf = self.projection._quantize_input(rows)
        qw = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
        kw = torch.randn_like(qw)
        mask = torch.ones(m, device="cuda", dtype=torch.bool)
        state = State(
            self.projection.weight,
            self.projection.weight_scales,
            prepare_tf32_weight(next_hc.fn),
            next_hc.scale,
            next_hc.base,
            case["norm"].weight,
            rows=m,
            gate_eps=1e-6,
            hc_norm_eps=1e-6,
            hc_eps=1e-6,
            sinkhorn_iters=20,
            rmsnorm_eps=1e-6,
        )

        def native():
            h = previous.post(x, residual.clone(), post, comb)
            kv = self.projection.forward_quantized(a, sf)
            h = gated_engram_residual(h, kv, qw, kw, 1e-6, mask, out=h)
            y, np, nc = next_hc.pre(h)
            y = case["norm"](y)
            q, qsf = sgl_per_token_group_quant_fp8_v3(y)
            return h, y, np, nc, next_hc.pre_mix_out.reshape(m, 4, 1), q, qsf

        def fused():
            return state.forward(a, sf, x, residual, post, comb, shifted, qw, kw, mask)

        return native, fused, (a, sf, x, residual, post, comb, shifted, mask)

    @torch.inference_mode()
    @patch.dict(
        os.environ, {"DSV4_MHC_PRE_GEMM_BACKEND": "auto", "DSV4_FP8_QUANT_KERNEL": "v3"}
    )
    def test_eager_and_graph_all_tile_and_split_boundaries(self):
        # Cover all twelve WKV configurations and all eight resident Pre split counts.
        for m in (
            1,
            6,
            33,
            65,
            128,
            129,
            193,
            257,
            289,
            321,
            353,
            385,
            417,
            449,
            481,
            512,
        ):
            with self.subTest(rows=m):
                native, fused, inputs = self.make_pair(m)
                for _ in range(3):
                    expected, actual = native(), fused()
                self.assert_exact(actual, expected)
                held = tuple(t.clone() for t in actual)
                again = fused()
                for got, before in zip(actual, held):
                    self.assertTrue(
                        torch.equal(
                            got.contiguous().view(torch.uint8),
                            before.contiguous().view(torch.uint8),
                        )
                    )
                for got, other in zip(actual, again):
                    self.assertNotEqual(got.data_ptr(), other.data_ptr())
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    # States are stream-owned; build and warm this pair on the capture stream.
                    native, fused, inputs = self.make_pair(m)
                    for _ in range(3):
                        native()
                        fused()
                    native_graph, fused_graph = (
                        torch.cuda.CUDAGraph(),
                        torch.cuda.CUDAGraph(),
                    )
                    with torch.cuda.graph(native_graph, stream=stream):
                        expected = native()
                    with torch.cuda.graph(fused_graph, stream=stream):
                        actual = fused()
                    for phase in range(3):
                        inputs[2].normal_(0, 0.25)
                        inputs[3].normal_(0, 0.25)
                        inputs[6].fill_(0.2 + 0.1 * phase)
                        inputs[7].fill_(phase != 1)
                        if phase == 2:
                            inputs[7][::3] = False
                        for _ in range(20):
                            native_graph.replay()
                            fused_graph.replay()
                        self.assert_exact(actual, expected)
                torch.cuda.current_stream().wait_stream(stream)

    @torch.inference_mode()
    def test_state_rejects_cross_stream_reuse(self):
        _, fused, _ = self.make_pair(6)
        with torch.cuda.stream(torch.cuda.Stream()):
            with self.assertRaisesRegex(RuntimeError, "another CUDA stream"):
                fused()


if __name__ == "__main__":
    unittest.main()
