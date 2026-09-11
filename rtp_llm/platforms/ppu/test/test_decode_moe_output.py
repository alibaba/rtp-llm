"""Native BF16 combine storage must preserve the FP32 shared-add boundary."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.shared_expert import fused_moe_epilogue
from rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency import (
    low_latency_mxfp4_moe,
)


class MoeOutputContractTest(unittest.TestCase):
    def test_invalid_output_dtype_fails_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, "routed output"):
            low_latency_mxfp4_moe(
                None,
                None,
                None,
                None,
                None,
                None,
                num_experts=256,
                max_dispatch_tokens=128,
                expected_m=24,
                output_dtype=torch.float16,
            )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class MoeOutputGraphTest(unittest.TestCase):
    @torch.inference_mode()
    def test_native_storage_and_materialized_fp32_have_identical_epilogues(self):
        torch.manual_seed(890432)
        for batch in (1, 3, 8, 32, 128):
            for shared_dtype in (torch.bfloat16, torch.float32):
                for strided in (False, True):
                    width = 4096
                    storage_width = width * (2 if strided else 1)
                    routed = torch.empty(
                        (batch, storage_width), device="cuda", dtype=torch.bfloat16
                    )[:, :width]
                    shared = torch.empty(
                        (batch, storage_width), device="cuda", dtype=shared_dtype
                    )[:, :width]
                    actual = torch.empty_like(routed)

                    def run():
                        return fused_moe_epilogue(
                            routed, shared, torch.bfloat16, out=actual
                        )

                    routed.normal_()
                    shared.normal_()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        run()
                    torch.cuda.current_stream().wait_stream(stream)
                    address = actual.data_ptr()
                    for step in range(5):
                        routed.normal_().mul_(2.0 ** (step * 4 - 8))
                        # Cancellation plus a small FP32 residual detects any
                        # premature cast of the shared result to BF16.
                        shared.copy_(
                            -routed.float() + torch.randn_like(routed.float()) / 128
                        )
                        expected = (routed.float() + shared.float()).bfloat16()
                        materialized = fused_moe_epilogue(
                            routed.float(), shared, torch.bfloat16
                        )
                        actual.fill_(float("nan"))
                        graph.replay()
                        self.assertEqual(actual.data_ptr(), address)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(actual, materialized, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
