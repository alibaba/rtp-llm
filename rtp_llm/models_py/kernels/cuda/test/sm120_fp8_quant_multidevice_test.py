"""Multi-GPU SM120 FP8 quant v2 and DeepGEMM linear device/stream guard test."""

from __future__ import annotations

import os
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    quant_weight_ue8m0_packed,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import _transform_scale_ue8m0
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.utils.arch import is_sm120


class Sm120Fp8QuantMultideviceTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("SM120 multi-device gate requires >= 2 CUDA devices")
        if not (is_sm120(0) and is_sm120(1)):
            self.skipTest(
                "SM120 multi-device gate requires SM120 on devices 0 and 1"
            )
        self._prev_device = torch.cuda.current_device()
        torch.cuda.set_device(0)
        torch.manual_seed(17)

    def tearDown(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(self._prev_device)

    @staticmethod
    def _assert_quantized_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
        actual_f = actual.float()
        expected_f = expected.float()
        assert torch.isfinite(actual_f).all(), "linear output is non-finite"
        rel_l2 = (
            (actual_f - expected_f).norm() / expected_f.norm().clamp_min(1e-10)
        ).item()
        assert rel_l2 < 0.06, f"relative L2 error {rel_l2:.5f} >= 0.06"

    def test_quant_v2_and_deepgemm_linear_nondefault_device_and_stream(self) -> None:
        if not is_deep_gemm_e8m0_used():
            self.skipTest("DeepGEMM E8M0 must be active for SM120 multidevice test")
        dev1 = torch.device("cuda", 1)
        with torch.cuda.device(dev1):
            raw_w = torch.randn((256, 256), dtype=torch.bfloat16, device=dev1) * 0.05
            weight, scales = quant_weight_ue8m0_packed(raw_w)
            linear = CudaFp8DeepGEMMLinear(weight, scales)

            src_a = torch.randn((17, 256), dtype=torch.bfloat16, device=dev1)
            src_b = src_a * 0.5 + 0.25
            ref_q_a, ref_s_a = sgl_per_token_group_quant_fp8(
                src_a,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=False,
                scale_ue8m0=False,
                enable_v2=True,
            )
            ref1 = src_a.float() @ raw_w.float().t()
            ref2 = src_b.float() @ raw_w.float().t()
            torch.cuda.synchronize(dev1)

        torch.cuda.set_device(0)
        self.assertEqual(torch.cuda.current_device(), 0)
        stream1 = torch.cuda.Stream(device=dev1)
        activation = torch.empty_like(src_a)

        with torch.cuda.device(dev1):
            prev_stream1 = torch.cuda.current_stream(dev1)
            torch.cuda.set_stream(stream1)
        try:
            torch.cuda.set_device(0)
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(torch.cuda.current_stream(dev1), stream1)

            q_mismatched, s_mismatched = sgl_per_token_group_quant_fp8(
                src_a,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=False,
                scale_ue8m0=False,
                enable_v2=True,
            )
            activation.copy_(src_a)
            out1 = linear(activation)
            stream1.synchronize()
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(out1.device, dev1)
            torch.testing.assert_close(
                q_mismatched.float(), ref_q_a.float(), rtol=0, atol=0
            )
            torch.testing.assert_close(s_mismatched, ref_s_a, rtol=0, atol=0)
            self._assert_quantized_close(out1, ref1)

            activation.mul_(0.5).add_(0.25)
            out2 = linear(activation)
            stream1.synchronize()
            self.assertEqual(torch.cuda.current_device(), 0)
            self.assertEqual(out2.device, dev1)
            self._assert_quantized_close(out2, ref2)
        finally:
            with torch.cuda.device(dev1):
                torch.cuda.set_stream(prev_stream1)
            torch.cuda.set_device(0)

    def test_transform_scale_ue8m0_honors_explicit_cuda_device_override(self) -> None:
        # Non-4-aligned mn=129 exercises TMA-padded strides (1, 132) across devices.
        sf_on_0 = torch.full((2, 8), 2.0, dtype=torch.float32, device="cuda:0")
        packed = _transform_scale_ue8m0(sf_on_0, mn=129, cuda_device=1)
        self.assertEqual(packed.device, sf_on_0.device)
        packed_ref = _transform_scale_ue8m0(sf_on_0.cuda(1), mn=129, cuda_device=1)
        self.assertEqual(packed.stride(), packed_ref.stride())
        torch.testing.assert_close(packed.cpu(), packed_ref.cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    cuda_version = torch.version.cuda
    if not cuda_version or cuda_version.split(".", 1)[0] != "13":
        raise RuntimeError(f"CUDA 13 runtime required, got {cuda_version}")
    if os.environ.get("RTP_LLM_REQUIRE_SM120") == "1":
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() < 2
            or not (is_sm120(0) and is_sm120(1))
        ):
            raise RuntimeError(
                "2-GPU SM120 CUDA 13 runtime required for multidevice test"
            )
    result = unittest.main(exit=False).result
    if result.testsRun == len(result.skipped):
        raise RuntimeError("No SM120 FP8 multidevice hardware test ran")
    raise SystemExit(int(not result.wasSuccessful()))
