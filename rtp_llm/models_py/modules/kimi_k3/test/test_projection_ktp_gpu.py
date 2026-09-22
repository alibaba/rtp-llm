"""Manual CUDA correctness for the K3 projection-KTP wrapper."""

import unittest
from functools import partial

import torch

import rtp_llm.models_py.modules.kimi_k3.projection_ktp as projection_ktp
from rtp_llm.model_loader.per_block_fp8_quant_weight import per_block_cast_to_fp8
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import _transform_scale_ue8m0
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)

assert_exact = partial(torch.testing.assert_close, rtol=0, atol=0)


def projection_options(ranks=8, rank=0):
    return dict(
        total_heads=96,
        head_dim=128,
        forget_latent_size=128,
        ktp_size=ranks,
        ktp_rank=rank,
    )


def bf16(*shape):
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectionKtpGpuTest(unittest.TestCase):
    def test_fp8_strided_forget_is_bitwise_exact(self):
        width, rows = 1536, 8
        projected = bf16(rows, 4 * width + 128 + 96)
        hidden = bf16(rows, 64)
        weight, scales = per_block_cast_to_fp8(
            bf16(width, 128) * 0.02, 128, use_ue8m0=True
        )
        forget = CudaFp8DeepGEMMLinear(weight, _transform_scale_ue8m0(scales, mn=width))
        expected_gate = forget(projected[:, 4 * width : 4 * width + 128].contiguous())
        expected = torch.cat(
            (projected[:, : 4 * width], expected_gate, projected[:, -96:-84]), dim=1
        )
        output = torch.empty_like(expected)

        def run():
            return projection_ktp.pack_ktp_projection_payload(
                hidden,
                lambda _: projected,
                forget,
                output=output,
                **projection_options()
            )

        actual = run()
        self.assertIs(actual, output)
        assert_exact(actual, expected)
        # Packing writes the communication buffer directly, without a temporary
        # concatenation, even when no optimization flag is configured.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            run()
        self.assertNotIn("aten::cat", {event.key for event in profile.key_averages()})


if __name__ == "__main__":
    unittest.main()
