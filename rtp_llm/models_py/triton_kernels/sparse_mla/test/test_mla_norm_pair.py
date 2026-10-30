import unittest
from unittest.mock import patch

import torch
from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
    fused_strided_rmsnorm,
    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
)
from rtp_llm.models_py.triton_kernels.common.mla_norm_pair import try_mla_norm_pair


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MlaNormPairTest(unittest.TestCase):
    def test_scale_view_preserves_padded_storage(self):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale as allocate,
        )

        for shape in ((0, 2048), (1, 128), (17, 2048), (2024, 2048), (2, 17, 512)):
            options = dict(
                group_size=128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
            with patch.dict(allocate.__globals__, {"_CPU_PATH": False}):
                before = allocate(shape, "cpu", **options)
            with patch.dict(allocate.__globals__, {"_CPU_PATH": True}):
                after = allocate(shape, "cpu", **options)
            self.assertEqual(before.shape, after.shape)
            self.assertEqual(before.stride(), after.stride())
            self.assertEqual(before.storage_offset(), after.storage_offset())
            self.assertEqual(
                before.untyped_storage().nbytes(), after.untyped_storage().nbytes()
            )

    def test_exact_outputs(self):
        torch.manual_seed(769)
        for n in (0, 1, 17, 2024, 4093):
            for fp8, ue8 in ((True, True), (True, False), (False, False)):
                x = torch.randn(n, 2624, device="cuda", dtype=torch.bfloat16)
                qw = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
                kw = torch.randn(512, device="cuda", dtype=torch.bfloat16)
                q, kv = x[:, :2048], x[:, 2048:2560]
                got = try_mla_norm_pair(
                    q, kv, qw, kw, 1.0e-5, 1.0e-6, fp8=fp8, scale_ue8m0=ue8
                )
                refq = (
                    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                        q, qw, 1.0e-5, scale_ue8m0=ue8
                    )
                    if fp8
                    else (fused_strided_rmsnorm(q, qw, 1.0e-5), None, None)
                )
                ref = (*refq, fused_strided_rmsnorm(kv, kw, 1.0e-6))
                for i, (a, b) in enumerate(zip(got, ref)):
                    if a is not None:
                        self.assertTrue(
                            torch.equal(
                                a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8),
                            ),
                            f"N={n} fp8={fp8} ue8={ue8} output={i}",
                        )


if __name__ == "__main__":
    unittest.main()
