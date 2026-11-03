import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
    fused_strided_rmsnorm,
    fused_strided_rmsnorm_per_token_fp8_quant,
    fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output,
)
from rtp_llm.models_py.triton_kernels.common.mla_norm_pair import try_mla_norm_pair


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MlaNormPairTest(unittest.TestCase):

    def test_mxfp8_group32_matches_separate_norms(self):
        torch.manual_seed(759)
        for n in (0, 1, 17, 257, 2024):
            x = torch.randn(n, 2624, device="cuda", dtype=torch.bfloat16)
            x[:, :32] = 0
            x[:, 32:64] *= 1e-7
            qw = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
            kw = torch.randn(512, device="cuda", dtype=torch.bfloat16)
            q, kv = x[:, :2048], x[:, 2048:2560]
            for dual in (False, True):
                got = try_mla_norm_pair(
                    q, kv, qw, kw, 1e-5, 1e-6, mxfp8=True, bf16_output=dual
                )
                self.assertIsNotNone(got)
                fp8, scale = fused_strided_rmsnorm_per_token_fp8_quant(
                    q, qw, 1e-5, group_size=32, scale_ue8m0=True, mxfp8_semantics=True
                )
                self.assertEqual(got[2].shape, (n, 16))
                self.assertEqual(got[2].stride(), scale.stride())
                for label, actual, expected in zip(
                    ("fp8", "scale", "kv"),
                    got[1:],
                    (fp8, scale, fused_strided_rmsnorm(kv, kw, 1e-6)),
                ):
                    self.assertTrue(
                        torch.equal(
                            actual.contiguous().view(torch.uint8),
                            expected.contiguous().view(torch.uint8),
                        ),
                        f"n={n}, dual={dual}, component={label}, different_values={(actual.float() != expected.float()).sum().item()}, max_abs={(actual.float()-expected.float()).abs().max().item() if actual.numel() else 0}",
                    )
                if dual:
                    original_dual = (
                        fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
                            q,
                            qw,
                            1e-5,
                            group_size=32,
                            scale_ue8m0=True,
                            mxfp8_semantics=True,
                        )
                    )
                    self.assertTrue(
                        torch.equal(got[0], original_dual[0]),
                        f"n={n}, BF16 output differs from existing MXFP8 dual-output kernel",
                    )
                else:
                    self.assertIsNone(got[0])

    def test_mxfp8_graph_replay_with_changed_inputs(self):
        x = torch.randn(17, 2624, device="cuda", dtype=torch.bfloat16)
        qw = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
        kw = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        q, kv = x[:, :2048], x[:, 2048:2560]

        def forward():
            return try_mla_norm_pair(
                q, kv, qw, kw, 1e-5, 1e-6, mxfp8=True, bf16_output=False
            )

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        for factor in (0.01, 2.0, 100.0):
            x.normal_().mul_(factor)
            graph.replay()
            expected = forward()
            self.assertIsNone(captured[0])
            for actual, reference in zip(captured[1:], expected[1:]):
                self.assertTrue(
                    torch.equal(
                        actual.contiguous().view(torch.uint8),
                        reference.contiguous().view(torch.uint8),
                    )
                )

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
