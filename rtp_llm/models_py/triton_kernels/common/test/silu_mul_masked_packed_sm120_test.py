"""SM120 packed activation contract: tails and padding must not depend on memory."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.common.silu_mul_masked_packed_sm120 import (
    create_packed_scale_tensor,
    silu_and_mul_masked_post_quant_packed_fwd,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulMaskedPackedSM120Test(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("SM120 required")
        torch.manual_seed(123)

    def test_qwen_six_groups(self):
        scale = create_packed_scale_tensor(3, 129, 1536, 128, torch.device("cuda"))
        self.assertEqual(scale.shape, (3, 129, 2))
        self.assertEqual(scale.stride(), (264, 1, 132))

    def test_padding_is_overwritten(self):
        x = torch.randn(2, 128, 1024, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        out = torch.empty((2, 128, 512), device="cuda", dtype=torch.float8_e4m3fn)
        scale = create_packed_scale_tensor(2, 128, 1024, 128, x.device)
        scale.fill_(-1)
        silu_and_mul_masked_post_quant_packed_fwd(x, out, scale, 128, counts)
        self.assertTrue(torch.all(scale[0] == 0).item())
        self.assertTrue(torch.all(scale[1, 1:] == 0).item())

    @staticmethod
    def _storage_view(scale, e, m, h):
        aligned_m = (m + 3) // 4 * 4
        packed_g = (h // 128 + 3) // 4
        return scale.as_strided(
            (e, aligned_m, packed_g), (packed_g * aligned_m, 1, aligned_m)
        )

    def _assert_reference(self, x, out, scale, counts, counts_cpu):
        e, m, twice_h = x.shape
        h = twice_h // 2
        ref_out = torch.empty_like(out)
        ref_scale = torch.zeros((e, m, h // 128), device=x.device)
        silu_mul_masked_fp8_post_quant_fwd(
            x,
            ref_out,
            ref_scale,
            128,
            counts,
            expected_m=max(1, sum(counts_cpu) // e),
            scale_ue8m0=True,
        )
        # Independent packing in torch, including physical alignment rows.
        aligned_m = (m + 3) // 4 * 4
        packed_g = (h // 128 + 3) // 4
        exp = torch.zeros(
            (e, aligned_m, packed_g * 4), device=x.device, dtype=torch.int64
        )
        exp[:, :m, : h // 128] = (
            ref_scale.view(torch.int32).to(torch.int64) >> 23
        ) & 255
        exp = exp.view(e, aligned_m, packed_g, 4)
        expected = sum(exp[..., i] << (8 * i) for i in range(4)).to(torch.int32)
        torch.testing.assert_close(
            self._storage_view(scale, e, m, h), expected, rtol=0, atol=0
        )
        for expert, n in enumerate(counts_cpu):
            torch.testing.assert_close(
                out[expert, :n].view(torch.uint8),
                ref_out[expert, :n].view(torch.uint8),
                rtol=0,
                atol=0,
            )

    def test_tails_and_capacity_boundaries_match_unpacked(self):
        # Six groups catches the Qwen3 shape; 1/2/3 cover TP-sized tail packs.
        for h in (128, 256, 384, 512, 768, 1024, 2048):
            for m in (1, 127, 128, 129, 257):
                with self.subTest(h=h, m=m):
                    counts_cpu = [0, 1, m]
                    counts = torch.tensor(counts_cpu, device="cuda", dtype=torch.int32)
                    x = torch.randn((3, m, 2 * h), device="cuda", dtype=torch.bfloat16)
                    x[0].fill_(float("nan"))
                    x[1, 1:].fill_(float("nan"))
                    out = torch.empty(
                        (3, m, h), device="cuda", dtype=torch.float8_e4m3fn
                    )
                    scale = create_packed_scale_tensor(3, m, 2 * h, 128, x.device)
                    self._storage_view(scale, 3, m, h).fill_(-1)
                    silu_and_mul_masked_post_quant_packed_fwd(
                        x, out, scale, 128, counts
                    )
                    self._assert_reference(x, out, scale, counts, counts_cpu)

    def test_single_expert_zero_and_scale_boundaries(self):
        h, m = 768, 128
        x = torch.zeros((1, m, 2 * h), device="cuda", dtype=torch.bfloat16)
        # Unequal halves detect accidentally swapping gate and up. Sweep scales
        # across powers of two and include the all-zero epsilon case.
        x[:, 1:, :h] = torch.logspace(-9, 4, m - 1, base=2, device="cuda")[:, None]
        x[:, 1:, h:] = 0.75
        counts = torch.tensor([m], device="cuda", dtype=torch.int32)
        out = torch.empty((1, m, h), device="cuda", dtype=torch.float8_e4m3fn)
        scale = create_packed_scale_tensor(1, m, 2 * h, 128, x.device)
        self._storage_view(scale, 1, m, h).fill_(-1)
        silu_and_mul_masked_post_quant_packed_fwd(x, out, scale, 128, counts)
        self._assert_reference(x, out, scale, counts, [m])

    def test_graph_replay_overwrites_previous_routes(self):
        e, m, h = 3, 129, 768
        x = torch.randn((e, m, 2 * h), device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([m, m, m], device="cuda", dtype=torch.int32)
        out = torch.empty((e, m, h), device="cuda", dtype=torch.float8_e4m3fn)
        scale = create_packed_scale_tensor(e, m, 2 * h, 128, x.device)

        def run():
            silu_and_mul_masked_post_quant_packed_fwd(x, out, scale, 128, counts)

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for counts_cpu in ([m, m, m], [1, 0, 127], [0, 0, 0], [m, 128, 1]):
            counts.copy_(torch.tensor(counts_cpu, device="cuda", dtype=torch.int32))
            self._storage_view(scale, e, m, h).fill_(-1)
            graph.replay()
            self._assert_reference(x, out, scale, counts, counts_cpu)

    def test_masked_down_gemm_consumes_tail_packs(self):
        import deep_gemm

        for e in (1, 3):
            m, h, n = 128, 768, 256
            x = torch.randn((e, m, h * 2), device="cuda", dtype=torch.bfloat16)
            counts_cpu = [1] if e == 1 else [0, 1, 127]
            counts = torch.tensor(counts_cpu, device="cuda", dtype=torch.int32)
            out = torch.empty((e, m, h), device="cuda", dtype=torch.float8_e4m3fn)
            scale = create_packed_scale_tensor(e, m, 2 * h, 128, x.device)
            self._storage_view(scale, e, m, h).fill_(-1)
            silu_and_mul_masked_post_quant_packed_fwd(x, out, scale, 128, counts)
            w = (torch.randn((e, n, h), device="cuda") * 0.25).to(torch.float8_e4m3fn)
            # Packed unit scales: no conversion kernel on either operand.
            ws = torch.full(
                (e, 2, n), 0x7F7F7F7F, device="cuda", dtype=torch.int32
            ).transpose(1, 2)
            d = torch.empty((e, m, n), device="cuda", dtype=torch.bfloat16)
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                (out, scale),
                (w, ws),
                d,
                counts,
                max(1, sum(counts_cpu) // e),
                disable_ue8m0_cast=False,
            )
            # Reference dequantization is independent of the GEMM consumer.
            exp = torch.stack(
                [(scale.to(torch.int64) >> (8 * i)) & 255 for i in range(4)], dim=-1
            ).flatten(-2)[..., : h // 128]
            sf = torch.exp2(exp.float() - 127).repeat_interleave(128, dim=-1)
            for expert, count in enumerate(counts_cpu):
                if count:
                    expected = (out[expert, :count].float() * sf[expert, :count]) @ w[
                        expert
                    ].float().T
                    torch.testing.assert_close(
                        d[expert, :count].float(), expected, rtol=0.01, atol=0.08
                    )

    def test_nonzero_offset_expert_slices(self):
        e, m, h = 3, 129, 768
        x = torch.randn((e, m, 2 * h), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((e, m, h), device="cuda", dtype=torch.float8_e4m3fn)
        scale = create_packed_scale_tensor(e, m, 2 * h, 128, x.device)
        counts = torch.tensor([m, 1, 127], device="cuda", dtype=torch.int32)
        storage = self._storage_view(scale, e, m, h)
        storage.fill_(-1)
        silu_and_mul_masked_post_quant_packed_fwd(
            x[1:], out[1:], scale[1:], 128, counts[1:]
        )
        self._assert_reference(x[1:], out[1:], scale[1:], counts[1:], [1, 127])
        self.assertTrue(torch.all(storage[0] == -1).item())

    def test_rejects_storage_missing_physical_padding(self):
        x = torch.zeros((3, 129, 1536), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((3, 129, 768), device="cuda", dtype=torch.float8_e4m3fn)
        counts = torch.tensor([0, 1, 129], device="cuda", dtype=torch.int32)
        # Correct logical layout but only 789 words; kernel requires 792.
        scale = torch.empty_strided(
            (3, 129, 2), (264, 1, 132), device="cuda", dtype=torch.int32
        )
        with self.assertRaises(AssertionError):
            silu_and_mul_masked_post_quant_packed_fwd(x, out, scale, 128, counts)


if __name__ == "__main__":
    unittest.main()
