"""Contract tests for the optional FlashInfer SM12x FP8 adapter."""

import os
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.flashinfer_sm12x_fp8 import (
    FlashInferSm12xFp8Moe,
    prepare_flashinfer_block_scales,
)


class FlashInferSm12xFp8ScaleTest(unittest.TestCase):
    def test_raw_scales_are_transposed_without_weight_requantization(self):
        source = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(2, 2, 3)
        got = prepare_flashinfer_block_scales(
            source, rows=256, cols=384, experts=2, name="w"
        )
        self.assertTrue(torch.equal(got, source.transpose(-1, -2).contiguous()))

    def test_unsupported_scale_does_not_silently_requantize(self):
        source = torch.ones((1, 1, 1), dtype=torch.float16)
        with self.assertRaisesRegex(TypeError, "Refuse"):
            prepare_flashinfer_block_scales(
                source, rows=128, cols=128, experts=1, name="w"
            )

    def test_packed_ue8m0_is_rejected_without_a_trusted_unpacker(self):
        # This adapter accepts only the canonical loader sidecar and must not
        # accidentally interpret an int32 packed scale as FP32 block scales.
        packed = torch.empty((1, 256, 1), dtype=torch.int32)
        with self.assertRaisesRegex(TypeError, "Refuse to unpack"):
            prepare_flashinfer_block_scales(
                packed, rows=256, cols=512, experts=1, name="w"
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA numerical test")
class FlashInferSm12xFp8NumericalTest(unittest.TestCase):
    @staticmethod
    def _dequant_blockwise(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Reference dequantization for canonical [Nblock, Kblock] scales."""

        n, k = values.shape
        expanded = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        return values.float() * expanded[:n, :k]

    @staticmethod
    def _quant_dequant_per_token_128(values: torch.Tensor) -> torch.Tensor:
        """Reference FP8 Q0/Q1 with FlashInfer's 128-wide group granularity."""

        rows, cols = values.shape
        blocks = values.reshape(rows, cols // 128, 128).float()
        scales = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4) / 448.0
        return (
            (blocks / scales)
            .to(torch.float8_e4m3fn)
            .float()
            .mul(scales)
            .reshape_as(values)
        )

    def _assert_fidelity(
        self,
        actual: torch.Tensor,
        reference: torch.Tensor,
        *,
        tokens: int,
        label: str,
        max_relative_l2: float,
        min_cosine: float,
        max_relative_peak: float,
    ) -> None:
        """Check scale-independent error as well as the worst output channel."""

        got, expected = actual.float(), reference.float()
        diff = got - expected
        expected_l2 = expected.norm().clamp_min(1.0e-6)
        relative_l2 = diff.norm() / expected_l2
        cosine = torch.nn.functional.cosine_similarity(
            got.reshape(1, -1), expected.reshape(1, -1)
        )[0]
        relative_peak = diff.abs().max() / expected.abs().max().clamp_min(1.0e-6)
        print(
            "[FlashInferSm12xFp8 fidelity] "
            f"tokens={tokens} compare={label} "
            f"relative_l2={relative_l2.item():.8g} "
            f"cosine={cosine.item():.8g} "
            f"relative_peak={relative_peak.item():.8g}"
        )
        self.assertLessEqual(
            relative_l2.item(),
            max_relative_l2,
            "relative L2 error exceeded the contract",
        )
        self.assertGreaterEqual(
            cosine.item(), min_cosine, "cosine similarity fell below the contract"
        )
        self.assertLessEqual(
            relative_peak.item(),
            max_relative_peak,
            "peak error normalized by reference amplitude exceeded the contract",
        )

    def _reference(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        route_weights: torch.Tensor,
        w1: torch.Tensor,
        s1: torch.Tensor,
        w2: torch.Tensor,
        s2: torch.Tensor,
    ) -> torch.Tensor:
        """FP32 route-order reference, with FP8 Q0/Q1 and BF16 accumulation."""

        q0 = self._quant_dequant_per_token_128(hidden)
        w1_f = torch.stack(
            [self._dequant_blockwise(w1[e], s1[e]) for e in range(w1.shape[0])]
        )
        w2_f = torch.stack(
            [self._dequant_blockwise(w2[e], s2[e]) for e in range(w2.shape[0])]
        )
        num_tokens, top_k = ids.shape
        pair_tokens = torch.arange(num_tokens, device=ids.device).repeat_interleave(
            top_k
        )
        pair_ids = ids.reshape(-1)
        q1_pairs = torch.empty(
            (pair_ids.numel(), w2.shape[-1]), device=hidden.device, dtype=torch.float32
        )
        for expert in range(w1.shape[0]):
            pair_mask = pair_ids == expert
            fc1 = q0[pair_tokens[pair_mask]].matmul(w1_f[expert].transpose(0, 1))
            # Fixed 14a9811 FC1 defaults to WG_S2R_QUANT: it converts the
            # FP32 activated accumulator to BF16 before deriving FP8 Q1.
            up, gate = fc1.chunk(2, dim=-1)
            activated_bf16 = (up * torch.nn.functional.silu(gate)).to(torch.bfloat16)
            q1_pairs[pair_mask] = self._quant_dequant_per_token_128(
                activated_bf16.float()
            )

        fc2_pairs = torch.empty(
            (pair_ids.numel(), hidden.shape[-1]),
            device=hidden.device,
            dtype=torch.bfloat16,
        )
        for expert in range(w2.shape[0]):
            pair_mask = pair_ids == expert
            # FC2's scatter path stores FP32 GEMM accumulators to BF16 shared
            # memory before multiplying each FP32 routing weight.
            fc2_acc_bf16 = (
                q1_pairs[pair_mask]
                .matmul(w2_f[expert].transpose(0, 1))
                .to(torch.bfloat16)
            )
            fc2_pairs[pair_mask] = fc2_acc_bf16

        out = torch.zeros_like(hidden)
        for slot in range(top_k):
            contribution = (
                fc2_pairs.view(num_tokens, top_k, -1)[:, slot].float()
                * route_weights[:, slot, None]
            ).to(torch.bfloat16)
            # `red.global.add.bf16x2` rounds the scaled contribution to BF16,
            # then performs a BF16 atomic reduction.  Slot order is only a
            # deterministic reference order; the device reduction is unordered.
            out = (out.float() + contribution.float()).to(torch.bfloat16)
        return out

    def test_qwen_i256_int64_routes_match_tolerant_reference_and_repeat(self):
        with mock.patch.dict(
            os.environ, {"MOE_TP_PREFILL_BACKEND": "flashinfer_sm12x"}
        ):
            if not FlashInferSm12xFp8Moe.is_supported():
                self.skipTest("requires SM120/121")
        import importlib.metadata

        from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
            per_block_cast_to_fp8,
        )
        from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

        try:
            import cutlass
        except ModuleNotFoundError:
            setup_cutlass_import_path()
            import cutlass
        import flashinfer

        print(
            "[FlashInferSm12xFp8 dependencies] "
            f"flashinfer={flashinfer.__version__} path={flashinfer.__file__} "
            f"cutlass_dsl_distribution={importlib.metadata.version('nvidia-cutlass-dsl')} "
            f"path={cutlass.__file__}"
        )
        torch.manual_seed(7)
        # Qwen3.5 TP2 local routed geometry: H=2048, I_local=256, k=8.
        # E=8 keeps the GPU contract compact while exercising all 8 routes.
        e, h, i, k = 8, 2048, 256, 8
        w1_b = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16) / 8
        w2_b = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) / 8
        w1, s1 = zip(
            *(per_block_cast_to_fp8(x, use_ue8m0=False) for x in w1_b), strict=True
        )
        w2, s2 = zip(
            *(per_block_cast_to_fp8(x, use_ue8m0=False) for x in w2_b), strict=True
        )
        adapter = FlashInferSm12xFp8Moe(
            torch.stack(w1), torch.stack(s1), torch.stack(w2), torch.stack(s2)
        )
        for t in (5, 129):
            hidden = torch.randn(t, h, device="cuda", dtype=torch.bfloat16) / 8
            ids = (
                torch.arange(t * k, device="cuda", dtype=torch.int64).reshape(t, k) % e
            )
            weights = torch.full((t, k), 1.0 / k, device="cuda", dtype=torch.float32)
            out_a = torch.empty_like(hidden)
            out_b = torch.empty_like(hidden)
            adapter.forward(hidden, ids, weights, out_a)
            adapter.forward(hidden, ids, weights, out_b)
            self.assertFalse(torch.isnan(out_a).any())
            reference = self._reference(
                hidden,
                ids,
                weights,
                torch.stack(w1),
                torch.stack(s1),
                torch.stack(w2),
                torch.stack(s2),
            )
            # E4M3 Q0/Q1 each introduce <= roughly 1/32 local relative roundoff;
            # BF16 stores/atomics and independent F32 tile reduction add error.
            # These scale-free thresholds are fixed from those operations, not
            # from a measured output: bulk L2 <= 8%, cosine >= .995, peak <=20%.
            self._assert_fidelity(
                out_a,
                reference,
                tokens=t,
                label="reference",
                max_relative_l2=0.08,
                min_cosine=0.995,
                max_relative_peak=0.20,
            )
            # Atomic arrivals are unordered, but variation must be small relative
            # to the output scale rather than a fixed absolute allowance.
            self._assert_fidelity(
                out_a,
                out_b,
                tokens=t,
                label="repeat",
                max_relative_l2=0.01,
                min_cosine=0.9999,
                max_relative_peak=0.03,
            )


if __name__ == "__main__":
    unittest.main()
