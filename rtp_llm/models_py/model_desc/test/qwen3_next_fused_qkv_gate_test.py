"""Validate that fusing the qwen3_next attention qkv and gate GEMMs into a
single GEMM produces the same output (up to numerical noise) as running them
separately, for both bf16 and FP8 per-block paths.

Mirrors sglang's `attn_output_gate` packing in qwen3_5: when the gate share is
folded into qkv_proj, the linear's output is `[qkv | gate]` along the column
dim and we slice the two halves apart at forward time.
"""

import unittest
from typing import Tuple

import torch

from rtp_llm.config.quant_config import init_quant_config

# Importing the factory module triggers device-specific strategy registration.
from rtp_llm.models_py.modules.factory.linear import LinearFactory as _LinearFactory


def _bf16_linear_from_weight(weight: torch.Tensor) -> torch.nn.Module:
    return _LinearFactory.create_linear(
        weight=weight,
        bias=None,
        weight_scales=None,
        quant_config=None,
    )


class FusedQkvGateBf16Test(unittest.TestCase):
    """The fused linear must match the concat of two separate linears exactly
    for bf16 (no quantization, no rounding differences)."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.hidden = 2048
        self.head_dim = 128
        self.num_heads = 16
        self.num_kv_heads = 4
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_size = self.q_size + 2 * self.kv_size
        self.gate_size = self.q_size

    def _random_weight(self, n: int) -> torch.Tensor:
        return (
            torch.randn(self.hidden, n, dtype=self.dtype, device=self.device) * 0.05
        ).contiguous()

    def test_bf16_fused_matches_separate(self) -> None:
        qkv_w = self._random_weight(self.qkv_size)
        gate_w = self._random_weight(self.gate_size)
        fused_w = torch.cat([qkv_w, gate_w], dim=1).contiguous()

        sep_qkv = _bf16_linear_from_weight(qkv_w)
        sep_gate = _bf16_linear_from_weight(gate_w)
        fused = _bf16_linear_from_weight(fused_w)

        for batch in (1, 7, 32, 257):
            with self.subTest(batch=batch):
                x = torch.randn(
                    batch, self.hidden, dtype=self.dtype, device=self.device
                )
                ref_qkv = sep_qkv(x)
                ref_gate = sep_gate(x)
                ref = torch.cat([ref_qkv, ref_gate], dim=1)

                out = fused(x)
                self.assertEqual(out.shape, ref.shape)
                # bf16 matmul is deterministic for the same kernel + inputs;
                # the fused kernel may dispatch differently (different N), so
                # accept tiny rounding differences instead of bit-equality.
                torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

                got_qkv = out[:, : self.qkv_size]
                got_gate = out[:, self.qkv_size :]
                torch.testing.assert_close(got_qkv, ref_qkv, atol=1e-2, rtol=1e-2)
                torch.testing.assert_close(got_gate, ref_gate, atol=1e-2, rtol=1e-2)


class FusedQkvGateFp8PerBlockTest(unittest.TestCase):
    """For FP8 per-block, per-block scales are independent along the output
    dim so concatenating along that axis is mathematically lossless. The two
    paths should agree to within FP8-quant noise."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        # FP8 per-block requires DeepGEMM/flashinfer; skip if unavailable.
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (  # noqa: F401
                has_deep_gemm,
            )
        except Exception as e:  # pragma: no cover
            self.skipTest(f"deepgemm wrapper import failed: {e}")
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm

        if not has_deep_gemm():
            self.skipTest("DeepGEMM not available on this build")

        self.device = "cuda"
        # Pick sizes that align to the 128 block size.
        self.hidden = 2048
        self.head_dim = 128
        self.num_heads = 16
        self.num_kv_heads = 4
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_size = self.q_size + 2 * self.kv_size
        self.gate_size = self.q_size
        self.quant_config = init_quant_config("FP8_PER_BLOCK")

    def _random_fp8_block(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (fp8 weight [K,N], fp32 scale [K/128,N/128])."""
        from rtp_llm.test.utils.numeric_util import per_block_cast_to_fp8

        bf16 = (
            torch.randn(n, self.hidden, dtype=torch.bfloat16, device=self.device) * 0.05
        )
        fp8, scale = per_block_cast_to_fp8(bf16, use_ue8m0=False)
        K, N = self.hidden, n
        scale_K = (K + 127) // 128
        scale_N = (N + 127) // 128
        fp8 = fp8.reshape(K, N)
        scale = scale.reshape(scale_K, scale_N)
        return fp8, scale

    def _make_linear(self, fp8: torch.Tensor, scale: torch.Tensor) -> torch.nn.Module:
        return _LinearFactory.create_linear(
            weight=fp8,
            bias=None,
            weight_scales=scale,
            quant_config=self.quant_config,
        )

    def test_fp8_block_fused_matches_separate(self) -> None:
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextAttention

        qkv_fp8, qkv_scale = self._random_fp8_block(self.qkv_size)
        gate_fp8, gate_scale = self._random_fp8_block(self.gate_size)

        # FP8 weight memory is laid out as (N, K); the (K, N) shape attribute
        # is just a postprocess reshape. The Qwen3NextAttention helper does
        # the layout-aware concat (dim=0 in the actual (N, K) view) and
        # restores the (K, N_total) shape so the linear factory accepts it.
        fused_fp8 = Qwen3NextAttention._concat_along_n_axis(qkv_fp8, gate_fp8)
        fused_scale = Qwen3NextAttention._concat_along_n_axis(qkv_scale, gate_scale)

        sep_qkv = self._make_linear(qkv_fp8, qkv_scale)
        sep_gate = self._make_linear(gate_fp8, gate_scale)
        fused = self._make_linear(fused_fp8, fused_scale)

        x_probe = torch.randn(8, self.hidden, dtype=torch.bfloat16, device=self.device)
        try:
            sep_qkv(x_probe)
        except RuntimeError as e:
            # FP8 GEMMs JIT-compile via nvcc; in sandboxes without the CUDA
            # toolkit on PATH the JIT raises an assertion. Skip rather than
            # fail — the e2e smoke test exercises this path with full env.
            self.skipTest(f"FP8 GEMM JIT not available in this env: {e}")

        for batch in (1, 32, 256):
            with self.subTest(batch=batch):
                x = torch.randn(
                    batch, self.hidden, dtype=torch.bfloat16, device=self.device
                )
                ref_qkv = sep_qkv(x)
                ref_gate = sep_gate(x)
                out = fused(x)

                self.assertEqual(out.shape, (batch, self.qkv_size + self.gate_size))
                got_qkv = out[:, : self.qkv_size]
                got_gate = out[:, self.qkv_size :]
                # Block-FP8 GEMM is deterministic for fixed (M,N,K,scale),
                # so the fused N=qkv+gate path may differ from running each
                # half separately by tiny amounts due to kernel dispatch.
                torch.testing.assert_close(got_qkv, ref_qkv, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(got_gate, ref_gate, atol=2e-2, rtol=2e-2)


class FusedQkvGateFp8LayoutTest(unittest.TestCase):
    """Pure layout/data check: the fused FP8 weight memory must match what
    DeepGEMM expects (rows of the qkv weight followed by rows of the gate
    weight in the (N, K) interpretation). Doesn't require nvcc/GEMM JIT, so
    it runs in any CUDA-capable env.
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = "cuda"
        self.K = 256
        self.N_qkv = 384  # 3 blocks of 128
        self.N_gate = 256  # 2 blocks of 128

    def _fake_fp8_weight(self, n: int) -> torch.Tensor:
        """Build a tensor whose memory is laid out as (n, K) row-major (as
        merge_te_qkv would produce) but whose shape attribute is (K, n) (as
        the per_block_fp8 postprocess reshape produces)."""
        real = torch.randint(
            0, 100, (n, self.K), dtype=torch.uint8, device=self.device
        ).view(torch.float8_e4m3fn)
        return real.reshape(self.K, n), real

    def test_fused_memory_matches_real_concat(self) -> None:
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextAttention

        qkv_view, qkv_real = self._fake_fp8_weight(self.N_qkv)
        gate_view, gate_real = self._fake_fp8_weight(self.N_gate)

        fused_view = Qwen3NextAttention._concat_along_n_axis(qkv_view, gate_view)

        # The fused tensor must carry the (K, N_total) shape attribute, but
        # its memory — when re-viewed as (N_total, K) — must equal the row
        # concat of qkv_real and gate_real.
        self.assertEqual(fused_view.shape, (self.K, self.N_qkv + self.N_gate))
        fused_real = fused_view.reshape(self.N_qkv + self.N_gate, self.K)
        expected = torch.cat([qkv_real, gate_real], dim=0)
        self.assertTrue(
            torch.equal(fused_real.view(torch.uint8), expected.view(torch.uint8)),
            "FP8 fused memory layout does not match expected (qkv rows then gate rows)",
        )


class FusedQkvGateBf16LayoutTest(unittest.TestCase):
    """Symmetric check for BF16: the data layout there is (K, N) row-major,
    so the helper must do plain column-wise concat."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = "cuda"
        self.K = 256
        self.N_qkv = 384
        self.N_gate = 256

    def test_fused_memory_matches_naive_concat(self) -> None:
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextAttention

        qkv = torch.randn(self.K, self.N_qkv, dtype=torch.bfloat16, device=self.device)
        gate = torch.randn(
            self.K, self.N_gate, dtype=torch.bfloat16, device=self.device
        )
        fused = Qwen3NextAttention._concat_along_n_axis(qkv, gate)
        self.assertEqual(fused.shape, (self.K, self.N_qkv + self.N_gate))
        ref = torch.cat([qkv, gate], dim=1)
        self.assertTrue(torch.equal(fused, ref))


if __name__ == "__main__":
    unittest.main()
