"""Bit-for-bit equivalence test for ``fused_qk_rope_fp4_quant`` (Blackwell).

The fused QK FP4 kernel is the unified FP4 sibling of the FP8
``fused_qk_rope_quant``. It collapses the two-launch FP4 decode path
(``apply_rope_and_rotate_k`` + ``fused_q_rope_fp4_quant``) into a single
Triton kernel. Equivalence is pinned against the already-tested single-
purpose kernels so any drift in either path is caught:

  * Q output (FP4 e2m1 + UE8M0/32 scale) must match ``fused_q_rope_fp4_quant``
    on the same ``(q, positions)`` byte-for-byte — same Hadamard, same
    quantizer, same packing.
  * K output (bf16 RoPE + Hadamard) must match the FP8 QK kernel's
    ``k_out`` on the same ``(k, positions)`` byte-for-byte — both Triton
    kernels share the K-prelude verbatim.

Blackwell SM100+ only.
"""

from unittest import SkipTest, TestCase, main

import torch


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (10, 0)


class FusedQKRopeFp4QuantTest(TestCase):
    HD = 128
    ROPE_DIM = 64
    N_HEADS = 64
    MAX_POS = 4096

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if not _is_blackwell():
            raise SkipTest("FP4 fused QK kernel requires Blackwell SM100+")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(0)

        half = self.ROPE_DIM // 2
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, half, dtype=torch.float32, device=self.device) / half)
        )
        positions_full = torch.arange(
            self.MAX_POS, dtype=torch.float32, device=self.device
        )
        freqs = positions_full[:, None] * inv_freq[None, :]
        self.cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(
            torch.bfloat16
        )

    def _run_qk_fp4(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_fp4_quant import (
            fused_qk_rope_fp4_quant,
        )

        return fused_qk_rope_fp4_quant(
            q,
            k,
            positions,
            self.cos_sin_cache,
            self.N_HEADS,
            self.HD,
            self.ROPE_DIM,
            is_neox_style=True,
        )

    def _run_q_only_fp4(
        self, q: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_fp4_quant import (
            fused_q_rope_fp4_quant,
        )

        return fused_q_rope_fp4_quant(
            q,
            positions,
            self.cos_sin_cache,
            self.N_HEADS,
            self.HD,
            self.ROPE_DIM,
            is_neox_style=True,
        )

    def _run_qk_fp8(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        from rtp_llm.models_py.triton_kernels.sparse_mla.fused_q_rope_quant import (
            fused_qk_rope_quant,
        )

        _, _, k_out = fused_qk_rope_quant(
            q,
            k,
            positions,
            self.cos_sin_cache,
            self.N_HEADS,
            self.HD,
            self.ROPE_DIM,
            is_neox_style=True,
        )
        return k_out

    def test_q_output_matches_fused_q_only_fp4(self):
        num_tokens = 13
        q = torch.randn(
            num_tokens, self.N_HEADS, self.HD, dtype=torch.bfloat16, device=self.device
        )
        k = torch.randn(num_tokens, self.HD, dtype=torch.bfloat16, device=self.device)
        positions = torch.randint(
            0, self.MAX_POS, (num_tokens,), dtype=torch.int64, device=self.device
        )

        qk_q_fp4, qk_q_scale, _ = self._run_qk_fp4(q, k, positions)
        q_fp4, q_scale = self._run_q_only_fp4(q, positions)
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(qk_q_fp4.view(torch.uint8), q_fp4.view(torch.uint8)),
            "fused QK Q-nibbles differ from Q-only FP4 kernel",
        )
        self.assertTrue(
            torch.equal(qk_q_scale.view(torch.uint8), q_scale.view(torch.uint8)),
            "fused QK Q-scales differ from Q-only FP4 kernel",
        )

    def test_k_output_matches_fused_qk_fp8(self):
        num_tokens = 13
        q = torch.randn(
            num_tokens, self.N_HEADS, self.HD, dtype=torch.bfloat16, device=self.device
        )
        k = torch.randn(num_tokens, self.HD, dtype=torch.bfloat16, device=self.device)
        positions = torch.randint(
            0, self.MAX_POS, (num_tokens,), dtype=torch.int64, device=self.device
        )

        _, _, qk_k_out = self._run_qk_fp4(q, k, positions)
        fp8_k_out = self._run_qk_fp8(q, k, positions)
        torch.cuda.synchronize()

        # Both kernels share the K prelude verbatim — bf16 storage and the
        # same butterfly stages produce identical bytes.
        self.assertTrue(
            torch.equal(qk_k_out.view(torch.uint8), fp8_k_out.view(torch.uint8)),
            "fused QK FP4 K-output differs from FP8 QK K-output",
        )

    def test_empty_input_returns_empty(self):
        q = torch.empty(
            0, self.N_HEADS, self.HD, dtype=torch.bfloat16, device=self.device
        )
        k = torch.empty(0, self.HD, dtype=torch.bfloat16, device=self.device)
        positions = torch.empty(0, dtype=torch.int64, device=self.device)
        q_fp4, q_scale, k_out = self._run_qk_fp4(q, k, positions)
        self.assertEqual(q_fp4.shape, (0, self.N_HEADS, self.HD // 2))
        self.assertEqual(q_scale.shape, (0, self.N_HEADS))
        self.assertEqual(k_out.shape, (0, self.HD))


if __name__ == "__main__":
    main()
