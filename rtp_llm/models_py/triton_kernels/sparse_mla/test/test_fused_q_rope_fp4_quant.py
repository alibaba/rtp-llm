"""Bit-for-bit equivalence test for ``fused_q_rope_fp4_quant`` (Blackwell).

Compares the fused Triton kernel against a step-by-step reference: torch
performs RoPE and the Hadamard transform, then ``deep_gemm.utils
.per_token_cast_to_fp4`` produces the final FP4 / UE8M0 bytes — mirroring
the unfused FP4 indexer path that the kernel is meant to replace. The
authoritative quantizer is deep_gemm, not torch.
"""

from unittest import SkipTest, TestCase, main

import torch


def _is_blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (10, 0)


def _walsh_hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    """Unnormalized Walsh-Hadamard matrix (Sylvester construction)."""
    assert n & (n - 1) == 0
    h = torch.tensor([[1.0]], device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def _apply_rope_neox(
    q: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    """NeOX-style RoPE on first ``rope_dim`` dims of last axis, in fp32 with a
    bf16 roundtrip after RoPE to match the kernel."""
    half = rope_dim // 2
    out = q.clone().to(torch.float32)
    cos = cos_sin[positions, :half].to(torch.float32)  # [T, half]
    sin = cos_sin[positions, half:].to(torch.float32)  # [T, half]
    # broadcast over heads
    cos = cos[:, None, :]
    sin = sin[:, None, :]
    x_first = out[..., :half]
    x_second = out[..., half:rope_dim]
    r_first = x_first * cos - x_second * sin
    r_second = x_second * cos + x_first * sin
    r_first = r_first.to(torch.bfloat16).to(torch.float32)
    r_second = r_second.to(torch.bfloat16).to(torch.float32)
    out[..., :half] = r_first
    out[..., half:rope_dim] = r_second
    return out


def _reference_rope_hadamard_deep_gemm_fp4(
    q_bf16: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference path: torch RoPE → torch Hadamard → deep_gemm.per_token_cast_to_fp4.

    The FP4 / UE8M0 byte layout is whatever deep_gemm produces; torch only
    handles the bf16-precision pre-processing. Same ``(q_fp4 int8, q_scale
    int32)`` shape contract as the kernel.
    """
    from deep_gemm.utils import per_token_cast_to_fp4

    T, H, D = q_bf16.shape
    rotated = _apply_rope_neox(q_bf16, positions, cos_sin_cache, rope_head_dim)
    # Walsh-Hadamard with normalization 1/sqrt(D) (matches kernel's data * D**-0.5).
    H_mat = _walsh_hadamard_matrix(D, q_bf16.device, torch.float32)
    flat = rotated.view(-1, D)
    transformed = (flat @ H_mat.t()) * (D**-0.5)

    q_fp4_packed, q_scale = per_token_cast_to_fp4(
        transformed.to(torch.bfloat16),
        use_ue8m0=True,
        gran_k=32,
        use_packed_ue8m0=True,
    )
    return (
        q_fp4_packed.view(T, H, D // 2),
        q_scale.view(T, H),
    )


class FusedQRopeFp4QuantTest(TestCase):
    HD = 128
    ROPE_DIM = 64
    N_HEADS = 64
    MAX_POS = 4096

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if not _is_blackwell():
            raise SkipTest("FP4 fused Q kernel requires Blackwell SM100+")
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        torch.manual_seed(0)

        # cos_sin layout matches what the FP8 fused kernel consumes:
        # first ``rope_dim/2`` cols cos, next ``rope_dim/2`` cols sin.
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
        )  # [MAX_POS, rope_dim]

    def _run_kernel(
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

    def test_matches_rope_hadamard_deep_gemm_reference(self):
        num_tokens = 13
        q = torch.randn(
            num_tokens, self.N_HEADS, self.HD, dtype=torch.bfloat16, device=self.device
        )
        positions = torch.randint(
            0, self.MAX_POS, (num_tokens,), dtype=torch.int64, device=self.device
        )

        actual_fp4, actual_scale = self._run_kernel(q, positions)
        torch.cuda.synchronize()

        ref_fp4, ref_scale = _reference_rope_hadamard_deep_gemm_fp4(
            q, positions, self.cos_sin_cache, self.ROPE_DIM
        )

        # FP4 packed nibbles must match bit-for-bit.
        self.assertTrue(
            torch.equal(actual_fp4.view(torch.uint8), ref_fp4.view(torch.uint8)),
            "fused FP4 nibbles differ from deep_gemm reference",
        )
        # UE8M0 packed scales must match bit-for-bit.
        self.assertTrue(
            torch.equal(actual_scale.view(torch.uint8), ref_scale.view(torch.uint8)),
            "fused UE8M0 scales differ from deep_gemm reference",
        )

    def test_empty_input_returns_empty(self):
        q = torch.empty(
            0, self.N_HEADS, self.HD, dtype=torch.bfloat16, device=self.device
        )
        positions = torch.empty(0, dtype=torch.int64, device=self.device)
        q_fp4, q_scale = self._run_kernel(q, positions)
        self.assertEqual(q_fp4.shape, (0, self.N_HEADS, self.HD // 2))
        self.assertEqual(q_scale.shape, (0, self.N_HEADS))


if __name__ == "__main__":
    main()
