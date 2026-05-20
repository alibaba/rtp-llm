"""Correctness UT for the vLLM-parity DSV4 Q/KV fused kernels.

Covers the two Triton entry points exposed by
``rtp_llm.models_py.modules.dsv4._fused_qkv_rmsnorm_rope_triton``:

  * :func:`fused_q_kv_rmsnorm` — RMSNorm both halves of the packed
    ``[wq_a | wkv]`` GEMM output. Parity vs torch RMSNorm on each
    slice.
  * :func:`fused_q_perhead_norm_qkv_rope` — per-head Q RMSNorm
    (no weight) + Q-RoPE + KV-RoPE. Parity vs torch RMSNorm +
    interleaved RoPE on the trailing ``rope_head_dim`` cols.

Shape sweep matches both the decode regime (small ``N_tok``,
batch concurrency) and the prefill regime (large ``N_tok``, full
sequence). Parametrized over both V4 variants so every
``GROUP_HEADS`` value baked into the dispatch table
(``_LAUNCH_CONFIGS``) is exercised:

  * V4-Flash: ``q_lora_rank=1024``, ``head_dim=512``, ``n_heads=64``
  * V4-Pro:   ``q_lora_rank=1536``, ``head_dim=512``, ``n_heads=128``

Both variants share ``rope_head_dim=64`` and ``head_dim=512``.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from rtp_llm.models_py.modules.dsv4._fused_qkv_rmsnorm_rope_triton import (
    _LAUNCH_CONFIGS,
    fused_q_kv_rmsnorm,
    fused_q_perhead_norm_qkv_rope,
)


@dataclass(frozen=True)
class V4Shape:
    name: str
    q_lora_rank: int
    head_dim: int
    n_heads: int
    rope_dim: int

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_dim

    @property
    def total_dim(self) -> int:
        return self.q_lora_rank + self.head_dim


V4_FLASH = V4Shape("v4_flash", q_lora_rank=1024, head_dim=512, n_heads=64, rope_dim=64)
V4_PRO = V4Shape("v4_pro", q_lora_rank=1536, head_dim=512, n_heads=128, rope_dim=64)
VARIANTS = (V4_FLASH, V4_PRO)

EPS = 1e-6

# Decode shape (per-request batched) + prefill shape (sequence tokens).
DECODE_NS = [1, 2, 4, 8, 16, 32]
PREFILL_NS = [128, 512, 4096, 16384]


def _rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Reference RMSNorm in fp32 (matches the kernel's accumulator dtype)."""
    x_f32 = x.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    y = x_f32 * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.to(torch.float32)
    return y.to(x.dtype)


def _rotate_pairs(
    real: torch.Tensor, imag: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to (real, imag) pairs given complex64 freqs.

    ``freqs_cis`` is ``[N_tok, half]``; broadcast over any extra leading
    dims (e.g. Q's head axis) by unsqueezing on the right.
    """
    cs = torch.view_as_real(freqs_cis).to(torch.float32)
    cos = cs[..., 0]
    sin = cs[..., 1]
    while cos.dim() < real.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    new_real = real * cos - imag * sin
    new_imag = real * sin + imag * cos
    return new_real, new_imag


def _apply_partial_rope_bf16_ref(
    x: torch.Tensor, freqs_cis: torch.Tensor, rope_head_dim: int
) -> torch.Tensor:
    """Reference partial RoPE on a bf16 tensor (KV path — input already RMSNormed).

    Matches the kernel's KV branch: bf16 in mem → cast to fp32 →
    rotate → cast back to bf16 on store. NOPE region passes through
    untouched.
    """
    out = x.clone()
    last_dim = x.shape[-1]
    nope = last_dim - rope_head_dim
    half = rope_head_dim // 2
    rope_block = x[..., nope:].reshape(*x.shape[:-1], half, 2).to(torch.float32)
    new_real, new_imag = _rotate_pairs(
        rope_block[..., 0], rope_block[..., 1], freqs_cis
    )
    rotated = torch.stack([new_real, new_imag], dim=-1).reshape(
        *x.shape[:-1], rope_head_dim
    )
    out[..., nope:] = rotated.to(x.dtype)
    return out


def _apply_q_rmsnorm_rope_ref(
    x_bf16: torch.Tensor, freqs_cis: torch.Tensor, rope_head_dim: int, eps: float
) -> torch.Tensor:
    """Reference for the Q branch of ``fused_q_perhead_norm_qkv_rope``.

    Critical: the kernel keeps the post-RMSNorm value in fp32 all the
    way through RoPE — it does NOT round-trip to bf16 between rmsnorm
    and rope. The naive ``rmsnorm → bf16 → cast back → rope`` reference
    accumulates a ~1-ULP error that exceeds the 4e-3 tolerance for
    values around 1-2. Match the kernel's "fp32 throughout" semantics
    by deferring the bf16 cast until the final store.
    """
    last_dim = x_bf16.shape[-1]
    nope = last_dim - rope_head_dim
    half = rope_head_dim // 2

    x_f32 = x_bf16.to(torch.float32)
    var = x_f32.pow(2).mean(dim=-1, keepdim=True)
    y_f32 = x_f32 * torch.rsqrt(var + eps)  # per-head RMSNorm, no weight

    # NOPE region: bf16 cast on store (matches `tl.store(out, y)` rounding).
    out_nope = y_f32[..., :nope].to(torch.bfloat16)

    # ROPE region: stay in fp32 through the rotation, only cast at the end.
    rope_f32 = y_f32[..., nope:].reshape(*y_f32.shape[:-1], half, 2)
    new_real, new_imag = _rotate_pairs(
        rope_f32[..., 0], rope_f32[..., 1], freqs_cis
    )
    rotated = torch.stack([new_real, new_imag], dim=-1).reshape(
        *y_f32.shape[:-1], rope_head_dim
    )
    out_rope = rotated.to(torch.bfloat16)

    return torch.cat([out_nope, out_rope], dim=-1)


def _make_freqs_cis(n_tok: int, rope_dim: int, device: torch.device) -> torch.Tensor:
    """Random complex64 freqs_cis [n_tok, rope_dim // 2] with |z| = 1."""
    torch.manual_seed(7 + n_tok + rope_dim)
    angles = (torch.rand(n_tok, rope_dim // 2, device=device) - 0.5) * 2 * 3.14159
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return torch.complex(cos, sin).contiguous()


def _make_qkv_a(n_tok: int, total_dim: int, device: torch.device) -> torch.Tensor:
    """Random bf16 [n_tok, total_dim] packed GEMM output (q_lora | kv)."""
    torch.manual_seed(11 + n_tok + total_dim)
    return (torch.randn(n_tok, total_dim, device=device, dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    )


def _make_q(n_tok: int, n_heads: int, head_dim: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(13 + n_tok + n_heads)
    return (
        torch.randn(n_tok, n_heads, head_dim, device=device, dtype=torch.float32) * 0.3
    ).to(torch.bfloat16)


def _make_kv(n_tok: int, head_dim: int, device: torch.device) -> torch.Tensor:
    torch.manual_seed(17 + n_tok + head_dim)
    return (
        torch.randn(n_tok, head_dim, device=device, dtype=torch.float32) * 0.3
    ).to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedQKVRmsnormTest(unittest.TestCase):
    def _check_q_kv_rmsnorm(self, shape: V4Shape, n_tok: int) -> None:
        device = torch.device("cuda")
        qkv_a = _make_qkv_a(n_tok, shape.total_dim, device)
        q_norm = torch.rand(shape.q_lora_rank, dtype=torch.bfloat16, device=device) + 0.5
        kv_norm = torch.rand(shape.head_dim, dtype=torch.bfloat16, device=device) + 0.5

        qr_out, kv_out = fused_q_kv_rmsnorm(
            qkv_a,
            q_norm,
            kv_norm,
            q_size=shape.q_lora_rank,
            kv_offset=shape.q_lora_rank,
            eps=EPS,
        )

        # Reference computed directly on the python slices.
        qr_ref = _rmsnorm_ref(qkv_a[:, : shape.q_lora_rank].contiguous(), q_norm, EPS)
        kv_ref = _rmsnorm_ref(qkv_a[:, shape.q_lora_rank :].contiguous(), kv_norm, EPS)

        self.assertEqual(qr_out.shape, (n_tok, shape.q_lora_rank))
        self.assertEqual(kv_out.shape, (n_tok, shape.head_dim))
        torch.testing.assert_close(qr_out, qr_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(kv_out, kv_ref, atol=2e-2, rtol=2e-2)

    def test_q_kv_rmsnorm_decode(self):
        for shape in VARIANTS:
            for n in DECODE_NS:
                with self.subTest(variant=shape.name, n_tok=n):
                    self._check_q_kv_rmsnorm(shape, n)

    def test_q_kv_rmsnorm_prefill(self):
        for shape in VARIANTS:
            for n in PREFILL_NS:
                with self.subTest(variant=shape.name, n_tok=n):
                    self._check_q_kv_rmsnorm(shape, n)

    def test_q_kv_rmsnorm_leading_shape_preserved(self):
        device = torch.device("cuda")
        # Caller may pass [B, S, total_dim] — the wrapper must thread the
        # leading shape through to the outputs.  Pick V4-Flash for this
        # check; the wrapper path is identical for V4-Pro.
        shape = V4_FLASH
        B, S = 3, 5
        qkv_a = _make_qkv_a(B * S, shape.total_dim, device).view(B, S, shape.total_dim)
        q_norm = torch.rand(shape.q_lora_rank, dtype=torch.bfloat16, device=device) + 0.5
        kv_norm = torch.rand(shape.head_dim, dtype=torch.bfloat16, device=device) + 0.5
        qr, kv = fused_q_kv_rmsnorm(
            qkv_a,
            q_norm,
            kv_norm,
            q_size=shape.q_lora_rank,
            kv_offset=shape.q_lora_rank,
            eps=EPS,
        )
        self.assertEqual(qr.shape, (B, S, shape.q_lora_rank))
        self.assertEqual(kv.shape, (B, S, shape.head_dim))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedQPerheadNormQKVRopeTest(unittest.TestCase):
    def _check_q_perhead_norm_qkv_rope(
        self,
        shape: V4Shape,
        n_tok: int,
        launch_override=None,
    ) -> None:
        device = torch.device("cuda")
        q = _make_q(n_tok, shape.n_heads, shape.head_dim, device)
        kv = _make_kv(n_tok, shape.head_dim, device)  # already RMSNormed (step-2 output)
        freqs = _make_freqs_cis(n_tok, shape.rope_dim, device)

        q_in = q.clone()
        q_out, kv_out = fused_q_perhead_norm_qkv_rope(
            q,
            kv,
            freqs,
            shape.rope_dim,
            eps=EPS,
            _launch_override=launch_override,
        )
        # Sanity: q is the same tensor (in-place).
        self.assertTrue(q_out.data_ptr() == q.data_ptr())

        # Q reference matches the kernel's "fp32 throughout" semantics
        # (see _apply_q_rmsnorm_rope_ref); KV path is bf16 in mem →
        # fp32 rotate → bf16 store with the NOPE region copied through.
        q_ref = _apply_q_rmsnorm_rope_ref(q_in, freqs, shape.rope_dim, EPS)
        kv_ref = _apply_partial_rope_bf16_ref(kv, freqs, shape.rope_dim)

        torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(kv_out, kv_ref, atol=2e-2, rtol=2e-2)

    def test_decode(self):
        for shape in VARIANTS:
            for n in DECODE_NS:
                with self.subTest(variant=shape.name, n_tok=n):
                    self._check_q_perhead_norm_qkv_rope(shape, n)

    def test_prefill(self):
        for shape in VARIANTS:
            for n in PREFILL_NS:
                with self.subTest(variant=shape.name, n_tok=n):
                    self._check_q_perhead_norm_qkv_rope(shape, n)

    def test_group_heads_values(self):
        """Every GROUP_HEADS baked into _LAUNCH_CONFIGS must produce
        bit-equivalent output (within the standard 2e-2 tolerance) to
        the G=1 path — the dispatch table can't introduce a numerical
        regression."""
        group_values = {
            cfg[0]
            for key, cfg in _LAUNCH_CONFIGS.items()
            if isinstance(key[0], int)  # skip wildcard rows
        }
        # Use a mid n_tok that fills the SM grid for any G.
        n_tok = 64
        for shape in VARIANTS:
            for g in sorted(group_values):
                with self.subTest(variant=shape.name, group_heads=g):
                    self._check_q_perhead_norm_qkv_rope(
                        shape, n_tok, launch_override=(g, 4, 3)
                    )

    def test_strided_kv_input(self):
        """KV may be a strided view into a wider packed tensor — the
        kernel must use ``kv.stride(0)`` rather than assuming D-contiguous."""
        device = torch.device("cuda")
        shape = V4_FLASH
        n_tok = 64
        q = _make_q(n_tok, shape.n_heads, shape.head_dim, device)
        # Build a wider tensor [n_tok, total_dim] and slice out the kv half
        # so kv.stride(0) == total_dim but kv.stride(-1) == 1.
        packed = _make_qkv_a(n_tok, shape.total_dim, device)
        kv = packed[:, shape.q_lora_rank : shape.q_lora_rank + shape.head_dim]
        self.assertEqual(kv.stride(0), shape.total_dim)
        self.assertEqual(kv.stride(-1), 1)
        freqs = _make_freqs_cis(n_tok, shape.rope_dim, device)

        q_in = q.clone()
        kv_in = kv.clone().contiguous()  # for the reference computation
        q_out, kv_out = fused_q_perhead_norm_qkv_rope(
            q, kv, freqs, shape.rope_dim, eps=EPS
        )
        q_ref = _apply_q_rmsnorm_rope_ref(q_in, freqs, shape.rope_dim, EPS)
        kv_ref = _apply_partial_rope_bf16_ref(kv_in, freqs, shape.rope_dim)
        torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(kv_out, kv_ref, atol=2e-2, rtol=2e-2)

    def test_leading_shape_preserved(self):
        device = torch.device("cuda")
        shape = V4_FLASH
        B, S = 2, 4
        q = (
            _make_q(B * S, shape.n_heads, shape.head_dim, device)
            .view(B, S, shape.n_heads, shape.head_dim)
            .contiguous()
        )
        kv = (
            _make_kv(B * S, shape.head_dim, device)
            .view(B, S, shape.head_dim)
            .contiguous()
        )
        freqs = (
            _make_freqs_cis(B * S, shape.rope_dim, device)
            .view(B, S, shape.rope_dim // 2)
            .contiguous()
        )
        q_out, kv_out = fused_q_perhead_norm_qkv_rope(
            q, kv, freqs, shape.rope_dim, eps=EPS
        )
        self.assertEqual(q_out.shape, (B, S, shape.n_heads, shape.head_dim))
        self.assertEqual(kv_out.shape, (B, S, shape.head_dim))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedQKVEndToEndParityTest(unittest.TestCase):
    """End-to-end parity: the two-kernel pipeline matches a single torch
    reference that does RMSNorm → (split for q-side wq_b skipped, that's a
    user-supplied tensor) → per-head Q RMSNorm + RoPE / KV RoPE.

    This protects against drift between the kernel impl and the comment
    in attention.py describing the vLLM 4-launch flow.
    """

    def test_pipeline(self):
        device = torch.device("cuda")
        shape = V4_FLASH
        n_tok = 17
        qkv_a = _make_qkv_a(n_tok, shape.total_dim, device)
        q_norm = torch.rand(shape.q_lora_rank, dtype=torch.bfloat16, device=device) + 0.5
        kv_norm = torch.rand(shape.head_dim, dtype=torch.bfloat16, device=device) + 0.5
        freqs = _make_freqs_cis(n_tok, shape.rope_dim, device)
        # Stand-in for wq_b output (the kernels treat q as opaque bf16
        # input from the GEMM; only its shape matters here).
        q_after_wqb = _make_q(n_tok, shape.n_heads, shape.head_dim, device)

        # Pipeline (vLLM-parity 4-launch, minus the GEMMs):
        qr, kv = fused_q_kv_rmsnorm(
            qkv_a,
            q_norm,
            kv_norm,
            q_size=shape.q_lora_rank,
            kv_offset=shape.q_lora_rank,
            eps=EPS,
        )
        q_in = q_after_wqb.clone()
        q_pipe, kv_pipe = fused_q_perhead_norm_qkv_rope(
            q_after_wqb, kv, freqs, shape.rope_dim, eps=EPS
        )

        # Reference: same math, plain torch.
        qr_ref = _rmsnorm_ref(qkv_a[:, : shape.q_lora_rank].contiguous(), q_norm, EPS)
        kv_normed_ref = _rmsnorm_ref(
            qkv_a[:, shape.q_lora_rank :].contiguous(), kv_norm, EPS
        )
        q_ref = _apply_q_rmsnorm_rope_ref(q_in, freqs, shape.rope_dim, EPS)
        kv_ref = _apply_partial_rope_bf16_ref(kv_normed_ref, freqs, shape.rope_dim)

        torch.testing.assert_close(qr, qr_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(q_pipe, q_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(kv_pipe, kv_ref, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
