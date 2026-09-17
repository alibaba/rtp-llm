"""Correctness UT for fused Q/K RMSNorm vs two flashinfer rmsnorm launches.

Mirrors ``FusedQKRMSNorm.forward`` (reshape to [M, Q+K+V, D], inplace
rmsnorm on Q then K). V must stay bit-identical.
"""

from __future__ import annotations

import os
import unittest

import torch

try:
    import flashinfer
except ImportError:  # pragma: no cover
    flashinfer = None


def _load_mod():
    from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fused_qk_rmsnorm import (
        fused_qk_rmsnorm,
        is_supported,
        maybe_fused_qk_rmsnorm,
    )

    return fused_qk_rmsnorm, is_supported, maybe_fused_qk_rmsnorm


def _last_dim(head_num: int, kv_head_num: int, size_per_head: int) -> int:
    return (head_num + kv_head_num * 2) * size_per_head


def _torch_qk_rmsnorm(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    eps: float,
) -> torch.Tensor:
    """Same last-dim RMS + weight as flashinfer / DSV4 ``ref_rmsnorm_rope``."""
    m, n = hidden.shape
    qkv = hidden.reshape(m, head_num + kv_head_num * 2, size_per_head)
    q = qkv[:, :head_num, :].float()
    k = qkv[:, head_num : head_num + kv_head_num, :].float()
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps) * q_weight.float()
    k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + eps) * k_weight.float()
    qkv[:, :head_num, :] = q.to(hidden.dtype)
    qkv[:, head_num : head_num + kv_head_num, :] = k.to(hidden.dtype)
    return qkv.reshape(m, n)


def _flashinfer_qk_rmsnorm(
    hidden: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    eps: float,
) -> torch.Tensor:
    if flashinfer is None:
        return _torch_qk_rmsnorm(
            hidden, q_weight, k_weight, head_num, kv_head_num, size_per_head, eps
        )
    m, n = hidden.shape
    qkv = hidden.reshape(m, head_num + kv_head_num * 2, size_per_head)
    q = qkv[:, :head_num, :]
    k = qkv[:, head_num : head_num + kv_head_num, :]
    flashinfer.norm.rmsnorm(q, q_weight, eps=eps, out=q)
    flashinfer.norm.rmsnorm(k, k_weight, eps=eps, out=k)
    return qkv.reshape(m, n)


def _make_inputs(
    m: int,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    n = _last_dim(head_num, kv_head_num, size_per_head)
    hidden = torch.randn(m, n, device="cuda", dtype=dtype)
    q_weight = torch.randn(size_per_head, device="cuda", dtype=dtype).abs() + 0.25
    k_weight = torch.randn(size_per_head, device="cuda", dtype=dtype).abs() + 0.25
    return hidden, q_weight, k_weight


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedQKRMSNormTest(unittest.TestCase):
    # Tight BF16: same fp32 RMS + bf16 store as flashinfer; stay within ~1 ULP.
    ATOL = 1e-3
    RTOL = 1.6e-2

    @classmethod
    def setUpClass(cls):
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"
        fused, is_supported_fn, maybe = _load_mod()
        cls.fused = staticmethod(fused)
        cls.is_supported = staticmethod(is_supported_fn)
        cls.maybe = staticmethod(maybe)

    def setUp(self) -> None:
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"

    def _assert_close(self, got: torch.Tensor, ref: torch.Tensor) -> None:
        torch.testing.assert_close(got, ref, atol=self.ATOL, rtol=self.RTOL)
        max_abs = (got.float() - ref.float()).abs().max().item()
        self.assertLessEqual(max_abs, 2e-2, f"max abs {max_abs} exceeds bf16 1-ULP")

    def _run_pair(
        self,
        m: int,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        eps: float = 1e-6,
    ) -> None:
        hidden, q_w, k_w = _make_inputs(m, head_num, kv_head_num, size_per_head)
        ref_in = hidden.clone()
        got_in = hidden.clone()
        ref = _flashinfer_qk_rmsnorm(
            ref_in, q_w, k_w, head_num, kv_head_num, size_per_head, eps
        )
        self.assertTrue(self.is_supported(got_in, q_w, k_w, head_num, kv_head_num, size_per_head))
        got = self.fused(got_in, q_w, k_w, head_num, kv_head_num, size_per_head, eps)
        self.assertIs(got, got_in)
        self._assert_close(got, ref)

        q_end = head_num * size_per_head
        k_end = q_end + kv_head_num * size_per_head
        v = hidden[:, k_end:]
        torch.testing.assert_close(got[:, k_end:], v, atol=0.0, rtol=0.0)
        self._assert_close(got[:, :q_end], ref[:, :q_end])
        self._assert_close(got[:, q_end:k_end], ref[:, q_end:k_end])

    def test_397b_typical_m(self) -> None:
        for m in (1, 8, 32, 128, 256):
            with self.subTest(m=m, geom="32/2/256"):
                self._run_pair(m, 32, 2, 256)

    def test_small_geometry(self) -> None:
        for m in (1, 8, 32, 128, 256):
            with self.subTest(m=m, geom="8/2/128"):
                self._run_pair(m, 8, 2, 128)

    def test_m_equals_3(self) -> None:
        self._run_pair(3, 32, 2, 256)
        self._run_pair(3, 8, 2, 128)

    def test_m_equals_0(self) -> None:
        hidden, q_w, k_w = _make_inputs(0, 32, 2, 256)
        self.assertTrue(self.is_supported(hidden, q_w, k_w, 32, 2, 256))
        got = self.fused(hidden, q_w, k_w, 32, 2, 256, 1e-6)
        self.assertEqual(tuple(got.shape), (0, _last_dim(32, 2, 256)))
        self.assertEqual(got.dtype, torch.bfloat16)
        maybe = self.maybe(hidden, q_w, k_w, 32, 2, 256, 1e-6)
        self.assertIsNotNone(maybe)
        self.assertEqual(tuple(maybe.shape), (0, _last_dim(32, 2, 256)))

    def test_v_slice_unchanged(self) -> None:
        head_num, kv_head_num, d = 32, 2, 256
        hidden, q_w, k_w = _make_inputs(32, head_num, kv_head_num, d, seed=7)
        k_end = (head_num + kv_head_num) * d
        v_before = hidden[:, k_end:].clone()
        self.fused(hidden, q_w, k_w, head_num, kv_head_num, d, 1e-6)
        self.assertTrue(torch.equal(hidden[:, k_end:], v_before))

    def test_unsupported_returns_none(self) -> None:
        hidden, q_w, k_w = _make_inputs(8, 32, 2, 256)
        self.assertIsNone(
            self.maybe(hidden.float(), q_w.float(), k_w.float(), 32, 2, 256, 1e-6)
        )
        bad_hidden = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(self.maybe(bad_hidden, q_w, k_w, 32, 2, 256, 1e-6))
        self.assertFalse(self.is_supported(bad_hidden, q_w, k_w, 32, 2, 256))
        cpu = hidden.cpu()
        cpu_q, cpu_k = q_w.cpu(), k_w.cpu()
        self.assertIsNone(self.maybe(cpu, cpu_q, cpu_k, 32, 2, 256, 1e-6))

    def test_env_kill_switch(self) -> None:
        hidden, q_w, k_w = _make_inputs(4, 32, 2, 256)
        os.environ["RTP_QWEN35_DECODE_FUSION"] = "0"
        try:
            self.assertFalse(self.is_supported(hidden, q_w, k_w, 32, 2, 256))
            self.assertIsNone(self.maybe(hidden, q_w, k_w, 32, 2, 256, 1e-6))
        finally:
            os.environ["RTP_QWEN35_DECODE_FUSION"] = "1"


if __name__ == "__main__":
    unittest.main()
