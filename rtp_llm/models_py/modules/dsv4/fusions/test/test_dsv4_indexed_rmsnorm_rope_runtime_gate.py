from __future__ import annotations

import os
import sys
import types
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.dsv4_indexed_rope import (
    dsv4_indexed_rmsnorm_rope,
    indexed_rmsnorm_rope_path,
    is_indexed_rmsnorm_rope_supported,
)

os.environ.setdefault("DSV4_INDEXED_ROPE_CUDA", "1")


def _freqs(max_pos: int, rd: int) -> torch.Tensor:
    pos = torch.arange(max_pos, dtype=torch.float32, device="cuda")[:, None]
    pair = torch.arange(rd // 2, dtype=torch.float32, device="cuda")[None, :]
    angle = (pos + 1) * (pair + 3) * 0.001
    return torch.polar(torch.ones_like(angle), angle).contiguous()


def _materialized_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs: torch.Tensor,
    position_ids: torch.Tensor,
    rd: int,
) -> torch.Tensor:
    selected = freqs.index_select(0, position_ids.to(dtype=torch.long)).contiguous()
    return _fake_fused_rmsnorm_rope(x, weight, selected, rd, eps=1e-6)


def _fake_fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    d = x.shape[-1]
    x_flat = x.reshape(-1, d).float()
    n_freq = int(freqs_cis.reshape(-1, freqs_cis.shape[-1]).shape[0])
    freq_stride_n = int(x_flat.shape[0] // n_freq)
    inv = torch.rsqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + eps)
    y = x_flat * inv
    if weight is not None:
        y = y * weight.float()[None, :]
    nope = d - rope_head_dim
    row_freq = torch.arange(x_flat.shape[0], device=x.device) // freq_stride_n
    selected = freqs_cis.reshape(-1, freqs_cis.shape[-1]).index_select(0, row_freq)
    tail = y[:, nope:].reshape(-1, rope_head_dim // 2, 2)
    real = tail[..., 0]
    imag = tail[..., 1]
    y_tail = torch.stack(
        (
            real * selected.real - imag * selected.imag,
            real * selected.imag + imag * selected.real,
        ),
        dim=-1,
    ).flatten(-2)
    return torch.cat((y[:, :nope], y_tail), dim=-1).reshape(x.shape).to(x.dtype)


_fake_triton_mod = types.ModuleType(
    "rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton"
)
_fake_triton_mod.fused_rmsnorm_rope = _fake_fused_rmsnorm_rope
sys.modules[_fake_triton_mod.__name__] = _fake_triton_mod


class _ScopedEnv:
    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value
        self.old = os.environ.get(key)

    def __enter__(self) -> None:
        os.environ[self.key] = self.value

    def __exit__(self, *args) -> None:
        if self.old is None:
            os.environ.pop(self.key, None)
        else:
            os.environ[self.key] = self.old


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSV4IndexedRmsnormRopeRuntimeGate(unittest.TestCase):
    def test_q_runtime_path_selects_by_fixed_dims_not_m(self) -> None:
        torch.manual_seed(699)
        h, rd = 64, 64
        freqs = _freqs(5000, rd)
        small_q = torch.randn(128, 1, h, 128, dtype=torch.bfloat16, device="cuda")
        large_q = torch.randn(3079, 1, h, 128, dtype=torch.bfloat16, device="cuda")
        q_d512 = torch.randn(8, 1, h, 512, dtype=torch.bfloat16, device="cuda")
        generic_q = torch.randn(8, 1, 32, 128, dtype=torch.bfloat16, device="cuda")
        small_pos = torch.randint(0, 5000, (128,), dtype=torch.int32, device="cuda")
        large_pos = torch.randint(0, 5000, (3079,), dtype=torch.int32, device="cuda")
        q_d512_pos = torch.randint(0, 5000, (8,), dtype=torch.int32, device="cuda")
        generic_pos = torch.randint(0, 5000, (8,), dtype=torch.int32, device="cuda")

        with _ScopedEnv("DSV4_INDEXED_RMSNORM_ROPE_Q_LARGE_M", "0"):
            self.assertEqual(indexed_rmsnorm_rope_path(small_q, None, rd), "indexed_small")
            self.assertTrue(is_indexed_rmsnorm_rope_supported(small_q, None, freqs, small_pos, rd))
            self.assertEqual(indexed_rmsnorm_rope_path(large_q, None, rd), "indexed_small")
            self.assertTrue(is_indexed_rmsnorm_rope_supported(large_q, None, freqs, large_pos, rd))
            self.assertEqual(indexed_rmsnorm_rope_path(q_d512, None, rd), "indexed_small")
            self.assertTrue(is_indexed_rmsnorm_rope_supported(q_d512, None, freqs, q_d512_pos, rd))
            self.assertEqual(indexed_rmsnorm_rope_path(generic_q, None, rd), "materialized_fallback")
            self.assertFalse(is_indexed_rmsnorm_rope_supported(generic_q, None, freqs, generic_pos, rd))

        with _ScopedEnv("DSV4_INDEXED_RMSNORM_ROPE_Q_LARGE_M", "1"):
            self.assertEqual(indexed_rmsnorm_rope_path(large_q, None, rd), "indexed_large")
            self.assertTrue(is_indexed_rmsnorm_rope_supported(large_q, None, freqs, large_pos, rd))

    def test_q_d128_threshold_env_does_not_disable_indexed_path(self) -> None:
        with _ScopedEnv("DSV4_INDEXED_RMSNORM_ROPE_Q_MAX_M", "4"):
            torch.manual_seed(700)
            m, h, d, rd = 8, 64, 128, 64
            q = torch.randn(m, 1, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
            freqs = _freqs(64, rd)
            pos = torch.randint(0, 64, (m,), dtype=torch.int32, device="cuda")

            self.assertEqual(indexed_rmsnorm_rope_path(q, None, rd), "indexed_small")
            self.assertTrue(is_indexed_rmsnorm_rope_supported(q, None, freqs, pos, rd))
            ref = _materialized_ref(q, None, freqs, pos, rd)
            cand = dsv4_indexed_rmsnorm_rope(q, None, freqs, pos, rd)
            self.assertLessEqual(float((cand.float() - ref.float()).abs().max()), 2e-2)

    def test_kv_d512_threshold_env_does_not_disable_indexed_path(self) -> None:
        with _ScopedEnv("DSV4_INDEXED_RMSNORM_ROPE_KV_MAX_M", "4"):
            torch.manual_seed(701)
            m, d, rd = 8, 512, 64
            kv = torch.randn(m, 1, d, dtype=torch.bfloat16, device="cuda") * 0.5
            weight = torch.randn(d, dtype=torch.bfloat16, device="cuda").abs() + 0.25
            freqs = _freqs(64, rd)
            pos = torch.randint(0, 64, (m,), dtype=torch.int64, device="cuda")

            self.assertEqual(indexed_rmsnorm_rope_path(kv, weight, rd), "indexed_small")
            self.assertTrue(is_indexed_rmsnorm_rope_supported(kv, weight, freqs, pos, rd))
            ref = _materialized_ref(kv, weight, freqs, pos, rd)
            cand = dsv4_indexed_rmsnorm_rope(kv, weight, freqs, pos, rd)
            self.assertLessEqual(float((cand.float() - ref.float()).abs().max()), 5e-2)


if __name__ == "__main__":
    unittest.main()
