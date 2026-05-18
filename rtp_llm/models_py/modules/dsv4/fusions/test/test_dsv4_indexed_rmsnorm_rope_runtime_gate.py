from __future__ import annotations

import os
import sys
import types
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.dsv4_indexed_rope import (
    dsv4_indexed_rmsnorm_rope,
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


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSV4IndexedRmsnormRopeRuntimeGate(unittest.TestCase):
    def test_q_d128_large_m_path_matches_materialized(self) -> None:
        torch.manual_seed(700)
        m, h, d, rd = 2049, 64, 128, 64
        q = torch.randn(m, 1, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
        freqs = _freqs(m + 64, rd)
        pos = torch.randint(0, int(freqs.shape[0]), (m,), dtype=torch.int32, device="cuda")

        ref = _materialized_ref(q, None, freqs, pos, rd)
        cand = dsv4_indexed_rmsnorm_rope(q, None, freqs, pos, rd)
        self.assertLessEqual(float((cand.float() - ref.float()).abs().max()), 2e-2)

    def test_q_d512_large_m_path_matches_materialized(self) -> None:
        torch.manual_seed(702)
        m, h, d, rd = 97, 64, 512, 64
        q = torch.randn(m, 1, h, d, dtype=torch.bfloat16, device="cuda") * 0.5
        freqs = _freqs(m + 64, rd)
        pos = torch.randint(0, int(freqs.shape[0]), (m,), dtype=torch.int32, device="cuda")

        ref = _materialized_ref(q, None, freqs, pos, rd)
        cand = dsv4_indexed_rmsnorm_rope(q, None, freqs, pos, rd)
        self.assertLessEqual(float((cand.float() - ref.float()).abs().max()), 5e-2)

    def test_kv_d512_indexed_path_matches_materialized(self) -> None:
        torch.manual_seed(701)
        m, d, rd = 8, 512, 64
        kv = torch.randn(m, 1, d, dtype=torch.bfloat16, device="cuda") * 0.5
        weight = torch.randn(d, dtype=torch.bfloat16, device="cuda").abs() + 0.25
        freqs = _freqs(64, rd)
        pos = torch.randint(0, 64, (m,), dtype=torch.int64, device="cuda")

        ref = _materialized_ref(kv, weight, freqs, pos, rd)
        cand = dsv4_indexed_rmsnorm_rope(kv, weight, freqs, pos, rd)
        self.assertLessEqual(float((cand.float() - ref.float()).abs().max()), 5e-2)


if __name__ == "__main__":
    unittest.main()
