from __future__ import annotations

import os
import unittest

import torch

import libth_transformer_config  # noqa: F401
from librtp_compute_ops import rtp_llm_ops

os.environ.setdefault("DSV4_INDEXED_RMSNORM_ROPE_Q_LARGE_M", "1")


def _make_freqs(max_pos: int, rd: int) -> torch.Tensor:
    pos = torch.arange(max_pos, dtype=torch.float32, device="cuda")[:, None]
    pair = torch.arange(rd // 2, dtype=torch.float32, device="cuda")[None, :]
    angle = (pos + 1) * (pair + 3) * 0.001
    return torch.polar(torch.ones_like(angle), angle).contiguous()


def _ref_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs: torch.Tensor,
    pos: torch.Tensor,
    rd: int,
) -> torch.Tensor:
    d = x.shape[-1]
    x_flat = x.reshape(-1, d).float()
    inv = torch.rsqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
    y = x_flat * inv
    if weight is not None:
        y = y * weight.float()[None, :]
    freq_stride_n = x.shape[2] if x.dim() == 4 else 1
    row_token = torch.arange(x_flat.shape[0], device=x.device, dtype=torch.long) // freq_stride_n
    selected = freqs.index_select(0, pos.to(torch.long).index_select(0, row_token))
    nope = d - rd
    tail = y[:, nope:].reshape(-1, rd // 2, 2)
    real = tail[..., 0]
    imag = tail[..., 1]
    cos = selected.real
    sin = selected.imag
    y_tail = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(-2)
    y = torch.cat((y[:, :nope], y_tail), dim=-1)
    return y.reshape(x.shape).to(torch.bfloat16)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDSV4IndexedRmsnormRopeDirectOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not hasattr(rtp_llm_ops, "dsv4_indexed_rmsnorm_rope"):
            raise AssertionError("rtp_llm_ops.dsv4_indexed_rmsnorm_rope is not registered")
        cls.freqs = _make_freqs(8192, 64)

    def _run_case(
        self,
        name: str,
        shape: tuple[int, ...],
        has_weight: bool,
        m_values: list[int],
        tol: float,
    ) -> None:
        for m in m_values:
            for dtype in (torch.int32, torch.int64):
                torch.manual_seed(1000 + m + len(shape))
                actual_shape = (m, *shape[1:])
                x = (torch.randn(actual_shape, dtype=torch.bfloat16, device="cuda") * 0.5).contiguous()
                weight = (
                    (torch.randn((actual_shape[-1],), dtype=torch.bfloat16, device="cuda").abs() + 0.25).contiguous()
                    if has_weight
                    else None
                )
                weight_arg = weight if weight is not None else torch.empty((0,), dtype=torch.bfloat16, device="cuda")
                pos = ((torch.arange(m, device="cuda") * 37 + 11) % 8192).to(dtype=dtype).contiguous()
                out = torch.empty_like(x)

                rtp_llm_ops.dsv4_indexed_rmsnorm_rope(
                    x,
                    weight_arg,
                    self.freqs,
                    pos,
                    out,
                    64,
                    1e-6,
                    has_weight,
                )
                torch.cuda.synchronize()

                ref = _ref_rmsnorm_rope(x, weight, self.freqs, pos, 64)
                diff = (out.float() - ref.float()).abs().max().item()
                self.assertLessEqual(diff, tol, f"{name} M={m} dtype={dtype} max_abs={diff}")

    def test_q_d128_no_weight_matches_reference(self) -> None:
        self._run_case(
            "q_d128_no_weight",
            (1, 1, 64, 128),
            False,
            [1, 3, 17, 127, 128, 1021, 3079, 4096],
            2e-2,
        )

    def test_q_d512_no_weight_matches_reference(self) -> None:
        self._run_case(
            "q_d512_no_weight",
            (1, 1, 64, 512),
            False,
            [1, 3, 17, 127, 1021],
            5e-2,
        )

    def test_kv_d512_weight_matches_reference(self) -> None:
        self._run_case("kv_d512_weight", (1, 1, 512), True, [1, 5, 127, 1021], 5e-2)


if __name__ == "__main__":
    unittest.main()
