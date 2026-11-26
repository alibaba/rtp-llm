"""Shared inference launcher for KDA gate kernels."""

import torch
import triton


def kda_gate_fwd(
    kernel,
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)

    yg = torch.empty_like(g, dtype=output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), H)

    kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        beta=None,
        yg=yg,
        yb=None,
        T=T,
        H=H,
        D=K,
        BD=triton.next_power_of_2(K),
        lower_bound=lower_bound,
    )
    return yg


class KDAGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kernel,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        lower_bound: float | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        yg = kda_gate_fwd(
            kernel,
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            output_dtype=output_dtype,
        )
        return yg
