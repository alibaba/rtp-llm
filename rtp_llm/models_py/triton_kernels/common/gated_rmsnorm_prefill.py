"""Output-only, tiled gated RMSNorm for inference with small head groups.

Several independent groups share a CTA. This reduces CTA scheduling overhead
without changing normalization groups or specializing for a sequence length.
The training/general API in layernorm_gated still returns its statistics.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gated_rmsnorm_rows(
    X,
    Z,
    W,
    B,
    Y,
    ROWS,
    SX: tl.constexpr,
    SZ: tl.constexpr,
    N: tl.constexpr,
    GROUPS: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SHARED: tl.constexpr,
    BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    r = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.arange(0, BD)
    token = r // GROUPS
    group = r % GROUPS
    mask = (r[:, None] < ROWS) & (c[None, :] < D)
    off = group[:, None] * D + c[None, :]
    x = tl.load(X + token[:, None].to(tl.int64) * SX + off, mask, 0).to(tl.float32)
    z = tl.load(Z + token[:, None].to(tl.int64) * SZ + off, mask, 0).to(tl.float32)
    woff = c[None, :] if SHARED else off
    w = tl.load(W + woff, c[None, :] < D, 0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / D
    y = (x * (1.0 / tl.sqrt(var + EPS))[:, None]) * w
    if BIAS:
        y += tl.load(B + woff, c[None, :] < D, 0).to(tl.float32)
    gate = tl.sigmoid(z)
    if SILU:
        gate = z * gate
    y *= gate
    tl.store(Y + r[:, None].to(tl.int64) * D + c[None, :], y, mask)


def gated_rmsnorm_prefill(
    x,
    gate,
    weight,
    bias=None,
    eps=1e-6,
    group_size=None,
    activation="silu",
    *,
    tile_rows=4
):
    """Same forward expression as layer_norm_fwd(norm_before_gate=True)."""
    if x.ndim != 2 or gate.shape != x.shape:
        raise ValueError("Expected matching two-dimensional input and gate")
    d = weight.numel() if group_size is None else group_size
    m, n = x.shape
    if d <= 0 or n % d or d > 1024 or activation not in ("silu", "sigmoid"):
        raise ValueError("Unsupported normalization groups or activation")
    if not (x.is_cuda and gate.device == weight.device == x.device):
        raise ValueError("Input, gate and weight must be on the same CUDA device")
    if x.stride(-1) != 1 or gate.stride(-1) != 1 or weight.stride(-1) != 1:
        raise ValueError("Feature dimensions must be contiguous")
    if weight.shape not in ((d,), (n,)) or (
        bias is not None and bias.shape != weight.shape
    ):
        raise ValueError("Invalid weight or bias shape")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = m * (n // d)
    if rows:
        _gated_rmsnorm_rows[(triton.cdiv(rows, tile_rows),)](
            x,
            gate,
            weight,
            bias,
            out,
            rows,
            x.stride(0),
            gate.stride(0),
            n,
            n // d,
            d,
            eps,
            weight.numel() == d,
            bias is not None,
            activation == "silu",
            tile_rows,
            triton.next_power_of_2(d),
            num_warps=4,
        )
    return out


def supports_gated_rmsnorm_prefill(x, gate, weight, bias, group_size):
    return (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim == 2
        and x.shape[0] >= 2048
        and gate.shape == x.shape
        and x.dtype == gate.dtype == weight.dtype == torch.bfloat16
        and x.device == gate.device == weight.device
        and 0 < group_size <= 256
        and x.shape[1] % group_size == 0
        and weight.shape in ((group_size,), (x.shape[1],))
        and x.stride(1) == gate.stride(1) == weight.stride(0) == 1
        and (
            bias is None
            or (
                bias.shape == weight.shape
                and bias.device == x.device
                and bias.dtype == weight.dtype
                and bias.stride(0) == 1
            )
        )
        and torch.cuda.get_device_capability(x.device)[0] == 10
    )
