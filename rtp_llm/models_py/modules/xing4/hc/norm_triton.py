"""Fused hc-row RMSNorm Triton kernel for the Xing4.0 hyper-connection unit.

The mathematical authority is ``Xing4HCUnit._pre_impl`` in
``rtp_llm/models_py/model_desc/xing4_0_model.py`` (read-only; this module must
stay equivalent to it):

    x_flat      = x.flatten(-2)            # [T, hc*H] bf16 (14336 for hc=4, H=3584)
    rsqrt       = torch.rsqrt(
        x_flat.float().square().mean(-1, keepdim=True) + norm_eps)          # fp32
    flat_normed = (x_flat.float() * rsqrt).to(x.dtype)   # bf16, feeds F.linear

The eager chain lowers to 6-8 CUDA micro kernels per call (aten::to /
aten::mul(square) / aten::mean / aten::add / aten::rsqrt / aten::to /
aten::mul / aten::to) and materializes [T, hc*H] fp32 intermediates; this
module fuses it into a single Triton row kernel:

    out[i, j] = bf16(fp32(x[i, j]) * rsqrt(mean_j(fp32(x[i, j])**2) + eps))

Design points:

* Squares are accumulated in fp32 registers with a block-cyclic loop over the
  14336-wide row (7 blocks of 2048); no fp32 tensor of shape [T, hc*H] is ever
  materialized.
* Consequently the torch path's chunking (``XING4_HC_FALLBACK_CHUNK_TOKENS``,
  16384 by default, which exists to bound the eager chain's fp32 intermediate)
  is NOT needed on the fused path: any T is handled in a single launch
  (T >= 65536 validated in the UT). The kernel takes an arbitrary row stride,
  so callers that keep chunking for unrelated reasons (e.g. GEMM tiling) can
  pass row slices without copies.
* Index-arithmetic contract (same pitfall as
  ``modules/dsv4/_fused_rmsnorm_rope_triton.py``): the program id is cast to
  int64 BEFORE the row-stride multiply. With 14336 columns, T >= 149726 rows
  puts the linear index past INT32_MAX and int32 offsets would silently wrap
  (validated at T=150000 in the UT).
* Equivalence is tolerance-level, not bitwise: the fp32 reduction order
  differs from torch's ``mean``, so rare elements can differ by 1 bf16 ulp --
  the same equivalence class the K1 research lane (R1) measured for every
  fused candidate, including torch's own ``aten::_fused_rms_norm``. The UT
  gates with rtol = atol = 2**-7.

An alternative backend selected with ``XING4_HC_NORM_IMPL=torch`` thin-wraps
``torch.nn.functional.rms_norm`` (the R1 reuse verdict: ``aten::_fused_rms_norm``
is a single fused CUDA kernel on the pinned torch 2.11.0+cu130; it requires
torch >= 2.10 for the fused CUDA dispatch, and the flattened last dim must
equal hc*H, which this wrapper guarantees). Both backends are covered by
``rtp_llm/models_py/modules/xing4/test/test_hc_norm_triton.py``.
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

_IMPL_ENV = "XING4_HC_NORM_IMPL"
_DEFAULT_BLOCK_N = 2048  # 14336 = 7 * 2048 -> block-cyclic row accumulation


@triton.jit
def _hc_row_rmsnorm_kernel(
    x_ptr,
    y_ptr,
    eps,
    n_cols,
    x_row_stride,
    y_row_stride,
    BLOCK_N: tl.constexpr,
):
    # Cast pid to int64 BEFORE the row-stride multiply: rows * 14336 exceeds
    # INT32_MAX from ~149726 rows on, and int32 offsets would silently wrap.
    # Same contract as modules/dsv4/_fused_rmsnorm_rope_triton.py.
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * x_row_stride
    y_row = y_ptr + row * y_row_stride

    # Pass 1: fp32 sum of squares over the row, block-cyclic (14336 = 7*2048).
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        v = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        acc += v * v
    mean_sq = tl.sum(acc, axis=0) / n_cols
    rstd = tl.rsqrt(mean_sq + eps)

    # Pass 2: scale and cast back in a single store (one bf16 materialization).
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        v = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_row + offs, (v * rstd).to(y_ptr.dtype.element_ty), mask=mask)


def _row_view(x: torch.Tensor):
    """Flatten trailing (.., hc, H) dims into rows, mirroring ``x.flatten(-2)``.

    Returns ``(rows_view, flat_shape)``: ``rows_view`` is a [rows, cols] view
    with unit column stride (its row stride may exceed cols, e.g. for chunked
    row slices); ``flat_shape`` is the reference ``flatten(-2)`` shape the
    output is viewed back into. Exotic non-row-major inputs fall back to a
    torch copy (one extra launch) -- the production path never hits this.
    """
    if x.dim() < 2:
        raise ValueError(
            f"hc_row_rms_norm expects dim >= 2 ([T, hc, H] or [T, hc*H]), got {x.dim()}"
        )
    x_flat = x.flatten(-2) if x.dim() > 2 else x
    flat_shape = x_flat.shape
    cols = flat_shape[-1]
    rows = x_flat.numel() // cols
    x_rows = x_flat.reshape(rows, cols)
    if x_rows.stride(-1) != 1:
        x_rows = x_rows.contiguous()
    return x_rows, flat_shape


def hc_row_rms_norm_triton(
    x: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Fused [.., hc, H] -> [.., hc*H] row RMSNorm in one Triton kernel launch.

    ``eps`` is a runtime parameter (fp32, added inside the rsqrt, matching the
    reference). ``out`` optionally provides the output buffer with the
    ``x.flatten(-2)`` shape, matching dtype/device and unit column stride.
    """
    if not x.is_cuda:
        raise ValueError("hc_row_rms_norm_triton requires a CUDA tensor")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported dtype {x.dtype}")
    x_rows, flat_shape = _row_view(x)
    rows, cols = x_rows.shape
    if out is not None:
        if (
            tuple(out.shape) != tuple(flat_shape)
            or out.dtype != x.dtype
            or out.device != x.device
        ):
            raise ValueError(
                "out must match x.flatten(-2) shape with x's dtype and device"
            )
        out_rows = out.reshape(rows, cols)
        if out_rows.stride(-1) != 1:
            raise ValueError("out requires unit stride on the last dim")
    else:
        out_rows = torch.empty((rows, cols), dtype=x.dtype, device=x.device)
    if rows == 0:
        return out_rows.view(*flat_shape)

    block_n = (
        _DEFAULT_BLOCK_N
        if cols >= _DEFAULT_BLOCK_N
        else triton.next_power_of_2(cols)
    )
    num_warps = 8 if block_n >= 2048 else (4 if block_n >= 512 else 1)
    _hc_row_rmsnorm_kernel[(rows,)](
        x_rows,
        out_rows,
        eps,
        cols,
        x_rows.stride(0),
        out_rows.stride(0),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out_rows.view(*flat_shape)


def hc_row_rms_norm_torch(x: torch.Tensor, eps: float) -> torch.Tensor:
    """R1 reuse alternative: thin wrapper over ``torch.nn.functional.rms_norm``.

    On the pinned bazel torch 2.11.0+cu130 this dispatches to the single fused
    CUDA kernel ``aten::_fused_rms_norm`` (weight=None is allowed); on older
    torch (< 2.10) or non-CUDA backends it decomposes into an equivalent
    composite chain instead of fusing.
    """
    x_rows, flat_shape = _row_view(x)
    cols = x_rows.shape[-1]
    out = torch.nn.functional.rms_norm(x_rows, [cols], weight=None, eps=eps)
    return out.view(*flat_shape)


def hc_row_rms_norm(
    x: torch.Tensor, eps: float, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Env-dispatched entry: ``XING4_HC_NORM_IMPL`` in {"triton", "torch"}.

    Defaults to the fused Triton kernel (this lane's product). The "torch"
    value selects the R1-reuse thin wrapper and does not accept ``out``.
    """
    impl = os.environ.get(_IMPL_ENV, "triton").strip().lower()
    if impl == "torch":
        if out is not None:
            raise ValueError("out= is only supported by the triton backend")
        return hc_row_rms_norm_torch(x, eps)
    if impl != "triton":
        raise ValueError(
            f"invalid {_IMPL_ENV}={impl!r}; expected 'triton' or 'torch'"
        )
    return hc_row_rms_norm_triton(x, eps, out)
