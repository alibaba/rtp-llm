"""Triton fused bias + GELU epilogue for the non-quantized CUDA Linear path.

Whisper Phase C kernel fusion, step 5 (lane contract). ``DenseMLP`` runs
``up_proj.forward_with_bias_gelu(x)``; the ``LinearBase`` default is
``F.gelu(self.forward(input))`` and ``CudaF16Linear.forward`` folds the bias
into the cuBLASLt GEMM epilogue (``F.linear(input, weight, bias)``), so the
default leaves one ATen-native GELU kernel per MLP per step on the decode
timeline (whisper: 32 decoder layers x 1 MLP x 20 steps = 640 events).
This module replaces that pair with one Triton kernel applying the bias add
and the exact erf GELU together:

    LinearBase default: F.gelu(F.linear(x, w, b))    # GEMM(bias) + ATen gelu
    this path:          bias_gelu(F.linear(x, w, None), b)
                       # GEMM(no bias) + one fused Triton bias+gelu kernel

Wiring choice (explicit): the bias is deferred from the cuBLASLt epilogue
into the Triton kernel, so the whole post-GEMM chain (bias add + activation)
is one custom kernel and the bias is still added exactly once. The
alternative (GEMM with bias + gelu-only Triton kernel) was rejected: it
forfeits the bias+gelu chain fusion (the kernel's bias operand would go
unused) while saving no launches - both options are two kernels - and it
buys no extra numerical fidelity beyond the fp32 erf evaluation that this
path already performs.

Numerics contract: GELU is the exact erf form, identical to
``F.gelu(..., approximate='none')`` and ATen's CUDA opmath - upcast to
fp32, ``y = x * 0.5 * (1 + erf(x * M_SQRT1_2))`` (the constant fp32-rounds
to the same value ATen uses), round back to the input dtype; both ATen and
Triton lower erf to libdevice ``erff``. No tanh approximation is used.
The deviation vs the default path is bounded: the no-bias GEMM result is
rounded to the activation dtype once *before* the bias add, where the
default rounds once *after* the epilogue add (and cuBLAS may pick a
different algorithm for mm vs addmm), so activations differ by at most
about one ulp of the activation dtype.

Shape contract: the row count T (= bsz * seq) only sizes the launch grid;
K (output features), row strides and the bias stride are runtime arguments.
BLOCK_K is derived from K (a model constant such as whisper-large-v3 ffn
5120), never from T, so one compiled variant per (dtype, HAS_BIAS,
BLOCK_K) serves every bsz/seq shape. The compiled kernel name is
``_bias_gelu_kernel`` - that is the name to look for in decode timelines
(where ``at::native`` gelu events must drop to zero).

Fallbacks (bit-exact with the LinearBase default composition): non-CUDA
tensors (CPU/PPU), missing triton, unsupported dtype, 1-D input, malformed
bias, or the kill switch ``RTP_LLM_LINEAR_BIAS_GELU_TRITON=0`` all take
``F.gelu(F.linear(input, weight, bias))``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - cpu/ppu containers without triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

__all__ = ["bias_gelu", "linear_bias_gelu"]

# Set to "0" to disable the Triton path (parity A/B, emergency rollback).
_ENV_KILL_SWITCH = "RTP_LLM_LINEAR_BIAS_GELU_TRITON"

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_MAX_BLOCK_K = 1024

if _TRITON_AVAILABLE:

    @triton.jit
    def _bias_gelu_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        K,
        x_row_stride,
        bias_stride,
        out_row_stride,
        HAS_BIAS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # int64 row base: large prefills flatten to T rows and
        # pid * row_stride silently overflows int32 past 2^31 elements
        # (same pitfall as the dsv4 _fused_rmsnorm_rope_kernel).
        row = tl.program_id(0).to(tl.int64)
        chunk = tl.program_id(1)
        cols = chunk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = cols < K

        x_row = x_ptr + row * x_row_stride
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + cols * bias_stride, mask=mask, other=0.0
            ).to(tl.float32)
            x = x + bias

        # Exact erf GELU (F.gelu default, approximate='none') in fp32 opmath,
        # matching ATen's GeluCUDAKernelImpl:
        #     y = x * 0.5 * (1 + erf(x * M_SQRT1_2))
        # 0.7071067811865476 fp32-rounds to ATen's constexpr M_SQRT1_2.
        y = x * (0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476)))

        out_row = out_ptr + row * out_row_stride
        tl.store(out_row + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def _triton_enabled() -> bool:
    return os.environ.get(_ENV_KILL_SWITCH, "1") != "0"


def _bias_gelu_triton_ready(x: torch.Tensor, bias: Optional[torch.Tensor]) -> bool:
    """Whether :func:`bias_gelu` can take the Triton path for this (x, bias)."""
    if not (_TRITON_AVAILABLE and _triton_enabled()):
        return False
    if not x.is_cuda or x.dim() < 2 or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.stride(-1) != 1:
        return False
    if x.dim() > 2 and not x.is_contiguous():
        return False  # leading dims cannot be flattened as a view
    if bias is not None:
        if bias.dim() != 1 or bias.numel() != x.shape[-1] or bias.device != x.device:
            return False
    return True


def bias_gelu(x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """Elementwise ``gelu(x + bias)`` over the last dimension of ``x``.

    ``x``: CUDA tensor with unit innermost stride; leading dims are
    flattened into rows and row strides may be arbitrary. ``bias``:
    optional 1-D tensor with ``numel() == x.shape[-1]``. Returns a new
    contiguous tensor shaped like ``x``.

    Falls back to eager torch (``F.gelu(x + bias)``) whenever the Triton
    path is unavailable, so the op is safe to call from any backend.
    """
    if not _bias_gelu_triton_ready(x, bias):
        if bias is None:
            return F.gelu(x)
        return F.gelu(x + bias)

    orig_shape = x.shape
    K = int(orig_shape[-1])
    if x.numel() == 0:
        return torch.empty(orig_shape, dtype=x.dtype, device=x.device)
    x2 = x if x.dim() == 2 else x.view(-1, K)
    T = int(x2.shape[0])
    out = torch.empty((T, K), dtype=x.dtype, device=x.device)

    block_k = min(_MAX_BLOCK_K, triton.next_power_of_2(K))
    grid = (T, triton.cdiv(K, block_k))
    # bias_ptr is never dereferenced when HAS_BIAS is False (the load sits
    # inside the constexpr branch), so x2 stands in as a dummy pointer.
    bias_arg = bias if bias is not None else x2
    _bias_gelu_kernel[grid](
        x2,
        bias_arg,
        out,
        K,
        x2.stride(0),
        bias.stride(0) if bias is not None else 1,
        out.stride(0),
        HAS_BIAS=bias is not None,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out.view(orig_shape)


def linear_bias_gelu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """``gelu(input @ weight^T + bias)`` with a fused Triton bias+gelu epilogue.

    ``weight`` uses the ``F.linear`` convention ``[out_features, in_features]``
    (``CudaF16Linear.weight`` is already stored transposed to this layout).
    The GEMM is issued without the bias (cuBLAS) and one Triton kernel
    applies the bias add and the exact erf GELU together; see the module
    docstring for the full wiring and numerics contract.

    Every condition the Triton path cannot satisfy falls back to the
    ``LinearBase`` default composition ``F.gelu(F.linear(input, weight,
    bias))`` bit-exactly (non-CUDA tensors, missing triton, unsupported
    dtype, 1-D input, malformed bias, or the kill-switch env).
    """
    if bias is not None and (
        bias.dim() != 1
        or bias.numel() != weight.shape[0]
        or bias.device != input.device
    ):
        # Keep the default behavior for malformed bias (F.linear raises the
        # same error the base implementation would).
        return F.gelu(F.linear(input, weight, bias))
    if not (_TRITON_AVAILABLE and _triton_enabled()):
        return F.gelu(F.linear(input, weight, bias))
    if not input.is_cuda or input.dtype not in _SUPPORTED_DTYPES or input.dim() < 2:
        return F.gelu(F.linear(input, weight, bias))
    # The GEMM output of F.linear is freshly allocated, contiguous and
    # CUDA-resident, so bias_gelu takes the Triton path here; its own
    # readiness check still guards the eager fallback for exotic cases.
    gemm = F.linear(input, weight, None)
    return bias_gelu(gemm, bias)
