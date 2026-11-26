"""SM100 FP8 / MXFP8 / W4A8 dense linear kernels for Qwen 4.5-Flash.

Provides regime-dispatched dense linear acceleration across Qwen 4.5-Flash
projections (``in_proj_qkvz``, ``q_proj``, ``o_proj``, ``latent_down/up``,
``shared_expert``, ``DenseMLP``, ``lm_head``) with ``cos >= 0.999`` parity in
FP8/MXFP8 modes and 4x weight compression in opt-in ``w4a8`` mode:

  - ``trtllm_ll`` (``flashinfer.mm_fp8(..., backend="trtllm_low_latency")``):
    Decode ``M <= 8``, large ``N >= 16384`` (``lm_head``, ``dense_gu``, ``q_proj``).
  - ``scaled_mm`` (fused Triton rowwise FP8 quant + ``torch._scaled_mm``):
    Decode/medium-M ``1 <= M < 2048`` (``cos >= 0.9993``; zero weight pre-shuffle).
  - ``mxfp8`` (``flashinfer.mxfp8_quantize`` + ``flashinfer.mm_mxfp8(cutlass)``):
    Compute-bound Prefill ``M >= 2048`` (or ``M >= 1024, N >= 32768``), ~2700-2800 TF.
  - ``deepgemm_fp8`` (``deep_gemm.per_token_cast_to_fp8`` + ``deep_gemm.fp8_gemm_nt``):
    SM100 block-128 fallback when FlashInfer MXFP8 is unavailable.
  - ``w4a8`` (``per_token_group_quant_fp8_ue8m0`` + ``deep_gemm.fp8_fp4_gemm_nt``):
    Opt-in W4A8 (~2800-2957 TF, 4x smaller weights, ``cos ~ 0.993``).
  - ``bf16`` (``F.linear``): Clean fallback for small Decode GEMMs (``M <= 64, N <= 8192``)
    or unaligned dimensions.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_5.activation import per_token_group_quant_fp8_ue8m0
from rtp_llm.models_py.modules.qwen4_5.mxfp4_ops import (
    has_deepgemm_fp8_fp4,
    prepare_fp4_weight_scale_for_deepgemm,
    quantize_mxfp4,
)

_FP8_E4M3_MAX = 448.0
_QUANT_EPS = 1.0e-12
_TRTLLM_PERM_CACHE: Dict[torch.Size, torch.Tensor] = {}
_MXFP8_PROBE_CACHE: Dict[int, bool] = {}
_TRTLLM_LL_PROBE_CACHE: Dict[int, bool] = {}


def has_flashinfer_mxfp8(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    dev_idx = int(device_index)
    cached = _MXFP8_PROBE_CACHE.get(dev_idx)
    if cached is not None:
        return cached
    cap = torch.cuda.get_device_capability(dev_idx)
    if cap[0] < 10:
        _MXFP8_PROBE_CACHE[dev_idx] = False
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        import flashinfer  # type: ignore

        if not (hasattr(flashinfer, "mxfp8_quantize") and hasattr(flashinfer, "mm_mxfp8")):
            _MXFP8_PROBE_CACHE[dev_idx] = False
            return False
        # One-shot JIT probe catches container CUDA header conflicts safely pre-capture.
        with torch.cuda.device(dev_idx):
            dev = torch.device("cuda", dev_idx)
            t_a = torch.zeros((16, 128), dtype=torch.bfloat16, device=dev)
            t_w = torch.zeros((128, 128), dtype=torch.bfloat16, device=dev)
            aq, asc = flashinfer.mxfp8_quantize(t_a, is_sf_swizzled_layout=True, alignment=32)
            wq, wsc = flashinfer.mxfp8_quantize(t_w, is_sf_swizzled_layout=True, alignment=32)
            _ = flashinfer.mm_mxfp8(
                aq, wq.t(), asc, wsc, out_dtype=torch.bfloat16, backend="cutlass"
            )
        _MXFP8_PROBE_CACHE[dev_idx] = True
        return True
    except Exception:
        _MXFP8_PROBE_CACHE[dev_idx] = False
        return False


def has_flashinfer_trtllm_ll(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    dev_idx = int(device_index)
    cached = _TRTLLM_LL_PROBE_CACHE.get(dev_idx)
    if cached is not None:
        return cached
    cap = torch.cuda.get_device_capability(dev_idx)
    if cap[0] != 10:
        _TRTLLM_LL_PROBE_CACHE[dev_idx] = False
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        import flashinfer  # type: ignore

        if not (
            hasattr(flashinfer, "mm_fp8")
            and hasattr(flashinfer, "prepare_low_latency_gemm_weights")
        ):
            _TRTLLM_LL_PROBE_CACHE[dev_idx] = False
            return False
        # One-shot cubin probe catches non-writable cubin cache PermissionError safely.
        with torch.cuda.device(dev_idx):
            dev = torch.device("cuda", dev_idx)
            ax = torch.zeros((8, 128), dtype=torch.float8_e4m3fn, device=dev)
            wq = torch.zeros((128, 128), dtype=torch.float8_e4m3fn, device=dev)
            w_prep = flashinfer.prepare_low_latency_gemm_weights(wq, _TRTLLM_PERM_CACHE)
            alpha = torch.ones((1,), dtype=torch.float32, device=dev)
            _ = flashinfer.mm_fp8(
                ax,
                w_prep,
                alpha=alpha,
                out_dtype=torch.bfloat16,
                backend="trtllm_low_latency",
            )
        _TRTLLM_LL_PROBE_CACHE[dev_idx] = True
        return True
    except Exception:
        _TRTLLM_LL_PROBE_CACHE[dev_idx] = False
        return False


@functools.lru_cache(maxsize=8)
def has_deepgemm_fp8(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(device_index)
    if cap[0] not in (9, 10):
        return False
    try:
        import deep_gemm  # type: ignore

        return hasattr(deep_gemm, "fp8_gemm_nt") and hasattr(
            deep_gemm, "per_token_cast_to_fp8"
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fast Fused Triton Rowwise & Tensorwise FP8 Quantization Kernels (CG-Safe)
# ---------------------------------------------------------------------------


@triton.jit
def _quant_fp8_rowwise_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    M,
    K,
    stride_x: tl.int64,
    stride_q: tl.int64,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    row_x = x_ptr + m * stride_x
    row_q = q_ptr + m * stride_q
    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mk = offs < K
        v = tl.load(row_x + offs, mask=mk, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))
    sc = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    inv_sc = 1.0 / sc
    tl.store(sc_ptr + m, sc)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mk = offs < K
        v = tl.load(row_x + offs, mask=mk, other=0.0).to(tl.float32)
        q = tl.clamp(v * inv_sc, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(row_q + offs, q, mask=mk)


@triton.jit
def _reduce_amax_rows_i32_kernel(
    x_ptr,
    amax_i32_ptr,
    M,
    K,
    stride_x: tl.int64,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    row_x = x_ptr + m * stride_x
    amax = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mk = offs < K
        v = tl.load(row_x + offs, mask=mk, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v), axis=0))
    # Positive IEEE-754 float32 bit pattern preserves numerical ordering in int32.
    amax_pos = tl.maximum(amax, 1e-12 * 448.0)
    tl.atomic_max(amax_i32_ptr, amax_pos.to(tl.int32, bitcast=True))


@triton.jit
def _apply_tensorwise_fp8_kernel(
    x_ptr,
    q_ptr,
    amax_i32_ptr,
    sc_out_ptr,
    w_scale: tl.float32,
    M,
    K,
    stride_x: tl.int64,
    stride_q: tl.int64,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    if m >= M:
        return
    amax = tl.load(amax_i32_ptr).to(tl.float32, bitcast=True)
    sc = tl.maximum(amax * (1.0 / 448.0), 1e-12)
    if m == 0:
        tl.store(sc_out_ptr, sc * w_scale)
    inv_sc = 1.0 / sc
    row_x = x_ptr + m * stride_x
    row_q = q_ptr + m * stride_q
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mk = offs < K
        v = tl.load(row_x + offs, mask=mk, other=0.0).to(tl.float32)
        q = tl.clamp(v * inv_sc, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(row_q + offs, q, mask=mk)


def quantize_fp8_rowwise_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D ``[M, K]`` tensor into FP8 E4M3FN + ``[M, 1]`` FP32 row scales."""
    x_2d = x if (x.ndim == 2 and x.stride(-1) == 1) else x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x_2d.shape
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x_2d.device)
    sc = torch.empty((M, 1), dtype=torch.float32, device=x_2d.device)
    if M == 0 or K == 0:
        return q, sc
    BK = min(4096, triton.next_power_of_2(K))
    _quant_fp8_rowwise_kernel[(M,)](
        x_2d, q, sc, M, K, x_2d.stride(0), q.stride(0), BLOCK_K=BK, num_warps=8
    )
    return q, sc


def quantize_fp8_tensorwise_triton(
    x: torch.Tensor, *, weight_scale: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` with one tensorwise scale and compute ``alpha = x_sc * weight_scale``."""
    x_2d = x if (x.ndim == 2 and x.stride(-1) == 1) else x.reshape(-1, x.shape[-1]).contiguous()
    M, K = x_2d.shape
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x_2d.device)
    if M == 0 or K == 0:
        return q, torch.ones((1,), dtype=torch.float32, device=x_2d.device)
    if M == 1:
        q, sc = quantize_fp8_rowwise_triton(x_2d)
        alpha = (sc.view(1) * float(weight_scale)).contiguous()
        return q, alpha
    BK = min(4096, triton.next_power_of_2(K))
    amax_i32 = torch.zeros((1,), dtype=torch.int32, device=x_2d.device)
    alpha = torch.empty((1,), dtype=torch.float32, device=x_2d.device)
    _reduce_amax_rows_i32_kernel[(M,)](
        x_2d, amax_i32, M, K, x_2d.stride(0), BLOCK_K=BK, num_warps=8
    )
    _apply_tensorwise_fp8_kernel[(M,)](
        x_2d,
        q,
        amax_i32,
        alpha,
        float(weight_scale),
        M,
        K,
        x_2d.stride(0),
        q.stride(0),
        BLOCK_K=BK,
        num_warps=8,
    )
    return q, alpha


# ---------------------------------------------------------------------------
# Weight Quantization Helpers
# ---------------------------------------------------------------------------


def quantize_weight_fp8_rowwise(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[N, K]`` weight into col-major ``w_q.t()`` and ``[1, N]`` FP32 scale."""
    amax = w.float().abs().amax(dim=-1, keepdim=True).clamp(min=_QUANT_EPS * _FP8_E4M3_MAX)
    sc = (amax / _FP8_E4M3_MAX).squeeze(-1).contiguous()
    q = (w.float() / sc[:, None]).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
    return q, sc.unsqueeze(0).contiguous()


def quantize_weight_fp8_tensorwise(w: torch.Tensor) -> Tuple[torch.Tensor, float]:
    amax = float(w.float().abs().max().clamp(min=_QUANT_EPS * _FP8_E4M3_MAX).item())
    sc = amax / _FP8_E4M3_MAX
    q = (w.float() / sc).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
    return q, sc


# ---------------------------------------------------------------------------
# Drop-in SM100FP8Linear Module & Functional API
# ---------------------------------------------------------------------------


class SM100FP8Linear(nn.Module):
    """Regime-dispatched SM100 dense linear layer (BF16 / FP8 / MXFP8 / W4A8).

    Maintains ``cos >= 0.999`` parity vs ``F.linear`` in default ``"auto"`` mode
    by combining ``trtllm_low_latency`` (``M=1..8``, huge N), ``torch._scaled_mm``
    (rowwise FP8 for Decode/medium M), ``flashinfer.mm_mxfp8`` (Prefill ``M>=2048``
    / large GEMMs), and BF16 for tiny Decode projections (``M <= 64, N <= 8192``).
    Opt-in ``mode="w4a8"`` dispatches ``deep_gemm.fp8_fp4_gemm_nt`` (4x weight
    compression, ``cos ~ 0.993``). All paths are 100% CUDA-Graph capture/replay safe.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        mode: str = "auto",
        keep_bf16: bool = True,
        prepare_trtllm_ll: bool = True,
        prepare_mxfp8: bool = True,
        prepare_deepgemm_fp8: bool = False,
        prepare_w4a8: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.mode = str(mode)
        self.dtype = dtype

        if keep_bf16 or self.mode == "bf16":
            self.weight: Optional[nn.Parameter] = nn.Parameter(
                torch.empty((self.out_features, self.in_features), dtype=dtype, device=device)
            )
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(
                torch.zeros((self.out_features,), dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

        # Cached quantized weight representations (prepared once at load time)
        self.register_buffer("w_fp8_row", None, persistent=False)
        self.register_buffer("w_sc_row", None, persistent=False)
        self.register_buffer("w_mxfp8", None, persistent=False)
        self.register_buffer("w_mxfp8_sc", None, persistent=False)
        self.register_buffer("w_dg_fp8", None, persistent=False)
        self.register_buffer("w_dg_sc", None, persistent=False)
        self.register_buffer("w_trtllm_ll", None, persistent=False)
        self._w_trtllm_ll_scale: float = 1.0
        self.register_buffer("w_fp4_i8", None, persistent=False)
        self.register_buffer("w_fp4_sc", None, persistent=False)

        self._prepare_trtllm_ll_flag = bool(prepare_trtllm_ll)
        self._prepare_mxfp8_flag = bool(prepare_mxfp8)
        self._prepare_dg_fp8_flag = bool(prepare_deepgemm_fp8)
        self._prepare_w4a8_flag = bool(prepare_w4a8 or mode == "w4a8")

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        mode: str = "auto",
        keep_bf16: bool = False,
        prepare_trtllm_ll: bool = True,
        prepare_mxfp8: bool = True,
        prepare_deepgemm_fp8: bool = False,
        prepare_w4a8: bool = False,
    ) -> "SM100FP8Linear":
        """Wrap and quantize an existing ``nn.Linear`` module."""
        w = linear.weight.data
        inst = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            mode=mode,
            keep_bf16=keep_bf16,
            prepare_trtllm_ll=prepare_trtllm_ll,
            prepare_mxfp8=prepare_mxfp8,
            prepare_deepgemm_fp8=prepare_deepgemm_fp8,
            prepare_w4a8=prepare_w4a8,
            dtype=w.dtype,
            device=w.device,
        )
        if linear.bias is not None and inst.bias is not None:
            inst.bias.data.copy_(linear.bias.data)
        inst.prepare_weights(w)
        return inst

    @torch.no_grad()
    def prepare_weights(self, w: Optional[torch.Tensor] = None) -> None:
        """Pre-quantize BF16 weight ``[N, K]`` into enabled SM100 tensor formats."""
        src = w if w is not None else self.weight
        if src is None:
            raise ValueError("Weight tensor is required when keep_bf16=False")
        if self.weight is not None and w is not None and w is not self.weight:
            self.weight.data.copy_(w)
        w_d = src.data
        N, K = self.out_features, self.in_features
        dev = w_d.device
        dev_idx = int(dev.index or 0) if dev.type == "cuda" else 0

        if not w_d.is_cuda or self.mode == "bf16":
            return

        auto_like = ("auto", "fp8")
        if K % 16 == 0 and N % 16 == 0 and (self.mode in auto_like or self.mode == "scaled_mm"):
            wq, wsc = quantize_weight_fp8_rowwise(w_d)
            self.w_fp8_row = wq
            self.w_sc_row = wsc

        if (
            self._prepare_mxfp8_flag
            and K % 128 == 0
            and N % 128 == 0
            and self.mode in ("auto", "fp8", "mxfp8")
            and has_flashinfer_mxfp8(dev_idx)
        ):
            try:
                import flashinfer  # type: ignore

                wm8, wms = flashinfer.mxfp8_quantize(w_d, is_sf_swizzled_layout=True, alignment=32)
                self.w_mxfp8 = wm8
                self.w_mxfp8_sc = wms
            except Exception:
                self.w_mxfp8 = None
                self.w_mxfp8_sc = None

        if (
            self._prepare_dg_fp8_flag
            and K % 128 == 0
            and N % 128 == 0
            and self.mode in ("auto", "fp8", "deepgemm_fp8")
            and has_deepgemm_fp8(dev_idx)
        ):
            try:
                import deep_gemm  # type: ignore

                wdq, wds = deep_gemm.per_block_cast_to_fp8(w_d, use_ue8m0=True)
                self.w_dg_fp8 = wdq
                self.w_dg_sc = wds
            except Exception:
                self.w_dg_fp8 = None
                self.w_dg_sc = None

        if (
            self._prepare_trtllm_ll_flag
            and K % 128 == 0
            and N % 128 == 0
            and self.mode in ("auto", "fp8", "trtllm_ll")
            and has_flashinfer_trtllm_ll(dev_idx)
        ):
            try:
                import flashinfer  # type: ignore

                wq_t, w_sc_t = quantize_weight_fp8_tensorwise(w_d)
                self.w_trtllm_ll = flashinfer.prepare_low_latency_gemm_weights(
                    wq_t, _TRTLLM_PERM_CACHE
                )
                self._w_trtllm_ll_scale = w_sc_t
            except Exception:
                self.w_trtllm_ll = None

        if (
            self._prepare_w4a8_flag
            and K % 128 == 0
            and N % 128 == 0
            and has_deepgemm_fp8_fp4(dev_idx)
        ):
            w_u8, w_sc = quantize_mxfp4(w_d)
            w_sc_dg = prepare_fp4_weight_scale_for_deepgemm(
                w_sc.unsqueeze(0), mn=N, k=K, num_groups=1
            ).squeeze(0)
            self.w_fp4_i8 = w_u8.view(torch.int8)
            self.w_fp4_sc = w_sc_dg

    def _resolve_backend(self, M: int, device: torch.device) -> str:
        env_mode = os.environ.get("QWEN4_5_DENSE_LINEAR_MODE", "").strip().lower()
        req = env_mode or self.mode
        if device.type != "cuda":
            return "bf16"
        N, K = self.out_features, self.in_features
        if req == "bf16":
            return "bf16" if self.weight is not None else "scaled_mm"
        if req == "trtllm_ll":
            return "trtllm_ll" if self.w_trtllm_ll is not None else "scaled_mm"
        if req == "scaled_mm":
            return "scaled_mm" if self.w_fp8_row is not None else "bf16"
        if req == "mxfp8":
            return "mxfp8" if self.w_mxfp8 is not None else "scaled_mm"
        if req == "deepgemm_fp8":
            return "deepgemm_fp8" if self.w_dg_fp8 is not None else "scaled_mm"
        if req == "w4a8":
            return "w4a8" if self.w_fp4_i8 is not None else "bf16"

        # Empirical SM100/SM103 Pareto frontier (cos >= 0.999):
        #   - Decode (M <= 512): keep all N < 16384 layers (GDN, QSA, MoE, DFlash fc)
        #     in BF16 so zero 1-ULP router flips occur and single-kernel BF16 wins
        #     over 2-kernel rowwise quant+FP8. Wide outputs (N >= 16384, lm_head_tp4)
        #     run scaled_mm (1.35x-1.6x faster than BF16, cos>=0.9993) or opt-in trtllm_ll.
        #   - Prefill (M > 512): use block-32 mxfp8 when available; otherwise keep
        #     N < 16384 in BF16 (avoids rowwise scaled_mm cos drop and latency regression).
        is_wide = N >= 16384
        if M <= 512:
            if not is_wide and self.weight is not None:
                return "bf16"
            prefer_ll = os.environ.get("QWEN4_5_FP8_PREFER_TRTLLM_LL", "0") == "1"
            if M <= 16 and prefer_ll and self.w_trtllm_ll is not None and (is_wide or self.weight is None):
                return "trtllm_ll"
            if self.w_fp8_row is not None:
                return "scaled_mm"
            if M <= 16 and self.w_trtllm_ll is not None:
                return "trtllm_ll"
            if self.weight is not None:
                return "bf16"
        if (M >= 4096 or (M >= 1024 and is_wide)) and self.w_mxfp8 is not None:
            return "mxfp8"
        if is_wide and self.w_fp8_row is not None:
            return "scaled_mm"
        if self.weight is not None:
            return "bf16"
        if self.w_fp8_row is not None:
            return "scaled_mm"
        if self.w_trtllm_ll is not None:
            return "trtllm_ll"
        if self.w_dg_fp8 is not None:
            return "deepgemm_fp8"
        if self.w_mxfp8 is not None:
            return "mxfp8"
        raise RuntimeError("SM100FP8Linear has no prepared weights for active backend.")

    def forward_prequant_rowwise(
        self, x_q: torch.Tensor, x_sc: torch.Tensor, *, orig_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Execute ``torch._scaled_mm`` reusing pre-quantized rowwise FP8 input."""
        if self.w_fp8_row is None or self.w_sc_row is None:
            raise RuntimeError("w_fp8_row is not prepared on SM100FP8Linear")
        out = torch._scaled_mm(
            x_q,
            self.w_fp8_row.t(),
            scale_a=x_sc,
            scale_b=self.w_sc_row,
            out_dtype=self.dtype,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape, self.out_features)

    def forward(self, x: torch.Tensor, *, mode: Optional[str] = None) -> torch.Tensor:
        orig_prefix = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        M = int(x_2d.shape[0])
        if M == 0:
            return torch.empty((*orig_prefix, self.out_features), dtype=self.dtype, device=x.device)

        chosen = (
            self._resolve_backend(M, x_2d.device)
            if mode is None
            else str(mode)
        )
        if chosen == "bf16" and self.weight is not None:
            return F.linear(x, self.weight, self.bias)

        if chosen == "trtllm_ll" and self.w_trtllm_ll is not None:
            import flashinfer  # type: ignore

            ax, alpha = quantize_fp8_tensorwise_triton(
                x_2d, weight_scale=self._w_trtllm_ll_scale
            )
            out = flashinfer.mm_fp8(
                ax,
                self.w_trtllm_ll,
                alpha=alpha,
                out_dtype=self.dtype,
                backend="trtllm_low_latency",
            )
        elif chosen == "mxfp8" and self.w_mxfp8 is not None and self.w_mxfp8_sc is not None:
            import flashinfer  # type: ignore

            ax, asc = flashinfer.mxfp8_quantize(
                x_2d, is_sf_swizzled_layout=True, alignment=32
            )
            out = flashinfer.mm_mxfp8(
                ax,
                self.w_mxfp8.t(),
                asc,
                self.w_mxfp8_sc,
                out_dtype=self.dtype,
                backend="cutlass",
            )
        elif chosen == "deepgemm_fp8" and self.w_dg_fp8 is not None and self.w_dg_sc is not None:
            import deep_gemm  # type: ignore

            ax, asc = deep_gemm.per_token_cast_to_fp8(x_2d, use_ue8m0=True)
            out = torch.empty((M, self.out_features), dtype=self.dtype, device=x_2d.device)
            deep_gemm.fp8_gemm_nt(
                (ax, asc), (self.w_dg_fp8, self.w_dg_sc), out, recipe=(1, 128, 128)
            )
        elif chosen == "w4a8" and self.w_fp4_i8 is not None and self.w_fp4_sc is not None:
            import deep_gemm  # type: ignore

            ax, asc = per_token_group_quant_fp8_ue8m0(x_2d, scale_format="deepgemm")
            out = torch.empty((M, self.out_features), dtype=self.dtype, device=x_2d.device)
            deep_gemm.fp8_fp4_gemm_nt(
                (ax, asc),
                (self.w_fp4_i8, self.w_fp4_sc),
                out,
                recipe_a=(1, 32),
                recipe_b=(1, 32),
            )
        elif self.w_fp8_row is not None and self.w_sc_row is not None:
            ax, asc = quantize_fp8_rowwise_triton(x_2d)
            out = torch._scaled_mm(
                ax,
                self.w_fp8_row.t(),
                scale_a=asc,
                scale_b=self.w_sc_row,
                out_dtype=self.dtype,
            )
        elif self.weight is not None:
            return F.linear(x, self.weight, self.bias)
        else:
            raise RuntimeError(f"Requested fp8_linear backend={chosen!r} not prepared.")

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_prefix, self.out_features)


def sm100_fp8_linear(
    x: torch.Tensor,
    w_q_t: torch.Tensor,
    w_scale_row: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Functional FP8 rowwise GEMM: ``x @ w_q_t`` using fused Triton quant + ``_scaled_mm``."""
    target_dtype = out_dtype or x.dtype
    orig_prefix = x.shape[:-1]
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.shape[0] == 0:
        return torch.empty((*orig_prefix, w_q_t.shape[1]), dtype=target_dtype, device=x.device)
    ax, asc = quantize_fp8_rowwise_triton(x_2d)
    out = torch._scaled_mm(ax, w_q_t, scale_a=asc, scale_b=w_scale_row, out_dtype=target_dtype)
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_prefix, w_q_t.shape[1])


__all__ = [
    "SM100FP8Linear",
    "has_deepgemm_fp8",
    "has_flashinfer_mxfp8",
    "has_flashinfer_trtllm_ll",
    "quantize_fp8_rowwise_triton",
    "quantize_fp8_tensorwise_triton",
    "quantize_weight_fp8_rowwise",
    "quantize_weight_fp8_tensorwise",
    "sm100_fp8_linear",
]
