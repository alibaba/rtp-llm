"""SiTU-GLU activation and fused block-32 UE8M0 FP8 quantization for Qwen 4.5-Flash.

Exact mathematical specification (computed in FP32, cast back to input dtype):
    g = gate_beta * tanh(gate / gate_beta) * sigmoid(gate)
    u = up_beta * tanh(up / up_beta)
    out = g * u

Default betas for Qwen 4.5-Flash: ``gate_beta = 4.0``, ``up_beta = 25.0``.

Also provides pure-Triton (zero ``sgl_kernel`` dependency) block-32 UE8M0
FP8 E4M3 quantization kernels:
  - ``per_token_group_quant_fp8_ue8m0`` (2D / 3D-masked; DeepGEMM / raw / FlashInfer layouts)
  - ``situ_glu_fp8_quant_packed`` (fused SiTU-GLU + block-32 UE8M0 FP8 quant in one pass)
  - ``situ_glu_fp8_quant_masked_packed`` (3D masked ``[E, max_m, 2*D]`` variant)
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

_FP8_E4M3_MAX = 448.0
_QUANT_EPS = 1.0e-10
_MASKED_PERSIST_MAX_E = 1024
_MASKED_CHUNK_M = 16
_NULL_CTX = contextlib.nullcontext()
_IS_MULTI_GPU: Optional[bool] = None
_SM_COUNT_CACHE: dict[int, int] = {}


def _cuda_device_ctx(device: torch.device):
    global _IS_MULTI_GPU
    if device.type == "cuda":
        multi = _IS_MULTI_GPU
        if multi is None:
            multi = torch.cuda.device_count() > 1
            _IS_MULTI_GPU = multi
        if multi and device.index is not None and device.index != torch.cuda.current_device():
            return torch.cuda.device(device)
    return _NULL_CTX


def _get_sm_count(device_index: int) -> int:
    cached = _SM_COUNT_CACHE.get(device_index)
    if cached is None:
        cached = int(torch.cuda.get_device_properties(device_index).multi_processor_count)
        _SM_COUNT_CACHE[device_index] = cached
    return cached


_FLASHINFER_MM_BF16 = None
_FLASHINFER_TGV_FN = None
_FLASHINFER_TGV_TACTIC_NUM = 0
_FLASHINFER_TGV_CHECKED = False


def get_sm10x_bf16_tgv_backend(device: torch.device) -> Tuple[Any, Any, int]:
    """Return ``(mm_fn, tgv_fn, tactic_num)`` when SM100+ TGV BF16 path is enabled."""
    global _FLASHINFER_MM_BF16, _FLASHINFER_TGV_FN, _FLASHINFER_TGV_TACTIC_NUM, _FLASHINFER_TGV_CHECKED
    if os.environ.get("QWEN4_5_BF16_TGV", "1") == "0":
        return None, None, 0
    if device.type != "cuda":
        return None, None, 0
    try:
        cap_major, _ = torch.cuda.get_device_capability(device)
        if cap_major < 10:
            return None, None, 0
    except Exception:
        return None, None, 0
    if not _FLASHINFER_TGV_CHECKED:
        _FLASHINFER_TGV_CHECKED = True
        try:
            import flashinfer

            _FLASHINFER_MM_BF16 = getattr(flashinfer, "mm_bf16", None)
        except Exception:
            _FLASHINFER_MM_BF16 = None
        if cap_major == 10:
            try:
                from flashinfer.gemm import gemm_base as _gb

                use_100f = bool(_gb.is_sm100f_supported(device))
                mod = _gb.gen_tgv_gemm_sm10x_module(torch.bfloat16, use_100f).build_and_load()
                _FLASHINFER_TGV_FN = mod.tgv_gemm
                _FLASHINFER_TGV_TACTIC_NUM = int(mod.tgv_gemm_tactic_num())
            except Exception:
                _FLASHINFER_TGV_FN = None
                _FLASHINFER_TGV_TACTIC_NUM = 0
    return _FLASHINFER_MM_BF16, _FLASHINFER_TGV_FN, _FLASHINFER_TGV_TACTIC_NUM


def select_sm10x_bf16_tgv_tactic(
    M: int, K: int, N: int, tactic_num: int = 11
) -> Optional[int]:
    """Select empirically optimal SM100+ TGV tactic (splitKreduce=0) or ``None`` if F.linear wins."""
    if M <= 0 or K < 320 or N < 1024 or (K % 16) != 0 or (N % 8) != 0:
        return None
    if N * K >= (1 << 30):
        return None
    if N > 8192:
        if M > 4:
            return None
        tactic = 3 if K >= 10240 else (2 if K >= 1280 else 1)
    elif M <= 8:
        tactic = 3 if K >= 10240 else (2 if K >= 1280 else 1)
    elif M <= 16:
        tactic = 6 if K >= 1280 else 5
    elif M <= 32:
        if N <= 2560 and M > 24:
            tactic = 5
        elif 1280 <= K < 2560:
            tactic = 7
        else:
            tactic = 8
    elif M <= 64 and N <= 2560:
        tactic = 8
    else:
        return None
    return tactic if tactic < tactic_num else None


def can_sm10x_bf16_tgv(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    tactic: Optional[int] = None,
    bias: Optional[torch.Tensor] = None,
) -> Optional[Tuple[Any, Any, Optional[int], int, int, int]]:
    """Return ``(mm_fn, tgv_fn, tactic, M, K, N)`` for zero-splitKreduce SM100+ TGV fast path."""
    mm_fn, tgv_fn, tactic_num = get_sm10x_bf16_tgv_backend(x.device)
    if (
        (tgv_fn is not None or mm_fn is not None)
        and x.is_cuda
        and w.is_cuda
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and w.is_contiguous()
        and not (
            torch.is_grad_enabled()
            and (x.requires_grad or w.requires_grad or (bias is not None and bias.requires_grad))
        )
    ):
        K = int(w.shape[1])
        N = int(w.shape[0])
        M = int(x.numel() // K) if K > 0 and x.shape[-1] == K else 0
        if M <= 0:
            return None
        if bias is not None and (
            not bias.is_cuda
            or bias.device != x.device
            or bias.dtype != torch.bfloat16
            or bias.ndim != 1
            or int(bias.shape[0]) != N
            or not bias.is_contiguous()
        ):
            return None
        if tgv_fn is not None:
            if tactic is not None:
                if (
                    0 <= int(tactic) < tactic_num
                    and (K % 16) == 0
                    and (N % 8) == 0
                    and (N * K) < (1 << 30)
                ):
                    return mm_fn, tgv_fn, int(tactic), M, K, N
                return None
            sel = select_sm10x_bf16_tgv_tactic(M, K, N, tactic_num)
            if sel is not None:
                return mm_fn, tgv_fn, sel, M, K, N
        elif (
            mm_fn is not None
            and tactic is None
            and 1 <= M <= 8
            and K >= 1280
            and N >= 1024
            and (K % 16) == 0
            and (N % 8) == 0
        ):
            return mm_fn, None, None, M, K, N
    return None


def run_sm10x_bf16_tgv_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    spec: Optional[Tuple[Any, Any, Optional[int], int, int, int]] = None,
    *,
    tactic: Optional[int] = None,
    bias: Optional[torch.Tensor] = None,
    pdl: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Execute explicit SM100+ TGV BF16 GEMM (or ``mm_bf16`` fallback) for ``spec``."""
    if spec is None:
        spec = can_sm10x_bf16_tgv(x, w, tactic=tactic, bias=bias)
        if spec is None:
            raise RuntimeError("SM100+ BF16 TGV path unavailable or invalid tensor/tactic spec.")
    mm_fn, tgv_fn, sel_tactic, M, K, N = spec
    x_2d = x.reshape(M, K)
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    with _cuda_device_ctx(x.device):
        if tgv_fn is not None and sel_tactic is not None:
            out_2d = (
                out.reshape(M, N)
                if (
                    out is not None
                    and out.is_cuda
                    and out.device == x.device
                    and out.dtype == torch.bfloat16
                    and out.is_contiguous()
                    and out.numel() == M * N
                )
                else torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
            )
            tgv_fn(w, x_2d.t(), bias, sel_tactic, out_2d, pdl)
        else:
            out_2d = (
                mm_fn(x_2d, w.t(), bias=bias, pdl=pdl, backend="tgv")
                if bias is not None
                else mm_fn(x_2d, w.t(), pdl=pdl, backend="tgv")
            )
            if out is not None:
                out.reshape(M, N).copy_(out_2d)
                out_2d = out.reshape(M, N)
    return out_2d.reshape(*x.shape[:-1], N) if x.ndim != 2 else out_2d


def bf16_linear_tgv_or_torch(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    tactic: Optional[int] = None,
    pdl: bool = True,
) -> torch.Tensor:
    """Execute ``x @ w.t() (+ bias)`` via SM100+ TGV when beneficial, else ``F.linear``."""
    spec = can_sm10x_bf16_tgv(x, w, tactic=tactic, bias=bias)
    if spec is not None:
        return run_sm10x_bf16_tgv_gemm(x, w, spec, bias=bias, pdl=pdl)
    return torch.nn.functional.linear(x, w, bias)


def situ_and_mul_ref(
    x: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    scale_gate: Optional[torch.Tensor] = None,
    scale_gate_col: int = 0,
) -> torch.Tensor:
    """Pure PyTorch FP32 reference for SiTU-GLU.

    Accepts either:
      - concatenated ``x`` of shape ``[..., 2 * D]`` (with ``up=None``), or
      - separate ``x`` (gate) and ``up`` each of shape ``[..., D]``.
    """
    if up is None:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Expected even last dimension for concatenated gate_up, got {x.shape[-1]}"
            )
        gate, up_t = x.chunk(2, dim=-1)
    else:
        if x.shape != up.shape:
            raise ValueError(f"gate shape {x.shape} != up shape {up.shape}")
        gate, up_t = x, up

    out_dtype = gate.dtype
    gate_f = gate.float()
    up_f = up_t.float()
    g = gate_beta * torch.tanh(gate_f / gate_beta) * torch.sigmoid(gate_f)
    u = up_beta * torch.tanh(up_f / up_beta)
    res = (g * u).to(out_dtype)
    if scale_gate is not None and res.numel() > 0:
        sg = scale_gate[:, scale_gate_col : scale_gate_col + 1] if scale_gate.ndim >= 2 else scale_gate.unsqueeze(-1)
        sg = sg.reshape(*res.shape[:-1], 1).to(res.dtype)
        sig = torch.sigmoid(sg.float()).to(res.dtype)
        res = (res.float() * sig.float()).to(out_dtype)
    return res


@triton.jit
def _situ_and_mul_cat_kernel(
    x_ptr,
    out_ptr,
    scale_gate_ptr,
    D: tl.constexpr,
    stride_x_row: tl.int64,
    stride_out_row: tl.int64,
    stride_sg_row: tl.int64,
    scale_gate_col: tl.constexpr,
    HAS_SCALE_GATE: tl.constexpr,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    inv_gate_beta: tl.constexpr,
    inv_up_beta: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_col = tl.program_id(1)
    offs = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D

    x_row_ptr = x_ptr + pid_row * stride_x_row
    gate = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row_ptr + D + offs, mask=mask, other=0.0).to(tl.float32)

    sig_gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    g = gate_beta * libdevice.tanh(gate * inv_gate_beta) * sig_gate
    u = up_beta * libdevice.tanh(up * inv_up_beta)
    out_val = (g * u).to(out_ptr.dtype.element_ty)
    if HAS_SCALE_GATE:
        sg_raw = tl.load(scale_gate_ptr + pid_row * stride_sg_row + scale_gate_col).to(tl.float32)
        sig_sg = tl.div_rn(1.0, 1.0 + libdevice.exp(-sg_raw)).to(out_ptr.dtype.element_ty)
        out_val = (out_val.to(tl.float32) * sig_sg.to(tl.float32)).to(out_ptr.dtype.element_ty)

    out_row_ptr = out_ptr + pid_row * stride_out_row
    tl.store(out_row_ptr + offs, out_val, mask=mask)


@triton.jit
def _situ_and_mul_split_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    D: tl.constexpr,
    stride_g_row: tl.int64,
    stride_u_row: tl.int64,
    stride_out_row: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    inv_gate_beta: tl.constexpr,
    inv_up_beta: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_row = tl.program_id(0).to(tl.int64)
    pid_col = tl.program_id(1)
    offs = pid_col * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D

    gate = tl.load(gate_ptr + pid_row * stride_g_row + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(up_ptr + pid_row * stride_u_row + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    sig_gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    g = gate_beta * libdevice.tanh(gate * inv_gate_beta) * sig_gate
    u = up_beta * libdevice.tanh(up * inv_up_beta)
    out = g * u

    tl.store(
        out_ptr + pid_row * stride_out_row + offs,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def situ_and_mul_triton(
    x: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    scale_gate: Optional[torch.Tensor] = None,
    scale_gate_col: int = 0,
) -> torch.Tensor:
    """Fused Triton kernel for SiTU-GLU activation (with optional fused sigmoid gate).

    Automatically falls back to :func:`situ_and_mul_ref` for non-CUDA tensors.
    """
    if not x.is_cuda or (up is not None and not up.is_cuda):
        return situ_and_mul_ref(
            x,
            up,
            gate_beta=gate_beta,
            up_beta=up_beta,
            scale_gate=scale_gate,
            scale_gate_col=scale_gate_col,
        )

    gate_beta_f = float(gate_beta)
    up_beta_f = float(up_beta)
    inv_gate_beta_f = 1.0 / gate_beta_f
    inv_up_beta_f = 1.0 / up_beta_f

    if up is None:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Expected even last dimension for concatenated gate_up, got {x.shape[-1]}"
            )
        orig_shape = x.shape[:-1]
        D = x.shape[-1] // 2
        if x.numel() == 0:
            return torch.empty((*orig_shape, D), dtype=x.dtype, device=x.device)
        x_2d = x.reshape(-1, 2 * D)
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        num_rows = x_2d.shape[0]
        out = torch.empty((num_rows, D), dtype=x.dtype, device=x.device)
        has_sg = scale_gate is not None
        if has_sg:
            assert scale_gate is not None
            sg_2d = scale_gate.reshape(num_rows, -1) if scale_gate.ndim >= 2 else scale_gate.reshape(num_rows, 1)
            if sg_2d.stride(-1) != 1:
                sg_2d = sg_2d.contiguous()
            sg_ptr = sg_2d
            sg_stride = int(sg_2d.stride(0))
            sg_col = int(scale_gate_col)
        else:
            sg_ptr = x_2d
            sg_stride = 0
            sg_col = 0

        # BLOCK_D=512 with 4 warps maximizes SM100/SM103 occupancy across D=320..12800.
        BLOCK_D = min(512, triton.next_power_of_2(max(D, 64)))
        grid = (num_rows, triton.cdiv(D, BLOCK_D))
        with _cuda_device_ctx(x_2d.device):
            _situ_and_mul_cat_kernel[grid](
                x_2d,
                out,
                sg_ptr,
                D,
                x_2d.stride(0),
                out.stride(0),
                sg_stride,
                scale_gate_col=sg_col,
                HAS_SCALE_GATE=has_sg,
                gate_beta=gate_beta_f,
                up_beta=up_beta_f,
                inv_gate_beta=inv_gate_beta_f,
                inv_up_beta=inv_up_beta_f,
                BLOCK_D=BLOCK_D,
                num_warps=4,
            )
        return out.reshape(*orig_shape, D)

    if x.shape != up.shape:
        raise ValueError(f"gate shape {x.shape} != up shape {up.shape}")
    orig_shape = x.shape[:-1]
    D = x.shape[-1]
    if x.numel() == 0:
        return torch.empty_like(x)
    g_2d = x.reshape(-1, D)
    u_2d = up.reshape(-1, D)
    if g_2d.stride(-1) != 1:
        g_2d = g_2d.contiguous()
    if u_2d.stride(-1) != 1:
        u_2d = u_2d.contiguous()
    num_rows = g_2d.shape[0]
    out = torch.empty((num_rows, D), dtype=x.dtype, device=x.device)

    BLOCK_D = min(512, triton.next_power_of_2(max(D, 64)))
    grid = (num_rows, triton.cdiv(D, BLOCK_D))
    with _cuda_device_ctx(g_2d.device):
        _situ_and_mul_split_kernel[grid](
            g_2d,
            u_2d,
            out,
            D,
            g_2d.stride(0),
            u_2d.stride(0),
            out.stride(0),
            gate_beta=gate_beta_f,
            up_beta=up_beta_f,
            inv_gate_beta=inv_gate_beta_f,
            inv_up_beta=inv_up_beta_f,
            BLOCK_D=BLOCK_D,
            num_warps=4,
        )
    res = out.reshape(*orig_shape, D)
    if scale_gate is not None and res.numel() > 0:
        sg = scale_gate[:, scale_gate_col : scale_gate_col + 1] if scale_gate.ndim >= 2 else scale_gate.unsqueeze(-1)
        sg = sg.reshape(*res.shape[:-1], 1).to(res.dtype)
        sig = torch.sigmoid(sg.float()).to(res.dtype)
        res = (res.float() * sig.float()).to(res.dtype)
    return res


class SituAndMul(nn.Module):
    """SiTU-GLU activation module for Qwen 4.5-Flash Dense MLP & Latent MoE."""

    def __init__(
        self,
        gate_beta: float = 4.0,
        up_beta: float = 25.0,
        use_triton: bool = True,
    ) -> None:
        super().__init__()
        self.gate_beta = float(gate_beta)
        self.up_beta = float(up_beta)
        self.use_triton = bool(use_triton)

    def forward(
        self,
        x: torch.Tensor,
        up: Optional[torch.Tensor] = None,
        *,
        use_triton: Optional[bool] = None,
        scale_gate: Optional[torch.Tensor] = None,
        scale_gate_col: int = 0,
    ) -> torch.Tensor:
        eff_triton = self.use_triton if use_triton is None else bool(use_triton)
        if eff_triton and x.is_cuda and (up is None or up.is_cuda):
            return situ_and_mul_triton(
                x,
                up,
                gate_beta=self.gate_beta,
                up_beta=self.up_beta,
                scale_gate=scale_gate,
                scale_gate_col=scale_gate_col,
            )
        return situ_and_mul_ref(
            x,
            up,
            gate_beta=self.gate_beta,
            up_beta=self.up_beta,
            scale_gate=scale_gate,
            scale_gate_col=scale_gate_col,
        )


# ---------------------------------------------------------------------------
# Pure-Triton Block-32 UE8M0 FP8 Quantization + Fused SiTU-GLU FP8 Quant
# ---------------------------------------------------------------------------


def _pack_ue8m0_scales_deepgemm_ref(scale_u8: torch.Tensor) -> torch.Tensor:
    """Reference packer: ``[..., M, K // 32] uint8`` -> MN-major TMA-aligned ``int32``."""
    prefix = scale_u8.shape[:-2]
    M, num_groups = scale_u8.shape[-2], scale_u8.shape[-1]
    num_packs = (num_groups + 3) // 4
    padded_groups = num_packs * 4
    if padded_groups != num_groups:
        pad = torch.zeros(
            (*prefix, M, padded_groups - num_groups),
            dtype=torch.uint8,
            device=scale_u8.device,
        )
        sc = torch.cat([scale_u8, pad], dim=-1)
    else:
        sc = scale_u8
    sc_u32 = sc.reshape(*prefix, M, num_packs, 4).to(torch.int64)
    packed = (
        sc_u32[..., 0]
        | (sc_u32[..., 1] << 8)
        | (sc_u32[..., 2] << 16)
        | (sc_u32[..., 3] << 24)
    ).to(torch.int32)
    tma_m = ((M + 3) // 4) * 4
    out_buf = torch.zeros(
        (*prefix, num_packs, tma_m), dtype=torch.int32, device=scale_u8.device
    )
    out_buf[..., :M] = packed.transpose(-2, -1)
    return out_buf.transpose(-2, -1)[..., :M, :]


def _swizzle_128x4_ref(
    scale_u8: torch.Tensor, m_indptr: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pure-PyTorch CUTLASS 128x4 swizzle matching ``flashinfer.quantization.block_scale_interleave``."""
    M, C = scale_u8.shape
    C_pad = ((C + 3) // 4) * 4
    if m_indptr is None:
        R_pad = ((M + 127) // 128) * 128
        padded = torch.zeros((R_pad, C_pad), dtype=torch.uint8, device=scale_u8.device)
        padded[:M, :C] = scale_u8
        return (
            padded.reshape(R_pad // 128, 4, 32, C_pad // 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(-1)
        )

    G = int(m_indptr.numel()) - 1
    offsets = m_indptr.tolist()
    total_rows = ((M + G * 127) // 128) * 128 if G > 0 else 0
    out = torch.zeros((total_rows * C_pad,), dtype=torch.uint8, device=scale_u8.device)
    for g in range(G):
        m0, m1 = offsets[g], offsets[g + 1]
        mg = m1 - m0
        if mg <= 0:
            continue
        sf_m0 = ((m0 + g * 127) // 128) * 128
        rg_pad = ((mg + 127) // 128) * 128
        tmp = torch.zeros((rg_pad, C_pad), dtype=torch.uint8, device=scale_u8.device)
        tmp[:mg, :C] = scale_u8[m0:m1]
        sw = (
            tmp.reshape(rg_pad // 128, 4, 32, C_pad // 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(-1)
        )
        out[sf_m0 * C_pad : (sf_m0 + rg_pad) * C_pad] = sw
    return out


def _quant_block32_ue8m0_ref(
    x: torch.Tensor,
    *,
    group_size: int = 32,
    scale_format: str = "deepgemm",
    masked_m: Optional[torch.Tensor] = None,
    m_indptr: Optional[torch.Tensor] = None,
    eps: float = _QUANT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for block-32 UE8M0 FP8 quantization."""
    if group_size != 32:
        raise ValueError(f"Expected group_size=32, got {group_size}")
    if x.shape[-1] % 32 != 0:
        raise ValueError(f"Last dim {x.shape[-1]} must be divisible by 32")

    orig_shape = x.shape
    K = orig_shape[-1]
    num_groups = K // 32
    x_blk = x.float().reshape(*orig_shape[:-1], num_groups, 32)
    amax = x_blk.abs().amax(dim=-1)
    raw = torch.clamp(amax / _FP8_E4M3_MAX, min=eps)
    # Ceil log2 via IEEE-754 bits matching Triton kernel
    bits = raw.view(torch.int32)
    biased = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    biased = biased.clamp(1, 254)
    inv_scale = ((254 - biased) << 23).to(torch.int32).view(torch.float32)
    q = (x_blk * inv_scale.unsqueeze(-1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    q = q.reshape(orig_shape)
    sc_u8 = biased.to(torch.uint8)

    if scale_format == "raw":
        return q, sc_u8
    if scale_format == "deepgemm":
        return q, _pack_ue8m0_scales_deepgemm_ref(sc_u8)
    if scale_format == "flashinfer":
        if sc_u8.ndim != 2:
            raise ValueError(
                f"flashinfer scale_format requires 2D input, got {orig_shape}"
            )
        return q, _swizzle_128x4_ref(sc_u8, m_indptr=m_indptr)
    raise ValueError(f"Unsupported scale_format={scale_format!r}")


def per_token_group_quant_fp8_ue8m0_ref(
    x: torch.Tensor,
    *,
    group_size: int = 32,
    scale_format: str = "deepgemm",
    masked_m: Optional[torch.Tensor] = None,
    m_indptr: Optional[torch.Tensor] = None,
    eps: float = _QUANT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for per-token group-32 UE8M0 FP8 E4M3 quantization."""
    return _quant_block32_ue8m0_ref(
        x,
        group_size=group_size,
        scale_format=scale_format,
        masked_m=masked_m,
        m_indptr=m_indptr,
        eps=eps,
    )


def situ_glu_fp8_quant_ref(
    x: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    group_size: int = 32,
    scale_format: str = "deepgemm",
    m_indptr: Optional[torch.Tensor] = None,
    eps: float = _QUANT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for SiTU-GLU followed by block-32 UE8M0 FP8 quantization."""
    y = situ_and_mul_ref(x, up, gate_beta=gate_beta, up_beta=up_beta)
    return _quant_block32_ue8m0_ref(
        y,
        group_size=group_size,
        scale_format=scale_format,
        m_indptr=m_indptr,
        eps=eps,
    )


@triton.jit
def _compute_ue8m0_fp8_block32(y_g, eps: tl.constexpr):
    """Quantize ``[GROUPS, 32]`` FP32 tensor into FP8 E4M3FN + biased UE8M0 int32."""
    amax = tl.max(tl.abs(y_g), axis=1)
    raw = tl.maximum(amax * (1.0 / 448.0), eps)
    bits = raw.to(tl.int32, bitcast=True)
    biased = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    biased = tl.maximum(tl.minimum(biased, 254), 1)
    inv_scale = ((254 - biased) << 23).to(tl.float32, bitcast=True)
    q = tl.clamp(y_g * inv_scale[:, None], -448.0, 448.0).to(tl.float8e4nv)
    return q, biased


@triton.jit
def _situ_glu_fp32_to_bf16(gate, up, gate_beta: tl.constexpr, up_beta: tl.constexpr):
    inv_gb: tl.constexpr = 1.0 / gate_beta
    inv_ub: tl.constexpr = 1.0 / up_beta
    sig_gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    g = gate_beta * libdevice.tanh(gate * inv_gb) * sig_gate
    u = up_beta * libdevice.tanh(up * inv_ub)
    return (g * u).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _store_deepgemm_row_packed(
    q_row_base,
    sc_ebase,
    m: tl.int64,
    q,
    biased,
    offs_d,
    mask_d,
    g_idx,
    num_groups,
    pack_offs,
    num_packs,
    stride_sck: tl.int64,
    BLOCK_D: tl.constexpr,
    GROUPS: tl.constexpr,
    PACKS: tl.constexpr,
):
    tl.store(q_row_base + offs_d, q.reshape((BLOCK_D,)), mask=mask_d)
    biased_u32 = tl.where(g_idx < num_groups, biased, 0).to(tl.uint8).to(tl.uint32)
    shifts = (tl.arange(0, 4)[None, :] * 8).to(tl.uint32)
    packed = tl.sum(biased_u32.reshape((PACKS, 4)) << shifts, axis=1).to(tl.int32)
    tl.store(
        sc_ebase + pack_offs.to(tl.int64) * stride_sck + m,
        packed,
        mask=pack_offs < num_packs,
    )


@triton.jit
def _masked_prefix_map(masked_m_ptr, max_m, E, E_PAD: tl.constexpr):
    offs_e = tl.arange(0, E_PAD)
    raw = tl.load(masked_m_ptr + offs_e, mask=offs_e < E, other=0)
    counts = tl.where(offs_e < E, tl.maximum(tl.minimum(raw, max_m), 0), 0)
    ends = tl.cumsum(counts, axis=0)
    total_rows = tl.sum(counts, axis=0)
    return offs_e, ends, total_rows


@triton.jit
def _store_flashinfer_128x4_group(
    out_sc_ptr,
    m_indptr_ptr,
    row_idx: tl.int64,
    g_idx,
    biased,
    num_groups,
    sf_cols_pad,
    num_ep_groups: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
    HAS_INDPTR: tl.constexpr,
    GROUPS: tl.constexpr,
):
    if HAS_INDPTR:
        lo = 0
        hi = num_ep_groups - 1
        for _ in tl.static_range(BSEARCH_ITERS):
            if lo < hi:
                mid = (lo + hi + 1) // 2
                m_mid = tl.load(m_indptr_ptr + mid).to(tl.int64)
                if m_mid <= row_idx:
                    lo = mid
                else:
                    hi = mid - 1
        m_off = tl.load(m_indptr_ptr + lo).to(tl.int64)
        r_local = row_idx - m_off
        sf_m_off = ((m_off + lo * 127) // 128) * 128
    else:
        r_local = row_idx
        sf_m_off = tl.cast(0, tl.int64)

    r_blk = r_local // 128
    r_rem = r_local % 128
    r4 = r_rem // 32
    r32 = r_rem % 32
    c_blk = g_idx // 4
    c4 = g_idx % 4
    sw_off = (
        (sf_m_off + r_blk * 128) * sf_cols_pad
        + ((c_blk * 32 + r32) * 4 + r4) * 4
        + c4
    )
    tl.store(out_sc_ptr + sw_off, biased.to(tl.uint8), mask=g_idx < num_groups)


@triton.jit(do_not_specialize=["M", "stride_x_row", "stride_q_row", "stride_sc_k", "stride_sc_m"])
def _per_token_group_quant_fp8_ue8m0_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    m_indptr_ptr,
    M,
    K,
    num_groups,
    num_packs,
    sf_cols_pad,
    stride_x_row: tl.int64,
    stride_q_row: tl.int64,
    stride_sc_k: tl.int64,
    stride_sc_m: tl.int64,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_FMT: tl.constexpr,  # 0=deepgemm, 1=raw, 2=flashinfer
    HAS_INDPTR: tl.constexpr,
    NUM_EP_GROUPS: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1)
    if pid_m >= M:
        return

    GROUPS: tl.constexpr = BLOCK_K // 32
    PACKS: tl.constexpr = GROUPS // 4

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    x = tl.load(x_ptr + pid_m * stride_x_row + offs_k, mask=mask_k, other=0.0).to(
        tl.float32
    )
    x = tl.where(mask_k, x, 0.0).reshape((GROUPS, 32))
    q, biased = _compute_ue8m0_fp8_block32(x, eps=eps)
    tl.store(
        q_ptr + pid_m * stride_q_row + offs_k,
        q.reshape((BLOCK_K,)),
        mask=mask_k,
    )

    g_idx = pid_k * GROUPS + tl.arange(0, GROUPS)
    if SCALE_FMT == 0:
        biased_u32 = tl.where(g_idx < num_groups, biased, 0).to(tl.uint8).to(tl.uint32)
        shifts = (tl.arange(0, 4)[None, :] * 8).to(tl.uint32)
        packed = tl.sum(biased_u32.reshape((PACKS, 4)) << shifts, axis=1).to(tl.int32)
        pack_offs = pid_k * PACKS + tl.arange(0, PACKS)
        tl.store(
            sc_ptr + pack_offs.to(tl.int64) * stride_sc_k + pid_m,
            packed,
            mask=pack_offs < num_packs,
        )
    elif SCALE_FMT == 1:
        tl.store(
            sc_ptr + pid_m * stride_sc_m + g_idx,
            biased.to(tl.uint8),
            mask=g_idx < num_groups,
        )
    else:
        _store_flashinfer_128x4_group(
            sc_ptr,
            m_indptr_ptr,
            pid_m,
            g_idx,
            biased,
            num_groups,
            sf_cols_pad,
            NUM_EP_GROUPS,
            BSEARCH_ITERS,
            HAS_INDPTR,
            GROUPS,
        )


@triton.jit(do_not_specialize=["max_m", "stride_xe", "stride_xm", "stride_qe", "stride_qm", "stride_sce", "stride_sck"])
def _per_token_group_quant_fp8_ue8m0_masked_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    masked_m_ptr,
    max_m,
    K,
    num_groups,
    num_packs,
    stride_xe: tl.int64,
    stride_xm: tl.int64,
    stride_qe: tl.int64,
    stride_qm: tl.int64,
    stride_sce: tl.int64,
    stride_sck: tl.int64,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_M: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    pid_mc = tl.program_id(1).to(tl.int64)
    pid_k = tl.program_id(2)
    m_cnt = tl.minimum(tl.load(masked_m_ptr + e).to(tl.int64), max_m)
    m_start = pid_mc * CHUNK_M
    if m_start >= m_cnt:
        return
    m_end = tl.minimum(m_start + CHUNK_M, m_cnt)

    GROUPS: tl.constexpr = BLOCK_K // 32
    PACKS: tl.constexpr = GROUPS // 4
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    g_idx = pid_k * GROUPS + tl.arange(0, GROUPS)
    pack_offs = pid_k * PACKS + tl.arange(0, PACKS)
    x_ebase = x_ptr + e * stride_xe
    q_ebase = q_ptr + e * stride_qe
    sc_ebase = sc_ptr + e * stride_sce

    for m in range(m_start, m_end):
        x = tl.load(x_ebase + m * stride_xm + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        x = tl.where(mask_k, x, 0.0).reshape((GROUPS, 32))
        q, biased = _compute_ue8m0_fp8_block32(x, eps=eps)
        _store_deepgemm_row_packed(
            q_ebase + m * stride_qm, sc_ebase, m,
            q, biased, offs_k, mask_k, g_idx, num_groups, pack_offs, num_packs,
            stride_sck, BLOCK_K, GROUPS, PACKS,
        )


@triton.jit(do_not_specialize=["max_m", "stride_xe", "stride_xm", "stride_qe", "stride_qm", "stride_sce", "stride_sck"])
def _per_token_group_quant_fp8_ue8m0_masked_persistent_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    masked_m_ptr,
    max_m,
    E,
    K,
    num_groups,
    num_packs,
    stride_xe: tl.int64,
    stride_xm: tl.int64,
    stride_qe: tl.int64,
    stride_qm: tl.int64,
    stride_sce: tl.int64,
    stride_sck: tl.int64,
    eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
    D_TILES: tl.constexpr,
    E_PAD: tl.constexpr,
):
    offs_e, ends, total_rows = _masked_prefix_map(masked_m_ptr, max_m, E, E_PAD)
    if total_rows == 0:
        return
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    total_tasks = total_rows * D_TILES

    GROUPS: tl.constexpr = BLOCK_K // 32
    PACKS: tl.constexpr = GROUPS // 4

    for task_id in range(pid, total_tasks, num_ctas):
        row_id = task_id // D_TILES
        pid_k = task_id - row_id * D_TILES
        e = tl.sum((ends <= row_id).to(tl.int32), axis=0).to(tl.int64)
        m_start_e = tl.sum(tl.where(offs_e == (e - 1), ends, 0), axis=0).to(tl.int64)
        m = row_id.to(tl.int64) - m_start_e

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(x_ptr + e * stride_xe + m * stride_xm + offs_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        x = tl.where(mask_k, x, 0.0).reshape((GROUPS, 32))
        q, biased = _compute_ue8m0_fp8_block32(x, eps=eps)
        g_idx = pid_k * GROUPS + tl.arange(0, GROUPS)
        pack_offs = pid_k * PACKS + tl.arange(0, PACKS)
        _store_deepgemm_row_packed(
            q_ptr + e * stride_qe + m * stride_qm, sc_ptr + e * stride_sce, m,
            q, biased, offs_k, mask_k, g_idx, num_groups, pack_offs, num_packs,
            stride_sck, BLOCK_K, GROUPS, PACKS,
        )


@triton.jit(do_not_specialize=["M", "stride_g_row", "stride_u_row", "stride_q_row", "stride_sc_k", "stride_sc_m"])
def _situ_glu_fp8_quant_packed_kernel(
    gate_ptr,
    up_ptr,
    q_ptr,
    sc_ptr,
    m_indptr_ptr,
    M,
    D,
    num_groups,
    num_packs,
    sf_cols_pad,
    stride_g_row: tl.int64,
    stride_u_row: tl.int64,
    stride_q_row: tl.int64,
    stride_sc_k: tl.int64,
    stride_sc_m: tl.int64,
    up_offset: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    active_rows_ptr,
    SCALE_FMT: tl.constexpr,
    HAS_INDPTR: tl.constexpr,
    HAS_ACTIVE_ROWS: tl.constexpr,
    NUM_EP_GROUPS: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
    if HAS_ACTIVE_ROWS:
        if pid_m >= tl.load(active_rows_ptr).to(tl.int64):
            return
    if pid_m >= M:
        return

    GROUPS: tl.constexpr = BLOCK_D // 32
    PACKS: tl.constexpr = GROUPS // 4

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    gate = tl.load(gate_ptr + pid_m * stride_g_row + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        up_ptr + pid_m * stride_u_row + up_offset + offs_d, mask=mask_d, other=0.0
    ).to(tl.float32)

    y = _situ_glu_fp32_to_bf16(gate, up, gate_beta=gate_beta, up_beta=up_beta)
    y = tl.where(mask_d, y, 0.0).reshape((GROUPS, 32))

    q, biased = _compute_ue8m0_fp8_block32(y, eps=eps)
    tl.store(
        q_ptr + pid_m * stride_q_row + offs_d,
        q.reshape((BLOCK_D,)),
        mask=mask_d,
    )

    g_idx = pid_d * GROUPS + tl.arange(0, GROUPS)
    if SCALE_FMT == 0:
        biased_u32 = tl.where(g_idx < num_groups, biased, 0).to(tl.uint8).to(tl.uint32)
        shifts = (tl.arange(0, 4)[None, :] * 8).to(tl.uint32)
        packed = tl.sum(biased_u32.reshape((PACKS, 4)) << shifts, axis=1).to(tl.int32)
        pack_offs = pid_d * PACKS + tl.arange(0, PACKS)
        tl.store(
            sc_ptr + pack_offs.to(tl.int64) * stride_sc_k + pid_m,
            packed,
            mask=pack_offs < num_packs,
        )
    elif SCALE_FMT == 1:
        tl.store(
            sc_ptr + pid_m * stride_sc_m + g_idx,
            biased.to(tl.uint8),
            mask=g_idx < num_groups,
        )
    else:
        _store_flashinfer_128x4_group(
            sc_ptr,
            m_indptr_ptr,
            pid_m,
            g_idx,
            biased,
            num_groups,
            sf_cols_pad,
            NUM_EP_GROUPS,
            BSEARCH_ITERS,
            HAS_INDPTR,
            GROUPS,
        )


@triton.jit(do_not_specialize=["M"])
def _situ_glu_fp8_quant_packed_1d_kernel(
    gate_ptr,
    up_ptr,
    q_ptr,
    sc_ptr,
    m_indptr_ptr,
    m_indices_ptr,
    offsets_ptr,
    counts_ptr,
    M,
    D: tl.constexpr,
    stride_g_row: tl.int64,
    stride_u_row: tl.int64,
    stride_q_row: tl.int64,
    stride_sc_k: tl.int64,
    stride_sc_m: tl.int64,
    up_offset: tl.int64,
    sf_cols_pad: tl.constexpr,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    active_rows_ptr,
    SCALE_FMT: tl.constexpr,
    HAS_INDPTR: tl.constexpr,
    HAS_ACTIVE_ROWS: tl.constexpr,
    HAS_ROW_BOUNDS: tl.constexpr,
    NUM_EP_GROUPS: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
):
    """Row-per-CTA SiTU-GLU + block-32 UE8M0 FP8 quant (exact D % 128 == 0 fast path)."""
    row = tl.program_id(0).to(tl.int64)
    if HAS_ACTIVE_ROWS:
        if row >= tl.load(active_rows_ptr).to(tl.int64):
            return
    if row >= M:
        return
    if HAS_ROW_BOUNDS:
        e = tl.load(m_indices_ptr + row)
        if e < 0:
            return
        if row >= tl.load(offsets_ptr + e).to(tl.int64) + tl.load(counts_ptr + e).to(tl.int64):
            return
    TILES: tl.constexpr = D // BLOCK_D
    GROUPS: tl.constexpr = BLOCK_D // 32
    PACKS: tl.constexpr = GROUPS // 4

    sw_row_base_i32 = tl.cast(0, tl.int64)
    if SCALE_FMT == 2:
        if HAS_INDPTR:
            lo = 0
            hi = NUM_EP_GROUPS - 1
            for _ in tl.static_range(BSEARCH_ITERS):
                if lo < hi:
                    mid = (lo + hi + 1) // 2
                    if tl.load(m_indptr_ptr + mid).to(tl.int64) <= row:
                        lo = mid
                    else:
                        hi = mid - 1
            m_off = tl.load(m_indptr_ptr + lo).to(tl.int64)
            r_local = row - m_off
            sf_m_off = ((m_off + lo * 127) // 128) * 128
        else:
            r_local = row
            sf_m_off = tl.cast(0, tl.int64)
        r_blk = r_local // 128
        r_rem = r_local % 128
        r4 = r_rem // 32
        r32 = r_rem % 32
        sw_row_base_i32 = ((sf_m_off + r_blk * 128) * sf_cols_pad + r32 * 16 + r4 * 4) // 4

    g_row = gate_ptr + row * stride_g_row
    u_row = up_ptr + row * stride_u_row + up_offset
    q_row = q_ptr + row * stride_q_row
    sc_i32_ptr = tl.cast(sc_ptr, tl.pointer_type(tl.int32))
    shifts = (tl.arange(0, 4)[None, :] * 8).to(tl.uint32)

    for t in tl.static_range(TILES):
        offs = t * BLOCK_D + tl.arange(0, BLOCK_D)
        gate = tl.load(g_row + offs).to(tl.float32)
        up = tl.load(u_row + offs).to(tl.float32)
        y = _situ_glu_fp32_to_bf16(gate, up, gate_beta=gate_beta, up_beta=up_beta)
        q, biased = _compute_ue8m0_fp8_block32(y.reshape((GROUPS, 32)), eps=eps)
        tl.store(q_row + offs, q.reshape((BLOCK_D,)))

        if SCALE_FMT == 0:
            packed = tl.sum(biased.to(tl.uint8).to(tl.uint32).reshape((PACKS, 4)) << shifts, axis=1).to(tl.int32)
            pack_offs = t * PACKS + tl.arange(0, PACKS)
            tl.store(sc_ptr + pack_offs.to(tl.int64) * stride_sc_k + row, packed)
        elif SCALE_FMT == 1:
            g_offs = t * GROUPS + tl.arange(0, GROUPS)
            tl.store(sc_ptr + row * stride_sc_m + g_offs, biased.to(tl.uint8))
        else:
            packed = tl.sum(biased.to(tl.uint8).to(tl.uint32).reshape((PACKS, 4)) << shifts, axis=1).to(tl.int32)
            c_blk = t * PACKS + tl.arange(0, PACKS)
            tl.store(sc_i32_ptr + sw_row_base_i32 + c_blk * 128, packed)


@triton.jit(do_not_specialize=["max_m", "stride_xe", "stride_xm", "stride_qe", "stride_qm", "stride_sce", "stride_sck"])
def _situ_glu_fp8_quant_masked_packed_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    masked_m_ptr,
    max_m,
    D,
    num_groups,
    num_packs,
    stride_xe: tl.int64,
    stride_xm: tl.int64,
    stride_qe: tl.int64,
    stride_qm: tl.int64,
    stride_sce: tl.int64,
    stride_sck: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CHUNK_M: tl.constexpr,
):
    e = tl.program_id(0).to(tl.int64)
    pid_mc = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2)
    m_cnt = tl.minimum(tl.load(masked_m_ptr + e).to(tl.int64), max_m)
    m_start = pid_mc * CHUNK_M
    if m_start >= m_cnt:
        return
    m_end = tl.minimum(m_start + CHUNK_M, m_cnt)

    GROUPS: tl.constexpr = BLOCK_D // 32
    PACKS: tl.constexpr = GROUPS // 4
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    g_idx = pid_d * GROUPS + tl.arange(0, GROUPS)
    pack_offs = pid_d * PACKS + tl.arange(0, PACKS)
    x_ebase = x_ptr + e * stride_xe
    q_ebase = q_ptr + e * stride_qe
    sc_ebase = sc_ptr + e * stride_sce

    for m in range(m_start, m_end):
        row_base = x_ebase + m * stride_xm
        gate = tl.load(row_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        up = tl.load(row_base + D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        y = _situ_glu_fp32_to_bf16(gate, up, gate_beta=gate_beta, up_beta=up_beta)
        y = tl.where(mask_d, y, 0.0).reshape((GROUPS, 32))
        q, biased = _compute_ue8m0_fp8_block32(y, eps=eps)
        _store_deepgemm_row_packed(
            q_ebase + m * stride_qm, sc_ebase, m,
            q, biased, offs_d, mask_d, g_idx, num_groups, pack_offs, num_packs,
            stride_sck, BLOCK_D, GROUPS, PACKS,
        )


@triton.jit
def _resolve_masked_row(offs_e, ends, row_id):
    """Map flattened active row index ``row_id`` to ``(expert_idx, local_row)``."""
    e = tl.sum((ends <= row_id).to(tl.int32), axis=0).to(tl.int64)
    m_start_e = tl.sum(tl.where(offs_e == (e - 1), ends, 0), axis=0).to(tl.int64)
    m = row_id.to(tl.int64) - m_start_e
    return e, m


@triton.jit
def _situ_glu_quant_masked_tile(
    x_base,
    q_base,
    sc_base,
    t,
    D: tl.constexpr,
    stride_sck: tl.constexpr,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    GROUPS: tl.constexpr,
    PACKS: tl.constexpr,
    shifts,
):
    """Compute one ``BLOCK_D``-wide SiTU-GLU + block-32 UE8M0 FP8 quant tile."""
    offs_d = t * BLOCK_D + tl.arange(0, BLOCK_D)
    gate = tl.load(x_base + offs_d).to(tl.float32)
    up = tl.load(x_base + D + offs_d).to(tl.float32)
    y = _situ_glu_fp32_to_bf16(gate, up, gate_beta=gate_beta, up_beta=up_beta)
    q, biased = _compute_ue8m0_fp8_block32(y.reshape((GROUPS, 32)), eps=eps)
    tl.store(q_base + offs_d, q.reshape((BLOCK_D,)))
    packed = tl.sum(
        biased.to(tl.uint8).to(tl.uint32).reshape((PACKS, 4)) << shifts,
        axis=1,
    ).to(tl.int32)
    pack_offs = t * PACKS + tl.arange(0, PACKS)
    tl.store(sc_base + pack_offs.to(tl.int64) * stride_sck, packed)


@triton.jit(
    do_not_specialize=[
        "max_m",
        "stride_xe",
        "stride_xm",
        "stride_qe",
        "stride_qm",
        "stride_sce",
        "stride_sck",
    ]
)
def _situ_glu_fp8_quant_masked_row_tile_persistent_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    masked_m_ptr,
    max_m,
    E,
    D: tl.constexpr,
    stride_xe: tl.int64,
    stride_xm: tl.int64,
    stride_qe: tl.int64,
    stride_qm: tl.int64,
    stride_sce: tl.int64,
    stride_sck: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    E_PAD: tl.constexpr,
):
    """Row-and-tile persistent masked SiTU-GLU + FP8 quant balancing hot skewed experts across SMs."""
    offs_e, ends, total_rows = _masked_prefix_map(masked_m_ptr, max_m, E, E_PAD)
    if total_rows == 0:
        return
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    TILES: tl.constexpr = D // BLOCK_D
    GROUPS: tl.constexpr = BLOCK_D // 32
    PACKS: tl.constexpr = GROUPS // 4
    shifts = (tl.arange(0, 4)[None, :] * 8).to(tl.uint32)
    total_tasks = total_rows * TILES
    for task_id in range(pid, total_tasks, num_ctas):
        row_id = task_id // TILES
        t = task_id % TILES
        e, m = _resolve_masked_row(offs_e, ends, row_id)
        _situ_glu_quant_masked_tile(
            x_ptr + e * stride_xe + m * stride_xm,
            q_ptr + e * stride_qe + m * stride_qm,
            sc_ptr + e * stride_sce + m,
            t,
            D=D,
            stride_sck=stride_sck,
            gate_beta=gate_beta,
            up_beta=up_beta,
            eps=eps,
            BLOCK_D=BLOCK_D,
            GROUPS=GROUPS,
            PACKS=PACKS,
            shifts=shifts,
        )


@triton.jit(do_not_specialize=["max_m", "stride_xe", "stride_xm", "stride_qe", "stride_qm", "stride_sce", "stride_sck"])
def _situ_glu_fp8_quant_masked_row_persistent_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    masked_m_ptr,
    max_m,
    E,
    D: tl.constexpr,
    stride_xe: tl.int64,
    stride_xm: tl.int64,
    stride_qe: tl.int64,
    stride_qm: tl.int64,
    stride_sce: tl.int64,
    stride_sck: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    E_PAD: tl.constexpr,
):
    """Row-per-task persistent masked SiTU-GLU + FP8 quant (computes expert offset once per row)."""
    offs_e, ends, total_rows = _masked_prefix_map(masked_m_ptr, max_m, E, E_PAD)
    if total_rows == 0:
        return
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    TILES: tl.constexpr = D // BLOCK_D
    GROUPS: tl.constexpr = BLOCK_D // 32
    PACKS: tl.constexpr = GROUPS // 4
    shifts = (tl.arange(0, 4)[None, :] * 8).to(tl.uint32)

    for row_id in range(pid, total_rows, num_ctas):
        e, m = _resolve_masked_row(offs_e, ends, row_id)
        x_base = x_ptr + e * stride_xe + m * stride_xm
        q_base = q_ptr + e * stride_qe + m * stride_qm
        sc_base = sc_ptr + e * stride_sce + m
        for t in tl.static_range(TILES):
            _situ_glu_quant_masked_tile(
                x_base,
                q_base,
                sc_base,
                t,
                D=D,
                stride_sck=stride_sck,
                gate_beta=gate_beta,
                up_beta=up_beta,
                eps=eps,
                BLOCK_D=BLOCK_D,
                GROUPS=GROUPS,
                PACKS=PACKS,
                shifts=shifts,
            )


@triton.jit(do_not_specialize=["max_m", "stride_xe", "stride_xm", "stride_qe", "stride_qm", "stride_sce", "stride_sck"])
def _situ_glu_fp8_quant_masked_packed_persistent_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    masked_m_ptr,
    max_m,
    E,
    D,
    num_groups,
    num_packs,
    stride_xe: tl.int64,
    stride_xm: tl.int64,
    stride_qe: tl.int64,
    stride_qm: tl.int64,
    stride_sce: tl.int64,
    stride_sck: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_TILES: tl.constexpr,
    E_PAD: tl.constexpr,
):
    offs_e, ends, total_rows = _masked_prefix_map(masked_m_ptr, max_m, E, E_PAD)
    if total_rows == 0:
        return
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    total_tasks = total_rows * D_TILES

    GROUPS: tl.constexpr = BLOCK_D // 32
    PACKS: tl.constexpr = GROUPS // 4

    for task_id in range(pid, total_tasks, num_ctas):
        row_id = task_id // D_TILES
        pid_d = task_id - row_id * D_TILES
        e = tl.sum((ends <= row_id).to(tl.int32), axis=0).to(tl.int64)
        m_start_e = tl.sum(tl.where(offs_e == (e - 1), ends, 0), axis=0).to(tl.int64)
        m = row_id.to(tl.int64) - m_start_e

        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        row_base = e * stride_xe + m * stride_xm
        gate = tl.load(x_ptr + row_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + row_base + D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        y = _situ_glu_fp32_to_bf16(gate, up, gate_beta=gate_beta, up_beta=up_beta)
        y = tl.where(mask_d, y, 0.0).reshape((GROUPS, 32))
        q, biased = _compute_ue8m0_fp8_block32(y, eps=eps)
        g_idx = pid_d * GROUPS + tl.arange(0, GROUPS)
        pack_offs = pid_d * PACKS + tl.arange(0, PACKS)
        _store_deepgemm_row_packed(
            q_ptr + e * stride_qe + m * stride_qm, sc_ptr + e * stride_sce, m,
            q, biased, offs_d, mask_d, g_idx, num_groups, pack_offs, num_packs,
            stride_sck, BLOCK_D, GROUPS, PACKS,
        )


def _alloc_scale_tensor(
    M: int,
    num_groups: int,
    scale_format: str,
    device: torch.device,
    m_indptr: Optional[torch.Tensor] = None,
    zero_unused_scales: bool = True,
) -> Tuple[torch.Tensor, int, int]:
    num_packs = (num_groups + 3) // 4
    sf_cols_pad = num_packs * 4
    alloc_fn = torch.zeros if zero_unused_scales else torch.empty
    if scale_format == "deepgemm":
        tma_m = ((M + 3) // 4) * 4
        sc = alloc_fn((num_packs, tma_m), dtype=torch.int32, device=device).T[:M]
        return sc, 0, sf_cols_pad
    if scale_format == "raw":
        sc = torch.empty((M, num_groups), dtype=torch.uint8, device=device)
        return sc, 1, sf_cols_pad
    if scale_format == "flashinfer":
        G = int(m_indptr.numel()) - 1 if m_indptr is not None else 1
        total_rows = (((M + max(G - 1, 0) * 127) + 127) // 128) * 128 if M > 0 else 0
        sc = alloc_fn((total_rows * sf_cols_pad,), dtype=torch.uint8, device=device)
        return sc, 2, sf_cols_pad
    raise ValueError(f"Unsupported scale_format={scale_format!r}")


def _masked_persist_ctas(device: torch.device, E: int, max_m: int, d_tiles: int) -> int:
    sm_count = _get_sm_count(int(device.index or 0))
    wave_mult = 4 if max_m <= 128 else 6
    return min(max(sm_count * wave_mult, 256), E * max_m * d_tiles)


def _pick_block_d(D: int) -> int:
    if D <= 128:
        return 128
    if D <= 256:
        return 256
    return 512


def per_token_group_quant_fp8_ue8m0(
    x: torch.Tensor,
    *,
    group_size: int = 32,
    scale_format: str = "deepgemm",
    masked_m: Optional[torch.Tensor] = None,
    m_indptr: Optional[torch.Tensor] = None,
    eps: float = _QUANT_EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-Triton per-token group-32 UE8M0 FP8 E4M3 quantization (zero sgl_kernel dep).

    Args:
        x: Input tensor ``[M, K]`` (or ``[E, max_m, K]`` when ``masked_m`` is provided).
        group_size: Must be ``32`` for SM100 MXFP8xMXFP4.
        scale_format: ``"deepgemm"`` (packed int32 MN-major TMA-aligned),
            ``"raw"`` (uint8 ``[..., K // 32]``), or ``"flashinfer"`` (128x4 swizzled uint8).
        masked_m: Optional ``[E]`` int32 active row counts for 3D ``[E, max_m, K]``.
        m_indptr: Optional ``[G + 1]`` int32 group offsets for ``scale_format="flashinfer"``.
        eps: Floor on scaled amax to avoid zero-division / subnormal UE8M0.
    """
    if group_size != 32:
        raise ValueError(f"Expected group_size=32, got {group_size}")
    K = int(x.shape[-1])
    if K % 32 != 0:
        raise ValueError(f"K={K} must be divisible by 32")

    if not x.is_cuda:
        return per_token_group_quant_fp8_ue8m0_ref(
            x,
            group_size=group_size,
            scale_format=scale_format,
            masked_m=masked_m,
            m_indptr=m_indptr,
            eps=eps,
        )

    if masked_m is not None:
        if x.ndim != 3:
            raise ValueError(f"masked_m requires 3D [E, max_m, K], got {x.shape}")
        if scale_format != "deepgemm":
            return per_token_group_quant_fp8_ue8m0_ref(
                x,
                group_size=group_size,
                scale_format=scale_format,
                masked_m=masked_m,
                m_indptr=m_indptr,
                eps=eps,
            )
        E, max_m, _ = x.shape
        x_c = x if x.stride(-1) == 1 else x.contiguous()
        q = torch.empty((E, max_m, K), dtype=torch.float8_e4m3fn, device=x.device)
        num_groups = K // 32
        num_packs = (num_groups + 3) // 4
        tma_m = ((max_m + 3) // 4) * 4
        sc = torch.zeros(
            (E, num_packs, tma_m), dtype=torch.int32, device=x.device
        ).transpose(1, 2)[:, :max_m, :]
        if E == 0 or max_m == 0 or K == 0:
            return q, sc
        BLOCK_K = _pick_block_d(K)
        d_tiles = triton.cdiv(K, BLOCK_K)
        mm_c = masked_m.contiguous()
        if E <= _MASKED_PERSIST_MAX_E:
            num_ctas = _masked_persist_ctas(x_c.device, E, max_m, d_tiles)
            _per_token_group_quant_fp8_ue8m0_masked_persistent_kernel[(num_ctas,)](
                x_c,
                q,
                sc,
                mm_c,
                max_m,
                E,
                K,
                num_groups,
                num_packs,
                x_c.stride(0),
                x_c.stride(1),
                q.stride(0),
                q.stride(1),
                sc.stride(0),
                sc.stride(2),
                eps=float(eps),
                BLOCK_K=BLOCK_K,
                D_TILES=d_tiles,
                E_PAD=max(32, triton.next_power_of_2(E)),
                num_warps=4,
            )
        else:
            _per_token_group_quant_fp8_ue8m0_masked_kernel[
                (E, triton.cdiv(max_m, _MASKED_CHUNK_M), d_tiles)
            ](
                x_c,
                q,
                sc,
                mm_c,
                max_m,
                K,
                num_groups,
                num_packs,
                x_c.stride(0),
                x_c.stride(1),
                q.stride(0),
                q.stride(1),
                sc.stride(0),
                sc.stride(2),
                eps=float(eps),
                BLOCK_K=BLOCK_K,
                CHUNK_M=_MASKED_CHUNK_M,
                num_warps=4,
            )
        return q, sc

    orig_prefix = x.shape[:-1]
    x_2d = x.reshape(-1, K)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    M = int(x_2d.shape[0])
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    num_groups = K // 32
    num_packs = (num_groups + 3) // 4
    sc, fmt_code, sf_cols_pad = _alloc_scale_tensor(
        M, num_groups, scale_format, x.device, m_indptr=m_indptr
    )
    if M == 0 or K == 0:
        if scale_format == "raw":
            return q.reshape(*orig_prefix, K), sc.reshape(*orig_prefix, num_groups)
        if scale_format == "deepgemm":
            return q.reshape(*orig_prefix, K), sc.reshape(*orig_prefix, num_packs)
        return q, sc

    indptr_t = m_indptr.contiguous() if m_indptr is not None else q
    num_ep_groups = (int(m_indptr.numel()) - 1) if m_indptr is not None else 1
    BLOCK_K = _pick_block_d(K)
    _per_token_group_quant_fp8_ue8m0_kernel[(M, triton.cdiv(K, BLOCK_K))](
        x_2d,
        q,
        sc,
        indptr_t,
        M,
        K,
        num_groups,
        num_packs,
        sf_cols_pad,
        x_2d.stride(0),
        q.stride(0),
        sc.stride(1) if fmt_code == 0 else 0,
        sc.stride(0) if fmt_code == 1 else 0,
        eps=float(eps),
        BLOCK_K=BLOCK_K,
        SCALE_FMT=fmt_code,
        HAS_INDPTR=(m_indptr is not None),
        NUM_EP_GROUPS=num_ep_groups,
        BSEARCH_ITERS=max(1, (num_ep_groups - 1).bit_length()),
        num_warps=(2 if M >= 4096 else 4),
    )
    if scale_format == "raw":
        return q.reshape(*orig_prefix, K), sc.reshape(*orig_prefix, num_groups)
    if scale_format == "deepgemm" and len(orig_prefix) > 1:
        return q.reshape(*orig_prefix, K), sc
    return q, sc


def situ_glu_fp8_quant_packed(
    x: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    group_size: int = 32,
    scale_format: str = "deepgemm",
    m_indptr: Optional[torch.Tensor] = None,
    eps: float = _QUANT_EPS,
    zero_unused_scales: bool = True,
    max_active_rows: Optional[torch.Tensor] = None,
    active_row_bounds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiTU-GLU(gate_beta=4.0, up_beta=25.0) + block-32 UE8M0 FP8 quantization.

    Computes ``SiTU-GLU`` in FP32, rounds at the BF16 boundary, quantizes into
    ``float8_e4m3fn``, and packs per-32 UE8M0 scale factors in a single Triton pass.
    """
    if group_size != 32:
        raise ValueError(f"Expected group_size=32, got {group_size}")
    if up is None:
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Expected even last dimension for concatenated gate_up, got {x.shape[-1]}"
            )
        D = int(x.shape[-1]) // 2
        orig_prefix = x.shape[:-1]
    else:
        if x.shape != up.shape:
            raise ValueError(f"gate shape {x.shape} != up shape {up.shape}")
        D = int(x.shape[-1])
        orig_prefix = x.shape[:-1]

    if D % 32 != 0:
        raise ValueError(f"Intermediate width D={D} must be divisible by 32")

    if not x.is_cuda or (up is not None and not up.is_cuda):
        return situ_glu_fp8_quant_ref(
            x,
            up,
            gate_beta=gate_beta,
            up_beta=up_beta,
            group_size=group_size,
            scale_format=scale_format,
            m_indptr=m_indptr,
            eps=eps,
        )

    if up is None:
        g_2d = x.reshape(-1, 2 * D)
        if g_2d.stride(-1) != 1:
            g_2d = g_2d.contiguous()
        u_2d = g_2d
        up_offset = D
    else:
        g_2d = x.reshape(-1, D)
        u_2d = up.reshape(-1, D)
        if g_2d.stride(-1) != 1:
            g_2d = g_2d.contiguous()
        if u_2d.stride(-1) != 1:
            u_2d = u_2d.contiguous()
        up_offset = 0

    M = int(g_2d.shape[0])
    q = torch.empty((M, D), dtype=torch.float8_e4m3fn, device=x.device)
    num_groups = D // 32
    num_packs = (num_groups + 3) // 4
    sc, fmt_code, sf_cols_pad = _alloc_scale_tensor(
        M,
        num_groups,
        scale_format,
        x.device,
        m_indptr=m_indptr,
        zero_unused_scales=zero_unused_scales,
    )
    if M == 0 or D == 0:
        if scale_format == "raw":
            return q.reshape(*orig_prefix, D), sc.reshape(*orig_prefix, num_groups)
        return q, sc

    indptr_t = m_indptr.contiguous() if m_indptr is not None else q
    num_ep_groups = (int(m_indptr.numel()) - 1) if m_indptr is not None else 1
    bsearch_iters = max(1, (num_ep_groups - 1).bit_length())
    bd_1d = 0
    if D % 128 == 0:
        for cand in (512, 256, 128):
            if D % cand == 0:
                bd_1d = cand
                break
    has_active_rows = max_active_rows is not None
    active_rows_t = max_active_rows if has_active_rows else q
    has_row_bounds = (
        active_row_bounds is not None and not zero_unused_scales
    )
    if has_row_bounds:
        rb_m_idx, rb_offs, rb_cnts = active_row_bounds
    else:
        rb_m_idx = rb_offs = rb_cnts = q
    with _cuda_device_ctx(g_2d.device):
        if bd_1d > 0 and D // bd_1d <= 16:
            nw_1d = 2 if (has_row_bounds or M >= 4096) else 4
            _situ_glu_fp8_quant_packed_1d_kernel[(M,)](
                g_2d,
                u_2d,
                q,
                sc,
                indptr_t,
                rb_m_idx,
                rb_offs,
                rb_cnts,
                M,
                D=D,
                stride_g_row=g_2d.stride(0),
                stride_u_row=u_2d.stride(0),
                stride_q_row=q.stride(0),
                stride_sc_k=sc.stride(1) if fmt_code == 0 else 0,
                stride_sc_m=sc.stride(0) if fmt_code == 1 else 0,
                up_offset=up_offset,
                sf_cols_pad=sf_cols_pad,
                gate_beta=float(gate_beta),
                up_beta=float(up_beta),
                eps=float(eps),
                BLOCK_D=bd_1d,
                active_rows_ptr=active_rows_t,
                SCALE_FMT=fmt_code,
                HAS_INDPTR=(m_indptr is not None),
                HAS_ACTIVE_ROWS=has_active_rows,
                HAS_ROW_BOUNDS=has_row_bounds,
                NUM_EP_GROUPS=num_ep_groups,
                BSEARCH_ITERS=bsearch_iters,
                num_warps=nw_1d,
            )
        else:
            BLOCK_D = _pick_block_d(D)
            _situ_glu_fp8_quant_packed_kernel[(M, triton.cdiv(D, BLOCK_D))](
                g_2d,
                u_2d,
                q,
                sc,
                indptr_t,
                M,
                D,
                num_groups,
                num_packs,
                sf_cols_pad,
                g_2d.stride(0),
                u_2d.stride(0),
                q.stride(0),
                sc.stride(1) if fmt_code == 0 else 0,
                sc.stride(0) if fmt_code == 1 else 0,
                up_offset,
                gate_beta=float(gate_beta),
                up_beta=float(up_beta),
                eps=float(eps),
                BLOCK_D=BLOCK_D,
                active_rows_ptr=active_rows_t,
                SCALE_FMT=fmt_code,
                HAS_INDPTR=(m_indptr is not None),
                HAS_ACTIVE_ROWS=has_active_rows,
                NUM_EP_GROUPS=num_ep_groups,
                BSEARCH_ITERS=bsearch_iters,
                num_warps=4,
            )
    if scale_format == "raw":
        return q.reshape(*orig_prefix, D), sc.reshape(*orig_prefix, num_groups)
    return q, sc


def situ_glu_fp8_quant_masked_packed(
    gate_up: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
    group_size: int = 32,
    eps: float = _QUANT_EPS,
    zero_unused_scales: bool = True,
    out_q: Optional[torch.Tensor] = None,
    out_sc: Optional[torch.Tensor] = None,
    actual_m: Optional[int] = None,
    actual_rows: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiTU-GLU + block-32 UE8M0 FP8 quant for 3D masked MoE ``[E, max_m, 2*D]``."""
    if group_size != 32:
        raise ValueError(f"Expected group_size=32, got {group_size}")
    if gate_up.ndim != 3 or gate_up.shape[-1] % 2 != 0:
        raise ValueError(f"Expected [E, max_m, 2*D] gate_up, got {gate_up.shape}")
    E, max_m, two_D = gate_up.shape
    D = two_D // 2
    if D % 32 != 0:
        raise ValueError(f"D={D} must be divisible by 32")

    if not gate_up.is_cuda:
        y = situ_and_mul_ref(gate_up, gate_beta=gate_beta, up_beta=up_beta)
        return _quant_block32_ue8m0_ref(
            y, group_size=32, scale_format="deepgemm", eps=eps
        )

    x_c = gate_up if gate_up.stride(-1) == 1 else gate_up.contiguous()
    q = (
        out_q
        if out_q is not None
        else torch.empty((E, max_m, D), dtype=torch.float8_e4m3fn, device=gate_up.device)
    )
    num_groups = D // 32
    num_packs = (num_groups + 3) // 4
    if out_sc is not None:
        sc = out_sc
        if zero_unused_scales and sc.numel() > 0:
            sc.zero_()
    else:
        tma_m = ((max_m + 3) // 4) * 4
        alloc_fn = torch.zeros if zero_unused_scales else torch.empty
        sc = alloc_fn(
            (E, num_packs, tma_m), dtype=torch.int32, device=gate_up.device
        ).transpose(1, 2)[:, :max_m, :]
    if E == 0 or max_m == 0 or D == 0:
        return q, sc

    BLOCK_D = _pick_block_d(D)
    d_tiles = triton.cdiv(D, BLOCK_D)
    mm_c = masked_m if masked_m.is_contiguous() else masked_m.contiguous()
    row_est = actual_rows if actual_rows is not None else actual_m
    use_row_tile = (row_est is not None and row_est <= 640) or (
        row_est is None and max_m <= 128
    )
    if E <= _MASKED_PERSIST_MAX_E and D >= 512 and D % 512 == 0 and E > 0 and max_m > 0:
        sm_count = _get_sm_count(int(x_c.device.index or 0))
        d_tiles_512 = D // 512
        e_pad = max(32, triton.next_power_of_2(E))
        if use_row_tile:
            # Spread (row, 512-col tile) pairs across SMs to saturate small/skewed batches
            # without hot-expert serialization.
            num_ctas = min(sm_count * 4, E * max_m * d_tiles_512)
            kernel = _situ_glu_fp8_quant_masked_row_tile_persistent_kernel
        else:
            num_ctas = min(sm_count * 6, E * max_m)
            kernel = _situ_glu_fp8_quant_masked_row_persistent_kernel
        with _cuda_device_ctx(x_c.device):
            kernel[(num_ctas,)](
                x_c,
                q,
                sc,
                mm_c,
                max_m,
                E,
                D=D,
                stride_xe=x_c.stride(0),
                stride_xm=x_c.stride(1),
                stride_qe=q.stride(0),
                stride_qm=q.stride(1),
                stride_sce=sc.stride(0),
                stride_sck=sc.stride(2),
                gate_beta=float(gate_beta),
                up_beta=float(up_beta),
                eps=float(eps),
                BLOCK_D=512,
                E_PAD=e_pad,
                num_warps=4,
            )
    elif E <= _MASKED_PERSIST_MAX_E:
        num_ctas = _masked_persist_ctas(x_c.device, E, max_m, d_tiles)
        _situ_glu_fp8_quant_masked_packed_persistent_kernel[(num_ctas,)](
            x_c,
            q,
            sc,
            mm_c,
            max_m,
            E,
            D,
            num_groups,
            num_packs,
            x_c.stride(0),
            x_c.stride(1),
            q.stride(0),
            q.stride(1),
            sc.stride(0),
            sc.stride(2),
            gate_beta=float(gate_beta),
            up_beta=float(up_beta),
            eps=float(eps),
            BLOCK_D=BLOCK_D,
            D_TILES=d_tiles,
            E_PAD=max(32, triton.next_power_of_2(E)),
            num_warps=4,
        )
    else:
        _situ_glu_fp8_quant_masked_packed_kernel[
            (E, triton.cdiv(max_m, _MASKED_CHUNK_M), d_tiles)
        ](
            x_c,
            q,
            sc,
            mm_c,
            max_m,
            D,
            num_groups,
            num_packs,
            x_c.stride(0),
            x_c.stride(1),
            q.stride(0),
            q.stride(1),
            sc.stride(0),
            sc.stride(2),
            gate_beta=float(gate_beta),
            up_beta=float(up_beta),
            eps=float(eps),
            BLOCK_D=BLOCK_D,
            CHUNK_M=_MASKED_CHUNK_M,
            num_warps=4,
        )
    return q, sc


@triton.jit(do_not_specialize=["M", "m_aligned", "stride_xm", "stride_qm"])
def _quant_fp8_1x128_dg_kernel(
    x_ptr,
    q_ptr,
    sc_ptr,
    M,
    m_aligned,
    stride_xm: tl.int64,
    stride_qm: tl.int64,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Coalesced 1x128 FP8 quant producing MN-major TMA-aligned packed UE8M0 int32 scales."""
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    g_offs = tl.arange(0, 4)
    g_idx = pid_p * 4 + g_offs
    g_mask = g_idx < NUM_GROUPS
    k_offs = g_idx[:, None] * 128 + tl.arange(0, 128)[None, :]
    load_mask = row_mask[:, None, None] & g_mask[None, :, None]
    vals = tl.load(
        x_ptr + rows[:, None, None].to(tl.int64) * stride_xm + k_offs[None, :, :],
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(vals), axis=2), 1.0e-4)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    biased = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    biased = tl.maximum(tl.minimum(biased, 254), 1)
    inv_scale = tl.div_rn(1.0, (biased << 23).to(tl.float32, bitcast=True))
    q_vals = tl.clamp(vals * inv_scale[:, :, None], -448.0, 448.0).to(tl.float8e4nv)
    tl.store(
        q_ptr + rows[:, None, None].to(tl.int64) * stride_qm + k_offs[None, :, :],
        q_vals,
        mask=load_mask,
    )
    biased_valid = tl.where(g_mask[None, :], biased & 0xFF, 0)
    packed = tl.sum(biased_valid << (g_offs[None, :] * 8), axis=1)
    tl.store(sc_ptr + pid_p * m_aligned.to(tl.int64) + rows.to(tl.int64), packed, mask=row_mask)


def quant_fp8_1x128_deepgemm_triton(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize [M, K] into E4M3FN + MN-major TMA-aligned packed UE8M0 int32 [M, ceil_div(K//128, 4)]."""
    K = int(x.shape[-1])
    if K % 128 != 0:
        raise ValueError(f"quant_fp8_1x128_deepgemm_triton requires K % 128 == 0, got K={K}")
    orig_shape = x.shape
    x_2d = x.reshape(-1, K)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    M = int(x_2d.shape[0])
    num_groups = K // 128
    num_packs = (num_groups + 3) // 4
    m_aligned = ((M + 3) // 4) * 4

    q = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    sc = torch.empty(
        (num_packs, max(m_aligned, 4)), dtype=torch.int32, device=x.device
    ).transpose(0, 1)[:M, :]
    if M == 0:
        return q.reshape(*orig_shape[:-1], K), sc

    if not x_2d.is_cuda:
        x_g = x_2d.float().view(M, num_groups, 128)
        amax = x_g.abs().amax(dim=2).clamp_min(1.0e-4)
        bits = (amax * (1.0 / 448.0)).view(torch.int32)
        exp = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)).clamp(1, 254)
        sf = (exp << 23).view(torch.float32)
        q.copy_((x_g * (1.0 / sf.unsqueeze(2))).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(M, K))
        pad_g = num_packs * 4 - num_groups
        if pad_g > 0:
            exp = torch.nn.functional.pad(exp, (0, pad_g), value=0)
        sc.copy_(exp.to(torch.uint8).contiguous().view(torch.int32))
        return q.reshape(*orig_shape[:-1], K), sc

    bm = 1 if M <= 4 else (4 if M <= 4096 else 8)
    grid = (triton.cdiv(M, bm), num_packs)
    with _cuda_device_ctx(x.device):
        _quant_fp8_1x128_dg_kernel[grid](
            x_2d,
            q,
            sc,
            M,
            m_aligned,
            x_2d.stride(0),
            q.stride(0),
            K=K,
            NUM_GROUPS=num_groups,
            BLOCK_M=bm,
            num_warps=4,
        )
    return q.reshape(*orig_shape[:-1], K), sc


__all__ = [
    "SituAndMul",
    "bf16_linear_tgv_or_torch",
    "can_sm10x_bf16_tgv",
    "get_sm10x_bf16_tgv_backend",
    "per_token_group_quant_fp8_ue8m0",
    "per_token_group_quant_fp8_ue8m0_ref",
    "quant_fp8_1x128_deepgemm_triton",
    "run_sm10x_bf16_tgv_gemm",
    "select_sm10x_bf16_tgv_tactic",
    "situ_and_mul_ref",
    "situ_and_mul_triton",
    "situ_glu_fp8_quant_masked_packed",
    "situ_glu_fp8_quant_packed",
    "situ_glu_fp8_quant_ref",
]
