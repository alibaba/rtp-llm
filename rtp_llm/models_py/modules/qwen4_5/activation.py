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

from functools import lru_cache
from typing import Optional, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

_FP8_E4M3_MAX = 448.0
_QUANT_EPS = 1.0e-10
_MASKED_PERSIST_MAX_E = 1024
_MASKED_CHUNK_M = 16


@lru_cache(maxsize=8)
def _get_sm_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def situ_and_mul_ref(
    x: torch.Tensor,
    up: Optional[torch.Tensor] = None,
    *,
    gate_beta: float = 4.0,
    up_beta: float = 25.0,
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
    return (g * u).to(out_dtype)


@triton.jit
def _situ_and_mul_cat_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    D,
    stride_x_row: tl.int64,
    stride_out_row: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
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
    g = gate_beta * libdevice.tanh(tl.div_rn(gate, gate_beta)) * sig_gate
    u = up_beta * libdevice.tanh(tl.div_rn(up, up_beta))
    out = g * u

    out_row_ptr = out_ptr + pid_row * stride_out_row
    tl.store(out_row_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _situ_and_mul_split_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    num_rows,
    D,
    stride_g_row: tl.int64,
    stride_u_row: tl.int64,
    stride_out_row: tl.int64,
    gate_beta: tl.constexpr,
    up_beta: tl.constexpr,
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
    g = gate_beta * libdevice.tanh(tl.div_rn(gate, gate_beta)) * sig_gate
    u = up_beta * libdevice.tanh(tl.div_rn(up, up_beta))
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
) -> torch.Tensor:
    """Fused Triton kernel for SiTU-GLU activation.

    Automatically falls back to :func:`situ_and_mul_ref` for non-CUDA tensors.
    """
    if not x.is_cuda or (up is not None and not up.is_cuda):
        return situ_and_mul_ref(x, up, gate_beta=gate_beta, up_beta=up_beta)

    gate_beta_f = float(gate_beta)
    up_beta_f = float(up_beta)

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

        BLOCK_D = 1024 if D >= 1024 else triton.next_power_of_2(max(D, 64))
        num_warps = 8 if BLOCK_D >= 1024 else 4
        grid = (num_rows, triton.cdiv(D, BLOCK_D))
        _situ_and_mul_cat_kernel[grid](
            x_2d,
            out,
            num_rows,
            D,
            x_2d.stride(0),
            out.stride(0),
            gate_beta=gate_beta_f,
            up_beta=up_beta_f,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
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

    BLOCK_D = 1024 if D >= 1024 else triton.next_power_of_2(max(D, 64))
    num_warps = 8 if BLOCK_D >= 1024 else 4
    grid = (num_rows, triton.cdiv(D, BLOCK_D))
    _situ_and_mul_split_kernel[grid](
        g_2d,
        u_2d,
        out,
        num_rows,
        D,
        g_2d.stride(0),
        u_2d.stride(0),
        out.stride(0),
        gate_beta=gate_beta_f,
        up_beta=up_beta_f,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out.reshape(*orig_shape, D)


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
        self, x: torch.Tensor, up: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.use_triton and x.is_cuda and (up is None or up.is_cuda):
            return situ_and_mul_triton(
                x, up, gate_beta=self.gate_beta, up_beta=self.up_beta
            )
        return situ_and_mul_ref(
            x, up, gate_beta=self.gate_beta, up_beta=self.up_beta
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
    sig_gate = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    g = gate_beta * libdevice.tanh(tl.div_rn(gate, gate_beta)) * sig_gate
    u = up_beta * libdevice.tanh(tl.div_rn(up, up_beta))
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
    SCALE_FMT: tl.constexpr,
    HAS_INDPTR: tl.constexpr,
    NUM_EP_GROUPS: tl.constexpr,
    BSEARCH_ITERS: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_d = tl.program_id(1)
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
) -> Tuple[torch.Tensor, int, int]:
    num_packs = (num_groups + 3) // 4
    sf_cols_pad = num_packs * 4
    if scale_format == "deepgemm":
        tma_m = ((M + 3) // 4) * 4
        sc = torch.zeros((num_packs, tma_m), dtype=torch.int32, device=device).T[:M]
        return sc, 0, sf_cols_pad
    if scale_format == "raw":
        sc = torch.empty((M, num_groups), dtype=torch.uint8, device=device)
        return sc, 1, sf_cols_pad
    if scale_format == "flashinfer":
        G = int(m_indptr.numel()) - 1 if m_indptr is not None else 1
        total_rows = (((M + max(G - 1, 0) * 127) + 127) // 128) * 128 if M > 0 else 0
        sc = torch.zeros((total_rows * sf_cols_pad,), dtype=torch.uint8, device=device)
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
        num_warps=4,
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
        M, num_groups, scale_format, x.device, m_indptr=m_indptr
    )
    if M == 0 or D == 0:
        if scale_format == "raw":
            return q.reshape(*orig_prefix, D), sc.reshape(*orig_prefix, num_groups)
        return q, sc

    indptr_t = m_indptr.contiguous() if m_indptr is not None else q
    num_ep_groups = (int(m_indptr.numel()) - 1) if m_indptr is not None else 1
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
        SCALE_FMT=fmt_code,
        HAS_INDPTR=(m_indptr is not None),
        NUM_EP_GROUPS=num_ep_groups,
        BSEARCH_ITERS=max(1, (num_ep_groups - 1).bit_length()),
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
    q = torch.empty((E, max_m, D), dtype=torch.float8_e4m3fn, device=gate_up.device)
    num_groups = D // 32
    num_packs = (num_groups + 3) // 4
    tma_m = ((max_m + 3) // 4) * 4
    alloc_fn = torch.zeros if zero_unused_scales else torch.empty
    sc = alloc_fn(
        (E, num_packs, tma_m), dtype=torch.int32, device=gate_up.device
    ).transpose(1, 2)[:, :max_m, :]
    if E == 0 or max_m == 0 or D == 0:
        return q, sc

    BLOCK_D = _pick_block_d(D)
    d_tiles = triton.cdiv(D, BLOCK_D)
    mm_c = masked_m.contiguous()
    if E <= _MASKED_PERSIST_MAX_E:
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


__all__ = [
    "SituAndMul",
    "per_token_group_quant_fp8_ue8m0",
    "per_token_group_quant_fp8_ue8m0_ref",
    "situ_and_mul_ref",
    "situ_and_mul_triton",
    "situ_glu_fp8_quant_masked_packed",
    "situ_glu_fp8_quant_packed",
    "situ_glu_fp8_quant_ref",
]
