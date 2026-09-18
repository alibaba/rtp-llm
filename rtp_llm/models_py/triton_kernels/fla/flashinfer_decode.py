"""Optional FlashInfer GDN decode adapter for T=1 V-first BF16 paged state.

Reuses FlashInfer ``gated_delta_rule_decode_pretranspose`` (gating + L2norm +
recurrent). Serving selects this backend with ``RTP_QWEN35_GDN_DECODE_BACKEND``.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

GDN_DECODE_BACKEND_ENV = "RTP_QWEN35_GDN_DECODE_BACKEND"
_VALID_BACKENDS = ("native", "flashinfer")


def gdn_decode_backend() -> str:
    backend = os.getenv(GDN_DECODE_BACKEND_ENV, "native")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown GDN decode backend {backend!r}; "
            f"expected one of {_VALID_BACKENDS}"
        )
    return backend


def supports_flashinfer_gdn_decode(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    seq: int,
) -> bool:
    if seq != 1:
        return False
    if q.ndim != 4 or v.ndim != 4 or initial_state.ndim != 4:
        return False
    if q.dtype != torch.bfloat16 or v.dtype != q.dtype:
        return False
    if initial_state.dtype != torch.bfloat16:
        return False
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    if initial_state.shape[-2:] != (128, 128):
        return False
    if initial_state.stride(-1) != 1:
        return False
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] < 9:
        return False
    return True


@triton.jit
def _fill_paged_decode_indices_kernel(
    seq_ptr,
    map_ptr,
    read_ptr,
    write_ptr,
    alog_in_ptr,
    alog_out_ptr,
    seq_size,
    map_stride,
    alog_stride,
    n_pages,
    batch,
    hv,
    BLOCK: tl.constexpr,
    CAST_ALOG: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch
    seq = tl.load(seq_ptr + offs, mask=mask, other=1)
    last = n_pages - 1
    read_page = (seq - 2) // seq_size
    write_page = (seq - 1) // seq_size
    read_page = tl.minimum(tl.maximum(read_page, 0), last)
    write_page = tl.minimum(tl.maximum(write_page, 0), last)
    row = offs.to(tl.int64) * map_stride
    read_idx = tl.load(map_ptr + row + read_page, mask=mask, other=0)
    write_idx = tl.load(map_ptr + row + write_page, mask=mask, other=0)
    invalid = read_idx <= 0
    read_idx = tl.where(invalid, -1, read_idx)
    write_idx = tl.where(invalid | (write_idx <= 0), -1, write_idx)
    tl.store(read_ptr + offs, read_idx, mask=mask)
    tl.store(write_ptr + offs, write_idx, mask=mask)
    if CAST_ALOG:
        mask_h = offs < hv
        alog = tl.load(alog_in_ptr + offs * alog_stride, mask=mask_h, other=0.0)
        tl.store(alog_out_ptr + offs, alog.to(tl.float32), mask=mask_h)


def fill_paged_decode_indices(
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
    A_log: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map RTP ``block_map`` + ``sequence_lengths_plus_1`` to FI pool indices.

    Matches fused_recurrent: read ``(L-2)//S``, write ``(L-1)//S``. Slot ids
    ``<= 0`` become ``-1`` (FI BF16 sacrificial slot 0, which is also RTP
    padding). A non-positive read forces the write index to ``-1``.

    When ``A_log`` is given, also emit an fp32 copy in the same launch so FI
    does not pay a separate ``aten::copy_`` / ``.float()``.
    """
    batch = sequence_lengths_plus_1.shape[0]
    read_idx = torch.empty(batch, device=block_map.device, dtype=torch.int32)
    write_idx = torch.empty_like(read_idx)
    cast_alog = A_log is not None and A_log.dtype != torch.float32
    if A_log is None:
        alog_in = read_idx
        alog_out = write_idx
        hv = 0
        alog_stride = 1
    elif cast_alog:
        alog_in = A_log
        alog_out = torch.empty(A_log.shape, device=A_log.device, dtype=torch.float32)
        hv = A_log.numel()
        alog_stride = A_log.stride(-1)
    else:
        alog_in = A_log
        alog_out = A_log
        hv = A_log.numel()
        alog_stride = A_log.stride(-1)
    _fill_paged_decode_indices_kernel[
        (triton.cdiv(max(batch, hv), 256),)
    ](
        sequence_lengths_plus_1,
        block_map,
        read_idx,
        write_idx,
        alog_in,
        alog_out,
        seq_size_per_block,
        block_map.stride(0),
        alog_stride,
        block_map.size(1),
        batch,
        hv,
        BLOCK=256,
        CAST_ALOG=cast_alog,
    )
    if A_log is None:
        return read_idx, write_idx
    return read_idx, write_idx, alog_out


def _gate_bt_hv(x: torch.Tensor, batch: int, hv: int) -> torch.Tensor:
    if x.ndim == 2:
        return x.view(batch, 1, hv)
    if x.ndim == 3:
        return x.view(batch, 1, hv)
    raise ValueError(f"Expected gate [B,HV] or [B,1,HV], got {tuple(x.shape)}")


def flashinfer_gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Run FlashInfer T=1 decode against a V-first paged SSM pool.

    ``q/k`` are ``[B,1,H,K]``, ``v`` is ``[B,1,HV,V]``, ``a/b`` are
    ``[B,HV]`` or ``[B,1,HV]``. ``initial_state`` is the RTP SSM view
    ``[P,HV,V,K]`` (page stride may include conv state).
    Returns ``[B,1,HV,V]``.
    """
    if q.ndim != 4 or q.shape[1] != 1:
        raise ValueError(f"Expected Q [B,1,H,K], got {tuple(q.shape)}")
    batch, _, _, _ = q.shape
    hv = v.shape[2]
    if not supports_flashinfer_gdn_decode(q, v, initial_state, seq=1):
        raise ValueError(
            "FlashInfer GDN decode requires SM90+, BF16 Q/K/V/SSM, K=V=128, T=1"
        )
    if k.shape != q.shape or v.shape[:2] != q.shape[:2]:
        raise ValueError("Q/K/V batch and sequence dims must match")
    if initial_state.shape[1] != hv:
        raise ValueError(
            f"SSM HV {initial_state.shape[1]} does not match V heads {hv}"
        )
    if sequence_lengths_plus_1.shape[0] != batch or block_map.shape[0] != batch:
        raise ValueError("block_map and sequence_lengths_plus_1 must be length B")

    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    read_idx, write_idx, alog = fill_paged_decode_indices(
        block_map, sequence_lengths_plus_1, seq_size_per_block, A_log=A_log
    )
    output, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,
        A_log=alog,
        a=_gate_bt_hv(a, batch, hv),
        dt_bias=dt_bias,
        b=_gate_bt_hv(b, batch, hv),
        scale=scale,
        output=None,
        use_qk_l2norm=True,
        initial_state=initial_state,
        initial_state_indices=read_idx,
        output_state_indices=write_idx,
    )
    return output
