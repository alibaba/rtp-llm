"""Fuse draft request/packed layout conversions without changing attention."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def _enabled(tensor):
    return (
        tensor.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(tensor.device) == (12, 0)
        and tensor.dtype == torch.bfloat16
        and not tensor.requires_grad
    )


@triton.jit
def _pack_draft_attention(
    Q,
    SELECTED,
    SOURCES,
    VISIBLE,
    QUERY,
    INDICES,
    QS: tl.constexpr,
    SS: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(SOURCES + row)
    value = tl.load(Q + source * QS + col, col < H * D, other=0)
    offset = ((row // L * H + col // D) * L + row % L) * D + col % D
    tl.store(QUERY + offset, value, col < H * D)
    selected = tl.load(SELECTED + source * SS + col, col < K, other=-1)
    visible = tl.load(VISIBLE + row)
    tl.store(INDICES + row * K + col, tl.where(visible > 0, selected, -1), col < K)


@triton.jit
def _unpack_draft_attention(
    ATTENTION,
    INVERSE,
    VALID,
    GATE,
    OUTPUT,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    L: tl.constexpr,
    WITH_GATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inverse = tl.load(INVERSE + row)
    valid = tl.load(VALID + row)
    offset = inverse // L * S0 + col // D * S1 + inverse % L * S2 + col % D
    value = tl.load(ATTENTION + offset, col < H * D, other=0)
    value = tl.where(valid, value, 0).to(tl.float32)
    if WITH_GATE:
        gate = tl.load(GATE + row * H * D + col, col < H * D, other=0).to(tl.float32)
        sigmoid = (1.0 / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        value *= sigmoid
    tl.store(OUTPUT + row * H * D + col, value, col < H * D)


def try_pack_inputs(qkv, selected, sources, visible, *, heads, head_dim):
    if not (
        _enabled(qkv)
        and qkv.ndim == 2
        and 1 <= qkv.shape[0] <= 32
        and (heads, head_dim) == (12, 256)
        and qkv.shape[1] >= heads * head_dim
        and qkv.stride(1) == 1
        and qkv.stride(0) >= qkv.shape[1]
        and selected.shape == (qkv.shape[0], 2051)
        and selected.dtype == torch.int32
        and selected.stride(1) == 1
        and selected.stride(0) >= 2051
        and sources.shape == (qkv.shape[0],)
        and sources.dtype == torch.int64
        and visible.ndim == 2
        and visible.shape[1] == 4
        and visible.numel() == qkv.shape[0]
        and visible.dtype == torch.int32
        and sources.is_contiguous()
        and visible.is_contiguous()
        and all(x.device == qkv.device for x in (selected, sources, visible))
    ):
        return None
    batch, lanes = visible.shape
    query = torch.empty(
        (batch, heads, lanes, head_dim), device=qkv.device, dtype=qkv.dtype
    )
    indices = torch.empty((batch, lanes, 2051), device=qkv.device, dtype=torch.int32)
    _pack_draft_attention[(qkv.shape[0], triton.cdiv(heads * head_dim, 256))](
        qkv,
        selected,
        sources,
        visible,
        query,
        indices,
        QS=qkv.stride(0),
        SS=selected.stride(0),
        H=heads,
        D=head_dim,
        L=lanes,
        K=2051,
        BLOCK=256,
        num_warps=4,
    )
    return query, indices


def try_unpack_output(attention, inverse, valid, *, gate=None):
    if not (
        _enabled(attention)
        and attention.ndim == 4
        and attention.shape[1:] == (12, 4, 256)
        and 1 <= attention.shape[0] <= 8
        and attention.stride(-1) == 1
        and all(stride > 0 for stride in attention.stride())
        and inverse.shape == valid.shape == (attention.shape[0] * 4,)
        and inverse.dtype == torch.int64
        and valid.dtype == torch.bool
        and all(
            x.device == attention.device and x.is_contiguous() for x in (inverse, valid)
        )
        and (
            gate is None
            or (
                gate.shape == (attention.shape[0] * 4, 3072)
                and gate.dtype == attention.dtype
                and gate.device == attention.device
                and gate.is_contiguous()
                and not gate.requires_grad
            )
        )
    ):
        return None
    batch, heads, lanes, dim = attention.shape
    output = torch.empty(
        (batch * lanes, heads * dim), device=attention.device, dtype=attention.dtype
    )
    _unpack_draft_attention[(batch * lanes, triton.cdiv(heads * dim, 256))](
        attention,
        inverse,
        valid,
        gate if gate is not None else output,
        output,
        S0=attention.stride(0),
        S1=attention.stride(1),
        S2=attention.stride(2),
        H=heads,
        D=dim,
        L=lanes,
        WITH_GATE=gate is not None,
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@triton.jit
def _restore_draft_selection(
    SELECTED, INVERSE, VALID, OUT, STRIDE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    inverse = tl.load(INVERSE + row)
    valid = tl.load(VALID + row)
    value = tl.load(SELECTED + inverse * STRIDE + col, (col < 2051) & valid, other=-1)
    tl.store(OUT + row * 2051 + col, value, col < 2051)


def try_restore_selection(selected, inverse, valid):
    if not (
        selected.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(selected.device) == (12, 0)
        and selected.ndim == 2
        and selected.shape[1] == 2051
        and selected.shape[0] in (4, 8, 12, 16, 20, 24, 28, 32)
        and selected.dtype == torch.int32
        and selected.stride(1) == 1
        and selected.stride(0) >= 2051
        and inverse.shape == valid.shape == (selected.shape[0],)
        and inverse.dtype == torch.int64
        and valid.dtype == torch.bool
        and all(
            t.device == selected.device and t.is_contiguous() for t in (inverse, valid)
        )
    ):
        return None
    output = torch.empty(selected.shape, device=selected.device, dtype=selected.dtype)
    _restore_draft_selection[(selected.shape[0], triton.cdiv(2051, 256))](
        selected,
        inverse,
        valid,
        output,
        STRIDE=selected.stride(0),
        BLOCK=256,
    )
    return output
