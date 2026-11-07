"""Packed prefill over request-local QSA selections and validated CU metadata."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

from .membership import _find_request, _write_membership
from .ragged_metadata import _locate
from .tma_pipeline import _KV_LAYOUT, _execute_prefill


@triton.jit(do_not_specialize=["B", "WORDS"])
def _build_ragged_membership(Ids, Mask, Cu, B, WORDS, MAX_WORDS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    request = _find_request(Cu, row, B)
    local_s = tl.load(Cu + request + 1) - tl.load(Cu + request)
    _write_membership(Ids, Mask, row, local_s, WORDS, MAX_WORDS)


@g.jit(do_not_specialize=["B", "WORDS"])
def _packed_prefill(
    Q,
    K,
    V,
    Mask,
    O,
    Cu,
    B,
    WORDS,
    G: gl.constexpr,
    QM: gl.constexpr,
    QH: gl.constexpr,
    QS: gl.constexpr,
    OS: gl.constexpr,
    PIPE: gl.constexpr,
):
    positions: gl.constexpr = QM // G
    _, base, length, tile = _locate(Cu, B, positions, G)
    _execute_prefill(
        Q,
        K,
        V,
        Mask,
        O,
        length,
        WORDS,
        256,
        G,
        QM,
        32,
        positions,
        QH,
        QS,
        256,
        OS,
        PIPE,
        QM // 16,
        QM == 32,
        base,
        tile,
        2 if G != 6 else 1,
    )


def run_prefill(q, k, v, selected, cu, lengths):
    _, heads, tokens, dim = q.shape
    longest = max(lengths)
    qm, group = (
        (16, 12) if heads == 12 and longest <= 512 else (32, 4 if heads == 12 else 6)
    )
    words = triton.cdiv(longest, 32)
    membership = torch.empty((tokens, words), dtype=torch.int32, device=q.device)
    _build_ragged_membership[(tokens,)](
        selected,
        membership,
        cu,
        len(lengths),
        words,
        16 if longest <= 512 else 128,
        num_warps=4,
    )
    output = torch.empty(
        (1, tokens, heads, dim), dtype=q.dtype, device=q.device
    ).transpose(1, 2)
    key = TensorDescriptor(k, [tokens, dim], [k.stride(2), 1], [32, dim], _KV_LAYOUT)
    value = TensorDescriptor(v, [tokens, dim], [v.stride(2), 1], [32, dim], _KV_LAYOUT)
    positions = qm // group
    # Cover every positive partition with the same total, including graph replays.
    # Surplus work resolves to an empty request and performs no Q/K/V reads.
    tiles = len(lengths) + (tokens - len(lengths)) // positions
    grid = (tiles, heads // group)
    if group == 4:
        grid = grid[::-1]
    _packed_prefill[grid](
        q,
        key,
        value,
        membership,
        output,
        cu,
        len(lengths),
        words,
        group,
        qm,
        q.stride(1),
        q.stride(2),
        output.stride(2),
        heads == 6,
        num_warps=4,
    )
    return output.transpose(1, 2).reshape(tokens, heads * dim)
