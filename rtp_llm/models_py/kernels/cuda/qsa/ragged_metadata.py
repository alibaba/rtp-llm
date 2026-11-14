"""Gluon work-item mapping for packed QSA prefill."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout


@g.jit
def _add(a, b):
    return a + b


@g.jit
def _locate(Cu, B, POSITIONS: gl.constexpr, G: gl.constexpr):
    layout: gl.constexpr = BlockedLayout([1], [32], [4], [0])
    b = gl.arange(0, 32, layout=layout)
    starts = gl.load(Cu + b, b < B, other=0)
    ends = gl.load(Cu + b + 1, b < B, other=0)
    lengths = ends - starts
    counts = gl.cdiv(lengths, POSITIONS)
    cumulative = gl.associative_scan(counts, 0, _add)
    work = gl.num_programs(1 if G == 4 else 0) - 1 - gl.program_id(1 if G == 4 else 0)
    request = gl.sum(((work >= cumulative) & (b < B)).to(gl.int32), 0)
    base = gl.sum(gl.where(b == request, starts, 0), 0)
    length = gl.sum(gl.where(b == request, lengths, 0), 0)
    first = gl.sum(gl.where(b == request, cumulative - counts, 0), 0)
    return request, base, length, work - first
