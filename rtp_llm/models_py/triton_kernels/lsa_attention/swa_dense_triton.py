"""Triton implementation of the LSA dense sliding-window + sink attention (SWA part).

Semantics (LongCat Sparse Attention, paper section 3.1): query row ``i`` at
absolute position ``p = positions[i]`` attends to the union of the sink tokens
and its causal sliding window over the *same* latent KV buffer that
:mod:`.sparse_mla_triton` gathers from::

    slots(i) = {0, ..., init_tokens - 1} u {max(0, p - window + 1), ..., p}

The attention math is the absorbed-MQA latent attention of
``reference.sparse_mla_attention`` evaluated on that (deduplicated) slot set::

    q_abs [T, H, DQ]   absorbed queries (DQ = kv_lora_rank + rope)
    kv    [N, DQ]      latent KV rows ([c_kv ; k_rope])
    -> out [T, H, DV]  (DV = kv_lora_rank), lse [T, H] (natural log)

Design (HFA split, paper section 4.1.1): the sink+window part is a *dense,
contiguous* range read -- deliberately the opposite of the gathered sparse part
-- so it can run on a second stream and be merged with the sparse partial via
online-softmax rescaling (:func:`merge_attention`).

Kernel layout:

* grid ``(cdiv(T, BLOCK_M), H)``: one query-row block x one head per program,
  FA2-style online softmax along the KV slots. The program loops over the range
  ``[lo, p_max]`` with ``lo = max(0, min(positions) - window + 1)`` and
  ``p_max = max(positions)`` of the token block, and applies the per-row mask
  ``(s <= p) & ((s >= p - window + 1) | (s < init_tokens))``.
* Sink slots below ``lo`` are covered by a second, tiny loop, so a program reads
  exactly one contiguous window instead of ``[0, p_max]`` (which matters as soon
  as ``min(positions)`` is a long way into the context).
* The ``c_kv`` head dim is split into 4 chunks with separate accumulators: the
  ``[BLOCK_N, DC]`` value tile is reused for both the QK scores and ``P @ V``
  (only the rope slice is loaded separately), and the smaller live operands keep
  the kernel under the 99 KiB (101376 B) shared-memory limit of SM86.

The wrapper :func:`swa_dense_fwd` follows the calling convention of
:func:`rtp_llm.models_py.triton_kernels.lsa_attention.sparse_mla_triton.sparse_mla_fwd`.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.lsa_attention.reference import (
    DEFAULT_INIT_TOKENS,
    DEFAULT_KV_LORA_RANK,
    DEFAULT_LOCAL_TOKENS,
)

_OUT_DTYPES = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}

_NEG_INF = float("-inf")


@triton.jit
def _swa_kv_block(
    KV,
    s_glob,
    valid,
    q0,
    q1,
    q2,
    q3,
    q_rope,
    m_i,
    l_i,
    acc_o0,
    acc_o1,
    acc_o2,
    acc_o3,
    softmax_scale,
    N,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    DC: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Score one ``BLOCK_N`` KV block and apply one FA2 online-softmax step.

    ``s_glob`` [BLOCK_N] holds the absolute KV slot ids, ``valid``
    [BLOCK_M, BLOCK_N] the per-row slot mask. The four ``c_kv`` chunks are loaded
    once and used both as the score keys and as the ``P @ V`` values; ``k_rope``
    is only needed for the scores.
    """
    offs_c = tl.arange(0, DC)
    offs_r = tl.arange(0, ROPE)
    load_ok = s_glob < N
    base = KV + s_glob[:, None] * DQ

    k0 = tl.load(base + offs_c[None, :], mask=load_ok[:, None], other=0.0)
    k1 = tl.load(base + (DC + offs_c)[None, :], mask=load_ok[:, None], other=0.0)
    k2 = tl.load(base + (2 * DC + offs_c)[None, :], mask=load_ok[:, None], other=0.0)
    k3 = tl.load(base + (3 * DC + offs_c)[None, :], mask=load_ok[:, None], other=0.0)
    k_rope = tl.load(base + (DV + offs_r)[None, :], mask=load_ok[:, None], other=0.0)

    s_block = tl.dot(q0, tl.trans(k0))
    s_block = tl.dot(q1, tl.trans(k1), s_block)
    s_block = tl.dot(q2, tl.trans(k2), s_block)
    s_block = tl.dot(q3, tl.trans(k3), s_block)
    s_block = tl.dot(q_rope, tl.trans(k_rope), s_block)
    s_block = s_block * softmax_scale
    s_block = tl.where(valid, s_block, float("-inf"))

    m_new = tl.maximum(m_i, tl.max(s_block, axis=1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_safe))
    p = tl.exp(s_block - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    # Rescale the accumulators to the new running max before folding in this block.
    p_k = p.to(k0.dtype)
    acc_o0 = tl.dot(p_k, k0, acc_o0 * alpha[:, None])
    acc_o1 = tl.dot(p_k, k1, acc_o1 * alpha[:, None])
    acc_o2 = tl.dot(p_k, k2, acc_o2 * alpha[:, None])
    acc_o3 = tl.dot(p_k, k3, acc_o3 * alpha[:, None])
    return m_new, l_i, acc_o0, acc_o1, acc_o2, acc_o3


@triton.jit
def _swa_dense_fwd_kernel(
    Q,
    KV,
    POS,
    OUT,
    LSE,
    T,
    N,
    softmax_scale,
    DQ: tl.constexpr,
    DV: tl.constexpr,
    ROPE: tl.constexpr,
    H: tl.constexpr,
    WINDOW: tl.constexpr,
    INIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """``BLOCK_M = tokens_per_prog x HEADS_PER_PROG`` (query, head) dot rows per program.

    The dot M axis is the flattened ``(token, head)`` pair list, so one KV tile
    serves ``BLOCK_M`` score/PV rows. With ``BLOCK_M = 32`` and
    ``HEADS_PER_PROG`` heads per program, each KV window row is read
    ``H / HEADS_PER_PROG`` times per token (``HEADS_PER_PROG == H`` reads it
    exactly once); the per-program KV traffic is
    ``H / HEADS_PER_PROG x (window + tokens_per_prog)`` rows.

    The KV range is the contiguous ``[lo, p_max]`` window of the program's token
    range (``lo`` from its smallest position); the sink prefix
    ``[0, min(init_tokens, lo))`` is visited by a second small loop so
    long-context programs never walk the ``[init_tokens, lo)`` hole in between.
    """
    DC: tl.constexpr = DV // 4
    TOKENS_PER_PROG: tl.constexpr = BLOCK_M // HEADS_PER_PROG

    t_base = tl.program_id(0) * TOKENS_PER_PROG
    head0 = tl.program_id(1) * HEADS_PER_PROG

    m = tl.arange(0, BLOCK_M)
    t_row = t_base + m // HEADS_PER_PROG
    row_ok = t_row < T
    pos = tl.load(POS + t_row, mask=row_ok, other=0).to(tl.int32)
    row = t_row * H + (head0 + m % HEADS_PER_PROG)

    offs_c = tl.arange(0, DC)
    offs_r = tl.arange(0, ROPE)
    q_base = Q + row[:, None] * DQ
    q0 = tl.load(q_base + offs_c[None, :], mask=row_ok[:, None], other=0.0)
    q1 = tl.load(q_base + (DC + offs_c)[None, :], mask=row_ok[:, None], other=0.0)
    q2 = tl.load(q_base + (2 * DC + offs_c)[None, :], mask=row_ok[:, None], other=0.0)
    q3 = tl.load(q_base + (3 * DC + offs_c)[None, :], mask=row_ok[:, None], other=0.0)
    q_rope = tl.load(q_base + (DV + offs_r)[None, :], mask=row_ok[:, None], other=0.0)

    # KV range shared by every row of the program: [lo, p_max] plus the sink prefix.
    p_hi = tl.max(tl.where(row_ok, pos, 0), axis=0)
    p_lo = tl.min(tl.where(row_ok, pos, 2147483647), axis=0)
    lo = tl.maximum(p_lo - WINDOW + 1, 0)
    sink_hi = tl.minimum(INIT, lo)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o0 = tl.zeros([BLOCK_M, DC], dtype=tl.float32)
    acc_o1 = tl.zeros([BLOCK_M, DC], dtype=tl.float32)
    acc_o2 = tl.zeros([BLOCK_M, DC], dtype=tl.float32)
    acc_o3 = tl.zeros([BLOCK_M, DC], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)

    # Sink prefix [0, sink_hi): the slots below the program's common window start.
    # Masking by ``s < sink_hi`` keeps this loop disjoint from the window loop
    # (slots inside [sink_hi, BLOCK_N) are already covered there as window slots).
    for n0 in range(0, sink_hi, BLOCK_N):
        s = n0 + offs_n
        valid = (s[None, :] <= pos[:, None]) & (s[None, :] < sink_hi)
        m_i, l_i, acc_o0, acc_o1, acc_o2, acc_o3 = _swa_kv_block(
            KV, s, valid, q0, q1, q2, q3, q_rope, m_i, l_i,
            acc_o0, acc_o1, acc_o2, acc_o3, softmax_scale, N,
            DQ=DQ, DV=DV, DC=DC, ROPE=ROPE, BLOCK_N=BLOCK_N,
        )

    # Window (and sink tokens inside it) over [lo, p_max].
    for n0 in range(lo, p_hi + 1, BLOCK_N):
        s = n0 + offs_n
        valid = (s[None, :] <= pos[:, None]) & (
            (s[None, :] >= pos[:, None] - WINDOW + 1) | (s[None, :] < INIT)
        )
        m_i, l_i, acc_o0, acc_o1, acc_o2, acc_o3 = _swa_kv_block(
            KV, s, valid, q0, q1, q2, q3, q_rope, m_i, l_i,
            acc_o0, acc_o1, acc_o2, acc_o3, softmax_scale, N,
            DQ=DQ, DV=DV, DC=DC, ROPE=ROPE, BLOCK_N=BLOCK_N,
        )

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    out_base = OUT + row[:, None] * DV
    tl.store(
        out_base + offs_c[None, :],
        (acc_o0 / l_safe[:, None]).to(OUT_DTYPE),
        mask=row_ok[:, None],
    )
    tl.store(
        out_base + (DC + offs_c)[None, :],
        (acc_o1 / l_safe[:, None]).to(OUT_DTYPE),
        mask=row_ok[:, None],
    )
    tl.store(
        out_base + (2 * DC + offs_c)[None, :],
        (acc_o2 / l_safe[:, None]).to(OUT_DTYPE),
        mask=row_ok[:, None],
    )
    tl.store(
        out_base + (3 * DC + offs_c)[None, :],
        (acc_o3 / l_safe[:, None]).to(OUT_DTYPE),
        mask=row_ok[:, None],
    )
    lse = tl.where(l_i > 0, m_i + tl.log(l_safe), float("-inf"))
    tl.store(LSE + row, lse, mask=row_ok)


def _validate(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    kv_lora_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    if not q_abs.is_cuda:
        raise ValueError("SWA dense kernel requires CUDA tensors")
    if q_abs.dim() != 3:
        raise ValueError(f"q_abs must be [T, H, DQ], got {tuple(q_abs.shape)}")
    if kv.dim() != 2 or kv.shape[1] != q_abs.shape[2]:
        raise ValueError("kv must be [N, DQ] matching q_abs' last dim")
    if positions.dim() != 1 or positions.shape[0] != q_abs.shape[0]:
        raise ValueError("positions must be [T] matching q_abs")
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"positions must be int32 or int64, got {positions.dtype}")
    if q_abs.dtype != kv.dtype:
        raise ValueError(
            f"q_abs and kv must share a dtype, got {q_abs.dtype} and {kv.dtype}"
        )

    q_abs = q_abs.contiguous()
    kv = kv.contiguous()
    positions = positions.contiguous()

    t_count, heads, dq = q_abs.shape
    dv = int(kv_lora_rank)
    rope = dq - dv
    if dv <= 0 or dv % 4 or (dv // 4) & (dv // 4 - 1):
        raise ValueError(f"kv_lora_rank {dv} must be 4 x a power of two")
    if rope <= 0 or (rope & (rope - 1)):
        raise ValueError(f"rope slice (DQ - kv_lora_rank) must be a power of two, got {rope}")
    return q_abs, kv, positions, t_count, heads, dq, dv


def swa_dense_fwd(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    positions: torch.Tensor,
    *,
    window: int = DEFAULT_LOCAL_TOKENS,
    init_tokens: int = DEFAULT_INIT_TOKENS,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    scale: Optional[float] = None,
    out_dtype: Optional[torch.dtype] = None,
    block_m: int = 32,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    heads_per_prog: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense sliding-window + sink attention over the latent KV buffer.

    Args:
        q_abs: ``[T, H, DQ]`` absorbed queries (``DQ = kv_lora_rank + rope``).
        kv: ``[N, DQ]`` latent KV rows ``[c_kv ; k_rope]``.
        positions: ``[T]`` int32/int64 absolute positions of the query rows.
        window: sliding-window size (``index_local_tokens``).
        init_tokens: number of sink tokens forced into the attended set.
        kv_lora_rank: width of ``c_kv`` / attention output (``DV``).
        scale: softmax scale; defaults to ``DQ ** -0.5`` (RTP convention).
        out_dtype: output dtype; defaults to ``q_abs.dtype``.
        block_m: query rows per program (>= 16, the ``tl.dot`` M minimum).
        block_n: KV slots per score block (>= 16).
        num_warps: warps per program.
        num_stages: software pipelining stages for the KV range loop.
        heads_per_prog: heads covered by one program (must divide ``block_m``
            and ``H``). ``block_m // heads_per_prog`` tokens share one KV window
            per program, so the window is read ``H / heads_per_prog`` times per
            token; ``heads_per_prog == H`` reads it exactly once (one token x all
            heads per program). ``None`` auto-selects ``gcd(H, block_m)``, which
            measured fastest on SM86/SM10x (fewest KV re-reads).

    Returns:
        ``(out, lse)`` with ``out`` of shape ``[T, H, kv_lora_rank]`` in
        ``out_dtype`` and float32 natural-log LSE ``[T, H]``. Rows with an empty
        attended set (possible for negative positions only) yield zeros and
        ``-inf``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if init_tokens < 0:
        raise ValueError(f"init_tokens must be >= 0, got {init_tokens}")
    if block_m < 16 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a power of two >= 16, got {block_m}")
    if block_n < 16 or block_n & (block_n - 1):
        raise ValueError(f"block_n must be a power of two >= 16, got {block_n}")
    if heads_per_prog is not None and (
        heads_per_prog < 1 or block_m % heads_per_prog
    ):
        raise ValueError(
            f"heads_per_prog must divide block_m, got {heads_per_prog} and {block_m}"
        )

    q_abs, kv, positions, t_count, heads, dq, dv = _validate(
        q_abs, kv, positions, kv_lora_rank
    )
    if heads_per_prog is None:
        heads_per_prog = math.gcd(heads, block_m)
    if heads % heads_per_prog:
        raise ValueError(f"heads_per_prog must divide H={heads}, got {heads_per_prog}")
    resolved_out_dtype = out_dtype or q_abs.dtype
    if resolved_out_dtype not in _OUT_DTYPES:
        raise ValueError(f"unsupported out_dtype {resolved_out_dtype}")

    out = torch.empty((t_count, heads, dv), device=q_abs.device, dtype=resolved_out_dtype)
    lse = torch.empty((t_count, heads), device=q_abs.device, dtype=torch.float32)
    if t_count == 0:
        return out, lse

    tokens_per_prog = block_m // heads_per_prog
    grid = (triton.cdiv(t_count, tokens_per_prog), heads // heads_per_prog)
    _swa_dense_fwd_kernel[grid](
        q_abs,
        kv,
        positions,
        out,
        lse,
        t_count,
        kv.shape[0],
        scale if scale is not None else dq**-0.5,
        DQ=dq,
        DV=dv,
        ROPE=dq - dv,
        H=heads,
        WINDOW=window,
        INIT=init_tokens,
        BLOCK_M=block_m,
        HEADS_PER_PROG=heads_per_prog,
        BLOCK_N=block_n,
        OUT_DTYPE=_OUT_DTYPES[resolved_out_dtype],
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse


def _check_parts(
    parts: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, ...]:
    """Validate a partial-attention list and return the ``out`` tensors."""
    if not parts:
        raise ValueError("merge requires at least one part")
    outs = [part[0] for part in parts]
    lses = [part[1] for part in parts]
    ref_out_shape = outs[0].shape
    if len(ref_out_shape) != 3:
        raise ValueError(f"partial out must be [T, H, DV], got {tuple(ref_out_shape)}")
    for out_i, lse_i in zip(outs, lses):
        if out_i.shape != ref_out_shape:
            raise ValueError("all partial outputs must share one shape")
        if lse_i.shape != ref_out_shape[:2]:
            raise ValueError(f"partial lse must be [T, H], got {tuple(lse_i.shape)}")
        if out_i.device != outs[0].device:
            raise ValueError("all partial outputs must live on one device")
    return tuple(outs)


def merge_attention(
    parts: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge partial attentions with the standard online-softmax rescaling.

    Args:
        parts: sequence of ``(out_i [T, H, DV], lse_i [T, H])`` partial results
            of attention over disjoint slot sets of the same query rows. Any
            dtype is accepted; all math is done in float32.

    Returns:
        ``(out, lse)`` -- the fused attention over the union of the slot sets,
        with ``out`` in the dtype of ``parts[0][0]`` and float32 natural-log
        ``lse``. Rows whose partial LSEs are all ``-inf`` yield zero output and
        ``-inf`` LSE.

    Notes:
        ``M = max_i lse_i``; ``denom = sum_i exp(lse_i - M)``;
        ``out = sum_i out_i * exp(lse_i - M) / denom``; ``lse = M + log(denom)``.
        This is the HFA merge used to recombine the SWA and SFA streams; the
        CUDA path is :func:`merge_attention_triton`.
    """
    outs = _check_parts(parts)
    lses = [part[1] for part in parts]

    lse_stack = torch.stack([lse.float() for lse in lses], dim=0)
    out_stack = torch.stack([out.float() for out in outs], dim=0)
    m = lse_stack.max(dim=0).values
    m_safe = torch.where(m == _NEG_INF, torch.zeros_like(m), m)
    w = torch.where(
        lse_stack == _NEG_INF, torch.zeros_like(lse_stack), torch.exp(lse_stack - m_safe)
    )
    denom = w.sum(dim=0)
    nonempty = denom > 0
    denom_safe = torch.where(nonempty, denom, torch.ones_like(denom))
    merged = (out_stack * w.unsqueeze(-1)).sum(dim=0) / denom_safe.unsqueeze(-1)
    lse = torch.where(nonempty, m + torch.log(denom_safe), torch.full_like(m, _NEG_INF))
    return merged.to(outs[0].dtype), lse


@triton.jit
def _merge_attention_kernel(
    OUTS,
    LSES,
    OUT,
    LSE,
    T,
    H: tl.constexpr,
    DV: tl.constexpr,
    P: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """Fuse ``P`` partial ``(out [T, H, DV], lse [T, H])`` pairs in one pass.

    One program per (token block, head). Each partial is read exactly once (the
    per-part weights are recomputed from the tiny ``lse`` vector per DV chunk),
    the merged row is written once, and rows whose partial LSEs are all ``-inf``
    come out as zeros with ``-inf`` LSE.
    """
    pid_m = tl.program_id(0)
    head = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_ok = rows < T
    lse_base = rows * H + head

    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    for i in tl.static_range(P):
        lse_i = tl.load(LSES[i] + lse_base, mask=row_ok, other=float("-inf"))
        m = tl.maximum(m, lse_i.to(tl.float32))
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    for i in tl.static_range(P):
        lse_i = tl.load(LSES[i] + lse_base, mask=row_ok, other=float("-inf"))
        denom += tl.exp(lse_i.to(tl.float32) - m_safe)
    denom_safe = tl.where(denom > 0, denom, 1.0)

    for d0 in range(0, DV, BLOCK_DV):
        offs_d = d0 + tl.arange(0, BLOCK_DV)
        d_ok = offs_d < DV
        out_offs = rows[:, None] * (H * DV) + head * DV + offs_d[None, :]
        acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
        for i in tl.static_range(P):
            lse_i = tl.load(LSES[i] + lse_base, mask=row_ok, other=float("-inf"))
            w = tl.exp(lse_i.to(tl.float32) - m_safe) / denom_safe
            o_i = tl.load(
                OUTS[i] + out_offs,
                mask=row_ok[:, None] & d_ok[None, :],
                other=0.0,
            )
            acc += o_i.to(tl.float32) * w[:, None]
        tl.store(OUT + out_offs, acc.to(OUT_DTYPE), mask=row_ok[:, None] & d_ok[None, :])

    lse = tl.where(denom > 0, m + tl.log(denom_safe), float("-inf"))
    tl.store(LSE + lse_base, lse, mask=row_ok)


def merge_attention_triton(
    parts: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    block_m: int = 16,
    block_dv: int = 512,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Triton merge of partial attentions (single read/write pass).

    Same contract as :func:`merge_attention` (which stays as the CPU/fallback
    path): ``M = max_i lse_i``; ``denom = sum_i exp(lse_i - M)``;
    ``out = sum_i out_i * exp(lse_i - M) / denom``; ``lse = M + log(denom)``.
    Returns ``out`` in the dtype of ``parts[0][0]``.

    Args:
        parts: sequence of ``(out_i [T, H, DV], lse_i [T, H])`` CUDA tensors.
        block_m: token rows per program.
        block_dv: ``c_kv`` columns per program (a tail chunk is masked).
        num_warps: warps per program.
        num_stages: software pipelining stages for the partial loop.
    """
    outs = _check_parts(parts)
    lses = tuple(part[1] for part in parts)
    t_count, heads, dv = outs[0].shape
    if not outs[0].is_cuda:
        raise ValueError("merge_attention_triton requires CUDA tensors")
    if block_m < 1 or block_dv < 1:
        raise ValueError("block_m and block_dv must be >= 1")

    outs = tuple(out.contiguous() for out in outs)
    lses = tuple(lse.contiguous() for lse in lses)
    out = torch.empty_like(outs[0])
    lse = torch.empty((t_count, heads), device=outs[0].device, dtype=torch.float32)
    if t_count == 0:
        return out, lse

    resolved_out_dtype = outs[0].dtype
    if resolved_out_dtype not in _OUT_DTYPES:
        raise ValueError(f"unsupported out dtype {resolved_out_dtype}")
    grid = (triton.cdiv(t_count, block_m), heads)
    _merge_attention_kernel[grid](
        outs,
        lses,
        out,
        lse,
        t_count,
        H=heads,
        DV=dv,
        P=len(outs),
        BLOCK_M=block_m,
        BLOCK_DV=block_dv,
        OUT_DTYPE=_OUT_DTYPES[resolved_out_dtype],
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse