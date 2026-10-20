"""Split-K windowed prefill attention Triton kernel (causal + sliding window).

Contract 4c (opt r8, approaching the >40% windowed-MFU gate): the DiffusionGemma
TTFT prefill's 25 sliding-layer invocations run Q=4096 with window 1024 through
flashinfer fa3 ragged prefill; its CTA packing yields only Q/128 = 32 CTAs per
layer on the 78 H20 SMs (41% occupancy, r7 gate row: 29.19% windowed MFU vs
39.56% full-causal reference). This kernel re-parallelizes the same problem:

  * per-q-head CTAs: grid (total_q_tiles, SPLIT_K, num_q_heads) instead of
    one CTA packing all heads — 16x the CTA count at equal tile size;
  * KV-dimension split: each q-tile's attended KV span
    [max(lo, m0-(window-1)), min(hi, m0+BM)) is cut into SPLIT_K contiguous
    chunks, each producing a partial (O_s, lse_s); a merge kernel combines
    them via LSE (the flashinfer merge_state math). SPLIT_K=1 writes the
    final output directly (no merge launch).

Semantics: ragged layout [T, H, D] with cu_seqlens request spans; request
b's query row m (global token index) attends keys j in the same request
with j <= m (causal) and j >= m-(window-1) (window>0). Tiles are
per-request (tile_cu + req_of_tile maps, computed on device), so any
request alignment is handled. GQA: q head h reads kv head h // (HQ/HKV).
bf16 in/out, fp32 softmax accumulation with the exp2/log2 running-max
form, sm_scale applied in fp32 — the same numeric class as the fa3 kernel
(the A/B test asserts agreement to bf16 noise: r8 first-remote measured
max_diff 3.9e-3 on absmax 3.297 = 0.0012 relative at split_k=1).

LSE convention: emitted in flashinfer's cascade convention (base-2
logsumexp, ``lse = m + log2(l)`` with the exp2 running max) and [T, HQ]
fp32 layout, matching what ``BatchPrefillWithRaggedKVCacheWrapper.run(
return_lse=True)`` returns — verified empirically (r8 first-remote: lse
median ratio 1.0000, shapes identical) — so the AttnOp's existing
merge_state_in_place with the paged-prefix segment composes unchanged.

Tuning: BM/BN/num_warps/num_stages are wrapper arguments (the tile
geometry follows BM; recompiles per constexpr combination, the standard
Triton JIT cache). The bench sweeps
(BM=128, BN=64, warps=16, stages=2) and (BM=64, BN=64, warps=8, stages=3);
per-CTA fixed-cost amortization (the 4c structural finding: windowed work
per CTA halves while fixed costs do not) favors the larger tile.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _splitk_window_attn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_ptr,
    tile_cu_ptr,
    req_of_tile_ptr,
    o_ptr,
    lse_ptr,
    sm_scale,
    window,
    T,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    SPLIT: tl.constexpr,
    MERGE: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
):
    # grid: (total_q_tiles over all requests, SPLIT, HQ) or the head-major
    # order (HQ, SPLIT, total_q_tiles) when HEAD_MAJOR (L2: consecutive
    # CTAs share the SAME k/v span across the GQA group instead of
    # overlapping spans of the same head). One CTA: one per-request q-tile
    # x one kv chunk x one q head. The tile -> request map and per-request
    # tile offsets are precomputed on device (torch.repeat_interleave /
    # cumsum — no host sync, no in-kernel scan).
    if HEAD_MAJOR:
        pid_h = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_m = tl.program_id(2)
    else:
        pid_m = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_h = tl.program_id(2)

    req = tl.load(req_of_tile_ptr + pid_m)
    lo = tl.load(cu_ptr + req)
    hi = tl.load(cu_ptr + req + 1)
    m0 = lo + (pid_m - tl.load(tile_cu_ptr + req)) * BM

    offs_m = m0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < hi
    q = tl.load(
        q_ptr + offs_m[:, None].to(tl.int64) * (HQ * D) + pid_h * D + offs_d[None, :],
        mask=row_ok[:, None],
        other=0.0,
    )
    kvh = pid_h // GROUP
    # request-relative row positions (hoisted: constant across the kv loop)
    rel_m = offs_m - lo

    # this tile's kv span: causal upper bound m0+BM (clipped to the request),
    # window lower bound max(lo, m0-(window-1))
    kv_hi = tl.minimum(m0 + BM, hi)
    kv_lo = lo
    if HAS_WINDOW:
        kv_lo = tl.maximum(kv_lo, m0 - (window - 1))
    span = tl.maximum(kv_hi - kv_lo, 0)
    # split the span into SPLIT contiguous chunks (widths differ by <=1);
    # chunks beyond the span are empty (l=0 partials, weight 0 at merge).
    chunk = tl.cdiv(span, SPLIT)
    k_start = kv_lo + pid_s * chunk
    k_end = tl.minimum(k_start + chunk, kv_hi)

    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    # combined score scale: exp2-domain softmax (p = exp2(qk*log2e - m)),
    # so the emitted lse is m + log2(l), the flashinfer cascade convention.
    qk_scale = sm_scale * 1.4426950408889634

    for start_n in range(k_start, k_end, BN):
        offs_n = start_n + tl.arange(0, BN)
        kv_mask = offs_n < k_end
        k = tl.load(
            k_ptr + offs_n[:, None].to(tl.int64) * (HKV * D) + kvh * D + offs_d[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale
        # NOTE: no data-dependent branch in the loop body — Triton's
        # software pipeliner requires a straight-line body to overlap the
        # k/v loads with the dots (an in-loop scalar `if` disabled it and
        # cost ~1.5x in the r8 second-remote measurement).
        rel_n = offs_n - lo
        keep = kv_mask[None, :] & row_ok[:, None]
        if IS_CAUSAL:
            keep = keep & (rel_n[None, :] <= rel_m[:, None])
        if HAS_WINDOW:
            keep = keep & (rel_n[None, :] >= rel_m[:, None] - (window - 1))
        qk = tl.where(keep, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # rows fully masked so far: keep m finite-safe, alpha=0
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.math.exp2(m_i - m_safe))
        p = tl.math.exp2(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = tl.load(
            v_ptr + offs_n[:, None].to(tl.int64) * (HKV * D) + kvh * D + offs_d[None, :],
            mask=kv_mask[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # partial state: self-normalized O_s plus lse_s = m + log2(l)
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    o_part = acc / l_safe[:, None]
    lse = tl.where(l_i == 0.0, float("-inf"), m_i + tl.math.log2(l_safe))

    if MERGE:
        # per-split partial buffers: [SPLIT, T, HQ(, D)]
        base = pid_s.to(tl.int64) * (T * HQ) + offs_m.to(tl.int64) * HQ + pid_h
        tl.store(
            o_ptr + base[:, None] * D + offs_d[None, :],
            o_part.to(o_ptr.dtype.element_ty),
            mask=row_ok[:, None],
        )
        tl.store(lse_ptr + base, lse, mask=row_ok)
    else:
        base = offs_m.to(tl.int64) * HQ + pid_h
        tl.store(
            o_ptr + base[:, None] * D + offs_d[None, :],
            o_part.to(o_ptr.dtype.element_ty),
            mask=row_ok[:, None],
        )
        tl.store(lse_ptr + base, lse, mask=row_ok)


@triton.jit
def _splitk_merge_kernel(
    o_part_ptr,
    lse_ptr,
    o_ptr,
    lse_out_ptr,
    T,
    HQ: tl.constexpr,
    D: tl.constexpr,
    SPLIT: tl.constexpr,
    BM: tl.constexpr,
):
    # grid: (rows/BM, HQ). Merge the SPLIT partial states per row:
    #   M = max(lse_s); w_s = 2^(lse_s - M); O = sum_s w_s O_s / sum_s w_s;
    #   lse = M + log2(sum_s w_s)  (the flashinfer merge_state math).
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < T
    m_max = tl.full([BM], float("-inf"), dtype=tl.float32)
    for s in range(SPLIT):
        base = s * T * HQ + offs_m.to(tl.int64) * HQ + pid_h
        s_val = tl.load(lse_ptr + base, mask=row_ok, other=float("-inf"))
        m_max = tl.maximum(m_max, s_val)
    m_safe = tl.where(m_max == float("-inf"), 0.0, m_max)
    denom = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, D], dtype=tl.float32)
    for s in range(SPLIT):
        base = s * T * HQ + offs_m.to(tl.int64) * HQ + pid_h
        s_val = tl.load(lse_ptr + base, mask=row_ok, other=float("-inf"))
        w = tl.where(s_val == float("-inf"), 0.0, tl.math.exp2(s_val - m_safe))
        o_part = tl.load(
            o_part_ptr + base[:, None] * D + offs_d[None, :],
            mask=row_ok[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += w[:, None] * o_part
        denom += w
    d_safe = tl.where(denom == 0.0, 1.0, denom)
    out = acc / d_safe[:, None]
    out_base = offs_m.to(tl.int64) * HQ + pid_h
    tl.store(
        o_ptr + out_base[:, None] * D + offs_d[None, :],
        out.to(o_ptr.dtype.element_ty),
        mask=row_ok[:, None],
    )
    lse_out = tl.where(denom == 0.0, float("-inf"), m_max + tl.math.log2(d_safe))
    tl.store(lse_out_ptr + out_base, lse_out, mask=row_ok)


def splitk_window_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    causal: bool,
    window: int,
    sm_scale: float = 1.0,
    split_k: int = 1,
    block_m: int = 128,
    block_n: int = 64,
    num_warps: int = 8,
    num_stages: int = 2,
    head_major: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Windowed causal prefill attention over a ragged [T, H, D] batch.

    Returns (out [T, HQ, D] bf16, lse [T, HQ] fp32 in the flashinfer
    cascade base-2 convention). ``split_k`` KV chunks per q-tile with an
    LSE merge pass (split_k=1 -> single launch, no merge). ``block_m`` /
    ``block_n`` / ``num_warps`` / ``num_stages`` are the tile tuning knobs
    (default BM=128/warps=8: two warpgroups x m64 — the wgmma-compatible
    M split for sm90; BM=64/warps=8 or BM=128/warps=16 cannot tile M=64
    per warpgroup and fall back to slower MMA paths — r8 remote
    measurements: 16.6-18.9% MFU vs fa3 29.26%).
    """
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v must be [T, H, D]")
    T, HQ, D = q.shape
    HKV = k.shape[1]
    if k.shape[0] != T or v.shape != k.shape:
        raise ValueError("k/v must share the ragged token count and head count")
    if HQ % HKV != 0:
        raise ValueError("q heads must be a multiple of kv heads")
    if D not in (64, 128, 256):
        raise ValueError("head_dim must be 64/128/256 for the tl.dot tile")
    if not q.is_cuda or q.dtype != torch.bfloat16:
        raise ValueError("splitk_window_attn requires CUDA bf16")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if cu_seqlens.device != q.device:
        cu_seqlens = cu_seqlens.to(q.device)
    cu = cu_seqlens.to(torch.int32).contiguous()
    B = cu.numel() - 1
    if B < 1:
        raise ValueError("cu_seqlens needs at least one request")
    out = torch.empty((T, HQ, D), dtype=q.dtype, device=q.device)
    lse = torch.empty((T, HQ), dtype=torch.float32, device=q.device)

    BM = int(block_m)
    BN = int(block_n)
    # per-request tile offsets + tile->request map (device-side; any
    # request alignment works; one host sync for the grid size).
    lens = cu[1:] - cu[:-1]
    tiles_per_req = (lens + (BM - 1)) // BM
    tile_cu = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    tile_cu[1:] = torch.cumsum(tiles_per_req, dim=0)
    req_of_tile = torch.repeat_interleave(
        torch.arange(B, dtype=torch.int32, device=q.device), tiles_per_req
    )
    total_tiles = int(req_of_tile.numel())
    if total_tiles < 1:
        raise ValueError("empty ragged batch")
    group = HQ // HKV
    merge = split_k > 1
    o_store = (
        torch.empty((split_k, T, HQ, D), dtype=q.dtype, device=q.device)
        if merge
        else out
    )
    lse_store = (
        torch.empty((split_k, T, HQ), dtype=torch.float32, device=q.device)
        if merge
        else lse
    )
    grid = (HQ, split_k, total_tiles) if head_major else (total_tiles, split_k, HQ)
    _splitk_window_attn_kernel[grid](
        q,
        k,
        v,
        cu,
        tile_cu,
        req_of_tile,
        o_store,
        lse_store,
        float(sm_scale),
        int(window),
        T,
        HQ=HQ,
        HKV=HKV,
        GROUP=group,
        D=D,
        BM=BM,
        BN=BN,
        IS_CAUSAL=causal,
        HAS_WINDOW=window > 0,
        SPLIT=split_k,
        MERGE=merge,
        HEAD_MAJOR=head_major,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if merge:
        _splitk_merge_kernel[(triton.cdiv(T, 64), HQ)](
            o_store,
            lse_store,
            out,
            lse,
            T,
            HQ=HQ,
            D=D,
            SPLIT=split_k,
            BM=64,
            num_warps=8,
        )
    _report_compilation()
    return out, lse


_compiled_reported: set = set()


def _report_compilation() -> None:
    """Log register/spill counts per compiled specialization (tuning
    evidence; once per specialization per process)."""
    try:
        for kernel in (_splitk_window_attn_kernel, _splitk_merge_kernel):
            cache = getattr(kernel, "cache", None) or getattr(kernel, "_cache", None)
            if not cache:
                continue
            for device_cache in cache.values() if isinstance(cache, dict) else []:
                entries = (
                    device_cache.values()
                    if isinstance(device_cache, dict)
                    else [device_cache]
                )
                for compiled in entries:
                    key = (getattr(compiled, "n_regs", -1), getattr(compiled, "n_spills", -1))
                    if key in _compiled_reported:
                        continue
                    _compiled_reported.add(key)
                    print(
                        f"[dgm-splitk] kernel={kernel.fn.__name__} "
                        f"n_regs={getattr(compiled, 'n_regs', '?')} "
                        f"n_spills={getattr(compiled, 'n_spills', '?')}",
                        flush=True,
                    )
    except Exception:
        pass
