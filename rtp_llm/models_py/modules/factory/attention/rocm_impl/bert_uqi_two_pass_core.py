"""AITER two-pass encoder attention. No RTP native ops or FlashInfer imports.

Inputs/outputs use the shared schedule's per-sequence [QI, profile] layout.
Metadata is prepared once per request, outside the layer loop. Empty segments
are removed on the host, so no kernel depends on zero-length varlen support.
"""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class VarlenPass:
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    max_q: int
    max_k: int


@dataclass
class TwoPassPlan:
    first: VarlenPass
    second: Optional[VarlenPass]
    b_rows: Optional[torch.Tensor]
    kv_rows: Optional[torch.Tensor]


def _indptr(lengths, device):
    ptr = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    ptr[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    return ptr.to(device, non_blocking=True)


def prepare_two_pass(schedule, device):
    """Read CPU schedule only; upload compact indptrs/indices once."""
    ptr = schedule.qo_indptr_p1
    if ptr.device.type != "cpu":
        raise ValueError("BERT UQI schedule indptrs must be on CPU")
    lengths = (ptr[1:] - ptr[:-1]).tolist()
    nonempty = [n for n in lengths if n > 0]
    cu = _indptr(nonempty, device)
    first = VarlenPass(cu, cu, max(nonempty, default=0), max(nonempty, default=0))
    if not schedule.has_b:
        return TwoPassPlan(first, None, None, None)
    qp, kp = schedule.qo_indptr_p2, schedule.kv_indptr_p2
    if qp.device.type != "cpu" or kp.device.type != "cpu":
        raise ValueError("BERT UQI schedule indptrs must be on CPU")
    qlens = (qp[1:] - qp[:-1]).tolist()
    klens = (kp[1:] - kp[:-1]).tolist()
    active = [i for i, n in enumerate(qlens) if n > 0]
    qlens_active = [qlens[i] for i in active]
    klens_active = [klens[i] for i in active]
    second = VarlenPass(
        _indptr(qlens_active, device), _indptr(klens_active, device),
        max(qlens_active), max(klens_active),
    )
    # Queries are already packed by b_rows. Compact KV only for mixed batches;
    # otherwise use the original KV directly (no extra per-layer gather).
    kv_rows = None
    if any(n > 0 and q == 0 for q, n in zip(qlens, klens)):
        offsets = kp.tolist()
        kv_rows = torch.cat([
            torch.arange(offsets[i], offsets[i + 1], dtype=torch.long)
            for i in active
        ]).to(device, non_blocking=True)
    return TwoPassPlan(first, second, schedule.b_rows, kv_rows)


def run_two_pass(plan, q, k, v, attention=None):
    """Run noncausal attention; B queries see all A+B, A queries only A."""
    if q.shape[0] == 0:
        return torch.empty_like(q)
    if attention is None:
        from aiter import flash_attn_varlen_func
        attention = flash_attn_varlen_func
    # Split projected QKV and strided callers need not be contiguous. AITER's
    # fast path requires the innermost dimension contiguous, not a full copy.
    q, k, v = (t.contiguous() if t.stride(-1) != 1 else t for t in (q, k, v))

    def run(meta, query, key, value):
        return attention(
            query, key, value, meta.cu_q, meta.cu_k, meta.max_q, meta.max_k,
            dropout_p=0.0, causal=False,
        )

    out = run(plan.first, q, k, v)
    if plan.second is not None:
        kb, vb = k, v
        if plan.kv_rows is not None:
            kb = k.index_select(0, plan.kv_rows)
            vb = v.index_select(0, plan.kv_rows)
        out_b = run(plan.second, q.index_select(0, plan.b_rows), kb, vb)
        out.index_copy_(0, plan.b_rows, out_b)
    return out
