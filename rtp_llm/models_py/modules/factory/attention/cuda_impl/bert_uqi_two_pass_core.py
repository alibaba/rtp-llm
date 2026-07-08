"""Two-pass user-profile attention core — torch + flashinfer only, no rtp_llm deps.

Numerics of the block mask M=[[1,0],[1,1]] via two native ragged passes over the
PERMUTED [A...|B...] per-sequence layout (see block_mask.build_bert_uqi_two_pass_schedule):

  pass 1: one ragged call over 2N segments [A_0][B_0]...[A_{N-1}][B_{N-1}] —
          each segment attends itself. A rows are exact (A sees only A); the
          tiny B self-attention rows are garbage, overwritten by pass 2.
  pass 2: B queries attend the FULL sequence (kv = the whole permuted q/k/v —
          attention is permutation-invariant over keys), scattered over the
          garbage rows.

No custom_mask anywhere => FlashInfer keeps FA3 eligibility and skips the
mask packbits path entirely. plan() is fed CPU int32 indptrs so its internal
`.to("cpu")` is a no-op — zero device syncs.

The `schedule` argument is duck-typed (block_mask.BertUqiTwoPassSchedule);
this module deliberately avoids importing rtp_llm packages so tests can load
it by file path without the package __init__ chain (compiled .so).
"""

from typing import Any

import torch


def plan_two_pass(
    wrapper_p1: Any,
    wrapper_p2: Any,
    schedule: Any,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_data_type: torch.dtype,
) -> None:
    """Plan both passes. schedule.qo_indptr_* must be CPU int32 (sync-free plan)."""
    wrapper_p1.plan(
        schedule.qo_indptr_p1,
        schedule.qo_indptr_p1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=False,
        q_data_type=q_data_type,
    )
    if schedule.has_b:
        wrapper_p2.plan(
            schedule.qo_indptr_p2,
            schedule.kv_indptr_p2,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            head_dim_vo=head_dim,
            causal=False,
            q_data_type=q_data_type,
        )


def run_two_pass(
    wrapper_p1: Any,
    wrapper_p2: Any,
    schedule: Any,
    q: torch.Tensor,  # [total, num_qo_heads, head_dim], PERMUTED layout
    k: torch.Tensor,  # [total, num_kv_heads, head_dim], PERMUTED layout
    v: torch.Tensor,  # [total, num_kv_heads, head_dim], PERMUTED layout
) -> torch.Tensor:
    """Attention output in the PERMUTED layout, [total, num_qo_heads, head_dim]."""
    out = wrapper_p1.run(q, k, v)
    if schedule.has_b:
        q_b = q[schedule.b_rows]
        out_b = wrapper_p2.run(q_b, k, v)
        out[schedule.b_rows] = out_b
    return out
