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
    # wrapper_p2=None => pass2 走 run_b_rows_eager(零 plan)。H20 实测: 合批
    # 150-seq 下 eager 的 padded gather(~1.8ms) 贵过省下的 plan(0.33ms), 默认
    # 保留 flashinfer pass2; eager 留给单请求/小批或无 flashinfer 的场景。
    if schedule.has_b and wrapper_p2 is not None:
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


def run_b_rows_eager(
    schedule: Any,
    q: torch.Tensor,  # [total, h, d] PERMUTED layout
    k: torch.Tensor,  # [total, kvh, d] PERMUTED layout
    v: torch.Tensor,  # [total, kvh, d] PERMUTED layout
) -> torch.Tensor:
    """pass 2 的 plan-free 实现: B 查询(每序列 ~几个 token)对全序列 attention。

    纯 torch, 无 flashinfer plan/wrapper: B 段极小, padded 批量 einsum 即可
    (bf16 matmul 默认 fp32 累加, 数值口径与 kernel 相当)。GQA 时按组广播 kv。
    Returns: [total_b, h, d], 行序与 schedule.b_rows 一致(按序列、段内顺序)。
    """
    h = q.shape[1]
    kvh = k.shape[1]
    kh = k[schedule.kv_pad_idx]  # [N, Lmax, kvh, d]
    vh = v[schedule.kv_pad_idx]
    qb = q[schedule.q_pad_idx]  # [N, Bmax, h, d]
    if kvh != h:  # GQA: 广播 kv 头到 q 头
        rep = h // kvh
        kh = kh.repeat_interleave(rep, dim=2)
        vh = vh.repeat_interleave(rep, dim=2)
    scale = q.shape[-1] ** -0.5
    lg = torch.einsum("nbhd,nlhd->nbhl", qb, kh) * scale  # [N, Bmax, h, Lmax]
    lg = lg.masked_fill(~schedule.kv_pad_mask[:, None, None, :], float("-inf"))
    p = lg.float().softmax(-1).to(q.dtype)
    out = torch.einsum("nbhl,nlhd->nbhd", p, vh)  # [N, Bmax, h, d]
    return out[schedule.q_pad_mask]  # [total_b, h, d] 按 (seq, within) 序


def run_two_pass(
    wrapper_p1: Any,
    wrapper_p2: Any,  # 兼容旧签名; eager pass2 后不再使用, 传 None 即可
    schedule: Any,
    q: torch.Tensor,  # [total, num_qo_heads, head_dim], PERMUTED layout
    k: torch.Tensor,  # [total, num_kv_heads, head_dim], PERMUTED layout
    v: torch.Tensor,  # [total, num_kv_heads, head_dim], PERMUTED layout
) -> torch.Tensor:
    """Attention output in the PERMUTED layout, [total, num_qo_heads, head_dim]."""
    out = wrapper_p1.run(q, k, v)
    if schedule.has_b:
        if wrapper_p2 is not None:
            out[schedule.b_rows] = wrapper_p2.run(q[schedule.b_rows], k, v)
        else:
            out[schedule.b_rows] = run_b_rows_eager(schedule, q, k, v)
    return out
