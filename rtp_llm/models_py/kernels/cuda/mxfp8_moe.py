"""MXFP8 contiguous grouped MoE for MiniMax-M3.

Runs the full per-expert FFN (gate/up grouped GEMM -> SwiGLU-OAI -> down
grouped GEMM) through DeepGEMM's ``m_grouped_fp8_fp4_gemm_nt_contiguous`` with
``recipe=(1, 32)``. Weight layout matches ``stack_moe_w1`` = ``[up_proj,
gate_proj]``:

  * ``w1_e4m3``: ``[E, 2*moe_inter, hidden]`` (up|gate)  + packed int32 scale
  * ``w2_e4m3``: ``[E, hidden, moe_inter]`` (down)        + packed int32 scale
"""

from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_grouped_gemm
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.common.swiglu_oai import swiglu_oai_torch


def _contiguous_alignment() -> int:
    import deep_gemm

    return deep_gemm.get_m_alignment_for_contiguous_layout()


def mxfp8_moe_contiguous_ffn(
    x: torch.Tensor,
    m_indices: torch.Tensor,
    w1_e4m3: torch.Tensor,
    w1_scale_packed: torch.Tensor,
    w2_e4m3: torch.Tensor,
    w2_scale_packed: torch.Tensor,
    alpha: Optional[float],
    limit: Optional[float],
    gate_first: bool = False,
) -> torch.Tensor:
    """x: [T, hidden] (permuted, per-expert blocks padded to 128). Returns [T, hidden]."""
    upgate = mxfp8_grouped_gemm(x, w1_e4m3, w1_scale_packed, m_indices)  # [T, 2*inter]
    if alpha is not None and limit is not None:
        act = swiglu_oai_torch(upgate, alpha, limit, gate_first=gate_first)
    else:
        act = torch.empty(
            upgate.size(0), upgate.size(1) // 2, device=upgate.device, dtype=upgate.dtype
        )
        silu_and_mul(act, upgate)
    down = mxfp8_grouped_gemm(act.contiguous(), w2_e4m3, w2_scale_packed, m_indices)
    return down


def mxfp8_moe_forward(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_e4m3: torch.Tensor,
    w1_scale_packed: torch.Tensor,
    w2_e4m3: torch.Tensor,
    w2_scale_packed: torch.Tensor,
    num_experts: int,
    alpha: Optional[float] = None,
    limit: Optional[float] = None,
    gate_first: bool = False,
) -> torch.Tensor:
    """Full MXFP8 MoE: route -> contiguous per-expert FFN -> weighted combine.

    hidden: ``[M, K]`` bf16. topk_ids/topk_weights: ``[M, top_k]``. Returns
    ``[M, K]`` bf16. Tokens are scattered into a contiguous per-expert layout
    (each expert block padded to 128), run through the grouped fp8_fp4 GEMMs,
    then gathered back with the router weights.
    """
    M, K = hidden.shape
    top_k = topk_ids.size(1)
    align = _contiguous_alignment()
    device = hidden.device

    flat_ids = topk_ids.reshape(-1).to(torch.long)
    flat_src = (
        torch.arange(M, device=device).repeat_interleave(top_k)
    )  # token idx per assignment
    flat_w = topk_weights.reshape(-1).to(torch.float32)

    # Under expert parallelism the router (recompute_topk) remaps expert IDs to
    # this rank's local range and marks non-local assignments as -1. Drop those:
    # the experts that own them live on other ranks and their contribution is
    # merged back via the TP all-reduce in the router's finalize(). At ep_size==1
    # nothing is marked -1, so this is a no-op.
    valid = flat_ids >= 0
    if not bool(valid.all()):
        flat_ids = flat_ids[valid]
        flat_src = flat_src[valid]
        flat_w = flat_w[valid]
        if flat_ids.numel() == 0:
            return torch.zeros(M, K, device=device, dtype=torch.bfloat16)

    counts = torch.bincount(flat_ids, minlength=num_experts)
    padded = ((counts + align - 1) // align * align).to(torch.long)
    expert_start = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
    expert_start[1:] = torch.cumsum(padded, 0)
    all_tokens = int(expert_start[-1].item())

    if all_tokens == 0:
        return torch.zeros(M, K, device=device, dtype=torch.bfloat16)

    x_perm = torch.zeros(all_tokens, K, device=device, dtype=torch.bfloat16)
    m_indices = torch.zeros(all_tokens, device=device, dtype=torch.int32)
    row_of_assign = torch.empty(flat_ids.numel(), device=device, dtype=torch.long)

    sort_idx = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[sort_idx]
    counts_list = counts.tolist()
    starts_list = expert_start.tolist()
    padded_list = padded.tolist()
    pos = 0
    for e in range(num_experts):
        c = counts_list[e]
        s = starts_list[e]
        p = padded_list[e]
        if p > 0:
            m_indices[s : s + p] = e
        if c > 0:
            idxs = sort_idx[pos : pos + c]
            x_perm[s : s + c] = hidden[flat_src[idxs]]
            row_of_assign[idxs] = s + torch.arange(c, device=device, dtype=torch.long)
            pos += c

    out_perm = mxfp8_moe_contiguous_ffn(
        x_perm,
        m_indices,
        w1_e4m3,
        w1_scale_packed,
        w2_e4m3,
        w2_scale_packed,
        alpha,
        limit,
        gate_first=gate_first,
    )

    out_assign = out_perm[row_of_assign].to(torch.float32) * flat_w.unsqueeze(1)
    output = torch.zeros(M, K, device=device, dtype=torch.float32)
    output.index_add_(0, flat_src, out_assign)
    return output.to(torch.bfloat16)
