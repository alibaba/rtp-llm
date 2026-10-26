"""Fused EPLB policy adapted from the DeepInfer Triton prototype (Apache-2.0).

One program per layer keeps replica selection and greedy placement on device.
The Torch prologue preserves stable sorting and inverse-map replica ordering.
Division rounding and reduction order must match policy.py: a single ULP can
change the expert or rank selected by argmin.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _redundant_replica_kernel(
    tpe_ptr,  # [S, L, E] int32
    tpe_diff_ptr,  # [S, L, E] float32, = tpe + arange(E) * 1e-4
    count_ptr,  # [L, E] int32, in/out, 初始为全 1
    ri_ptr,  # [L, R] int32 out, 每轮被复制的 logical expert
    slot_ptr,  # [L, R] int32 out, 每轮写入 l2p 的副本槽位(= 更新前 count)
    L,
    E,
    R,
    S: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    l = tl.program_id(0).to(tl.int64)
    offs_e = tl.arange(0, BLOCK_E)
    rows = tl.arange(0, S)
    mask_e = offs_e < E

    # 只读 live-in:[S, BLOCK_E];padding 列 tpe_diff=-inf、tpe=0、count=1
    tpe_diff = tl.load(
        tpe_diff_ptr + rows[:, None] * (L * E) + l * E + offs_e[None, :],
        mask=mask_e[None, :],
        other=-float("inf"),
    )
    tpe = tl.load(
        tpe_ptr + rows[:, None] * (L * E) + l * E + offs_e[None, :],
        mask=mask_e[None, :],
        other=0,
    )
    tpe_f = tpe.to(tl.float32)
    count = tl.load(count_ptr + l * E + offs_e, mask=mask_e, other=1)

    for i in range(R):
        count_f = count.to(tl.float32)
        count_p1 = (count + 1).to(tl.float32)
        # 除法必须用 div_rn:Triton `/` 编译为 div.full,约 3% 操作数差 1 ulp,
        # 会翻转下游精确同分的 argmin(EP×冗余度扫描 ep=32 完全重复场景实测)。
        # torch `/` 是 div.rn,libdevice.div_rn 与其逐位一致(探针 0 bit-diffs)。
        score = libdevice.div_rn(tpe_diff, count_f[None, :])  # [S, BLOCK_E]
        added = libdevice.div_rn(tpe_f, count_p1[None, :])  # [S, BLOCK_E]
        # score.max(-1):每 step 的最大每副本负载及其位置(first-index)
        am = tl.argmax(score, axis=1, tie_break_left=True)  # [S]
        om = tl.max(score, axis=1)  # [S]
        # 把 argmax 位置替换成 score_added 后重算 max:new_max = max(rest, added_at)
        eq = offs_e[None, :] == am[:, None]  # [S, BLOCK_E]
        rest = tl.max(tl.where(eq, -float("inf"), score), axis=1)
        added_at = tl.sum(tl.where(eq, added, 0.0), axis=1)  # 精确提取
        nm = tl.maximum(rest, added_at)  # [S]
        # values[s, e] = (e == am[s]) ? nm[s] : om[s]
        val = tl.where(eq, nm[:, None], om[:, None])  # [S, BLOCK_E]
        # values.sum(0):顺序累加;padding 列恒为 +inf,不会被 argmin 选中
        acc = tl.where(mask_e, 0.0, float("inf"))  # [BLOCK_E]
        for s in tl.static_range(S):
            row_s = tl.sum(tl.where(rows[:, None] == s, val, 0.0), axis=0)
            acc = acc + row_s
        chosen = tl.argmin(acc, axis=0, tie_break_left=True)
        count_before = tl.sum(tl.where(offs_e == chosen, count, 0), axis=0)
        tl.store(ri_ptr + l * R + i, chosen)
        tl.store(slot_ptr + l * R + i, count_before)
        count = count + tl.where(offs_e == chosen, 1, 0)
    tl.store(count_ptr + l * E + offs_e, count, mask=mask_e)


@triton.jit
def _greedy_placement_kernel(
    vectors_ptr,  # [L, E, S] float32
    eo_ptr,  # [L, P] int64, expert_order
    po_ptr,  # [L, P] int64, physical_order
    indices_ptr,  # [L, P] int64 out
    E,
    P,
    capacity,
    num_ranks,
    S: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    l = tl.program_id(0).to(tl.int64)
    offs_r = tl.arange(0, BLOCK_R)
    rows = tl.arange(0, S)

    # loop-carried 状态全部是单名 tile:loads [S, R], peaks [S], occ [R]
    loads = tl.zeros([S, BLOCK_R], dtype=tl.float32)
    peaks = tl.zeros([S], dtype=tl.float32)
    # padding rank 视为已满(occ == capacity),得分恒为 inf
    occ = tl.where(offs_r < num_ranks, 0, capacity)

    for k in range(P):
        e = tl.load(eo_ptr + l * P + k)
        v = tl.load(vectors_ptr + (l * E + e) * S + rows)  # [S]
        # candidates[s, r] = max(loads[s, r] + v[s], peaks[s])
        cand = tl.maximum(loads + v[:, None], peaks[:, None])  # [S, BLOCK_R]
        # scores[r] = sum_s cand[s, r],按 torch 的跨步配对蝶形顺序
        if S == 1:
            scores = tl.sum(cand, axis=0)
        elif S == 2:
            scores = tl.sum(tl.where(rows[:, None] == 0, cand, 0.0), axis=0) + tl.sum(
                tl.where(rows[:, None] == 1, cand, 0.0), axis=0
            )
        elif S == 4:
            even = tl.sum(
                tl.where((rows[:, None] == 0) | (rows[:, None] == 2), cand, 0.0), axis=0
            )
            odd = tl.sum(
                tl.where((rows[:, None] == 1) | (rows[:, None] == 3), cand, 0.0), axis=0
            )
            scores = even + odd
        scores = tl.where(occ == capacity, float("inf"), scores)
        rank = tl.argmin(scores, axis=0, tie_break_left=True)
        rank_mask = offs_r == rank
        occ_rank = tl.sum(tl.where(rank_mask, occ, 0), axis=0)
        po_k = tl.load(po_ptr + l * P + k)
        tl.store(indices_ptr + l * P + rank * capacity + occ_rank, po_k)
        occ = occ + tl.where(rank_mask, 1, 0)
        # loads[rank, :] += v;peaks = max(peaks, loads[rank] + v)
        loads = loads + tl.where(rank_mask[None, :], v[:, None], 0.0)
        new_row = tl.sum(
            tl.where(rank_mask[None, :], loads, 0.0), axis=1
        )  # [S] 精确提取
        peaks = tl.maximum(peaks, new_row)


def _batch_greedy_indices_fused(
    tokens_per_expert: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    logical_count: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """与 policy._batch_greedy_indices 相同的接口,贪心循环换成单个 Triton kernel。"""
    weights = tokens_per_expert.sum(dim=0, dtype=torch.float32) / logical_count
    physical_experts = physical_to_logical_map.to(torch.int64)
    physical_order = weights.gather(-1, physical_experts).argsort(
        dim=-1, descending=True, stable=True
    )
    expert_order = physical_experts.gather(-1, physical_order)
    vectors = tokens_per_expert.permute(1, 2, 0).to(
        dtype=torch.float32, memory_format=torch.contiguous_format, copy=True
    )
    vectors.div_(logical_count.unsqueeze(-1))

    num_layers, num_logical, num_steps = vectors.shape
    num_physical = physical_order.size(1)
    num_ranks = num_physical // capacity

    indices = torch.empty_like(physical_order)
    _greedy_placement_kernel[(num_layers,)](
        vectors,
        expert_order,
        physical_order,
        indices,
        num_logical,
        num_physical,
        capacity,
        num_ranks,
        S=num_steps,
        BLOCK_R=triton.next_power_of_2(num_ranks),
        num_warps=1,
    )
    return indices


def rebalance_experts_global_fused(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plan a validated contiguous CUDA int32 history with 1, 2 or 4 steps.

    Called by policy.rebalance_experts_global with the FP32 default dtype.
    Output ordering and floating-point ties follow the Torch implementation.
    """
    num_steps, num_moe_layers, num_logical_experts = tokens_per_expert.shape
    num_redundant_experts = num_physical_experts - num_logical_experts
    device = tokens_per_expert.device

    physical_to_logical_map = torch.empty(
        num_moe_layers, num_physical_experts, dtype=torch.int, device=device
    )
    logical_to_physical_map = torch.full(
        (num_moe_layers, num_logical_experts, num_redundant_experts + 1),
        -1,
        dtype=torch.int,
        device=device,
    )
    logical_count = torch.ones(
        num_moe_layers, num_logical_experts, dtype=torch.int, device=device
    )

    arange_layers = torch.arange(num_moe_layers, dtype=torch.int, device=device)
    arange_logical = torch.arange(num_logical_experts, dtype=torch.int, device=device)
    physical_to_logical_map[:, :num_logical_experts] = arange_logical
    logical_to_physical_map[:, :, 0] = arange_logical

    tokens_per_expert_all_diff = tokens_per_expert + arange_logical * 1e-4
    if num_redundant_experts > 0:
        L, E, R = num_moe_layers, num_logical_experts, num_redundant_experts
        ri = torch.empty(L, R, dtype=torch.int32, device=device)
        slot = torch.empty(L, R, dtype=torch.int32, device=device)
        _redundant_replica_kernel[(L,)](
            tokens_per_expert,
            tokens_per_expert_all_diff,
            logical_count,
            ri,
            slot,
            L,
            E,
            R,
            S=num_steps,
            BLOCK_E=triton.next_power_of_2(E),
            num_warps=4,
        )
        physical_to_logical_map[:, num_logical_experts:] = ri
        # 重建 logical_to_physical_map:l2p[l, ri[l,i], slot[l,i]] = num_logical + i
        pos = ri.to(torch.int64) * (R + 1) + slot.to(torch.int64)
        physical_ids = num_logical_experts + torch.arange(
            R, dtype=torch.int32, device=device
        )
        logical_to_physical_map.view(num_moe_layers, -1)[
            arange_layers.to(torch.int64).unsqueeze(1), pos
        ] = physical_ids

    if num_local_physical_experts > 1:
        indices = _batch_greedy_indices_fused(
            tokens_per_expert,
            physical_to_logical_map,
            logical_count,
            num_local_physical_experts,
        )
        physical_to_logical_map = physical_to_logical_map.gather(-1, indices)
        mask = logical_to_physical_map == -1
        logical_to_physical_map[mask] = 0
        logical_to_physical_map = (
            indices.argsort(-1)
            .gather(
                -1, logical_to_physical_map.view(num_moe_layers, -1).to(torch.int64)
            )
            .view_as(logical_to_physical_map)
            .to(torch.int)
        )
        logical_to_physical_map[mask] = -1

    return physical_to_logical_map, logical_to_physical_map, logical_count
