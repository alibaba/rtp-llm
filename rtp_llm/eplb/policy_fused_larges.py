"""Streaming EPLB decisions for statistics windows outside 1, 2 and 4.

The two reductions over steps use a fixed tree. Near-tied decisions can differ
from Torch, while each rank must compute identical maps from identical inputs.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# These constants also define the reduction tree and must agree across ranks.
_REPLICA_CHUNK_S = 16
_REPLICA_WARPS = 8
_REPLICA_TARGET_CTAS = 4096
_GREEDY_CHUNK_S = 512
_GREEDY_WARPS = 4


@triton.jit
def _replica_scan_kernel(
    tpe_ptr,  # [S, L, E] int32
    tpe_diff_ptr,  # [S, L, E] float32, = tpe + arange(E) * 1e-4
    count_ptr,  # [L, E] int32, decide kernel 每轮更新,scan 每轮重读
    partial_ptr,  # [L, n_split, E] float32 out, 本切片的 partial acc
    L,
    E,
    S,
    n_split,
    slice_rows,
    n_chunks,
    CHUNK_S: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    l = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    count = tl.load(count_ptr + l * E + offs_e, mask=mask_e, other=1)
    count_f = count.to(tl.float32)
    # partial[e] = sum_{s in slice} values[s, e];padding 列 +inf,由 decide 的 argmin 排除。
    # 归约顺序:切片内 chunk 顺序累加 + 块内 tl.sum 树,编译期固定
    acc = tl.where(mask_e, 0.0, float("inf"))
    for c in range(n_chunks):
        row = split * slice_rows + c.to(tl.int64) * CHUNK_S + tl.arange(0, CHUNK_S)
        # 行掩码必须同时卡 S 上界和切片上界:CHUNK_S > slice_rows 时最后一个 chunk
        # 会越过切片边界,把下一片的行重复计入 acc(S=100 时 slice_rows=15 < 16,
        # 每片多算 1 行,parity 大面积崩;S=1000 时 84 vs 96,每片多算 12 行)
        row_ok = (row < S) & (row < (split + 1) * slice_rows)
        tile_mask = row_ok[:, None] & mask_e[None, :]
        tpe_diff = tl.load(
            tpe_diff_ptr + row[:, None] * L * E + l * E + offs_e[None, :],
            mask=tile_mask,
            other=-float("inf"),
        )
        # 除法必须 div_rn(与 torch `/` 逐位一致;Triton `/` 是 div.full,差 1 ulp
        # 会翻转下游精确同分的 argmin,EP 扫描 ep=32 完全重复场景实测抓到)
        score = libdevice.div_rn(tpe_diff, count_f[None, :])
        # score.max(-1):每 step 的最大每副本负载及其位置(first-index)
        am = tl.argmax(score, axis=1, tie_break_left=True)
        om = tl.max(score, axis=1)
        eq = offs_e[None, :] == am[:, None]
        rest = tl.max(tl.where(eq, -float("inf"), score), axis=1)
        # tpe 只在 argmax 位置需要(score_added),按 [CHUNK_S] gather,traffic 远小于整 tile
        tpe_g = tl.load(tpe_ptr + row * L * E + l * E + am, mask=row_ok, other=0)
        # count 从全局 tile 提取:其余位置补 0 再求和,整数加法精确
        cnt_g = tl.sum(tl.where(eq, count[None, :], 0), axis=1)
        added = libdevice.div_rn(tpe_g.to(tl.float32), (cnt_g + 1).to(tl.float32))
        # values[s, e] = (e == am[s]) ? nm[s] : om[s];padding 行 om/nm 置 0,不污染 acc
        nm = tl.maximum(rest, added)
        om = tl.where(row_ok, om, 0.0)
        val = tl.where(eq, nm[:, None], om[:, None])
        acc = acc + tl.sum(val, axis=0)
    tl.store(partial_ptr + (l * n_split + split) * E + offs_e, acc, mask=mask_e)


@triton.jit
def _replica_decide_kernel(
    partial_ptr,  # [L, n_split, E] float32
    count_ptr,  # [L, E] int32 in/out
    ri_ptr,  # [L, R] int32 out
    slot_ptr,  # [L, R] int32 out
    L,
    E,
    R,
    i,
    n_split,
    BLOCK_E: tl.constexpr,
):
    l = tl.program_id(0).to(tl.int64)
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < E

    count = tl.load(count_ptr + l * E + offs_e, mask=mask_e, other=1)
    # 切片间按固定顺序(slice 0..n_split-1)累加;padding 列必须 +inf:
    # partials 只写真实列(masked store),masked load 补 0,若 acc 从 0 起步,
    # padding 列(0)会赢过任何非负真实列 → chosen ≥ E → 下游 l2p scatter 越界
    acc = tl.where(mask_e, 0.0, float("inf"))
    for t in range(n_split):
        acc = acc + tl.load(
            partial_ptr + (l * n_split + t) * E + offs_e, mask=mask_e, other=0.0
        )
    chosen = tl.argmin(acc, axis=0, tie_break_left=True)
    count_before = tl.sum(tl.where(offs_e == chosen, count, 0), axis=0)
    tl.store(ri_ptr + l * R + i, chosen)
    tl.store(slot_ptr + l * R + i, count_before)
    count = count + tl.where(offs_e == chosen, 1, 0)
    tl.store(count_ptr + l * E + offs_e, count, mask=mask_e)


@triton.jit
def _greedy_placement_kernel_stream(
    vectors_ptr,  # [L, E, S] float32
    eo_ptr,  # [L, P] int64, expert_order
    po_ptr,  # [L, P] int64, physical_order
    indices_ptr,  # [L, P] int64 out
    loads_ptr,  # [L, R, S] float32 scratch, 初始全 0
    peaks_ptr,  # [L, S] float32 scratch, 初始全 0
    E,
    P,
    capacity,
    num_ranks,
    S,
    n_chunks,
    CHUNK_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    l = tl.program_id(0).to(tl.int64)
    offs_r = tl.arange(0, BLOCK_R).to(tl.int64)
    mask_r = offs_r < num_ranks
    # padding rank 视为已满(occ == capacity),得分恒为 inf
    occ = tl.where(offs_r < num_ranks, 0, capacity)

    for k in range(P):
        e = tl.load(eo_ptr + l * P + k)
        po_k = tl.load(po_ptr + l * P + k)
        v_base = vectors_ptr + (l * E + e) * S
        # partial[r] = sum_s max(loads[r, s] + v[s], peaks[s]);
        # 归约顺序自定义且固定:chunk 顺序累加 + 块内 tl.sum 树
        partial = tl.zeros([BLOCK_R], dtype=tl.float32)
        for c in range(n_chunks):
            col = c.to(tl.int64) * CHUNK_S + tl.arange(0, CHUNK_S)
            col_ok = col < S
            v = tl.load(v_base + col, mask=col_ok, other=0.0)
            ld = tl.load(
                loads_ptr + l * num_ranks * S + offs_r[:, None] * S + col[None, :],
                mask=mask_r[:, None] & col_ok[None, :],
                other=0.0,
            )
            pk = tl.load(peaks_ptr + l * S + col, mask=col_ok, other=0.0)
            cand = tl.maximum(ld + v[None, :], pk[None, :])
            partial = partial + tl.sum(cand, axis=1)
        # 排除已满 rank 后取最低分;argmin 同分取 first-index,与 torch 一致
        scores = tl.where(occ == capacity, float("inf"), partial)
        rank = tl.argmin(scores, axis=0, tie_break_left=True).to(tl.int64)
        rank_mask = offs_r == rank
        occ_rank = tl.sum(tl.where(rank_mask, occ, 0), axis=0)
        tl.store(indices_ptr + l * P + rank * capacity + occ_rank, po_k)
        occ = occ + tl.where(rank_mask, 1, 0)
        # loads[l, rank, :] += v;peaks = max(peaks, loads[l, rank, :] + v)
        # 加法序列与参考逐轮相同 → loads/peaks/cand 与 torch 逐位一致
        row_base = loads_ptr + l * num_ranks * S + rank * S
        for c in range(n_chunks):
            col = c.to(tl.int64) * CHUNK_S + tl.arange(0, CHUNK_S)
            col_ok = col < S
            v = tl.load(v_base + col, mask=col_ok, other=0.0)
            old = tl.load(row_base + col, mask=col_ok, other=0.0)
            new = old + v
            tl.store(row_base + col, new, mask=col_ok)
            pk = tl.load(peaks_ptr + l * S + col, mask=col_ok, other=0.0)
            tl.store(peaks_ptr + l * S + col, tl.maximum(pk, new), mask=col_ok)


def _replica_phase_streaming(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """副本轮 ping-pong 段。返回 (ri [L,R] int32, slot [L,R] int32, logical_count [L,E] int)。

    单独成函数:UT 的间隔归因需要直接取回 ri 逐轮比对(最终输出里 per-round
    信息已被 gather 抹掉),也便于单独计时。
    """
    S, L, E = tokens_per_expert.shape
    R = num_physical_experts - E
    device = tokens_per_expert.device
    arange_logical = torch.arange(E, dtype=torch.int, device=device)
    count = torch.ones(L, E, dtype=torch.int, device=device)
    tpe_diff = (tokens_per_expert + arange_logical * 1e-4).contiguous()
    ri = torch.empty(L, R, dtype=torch.int32, device=device)
    slot = torch.empty(L, R, dtype=torch.int32, device=device)
    chunk = _REPLICA_CHUNK_S
    warps = _REPLICA_WARPS
    # ping-pong 分片:n_split 由 (S, L, chunk) 决定,各 rank 一致
    n_split = max(1, min((S + chunk - 1) // chunk, _REPLICA_TARGET_CTAS // max(L, 1)))
    slice_rows = (S + n_split - 1) // n_split
    n_chunks_slice = (slice_rows + chunk - 1) // chunk
    partials = torch.empty(L, n_split, E, dtype=torch.float32, device=device)
    for i in range(R):
        _replica_scan_kernel[(L, n_split)](
            tokens_per_expert,
            tpe_diff,
            count,
            partials,
            L,
            E,
            S,
            n_split,
            slice_rows,
            n_chunks_slice,
            CHUNK_S=chunk,
            BLOCK_E=triton.next_power_of_2(E),
            num_warps=warps,
        )
        _replica_decide_kernel[(L,)](
            partials,
            count,
            ri,
            slot,
            L,
            E,
            R,
            i,
            n_split,
            BLOCK_E=triton.next_power_of_2(E),
            num_warps=4,
        )
    return ri, slot, count


def _batch_greedy_indices_stream(
    tokens_per_expert: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    logical_count: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """与 policy._batch_greedy_indices 相同的接口,贪心循环换成流式 Triton kernel。"""
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

    chunk = _GREEDY_CHUNK_S
    warps = _GREEDY_WARPS
    # 循环状态驻全局 scratch;每层一个 program,层内 round/chunk 循环不出 kernel
    loads = vectors.new_zeros((num_layers, num_ranks, num_steps))
    peaks = vectors.new_zeros((num_layers, num_steps))
    indices = torch.empty_like(physical_order)
    _greedy_placement_kernel_stream[(num_layers,)](
        vectors,
        expert_order,
        physical_order,
        indices,
        loads,
        peaks,
        num_logical,
        num_physical,
        capacity,
        num_ranks,
        num_steps,
        (num_steps + chunk - 1) // chunk,
        CHUNK_S=chunk,
        BLOCK_R=triton.next_power_of_2(max(num_ranks, 2)),
        num_warps=warps,
    )
    return indices


def rebalance_experts_global_streaming(
    tokens_per_expert: torch.Tensor,
    num_physical_experts: int,
    num_local_physical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """大 S 的流式融合实现。

    由 policy.rebalance_experts_global 分派,输入为连续 CUDA int32 历史,
    物理专家覆盖逻辑专家并能均分到各 rank;这些条件已在入口校验。
    与参考的数值关系:prologue/epilogue 的 torch 逐元素运算(sort、除法、scatter)与参考
    逐位一致;kernel 内逐元素运算逐位一致;两处 sum-over-S 归约树不同(顺序固定)。
    鲁棒输入上决策与参考一致,输出逐位相同;验收见 test/policy_larges_test.py。
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

    arange_layers = torch.arange(num_moe_layers, dtype=torch.int, device=device)
    arange_logical = torch.arange(num_logical_experts, dtype=torch.int, device=device)
    physical_to_logical_map[:, :num_logical_experts] = arange_logical
    logical_to_physical_map[:, :, 0] = arange_logical

    if num_redundant_experts > 0:
        ri, slot, logical_count = _replica_phase_streaming(
            tokens_per_expert,
            num_physical_experts,
        )
        R = num_redundant_experts
        physical_to_logical_map[:, num_logical_experts:] = ri
        # 重建 logical_to_physical_map:l2p[l, ri[l,i], slot[l,i]] = num_logical + i
        pos = ri.to(torch.int64) * (R + 1) + slot.to(torch.int64)
        physical_ids = num_logical_experts + torch.arange(
            R, dtype=torch.int32, device=device
        )
        logical_to_physical_map.view(num_moe_layers, -1)[
            arange_layers.to(torch.int64).unsqueeze(1), pos
        ] = physical_ids
    else:
        logical_count = torch.ones(
            num_moe_layers, num_logical_experts, dtype=torch.int, device=device
        )

    if num_local_physical_experts > 1:
        indices = _batch_greedy_indices_stream(
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
