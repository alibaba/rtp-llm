from dataclasses import dataclass
from typing import Optional

import torch

@dataclass
class BertUqiTwoPassSchedule:
    has_b: bool
    perm: Optional[torch.Tensor]  # [total] int64 device; None => identity
    inv_perm: Optional[torch.Tensor]  # [total] int64 device
    b_rows: Optional[torch.Tensor]  # [total_b] int64 device, rows in PERMUTED layout
    qo_indptr_p1: torch.Tensor  # int32 CPU: [2N+1] if has_b else [N+1] (=cu_seqlens_host)
    qo_indptr_p2: Optional[torch.Tensor]  # [N+1] int32 CPU
    kv_indptr_p2: Optional[torch.Tensor]  # [N+1] int32 CPU (= cu_seqlens_host)


def derive_bert_uqi_segment_ids_hostlen(
    input_ids: torch.Tensor,
    cu_seqlens_dev: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    b_start_token_id: int = 2,
    sep_token_id: int = 102,
) -> torch.Tensor:
    """与 `derive_bert_uqi_segment_ids` 逐位等价, 但所有数据相关形状取自
    cu_seqlens_host —— 无隐藏 device sync (repeat_interleave 的 lengths 在 host)。
    """
    device = input_ids.device
    total = int(input_ids.shape[0])
    out = torch.zeros(total, dtype=torch.int32, device=device)
    num_seq = int(cu_seqlens_host.numel()) - 1
    if total == 0 or num_seq <= 0:
        return out
    cu_host = cu_seqlens_host.to(dtype=torch.long)
    lengths_host = cu_host[1:] - cu_host[:-1]
    seq_of_tok = torch.repeat_interleave(
        torch.arange(num_seq), lengths_host
    ).to(device, non_blocking=True)
    cu = cu_seqlens_dev.to(device=device, dtype=torch.long)
    lengths = cu[1:] - cu[:-1]
    local_pos = torch.arange(total, device=device) - cu[seq_of_tok]
    BIG = total + 1
    big = torch.full((total,), BIG, dtype=torch.long, device=device)
    cls_src = torch.where(input_ids == b_start_token_id, local_pos, big)
    first_cls = torch.full(
        (num_seq,), BIG, dtype=torch.long, device=device
    ).scatter_reduce(0, seq_of_tok, cls_src, reduce="amin", include_self=True)
    first_cls_tok = first_cls[seq_of_tok]
    sep_after = (input_ids == sep_token_id) & (local_pos >= first_cls_tok)
    sep_src = torch.where(sep_after, local_pos, big)
    first_sep = torch.full(
        (num_seq,), BIG, dtype=torch.long, device=device
    ).scatter_reduce(0, seq_of_tok, sep_src, reduce="amin", include_self=True)
    first_sep_tok = first_sep[seq_of_tok]
    has_sep = first_sep_tok < BIG
    b_end_tok = torch.where(has_sep, first_sep_tok + 1, lengths[seq_of_tok])
    in_b = (local_pos >= first_cls_tok) & (local_pos < b_end_tok)
    out[in_b] = 1
    return out


def build_bert_uqi_two_pass_schedule_from_bounds(
    b_starts_host: torch.Tensor,
    b_lens_host: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
    device: torch.device,
) -> BertUqiTwoPassSchedule:
    """C++ 元数据路径: 段边界已由引擎在 host token 上扫出(PyWrappedModel),
    schedule 全部 host 构建 —— 零 GPU derive kernel、零 D2H 同步、零排序
    (permutation 由边界闭式公式直接算出)。语义与 derive+build 等价。

    Args:
        b_starts_host: [N] int32 CPU, 画像段起点(序列内), -1=无画像段。
        b_lens_host:   [N] int32 CPU, 画像段长度, 0=无。
        cu_seqlens_host: [N+1] int32 CPU (pinned 镜像)。
        device: perm/b_rows 的目标设备。
    """
    num_seq = int(cu_seqlens_host.numel()) - 1
    cu_i32_host = cu_seqlens_host.to(dtype=torch.int32)
    cu_host = cu_seqlens_host.to(dtype=torch.long)
    total = int(cu_host[-1]) if num_seq > 0 else 0
    b_lens = b_lens_host.to(torch.long)
    b_starts = b_starts_host.to(torch.long)
    total_b = int(b_lens.sum()) if num_seq > 0 else 0
    if num_seq <= 0 or total == 0 or total_b == 0:
        return BertUqiTwoPassSchedule(
            has_b=False, perm=None, inv_perm=None, b_rows=None,
            qo_indptr_p1=cu_i32_host, qo_indptr_p2=None, kv_indptr_p2=None,
        )
    lengths = cu_host[1:] - cu_host[:-1]
    a_lens = lengths - b_lens
    seg_lens = torch.stack([a_lens, b_lens], dim=1).reshape(-1)  # [2N]
    qo_p1 = torch.zeros(2 * num_seq + 1, dtype=torch.int32)
    qo_p1[1:] = seg_lens.cumsum(0).to(torch.int32)
    qo_p2 = torch.zeros(num_seq + 1, dtype=torch.int32)
    qo_p2[1:] = b_lens.cumsum(0).to(torch.int32)
    # permutation 闭式构造(免排序): token t(序列内位置 local)的新位置 =
    #   B 段内:  a_len + (local - b_start)
    #   B 段前:  local
    #   B 段后:  local - b_len
    pos = torch.arange(total)
    seq_of_tok = torch.repeat_interleave(torch.arange(num_seq), lengths)
    local = pos - cu_host[:-1][seq_of_tok]
    bs_t = b_starts[seq_of_tok]
    bl_t = b_lens[seq_of_tok]
    has_b_t = bs_t >= 0
    in_b = has_b_t & (local >= bs_t) & (local < bs_t + bl_t)
    after_b = has_b_t & (local >= bs_t + bl_t)
    new_local = torch.where(
        in_b, a_lens[seq_of_tok] + (local - bs_t), local - after_b * bl_t
    )
    new_pos = cu_host[:-1][seq_of_tok] + new_local  # = inv_perm (token t 的新位置)
    perm_host = torch.empty(total, dtype=torch.long)
    perm_host[new_pos] = pos  # perm[新位置] = 旧下标
    # B rows in permuted layout: 序列 i 尾部 [cu[i]+a_i, cu[i+1])
    seq_of_b = torch.repeat_interleave(torch.arange(num_seq), b_lens)
    within = torch.arange(total_b) - qo_p2[:-1].to(torch.long)[seq_of_b]
    b_rows_host = cu_host[:-1][seq_of_b] + a_lens[seq_of_b] + within
    return BertUqiTwoPassSchedule(
        has_b=True,
        perm=perm_host.to(device, non_blocking=True),
        inv_perm=new_pos.to(device, non_blocking=True),
        b_rows=b_rows_host.to(device, non_blocking=True),
        qo_indptr_p1=qo_p1, qo_indptr_p2=qo_p2, kv_indptr_p2=cu_i32_host,
    )


def build_bert_uqi_two_pass_schedule(
    seg_ids: torch.Tensor,
    cu_seqlens_dev: torch.Tensor,
    cu_seqlens_host: torch.Tensor,
) -> BertUqiTwoPassSchedule:
    """从段标记构建两趟 attention 的执行计划。

    perm 把每条序列重排为 [A...|B...] (A/B 内部保持原相对顺序, stable);
    qo_indptr_p1 为 2N 段 [A_0][B_0]...[A_{N-1}][B_{N-1}] 的前缀和 (CPU int32,
    直接喂 FlashInfer plan, 其内部 `.to("cpu")` 变 no-op); pass 2 的 kv indptr
    就是 cu_seqlens_host。唯一 device->host 同步 = b_lens 的 [N] int 拷贝。
    """
    device = seg_ids.device
    total = int(seg_ids.shape[0])
    num_seq = int(cu_seqlens_host.numel()) - 1
    cu_i32_host = cu_seqlens_host.to(dtype=torch.int32)
    if total == 0 or num_seq <= 0:
        return BertUqiTwoPassSchedule(
            has_b=False, perm=None, inv_perm=None, b_rows=None,
            qo_indptr_p1=cu_i32_host, qo_indptr_p2=None, kv_indptr_p2=None,
        )
    cu_host = cu_seqlens_host.to(dtype=torch.long)
    lengths_host = cu_host[1:] - cu_host[:-1]
    seq_of_tok_host = torch.repeat_interleave(torch.arange(num_seq), lengths_host)
    seq_of_tok = seq_of_tok_host.to(device, non_blocking=True)
    b_lens_dev = torch.zeros(num_seq, dtype=torch.long, device=device).scatter_add_(
        0, seq_of_tok, seg_ids.to(torch.long)
    )
    b_lens_host = b_lens_dev.cpu()  # 全路径唯一的 device->host 同步 ([N] ints)
    total_b = int(b_lens_host.sum())
    if total_b == 0:
        return BertUqiTwoPassSchedule(
            has_b=False, perm=None, inv_perm=None, b_rows=None,
            qo_indptr_p1=cu_i32_host, qo_indptr_p2=None, kv_indptr_p2=None,
        )
    a_lens_host = lengths_host - b_lens_host
    seg_lens = torch.stack([a_lens_host, b_lens_host], dim=1).reshape(-1)  # [2N]
    qo_p1 = torch.zeros(2 * num_seq + 1, dtype=torch.int32)
    qo_p1[1:] = seg_lens.cumsum(0).to(torch.int32)
    qo_p2 = torch.zeros(num_seq + 1, dtype=torch.int32)
    qo_p2[1:] = b_lens_host.cumsum(0).to(torch.int32)
    # stable sort by (seq, seg): A 保持相对顺序在前, B 在后
    key = seq_of_tok * 2 + seg_ids.to(torch.long)
    perm = torch.argsort(key, stable=True)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(total, device=device)
    # B rows in permuted layout: 序列 i 的尾部 [cu[i]+a_i, cu[i+1]); 全在 host 构建
    seq_of_b = torch.repeat_interleave(torch.arange(num_seq), b_lens_host)
    within = torch.arange(total_b) - qo_p2[:-1].to(torch.long)[seq_of_b]
    b_rows_host = cu_host[:-1][seq_of_b] + a_lens_host[seq_of_b] + within
    b_rows = b_rows_host.to(device, non_blocking=True)
    return BertUqiTwoPassSchedule(
        has_b=True, perm=perm, inv_perm=inv_perm, b_rows=b_rows,
        qo_indptr_p1=qo_p1, qo_indptr_p2=qo_p2, kv_indptr_p2=cu_i32_host,
    )
