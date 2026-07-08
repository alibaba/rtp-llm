"""User-profile block-attention-mask helpers — pure torch, zero rtp_llm deps.

Production utilities for the BERT reranker user-profile branch:
  - `derive_bert_uqi_segment_ids`: derive the A/B segment tag from combo token ids.
  - `build_bert_uqi_flashinfer_mask`: build the logical boolean mask fed to
    FlashInfer's `plan(custom_mask=...)`.

Visibility rule (M = [[1, 0], [1, 1]]):
    i(query) can attend j(key)  <=>  qi_uqi_segment_ids[j] == 0  OR  qi_uqi_segment_ids[i] == 1
i.e. A sees A, A does NOT see B, B sees A, B sees B. All-zero segment => no
masking (original full attention, byte-identical to the QI-only path).

The numerical eager reference (`eager_masked_attention`) used to cross-check
these in tests lives in `eager_impl/eager_block_mask_core.py` (test oracle only).
"""

from dataclasses import dataclass
from typing import Optional

import torch


def derive_bert_uqi_segment_ids(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    b_start_token_id: int = 2,
    sep_token_id: int = 102,
) -> torch.Tensor:
    """从 combo token ids 推导 qi_uqi_segment_ids (每 token 0=A / 1=B)。

    B 段 = 每条序列内第一个 `b_start_token_id` (CLS_UQI=2) 起, 到其后第一个
    `sep_token_id` (SEP=102) 止 (闭区间)。**B 段只覆盖 `[CLS_UQI] user [SEP]`**;
    user 段之后若还有 token (例如 vision 占位负数) 会回到 A 段。
    序列内不含 `b_start_token_id` (老 QI-only 输入) 则整条 A 段(全 0)。
    若 CLS_UQI 之后找不到 SEP, 退化为到序列尾。

    Args:
        input_ids:   [total_tokens] 拼接的 combo token ids。
        cu_seqlens:  [batch+1] 各序列边界前缀和。
        b_start_token_id: B 段起始标记 token id (CLS_UQI)。
        sep_token_id:     B 段结束标记 token id (SEP)。
    Returns:
        [total_tokens] int32, 0/1。
    """
    # 全向量化: 无 Python 循环 / 无 nonzero / 无 D2H, 设备跟随 input_ids。
    # ragged "每条序列首个匹配" 用 scatter_reduce(amin) 表达; 与上面逐字节等价
    # (eager_block_mask_core_test + block_mask_vectorized_parity 逐位对拍)。
    device = input_ids.device
    total = int(input_ids.shape[0])
    out = torch.zeros(total, dtype=torch.int32, device=device)
    num_seq = cu_seqlens.numel() - 1
    if total == 0 or num_seq <= 0:
        return out
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    lengths = cu[1:] - cu[:-1]  # [num_seq] 各序列长度
    seq_of_tok = torch.repeat_interleave(
        torch.arange(num_seq, device=device), lengths
    )  # [total] 每 token 属于哪条序列
    local_pos = torch.arange(total, device=device) - cu[seq_of_tok]  # 序列内位置
    BIG = total + 1
    big = torch.full((total,), BIG, dtype=torch.long, device=device)
    # 每条序列首个 CLS_UQI 的序列内位置 (无则保持 BIG)
    cls_src = torch.where(input_ids == b_start_token_id, local_pos, big)
    first_cls = torch.full(
        (num_seq,), BIG, dtype=torch.long, device=device
    ).scatter_reduce(0, seq_of_tok, cls_src, reduce="amin", include_self=True)
    first_cls_tok = first_cls[seq_of_tok]  # [total]
    # 每条首个 "在 CLS_UQI 之后(含)" 的 SEP 的序列内位置 (无则 BIG)
    sep_after = (input_ids == sep_token_id) & (local_pos >= first_cls_tok)
    sep_src = torch.where(sep_after, local_pos, big)
    first_sep = torch.full(
        (num_seq,), BIG, dtype=torch.long, device=device
    ).scatter_reduce(0, seq_of_tok, sep_src, reduce="amin", include_self=True)
    first_sep_tok = first_sep[seq_of_tok]  # [total]
    # B 段闭区间 [first_cls, b_end): 有 SEP -> first_sep+1; 无 SEP -> 序列尾(length)。
    # 无 CLS_UQI 时 first_cls_tok==BIG -> in_b 全 False -> 整条 A 段。
    has_sep = first_sep_tok < BIG
    b_end_tok = torch.where(has_sep, first_sep_tok + 1, lengths[seq_of_tok])
    in_b = (local_pos >= first_cls_tok) & (local_pos < b_end_tok)
    out[in_b] = 1
    return out


def build_bert_uqi_flashinfer_mask(
    qi_uqi_segment_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """造 FlashInfer ragged `custom_mask`: 逻辑布尔 mask, per-req [q,kv] flatten 拼接。

    FlashInfer 吃**逻辑顺序**(行优先 [q_len, kv_len])的布尔 mask, True=可见;
    tensor-core swizzle / packbits 由库内部按架构自适应完成(与 TRT 需 host 端预排相反)。
    可见性: visible[i,j] = (qi_uqi_segment_ids[i]==1) OR (qi_uqi_segment_ids[j]==0)
    (A看A、A不看B、B看全部)。自注意力 q_len==kv_len, 返回 dtype=torch.bool,
    长度 = sum_b (n_b * n_b), 设备跟随 qi_uqi_segment_ids。
    """
    # 全向量化(无 Python 循环): 为每个输出位置算出 (序列, 行 i, 列 j), 再施可见性规则。
    # 输出顺序与逐序列 cat 完全一致 (按序列、块内 row-major [i,j])。
    device = qi_uqi_segment_ids.device
    num_seq = cu_seqlens.numel() - 1
    if num_seq <= 0:
        return torch.empty(0, dtype=torch.bool, device=device)
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    lengths = cu[1:] - cu[:-1]  # [num_seq]
    sizes = lengths * lengths  # 每条 ragged mask 的元素数 n_b^2
    total_mask = int(sizes.sum())
    if total_mask == 0:
        return torch.empty(0, dtype=torch.bool, device=device)
    seq_of_pos = torch.repeat_interleave(
        torch.arange(num_seq, device=device), sizes
    )  # [total_mask] 每个 mask 元素属于哪条序列
    block_start = torch.cumsum(sizes, 0) - sizes  # 各序列 mask 块在输出里的起点
    within = torch.arange(total_mask, device=device) - block_start[seq_of_pos]  # 块内 0..n^2-1
    n = lengths[seq_of_pos]
    qi = torch.div(within, n, rounding_mode="floor")  # 行(序列内 query)
    kj = within - qi * n  # 列(序列内 key)
    base = cu[seq_of_pos]
    seg_i = qi_uqi_segment_ids[base + qi]
    seg_j = qi_uqi_segment_ids[base + kj]
    return (seg_i == 1) | (seg_j == 0)  # [total_mask] bool, row-major [q,kv] per-req 拼接


# ---------------------------------------------------------------------------
# Two-pass (no custom mask) schedule
# ---------------------------------------------------------------------------
#
# The block mask M=[[1,0],[1,1]] admits an exact two-pass decomposition once
# each sequence is permuted to [A... | B...]:
#   pass 1: one ragged attention over 2N segments [A_0][B_0][A_1][B_1]... —
#           every segment attends itself. A rows come out CORRECT (A sees only
#           A); the tiny B self-attention rows are garbage and get overwritten.
#   pass 2: B queries attend the FULL sequence (kv order is irrelevant —
#           attention is permutation-invariant over keys); scatter over the
#           garbage rows.
# All plan indptrs are built from the pinned cu_seqlens_host; the ONLY
# device->host sync in this path is the [num_seq]-int b_lens copy below.
# (`torch.repeat_interleave` with device-side lengths hides a `.sum().item()`
# sync — every data-dependent shape here comes from host lengths instead.)


@dataclass
class BertUqiTwoPassSchedule:
    has_b: bool
    perm: Optional[torch.Tensor]  # [total] int64 device; None => identity
    inv_perm: Optional[torch.Tensor]  # [total] int64 device
    b_rows: Optional[torch.Tensor]  # [total_b] int64 device, rows in PERMUTED layout
    qo_indptr_p1: torch.Tensor  # int32 CPU: [2N+1] if has_b else [N+1] (=cu_seqlens_host)
    qo_indptr_p2: Optional[torch.Tensor]  # [N+1] int32 CPU
    kv_indptr_p2: Optional[torch.Tensor]  # [N+1] int32 CPU (= cu_seqlens_host)
    # eager pass-2 的 padded 索引(全 host 构建后 H2D, 形状 host 已知, 零同步)。
    # kv_pad_idx/[mask]: [N, Lmax] 每序列 kv 行号(PERMUTED 布局)/有效位;
    # q_pad_idx/[mask]:  [N, Bmax] 每序列 B 查询行号/有效位。
    kv_pad_idx: Optional[torch.Tensor] = None  # [N, Lmax] int64 device
    kv_pad_mask: Optional[torch.Tensor] = None  # [N, Lmax] bool device
    q_pad_idx: Optional[torch.Tensor] = None  # [N, Bmax] int64 device
    q_pad_mask: Optional[torch.Tensor] = None  # [N, Bmax] bool device


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
    # eager pass-2 padded 索引 (host 构建, 形状由 host 长度决定, 零同步)
    l_max = int(lengths_host.max())
    b_max = int(b_lens_host.max())
    pos = torch.arange(l_max)
    kv_pad_mask_h = pos[None, :] < lengths_host[:, None]  # [N, Lmax]
    kv_pad_idx_h = cu_host[:-1, None] + torch.minimum(
        pos[None, :], (lengths_host - 1)[:, None]
    )  # pad 槽 clamp 到序列内(反正被 mask), 保证 gather 不越界
    bpos = torch.arange(b_max)
    q_pad_mask_h = bpos[None, :] < b_lens_host[:, None]  # [N, Bmax]
    q_start = cu_host[:-1] + a_lens_host
    q_pad_idx_h = torch.where(
        q_pad_mask_h, q_start[:, None] + bpos[None, :], torch.zeros((), dtype=torch.long)
    )
    return BertUqiTwoPassSchedule(
        has_b=True, perm=perm, inv_perm=inv_perm, b_rows=b_rows,
        qo_indptr_p1=qo_p1, qo_indptr_p2=qo_p2, kv_indptr_p2=cu_i32_host,
        kv_pad_idx=kv_pad_idx_h.to(device, non_blocking=True),
        kv_pad_mask=kv_pad_mask_h.to(device, non_blocking=True),
        q_pad_idx=q_pad_idx_h.to(device, non_blocking=True),
        q_pad_mask=q_pad_mask_h.to(device, non_blocking=True),
    )
