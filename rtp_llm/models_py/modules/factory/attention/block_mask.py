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
