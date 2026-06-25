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
    uqi_segment_ids = torch.zeros(input_ids.shape[0], dtype=torch.int32)
    for b in range(cu_seqlens.numel() - 1):
        s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        seq = input_ids[s:e]
        hits = (seq == b_start_token_id).nonzero(as_tuple=False)
        if hits.numel() == 0:
            continue
        b_start = int(hits[0])
        # B 段结束 = b_start 之后第一个 SEP (闭区间); 找不到则到序列尾。
        sep_hits = (seq[b_start:] == sep_token_id).nonzero(as_tuple=False)
        b_end = b_start + int(sep_hits[0]) + 1 if sep_hits.numel() > 0 else e - s
        uqi_segment_ids[s + b_start : s + b_end] = 1
    return uqi_segment_ids


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
    parts = []
    for b in range(cu_seqlens.numel() - 1):
        s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        uqi_segment_ids = qi_uqi_segment_ids[s:e]
        n = e - s
        vis = (uqi_segment_ids.view(n, 1) == 1) | (uqi_segment_ids.view(1, n) == 0)  # [n, n] bool
        parts.append(vis.reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.bool, device=qi_uqi_segment_ids.device)
    return torch.cat(parts)
