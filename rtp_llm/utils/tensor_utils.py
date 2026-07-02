import torch


def get_first_token_from_combo_tokens(
    tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    start_indices = torch.cumsum(torch.cat((torch.tensor([0]), lengths[:-1])), dim=0)
    return tensor[start_indices]


def get_last_token_from_combo_tokens(
    tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    end_indices = torch.cumsum(lengths, dim=0) - 1
    return tensor[end_indices]


def get_token_at_first_id_from_combo_tokens(
    tensor: torch.Tensor,
    input_ids: torch.Tensor,
    lengths: torch.Tensor,
    token_id: int,
) -> torch.Tensor:
    """取每条序列内首个值为 `token_id` 的 token 处的 hidden (用于 CLS_UQI=2 个性化头)。

    Args:
        tensor:    [total_tokens, hidden] 拼接的 combo hidden。
        input_ids: [total_tokens] 拼接的 combo token ids。
        lengths:   [batch] 各序列长度。
        token_id:  目标标记 token id (CLS_UQI 默认 2)。
    Returns:
        [batch, hidden] 各序列对应位置的 hidden。
    """
    # 全向量化: 无 Python 循环 / 无 per-seq D2H 同步 (老路每条 nonzero+int(hits[0]) 是线性
    # 开销, 大 batch 下随候选数放大)。ragged "每条首个 token_id 位置" 用 scatter_reduce(amin)
    # 表达, 与老循环逐位等价 (tensor_utils_cls_uqi_test 逐位对拍); 找不到时回退 offset 0。
    device = tensor.device
    num_seq = int(lengths.shape[0])
    lengths_l = lengths.to(device=device, dtype=torch.long)
    total = int(input_ids.shape[0])
    start = torch.cumsum(lengths_l, 0) - lengths_l  # [num_seq] 各序列起点
    seq_of_tok = torch.repeat_interleave(
        torch.arange(num_seq, device=device), lengths_l
    )  # [total] 每 token 属于哪条序列
    local_pos = torch.arange(total, device=device) - start[seq_of_tok]  # 序列内位置
    BIG = total + 1
    big = torch.full((total,), BIG, dtype=torch.long, device=device)
    match_pos = torch.where(input_ids.to(device) == token_id, local_pos, big)
    first_local = torch.full(
        (num_seq,), BIG, dtype=torch.long, device=device
    ).scatter_reduce(0, seq_of_tok, match_pos, reduce="amin", include_self=True)
    # 找不到 (BIG) -> offset 0, 与老 fallback 逐位一致
    first_local = torch.where(
        first_local < BIG, first_local, torch.zeros_like(first_local)
    )
    return tensor[start + first_local]
