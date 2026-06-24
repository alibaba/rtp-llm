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
    start_indices = torch.cumsum(
        torch.cat((torch.tensor([0], device=lengths.device), lengths[:-1])), dim=0
    )
    picked = []
    for b in range(lengths.shape[0]):
        s = int(start_indices[b])
        e = s + int(lengths[b])
        hits = (input_ids[s:e] == token_id).nonzero(as_tuple=False)
        offset = int(hits[0]) if hits.numel() > 0 else 0
        picked.append(tensor[s + offset])
    return torch.stack(picked)
