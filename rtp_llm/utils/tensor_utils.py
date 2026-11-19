import torch


def first_token_indices(lengths: torch.Tensor) -> torch.Tensor:
    return lengths.cumsum(0) - lengths


def last_token_indices(lengths: torch.Tensor) -> torch.Tensor:
    return lengths.cumsum(0) - 1


def get_first_token_from_combo_tokens(
    tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    return tensor[first_token_indices(lengths)]


def get_last_token_from_combo_tokens(
    tensor: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    return tensor[last_token_indices(lengths)]
