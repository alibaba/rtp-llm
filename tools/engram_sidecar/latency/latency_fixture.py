"""Deterministic addresses/content for latency experiments, not model weights."""
import torch


def raw_rows(row_ids, dim):
    rows = torch.as_tensor(row_ids, dtype=torch.int64, device='cpu')
    columns = torch.arange(dim, dtype=torch.int64)
    groups = torch.arange(dim // 32, dtype=torch.int64)
    values = ((rows[..., None] * 17 + columns * 13) % 31 - 15).float()
    weights = values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    scales = ((rows[..., None] * 3 + groups) % 7 + 122).to(torch.uint8).contiguous()
    return weights, scales


def local_raw_table(rows, dim, chunk_rows=16384):
    weight = torch.empty((rows, dim), dtype=torch.uint8)
    scale = torch.empty((rows, dim // 32), dtype=torch.uint8)
    for begin in range(0, rows, chunk_rows):
        end = min(rows, begin + chunk_rows)
        weight[begin:end], scale[begin:end] = raw_rows(torch.arange(begin, end), dim)
    return weight, scale
