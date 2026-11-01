"""Deterministic small Engram tables and independent CPU lookup expectations."""
import hashlib

LAYERS = ((1, 4097), (2, 514))
DIM = 64
MODEL_REVISION = "engram-integration-main-492ecf45"
HASH_ABI = "sha256:" + hashlib.sha256(b"engram-fixture-v1").hexdigest()


def expected_layout():
    from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout

    return ExpectedLayout(layers=LAYERS, head_dim=DIM,
                          model_revision=MODEL_REVISION, hash_abi=HASH_ABI)


def raw_rows(row_ids):
    """Return selected (FP8 weight bytes, E8M0 scale bytes), both CPU uint8.

    The leading shape follows row_ids; this function does not apply table bounds.
    Consumers can verify sampled rows without materializing an entire table.
    """
    import torch

    rows = torch.as_tensor(row_ids, dtype=torch.int64, device="cpu")
    columns = torch.arange(DIM, dtype=torch.int64)
    groups = torch.arange(DIM // 32, dtype=torch.int64)
    values = ((rows[..., None] * DIM + columns) % 17 - 8).to(torch.float32)
    weight = values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    scale = ((rows[..., None] * (DIM // 32) + groups) % 5 + 125).to(torch.uint8)
    return weight, scale.contiguous()


def make_table(rows):
    """Owner-only complete CPU table; consumers should use raw_rows instead."""
    import torch

    if type(rows) is not int or rows < 1:
        raise ValueError("rows must be a positive integer")
    return raw_rows(torch.arange(rows, dtype=torch.int64))


def expected_rows(ids, table_rows):
    """BF16 CPU oracle with every out-of-range row (including negatives) zero."""
    import torch

    if type(table_rows) is not int or table_rows < 1:
        raise ValueError("table_rows must be a positive integer")
    indices = torch.as_tensor(ids, dtype=torch.int64, device="cpu")
    valid = (indices >= 0) & (indices < table_rows)
    # Mask before arithmetic, including for very large invalid indices.
    safe = torch.where(valid, indices, 0)
    weight, scale = raw_rows(safe)
    values = weight.view(torch.float8_e4m3fn).to(torch.float32)
    factors = torch.ldexp(torch.ones_like(scale, dtype=torch.float32),
                          scale.to(torch.int32) - 127)
    result = (values * factors.repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    return result.masked_fill(~valid[..., None], 0)
