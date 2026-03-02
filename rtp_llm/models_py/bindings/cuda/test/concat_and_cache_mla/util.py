import torch


def create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    """Create MLA cache tensor"""
    cache_dtype = (
        torch.uint8
        if kv_cache_dtype in ["fp8_e4m3", "fp8_ds_mla", "fp8_model1_mla"]
        else dtype
    )
    return torch.zeros(
        num_blocks, block_size, entry_size, dtype=cache_dtype, device=device
    )
