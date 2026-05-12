import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class AscendAttnParams:
    """Ascend attention operator parameters (mapped to torch_npu interface).

    Also serves as a lightweight params container for RoPE/KVCacheWrite components,
    replacing FlashInferMlaAttnParams on Ascend platform.
    """
    block_table: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None
    actual_seq_lengths_q: Optional[torch.Tensor] = None
    actual_seq_lengths_kv: Optional[torch.Tensor] = None
    num_kv_heads: int = 0
    num_heads: int = 0
    head_dim: int = 0
    block_size: int = 128
    scale: float = 1.0
    positions_d: Optional[torch.Tensor] = None  # RoPE position IDs (device tensor)


def build_ascend_params(attn_inputs, page_size: int) -> AscendAttnParams:
    """Build AscendAttnParams from PyAttentionInputs."""
    params = AscendAttnParams()

    params.block_table = attn_inputs.kv_cache_block_id_host

    if attn_inputs.sequence_lengths.numel() > 0:
        params.seq_lens = attn_inputs.prefix_lengths + attn_inputs.input_lengths

    if attn_inputs.is_prefill:
        prefix_len = attn_inputs.prefix_lengths
        input_len = attn_inputs.input_lengths
        kv_len = prefix_len + input_len

        zero = torch.zeros(1, dtype=torch.int32, device=input_len.device)
        params.actual_seq_lengths_q = torch.cat([zero, torch.cumsum(input_len, dim=0)])
        params.actual_seq_lengths_kv = torch.cat([zero, torch.cumsum(kv_len, dim=0)])

    params.block_size = page_size
    return params


def compute_ascend_attn_params(attn_inputs):
    """Compute RoPE positions and KV cache slot_mapping in pure Python.

    Replaces C++ FlashInferMlaAttnParams.fill_params() on Ascend platform.
    Computation is on CPU (device-independent integer ops).
    Caller should move returned tensors to the target device (NPU).

    Args:
        attn_inputs: PyAttentionInputs with fields:
            - is_prefill: bool
            - prefix_lengths: [B] int32 (CPU or NPU)
            - input_lengths: [B] int32 (CPU or NPU)
            - sequence_lengths: [B] int32 (CPU or NPU)
            - kv_cache_block_id_host: [B, max_blocks] int32 (CPU)
            - kv_cache: object with seq_size_per_block

    Returns:
        positions: [num_tokens] int32, CPU
        slot_mapping: [num_tokens] int64, CPU
    """
    is_prefill = attn_inputs.is_prefill
    block_table = attn_inputs.kv_cache_block_id_host  # always on CPU
    page_size = (attn_inputs.kv_cache.seq_size_per_block
                 if attn_inputs.kv_cache is not None else 128)

    if is_prefill:
        prefix_lens = attn_inputs.prefix_lengths.cpu() if attn_inputs.prefix_lengths is not None else None
        input_lens = attn_inputs.input_lengths.cpu()

        batch_ids_list = []
        pos_list = []
        for i in range(len(input_lens)):
            prefix = int(prefix_lens[i]) if prefix_lens is not None else 0
            inp_len = int(input_lens[i])
            for j in range(inp_len):
                batch_ids_list.append(i)
                pos_list.append(prefix + j)

        positions = torch.tensor(pos_list, dtype=torch.int32)
        batch_ids = torch.tensor(batch_ids_list, dtype=torch.int32)
    else:
        positions = attn_inputs.sequence_lengths.cpu().clone()
        batch_ids = torch.arange(len(positions), dtype=torch.int32)

    if (block_table is not None and block_table.numel() > 0
            and positions.numel() > 0):
        max_blocks = block_table.shape[1]
        block_index = positions // page_size
        if max_blocks > 0:
            block_index = block_index.clamp(max=max_blocks - 1)
        block_offset = positions % page_size
        slot_block_numbers = block_table[batch_ids, block_index]
        slot_block_numbers = slot_block_numbers.clamp(min=0)  # replace -1 with 0
        slot_mapping = (slot_block_numbers * page_size + block_offset).to(torch.int64)
    else:
        slot_mapping = torch.empty(0, dtype=torch.int64)

    return positions, slot_mapping
