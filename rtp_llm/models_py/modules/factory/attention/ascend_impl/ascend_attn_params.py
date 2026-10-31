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
    # MRoPE three-axis (t/h/w) position IDs [num_tokens, 3] (device tensor).
    # Populated from combo_position_ids when the model uses MROPE-style
    # position generation; text-only requests carry identical rows.
    mrope_positions_d: Optional[torch.Tensor] = None
    # Kernel blocks per physical block; >1 when the attention kernel needs a
    # finer block granularity than the allocated KV block.
    blocks_per_phys: int = 1


def _squeeze_block_table(block_table):
    """Ensure block_table is 2-D [batch, max_blocks].

    The C++ side may pass a 3-D tensor [group, batch, max_blocks].
    For non-hybrid models group=1, so we squeeze the leading dim.
    """
    if block_table is not None and block_table.dim() == 3:
        block_table = block_table.squeeze(0)
    return block_table


def infer_blocks_per_phys(attn_inputs) -> int:
    """Kernel blocks per physical block, derived from the two block tables.

    ``attn_inputs.kv_cache`` is not populated on this path, so the physical
    block size cannot be read directly.  The kernel table is the physical table
    expanded by that factor, so their widths give the ratio.
    """
    phys = attn_inputs.kv_cache_block_id
    kernel = attn_inputs.kv_cache_kernel_block_id
    if phys is None or kernel is None or phys.numel() == 0 or kernel.numel() == 0:
        return 1
    phys_cols = int(phys.shape[-1])
    kernel_cols = int(kernel.shape[-1])
    if phys_cols <= 0 or kernel_cols % phys_cols != 0:
        return 1
    return max(1, kernel_cols // phys_cols)


def _blocks_per_phys_from_config(attn_configs, attn_inputs) -> int:
    """Kernel blocks per physical block.

    Derived from tokens_per_block / kernel_tokens_per_block. During aclgraph
    capture the C++ runner allocates both block tables with identical widths,
    so shape-based inference (infer_blocks_per_phys) returns 1 there and breaks
    the slot mapping for subdivided blocks (bpk > 1).
    """
    tpb = getattr(attn_configs, "tokens_per_block", 0) or 0
    ktpb = getattr(attn_configs, "kernel_tokens_per_block", 0) or 0
    if tpb > 0 and ktpb > 0 and tpb % ktpb == 0:
        return tpb // ktpb
    return infer_blocks_per_phys(attn_inputs)


def build_ascend_params(attn_inputs, page_size: int) -> AscendAttnParams:
    """Build AscendAttnParams from PyAttentionInputs."""
    params = AscendAttnParams()

    params.block_table = _squeeze_block_table(attn_inputs.kv_cache_block_id)

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


def compute_ascend_attn_params(attn_inputs, layer_idx: int = 0, phys_page_size: int = 0):
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
            - kv_cache_block_id: [B, max_blocks] int32 (CPU)
            - kv_cache: object with seq_size_per_block

    Returns:
        positions: [num_tokens] int32, CPU
        slot_mapping: [num_tokens] int64, CPU
    """
    is_prefill = attn_inputs.is_prefill
    block_table = _squeeze_block_table(attn_inputs.kv_cache_block_id)  # always on CPU
    # Hybrid models: with the tag-based cache layout the engine already splits
    # per-group inputs (each PyAttentionInputs carries this group's 2-D table),
    # so no [group, batch, blocks] selection is needed here anymore.
    # slot_mapping indexes the physical block storage, so it must use the
    # physical block size.  attn_inputs.kv_cache is not populated here, hence
    # the caller passes the size it derived from the block tables.
    page_size = int(phys_page_size) if phys_page_size else (
        attn_inputs.kv_cache.seq_size_per_block
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

    # MRoPE positions: the engine (PositionIdsGenerator MROPE style) provides
    # combo_position_ids [num_tokens, 3]; text-only requests degenerate to
    # identical rows (standard sequential positions). slot_mapping above keeps
    # using the 1-D seq positions — KV slot addressing is independent of the
    # RoPE axis layout.
    mrope_positions = None
    combo_position_ids = getattr(attn_inputs, "combo_position_ids", None)
    if combo_position_ids is not None and combo_position_ids.numel() > 0:
        if combo_position_ids.dim() == 1:
            # index_factor=3 flat layout [T*3] -> [T, 3]
            combo_position_ids = combo_position_ids.reshape(-1, 3)
        if combo_position_ids.dim() == 2 and combo_position_ids.size(1) == 3:
            mrope_positions = combo_position_ids.to(torch.int64)

    return positions, slot_mapping, mrope_positions