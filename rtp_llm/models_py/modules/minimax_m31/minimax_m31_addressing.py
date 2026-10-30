"""M3.1 model-owned addressing of independent main and indexer KV pages."""

from dataclasses import dataclass

import torch

from rtp_llm.ops.compute_ops import PyAttentionInputs


@dataclass(frozen=True)
class M31DecodeAddressing:
    main_table: torch.Tensor
    index_table: torch.Tensor
    main_slots: torch.Tensor
    index_slots: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    valid_rows: torch.Tensor


def _decode_table(inputs: PyAttentionInputs, tag: str) -> torch.Tensor:
    table = inputs.kv_cache_kernel_block_id_device
    if table is None or table.numel() == 0:
        raise ValueError(f"M3.1 {tag} cache requires a device block table")
    if table.ndim != 2 or table.dtype != torch.int32 or table.stride(1) != 1:
        raise ValueError(
            f"M3.1 {tag} block table must be contiguous int32 [batch,pages]"
        )
    return table


def m31_decode_addressing(
    main_inputs: PyAttentionInputs,
    index_inputs: PyAttentionInputs,
) -> M31DecodeAddressing:
    """Resolve the current token in each group's own 128-token page table.

    main publishes the current length in sequence_lengths_plus_1_device.
    CUDA Graph replay updates this device tensor in the capture buffer.
    """
    if main_inputs.is_prefill or index_inputs.is_prefill:
        raise ValueError("M3.1 decode addressing requires decode inputs")
    main = _decode_table(main_inputs, "main")
    index = _decode_table(index_inputs, "indexer")
    if main.shape != index.shape or main.device != index.device:
        raise ValueError("M3.1 cache tables must have matching batch/page geometry")
    seq_lens = main_inputs.sequence_lengths_plus_1_device
    index_lens = index_inputs.sequence_lengths_plus_1_device
    if (
        seq_lens is None
        or seq_lens.shape != (main.shape[0],)
        or seq_lens.dtype != torch.int32
        or seq_lens.device != main.device
    ):
        raise ValueError("M3.1 decode requires device int32 current lengths")
    if (
        index_lens is None
        or index_lens.shape != seq_lens.shape
        or index_lens.dtype != seq_lens.dtype
        or index_lens.device != seq_lens.device
    ):
        raise ValueError("M3.1 indexer current lengths must match main geometry")
    # Graph replay clears padded block-table rows to reserved physical block 0.
    # input_lengths_device is not cleared for ordinary Decode padding.
    # Tagged inputs share request metadata. No device-to-host equality check
    # is allowed on the decode or CUDA Graph replay path.
    raw_positions = seq_lens.to(torch.int64) - 1
    page = torch.div(raw_positions, 128, rounding_mode="floor")
    if main.shape[1] == 0:
        raise ValueError("M3.1 decode requires allocated pages")
    # Invalid and out-of-range requests remain -1 for the masked writer.
    in_range = (raw_positions >= 0) & (page < main.shape[1])
    safe_page = page.clamp(0, main.shape[1] - 1).unsqueeze(1)
    main_id = main.gather(1, safe_page).squeeze(1).to(torch.int64)
    index_id = index.gather(1, safe_page).squeeze(1).to(torch.int64)
    valid_rows = in_range & (main_id > 0) & (index_id > 0)
    offset = raw_positions.remainder(128)
    main_slots = torch.where(valid_rows, main_id * 128 + offset, -1)
    index_slots = torch.where(valid_rows, index_id * 128 + offset, -1)
    return M31DecodeAddressing(
        main,
        index,
        main_slots,
        index_slots,
        torch.where(valid_rows, seq_lens, 0),
        raw_positions.clamp_min(0).to(torch.int32),
        valid_rows,
    )
