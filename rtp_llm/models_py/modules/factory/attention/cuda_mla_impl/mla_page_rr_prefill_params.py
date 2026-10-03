"""Device MLA Prefill metadata for a page round-robin KV cache.

Adapted from feat/k3_dev's FlashMLADeviceParams builder. The current main
names CUDA mirrors explicitly with a ``_device`` suffix.
"""

from typing import Any

import torch


class MlaPageRRPrefillParams:
    def __init__(
        self,
        *,
        attn_inputs: Any,
        q_lens_host: list[int],
        kv_lens_host: list[int],
        prefix_lens_host: list[int],
        qo_indptr_d: torch.Tensor,
        kv_indptr_d: torch.Tensor,
        positions_d: torch.Tensor,
        batch_indice_d: torch.Tensor,
        batch_reuse_info_vec_d: torch.Tensor,
    ) -> None:
        self.attn_inputs = attn_inputs
        self.q_lens_host = q_lens_host
        self.kv_lens_host = kv_lens_host
        self.prefix_lens_host = prefix_lens_host
        self.qo_indptr_d = qo_indptr_d
        self.prefill_ragged_kv_len_indptr_d = kv_indptr_d
        self.positions_d = positions_d
        self.batch_indice_d = batch_indice_d
        self.batch_reuse_info_vec_d = batch_reuse_info_vec_d
        self.has_reuse_cache = any(prefix_lens_host)
        self.slot_mapping = None
        self.qo_indptr_h = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(q_lens_host), 0).tolist()),
            dtype=torch.int32,
        )
        self.prefill_ragged_kv_len_indptr_h = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(kv_lens_host), 0).tolist()),
            dtype=torch.int32,
        )


def build_mla_page_rr_prefill_params(
    attn_inputs: Any, page_size: int
) -> MlaPageRRPrefillParams:
    """Construct the exact feat/k3_dev device plan from main's tensor names."""

    q_lens = [int(value) for value in attn_inputs.input_lengths.tolist()]
    prefix_lens = [int(value) for value in attn_inputs.prefix_lengths.tolist()]
    if not q_lens or len(q_lens) != len(prefix_lens):
        raise ValueError("MLA Page-RR requires matched nonempty Q/prefix lengths")
    kv_lens = [q + prefix for q, prefix in zip(q_lens, prefix_lens)]
    input_lengths_d = attn_inputs.input_lengths_device
    prefix_lengths_d = attn_inputs.prefix_lengths_device
    qo_indptr_d = attn_inputs.cu_seqlens_device
    if any(not tensor.is_cuda for tensor in (
        input_lengths_d, prefix_lengths_d, qo_indptr_d
    )):
        raise ValueError("MLA Page-RR requires CUDA attention metadata")

    device = input_lengths_d.device
    # New main can hand the MTP Prefill consumer a stale cu_kv_seqlens_device
    # when the length producer and attention run on different streams. The
    # host Q/prefix lengths above already determine this eager plan; own its
    # tiny device indptr instead of borrowing mutable Cache metadata.
    kv_offsets = [0]
    for length in kv_lens:
        kv_offsets.append(kv_offsets[-1] + length)
    kv_indptr_d = torch.tensor(kv_offsets, dtype=torch.int32, device=device)
    total_q = sum(q_lens)
    packed_indices = torch.arange(total_q, dtype=torch.int32, device=device)
    fixed_q_len = q_lens[0] if all(q == q_lens[0] for q in q_lens) else 0
    if fixed_q_len:
        batch_indice_d = torch.div(packed_indices, fixed_q_len, rounding_mode="floor")
        local_positions = torch.remainder(packed_indices, fixed_q_len)
    else:
        max_q_len = max(q_lens)
        padded_indices = packed_indices + attn_inputs.padding_offset
        batch_indice_d = torch.div(padded_indices, max_q_len, rounding_mode="floor")
        local_positions = torch.remainder(padded_indices, max_q_len)
    positions_d = (
        prefix_lengths_d.index_select(0, batch_indice_d.to(torch.int64))
        + local_positions
    )

    block_table = attn_inputs.kv_cache_kernel_block_id_device
    if block_table is None:
        raise ValueError("MLA Page-RR requires a local device block table")
    max_blocks = int(block_table.shape[1])
    batch_ids_d = torch.arange(len(q_lens), dtype=torch.int32, device=device)
    page_counts_d = torch.div(
        prefix_lengths_d + page_size - 1, page_size, rounding_mode="floor"
    )
    batch_reuse_info_vec_d = torch.stack(
        (batch_ids_d, prefix_lengths_d, batch_ids_d * max_blocks, page_counts_d),
        dim=1,
    )
    current_stream = torch.cuda.current_stream(device)
    for tensor in (
        input_lengths_d,
        prefix_lengths_d,
        qo_indptr_d,
        kv_indptr_d,
        block_table,
        positions_d,
        batch_indice_d,
        batch_reuse_info_vec_d,
    ):
        tensor.record_stream(current_stream)
    return MlaPageRRPrefillParams(
        attn_inputs=attn_inputs,
        q_lens_host=q_lens,
        kv_lens_host=kv_lens,
        prefix_lens_host=prefix_lens,
        qo_indptr_d=qo_indptr_d,
        kv_indptr_d=kv_indptr_d,
        positions_d=positions_d,
        batch_indice_d=batch_indice_d,
        batch_reuse_info_vec_d=batch_reuse_info_vec_d,
    )
