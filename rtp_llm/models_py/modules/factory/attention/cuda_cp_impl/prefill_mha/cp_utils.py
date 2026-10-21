import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper


def plan_prefix_paged_attention(
    wrapper: BatchPrefillWithPagedKVCacheWrapper,
    qo_indptr: torch.Tensor,
    prefix_lengths: torch.Tensor,
    params,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    device,
) -> None:
    """Plan paged attention for the prefix portion of KV cache.

    All prefix positions precede every new-token position, so causal=False.
    This is shared by all CP implementations that need prefix cache support.
    """
    batch_size = params.kvlen_h.shape[0]
    prefix_lens = prefix_lengths.cpu().to(torch.int32)
    assert (prefix_lens % page_size == 0).all(), (
        f"prefix lengths must be multiples of page_size({page_size}), "
        f"got {prefix_lens}"
    )

    prefix_pages = prefix_lens // page_size
    page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    page_indptr[1:] = prefix_pages.cumsum(0)

    full_page_starts = params.decode_page_indptr_h[:batch_size].to(torch.int32)
    all_page_indices = params.page_indice_d

    total_pages = page_indptr[-1].item()
    if total_pages > 0:
        expanded_starts = torch.repeat_interleave(full_page_starts, prefix_pages)
        local_offsets = torch.arange(
            total_pages, dtype=torch.int32
        ) - torch.repeat_interleave(page_indptr[:batch_size], prefix_pages)
        gather_idx = (expanded_starts + local_offsets).long().to(device)
        prefix_page_indices = all_page_indices[gather_idx]
    else:
        prefix_page_indices = torch.tensor([], dtype=torch.int32, device=device)

    last_page_len = torch.where(prefix_pages > 0, page_size, 0).to(torch.int32)

    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=page_indptr.to(device),
        paged_kv_indices=prefix_page_indices,
        paged_kv_last_page_len=last_page_len.to(device),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=False,
        q_data_type=torch.bfloat16,
    )


def generate_full_causal_kv_indices(cp_chunk_lengths, cp_rank, cp_size):
    """Generate KV indices covering the **full causal range** for each Q-half.

    Used by the **all-gather (non-overlap)** implementation, where a single
    attention call per Q-half handles both local and non-local KV.

    Under zigzag load balancing, queries are split into two halves:
      - part0 (first-half Q): positions [rank*h, (rank+1)*h)
        -> causal KV range is [0, (rank+1)*h), i.e. h*(rank+1) tokens
      - part1 (second-half Q): positions [(2S-rank-1)*h, (2S-rank)*h)
        -> causal KV range is [0, (2S-rank)*h), i.e. h*(2*cp_size-rank) tokens

    where h = chunk_len // 2, S = cp_size.

    Args:
        cp_chunk_lengths: Per-batch chunk length (tokens assigned to one rank).
        cp_rank: Rank of the current process.
        cp_size: Total number of CP ranks.
    Returns:
        (kv_part0_indices, kv_part1_indices): indices into the all-gathered
        KV buffer for the first-half and second-half Q respectively.
    """
    total_seq_lengths = [x * cp_size for x in cp_chunk_lengths]

    kv_part_0_indices = []
    kv_part_1_indices = []
    seq_offset = 0
    for i in range(len(total_seq_lengths)):
        assert cp_chunk_lengths[i] % 2 == 0
        h = cp_chunk_lengths[i] // 2

        end_part0 = h * (cp_rank + 1)
        if end_part0 > 0:
            kv_part_0_indices.extend(range(seq_offset, seq_offset + end_part0))

        end_part1 = h * (2 * cp_size - cp_rank)
        if end_part1 > 0:
            kv_part_1_indices.extend(range(seq_offset, seq_offset + end_part1))

        seq_offset += total_seq_lengths[i]
    return kv_part_0_indices, kv_part_1_indices


def generate_nonlocal_causal_kv_indices(cp_chunk_lengths, cp_rank, cp_size):
    """Generate KV indices covering only the **non-local** causal range.

    Used by the **all-gather with overlap** implementation, where local
    causal attention is computed separately and the results are merged via
    ``merge_state``.  The indices here must *exclude* all local KV positions
    (both first-half and second-half of the local chunk) to avoid
    double-counting in the merge.

    Under zigzag load balancing (h = chunk_len // 2, S = cp_size):
      - part0 (first-half Q): full causal range [0, (rank+1)*h)
        minus local first-half [rank*h, (rank+1)*h)
        -> non-local count = h * rank
      - part1 (second-half Q): full causal range [0, (2S-rank)*h)
        minus local first-half [rank*h, (rank+1)*h)
        minus local second-half [(2S-rank-1)*h, (2S-rank)*h)
        -> non-local count = h * (2*cp_size - rank - 2)

    Args:
        cp_chunk_lengths: Per-batch chunk length (tokens assigned to one rank).
        cp_rank: Rank of the current process.
        cp_size: Total number of CP ranks.
    Returns:
        (kv_part0_indices, kv_part1_indices): non-local KV indices for the
        first-half and second-half Q respectively.
    """
    total_seq_lengths = [x * cp_size for x in cp_chunk_lengths]

    kv_part_0_indices = []
    kv_part_1_indices = []
    seq_offset = 0
    for i in range(len(total_seq_lengths)):
        assert cp_chunk_lengths[i] % 2 == 0
        h = cp_chunk_lengths[i] // 2

        # part0: [0, rank*h) — the local first-half [rank*h, (rank+1)*h) is excluded
        end_part0 = h * cp_rank
        if end_part0 > 0:
            kv_part_0_indices.extend(range(seq_offset, seq_offset + end_part0))

        # part1: [0, (2S-rank-1)*h) minus [rank*h, (rank+1)*h)
        local_fh_start = h * cp_rank
        local_fh_end = h * (cp_rank + 1)
        end_part1 = h * (2 * cp_size - cp_rank - 1)
        if local_fh_start > 0:
            kv_part_1_indices.extend(range(seq_offset, seq_offset + local_fh_start))
        if local_fh_end < end_part1:
            kv_part_1_indices.extend(
                range(seq_offset + local_fh_end, seq_offset + end_part1)
            )

        seq_offset += total_seq_lengths[i]
    return kv_part_0_indices, kv_part_1_indices


def generate_q_indices(cp_chunk_lengths):
    """Split local Q tokens into first-half and second-half per batch item.

    Under zigzag, each rank's chunk is [first_half, second_half] where
    first_half holds early-position tokens and second_half holds late ones.
    This function produces indices to separate them for the two ragged
    attention calls (part0 and part1).

    Args:
        cp_chunk_lengths: Per-batch chunk lengths on this rank.
    Returns:
        (indices0, indices1): flat index lists into the concatenated local Q.

    Example:
        cp_chunk_lengths = [8, 4, 4]
        indices0 = [0, 1, 2, 3, 8, 9, 12, 13]   (first halves)
        indices1 = [4, 5, 6, 7, 10, 11, 14, 15]  (second halves)
    """
    indices0 = []
    indices1 = []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        # Use ceiling division for first half (gets extra element if odd)
        half0 = (chunk_len + 1) // 2
        indices0.extend(range(offset, offset + half0))
        indices1.extend(range(offset + half0, offset + chunk_len))
        offset += chunk_len

    return indices0, indices1


def generate_half_q_indices(cp_chunk_lengths):
    """Return indices of the **second-half** Q tokens from each chunk.

    Used by the all-to-all CP implementation: after the first round of ring
    attention covers the first-half Q with full causal context, only
    second-half Q tokens need additional KV from other ranks.

    Args:
        cp_chunk_lengths: Per-batch chunk lengths on this rank.
    Returns:
        half_q_indices: flat list of second-half Q indices.
    """
    half_q_indices = []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        assert chunk_len % 2 == 0
        half_q_indices.extend(range(offset + (chunk_len) // 2, offset + chunk_len))
        offset += chunk_len
    return half_q_indices


def generate_half_kv_indices(cp_chunk_lengths):
    """Return indices of the **first-half** KV tokens from each chunk.

    Used by the all-to-all CP implementation: these are the early-position
    KV tokens that second-half Q on other ranks still need to attend to.

    Args:
        cp_chunk_lengths: Per-batch chunk lengths on this rank.
    Returns:
        half_kv_indices: flat list of first-half KV indices.
    """
    half_kv_indices = []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        assert chunk_len % 2 == 0
        half_kv_indices.extend(range(offset, offset + (chunk_len) // 2))
        offset += chunk_len
    return half_kv_indices
