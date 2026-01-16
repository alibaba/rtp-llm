import torch

# Global workspace buffer shared across all wrappers
_g_workspace_buffer = None
_g_workspace_size = 512 * 1024 * 1024  # 512MB


def get_workspace_buffer(device: torch.device) -> torch.Tensor:
    """Get or create global workspace buffer for FlashInfer."""
    global _g_workspace_buffer
    if _g_workspace_buffer is None:
        _g_workspace_buffer = torch.empty(
            _g_workspace_size,
            dtype=torch.uint8,
            device=device,
        )
    return _g_workspace_buffer


# generate kv_indices for zigzag load balance with partial q and full kv
def generate_kv_indices(cp_chunk_lengths, cp_rank, cp_size, is_non_local=False):
    """
    Generate kv_indices for zigzag load balance with partial q and full kv
    Args:
        cp_chunk_lengths: List of chunk lengths for each CP rank
        cp_rank: Rank of the current process
        cp_size: Size of the context parallel group
        is_non_local: Whether the kv_indices include the local kv index
    Returns:
        kv_part_0_indices: List of indices for the first part of the kv
        kv_part_1_indices: List of indices for the second part of the kv
    """
    total_seq_lengths = [x * cp_size for x in cp_chunk_lengths]

    kv_part_0_indices = []
    kv_part_1_indices = []
    seq_offset = 0
    for i in range(len(total_seq_lengths)):
        assert cp_chunk_lengths[i] % 2 == 0
        half_chunk_len = cp_chunk_lengths[i] // 2
        # with out prefix cache, the start kv position is always 0
        start_pos_part0 = 0
        end_pos_part0 = half_chunk_len * (cp_rank + 1 - int(is_non_local))
        start_pos_part1 = 0
        end_pos_part1 = half_chunk_len * (2 * cp_size - cp_rank - int(is_non_local))

        if end_pos_part0 > start_pos_part0:
            kv_part_0_indices.extend(
                range(start_pos_part0 + seq_offset, end_pos_part0 + seq_offset)
            )
        if end_pos_part1 > start_pos_part1:
            kv_part_1_indices.extend(
                range(start_pos_part1 + seq_offset, end_pos_part1 + seq_offset)
            )
        seq_offset += total_seq_lengths[i]
    return kv_part_0_indices, kv_part_1_indices


# generate q_indices for zigzag load balance with partial q and full kv
def generate_q_indices(cp_chunk_lengths):
    """Generate two sets of indices by splitting each chunk in half.
    Args:
        cp_chunk_lengths: List of chunk lengths for each CP rank
    Returns:
        indices0: List of first half indices from each chunk
        indices1: List of second half indices from each chunk
    Example 1:
        cp_chunk_lengths = [8, 4, 4]
        indices0 = [0, 1, 2, 3, 8, 9, 12, 13]
        indices1 = [4, 5, 6, 7, 10, 11, 14, 15]
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


# for all2all mode with zigzag loadbalance
def generate_half_q_indices(cp_chunk_lengths):
    """
    Generate half q indices for all2all with zigzag loadbalance
    Args:
        cp_chunk_lengths: List of chunk lengths for each CP rank
    Returns:
        half_q_indices: List of indices for the first half of the q
    """
    half_q_indices = []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        assert chunk_len % 2 == 0
        half_q_indices.extend(range(offset + (chunk_len) // 2, offset + chunk_len))
        offset += chunk_len
    return half_q_indices


# for all2all mode with zigzag loadbalance
def generate_half_kv_indices(cp_chunk_lengths):
    """
    Generate half kv indices for all2all with zigzag loadbalance
    Args:
        cp_chunk_lengths: List of chunk lengths for each CP rank
    Returns:
        half_kv_indices: List of indices for the first half of the kv
    """
    half_kv_indices = []
    offset = 0
    for chunk_len in cp_chunk_lengths:
        assert chunk_len % 2 == 0
        half_kv_indices.extend(range(offset, offset + (chunk_len) // 2))
        offset += chunk_len
    return half_kv_indices
