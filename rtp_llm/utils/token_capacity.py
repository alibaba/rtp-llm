"""Token-layout bounds shared by model buffers and warmup configuration."""

from math import gcd


def align_up(tokens: int, alignment: int) -> int:
    return (tokens + alignment - 1) // alignment * alignment


def sp_padded_tokens(batch_size: int, query_width: int, tp_size: int) -> int:
    """Global rows after padding complete, fixed-width requests for SP."""
    request_alignment = tp_size // gcd(tp_size, query_width)
    return align_up(batch_size, request_alignment) * query_width


def cp_padded_tokens_per_rank_bound(max_seq_len: int, cp_size: int) -> int:
    """Local rows for one sequence; multi-rank ZigZag CP pads to 2 * CP."""
    cp_size = max(int(cp_size), 1)
    max_seq_len = max(int(max_seq_len), 0)
    if cp_size == 1:
        return max_seq_len
    return align_up(max_seq_len, 2 * cp_size) // cp_size


def cp_padded_batch_tokens_per_rank_bound(
    max_batch_tokens: int, cp_size: int, max_batch_size: int
) -> int:
    """Local bound for requests padded independently before CP splitting.

    Each nonempty row adds at most 2 * CP - 1 tokens. The padded total is a
    multiple of 2 * CP, so round that upper bound down before splitting.
    """
    max_batch_tokens = max(int(max_batch_tokens), 0)
    cp_size = max(int(cp_size), 1)
    if cp_size == 1:
        return max_batch_tokens
    max_batch_size = min(max(int(max_batch_size), 1), max_batch_tokens)
    alignment = 2 * cp_size
    return 2 * ((max_batch_tokens + max_batch_size * (alignment - 1)) // alignment)
