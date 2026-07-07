import argparse
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BlockCandidateConfig:
    block_size: int
    sink_blocks: int
    local_blocks: int
    salience_blocks: int
    max_candidate_blocks: int


@dataclass(frozen=True)
class DecodeBenchmarkResult:
    dense_ms: float
    sparse_ms: float
    speedup: float
    seq_len: int
    selected_tokens: int


def build_key_block_centroids(key: torch.Tensor, block_size: int) -> torch.Tensor:
    """Average K-cache vectors into block centroids.

    Accepts `[batch, heads, seq, dim]`, `[heads, seq, dim]`, or `[seq, dim]`.
    Batch and head lanes are both averaged; the benchmark uses batch=1 so the
    returned centroids describe one request's token geometry.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if key.ndim == 4:
        normalized_key = key
    elif key.ndim == 3:
        normalized_key = key.unsqueeze(0)
    elif key.ndim == 2:
        normalized_key = key.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("key must have shape [batch, heads, seq, dim], [heads, seq, dim], or [seq, dim]")

    batch, heads, seq_len, dim = normalized_key.shape
    if seq_len <= 0:
        raise ValueError("key sequence length must be positive")

    block_count = (seq_len + block_size - 1) // block_size
    padded_len = block_count * block_size
    if padded_len != seq_len:
        padding = normalized_key.new_zeros(batch, heads, padded_len - seq_len, dim)
        normalized_key = torch.cat([normalized_key, padding], dim=2)

    accumulation_key = (
        normalized_key.float()
        if normalized_key.dtype in (torch.float16, torch.bfloat16)
        else normalized_key
    )
    blocks = accumulation_key.reshape(batch, heads, block_count, block_size, dim)
    block_sums = blocks.sum(dim=(0, 1, 3))
    token_counts = torch.full(
        (block_count,),
        block_size,
        dtype=block_sums.dtype,
        device=block_sums.device,
    )
    token_counts[-1] = seq_len - (block_count - 1) * block_size
    denominator = token_counts.unsqueeze(1) * batch * heads
    return block_sums / denominator


def block_schedule_to_token_indices(
    block_schedule: torch.Tensor,
    block_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Expand rectangular block ids to rectangular token ids with -1 padding."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if block_schedule.ndim != 2:
        raise ValueError("block_schedule must have shape [rows, candidate_blocks]")

    width = block_schedule.shape[1] * block_size
    block_offsets = torch.arange(block_size, device=block_schedule.device)
    token_indices = block_schedule.unsqueeze(-1) * block_size + block_offsets
    valid_tokens = (block_schedule.unsqueeze(-1) >= 0) & (token_indices < seq_len)
    token_indices = torch.where(
        valid_tokens,
        token_indices,
        torch.full_like(token_indices, -1),
    )
    return token_indices.reshape(block_schedule.shape[0], width)


def _validate_config(config: BlockCandidateConfig) -> None:
    fields = {
        "block_size": config.block_size,
        "sink_blocks": config.sink_blocks,
        "local_blocks": config.local_blocks,
        "salience_blocks": config.salience_blocks,
        "max_candidate_blocks": config.max_candidate_blocks,
    }
    for name, value in fields.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.block_size <= 0:
        raise ValueError("block_size must be positive")
    if config.max_candidate_blocks <= 0:
        raise ValueError("max_candidate_blocks must be positive")


def _append_unique(target: List[int], values: Iterable[int], limit: int) -> None:
    for value in values:
        if value < 0 or value in target:
            continue
        if len(target) >= limit:
            return
        target.append(value)


def _build_drift_score_values(key_block_centroids: torch.Tensor) -> List[float]:
    if key_block_centroids.shape[0] == 1:
        drift_scores = torch.zeros(1, device=key_block_centroids.device)
    else:
        first_score = torch.zeros(1, device=key_block_centroids.device)
        drift_scores = torch.cat(
            [
                first_score,
                torch.linalg.vector_norm(
                    key_block_centroids[1:] - key_block_centroids[:-1],
                    dim=1,
                ),
            ]
        )
    return drift_scores.detach().cpu().tolist()


def _build_candidate_row(
    query_block: int,
    config: BlockCandidateConfig,
    drift_score_values: Sequence[float],
) -> List[int]:
    row: List[int] = []
    limit = config.max_candidate_blocks

    _append_unique(row, [query_block], limit)

    sink_end = min(config.sink_blocks, query_block + 1)
    _append_unique(row, range(sink_end), limit)

    local_start = max(0, query_block - config.local_blocks + 1)
    _append_unique(row, range(query_block, local_start - 1, -1), limit)

    eligible = [block for block in range(query_block + 1) if block not in row]
    eligible.sort(key=lambda block: (-drift_score_values[block], block))
    _append_unique(row, eligible[: config.salience_blocks], limit)

    row = sorted(row)
    row.extend([-1] * (limit - len(row)))
    return row


def _prioritize_latest_blocks_for_partial_budget(
    row: Sequence[int],
    query_block: int,
    local_blocks: int,
) -> List[int]:
    valid_row = [block for block in row if block >= 0]
    local_start = max(0, query_block - local_blocks + 1)
    local_row = [
        block
        for block in range(query_block, local_start - 1, -1)
        if block in valid_row
    ]
    other_row = [block for block in valid_row if block not in local_row]
    ordered_row = local_row + other_row
    ordered_row.extend([-1] * (len(row) - len(ordered_row)))
    return ordered_row


def _candidate_block_budget_for_tokens(seq_len: int, selected_tokens: int, block_size: int) -> int:
    block_count = (seq_len + block_size - 1) // block_size
    final_block_tokens = seq_len % block_size or block_size
    padding_tokens = block_size - final_block_tokens
    blocks = (selected_tokens + padding_tokens + block_size - 1) // block_size
    return max(1, min(block_count, blocks))


def build_block_candidate_schedule(
    key_block_centroids: torch.Tensor,
    config: BlockCandidateConfig,
) -> torch.Tensor:
    """Build rectangular causal candidate block rows from key-block centroids.

    The schedule is intentionally backend-neutral: each row contains request-local
    block ids and is padded with -1. Candidate priority is the query block, sink
    blocks, remaining local causal blocks, then high-drift blocks. The high-drift
    score is a cheap change-point proxy: blocks where the centroid moves sharply
    from the previous block are treated as distinctive witness blocks worth
    replaying.
    """

    _validate_config(config)
    if key_block_centroids.ndim != 2:
        raise ValueError("key_block_centroids must have shape [blocks, dim]")

    block_count = key_block_centroids.shape[0]
    if block_count == 0:
        raise ValueError("key_block_centroids must not be empty")

    drift_score_values = _build_drift_score_values(key_block_centroids)

    rows: List[List[int]] = []
    for query_block in range(block_count):
        rows.append(_build_candidate_row(query_block, config, drift_score_values))

    return torch.tensor(rows, dtype=torch.long, device=key_block_centroids.device)


def dense_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    # q_len=1 decode already has no future query positions. `is_causal=True`
    # would mask as if this query were at position 0 without an offset mask.
    return F.scaled_dot_product_attention(query, key, value, is_causal=False)


def _sparse_decode_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    selected_key = key.index_select(2, indices)
    selected_value = value.index_select(2, indices)
    # See dense_decode_attention: selected decode rows use explicit candidates.
    return F.scaled_dot_product_attention(query, selected_key, selected_value, is_causal=False)


def build_topology_candidate_token_indices(
    key: torch.Tensor,
    selected_tokens: int,
    block_size: int = 64,
) -> torch.Tensor:
    if selected_tokens <= 0:
        raise ValueError("selected_tokens must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if key.ndim == 4:
        seq_len = key.shape[2]
    elif key.ndim == 3:
        seq_len = key.shape[1]
    elif key.ndim == 2:
        seq_len = key.shape[0]
    else:
        raise ValueError("key must have shape [batch, heads, seq, dim], [heads, seq, dim], or [seq, dim]")
    if selected_tokens > seq_len:
        raise ValueError("selected_tokens must not exceed seq_len")

    max_candidate_blocks = _candidate_block_budget_for_tokens(
        seq_len=seq_len,
        selected_tokens=selected_tokens,
        block_size=block_size,
    )
    sink_blocks = 0 if max_candidate_blocks == 1 else 1
    remaining_blocks = max_candidate_blocks - sink_blocks
    local_blocks = min(2, remaining_blocks)
    salience_blocks = max(0, remaining_blocks - local_blocks)
    centroids = build_key_block_centroids(key, block_size=block_size)
    drift_score_values = _build_drift_score_values(centroids)
    config = BlockCandidateConfig(
        block_size=block_size,
        sink_blocks=sink_blocks,
        local_blocks=local_blocks,
        salience_blocks=salience_blocks,
        max_candidate_blocks=max_candidate_blocks,
    )
    query_block = centroids.shape[0] - 1
    row = _build_candidate_row(
        query_block,
        config,
        drift_score_values,
    )
    if selected_tokens % block_size != 0 or seq_len % block_size != 0:
        row = _prioritize_latest_blocks_for_partial_budget(
            row,
            query_block,
            config.local_blocks,
        )
    token_indices = block_schedule_to_token_indices(
        torch.tensor([row], dtype=torch.long, device=centroids.device),
        block_size=block_size,
        seq_len=seq_len,
    ).reshape(-1)
    token_indices = token_indices[token_indices >= 0]
    if token_indices.numel() == 0:
        raise ValueError("topology schedule did not select any tokens")
    if token_indices.numel() < selected_tokens:
        raise ValueError("topology schedule selected fewer tokens than requested")
    return token_indices[:selected_tokens]


def sparse_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    """Run decode attention over one shared candidate token list.

    `candidate_indices` must be flat `[tokens]` or rectangular `[1, tokens]`.
    Per-batch or per-head candidate schedules are intentionally not supported by
    this benchmark helper.
    """

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, seq, dim]")
    if query.shape[2] != 1:
        raise ValueError("sparse_decode_attention currently supports one decode query")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query, key, and value batch/head/head_dim must match")

    if candidate_indices.ndim == 1:
        indices = candidate_indices
    elif candidate_indices.ndim == 2 and candidate_indices.shape[0] == 1:
        indices = candidate_indices.reshape(-1)
    else:
        raise ValueError("candidate_indices must have shape [tokens] or [1, tokens]")
    indices = indices.to(device=key.device, dtype=torch.long)
    indices = indices[indices >= 0]
    if indices.numel() == 0:
        raise ValueError("candidate_indices must select at least one token")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError("candidate_indices must not contain duplicate tokens")
    if int(indices.max()) >= key.shape[2]:
        raise ValueError("candidate_indices contain tokens outside key/value length")

    return _sparse_decode_attention_impl(query, key, value, indices)


def benchmark_decode_attention(
    seq_len: int,
    selected_tokens: int,
    heads: int,
    head_dim: int,
    rounds: int,
    warmup: int,
    dtype: torch.dtype,
    device: str,
) -> DecodeBenchmarkResult:
    if selected_tokens > seq_len:
        raise ValueError("selected_tokens must not exceed seq_len")
    if rounds <= 0 or warmup < 0:
        raise ValueError("rounds must be positive and warmup must be non-negative")

    benchmark_device = torch.device(device)
    generator = torch.Generator(device=benchmark_device)
    generator.manual_seed(0)
    query = torch.randn(
        1, heads, 1, head_dim,
        device=benchmark_device,
        dtype=dtype,
        generator=generator,
    )
    key = torch.randn(
        1, heads, seq_len, head_dim,
        device=benchmark_device,
        dtype=dtype,
        generator=generator,
    )
    value = torch.randn(
        1, heads, seq_len, head_dim,
        device=benchmark_device,
        dtype=dtype,
        generator=generator,
    )
    candidate_indices = build_topology_candidate_token_indices(
        key,
        selected_tokens=selected_tokens,
    )

    for _ in range(warmup):
        dense_decode_attention(query, key, value)
        _sparse_decode_attention_impl(query, key, value, candidate_indices)
    if benchmark_device.type == "cuda":
        torch.cuda.synchronize(benchmark_device)

    dense_start = time.perf_counter()
    for _ in range(rounds):
        dense_decode_attention(query, key, value)
    if benchmark_device.type == "cuda":
        torch.cuda.synchronize(benchmark_device)
    dense_ms = (time.perf_counter() - dense_start) * 1000 / rounds

    sparse_start = time.perf_counter()
    for _ in range(rounds):
        _sparse_decode_attention_impl(query, key, value, candidate_indices)
    if benchmark_device.type == "cuda":
        torch.cuda.synchronize(benchmark_device)
    sparse_ms = (time.perf_counter() - sparse_start) * 1000 / rounds
    speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")

    return DecodeBenchmarkResult(
        dense_ms=dense_ms,
        sparse_ms=sparse_ms,
        speedup=speedup,
        seq_len=seq_len,
        selected_tokens=selected_tokens,
    )


def run_decode_attention_grid(
    seq_lens: Sequence[int],
    selected_tokens: Sequence[int],
    heads: int,
    head_dim: int,
    rounds: int,
    warmup: int,
    dtype: torch.dtype,
    device: str,
) -> List[DecodeBenchmarkResult]:
    results = []
    for seq_len in seq_lens:
        for selected in selected_tokens:
            if selected > seq_len:
                continue
            results.append(
                benchmark_decode_attention(
                    seq_len=seq_len,
                    selected_tokens=selected,
                    heads=heads,
                    head_dim=head_dim,
                    rounds=rounds,
                    warmup=warmup,
                    dtype=dtype,
                    device=device,
                )
            )
    return results


def format_benchmark_results(results: Sequence[DecodeBenchmarkResult]) -> str:
    lines = [
        "| seq_len | selected_tokens | dense_sdpa_ms | sparse_selected_ms | speedup |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result.seq_len} | {result.selected_tokens} | {result.dense_ms:.4f} | "
            f"{result.sparse_ms:.4f} | {result.speedup:.2f}x |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense decode attention against selected-token decode attention."
    )
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--selected-tokens", type=int, nargs="+", default=[512])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is not available")

    results = run_decode_attention_grid(
        seq_lens=[args.seq_len],
        selected_tokens=args.selected_tokens,
        heads=args.heads,
        head_dim=args.head_dim,
        rounds=args.rounds,
        warmup=args.warmup,
        dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device=args.device,
    )
    print(format_benchmark_results(results))


if __name__ == "__main__":
    main()
