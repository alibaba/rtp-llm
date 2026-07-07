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
    Batch and head lanes are averaged so the returned centroids describe the
    request-local token geometry independently of lane count.
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

    seq_len = normalized_key.shape[2]
    centroids = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        centroids.append(normalized_key[:, :, start:end, :].mean(dim=(0, 1, 2)))
    return torch.stack(centroids, dim=0)


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

    rows = []
    width = block_schedule.shape[1] * block_size
    for schedule_row in block_schedule.tolist():
        token_row: List[int] = []
        for block in schedule_row:
            if block < 0:
                token_row.extend([-1] * block_size)
                continue
            start = block * block_size
            end = min(start + block_size, seq_len)
            token_row.extend(range(start, end))
            token_row.extend([-1] * (block_size - max(0, end - start)))
        token_row = token_row[:width]
        token_row.extend([-1] * (width - len(token_row)))
        rows.append(token_row)
    return torch.tensor(rows, dtype=torch.long, device=block_schedule.device)


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


def build_block_candidate_schedule(
    key_block_centroids: torch.Tensor,
    config: BlockCandidateConfig,
) -> torch.Tensor:
    """Build rectangular causal candidate block rows from key-block centroids.

    The schedule is intentionally backend-neutral: each row contains request-local
    block ids and is padded with -1. Candidate priority is sink blocks, local
    causal blocks, then high-drift blocks. The high-drift score is a cheap
    0D-persistence proxy: blocks where the centroid moves sharply from the
    previous block are treated as stable witnesses worth replaying.
    """

    _validate_config(config)
    if key_block_centroids.ndim != 2:
        raise ValueError("key_block_centroids must have shape [blocks, dim]")

    block_count = key_block_centroids.shape[0]
    if block_count == 0:
        raise ValueError("key_block_centroids must not be empty")

    if block_count == 1:
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

    rows: List[List[int]] = []
    for query_block in range(block_count):
        row: List[int] = []
        limit = config.max_candidate_blocks

        sink_end = min(config.sink_blocks, query_block + 1)
        _append_unique(row, range(sink_end), limit)

        local_start = max(0, query_block - config.local_blocks + 1)
        _append_unique(row, range(query_block, local_start - 1, -1), limit)

        eligible = [block for block in range(query_block + 1) if block not in row]
        eligible.sort(key=lambda block: (-float(drift_scores[block]), block))
        _append_unique(row, eligible[: config.salience_blocks], limit)

        row = sorted(row)
        row.extend([-1] * (limit - len(row)))
        rows.append(row)

    return torch.tensor(rows, dtype=torch.long, device=key_block_centroids.device)


def dense_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(query, key, value, is_causal=False)


def _sparse_decode_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    selected_key = key.index_select(2, indices)
    selected_value = value.index_select(2, indices)
    return F.scaled_dot_product_attention(query, selected_key, selected_value, is_causal=False)


def build_topology_candidate_token_indices(
    key: torch.Tensor,
    selected_tokens: int,
    block_size: int = 64,
) -> torch.Tensor:
    if selected_tokens <= 0:
        raise ValueError("selected_tokens must be positive")

    seq_len = key.shape[2]
    max_candidate_blocks = max(1, (selected_tokens + block_size - 1) // block_size)
    centroids = build_key_block_centroids(key, block_size=block_size)
    schedule = build_block_candidate_schedule(
        centroids,
        BlockCandidateConfig(
            block_size=block_size,
            sink_blocks=min(1, max_candidate_blocks),
            local_blocks=min(2, max_candidate_blocks),
            salience_blocks=max_candidate_blocks,
            max_candidate_blocks=max_candidate_blocks,
        ),
    )
    token_indices = block_schedule_to_token_indices(
        schedule[-1:].contiguous(),
        block_size=block_size,
        seq_len=seq_len,
    ).reshape(-1)
    token_indices = token_indices[token_indices >= 0]
    if token_indices.numel() == 0:
        raise ValueError("topology schedule did not select any tokens")
    return token_indices[:selected_tokens]


def sparse_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, seq, dim]")
    if query.shape[2] != 1:
        raise ValueError("sparse_decode_attention currently supports one decode query")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query, key, and value batch/head/head_dim must match")

    indices = candidate_indices.reshape(-1).to(device=key.device, dtype=torch.long)
    indices = indices[indices >= 0]
    if indices.numel() == 0:
        raise ValueError("candidate_indices must select at least one token")
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

    torch.manual_seed(0)
    query = torch.randn(1, heads, 1, head_dim, device=device, dtype=dtype)
    key = torch.randn(1, heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(1, heads, seq_len, head_dim, device=device, dtype=dtype)
    candidate_indices = build_topology_candidate_token_indices(
        key,
        selected_tokens=selected_tokens,
    )

    for _ in range(warmup):
        dense_decode_attention(query, key, value)
        _sparse_decode_attention_impl(query, key, value, candidate_indices)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    dense_start = time.perf_counter()
    for _ in range(rounds):
        dense_decode_attention(query, key, value)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dense_ms = (time.perf_counter() - dense_start) * 1000 / rounds

    sparse_start = time.perf_counter()
    for _ in range(rounds):
        _sparse_decode_attention_impl(query, key, value, candidate_indices)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    sparse_ms = (time.perf_counter() - sparse_start) * 1000 / rounds

    return DecodeBenchmarkResult(
        dense_ms=dense_ms,
        sparse_ms=sparse_ms,
        speedup=dense_ms / sparse_ms,
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
