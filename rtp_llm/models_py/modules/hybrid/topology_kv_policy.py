"""Topology-aware sparse KV candidate policy.

The policy is an opt-in post-processing layer for indexer top-k results. Learned
`topk_indices` must already be absolute token indices in the downstream sparse
attention coordinate system. For ragged and CP-prefill paths, `row_starts` plus
`topk_indices_offset` define the valid absolute interval for each row:
`[row_start + offset, row_start + offset + length)`. Negative values are padding.
Non-padding learned indices outside that interval fail fast because silently
dropping them can hide a coordinate-system mismatch.

`stable_scaffold` and `output_contract` describe repeated prompt structure such
as system instructions, tool schemas, and response contracts. Their fingerprint
is only an observability key for repeated stable regions; this module does not
physically compress the KV cache. Counters report selected-token shape and how
many stable tokens were represented by that fingerprint but not selected as raw
top-k candidates.
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Literal, Optional

import torch

TopologyKvPolicy = Literal[
    "disabled",
    "topology_sparse_merge",
    "topology_compress_sparse",
    "topology_only",
]
SUPPORTED_TOPOLOGY_KV_POLICIES = frozenset(
    {
        "disabled",
        "topology_sparse_merge",
        "topology_compress_sparse",
        "topology_only",
    }
)


def normalize_topology_kv_policy(policy: str) -> TopologyKvPolicy:
    normalized = policy.strip()
    if normalized not in SUPPORTED_TOPOLOGY_KV_POLICIES:
        raise ValueError(f"unknown topology KV policy: {policy}")
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class TopologyKvPolicyConfig:
    policy: TopologyKvPolicy
    sink_blocks: int
    local_blocks: int
    witness_blocks: int
    block_size: int

    def __post_init__(self) -> None:
        normalize_topology_kv_policy(self.policy)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.sink_blocks < 0:
            raise ValueError("sink_blocks must be non-negative")
        if self.local_blocks < 0:
            raise ValueError("local_blocks must be non-negative")
        if self.witness_blocks < 0:
            raise ValueError("witness_blocks must be non-negative")


@dataclass(frozen=True)
class TopologyKvCounters:
    raw_selected_tokens: int
    compressed_tokens_represented: int
    unselected_stable_tokens: int
    compression_hits: int
    schedule_ms: float


@dataclass(frozen=True)
class TopologyKvPolicyResult:
    topk_indices: torch.Tensor
    counters: TopologyKvCounters
    stable_fingerprint: str


def _stable_fingerprint(
    stable_scaffold: Optional[str], output_contract: Optional[str]
) -> str:
    if not stable_scaffold and not output_contract:
        return ""
    material = f"{stable_scaffold or ''}\n---contract---\n{output_contract or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _append_block_tokens(
    selected: list[int],
    seen: set[int],
    block_id: int,
    block_size: int,
    length: int,
) -> None:
    start = block_id * block_size
    end = min(start + block_size, length)
    for token in range(start, end):
        if token not in seen:
            selected.append(token)
            seen.add(token)


def _structural_tokens(
    length: int,
    config: TopologyKvPolicyConfig,
    drift_row: Optional[torch.Tensor],
) -> list[int]:
    if length <= 0:
        return []
    block_count = (length + config.block_size - 1) // config.block_size
    selected: list[int] = []
    seen: set[int] = set()
    ordered_blocks: list[int] = []

    def add_block(block_id: int) -> None:
        if block_id not in ordered_blocks:
            ordered_blocks.append(block_id)

    for block_id in range(min(config.sink_blocks, block_count)):
        add_block(block_id)

    local_start = max(0, block_count - config.local_blocks)
    for block_id in range(local_start, block_count):
        add_block(block_id)

    if config.witness_blocks > 0:
        if drift_row is not None and drift_row.numel() >= block_count:
            witness_blocks = (
                torch.topk(
                    drift_row[:block_count].float(),
                    k=min(config.witness_blocks, block_count),
                )
                .indices.cpu()
                .tolist()
            )
        else:
            step = max(1, block_count // (config.witness_blocks + 1))
            witness_blocks = list(range(step, block_count, step))[
                : config.witness_blocks
            ]
        for block_id in witness_blocks:
            add_block(int(block_id))

    # Representatives first keep both sink and latest-local evidence under tiny budgets.
    for block_id in ordered_blocks:
        token = min((block_id + 1) * config.block_size, length) - 1
        if token not in seen:
            selected.append(token)
            seen.add(token)

    for block_id in ordered_blocks:
        _append_block_tokens(selected, seen, block_id, config.block_size, length)

    return selected


def _merge_row(
    learned_row: torch.Tensor,
    structural: list[int],
    local_start: int,
    local_length: int,
) -> list[int]:
    budget = learned_row.numel()
    selected: list[int] = []
    seen: set[int] = set()
    local_end = local_start + local_length

    for token in structural:
        if local_start <= token < local_end and token not in seen:
            selected.append(token)
            seen.add(token)
            if len(selected) == budget:
                return selected

    for raw_value in learned_row.detach().cpu().tolist():
        value = int(raw_value)
        if value < 0:
            continue
        if local_start <= value < local_end and value not in seen:
            selected.append(value)
            seen.add(value)
            if len(selected) == budget:
                return selected

    while len(selected) < budget:
        selected.append(-1)
    return selected


def _validate_learned_row_coordinates(
    learned_row: torch.Tensor,
    *,
    row_idx: int,
    local_start: int,
    local_length: int,
) -> None:
    local_end = local_start + local_length
    valid_values = learned_row[learned_row >= 0]
    if valid_values.numel() == 0:
        return
    invalid = (valid_values < local_start) | (valid_values >= local_end)
    if bool(invalid.any().item()):
        first_invalid = int(valid_values[invalid][0].item())
        raise ValueError(
            "learned topk index outside valid topology KV range: "
            f"row={row_idx}, value={first_invalid}, expected absolute index in "
            f"[{local_start}, {local_end})"
        )


def _zero_counters(topk_indices: torch.Tensor) -> TopologyKvCounters:
    return TopologyKvCounters(
        raw_selected_tokens=int((topk_indices >= 0).sum().item()),
        compressed_tokens_represented=0,
        unselected_stable_tokens=0,
        compression_hits=0,
        schedule_ms=0.0,
    )


def apply_topology_kv_policy(
    topk_indices: torch.Tensor,
    lengths: torch.Tensor,
    *,
    config: TopologyKvPolicyConfig,
    row_starts: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    stable_scaffold: Optional[str] = None,
    output_contract: Optional[str] = None,
    block_drift_scores: Optional[torch.Tensor] = None,
    previous_fingerprint: Optional[str] = None,
) -> TopologyKvPolicyResult:
    """Apply the topology policy to absolute learned top-k token indices.

    Returned indices use the same absolute sparse-attention coordinate system as
    the input `topk_indices`. `row_starts` and `topk_indices_offset` are only used
    to derive the row-local valid interval; they are not subtracted from learned
    values.
    """

    started = time.perf_counter()
    fingerprint = _stable_fingerprint(stable_scaffold, output_contract)

    if config.policy == "disabled":
        return TopologyKvPolicyResult(
            topk_indices, _zero_counters(topk_indices), fingerprint
        )
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must have shape [rows, topk]")
    if lengths.ndim != 1 or lengths.numel() != topk_indices.size(0):
        raise ValueError("lengths must have shape [rows]")

    lengths_cpu = lengths.detach().cpu()
    starts_cpu = (
        row_starts.detach().cpu()
        if row_starts is not None
        else torch.zeros_like(lengths_cpu)
    )
    offsets_cpu = (
        topk_indices_offset.detach().cpu()
        if topk_indices_offset is not None
        else torch.zeros_like(lengths_cpu)
    )
    raw_selected_before = int((topk_indices >= 0).sum().item())
    rows = []

    compressed_tokens = 0
    compression_hits = 0
    if config.policy == "topology_compress_sparse" and fingerprint:
        stable_tokens = int(lengths_cpu.sum().item())
        compressed_tokens = max(0, stable_tokens - raw_selected_before)
        if previous_fingerprint and previous_fingerprint == fingerprint:
            compression_hits = 1

    for row_idx in range(topk_indices.size(0)):
        row_start = int(starts_cpu[row_idx].item())
        row_length = int(lengths_cpu[row_idx].item())
        offset = int(offsets_cpu[row_idx].item())
        drift_row = (
            block_drift_scores[row_idx]
            if block_drift_scores is not None and block_drift_scores.ndim == 2
            else None
        )
        structural = _structural_tokens(row_length, config, drift_row)
        structural = [token + row_start + offset for token in structural]
        learned = (
            topk_indices.new_full(topk_indices[row_idx].shape, -1)
            if config.policy == "topology_only"
            else topk_indices[row_idx]
        )
        _validate_learned_row_coordinates(
            learned,
            row_idx=row_idx,
            local_start=row_start + offset,
            local_length=row_length,
        )
        rows.append(
            torch.tensor(
                _merge_row(learned, structural, row_start + offset, row_length),
                dtype=topk_indices.dtype,
            )
        )

    merged = torch.stack(rows, dim=0).to(device=topk_indices.device)
    raw_selected_after = int((merged >= 0).sum().item())
    counters = TopologyKvCounters(
        raw_selected_tokens=raw_selected_after,
        compressed_tokens_represented=compressed_tokens,
        unselected_stable_tokens=compressed_tokens,
        compression_hits=compression_hits,
        schedule_ms=(time.perf_counter() - started) * 1000,
    )
    return TopologyKvPolicyResult(merged, counters, fingerprint)
